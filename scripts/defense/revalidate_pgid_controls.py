"""Evaluate authentic/cover controls through the unchanged PGID-F defense."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job', choices=['pgid0', 'pgid5'], required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    state_path = run / (args.job + '_controls_state.json')
    with state_path.open('x') as f:
        json.dump({'status': 'waiting_for_parent', 'pid': os.getpid()}, f)
    state = dict(status='waiting_for_parent', pid=os.getpid(), parent_pid=args.pid)
    try:
        control = json.loads((run / 'controls_protocol.json').read_text())
        assert sha(__file__) == control['script_sha256']
        assert sha(run / 'protocol.json') == control['parent_protocol_sha256']
        plan = json.loads((run / 'protocol.json').read_text())
        root = Path(plan['root'])
        parent_path = run / (args.job + '_state.json')
        parent = json.loads(parent_path.read_text())
        if parent['status'] != 'complete':
            cmd = Path(f'/proc/{args.pid}/cmdline').read_bytes()
            assert str(run).encode() in cmd and args.job.encode() in cmd
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.syscall(434, args.pid, 0)
            if fd < 0:
                assert json.loads(parent_path.read_text())['status'] == 'complete'
            else:
                select.select([fd], [], [])
                os.close(fd)
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'complete', parent.get('error')
        for name, digest in plan['source_sha256'].items():
            assert sha(run / name) == digest, name
        start = int(args.job[4:])
        shard = next(x for x in plan['pgid']['shards'] if x['start_index'] == start)
        gpu = shard['gpu']
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert gpu in [7, 5] and int(info[2]) < 50 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(root / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        state.update(status='running', gpu=gpu, gpu_uuid=info[1])
        save(state_path, state)
        sys.path.insert(0, str(run / 'PGID'))
        import torch
        from PIL import Image
        from pipe import pipe_utils
        from pgid_defense import PGIDPipeline
        from watermarks.gs_watermark import parser as gs_parser
        from watermarks.watermark_utils import SemanticWatermarks, validate, get_detection_threshold
        from utils import get_watermark_args, set_random_seed
        torch.set_num_threads(4)
        gs_args = gs_parser.parse_args([])
        gs_args.wm_type = 'GS'
        kwargs = get_watermark_args(gs_args, 'cuda:0')
        assert kwargs['message_width_in_bytes'] == 32 and kwargs['num_replications'] == 64 and kwargs['l'] == 1
        set_random_seed(control['evaluation_seed'])
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=plan['pgid']['modelid'],
                    resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM', eager_loading=True)
        defense = PGIDPipeline(pipe=provider, scheduler_config=provider.scheduler.config,
                    num_inference_steps=50, stopping_timestep=15, num_skip_inversion=3, guidance_strength=0.001)
        threshold = get_detection_threshold('GS', plan['pgid']['modelid'])
        assert threshold == 181 / 256
        originals = list(csv.DictReader((run / args.job / 'attack=forgery/all_results.csv').open()))
        rows = []
        for wm, idx in SemanticWatermarks.GS.value.generate_watermark_signals(
                total_num_signals=start + 5, latent_shape=provider.get_latent_shape(), batch_size=1,
                target_start_index=start, target_end_index=start + 5, w_seed=1, **kwargs):
            reference = next(r for r in originals if int(r['image_idx']) == idx and r['stage'] == 'watermarked')
            initial = wm.get_wm_latents()['message_bits_str_list'][0]
            assert initial == reference['message_bits_str_initial']
            clean = run / args.job / 'attack=forgery/watermarked_images' / f'image_{idx:03d}.png'
            cover_item = next(x for x in plan['pgid_inputs'] if x['index'] == idx)
            cover = run / cover_item['cover']
            assert sha(cover) == cover_item['cover_sha256']
            for kind, path in [('genuine', clean), ('cover', cover)]:
                image = Image.open(path).resize((512, 512)).convert('RGB')
                for defended in [False, True]:
                    set_random_seed(control['evaluation_seed'] + idx)
                    latent = defense.run_defense(image)['zT'] if defended else None
                    result = validate(image_to_verify_PIL=image, watermark=wm, pipe_provider_target=provider,
                              num_inference_steps_target=50, message_bits_str_initial=initial, provided_zT=latent)
                    bits = result['message_bits_str_recovered']
                    assert len(bits) == len(initial) == 256
                    ba = sum(a == b for a, b in zip(initial, bits)) / 256
                    assert ba == float(result['bit_accuracy'])
                    if kind == 'genuine' and not defended:
                        assert ba == float(reference['bit_accuracy']), 'Fresh reference replay mismatch'
                    rows.append(dict(index=idx, kind=kind, defended=defended, ba=ba, accepted=ba > threshold,
                                     input_sha256=sha(path)))
            print('CONTROL_COMPLETE', idx, flush=True)
        assert len(rows) == 20
        summary = [dict(kind=k, defended=d, n=5, accepted=sum(r['accepted'] for r in rows if r['kind'] == k and r['defended'] == d),
                   mean_ba=sum(r['ba'] for r in rows if r['kind'] == k and r['defended'] == d) / 5)
                   for k in ['genuine', 'cover'] for d in [False, True]]
        state.update(status='complete', rows=rows, summary=summary, protocol_sha256=sha(run / 'controls_protocol.json'),
                     parent_state_sha256=sha(parent_path), reference_replay_matches=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(state_path, state)


if __name__ == '__main__':
    main()
