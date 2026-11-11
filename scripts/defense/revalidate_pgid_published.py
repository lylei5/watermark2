"""Fixed published distortion suite on freshly generated PGID references."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'protocol.json').read_text())
    parent = Path(plan['parent_run'])
    output = run / f'shard_{args.shard}'
    output.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    try:
        assert sha(__file__) == plan['script_sha256']
        parent_plan = json.loads((parent / 'protocol.json').read_text())
        for name, digest in parent_plan['source_sha256'].items():
            assert sha(parent / name) == digest, name
        assert sha(parent / 'gs_image_utils.py') == plan['distortion_source_sha256']
        gpu = [7, 5][args.shard]
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 50 and int(info[3]) == 0, info
        root = Path(parent_plan['root'])
        ref = root / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip() == plan['model_revision']
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(root / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        sys.path.insert(0, str(parent / 'PGID'))
        import numpy as np
        import torch
        from PIL import Image
        from pipe import pipe_utils
        from pgid_defense import PGIDPipeline
        from watermarks.gs_watermark import parser as gs_parser
        from watermarks.watermark_utils import SemanticWatermarks, validate
        from utils import get_watermark_args, set_random_seed
        spec = importlib.util.spec_from_file_location('official_gs_distortions', parent / 'gs_image_utils.py')
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state.update(status='running', gpu=gpu, gpu_uuid=info[1])
        write(output / 'state.json', state)
        (output / 'work').mkdir()
        (output / 'latents').mkdir()
        os.chdir(output / 'work')
        gs_args = gs_parser.parse_args([])
        gs_args.wm_type = 'GS'
        kwargs = get_watermark_args(gs_args, 'cuda:0')
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent_plan['pgid']['modelid'],
                    resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM', eager_loading=True)
        defense = PGIDPipeline(provider, provider.scheduler.config, num_inference_steps=50,
                               stopping_timestep=15, num_skip_inversion=3, guidance_strength=0.001)
        start = args.shard * 5
        rows = []
        for wm, idx in SemanticWatermarks.GS.value.generate_watermark_signals(total_num_signals=start + 5,
                latent_shape=provider.get_latent_shape(), batch_size=1, target_start_index=start,
                target_end_index=start + 5, w_seed=1, **kwargs):
            item = next(x for x in plan['inputs'] if x['index'] == idx)
            path = parent / item['image']
            assert sha(path) == item['sha256']
            initial = wm.get_wm_latents()['message_bits_str_list'][0]
            original_rows = list(csv.DictReader((parent / ('pgid0' if idx < 5 else 'pgid5') / 'attack=forgery/all_results.csv').open()))
            reference = next(r for r in original_rows if int(r['image_idx']) == idx and r['stage'] == 'watermarked')
            assert initial == reference['message_bits_str_initial']
            source = Image.open(path).convert('RGB')
            for c, condition in enumerate(plan['conditions']):
                seed = plan['seed_base'] + idx * 100 + c
                options = SimpleNamespace(**{x['name']: None for x in plan['conditions']})
                setattr(options, condition['name'], condition['value'])
                distort.set_random_seed(seed)
                query = distort.image_distortion(source.copy(), seed, options).convert('RGB')
                for defended in [False, True]:
                    set_random_seed(seed)
                    if defended:
                        z = defense.run_defense(query)['zT']
                    else:
                        with torch.no_grad():
                            z = provider.invert_images(query, num_inference_steps=50)['zT_torch']
                    result = validate(query, wm, provider, 50, initial, provided_zT=z)
                    recovered = result['message_bits_str_recovered']
                    assert len(initial) == len(recovered) == 256
                    ba = sum(a == b for a, b in zip(initial, recovered)) / 256
                    assert ba == float(result['bit_accuracy'])
                    latent_path = output / 'latents' / f'{idx:03}_{c:02}_{int(defended)}.npz'
                    np.savez_compressed(latent_path, z=z.detach().cpu().numpy())
                    rows.append(dict(index=idx, condition=condition['name'], value=condition['value'], seed=seed,
                                     defended=defended, ba=ba, accepted=ba > 181 / 256,
                                     input_sha256=item['sha256'], pixels_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
                                     latent=str(latent_path.relative_to(run)), latent_sha256=sha(latent_path)))
                write(output / 'progress.json', dict(rows=rows))
            print('SOURCE_COMPLETE', idx, flush=True)
        assert len(rows) == 90
        summary = []
        for condition in plan['conditions']:
            for defended in [False, True]:
                group = [r for r in rows if r['condition'] == condition['name'] and r['defended'] == defended]
                assert len(group) == 5
                summary.append(dict(condition=condition['name'], defended=defended, n=5,
                                    accepted=sum(r['accepted'] for r in group), mean_ba=sum(r['ba'] for r in group) / 5))
        state.update(status='complete', rows=rows, summary=summary)
        write(output / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(output / 'state.json', state)


if __name__ == '__main__':
    main()
