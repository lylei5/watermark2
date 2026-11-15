"""SEAL500 clean evaluation, gated by full natural calibration process exit."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')
CALIBRATION = ROOT/'runs/unified_seal_nulls_20261115'
COHORT = ROOT/'runs/unified_main_sources_20261116'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    assert sha(CALIBRATION/'protocol.json') == '1a3283f12074a95cbfede7a2bb64095dfa7cb812c65c00cceb363ca34d2de20a'
    assert sha(COHORT/'protocol.json') == '58cdf0f60db1fe6a6d1b2c45f94846b5f4f01eeb55fb75a40bd960d03118ed18'
    calibrated_plan = json.loads((CALIBRATION/'protocol.json').read_text())
    assert sha(CALIBRATION/'model_manifest.json') == calibrated_plan['model_manifest_sha256']
    for name in ['seal_paper_simhash.py', 'empirical_calibration.py']:
        assert sha(CALIBRATION/name) == calibrated_plan['helper_sha256'][name]
    cohort = json.loads((COHORT/'protocol.json').read_text())
    predecessor = json.loads((CALIBRATION/'pipeline_state.json').read_text())
    pid = predecessor['pid']
    cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
    assert str(CALIBRATION).encode() in cmdline and b'--coordinate' in cmdline
    shutil.copy2(__file__, run/Path(__file__).name)
    for name in ['seal_paper_simhash.py', 'empirical_calibration.py', 'model_manifest.json']:
        shutil.copy2(CALIBRATION/name, run/name)
    plan = dict(scope='Native SEAL paper-salt system;500 common sources, clean legitimate and500 preview unwatermarked controls; no attacks or edits',
        frozen_at=datetime.now(timezone.utc).isoformat(),
        calibration_run=str(CALIBRATION), calibration_protocol_sha256=sha(CALIBRATION/'protocol.json'),
        cohort_protocol_sha256=sha(COHORT/'protocol.json'), sources=cohort['sources'],
        predecessor=dict(pid=pid, cmdline_sha256=hashlib.sha256(cmdline).hexdigest()),
        worker=Path(__file__).name,
        helper_sha256={p.name: sha(p) for p in run.iterdir() if p.is_file()},
        detector=calibrated_plan,
        generation=dict(steps=50, expected_unet_calls=51, cfg=7.5, height=512, width=512,
                        preview='One original unwatermarked preview; original caption then salted-noise generation; no iterative refinement',
                        seed='Original SEAL set_random_seed with cohort generation_seed, including its documented offsets'),
        analysis=dict(denominator='All500 for each kind; no exclusion on detection outcomes',
                      legitimate_primary='Candidate-own-caption reference only',
                      diagnostic='Compare recovered noise against original embedding noise; diagnostic only, never deployed detection',
                      thresholds='Copy both natural-calibration thresholds once after predecessor success; never tune on these500 sources',
                      intervals='95% exact Clopper-Pearson; per-stratum counts also reported'),
        gpu_indices=[7, 5, 4], no_gpu_before_predecessor_exit=True, attacks=0)
    save(run/'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run/'protocol.json'), sources=500, queued_after_pid=pid)))


def check_run(run):
    plan = json.loads((run/'protocol.json').read_text())
    assert Path(__file__).resolve() == run/plan['worker']
    for name, digest in plan['helper_sha256'].items():
        assert sha(run/name) == digest, name
    return plan


def worker(run, shard):
    plan = check_run(run)
    cfg = plan['detector']
    thresholds_path = run/'frozen_working_points.json'
    thresholds = json.loads(thresholds_path.read_text())
    assert thresholds['calibration_protocol_sha256'] == plan['calibration_protocol_sha256']
    output = run/f'shard_{shard}'
    output.mkdir(exist_ok=True)
    lock = (output/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for directory in ['records', 'images', 'features']:
        (output/directory).mkdir(exist_ok=True)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'),
                 working_points_sha256=sha(thresholds_path), completed=0)
    save(output/'state.json', state)
    started = time.monotonic()
    try:
        assignment = json.loads((run/'gpu_assignment.json').read_text())
        gpu = assignment['gpu_indices'][shard]
        assert gpu != 6
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert fields[1] == assignment['gpu_uuids'][shard] and int(fields[3]) < 100 and int(fields[4]) == 0
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(ROOT/'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                          MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
        official = Path(cfg['source'])
        for name, digest in cfg['source_sha256'].items():
            assert sha(official/name) == digest, name
        assert sha(cfg['private_salt_file']) == cfg['private_salt_sha256']
        models = json.loads((run/'model_manifest.json').read_text())
        for model in models.values():
            for name, digest in model['files'].items():
                assert sha(Path(model['root'])/name) == digest, name
        sys.path.insert(0, str(official))
        import numpy as np
        import torch
        import diffusers
        import transformers
        import PIL
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from utils import generate_caption, transform_img, calculate_patch_l2, set_random_seed
        from seal_paper_simhash import SealPaperSimHash
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pipe = InversableStableDiffusionPipeline.from_pretrained(models['diffusion_model']['root'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        assert type(pipe.scheduler).__name__ == cfg['expected_scheduler']
        processor = Blip2Processor.from_pretrained(models['caption_model']['root'], local_files_only=True)
        captioner = Blip2ForConditionalGeneration.from_pretrained(models['caption_model']['root'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        encoder = SentenceTransformer(models['embedding_model']['root'], device='cuda')
        sampler = SealPaperSimHash(bytes.fromhex(json.loads(Path(cfg['private_salt_file']).read_text())['salt']))
        nfe = [0]
        original_forward = pipe.unet.forward
        def counted_forward(*args, **kwargs):
            nfe[0] += 1
            return original_forward(*args, **kwargs)
        pipe.unet.forward = counted_forward

        def measure(fn, expected_calls=None):
            nfe[0] = 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            value = fn()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tick
            if expected_calls is not None:
                assert nfe[0] == expected_calls, (expected_calls, nfe[0])
            return value, dict(seconds=elapsed, unet_calls=nfe[0], peak_allocated_bytes=torch.cuda.max_memory_allocated())

        def describe(image):
            caption = generate_caption(image, processor, captioner)
            embedding = encoder.encode(caption, convert_to_tensor=True).to('cuda')
            reference = sampler.sample(embedding/torch.norm(embedding)).to(dtype=pipe.vae.dtype)
            return caption, reference

        def invert(image):
            with torch.no_grad():
                tensor = transform_img(image).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
                latent = pipe.get_image_latents(tensor, sample=False)
                return pipe.forward_diffusion(latents=latent, text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=cfg['inversion_cfg'], num_inference_steps=cfg['inference_steps'])

        generation = dict(num_inference_steps=50, guidance_scale=7.5, height=512, width=512)
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1], gpu_name=fields[2],
            loading_seconds=time.monotonic()-started,
            versions=dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__, numpy=np.__version__, pillow=PIL.__version__))
        save(output/'state.json', state)
        finished = []
        for item in [x for x in plan['sources'] if x['index'] % 3 == shard]:
            index = item['index']
            record_path = output/'records'/f'{index:03}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                assert record['protocol_sha256'] == state['protocol_sha256']
                assert record['working_points_sha256'] == state['working_points_sha256']
                for artifact in list(record['images'].values()) + [record['feature']]:
                    assert sha(run/artifact['path']) == artifact['sha256']
                finished.append(record)
                continue
            set_random_seed(item['generation_seed'])
            preview, cost_preview = measure(lambda: pipe(item['prompt'], **generation).images[0], 51)
            (preview_caption, initial), cost_caption_preview = measure(lambda: describe(preview), 0)
            genuine, cost_genuine = measure(lambda: pipe(item['prompt'], latents=initial, **generation).images[0], 51)
            (genuine_caption, reference), cost_caption_genuine = measure(lambda: describe(genuine), 0)
            recovered, cost_inverse_genuine = measure(lambda: invert(genuine), 51)
            preview_recovered, cost_inverse_preview = measure(lambda: invert(preview), 51)
            arrays = dict(initial=initial.cpu().numpy(), query_reference=reference.cpu().numpy(),
                          recovered=recovered.cpu().numpy(), preview_recovered=preview_recovered.cpu().numpy())
            for value in arrays.values():
                assert value.shape == (1, 4, 64, 64) and np.isfinite(value).all()
            arrays.update(genuine_distances=np.asarray(calculate_patch_l2(recovered, reference, 1024)),
                          preview_null_distances=np.asarray(calculate_patch_l2(preview_recovered, initial, 1024)),
                          diagnostic_original_noise_distances=np.asarray(calculate_patch_l2(recovered, initial, 1024)),
                          reference_drift_distances=np.asarray(calculate_patch_l2(initial, reference, 1024)))
            for name in ['genuine_distances', 'preview_null_distances', 'diagnostic_original_noise_distances', 'reference_drift_distances']:
                assert arrays[name].shape == (1024,) and np.isfinite(arrays[name]).all()
            images = {}
            for kind, image in [('preview_null', preview), ('genuine', genuine)]:
                destination = output/'images'/f'{index:03}_{kind}.png'
                image.save(destination)
                images[kind] = dict(path=str(destination.relative_to(run)), sha256=sha(destination),
                                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            feature = output/'features'/f'{index:03}.npz'
            temporary = feature.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(feature)
            scores = {kind: int((arrays[name] < cfg['patch_distance_threshold']).sum()) for kind, name in
                [('genuine', 'genuine_distances'), ('preview_null', 'preview_null_distances'),
                 ('diagnostic_original_noise', 'diagnostic_original_noise_distances')]}
            record = dict(index=index, dataset=item['dataset'], source_id=item['source_id'],
                protocol_sha256=state['protocol_sha256'], working_points_sha256=state['working_points_sha256'],
                captions=dict(preview=preview_caption, genuine=genuine_caption), images=images,
                feature=dict(path=str(feature.relative_to(run)), sha256=sha(feature)), matches=scores,
                identical_reference_patches=int((arrays['reference_drift_distances'] == 0).sum()),
                costs=dict(preview_generation=cost_preview, genuine_generation=cost_genuine,
                           preview_caption=cost_caption_preview, genuine_caption=cost_caption_genuine,
                           genuine_inverse=cost_inverse_genuine, preview_inverse=cost_inverse_preview))
            save(record_path, record)
            finished.append(record)
            state['completed'] = len(finished)
        assert len(finished) == len([x for x in plan['sources'] if x['index'] % 3 == shard])
        save(output/'rows.json', finished)
        state.update(status='complete', completed=len(finished), seconds_this_session=time.monotonic()-started,
                     rows_sha256=sha(output/'rows.json'))
        save(output/'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(output/'state.json', state)


def summarize(run):
    from empirical_calibration import exact_binomial_interval
    plan = check_run(run)
    rules = json.loads((run/'frozen_working_points.json').read_text())
    records = []
    for shard in range(3):
        directory = run/f'shard_{shard}'
        receipt = json.loads((directory/'complete.json').read_text())
        assert receipt['status'] == 'complete' and receipt['rows_sha256'] == sha(directory/'rows.json')
        records.extend(json.loads((directory/'rows.json').read_text()))
    assert sorted(x['index'] for x in records) == list(range(500))
    for row in records:
        assert row['protocol_sha256'] == sha(run/'protocol.json')
        assert row['working_points_sha256'] == sha(run/'frozen_working_points.json')
        for artifact in list(row['images'].values()) + [row['feature']]:
            assert sha(run/artifact['path']) == artifact['sha256']
        assert row['dataset'] == plan['sources'][row['index']]['dataset']
        assert row['source_id'] == plan['sources'][row['index']]['source_id']
        for stage, cost in row['costs'].items():
            assert cost['unet_calls'] == (0 if 'caption' in stage else 51)
    working_points = [dict(name=f"natural_fpr_{x['target_fpr']}", threshold=x['threshold']) for x in rules['calibration_results']['working_points']]
    working_points.append(dict(name='paper_threshold12', threshold=12))
    rows, diagnostics = [], []
    for point in working_points:
        for dataset in ['all', 'COCO', 'SDP']:
            subset = [r for r in records if dataset == 'all' or r['dataset'] == dataset]
            for kind in ['genuine', 'preview_null']:
                n = len(subset)
                accepted = sum(r['matches'][kind] >= point['threshold'] for r in subset)
                lower, upper = exact_binomial_interval(accepted, n)
                rows.append(dict(system='SEAL paper-salt native', dataset=dataset, kind=kind, working_point=point['name'],
                    threshold=point['threshold'], accepted=accepted, sources=n, rate=accepted/n, lower95=lower, upper95=upper))
            failures = [r for r in subset if r['matches']['genuine'] < point['threshold']]
            diagnostics.append(dict(working_point=point['name'], dataset=dataset, genuine_failures=len(failures),
                failure_with_original_noise_pass=sum(r['matches']['diagnostic_original_noise'] >= point['threshold'] for r in failures),
                failure_with_original_noise_fail=sum(r['matches']['diagnostic_original_noise'] < point['threshold'] for r in failures),
                interpretation='Descriptive decomposition only; initial-noise reference is unavailable to the deployed stateless detector'))
    save(run/'clean_results.json', dict(status='complete', protocol_sha256=sha(run/'protocol.json'),
         working_points_sha256=sha(run/'frozen_working_points.json'), rows=rows, diagnostics=diagnostics,
         full_budget_generations=1000, full_budget_inversions=1000, expected_total_unet_calls=204000,
         actual_total_unet_calls=sum(c['unet_calls'] for r in records for c in r['costs'].values()), attacks=0))
    with (run/'clean_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coordinate(run):
    plan = check_run(run)
    lock = (run/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run/'pipeline_state.json').exists()
    state = dict(status='waiting_for_calibration_exit', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'), children=[])
    save(run/'pipeline_state.json', state)
    try:
        prior = plan['predecessor']
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.syscall(434, prior['pid'], 0)
        if descriptor < 0:
            if ctypes.get_errno() != 3:
                raise OSError(ctypes.get_errno(), 'pidfd_open')
        else:
            try:
                path = Path(f"/proc/{prior['pid']}/cmdline")
                if path.exists():
                    command = path.read_bytes()
                    assert not command or hashlib.sha256(command).hexdigest() == prior['cmdline_sha256']
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
        previous_run = Path(plan['calibration_run'])
        assert sha(previous_run/'protocol.json') == plan['calibration_protocol_sha256']
        previous_state = json.loads((previous_run/'pipeline_state.json').read_text())
        result = json.loads((previous_run/'calibration_results.json').read_text())
        assert previous_state['status'] == result['status'] == 'complete'
        assert result['protocol_sha256'] == plan['calibration_protocol_sha256']
        assert {x['target_fpr'] for x in result['working_points']} == {.01, .001}
        for point in result['working_points']:
            assert point['comparator'] == '>=' and point['validation_n'] == 20000
        assert not (run/'frozen_working_points.json').exists()
        save(run/'frozen_working_points.json', dict(calibration_protocol_sha256=plan['calibration_protocol_sha256'],
            calibration_results_sha256=sha(previous_run/'calibration_results.json'), calibration_results=result,
            policy='Copied before evaluation; no retuning based on evaluation or validation outcomes'))
        time.sleep(15)  # One GPU context-release interval; no progress polling.
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        idle = {}
        for line in inventory.splitlines():
            index, uuid, memory, util = [x.strip() for x in line.split(',')]
            if int(index) != 6 and int(memory) < 100 and int(util) == 0:
                idle[int(index)] = uuid
        available = [i for i in [7, 5, 4, 3, 2, 1, 0] if i in idle]
        assert len(available) >= 3, 'Require three currently idle permitted GPUs; never preempt a job'
        save(run/'gpu_assignment.json', dict(gpu_indices=available[:3], gpu_uuids=[idle[i] for i in available[:3]]))
        children = []
        for shard in range(3):
            log = (run/f'shard_{shard}.log').open('x')
            child = subprocess.Popen([plan['detector']['environment'], str(run/plan['worker']), '--run', str(run), '--shard', str(shard)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            children.append(child)
        state.update(status='running', children=[c.pid for c in children])
        save(run/'pipeline_state.json', state)
        codes = [child.wait() for child in children]
        state['exit_codes'] = codes
        assert all(code == 0 for code in codes), codes
        summarize(run)
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(run/'pipeline_state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--coordinate', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--shard', type=int, choices=range(3))
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT/'runs'
    if args.prepare:
        prepare(run)
    elif args.coordinate:
        coordinate(run)
    elif args.summarize:
        summarize(run)
    else:
        assert args.shard is not None
        worker(run, args.shard)
