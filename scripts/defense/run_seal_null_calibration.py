"""Full SEAL natural-image calibration and independent FPR validation on249."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(run):
    assert run.parent == ROOT / 'runs'
    run.mkdir(exist_ok=False)
    data = ROOT / 'datasets/unified_coco_nulls_20261115'
    receipt = json.loads((data / 'complete.json').read_text())
    assert receipt['status'] == 'complete' and sha(data / 'manifest.json') == receipt['manifest_sha256']
    cfg = json.loads((ROOT / 'protocols/seal_paper_native_20261113.json').read_text())
    source = Path(cfg['source'])
    for name, digest in cfg['source_sha256'].items():
        assert sha(source / name) == digest
    private = ROOT / 'private_eval' / run.name
    private.mkdir(mode=0o700, exist_ok=False)
    salt = private / 'salt.json'
    salt.write_text(json.dumps({'salt': secrets.token_hex(32)}) + '\n')
    salt.chmod(0o600)
    helpers = ['run_seal_null_calibration.py', 'seal_paper_simhash.py',
               'empirical_calibration.py', 'prepare_coco_nulls.py']
    for name in helpers:
        shutil.copy2(Path(__file__).parent / name, run / name)
    models = {}
    for name in ['diffusion_model', 'caption_model', 'embedding_model']:
        directory = Path(cfg[name])
        models[name] = dict(root=str(directory), files={str(p.relative_to(directory)): sha(p)
                            for p in sorted(directory.rglob('*')) if p.is_file()})
    save(run / 'model_manifest.json', models)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='SEAL paper-salt semantic detector; 20000 natural calibration and 20000 disjoint natural validation; no attacks',
        worker='run_seal_null_calibration.py', helper_sha256={name: sha(run / name) for name in helpers},
        data_manifest=str(data / 'manifest.json'), data_manifest_sha256=sha(data / 'manifest.json'),
        data_receipt_sha256=sha(data / 'complete.json'), source=str(source), source_sha256=cfg['source_sha256'],
        source_commit=cfg['source_commit'], model_manifest_sha256=sha(run / 'model_manifest.json'),
        private_salt_file=str(salt), private_salt_sha256=sha(salt),
        key_setting='one randomly sampled provider salt, fixed before image inference; conditional FPR for this identity',
        target_fpr=[.01, .001], patch_distance_threshold=2.3, paper_minimum_matches=12,
        patches=1024, bits_per_patch=7, inference_steps=50, expected_unet_calls=51, inversion_cfg=1,
        expected_scheduler='PNDMScheduler', dtype='float16', tf32=False,
        preprocessing=receipt['preprocessing'], gpu_indices=[7, 5, 4],
        environment=str(ROOT / 'envs/seal_reproduction_py310/bin/python'),
        numerical_scope='Original SEAL inversion/caption/embedding; project paper-salt sampler with declared SHA256/PRNG convention',
        threshold_policy='Calibrate >= matching-block count on calibration only; conservative ties; validate once on held-out natural images',
        main_table_complete=False)
    save(run / 'protocol.json', plan)
    print(json.dumps({'prepared': str(run), 'protocol_sha256': sha(run / 'protocol.json')}), flush=True)


def worker(run, shard):
    plan = json.loads((run / 'protocol.json').read_text())
    assert Path(__file__).resolve() == run / plan['worker']
    for name, digest in plan['helper_sha256'].items():
        assert sha(run / name) == digest
    assert sha(plan['data_manifest']) == plan['data_manifest_sha256']
    assert sha(plan['private_salt_file']) == plan['private_salt_sha256']
    source = Path(plan['source'])
    for name, digest in plan['source_sha256'].items():
        assert sha(source / name) == digest
    output = run / f'shard_{shard}'
    output.mkdir(exist_ok=True)
    lock = (output / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    features = output / 'features'
    features.mkdir(exist_ok=True)
    record_dir = output / 'records'
    record_dir.mkdir(exist_ok=True)
    source_rows = json.loads(Path(plan['data_manifest']).read_text())
    source_rows = [row for row in source_rows if row['index'] % 3 == shard]
    by_index = {row['index']: row for row in source_rows}
    finished = {}
    for path in sorted(record_dir.glob('*.json')):
        record = json.loads(path.read_text())
        row = by_index[record['index']]
        assert record['index'] not in finished
        assert record['protocol_sha256'] == sha(run / 'protocol.json')
        assert record['input_file_sha256'] == row['file_sha256']
        assert record['input_pixel_sha256'] == row['pixel_sha256']
        assert record['split'] == row['split'] and record['source_id'] == row['source_id']
        artifact = run / record['artifact']
        assert artifact.parent == features and sha(artifact) == record['artifact_sha256']
        finished[record['index']] = record
    state = dict(status='preflight', pid=os.getpid(), shard=shard, protocol_sha256=sha(run / 'protocol.json'))
    save(output / 'state.json', state)
    completed = len(finished)
    start = time.monotonic()
    try:
        assignment = json.loads((run / 'gpu_assignment.json').read_text())
        gpu = assignment['gpu_indices'][shard]
        assert gpu != 6
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == gpu and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        assert fields[1] == assignment['gpu_uuids'][shard]
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(ROOT / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                          MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        import diffusers
        import transformers
        import PIL
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from utils import generate_caption, transform_img, calculate_patch_l2
        from seal_paper_simhash import SealPaperSimHash
        from prepare_coco_nulls import common_image
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert sha(run / 'model_manifest.json') == plan['model_manifest_sha256']
        models = json.loads((run / 'model_manifest.json').read_text())
        for model in models.values():
            for name, digest in model['files'].items():
                assert sha(Path(model['root']) / name) == digest, name
        pipe = InversableStableDiffusionPipeline.from_pretrained(models['diffusion_model']['root'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        assert type(pipe.scheduler).__name__ == plan['expected_scheduler']
        nfe = [0]
        original_forward = pipe.unet.forward
        def counted_forward(*args, **kwargs):
            nfe[0] += 1
            return original_forward(*args, **kwargs)
        pipe.unet.forward = counted_forward
        processor = Blip2Processor.from_pretrained(models['caption_model']['root'], local_files_only=True)
        captioner = Blip2ForConditionalGeneration.from_pretrained(models['caption_model']['root'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        encoder = SentenceTransformer(models['embedding_model']['root'], device='cuda')
        sampler = SealPaperSimHash(bytes.fromhex(json.loads(Path(plan['private_salt_file']).read_text())['salt']))
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1], expected_rows=len(source_rows),
                     versions=dict(python=sys.version, torch=torch.__version__, diffusers=diffusers.__version__,
                                   transformers=transformers.__version__, pillow=PIL.__version__, numpy=np.__version__),
                     resumed_rows=completed, loading_seconds=time.monotonic()-start)
        save(output / 'state.json', state)
        with (output / 'events.jsonl').open('a') as log:
            for row in source_rows:
                if row['index'] in finished:
                    continue
                assert sha(row['path']) == row['file_sha256']
                with Image.open(row['path']) as source_image:
                    image = common_image(source_image)
                assert hashlib.sha256(image.tobytes()).hexdigest() == row['pixel_sha256']
                torch.manual_seed(202611150000 + row['index'])
                np.random.seed(20261115 + row['index'])
                nfe[0] = 0
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.monotonic()
                caption = generate_caption(image, processor, captioner)
                embedding = encoder.encode(caption, convert_to_tensor=True).to('cuda')
                reference = sampler.sample(embedding / torch.norm(embedding)).to(dtype=pipe.vae.dtype)
                with torch.no_grad():
                    tensor = transform_img(image).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
                    latent = pipe.get_image_latents(tensor, sample=False)
                    recovered = pipe.forward_diffusion(latents=latent, text_embeddings=pipe.get_text_embedding(''),
                        guidance_scale=plan['inversion_cfg'], num_inference_steps=plan['inference_steps'])
                assert recovered.shape == (1, 4, 64, 64) and torch.isfinite(recovered).all()
                assert nfe[0] == plan['expected_unet_calls'], nfe[0]
                distances = np.asarray(calculate_patch_l2(recovered, reference, plan['patches']))
                assert distances.shape == (1024,) and np.isfinite(distances).all()
                matches = int((distances < plan['patch_distance_threshold']).sum())
                torch.cuda.synchronize()
                elapsed = time.monotonic()-tick
                artifact = features / f"{row['index']:05}.npz"
                temporary = artifact.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, recovered=recovered.cpu().numpy(),
                                        reference=reference.cpu().numpy(), distances=distances)
                temporary.replace(artifact)
                record = dict(index=row['index'], source_id=row['source_id'], split=row['split'],
                    protocol_sha256=sha(run / 'protocol.json'),
                    input_file_sha256=row['file_sha256'], input_pixel_sha256=row['pixel_sha256'],
                    caption=caption, matches=matches, paper_accepted=matches >= plan['paper_minimum_matches'],
                    unet_calls=nfe[0], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    seconds=elapsed, artifact=str(artifact.relative_to(run)), artifact_sha256=sha(artifact))
                save(record_dir / f"{row['index']:05}.json", record)
                finished[row['index']] = record
                log.write(json.dumps(record) + '\n')
                log.flush()
                completed += 1
                if completed % 100 == 0:
                    save(output / 'progress.json', dict(completed=completed, expected=len(source_rows)))
        assert sorted(finished) == sorted(by_index)
        (output / 'rows.jsonl').write_text(''.join(json.dumps(finished[i]) + '\n' for i in sorted(finished)))
        state.update(status='complete', completed=completed, seconds_this_session=time.monotonic()-start,
                     rows_sha256=sha(output / 'rows.jsonl'))
        save(output / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', completed=completed, error=repr(exc))
        raise
    finally:
        save(output / 'state.json', state)


def summarize(run):
    from empirical_calibration import calibrate_upper_tail, exact_binomial_interval
    plan = json.loads((run / 'protocol.json').read_text())
    records = []
    for shard in range(3):
        output = run / f'shard_{shard}'
        receipt = json.loads((output / 'complete.json').read_text())
        assert receipt['status'] == 'complete' and receipt['protocol_sha256'] == sha(run / 'protocol.json')
        assert sha(output / 'rows.jsonl') == receipt['rows_sha256']
        records.extend(json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines())
    assert sorted(row['index'] for row in records) == list(range(40000))
    calibration = [row['matches'] for row in records if row['split'] == 'calibration']
    validation = [row['matches'] for row in records if row['split'] == 'validation']
    assert len(calibration) == len(validation) == 20000
    results = []
    for target in plan['target_fpr']:
        rule = calibrate_upper_tail(calibration, target, minimum_threshold=0)
        assert rule['status'] == 'calibrated'
        threshold = math.ceil(rule['threshold'])
        accepted = sum(score >= threshold for score in validation)
        results.append(dict(target_fpr=target, threshold=threshold, comparator='>=', calibration=rule,
                            validation_accepted=accepted, validation_n=len(validation),
                            validation_fpr=accepted/len(validation), validation_95ci=exact_binomial_interval(accepted, len(validation))))
    paper_accepted = sum(score >= plan['paper_minimum_matches'] for score in validation)
    save(run / 'calibration_results.json', dict(status='complete', protocol_sha256=sha(run / 'protocol.json'),
        scope='SEAL semantic natural-image FPR only; no legitimate TPR or attacks yet', working_points=results,
        paper_working_point=dict(threshold=plan['paper_minimum_matches'], accepted=paper_accepted, n=len(validation),
                                confidence_interval=exact_binomial_interval(paper_accepted, len(validation)))))


def coordinate(run):
    lock = (run / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((run / 'protocol.json').read_text())
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
    idle = {}
    for line in inventory.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if int(fields[0]) != 6 and int(fields[2]) < 100 and int(fields[3]) == 0:
            idle[int(fields[0])] = fields[1]
    available = [gpu for gpu in [7, 5, 4, 3, 2, 1, 0] if gpu in idle]
    if len(available) < 3:
        raise RuntimeError('Three idle permitted GPUs are required for this fixed three-shard launch')
    save(run / 'gpu_assignment.json', dict(gpu_indices=available[:3], gpu_uuids=[idle[gpu] for gpu in available[:3]]))
    children = []
    state = dict(status='launching', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    save(run / 'pipeline_state.json', state)
    for shard in range(3):
        if (run / f'shard_{shard}/complete.json').exists():
            continue
        log = (run / f'shard_{shard}.log').open('a')
        child = subprocess.Popen([plan['environment'], str(run / plan['worker']), '--run', str(run), '--shard', str(shard)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        children.append(child)
    state.update(status='running', children=[p.pid for p in children])
    save(run / 'pipeline_state.json', state)
    # Blocking process exits, no progress polling.
    codes = [child.wait() for child in children]
    state['exit_codes'] = codes
    if all(code == 0 for code in codes):
        summarize(run)
        state['status'] = 'complete'
    else:
        state['status'] = 'failed'
    save(run / 'pipeline_state.json', state)
    if state['status'] != 'complete':
        raise RuntimeError(f'One or more extraction shards failed: {codes}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--coordinate', action='store_true')
    parser.add_argument('--shard', type=int, choices=range(3))
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / 'runs'
    if args.prepare:
        prepare(run)
    elif args.coordinate:
        coordinate(run)
    else:
        worker(run, args.shard)
