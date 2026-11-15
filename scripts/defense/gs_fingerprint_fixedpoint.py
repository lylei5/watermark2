"""Bounded 256-bit GS fixed-point feasibility experiment, server 249 only."""
import argparse
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
PYTHON = ROOT / 'envs/semantic_forgery_py310/bin/python'
RUN_NAME = 'gs_fingerprint_fixedpoint_20261116'
ASSETS = ROOT / 'runs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def material(private):
    import numpy as np
    key = np.packbits(private['key'].reshape(-1)).tobytes()
    pad = np.unpackbits(np.frombuffer(hmac.digest(key, b'gs-fingerprint-pad-v1', 'sha256'), dtype=np.uint8))
    seed = int.from_bytes(hmac.digest(key, b'gs-fingerprint-positions-v1', 'sha256')[:16], 'big')
    positions = np.random.Generator(np.random.PCG64(seed)).permutation(16384)
    return pad, positions


def embed(base, bits, pad, positions, count, width):
    import numpy as np
    from scipy.special import ndtr, ndtri
    output = base.copy().reshape(-1)
    where = positions[:count]
    slots = np.resize(np.asarray(bits, dtype=np.uint8) ^ pad, count * width).reshape(count, width)
    magnitude = np.abs(output[where].astype(np.float64))
    if width == 2:
        residual = np.mod(2 * (2 * ndtr(magnitude) - 1), 1)
        magnitude = ndtri(.5 + .25 * (slots[:, 1] + residual))
        # Ensure the requested Gaussian quartile survives native float16 storage.
        cut = ndtri(.75)
        low = np.float16(cut)
        if low >= cut:
            low = np.nextafter(low, np.float16(-np.inf))
        high = np.nextafter(low, np.float16(np.inf))
        magnitude = np.where(slots[:, 1] == 0, np.minimum(magnitude, low), np.maximum(magnitude, high))
    magnitude = np.maximum(magnitude, np.finfo(np.float16).smallest_subnormal)
    output[where] = ((2 * slots[:, 0].astype(float) - 1) * magnitude).astype(base.dtype)
    return output.reshape(base.shape)


def decode(latent, pad, positions, count, width):
    import numpy as np
    from scipy.special import ndtri
    values = np.asarray(latent).reshape(-1)[positions[:count]].astype(np.float64)
    slots = np.column_stack([values > 0, np.abs(values) >= ndtri(.75)])[:, :width].reshape(-1)
    indices = np.arange(len(slots)) % 256
    votes = np.bincount(indices, weights=slots, minlength=256)
    totals = np.bincount(indices, minlength=256)
    return (votes > totals / 2).astype(np.uint8) ^ pad


def asym(bits, feature, projection):
    import numpy as np
    projected = feature.astype(np.float64) @ projection
    return float((2 * bits.astype(float) - 1) @ projected / (16 * np.linalg.norm(projected)))


def prepare(run):
    import numpy as np
    run.mkdir(exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    old = ASSETS / 'sscd_public_binary_20261116'
    shutil.copy2(old / 'public_projection.npy', run / 'public_projection.npy')
    source = ASSETS / 'phase1_trajectory_native_gs_20261113'
    split = read(ASSETS / 'instance_representation_20261113_v2/protocol.json')
    sources = [row for row in read(source / 'protocol.json')['sources'] if row['task'] in split['evaluation']]
    assert len(sources) == 50
    key = np.unpackbits(np.frombuffer(os.urandom(2048), dtype=np.uint8)).reshape(1, 4, 64, 64)
    watermark = np.unpackbits(np.frombuffer(os.urandom(32), dtype=np.uint8)).reshape(1, 4, 8, 8)
    prior = ASSETS / RUN_NAME
    prior_seconds = 0.
    if run != prior:
        for attempt in sorted(ASSETS.glob(RUN_NAME + '*')):
            if attempt == run:
                continue
            receipt = read(attempt / 'supervisor_complete.json')
            assert receipt['exit_code'] != 0 and not (attempt / 'images').exists()
            prior_seconds += receipt['allocated_gpu_seconds_upper_bound']
        original_private = np.load(prior / 'private_material.npz')
        key, watermark = original_private['key'], original_private['watermark']
    np.savez(run / 'private_material.npz', key=key, watermark=watermark)
    os.chmod(run / 'private_material.npz', 0o600)
    private = dict(key=key, watermark=watermark)
    pad, positions = material(private)
    ratios = [dict(name='p01', count=163, width=2), dict(name='p03', count=491, width=1), dict(name='p06', count=983, width=1)]
    checks = []
    rng = np.random.default_rng(2026111601)
    for ratio in ratios:
        for i in range(50):
            base = rng.standard_normal((1, 4, 64, 64)).astype(np.float16)
            bits = rng.integers(0, 2, 256, dtype=np.uint8)
            z = embed(base, bits, pad, positions, ratio['count'], ratio['width'])
            assert np.array_equal(decode(z, pad, positions, ratio['count'], ratio['width']), bits)
            outside = np.ones(16384, dtype=bool)
            outside[positions[:ratio['count']]] = False
            assert np.array_equal(base.reshape(-1)[outside], z.reshape(-1)[outside])
        checks.append(dict(ratio=ratio['name'], roundtrips=50, failures=0))
    files = [source / 'gs_watermark.py', source / 'protocol.json',
             ASSETS / 'instance_representation_20261113_v2/protocol.json',
             ASSETS / 'defense_revalidation_20261111/gs_image_utils.py',
             ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt',
             old / 'thresholds.json', run / 'public_projection.npy', run / 'private_material.npz']
    provider = ROOT / 'third_party/frozen/PGID_dc4ad96'
    files.extend(provider / 'pipe' / name for name in ['base_pipe.py', 'SD_pipe.py', 'pipe_utils.py'])
    tau = .29069975629559697
    plan = dict(run=run.name, scope='Authorized bounded defense mechanism feasibility; no attacks',
        prior_launch_seconds_upper_bound=prior_seconds,
        worker_sha256=sha(__file__), input_sha256={str(p): sha(p) for p in files},
        sources=sources, ratios=ratios, conditions=split['conditions'],
        gpu_index=3, excluded_gpu_indices=[6], gpu_seconds_limit=10800,
        watchdog_worker_timeout_seconds=10620, worker_soft_limit_seconds=10440,
        generation=dict(steps=50, cfg=7.5, dtype='float16', resolution=512),
        inversion=dict(steps=50, cfg=1, prompt='', backend='PGID original plain DDIM; no defense, no VAE optimization'),
        model_path=str(ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'),
        sscd_model=str(ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'),
        extra_dependencies=str(ROOT / 'envs/phase1_extra'),
        provider=str(provider), native_gs=str(source / 'gs_watermark.py'),
        distortions=str(ASSETS / 'defense_revalidation_20261111/gs_image_utils.py'),
        public_projection_sha256=sha(run / 'public_projection.npy'),
        fingerprint='256 signs of frozen public projection; zero positive; float64 scoring',
        gs_accept='original GS BA > 181/256', similarity_threshold=tau,
        drift='1-asymmetric(written plaintext code, generated continuous SSCD); also report Hamming and SSCD cosine drift',
        drift_limit=(1-tau)/3, rounds=3, convergence='All256 bits equal to written plaintext in a round',
        iteration='Each ratio starts from baseline fingerprint; overwrite same original latent coordinates each round',
        encryption='GS-style keyed XOR: HMAC-SHA256( packed native GS OTP key, domain ) yields256-bit fixed pad; deterministic masking, no authentication or semantic-security claim',
        locations='HMAC-derived PCG64 permutation, fixed across50 prompts; prefix by ratio; provider key only',
        coding='p01: sign plus magnitude-half Gaussian quartile; p03/p06: sign only; cyclic repetitions, majority ties ciphertext0; no ECC',
        private_material='One provider GS OTP key and watermark reused; no per-image enrolled fingerprint at verification',
        distortion_seed_base=2026111300, generation_seed_base=2026111600,
        pass_rule=dict(converged_and_drift_at_most_round3=45, denominator=50,
                      joint_nine_edits_min=405, joint_nine_edits_total=450, joint_clean_min=45,
                      gs_no_drop='At every clean/individual edit condition, accepted count must be at least paired native GS baseline'),
        denominator='All50 prompts for every condition, including nonconverged outputs; clean separate from450 edits',
        visual_tasks=[1, 2, 6, 52, 53, 55],
        stop='No mechanism changes or attacks after this bounded experiment, irrespective of result',
        cpu_codec_checks=checks, new_attacks=0, main_benchmark_priority=True,
        gpu_concurrency_exception='One idle GPU3 for this user-authorized bounded probe while SEAL retains GPUs7/5/4')
    assert Path(plan['model_path']).is_dir()
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(prepared=str(run), protocol_sha256=sha(run / 'protocol.json'), cpu_codec_checks=checks)), flush=True)


def worker(run):
    plan = read(run / 'protocol.json')
    assert sha(__file__) == plan['worker_sha256']
    for path, digest in plan['input_sha256'].items():
        assert sha(path) == digest, path
    info = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu_index']), '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert int(info[0]) == 3 and int(info[3]) < 100 and int(info[4]) == 0, info
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT / 'cache/huggingface'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    import torch.nn.functional as F
    import diffusers
    import torchvision
    from PIL import Image
    from torchvision import transforms
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, plan['provider'])
    sys.path.insert(0, plan['extra_dependencies'])
    from pipe import pipe_utils
    provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base', resolution=512,
        device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM', dtype=torch.float16,
        eager_loading=True, disable_tqdm=True)
    provider.pipe.set_progress_bar_config(disable=True)
    model = torch.jit.load(plan['sscd_model'], map_location='cuda').eval()
    transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    native = load_module('native_gs', plan['native_gs']).Gaussian_Shading(1, 8, 1e-6, 1)
    private = np.load(run / 'private_material.npz')
    native.key = torch.from_numpy(private['key'].astype(np.int64)).cuda()
    native.watermark = torch.from_numpy(private['watermark'].astype(np.int64)).cuda()
    message = ((native.watermark.repeat(1, 1, 8, 8) + native.key) % 2).reshape(-1).cpu().numpy()
    pad, positions = material(private)
    projection = np.load(run / 'public_projection.npy')
    distortion = load_module('original_distortion', plan['distortions'])
    nfe = [0]
    provider.pipe.unet.register_forward_hook(lambda *unused: nfe.__setitem__(0, nfe[0] + 1))
    work = run / 'work'
    work.mkdir(exist_ok=False)
    os.chdir(work)
    write(run / 'environment.json', dict(gpu=info, pid=os.getpid(), torch=torch.__version__,
          diffusers=diffusers.__version__, torchvision=torchvision.__version__, dtype='float16', tf32=False))

    def budget():
        if time.monotonic() - started > plan['worker_soft_limit_seconds']:
            raise TimeoutError('Frozen GPU budget reached; incomplete results cannot pass')

    def feature(image):
        value = F.normalize(model(transform(image).unsqueeze(0).cuda()), dim=1)[0].float().cpu().numpy()
        assert np.isfinite(value).all()
        return value

    def code(value):
        return (value.astype(np.float64) @ projection >= 0).astype(np.uint8)

    def generate(z, prompt, seed):
        budget()
        torch.manual_seed(seed)
        before, begin = nfe[0], time.monotonic()
        image = provider.generate([prompt], num_inference_steps=50, guidance_scale=7.5,
                                  latents=torch.from_numpy(z).cuda(), return_latents=False)['images_PIL'][0].convert('RGB')
        torch.cuda.synchronize()
        assert nfe[0] - before == 50
        return image, time.monotonic() - begin

    rounds, finals = [], []
    with torch.no_grad():
        for source in plan['sources']:
            task = source['task']
            folder = run / 'images' / str(task)
            folder.mkdir(parents=True)
            np.random.seed(plan['generation_seed_base'] + task)
            base = native.truncSampling(message).cpu().numpy()
            assert native.eval_watermark(torch.from_numpy(base).cuda()) == 1.
            baseline, seconds = generate(base, source['caption'], plan['generation_seed_base'] + task)
            baseline.save(folder / 'baseline.png')
            base_feature = feature(baseline)
            np.savez(folder / 'baseline.npz', initial_latent=base, feature=base_feature, fingerprint=code(base_feature))
            finals.append(dict(task=task, ratio='baseline', rounds=0, converged=True, generation_seconds=seconds))
            for ratio in plan['ratios']:
                bits, previous = code(base_feature), base_feature
                for iteration in range(1, 4):
                    z = embed(base, bits, pad, positions, ratio['count'], ratio['width'])
                    assert np.array_equal(decode(z, pad, positions, ratio['count'], ratio['width']), bits)
                    initial_ba = native.eval_watermark(torch.from_numpy(z).cuda())
                    image, seconds = generate(z, source['caption'], plan['generation_seed_base'] + task)
                    value = feature(image)
                    new_bits = code(value)
                    drift = 1 - asym(bits, value, projection)
                    row = dict(task=task, ratio=ratio['name'], iteration=iteration, flips=int(np.sum(bits != new_bits)),
                        converged=bool(np.array_equal(bits, new_bits)), drift=drift,
                        strict_drift_pass=bool(drift < plan['drift_limit']),
                        previous_sscd_distance=float(1 - previous.astype(float) @ value.astype(float)),
                        baseline_sscd_distance=float(1 - base_feature.astype(float) @ value.astype(float)),
                        initial_gs_ba=initial_ba, generation_seconds=seconds,
                        changed_coordinates=int(np.sum(base != z)), unet_calls=50)
                    stem = folder / (ratio['name'] + '_r' + str(iteration))
                    image.save(stem.with_suffix('.png'))
                    np.savez(stem.with_suffix('.npz'), initial_latent=z, feature=value, written_bits=bits, fingerprint=new_bits)
                    row['image_sha256'] = sha(stem.with_suffix('.png'))
                    rounds.append(row)
                    write(run / 'rounds.json', rounds)
                    if row['converged']:
                        break
                    bits, previous = new_bits, value
                finals.append(dict(row, rounds=iteration, image=str(stem.with_suffix('.png')), latent=str(stem.with_suffix('.npz'))))
            write(run / 'finals.json', finals)
        write(run / 'generation_complete.json', dict(sources=50, generation_calls=50+len(rounds), unet_calls=nfe[0], seconds=time.monotonic()-started))
        print(json.dumps(dict(event='generation_complete', calls=50+len(rounds), seconds=time.monotonic()-started,
            converged={r['name']: sum(f['converged'] for f in finals if f['ratio']==r['name']) for r in plan['ratios']})), flush=True)
        results = []
        configs = {r['name']: r for r in plan['ratios']}
        for source in plan['sources']:
            task = source['task']
            for final in [r for r in finals if r['task'] == task]:
                name = final['ratio']
                folder = run / 'queries' / str(task) / name
                folder.mkdir(parents=True)
                if name == 'baseline':
                    image_path = run / 'images' / str(task) / 'baseline.png'
                    written = None
                else:
                    image_path = Path(final['image'])
                    written = np.load(final['latent'])['written_bits']
                original = Image.open(image_path).convert('RGB')
                for i, condition in enumerate(['clean', *plan['conditions']]):
                    budget()
                    seed = plan['distortion_seed_base'] + task * 100 + i - 1
                    distortion.set_random_seed(seed)
                    query = original.copy()
                    if condition != 'clean':
                        args = SimpleNamespace(**{k: None for k in plan['conditions']})
                        setattr(args, condition, plan['conditions'][condition])
                        query = distortion.image_distortion(query, seed, args).convert('RGB')
                    query.save(folder / (condition + '.png'))
                    before, begin = nfe[0], time.monotonic()
                    z = provider.invert_images(query, num_inference_steps=50)['zT_torch']
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - begin
                    assert nfe[0] - before == 50
                    ba = native.eval_watermark(z)
                    z_numpy = z.float().cpu().numpy()
                    value = feature(query)
                    row = dict(task=task, ratio=name, condition=condition, gs_ba=ba, gs_accept=ba > 181/256,
                               inversion_seconds=elapsed, unet_calls=50, image_sha256=sha(folder / (condition + '.png')))
                    arrays = dict(inverted_latent=z_numpy, feature=value)
                    if name != 'baseline':
                        cfg = configs[name]
                        recovered = decode(z_numpy, pad, positions, cfg['count'], cfg['width'])
                        similarity = asym(recovered, value, projection)
                        row.update(bit_errors=int(np.sum(recovered != written)), similarity=similarity,
                                   fingerprint_accept=similarity >= plan['similarity_threshold'],
                                   joint=bool(ba > 181/256 and similarity >= plan['similarity_threshold']),
                                   converged=final['converged'])
                        arrays.update(recovered_bits=recovered, written_bits=written)
                    np.savez(folder / (condition + '.npz'), **arrays)
                    results.append(row)
                write(run / 'queries.json', results)
        write(run / 'gpu_complete.json', dict(status='complete', queries=len(results), generation_calls=50+len(rounds),
            unet_calls=nfe[0], gpu_worker_seconds=time.monotonic()-started, gpu=info, new_attacks=0))
        print(json.dumps(dict(event='gpu_complete', queries=len(results), seconds=time.monotonic()-started)), flush=True)


def supervise(run):
    assert not (run / 'supervisor.json').exists(), 'One launch only; no unaccounted restart'
    started = time.monotonic()
    plan = read(run / 'protocol.json')
    worker_process = subprocess.Popen([str(PYTHON), '-u', str(run / Path(__file__).name), 'worker', str(run)], start_new_session=True)
    write(run / 'supervisor.json', dict(status='running', pid=os.getpid(), worker_pid=worker_process.pid,
                                      protocol_sha256=sha(run / 'protocol.json'), started_unix=time.time()))
    expired = False
    try:
        code = worker_process.wait(timeout=plan['watchdog_worker_timeout_seconds'])
    except subprocess.TimeoutExpired:
        expired = True
        os.killpg(worker_process.pid, signal.SIGTERM)
        try:
            code = worker_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(worker_process.pid, signal.SIGKILL)
            code = worker_process.wait(timeout=30)
    elapsed = time.monotonic() - started
    write(run / 'supervisor_complete.json', dict(exit_code=code, budget_expired=expired,
         allocated_gpu_seconds_upper_bound=elapsed,
         cumulative_gpu_hours_upper_bound=(elapsed+plan['prior_launch_seconds_upper_bound'])/3600,
         complete=code == 0 and (run / 'gpu_complete.json').exists(), protocol_sha256=sha(run / 'protocol.json')))
    print(json.dumps(dict(event='worker_exited', code=code, expired=expired, seconds=elapsed)), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'supervise'])
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    assert args.run.parent.resolve() == (ROOT / 'runs').resolve()
    if args.mode == 'prepare':
        prepare(args.run)
    elif args.mode == 'worker':
        worker(args.run)
    else:
        sys.exit(supervise(args.run))
