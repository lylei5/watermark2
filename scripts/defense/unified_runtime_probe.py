"""Extract full detector features under frozen inputs and original inference budgets."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(run):
    assert run.parent == ROOT / 'runs'
    assert not (run / 'protocol.json').exists()
    images = sorted((ROOT / 'runs/content_auth_20261109/assets/independent_coco2017/images').glob('*.png'))[:8]
    assert len(images) == 8
    plans = {}
    for method, name in [('sembind', 'sembind_prc_native_20261113'), ('seal', 'seal_paper_native_20261113')]:
        path = ROOT / 'protocols' / (name + '.json')
        plans[method] = json.loads(path.read_text())
    sources = {
        'pgid': ROOT / 'third_party/frozen/PGID_dc4ad96',
        'sembind': Path(plans['sembind']['paths']['official']),
        'seal': Path(plans['seal']['source']),
    }
    plan = dict(scope='development-only full-runtime extraction; no attacks, no FPR or TPR claim',
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                inputs=[dict(path=str(p), sha256=sha(p)) for p in images],
                methods=plans, sources={k: str(v) for k, v in sources.items()},
                source_sha256={k: {str(p.relative_to(v)): sha(p) for p in sorted(v.rglob('*.py'))}
                               for k, v in sources.items()},
                worker_sha256=sha(__file__),
                helper_sha256={p.name: sha(p) for p in run.glob('*.py') if p.name != Path(__file__).name},
                gpu_indices={'pgid': 7, 'sembind': 5, 'seal': 4},
                budgets={'pgid_default': {'steps': 50},
                         'pgid_f': {'steps': 50, 'k': 15, 's': 3, 'gamma': .001},
                         'pgid_r': {'steps': 50, 'k': 10, 's': 1, 'gamma': .045},
                         'sembind': {'steps': 50, 'vae_optimization_steps': 100, 'cfg': 3, 'inv_order': 0},
                         'seal': {'steps': 50, 'cfg': 1, 'caption': 'official BLIP2'}})
    save(run / 'protocol.json', plan)
    print(json.dumps({'prepared': str(run), 'sources': len(images), 'protocol_sha256': sha(run / 'protocol.json')}))


def execute(run, method):
    assert run.is_relative_to(ROOT / 'runs')
    assert Path(__file__).resolve() == run / 'unified_runtime_probe.py'
    plan = json.loads((run / 'protocol.json').read_text())
    assert sha(__file__) == plan['worker_sha256']
    for name, digest in plan['helper_sha256'].items():
        assert sha(run / name) == digest, name
    source = Path(plan['sources'][method])
    for name, digest in plan['source_sha256'][method].items():
        assert sha(source / name) == digest, name
    output = run / method
    output.mkdir(exist_ok=True)
    state = dict(status='preflight', pid=os.getpid(), method=method, protocol_sha256=sha(run / 'protocol.json'))
    rows = []
    completed = set()
    if (output / 'progress.json').exists():
        rows = json.loads((output / 'progress.json').read_text())['rows']
        for row in rows:
            assert sha(output / row['artifact']) == row['artifact_sha256']
            assert row['input_sha256'] == plan['inputs'][row['source']]['sha256']
            assert row.get('protocol_sha256') == sha(run / 'protocol.json')
            pair = (row['source'], row['arm'])
            assert pair not in completed
            completed.add(pair)
    started = time.monotonic()
    save(output / 'state.json', state)
    try:
        gpu = plan['gpu_indices'][method]
        assert gpu != 6
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[3]) < 100 and int(info[4]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        import diffusers
        import transformers
        from PIL import Image
        torch.set_num_threads(4)
        torch.manual_seed(20261115)
        np.random.seed(20261115)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state.update(status='loading', gpu=gpu, gpu_uuid=info[1], gpu_name=info[2],
                     versions={'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__})
        save(output / 'state.json', state)
        nfe = [0]

        def count_forward(module):
            original = module.forward
            def measured(*args, **kwargs):
                nfe[0] += 1
                return original(*args, **kwargs)
            module.forward = measured

        if method == 'pgid':
            from pipe import pipe_utils
            from pgid_defense import PGIDPipeline
            provider = pipe_utils.create_diffusion_pipe(
                pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base', resolution=512,
                device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM', eager_loading=True)
            count_forward(provider.pipe.unet)
            defenses = {name: PGIDPipeline(pipe=provider, scheduler_config=provider.scheduler.config,
                        num_inference_steps=50, stopping_timestep=budget['k'], num_skip_inversion=budget['s'],
                        guidance_strength=budget['gamma'])
                        for name, budget in plan['budgets'].items() if name in ('pgid_f', 'pgid_r')}
            def extract(image, arm):
                if arm == 'pgid_default':
                    with torch.no_grad():
                        return provider.invert_images(image, num_inference_steps=50)['zT_torch'], {}
                return defenses[arm].run_defense(image)['zT'], {}
            arms = ['pgid_default', 'pgid_f', 'pgid_r']
        elif method == 'sembind':
            from semantic_code import SemanticCodeExtractor
            from sembind_official_audit import load_official
            paths = {k: Path(v) for k, v in plan['methods']['sembind']['paths'].items()}
            assert sha(paths['scheduler']) == plan['methods']['sembind']['scheduler_sha256']
            spec = importlib.util.spec_from_file_location('model', paths['official'] / 'model.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules['model'] = module
            spec.loader.exec_module(module)
            extractor = SemanticCodeExtractor(paths['semantic'] / 'dinov2-giant', paths['semantic'] / 'sembind_masker.pth')
            inversion, pipe = load_official(SimpleNamespace(**paths))
            pipe.set_progress_bar_config(disable=True)
            count_forward(pipe.unet)
            def extract(image, arm):
                code = extractor.code(image).detach().cpu().numpy().astype(np.uint8)
                recovered = inversion.exact_inversion(image, prompt='', guidance_scale=3., num_inference_steps=50,
                    test_num_inference_steps=50, inv_order=0, decoder_inv=True, pipe=pipe).detach()
                return recovered, {'semantic_code': code}
            arms = ['sembind']
        else:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            from sentence_transformers import SentenceTransformer
            from inverse_stable_diffusion import InversableStableDiffusionPipeline
            from utils import generate_caption, transform_img, calculate_patch_l2
            from seal_paper_simhash import SealPaperSimHash
            cfg = plan['methods']['seal']
            assert sha(cfg['private_salt_file']) == cfg['private_salt_sha256']
            sampler = SealPaperSimHash(bytes.fromhex(json.loads(Path(cfg['private_salt_file']).read_text())['salt']))
            pipe = InversableStableDiffusionPipeline.from_pretrained(cfg['diffusion_model'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
            pipe.set_progress_bar_config(disable=True)
            processor = Blip2Processor.from_pretrained(cfg['caption_model'], local_files_only=True)
            captioner = Blip2ForConditionalGeneration.from_pretrained(cfg['caption_model'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
            encoder = SentenceTransformer(cfg['embedding_model'], device='cuda')
            count_forward(pipe.unet)
            def extract(image, arm):
                caption = generate_caption(image, processor, captioner)
                embedding = encoder.encode(caption, convert_to_tensor=True).to('cuda')
                reference = sampler.sample(embedding / torch.norm(embedding)).to(dtype=pipe.vae.dtype)
                with torch.no_grad():
                    tensor = transform_img(image).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
                    latent = pipe.get_image_latents(tensor, sample=False)
                    recovered = pipe.forward_diffusion(latents=latent, text_embeddings=pipe.get_text_embedding(''), guidance_scale=1, num_inference_steps=50)
                distances = calculate_patch_l2(recovered, reference, 1024)
                return recovered, {'distances': np.asarray(distances), 'semantic_matches': np.asarray(sum(d < 2.3 for d in distances))}
            arms = ['seal']
        torch.cuda.synchronize()
        state.update(status='running', loading_seconds=time.monotonic()-started)
        save(output / 'state.json', state)
        for index, item in enumerate(plan['inputs']):
            assert sha(item['path']) == item['sha256']
            with Image.open(item['path']) as source_image:
                if plan.get('preprocessing') == 'common_natural_512':
                    from prepare_coco_nulls import common_image
                    image = common_image(source_image)
                    assert hashlib.sha256(image.tobytes()).hexdigest() == item['pixel_sha256']
                else:
                    image = source_image.convert('RGB')
            for arm in arms:
                if (index, arm) in completed:
                    continue
                torch.manual_seed(202611150000 + index)
                torch.cuda.reset_peak_memory_stats()
                nfe[0] = 0
                torch.cuda.synchronize()
                tick = time.monotonic()
                recovered, auxiliary = extract(image, arm)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                assert recovered.shape == (1, 4, 64, 64) and torch.isfinite(recovered).all()
                if 'expected_unet_calls' in plan:
                    assert nfe[0] == plan['expected_unet_calls'][arm], (arm, nfe[0])
                destination = output / f'{index:03}_{arm}.npz'
                np.savez_compressed(destination, recovered=recovered.detach().cpu().numpy(), **auxiliary)
                rows.append(dict(source=index, arm=arm, input_sha256=item['sha256'], seconds=elapsed,
                                 global_source=item.get('index', index), split=item.get('split', 'development'),
                                 protocol_sha256=sha(run / 'protocol.json'),
                                 input_pixel_sha256=item.get('pixel_sha256'), unet_calls=nfe[0],
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                 artifact=destination.name, artifact_sha256=sha(destination), artifact_bytes=destination.stat().st_size))
                save(output / 'progress.json', {'rows': rows})
                print(json.dumps(rows[-1]), flush=True)
        state.update(status='complete', rows=len(rows), total_seconds=time.monotonic()-started)
        save(output / 'complete.json', dict(state, measurements=rows))
    except Exception as exc:
        state.update(status='failed', error=repr(exc), completed_rows=len(rows))
        raise
    finally:
        save(output / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', choices=['pgid', 'sembind', 'seal'])
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare(args.run.resolve()) if args.prepare else execute(args.run.resolve(), args.method)
