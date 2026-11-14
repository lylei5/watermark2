"""CVPR2025 author's unmodified Reprompt phase2/3 on public reference images."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')
AUTHOR = ROOT / 'third_party/semantic-forgery'
AUTHOR_COMMIT = 'ca68950cd7b53b8438a49580f134b2a0db28d778'
PYTHON = ROOT / 'envs/semantic_forgery_py310/bin/python'
MODELS = {
    'sd21': ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c',
    'sd15': Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def author_check():
    assert subprocess.check_output(['git', '-C', str(AUTHOR), 'rev-parse', 'HEAD'], text=True).strip() == AUTHOR_COMMIT
    assert not subprocess.check_output(['git', '-C', str(AUTHOR), 'status', '--porcelain'], text=True).strip()


def prepare(run):
    author_check()
    parent = ROOT / 'runs/shared_channel_binding_20261114_v2/feature_rows.json'
    inputs = ROOT / 'runs/phase1_public_inputs_20261113/protocol.json'
    targets = read(inputs)['targets']
    references = []
    for row in read(parent):
        if row['kind'] != 'clean':
            continue
        target = targets[row['task']]
        assert sha(row['image']) == row['image_sha256']
        references.append(dict(id=row['id'], carrier=row['carrier'], task=row['task'],
            image=row['image'], image_sha256=row['image_sha256'],
            pixels_sha256=row['pixels_sha256'], target_prompt=target['prompt'],
            target_source_index=target['source_index']))
    assert len(references) == 30 and len({r['id'] for r in references}) == 30
    run.mkdir(exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    source_hashes = {str(p.relative_to(AUTHOR)): sha(p) for p in AUTHOR.rglob('*.py') if '.git' not in p.parts}
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(run / Path(__file__).name), author_root=str(AUTHOR), author_commit=AUTHOR_COMMIT,
        author_source_sha256=source_hashes, parent_rows_sha256=sha(parent), input_protocol_sha256=sha(inputs),
        references=references, models={k: str(v) for k, v in MODELS.items()},
        model_configs={k: {str(p.relative_to(v)): sha(p) for p in v.rglob('*.json')} for k, v in MODELS.items()},
        resolution=512, inverse_steps=50, forward_steps=50, inverse_prompt='', inverse_guidance=1.,
        forward_guidance=7.5, dtype='float32', seed=1, resampling=False, candidates_per_reference=1,
        author_functions=['SDPipeProvider.invert_images', 'SDPipeProvider.generate'],
        adaptation='Public image input and local model path; instantiate author SDPipeProvider directly for SD1.5 alias.',
        private_inputs=False, isolation='Logical input separation, shared Unix user; no OS sandbox claim.',
        cohort='Paired development sources:10 each GS/TR/PRC, same as complete Imprint and genuine controls.',
        scope='Complete 50-step basic Reprompt workflow, two proxy models,60 outputs; not the paper-scale matrix.',
        differences=['Filtered I2P targets from existing reviewed protocol rather than unrestricted I2P.',
            'SD2.1 target carrier cohort; CVPR original main study uses other target models and1000 sources.',
            'SemBind Table4 uses100COCO+100SDP per arm; current matched cohort has10 per carrier.',
            'Reprompt+ best-of3 prompts and GS resampling is a separate published setting, not run here.'],
        paper='https://arxiv.org/html/2412.03283v3',
        defense_protocol='https://arxiv.org/html/2601.20310#S4.SS3',
        allowed_gpu_indices=[7, 5, 4], excluded_gpu_indices=[6])
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), references=len(references), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def worker(run, model, gpu):
    out = run / model
    out.mkdir(exist_ok=False)
    plan = read(run / 'protocol.json')
    state = dict(status='preflight', pid=os.getpid(), model=model, protocol_sha256=sha(run / 'protocol.json'))
    write(out / 'state.json', state)
    try:
        assert sha(__file__) == plan['source_sha256']
        assert gpu in plan['allowed_gpu_indices'] and gpu != 6
        author_check()
        for name, digest in plan['author_source_sha256'].items():
            assert sha(AUTHOR / name) == digest, name
        for name, digest in plan['model_configs'][model].items():
            assert sha(Path(plan['models'][model]) / name) == digest, name
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 100 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT / 'cache/huggingface'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        sys.path.insert(0, str(AUTHOR))
        import torch
        import diffusers
        import transformers
        import numpy as np
        from PIL import Image
        from utils.pipe.SD_provider import SDPipeProvider
        from utils.utils import set_random_seed
        torch.set_num_threads(4)
        provider = SDPipeProvider(pretrained_model_name_or_path=plan['models'][model], resolution=512, device=torch.device('cuda:0'),
            eager_loading=True, unet_id_or_checkpoint_dir=None, lora_checkpoint_dir=None,
            vae_id=None, zero_unet=False, disable_tqdm=True)
        assert provider.get_dtype() == torch.float32
        state.update(status='running', gpu=gpu, gpu_uuid=info[1], started_at=time.time(),
            torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
            forward_scheduler=type(provider.scheduler).__module__, inverse_scheduler=type(provider.scheduler_inverse).__module__)
        write(out / 'state.json', state)
        (out / 'images').mkdir()
        (out / 'latents').mkdir()
        rows = []
        for ref in plan['references']:
            begin = time.perf_counter()
            assert sha(ref['image']) == ref['image_sha256']
            image = Image.open(ref['image']).convert('RGB')
            assert image.size == (512, 512) and hashlib.sha256(image.tobytes()).hexdigest() == ref['pixels_sha256']
            set_random_seed(plan['seed'])
            # The same phase2/3 method calls and clone as the author's run_reprompting.py.
            with torch.no_grad():
                res_2 = provider.invert_images(images=provider.PIL_to_torch([image]), num_inference_steps=plan['inverse_steps'])
                recovered_zT = res_2['zT_torch'].clone()
                res_3 = provider.generate(prompts=[ref['target_prompt']], num_inference_steps=plan['forward_steps'],
                    guidance_scale=plan['forward_guidance'], latents=recovered_zT)
            path = out / 'images' / (ref['id'] + '.png')
            latent_path = out / 'latents' / (ref['id'] + '.npz')
            res_3['images_PIL'][0].save(path)
            np.savez_compressed(latent_path, proxy_zT=recovered_zT.cpu().numpy())
            row = dict(id=ref['id'], carrier=ref['carrier'], task=ref['task'], proxy=model,
                reference_sha256=ref['image_sha256'], target_source_index=ref['target_source_index'],
                image=str(path), image_sha256=sha(path), pixels_sha256=hashlib.sha256(res_3['images_PIL'][0].convert('RGB').tobytes()).hexdigest(),
                latent=str(latent_path), latent_sha256=sha(latent_path), seconds=time.perf_counter()-begin)
            rows.append(row)
            write(out / 'rows.json', rows)
            print(json.dumps(row), flush=True)
        author_check()
        state.update(status='complete', count=len(rows), seconds=time.time()-state['started_at'], rows_sha256=sha(out / 'rows.json'))
        write(out / 'complete.json', state)
        write(out / 'state.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        write(out / 'state.json', state)
        raise


def launch(run):
    jobs = []
    for model, gpu in [('sd21', 7), ('sd15', 5)]:
        log = open(run / (model + '.log'), 'xb')
        process = subprocess.Popen([str(PYTHON), str(run / Path(__file__).name), 'worker', '--run', str(run),
            '--model', model, '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        jobs.append(dict(model=model, gpu=gpu, pid=process.pid))
    write(run / 'launch.json', dict(jobs=jobs, protocol_sha256=sha(run / 'protocol.json')))
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'launch', 'worker'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', choices=['sd21', 'sd15'])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.run)
    elif args.mode == 'launch':
        launch(args.run)
    else:
        worker(args.run, args.model, args.gpu)
