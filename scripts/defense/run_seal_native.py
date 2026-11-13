"""SEAL official generation/inversion with the ICCV supplement decision rule."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert str(run.parent) == '/data0/liyanlei/watermark_research/runs'
    plan = json.loads((run / 'protocol.json').read_text())
    assert Path(__file__).resolve() == run / plan['worker'], 'Execute the frozen run worker'
    assert sha(__file__) == plan['worker_sha256']
    assert sha(run / 'environment_assets.json') == plan['environment_assets_sha256']
    assert sha(run / 'source_manifest.json') == plan['source_manifest_sha256']
    output = run / f'native_{args.shard}'
    output.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    save(output / 'state.json', state)
    started = time.monotonic()
    rows = []
    try:
        gpu = plan['gpu_indices'][args.shard]
        assert gpu in [7, 5, 4] and gpu != 6
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
               '--query-gpu=index,uuid,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 100 and int(info[3]) == 0, info
        root = Path('/data0/liyanlei/watermark_research')
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(root / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                          TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        official = Path(plan['source'])
        for name, digest in plan['source_sha256'].items():
            assert sha(official / name) == digest, name
        sys.path.insert(0, str(official))
        import numpy as np
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from utils import (generate_caption, generate_initial_noise, transform_img,
                           calculate_patch_l2, set_random_seed)
        paper_sampler = None
        if plan.get('noise_variant') == 'paper_secret_salt':
            assert sha(run / 'seal_paper_simhash.py') == plan['salted_sampler_sha256']
            salt_file = Path(plan['private_salt_file'])
            assert sha(salt_file) == plan['private_salt_sha256']
            from seal_paper_simhash import SealPaperSimHash
            paper_sampler = SealPaperSimHash(bytes.fromhex(json.loads(salt_file.read_text())['salt']), device='cuda')
        torch.set_num_threads(4)
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            plan['diffusion_model'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        processor = Blip2Processor.from_pretrained(plan['caption_model'], local_files_only=True)
        captioner = Blip2ForConditionalGeneration.from_pretrained(
            plan['caption_model'], torch_dtype=torch.float16, local_files_only=True).to('cuda')
        encoder = SentenceTransformer(plan['embedding_model'], device='cuda')
        state.update(status='running', gpu=gpu, gpu_uuid=info[1],
                     scheduler=type(pipe.scheduler).__name__, dtype=str(pipe.vae.dtype),
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32)
        assert state['scheduler'] == plan['expected_scheduler']
        save(output / 'state.json', state)
        (output / 'images').mkdir()
        (output / 'latents').mkdir()

        def describe(image):
            caption = generate_caption(image, processor, captioner)
            embedding = encoder.encode(caption, convert_to_tensor=True).to('cuda')
            return caption, embedding / torch.norm(embedding)

        def noise(embedding):
            if paper_sampler is not None:
                return paper_sampler.sample(embedding).to(dtype=pipe.vae.dtype)
            return generate_initial_noise(embedding, plan['patches'], plan['bits_per_patch'],
                                          plan['official_projection_seed'], 'cuda').to(dtype=pipe.vae.dtype)

        def invert(image):
            tensor = transform_img(image).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
            latent = pipe.get_image_latents(tensor, sample=False)
            return pipe.forward_diffusion(latents=latent, text_embeddings=pipe.get_text_embedding(''),
                                          guidance_scale=1, num_inference_steps=50)

        for item in (x for x in plan['sources'] if x['task'] % 3 == args.shard):
            tick = time.monotonic()
            task = item['task']
            set_random_seed(item['generation_seed'])
            preview = pipe(item['prompt']).images[0]
            preview_caption, embedding = describe(preview)
            initial = noise(embedding)
            genuine = pipe(item['prompt'], latents=initial).images[0]
            genuine_caption, genuine_embedding = describe(genuine)
            genuine_reference = noise(genuine_embedding)
            recovered = invert(genuine)
            # Preserve SEAL.py's order: null generation precedes null caption/noise.
            null_image = pipe(item['null_prompt']).images[0]
            null_recovered = invert(null_image)
            null_caption, null_embedding = describe(null_image)
            null_reference = noise(null_embedding)
            images = {}
            for kind, image in [('preview', preview), ('genuine', genuine), ('null', null_image)]:
                path = output / 'images' / f'{task:03}_{kind}.png'
                image.save(path)
                images[kind] = dict(path=str(path.relative_to(run)), sha256=sha(path))
            preview_control_equal = None
            if paper_sampler is not None:
                control = Path(plan['released_code_run']) / f'native_{args.shard}' / 'images' / f'{task:03}_preview.png'
                preview_control_equal = sha(control) == images['preview']['sha256']
                assert preview_control_equal, 'The paired preview image changed'
            latent_path = output / 'latents' / f'{task:03}.npz'
            np.savez_compressed(latent_path, initial=initial.cpu().numpy(),
                                genuine_reference=genuine_reference.cpu().numpy(),
                                recovered=recovered.cpu().numpy(), null_recovered=null_recovered.cpu().numpy(),
                                null_reference=null_reference.cpu().numpy())
            decisions = []
            for kind, z, reference in [('genuine', recovered, genuine_reference),
                                       ('null', null_recovered, null_reference)]:
                distances = calculate_patch_l2(z, reference, plan['patches'])
                assert len(distances) == 1024 and np.isfinite(distances).all()
                matches = sum(d < plan['tau'] for d in distances)
                decisions.append(dict(kind=kind, distances=distances, matches=matches,
                                      accepted=matches >= plan['minimum_matches']))
            rows.append(dict(task=task, dataset=item['dataset'], images=images,
                             preview_control_equal=preview_control_equal,
                             captions=dict(preview=preview_caption, genuine=genuine_caption, null=null_caption),
                             latents=dict(path=str(latent_path.relative_to(run)), sha256=sha(latent_path)),
                             decisions=decisions, seconds=time.monotonic()-tick))
            save(output / 'progress.json', dict(rows=rows, protocol_sha256=state['protocol_sha256']))
            print('SOURCE_COMPLETE', task, round(rows[-1]['seconds'], 2), flush=True)
        state.update(status='complete', rows=rows, seconds=time.monotonic()-started,
                     completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(output / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), completed_sources=len(rows))
        raise
    finally:
        save(output / 'state.json', {k: v for k, v in state.items() if k != 'rows'})


if __name__ == '__main__':
    main()
