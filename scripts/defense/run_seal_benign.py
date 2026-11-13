"""Evaluate saved SEAL images with the seven unchanged released conditions."""
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


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=range(3), required=True)
    args = p.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    run = args.run.resolve()
    assert run.parent == root / 'runs'
    plan = json.loads((run / 'protocol.json').read_text())
    assert Path(__file__).resolve() == run / plan['worker'], 'Execute the frozen run worker'
    assert sha(__file__) == plan['worker_sha256']
    for name, digest in plan['local_sha256'].items():
        assert sha(run / name) == digest, name
    out = run / f'native_{args.shard}'
    out.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    rows = []
    started = time.monotonic()
    try:
        gpu = plan['gpu_indices'][args.shard]
        assert gpu in [7, 5, 4]
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 100 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(root / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        official = Path(plan['source'])
        for name, digest in plan['source_sha256'].items():
            assert sha(official / name) == digest, name
        sys.path.insert(0, str(official))
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from utils import generate_caption, generate_initial_noise, transform_img, calculate_patch_l2, set_random_seed, image_distortion
        from seal_paper_simhash import SealPaperSimHash
        torch.set_num_threads(4)
        pipe = InversableStableDiffusionPipeline.from_pretrained(plan['diffusion_model'],
                torch_dtype=torch.float16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        assert type(pipe.scheduler).__name__ == 'PNDMScheduler'
        processor = Blip2Processor.from_pretrained(plan['caption_model'], local_files_only=True)
        captioner = Blip2ForConditionalGeneration.from_pretrained(plan['caption_model'],
                    torch_dtype=torch.float16, local_files_only=True).to('cuda')
        encoder = SentenceTransformer(plan['embedding_model'], device='cuda')
        salt = Path(plan['private_salt_file'])
        assert sha(salt) == plan['private_salt_sha256']
        sampler = SealPaperSimHash(bytes.fromhex(json.loads(salt.read_text())['salt']), device='cuda')
        (out / 'images').mkdir()
        (out / 'latents').mkdir()
        state.update(status='running', gpu=gpu, gpu_uuid=info[1])
        save(out / 'state.json', state)
        for item in (x for x in plan['sources'] if x['task'] % 3 == args.shard):
            image_path = Path(item['image'])
            assert sha(image_path) == item['image_sha256']
            genuine = Image.open(image_path).convert('RGB')
            for condition in plan['conditions']:
                name = condition['name']
                set_random_seed(item['task'])
                image = genuine if name == 'Clean' else image_distortion(None, genuine, seed=item['task'], **condition['params'])[1]
                caption = generate_caption(image, processor, captioner)
                embedding = encoder.encode(caption, convert_to_tensor=True).to('cuda')
                embedding = embedding / torch.norm(embedding)
                reference = (sampler.sample(embedding) if item['variant'] == 'paper' else
                             generate_initial_noise(embedding, 1024, 7, 42, 'cuda')).to(dtype=pipe.vae.dtype)
                tensor = transform_img(image).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
                recovered = pipe.forward_diffusion(latents=pipe.get_image_latents(tensor, sample=False),
                          text_embeddings=pipe.get_text_embedding(''), guidance_scale=1, num_inference_steps=50)
                distances = calculate_patch_l2(recovered, reference, 1024)
                assert len(distances) == 1024 and np.isfinite(distances).all()
                matches = sum(x < 2.3 for x in distances)
                stem = f"{item['variant']}_{item['task']:03}_{name}"
                file = out / 'images' / (stem + '.png')
                image.save(file)
                latent = out / 'latents' / (stem + '.npz')
                np.savez_compressed(latent, recovered=recovered.cpu().numpy(), reference=reference.cpu().numpy())
                row = dict(task=item['task'], variant=item['variant'], dataset=item['dataset'], condition=name,
                    caption=caption, distances=distances, matching_patches=matches, accepted=matches >= 12,
                    image=str(file.relative_to(run)), image_sha256=sha(file),
                    latent=str(latent.relative_to(run)), latent_sha256=sha(latent),
                    clean_accepted=item['clean_accepted'], clean_matching_patches=item['clean_matching_patches'])
                if name == 'Clean':
                    assert caption == item['clean_caption']
                    assert matches == item['clean_matching_patches'] and row['accepted'] == item['clean_accepted'], 'Clean replay changed'
                rows.append(row)
                save(out / 'progress.json', dict(completed=len(rows), last=stem))
            print('SOURCE_COMPLETE', item['variant'], item['task'], flush=True)
        state.update(status='complete', rows=rows, seconds=time.monotonic()-started,
                     completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), completed_rows=len(rows))
        save(out / 'partial.json', dict(rows=rows))
        raise
    finally:
        save(out / 'state.json', {k:v for k,v in state.items() if k != 'rows'})


if __name__ == '__main__':
    main()
