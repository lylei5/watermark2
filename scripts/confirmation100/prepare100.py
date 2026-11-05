import os
import sys
import time
from pathlib import Path
from common100 import c, r, ROOT, BASE, gate, module


def main(runtime, private_folder, output):
    out = Path(output)
    cfg, d = gate(runtime, out, 'evaluator')
    import torch
    from PIL import Image
    private = module('confirmation100_private', BASE / 'reference_reliability/private_state.py')
    provider = r.official_provider(d['target'])
    from utils.wm.gs_provider import GsProvider
    rows = []
    for task in d['tasks']:
        sid = task['id']
        folder = out / str(sid)
        folder.mkdir()
        source = ROOT / task['cover']
        if c.sha(source) != task['cover_sha256']:
            raise ValueError('Natural cover source changed')
        with (folder / 'original_cover.png').open('xb') as f:
            f.write(source.read_bytes())
        with Image.open(source) as im:
            cover = im.resize((512, 512)).convert('RGB')
        with (folder / 'cover.png').open('xb') as f:
            cover.save(f, format='PNG')
        r.deterministic(task['reference_seed'])
        wm = GsProvider(latent_shape=(1, 4, 64, 64), device='cuda:0', dtype=torch.float32,
                        message_width_in_bytes=32, num_replications=64, l=1, offset=0)
        raw = wm.messages[0] + wm.keys[0] + wm.nonces[0]
        fingerprints = private.fingerprints(raw)
        fd = os.open(Path(private_folder) / (str(sid) + '.bin'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
        z = wm.get_wm_latents()['zT_torch']
        noise = r.tensor(folder / 'original_noise.npy', z)
        start = time.perf_counter()
        with torch.no_grad():
            image = provider.generate(prompts=[task['reference_prompt']], num_inference_steps=50,
                                      guidance_scale=7.5, latents=z, return_latents=False)['images_PIL'][0]
        torch.cuda.synchronize()
        timesteps = provider.scheduler.timesteps.tolist()
        if timesteps != list(range(981, 0, -20)):
            raise ValueError('Unexpected actual generation timesteps')
        with (folder / 'reference.png').open('xb') as f:
            image.save(f, format='PNG')
        row = dict(task_id=sid, reference=c.png_record(folder / 'reference.png'),
                   cover=c.png_record(folder / 'cover.png'),
                   original_cover_sha256=c.sha(source), original_noise=noise,
                   encryption_fingerprints=fingerprints, seed=task['reference_seed'],
                   prompt=task['reference_prompt'], generation_seconds=time.perf_counter() - start,
                   actual_generation_timesteps=timesteps,
                   endpoint_alpha_cumprod=float(provider.scheduler.alphas_cumprod[timesteps[-1]]))
        c.write(folder / 'record.json', row)
        rows.append(row)
        print('PREPARED', sid, 'seconds', round(row['generation_seconds'], 3), flush=True)
    c.write(out / 'summary.json', dict(completed=True, rows=rows))
    print('COMPLETED_NATIVE_PREPARE', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
