"""SemBind smoke: GS-native vs SemBind-GS, self-verification and one Imprint forgery arm per variant.

Deviation: the forgery uses this project's pinned 50-step DDIM inversion and a
short 30-step optimisation budget, not the paper's 150-step / exact-inversion setup.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
sys.path.insert(0, str(ROOT / 'scripts/defense'))
import common100 as m  # noqa: E402
from sembind import SemanticMasker  # noqa: E402


def imprint_attack(provider, target_snapshot, cover, reference, steps, device):
    import typing
    import PIL.Image as PILImage
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor, to_pil_image
    from torch.utils.checkpoint import checkpoint
    from diffusers import (StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler,
                           DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler)
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
    scope = dict(typing=typing, torch=torch, PIL=PILImage, Optional=typing.Optional, Union=typing.Union,
                 List=typing.List, checkpoint=checkpoint, to_tensor=to_tensor, to_pil_image=to_pil_image,
                 retrieve_timesteps=retrieve_timesteps, rescale_noise_cfg=rescale_noise_cfg,
                 StableDiffusionPipeline=StableDiffusionPipeline, DDIMScheduler=DDIMScheduler,
                 DDIMInverseScheduler=DDIMInverseScheduler, DPMSolverMultistepScheduler=DPMSolverMultistepScheduler,
                 DPMSolverMultistepInverseScheduler=DPMSolverMultistepInverseScheduler)
    m.r.definitions(m.r.UPSTREAM / 'utils/imprint_utils.py',
                    {'load_attacker_pipe', 'DiffPipe', 'UnetWrapper', 'pixel_to_latent', 'latent_to_pil', 'invert_image'}, scope)
    pixel_scope = dict(typing=typing, torch=torch, Image=PILImage, transforms=transforms)
    m.r.definitions(m.r.UPSTREAM / 'utils/image_utils.py', {'PIL_to_torch'}, pixel_scope)
    pipe, scheduler, inverse = scope['load_attacker_pipe'](modelid=target_snapshot, scheduler='DDIM', device=device)
    diff = scope['DiffPipe'](pipe, scheduler=inverse, device=device)
    inverse.set_timesteps(50, device=device)
    z0 = torch.nn.Parameter(scope['pixel_to_latent'](cover, pipe).detach().clone())
    optim = torch.optim.Adam([z0], lr=.01)
    ref_pt = pixel_scope['PIL_to_torch']([reference], dtype=torch.float32, device=device)
    with torch.no_grad():
        target = scope['invert_image'](pipe=pipe, image_pt=ref_pt, scheduler=inverse, num_inference_steps=50)
    code = m.update_code(m.r.UPSTREAM / 'run_imprint_forgery.py')
    state = dict(optim=optim, diffpipe=diff, z0=z0, torch=torch, zT_retrieved=target)
    for _ in range(steps):
        exec(code, state)
    return scope['latent_to_pil'](z0, pipe)[0]


def main(out_dir, steps=30, device='cuda:0'):
    steps = int(steps)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    design = m.c.read(ROOT / 'protocols/confirmation100_design_v7.json')
    proxy = dict(design['proxy'])
    provider = m.r.official_provider(proxy)
    from utils.wm.gs_provider import GsProvider

    masker = SemanticMasker(ROOT / 'models/sembind/dinov2-giant',
                            ROOT / 'models/sembind/sembind_masker.pth', device=device)
    wm = GsProvider(latent_shape=(1, 4, 64, 64), device=device, dtype=torch.float32,
                    message_width_in_bytes=32, num_replications=64, l=1, offset=0)

    prompt = 'A cat sitting on a wooden bench in a garden.'
    cover_path = ROOT / 'datasets/confirmation100_relay_v11/rgb512/000000084477.png'
    result = {'prompt': prompt, 'cover': str(cover_path), 'steps': steps,
              'deviations': ['DDIM 50-step inversion instead of exact DPM-Solver inv_order=0',
                             'short 30-step Imprint budget instead of 150']}

    torch.manual_seed(7)
    aux_z = torch.randn(1, 4, 64, 64, device=device)
    aux = provider.generate(prompts=[prompt], num_inference_steps=50, guidance_scale=7.5,
                            latents=aux_z, return_latents=False)['images_PIL'][0]
    aux.save(out / 'aux_clean.png')
    sign = masker.sign_mask(aux).to(device)
    bits = masker.code(aux)
    result['mask'] = dict(bits=int(bits.numel()), ones=float(bits.float().mean()),
                          minus_one_fraction=float((sign == -1).float().mean()))
    zT = wm.get_wm_latents()['zT_torch']
    z_sem = sign * zT

    t0 = time.time()
    sem_img = provider.generate(prompts=[prompt], num_inference_steps=50, guidance_scale=7.5,
                                latents=z_sem, return_latents=False)['images_PIL'][0]
    sem_img.save(out / 'sembind_gs.png')
    van_img = provider.generate(prompts=[prompt], num_inference_steps=50, guidance_scale=7.5,
                                latents=zT, return_latents=False)['images_PIL'][0]
    van_img.save(out / 'vanilla_gs.png')
    result['generation_seconds'] = time.time() - t0

    def verify(image, label):
        s_hat = masker.sign_mask(image).to(device)
        z_inv = provider.invert_images(image, num_inference_steps=50)['zT_torch']
        bound = float(wm.get_accuracies(z_inv)['bit_accuracies'][0])
        unbound = float(wm.get_accuracies(s_hat * z_inv)['bit_accuracies'][0])
        return dict(label=label, ba_bound=bound, ba_unbound=unbound)

    result['self_check'] = [verify(sem_img, 'sembind_gs'), verify(van_img, 'vanilla_gs')]

    with Image.open(cover_path) as im:
        cover = im.convert('RGB')
    result['forgery'] = []
    for label, ref_img in (('sembind_gs', sem_img), ('vanilla_gs', van_img)):
        start = time.time()
        candidate = imprint_attack(provider, proxy['snapshot'], cover, ref_img, steps, device)
        candidate.save(out / ('forged_' + label + '.png'))
        check = verify(candidate, label + '_forged')
        check['seconds'] = time.time() - start
        result['forgery'].append(check)
        print('FORGERY', label, check, flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
