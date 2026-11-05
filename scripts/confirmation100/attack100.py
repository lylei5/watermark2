import sys
import time
from pathlib import Path
from common100 import c, r, gate, validate_job, update_code, guard, POINTS


def main(runtime, job_path, output):
    path = Path(job_path)
    out = Path(output)
    cfg, d = gate(runtime, out, 'attacker')
    job = validate_job(path, cfg, d)
    job_hash = c.sha(path)
    r.deterministic(job['seed'])
    import typing
    import numpy as np
    import torch
    import PIL.Image
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor, to_pil_image
    from torch.utils.checkpoint import checkpoint
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
    scope = dict(typing=typing, torch=torch, PIL=PIL, Optional=typing.Optional, Union=typing.Union, List=typing.List,
                 checkpoint=checkpoint, to_tensor=to_tensor, to_pil_image=to_pil_image,
                 retrieve_timesteps=retrieve_timesteps, rescale_noise_cfg=rescale_noise_cfg,
                 StableDiffusionPipeline=StableDiffusionPipeline, DDIMScheduler=DDIMScheduler,
                 DDIMInverseScheduler=DDIMInverseScheduler, DPMSolverMultistepScheduler=DPMSolverMultistepScheduler,
                 DPMSolverMultistepInverseScheduler=DPMSolverMultistepInverseScheduler)
    r.definitions(r.UPSTREAM / 'utils/imprint_utils.py',
                  {'load_attacker_pipe', 'DiffPipe', 'UnetWrapper', 'pixel_to_latent', 'latent_to_pil', 'invert_image'}, scope)
    pixel_scope = dict(typing=typing, torch=torch, Image=PIL.Image, transforms=transforms)
    r.definitions(r.UPSTREAM / 'utils/image_utils.py', {'PIL_to_torch'}, pixel_scope)
    start = time.perf_counter()
    pipe, scheduler, inverse = scope['load_attacker_pipe'](modelid=job['proxy']['snapshot'], scheduler='DDIM', device='cuda:0')
    diff = scope['DiffPipe'](pipe, scheduler=inverse, device='cuda:0')
    arrays = []
    guard(inverse, arrays)
    inverse.set_timesteps(50, device='cuda:0')
    torch.cuda.synchronize()
    load = time.perf_counter() - start
    start = time.perf_counter()
    with PIL.Image.open(path.parent / 'cover.png') as im:
        cover = im.convert('RGB')
    with PIL.Image.open(path.parent / 'reference.png') as im:
        reference = im.convert('RGB')
    z0_original = scope['pixel_to_latent'](cover, pipe)
    z0 = torch.nn.Parameter(z0_original.detach().clone())
    optim = torch.optim.Adam([z0], lr=.01)
    generated_pt = pixel_scope['PIL_to_torch']([reference], dtype=torch.float32, device='cuda:0')
    if generated_pt.min() < 0 or generated_pt.max() > 1:
        raise ValueError('Official reference RGB01 path violated')
    with torch.no_grad():
        target = scope['invert_image'](pipe=pipe, image_pt=generated_pt, scheduler=inverse, num_inference_steps=50)
    r.tensor(out / 'initial_z0.npy', z0)
    r.tensor(out / 'reference_proxy_inverse.npy', target)
    torch.cuda.synchronize()
    preparation = time.perf_counter() - start
    c.write(out / 'parameter_flags.json', {name: dict(total=sum(1 for _ in model.parameters()),
                                                      requires_grad=sum(p.requires_grad for p in model.parameters()))
                                           for name, model in [('unet', pipe.unet), ('vae', pipe.vae), ('text_encoder', pipe.text_encoder)]})
    code = update_code(r.UPSTREAM / 'run_imprint_forgery.py')
    state = dict(optim=optim, diffpipe=diff, z0=z0, torch=torch, zT_retrieved=target)
    history = []
    images = {}
    point_records = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    updates = int(job['native']['actual_adam_updates'])
    for actual in range(1, updates + 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        exec(code, state)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        if not torch.isfinite(z0).all() or not torch.isfinite(state['l']):
            raise RuntimeError('Nonfinite original update')
        history.append(dict(actual_update=actual, official_loop_index=actual - 1,
                            loss_before_update=float(state['l'].detach()), update_seconds=elapsed))
        print('NATIVE_UPDATE', actual, 'LOSS', history[-1]['loss_before_update'], flush=True)
        if actual in POINTS:
            tick = time.perf_counter()
            latent = r.tensor(out / ('update_' + str(actual) + '_latent.npy'), z0)
            image = scope['latent_to_pil'](z0, pipe)[0]
            with (out / ('update_' + str(actual) + '.png')).open('xb') as f:
                image.save(f, format='PNG')
            with PIL.Image.open(out / ('update_' + str(actual) + '.png')) as im:
                if not np.array_equal(np.asarray(im), np.asarray(image)):
                    raise ValueError('PNG roundtrip changed')
            avg = r.tensor(out / ('update_' + str(actual) + '_adam_avg.npy'), optim.state[z0]['exp_avg'])
            sq = r.tensor(out / ('update_' + str(actual) + '_adam_sq.npy'), optim.state[z0]['exp_avg_sq'])
            if int(optim.state[z0]['step']) != actual:
                raise ValueError('Optimizer step fencepost')
            images[str(actual)] = c.png_record(out / ('update_' + str(actual) + '.png'))
            point_records[str(actual)] = dict(latent=latent, adam_avg=avg, adam_sq=sq, adam_step=actual,
                                              export_seconds=time.perf_counter() - tick)
    torch.cuda.synchronize()
    optimization_and_exports = time.perf_counter() - start
    if c.sha(path) != job_hash:
        raise ValueError('Job changed')
    validate_job(path, cfg, d)
    c.write(out / 'optimization.json', dict(updates=history, checkpoints=point_records, scheduler_calls=arrays,
                                            reference_input_range='RGB01 exact official CLI', mask=None,
                                            target_scores_seen=0))
    final_png = out / ('update_' + str(updates) + '.png')
    with (out / 'candidate.png').open('xb') as f:
        f.write(final_png.read_bytes())
    frozen = dict(task_id=job['task_id'], job_sha256=job_hash, images=images)
    c.write(out / 'candidates_frozen.json', frozen)
    c.write(out / 'summary.json', dict(completed=True, actual_updates=updates, candidate_checkpoints=len(POINTS),
                                       timing=dict(model_load_seconds=load,
                                                   reference_and_cover_preparation_seconds=preparation,
                                                   optimization_and_export_seconds=optimization_and_exports,
                                                   core_seconds=preparation + optimization_and_exports,
                                                   peak_allocated_bytes=torch.cuda.max_memory_allocated()),
                                       images=images, target_scores_seen=0))
    print('COMPLETED_NATIVE_ATTACK', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])

