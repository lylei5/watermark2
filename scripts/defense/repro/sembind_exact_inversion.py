"""Reproduction check 2 for SemBind: official exact inversion on the paper's setting.

What this tests
---------------
SemBind's released code ships `inversion.py::exact_inversion`, built on
`InversableStableDiffusionPipeline` + DPM-Solver. It only exists for SD2.1, so it
cannot be applied to our SDXL target. A faithful reproduction therefore has to
move to the paper's own setting: target = proxy = SD2.1.

The paper's claimed self-check behaviour is:
  SemBind-bound reference  -> bound-state bit accuracy ~= 0.5 (chance)
                           -> unbound-state bit accuracy ~= 1.0
  vanilla GS reference     -> bound-state ~= 1.0, unbound-state ~= 0.5

Our earlier SemBind runs used this project's pinned DDIM 50-step inversion on
SDXL, and the official default is `inv_order=1` while the paper uses
`inv_order=0`. This script quantifies both by running the official pipeline.

Claim boundary
--------------
- The binding/unbinding algebra (S, pi, sigma) is still OUR reimplementation;
  the authors released only the masker and the inversion. So this reproduces the
  inversion component and the behavioural self-check, not the authors' full
  pipeline.
- One prompt, one image per arm: a component check, not a rate estimate.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'third_party/SemBind'))
sys.path.insert(0, str(ROOT / 'third_party/semantic-forgery'))
sys.path.insert(0, str(ROOT / 'scripts/defense'))

SD21 = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
PROMPT = 'A cat sitting on a wooden bench in a garden.'


def acc(wm, z):
    return float(wm.get_accuracies(z)['bit_accuracies'][0])


def main(out_path, inv_order='0', device='cuda:0'):
    # The released inversion.py imports `datasets` at module level only to build a
    # prompt loader. We always pass an explicit prompt, so that path is never
    # reached; stub it rather than modify third-party code.
    import types
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        import importlib.machinery
        stub = types.ModuleType('datasets')
        # transformers probes availability via importlib.util.find_spec, which
        # raises when __spec__ is None; give the stub a valid spec.
        stub.__spec__ = importlib.machinery.ModuleSpec('datasets', loader=None)
        stub.__version__ = '0.0.0-stub'

        def _unused(*a, **k):
            raise RuntimeError('datasets stub: get_dataset must not be called')

        stub.load_dataset = _unused
        sys.modules['datasets'] = stub

    from inversion import stable_diffusion_pipe, generate, exact_inversion
    from utils.wm.gs_provider import GsProvider
    from sembind import SemanticMasker

    inv_order = int(inv_order)
    pipe = stable_diffusion_pipe(solver_order=1, model_id=str(SD21), cache_dir=str(SD21))
    pipe.set_progress_bar_config(disable=True)
    masker = SemanticMasker(str(ROOT / 'models/sembind/dinov2-giant'),
                            str(ROOT / 'models/sembind/sembind_masker.pth'), device=device)
    wm = GsProvider(latent_shape=(1, 4, 64, 64), device=device, dtype=torch.float32,
                    message_width_in_bytes=32, num_replications=64, l=1, offset=0)
    zT = wm.get_wm_latents()['zT_torch']

    torch.manual_seed(7)
    aux_z = torch.randn(1, 4, 64, 64, device=device)
    aux, _, _ = generate(image_num=0, prompt=PROMPT, guidance_scale=7.5, num_inference_steps=50,
                         solver_order=1, pipe=pipe, init_latents=aux_z)
    aux.save(Path(out_path).parent / 'aux_clean.png')
    sign_ref = masker.sign_mask(aux)

    arms = {}
    for name, latents in (('sembind_gs', sign_ref * zT), ('vanilla_gs', zT)):
        img, _, _ = generate(image_num=0, prompt=PROMPT, guidance_scale=7.5, num_inference_steps=50,
                             solver_order=1, pipe=pipe, init_latents=latents)
        img.save(Path(out_path).parent / (name + '.png'))
        s_hat = masker.sign_mask(img)
        row = {}
        for order in sorted({inv_order, 0, 1}):
            z = exact_inversion(img, prompt=PROMPT, guidance_scale=7.5,
                                num_inference_steps=50, solver_order=1,
                                test_num_inference_steps=50, inv_order=order,
                                decoder_inv=True, model_id=str(SD21), pipe=pipe)
            row[f'inv_order_{order}'] = dict(
                ba_bound=acc(wm, z),
                ba_unbound=acc(wm, s_hat * z),
            )
        arms[name] = row
        print('ARM', name, json.dumps(row), flush=True)

    result = dict(
        check='sembind_official_exact_inversion_selfcheck',
        target_model='Manojb/stable-diffusion-2-1-base',
        revision='0094d483a120f3f33dafbd187ea4aa60d10de75c',
        prompt=PROMPT,
        inversion='official inversion.py::exact_inversion, decoder_inv=True, 50 steps',
        paper_inv_order=0,
        official_default_inv_order=1,
        binding_algebra='project reimplementation (authors released masker + inversion only)',
        arms=arms,
        n_images_per_arm=1,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT', json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
