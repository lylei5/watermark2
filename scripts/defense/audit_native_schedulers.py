"""Trace released scheduler grids and PGID's full network-call schedule on249 CPU."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['seal', 'sembind_pgid'], required=True)
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name == 'posix'
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from diffusers import PNDMScheduler, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
    torch.set_num_threads(4)
    model = root / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
    config = json.loads((model / 'scheduler/scheduler_config.json').read_text())
    output = dict(stage=args.stage, diffusers=importlib.metadata.version('diffusers'),
                  scope='CPU scheduler and released PGID control-flow trace with a deterministic stand-in noise predictor; no image model or performance result')
    if args.stage == 'seal':
        scheduler = PNDMScheduler.from_config(config)
        scheduler.set_timesteps(50)
        output.update(requested_steps=50, actual_timesteps=len(scheduler.timesteps),
                      generation_timesteps=scheduler.timesteps.tolist(), inversion_timesteps=list(reversed(scheduler.timesteps.tolist())))
        assert output['actual_timesteps'] == 51
    else:
        dpm_kwargs = dict(beta_end=.012, beta_schedule='scaled_linear', beta_start=.00085,
                          num_train_timesteps=1000, prediction_type='epsilon', steps_offset=1, solver_order=1)
        file = root / 'runs/sembind_chain_revalidation_20261111/scheduling_dpmsolver_multistep.py'
        spec = importlib.util.spec_from_file_location('diffusers.schedulers.native_legacy_dpm', file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output['sembind_scheduler_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        grids = {}
        for name, cls, kwargs in [('SemBind_DPM', module.DPMSolverMultistepScheduler, dpm_kwargs),
                                  ('PGID_DDIM_forward', DDIMScheduler, config),
                                  ('PGID_DDIM_inverse', DDIMInverseScheduler, config),
                                  ('PGID_PRC_DPM_forward', DPMSolverMultistepScheduler, dict(config, **dpm_kwargs)),
                                  ('PGID_PRC_DPM_inverse', DPMSolverMultistepInverseScheduler, dict(config, **dpm_kwargs))]:
            scheduler = cls.from_config(kwargs)
            scheduler.set_timesteps(50)
            grids[name] = scheduler.timesteps.tolist()
        output['grids'] = grids
        output['sembind_inverse_network_timesteps'] = list(reversed(grids['SemBind_DPM']))
        output['sembind_inverse_conversion_timesteps'] = [t - 20 for t in output['sembind_inverse_network_timesteps']]
        sys.path.insert(0, str(root / 'third_party/frozen/PGID_dc4ad96'))
        from pgid_defense import PGIDPipeline

        class TracePipe:
            device = torch.device('cpu')

            def __init__(self):
                self.trace = []

            def get_dtype(self):
                return torch.float32

            def imgs_to_latents(self, image):
                return torch.ones(1, 4, 2, 2)

            def prepare_null_prompt_embeds(self, latent):
                return torch.zeros(1, 1, 1)

            def predict_noise_with_null(self, latent, timestep, conditioning):
                self.trace.append(int(timestep))
                return .01 * latent

        traces = []
        for name, k, skip, gamma, expected in [('F_GS_TR_T2SMark', 15, 3, .001, 298),
                                               ('F_PRC', 7, 3, .001, 138), ('R', 10, 1, .045, 200)]:
            provider = TracePipe()
            defense = PGIDPipeline(provider, config, stopping_timestep=k, num_skip_inversion=skip, guidance_strength=gamma)
            z = defense.run_defense(None)['zT']
            assert torch.isfinite(z).all() and len(provider.trace) == expected
            assert provider.trace[:50] == provider.trace[-50:] == grids['PGID_DDIM_inverse']
            traces.append(dict(variant=name, k=k, skip=skip, gamma=gamma,
                               network_calls=len(provider.trace), timesteps=provider.trace, finite_output=True))
        output['pgid_traces'] = traces
    output['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = root / 'runs/three_baselines_20261113' / ('scheduler_' + args.stage + '.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['grids', 'pgid_traces', 'generation_timesteps', 'inversion_timesteps']}))


if __name__ == '__main__':
    main()
