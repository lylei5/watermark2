"""Production child gate, executed before loading any CUDA-capable model code."""
import argparse
import os
from pathlib import Path
import runpy
import sys

import launcher as launch
import scheduler as s


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'plan', 'plan-sha256', 'job', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu-index', required=True, type=int)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    launch.check_host(project)
    plan, devices, _, jobs, phases = launch.load_plan(project, args.plan, args.plan_sha256)
    device = next(d for d in devices if d.index == args.gpu_index)
    expected_env = dict(CUDA_VISIBLE_DEVICES=device.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise ValueError('Child GPU/offline environment differs')
    output = Path(args.output).resolve()
    if output != project / 'runs' / plan['run_name'] / args.job or not output.is_dir():
        raise ValueError('Child output differs from its reserved directory')
    observations = []
    idle = s.verify_idle(device, launch.sample, observations.append)
    s.write_once(output / 'child_preimport.json', dict(idle=idle, observations=observations))
    if not idle:
        raise RuntimeError('Child device is busy; model import refused')
    if phases[args.job] == 'evaluation':
        seal = launch.read(output.parent / 'candidates_sealed.json')
        for name, digest in seal['candidates'].items():
            if launch.sha(output.parent / name / 'candidate.png') != digest:
                raise ValueError('Candidate changed before evaluation import')
    spec = jobs[args.job]
    entry = launch.local(project, spec['entry'])
    replacements = {'{output}': str(output), '{parameters}': str(launch.local(project, spec['parameters'])),
                    '{project}': str(project), '{run}': str(output.parent)}
    argv = [replacements.get(arg, arg) for arg in spec['argv']]
    s.write_once(output / 'entry_command.json', dict(entry=str(entry), argv=argv,
                                                  plan_sha256=args.plan_sha256, role=phases[args.job]))
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry)] + argv
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
