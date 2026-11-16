"""Compare frozen legacy and current extraction under deterministic backends on249."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data0/liyanlei/watermark_research')
PROBE = ROOT/'runs/unified_runtime_probe_20261115'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT/'runs' and not run.exists()
    run.mkdir()
    shutil.copy2(__file__, run/Path(__file__).name)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    idle = {int(a): b for a,b,c,d in [line.split(', ') for line in inventory.splitlines()]
            if int(a) != 6 and int(c) < 100 and int(d) == 0}
    gpus = [g for g in [7,5,4,3,2,1,0] if g in idle][:3]
    assert len(gpus) == 3
    template = json.loads((PROBE/'protocol.json').read_text())
    children = []
    plans = []
    wrapper = '''import json, os, runpy, subprocess, sys
from pathlib import Path
run=Path(__file__).parent
plan=json.loads((run/'protocol.json').read_text())
gpu=plan['gpu_indices']['sembind']
uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
os.environ.update(CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8')
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
assert not torch.cuda.is_initialized()
sys.argv=[str(run/'unified_runtime_probe.py'),'--run',str(run),'--method','sembind']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    for label, gpu in zip(['legacy', 'current_a', 'current_b'], gpus):
        target = ROOT/'runs'/f'{run.name}_{label}'
        target.mkdir()
        for name, digest in template['helper_sha256'].items():
            assert sha(PROBE/name) == digest
            shutil.copy2(PROBE/name, target/name)
        worker = PROBE/'unified_runtime_probe.py' if label == 'legacy' else Path(__file__).parent/'unified_runtime_probe.py'
        shutil.copy2(worker, target/'unified_runtime_probe.py')
        plan = dict(template, scope='Eight existing development inputs; fixed50 inversion steps and100 VAE updates; deterministic old/current implementation parity; no calibration or attacks',
                    worker_sha256=sha(target/'unified_runtime_probe.py'), deterministic_backend=True,
                    expected_unet_calls=dict(sembind=50))
        plan['gpu_indices'] = dict(template['gpu_indices'], sembind=gpu)
        save(target/'protocol.json', plan)
        (target/'deterministic_entry.py').write_text(wrapper)
        plans.append(dict(label=label, run=str(target), gpu=gpu, gpu_uuid=idle[gpu],
                          protocol_sha256=sha(target/'protocol.json'), entry_sha256=sha(target/'deterministic_entry.py')))
    save(run/'protocol.json', dict(created_at=datetime.now(timezone.utc).isoformat(), sources=8, plans=plans,
         script_sha256=sha(Path(__file__)), reference_protocol_sha256=sha(PROBE/'protocol.json'),
         tolerance=dict(atol=1e-5, rtol=1e-5, semantic_exact=True), attacks=0, gpu_cap=3))
    state = dict(status='running', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'), children=[])
    save(run/'pipeline_state.json', state)
    try:
        for plan in plans:
            target = Path(plan['run'])
            with (target/'worker.log').open('x') as log:
                child = subprocess.Popen([str(ROOT/'envs/semantic_forgery_py310/bin/python'), str(target/'deterministic_entry.py')],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OPENBLAS_NUM_THREADS='4'))
            children.append(child)
            state['children'].append(child.pid)
            save(run/'pipeline_state.json', state)
        state['exit_codes'] = [child.wait() for child in children]
        assert state['exit_codes'] == [0, 0, 0], state['exit_codes']
        import numpy as np
        comparison = []
        for index in range(8):
            for plan in plans[1:]:
                original = np.load(Path(plans[0]['run'])/'sembind'/f'{index:03}_sembind.npz')
                actual = np.load(Path(plan['run'])/'sembind'/f'{index:03}_sembind.npz')
                delta = np.abs(original['recovered'].astype(float)-actual['recovered'].astype(float))
                comparison.append(dict(source=index, candidate=plan['label'], maximum=float(delta.max()), mean=float(delta.mean()),
                    sign_changes=int((np.sign(original['recovered']) != np.sign(actual['recovered'])).sum()),
                    allclose=bool(np.allclose(original['recovered'], actual['recovered'], atol=1e-5, rtol=1e-5)),
                    semantic_equal=bool(np.array_equal(original['semantic_code'], actual['semantic_code']))))
        passed = all(r['allclose'] and r['semantic_equal'] for r in comparison)
        save(run/'parity.json', dict(passed=passed, comparisons=comparison, protocol_sha256=state['protocol_sha256']))
        assert passed, 'Deterministic legacy/current parity failed'
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(run/'pipeline_state.json', state)


if __name__ == '__main__':
    main()
