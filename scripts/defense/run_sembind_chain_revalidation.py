"""Fresh staged SemBind/GS generation and full inversion diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'execution_protocol.json').read_text())
    paths = json.loads((run / 'paths.json').read_text())
    out = Path(paths['output'])
    state = run / 'state.json'
    with state.open('x') as f:
        json.dump({'status': 'preflight'}, f)
    record = dict(status='preflight', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(run / 'execution_protocol.json'), completed_stages=[])
    try:
        assert not (out / 'images').exists() and not (out / 'inversions').exists()
        for name, digest in plan['source_sha256'].items():
            assert sha(run / name) == digest, name
        assert sha(out / 'protocol.json') == plan['scientific_protocol_sha256']
        for name, digest in plan['semantic_assets_sha256'].items():
            assert sha(Path(paths['semantic']) / name) == digest, name
        for name, digest in plan['private_inputs_sha256'].items():
            assert sha(Path(paths['root']) / name) == digest, name
        info = subprocess.check_output(['nvidia-smi', '-i', '4',
                '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == 4 and int(info[2]) < 50 and int(info[3]) == 0, info
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=info[1], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   HF_HOME=plan['hf_home'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple/', PYTHONPATH='',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        base = [plan['python'], str(run / 'scripts/sembind_official_audit.py')]
        for key, value in paths.items():
            base += ['--' + key, value]
        base += ['--count', '4', '--semantic-only']
        record.update(status='running', gpu=4, gpu_uuid=info[1])
        write(state, record)
        stages = [('aux', 'aux'), ('codes', 'aux'), ('reference', 'aux'),
                  ('codes', 'reference'), ('invert', 'aux'), ('score', 'aux')]
        for stage, features in stages:
            name = stage + ('_' + features if stage == 'codes' else '')
            command = base + ['--stage', stage, '--features', features]
            with (run / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record.update(stage=name, child_pid=child.pid, command=command)
                write(state, record)
                result = child.wait()
            if result:
                raise RuntimeError(f'{name} failed with exit {result}; full log retained')
            record['completed_stages'].append(name)
            write(state, record)
        result = json.loads((out / 'results.json').read_text())
        assert len(result['rows']) == 18
        expected = {(i, m, s) for i in range(4) for m in ['sembind', 'gs'] for s in ['blind_encoder', 'blind_decoder']}
        expected |= {(0, m, 'oracle_decoder') for m in ['sembind', 'gs']}
        assert {(r['task'], r['method'], r['setting']) for r in result['rows']} == expected
        assert all(r['final_accepted'] == (r['final_ba'] > 181 / 256) for r in result['rows'])
        record.update(status='complete', results_sha256=sha(out / 'results.json'),
                      images=12, inversion_records=18, full_decoder_optimization_steps=100,
                      full_generation_and_inversion_steps=50, full_paper_reproduction=False)
        write(run / 'complete.json', record)
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['last_event_utc'] = datetime.now(timezone.utc).isoformat()
        write(state, record)


if __name__ == '__main__':
    main()
