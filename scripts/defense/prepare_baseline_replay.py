"""Freeze a complete carrier-chain replay, retaining its numeric code and private inputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--after-run')
    parser.add_argument('--deterministic-backend', action='store_true')
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research/runs')
    source, target = [(root / name).resolve() for name in [args.source_run, args.run_name]]
    assert source.parent == target.parent == root
    state = json.loads((source / 'pipeline_state.json').read_text())
    assert state['status'] in ['complete', 'failed']
    plan = json.loads((source / 'protocol.json').read_text())
    assert plan['worker'] in ['run_sembind_prc_native.py', 'run_sembind_tr_native.py']
    target.mkdir(exist_ok=False)
    for name, digest in plan['local_sha256'].items():
        assert sha(source / name) == digest, name
        shutil.copy2(source / name, target / name)
    shutil.copy2(__file__, target / Path(__file__).name)
    # The model worker and private inputs stay frozen; resource-release handling can advance.
    coordinator = Path(__file__).parent / 'run_baseline_after_exit.py'
    shutil.copy2(coordinator, target / coordinator.name)
    plan['coordinator_sha256'] = sha(target / coordinator.name)
    plan['local_sha256'][coordinator.name] = plan['coordinator_sha256']
    plan['local_sha256'][Path(__file__).name] = sha(target / Path(__file__).name)
    if args.deterministic_backend:
        worker = Path(__file__).parent / plan['worker']
        shutil.copy2(worker, target / worker.name)
        plan['worker_sha256'] = sha(target / worker.name)
        plan['local_sha256'][worker.name] = plan['worker_sha256']
        plan['deterministic_backend'] = True
    plan['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan['replay_of'] = dict(run=str(source), protocol_sha256=sha(source / 'protocol.json'),
                             state_sha256=sha(source / 'pipeline_state.json'),
                             same_official_numeric_code_private_keys_inputs=True,
                             worker_bytes_unchanged=plan['worker_sha256'] == sha(source / plan['worker']),
                             deterministic_backend_added=args.deterministic_backend,
                             complete_cohort_repeated=True)
    after = (root / args.after_run).resolve() if args.after_run else source
    assert after.parent == root
    predecessor = json.loads((after / 'pipeline_state.json').read_text())
    file = Path(f"/proc/{predecessor['pid']}/cmdline")
    command = file.read_bytes() if file.exists() else b''
    assert not command or str(after).encode() in command, 'Predecessor PID reused'
    plan['predecessors'] = [dict(pid=predecessor['pid'], cmdline_sha256=hashlib.sha256(command).hexdigest())] * 3
    (target / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(run=str(target), protocol_sha256=sha(target / 'protocol.json'))))


if __name__ == '__main__':
    main()
