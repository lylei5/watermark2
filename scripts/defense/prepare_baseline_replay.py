"""Freeze a complete replay of an ended PRC-S run, retaining its private inputs."""
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
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research/runs')
    source, target = [(root / name).resolve() for name in [args.source_run, args.run_name]]
    assert source.parent == target.parent == root
    state = json.loads((source / 'pipeline_state.json').read_text())
    assert state['status'] in ['complete', 'failed']
    assert not Path(f"/proc/{state['pid']}").exists(), 'Predecessor PID is still present or reused'
    plan = json.loads((source / 'protocol.json').read_text())
    assert plan['worker'] == 'run_sembind_prc_native.py'
    target.mkdir(exist_ok=False)
    for name, digest in plan['local_sha256'].items():
        assert sha(source / name) == digest, name
        shutil.copy2(source / name, target / name)
    shutil.copy2(__file__, target / Path(__file__).name)
    plan['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan['replay_of'] = dict(run=str(source), protocol_sha256=sha(source / 'protocol.json'),
                             state_sha256=sha(source / 'pipeline_state.json'),
                             same_numeric_code_private_keys_inputs=True,
                             complete_cohort_repeated=True)
    plan['predecessors'] = [dict(pid=state['pid'], cmdline_sha256=hashlib.sha256(b'').hexdigest())] * 3
    (target / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(run=str(target), protocol_sha256=sha(target / 'protocol.json'))))


if __name__ == '__main__':
    main()
