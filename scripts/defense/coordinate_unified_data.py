"""Event-driven data acquisition, audit and first null-calibration stage on249."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--launch-seal', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / 'runs'
    run.mkdir(exist_ok=False)
    names = ['coordinate_unified_data.py', 'fetch_coco_archive.py', 'prepare_coco_nulls.py',
             'run_seal_null_calibration.py', 'seal_paper_simhash.py', 'empirical_calibration.py']
    for name in names:
        shutil.copy2(Path(__file__).parent / name, run / name)
    save(run / 'source_manifest.json', {name: sha(run / name) for name in names})
    state = dict(status='starting', pid=os.getpid(), launch_seal=args.launch_seal)
    save(run / 'state.json', state)
    def step(name, command):
        state.update(status=name)
        save(run / 'state.json', state)
        with (run / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save(run / 'state.json', state)
            result = child.wait()
        assert result == 0, (name, result)
    try:
        archive = ROOT / 'datasets/coco_test2017_official'
        if not (archive / 'complete.json').exists():
            previous = archive / 'state.json'
            if previous.exists():
                old = json.loads(previous.read_text())
                command = Path(f"/proc/{old['pid']}/cmdline")
                assert not command.exists() or b'fetch_coco_archive.py' not in command.read_bytes(), 'An existing downloader is still active'
            step('download', [sys.executable, str(run / 'fetch_coco_archive.py'), '--output', str(archive)])
        data = ROOT / 'datasets/unified_coco_nulls_20261115'
        if not (data / 'complete.json').exists():
            step('audit', [sys.executable, str(run / 'prepare_coco_nulls.py'), '--output', str(data)])
        receipt = json.loads((data / 'complete.json').read_text())
        assert receipt['status'] == 'complete'
        if args.launch_seal:
            target = ROOT / 'runs/unified_seal_nulls_20261115'
            step('prepare_seal', [sys.executable, str(run / 'run_seal_null_calibration.py'), '--run', str(target), '--prepare'])
            with (target / 'coordinator.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(target / 'run_seal_null_calibration.py'), '--run', str(target), '--coordinate'],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='data_complete_seal_dispatched', seal_run=str(target), seal_coordinator_pid=child.pid)
        else:
            state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(run / 'state.json', state)


if __name__ == '__main__':
    main()
