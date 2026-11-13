"""Wait for one native coordinator's exit event and audit its frozen artifacts."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', choices=['seal', 'prc_s', 'tr_s', 'pgid', 'seal_benign', 'determinism'], required=True)
    parser.add_argument('--auditor', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    run = args.run.resolve()
    assert run.parent == root / 'runs'
    state = json.loads((run / 'pipeline_state.json').read_text())
    if state['status'] in ['queued', 'running']:
        assert os.uname().machine == 'x86_64'
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.syscall(434, state['pid'], 0)
        if descriptor < 0:
            if ctypes.get_errno() != 3:
                raise OSError(ctypes.get_errno(), 'pidfd_open')
        else:
            try:
                command = Path(f"/proc/{state['pid']}/cmdline")
                value = command.read_bytes() if command.exists() else b''
                assert not value or str(run).encode() in value, 'Coordinator PID reused'
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    state = json.loads((run / 'pipeline_state.json').read_text())
    assert state['status'] == 'complete', state
    receipt = dict(pipeline_state_sha256=hashlib.sha256((run / 'pipeline_state.json').read_bytes()).hexdigest(),
                   auditor=str(args.auditor.resolve()),
                   auditor_sha256=hashlib.sha256(args.auditor.read_bytes()).hexdigest())
    print(json.dumps(receipt), flush=True)
    subprocess.run([str(root / 'envs/semantic_forgery_py310/bin/python'), str(args.auditor),
                    '--run', str(run), '--method', args.method], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4'))


if __name__ == '__main__':
    main()
