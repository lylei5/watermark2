"""Audit and plot a successful experiment after one kernel process-exit event."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    run = args.run
    if not (run / 'complete.json').exists():
        command = Path(f'/proc/{args.pid}/cmdline').read_bytes()
        assert str(run).encode() in command and b'run_geometry_confirmation.py' in command
        # Linux x86_64 pidfd_open; no repeated status queries.
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, args.pid, 0)
        if fd < 0:
            assert (run / 'complete.json').exists(), os.strerror(ctypes.get_errno())
        else:
            try:
                select.select([fd], [], [])
            finally:
                os.close(fd)
    assert json.loads((run / 'complete.json').read_text())['completed']
    here = Path(__file__).parent
    subprocess.run([sys.executable, str(here / 'audit_geometry_confirmation.py'), '--run', str(run)], check=True)
    subprocess.run([sys.executable, str(here / 'plot_frozen_confirmation.py'), '--summary',
                    str(run / 'audited_summary.json'), '--output', str(run / 'figures')], check=True)
    summary = json.loads((run / 'audited_summary.json').read_text())
    record = dict(completed=True, wait_mode='kernel process-exit event',
                  summary_sha256=hashlib.sha256((run / 'audited_summary.json').read_bytes()).hexdigest(),
                  analysis_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), here / 'audit_geometry_confirmation.py', here / 'plot_frozen_confirmation.py']})
    (run / 'analysis_complete.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(summary['summary'], indent=2))


if __name__ == '__main__':
    main()
