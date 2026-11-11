"""Continue once a predecessor exits successfully, using kernel exit events."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);args=p.parse_args()
    if not (args.predecessor/'complete.json').exists():
        cmd=Path(f'/proc/{args.pid}/cmdline').read_bytes()
        assert str(args.predecessor).encode() in cmd
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,args.pid,0)
        if fd<0:
            assert (args.predecessor/'complete.json').exists(),os.strerror(ctypes.get_errno())
        else:select.select([fd],[],[]);os.close(fd)
    assert json.loads((args.predecessor/'complete.json').read_text())['completed']
    command=[str(args.root/'envs/semantic_forgery_py310/bin/python'),str(args.run/'scripts/run_geometry_confirmation.py'),
             '--root',str(args.root),'--run',str(args.run)]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
