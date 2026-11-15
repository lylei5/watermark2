"""Detached frozen pipeline and Linux process-exit notification; no polling."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main(action,run):
    if action=='launch':
        assert not (run/'launch.json').exists()
        with (run/'worker.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run',str(run)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,started_at=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completion='inotify event on exit.json')
        write(run/'launch.json',value);print(json.dumps(value));return
    if action=='wait':
        library=ctypes.CDLL(None,use_errno=True)
        descriptor=library.inotify_init1(0)
        if descriptor<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        watch=library.inotify_add_watch(descriptor,os.fsencode(run),8|128)
        if watch<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        write(run/'completion_observer.json',dict(mechanism='inotify close-write/moved-to event; no periodic checks',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        try:
            while not (run/'exit.json').exists():
                select.select([descriptor],[],[]);os.read(descriptor,65536)
            print((run/'exit.json').read_text())
        finally:os.close(descriptor)
        return
    started=time.time();exit_code=0
    try:
        for stage in ['controls','execute']:
            subprocess.run([sys.executable,str(run/'run_order_recovery.py'),stage,'--run',str(run)],check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_order_recovery.py')),'--run',str(run)],check=True)
    except BaseException as exc:
        exit_code=1;print(repr(exc),flush=True)
    finally:write(run/'exit.json',dict(exit_code=exit_code,seconds=time.time()-started,finished_at=time.time()))
    sys.exit(exit_code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','worker','wait']);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();main(args.action,args.run)
