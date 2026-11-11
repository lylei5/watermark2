"""Attach defense scoring to completion of a published-baseline process."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent-pid',type=int,required=True)
    args=p.parse_args();run=args.run.resolve();plan=json.loads((run/'protocol.json').read_text())
    marker=Path(plan['baseline_run'])/'verified_results.json'
    model_hash=sha(run/'frozen_models.json')
    def state(status,**kw):
        (run/'pipeline_state.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),models_sha256=model_hash,**kw),indent=2)+'\n')
    state('waiting',parent_pid=args.parent_pid,mode='kernel process-exit event; no polling')
    if not marker.exists():
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,args.parent_pid,0)
        if fd<0:
            if not marker.exists():raise OSError(ctypes.get_errno(),'Cannot attach baseline exit event')
        else:
            select.select([fd],[],[]);os.close(fd)
    try:
        assert marker.exists(),'Baseline did not produce an audited completion'
        assert sha(run/'frozen_models.json')==model_hash
        state('running',stage='defense_evaluation')
        subprocess.run([sys.executable,str(run/'run_conditional_content_experiment.py'),'--run',str(run),'--stage','evaluate'],check=True)
        state('complete',stage='defense_evaluation')
    except Exception as exc:
        state('failed',error=repr(exc));raise


if __name__=='__main__':main()
