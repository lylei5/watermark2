"""Wait for frozen generation processes by exit events, then run CPU analysis."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    state=json.loads((run/'pipeline_state.json').read_text())
    assert state['stage']=='generate' and state['status']=='running'
    libc=ctypes.CDLL(None,use_errno=True);waited=[]
    for pid in state['worker_pids']:
        fd=libc.syscall(434,pid,0)
        if fd<0:
            if ctypes.get_errno()!=3:raise OSError(ctypes.get_errno(),'pidfd_open')
        else:
            select.select([fd],[],[]);os.close(fd)
        waited.append(pid)
    for shard in range(3):assert json.loads((run/f'generate_{shard}/complete.json').read_text())['status']=='complete'
    scripts=Path(__file__).parent
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for script in ['measure_phase1_quality.py','diagnose_phase1_semantic_votes.py','plot_phase1_trajectory.py']:
        source=run/script if (run/script).exists() else scripts/script
        subprocess.run([sys.executable,str(source),'--run',str(run)],check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='4'))
        print('CPU_STAGE_COMPLETE',script,flush=True)
    (run/'postprocess_complete.json').write_text(json.dumps(dict(status='complete',waited_processes=waited,mechanism='pidfd process-exit events; no status polling'),indent=2)+'\n')


if __name__=='__main__':main()
