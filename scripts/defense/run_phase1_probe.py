"""Process-exit driven generation, summary and exact-inversion verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items(): assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
    state=dict(status='running',pid=os.getpid(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=[])
    def save():
        temp=run/'pipeline_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(run/'pipeline_state.json')
    try:
        for stage in ['generate','verify']:
            children=[]
            for shard in range(3):
                with (run/f'{stage}_{shard}.log').open('w') as log:
                    children.append(subprocess.Popen([sys.executable,str(run/'run_phase1_probe_worker.py'),
                        '--run',str(run),'--stage',stage,'--shard',str(shard)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OPENBLAS_NUM_THREADS='4')))
            state.update(stage=stage,worker_pids=[c.pid for c in children]);save()
            codes=[]
            with ThreadPoolExecutor(max_workers=3) as pool:
                waiting={pool.submit(child.wait):i for i,child in enumerate(children)}
                for future in as_completed(waiting):
                    rc=future.result();codes.append(rc)
                    event=dict(stage=stage,shard=waiting[future],exit_code=rc,time=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    state['events'].append(event);save();print(json.dumps(event),flush=True)
            assert not any(codes),codes
            subprocess.run([sys.executable,str(run/'summarize_phase1_probe.py'),'--run',str(run),'--stage',stage],check=True)
            print('STAGE_COMPLETE',stage,flush=True)
        state.update(status='complete',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__': main()
