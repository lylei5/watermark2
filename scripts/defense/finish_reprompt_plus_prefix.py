"""Close a redundant worker after fixed shards finish; no progress polling."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import time

ROOT=Path('/data0/liyanlei/watermark_research/runs')
RUN=ROOT/'reprompt_plus_evaluation_20261115_v2'
OLD=ROOT/'reprompt_plus_evaluation_20261115'
ASSIST=ROOT/'reprompt_plus_native_assist_20261115'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fd_for(pid,expected):
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
    if fd<0:
        assert ctypes.get_errno()==3
        return None
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if command:assert expected.encode() in command
    return fd


def main():
    state=dict(source_sha256=sha(Path(__file__)),started_at=time.time(),pid=os.getpid(),
        policy='Wait for both shard process exits, one180s grace event, one coverage check; no repeated polling.',
        original_pid=26006,fixed_source_prefix=list(range(6)))
    try:
        original=fd_for(26006,str(OLD/'evaluate_reprompt_plus.py'))
        for role,folder in [('TR',RUN),('GS',ASSIST)]:
            pid=next(j['pid'] for j in read(RUN/'workers.json') if j['role']==role)
            fd=fd_for(pid,str(folder/'evaluate_reprompt_plus.py'))
            if fd is not None:select.select([fd],[],[]);os.close(fd)
            done=read(folder/role/'complete.json')
            assert done['status']=='complete' and sha(folder/role/'rows.json')==done['rows_sha256']
        if original is None or select.select([original],[],[],180)[0]:
            state['status']='original_already_exited'
        else:
            expected={(q['trial'],d) for q in read(RUN/'queries.json') if q['carrier']=='GS' and q['task']<6 for d in [False,True]}
            rows=[r for r in read(OLD/'GS/rows.json') if r['task']<6]
            covered={(r['trial'],r['defended']) for r in rows}
            state.update(expected_rows=len(expected),covered_rows=len(rows))
            if covered==expected and len(rows)==216:
                for r in rows:assert sha(Path(r['latent']))==r['latent_sha256']
                assert str(OLD/'evaluate_reprompt_plus.py').encode() in Path('/proc/26006/cmdline').read_bytes()
                os.kill(26006,signal.SIGTERM);select.select([original],[],[])
                state['status']='fixed_prefix_completed_redundant_tail_stopped'
            else:state['status']='prefix_incomplete_original_left_running'
        if original is not None:os.close(original)
    except BaseException as exc:
        state.update(status='failed_without_fallback_termination',error=repr(exc));raise
    finally:
        state['finished_at']=time.time()
        (RUN/'prefix_completion_event.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
