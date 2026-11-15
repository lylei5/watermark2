"""Finish the frozen native budget with fixed source shards and exit events."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time

ROOT = Path('/data0/liyanlei/watermark_research')
OLD = ROOT/'runs/reprompt_plus_evaluation_20261115'
RUN = ROOT/'runs/reprompt_plus_evaluation_20261115_v2'
ASSIST = ROOT/'runs/reprompt_plus_native_assist_20261115'
PYTHON = ROOT/'envs/pgid_revalidation_py310/bin/python'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def process_fd(pid, expected):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        assert ctypes.get_errno() == 3
        return None
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    if command:
        assert expected.encode() in command, (pid, command)
    return fd


def wait_fd(fd):
    if fd is not None:
        select.select([fd], [], [])
        os.close(fd)


def stop_owned(pid, expected):
    fd = process_fd(pid, expected)
    if fd is not None:
        if not select.select([fd], [], [], 0)[0]:
            os.kill(pid, signal.SIGTERM)
        wait_fd(fd)


def prepare():
    RUN.mkdir(exist_ok=False)
    ASSIST.mkdir(exist_ok=False)
    previous = read(OLD/'protocol.json')
    for name, digest in previous['scripts'].items():
        assert sha(OLD/name) == digest
        shutil.copy2(OLD/name, RUN/name)
        shutil.copy2(OLD/name, ASSIST/name)
    shutil.copy2(__file__, RUN/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('evaluate_reprompt_plus_recovery.py'), RUN/'evaluate_reprompt_plus_recovery.py')
    queries = read(OLD/'queries.json')
    write(RUN/'queries.json', queries)
    write(ASSIST/'queries.json', [q for q in queries if q['carrier']=='GS' and q['task']>=6])
    assert len(read(ASSIST/'queries.json')) == 72
    for name in ['content_rows.json', 'features.npz', 'content_complete.json']:
        shutil.copy2(OLD/name, RUN/name)
        assert sha(RUN/name) == sha(OLD/name)
    for name, digest in read(RUN/'content_complete.json')['artifacts'].items():
        assert sha(RUN/name) == digest
    plan = dict(previous, created_at=time.time(), predecessor=str(OLD),
        predecessor_protocol_sha256=sha(OLD/'protocol.json'),
        recovery_reason='TR idle guard rejected GPU5 utilization51 before model loading; retry unchanged on idle GPU5.',
        tr_failure_log_sha256=sha(OLD/'TR.log'),
        gs_shards={'original_tasks':list(range(6)), 'assist_tasks':list(range(6,10)),
                   'original_pid':26006, 'assist_gpu':4, 'carrier_material':'Author fixed key/nonce/message indexed by source offset; each query reseeds inversion.'},
        coordinator_sha256=sha(RUN/Path(__file__).name),
        recovery_source_sha256=sha(RUN/'evaluate_reprompt_plus_recovery.py'),
        reuse_content_sha256=sha(RUN/'content_complete.json'),
        selection='Fixed source indices, independent of acceptance. Verify duplicate decisions and complete trial coverage before merge.')
    write(RUN/'protocol.json', plan)
    write(ASSIST/'protocol.json', dict(previous, native_decisions=144, content_candidates=0,
        source_tasks=list(range(6,10)), parent_protocol_sha256=sha(RUN/'protocol.json')))
    print(json.dumps({'run':str(RUN),'protocol_sha256':sha(RUN/'protocol.json')}), flush=True)


def execute():
    plan = read(RUN/'protocol.json')
    assert sha(__file__)==plan['coordinator_sha256']
    assert sha(RUN/'evaluate_reprompt_plus_recovery.py')==plan['recovery_source_sha256']
    state = dict(status='evaluating', pid=os.getpid(), started_at=time.time(), protocol_sha256=sha(RUN/'protocol.json'))
    write(RUN/'state.json', state)
    try:
        # These two project coordinators have no GPU work of their own.
        stop_owned(33165, str(OLD/'postprocess/waiter.py'))
        write(OLD/'postprocess/state.json', dict(status='superseded_before_recovery', successor=str(RUN)))
        stop_owned(16874, str(OLD/'evaluate_reprompt_plus.py'))
        old_state=read(OLD/'state.json')
        old_state.update(status='partial', successor=str(RUN), reason=plan['recovery_reason'])
        write(OLD/'state.json',old_state)
        original_fd=process_fd(26006, str(OLD/'evaluate_reprompt_plus.py'))
        jobs=[]
        for run,carrier,gpu in [(RUN,'TR',5),(ASSIST,'GS',4)]:
            log=open(run/(carrier+'.log'),'xb')
            p=subprocess.Popen([str(PYTHON),str(run/'evaluate_reprompt_plus.py'),'native','--run',str(run),'--gpu',str(gpu),'--carrier',carrier],stdout=log,stderr=subprocess.STDOUT)
            log.close();jobs.append((carrier,p))
        write(RUN/'workers.json',[dict(role=name,pid=p.pid) for name,p in jobs]+[dict(role='GS_original',pid=26006)])
        exits={name:p.wait() for name,p in jobs}
        assert not any(exits.values()),exits
        queries=read(RUN/'queries.json')
        expected_prefix={(q['trial'],d) for q in queries if q['carrier']=='GS' and q['task']<6 for d in [False,True]}
        prefix=read(OLD/'GS/rows.json')
        covered={(r['trial'],r['defended']) for r in prefix if r['task']<6}
        if covered==expected_prefix:
            if original_fd is not None and not select.select([original_fd],[],[],0)[0]:
                assert str(OLD/'evaluate_reprompt_plus.py').encode() in Path('/proc/26006/cmdline').read_bytes()
                os.kill(26006,signal.SIGTERM)
            state['original_stop_reason']='All fixed prefix decisions complete; tail completed by GPU4 shard.'
        wait_fd(original_fd)
        prefix=read(OLD/'GS/rows.json');tail=read(ASSIST/'GS/rows.json')
        done=read(ASSIST/'GS/complete.json')
        assert sha(ASSIST/'GS/rows.json')==done['rows_sha256'] and len(tail)==144
        duplicate={(r['trial'],r['defended']):r for r in prefix if r['task']>=6}
        overlap=[]
        for r in tail:
            other=duplicate.get((r['trial'],r['defended']))
            if other:
                assert other['accepted']==r['accepted'] and abs(other['metric']-r['metric'])<1e-6
                overlap.append(dict(trial=r['trial'],defended=r['defended'],metric_difference=abs(other['metric']-r['metric'])))
        rows=[r for r in prefix if r['task']<6]+tail
        expected={(q['trial'],d) for q in queries if q['carrier']=='GS' for d in [False,True]}
        assert len(rows)==360 and {(r['trial'],r['defended']) for r in rows}==expected
        for r in rows: assert sha(Path(r['latent']))==r['latent_sha256']
        (RUN/'GS').mkdir();write(RUN/'GS/rows.json',rows)
        write(RUN/'GS/complete.json',dict(done,rows=360,gpu=None,gpu_uuids=['GPU-9644fb82-cb6f-7311-8017-eaee41409bb7',done['gpu_uuid']],
            rows_sha256=sha(RUN/'GS/rows.json'),shards=[str(OLD/'GS'),str(ASSIST/'GS')],duplicate_checks=overlap,
            timing_scope='seconds inherited from assist shard only; coordinator reports elapsed completion time'))
        tr=read(RUN/'TR/rows.json');td=read(RUN/'TR/complete.json')
        assert len(tr)==120 and sha(RUN/'TR/rows.json')==td['rows_sha256']
        assert {(r['trial'],r['defended']) for r in tr}=={(q['trial'],d) for q in queries if q['carrier']=='TR' for d in [False,True]}
        for r in tr: assert sha(Path(r['latent']))==r['latent_sha256']
        write(RUN/'merge_audit.json',dict(original_rows=len(prefix),fixed_prefix_rows=216,assist_rows=144,duplicate_checks=overlap,
            original_rows_sha256=sha(OLD/'GS/rows.json'),assist_rows_sha256=sha(ASSIST/'GS/rows.json'),tr_rows=120,
            original_protocol_sha256=sha(OLD/'protocol.json'),queries_sha256=sha(RUN/'queries.json'),old_queries_sha256=sha(OLD/'queries.json')))
        state.update(status='complete',native_rows=480,content_rows=240,seconds=time.time()-state['started_at'])
        write(RUN/'complete.json',state);write(RUN/'state.json',state)
        output=ROOT/'runs/reprompt_plus_recovery_20261115'
        with open(RUN/'recovery.log','xb') as log:
            result=subprocess.run([str(PYTHON),str(RUN/'evaluate_reprompt_plus_recovery.py'),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
        write(RUN/'recovery_exit.json',dict(returncode=result.returncode,output=str(output)))
        assert result.returncode==0
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(RUN/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute']);a=p.parse_args()
    prepare() if a.mode=='prepare' else execute()
