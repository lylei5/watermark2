"""Paired native/PGID and content verification of the published Reprompt+ budget."""
import argparse
import ctypes
import os
from pathlib import Path
import select
import shutil
import subprocess
import time

import evaluate_author_reprompt as evaluation

ROOT=evaluation.ROOT
ATTACK=ROOT/'runs/author_reprompt_plus_20261115'
read,write,sha=evaluation.read,evaluation.write,evaluation.sha


def prepare(run):
    run.mkdir(exist_ok=False)
    for name in ['evaluate_reprompt_plus.py','evaluate_author_reprompt.py','copynce_control.py']:
        shutil.copy2(Path(__file__).with_name(name),run/name)
    write(run/'protocol.json',dict(created_at=time.time(),attack=str(ATTACK),attack_protocol_sha256=sha(ATTACK/'protocol.json'),
        scripts={p.name:sha(p) for p in run.glob('*.py')},native_decisions=480,content_candidates=240,
        inversion='Unchanged author native GS/TR DDIM50 and PGID-F50, published per-carrier parameters.',
        thresholds='OriginalGS BA>181/256; originalTR p threshold; content uses frozenGS30 calibration thresholds.',
        aggregation='Same candidate must pass both content recovery and native detection; then any candidate within fixed budget.',
        full_reference_denominator=True,attack_modified=False,allowed_gpu_indices=[7,5,4],excluded_gpu_indices=[6],
        scope='10sources perGS/TR and two proxies, complete9/3candidate budget; noPRCextension or paper-scale claim.'))
    print({'run':str(run),'protocol_sha256':sha(run/'protocol.json')},flush=True)


def wait_attacks():
    for job in read(ATTACK/'launch.json')['jobs']:
        # A written completion file can precede CUDA process teardown.
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,job['pid'],0)
        if fd>=0:
            command=Path(f"/proc/{job['pid']}/cmdline").read_bytes()
            if command:
                assert str(ATTACK).encode() in command
            select.select([fd],[],[]);os.close(fd)
        else:assert ctypes.get_errno()==3
    queries=[]
    for model in ['sd21','sd15']:
        done=read(ATTACK/model/'complete.json');assert done['status']=='complete' and done['count']==120
        assert sha(ATTACK/model/'rows.json')==done['rows_sha256']
        queries.extend(read(ATTACK/model/'rows.json'))
    assert len(queries)==240 and len({q['trial'] for q in queries})==240
    return queries


def pipeline(run):
    state=dict(status='waiting_for_attack_exit',pid=os.getpid(),started_at=time.time(),protocol_sha256=sha(run/'protocol.json'))
    write(run/'state.json',state)
    try:
        write(run/'queries.json',wait_attacks())
        state['status']='evaluating';write(run/'state.json',state)
        jobs=[]
        for mode,carrier,gpu in [('native','GS',7),('native','TR',5),('content',None,4)]:
            log=open(run/((carrier or mode)+'.log'),'xb')
            args=[str(evaluation.PYTHON),str(run/Path(__file__).name),mode,'--run',str(run),'--gpu',str(gpu)]
            if carrier:args+=['--carrier',carrier]
            p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);log.close();jobs.append((carrier or mode,p))
        write(run/'workers.json',[dict(role=name,pid=p.pid) for name,p in jobs])
        exits={name:p.wait() for name,p in jobs};assert not any(exits.values()),exits
        state.update(status='complete',native_rows=480,content_rows=240,seconds=time.time()-state['started_at'])
        write(run/'complete.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pipeline','content','native']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gpu',type=int);p.add_argument('--carrier',choices=['GS','TR']);a=p.parse_args()
    assert a.run.resolve().parent==ROOT/'runs'
    if a.mode=='prepare':prepare(a.run)
    else:
        plan=read(a.run/'protocol.json')
        for name,digest in plan['scripts'].items():assert sha(a.run/name)==digest
        assert sha(ATTACK/'protocol.json')==plan['attack_protocol_sha256']
        evaluation.ATTACK=ATTACK
        if a.mode=='pipeline':pipeline(a.run)
        elif a.mode=='content':evaluation.content(a.run,a.gpu)
        else:evaluation.native(a.run,a.carrier,a.gpu)
