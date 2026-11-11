"""Finish the fixed three-GPU experiment with separate full verification."""
import json
import os
from pathlib import Path
import subprocess
import ctypes
import platform
import select

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/published_imprint_dpm_full_20261110'
PYTHON=ROOT/'envs/semantic_forgery_py310/bin/python'


def exit_event(pid):
    # Linux pidfd_open (x86_64 syscall 434) gives an exit event for an existing
    # worker without periodic process or filesystem polling.
    assert platform.machine() == 'x86_64'
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return fd


def save(name,data):
    path=RUN/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    launches={i:json.loads((RUN/f'launch_{i}.json').read_text()) for i in range(3)}
    assert [launches[i]['gpu_index'] for i in range(3)]==[7,5,4]
    processes={};finished=set();failures={}
    events={exit_event(v['pid']):(i,'generation') for i,v in launches.items()}
    while events:
        save('pipeline_status.json',dict(finished_shards=sorted(finished),failed_shards=failures,
            evaluator_pids={str(k):v.pid for k,v in processes.items()},max_concurrent_gpus=3,
            gpu_indices=[7,5,4],completion_mode='kernel process-exit events; no periodic polling'))
        ready,_,_=select.select(list(events),[],[])
        for fd in ready:
            i,stage=events.pop(fd);os.close(fd);launch=launches[i]
            if stage=='evaluation':
                result=processes[i].wait()
                if result==0 and (RUN/'evaluation'/f'shard_{i}'/'results.json').is_file():finished.add(i)
                else:failures[i]=dict(stage='evaluation',returncode=result)
                continue
            marker=RUN/'outputs'/f'shard_{i}_complete.json'
            if not marker.is_file():
                failures[i]=dict(stage='generation',reason='worker exited without full completion');continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=launch['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache/huggingface'))
            cmd=[str(PYTHON),str(RUN/'evaluator/evaluate_published_imprint_full.py'),'--root',str(ROOT),'--run',str(RUN),
                 '--identity-root',str(ROOT/'private_eval/published_imprint_dpm_full_20261110'),'--shard',str(i)]
            with (RUN/f'evaluator_{i}.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[i]=proc;events[exit_event(proc.pid)]=(i,'evaluation')
            save(f'evaluator_launch_{i}.json',dict(pid=proc.pid,command=cmd,gpu_index=launch['gpu_index']))
            print('EVALUATOR_LAUNCHED',i,proc.pid,flush=True)
    save('pipeline_status.json',dict(finished_shards=sorted(finished),failed_shards=failures,
        max_concurrent_gpus=3,gpu_indices=[7,5,4],completion_mode='kernel process-exit events'))
    if failures:
        save('pipeline_failed.json',dict(failures=failures,completed=False));return
    rows=[]
    for i in range(3):rows.extend(json.loads((RUN/'evaluation'/f'shard_{i}'/'results.json').read_text())['rows'])
    assert len(rows)==144
    summary=[]
    for method in ['sembind','gs']:
        for kind in ['reference','cover','candidate']:
            group=[r for r in rows if r['method']==method and r['kind']==kind and (kind!='candidate' or r['update']==151)]
            assert len(group)==4
            summary.append(dict(method=method,kind=kind,n=4,accepted=sum(r['accepted'] for r in group),
                mean_ba=sum(r['ba'] for r in group)/4,
                **({m:sum(r[m] for r in group)/4 for m in ['psnr','ssim','ms_ssim','lpips']} if kind=='candidate' else {})))
    save('complete.json',dict(completed=True,full_update_jobs=8,updates_per_job=151,checkpoint_images=128,
        evaluated_images=144,summary=summary,rows=rows,scope='Small fixed development comparison; not paper-scale confirmation'))
    print('FULL_PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':main()
