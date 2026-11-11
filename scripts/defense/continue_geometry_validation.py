"""Wait once for the full pipeline exit, then run frozen validation stages."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import select
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def exit_event(pid):
    assert platform.machine()=='x86_64'
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return fd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--predecessor-pid',type=int,required=True)
    p.add_argument('--gpu-indices',type=int,nargs='+',choices=[7,5,4],default=[7,5,4])
    p.add_argument('--cache-python',type=Path)
    args=p.parse_args();root=args.root;run=args.run
    assert len(set(args.gpu_indices))==len(args.gpu_indices)<=3
    predecessor=root/'runs/published_imprint_dpm_full_20261110'
    python=root/'envs/semantic_forgery_py310/bin/python';scripts=run/'scripts'
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(root/'cache/huggingface'),
        PYTHONPATH=str(scripts),CUDA_VISIBLE_DEVICES='')
    try:
        if not (predecessor/'complete.json').exists():
            cmdline=Path(f'/proc/{args.predecessor_pid}/cmdline').read_bytes()
            if str(predecessor/'finish_published_imprint_full.py').encode() not in cmdline:
                raise RuntimeError('Predecessor PID does not identify the expected supervisor')
            try:fd=exit_event(args.predecessor_pid)
            except ProcessLookupError:
                if not (predecessor/'complete.json').exists():raise
            else:
                save(run/'continuation_status.json',dict(stage='waiting_for_full_pipeline_exit',pid=os.getpid(),predecessor_pid=args.predecessor_pid,
                    wait_mode='blocking kernel process-exit event; no periodic polling',gpu_indices=[7,5,4]))
                select.select([fd],[],[]);os.close(fd)
        if not (predecessor/'complete.json').exists():raise RuntimeError('Predecessor ended without complete results')
        save(run/'continuation_status.json',dict(stage='scoring_complete_baseline',pid=os.getpid()))
        cmd=[str(python),str(scripts/'score_frozen_geometry.py'),'--run',str(predecessor),'--mode','full_baseline',
            '--identity-root',str(root/'private_eval/published_imprint_dpm_full_20261110'),
            '--output',str(run/'full_baseline_scores'),'--frozen-protocol',str(run/'protocol.json')]
        with (run/'full_baseline_score.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        launches=[json.loads((predecessor/f'launch_{i}.json').read_text()) for i in range(3)]
        assert [x['gpu_index'] for x in launches]==[7,5,4]
        by_gpu={launch['gpu_index']:launch for launch in launches}
        def run_assigned(slot,gpu):
            codes=[]
            for i in range(slot,3,len(args.gpu_indices)):
                cmd=[str(args.cache_python or python),str(scripts/'cache_geometry_calibration.py'),'--run',str(run),'--root',str(root),
                    '--shard',str(i),'--gpu-index',str(gpu)]
                with (run/f'cache_{i}.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=by_gpu[gpu]['gpu_uuid']),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(run/f'cache_launch_{i}.json',dict(pid=proc.pid,gpu_index=gpu,command=cmd))
                code=proc.wait();codes.append(code)
                if code!=0:break
            return codes
        save(run/'continuation_status.json',dict(stage='caching_calibration',gpu_indices=args.gpu_indices,
            shards=3,scheduling='Sequential shards per assigned idle GPU; no progress polling'))
        with ThreadPoolExecutor(max_workers=len(args.gpu_indices)) as pool:
            futures=[pool.submit(run_assigned,i,gpu) for i,gpu in enumerate(args.gpu_indices)]
            codes=[code for future in futures for code in future.result()]
        if any(c!=0 for c in codes):raise RuntimeError(f'Calibration cache return codes: {codes}; no scoring performed')
        save(run/'continuation_status.json',dict(stage='scoring_calibration',pid=os.getpid()))
        cmd=[str(python),str(scripts/'score_frozen_geometry.py'),'--run',str(run),'--mode','calibration',
            '--identity-root',str(root/'private_eval/geometry_calibration_20261110'),
            '--output',str(run/'calibration_scores'),'--frozen-protocol',str(run/'protocol.json')]
        with (run/'calibration_score.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(run/'complete.json',dict(completed=True,calibration_sources=32,calibration_queries=192,
            calibration_identity_decisions=6144,full_baseline_queries=144,confirmation_scored=False))
        save(run/'continuation_status.json',dict(stage='complete',pid=os.getpid()))
    except Exception as exc:
        save(run/'failed.json',dict(completed=False,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
