"""Execute a frozen independent confirmation with process-exit continuation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,data):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.root;run=args.run;plan=json.loads((run/'protocol.json').read_text())
    assert plan['evaluation_split']=='confirmation'
    assert not any((run/name).exists() for name in ['complete.json','status.json','failed.json'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             HF_HOME=str(root/'cache/huggingface'),PYTHONPATH=str(run/'scripts'))
    py=root/'envs/ddim_replay_py310/bin/python';primary=root/'envs/semantic_forgery_py310/bin/python'
    try:
        for name,digest in plan['source_sha256'].items():assert sha(run/'scripts'/name)==digest,name
        expected=plan['confirmation_queries']
        assert len(plan['rows'])==expected and plan['identity_tasks']==list(range(48,112))
        assert len({r['id'] for r in plan['rows']})==expected
        labels={r['label'] for r in plan['rows']}
        assert expected==64*len(labels)
        assert all({r['label'] for r in plan['rows'] if r['task']==task}==labels for task in plan['identity_tasks'])
        assert set(plan['source_image_ids']).isdisjoint(plan['excluded_source_image_ids'])
        for row in plan['rows']:assert sha(run/row['image'])==row['image_sha256']
        for task in plan['identity_tasks']:
            p=Path(plan['identity_root'])/'private'/f'{task:03}.json'
            assert p.is_file() and p.stat().st_mode&0o077==0
        devices=[]
        for index in plan['execution_gpus']:
            assert index in [7,5,4]
            raw=subprocess.check_output(['nvidia-smi','-i',str(index),
                '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            i,uuid,memory,util=[x.strip() for x in raw.strip().split(',')]
            assert int(i)==index and int(memory)<=50 and int(util)==0,'Assigned GPU is occupied'
            devices.append((index,uuid))
        assert 1<=len(devices)<=3 and len({i for i,_ in devices})==len(devices)
        write(run/'status.json',dict(stage='caching_confirmation',pid=os.getpid(),devices=devices,
                                     wait_mode='blocking child process exit; no progress polling'))
        def worker(slot,device):
            index,uuid=device
            for shard in range(slot,3,len(devices)):
                command=[str(py),str(run/'scripts/cache_geometry_calibration.py'),'--root',str(root),
                         '--run',str(run),'--shard',str(shard),'--gpu-index',str(index)]
                with (run/f'cache_{shard}.log').open('w') as log:
                    child=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=uuid),
                        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    write(run/f'cache_launch_{shard}.json',dict(pid=child.pid,gpu_index=index,gpu_uuid=uuid,command=command))
                    code=child.wait()
                if code:raise RuntimeError(f'Confirmation cache shard {shard} exited {code}')
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures=[pool.submit(worker,slot,device) for slot,device in enumerate(devices)]
            for future in futures:future.result()
        write(run/'status.json',dict(stage='scoring_confirmation',pid=os.getpid()))
        command=[str(primary),str(run/'scripts/score_frozen_geometry.py'),'--run',str(run),
                 '--mode','confirmation','--identity-root',plan['identity_root'],'--output',str(run/'scores'),
                 '--frozen-protocol',str(run/'protocol.json')]
        with (run/'score.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        scores=json.loads((run/'scores/results.json').read_text())
        assert scores['completed'] and scores['identity_decisions']==expected*64 and scores['unique_queries']==expected
        write(run/'complete.json',dict(completed=True,confirmation_sources=64,queries=expected,identity_decisions=expected*64,
              wrong_identity_decisions=expected*63,protocol_sha256=sha(run/'protocol.json'),results_sha256=sha(run/'scores/results.json'),
              new_attacks_run=plan.get('published_distortion_instances_generated',False),novel_or_modified_attacks=False))
        write(run/'status.json',dict(stage='complete',pid=os.getpid()))
    except Exception as exc:
        write(run/'failed.json',dict(error=str(exc),traceback=traceback.format_exc()))
        write(run/'status.json',dict(stage='failed',pid=os.getpid(),error=str(exc)))
        raise


if __name__=='__main__':main()
