"""Execute frozen published baseline then full verification; wait on child exits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    run=args.run.resolve()
    plan=json.loads((run/'protocol.json').read_text())
    root=Path(plan['root'])
    chain=Path(plan['chain_run'])
    paths=json.loads((chain/'paths.json').read_text())
    for name,digest in plan['pipeline_source_sha256'].items():
        assert sha(run/name)==digest,name
    save(run/'pipeline_state.json',dict(status='running',pid=os.getpid(),mode='blocking child-exit waits, no periodic polling'))
    def worker(i):
        gpu=plan['worker_gpu_indices'][i]
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HF_HOME=str(root/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple/',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH='')
        commands=[('generation',[plan['generation_python'],str(run/'scripts/run_published_imprint_full.py'),
            '--protocol',str(run/'protocol.json'),'--upstream',plan['upstream'],'--model',paths['model'],
            '--output',str(run/'outputs'),'--shard',str(i)]),
            ('verification',[plan['verification_python'],str(run/'scripts/evaluate_published_imprint_full.py'),
             '--root',str(root),'--run',str(run),'--identity-root',paths['root'],'--chain-run',str(chain),'--shard',str(i)])]
        for stage,command in commands:
            # One fixed settling interval after process teardown, not utilization polling.
            time.sleep(10)
            with (run/f'{stage}_{i}.log').open('w') as log:
                proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(run/f'shard_{i}_state.json',dict(status='running',stage=stage,pid=proc.pid,gpu=gpu,command=command))
            code=proc.wait()
            if code:
                save(run/f'shard_{i}_state.json',dict(status='failed',stage=stage,returncode=code))
                return dict(shard=i,stage=stage,returncode=code)
        save(run/f'shard_{i}_state.json',dict(status='complete',gpu=gpu))
        return None
    with ThreadPoolExecutor(max_workers=3) as pool:
        failures=[r for r in pool.map(worker,range(3)) if r]
    if failures:
        save(run/'pipeline_state.json',dict(status='failed',failures=failures))
        raise RuntimeError(failures)
    try:
        import numpy as np
        from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
        identities={}
        for task in range(4):
            secret=json.loads((Path(paths['root'])/'private'/f'{task:03}.json').read_text())
            key,nonce,msg,bkey=[bytes.fromhex(secret[k]) for k in ['key','nonce','message','binding_key']]
            cipher=np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64),dtype=np.uint8))
            message=np.unpackbits(np.frombuffer(msg,dtype=np.uint8))
            tags=[hmac.new(bkey,b'sembind/permutation/'+j.to_bytes(4,'big'),hashlib.sha256).digest() for j in range(16384)]
            identities[task]=(cipher^np.tile(message,64),message,np.asarray(sorted(range(16384),key=tags.__getitem__)))
        rows=[]
        for i in range(3):
            directory=run/'evaluation'/f'shard_{i}'
            result=json.loads((directory/'results.json').read_text())
            assert result['completed'] and result['protocol_sha256']==sha(run/'protocol.json')
            for r in result['rows']:
                latent=directory/(r['id']+'.npy')
                features=directory/(r['id']+'.npz')
                assert sha(latent)==r['inversion_sha256'] and sha(features)==r['features_sha256']
                assert sha(r['image_path'])==r['image_sha256']
                stream,message,perm=identities[r['task']]
                code=np.load(features)['sembind_code']
                mask=np.tile(code,16)[perm] if r['method']=='sembind' else np.zeros(16384,dtype=np.uint8)
                signs=(np.load(latent).reshape(-1)>0).astype(np.uint8)
                ba=float((((signs^mask^stream).reshape(64,256).sum(axis=0)>32)==message).mean())
                assert ba==r['ba'] and (ba>181/256)==r['accepted']
                rows.append(r)
        expected={j['id']+'_'+k for j in plan['jobs'] for k in ['reference','cover']}
        expected|={j['id']+f'_update_{u:03}' for j in plan['jobs'] for u in range(1,152,10)}
        assert len(rows)==144 and {r['id'] for r in rows}==expected
        summary=[]
        for method in ['sembind','gs']:
            for kind in ['reference','cover','candidate']:
                group=[r for r in rows if r['method']==method and r['kind']==kind and (kind!='candidate' or r['update']==151)]
                assert len(group)==4
                summary.append(dict(method=method,kind=kind,n=4,accepted=sum(r['accepted'] for r in group),mean_ba=sum(r['ba'] for r in group)/4))
        save(run/'verified_results.json',dict(completed=True,audit_passed=True,summary=summary,rows=rows,
            full_update_jobs=8,updates_per_job=151,checkpoint_images=128,independently_recomputed_scores=144,
            scope='Four reused development inputs, fresh generation and inference. No independent confirmation or SOTA claim.'))
        lines=['# Fresh full-budget published Imprint evaluation','','All 144 saved-latent scores independently recomputed. Every job completed 151 updates and all 16 checkpoints.','',
            '| Method | Input | Accepted | Mean BA |','|---|---|---:|---:|']
        for r in summary:
            lines.append(f"| {r['method']} | {r['kind']} | {r['accepted']}/4 | {r['mean_ba']:.6f} |")
        lines+=['','Primary candidate endpoint is update151; all intermediate checkpoints remain archived. These four identities are development inputs, not independent confirmation.','',
            'Next decision: compare the audited genuine-distortion decomposition with these unchanged-baseline outcomes before selecting a defense hypothesis. Any new method requires a separate frozen protocol and new sources.']
        (run/'analysis.md').write_text('\n'.join(lines)+'\n')
        save(run/'pipeline_state.json',dict(status='complete',stage='independently_audited',records=144))
    except Exception as exc:
        save(run/'pipeline_state.json',dict(status='failed',stage='audit',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
