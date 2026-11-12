"""Event-driven defense pilot, independent audit, and gated published evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def audit(run,stage,plan):
    import numpy as np
    from scipy.fft import dctn
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
    rows=[];refs=[];identities={}
    for i in range(6):
        s=json.loads((Path(plan['identity_root'])/f'{i:03}.json').read_text())
        key,nonce,msg,bkey=[bytes.fromhex(s[k]) for k in ['key','nonce','message','binding_key']]
        cipher=np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64),dtype=np.uint8))
        message=np.unpackbits(np.frombuffer(msg,dtype=np.uint8))
        permutations={}
        for n,domain in [(16384,b'sembind/permutation/'),(8192,b'anchor/permutation/')]:
            tags=[hmac.new(bkey,domain+j.to_bytes(4,'big'),hashlib.sha256).digest() for j in range(n)]
            permutations[n]=np.asarray(sorted(range(n),key=tags.__getitem__))
        identities[i]=(cipher,message,permutations)
    for i in range(3):
        done=json.loads((run/f'{stage}_{i}/complete.json').read_text())
        assert done['status']=='complete' and done['protocol_sha256']==sha(run/'protocol.json')
        rows.extend(done['rows']);refs.extend(done['references'])
    if stage=='genuine':
        expected={(i,a,c['name']) for i in range(6) for a in plan['arms'] for c in plan['conditions']}
        assert len(rows)==240 and {(r['task'],r['arm'],r['condition']) for r in rows}==expected
        assert len(refs)==24
        import torch
        import lpips
        from PIL import Image
        torch.set_num_threads(4)
        sys.path.insert(0,plan['upstream'])
        from utils.image_utils import lpips_PIL
        quality=lpips.LPIPS(net='alex').eval().cpu()
        for ref in refs:
            directory=run/f"genuine_{ref['task']%3}"
            q=np.load(directory/'codes'/f"{ref['task']:03}_{ref['arm']}.npz")['code']
            h=np.load(directory/'codes'/f"{ref['task']:03}_aux.npz")['code']
            assert float((q!=h).mean())==ref['semantic_mismatch']
            assert sha(run/ref['image'])==ref['image_sha256']
            with torch.no_grad():
                value=float(lpips_PIL(Image.open(run/ref['image']).convert('RGB'),
                                     Image.open(run/ref['auxiliary']).convert('RGB'),quality,'cpu'))
            assert abs(value-ref['lpips'])<1e-6
        del quality
    else:
        attack=json.loads((run/'attack_protocol.json').read_text())
        expected={j['id']+'_'+k for j in attack['jobs'] for k in ['reference','cover']}
        expected|={j['id']+f'_update_{u:03}' for j in attack['jobs'] for u in range(1,152,10)}
        assert len(rows)==216 and {r['id'] for r in rows}==expected
    for r in rows:
        latent=run/r['latent'];codepath=run/r['code'];imagepath=run/r['image']
        assert sha(latent)==r['latent_sha256'] and sha(codepath)==r['code_sha256'] and sha(imagepath)==r['image_sha256']
        z=np.load(latent);code=np.load(codepath)['code'];cipher,message,perms=identities[r['task']]
        if r['arm']=='full':values=z.ravel()
        elif r['arm'] in ['quantile_coupled','quantile_independent']:
            # Independent implementation: recover the Gaussian bit plane directly.
            from scipy.stats import norm
            low=sorted(range(4096),key=lambda j:(j//64+j%64,j))[:2048]
            anchor=np.asarray([j+4096*c for c in range(4) for j in low])
            coordinates=dctn(z.astype(np.float64),axes=(-2,-1),norm='ortho').ravel()
            raw=(coordinates>0).astype(np.uint8)
            raw[anchor]=(np.abs(coordinates[anchor])>=norm.ppf(.75)).astype(np.uint8)
            payload_order=np.concatenate([anchor,np.setdiff1d(np.arange(16384),anchor)])
            values=2*raw[payload_order].astype(float)-1
        else:
            if r['arm']=='anchor_random':a=np.sort(np.random.default_rng(2026111201).permutation(4096)[:2048])
            else:a=np.asarray(sorted(range(4096),key=lambda j:(j//64+j%64,j))[:2048])
            anchor=np.concatenate([a+4096*c for c in range(4)])
            payload=np.setdiff1d(np.arange(16384),anchor)
            values=dctn(z,axes=(-2,-1),norm='ortho').ravel()[payload]
        n=len(values);mask=np.tile(code,n//1024)[perms[n]]
        bits=(values>0).astype(np.uint8)^mask^cipher[:n]^np.tile(message,n//256)
        recovered=bits.reshape(n//256,256).sum(0)>(n//256)/2
        ba=float(np.mean(recovered==message));assert ba==r['ba'] and (ba>181/256)==r['accepted']
    summary=[]
    if stage=='genuine':
        for arm in plan['arms']:
            g=[r for r in rows if r['arm']==arm];f=[r for r in refs if r['arm']==arm]
            summary.append(dict(arm=arm,n=60,accepted=sum(r['accepted'] for r in g),mean_ba=float(np.mean([r['ba'] for r in g])),
                mean_binding_mismatch=float(np.mean([r['semantic_mismatch'] for r in f])),mean_reference_lpips=float(np.mean([r['lpips'] for r in f]))))
    else:
        for arm in plan['after_positive_gate']['arms']:
            for kind in ['reference','cover','candidate']:
                g=[r for r in rows if r['arm']==arm and r['kind']==kind and (kind!='candidate' or r['update']==151)]
                assert len(g)==6
                summary.append(dict(arm=arm,kind=kind,n=6,accepted=sum(r['accepted'] for r in g),mean_ba=float(np.mean([r['ba'] for r in g]))))
    result=dict(completed=True,audit_passed=True,rows=rows,references=refs,summary=summary,unique_sources=6,
                protocol_sha256=sha(run/'protocol.json'),independent_paper_scale_confirmation=False)
    save(run/f'{stage}_verified.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent-pid',type=int,required=True)
    args=p.parse_args();run=args.run.resolve();plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(run/name)==digest,name
    def state(status,**kw):save(run/'pipeline_state.json',dict(status=status,pid=os.getpid(),**kw))
    def worker_command(stage,i):return [sys.executable,str(run/'run_anchor_binding_worker.py'),'--run',str(run),'--stage',stage,'--shard',str(i)]
    def launch(command,logpath,env=None):
        with logpath.open('w') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        return proc
    marker=Path(plan['parent_baseline_run'])/'verified_results.json'
    state('waiting',stage='parent_full_baseline',parent_pid=args.parent_pid,mode='kernel process-exit event; no polling')
    try:
        if not marker.exists():
            libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,args.parent_pid,0)
            if fd<0:
                if not marker.exists():raise OSError(ctypes.get_errno(),'Cannot attach baseline completion event')
            else:select.select([fd],[],[]);os.close(fd)
        assert marker.exists() and json.loads(marker.read_text())['audit_passed']
        time.sleep(10)
        processes=[]
        for i in range(3):
            proc=launch(worker_command('genuine',i),run/f'genuine_{i}.log');processes.append(proc)
        state('running',stage='genuine_defense_pilot',worker_pids=[p.pid for p in processes])
        codes=[p.wait() for p in processes];assert not any(codes),codes
        genuine=audit(run,'genuine',plan);s={x['arm']:x for x in genuine['summary']}
        if plan.get('carrier_module')=='quantile_binding':
            a,b,c,f=[s[k] for k in ['quantile_coupled','quantile_independent','anchor_low','full']]
            gates=dict(binding_vs_independent=b['mean_binding_mismatch']-a['mean_binding_mismatch']>=.02,
                       genuine_retention=a['accepted']>=max(b['accepted'],c['accepted'],f['accepted']),
                       clean_retention=all(r['accepted'] for r in genuine['rows'] if r['arm']=='quantile_coupled' and r['condition']=='clean'),
                       reference_fidelity=a['mean_reference_lpips']<b['mean_reference_lpips'])
        else:
            a,b,c,f=[s[k] for k in ['anchor_low','independent_low','anchor_random','full']]
            gates=dict(binding_vs_independent=b['mean_binding_mismatch']-a['mean_binding_mismatch']>=.02,
                       binding_vs_random=c['mean_binding_mismatch']-a['mean_binding_mismatch']>=.01,
                       genuine_retention=a['accepted']>=max(b['accepted'],f['accepted']),
                       reference_fidelity=a['mean_reference_lpips']<b['mean_reference_lpips'])
        advance=all(gates.values());save(run/'pilot_decision.json',dict(advance_to_published_forgery=advance,gates=gates,
            interpretation='Prespecified feasibility gate on six new pilot sources; not a novelty or superiority claim.'))
        lines=['# Content-coupled defense pilot','','All240 genuine-query scores independently recomputed. Six new neutral prompts, four embedding arms, ten conditions, full50-step generation/inversion and100-step decoder optimization.','',
            '| Arm | Accepted | Mean BA | Binding mismatch | LPIPS to auxiliary |','|---|---:|---:|---:|---:|']
        for r in genuine['summary']:lines.append(f"| {r['arm']} | {r['accepted']}/60 | {r['mean_ba']:.6f} | {r['mean_binding_mismatch']:.6f} | {r['mean_reference_lpips']:.6f} |")
        lines+=['',f'Prespecified feasibility gates: {gates}. Advance: {advance}.',
            'LPIPS is a paired auxiliary-image fidelity diagnostic, not FID or a general quality guarantee. Carrier budgets and matched controls are specified in protocol.json. Shared and independent variants isolate coupling; full-carrier comparisons also change the encoding geometry.']
        (run/'analysis.md').write_text('\n'.join(lines)+'\n')
        if not advance:
            state('complete',stage='pilot_failed_feasibility_gate',decision='Do not expand this embedding candidate; preserve all negative results.')
            return
        jobs=[]
        for item in plan['sources']:
            for arm in plan['after_positive_gate']['arms']:
                ref=next(r for r in genuine['references'] if r['task']==item['task'] and r['arm']==arm)
                cover=plan['covers'][item['task']%len(plan['covers'])]
                assert sha(cover['path'])==cover['sha256']
                jobs.append(dict(id=f"{item['task']:03}_{arm}",task=item['task'],method=arm,
                    reference=ref['image'],reference_sha256=ref['image_sha256'],cover=cover['path'],cover_sha256=cover['sha256']))
        attack=dict(scope='Unchanged CVPR2025 Imprint on frozen defensive carrier and full-carrier control',jobs=jobs,updates=151,
                    inverse_steps=50,lr=.01,max_concurrent_gpus=3,excluded_gpu_indices=[6],worker_gpu_indices=[7,5,4],
                    checkpoint_source_steps=list(range(0,151,10)),early_stop=False,shortened_budget_allowed=False,
                    parent_protocol_sha256=sha(run/'protocol.json'),source='https://arxiv.org/abs/2412.03283')
        save(run/'attack_protocol.json',attack)
        def full_worker(i):
            uuid=subprocess.check_output(['nvidia-smi','-i',str(plan['gpu_indices'][i]),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            command=[plan['attack_python'],str(run/'run_published_imprint_full.py'),'--protocol',str(run/'attack_protocol.json'),
                     '--upstream',plan['upstream'],'--model',plan['paths']['model'],'--output',str(run/'attack_outputs'),'--shard',str(i)]
            time.sleep(10);proc=launch(command,run/f'imprint_{i}.log',env)
            save(run/f'attack_launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i],command=command))
            result=proc.wait()
            if result:return result
            time.sleep(10);proc=launch(worker_command('forgery',i),run/f'forgery_{i}.log')
            save(run/f'verifier_launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i]))
            return proc.wait()
        state('running',stage='published_full_forgery_evaluation')
        with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(full_worker,range(3)))
        assert not any(codes),codes
        forged=audit(run,'forgery',plan)
        with (run/'analysis.md').open('a') as out:
            out.write('\nAll216 published-baseline verification scores independently recomputed. Every job completed151 updates and all16 checkpoints.\n\n')
            out.write('| Arm | Input | Accepted |\n|---|---|---:|\n')
            for r in forged['summary']:out.write(f"| {r['arm']} | {r['kind']} | {r['accepted']}/6 |\n")
            out.write('\nNext: independent larger-source confirmation only if the genuine gain is retained without increased forgery acceptance. Six prompts and four repeated covers do not establish a population false-positive rate.\n')
        state('complete',stage='defense_and_published_baseline_audited')
    except Exception as exc:state('failed',error=repr(exc));raise


if __name__=='__main__':main()
