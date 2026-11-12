"""Full model experiment with process-exit scheduling and published-only baselines."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from anchor_binding import identity,order
from run_anchor_binding_pipeline import audit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)


def audit_projection(run,plan,genuine):
    keys={t:identity(plan['identity_root'],t) for t in range(6)};perms={t:order(k[3],16384) for t,k in keys.items()}
    details=[]
    for item in plan['sources']:
        t=item['task'];z=np.random.default_rng(item['aux_seed']).standard_normal((1,4,64,64)).astype(np.float32)
        directory=run/f'genuine_{t%3}'
        h=np.load(directory/'codes'/f'{t:03}_aux.npz')['code']
        secret,c,_,_=keys[t];target=c^np.tile(h,16)[perms[t]]
        base_wrong=(z.ravel()>0)!=target
        ranks=np.random.default_rng(secret.get('projection_order_seed',0)).random(16384)
        for arm in plan['arms']:
            ref=next(r for r in genuine['references'] if r['task']==t and r['arm']==arm)
            initial=np.load(run/ref['initial_latent']);assert sha(run/ref['initial_latent'])==ref['initial_latent_sha256']
            if arm=='full':
                amp=np.abs(np.random.default_rng(secret['amplitude_seed']).standard_normal(16384))
                expected=(amp*(2*target.astype(float)-1)).astype(np.float32)
            elif arm=='saturated':expected=np.abs(z.ravel())*(2*target.astype(np.float32)-1)
            elif arm in ['transport','shuffled_transport']:
                from scipy.stats import norm
                expected=norm.ppf((norm.cdf(z.ravel().astype(float))+target)/2).astype(np.float32)
                if arm=='shuffled_transport':
                    rng=np.random.default_rng(secret['transport_shuffle_seed'])
                    for bit in [0,1]:
                        positions=np.flatnonzero(target==bit)
                        expected[positions]=expected[rng.permutation(positions)]
            else:
                expected=z.ravel().copy()
                for b in range(256):
                    wrong=[i for i in range(b,16384,256) if base_wrong[i]]
                    ranked=sorted(wrong,key=lambda i:abs(float(z.ravel()[i])) if arm=='minimal56' else float(ranks[i]))
                    for i in ranked[:max(0,len(wrong)-8)]:expected[i]=-expected[i]
            if arm in ['transport','shuffled_transport']:
                np.testing.assert_allclose(initial.ravel(),expected,rtol=2e-6,atol=1e-7)
            else:assert np.array_equal(initial.ravel(),expected),(t,arm)
            mismatch=((initial.ravel()>0)!=target).reshape(64,256).sum(0)
            assert np.all(mismatch<=8) if arm.endswith('56') else np.all(mismatch==0)
            details.append(dict(task=t,arm=arm,latent_l2=float(np.mean((initial.astype(float)-z)**2)),
                                reflected_symbols=int(np.sum((initial.ravel()>0)!=(z.ravel()>0))),maximum_group_errors=int(mismatch.max())))
    wrong=[]
    for r in genuine['rows']:
        z=np.load(run/r['latent']).ravel();q=np.load(run/r['code'])['code']
        for t,(_,c,m,_) in keys.items():
            if t==r['task']:continue
            bits=(z>0).astype(np.uint8)^np.tile(q,16)[perms[t]]^c^np.tile(m,64)
            ba=float(np.mean((bits.reshape(64,256).sum(0)>32)==m))
            wrong.append(dict(id=r['id'],arm=r['arm'],identity=t,ba=ba,accepted=ba>181/256))
    assert len(wrong)==1200
    result=dict(passed=True,projection_records=details,wrong_identity=wrong,
        wrong_accepts={a:sum(r['accepted'] for r in wrong if r['arm']==a) for a in plan['arms']})
    save(run/'projection_audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve();plan=json.loads((run/'protocol.json').read_text())
    for n,d in plan['source_sha256'].items():assert sha(run/n)==d,n
    for n,d in plan['identity_sha256'].items():assert sha(Path(plan['identity_root'])/n)==d
    assert sha(run/'unit_checks.json')==plan['unit_checks_sha256'] and json.loads((run/'unit_checks.json').read_text())['passed']
    def state(status,**kw):save(run/'pipeline_state.json',dict(status=status,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    def launch(cmd,log,env=None):
        with log.open('w') as f:return subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
    def worker(stage,i):return [sys.executable,str(run/'run_anchor_binding_worker.py'),'--run',str(run),'--stage',stage,'--shard',str(i)]
    try:
        processes=[launch(worker('genuine',i),run/f'genuine_{i}.log') for i in range(3)]
        state('running',stage='genuine',worker_pids=[p.pid for p in processes]);codes=[p.wait() for p in processes];assert not any(codes),codes
        genuine=audit(run,'genuine',plan);detail=audit_projection(run,plan,genuine);s={r['arm']:r for r in genuine['summary']}
        primary=plan.get('primary','minimal56');coupling=plan.get('coupling_control','random56');controls=[k for k in plan['arms'] if k!=primary];a=s[primary]
        gates=dict(genuine=a['accepted']>=max(s[k]['accepted'] for k in controls),
            clean=all(r['accepted'] for r in genuine['rows'] if r['arm']==primary and r['condition']=='clean'),
            binding=a['mean_binding_mismatch']<=s['saturated']['mean_binding_mismatch']-.02 and a['mean_binding_mismatch']<=s[coupling]['mean_binding_mismatch'],
            fidelity=a['mean_reference_lpips']<=s['saturated']['mean_reference_lpips']-.05,
            wrong_identity=detail['wrong_accepts'][primary]<=min(detail['wrong_accepts'][k] for k in controls))
        advance=all(gates.values());save(run/'pilot_decision.json',dict(advance=advance,gates=gates))
        lines=['# Decision-region embedding experiment','','All240 full genuine inversions audited. Threshold BA>181/256 unchanged.','',
            '| Arm | Genuine | Binding mismatch | Preview LPIPS | Wrong identity |','|---|---:|---:|---:|---:|']
        for r in genuine['summary']:lines.append(f"| {r['arm']} | {r['accepted']}/60 | {r['mean_binding_mismatch']:.6f} | {r['mean_reference_lpips']:.6f} | {detail['wrong_accepts'][r['arm']]}/300 |")
        lines += ['',f'Frozen gates: {gates}. Advance: {advance}.','Six-source screen; LPIPS is paired preview fidelity, not FID. No superiority or empirical low-FPR claim.']
        (run/'analysis.md').write_text('\n'.join(lines)+'\n')
        if not advance:state('complete',stage='genuine_audited_gate_failed',genuine_records=240,forgery_jobs=0);return
        jobs=[]
        for item in plan['sources']:
            for arm in plan['arms']:
                ref=next(r for r in genuine['references'] if r['task']==item['task'] and r['arm']==arm)
                cover=plan['covers'][item['task']%len(plan['covers'])];assert sha(cover['path'])==cover['sha256']
                jobs.append(dict(id=f"{item['task']:03}_{arm}",task=item['task'],method=arm,reference=ref['image'],reference_sha256=ref['image_sha256'],cover=cover['path'],cover_sha256=cover['sha256']))
        attack=dict(scope='Unchanged published CVPR2025 Imprint on frozen defensive embeddings',jobs=jobs,updates=151,inverse_steps=50,lr=.01,max_concurrent_gpus=3,excluded_gpu_indices=[6],worker_gpu_indices=[7,5,4],checkpoint_source_steps=list(range(0,151,10)),early_stop=False,shortened_budget_allowed=False,parent_protocol_sha256=sha(run/'protocol.json'),source='https://arxiv.org/abs/2412.03283')
        save(run/'attack_protocol.json',attack)
        def baseline(i):
            uuid=subprocess.check_output(['nvidia-smi','-i',str(plan['gpu_indices'][i]),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            cmd=[plan['attack_python'],str(run/'run_published_imprint_full.py'),'--protocol',str(run/'attack_protocol.json'),'--upstream',plan['upstream'],'--model',plan['paths']['model'],'--output',str(run/'attack_outputs'),'--shard',str(i)]
            proc=launch(cmd,run/f'imprint_{i}.log',env);save(run/f'baseline_launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i]));code=proc.wait()
            if code:return code
            time.sleep(10);return launch(worker('forgery',i),run/f'forgery_{i}.log').wait()
        state('running',stage='published_full_imprint')
        with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(baseline,range(3)))
        assert not any(codes),codes
        forged=audit(run,'forgery',plan)
        with (run/'analysis.md').open('a') as out:
            out.write('\nAll24 published151-update jobs and432 verification records completed.\n')
            out.write(json.dumps(forged['summary'],indent=2)+'\n')
        state('complete',stage='all_endpoints_audited',genuine_records=240,forgery_records=432)
    except Exception as exc:state('failed',error=repr(exc));raise


if __name__=='__main__':main()
