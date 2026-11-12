"""Process-exit scheduled semantic-feedback mechanism experiment."""
import argparse,ctypes,datetime,hashlib,hmac,json,os,select,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
from run_anchor_binding_pipeline import audit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def audit_feedback(run,plan,genuine):
    trace=[];auxiliary=[];keys={}
    for i in range(3):
        d=json.loads((run/f'genuine_{i}/complete.json').read_text());trace.extend(d['feedback_trace']);auxiliary.extend(d['auxiliary_records'])
    assert len(trace)==18 and {(x['task'],x['round']) for x in trace}=={(i,j) for i in range(6) for j in [1,2,3]}
    assert len(auxiliary)==18 and {(x['task'],x['index']) for x in auxiliary}=={(i,j) for i in range(6) for j in [0,1,2]}
    for entry in auxiliary:
        for name in ['image','code']:assert sha(run/entry[name])==entry[name+'_sha256']
    source_summary=[]
    for task in range(6):
        s=json.loads((Path(plan['identity_root'])/f'{task:03}.json').read_text());k,nonce,msg,bkey=[bytes.fromhex(s[x]) for x in ['key','nonce','message','binding_key']]
        c=np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(k,nonce),mode=None).encryptor().update(msg*64),dtype=np.uint8));m=np.unpackbits(np.frombuffer(msg,dtype=np.uint8))
        tags=[hmac.new(bkey,b'sembind/permutation/'+j.to_bytes(4,'big'),hashlib.sha256).digest() for j in range(16384)];perm=np.asarray(sorted(range(16384),key=tags.__getitem__))
        amplitude=np.abs(np.random.default_rng(s['amplitude_seed']).standard_normal(16384));keys[task]=(c,m,perm)
        aux=sorted([x for x in auxiliary if x['task']==task],key=lambda x:x['index']);auxcodes=[np.load(run/x['code'])['code'] for x in aux]
        previous=auxcodes[0];history=[previous.copy()];records=sorted([x for x in trace if x['task']==task],key=lambda x:x['round'])
        for entry in records:
            for name in ['input_code','output_code','image','initial_latent']:assert sha(run/entry[name])==entry[name+'_sha256']
            before=np.load(run/entry['input_code'])['code'];after=np.load(run/entry['output_code'])['code'];latent=np.load(run/entry['initial_latent'])
            assert np.array_equal(previous,before)
            expected=(amplitude*(2*(c^np.tile(before,16)[perm]).astype(float)-1)).reshape(1,4,64,64).astype(np.float32)
            assert np.array_equal(latent,expected)
            assert float(np.mean(before!=after))==entry['binding_mismatch']
            history.append(after.copy());previous=after
        refs={x['arm']:x for x in genuine['references'] if x['task']==task}
        for arm,step in [('full',1),('feedback',3)]:
            record=records[step-1];assert refs[arm]['image_sha256']==record['image_sha256']
            assert refs[arm]['enrollment_code_sha256']==record['input_code_sha256']
        consensus=(np.stack(auxcodes).sum(0)>=2).astype(np.uint8)
        assert np.array_equal(consensus,np.load(run/refs['consensus']['enrollment_code'])['code'])
        source_summary.append(dict(task=task,round_mismatch=[x['binding_mismatch'] for x in records],final_fixed_point=bool(np.array_equal(history[-1],history[-2])),two_cycle=bool(np.array_equal(history[-1],history[-3]) and not np.array_equal(history[-1],history[-2]))))
    wrong=[]
    for x in genuine['rows']:
        z=np.load(run/x['latent']).ravel();q=np.load(run/x['code'])['code']
        for task,(c,m,perm) in keys.items():
            if task==x['task']:continue
            bits=(z>0).astype(np.uint8)^np.tile(q,16)[perm]^c^np.tile(m,64)
            ba=float(np.mean((bits.reshape(64,256).sum(0)>32)==m))
            wrong.append(dict(id=x['id'],arm=x['arm'],identity=task,ba=ba,accepted=ba>181/256))
    assert len(wrong)==900
    result=dict(audit_passed=True,feedback_trace=trace,auxiliary_records=auxiliary,sources=source_summary,wrong_identity=wrong,wrong_accepts={a:sum(x['accepted'] for x in wrong if x['arm']==a) for a in plan['arms']},wrong_pairs_per_arm=300)
    save(run/'feedback_audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve();plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(run/name)==digest,name
    assert sha(run/'unit_checks.json')==plan['unit_checks_sha256']
    assert json.loads((run/'unit_checks.json').read_text())['passed']
    def state(status,**kw):save(run/'pipeline_state.json',dict(status=status,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    def launch(command,log,env=None):
        with log.open('w') as stream:return subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    def worker(stage,i):return [sys.executable,str(run/'run_semantic_feedback_worker.py'),'--run',str(run),'--stage',stage,'--shard',str(i)]
    try:
        parent=Path(plan['parent_run']);assert sha(parent/'protocol.json')==plan['parent_protocol_sha256']
        parent_state=json.loads((parent/'pipeline_state.json').read_text())
        state('waiting',stage='parent_full_evaluation',parent_pid=plan['parent_pid'],mode='pidfd process-exit event; no polling')
        if parent_state['status']!='complete':
            proc=Path('/proc')/str(plan['parent_pid'])
            assert proc.joinpath('stat').read_text().split()[21]==plan['parent_start_ticks']
            assert str(parent) in proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,plan['parent_pid'],0)
            if fd<0:raise OSError(ctypes.get_errno(),'Cannot attach parent completion event')
            select.select([fd],[],[]);os.close(fd)
        done=json.loads((parent/'pipeline_state.json').read_text());assert done['status']=='complete',done
        for stage,n in [('genuine',120),('forgery',216)]:
            evidence=json.loads((parent/f'{stage}_defense_results.json').read_text());assert evidence['completed'] and evidence['independent_grouped_score_audit'] and len(evidence['rows'])==n
        time.sleep(10);ps=[launch(worker('genuine',i),run/f'genuine_{i}.log') for i in range(3)]
        state('running',stage='full_genuine_feedback_pilot',worker_pids=[x.pid for x in ps]);codes=[x.wait() for x in ps];assert not any(codes),codes
        genuine=audit(run,'genuine',plan);details=audit_feedback(run,plan,genuine);summaries={x['arm']:x for x in genuine['summary']};a,b,c=[summaries[k] for k in ['feedback','full','consensus']]
        gates=dict(binding_vs_full=b['mean_binding_mismatch']-a['mean_binding_mismatch']>=.02,binding_vs_consensus=a['mean_binding_mismatch']<=c['mean_binding_mismatch'],genuine_gain=a['accepted']>=b['accepted']+1 and a['accepted']>=c['accepted'],clean_retention=all(x['accepted'] for x in genuine['rows'] if x['arm']=='feedback' and x['condition']=='clean'),source_consistency=sum(x['round_mismatch'][-1]<x['round_mismatch'][0] for x in details['sources'])>=4,preview_fidelity=a['mean_reference_lpips']<=b['mean_reference_lpips']+.05,wrong_identity=details['wrong_accepts']['feedback']<=min(details['wrong_accepts']['full'],details['wrong_accepts']['consensus']))
        advance=all(gates.values());save(run/'pilot_decision.json',dict(advance=advance,gates=gates,scope='Six-source feasibility screen, not statistical superiority'))
        lines=['# Semantic feedback mechanism pilot','','All180 genuine records and18 feedback transitions audited. Fixed three full50-step reference generations; endpoint is always the third reference. Query verification remains originalhardBA>181/256.','', '| Arm | Genuine | Binding mismatch | LPIPS to initial auxiliary | Wrong identities |','|---|---:|---:|---:|---:|']
        for x in genuine['summary']:lines.append(f"| {x['arm']} | {x['accepted']}/60 | {x['mean_binding_mismatch']:.6f} | {x['mean_reference_lpips']:.6f} | {details['wrong_accepts'][x['arm']]}/300 |")
        lines+=['',f'Frozen feasibility gates: {gates}. Advance: {advance}.','', 'Full uses two generation calls standalone; feedback and three-auxiliary consensus each use four. Shared preliminary outputs reduce experiment duplication only. Feedback couples later masks to ciphertext; the one-pass Gaussian-independence argument does not apply. LPIPS is preview fidelity, not FID or a quality guarantee.']
        (run/'analysis.md').write_text('\n'.join(lines)+'\n')
        if not advance:state('complete',stage='failed_frozen_feasibility_gate');return
        jobs=[]
        for item in plan['sources']:
            for arm in plan['arms']:
                ref=next(x for x in genuine['references'] if x['task']==item['task'] and x['arm']==arm);cover=plan['covers'][item['task']%len(plan['covers'])];assert sha(cover['path'])==cover['sha256']
                jobs.append(dict(id=f"{item['task']:03}_{arm}",task=item['task'],method=arm,reference=ref['image'],reference_sha256=ref['image_sha256'],cover=cover['path'],cover_sha256=cover['sha256']))
        attack=dict(scope='Unchanged CVPR2025 Imprint on frozen full,feedback,and consensus references',jobs=jobs,updates=151,inverse_steps=50,lr=.01,max_concurrent_gpus=3,excluded_gpu_indices=[6],worker_gpu_indices=[7,5,4],checkpoint_source_steps=list(range(0,151,10)),early_stop=False,shortened_budget_allowed=False,parent_protocol_sha256=sha(run/'protocol.json'),source='https://arxiv.org/abs/2412.03283');save(run/'attack_protocol.json',attack)
        def baseline(i):
            uuid=subprocess.check_output(['nvidia-smi','-i',str(plan['gpu_indices'][i]),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            time.sleep(10);cmd=[plan['attack_python'],str(run/'run_published_imprint_full.py'),'--protocol',str(run/'attack_protocol.json'),'--upstream',plan['upstream'],'--model',plan['paths']['model'],'--output',str(run/'attack_outputs'),'--shard',str(i)]
            proc=launch(cmd,run/f'imprint_{i}.log',env);save(run/f'baseline_launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i]));code=proc.wait()
            if code:return code
            time.sleep(10);return launch(worker('forgery',i),run/f'forgery_{i}.log').wait()
        state('running',stage='published_full_imprint')
        with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(baseline,range(3)))
        assert not any(codes),codes
        forged=audit(run,'forgery',plan)
        with (run/'analysis.md').open('a') as out:
            out.write('\nAll18 published jobs completed151 updates and all16 checkpoints; all324 verification records independently audited.\n\n| Arm | Input | Accepted |\n|---|---|---:|\n')
            for x in forged['summary']:out.write(f"| {x['arm']} | {x['kind']} | {x['accepted']}/{x['n']} |\n")
            out.write('\nPrimary forgery endpoint is update151. No best-checkpoint selection. A positive mechanism pilot still requires larger independent sources, broader native comparisons and image-distribution evaluation.\n')
        state('complete',stage='genuine_and_published_forgery_audited',genuine_records=180,forgery_records=324)
    except Exception as exc:state('failed',error=repr(exc));raise


if __name__=='__main__':main()
