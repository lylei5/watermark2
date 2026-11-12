"""Source-separated semantic evidence experiment with full published evaluation."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from anchor_binding import identity,order
from fit_semantic_evidence import fit
from semantic_evidence import scores,decisions
from semantic_evidence_reference import grouped_reference
from run_anchor_binding_pipeline import audit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,data):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def evaluate(run,stage,plan,verified,model):
    identities={}
    for item in plan['sources']:
        t=item['task'];_,c,m,key=identity(plan['identity_root'],t);identities[t]=(c,m,order(key,16384))
    outputs=[];wrong=[];max_error=0.
    for x in verified['rows']:
        feature=np.load(run/x['code']);z=np.load(run/x['latent']);assert 'logits' in feature
        value=scores(z,feature['logits'],feature['code'],*identities[x['task']],model);assert value['ba']==x['ba']
        check=grouped_reference(z,feature['logits'],feature['code'],identities[x['task']][0],identities[x['task']][2],model)
        for k in check:
            error=abs(value[k]-check[k]);assert error<1e-6,(x['id'],k,error);max_error=max(max_error,error)
        outputs.append(dict(id=x['id'],task=x['task'],kind=x.get('kind','genuine'),condition=x.get('condition'),update=x.get('update'),scores=value,decisions=decisions(value,model)))
        if stage=='genuine':
            for t in identities:
                if t==x['task']:continue
                v=scores(z,feature['logits'],feature['code'],*identities[t],model)
                wrong.append(dict(id=x['id'],identity=t,decisions=decisions(v,model)))
    methods=list(outputs[0]['decisions']);summary=[]
    for name in methods:
        groups={'genuine':outputs} if stage=='genuine' else {'reference':[x for x in outputs if x['kind']=='reference'],'cover':[x for x in outputs if x['kind']=='cover'],'final_candidate':[x for x in outputs if x['kind']=='candidate' and x['update']==151],'all_checkpoints':[x for x in outputs if x['kind']=='candidate']}
        row=dict(method=name)
        for label,group in groups.items():row[label]=dict(accepted=sum(x['decisions'][name] for x in group),n=len(group))
        if wrong:row['wrong_identity']=dict(accepted=sum(x['decisions'][name] for x in wrong),n=len(wrong))
        summary.append(row)
    result=dict(completed=True,independent_grouped_score_audit=True,maximum_grouped_audit_error=max_error,model_sha256=sha(run/'decoder_model.json'),protocol_sha256=sha(run/'protocol.json'),rows=outputs,wrong_identity=wrong,summary=summary)
    save(run/f'{stage}_defense_results.json',result)
    return result


def report(run,genuine,forgery=None):
    lines=['# Shared semantic evidence experiment','',
      'Twelve new source images; full50-step generation/inversion and100-step decoder optimization. Model parameters and95%-target calibration thresholds frozen on separate prior development sources before generating test images. Actual test acceptance rates are reported; calibration targets do not imply matched test TPR.','',
      '| Rule | Genuine | Wrong identity | Final Imprint | All checkpoints | Covers |','|---|---:|---:|---:|---:|---:|']
    for g in genuine['summary']:
        def fmt(x):return str(x['accepted'])+'/'+str(x['n'])
        f=next((x for x in forgery['summary'] if x['method']==g['method']),None) if forgery else None
        lines.append('| '+g['method']+' | '+fmt(g['genuine'])+' | '+fmt(g['wrong_identity'])+' | '+(' | '.join(fmt(f[k]) for k in ['final_candidate','all_checkpoints','cover']) if f else 'pending | pending | pending')+' |')
    lines+=['','default_ba keeps BA>181/256; all other rules use their genuine-only calibrated thresholds. latent_soft_ba is a GS++-inspired decoder control on this adapter, not a full native GS++ reproduction. group_constant is the fixed semantic-error control; group_query is the candidate. No stored enrollment code is provided at test inference.','',
      'Source-level dependence, only12 test prompts, and four repeated public covers prevent population low-FPR or SOTA claims. The source-separated evaluation is an initial held-out pilot, not paper-scale confirmation.']
    if forgery:
        gs={x['method']:x for x in genuine['summary']};fs={x['method']:x for x in forgery['summary']};c='group_query'
        dominates=all(gs[c]['genuine']['accepted']>=gs[k]['genuine']['accepted'] and fs[c]['final_candidate']['accepted']<=fs[k]['final_candidate']['accepted'] and gs[c]['wrong_identity']['accepted']<=gs[k]['wrong_identity']['accepted'] for k in ['ba','latent_soft_ba','coordinate_soft_ba','group_constant'])
        improves=any(gs[c]['genuine']['accepted']>gs[k]['genuine']['accepted'] or fs[c]['final_candidate']['accepted']<fs[k]['final_candidate']['accepted'] for k in ['ba','latent_soft_ba','coordinate_soft_ba','group_constant'])
        lines+=['',f'Exploratory operating-count dominance over all four controls: {dominates}; at least one strict improvement: {improves}. These are empirical counts, not a statistical superiority test.',
          'Next decision: only a reproducible gain without added false acceptance merits larger independent confirmation; otherwise retain the negative result and do not tune on this test cohort.']
    (run/'analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(run/name)==digest,name
    assert json.loads((run/'unit_checks.json').read_text())['passed']
    def state(status,**kw):save(run/'pipeline_state.json',dict(status=status,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    def launch(command,path,env=None):
        with path.open('w') as f:return subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    def worker(stage,i):return [sys.executable,str(run/'run_anchor_binding_worker.py'),'--run',str(run),'--stage',stage,'--shard',str(i)]
    try:
        state('running',stage='development_logit_extraction')
        ps=[launch([sys.executable,str(run/'extract_semantic_evidence.py'),'--run',str(run),'--shard',str(i)],run/f'features_{i}.log') for i in range(3)]
        state('running',stage='development_logit_extraction',worker_pids=[x.pid for x in ps]);codes=[x.wait() for x in ps];assert not any(codes),codes
        state('running',stage='genuine_only_fit_and_calibration');model=fit(run,plan)
        save(run/'model_freeze.json',dict(model_sha256=sha(run/'decoder_model.json'),frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_generation_started=False,forgery_observed=False))
        time.sleep(10);ps=[launch(worker('genuine',i),run/f'genuine_{i}.log') for i in range(3)]
        state('running',stage='heldout_full_genuine',worker_pids=[x.pid for x in ps]);codes=[x.wait() for x in ps];assert not any(codes),codes
        genuine=audit(run,'genuine',plan);g=evaluate(run,'genuine',plan,genuine,model);report(run,g)
        jobs=[]
        for item in plan['sources']:
            ref=next(x for x in genuine['references'] if x['task']==item['task']);cover=plan['covers'][item['task']%len(plan['covers'])];assert sha(cover['path'])==cover['sha256']
            jobs.append(dict(id=f"{item['task']:03}_full",task=item['task'],method='full',reference=ref['image'],reference_sha256=ref['image_sha256'],cover=cover['path'],cover_sha256=cover['sha256']))
        attack=dict(scope='Unchanged CVPR2025 Imprint on12 frozen SemBind-adapter references; shared outputs for all defense scoring rules',jobs=jobs,updates=151,inverse_steps=50,lr=.01,max_concurrent_gpus=3,excluded_gpu_indices=[6],worker_gpu_indices=[7,5,4],checkpoint_source_steps=list(range(0,151,10)),early_stop=False,shortened_budget_allowed=False,parent_protocol_sha256=sha(run/'protocol.json'),source='https://arxiv.org/abs/2412.03283')
        save(run/'attack_protocol.json',attack)
        def baseline(i):
            uuid=subprocess.check_output(['nvidia-smi','-i',str(plan['gpu_indices'][i]),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            time.sleep(10);cmd=[plan['attack_python'],str(run/'run_published_imprint_full.py'),'--protocol',str(run/'attack_protocol.json'),'--upstream',plan['upstream'],'--model',plan['paths']['model'],'--output',str(run/'attack_outputs'),'--shard',str(i)]
            proc=launch(cmd,run/f'imprint_{i}.log',env);save(run/f'baseline_launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i]));code=proc.wait()
            if code:return code
            time.sleep(10);proc=launch(worker('forgery',i),run/f'forgery_{i}.log');return proc.wait()
        state('running',stage='published_full_imprint')
        with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(baseline,range(3)))
        assert not any(codes),codes
        forged=audit(run,'forgery',plan);f=evaluate(run,'forgery',plan,forged,model);report(run,g,f)
        state('complete',stage='all_endpoints_independently_audited',genuine_records=120,forgery_records=216)
    except Exception as exc:state('failed',error=repr(exc));raise


if __name__=='__main__':main()
