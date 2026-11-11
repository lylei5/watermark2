"""Source-disjoint fit/calibration/test for a defense-only likelihood experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import numpy as np
from conditional_content_likelihood import channel_features,fit_channel,predict_channel,score


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['fit','evaluate'],required=True)
    args=p.parse_args();run=args.run.resolve()
    plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items():
        assert sha(run/name)==digest,name
    for name,digest in plan['external_sha256'].items():
        assert sha(name)==digest,name
    chain=Path(plan['chain_run']);conditions=Path(plan['conditions_run']);baseline=Path(plan['baseline_run'])
    paths={k:Path(v) for k,v in json.loads((chain/'paths.json').read_text()).items()}
    sys.path.insert(0,str(chain/'scripts'))
    sys.path.insert(1,str(run/'helpers'))
    from sembind_official_audit import carrier
    from content_authentication import recover_groups,content_scores
    identities={i:carrier(paths['root'],i) for i in range(4)}
    def metrics(z,code,i,model,d):
        _,cipher,message,perm=identities[i]
        signs=(z.reshape(-1)>0).astype(np.uint8)
        bits=signs ^ np.tile(code,16)[perm] ^ cipher ^ np.tile(message,64)
        ba=float(((bits.reshape(64,256).sum(0)>32)==message).mean())
        old=content_scores(recover_groups(z,cipher,perm),code)
        return dict(ba=ba,**score(z,cipher,perm,code,model,d),
                    content_agreement=old['content_agreement'],
                    profile_content_lower_bound=old['profile_content_lower_bound'])
    names=plan['score_names']
    if args.stage=='fit':
        assert not (run/'frozen_models.json').exists()
        original=json.loads((conditions/'verified_results.json').read_text())
        assert original['audit_passed']
        records=[]
        for r in original['rows']:
            if r['method']!='sembind' or r['setting']!='blind_decoder':continue
            zpath=conditions/r['latent'];base=zpath.parent.parent
            codepath=base/'codes'/(r['id']+'.npz')
            assert sha(zpath)==r['latent_sha256'] and sha(codepath)==r['code_sha256']
            records.append(dict(task=r['task'],condition=r['condition'],latent=str(zpath),code=str(codepath),ba=r['query_ba']))
        clean=json.loads((chain/'verified_results.json').read_text())
        assert clean['passed']
        for r in clean['results']['rows']:
            if r['method']=='sembind' and r['setting']=='blind_decoder':
                records.append(dict(task=r['task'],condition='clean',
                    latent=str(paths['output']/'inversions'/f"{r['task']:03}_sembind_blind_decoder.npy"),
                    code=str(paths['output']/'codes'/f"{r['task']:03}_sembind.npz"),ba=r['final_ba']))
        assert len(records)==40 and len({(r['task'],r['condition']) for r in records})==40
        arrays=[]
        for r in records:
            z=np.load(r['latent']);q=np.load(r['code'])['sembind_code'];i=r['task']
            _,cipher,_,perm=identities[i]
            enrollment=np.load(paths['output']/'codes'/f'{i:03}_aux.npz')['sembind_code']
            errors=(z.reshape(-1)>0).astype(np.uint8) ^ cipher ^ np.tile(enrollment,16)[perm]
            arrays.append(dict(z=z,q=q,x=channel_features(z),errors=errors,d=float((q!=enrollment).mean())))
            r.update(latent_sha256=sha(r['latent']),code_sha256=sha(r['code']))
        folds=[];rows=[]
        for test in range(4):
            calibration=(test+1)%4;training=[i for i in range(4) if i not in [test,calibration]]
            selected=[a for r,a in zip(records,arrays) if r['task'] in training]
            model=fit_channel([a['x'] for a in selected],[a['errors'] for a in selected],l2=plan['l2'])
            d=float(np.mean([a['d'] for a in selected]))
            scores=[]
            for r,a in zip(records,arrays):
                if r['task'] not in [test,calibration]:continue
                m=metrics(a['z'],a['q'],r['task'],model,d);assert m['ba']==r['ba']
                pr=predict_channel(model,a['x']);y=a['errors'];e=np.clip(model['training_mean_error'],.005,.495)
                m.update(channel_logloss=float(-(y*np.log(pr)+(1-y)*np.log1p(-pr)).mean()),
                    constant_channel_logloss=float(-(y*np.log(e)+(1-y)*np.log1p(-e)).mean()),
                    channel_brier=float(np.mean((pr-y)**2)),
                    true_channel_error=float(y.mean()))
                scores.append(dict(task=r['task'],condition=r['condition'],**m))
            thresholds={name:min(s[name] for s in scores if s['task']==calibration) for name in names}
            folds.append(dict(test_source=test,calibration_source=calibration,training_sources=training,
                              model=model,benign_mismatch=d,thresholds=thresholds))
            for row in scores:
                if row['task']!=test:continue
                row['decisions']={name:row[name]>=thresholds[name] for name in names}
                row['default_ba_accepted']=row['ba']>181/256;rows.append(row)
            print('FOLD_COMPLETE',test,flush=True)
        result=dict(protocol_sha256=sha(run/'protocol.json'),folds=folds,inputs=records,
                    test_genuine_rows=rows,scope='Four-source development cross-fitting, not independent confirmation')
        save(run/'frozen_models.json',result)
        summary={name:dict(accepted=sum(r['decisions'][name] for r in rows),n=len(rows)) for name in names}
        save(run/'genuine_results.json',dict(completed=True,summary=summary,rows=rows,
            mean_channel_logloss=float(np.mean([r['channel_logloss'] for r in rows])),
            mean_constant_channel_logloss=float(np.mean([r['constant_channel_logloss'] for r in rows])),
            model_sha256=sha(run/'frozen_models.json'),protocol_sha256=sha(run/'protocol.json')))
        print(json.dumps(summary),flush=True)
        return
    frozen=json.loads((run/'frozen_models.json').read_text())
    assert frozen['protocol_sha256']==sha(run/'protocol.json')
    result=json.loads((baseline/'verified_results.json').read_text())
    assert result['audit_passed'] and result['updates_per_job']==151
    rows=[]
    for r in result['rows']:
        if r['method']!='sembind':continue
        fold=frozen['folds'][r['task']]
        assert fold['test_source']==r['task'] and r['task'] not in fold['training_sources']
        found=list((baseline/'evaluation').glob('shard_*/'+r['id']+'.npy'));assert len(found)==1
        path=found[0];codepath=path.with_suffix('.npz')
        assert sha(path)==r['inversion_sha256'] and sha(codepath)==r['features_sha256']
        z=np.load(path);q=np.load(codepath)['sembind_code']
        m=metrics(z,q,r['task'],fold['model'],fold['benign_mismatch']);assert m['ba']==r['ba']
        rows.append(dict(id=r['id'],task=r['task'],kind=r['kind'],update=r.get('update'),**m,
            decisions={name:m[name]>=fold['thresholds'][name] for name in names},
            default_ba_accepted=m['ba']>181/256))
    assert len(rows)==72
    genuine=frozen['test_genuine_rows'];summary=[]
    for name in names:
        final=[r for r in rows if r['kind']=='candidate' and r['update']==151]
        checkpoints=[r for r in rows if r['kind']=='candidate']
        covers=[r for r in rows if r['kind']=='cover']
        summary.append(dict(score=name,genuine_accepted=sum(r['decisions'][name] for r in genuine),genuine_n=40,
            final_forgery_accepted=sum(r['decisions'][name] for r in final),final_forgery_n=4,
            all_checkpoint_accepted=sum(r['decisions'][name] for r in checkpoints),all_checkpoint_n=64,
            cover_accepted=sum(r['decisions'][name] for r in covers),cover_n=4))
    base=next(s for s in summary if s['score']=='ba');candidate=next(s for s in summary if s['score']=='conditional_content_llr')
    dominates=(candidate['genuine_accepted']>=base['genuine_accepted'] and candidate['final_forgery_accepted']<=base['final_forgery_accepted']
        and (candidate['genuine_accepted']>base['genuine_accepted'] or candidate['final_forgery_accepted']<base['final_forgery_accepted']))
    save(run/'complete.json',dict(completed=True,summary=summary,rows=rows,genuine_rows=genuine,
        empirical_dominance_over_calibrated_ba=dominates,independent_confirmation=False,
        protocol_sha256=sha(run/'protocol.json'),models_sha256=sha(run/'frozen_models.json'),
        inference='No superiority if operating counts tie; checkpoints share sources. Composite likelihood has no claimed probability calibration.'))
    lines=['# Conditional content likelihood experiment','','Genuine-only fit, separate calibration source, and held-out source per fold. No forged image enters fitting or threshold choice. Four reused development sources; not independent confirmation.','',
        '| Detector | Genuine | Final Imprint | All checkpoints | Covers |','|---|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['score']} | {s['genuine_accepted']}/40 | {s['final_forgery_accepted']}/4 | {s['all_checkpoint_accepted']}/64 | {s['cover_accepted']}/4 |")
    lines+=['',f'Empirical operating-count dominance over calibrated BA: {dominates}. Ties do not establish added defense benefit.',
        'Next decision: only a reproducible advantage over both calibrated BA and the global-likelihood/weighted-vote ablations merits independent confirmation. Otherwise retain this as a negative method result.']
    (run/'analysis.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
