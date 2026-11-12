"""Server-only genuine endpoint diagnosis; no changes to frozen inference."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run;sys.path.insert(0,str(r))
    from anchor_binding import identity,order,embed
    from semantic_evidence import scores,carrier_posterior
    from semantic_evidence_reference import grouped_reference
    p=json.loads((r/'protocol.json').read_text());m=json.loads((r/'decoder_model.json').read_text());d=json.loads((r/'genuine_verified.json').read_text());actual=json.loads((r/'genuine_defense_results.json').read_text());lookup={x['id']:x for x in actual['rows']};cache={};rows=[];saturated=0
    for x in d['rows']:
        t=x['task'];f=np.load(r/x['code']);original=np.load(r/x['latent']);z=original.astype(np.float64);h=np.load(r/f'genuine_{t%3}/codes/{t:03}_aux.npz')['code']
        saturated+=int(np.sum(np.abs(carrier_posterior(original,m))>=1))
        if t not in cache:
            _,c,msg,k=identity(p['identity_root'],t);cache[t]=(c,msg,order(k,16384),embed(None,h,p['identity_root'],t,'full').ravel().astype(float))
        c,msg,perm,initial=cache[t];y=z.ravel();a=float(initial@y/(initial@initial));v=float(np.mean((y-a*initial)**2));logits=f['logits'].astype(float)
        objective=lambda logt:float(np.mean(np.logaddexp(0,logits/np.exp(logt))-h*logits/np.exp(logt)))
        optimum=minimize_scalar(objective,bounds=(-8,8),method='bounded');assert optimum.success
        row=dict(task=t,condition=x['condition'],decisions=lookup[x['id']]['decisions'],semantic_ber=float(np.mean(h!=f['code'])),frozen_semantic_nll=objective(np.log(m['temperature'])),oracle_semantic_nll=objective(optimum.x),oracle_gain=a,oracle_variance=v)
        for label,change in [('oracle_channel',dict(gain=max(a,1e-6),noise_variance=max(v,1e-6))),('oracle_semantic',dict(temperature=float(np.exp(optimum.x)))),('oracle_both',dict(gain=max(a,1e-6),noise_variance=max(v,1e-6),temperature=float(np.exp(optimum.x))))]:
            model=dict(m,**change);value=scores(z,f['logits'],f['code'],c,msg,perm,model)['group_query']
            check=grouped_reference(z,f['logits'],f['code'],c,perm,model)['group_query']
            assert np.isfinite(value) and np.isfinite(check) and abs(value-check)<1e-6,(x['id'],label,value,check)
            row[label]=value;row[label+'_accepted']=value>=m['thresholds']['group_query']
        rows.append(row)
    summary=dict(default_ba=sum(x['decisions']['default_ba'] for x in rows),candidate=sum(x['decisions']['group_query'] for x in rows),oracle_accepts={k:sum(x[k+'_accepted'] for x in rows) for k in ['oracle_channel','oracle_semantic','oracle_both']},frozen_semantic_nll=float(np.mean([x['frozen_semantic_nll'] for x in rows])),oracle_semantic_nll=float(np.mean([x['oracle_semantic_nll'] for x in rows])),original_frozen_saturated_coordinates=saturated)
    conditions=[]
    for c in p['conditions']:
        v=[x for x in rows if x['condition']==c['name']]
        conditions.append(dict(condition=c['name'],default_ba=sum(x['decisions']['default_ba'] for x in v),candidate=sum(x['decisions']['group_query'] for x in v),oracle_both=sum(x['oracle_both_accepted'] for x in v),semantic_ber=float(np.mean([x['semantic_ber'] for x in v]))))
    result=dict(scope='Post-hoc diagnostic on consumed genuine test cohort; oracle parameters use unavailable enrollment information, original endpoint decisions unchanged',numerics='Oracle evaluations use float64 and independent probability-space checks; frozen production source remains unchanged',summary=summary,conditions=conditions,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (r/'genuine_failure_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,conditions=conditions),indent=2))


if __name__=='__main__':main()
