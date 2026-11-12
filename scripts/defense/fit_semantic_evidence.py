"""Fit only on declared genuine sources; calibrate once on separate sources."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from anchor_binding import identity,order,embed
from semantic_evidence import scores,decisions,NAMES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit(run,plan):
    features={}
    for i in range(3):
        d=json.loads((run/f'features_{i}/complete.json').read_text());assert d['completed'] and d['protocol_sha256']==sha(run/'protocol.json')
        for row in d['rows']:
            assert row['index'] not in features and sha(run/row['path'])==row['sha256'];features[row['index']]=row
    assert set(features)==set(range(len(plan['development'])))
    data=[];identities={}
    for i,item in enumerate(plan['development']):
        for name in ['latent','enrollment_code','code']:assert sha(item[name])==item[name+'_sha256']
        f=np.load(run/features[i]['path']);h=np.load(item['enrollment_code'])['code'];z=np.load(item['latent'])
        key=(item['run'],item['task'])
        if key not in identities:
            _,c,m,b=identity(item['identity_root'],item['task']);identities[key]=(c,m,order(b,16384))
        data.append(dict(item=item,h=h,z=z,code=f['code'],logits=f['logits'],identity=identities[key]))
    training=[x for x in data if x['item']['split']=='fit'];cal=[x for x in data if x['item']['split']=='calibration']
    assert len(training)==len(cal)==60
    assert {x['item']['run'] for x in training}.isdisjoint({x['item']['run'] for x in cal})
    xx=xy=0.;pairs=[]
    for x in training:
        initial=embed(None,x['h'],x['item']['identity_root'],x['item']['task'],'full').ravel().astype(float);observed=x['z'].ravel().astype(float)
        xx+=float(initial@initial);xy+=float(initial@observed);pairs.append((initial,observed))
    gain=xy/xx;variance=sum(float(np.sum((y-gain*z)**2)) for z,y in pairs)/sum(len(z) for z,y in pairs)
    assert gain>0 and variance>0
    logits=np.stack([x['logits'] for x in training]).astype(float);labels=np.stack([x['h'] for x in training]).astype(float)
    def nll(logtemp):
        value=logits/np.exp(logtemp)
        return float(np.mean(np.logaddexp(0,value)-labels*value))
    optimum=minimize_scalar(nll,bounds=(-8,8),method='bounded',options={'xatol':1e-8,'maxiter':500});assert optimum.success
    error=float(np.mean(np.stack([x['code']!=x['h'] for x in training])));assert 0<error<.5
    model=dict(gain=gain,noise_variance=variance,temperature=float(np.exp(optimum.x)),semantic_error=error,thresholds={},
               source_splits=dict(fit_sources=6,calibration_sources=6),calibration_target=.95,threshold_operator='>=',default_ba_operator='>',default_ba_threshold=181/256)
    calrows=[]
    for x in cal:
        value=scores(x['z'],x['logits'],x['code'],*x['identity'],model)
        calrows.append(dict(task=x['item']['task'],condition=x['item']['condition'],scores=value))
    for name in NAMES:model['thresholds'][name]=float(sorted(x['scores'][name] for x in calrows)[3])
    for x in calrows:x['decisions']=decisions(x['scores'],model)
    model['protocol_sha256']=sha(run/'protocol.json')
    model['feature_manifests_sha256']={str(i):sha(run/f'features_{i}/complete.json') for i in range(3)}
    report=dict(model=model,calibration=calrows,fit_semantic_nll=nll(optimum.x),
                fit_constant_nll=float(-np.mean(np.where(np.stack([x['code'] for x in training])==labels,np.log(1-error),np.log(error)))),
                scope='Six fit sources and six separate calibration sources are prior development cohorts; only forthcoming sources test the frozen method. Correlated coordinates are not independent samples.')
    (run/'decoder_model.json').write_text(json.dumps(model,indent=2)+'\n');(run/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    return model
