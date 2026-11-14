"""Join same-pixel Reprompt controls without fitting thresholds on attacks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'
COPY=ROOT/'runs/copynce_control_20261114'
EVAL=ROOT/'runs/author_reprompt_evaluation_20261114_v2'
WITNESS=ROOT/'runs/spatial_witness_recovery_20261114_v2'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main(output):
    assert ROOT.is_dir() and output.resolve().parent==ROOT/'runs'
    output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
    import numpy as np
    from scipy.stats import beta
    assert read(EVAL/'complete.json')['status']=='complete'
    assert read(WITNESS/'complete.json')['status']=='complete'
    done=read(EVAL/'content_complete.json')
    for n,h in done['artifacts'].items():assert sha(EVAL/n)==h
    for n,h in read(WITNESS/'complete.json')['artifacts'].items():assert sha(WITNESS/n)==h
    plan=read(PARENT/'protocol.json')
    old=read(PARENT/'controls.json')
    copy={(r['id'],r['query_id'],r['kind']):r for r in read(COPY/'rows.json')}
    for r in old:
        c=copy[r['id'],r['query_id'],r['kind']];assert c['pixels_sha256']==r['pixels_sha256']
        r['copynce_descriptor']=c['descriptor_distance'];r['copynce_matcher']=c['matcher_distance']
    rows=read(EVAL/'content_rows.json');data=np.load(EVAL/'features.npz')['sscd']
    references=read(ROOT/'runs/author_reprompt_control_20261114_v2/protocol.json')['references']
    indices={r['id']:i for i,r in enumerate(references)}
    center=np.load(PARENT/'centers.npz')['sscd_centered']
    templates=np.load(PARENT/'sscd_templates.npz')
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    native={};native_rows=[]
    for carrier in ['GS','TR','PRC']:
        done=read(EVAL/carrier/'complete.json');assert sha(EVAL/carrier/'rows.json')==done['rows_sha256']
        part=read(EVAL/carrier/'rows.json');native_rows.extend(part)
        for r in part:native[r['id'],r['proxy'],r['defended']]=r
    new=[]
    for i,r in enumerate(rows):
        j=indices[r['id']];q=data[30+i];ref=data[j];ts=templates[r['id']]
        int8=unit(np.round(ts/np.max(np.abs(ts),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32))
        n=native[r['id'],r['proxy'],False];assert n['pixels_sha256']==r['pixels_sha256']
        single=float(np.clip(1-ref@q,0,2));assert abs(single-r['sscd_distance'])<1e-6
        new.append(dict(id=r['id'],query_id=r['id'],carrier=r['carrier'],task=r['task'],kind='reprompt_'+r['proxy'],
            split='transfer' if r['carrier']!='GS' else 'fit' if r['task'] in plan['fit_sources'] else 'calibration' if r['task'] in plan['calibration_sources'] else 'evaluation',
            pixels_sha256=r['pixels_sha256'],native_accepted=n['accepted'],native_metric=n['metric'],
            sscd=single,sscd_centered=float(np.clip(1-unit(ref-center)@unit(q-center),0,2)),
            sscd_templates=float(np.clip(1-np.max(ts@q),0,2)),sscd_int8_templates=float(np.clip(1-np.max(int8@q),0,2)),
            copynce_descriptor=r['descriptor_distance'],copynce_matcher=r['matcher_distance']))
    allrows=old+new
    methods=['sscd','sscd_centered','sscd_templates','sscd_int8_templates','copynce_descriptor','copynce_matcher']
    def genuine(r):return r['kind'] in ['clean',*plan['conditions']]
    thresholds={}
    calibrated=[]
    for method in methods:
        values=sorted(r[method] for r in old if r['split']=='calibration' and genuine(r));assert len(values)==30
        for target,rank in [(.9,26),(.95,28)]:
            cutoff=values[rank];thresholds[method,str(target)]=cutoff
            for carrier in ['GS','TR','PRC']:
                part=[r for r in allrows if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
                for kind in ['all_genuine','same_prompt','wrong_record','imprint_150','reprompt_sd21','reprompt_sd15']:
                    group=[r for r in part if genuine(r)] if kind=='all_genuine' else [r for r in part if r['kind']==kind]
                    if group:calibrated.append(dict(method=method,target=target,threshold=cutoff,carrier=carrier,kind=kind,total=len(group),
                        accepted=sum(r[method]<=cutoff for r in group),joint=sum(r[method]<=cutoff and r['native_accepted'] for r in group)))
    witnesses=read(WITNESS/'rows.json');matched=[]
    for carrier in ['GS','TR','PRC']:
        for wm in ['identity','registered','heldout']:
            w=[r for r in witnesses if r['carrier']==carrier and r['method']==wm and r['split'] in ['evaluation','transfer']]
            genuine_w=[r for r in w if genuine(r)];count=sum(r['joint'] for r in genuine_w)
            c=[r for r in allrows if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
            for method in methods:
                values=sorted(r[method] for r in c if genuine(r) and r['native_accepted'])
                cutoff=values[count-1] if count else -1.
                attained=sum(r['native_accepted'] and r[method]<=cutoff for r in c if genuine(r))
                for kind in ['same_prompt','wrong_record','imprint_150','reprompt_sd21','reprompt_sd15']:
                    a=[r for r in w if r['kind']==kind];b=[r for r in c if r['kind']==kind]
                    assert len(a)==len(b)
                    if a:matched.append(dict(carrier=carrier,witness_method=wm,control=method,genuine_total=len(genuine_w),
                        witness_genuine=count,control_genuine=attained,exact_match=count==attained,control_threshold=cutoff,kind=kind,total=len(a),
                        witness_joint=sum(r['joint'] for r in a),control_joint=sum(r['native_accepted'] and r[method]<=cutoff for r in b)))
    # Reuse all three previously frozen SSCD code-offset helpers, with fresh query features.
    sys.path.insert(0,str(PARENT));from instance_coset_recovery import InstanceRecovery
    private=Path(plan['private']);provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==plan['provider_sha256']
    enrollment={(r['id'],r['repetition'],r['method']):r for r in read(PARENT/'enrollment.json')}
    recovery=[]
    for rep in range(3):
        assert sha(private/f'projection_{rep}.npy')==plan['projection_sha256'][rep]
        codec=InstanceRecovery(np.load(private/f'projection_{rep}.npy'))
        for method in ['sscd','sscd_centered']:
            for i,r in enumerate(new):
                helper=private/f"{r['id']}_{rep}_{method}.json";assert sha(helper)==enrollment[r['id'],rep,method]['helper_sha256']
                t=time.perf_counter();q=data[30+i] if method=='sscd' else unit(data[30+i]-center)
                recovered,detail=codec.recover(read(helper),q,provider)
                recovery.append(dict(id=r['id'],carrier=r['carrier'],kind=r['kind'],split=r['split'],repetition=rep,method=method,
                    key_recovered=recovered is not None,joint=bool(recovered is not None and r['native_accepted']),detail=detail,seconds=time.perf_counter()-t))
    summary=[]
    for c in ['GS','TR','PRC']:
        for proxy in ['sd21','sd15']:
            for defended in [False,True]:
                p=[r for r in native_rows if r['carrier']==c and r['proxy']==proxy and r['defended']==defended]
                k=sum(r['accepted'] for r in p);n=len(p);assert n==10
                summary.append(dict(carrier=c,proxy=proxy,defended=defended,accepted=k,total=n,
                    interval95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]))
    write(output/'content_rows.json',allrows);write(output/'calibrated.json',calibrated);write(output/'matched.json',matched)
    write(output/'frozen_key_recovery.json',recovery);write(output/'native_summary.json',summary)
    write(output/'complete.json',dict(status='complete',source_sha256=sha(__file__),author_attack_protocol_sha256=sha(ROOT/'runs/author_reprompt_control_20261114_v2/protocol.json'),
        evaluation_complete_sha256=sha(EVAL/'complete.json'),witness_complete_sha256=sha(WITNESS/'complete.json'),
        content_rows=len(allrows),calibrated_rows=len(calibrated),matched_comparisons=len(matched),additional_actual_key_decodes=len(recovery),
        matched_scope='Descriptive evaluation-selected absolute genuine counts, not independent confirmation; retain ties.',
        artifacts={n:sha(output/n) for n in ['content_rows.json','calibrated.json','matched.json','frozen_key_recovery.json','native_summary.json']}))
    print(json.dumps(dict(native=summary,matched=len(matched),recovery_success=sum(r['key_recovered'] for r in recovery))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
