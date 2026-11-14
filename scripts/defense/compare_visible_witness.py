"""Same-pixel, absolute-TPR comparisons for the visibility recovery experiment."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/visible_witness_recovery_20261114'
BASE=ROOT/'runs/reprompt_comparison_20261114'
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert ROOT.is_dir()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    import numpy as np
    plan=read(PARENT/'protocol.json');done=read(RUN/'complete.json');base_done=read(BASE/'complete.json')
    assert done['status']==base_done['status']=='complete'
    for n,h in done['artifacts'].items():assert sha(RUN/n)==h
    assert sha(BASE/'content_rows.json')==base_done['artifacts']['content_rows.json']
    controls=read(BASE/'content_rows.json');rows=read(RUN/'rows.json')
    lookup={(r['id'],r['query_id'],r['kind']):r for r in controls}
    for r in rows:
        c=lookup[r['id'],r['query_id'],r['kind']]
        assert c['pixels_sha256']==r['pixels_sha256'] and c['native_accepted']==r['native_accepted']
        if r['key_recovered'] and r['method']=='visible':
            d=r['detail'];m=d['observed_symbols'];e=d['symbol_errors']
            assert m>=64 and 2*e+255-m<=239 and m-e>=(m+17)//2
    def genuine(r):return r['kind'] in ['clean',*plan['conditions']]
    methods=['sscd','sscd_centered','sscd_templates','sscd_int8_templates','copynce_descriptor','copynce_matcher']
    matched=[];bootstrap=[]
    for carrier in ['GS','TR','PRC']:
        c=[r for r in controls if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
        for method in ['identity','visible']:
            w=[r for r in rows if r['carrier']==carrier and r['method']==method and r['split'] in ['evaluation','transfer']]
            gg=[r for r in w if genuine(r)];count=sum(r['joint'] for r in gg)
            for control in methods:
                values=sorted(r[control] for r in c if genuine(r) and r['native_accepted'])
                cutoff=values[count-1] if count else -1
                actual=sum(r['native_accepted'] and r[control]<=cutoff for r in c if genuine(r))
                for kind in ['same_prompt','wrong_record','imprint_150','reprompt_sd21','reprompt_sd15']:
                    a=[r for r in w if r['kind']==kind];b=[r for r in c if r['kind']==kind];assert len(a)==len(b)
                    if a:matched.append(dict(carrier=carrier,method=method,control=control,genuine_total=len(gg),
                        candidate_genuine=count,control_genuine=actual,exact_match=count==actual,cutoff=cutoff,kind=kind,total=len(a),
                        candidate_joint=sum(r['joint'] for r in a),control_joint=sum(r['native_accepted'] and r[control]<=cutoff for r in b)))
        for kind in ['all_genuine','non_enrollment_edits','random_crop_ratio','random_drop_ratio','brightness_factor']:
            select=lambda r:genuine(r) if kind=='all_genuine' else r['kind'] in plan['conditions'] and r['kind'] not in plan['enrollment_families'] if kind=='non_enrollment_edits' else r['kind']==kind
            a=[r for r in rows if r['carrier']==carrier and r['method']=='visible' and r['split'] in ['evaluation','transfer'] and select(r)]
            b={(r['id'],r['kind']):r for r in rows if r['carrier']==carrier and r['method']=='identity' and r['split'] in ['evaluation','transfer'] and select(r)}
            tasks=sorted({r['task'] for r in a})
            delta=np.array([sum(int(r['joint'])-int(b[r['id'],r['kind']]['joint']) for r in a if r['task']==task)/sum(r['task']==task for r in a) for task in tasks])
            rng=np.random.default_rng(2026111413);draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            bootstrap.append(dict(carrier=carrier,kind=kind,sources=len(tasks),queries=len(a),difference_pp=float(100*delta.mean()),
                interval95_pp=(100*np.quantile(draws,[.025,.975])).tolist()))
    result=dict(status='complete',source_sha256=sha(__file__),base_complete_sha256=sha(BASE/'complete.json'),
        witness_complete_sha256=sha(RUN/'complete.json'),same_pixel_checks=len(rows),matched=matched,source_bootstrap=bootstrap,
        certificate_checks=sum(r['key_recovered'] and r['method']=='visible' for r in rows),
        scope='Development data; post-hoc descriptive absolute-TPR matching; no independent confirmation or FPR guarantee.')
    p=RUN/'comparison.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(matched=len(matched),certificates=result['certificate_checks'],bootstrap=bootstrap)),flush=True)


if __name__=='__main__':main()
