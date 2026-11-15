"""Genuine-only code-budget calibration on completed recovery observations."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research/runs')
RUN=ROOT/'observable_evidence_recovery_20261115'
PARENT=ROOT/'shared_channel_binding_20261114_v2'
CONTROL=ROOT/'reprompt_comparison_20261114'
METHODS=['coarse','any_scale','all_scales','observable_scales']
CONTROLS=['sscd','sscd_centered','sscd_templates','sscd_int8_templates','copynce_descriptor','copynce_matcher']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def score(r):
    if not r['key_recovered']:return math.inf
    d=r['detail'];values=[b['code_budget'] for i,b in enumerate(d['bands']) if i in d['required_bands'] and b['reason']=='recovered']
    return min(values) if r['method']=='any_scale' else max(values)


def main():
    done=read(RUN/'complete.json')
    for name,digest in done['artifacts'].items():assert sha(RUN/name)==digest
    rows=read(RUN/'rows.json');controls=read(CONTROL/'content_rows.json');parent=read(PARENT/'protocol.json')
    genuine=lambda r:r['kind'] in ['clean',*parent['conditions']]
    result=[];matched=[]
    for method in METHODS:
        cal=[r for r in rows if r['carrier']=='GS' and r['split']=='calibration' and r['method']==method and genuine(r)]
        assert len(cal)==30
        for target in [.9,.95]:
            threshold=sorted(score(r) for r in cal)[math.ceil(target*len(cal))-1]
            feasible=math.isfinite(threshold)
            for carrier in ['GS','TR','PRC']:
                part=[r for r in rows if r['carrier']==carrier and r['method']==method and r['split'] in ['evaluation','transfer']]
                gs=[r for r in part if genuine(r)]
                count=sum(r['joint'] and score(r)<=threshold for r in gs) if feasible else None
                groups={k:[r for r in part if r['kind']==k] for k in ['imprint_150','reprompt_sd21','reprompt_sd15','same_prompt','wrong_record']}
                result.append(dict(method=method,target=target,threshold=threshold if feasible else None,feasible=feasible,carrier=carrier,
                    genuine_joint=count,genuine_total=len(gs),negatives={k:dict(accepted=sum(r['joint'] and score(r)<=threshold for r in rr) if feasible else None,total=len(rr)) for k,rr in groups.items()}))
                if not feasible:continue
                cc=[r for r in controls if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
                for control in CONTROLS:
                    values=sorted(r[control] for r in cc if genuine(r) and r['native_accepted'])
                    cutoff=values[count-1] if count else -1
                    attained=sum(r['native_accepted'] and r[control]<=cutoff for r in cc if genuine(r))
                    for kind,rr in groups.items():
                        ca=[r for r in cc if r['kind']==kind];assert len(ca)==len(rr)
                        if rr:matched.append(dict(method=method,target=target,carrier=carrier,control=control,kind=kind,genuine=count,control_genuine=attained,
                            candidate_joint=sum(r['joint'] and score(r)<=threshold for r in rr),control_joint=sum(r['native_accepted'] and r[control]<=cutoff for r in ca),negative_total=len(rr)))
    out=dict(status='complete',source_sha256=sha(Path(__file__)),recovery_complete_sha256=sha(RUN/'complete.json'),
        scope='Post-hoc score control on completed development observations; no new decoder or independent confirmation. Thresholds use GS30genuine calibration only.',
        score='Maximum required-band RS error/erasure budget 2e+s after correct recovery; any_scale uses minimum recovered-band budget. Failed recovery has infinite score.',
        calibrated=result,matched=matched)
    path=RUN/'budget_calibration.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(calibrated=result,matched=len(matched),better=sum(r['candidate_joint']<r['control_joint'] for r in matched),equal=sum(r['candidate_joint']==r['control_joint'] for r in matched),worse=sum(r['candidate_joint']>r['control_joint'] for r in matched)),allow_nan=False))


if __name__=='__main__':main()
