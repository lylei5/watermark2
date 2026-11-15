"""Audit evidence requirements and compare at equal absolute genuine counts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research')
CONTROLS=ROOT/'runs/reprompt_comparison_20261114'
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'
METHODS=['coarse','any_scale','all_scales','observable_scales']
CONTROL_NAMES=['sscd','sscd_centered','sscd_templates','sscd_int8_templates','copynce_descriptor','copynce_matcher']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run):
    assert run.resolve().parent==ROOT/'runs'
    plan=read(PARENT/'protocol.json');done=read(run/'complete.json');assert done['status']=='complete'
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    control_done=read(CONTROLS/'complete.json')
    assert sha(CONTROLS/'content_rows.json')==control_done['artifacts']['content_rows.json']
    controls=read(CONTROLS/'content_rows.json');rows=read(run/'rows.json')
    lookup={(r['id'],r['query_id'],r['kind']):r for r in controls}
    groups={};certificates=0
    for r in rows:
        key=(r['id'],r['query_id'],r['kind']);c=lookup[key]
        assert r['pixels_sha256']==c['pixels_sha256'] and r['native_accepted']==c['native_accepted']
        groups.setdefault(key,{})[r['method']]=r
        if r['key_recovered']:
            d=r['detail'];required=d['required_bands']
            if r['method']=='any_scale':required=[i for i in required if d['bands'][i]['reason']=='recovered']
            assert required
            for i in required:
                b=d['bands'][i];m=b['observed_symbols'];e=b['symbol_errors']
                assert b['reason']=='recovered' and m>=64 and 2*e+255-m<=239 and m-e>=(m+17)//2
                certificates+=1
    for g in groups.values():
        assert set(g)==set(METHODS)
        assert int(g['all_scales']['key_recovered'])<=int(g['observable_scales']['key_recovered'])<=int(g['coarse']['key_recovered'])<=int(g['any_scale']['key_recovered'])
    genuine=lambda r:r['kind'] in ['clean',*plan['conditions']]
    matched=[]
    for carrier in ['GS','TR','PRC']:
        c=[r for r in controls if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
        for method in METHODS:
            w=[r for r in rows if r['carrier']==carrier and r['method']==method and r['split'] in ['evaluation','transfer']]
            gg=[r for r in w if genuine(r)];count=sum(r['joint'] for r in gg)
            for control in CONTROL_NAMES:
                values=sorted(r[control] for r in c if genuine(r) and r['native_accepted'])
                cutoff=values[count-1] if count else -1
                actual=sum(r['native_accepted'] and r[control]<=cutoff for r in c if genuine(r))
                for kind in ['same_prompt','wrong_record','imprint_150','reprompt_sd21','reprompt_sd15']:
                    a=[r for r in w if r['kind']==kind];b=[r for r in c if r['kind']==kind];assert len(a)==len(b)
                    if a:matched.append(dict(carrier=carrier,method=method,control=control,genuine_total=len(gg),candidate_genuine=count,
                        control_genuine=actual,exact_match=count==actual,cutoff=cutoff,kind=kind,total=len(a),
                        candidate_joint=sum(r['joint'] for r in a),control_joint=sum(r['native_accepted'] and r[control]<=cutoff for r in b)))
    activation=[]
    for carrier in ['GS','TR','PRC']:
        w=[r for r in rows if r['carrier']==carrier and r['method']=='observable_scales' and r['split'] in ['evaluation','transfer']]
        for kind in sorted({r['kind'] for r in w}):
            part=[r for r in w if r['kind']==kind]
            for i,name in enumerate(['coarse','middle','fine']):
                active=[r for r in part if r['detail']['bands'][i]['active']]
                activation.append(dict(carrier=carrier,kind=kind,band=name,total=len(part),active=len(active),
                    decoded=sum(r['detail']['bands'][i]['reason']=='recovered' for r in active)))
    result=dict(status='complete',source_sha256=sha(Path(__file__)),witness_complete_sha256=sha(run/'complete.json'),
        control_complete_sha256=sha(CONTROLS/'complete.json'),same_pixel_checks=len(rows),inclusion_checks=len(groups),
        successful_band_certificate_checks=certificates,fresh_band_observations=3*sum(not r['detail']['decode_reused'] for r in rows),
        actual_rs_decode_attempts=sum(sum(b['active'] for b in r['detail']['bands']) for r in rows if not r['detail']['decode_reused']),
        matched=matched,activation=activation,scope='Existing development sources. Post-hoc descriptive equal absolute genuine count matching, not independent confirmation.')
    out=run/'comparison.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(matched=len(matched),better=sum(r['candidate_joint']<r['control_joint'] for r in matched),
        equal=sum(r['candidate_joint']==r['control_joint'] for r in matched),worse=sum(r['candidate_joint']>r['control_joint'] for r in matched),
        certificates=certificates,rs_attempts=result['actual_rs_decode_attempts'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
