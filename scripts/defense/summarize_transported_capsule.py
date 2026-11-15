"""Audit capsule recovery and same-pixel controls on249; no attack generation."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CONTROLS = ['sscd', 'sscd_centered', 'sscd_templates', 'sscd_int8_templates',
            'copynce_descriptor', 'copynce_matcher']
GENUINE = ['clean', 'random_crop_ratio', 'random_drop_ratio', 'brightness_factor',
           'jpeg_ratio', 'gaussian_blur_r', 'gaussian_std', 'median_blur_k', 'sp_prob', 'resize_ratio']


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def main(run):
    assert str(run).startswith('/data0/liyanlei/watermark_research/runs/')
    done = read(run/'complete.json'); assert done['status'] == 'complete'
    for name, digest in done['artifacts'].items(): assert sha(run/name) == digest
    rows = read(run/'rows.json'); cases = read(run/'cases.json'); decisions = read(run/'decisions.json')
    by_case = {c['base']['case_id']: c['base'] for c in cases}
    assert len(by_case) == len(cases) == 1010
    methods = sorted({r['method'] for r in rows}); seen = set()
    for r in rows:
        k = (r['case_id'], r['method']); assert k not in seen; seen.add(k)
        assert r['pixels_sha256'] == by_case[r['case_id']]['pixels_sha256']
        assert r['native_accepted'] == by_case[r['case_id']]['native_accepted']
        if r['key_recovered']: assert r['detail']['key_confirmed']
    assert len(rows) == len(cases)*len(methods)
    assert sum(r['key_recovered'] for r in rows) == done['exact_key_returns']
    row_index = {(r['case_id'], r['method']): r for r in rows}
    for d in decisions:
        r = row_index[d['case_id'], d['method']]
        assert not d['returned'] or r['key_recovered']
        assert d['joint'] == (d['returned'] and r['native_accepted'])
    assert sum(d['returned'] for d in decisions) == done['thresholded_payload_returns']
    primary = [r for r in rows if r['split'] in ['evaluation', 'transfer']]
    detail = []
    for method in methods:
        for carrier in ['GS', 'TR', 'PRC']:
            part = [r for r in primary if r['method']==method and r['carrier']==carrier]
            for kind in ['all_genuine'] + sorted({r['kind'] for r in part}):
                rr = [r for r in part if r['kind'] in GENUINE] if kind=='all_genuine' else [r for r in part if r['kind']==kind]
                detail.append(dict(method=method, carrier=carrier, kind=kind, total=len(rr),
                    keys=sum(r['key_recovered'] for r in rr),
                    key_and_saved_native=sum(r['key_recovered'] and r['native_accepted'] for r in rr),
                    reasons=dict(Counter(r['detail']['reason'] for r in rr))))
    calibration = read(run/'calibration.json')
    frozen = []; matched = []; operating = []
    canonical = [c['base'] for c in cases]
    cal = [r for r in canonical if r['carrier']=='GS' and r['split']=='calibration' and r['kind'] in GENUINE]
    assert len(cal)==30
    for method in methods:
        part = [r for r in primary if r['method']==method]
        rr = [r for r in rows if r['method']==method]
        for target in [.9, .95]:
            point = next(c for c in calibration if c['method']==method and c['score_name']=='score' and c['target']==target)
            dd = {d['case_id']:d for d in decisions if d['method']==method and d['target']==target}
            for carrier in ['GS','TR','PRC']:
                genuine = [r for r in part if r['carrier']==carrier and r['kind'] in GENUINE]
                count = sum(dd[r['case_id']]['joint'] for r in genuine)
                operating.append(dict(method=method,target=target,carrier=carrier,feasible=point['feasible'],
                    genuine_joint=count if point['feasible'] else None,total=len(genuine)))
                if not point['feasible']: continue
                for control in CONTROLS + ['reference_residual','pixel_residual']:
                    available = sorted(r[control] for r in genuine if r['native_accepted'] and r[control] is not None)
                    if len(available)<count: continue
                    cutoff = available[count-1] if count else -1.
                    attained = sum(r['native_accepted'] and r[control] is not None and r[control]<=cutoff for r in genuine)
                    for kind in ['imprint_150','reprompt_sd21','reprompt_sd15','reprompt_plus_sd21','reprompt_plus_sd15','same_prompt','wrong_record']:
                        group = [r for r in part if r['carrier']==carrier and r['kind']==kind]
                        if not group: continue
                        ids = sorted({r['id'] for r in group})
                        candidate = sum(any(dd[r['case_id']]['joint'] for r in group if r['id']==i) for i in ids)
                        baseline = sum(any(r['native_accepted'] and r[control] is not None and r[control]<=cutoff for r in group if r['id']==i) for i in ids)
                        matched.append(dict(method=method,target=target,carrier=carrier,control=control,kind=kind,
                            candidate_genuine=count,control_genuine=attained,genuine_total=len(genuine),exact_match=count==attained,
                            threshold=cutoff,candidate_sources=candidate,control_sources=baseline,source_total=len(ids),
                            scope='Descriptive evaluation-genuine-count matching; not independent calibration.'))
    for control in CONTROLS:
        for target in [.9,.95]:
            cutoff = sorted(r[control] for r in cal)[math.ceil(target*30)-1]
            for carrier in ['GS','TR','PRC']:
                part = [r for r in canonical if r['carrier']==carrier and r['split'] in ['evaluation','transfer']]
                for kind in ['all_genuine','imprint_150','reprompt_sd21','reprompt_sd15','reprompt_plus_sd21','reprompt_plus_sd15']:
                    group = [r for r in part if r['kind'] in GENUINE] if kind=='all_genuine' else [r for r in part if r['kind']==kind]
                    if not group: continue
                    ids = sorted({r['id'] for r in group})
                    frozen.append(dict(method=control,target=target,carrier=carrier,kind=kind,threshold=cutoff,total=len(group),
                        content=sum(r[control]<=cutoff for r in group),joint=sum(r['native_accepted'] and r[control]<=cutoff for r in group),
                        accepted_sources=sum(any(r['native_accepted'] and r[control]<=cutoff for r in group if r['id']==i) for i in ids),source_total=len(ids)))
    paired = [r for r in rows if r['key_recovered'] and r['reference_residual'] is not None]
    equivalence = dict(pairs=len(paired),max_absolute_difference=max(abs(r['score']-r['reference_residual']) for r in paired),
        identity='For an exact decoded s, alpha*A*B*s-(A*h+b-y)=y-(A*x+b). Rounding of stored float32 h explains small differences.',
        implication='Certificate is a direct forward-reference residual, not an independent image-discrimination mechanism.')
    enroll = read(run/'enrollment.json')
    audit = dict(status='complete',source_sha256=sha(Path(__file__)),complete_sha256=sha(run/'complete.json'),
        cases=len(cases),same_pixel_checks=len(rows),key_confirmations=sum(r['key_recovered'] for r in rows),
        feasible_calibration_points=sum(c['feasible'] for c in calibration if c['score_name']=='score'),
        residual_equivalence=equivalence,helper_bytes=[min(r['helper_bytes'] for r in enroll),max(r['helper_bytes'] for r in enroll)],
        matched_comparisons=dict(Counter('better' if r['candidate_sources']<r['control_sources'] else 'equal' if r['candidate_sources']==r['control_sources'] else 'worse' for r in matched)),
        source_scope='Previously examined development sources; GS3 evaluation sources, TR/PRC10 each; no FPR1e-6 or independent-confirmation claim.')
    # Recalibration stress test of saved observations, not new decode/native calls.
    heldout=[]
    for method in methods+CONTROLS:
        data=[r for r in rows if r['method']==method] if method in methods else canonical
        score='score' if method in methods else method
        for family in GENUINE[1:]:
            cal=[r for r in data if r['carrier']=='GS' and r['split']=='calibration' and r['kind'] in GENUINE and r['kind']!=family]
            assert len(cal)==27
            finite=sorted(r[score] for r in cal if r[score] is not None)
            for target in [.9,.95]:
                needed=math.ceil(target*len(cal));cutoff=finite[needed-1] if len(finite)>=needed else None
                for carrier in ['GS','TR','PRC']:
                    part=[r for r in data if r['carrier']==carrier and r['split'] in ['evaluation','transfer'] and r['kind']==family]
                    heldout.append(dict(method=method,family=family,target=target,carrier=carrier,calibration_total=len(cal),
                        threshold=cutoff,feasible=cutoff is not None,total=len(part),
                        content=sum(r[score] is not None and r[score]<=cutoff for r in part) if cutoff is not None else None,
                        joint=sum(r['native_accepted'] and r[score] is not None and r[score]<=cutoff for r in part) if cutoff is not None else None))
    heldout_result=dict(scope='Post-hoc exclusion of one edit family from threshold calibration; existing development sources and frozen observations. No claim of unseen development transformations or fresh payload/native calls.',rows=heldout)
    for name,value in [('audit',audit),('groups',detail),('operating_points',operating),('fixed_controls',frozen),('matched',matched),('heldout_calibration',heldout_result)]:write(run/(name+'.json'),value)
    print(json.dumps(audit)); print(json.dumps([r for r in detail if r['kind']=='all_genuine']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
