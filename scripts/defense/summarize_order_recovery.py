"""Audit and summarize completed order recovery; execute on249 only."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(run):
    done=read(run/'complete.json');plan=read(run/'protocol.json');assert done['status']=='complete'
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    rows=read(run/'rows.json');control=read(run/'controls_complete.json');assert control['status']=='complete'
    assert sha(run/'controls.json')==control['artifacts']['controls.json']
    paired={r['case_id']:r for r in read(run/'controls.json')}
    methods=[m['name'] for m in plan['methods']];aggregate=[];matched=[]
    conditions=['clean','bound','random_crop_ratio','random_drop_ratio','brightness_factor','jpeg_ratio','gaussian_blur_r','gaussian_std','median_blur_k','sp_prob','resize_ratio']
    def genuine(r):return r.get('is_genuine',r['kind'] in conditions)
    for r in rows:
        assert r['returned']==r['key_recovered']
        assert r['pixels_sha256']==paired[r['case_id']]['pixels_sha256']
        assert all(r[n]==paired[r['case_id']][n] for n in ['sscd','sscd_templates'])
        if r['returned']:assert r['detail']['key_confirmed'] and 1<=r['detail']['tested_keys']<=64
    cohorts=['GS','TR','PRC'] if plan['stage']=='development' else ['all',*sorted({r['dataset'] for r in rows})]
    for cohort in cohorts:
        if plan['stage']=='development':
            part=[r for r in rows if r['carrier']==cohort and (cohort!='GS' or r['task']>=7)]
        else:part=[r for r in rows if cohort=='all' or r['dataset']==cohort]
        kinds=sorted({r['kind'] for r in part})
        groups=['all_genuine','all_edits','unseen_edits',*kinds]
        def select(kind):
            if kind=='all_genuine':return [r for r in part if genuine(r)]
            if kind=='all_edits':return [r for r in part if genuine(r) and r['kind'] not in ['clean','bound']]
            if kind=='unseen_edits':return [r for r in part if r.get('is_unseen',False)]
            return [r for r in part if r['kind']==kind]
        for method in methods:
            for kind in groups:
                rr=[r for r in select(kind) if r['method']==method]
                if not rr:continue
                aggregate.append(dict(cohort=cohort,method=method,kind=kind,total=len(rr),accepted=sum(r['returned'] for r in rr),rounded=sum(r['rounded_key'] for r in rr),forward_keys=sum(r['forward_key'] for r in rr),joint=sum(r['returned'] and r.get('native_accepted',True) for r in rr) if plan['stage']=='development' else None))
            gg=[r for r in part if r['method']==method and genuine(r)];count=sum(r['returned'] for r in gg)
            # Exact observed genuine counts; record ties instead of breaking them.
            for name in plan['controls']:
                finite=sorted(r[name] for r in gg if r[name] is not None)
                threshold=finite[count-1] if count and len(finite)>=count else (-1. if not count else None)
                attained=sum(r[name] is not None and threshold is not None and r[name]<=threshold for r in gg)
                for kind in [k for k in kinds if not genuine(next(r for r in part if r['kind']==k))]:
                    negatives=[r for r in part if r['method']==method and r['kind']==kind]
                    candidate={r['id'] for r in negatives if r['returned']};baseline={r['id'] for r in negatives if r[name] is not None and threshold is not None and r[name]<=threshold}
                    matched.append(dict(cohort=cohort,method=method,control=name,kind=kind,genuine_total=len(gg),candidate_genuine=count,control_genuine=attained,exact_match=attained==count,threshold=threshold,
                                        queries=len(negatives),candidate_queries=sum(r['returned'] for r in negatives),control_queries=sum(r[name] is not None and threshold is not None and r[name]<=threshold for r in negatives),
                                        sources=len({r['id'] for r in negatives}),candidate_sources=len(candidate),control_sources=len(baseline)))
        for point in read(run/'calibration.json'):
            for kind in groups:
                rr=[r for r in select(kind) if r['method']==methods[0]]
                if not rr:continue
                name=point['method'];cutoff=point['threshold'];passed=lambda r:r[name] is not None and cutoff is not None and r[name]<=cutoff
                aggregate.append(dict(cohort=cohort,method=name,target=point['target'],kind=kind,total=len(rr),accepted=sum(passed(r) for r in rr),joint=sum(passed(r) and r.get('native_accepted',True) for r in rr) if plan['stage']=='development' else None))
    write(run/'aggregate.json',aggregate);write(run/'matched.json',matched)
    report=dict(status='complete',input_complete_sha256=sha(run/'complete.json'),source_sha256=sha(Path(__file__)),rows=len(rows),paired_queries=len(paired),methods=methods,exact_key_returns=sum(r['returned'] for r in rows),
                source_instances=len({r['id'] for r in rows}),exact_count_comparisons=sum(r['exact_match'] for r in matched),matched_comparisons=len(matched),native_calls=0,
                gpu=control['gpu'],fresh_model_images=control['fresh_model_images'],stage=plan['stage'],artifacts={n:sha(run/n) for n in ['aggregate.json','matched.json']})
    write(run/'audit.json',report)
    print(json.dumps(report))
    print(json.dumps([r for r in aggregate if r['kind'] in ['all_genuine','gaussian_std','brightness_factor','unseen_edits'] and (r.get('target') in [None,.9]) and r['cohort'] in ['TR','all']],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run)
