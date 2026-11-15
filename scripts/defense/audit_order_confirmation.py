"""Same-pixel legacy joins and per-family matched controls after completion."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(run):
    complete=read(run/'complete.json');legacy=read(run/'legacy_complete.json');assert complete['status']==legacy['status']=='complete'
    assert sha(run/'rows.json')==complete['artifacts']['rows.json']
    assert sha(run/'legacy_rows.json')==legacy['rows_sha256']
    rows=read(run/'rows.json');old={r['case_id']:r for r in read(run/'legacy_rows.json')};plan=read(run/'protocol.json');analysis=read(run/'analysis_plan.json')
    primary=analysis['primary_candidate'];rr=[r for r in rows if r['method']==primary];assert len(rr)==3800
    controls=plan['controls'];cal={p['method']:p['threshold'] for p in read(run/'calibration.json') if p['target']==analysis['primary_control_calibration_target']}
    for row in rows:assert row['pixels_sha256']==old[row['case_id']]['pixels_sha256']
    def subset(name):
        if name=='known_genuine':return [r for r in rr if r['is_genuine'] and not r['kind'].startswith('waves_')]
        if name=='all_genuine':return [r for r in rr if r['is_genuine']]
        if name=='unseen_edits':return [r for r in rr if r['is_unseen']]
        if name in ['contrast','rotation','resizedcrop','brightness','noise']:return [r for r in rr if r.get('family')==name]
        return [r for r in rr if r['kind']==name]
    categories=['known_genuine','all_genuine','unseen_edits','contrast','rotation','resizedcrop','brightness','noise',*sorted({r['kind'] for r in rr})]
    fixed=[];matched=[]
    for category in categories:
        selected=subset(category)
        if not selected:continue
        measures={'order':sum(r['returned'] for r in selected),'order_rounded':sum(r['rounded_key'] for r in selected),
                  'forward_same_alpha_keys':sum(r['forward_key'] for r in selected),
                  'legacy_keys':sum(old[r['case_id']]['key_recovered'] for r in selected),'legacy_final':sum(old[r['case_id']]['returned'] for r in selected)}
        measures.update({name:sum(r[name] is not None and cal[name] is not None and r[name]<=cal[name] for r in selected) for name in controls})
        fixed.append(dict(category=category,total=len(selected),sources=len({r['id'] for r in selected}),primary=primary,control_calibration_target=.95,**measures))
        if category in ['known_genuine','all_genuine','unseen_edits','contrast','rotation','resizedcrop','brightness','noise']:
            count=measures['order']
            for name in controls:
                finite=sorted(r[name] for r in selected if r[name] is not None)
                threshold=finite[count-1] if count and len(finite)>=count else (-1. if count==0 else None)
                attained=sum(r[name] is not None and r[name]<=threshold for r in selected) if threshold is not None else None
                for kind in ['same','aux','gs']:
                    negative=subset(kind)
                    matched.append(dict(genuine_category=category,control=name,negative_kind=kind,genuine_total=len(selected),candidate_genuine=count,available_control_genuine=len(finite),control_genuine=attained,exact_match=count==attained,threshold=threshold,negative_total=len(negative),candidate_negative=sum(r['returned'] for r in negative),control_negative=sum(r[name] is not None and r[name]<=threshold for r in negative) if threshold is not None else None))
    write(run/'comparison95.json',fixed);write(run/'family_matched.json',matched)
    audit=dict(status='complete',source_sha256=sha(Path(__file__)),analysis_plan_sha256=sha(run/'analysis_plan.json'),
               main_complete_sha256=sha(run/'complete.json'),legacy_complete_sha256=sha(run/'legacy_complete.json'),
               same_pixel_joins=len(rows),distinct_pairs=len(old),methods=len(plan['methods']),
               legacy_original_matches=legacy['original_same_pixel_matches'],
               optimizer_status=dict(Counter(str(r['detail'].get('optimizer_status','not_run')) for r in rows)),
               primary_optimizer_failures=sum(not r['detail'].get('optimizer_success',False) for r in rr),
               primary_successful_keys_with_optimizer_failure=sum(r['returned'] and not r['detail'].get('optimizer_success',False) for r in rr),
               list_rescues=sum(r['returned'] and not r['rounded_key'] for r in rr),
               artifacts={n:sha(run/n) for n in ['comparison95.json','family_matched.json']})
    write(run/'comparison_audit.json',audit);print(json.dumps(audit))
    print(json.dumps([r for r in fixed if r['category'] in ['known_genuine','unseen_edits','contrast','rotation','resizedcrop','gaussian_std','brightness_factor','same']],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run)
