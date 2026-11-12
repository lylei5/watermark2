"""Descriptive paired summaries of completed development experiments on server249.

This reuses audited artifacts; it is not fresh model execution or new validation.
"""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    names=['conditional_anchor_pilot_20261112','conditional_quantile_pilot_20261112_v2','semantic_feedback_pilot_20261112']
    outputs=[]
    for name in names:
        path=a.root/name/'genuine_verified.json';data=json.loads(path.read_text());assert data['audit_passed']
        rows=data['rows'];arms=[s['arm'] for s in data['summary']]
        base={(r['task'],r['condition']):r for r in rows if r['arm']=='full'}
        pairs=[]
        for arm in arms:
            if arm=='full':continue
            queries=[r for r in rows if r['arm']==arm]
            assert len(queries)==len(base) and {(r['task'],r['condition']) for r in queries}==set(base)
            counts={}
            for condition in sorted({r['condition'] for r in queries}):
                rr=[r for r in queries if r['condition']==condition]
                counts[condition]=dict(n=len(rr),full=sum(base[r['task'],condition]['accepted'] for r in rr),
                    candidate=sum(r['accepted'] for r in rr),
                    recovered=sum(r['accepted'] and not base[r['task'],condition]['accepted'] for r in rr),
                    lost=sum(not r['accepted'] and base[r['task'],condition]['accepted'] for r in rr))
            per_source=[]
            for task in sorted({r['task'] for r in queries}):
                rr=[r for r in queries if r['task']==task]
                per_source.append(dict(task=task,full=sum(base[task,r['condition']]['accepted'] for r in rr),
                    candidate=sum(r['accepted'] for r in rr),n=len(rr)))
            pairs.append(dict(arm=arm,conditions=counts,sources=per_source,
                recovered=sum(v['recovered'] for v in counts.values()),lost=sum(v['lost'] for v in counts.values())))
        outputs.append(dict(run=name,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            protocol_sha256=data['protocol_sha256'],n_records=len(rows),summary=data['summary'],paired=pairs))
    result=dict(completed=True,runs=outputs,unique_model_records=sum(x['n_records'] for x in outputs),
        inference='Within-run paired descriptive comparison only. Different source cohorts are not pooled or ranked across runs. No new model execution; consumed development data.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['run','n_records','paired']} for r in outputs],indent=2))


if __name__=='__main__':main()
