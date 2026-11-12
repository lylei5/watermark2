"""Paired summary of the completed Gaussian transport experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
    names=['protocol.json','genuine_verified.json','projection_audit.json','channel_diagnostic.json']
    plan,g,projection,d=[json.loads((r/name).read_text()) for name in names]
    assert g['audit_passed'] and projection['passed'] and len(g['rows'])==240
    indexed={(q['task'],q['arm'],q['condition']):q for q in g['rows']}
    pairs={}
    for arm in ['full','saturated','shuffled_transport']:
        items=[]
        for q in g['rows']:
            if q['arm']!='transport':continue
            control=indexed[q['task'],arm,q['condition']]
            items.append(dict(task=q['task'],condition=q['condition'],candidate=q['accepted'],control=control['accepted']))
        pairs[arm]=dict(restored=sum(q['candidate'] and not q['control'] for q in items),
            lost=sum(q['control'] and not q['candidate'] for q in items),rows=items)
    summary={arm:dict(initial_mse=float(np.mean([q['latent_l2'] for q in projection['projection_records'] if q['arm']==arm])),
        clean=sum(q['accepted'] for q in g['rows'] if q['arm']==arm and q['condition']=='clean'),
        wrong=projection['wrong_accepts'][arm],channel=d['summary'][arm]) for arm in plan['arms']}
    result=dict(completed=True,summary=summary,paired=pairs,script_sha256=sha(__file__),
                inputs={name:sha(r/name) for name in names},scope='Six source paired descriptive audit; no new model execution, threshold changes or independent-sample intervals.')
    (r/'paired_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,paired={arm:{k:v for k,v in values.items() if k!='rows'} for arm,values in pairs.items()}),indent=2))


if __name__=='__main__':main()
