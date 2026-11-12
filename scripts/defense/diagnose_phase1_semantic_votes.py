"""Original-GS noise-free decoder response to measured semantic-code differences.

This is an algebraic negative control, not an image attack or a defense change.
No candidate image is constructed and no search or optimization is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    plan=json.loads((run/'protocol.json').read_text())
    extra=json.loads((run/'semantic_votes_protocol.json').read_text())
    assert sha(__file__)==extra['script_sha256'] and extra['parent_protocol_sha256']==sha(run/'protocol.json')
    assert not (run/'semantic_votes.json').exists()
    from codeword_binding import keyed_order
    private=Path(plan['identity_root']);secrets=json.loads((private/'seeds.json').read_text())
    assert sha(private/'seeds.json')==plan['private_seed_sha256']
    perm=keyed_order(bytes.fromhex(secrets['binding_key']),b'sembind/permutation/',16384)
    codes={};identities={}
    for shard in range(3):
        result=json.loads((run/f'generate_{shard}/complete.json').read_text())
        assert result['status']=='complete' and result['protocol_sha256']==sha(run/'protocol.json')
        for row in result['rows']:
            task=row['task'];path=run/row['codes']['path']
            assert sha(path)==row['codes']['sha256']
            with np.load(path) as data: codes[task]={k:data[k] for k in data.files}
            assert sha(private/f'{task:03}.npz')==row['identity_sha256']
            with np.load(private/f'{task:03}.npz') as data: identities[task]={k:data[k] for k in data.files}
    rows=[]
    for item in plan['sources']:
        task=item['task'];c=codes[task];identity=identities[task]
        enroll=c['aux_code'];next_task=(task//50)*50+(task+1)%50
        controls={'same_prompt':c['same_code'],'aux_final':c['bound_code'],
                  'different_prompt':codes[next_task]['aux_code'],'trajectory35':c['trajectory_codes'][30]}
        for name,q in controls.items():
            # Compare the two original semantic masks at a noise-free carrier.
            mismatch=np.tile(enroll^q,16)[perm].reshape(1,4,64,64)
            observations=(identity['initial']>0).astype(np.uint8)^identity['key']^mismatch
            decoded=observations.reshape(1,4,8,8,8,8).sum(axis=(2,4))>32
            ba=float(np.mean(decoded.reshape(-1)==identity['message'].reshape(-1)))
            rows.append(dict(task=task,dataset=item['dataset'],control=name,code_agreement=float(np.mean(enroll==q)),ba=ba,
                accepts={str(k):ba>=k/256 for k in [166,167,182]}))
    summary=[]
    for dataset in ['COCO','SDP']:
        for name in controls:
            subset=[r for r in rows if r['dataset']==dataset and r['control']==name]
            summary.append(dict(dataset=dataset,control=name,n=len(subset),mean_code_agreement=float(np.mean([r['code_agreement'] for r in subset])),
                mean_ba=float(np.mean([r['ba'] for r in subset])),accepted_at167=sum(r['accepts']['167'] for r in subset)))
    result=dict(scope=extra['scope'],protocol_sha256=sha(run/'semantic_votes_protocol.json'),rows=rows,summary=summary,attack_jobs=0)
    (run/'semantic_votes.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
