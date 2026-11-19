"""Assign prompts, seeds and achievable RGB moment targets before any optimization."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/rgb_moments_phase1_20261120'


def read(p):
    return json.loads(p.read_bytes())


def put(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(p)


def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def moments(im):
    x=np.asarray(im.convert('RGB'),dtype=np.float64).reshape(-1,3)/255.
    return np.r_[x.mean(0),x.var(0)].tolist()


def main():
    seen=set()
    for stage,archive,n in [('pilot','color_budget_20261119',3),('test','color_veto_20261119',16)]:
        parent=ROOT/'runs'/archive
        records=read(parent/'sources.json')
        chosen=[]
        for dataset in ('COCO','SDP'):
            pool=sorted((s for s in records if s['dataset']==dataset),
                        key=lambda s:hashlib.sha256(('rgb-moment-v1/'+s['prompt']).encode()).digest())
            for s in pool[:n]:
                assert s['prompt'] not in seen
                seen.add(s['prompt'])
                folder=parent/'images'/f"{s['task']:03d}"
                receipt=read(folder/'receipt.json')
                for j in (0,1):
                    p=folder/f'{j}.png'
                    assert hashfile(p)==receipt['sha256'][j]
                    seed=int(hashlib.sha256(f"moment-20261120/{s['prompt']}/{j}".encode()).hexdigest()[:15],16)
                    assert seed not in receipt['seeds']
                    with Image.open(p) as im:
                        target=moments(im)
                    chosen.append(dict(id=len(chosen),pair=len(chosen)//2,idx=j,dataset=dataset,
                        prompt=s['prompt'],seed=seed,donor_seed=receipt['seeds'][j],donor=str(p),
                        donor_sha256=receipt['sha256'][j],target=target))
        path=RUN/f'{stage}_sources.json'
        assert not path.exists()
        put(path,chosen)


if __name__=='__main__':
    main()
