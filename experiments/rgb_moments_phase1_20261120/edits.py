"""Evaluate clean and published benign edits against assigned and sibling targets."""
import ast
import csv
import gzip
import hashlib
import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional
from prepare import ROOT,RUN,read,put,moments,hashfile


def official_geometry():
    path=RUN/'assets/stable_signature_utils_img.py'
    raw=path.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='a412156bd8368c3aa8cbeaa6b1fbd3ad2fb7e0a5'
    tree=ast.parse(raw)
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('center_crop','resize')]
    ns=dict(np=np,functional=functional)
    exec(compile(tree,str(path),'exec'),ns)
    return ns


def hit(a,b):
    d=np.abs(a-b)
    return bool(d[:3].max()<=.02),bool(d[:3].max()<=.02 and d[3:].max()<=.01)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'runs/order_capsule_confirmation_20261115'))
    from order_recovery_data import waves_module
    waves=waves_module()
    geom=official_geometry()
    samples=read(RUN/'test_sources.json')
    records=[read(RUN/'test'/f"{s['id']:03d}"/'record.json') for s in samples]
    rows=[]
    for s in samples:
        i=s['id']
        t=np.array(s['target'])
        sib=np.array(samples[i^1]['target'])
        for name,rec in records[i]['arms'].items():
            file=RUN/'test'/f'{i:03d}'/(name+'.png')
            assert hashfile(file)==rec['sha256']
            with Image.open(file) as img:
                im=img.convert('RGB')
            base=np.array(moments(im))
            x=functional.to_tensor(im)
            conditions={'clean':im,'crop50':functional.to_pil_image(geom['center_crop'](x,.5)),
                        'resize70':functional.to_pil_image(geom['resize'](x,.7))}
            for family in ('compression','blurring','noise'):
                conditions[family]=waves.apply_distortion([im],family,strength=.4,distortion_seed=20261120+s['pair'])[0]
            for edit,im2 in conditions.items():
                a=np.array(moments(im2))
                e=np.abs(a-t)
                delta=np.abs(a-base)
                own,own6=hit(a,t)
                wrong,wrong6=hit(a,sib)
                twin,twin6=hit(a,np.array(records[i^1]['arms'][name]['moments']))
                rows.append(dict(id=i,pair=s['pair'],arm=name,edit=edit,mu_mae=e[:3].mean(),var_mae=e[3:].mean(),
                    mu_drift=delta[:3].mean(),var_drift=delta[3:].mean(),mean_hit=own,joint_hit=own6,
                    wrong_mean=wrong,wrong_joint=wrong6,sibling_image_mean=twin,sibling_image_joint=twin6,
                    **{f'moment{j}':val for j,val in enumerate(a)}))
    with gzip.open(RUN/'test_edits.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary={}
    for arm in records[0]['arms']:
        summary[arm]={}
        for edit in ('clean','compression','blurring','noise','crop50','resize70'):
            sub=[r for r in rows if r['arm']==arm and r['edit']==edit]
            summary[arm][edit]={k:float(np.mean([r[k] for r in sub])) for k in
                 ('mu_mae','var_mae','mu_drift','var_drift','mean_hit','joint_hit','wrong_mean','wrong_joint',
                  'sibling_image_mean','sibling_image_joint')}
    pairs=[hit(np.array(samples[i]['target']),np.array(samples[i+1]['target'])) for i in range(0,len(samples),2)]
    put(RUN/'test_stability.json',dict(summary=summary,wrong_target_pairs=np.sum(pairs,axis=0).tolist(),pairs=len(pairs)))


if __name__=='__main__':
    main()
