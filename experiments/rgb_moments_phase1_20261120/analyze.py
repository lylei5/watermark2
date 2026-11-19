"""Audit measurements, estimate paired uncertainty, and export compact reproducibility data."""
import csv
import gzip
import json
import platform
import sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from prepare import ROOT,RUN,read,put,moments,hashfile


def main():
    samples=read(RUN/'test_sources.json')
    q=read(RUN/'test_quality.json')
    rows=q['rows']
    arms=['baseline','mean_gn8','joint_gn8','joint_adam20']
    assert len(rows)==len({(r['id'],r['arm']) for r in rows})==64*4
    assert len({s['seed'] for s in samples})==64
    pilot=read(RUN/'pilot_sources.json')
    assert not ({s['prompt'] for s in samples}&{s['prompt'] for s in pilot})
    assert not ({s['seed'] for s in samples}&{s['seed'] for s in pilot})
    traces=[]
    for stage in ('pilot','refine','test'):
        src=read(RUN/f'{stage}_sources.json')
        for s in src:
            assert hashfile(Path(s['donor']))==s['donor_sha256']
            with Image.open(s['donor']) as im:
                assert np.allclose(moments(im),s['target'],atol=1e-10)
            d=RUN/stage/f"{s['id']:03d}"
            rec=read(d/'record.json')
            assert rec['seed']==s['seed'] and rec['gpu'] in (4,5,7)
            assert rec['config_sha256']==hashfile(RUN/f'{stage}_config.json')
            for k,r in rec['arms'].items():
                assert hashfile(d/(k+'.png'))==r['sha256']
                if k!='baseline':
                    assert r['curve'][r['best']]['loss']==min(x['loss'] for x in r['curve'])
            traces.append(dict(stage=stage,**rec))
    with gzip.open(RUN/'records.json.gz','wt') as f:
        json.dump(traces,f,allow_nan=False)
    config=read(RUN/'test_config.json')
    for r in rows:
        item=traces[-64+r['id']]['arms'][r['arm']]
        e=np.abs(np.array(item['moments'])-samples[r['id']]['target'])
        assert r['joint_hit']==(max(e[:3])<=.02 and max(e[3:])<=.01)
        assert np.isclose(e[:3].mean(),r['mu_mae']) and np.isclose(e[3:].mean(),r['var_mae'])
    with gzip.open(RUN/'test_edits.csv.gz','rt') as f:
        edits=list(csv.DictReader(f))
    assert len(edits)==len({(r['id'],r['arm'],r['edit']) for r in edits})==64*4*6
    for r in edits:
        i=int(r['id'])
        a=np.array([float(r[f'moment{j}']) for j in range(6)])
        for label,j in [('joint_hit',i),('wrong_joint',i^1)]:
            e=np.abs(a-samples[j]['target'])
            assert (r[label]=='True')==(max(e[:3])<=.02 and max(e[3:])<=.01)
    rng=np.random.default_rng(20261120)
    ix=np.c_[rng.integers(0,16,(5000,16)),rng.integers(16,32,(5000,16))]
    arrays={}
    for a in arms:
        rr=sorted((r for r in rows if r['arm']==a),key=lambda r:r['id'])
        arrays[a]={k:np.array([r[k] for r in rr]).reshape(32,2).mean(1) for k in
            ['mu_mae','var_mae','mean_hit','joint_hit','seconds','lpips','clip_text','clip_image']}
        arrays[a]['seed_lpips']=np.array([r['seed_lpips'] for r in rr if 'seed_lpips' in r])
    def interval(x):
        return np.quantile(x[ix].mean(1),[.025,.975]).tolist()
    ci={a:{k:interval(x) for k,x in v.items()} for a,v in arrays.items()}
    paired={a+'-'+b:{k:interval(x-arrays[b][k]) for k,x in arrays[a].items()}
            for a,b in [('mean_gn8','baseline'),('joint_gn8','baseline'),('joint_gn8','mean_gn8'),('joint_gn8','joint_adam20')]}
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,arm,title in zip(axes,arms[1:],['Mean · Gauss–Newton','Mean + variance · Gauss–Newton','Mean + variance · Adam']):
        for rec in traces[-64:]:
            curve=rec['arms'][arm]['curve']
            y=[x['loss'] for x in curve]
            ax.plot(range(len(y)),y,color='#176677',alpha=.25,lw=.75)
        ax.set(title=title,xlabel='Latent update',yscale='log')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Weighted RGB loss (log scale)')
    fig.savefig(RUN/'curves.png',dpi=180)
    plt.close(fig)
    rr=sorted((r for r in rows if r['arm']=='joint_gn8' and samples[r['id']]['dataset']=='COCO'),key=lambda r:r['lpips'])
    picks=[rr[0],rr[len(rr)//2],rr[-1]]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    size=194
    sheet=Image.new('RGB',(size*4,34+3*(size+24)),'white')
    pen=ImageDraw.Draw(sheet)
    for j,title in enumerate(['Baseline','RGB mean','RGB mean + variance','Target donor']):
        pen.text((j*size+4,6),title,font=font,fill='black')
    for j,r in enumerate(picks):
        idx=r['id']
        paths=[RUN/'test'/f'{idx:03d}'/(k+'.png') for k in arms[:3]]+[Path(samples[idx]['donor'])]
        y=34+j*(size+24)
        for k,path in enumerate(paths):
            with Image.open(path) as im:
                sheet.paste(im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS),(k*size,y))
        pen.text((4,y+size),f"ID {idx:02d}   Joint LPIPS {r['lpips']:.3f}",fill='black',font=font)
    sheet.save(RUN/'paired_examples.jpg',quality=90)
    # Independently compute the cross-covariance trace via eigenvalues in sample space.
    real=np.load(RUN/'assets/real_pool3.npy').astype(float)
    for arm in arms:
        a=np.load(RUN/'assets'/f'{arm}_pool3.npy').astype(float)
        mx,my=a.mean(0),real.mean(0)
        x,y=(a-mx)/np.sqrt(len(a)-1),(real-my)/np.sqrt(len(real)-1)
        c=x@y.T
        val=((mx-my)**2).sum()+(x*x).sum()+(y*y).sum()-2*np.sqrt(np.maximum(0,np.linalg.eigvalsh(c@c.T))).sum()
        assert np.isclose(val,read(RUN/'fid.json')[arm]['vs_real_mixed'],atol=.0001)
    put(RUN/'uncertainty.json',dict(ci95=ci,paired_delta_ci95=paired))
    keep=['pilot_config.json','refine_config.json','test_config.json','pilot_sources.json','test_sources.json',
          'pilot_quality.json','refine_quality.json','test_quality.json','fid.json','test_stability.json',
          'records.json.gz','test_edits.csv.gz','real_reference.json','curves.png','paired_examples.jpg','uncertainty.json']
    dep=read(ROOT/'runs/unified_seal_clean_20261116/model_manifest.json')['diffusion_model']
    put(RUN/'audit.json',dict(pilot_prompts=6,test_prompts=32,test_seeds=64,pngs=sum(len(r['arms']) for r in traces),
        written_records=len(traces),edit_rows=len(edits),example_ids=[r['id'] for r in picks],fid_real_n=len(real),
        gpu_indices=sorted({r['gpu'] for r in traces}),diffusion_manifest=dep,torch=torch.__version__,
        python=platform.python_version(),artifacts={f:hashfile(RUN/f) for f in keep},
        code={p.name:hashfile(p) for p in RUN.glob('*.py')},protocol=hashfile(RUN/'protocol.json'),
        fid_weights=hashfile(RUN/'assets/fid.pth'),geometry_sha256=hashfile(RUN/'assets/stable_signature_utils_img.py')))


if __name__=='__main__':
    main()
