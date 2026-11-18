"""Audit paired color-feature scores and render the per-edit comparison."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def main(run):
    summary = json.loads((run/'summary.json').read_bytes())
    cut = json.loads((run/'thresholds.json').read_bytes())
    plan = json.loads((run/'color_identity_20261119.json').read_bytes())
    receipt = json.loads((run/'complete.json').read_bytes())
    for filename, digest in receipt['sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    scores = (run/'scores.csv').read_bytes()
    rows = list(csv.DictReader(scores.decode().splitlines()))
    key = ['split','task','seed_image','family','strength','method']
    assert len(rows)==len({tuple(r[k] for k in key) for r in rows})==17000
    for split,n in [('test',60),('calibration',40)]:
        tasks={r['task'] for r in rows if r['split']==split}
        assert len(tasks)==n
        assert not tasks & {r['task'] for r in rows if r['split']!=split}
    for m,(a,b) in cut.items():
        cal = [r for r in rows if r['method']==m and r['split']=='calibration' and r['family']!='clean']
        assert len(cal)==1280 and sum(float(r['sibling'])<=a for r in cal)<=64
        assert sum(float(r['own'])<=b for r in cal)>=1216
        part = [r for r in rows if r['method']==m and r['family']!='clean' and r['split']=='test']
        own = np.array([float(r['own']) for r in part])
        sibling = np.array([float(r['sibling']) for r in part])
        assert len(own)==1920
        for name,value in [('true_accept',(own<=a).mean()),('sibling_accept',(sibling<=a).mean()),
                           ('pair_top1',((own<sibling)+0.5*(own==sibling)).mean()),
                           ('gallery_top1',np.mean([float(r['gallery']) for r in part])),
                           ('true_at95',(own<=b).mean()),('sibling_at95',(sibling<=b).mean())]:
            assert abs(value-summary['aggregate'][m][name])<1e-10
    (run/'scores.csv.gz').write_bytes(gzip.compress(scores,mtime=0))
    methods = list(cut)
    families = plan['edit_families']
    fig, axes = plt.subplots(1,2,figsize=(10.6,5.7))
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.79,wspace=.2)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    cn = FontProperties(fname=font)
    labels = ['RGB 占比','RGB 直方图','全局色度','固定亮度区','相对亮度区']
    names = ['JPEG','裁剪缩放','旋转','亮度','对比度','模糊','噪声','局部擦除']
    for ax,strength in zip(axes,plan['strengths']):
        matrix = np.array([[summary['by_condition'][m][f'{f}/{strength}']['true_accept']
                            for m in methods] for f in families])*100
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto',interpolation='nearest')
        ax.set_xticks(range(5),labels,fontproperties=cn,fontsize=10,rotation=22,ha='right')
        ax.set_yticks(range(8),names,fontproperties=cn,fontsize=10)
        ax.tick_params(length=0)
        ax.set_title(f'WAVES 相对强度 {strength}',fontproperties=cn,fontsize=13)
        for y,x in np.ndindex(matrix.shape):
            ax.text(x,y,f'{matrix[y,x]:.0f}',ha='center',va='center',
                    color='white' if matrix[y,x]>=62 else '#16283c',fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.035,.94,'合法编辑后，原图仍被接受的比例（%）',fontproperties=cn,fontsize=19)
    fig.text(.035,.875,'按特征分别校准同提示词错认率，校准上限 5%。每格为留出集 60 个提示词、120 张原图。',fontproperties=cn,fontsize=11)
    fig.savefig(run/'edit_acceptance.png',dpi=150)
    plt.close(fig)
    print('Verified 17,000 rows and all 30 aggregate rates.')
    for m in methods:
        print(m,{k:round(100*v,2) for k,v in summary['aggregate'][m].items()})


if __name__=='__main__':
    main(Path(sys.argv[1]))
