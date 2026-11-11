"""Describe all completed published checkpoints; no experiment generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    original=json.loads(args.baseline.read_text());scored=json.loads(args.scores.read_text())
    assert original['completed'] and scored['completed']
    rows=original['rows'];assert len(rows)==144
    keys=['default_accepted','identity_accepted','identity_penalty_accepted','all_fit_accepted','split_fit_accepted']
    assert all(len({r[k] for k in keys})==1 for r in scored['rows'])
    table=[]
    for method in ['sembind','gs']:
        for update in range(1,152,10):
            subset=[r for r in rows if r['method']==method and r['kind']=='candidate' and r['update']==update]
            assert len(subset)==4
            table.append(dict(method=method,update=update,n=4,accepted=sum(r['accepted'] for r in subset),
                **{f'mean_{k}':float(np.mean([r[k] for r in subset])) for k in ['ba','psnr','ssim','ms_ssim','lpips']}))
    with (out/'trajectory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result=dict(completed=True,source_count_per_method=4,paired_sources=True,checkpoints_per_source=16,
        all_five_rules_identical_on_all_queries=True,full_budget=151,
        final_summary=original['summary'],trajectory=table,
        checkpoint_independence=False,independent_confirmation=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.baseline,args.scores]},
        figure=dict(destination='Research review; journal specifications not selected',unit='Four paired sources, 16 repeated checkpoints each',
            thin_lines='Individual sources',thick_lines='Arithmetic mean over four sources',
            smoothing=False,exclusions=0,confidence_interval=None))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(9.4,3.5),layout='constrained')
        for method,color,style,marker in [('sembind','#0072B2','-','o'),('gs','#D55E00','--','s')]:
            for task in range(4):
                group=sorted([r for r in rows if r['method']==method and r['task']==task and r['kind']=='candidate'],key=lambda r:r['update'])
                assert len(group)==16
                for ax,metric in zip(axes,['ba','psnr']):
                    ax.plot([r['update'] for r in group],[r[metric] for r in group],color=color,ls=style,alpha=.25,lw=.8)
            means=[r for r in table if r['method']==method]
            for ax,metric in zip(axes,['ba','psnr']):
                ax.plot([r['update'] for r in means],[r['mean_'+metric] for r in means],color=color,ls=style,
                    marker=marker,ms=3,lw=1.7,label='SemBind adapter' if method=='sembind' else 'GS')
        axes[0].axhline(181/256,color='#555555',ls=':',lw=1,label='Default BA threshold')
        axes[0].set(ylabel='Bit accuracy',ylim=(0,1.03),title='a  Detection evidence')
        axes[1].set(ylabel='PSNR to original cover (dB)',title='b  Candidate distortion')
        for ax in axes:
            ax.set(xlabel='Published optimization update',xlim=(1,151),xticks=[1,31,61,91,121,151]);ax.grid(axis='y',alpha=.15)
        axes[0].legend(fontsize=8,loc='lower right')
        fig.savefig(out/'trajectory.png',dpi=180)
        fig.savefig(out/'trajectory.pdf')
    print(json.dumps(dict(all_rules_identical=True,trajectory_rows=len(table),sources_per_method=4),indent=2))


if __name__=='__main__':main()
