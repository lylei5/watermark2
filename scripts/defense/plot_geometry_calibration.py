"""Render observed calibration counts; no independent-trial uncertainty claim."""
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
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.summary.read_text());args.output.mkdir(parents=True,exist_ok=True)
    order=['clean','jpeg50','jpeg75','blur1_resize80','crop90','noise0.01']
    names=['Clean','JPEG 50','JPEG 75','Blur + resize','Crop 90%','Noise 0.01']
    rows={r['group']:r for r in data['genuine_by_condition']}
    for r in rows.values():assert r['all_fit_accepted']==r['split_fit_accepted']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'figure.dpi':160}):
        fig,axes=plt.subplots(1,2,figsize=(10.6,4.3),layout='constrained')
        x=np.arange(6);ax=axes[0]
        ax.bar(x-.18,[rows[k]['default_accepted'] for k in order],width=.36,color='#0072B2',label='Default BA')
        ax.bar(x+.18,[rows[k]['all_fit_accepted'] for k in order],width=.36,color='#D55E00',hatch='//',label='Both geometry rules')
        ax.set(xticks=x,xticklabels=names,ylim=(0,36),yticks=[0,8,16,24,32],ylabel='Accepted genuine queries (out of 32)',title='A  New-source calibration')
        ax.tick_params(axis='x',rotation=25);ax.legend(loc='lower left',fontsize=8)
        ax.annotate('0 → 32',xy=(4,32),xytext=(4,34),ha='center',fontsize=9)
        ax=axes[1]
        points=data['supplementary_ba_controls']
        for i,r in enumerate(points):
            xx=100*r['calibration_tpr'];yy=100*r['wrong_identity_accepted']/5952
            ax.scatter(xx,yy,marker='o',s=45,color='#0072B2')
            ax.annotate(f"BA target {r['target_absolute_tpr']:.0%}",(xx,yy),
                xytext=[(4,10),(4,8),(4,8),(-80,-4)][i],textcoords='offset points',fontsize=8)
        default=ax.scatter(100*160/192,0,marker='x',s=60,color='#222222',label='Default BA: 160/192')
        ax.legend(handles=[default],loc='upper left',frameon=False,fontsize=8)
        ax.scatter(100,0,marker='D',s=60,color='#D55E00',label='Both geometry rules')
        ax.annotate('Geometry\n192/192; 0/5952',(100,0),xytext=(99,55),textcoords='data',ha='right',fontsize=8,
                    arrowprops={'arrowstyle':'-','color':'#D55E00'})
        ax.set(xlim=(76,103),ylim=(-8,108),xticks=[80,85,90,95,100],yticks=[0,25,50,75,100],
               xlabel='Observed genuine acceptance (%)',ylabel='Observed wrong-identity acceptance (%)',
               title='B  Calibration-frozen BA controls')
        ax.grid(alpha=.15)
        fig.suptitle('Query-only geometry restores crop detection at the frozen working point',fontsize=12)
        fig.get_layout_engine().set(rect=(0,.1,1,.88))
        fig.text(.5,.025,'32 sources × 6 views; 5,952 correlated wrong-identity pairs. Zero observed errors is not an ultra-low FPR guarantee.',
                 ha='center',fontsize=8)
        for ext in ['png','pdf']:fig.savefig(args.output/f'calibration.{ext}',dpi=300)
        plt.close(fig)
    with (args.output/'condition_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))));w.writeheader();w.writerows(rows[k] for k in order)
    provenance=dict(source_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        sources=32,views_per_source=6,wrong_identity_pairs=5952,error_bars=None,
        uncertainty_note='Repeated views and shared identities are correlated; descriptive counts only',
        general_research_figure=True,publisher_specific_format_verified=False,
        size_inches=[10.6,4.3],png_dpi=300,matplotlib_version=matplotlib.__version__)
    (args.output/'figure_provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
