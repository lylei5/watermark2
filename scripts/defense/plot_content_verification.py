"""Evidence figure; raw image-level observations and exact binomial intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    genuine=json.loads((args.report/'genuine_full_domain.json').read_text())
    attack=json.loads((args.report/'reprompt_full_domain.json').read_text())
    sets=[('Benign, no crop',[r for r in genuine['rows'] if r['method']=='sembind_scattered' and r['label']!='crop90'],'#0072B2','o'),
          ('Center crop',[r for r in genuine['rows'] if r['method']=='sembind_scattered' and r['label']=='crop90'],'#B14A00','x'),
          ('Published Reprompt',[r for r in attack['rows'] if r['method']=='sembind_scattered'],'#006D52','^')]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(12.8,4.3),layout='constrained')
        for label,rows,color,marker in sets:
            axes[0].scatter([1-r['diagnostic_actual_content_agreement'] for r in rows],
                            [1-r['profiled_content_agreement'] for r in rows],s=24,color=color,marker=marker,label=label,alpha=.8)
            axes[1].scatter([r['diagnostic_actual_channel_error'] for r in rows],
                            [r['estimated_channel_error'] for r in rows],s=24,color=color,marker=marker,alpha=.8)
        axes[0].plot([0,1],[0,1],ls='--',color='#777777',lw=1)
        axes[0].set(xlim=(-.025,1.025),ylim=(-.025,1.025),xlabel='Actual content mismatch',ylabel='Estimated content mismatch',title='a  Content evidence')
        axes[0].legend(loc='upper left',fontsize=8,frameon=False)
        axes[1].plot([0,.52],[0,.52],ls='--',color='#777777',lw=1)
        axes[1].set(xlim=(-.015,.52),ylim=(-.015,.52),xlabel='Actual inversion error',ylabel='Estimated inversion error',title='b  Channel evidence')
        labels=['clean','jpeg50','jpeg75','blur1_resize80','noise0.01','crop90']
        display=['Clean','JPEG 50','JPEG 75','Blur + resize','Noise 0.01','Center crop']
        for method,color,marker,offset,name in [('sembind_scattered','#0072B2','o',-.10,'SemBind-DDIM'),('coverlock_g_message','#B14A00','s',.10,'CoverLock-GS adapter')]:
            for i,label in enumerate(labels):
                row=next(r for r in genuine['summary'] if r['method']==method and r['label']==label)
                k,n=row['accepted'],row['n'];point=k/n
                lo=beta.ppf(.025,k,n-k+1) if k else 0
                hi=beta.ppf(.975,k+1,n-k) if k<n else 1
                axes[2].errorbar(point,i+offset,xerr=[[point-lo],[hi-point]],fmt=marker,color=color,capsize=3,label=name if i==0 else None)
        axes[2].set(yticks=range(6),yticklabels=display,xlim=(-.035,1.035),xlabel='Genuine acceptance rate',title='c  Genuine utility, n = 16 sources')
        axes[2].invert_yaxis();axes[2].legend(loc='lower right',fontsize=8,frameon=False)
        fig.suptitle('Content verification: real SD2.1/DDIM development data',fontsize=13)
        fig.savefig(args.report/'evidence.png',dpi=300,facecolor='white')
        fig.savefig(args.report/'evidence.svg',facecolor='white')
        svg=args.report/'evidence.svg'
        svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode())
        plt.close(fig)
    provenance=dict(audience='research review; target venue not specified',
                    sources={name:hashlib.sha256((args.report/name).read_bytes()).hexdigest() for name in ['genuine_full_domain.json','reprompt_full_domain.json']},
                    raw_observations='80 noncrop +16 crop genuine queries from16 sources,16 Reprompt candidates',
                    transformations='content mismatch=1-agreement; no filtering beyond labeled groups',
                    uncertainty='panel c: per-condition Clopper-Pearson two-sided95% intervals,16 sources; conditions are repeated measurements',
                    image_dimensions_px=[3840,1290],dpi=300,matplotlib=matplotlib.__version__,
                    limitation='development evidence; no superiority or journal-compliance claim')
    (args.report/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
