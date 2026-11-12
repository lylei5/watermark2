"""Render measured trajectory distributions and paired effects; no smoothing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    source=run/'trajectory_summary.json';data=json.loads(source.read_text())
    assert data['status']=='complete' and len(data['rows'])==100
    output=run/'figures';output.mkdir(exist_ok=False)
    styles={'mean':('#0072B2','-'),'p5':('#D55E00','--'),'minimum':('#666666',':')}
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'}):
        fig,axes=plt.subplots(1,3,figsize=(11.7,3.45),layout='constrained')
        for ax,group in zip(axes[:2],['COCO','SDP']):
            rows=[r for r in data['curves'] if r['dataset']==group];steps=[r['step'] for r in rows]
            for metric,(color,style) in styles.items():
                ax.plot(steps,[r[metric] for r in rows],color=color,ls=style,lw=1.6,label=metric.capitalize())
            ax.fill_between(steps,[r['mean_ci'][0] for r in rows],[r['mean_ci'][1] for r in rows],color=styles['mean'][0],alpha=.15,label='Mean 95% CI')
            ax.axhline(.95,color='black',ls='-.',lw=.8,label='P5 criterion')
            ax.axvline(35,color='black',ls=':',lw=.8)
            ax.set(xlim=(5,50),ylim=(0,1.01),xlabel='UNet evaluation (of 50)',ylabel='Code agreement with final image',title=f'{group}: 50 prompts')
            ax.set_xticks([5,15,25,35,45,50]);ax.grid(axis='y',alpha=.15)
        axes[0].legend(loc='lower right',frameon=False,fontsize=8)
        positions=[];labels=[]
        for j,group in enumerate(['COCO','SDP']):
            subset=[r for r in data['rows'] if r['dataset']==group]
            primary=next(r for r in data['primary'] if r['dataset']==group)
            for k,name in enumerate(['aux_final','same_prompt']):
                y=j*3+k;delta=np.array([r['agreement'][30]-r['controls'][name] for r in subset])
                offset=np.linspace(-.16,.16,len(delta))
                axes[2].scatter(delta,y+offset,s=9,c=styles['mean'][0] if j==0 else styles['p5'][0],alpha=.45)
                item=primary['paired'][name];lo,hi=item['ci95'];mu=item['mean_advantage']
                axes[2].errorbar(mu,y,xerr=[[mu-lo],[hi-mu]],fmt='D',color='black',ms=4,capsize=3)
                positions.append(y);labels.append(group+' / '+('aux-final' if k==0 else 'same prompt'))
        axes[2].axvline(0,color='black',ls=':',lw=.9)
        axes[2].set(yticks=positions,yticklabels=labels,xlabel='Step35 agreement minus control',title='Paired difference by prompt')
        axes[2].invert_yaxis();axes[2].grid(axis='x',alpha=.15)
        for ax,label in zip(axes,['a','b','c']): ax.text(-.12,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
        fig.savefig(output/'trajectory.png',dpi=300,facecolor='white')
        fig.savefig(output/'trajectory.pdf',facecolor='white')
        plt.close(fig)
    manifest=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dimensions_inches=[11.7,3.45],png_dpi=300,matplotlib=matplotlib.__version__,
        uncertainty='Pointwise95% percentile bootstrap intervals;10000 prompt resamples; not simultaneous bands',
        transformation='No smoothing; arithmetic mean, empirical5th percentile and minimum of measured1024-bit agreements; fixed step35 paired contrasts',
        points='Each point is one prompt; deterministic vertical spacing only',
        scope='Provisional research figure; no publisher-specific compliance claim',
        alt_text='COCO and SDP trajectory code agreement from evaluations5 to50 with mean,5th percentile,minimum and mean confidence band. Third panel shows per-prompt step35 advantage over auxiliary-final and same-prompt independent-seed controls.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
