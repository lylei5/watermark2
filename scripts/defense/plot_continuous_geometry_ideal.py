"""Export analytic channel response with Monte Carlo checks on server249."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--checks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.checks.read_text());assert data['passed']
    a.output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#333333','#0072B2','#D55E00'];styles=['-', '--', '-.']
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.7))
    records=[]
    grid=np.linspace(0,1,201)
    local=np.geomspace(.001,.5,201)
    for sigma,color,style in zip([0.,.4,1.],colors,styles):
        rho=1/np.sqrt(1+sigma**2)
        exact=.5+np.arcsin(rho*np.cos(grid))/np.pi
        penalty=(np.arcsin(rho)-np.arcsin(rho*np.cos(local)))/np.pi
        axes[0].plot(grid,exact,color=color,linestyle=style,label=fr'$\sigma={sigma:g}$')
        axes[1].loglog(local,penalty,color=color,linestyle=style)
        checked=[r for r in data['channel'] if r['sigma']==sigma]
        axes[0].errorbar([r['angle'] for r in checked],[r['measured'] for r in checked],
            yerr=[1.96*r['mc_standard_error'] for r in checked],fmt='o',ms=3,color=color,elinewidth=.7,capsize=2)
        for delta,value in zip(grid,exact):records.append(dict(panel='sign_correctness',sigma=sigma,angle=delta,value=value))
        for delta,value in zip(local,penalty):records.append(dict(panel='additional_error',sigma=sigma,angle=delta,value=value))
    axes[0].set(xlabel='Fixed decoding angle mismatch (radians)',ylabel='Single-coordinate sign correctness',ylim=(.57,1.02),xlim=(-.02,1.02),title='A  Ideal independent Gaussian channel')
    axes[1].set(xlabel='Fixed decoding angle mismatch (radians)',ylabel='Additional sign error over zero mismatch',title='B  Local response to angle mismatch')
    axes[1].text(.04,.91,'Noise-free: linear',color=colors[0],transform=axes[1].transAxes)
    axes[1].text(.04,.81,'Positive noise: quadratic',color=colors[1],transform=axes[1].transAxes)
    for ax in axes:ax.grid(alpha=.18,linewidth=.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False)
    fig.text(.5,.012,'Curves: analytic probabilities. Points: 300,000 draws; bars: pointwise ±1.96 MC SE. Not image-defense results.',ha='center',fontsize=7.2)
    fig.tight_layout(rect=[0,.12,1,1],w_pad=2.5)
    for suffix in ['png','pdf']:fig.savefig(a.output/f'ideal_response.{suffix}',dpi=300)
    plt.close(fig)
    path=a.output/'ideal_response_data.csv'
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=['panel','sigma','angle','value']);writer.writeheader();writer.writerows(records)
    manifest=dict(input=str(a.checks),input_sha256=hashlib.sha256(a.checks.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        size_inches=[8.2,3.7],dpi=300,scope='Analytic ideal channel; paired Monte Carlo samples across angles, pointwise intervals only; no image or forgery observations.',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file() and p.name!='ideal_figure_manifest.json'})
    (a.output/'ideal_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
