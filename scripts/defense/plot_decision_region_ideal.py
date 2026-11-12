"""Illustrative Gaussian measurement-channel simulation, not a model evaluation.

Run on the authorized research server. All samples and plot transformations saved.
"""
import argparse, csv, hashlib, json, platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def project(z, allowed, ranks=None):
    wrong = z < 0
    costs = np.abs(z) if ranks is None else ranks
    order = np.argsort(np.where(wrong, costs, np.inf), axis=1, kind='stable')
    number = np.maximum(wrong.sum(1) - allowed, 0)
    take = np.arange(z.shape[1])[None, :] < number[:, None]
    flip = np.zeros(z.shape, dtype=bool)
    np.put_along_axis(flip, order, take, axis=1)
    return np.where(flip, -z, z)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    seed=202611120; n=50000; rng=np.random.default_rng(seed)
    z=rng.standard_normal((n,64)).astype(np.float32)
    noise=rng.standard_normal(z.shape).astype(np.float32)
    ranks=rng.random(z.shape).astype(np.float32)
    full=np.abs(rng.standard_normal(z.shape)).astype(np.float32)
    raw=a.output/'ideal_samples.npz'
    np.savez_compressed(raw,z=z,noise=noise,ranks=ranks,full=full)
    rows=[]
    def record(panel,arm,x,v):
        mean=float(v.mean()); se=float(v.std(ddof=1)/np.sqrt(n))
        if panel=='channel':
            den=1+1.96**2/n; center=(mean+1.96**2/(2*n))/den
            half=1.96*np.sqrt(mean*(1-mean)/n+1.96**2/(4*n*n))/den
            low,high=center-half,center+half
        else: low,high=mean-1.96*se,mean+1.96*se
        rows.append(dict(panel=panel,arm=arm,x=float(x),mean=mean,
                         se=se,low=float(low),high=float(high),n=n))
    for required in [33,40,48,56,60,64]:
        for arm,order in [('minimal',None),('random',ranks)]:
            out=project(z,64-required,order)
            record('distortion',arm,required,np.mean((out-z)**2,axis=1))
    arms={'full':full,'saturated':np.abs(z),'minimal56':project(z,8),
          'random56':project(z,8,ranks)}
    for sigma in np.arange(0,3.01,.25):
        for arm,out in arms.items():
            record('channel',arm,sigma,((out+sigma*noise>0).sum(1)>32).astype(float))
    with (a.output/'ideal_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    styles={'minimal':('#0072B2','o','-'),'random':('#D55E00','s','--'),
            'full':('#000000','v',':'),'saturated':('#009E73','^','-.'),
            'minimal56':('#0072B2','o','-'),'random56':('#D55E00','s','--')}
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,ax=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
        for j,panel in enumerate(['distortion','channel']):
            names=['minimal','random'] if j==0 else list(arms)
            for arm in names:
                rr=[r for r in rows if r['panel']==panel and r['arm']==arm]
                x=np.array([r['x'] for r in rr]);y=np.array([r['mean'] for r in rr])
                c,m,l=styles[arm]
                ax[j].plot(x,y,color=c,marker=m,linestyle=l,markersize=3,label=arm)
                ax[j].fill_between(x,[r['low'] for r in rr],[r['high'] for r in rr],color=c,alpha=.12)
            ax[j].legend(frameon=False,fontsize=8)
            ax[j].set_ylim(bottom=0)
        ax[0].set(title='a  Cost of sign projection',xlabel='Required matching signs / 64',ylabel='Mean squared latent change')
        ax[0].axvline(56,color='#777777',linewidth=.8,linestyle=':')
        ax[1].set(title='b  Majority decoding',xlabel='Gaussian measurement noise SD',ylabel='Probability of correct bit',ylim=(0,1.03))
        fig.suptitle('Ideal Gaussian channel only — no diffusion or semantic extractor',fontsize=10)
        fig.savefig(a.output/'ideal_mechanism.png',dpi=300)
        fig.savefig(a.output/'ideal_mechanism.pdf')
        plt.close(fig)
    manifest=dict(seed=seed,n=n,unit='independent 64-coordinate group',
        uncertainty='Pointwise: distortion ±1.96 Monte Carlo SE; bit correctness 95% Wilson interval. Not simultaneous; same groups paired across arms and noise levels.',
        target='All positive by random-key sign symmetry; no semantic mismatch in this illustrative channel.',
        measurement='Additive independent standard Gaussian measurement error times sigma. This is a mathematical channel simulation, not an image transformation or an attack baseline.',
        limitation='No evidence of model robustness or defense superiority; publisher formatting pending target selection.',
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packages=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        figure_inches=[7.2,3.25],png_dpi=300,
        alt_text='Two panels compare Gaussian latent distortion and ideal majority decoding. Curves show the tension between small embedding change and reliable bit transmission. Exact plotted data are in ideal_summary.csv.')
    (a.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([r for r in rows if (r['panel']=='distortion' and r['x']==56) or (r['panel']=='channel' and r['x']==1)],indent=2))


if __name__=='__main__':main()
