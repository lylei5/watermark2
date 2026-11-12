"""Plot every source in the completed, audited decision-region model pilot."""
import argparse,csv,hashlib,json,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run
    data=json.loads((run/'genuine_verified.json').read_text())
    plan=json.loads((run/'protocol.json').read_text())
    assert data['audit_passed'] and len(data['rows'])==240 and len(data['references'])==24
    out=run/'figures';out.mkdir(exist_ok=True)
    records=[]
    for ref in data['references']:
        rr=[r for r in data['rows'] if (r['task'],r['arm'])==(ref['task'],ref['arm'])]
        assert len(rr)==10
        records.append(dict(task=ref['task'],arm=ref['arm'],accepted=sum(r['accepted'] for r in rr),n=10,
            clean_accepted=next(r['accepted'] for r in rr if r['condition']=='clean'),
            binding_mismatch=ref['semantic_mismatch'],preview_lpips=ref['lpips']))
    with (out/'source_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    colors=['#000000','#009E73','#0072B2','#D55E00'];markers=['v','^','o','s'];arms=plan['arms']
    keys=['accepted','binding_mismatch','preview_lpips'];limits=[(0,10.5),(0,None),(0,None)]
    labels=['Accepted genuine queries / 10','Enrollment-to-reference Hamming rate','LPIPS to auxiliary image']
    with plt.rc_context({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(9,3.35),layout='constrained')
        for j,ax in enumerate(axes):
            for task in range(6):
                values=[next(r[keys[j]] for r in records if r['task']==task and r['arm']==a) for a in arms]
                ax.plot(range(4),values,color='#BBBBBB',lw=.6,zorder=1)
                for i,value in enumerate(values):
                    # Small fixed source offsets reveal overlapping raw observations.
                    ax.scatter(i+(task-2.5)*.025,value,color=colors[i],marker=markers[i],s=20,zorder=2)
            means=[np.mean([r[keys[j]] for r in records if r['arm']==a]) for a in arms]
            for i,value in enumerate(means):ax.plot([i-.18,i+.18],[value,value],color=colors[i],lw=2.5,zorder=3)
            ax.set_xticks(range(4),arms,rotation=25,ha='right')
            ax.set(ylabel=labels[j],ylim=limits[j],title=['a  Genuine detection','b  Binding consistency','c  Preview fidelity'][j])
        fig.suptitle('Six paired sources; points = sources, lines = pairs, short bars = means',fontsize=10)
        fig.savefig(out/'model_mechanism.png',dpi=300);fig.savefig(out/'model_mechanism.pdf');plt.close(fig)
    manifest=dict(protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((run/'genuine_verified.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n=6,unit='independent source, paired across arms; each source has ten correlated genuine conditions',
        uncertainty='All six sources shown. No confidence bands or significance claims.',
        transformations='Accepted count summed within source. Reference Hamming and LPIPS directly from audited references. Equal source means. Horizontal jitter is fixed by task ID.',
        exclusions='None. All 240 genuine queries retained, including invalid clean references.',
        packages=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        figure_inches=[9,3.35],png_dpi=300,publisher='Provisional research figure; no submission target specified',
        alt_text='Three paired-source plots show genuine detection counts, semantic mismatch, and preview fidelity for all four embedding arms. Lower mismatch or LPIPS alone does not establish stronger authentication.')
    (out/'model_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(plotted=True,source_rows=len(records),output=str(out))))


if __name__=='__main__':main()
