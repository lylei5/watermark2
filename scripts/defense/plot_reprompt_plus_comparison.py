"""Export descriptive fixed-calibration TR defense counts with source data."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/reprompt_plus_recovery_20261115'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    done=read(RUN/'complete.json');assert done['status']=='complete'
    for name,digest in done['artifacts'].items():assert sha(RUN/name)==digest
    records=read(RUN/'aggregate.json');out=RUN/'figure';out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    names=[('sscd','SSCD'),('sscd_centered','Centered SSCD'),('sscd_templates','SSCD: 9 templates'),
        ('sscd_int8_templates','SSCD: int8 templates'),('copynce_descriptor','CopyNCE descriptor'),
        ('copynce_matcher','CopyNCE matcher'),('coarse','Coarse evidence'),('any_scale','Any scale'),
        ('all_scales','All scales'),('observable_scales','Observable scales')]
    table=[]
    for method,label in names:
        part=[r for r in records if r['carrier']=='TR' and r['method']==method and ('target' not in r or r['target']==.95)]
        assert len(part)==2
        a=next(r for r in part if r['proxy']=='sd21');b=next(r for r in part if r['proxy']=='sd15')
        assert (a['genuine_joint'],a['genuine_total'])==(b['genuine_joint'],b['genuine_total'])
        assert a['references']==b['references']==10 and a['genuine_total']==100
        table.append(dict(method=method,label=label,genuine=a['genuine_joint'],genuine_total=100,
            sd21=a['accepted_references'],sd15=b['accepted_references'],reference_total=10,
            calibration='Frozen GS30 target95%' if 'target' in a else 'Frozen algebraic recovery rule'))
    with (out/'source_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with plt.rc_context({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(11,6.1),sharey=True)
        fig.subplots_adjust(left=.23,right=.97,top=.79,bottom=.17,wspace=.25)
        for ax,key,den,title,color in zip(axes,['genuine','sd21','sd15'],[100,10,10],
                ['Legitimate joint acceptance\nHigher is better','Reprompt+ / SD2.1\nLower is better','Reprompt+ / SD1.5\nLower is better'],
                ['#0072B2','#D55E00','#D55E00']):
            for i,row in enumerate(table):
                x=100*row[key]/den
                ax.plot(x,i,'o' if i<6 else 's',color=color,markersize=6)
                ax.annotate(f"{row[key]}/{den}",(x,i),xytext=(-7 if x>80 else 7,0),textcoords='offset points',
                    ha='right' if x>80 else 'left',va='center',fontsize=9)
            ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Accepted (%)');ax.set_title(title,fontsize=11,pad=12)
            ax.grid(axis='x',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
            ax.axhline(5.5,color='#777777',linewidth=.7,linestyle='--')
        axes[0].set_yticks(range(len(table)),[r['label'] for r in table]);axes[0].invert_yaxis()
        fig.suptitle('Tree-Ring: legitimate coverage and fixed-budget Reprompt+ acceptance',fontsize=14,y=.97)
        fig.text(.23,.87,'Same query pixels and original native detector; three candidates per source and proxy.',fontsize=10)
        fig.text(.23,.07,'10 development sources; 100 legitimate images. A source is accepted if one candidate passes both checks.\n'
            'Circles: frozen GS30 calibration at 95% target. Squares: frozen recovery rules. Descriptive counts, no confidence intervals.',fontsize=9)
        fig.savefig(out/'tr_reprompt_plus.png',dpi=300)
        fig.savefig(out/'tr_reprompt_plus.pdf')
        plt.close(fig)
    with Image.open(out/'tr_reprompt_plus.png') as im:dimensions=list(im.size)
    manifest=dict(status='complete',source_sha256=sha(Path(__file__)),input_complete_sha256=sha(RUN/'complete.json'),
        cohort='TR10sources,100genuine; each proxy30candidates grouped into10sources',
        selection='All6strong controls at frozenGS95calibration and all4recovery rules; no best-result selection.',
        uncertainty='Descriptive counts. Genuine edits repeat within sources; no row-independent interval.',
        publisher='General research figure; journal-specific requirements not assessed.',png_dimensions=dimensions,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        alt_text='Three aligned panels compare legitimate joint acceptance and Reprompt+ source acceptance for all six frozen strong controls and four actual key-recovery rules.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
