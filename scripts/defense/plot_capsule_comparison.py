"""Descriptive research figure; matched TR attacks and fixed source transfer."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['MPLBACKEND']='Agg'
ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/transported_capsule_observability_20261115'
TRANSFER=ROOT/'runs/capsule_source_transfer_20261115_v3'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from matplotlib.text import Text
    assert read(RUN/'complete.json')['status']==read(TRANSFER/'complete.json')['status']=='complete'
    out=TRANSFER/'figure_v2';out.mkdir(exist_ok=False)
    matches=read(RUN/'matched.json');aggregate=read(TRANSFER/'aggregate.json');data=[]
    names=[('candidate','Forward key recovery'),('sscd','SSCD single reference'),('sscd_centered','Centered SSCD'),
        ('sscd_templates','SSCD nine templates'),('sscd_int8_templates','SSCD int8 nine templates'),
        ('copynce_descriptor','CopyNCE descriptor'),('copynce_matcher','CopyNCE matcher'),
        ('reference_residual','Forward reference residual'),('pixel_residual','Pixel residual')]
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
           'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}
    with plt.rc_context(style):
        fig=plt.figure(figsize=(12,9));grid=fig.add_gridspec(2,2,height_ratios=[1.4,1],hspace=.50,wspace=.50)
        a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
        for proxy,offset,color,marker in [('sd21',-.13,'#0072B2','o'),('sd15',.13,'#B34A00','s')]:
            values=[]
            for method,label in names:
                control='sscd' if method=='candidate' else method
                r=next(r for r in matches if r['method']=='spread_order_a03' and r['carrier']=='TR' and r['control']==control and r['kind']=='reprompt_plus_'+proxy)
                assert r['candidate_genuine']==r['control_genuine']==95 and r['source_total']==10
                count=r['candidate_sources'] if method=='candidate' else r['control_sources'];values.append(count)
                data.append(dict(panel='A',method=method,condition=proxy,accepted=count,total=10))
            a.scatter(values,np.arange(len(names))+offset,label='SD2.1 proxy' if proxy=='sd21' else 'SD1.5 proxy',color=color,marker=marker,s=44,zorder=3)
        a.set(yticks=np.arange(len(names)),yticklabels=[v for _,v in names],xlim=(-.3,10.5),xticks=range(0,11,2),xlabel='Accepted Reprompt+ sources / 10 (lower is better)')
        a.invert_yaxis();a.grid(axis='x',alpha=.20);a.legend(loc='center',bbox_to_anchor=(.65,.44),frameon=False)
        a.set_title('A  TR: all methods accept 95 / 100 genuine images',loc='left',fontweight='bold',pad=10)
        labels=[('spread_order_a03','Forward recovery'),('spread_order_identity_a03','Identity ablation'),('sscd','SSCD single'),('copynce_descriptor','CopyNCE descriptor'),('copynce_matcher','CopyNCE matcher')]
        for ax,kind,panel,title in [(b,'all_genuine','B','Genuine images'),(c,'same','C','Same-prompt different instances')]:
            counts=[];totals=[]
            for method,label in labels:
                r=next(r for r in aggregate if r['dataset']=='all' and r['method']==method and r['kind']==kind);counts.append(r['accepted']);totals.append(r['total'])
                data.append(dict(panel=panel,method=method,condition=kind,accepted=r['accepted'],total=r['total']))
            values=100*np.asarray(counts)/np.asarray(totals)
            ax.barh(np.arange(len(labels)),values,color=['#0072B2','#999999','#444444','#666666','#888888'],height=.60)
            for y,(v,n,d) in enumerate(zip(values,counts,totals)):ax.text(v+1.4,y,f'{n}/{d}',va='center',fontsize=9)
            ax.set(yticks=np.arange(len(labels)),yticklabels=[v for _,v in labels],xlim=(0,119),xticks=[0,25,50,75,100],xlabel='Observed acceptance (%)')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.set_title(panel+'  '+title,loc='left',fontweight='bold',pad=10)
        fig.subplots_adjust(left=.23,right=.97,top=.92,bottom=.23)
        fig.suptitle('Forward key recovery: development attacks and source transfer',x=.23,y=.975,ha='left',fontsize=14,fontweight='bold')
        note=fig.text(.23,.06,'A: descriptive equal-genuine-count thresholds; three candidates per source and proxy.\nB–C: unchanged GS-development thresholds; 100 reference images, 10 genuine observations each.\nSource-transfer assets were used in earlier representation studies. Counts describe these cohorts; no population CI.',fontsize=9,ha='left',va='bottom')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
        assert not note.get_window_extent(renderer).overlaps(b.xaxis.label.get_window_extent(renderer))
        assert not note.get_window_extent(renderer).overlaps(c.xaxis.label.get_window_extent(renderer))
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text():continue
            box=artist.get_window_extent(renderer)
            if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:outside.append(artist.get_text())
        assert not outside,outside
        fig.savefig(out/'capsule_comparison.png',dpi=300);fig.savefig(out/'capsule_comparison.pdf');plt.close(fig)
    with (out/'source_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','method','condition','accepted','total']);writer.writeheader();writer.writerows(data)
    with Image.open(out/'capsule_comparison.png') as im:shape=im.size
    manifest=dict(source_sha256=sha(Path(__file__)),inputs={str(p):sha(p) for p in [RUN/'matched.json',TRANSFER/'aggregate.json']},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},pixels=shape,figure_inches=[12,9],dpi=300,matplotlib=matplotlib.__version__,
        destination='General research report, publisher-specific requirements not selected.',
        aggregation='A:any candidate accepted per source/proxy at95/100genuine. B/C: observed counts from frozen transfer; genuine repeats share a source.',
        uncertainty='Observed cohort counts, no population inference or significance claim.',missing='No absent values plotted as zero.',text_bounds_passed=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
