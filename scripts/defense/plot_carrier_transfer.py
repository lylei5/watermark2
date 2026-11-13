"""Render completed native/PGID/SSCD/recovery comparison with explicit working points."""
import argparse
import csv
import json
import os
from pathlib import Path

from recovered_carrier_transfer import ROOT,sha,write,wait


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parents',type=Path)
    args=parser.parse_args();run=args.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    if args.parents:
        for parent in json.loads(args.parents.read_text()):wait(parent)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    output=run/'figures';output.mkdir(exist_ok=False)
    state=dict(status='running',source_sha256=sha(__file__))
    try:
        data=json.loads((run/'audited_scores_v3.json').read_text());assert data['status']=='complete'
        names=[('native_fresh','Native carrier'),('PGID','PGID'),('sscd_single','SSCD: single'),
               ('sscd_templates','SSCD: 9 templates'),('sscd_int8_templates','SSCD: 9 templates (int8)'),
               ('orbit_margin_fixed','Edit-margin key'),('tangent4_margin_fixed','4-direction key'),
               ('tangent8_margin_fixed','8-direction key'),('tangent4_orbit_fixed','4-direction + edit margin')]
        table=[]
        for carrier in ['TR','PRC']:
            for method,label in names:
                selected=[r for r in data['summaries']if r['method']==method and
                          (method in ['native_fresh','PGID']or r.get('target')==.9)]
                assert len(selected)in[1,3]
                for row in selected:
                    summary=row.get('summary')
                    def count(kind):
                        if summary is None:return None
                        item=next(x for x in summary if x['carrier']==carrier and x['kind']==kind)
                        assert item['total']==(100 if kind=='all_genuine'else 10)
                        return item['accepted']
                    table.append(dict(carrier=carrier,method=method,label=label,repetition=row.get('repetition'),
                        legitimate=count('all_genuine'),final_imprint=count('imprint_150'),
                        working_point='Author setting'if method in ['native_fresh','PGID']else'Frozen GS90% calibration'))
        with(output/'source_data.csv').open('w',newline='')as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        style={'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,
               'axes.spines.right':False,'axes.spines.left':False,'axes.axisbelow':True,'figure.facecolor':'white'}
        with plt.rc_context(style):
            fig,axes=plt.subplots(1,2,figsize=(10.8,5.4),sharey=True)
            fig.subplots_adjust(left=.245,right=.98,bottom=.24,top=.87,wspace=.14)
            for ax,carrier in zip(axes,['TR','PRC']):
                for i,(method,label)in enumerate(names):
                    part=[r for r in table if r['carrier']==carrier and r['method']==method]
                    if part[0]['legitimate']is None:
                        ax.text(48,i,'Calibration target not reached',ha='center',va='center',fontsize=8,color='#595959')
                        continue
                    is_recovery=method.endswith('_fixed')
                    color='#B74D00'if is_recovery else'#005E8A'
                    marker='o'if is_recovery else's'
                    for j,row in enumerate(part):
                        y=i+(j-1)*.16 if len(part)==3 else i
                        ax.scatter(row['legitimate'],y,color=color,marker=marker,s=28,zorder=3)
                    values=[r['legitimate']for r in part]
                    text=' / '.join(map(str,values))
                    ax.text(3,i,text,ha='left',va='center',fontsize=9,color='#222222')
                ax.set_xlim(0,103);ax.set_xticks([0,20,40,60,80,100]);ax.set_ylim(len(names)-.4,-.7)
                ax.set_title(carrier,loc='left',fontweight='bold',fontsize=13)
                ax.set_xlabel('Legitimate queries accepted (of 100)')
                ax.set_yticks(range(len(names)),[x[1]for x in names]);ax.tick_params(axis='y',length=0,pad=12)
                ax.grid(axis='x',color='#DDDDDD',linewidth=.65)
            fig.suptitle('Native-carrier transfer at frozen operating points',x=.02,ha='left',fontsize=15,fontweight='bold')
            note=('SSCD and key recovery: frozen GS90% calibration; native and PGID: original author settings.\n'
                  '10 sources per carrier; each contributes 10 legitimate queries. Circles show all 3 projection repetitions, not confidence intervals.\n'
                  'Final Imprint: native 10/10 accepted per carrier; PGID and every feasible content gate shown accept 0/10.')
            fig.text(.02,.045,note,ha='left',va='bottom',fontsize=8.5,linespacing=1.5,color='#333333')
            for row in table:
                if row['final_imprint']is not None:assert row['final_imprint']==(10 if row['method']=='native_fresh'else 0)
            fig.savefig(output/'carrier_comparison.png',dpi=300,facecolor='white',transparent=False)
            fig.savefig(output/'carrier_comparison.pdf',facecolor='white',transparent=False)
            plt.close(fig)
        with Image.open(output/'carrier_comparison.png')as im:
            metadata=dict(pixels=list(im.size),dpi=list(im.info.get('dpi',[])),mode=im.mode,
                          fully_opaque=im.mode!='RGBA'or im.getchannel('A').getextrema()==(255,255))
        manifest=dict(status='complete',purpose='Internal research comparison; publisher-specific requirements not selected',
            source_sha256=sha(__file__),audit_sha256=sha(run/'audited_scores_v3.json'),
            matplotlib_version=matplotlib.__version__,physical_size_inches=[10.8,5.4],png=metadata,
            aggregation='Per-query accepted count; denominator100 across10sources, repeated queries retained',
            uncertainty='All3projection repetitions shown separately; no confidence interval or populationFPR estimate',
            missing_values='Infeasible GS90% calibration displayed as text; not plotted as zero',
            alt_text='Two panels compare legitimate acceptance for TR and PRC across native, PGID, SSCD and actual key recovery. The PRC raw detector limits the joint acceptance of SSCD and recovery. All feasible defenses shown reject the final Imprint outputs in this cohort.',
            outputs={p.name:sha(p)for p in output.iterdir()if p.is_file()})
        write(output/'manifest.json',manifest);state.update(status='complete',manifest_sha256=sha(output/'manifest.json'))
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'figure_state.json',state)


if __name__=='__main__':main()
