"""Artifact-checked prompt-cluster statistics for the frozen phase1 probe."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import beta


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def cp(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['generate','verify'],required=True)
    a=p.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    plan=json.loads((run/'protocol.json').read_text());records=[]
    for shard in range(3):
        result=json.loads((run/f'{a.stage}_{shard}/complete.json').read_text())
        assert result['status']=='complete' and result['protocol_sha256']==sha(run/'protocol.json')
        records.extend(result['rows'])
    records.sort(key=lambda x:x['task'])
    assert [r['task'] for r in records]==list(range(100))
    def check(artifact):
        assert sha(run/artifact['path'])==artifact['sha256'],artifact['path']
    if a.stage=='verify':
        rows=[r for rec in records for r in rec['decisions']]
        for row in rows:
            check(row['image']);check(row['latent'])
            assert row['matching_bits']/256==row['ba']
        summaries=[]
        for dataset in ['COCO','SDP']:
            for kind,detector in [('gs','gs'),('bound','bound'),('aux','gs'),('aux','bound')]:
                subset=[r for r in rows if r['dataset']==dataset and r['image_kind']==kind and r['detector']==detector]
                assert len(subset)==50
                for threshold in plan['working_points']:
                    count=sum(r['matching_bits']>=threshold for r in subset)
                    summaries.append(dict(dataset=dataset,image_kind=kind,detector=detector,threshold=threshold,
                        accepted=count,n=50,rate=count/50,cp95=cp(count,50),mean_ba=float(np.mean([r['ba'] for r in subset]))))
        write(run/'baseline_summary.json',dict(status='complete',rows=rows,summaries=summaries,
            protocol_sha256=sha(run/'protocol.json'),scope='Clean TPR and unwatermarked generated-image randomized-identity null; no attacks',
            null_limit='Fresh identity per prompt; not a fixed-user natural-image FPR; n50 per dataset cannot validate1e-6'))
        print(json.dumps(summaries));return
    codes={};times=[]
    for record in records:
        check(record['codes']);check(record['trajectory'])
        for image in list(record['images'].values())+record['trajectory_images']: check(image)
        with np.load(run/record['codes']['path']) as data: codes[record['task']]={k:data[k] for k in data.files}
        assert codes[record['task']]['trajectory_codes'].shape==(46,1024)
        times.append(record['seconds'])
    assert sum(r['observer_check'] is not None for r in records)==3
    for record in records:
        if record['observer_check']: assert all(record['observer_check'].values())
    rows=[]
    for source in plan['sources']:
        task=source['task'];data=codes[task]
        next_task=(task//50)*50+(task+1)%50
        agreement=(data['trajectory_codes']==data['aux_code'][None,:]).mean(axis=1)
        control=dict(aux_final=float(np.mean(data['aux_code']==data['bound_code'])),
            same_prompt=float(np.mean(data['aux_code']==data['same_code'])),
            different_prompt=float(np.mean(data['aux_code']==codes[next_task]['aux_code'])),
            jpeg_aux=float(np.mean(data['aux_code']==data['jpeg_aux_code'])),
            jpeg_bound=float(np.mean(data['bound_code']==data['jpeg_bound_code'])))
        rows.append(dict(task=task,dataset=source['dataset'],agreement=agreement.tolist(),controls=control))
    summary=[];curves=[];control_summary=[];gates={}
    for group in ['COCO','SDP','ALL']:
        subset=[r for r in rows if group=='ALL' or r['dataset']==group]
        values=np.array([r['agreement'] for r in subset]);n=len(subset)
        indices=np.random.default_rng(plan['statistics']['seed']).integers(0,n,(10000,n))
        for j,step in enumerate(plan['trajectory_steps']):
            v=values[:,j];boot=v[indices]
            curves.append(dict(dataset=group,step=step,n=n,mean=float(v.mean()),p5=float(np.quantile(v,.05)),minimum=float(v.min()),
                mean_ci=np.quantile(boot.mean(axis=1),[.025,.975]).tolist(),p5_ci=np.quantile(boot,.05,axis=1).tolist()))
            curves[-1]['p5_ci']=np.quantile(curves[-1]['p5_ci'],[.025,.975]).tolist()
        point=values[:,plan['trajectory_steps'].index(plan['primary_step'])]
        paired={}
        for name in rows[0]['controls']:
            v=np.array([r['controls'][name] for r in subset])
            control_summary.append(dict(dataset=group,control=name,n=n,mean=float(v.mean()),p5=float(np.quantile(v,.05)),minimum=float(v.min())))
            if name in plan['gates']['required_controls']:
                delta=point-v
                paired[name]=dict(mean_advantage=float(delta.mean()),ci95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist())
        gate=bool(np.quantile(point,.05)>.95 and all(v['ci95'][0]>0 for v in paired.values()))
        gates[group]=gate
        summary.append(dict(dataset=group,n=n,step=35,mean=float(point.mean()),p5=float(np.quantile(point,.05)),minimum=float(point.min()),paired=paired,pass_gate=gate))
    result=dict(status='complete',protocol_sha256=sha(run/'protocol.json'),rows=rows,curves=curves,controls=control_summary,
        primary=summary,gates=gates,advance_necessary_condition=gates['COCO'] and gates['SDP'],
        mean_generation_seconds_per_prompt=float(np.mean(times)),observer_equality_checks=3,
        scope='Observational necessary condition only; no in-trajectory binding, no defense success or attack robustness claim')
    write(run/'trajectory_summary.json',result)
    with (run/'trajectory_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','step','n','mean','p5','minimum','mean_ci','p5_ci']);writer.writeheader();writer.writerows(curves)
    print(json.dumps(dict(primary=summary,controls=control_summary,advance_necessary_condition=result['advance_necessary_condition'])))


if __name__=='__main__': main()
