"""Audit spectral recovery and quantify real-edit common-channel cancellation."""
import argparse
import json
import os
from pathlib import Path

from shared_channel_experiment import ROOT, read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();assert ROOT.is_dir() and run.parent==ROOT/'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    import numpy as np
    from shared_channel_features import spectrum,phase
    plan=read(run/'protocol.json');done=read(run/'complete.json');assert done['status']=='complete'
    assert not (run/'audit.json').exists()
    checked={}
    for manifest in [read(run/'extract_complete.json'),done]:
        for name,digest in manifest['artifacts'].items():
            assert sha(run/name)==digest;checked[name]=digest
    for name,digest in plan['scripts'].items():assert sha(run/name)==digest
    rows,controls,enrollment=read(run/'rows.json'),read(run/'controls.json'),read(run/'enrollment.json')
    assert len(rows)==17040 and len(controls)==710 and len(enrollment)==720
    assert len({(x['id'],x['query_id'],x['kind'],x['method'],x['repetition'])for x in rows})==17040
    assert all(x['key_recovered']==(x['region_distance']<=127)for x in rows)
    assert all(x['joint_accepted']==(x['key_recovered'] and x['native_accepted'])for x in rows)
    assert all(x['recovered_detector_equal']is True for x in rows if x['key_recovered'])
    private=Path(plan['private']);blobs={}
    for x in enrollment:
        helper=private/f"{x['id']}_{x['repetition']}_{x['method']}.json"
        assert sha(helper)==x['helper_sha256']
        blobs[x['id']]=x['native_blob_sha256']
    for identity,digest in blobs.items():assert sha(private/f'{identity}_native.bin')==digest
    # Joint absolute calibration coincides with content calibration only if every
    # native calibration query is accepted; enforce that condition explicitly.
    calibration=[r for r in controls if r['split']=='calibration'and r['kind']in ['clean',*plan['conditions']]]
    assert len(calibration)==30 and all(r['native_accepted']for r in calibration)
    meta=read(run/'feature_rows.json');data=np.load(run/'features.npz')
    index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
    individual=[]
    for i,r in enumerate(meta):
        if r['kind']not in plan['conditions']:continue
        ref=index[r['id'],'clean']
        aq,bq=spectrum(data['content'][i]),spectrum(data['noise'][i])
        ar,br=spectrum(data['content'][ref]),spectrum(data['noise'][ref])
        da,db=phase(aq)*np.conj(phase(ar)),phase(bq)*np.conj(phase(br))
        content_phase=float(np.mean(1-da.real));noise_phase=float(np.mean(1-db.real))
        common_phase=float(np.mean(1-(da*np.conj(db)).real))
        la=np.log(np.maximum(np.abs(aq),1e-12))-np.log(np.maximum(np.abs(ar),1e-12))
        lb=np.log(np.maximum(np.abs(bq),1e-12))-np.log(np.maximum(np.abs(br),1e-12))
        individual.append(dict(id=r['id'],carrier=r['carrier'],task=r['task'],condition=r['kind'],
            content_phase_drift=content_phase,noise_phase_drift=noise_phase,relative_phase_drift=common_phase,
            relative_phase_below_both=common_phase<min(content_phase,noise_phase),
            content_log_rms=float(np.sqrt(np.mean(la**2))),noise_log_rms=float(np.sqrt(np.mean(lb**2))),
            relative_log_rms=float(np.sqrt(np.mean((la-lb)**2)))))
    summaries=[]
    for carrier in ['GS','TR','PRC']:
        for condition in ['all_edits','known_edits','unknown_edits',*plan['conditions']]:
            part=[r for r in individual if r['carrier']==carrier and
                (carrier!='GS'or r['task']in plan['evaluation_sources'])and
                (True if condition=='all_edits'else r['condition']in plan['enrollment_families']if condition=='known_edits'else
                 r['condition']not in plan['enrollment_families']if condition=='unknown_edits'else r['condition']==condition)]
            if not part:continue
            summaries.append(dict(carrier=carrier,condition=condition,total=len(part),
                phase_improved_over_both=sum(r['relative_phase_below_both']for r in part),
                **{k:float(np.mean([r[k]for r in part]))for k in ['content_phase_drift','noise_phase_drift',
                    'relative_phase_drift','content_log_rms','noise_log_rms','relative_log_rms']}))
    comparison=[]
    scores=read(run/'scores.json')
    for candidate in scores['scores']:
        if candidate['repetition']is None or not candidate['feasible']:continue
        for carrier,split in [('GS','evaluation'),('TR','transfer'),('PRC','transfer')]:
            row=next(r for r in candidate['summary']if r['carrier']==carrier and r['split']==split and r['kind']=='all_genuine')
            cc=[r for r in controls if r['carrier']==carrier and r['split']==split]
            for baseline in ['sscd','sscd_centered','sscd_templates','sscd_int8_templates']:
                values=sorted(r[baseline]for r in cc if r['kind']in ['clean',*plan['conditions']]and r['native_accepted'])
                assert row['joint']<=len(values)
                threshold=values[row['joint']-1]if row['joint']else-1.
                result=[]
                for kind in ['all_genuine','same_prompt','wrong_record','cover','imprint_150']:
                    group=[r for r in cc if(r['kind']in ['clean',*plan['conditions']]if kind=='all_genuine'else r['kind']==kind)]
                    if group:result.append(dict(kind=kind,total=len(group),accepted=sum(r['native_accepted']and r[baseline]<=threshold for r in group)))
                comparison.append(dict(method=candidate['method'],repetition=candidate['repetition'],target=candidate['target'],
                    carrier=carrier,split=split,baseline=baseline,threshold=threshold,candidate_genuine=row['joint'],summary=result))
    write(run/'audit.json',dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        artifact_hashes=checked,actual_recoveries=len(rows),unique_region_agreements=len(rows),helper_hash_checks=len(enrollment),
        native_blob_hash_checks=len(blobs),native_calibration_all_accepted=True,
        common_channel_scope='Post-hoc mechanism explanation on actual published edits; no new image perturbation or method changes',
        common_channel_summary=summaries,common_channel_rows=individual,
        descriptive_matching_scope='Thresholds selected on evaluation genuine scores only for descriptive equal-absolute-TPR comparison',
        matched_comparisons=comparison))
    print(json.dumps(dict(status='complete',actual_recoveries=len(rows),common_channel_summary=summaries[:3],
                         descriptive_comparisons=len(comparison))),flush=True)


if __name__=='__main__':main()
