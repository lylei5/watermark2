"""Freeze a new-source decision-region embedding experiment on server249."""
import argparse,datetime,hashlib,json,secrets,shutil,subprocess,sys
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    run=root/'runs/decision_region_pilot_20261112';run.mkdir(exist_ok=False)
    private=root/'private_eval/decision_region_pilot_20261112';private.mkdir(mode=0o700,exist_ok=False)
    plan=json.loads((root/'runs/semantic_feedback_pilot_20261112/protocol.json').read_text())
    keep=['paths','chain_run','distortion_source','conditions','covers','hf_home','upstream','attack_python','external_sha256']
    plan={k:plan[k] for k in keep}
    captions=[
        'A brown violin resting in an open case on a wooden stage beside a music stand.',
        'A yellow excavator parked beside piles of gravel beneath an overcast sky.',
        'A bowl of green apples on a kitchen counter beside a blue enamel jug.',
        'Two penguins standing on a flat icy shore with distant mountains behind them.',
        'A white lighthouse on a rocky headland above turquoise ocean waves.',
        'A vintage sewing machine on a table near a window with red curtains.']
    plan.update(purpose='Minimum-distortion sign reflections into a majority-decoding region; fixed56/64 agreement',
        freeze_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=[dict(task=t,caption=c,aux_seed=202611127100+t) for t,c in enumerate(captions)],
        identity_root=str(private),arms=['full','saturated','minimal56','random56'],primary='minimal56',
        carrier_module='decision_region_binding',capture_initial=True,gpu_indices=[7,5,4],max_gpus=3,excluded_gpus=[6],
        generation_steps=50,inversion_steps=50,decoder_optimization_steps=100,
        margin_correct=56,repetitions=64,message_bits=256,semantic_bits=1024,
        generation_calls_per_arm=2,unique_generation_calls=30,genuine_records=240,wrong_identity_pairs=1200,
        after_positive_gate=dict(arms=['full','saturated','minimal56','random56'],jobs=24,updates=151,verification_records=432),
        gates=dict(genuine='candidate >= all three controls',clean='6/6',
                   binding='candidate <= saturated-0.02 and candidate <= random56',
                   fidelity='candidate LPIPS <= saturated-0.05',wrong_identity='candidate <= minimum control accepts'),
        scope='Six new source feasibility screen, not paper-scale confirmation; no threshold tuning or attack modifications',
        isolation='Verifier inputs excluded from baseline process arguments and loading; same OS account, not an OS sandbox')
    scripts=['anchor_binding.py','decision_region_binding.py','test_decision_region_binding.py',
             'run_anchor_binding_worker.py','run_anchor_binding_pipeline.py','run_decision_region_pipeline.py',
             'run_published_imprint_full.py','prepare_decision_region.py']
    for n in scripts:shutil.copyfile(Path(__file__).parent/n,run/n)
    plan['source_sha256']={n:sha(run/n) for n in scripts}
    plan['identity_sha256']={}
    for t in range(6):
        s={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
        s.update(amplitude_seed=202611127200+t,projection_order_seed=202611127300+t)
        dest=private/f'{t:03}.json';dest.write_text(json.dumps(s,indent=2)+'\n');dest.chmod(0o600)
        plan['identity_sha256'][dest.name]=sha(dest)
    subprocess.run([sys.executable,str(run/'test_decision_region_binding.py'),'--output',str(run/'unit_checks.json')],check=True)
    plan['unit_checks_sha256']=sha(run/'unit_checks.json')
    for n,d in plan['external_sha256'].items():assert sha(n)==d,n
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'),unit_checks=True)))


if __name__=='__main__':main()
