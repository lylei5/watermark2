"""Freeze a full-codeword Gaussian coupling experiment on server249."""
import argparse,datetime,hashlib,json,secrets,shutil,subprocess,sys
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    run=root/'runs/semantic_transport_pilot_20261112';run.mkdir(exist_ok=False)
    private=root/'private_eval/semantic_transport_pilot_20261112';private.mkdir(mode=0o700,exist_ok=False)
    source=json.loads((root/'runs/semantic_feedback_pilot_20261112/protocol.json').read_text())
    plan={k:source[k] for k in ['paths','chain_run','distortion_source','conditions','covers','hf_home','upstream','attack_python','external_sha256']}
    captions=[
        'A red tram passing through a rainy city street with reflections on the pavement.',
        'A ceramic tea set on a wooden table beside a bamboo garden.',
        'A small sailboat on a calm lake beneath a pale sunrise.',
        'A snow-covered log cabin surrounded by tall pine trees.',
        'An orange tabby cat stretching on a grey sofa beside a bookcase.',
        'A silver microscope on a laboratory bench against a blue background.']
    arms=['full','saturated','transport','shuffled_transport']
    plan.update(purpose='Monotone conditional Gaussian transport couples the semantic preview to a fully matching sign codeword',
        freeze_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=[dict(task=t,caption=c,aux_seed=202611128100+t) for t,c in enumerate(captions)],
        identity_root=str(private),arms=arms,primary='transport',coupling_control='shuffled_transport',
        carrier_module='transport_binding',capture_initial=True,gpu_indices=[7,5,4],max_gpus=3,excluded_gpus=[6],
        generation_steps=50,inversion_steps=50,decoder_optimization_steps=100,
        margin_correct=64,repetitions=64,message_bits=256,semantic_bits=1024,
        generation_calls_per_arm=2,unique_generation_calls=30,genuine_records=240,wrong_identity_pairs=1200,
        after_positive_gate=dict(arms=arms,jobs=24,updates=151,verification_records=432),
        gates=dict(genuine='candidate >= all three controls',clean='6/6',
            binding='candidate <= saturated-0.02 and candidate <= shuffled_transport',
            fidelity='candidate LPIPS <= saturated-0.05',wrong_identity='candidate <= minimum control accepts'),
        decoder='Original hard majority and BA>181/256; unchanged for all arms',
        distribution_scope='Joint Gaussian only averaged over ideal fresh independent uniform ciphertext; not a fixed-key or empirical security proof',
        scope='Six new source feasibility screen; no tuning after outcomes; neither a native complete SemBind reproduction nor paper-scale confirmation',
        isolation='Baseline arguments and loading exclude private verifier inputs; same OS account, not an OS sandbox')
    scripts=['anchor_binding.py','transport_binding.py','test_transport_binding.py','prepare_semantic_transport.py',
        'run_anchor_binding_worker.py','run_anchor_binding_pipeline.py','run_decision_region_pipeline.py','run_published_imprint_full.py']
    for n in scripts:shutil.copyfile(Path(__file__).parent/n,run/n)
    plan['source_sha256']={n:sha(run/n) for n in scripts};plan['identity_sha256']={}
    for t in range(6):
        s={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
        s.update(amplitude_seed=202611128200+t,transport_shuffle_seed=202611128300+t)
        dest=private/f'{t:03}.json';dest.write_text(json.dumps(s,indent=2)+'\n');dest.chmod(0o600);plan['identity_sha256'][dest.name]=sha(dest)
    subprocess.run([sys.executable,str(run/'test_transport_binding.py'),'--output',str(run/'unit_checks.json')],check=True)
    plan['unit_checks_sha256']=sha(run/'unit_checks.json')
    for n,d in plan['external_sha256'].items():assert sha(n)==d,n
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'),unit_checks=True)))


if __name__=='__main__':main()
