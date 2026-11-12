"""Freeze six-source full-step clean registration screen on server249."""
import argparse
import datetime
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    run = root / 'runs/continuous_geometry_screen_20261113_v2'
    run.mkdir(exist_ok=False)
    private = root / 'private_eval/continuous_geometry_screen_20261113_v2'
    private.mkdir(mode=0o700, exist_ok=False)
    source = json.loads((root / 'runs/semantic_transport_pilot_20261112/protocol.json').read_text())
    plan = {k: source[k] for k in ['paths', 'chain_run', 'distortion_source', 'hf_home', 'upstream', 'external_sha256']}
    captions = [
        'A waterfall cascading over dark rocks into a green pool beneath a rainbow.',
        'A quiet library with curved bookshelves and sunlight on a reading desk.',
        'A wooden cello resting against a chair in a rehearsal room.',
        'A white windmill overlooking a field of yellow flowers under a cloudy sky.',
        'A bowl of sliced citrus fruit beside a glass pitcher on a kitchen counter.',
        'A blue bicycle leaning against a stone wall beside a flowering tree.']
    plan.update(purpose='Full-step clean feasibility of continuous content-dependent Gaussian decoding regions',
        execution_repair='Original coordinator lacked Python os.pidfd_open before GPU launch; syscall434 event wait used. Same frozen method, sources and gates; original artifacts preserved.',
        freeze_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=[dict(task=t, caption=c, aux_seed=202611130200+t) for t,c in enumerate(captions)],
        identity_root=str(private), arms=['full', 'quantized_geometry', 'continuous_independent', 'continuous_transport'],
        primary='continuous_transport', carrier_module='continuous_geometry_binding', carrier_feature='logits',
        capture_initial=True, capture_semantic_logits=True, conditions=[dict(name='clean', value=None)],
        gpu_indices=[7,5,4], max_gpus=3, excluded_gpus=[6],
        generation_steps=50, inversion_steps=50, decoder_optimization_steps=100,
        message_bits=256, repetitions=64, semantic_dimension=1024, frequency_vectors=1024,
        phase_pairs=8192, alpha=2.0, alpha_reason='For orthogonal unit features, expected orientation cosine exp(-alpha^2)=exp(-4); fixed from this illustrative design target, not fitted outcomes.',
        generation_calls_per_arm=2, unique_generation_calls=30, genuine_records=24, wrong_identity_pairs=120,
        gates=dict(clean='candidate6/6 and no less than controls',
            binding='mean auxiliary-to-delivered unit-logit distance <= full and <= continuous_independent',
            continuous_vs_quantized='mean clean BA >= quantized_geometry',
            fidelity='mean preview LPIPS <= continuous_independent-0.05',
            wrong_identity='candidate <= minimum control accepts'),
        detector='Original hard64-vote decoder; BA>181/256 unchanged. Candidate first inverts query-conditioned rotation.',
        scope='Six new source clean mechanism screen. No robustness, forgery, matched-TPR or SOTA claim. Any later published-baseline evaluation needs a separately frozen full protocol.',
        attack_jobs=0, no_new_attacks=True,
        distribution_scope='Ideal fresh independent uniform ciphertext and independent uniform pair phases; keyed finite-precision implementation is not a proof of unconditional security.',
        parent_run=str(root/'runs/semantic_transport_pilot_20261112'))
    scripts=['semantic_code.py','sembind_official_audit.py','codeword_binding.py','anchor_binding.py','transport_binding.py','continuous_geometry_binding.py',
             'test_continuous_geometry.py','run_anchor_binding_worker.py',
             'run_continuous_geometry_screen.py','prepare_continuous_geometry.py']
    for name in scripts:
        shutil.copyfile(Path(__file__).parent/name, run/name)
    plan['source_sha256']={name:sha(run/name) for name in scripts}
    plan['identity_sha256']={}
    for t in range(6):
        secret={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
        secret['amplitude_seed']=202611130300+t
        path=private/f'{t:03}.json'
        path.write_text(json.dumps(secret,indent=2)+'\n');path.chmod(0o600)
        plan['identity_sha256'][path.name]=sha(path)
    subprocess.run([sys.executable,str(run/'test_continuous_geometry.py'),'--output',str(run/'unit_checks.json')],check=True)
    plan['unit_checks_sha256']=sha(run/'unit_checks.json')
    for path,digest in plan['external_sha256'].items():
        assert sha(path)==digest
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))))


if __name__=='__main__':
    main()
