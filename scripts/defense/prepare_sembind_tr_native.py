"""Freeze SemBind TR-S with the original Tree-Ring carrier and private binding."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
from types import SimpleNamespace, ModuleType


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-name',required=True)
    p.add_argument('--after-run',required=True)
    args=p.parse_args()
    root=Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name=='posix'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    run=(root/'runs'/args.run_name).resolve();parent=(root/'runs'/args.after_run).resolve()
    assert run.parent==parent.parent==root/'runs'
    run.mkdir(exist_ok=False)
    private=root/'private_eval'/run.name;private.mkdir(mode=0o700,exist_ok=False)
    source=root/'third_party/frozen/TreeRing_3015283'
    if not source.exists():
        shutil.copytree(root/'tools/tree_ring_pinned',source)
    manifest=json.loads((source/'source_manifest.json').read_text())
    for name,entry in manifest['files'].items():
        data=(source/name).read_bytes()
        assert sha(source/name)==entry['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['git_blob']
    # Original helper imports an unused online dataset loader. No numeric function is changed.
    if importlib.util.find_spec('datasets') is None:
        stub=ModuleType('datasets');stub.__spec__=importlib.util.spec_from_loader('datasets',loader=None)
        def unavailable(*a,**kw):
            raise RuntimeError('Dataset loading is outside this frozen cohort')
        stub.load_dataset=unavailable;sys.modules['datasets']=stub
    spec=importlib.util.spec_from_file_location('native_tree_ring',source/'optim_utils.py')
    tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
    parameters=dict(w_channel=3,w_pattern='ring',w_mask_shape='circle',w_radius=10,w_measurement='l1_complex',w_injection='complex',w_pattern_const=0)
    seed=secrets.randbelow(2**32-5)
    config=SimpleNamespace(**parameters,w_seed=seed)
    target=tr.get_watermarking_pattern(None,config,'cpu',shape=(1,4,64,64))
    mask=tr.get_watermarking_mask(torch.zeros(1,4,64,64),config,'cpu')
    (private/'binding.json').write_text(json.dumps(dict(key=secrets.token_hex(32),carrier_seed=seed))+'\n')
    (private/'binding.json').chmod(0o600)
    previous=root/'runs/sembind_prc_native_20261113_v3'
    old=json.loads((previous/'protocol.json').read_text());sources=[];roundtrips=[]
    for item in old['sources']:
        generator=torch.Generator().manual_seed(secrets.randbits(63))
        initial=tr.inject_watermark(torch.randn(1,4,64,64,generator=generator),mask,target,config)
        _,pv=tr.get_p_value(initial,initial,mask,target,config)
        assert np.isfinite(pv) and pv<=1e-6
        file=private/f"{item['task']:03}.npz"
        np.savez_compressed(file,initial=initial.numpy(),target=target.numpy(),mask=mask.numpy())
        file.chmod(0o600)
        sources.append(dict(item,private_carrier_sha256=sha(file)))
        roundtrips.append(dict(task=item['task'],p_value=float(pv),accepted=True))
    names=['run_sembind_tr_native.py','prepare_sembind_tr_native.py','run_baseline_after_exit.py',
           'sembind_binding.py','semantic_code.py','sembind_official_audit.py','codeword_binding.py']
    for name in names:
        shutil.copy2(Path(__file__).parent/name,run/name)
    pid=json.loads((parent/'pipeline_state.json').read_text())['pid']
    file=Path(f'/proc/{pid}/cmdline');command=file.read_bytes() if file.exists() else b''
    assert not command or str(parent).encode() in command
    predecessor=dict(pid=pid,cmdline_sha256=hashlib.sha256(command).hexdigest())
    plan={k:old[k] for k in ['python','paths','official_sha256','scheduler_sha256','source_commit','generation','inversion','auxiliary_source']}
    plan.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='6-source TR/TR-S normal-image integration using original Tree-Ring source and paper SemBind binding; not a full numerical reproduction',
        sources=sources,private=str(private),binding_key_sha256=sha(private/'binding.json'),sigma=1,
        tree_ring_file=str(source/'optim_utils.py'),tree_ring_sha256=sha(source/'optim_utils.py'),tree_ring_manifest=manifest,
        tr_parameters=parameters,decision='Original noncentral-chi-square p-value <=1e-6; theoretical model, not empirically certified FPR',
        carrier_seed_policy='Private32-bit seed required by original NumPy seeding API; no cryptographic-strength claim for the carrier seed',
        binding_seed_policy='Separate private256-bit HMAC permutation key',carrier_roundtrips=roundtrips,
        local_sha256={name:sha(run/name) for name in names},worker='run_sembind_tr_native.py',
        worker_sha256=sha(run/'run_sembind_tr_native.py'),coordinator_sha256=sha(run/'run_baseline_after_exit.py'),
        gpu_indices=[7,5,4],predecessors=[predecessor]*3,attack_jobs=0)
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'),carrier_roundtrips=roundtrips)))


if __name__=='__main__':
    main()
