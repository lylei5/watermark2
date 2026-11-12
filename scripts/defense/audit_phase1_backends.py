"""Read-only asset provenance and original PRC component roundtrips on249 CPU."""
import ast
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import secrets
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def write(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',choices=['phase1_backend_audit_20261113_v2'],required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
    sys.dont_write_bytecode=True
    root=Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and sys.platform=='linux'
    run=root/'runs'/args.run_name;run.mkdir(exist_ok=False)
    private=root/'private_eval'/args.run_name;private.mkdir(mode=0o700,exist_ok=False)
    sys.path.insert(0,str(root/'envs/phase1_prc_extra'))
    upstream=Path('/data0/liyanlei/genstego_sota_assets/robust_diffusion_watermarks/repos/PRC-Watermark')
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert commit=='3f372c1d9cafcd0732331c0d00f0430ee822a389'
    sources={}
    for name in ['src/prc.py','src/pseudogaussians.py','decode.py','encode.py','requirements.txt']:
        expected=subprocess.check_output(['git','-C',str(upstream),'show',commit+':'+name])
        assert hashlib.sha256(expected).hexdigest()==sha(upstream/name)
        dst=run/Path(name).name;shutil.copy2(upstream/name,dst);sources[name]=sha(dst)
    same_ast=ast.dump(ast.parse((run/'prc.py').read_text()))==ast.dump(ast.parse((root/'third_party/SemBind/src/prc.py').read_text()))
    plan=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),prc_commit=commit,source_sha256=sources,
        script_sha256=sha(__file__),samples=8,codeword_length=16384,message_length=512,nominal_fpr=1e-6,
        posterior_variance=1.5,sigma=.5,attack_jobs=0,gpu_used=False,
        extra_dependencies=str(root/'envs/phase1_prc_extra'),
        scope='Original PRC CPU sampling/unbinding/detection/decoding checks only; no image or native SemBind experiment')
    write(run/'protocol.json',plan)
    import numpy as np
    import torch
    import galois
    import importlib.metadata as metadata
    torch.set_num_threads(4)
    prc=load('audited_prc',run/'prc.py');gaussian=load('audited_gaussian',run/'pseudogaussians.py')
    seed=secrets.randbits(32);np.random.seed(seed);torch.manual_seed(seed)
    write(private/'rng_seed.json',dict(seed=seed));(private/'rng_seed.json').chmod(0o600)
    started=time.monotonic()
    encoding,decoding=prc.KeyGen(16384,message_length=512,false_positive_rate=1e-6,t=3)
    with (private/'prc_key.pkl').open('wb') as file: pickle.dump((encoding,decoding),file)
    (private/'prc_key.pkl').chmod(0o600)
    rows=[]
    for i in range(8):
        message=np.random.randint(0,2,512)
        coded=prc.Encode(encoding,message)
        z=gaussian.sample(coded)
        mask=np.ones(16384);mask[np.random.permutation(16384)[:8192]]=np.random.choice([-1,1],8192)
        unbound=(z*torch.from_numpy(mask))*torch.from_numpy(mask)
        assert torch.equal(z,unbound)
        posterior=gaussian.recover_posteriors(unbound,variances=1.5)
        detected=bool(prc.Detect(decoding,posterior))
        np.savez_compressed(private/f'{i:03}.npz',message=message,codeword=coded.numpy(),latent=z.numpy())
        (private/f'{i:03}.npz').chmod(0o600)
        write(run/'progress.json',dict(task=i,stage='before_decode',detected=detected,completed=rows))
        recovered=prc.Decode(decoding,posterior)
        correct=bool(recovered is not None and np.array_equal(recovered,message))
        rows.append(dict(task=i,detected=detected,decoded_correctly=correct))
        print('PRC_ROUNDTRIP',i,detected,correct,flush=True)
    model_roots={
        'sd15':Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'),
        'clip_vitl14':Path('/data0/liyanlei/genstego_sota_assets/model_weights/mas_grdh/clip/clip-vit-large-patch14-local')}
    model_files={}
    for name,path in model_roots.items():
        model_files[name]=dict(root=str(path),files={str(f.relative_to(path)):dict(bytes=f.stat().st_size,sha256=sha(f))
            for f in path.rglob('*') if f.is_file() and (f.suffix=='.json' or f.name.endswith('.safetensors') and '.fp16.' not in f.name)})
    result=dict(status='complete',protocol_sha256=sha(run/'protocol.json'),prc_matches_sembind_ast=same_ast,
        rows=rows,passed=all(r['detected'] and r['decoded_correctly'] for r in rows),
        dependencies={k:metadata.version(k) for k in ['torch','numpy','scipy','galois','ldpc','numba']},
        original_requirements={'galois':'0.4.1','ldpc':'0.1.51'},model_files=model_files,
        private_key_sha256=sha(private/'prc_key.pkl'),seconds=time.monotonic()-started,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=plan['scope'],
        limitation='Roundtrip success is not dependency equivalence or empirical FPR calibration. Original decode.py uses Detect OR Decode; individual nominal1e-6 thresholds do not automatically prove the union FPR1e-6.')
    write(run/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
