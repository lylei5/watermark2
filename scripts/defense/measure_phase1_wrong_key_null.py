"""Conditional randomized-identity GS null; CPU on249, no attack or model calls."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np
from scipy.stats import beta


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2)+'\n')


def interval(k, n):
    return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    assert str(a.root.resolve()) == '/data0/liyanlei/watermark_research'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(a.root/'envs/phase1_extra'))
    from Crypto.Cipher import ChaCha20
    import torch
    torch.set_num_threads(4)
    run = a.root/'runs/phase1_wrong_key_null_20261113'
    run.mkdir(exist_ok=False)
    private = a.root/'private_eval/phase1_wrong_key_null_20261113'
    private.mkdir(mode=0o700, exist_ok=False)
    upstream = a.root/'runs/phase1_contract_20261113/gs_watermark.py'
    spec = importlib.util.spec_from_file_location('native_gs', upstream)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    native = mod.Gaussian_Shading(1, 8, 1e-6, 1)
    observed = a.root/'runs/continuous_geometry_screen_20261113_v2'
    audited = json.loads((observed/'genuine_verified.json').read_text())
    selected = [r for r in audited['rows'] if r['arm'] == 'full']
    assert len(selected) == 6
    total, per_source, batch = 3_000_000, 500_000, 512
    sources = []
    packed = []
    for row in selected:
        path = observed/row['latent']
        assert sha(path) == row['latent_sha256']
        packed.append(np.packbits(np.load(path).reshape(-1) > 0))
        sources.append(dict(path=str(path), sha256=sha(path), source_task=row['task']))
    seeds = {k: dict(key=secrets.token_bytes(32).hex(), nonce=secrets.token_bytes(12).hex())
             for k in ['identity_key', 'message', 'random_tie']}
    seed_file = private/'audit_rng.json'
    write(seed_file, seeds)
    seed_file.chmod(0o600)
    streams = {k: ChaCha20.new(key=bytes.fromhex(v['key']), nonce=bytes.fromhex(v['nonce']))
               for k,v in seeds.items()}
    plan = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        trials=total, trials_per_source=per_source, batch=batch, sources=sources,
        generator='Separate ChaCha20 streams for fresh OTP identities, messages, and diagnostic fair ties',
        source_sha256={str(upstream):sha(upstream), str(Path(__file__)):sha(__file__)},
        private_rng_sha256=sha(seed_file), message_bits=256, repetitions=64,
        working_points=[166,167,182], original_gs_tau=native.tau_onebit,
        main='Fresh random identity and message for every trial, conditional on six fixed historical DPM image inversions',
        controls='Fixed messages with112/128/144 ones; original tie0 and independent fair-tie diagnostic',
        claim_limit='Not a fixed-user natural-image FPR or fresh image-generation benchmark. Published source remains unchanged.',
        gpu_used=False, model_calls=0, attack_jobs=0)
    write(run/'protocol.json', plan)
    labels=['fresh_message','fixed_ones112','fixed_ones128','fixed_ones144']
    fixed={label:(np.arange(256)<ones)[None,:] for label,ones in zip(labels[1:],[112,128,144])}
    counts={tie:{label:{str(k):0 for k in plan['working_points']} for label in labels}
            for tie in ['original_tie0','fair_tie_diagnostic']}
    tie_count=one_count=fair_one_count=checked=0
    per_image=[]
    started=time.monotonic()
    print('START', total, 'conditional fresh-identity trials; CPU only', flush=True)
    for source_index, signs in enumerate(packed):
        source_hits=0
        for offset in range(0,per_source,batch):
            size=min(batch,per_source-offset)
            key=np.frombuffer(streams['identity_key'].encrypt(bytes(size*2048)),dtype=np.uint8).reshape(size,2048)
            decrypted=np.unpackbits(key^signs[None,:],axis=1).reshape(size,4,8,8,8,8)
            votes=decrypted.sum(axis=(2,4)).reshape(size,256)
            recovered=votes>32
            if source_index==0 and offset==0:
                for j in range(16):
                    tensor=torch.from_numpy(decrypted[j].reshape(1,4,64,64))
                    actual=native.diffusion_inverse(tensor).numpy().astype(bool).reshape(-1)
                    assert np.array_equal(actual,recovered[j])
                    checked+=1
            tie=votes==32
            random_bits=np.unpackbits(np.frombuffer(streams['random_tie'].encrypt(bytes(size*32)),dtype=np.uint8)).reshape(size,256).astype(bool)
            fair=np.where(tie,random_bits,recovered)
            message=np.unpackbits(np.frombuffer(streams['message'].encrypt(bytes(size*32)),dtype=np.uint8)).reshape(size,256).astype(bool)
            tie_count+=int(tie.sum());one_count+=int(recovered.sum());fair_one_count+=int(fair.sum())
            for decoder,bits in [('original_tie0',recovered),('fair_tie_diagnostic',fair)]:
                for label,target in [('fresh_message',message)]+list(fixed.items()):
                    score=(bits==target).sum(axis=1)
                    for k in plan['working_points']:
                        count=int((score>=k).sum())
                        counts[decoder][label][str(k)]+=count
                        if decoder=='original_tie0' and label=='fresh_message' and k==167:
                            source_hits+=count
        per_image.append(dict(source_index=source_index, trials=per_source, fresh_message_hits_at167=source_hits))
    summaries=[]
    for decoder,groups in counts.items():
        for message,points in groups.items():
            for threshold,count in points.items():
                summaries.append(dict(decoder=decoder,message=message,minimum_matching_bits=int(threshold),
                    accepted=count,n=total,rate=count/total,cp95=interval(count,total)))
    result=dict(status='complete',protocol_sha256=sha(run/'protocol.json'),summaries=summaries,
        per_image=per_image,bit_observations=total*256,tie_bits=tie_count,tie_fraction=tie_count/(total*256),
        decoded_one_fraction=one_count/(total*256),fair_decoded_one_fraction=fair_one_count/(total*256),
        original_source_decoder_checks=checked,seconds=time.monotonic()-started,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=plan['claim_limit'],gpu_used=False)
    write(run/'complete.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
