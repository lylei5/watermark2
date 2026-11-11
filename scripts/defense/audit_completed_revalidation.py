"""Recompute completed research scores from saved latents on server CPU."""
import argparse
from collections import defaultdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import statistics
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    parent = root / 'runs/defense_revalidation_20261111'
    run = root / 'runs/pgid_published_revalidation_20261112'
    plan = json.loads((run / 'protocol.json').read_text())
    assert sha(parent / 'protocol.json') == plan['parent_protocol_sha256']
    sys.path.insert(0, str(parent / 'PGID'))
    from watermarks.watermark_utils import SemanticWatermarks
    rows = []
    for shard in [0, 1]:
        d = json.loads((run / f'shard_{shard}/complete.json').read_text())
        assert d['status'] == 'complete' and d['protocol_sha256'] == sha(run / 'protocol.json')
        rows.extend(d['rows'])
    expected = {(i, c['name'], d) for i in range(10) for c in plan['conditions'] for d in [False, True]}
    assert len(rows) == 180 and {(r['index'], r['condition'], r['defended']) for r in rows} == expected
    for wm, idx in SemanticWatermarks.GS.value.generate_watermark_signals(total_num_signals=10,
            latent_shape=(1, 4, 64, 64), batch_size=1, w_seed=1, device='cpu',
            message_width_in_bytes=32, num_replications=64, l=1):
        for row in [r for r in rows if r['index'] == idx]:
            path = run / row['latent']
            assert sha(path) == row['latent_sha256']
            z = np.load(path)['z']
            ba = float(wm.get_accuracies(torch.from_numpy(z))['bit_accuracies'][0])
            assert ba == row['ba'] and (ba > 181 / 256) == row['accepted']
    summary = []
    for c in plan['conditions']:
        pair = {d: [r for r in rows if r['condition'] == c['name'] and r['defended'] == d] for d in [False, True]}
        before = {r['index']: r for r in pair[False]}
        after = {r['index']: r for r in pair[True]}
        summary.append(dict(condition=c['name'], n=10, standard_accepted=sum(r['accepted'] for r in before.values()),
                     pgid_accepted=sum(r['accepted'] for r in after.values()),
                     lost=sum(before[i]['accepted'] and not after[i]['accepted'] for i in before),
                     restored=sum(not before[i]['accepted'] and after[i]['accepted'] for i in before),
                     standard_mean_ba=statistics.mean(r['ba'] for r in before.values()),
                     pgid_mean_ba=statistics.mean(r['ba'] for r in after.values())))
    result = dict(passed=True, all_180_scores_recomputed_from_latents=True, protocol_sha256=sha(run/'protocol.json'),
                  summary=summary, rows=rows, standard_accepted=sum(r['standard_accepted'] for r in summary),
                  pgid_accepted=sum(r['pgid_accepted'] for r in summary), n=90, unique_sources=10)
    (run / 'verified_results.json').write_text(json.dumps(result, indent=2) + '\n')

    chain = root / 'runs/sembind_chain_revalidation_20261111'
    paths = json.loads((chain / 'paths.json').read_text())
    output = Path(paths['output'])
    result = json.loads((output / 'results.json').read_text())
    for row in result['rows']:
        i, method, setting = row['task'], row['method'], row['setting']
        secret = json.loads((Path(paths['root']) / 'private' / f'{i:03}.json').read_text())
        key, nonce, msg, bkey = [bytes.fromhex(secret[k]) for k in ['key','nonce','message','binding_key']]
        cipher = np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64), dtype=np.uint8))
        message = np.unpackbits(np.frombuffer(msg, dtype=np.uint8))
        stream = cipher ^ np.tile(message, 64)
        tags = [hmac.new(bkey, b'sembind/permutation/' + j.to_bytes(4,'big'), hashlib.sha256).digest() for j in range(16384)]
        perm = np.asarray(sorted(range(16384), key=tags.__getitem__))
        zpath = output / 'inversions' / f'{i:03}_{method}_{setting}.npy'
        z = np.load(zpath); meta = json.loads(zpath.with_suffix('.json').read_text())
        assert sha(zpath) == meta['inversion_sha256']
        assert sha(output/'images'/f'{i:03}_{method}.png') == meta['image_sha256']
        signs = (z.reshape(-1)>0).astype(np.uint8)
        for label, codepath in [('auxiliary',output/'codes'/f'{i:03}_aux.npz'),('final',output/'codes'/f'{i:03}_{method}.npz')]:
            code = np.load(codepath)['sembind_code']
            mask = np.tile(code,16)[perm] if method == 'sembind' else np.zeros(16384,dtype=np.uint8)
            recovered = ((signs ^ mask ^ stream).reshape(64,256).sum(axis=0)>32).astype(np.uint8)
            ba = float(np.mean(recovered == message))
            assert ba == row[label+'_ba'] and (ba>181/256) == row[label+'_accepted']
    groups=[]
    for method in ['sembind','gs']:
        for setting in ['blind_encoder','blind_decoder','oracle_decoder']:
            g=[r for r in result['rows'] if r['method']==method and r['setting']==setting]
            groups.append(dict(method=method,setting=setting,n=len(g),accepted=sum(r['final_accepted'] for r in g),
                               mean_ba=statistics.mean(r['final_ba'] for r in g),mean_sign_error=statistics.mean(r['channel_sign_error'] for r in g)))
    audit=dict(passed=True,records=18,all_message_scores_recomputed_from_latents=True,summary=groups,
               results_sha256=sha(output/'results.json'),results=result)
    (chain/'verified_results.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'pgid_standard':90,'pgid_defended':sum(r['pgid_accepted'] for r in summary),'chain':groups}))


if __name__ == '__main__':
    main()
