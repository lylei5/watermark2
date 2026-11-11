"""Evaluate frozen detectors on all predeclared natural, unwatermarked queries."""
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import time

import numpy as np
import torch
from blind_geometry_verifier import ALPHA, geometry_bank, query_observations, verify
from codeword_binding import sembind_mask_bits, bit_accuracy
from score_frozen_geometry import bind_frozen_alpha
from sembind_official_audit import carrier, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    plan = json.loads((run / 'protocol.json').read_text())
    assert plan['evaluation_split'] == 'unwatermarked'
    assert not (run / 'scores/results.json').exists()
    torch.set_num_threads(4)
    for name, digest in plan['source_sha256'].items():
        assert sha(run / 'scripts' / name) == digest
    records = []
    for shard in range(3):
        folder = run / 'cache' / f'shard_{shard}'
        part = json.loads((folder / 'complete.json').read_text())
        assert part['completed'] and part['steps'] == 50 and part['protocol_sha256'] == sha(run / 'protocol.json')
        records.extend(dict(r, cache=folder) for r in part['rows'] if r['split'] == 'unwatermarked')
    assert len(records) == len(plan['rows']) == 100
    assert {r['id'] for r in records} == {r['id'] for r in plan['rows']}
    identities = {task: carrier(Path(plan['identity_root']), task) for task in plan['identity_tasks']}
    poses, grid, support, counts, folds = geometry_bank()
    alpha = bind_frozen_alpha(plan['nominal_alpha'], ALPHA)
    assert len(poses) == 251
    output = []
    started = time.monotonic()
    for row in records:
        folder = row['cache']
        assert sha(run / row['image']) == row['image_sha256']
        for suffix, key in [('.npy', 'inversion_sha256'), ('.npz', 'features_sha256')]:
            assert sha(folder / (row['id'] + suffix)) == row[key]
        latent = np.load(folder / (row['id'] + '.npy'), allow_pickle=False)
        code = np.load(folder / (row['id'] + '.npz'), allow_pickle=False)['sembind_code']
        signs, valid = query_observations(latent, grid, support)
        for task, (_, cipher, message, perm) in identities.items():
            mask = sembind_mask_bits(code[None], perm)[0]
            stream = cipher ^ np.tile(message, 64)
            ba = float(bit_accuracy((latent.reshape(-1) > 0).astype(np.uint8) ^ mask ^ stream, message)[0])
            value = verify(signs, valid, counts, cipher ^ mask, folds, poses)
            output.append(dict(id=row['id'], claimed_identity=task, kind='unwatermarked_natural', ba=ba,
                default_accepted=ba > 181/256, identity_accepted=value['identity_accepted'],
                identity_penalty_accepted=min(1, 251 * value['identity']['p']) <= alpha,
                all_fit_accepted=value['all_fit']['accepted'], split_fit_accepted=value['split_fit']['accepted'],
                all_fit_p=value['all_fit']['p_adjusted'], split_fit_p=value['split_fit']['p_adjusted']))
    assert len(output) == 6400
    keys = ['default_accepted', 'identity_accepted', 'identity_penalty_accepted', 'all_fit_accepted', 'split_fit_accepted']
    summary = {key: dict(accepted=sum(r[key] for r in output), n=len(output),
                        any_identity_accepted_queries=len({r['id'] for r in output if r[key]}), queries=100)
               for key in keys}
    controls = [dict(target_calibration_tpr=c['target_absolute_tpr'], threshold=c['threshold'], operator='>=',
                     accepted=sum(r['ba'] >= c['threshold'] for r in output), n=len(output),
                     any_identity_accepted_queries=len({r['id'] for r in output if r['ba'] >= c['threshold']}), queries=100)
                for c in plan['supplementary_ba_controls']]
    result = dict(completed=True, queries=100, identity_decisions=6400, summary=summary,
                  supplementary_ba_controls=controls, rows=output, seconds=time.monotonic()-started,
                  protocol_sha256=sha(run / 'protocol.json'), thresholds_changed=False, new_attacks_run=False,
                  scope='Existing COCO development assets; correlated identities; no independent confirmation or ultra-low FPR claim')
    write(run / 'scores/results.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
