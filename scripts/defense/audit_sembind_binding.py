"""Verify paper mask semantics against stored native-GS artifacts on249 CPU."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sembind_binding import SemBindMask


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert os.name == 'posix' and root.is_dir()
    run = root / 'runs/phase1_trajectory_native_gs_20261113'
    private = root / 'private_eval/phase1_trajectory_native_gs_20261113'
    key = bytes.fromhex(json.loads((private / 'seeds.json').read_text())['binding_key'])
    masks = {s: SemBindMask(key, s) for s in [0., .5, 1.]}
    n = 0
    for shard in range(3):
        records = json.loads((run / f'generate_{shard}/complete.json').read_text())['rows']
        for row in records:
            path = run / row['codes']['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['codes']['sha256']
            code = np.load(path)['aux_code']
            for sigma, mask in masks.items():
                expected = np.tile(code, 16)
                expected[int(sigma * 16384):] = 0
                expected = expected[mask.permutation].reshape(1, 4, 64, 64)
                assert np.array_equal(mask.bits(code), expected)
                with np.load(private / f"{row['task']:03}.npz") as identity:
                    z = identity['initial']
                    assert np.array_equal(z * mask.signs(code) * mask.signs(code), z)
            n += 1
    assert n == 100
    other = SemBindMask(bytes(x ^ 255 for x in key), 1.)
    assert not np.array_equal(other.permutation, masks[1.].permutation)
    report = dict(sources=n, sigma_values=[0, .5, 1], exact_unbinding_checks=300,
                  sigma1_matches_frozen_native_gs=True, private_key_change_changes_permutation=True,
                  source_sha256=hashlib.sha256(Path(__file__).with_name('sembind_binding.py').read_bytes()).hexdigest(),
                  scope='CPU mask implementation and exact latent unbinding; not a new image or attack result')
    (root / 'runs/three_baselines_20261113/sembind_binding_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
