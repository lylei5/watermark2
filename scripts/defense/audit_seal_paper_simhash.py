"""Check paper SEAL salt behavior and RNG isolation on249, without images."""
import hashlib
import json
import os
from pathlib import Path
import secrets

import torch

from seal_paper_simhash import SealPaperSimHash


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name == 'posix'
    torch.set_num_threads(4)
    vector = torch.linspace(-1, 1, 768)
    key = secrets.token_bytes(32)
    first = SealPaperSimHash(key, device='cpu')
    before = torch.random.get_rng_state().clone()
    initial = first.sample(vector)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(initial, first.sample(vector))
    assert torch.equal(initial, SealPaperSimHash(key, device='cpu').sample(vector))
    second = SealPaperSimHash(secrets.token_bytes(32), device='cpu')
    changed = second.sample(vector)
    assert not torch.equal(initial, changed)
    assert not torch.equal(first._projections, second._projections)
    assert initial.shape == (1, 4, 64, 64) and initial.isfinite().all()
    try:
        SealPaperSimHash(b'', device='cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Missing salt must be rejected')
    record = dict(repeated_noise_exact=True, fresh_instance_replay_exact=True,
                  global_rng_unchanged=True, salt_changes_projections_and_noise=True,
                  missing_salt_rejected=True, shape=[1, 4, 64, 64], finite=True,
                  scope='CPU component verification of final Algorithm2; no model, image, defense-performance or attack result',
                  source_sha256=hashlib.sha256(Path(__file__).with_name('seal_paper_simhash.py').read_bytes()).hexdigest())
    path = root / 'runs/three_baselines_20261113/seal_paper_simhash_checks.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
