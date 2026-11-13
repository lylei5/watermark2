"""Audit split isolation, nuisance generalization and BCH parameters on 249."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir()
    run = root / 'runs/instance_nuisance_geometry_20261113'
    parent = root / 'runs/instance_representation_20261113_v2'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    from scipy.stats import beta
    plan = json.loads((run / 'protocol.json').read_text())
    done = json.loads((run / 'complete.json').read_text())
    assert done['status'] == 'complete'
    assert sha(run / 'protocol.json') == done['protocol_sha256']
    assert sha(run / 'rows.json') == done['rows_sha256']
    assert sha(run / 'basis.npz') == done['basis_sha256']
    splits = [set(plan[name]) for name in ['fit', 'calibration', 'evaluation']]
    assert list(map(len, splits)) == [24, 26, 50]
    assert len(set.union(*splits)) == sum(map(len, splits)) == 100
    assert sha(parent / 'sscd/features.npz') == plan['features_sha256']
    assert sha(parent / 'sscd/metadata.json') == plan['metadata_sha256']
    features = np.load(parent / 'sscd/features.npz')['features'].astype(np.float64)
    index = {tuple(k): i for i, k in enumerate(json.loads((parent / 'sscd/metadata.json').read_text())['keys'])}
    basis = np.load(run / 'basis.npz')['basis']
    energy = []
    for split in ['fit', 'calibration', 'evaluation']:
        residuals = np.stack([features[index[i, c]]-features[index[i, 'bound']]
                             for i in plan[split] for c in plan['conditions']])
        for rank in plan['ranks']:
            energy.append(dict(split=split, rank=rank,
                residual_energy_removed=float(np.sum((residuals @ basis[:rank].T)**2)/np.sum(residuals**2))))
    bounds = []
    for item in done['results']:
        t = item['threshold']
        if item['metric'] == 'margin_hamming1024':
            roots = set()
            for i in range(1, 2*t+1):
                j = i % 1023
                while j not in roots:
                    roots.add(j)
                    j = 2*j % 1023
            bounds.append(dict(rank=item['rank'], repetition=item['repetition'], t=t, n=1023,
                               designed_distance=2*t+1, dimension=1023-len(roots)))
    result = dict(source_sha256=sha(__file__), completed_sha256=sha(run / 'complete.json'),
        all_splits_disjoint=True, rows=len(json.loads((run / 'rows.json').read_text())), energy=energy,
        bch_dimensions=bounds, zero_of_50_clopper_pearson_95=[0., float(beta.ppf(.975, 1, 50))],
        caveat='Descriptive development configurations; no independent FPR measurement, BCH decoder or key recovery')
    with (run / 'audit.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
