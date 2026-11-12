"""Read-only continuous-feature registration diagnostic on consumed records.

No image generation, new attack, threshold fit, or changed authentication decision.
Run on server249 only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_feature(path):
    with np.load(path) as data:
        value = data['logits'].astype(np.float64).reshape(-1)
        assert value.shape == (1024,) and np.isfinite(value).all()
        assert np.array_equal(value > 0, data['code'].astype(bool))
    length = np.linalg.norm(value)
    assert length > 0
    return value / length


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifests = {phase: a.run / f'{phase}_verified.json' for phase in ['genuine', 'forgery']}
    data = {phase: json.loads(path.read_text()) for phase, path in manifests.items()}
    assert all(d['audit_passed'] for d in data.values())
    assert len(data['genuine']['rows']) == 120 and len(data['forgery']['rows']) == 216
    auxiliary = {t: a.run / f'genuine_{t % 3}/codes/{t:03}_aux.npz' for t in range(12)}
    enrolled = {t: unit_feature(path) for t, path in auxiliary.items()}
    clean = {r['task']: unit_feature(a.run / r['code']) for r in data['genuine']['rows'] if r['condition'] == 'clean'}
    rows = []
    for phase, d in data.items():
        for r in d['rows']:
            path = a.run / r['code']
            assert sha(path) == r['code_sha256']
            q = unit_feature(path)
            row = dict(id=r['id'], task=r['task'], phase=phase, kind=r.get('kind', 'genuine'),
                       condition=r.get('condition'), update=r.get('update'),
                       original_ba=r['ba'], original_accepted=r['accepted'])
            for name, reference in [('auxiliary', enrolled[r['task']]), ('delivered_reference', clean[r['task']])]:
                row[name + '_cosine'] = float(q @ reference)
                row[name + '_distance'] = float(np.linalg.norm(q - reference))
            rows.append(row)
    genuine = [r for r in rows if r['phase'] == 'genuine' and r['original_accepted']]
    endpoints = [r for r in rows if r['kind'] == 'candidate' and r['update'] == 151 and r['original_accepted']]
    overlap = {}
    for name in ['auxiliary', 'delivered_reference']:
        key = name + '_distance'
        overlap[name] = [dict(imprint_id=r['id'], distance=r[key],
            accepted_genuine_at_least_as_distant=[dict(id=q['id'], distance=q[key]) for q in genuine if q[key] >= r[key]]) for r in endpoints]
    result = dict(completed=True, rows=rows, overlap=overlap, script_sha256=sha(__file__),
        protocol_sha256=sha(a.run / 'protocol.json'),
        input_manifest_sha256={phase: sha(path) for phase, path in manifests.items()},
        auxiliary_sha256={str(t): sha(path) for t, path in auxiliary.items()},
        scope='All336 consumed records; original decisions retained; unit-normalized released hash-head logits; no threshold selection or attack changes.',
        limitation='A distance-only monotone acceptance rule cannot remove these overlaps. This does not determine the outcome of a new embedding or feature extractor.')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(records=len(rows), overlap=overlap), indent=2))


if __name__ == '__main__':
    main()
