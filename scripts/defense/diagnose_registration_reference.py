"""Compare auxiliary and delivered-image registration on consumed artifacts.

This is an oracle diagnostic: delivered reference codes are extra verifier state.
It does not choose thresholds, change decisions, or produce attack images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    g = json.loads((a.run / 'genuine_verified.json').read_text())
    f = json.loads((a.run / 'forgery_verified.json').read_text())
    assert g['audit_passed'] and f['audit_passed']
    assert len(g['rows']) == 120 and len(f['rows']) == 216
    refs = {r['task']: r for r in g['rows'] if r['condition'] == 'clean'}
    assert len(refs) == 12
    reference_codes = {}
    for t, r in refs.items():
        assert sha(a.run / r['code']) == r['code_sha256']
        reference_codes[t] = np.load(a.run / r['code'])['code']
    aux_paths = {t: a.run / f'genuine_{t % 3}/codes/{t:03}_aux.npz' for t in refs}
    auxiliary_codes = {t: np.load(path)['code'] for t, path in aux_paths.items()}
    rows = []
    for phase, data in [('genuine', g), ('forgery', f)]:
        for r in data['rows']:
            assert sha(a.run / r['code']) == r['code_sha256']
            q = np.load(a.run / r['code'])['code']
            t = r['task']
            assert q.shape == (1024,)
            rows.append(dict(id=r['id'], task=t, phase=phase, kind=r.get('kind', 'genuine'),
                update=r.get('update'), condition=r.get('condition'), original_ba=r['ba'],
                original_accepted=r['accepted'],
                auxiliary_hamming=float(np.mean(q != auxiliary_codes[t])),
                reference_hamming=float(np.mean(q != reference_codes[t]))))
    groups = dict(genuine=[r for r in rows if r['phase'] == 'genuine'],
        accepted_genuine=[r for r in rows if r['phase'] == 'genuine' and r['original_accepted']],
        final_imprint=[r for r in rows if r['kind'] == 'candidate' and r['update'] == 151],
        accepted_final_imprint=[r for r in rows if r['kind'] == 'candidate' and r['update'] == 151 and r['original_accepted']],
        accepted_checkpoints=[r for r in rows if r['kind'] == 'candidate' and r['original_accepted']])
    summary = {}
    for name, rr in groups.items():
        summary[name] = dict(n=len(rr))
        for key in ['auxiliary_hamming', 'reference_hamming']:
            summary[name][key] = dict(min=min(r[key] for r in rr), max=max(r[key] for r in rr),
                mean=float(np.mean([r[key] for r in rr]))) if rr else None
    overlap = {}
    for key in ['auxiliary_hamming', 'reference_hamming']:
        endpoints = groups['accepted_final_imprint']
        overlap[key] = [dict(imprint_id=r['id'], distance=r[key],
            accepted_genuine_at_least_as_distant=[q['id'] for q in groups['accepted_genuine'] if q[key] >= r[key]])
            for r in endpoints]
    result = dict(completed=True, rows=rows, summary=summary, overlap=overlap,
        script_sha256=sha(__file__), protocol_sha256=sha(a.run / 'protocol.json'),
        genuine_manifest_sha256=sha(a.run / 'genuine_verified.json'),
        forgery_manifest_sha256=sha(a.run / 'forgery_verified.json'),
        auxiliary_code_sha256={str(t): sha(path) for t, path in aux_paths.items()},
        scope='Read-only CPU diagnostic of all336 consumed observations; original BA decisions unchanged, no threshold selected, no attacks generated or modified.',
        limitation='Reference-code availability changes verifier state and is an oracle intervention here. Observed separation on consumed sources is not an implementable or independently validated defense, and does not prove causal uniqueness.')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, overlap=overlap), indent=2))


if __name__ == '__main__':
    main()
