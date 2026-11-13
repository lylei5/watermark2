"""Paired source-level bootstrap for frozen legitimate calibration comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT/'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4')
    import numpy as np
    scores = json.loads((run/'scores.json').read_text())
    controls = json.loads((run/'repetition_0/controls.json').read_text())
    rows = [json.loads((run/f'repetition_{r}/rows.json').read_text()) for r in range(3)]
    conditions = list(json.loads((run/'protocol.json').read_text())['conditions'])
    genuine = [r for r in controls if r['split'] == 'evaluation' and r['kind'] in ['clean',*conditions]]
    identities = sorted({r['id'] for r in genuine})
    assert len(identities) == 50 and len(genuine) == 500
    rng = np.random.default_rng(2026111317)
    samples = rng.integers(0,50,size=(10000,50))
    output = []
    for target in [.90,.95]:
        baseline_point = next(p for p in scores['operating_points'] if p['target'] == target and p['method'] == 'sscd_int8_templates')
        reference = {r['id']:np.mean([x['sscd_int8_templates'] <= baseline_point['cutoff']
                     for x in genuine if x['id'] == r['id']]) for r in genuine}
        candidates = [p for p in scores['operating_points'] if p['target'] == target and p['feasible']
                      and (p['method'].startswith('tangent') or p['method'] == 'orbit_margin_fixed')]
        for point in candidates:
            if point['repetition'] is None:
                part = genuine
                decision = lambda r:r[point['method']] <= point['cutoff']
            else:
                part = [r for r in rows[point['repetition']] if r['method'] == point['method']
                        and r['split'] == 'evaluation' and r['kind'] in ['clean',*conditions]]
                decision = lambda r:r['key_recovered'] and r['region_distance'] <= point['cutoff']
            candidate = {identity:np.mean([decision(r)for r in part if r['id'] == identity])for identity in identities}
            delta = np.array([candidate[i]-reference[i]for i in identities])
            interval = np.quantile(delta[samples].mean(axis=1),[.025,.975])
            output.append(dict(target=target,method=point['method'],repetition=point['repetition'],
                difference=float(delta.mean()),percentile95=[float(x)for x in interval],
                source_count=50,queries_per_source=10))
    result = dict(status='complete',bootstrap_seed=2026111317,resamples=10000,
        comparator='SSCD int8 nine-template control at same calibration target',results=output,
        scope='Descriptive source-paired uncertainty on development evaluation split; no multiplicity correction or independent confirmation',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scores_sha256=hashlib.sha256((run/'scores.json').read_bytes()).hexdigest())
    with (run/'source_uncertainty.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
