"""Post-hoc descriptive semantic drift; never feeds reference codes to a detector."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    output = run / 'failure_description.json'
    if output.exists():
        raise FileExistsError(output)
    result = json.loads((run / 'scores/results.json').read_text())
    genuine = [r for r in result['rows'] if r['kind'] == 'genuine']
    codes = {}
    for row in genuine:
        paths = list((run / 'cache').glob('shard_*/' + row['id'] + '.npz'))
        assert len(paths) == 1
        assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == row['features_sha256']
        codes[row['id']] = np.load(paths[0], allow_pickle=False)['sembind_code'].reshape(-1)
    clean = {r['task']: codes[r['id']] for r in genuine if r['label'] == 'clean'}
    assert set(clean) == {r['task'] for r in genuine}
    rows = []
    for row in genuine:
        code = codes[row['id']]
        ref = clean[row['task']]
        assert code.shape == ref.shape
        fit = row['detail']['all_fit']
        rows.append(dict(id=row['id'], task=row['task'], condition=row['label'],
            semantic_flips_vs_clean=int(np.count_nonzero(code != ref)), semantic_bits=int(code.size),
            ba=row['ba'], accepted=row['all_fit_accepted'], split_accepted=row['split_fit_accepted'],
            selected_pose=fit['pose'], decisive_groups=fit['n_decisive'], matching_groups=fit['successes'],
            adjusted_p=fit['p_adjusted']))
    groups = []
    for label in sorted({r['condition'] for r in rows}):
        group = [r for r in rows if r['condition'] == label]
        groups.append(dict(condition=label, n=len(group),
            semantic_flip_quantiles=np.quantile([r['semantic_flips_vs_clean'] for r in group], [0, .5, .9, 1]).tolist(),
            failures=[r for r in group if not r['accepted'] or not r['split_accepted']]))
    report = dict(post_hoc=True, thresholds_changed=False, reference_features_used_by_detector=False,
        scope='Descriptive association only; clean code is not the enrollment code and drift is not a causal decomposition',
        quantile_levels=[0, .5, .9, 1], conditions=groups, rows=rows,
        source_results_sha256=hashlib.sha256((run / 'scores/results.json').read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
