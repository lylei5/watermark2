"""Descriptive radius overlap on frozen codes; no extractor or attack execution."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    expected = Path('/data0/liyanlei/watermark_research/runs')
    assert run.parent == expected and output.parent == expected
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from scipy.stats import beta

    summary_path = run / 'trajectory_summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete'
    assert summary['protocol_sha256'] == sha(run / 'protocol.json')
    output.mkdir(exist_ok=False)
    protocol = dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(run), source_summary_sha256=sha(summary_path),
        script_sha256=sha(__file__), bits=1024,
        fixed_radii=[51, 102, 154, 195, 205, 256, 307],
        empirical_coverage_targets=[.95, 1.0],
        scope='Post-hoc descriptive overlap on an already inspected cohort; '
              'no independent calibration, no fuzzy extractor, no key derivation, '
              'no newly generated images and no attacks',
        interpretation='Same-prompt independent images are an instance-separation '
                       'control, not measured forgeries; aux/bound codes come from '
                       'SemBind sign modulation, not the proposed key-derived carrier',
        uncertainty='Pointwise Clopper-Pearson intervals at fixed radii only; '
                    'not simultaneous bands. Data-selected coverage radii have '
                    'descriptive counts only and no validation confidence claim.'
    )
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    arrays = {}
    for shard in range(3):
        record = json.loads((run / f'generate_{shard}/complete.json').read_text())
        assert record['status'] == 'complete'
        assert record['protocol_sha256'] == summary['protocol_sha256']
        for row in record['rows']:
            path = run / row['codes']['path']
            assert sha(path) == row['codes']['sha256']
            with np.load(path) as data:
                arrays[row['task']] = {k: data[k] for k in data.files}
    assert sorted(arrays) == list(range(100))
    rows = []
    for row in summary['rows']:
        data = arrays[row['task']]
        distances = {}
        for name, target in [('aux_final', 'bound_code'), ('same_prompt', 'same_code'),
                             ('jpeg_aux', 'jpeg_aux_code')]:
            count = int(np.count_nonzero(data['aux_code'] != data[target]))
            assert count == round((1 - row['controls'][name]) * 1024)
            distances[name] = count
        distances['jpeg_bound'] = int(np.count_nonzero(data['bound_code'] != data['jpeg_bound_code']))
        rows.append(dict(task=row['task'], dataset=row['dataset'], distances=distances))

    def stats(values, radius, interval):
        n = len(values)
        k = sum(v <= radius for v in values)
        result = dict(accepted=k, n=n, rate=k/n)
        if interval:
            result['pointwise_cp95'] = [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                                       1. if k == n else float(beta.ppf(.975, k+1, n-k))]
        return result

    fixed, selected = [], []
    for group in ['COCO', 'SDP', 'ALL']:
        subset = [r for r in rows if group == 'ALL' or r['dataset'] == group]
        values = {name: [r['distances'][name] for r in subset] for name in rows[0]['distances']}
        for radius in protocol['fixed_radii']:
            fixed.append(dict(dataset=group, radius=radius,
                              counts={k: stats(v, radius, True) for k, v in values.items()}))
        for coverage in protocol['empirical_coverage_targets']:
            radius = sorted(values['aux_final'])[math.ceil(coverage*len(subset))-1]
            selected.append(dict(dataset=group, target_coverage=coverage, radius=radius,
                                 normalized_radius=radius/1024,
                                 counts={k: stats(v, radius, False) for k, v in values.items()}))
    result = dict(status='complete', protocol_sha256=sha(output / 'protocol.json'),
                  completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  rows=rows, fixed_radii=fixed, selected_radii=selected,
                  scope=protocol['scope'], interpretation=protocol['interpretation'])
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='complete', selected_radii=selected)))


if __name__ == '__main__':
    main()
