"""Recount frozen PGID development decisions on249; no new model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT / 'runs'
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    sources = {'TR': 'pgid_carrier_genuine_20261113_v3',
               'PRC': 'pgid_carrier_genuine_20261113_v2'}
    results = {}
    for carrier, name in sources.items():
        run = ROOT / 'runs' / name
        directory = run / carrier
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['status'] == 'complete'
        assert sha(directory / 'rows.json') == complete['rows_sha256']
        assert sha(run / 'protocol.json') == complete['protocol_sha256']
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(run / 'pgid_carrier_genuine_controls.py') == plan['source_sha256']
        parent = Path(plan['parent'])
        assert sha(parent / 'protocol.json') == plan['parent_protocol_sha256']
        parent_plan = json.loads((parent / 'protocol.json').read_text())
        records = json.loads((directory / 'rows.json').read_text())
        assert len(records) == 220
        pairs = {}
        for row in records:
            key = row['task'], row['kind']
            by_method = pairs.setdefault(key, {})
            assert row['defended'] not in by_method
            by_method[row['defended']] = row
        assert len(pairs) == 110
        for pair in pairs.values():
            assert set(pair) == {False, True}
            assert pair[False]['pixels_sha256'] == pair[True]['pixels_sha256']
        source_ids = sorted({row['task'] for row in records})
        assert len(source_ids) == 10
        conditions = ['clean', *plan['conditions'], 'cover']
        assert len(conditions) == 11
        summary = []
        for name in [*conditions, 'all_edits', 'clean_plus_edits']:
            for defended in [False, True]:
                kinds = (set(plan['conditions']) if name == 'all_edits' else
                         {'clean', *plan['conditions']} if name == 'clean_plus_edits' else {name})
                rows = [row for row in records if row['defended'] == defended and row['kind'] in kinds]
                summary.append(dict(condition=name, detector='PGID-F' if defended else 'native',
                                    accepted=sum(row['accepted'] for row in rows),
                                    query_count=len(rows), distinct_sources=len({row['task'] for row in rows})))
        results[carrier] = dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'),
            rows_sha256=sha(directory / 'rows.json'), complete_sha256=sha(directory / 'complete.json'),
            decision_records=len(records), paired_pixel_queries=len(pairs),
            independent_source_count=len(source_ids), per_source_genuine_queries=10,
            scheduler=parent_plan['target_schedulers'][carrier],
            pgid=dict(k=parent_plan['defense']['k'][carrier], s=parent_plan['defense']['s'],
                      gamma=parent_plan['defense']['gamma'], steps=parent_plan['defense']['steps']),
            published_threshold=parent_plan['thresholds'][carrier], conditions=summary)
    receipt = dict(status='complete', worker_sha256=sha(__file__),
        scope='Historical artifact/denominator recount only; no model re-execution, new attack or calibrated FPR',
        decision_records=sum(result['decision_records'] for result in results.values()),
        paired_pixel_queries=sum(result['paired_pixel_queries'] for result in results.values()),
        reused_development_sources_per_carrier=10, results=results,
        inference_limit='Pooled clean-plus-edit queries share source images; do not use a binomial interval with n=100 as if independent')
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
