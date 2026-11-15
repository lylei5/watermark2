"""Read completed public artifacts and summarize recovery mechanisms on249."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run, ablation):
    complete = read(run / 'complete.json')
    assert sha(run / 'rows.json') == complete['artifacts']['rows.json']
    rows = read(run / 'rows.json')
    plan = read(run / 'analysis_plan.json')
    primary = [r for r in rows if r['method'] == plan['primary_candidate']]
    diagnostic = {}
    for kind in ['brightness_factor', 'gaussian_std']:
        selected = [r for r in primary if r['kind'] == kind]
        extra = [r for r in selected if r['returned'] and not r['forward_key']]
        diagnostic[kind] = dict(
            order_recovered=sum(r['returned'] for r in selected),
            forward_recovered=sum(r['forward_key'] for r in selected),
            order_only_successes=len(extra),
            forward_failure_reasons_for_order_only_successes=dict(Counter(r['forward_detail']['reason'] for r in extra)),
            order_failures=[dict(task=r['task'], reason=r['detail']['reason'],
                                 observable_coordinates=r['detail']['observable_coordinates'],
                                 unique_query_levels=r['detail']['unique_query_levels'])
                            for r in selected if not r['returned']])
    assert diagnostic == read(run / 'mechanism_diagnostic.json')
    methods = []
    for name in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == name]
        methods.append(dict(method=name,
                            known_genuine=sum(r['returned'] for r in selected if r['is_genuine'] and not r['kind'].startswith('waves_')),
                            unseen_genuine=sum(r['returned'] for r in selected if r['is_unseen']),
                            same_prompt=sum(r['returned'] for r in selected if r['kind'] == 'same'),
                            median_seconds=statistics.median(r['seconds'] for r in selected)))
    strength_rows = [r for r in read(run / 'comparison95.json') if r['category'].startswith('waves_')]
    matched = [r for r in read(run / 'family_matched.json') if r['genuine_category'] == 'known_genuine' and r['negative_kind'] == 'same']
    helpers = [r['helper_bytes'] for r in read(run / 'enrollment.json') if r['alpha'] == .05]
    templates = [r['bytes'] for r in read(run / 'templates.json')]
    ablation_complete = read(ablation / 'complete.json')
    assert sha(ablation / 'rows.json') == ablation_complete['artifacts']['rows.json']
    ablation_rows = read(ablation / 'rows.json')
    retained = {(r['task'], r['kind']): r for r in ablation_rows if not r['drop_white']}
    paired_changes = []
    for r in ablation_rows:
        if r['drop_white']:
            other = retained[r['task'], r['kind']]
            assert r['pixels_sha256'] == other['pixels_sha256']
            if r['returned'] != other['returned']:
                paired_changes.append(dict(task=r['task'], kind=r['kind']))
    output = dict(
        status='complete', source_sha256=sha(Path(__file__)),
        inputs={str(p): sha(p) for p in [run / 'complete.json', run / 'comparison95.json', run / 'family_matched.json', ablation / 'complete.json']},
        mechanism=diagnostic, methods=methods, by_strength=strength_rows,
        matched_known_genuine=matched,
        storage=dict(helper_json_bytes_min=min(helpers), helper_json_bytes_max=max(helpers),
                     nine_template_float32_bytes=sorted(set(templates)),
                     direct_order_reference_uint8_bytes=64 * 64 * 3,
                     direct_order_reference_float32_bytes=64 * 64 * 3 * 4,
                     note='Representation bytes only. Model weights, common secrets, metadata, and transient solver bases are separate; no matched latency benchmark.'),
        saturation_ablation_decision_changes=paired_changes,
        primary_successful_returns=sum(r['returned'] for r in primary),
        primary_rounded_returns=sum(r['rounded_key'] for r in primary))
    destination = run / 'mechanism_summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    manifest_path = run / 'public_artifact_manifest.json'
    manifest = read(manifest_path)
    for name in ['mechanism_diagnostic.json', 'mechanism_summary.json']:
        manifest[name] = sha(run / name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['inputs', 'mechanism', 'by_strength']}, indent=2))
    print(json.dumps(strength_rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ablation', type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.ablation)
