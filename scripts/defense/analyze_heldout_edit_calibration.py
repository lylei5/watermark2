"""Development comparison with each unregistered edit omitted from calibration."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--copy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert ROOT.is_dir()
    assert all(x.resolve().is_relative_to(ROOT / 'runs') for x in [a.parent, a.copy, a.output])
    assert not a.output.exists()
    for folder, name in [(a.parent, 'complete.json'), (a.parent, 'audit.json'), (a.copy, 'summary.json')]:
        assert read(folder / name)['status'] == 'complete'
    manifest = read(a.parent / 'complete.json')
    for name in ['controls.json', 'rows.json']:
        assert sha(a.parent / name) == manifest['artifacts'][name]
    assert sha(a.copy / 'rows.json') == read(a.copy / 'summary.json')['artifacts']['rows.json']
    plan = read(a.parent / 'protocol.json')
    controls, recoveries = read(a.parent / 'controls.json'), read(a.parent / 'rows.json')
    copy = {(r['id'], r['query_id'], r['kind']): r for r in read(a.copy / 'rows.json')}
    for r in controls:
        c = copy[r['id'], r['query_id'], r['kind']]
        assert c['pixels_sha256'] == r['pixels_sha256'] and c['native_accepted'] == r['native_accepted']
        r['copynce_descriptor'], r['copynce_matcher'] = c['descriptor_distance'], c['matcher_distance']
    unknown = [k for k in plan['conditions'] if k not in plan['enrollment_families']]
    assert len(unknown) == 5
    continuous = [m for m in plan['methods'] if m != 'permuted_phase'] + [
        'sscd_templates', 'sscd_int8_templates', 'copynce_descriptor', 'copynce_matcher']
    groups = [(m, None, controls) for m in continuous]
    for method in plan['methods']:
        if method == 'permuted_phase':
            continue
        for rep in range(3):
            groups.append((method, rep, [r for r in recoveries if r['method'] == method and r['repetition'] == rep]))
    rows = []
    for heldout in unknown:
        for method, rep, data in groups:
            cal = [r for r in data if r['split'] == 'calibration' and r['kind'] in ['clean', *plan['conditions']] and r['kind'] != heldout]
            assert len(cal) == 27 and all(r['native_accepted'] for r in cal)
            values = sorted(r[method] if rep is None else r['region_distance'] if r['key_recovered'] else math.inf for r in cal)
            for target in [.9, .95]:
                threshold = values[math.ceil(target * len(cal)) - 1]
                feasible = math.isfinite(threshold)
                for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
                    carrier_rows = [r for r in data if r['carrier'] == carrier and r['split'] == split]
                    part = [r for r in carrier_rows if r['kind'] == heldout]
                    assert len(part) == (3 if carrier == 'GS' else 10)
                    def accept(r):
                        return r[method] <= threshold if rep is None else r['key_recovered'] and r['region_distance'] <= threshold
                    negatives = []
                    for kind in ['same_prompt', 'wrong_record', 'cover', 'imprint_150']:
                        negative = [r for r in carrier_rows if r['kind'] == kind]
                        if negative:
                            negatives.append(dict(kind=kind, total=len(negative),
                                accepted=sum(accept(r) for r in negative) if feasible else None,
                                joint=sum(accept(r) and r['native_accepted'] for r in negative) if feasible else None))
                    rows.append(dict(heldout=heldout, method=method, repetition=rep, target=target,
                        carrier=carrier, calibration_count=27, threshold=threshold if feasible else None,
                        feasible=feasible, total=len(part), native=sum(r['native_accepted'] for r in part),
                        accepted=sum(accept(r) for r in part) if feasible else None,
                        joint=sum(accept(r) and r['native_accepted'] for r in part) if feasible else None,
                        negatives=negatives))
    aggregated = []
    for method, rep, _ in groups:
        for target in [.9, .95]:
            for carrier in ['GS', 'TR', 'PRC']:
                rr = [r for r in rows if r['method'] == method and r['repetition'] == rep and r['target'] == target and r['carrier'] == carrier]
                assert len(rr) == 5
                feasible = all(r['feasible'] for r in rr)
                aggregated.append(dict(method=method, repetition=rep, target=target, carrier=carrier,
                    feasible_families=sum(r['feasible'] for r in rr), total_families=5,
                    total=sum(r['total'] for r in rr), native=sum(r['native'] for r in rr),
                    accepted=sum(r['accepted'] for r in rr) if feasible else None,
                    joint=sum(r['joint'] for r in rr) if feasible else None))
    result = dict(status='complete', source_sha256=sha(Path(__file__)),
        inputs={str(x): sha(x) for x in [a.parent / 'protocol.json', a.parent / 'controls.json', a.parent / 'rows.json', a.copy / 'rows.json']},
        scope='Post-hoc development analysis: each of five non-enrollment edit families is excluded from GS threshold calibration in turn; representations and actual decode outcomes remain fixed',
        pretrained_augmentation_exclusion_claimed=False, independent_confirmation=False,
        rows=rows, aggregated=aggregated)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', rows=len(rows), aggregates=len(aggregated))), flush=True)


if __name__ == '__main__':
    main()
