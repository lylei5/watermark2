"""Check completed recovery decisions, candidate budgets and paired controls on249."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/reprompt_plus_recovery_20261115'
WITNESS = ROOT / 'runs/observable_evidence_recovery_20261115'
METHODS = ['coarse', 'any_scale', 'all_scales', 'observable_scales']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    done = read(RUN / 'complete.json')
    assert done['status'] == 'complete'
    for name, digest in done['artifacts'].items():
        assert sha(RUN / name) == digest
    rows = read(RUN / 'rows.json')
    controls = read(RUN / 'controls.json')
    matched = read(RUN / 'matched.json')
    by_trial = defaultdict(dict)
    for row in rows:
        assert row['method'] not in by_trial[row['trial']]
        by_trial[row['trial']][row['method']] = row
        assert row['joint'] == (row['key_recovered'] and row['native_accepted'])
    assert len(rows) == 960 and len(by_trial) == 240
    assert sum(r['key_recovered'] for r in rows) == done['exact_successes']
    assert sum(any(r['key_recovered'] for r in group.values()) for group in by_trial.values()) == done['restored_native_calls']
    budgets = Counter()
    equivalence = 0
    certificate_checks = 0
    for trial, group in by_trial.items():
        assert set(group) == set(METHODS)
        assert len({r['pixels_sha256'] for r in group.values()}) == 1
        a = group['coarse']
        budgets[a['id'], a['proxy'], a['carrier']] += 1
        equivalence += all(group['all_scales'][k] == group['observable_scales'][k] for k in ['key_recovered', 'joint'])
        assert not group['all_scales']['key_recovered'] or group['observable_scales']['key_recovered']
        assert not group['observable_scales']['key_recovered'] or group['coarse']['key_recovered']
        assert not group['coarse']['key_recovered'] or group['any_scale']['key_recovered']
        for row in group.values():
            for band in row['detail']['bands']:
                if band['reason'] == 'recovered':
                    assert band['code_budget'] == 2 * band['symbol_errors'] + band['erasures']
                    assert band['code_budget'] <= 239
                    assert band['observed_symbols'] + band['erasures'] == 255
                    certificate_checks += 1
    assert len(budgets) == 40
    assert all(n == (9 if carrier == 'GS' else 3) for (_, _, carrier), n in budgets.items())
    control_trials = {r['trial']: r for r in controls}
    assert len(control_trials) == len(controls) == 240
    for trial, group in by_trial.items():
        assert group['coarse']['pixels_sha256'] == control_trials[trial]['pixels_sha256']
        assert group['coarse']['native_accepted'] == control_trials[trial]['native_accepted']
    old = {(r['id'], r['kind'], r['method']): r for r in read(WITNESS / 'rows.json') if r['kind'].startswith('reprompt_')}
    basic_checks = 0
    for row in rows:
        if row['carrier'] == 'TR' and '_p0_' in row['trial']:
            previous = old[row['id'], 'reprompt_' + row['proxy'], row['method']]
            assert all(row[k] == previous[k] for k in ['pixels_sha256', 'key_recovered', 'native_accepted', 'joint'])
            basic_checks += 1
    assert basic_checks == 80
    comparisons = Counter('better' if r['candidate_joint'] < r['control_joint'] else 'equal' if r['candidate_joint'] == r['control_joint'] else 'worse' for r in matched)
    assert len(matched) == 96 and all(r['exact_match'] for r in matched)
    source_pairs = []
    for comparison in matched:
        if comparison['method'] != 'observable_scales' or comparison['control'] != 'sscd':
            continue
        part = [r for r in rows if r['carrier'] == comparison['carrier'] and r['proxy'] == comparison['proxy'] and r['method'] == 'observable_scales' and r['split'] in ['evaluation', 'transfer']]
        counts = Counter()
        sources = []
        for identity in sorted({r['id'] for r in part}):
            own = [r for r in part if r['id'] == identity]
            accepted = any(r['joint'] for r in own)
            control = any(control_trials[r['trial']]['native_accepted'] and control_trials[r['trial']]['sscd'] <= comparison['cutoff'] for r in own)
            counts[str(int(accepted)) + str(int(control))] += 1
            sources.append(dict(id=identity, candidate=accepted, control=control))
        assert sum(r['candidate'] for r in sources) == comparison['candidate_joint']
        assert sum(r['control'] for r in sources) == comparison['control_joint']
        source_pairs.append(dict(carrier=comparison['carrier'], proxy=comparison['proxy'], genuine_joint=comparison['candidate_genuine'], counts=dict(counts), sources=sources))
    out = dict(status='complete', source_sha256=sha(Path(__file__)), input_complete_sha256=sha(RUN / 'complete.json'),
        recovery_rows=len(rows), distinct_candidates=len(by_trial), source_proxy_budgets=len(budgets),
        all_observable_decision_agreement=[equivalence, len(by_trial)], repeated_certificate_checks=certificate_checks,
        basic_tr_same_pixel_decision_checks=basic_checks, matched_comparisons=dict(comparisons),
        matched_exact_genuine_counts=96, observable_vs_matched_sscd_source_pairs=source_pairs,
        scope='Post-completion audit of saved decisions; no new recovery, inversion or attack. Certificate checks reuse bands across rules. Counts10/01 are candidate-only/control-only source acceptance.')
    target = RUN / 'decision_audit.json'
    assert not target.exists()
    target.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'observable_vs_matched_sscd_source_pairs'}))


if __name__ == '__main__':
    main()
