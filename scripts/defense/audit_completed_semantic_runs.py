"""Read-only server audit of completed semantic evidence and feedback artifacts."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'runs/shared_semantic_evidence_20261112'))
    from anchor_binding import identity, order, decode, embed
    from semantic_evidence import scores, decisions
    from semantic_evidence_reference import grouped_reference
    report = dict(audit_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  execution_host='249', audit_script_sha256=sha(__file__),
                  scope='Artifact and arithmetic audit; no new model execution', runs={})
    for name in ['shared_semantic_evidence_20261112', 'semantic_feedback_pilot_20261112']:
        run = args.root / 'runs' / name
        plan = json.loads((run / 'protocol.json').read_text())
        state = json.loads((run / 'pipeline_state.json').read_text())
        assert state['status'] == 'complete'
        for file, digest in plan['source_sha256'].items():
            assert sha(run / file) == digest, file
        for file, digest in plan['external_sha256'].items():
            assert sha(file) == digest, file
        ids = {}
        for source in plan['sources']:
            t = source['task']
            _, c, m, key = identity(plan['identity_root'], t)
            ids[t] = c, m, order(key, 16384)
        checks = dict(completion=state, protocol_sha256=sha(run/'protocol.json'),
                      source_hashes_verified=True, external_hashes_verified=True,
                      records={}, artifacts={}, conditions={}, sources={}, maximum_score_difference=0.0)
        shared = name.startswith('shared')
        model = json.loads((run/'decoder_model.json').read_text()) if shared else None
        stages = ['genuine', 'forgery'] if shared else ['genuine']
        for stage in stages:
            verified = json.loads((run/f'{stage}_verified.json').read_text())
            expected_count = (120 if stage == 'genuine' else 216) if shared else 180
            assert verified['audit_passed'] and len(verified['rows']) == expected_count
            rows = verified['rows']
            assert len({r['id'] for r in rows}) == expected_count
            shards = [json.loads((run/f'{stage}_{i}/complete.json').read_text()) for i in range(3)]
            for shard in shards:
                assert shard['status'] == 'complete' and shard['protocol_sha256'] == sha(run/'protocol.json')
            assert {r['id']: r for shard in shards for r in shard['rows']} == {r['id']: r for r in rows}
            evidence = json.loads((run/f'{stage}_defense_results.json').read_text()) if shared else None
            if shared:
                assert evidence['completed'] and evidence['independent_grouped_score_audit']
                assert evidence['model_sha256'] == sha(run/'decoder_model.json')
                by_id = {r['id']: r for r in evidence['rows']}
                assert set(by_id) == {r['id'] for r in rows}
            wrong_count = 0
            wrong_accepts = {}
            wrong_expected = {(r['id'], r['identity']): r['decisions'] for r in evidence['wrong_identity']} if shared else {}
            for r in rows:
                for key in ['image', 'code', 'latent']:
                    assert sha(run/r[key]) == r[key+'_sha256'], (r['id'], key)
                f = np.load(run/r['code']); z = np.load(run/r['latent'])
                # All feedback arms use exactly the full SemBind carrier.
                ba = decode(z, f['code'], plan['identity_root'], r['task'], 'full')
                assert ba == r['ba'] and (ba > 181/256) == r['accepted']
                if shared:
                    value = scores(z, f['logits'], f['code'], *ids[r['task']], model)
                    stored = by_id[r['id']]['scores']
                    assert value['ba'] == stored['ba']
                    for k in value:
                        difference = abs(value[k]-stored[k])
                        assert difference < 1e-7, (r['id'], k, value[k], stored[k])
                        checks['maximum_score_difference'] = max(checks['maximum_score_difference'], difference)
                    assert decisions(value, model) == by_id[r['id']]['decisions']
                    c, _, perm = ids[r['task']]
                    independent = grouped_reference(z, f['logits'], f['code'], c, perm, model)
                    assert all(abs(value[k]-independent[k]) < 1e-6 for k in independent)
                if stage == 'genuine':
                    for t in ids:
                        if t == r['task']:
                            continue
                        wrong_count += 1
                        if shared:
                            d = decisions(scores(z, f['logits'], f['code'], *ids[t], model), model)
                            assert d == wrong_expected[(r['id'], t)]
                            for k, v in d.items(): wrong_accepts[k] = wrong_accepts.get(k, 0)+int(v)
                        else:
                            d = decode(z, f['code'], plan['identity_root'], t, 'full') > 181/256
                            wrong_accepts[r['arm']] = wrong_accepts.get(r['arm'], 0)+int(d)
                    condition = checks['conditions'].setdefault(r['arm'], {}).setdefault(r['condition'], [0, 0])
                    condition[0] += int(r['accepted']); condition[1] += 1
                    source = checks['sources'].setdefault(r['arm'], {}).setdefault(str(r['task']), [0, 0])
                    source[0] += int(r['accepted']); source[1] += 1
            checks['records'][stage] = dict(n=len(rows), wrong_identity_pairs=wrong_count, wrong_accepts=wrong_accepts)
            if stage == 'genuine':
                expected = {(s['task'], a, c['name']) for s in plan['sources'] for a in plan['arms'] for c in plan['conditions']}
                assert {(r['task'], r['arm'], r['condition']) for r in rows} == expected
            if shared:
                for summary in evidence['summary']:
                    k = summary['method']
                    groups = {'genuine': rows} if stage == 'genuine' else {
                        'reference': [r for r in rows if r['kind']=='reference'],
                        'cover': [r for r in rows if r['kind']=='cover'],
                        'final_candidate': [r for r in rows if r['kind']=='candidate' and r['update']==151],
                        'all_checkpoints': [r for r in rows if r['kind']=='candidate']}
                    for group, rs in groups.items():
                        assert summary[group] == dict(n=len(rs), accepted=sum(by_id[r['id']]['decisions'][k] for r in rs))
                checks['artifacts'][f'{stage}_defense_results.json'] = sha(run/f'{stage}_defense_results.json')
            checks['artifacts'][f'{stage}_verified.json'] = sha(run/f'{stage}_verified.json')
        if shared:
            attack = json.loads((run/'attack_protocol.json').read_text())
            assert len(attack['jobs']) == 12
            for job in attack['jobs']:
                out = run/'attack_outputs'/job['id']
                done = json.loads((out/'complete.json').read_text())
                assert done['updates_executed'] == 151 and len(done['trace']) == 151
                assert done['protocol_sha256'] == sha(run/'attack_protocol.json')
                assert [c['update'] for c in done['checkpoints']] == list(range(1,152,10))
                for c in done['checkpoints']: assert sha(out/c['path']) == c['sha256']
            checks['published_imprint'] = dict(jobs=12, updates_per_job=151, checkpoints_per_job=16)
        else:
            detail = json.loads((run/'feedback_audit.json').read_text())
            assert detail['audit_passed'] and len(detail['feedback_trace']) == 18
            assert len(detail['wrong_identity']) == 900
            for entry in detail['feedback_trace']:
                for key in ['input_code','output_code','image','initial_latent']:
                    assert sha(run/entry[key]) == entry[key+'_sha256']
                before = np.load(run/entry['input_code'])['code']
                after = np.load(run/entry['output_code'])['code']
                initial = np.load(run/entry['initial_latent'])
                assert np.array_equal(initial, embed(None, before, plan['identity_root'], entry['task'], 'full'))
                assert float(np.mean(before != after)) == entry['binding_mismatch']
            decision = json.loads((run/'pilot_decision.json').read_text())
            assert not decision['advance'] and not all(decision['gates'].values())
            assert not (run/'attack_outputs').exists()
            checks['decision'] = decision
            checks['feedback_sources'] = detail['sources']
            checks['artifacts'].update({p: sha(run/p) for p in ['feedback_audit.json','pilot_decision.json']})
        checks['artifacts']['pipeline_state.json'] = sha(run/'pipeline_state.json')
        checks['passed'] = True
        report['runs'][name] = checks
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, output=str(args.output), records=516)))


if __name__ == '__main__':
    main()
