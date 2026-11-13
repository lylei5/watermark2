"""Build auditable result tables from completed, immutable recovery artifacts."""
import argparse
import json
from pathlib import Path

from shared_channel_experiment import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    complete, audit, paired = [read(run / name) for name in ['complete.json', 'audit.json', 'pairing_control_complete.json']]
    assert all(x['status'] == 'complete' for x in [complete, audit, paired])
    for manifest in [complete, paired]:
        for name, digest in manifest['artifacts'].items():
            assert sha(run / name) == digest
    result = read(run / 'scores.json')
    scores, controls, enrollment = result['scores'], read(run / 'controls.json'), read(run / 'enrollment.json')
    plan = read(run / 'protocol.json')
    all_methods = plan['methods']
    methods = [m for m in all_methods if m != 'permuted_phase']

    def find(point, carrier, split, kind):
        return next((r for r in point['summary'] if r.get('carrier', 'GS') == carrier and r['split'] == split and r['kind'] == kind), None)

    def cell(row):
        return '--' if row is None else f"{row['accepted']}/{row['total']} ; {row['joint']}/{row['total']}"

    text = ['# 共同通道实例恢复：完整结果表', '',
            '所有表格由249上的逐样本结果生成。数对依次为“内容/密钥通过；与原生水印联合通过”。',
            'GS校准仅使用来源4–6的30张合法图；该组原生检测全部通过。评估使用来源7–9，TR/PRC冻结迁移。',
            '三次投影是同一批来源的计算重复。全局打乱作为诊断独立保存；正式消融使用划分内部打乱。', '']
    for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
        text += [f'## {carrier}：连续距离，GS95%校准', '',
                 '| 表示 | 合法图 | 未登记处理 | 同提示词其他实例 | 错误记录 | 完整Imprint |',
                 '|---|---|---|---|---|---|']
        for method in [*methods, 'sscd_templates', 'sscd_int8_templates']:
            point = next(s for s in scores if s['method'] == method and s['repetition'] is None and s['target'] == .95)
            text.append('| ' + method + ' | ' + ' | '.join(cell(find(point, carrier, split, kind)) for kind in
                        ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'imprint_150']) + ' |')
        text += ['', f'## {carrier}：实际密钥恢复', '',
                 '| 表示 | 投影 | GS校准目标 | 可行 | 合法图 | 未登记处理 | 同提示词其他实例 | 错误记录 | 完整Imprint |',
                 '|---|---|---|---|---|---|---|---|---|']
        for method in methods:
            for target in [None, .9, .95]:
                for rep in range(3):
                    point = next(s for s in scores if s['method'] == method and s['repetition'] == rep and s['target'] == target)
                    prefix = f"| {method} | {rep} | {target if target else '最大半径127'} | {point['feasible']} | "
                    text.append(prefix + (' | '.join(cell(find(point, carrier, split, kind)) for kind in
                                ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'imprint_150']) if point['feasible'] else '不可行 | -- | -- | -- | --') + ' |')
        text.append('')
    text += ['## GS划分内部打乱消融', '', '| 投影 | 目标 | 可行 | 合法图 | 未登记处理 | 同提示词其他实例 | 错误记录 | 完整Imprint |',
             '|---|---|---|---|---|---|---|---|']
    for point in paired['scores']:
        prefix = f"| {point['repetition'] if point['repetition'] is not None else '连续'} | {point['target'] or '最大半径127'} | {point['feasible']} | "
        text.append(prefix + (' | '.join(cell(find(point, 'GS', 'evaluation', kind)) for kind in
                    ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'imprint_150']) if point['feasible'] else '不可行 | -- | -- | -- | --') + ' |')
    comparison = []
    for match in audit['matched_comparisons']:
        if match['method'] == 'permuted_phase':
            continue
        point = next(s for s in scores if s['method'] == match['method'] and s['repetition'] == match['repetition'] and s['target'] == match['target'])
        genuine = next(r for r in match['summary'] if r['kind'] == 'all_genuine')
        checks = []
        for row in match['summary']:
            if row['kind'] == 'all_genuine':
                continue
            candidate = find(point, match['carrier'], match['split'], row['kind'])
            assert candidate['total'] == row['total']
            checks.append(dict(kind=row['kind'], total=row['total'], baseline=row['accepted'], candidate=candidate['joint']))
        comparison.append(dict(method=match['method'], repetition=match['repetition'], target=match['target'],
                               carrier=match['carrier'], baseline=match['baseline'], candidate_genuine=match['candidate_genuine'],
                               baseline_genuine=genuine['accepted'], negatives=checks,
                               weakly_dominated=genuine['accepted'] >= match['candidate_genuine'] and all(x['baseline'] <= x['candidate'] for x in checks)))
    costs = {}
    for method in all_methods:
        sizes = [r['helper_bytes'] for r in enrollment if r['method'] == method]
        costs[method] = dict(helper_bytes_min=min(sizes), helper_bytes_max=max(sizes), records=len(sizes))
    native = []
    for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
        for kind in ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'imprint_150']:
            part = [r for r in controls if r['carrier'] == carrier and r['split'] == split and
                    (r['kind'] in ['clean', *plan['conditions']] if kind == 'all_genuine' else
                     r['kind'] in plan['conditions'] and r['kind'] not in plan['enrollment_families'] if kind == 'unknown_edits' else r['kind'] == kind)]
            native.append(dict(carrier=carrier, split=split, kind=kind, total=len(part), accepted=sum(r['native_accepted'] for r in part)))
    summary = dict(status='complete', source_sha256=sha(__file__), actual_recoveries=result['actual_recovery_decisions'],
                   successful_recoveries=result['success_count'], additional_pairing_recoveries=paired['actual_recoveries'],
                   native_calls=result['original_native_calls'], restored_native_calls=result['restored_native_calls'],
                   additional_pairing_native_calls=paired['restored_native_calls'], native_acceptance_disagreements=result['native_acceptance_disagreements'],
                   native=native, costs=costs, matched_comparisons=comparison,
                   weakly_dominated_comparisons=sum(x['weakly_dominated'] for x in comparison), total_comparisons=len(comparison),
                   comparison_scope='Descriptive thresholds on evaluation genuine data; not independently calibrated FPR',
                   artifacts={name: sha(run / name) for name in ['complete.json', 'audit.json', 'pairing_control_complete.json', 'scores.json', 'controls.json', 'enrollment.json']})
    write(run / 'review_summary.json', summary)
    (run / 'result_tables.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['native', 'costs', 'matched_comparisons', 'artifacts']}), flush=True)


if __name__ == '__main__':
    main()
