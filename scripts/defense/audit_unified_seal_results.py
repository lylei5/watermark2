"""Independently audit completed SEAL artifacts on249; no model inference."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
PINS = {
    'unified_seal_nulls_20261115': '1a3283f12074a95cbfede7a2bb64095dfa7cb812c65c00cceb363ca34d2de20a',
    'unified_seal_clean_20261116': '14d0ac04f12803865ae614ad7697200ed0f5bbd2a4c86c31ad68b4e0d25c4475',
    'unified_null_features_20261116': 'e0ebe9aad974a310eac591a915b946a4eab997ffa4883d2a80bf463a7b078d22',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def artifact(run, relative, digest):
    path = (run / relative).resolve()
    require(path.is_relative_to(run.resolve()), f'Artifact outside run: {relative}')
    require(sha(path) == digest, f'Artifact hash mismatch: {relative}')
    return path


def binomial(k, n):
    from scipy.stats import beta
    require(0 <= k <= n and n > 0, 'Invalid source counts')
    return dict(accepted=k, sources=n, rate=k/n,
                lower95=0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                upper95=1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def integer_threshold(scores, target):
    """Independent integer-count implementation; never accesses validation."""
    require(scores and all(type(s) is int and 0 <= s <= 1024 for s in scores), 'Invalid block counts')
    require(0 < target < 1, 'Invalid FPR target')
    budget = math.floor(len(scores) * target)
    counts = Counter(scores)
    accepted = len(scores)
    for threshold in range(1026):
        if accepted <= budget:
            return threshold
        accepted -= counts[threshold]
    raise ValueError('No integer threshold')


def check_intervals(actual, expected):
    for key in ['accepted', 'sources', 'rate', 'lower95', 'upper95']:
        require(math.isclose(actual[key], expected[key], rel_tol=1e-10, abs_tol=1e-12), f'Count/interval mismatch: {key}')


def clean_budget(receipt, actual_nfe, source_count):
    expected_nfe = source_count * 4 * 51
    require(actual_nfe == receipt['actual_total_unet_calls'] == expected_nfe, 'Wrong measured total NFE')
    require(receipt['full_budget_generations'] == receipt['full_budget_inversions'] == source_count * 2 and receipt['attacks'] == 0, 'Wrong clean scope')
    discrepancies = []
    if receipt['expected_total_unet_calls'] != expected_nfe:
        discrepancies.append(dict(field='expected_total_unet_calls', recorded=receipt['expected_total_unet_calls'],
                                  corrected=expected_nfe, evidence='All per-stage counts verified: two generations and two inversions per source,51 calls each'))
    return discrepancies


def snapshot(root):
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), runs={})
    for name, digest in PINS.items():
        run = root/'runs'/name
        require(sha(run/'protocol.json') == digest, f'Changed frozen protocol: {name}')
        state = read(run/'pipeline_state.json')
        processes = []
        pids = [state['pid']] + state.get('children', []) + [x['pid'] for x in state.get('active', [])]
        for pid in sorted(set(pids)):
            path = Path(f'/proc/{pid}/cmdline')
            try:
                command = path.read_bytes()
            except FileNotFoundError:
                command = b''
            processes.append(dict(pid=pid, process_present=bool(command), run_in_command=str(run).encode() in command))
        result['runs'][name] = dict(pipeline_state=state, processes=processes,
            saved_records=sum(1 for _ in run.glob('shard_*/records/*.json')))
    return result


def audit_calibration(run):
    import numpy as np
    plan = read(run/'protocol.json')
    pin = PINS[run.name]
    require(read(run/'pipeline_state.json')['status'] == 'complete', 'Calibration is incomplete')
    published = read(run/'calibration_results.json')
    require(published['status'] == 'complete' and published['protocol_sha256'] == pin, 'Invalid calibration receipt')
    manifest_path = Path(plan['data_manifest'])
    require(sha(manifest_path) == plan['data_manifest_sha256'], 'Changed natural-image manifest')
    manifest = read(manifest_path)
    require(len(manifest) == 40000, 'Unexpected natural-image cohort size')
    sources = {r['index']: r for r in manifest}
    require(set(sources) == set(range(40000)), 'Repeated or missing manifest index')
    records = []
    for shard in range(3):
        directory = run/f'shard_{shard}'
        receipt = read(directory/'complete.json')
        require(receipt['status'] == 'complete' and receipt['protocol_sha256'] == pin, 'Incomplete calibration shard')
        require(sha(directory/'rows.jsonl') == receipt['rows_sha256'], 'Changed calibration rows')
        rows = [json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
        require(all(r['index'] % 3 == shard for r in rows), 'Wrong calibration shard')
        records.extend(rows)
    require(sorted(r['index'] for r in records) == list(range(40000)), 'Duplicate or missing calibration records')
    for row in records:
        source = sources[row['index']]
        require(row['protocol_sha256'] == pin and row['unet_calls'] == 51, 'Wrong protocol or truncated inverse')
        for key in ['source_id', 'split']:
            require(row[key] == source[key], f'Wrong source metadata: {key}')
        require(row['input_file_sha256'] == source['file_sha256'] and
                row['input_pixel_sha256'] == source['pixel_sha256'], 'Wrong natural-image identity')
        path = artifact(run, row['artifact'], row['artifact_sha256'])
        with np.load(path, allow_pickle=False) as values:
            for key in ['recovered', 'reference', 'distances']:
                shape = (1024,) if key == 'distances' else (1, 4, 64, 64)
                require(values[key].shape == shape and np.isfinite(values[key]).all(), f'Invalid {key}')
            matches = int((values['distances'] < plan['patch_distance_threshold']).sum())
        require(matches == row['matches'], 'Saved distance score mismatch')
        require(row['paper_accepted'] == (matches >= plan['paper_minimum_matches']), 'Paper decision mismatch')
    fit = [r['matches'] for r in records if r['split'] == 'calibration']
    test = [r['matches'] for r in records if r['split'] == 'validation']
    require(len(fit) == len(test) == 20000, 'Invalid calibration/validation split')
    require(len(published['working_points']) == 2 and
            {p['target_fpr'] for p in published['working_points']} == {.01, .001}, 'Unexpected working points')
    points = []
    for point in published['working_points']:
        threshold = integer_threshold(fit, point['target_fpr'])
        require(point['comparator'] == '>=' and point['threshold'] == threshold, 'Threshold differs from calibration-only reconstruction')
        require(point['calibration']['accepted'] == sum(x >= threshold for x in fit), 'Calibration accepts differ')
        interval = binomial(sum(x >= threshold for x in test), len(test))
        check_intervals(dict(accepted=point['validation_accepted'], sources=point['validation_n'], rate=point['validation_fpr'],
                             lower95=point['validation_95ci'][0], upper95=point['validation_95ci'][1]), interval)
        points.append(dict(target_fpr=point['target_fpr'], threshold=threshold, calibration_accepted=sum(x >= threshold for x in fit),
                           calibration_n=len(fit), **interval))
    paper = published['paper_working_point']
    paper_check = binomial(sum(x >= plan['paper_minimum_matches'] for x in test), len(test))
    require(paper['threshold'] == plan['paper_minimum_matches'], 'Changed paper threshold')
    check_intervals(dict(accepted=paper['accepted'], sources=paper['n'], rate=paper['accepted']/paper['n'],
        lower95=paper['confidence_interval'][0], upper95=paper['confidence_interval'][1]), paper_check)
    return dict(status='verified', records=len(records), working_points=points,
                paper_working_point=dict(threshold=paper['threshold'], **paper_check),
                total_unet_calls=sum(r['unet_calls'] for r in records),
                summed_image_seconds=sum(r['seconds'] for r in records),
                calibration_results_sha256=sha(run/'calibration_results.json'))


def audit_clean(run, calibration):
    import numpy as np
    from PIL import Image
    pin = PINS[run.name]
    plan = read(run/'protocol.json')
    rules = read(run/'frozen_working_points.json')
    rules_sha = sha(run/'frozen_working_points.json')
    require(rules['calibration_results_sha256'] == calibration['calibration_results_sha256'], 'Changed threshold provenance')
    require(rules['calibration_protocol_sha256'] == PINS['unified_seal_nulls_20261115'], 'Wrong calibration protocol')
    expected_calibration = read(Path(plan['calibration_run'])/'calibration_results.json')
    require(rules['calibration_results'] == expected_calibration, 'Changed copied working points')
    published = read(run/'clean_results.json')
    require(published['status'] == 'complete' and published['protocol_sha256'] == pin and
            published['working_points_sha256'] == rules_sha, 'Invalid clean completion receipt')
    records = []
    for shard in range(3):
        directory = run/f'shard_{shard}'
        receipt = read(directory/'complete.json')
        require(receipt['status'] == 'complete' and receipt['protocol_sha256'] == pin, 'Incomplete clean shard')
        require(sha(directory/'rows.json') == receipt['rows_sha256'], 'Changed clean rows')
        rows = read(directory/'rows.json')
        require(all(r['index'] % 3 == shard for r in rows), 'Wrong clean shard')
        records.extend(rows)
    require(sorted(r['index'] for r in records) == list(range(500)), 'Duplicate or missing clean source')
    require(Counter(r['dataset'] for r in records) == {'COCO': 250, 'SDP': 250}, 'Wrong clean strata')
    for row in records:
        require(row['protocol_sha256'] == pin and row['working_points_sha256'] == rules_sha, 'Wrong clean record protocol')
        source = plan['sources'][row['index']]
        require((row['dataset'], row['source_id']) == (source['dataset'], source['source_id']), 'Wrong clean source')
        for item in row['images'].values():
            image_path = artifact(run, item['path'], item['sha256'])
            with Image.open(image_path) as image:
                require(hashlib.sha256(image.tobytes()).hexdigest() == item['pixel_sha256'], 'Wrong clean pixels')
        path = artifact(run, row['feature']['path'], row['feature']['sha256'])
        with np.load(path, allow_pickle=False) as values:
            for name in ['initial', 'query_reference', 'recovered', 'preview_recovered']:
                require(values[name].shape == (1, 4, 64, 64) and np.isfinite(values[name]).all(), f'Invalid clean {name}')
            for kind, name in [('genuine', 'genuine_distances'), ('preview_null', 'preview_null_distances'),
                               ('diagnostic_original_noise', 'diagnostic_original_noise_distances')]:
                require(values[name].shape == (1024,) and np.isfinite(values[name]).all(), f'Invalid clean {name}')
                require(int((values[name] < plan['detector']['patch_distance_threshold']).sum()) == row['matches'][kind], 'Wrong clean score')
            require(values['reference_drift_distances'].shape == (1024,) and np.isfinite(values['reference_drift_distances']).all(), 'Invalid reference drift')
            require(int((values['reference_drift_distances'] == 0).sum()) == row['identical_reference_patches'], 'Wrong reference drift count')
        expected_stages = {'preview_generation', 'genuine_generation', 'preview_caption', 'genuine_caption', 'genuine_inverse', 'preview_inverse'}
        require(set(row['costs']) == expected_stages, 'Missing cost stage')
        for stage, cost in row['costs'].items():
            require(cost['unet_calls'] == (0 if 'caption' in stage else 51), 'Truncated generation/inverse')
            require(math.isfinite(cost['seconds']) and cost['seconds'] >= 0, 'Invalid runtime')
    for result in published['rows']:
        threshold = next((p['threshold'] for p in calibration['working_points'] if result['working_point'] == f"natural_fpr_{p['target_fpr']}"), None)
        if result['working_point'] == 'paper_threshold12':
            threshold = 12
        require(threshold is not None and result['threshold'] == threshold, 'Changed clean threshold')
        subset = [r for r in records if result['dataset'] == 'all' or r['dataset'] == result['dataset']]
        check_intervals(result, binomial(sum(r['matches'][result['kind']] >= threshold for r in subset), len(subset)))
    expected_rows = {(p, d, k) for p in ['natural_fpr_0.01', 'natural_fpr_0.001', 'paper_threshold12']
                     for d in ['all', 'COCO', 'SDP'] for k in ['genuine', 'preview_null']}
    require(len(published['rows']) == 18 and {(r['working_point'], r['dataset'], r['kind']) for r in published['rows']} == expected_rows, 'Missing or duplicate clean cells')
    diagnostics = []
    require(len(published['diagnostics']) == 9 and
            {(d['working_point'], d['dataset']) for d in published['diagnostics']} ==
            {(p, d) for p in ['natural_fpr_0.01', 'natural_fpr_0.001', 'paper_threshold12'] for d in ['all', 'COCO', 'SDP']},
            'Missing or duplicate diagnostic cells')
    for diagnostic in published['diagnostics']:
        threshold = next((p['threshold'] for p in calibration['working_points']
                          if diagnostic['working_point'] == f"natural_fpr_{p['target_fpr']}"), None)
        if diagnostic['working_point'] == 'paper_threshold12':
            threshold = 12
        subset = [r for r in records if diagnostic['dataset'] == 'all' or r['dataset'] == diagnostic['dataset']]
        failures = [r for r in subset if r['matches']['genuine'] < threshold]
        original_pass = sum(r['matches']['diagnostic_original_noise'] >= threshold for r in failures)
        require(diagnostic['genuine_failures'] == len(failures) and
                diagnostic['failure_with_original_noise_pass'] == original_pass and
                diagnostic['failure_with_original_noise_fail'] == len(failures)-original_pass, 'Diagnostic mismatch')
    for point in calibration['working_points']:
        threshold = point['threshold']
        failures = [r for r in records if r['matches']['genuine'] < threshold]
        diagnostics.append(dict(target_fpr=point['target_fpr'], failed_sources=len(failures),
            original_reference_pass=sum(r['matches']['diagnostic_original_noise'] >= threshold for r in failures),
            original_reference_fail=sum(r['matches']['diagnostic_original_noise'] < threshold for r in failures)))
    total_nfe = sum(c['unet_calls'] for r in records for c in r['costs'].values())
    discrepancies = clean_budget(published, total_nfe, len(records))
    costs = {}
    for label, stages in [('issuance', ['preview_generation', 'preview_caption', 'genuine_generation']),
                           ('verification', ['genuine_caption', 'genuine_inverse'])]:
        seconds = [sum(r['costs'][s]['seconds'] for s in stages) for r in records]
        costs[label] = dict(mean_seconds=float(np.mean(seconds)), median_seconds=float(np.median(seconds)),
                            p95_seconds=float(np.quantile(seconds, .95)),
                            peak_allocated_bytes=max(r['costs'][s]['peak_allocated_bytes'] for r in records for s in stages))
    return dict(status='verified', sources=500, rows=published['rows'], diagnostics=diagnostics, costs=costs,
                total_unet_calls=total_nfe, metadata_discrepancies=discrepancies, clean_results_sha256=sha(run/'clean_results.json'))


def export(out, result):
    lines = ['# SEAL统一工作点与合法检出核验', '',
             '本报告只纳入完成逐记录核验的结果。自然图FPR按固定提供方身份报告；尚无本批攻击或常规失真结果。', '']
    if 'calibration' not in result:
        lines.append('校准尚无正常完成回执；不读取部分分数，不填写阈值或准确率。')
    else:
        rows = []
        for point in result['calibration']['working_points']:
            row = dict(target_fpr=point['target_fpr'], threshold=point['threshold'], natural_false_accepts=point['accepted'],
                       natural_sources=point['sources'], natural_fpr=point['rate'], natural_lower95=point['lower95'], natural_upper95=point['upper95'],
                       legitimate_status='pending', legitimate_accepts=None, legitimate_sources=None, legitimate_tpr=None,
                       legitimate_lower95=None, legitimate_upper95=None, attack_status='not_run')
            if 'clean' in result:
                clean = next(r for r in result['clean']['rows'] if r['dataset'] == 'all' and r['kind'] == 'genuine'
                             and r['working_point'] == f"natural_fpr_{point['target_fpr']}")
                row.update(legitimate_status='verified', legitimate_accepts=clean['accepted'], legitimate_sources=clean['sources'],
                           legitimate_tpr=clean['rate'], legitimate_lower95=clean['lower95'], legitimate_upper95=clean['upper95'])
            rows.append(row)
        with (out/'main_table_seal.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines += ['|目标FPR|匹配块阈值|独立自然图误报（95%区间）|干净合法图通过（95%区间）|', '|---|---:|---|---|']
        tex = ['\\begin{tabular}{rrrr}', '\\hline', 'Target FPR & Threshold & Natural FPR (95\\% CI) & Clean TPR (95\\% CI) \\\\']
        for row in rows:
            fpr = f"{row['natural_false_accepts']}/{row['natural_sources']}; {100*row['natural_fpr']:.3f}% [{100*row['natural_lower95']:.3f}, {100*row['natural_upper95']:.3f}]"
            tpr = '待完成'
            if row['legitimate_status'] == 'verified':
                tpr = f"{row['legitimate_accepts']}/{row['legitimate_sources']}; {100*row['legitimate_tpr']:.2f}% [{100*row['legitimate_lower95']:.2f}, {100*row['legitimate_upper95']:.2f}]"
            lines.append(f"|{100*row['target_fpr']:g}%|{row['threshold']}|{fpr}|{tpr}|")
            tex.append(f"{100*row['target_fpr']:g}\\% & {row['threshold']} & " + fpr.replace('%', '\\%') + ' & ' + (tpr.replace('%', '\\%') if tpr != '待完成' else '--') + ' \\\\')
        tex += ['\\hline', '\\end{tabular}']
        (out/'main_table_seal.tex').write_text('\n'.join(tex)+'\n')
        if 'clean' in result:
            if result['clean'].get('metadata_discrepancies'):
                lines += ['', '计算量核对：原完成回执的预期UNet总调用数写成204000；逐阶段实测总计102000，与1000次生成、1000次反演各51次调用一致。原回执保留，准确率和完整推理预算不受这项汇总笔误影响。']
            lines += ['', '干净图失败分解（原始嵌入噪声仅用于诊断，不计入无状态检测成绩）：', '',
                      '|目标FPR|合法拒绝|其中原参考诊断通过|其中原参考诊断也失败|', '|---|---:|---:|---:|']
            for row in result['clean']['diagnostics']:
                lines.append(f"|{100*row['target_fpr']:g}%|{row['failed_sources']}|{row['original_reference_pass']}|{row['original_reference_fail']}|")
            lines += ['', '该分解是描述性观察，不单独建立失败原因的因果结论。检测和发行耗时含相应caption及预览生成；模型加载与预览负例检测另行记账。', '',
                      '|操作|平均秒/图|中位秒/图|P95秒/图|峰值已分配显存/字节|', '|---|---:|---:|---:|---:|']
            for label, cost in result['clean']['costs'].items():
                lines.append(f"|{label}|{cost['mean_seconds']:.3f}|{cost['median_seconds']:.3f}|{cost['p95_seconds']:.3f}|{cost['peak_allocated_bytes']}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(ROOT.is_dir() and Path('/proc').is_dir(), 'Execute only on249')
    out = args.out.resolve()
    require(out.parent == ROOT/'runs' and not out.exists(), 'Use a fresh audit directory inside249 runs')
    out.mkdir()
    result = dict(status='started', script_sha256=sha(Path(__file__)), observed_at=datetime.now(timezone.utc).isoformat(),
                  inference_calls=0, attacks=0, protocol_pins=PINS)
    try:
        state = snapshot(ROOT)
        save(out/'snapshot.json', state)
        if state['runs']['unified_seal_nulls_20261115']['pipeline_state']['status'] == 'complete':
            result['calibration'] = audit_calibration(ROOT/'runs/unified_seal_nulls_20261115')
        if state['runs']['unified_seal_clean_20261116']['pipeline_state']['status'] == 'complete':
            require('calibration' in result, 'Clean completion without completed calibration')
            result['clean'] = audit_clean(ROOT/'runs/unified_seal_clean_20261116', result['calibration'])
        result['status'] = 'audit_complete' if 'calibration' in result else 'awaiting_calibration_completion'
        export(out, result)
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(out/'audit.json', result)
    print(json.dumps(dict(status=result['status'], out=str(out), calibration_verified='calibration' in result, clean_verified='clean' in result)))


if __name__ == '__main__':
    main()
