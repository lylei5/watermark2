"""Run fixed defense revalidation jobs and summarize on process exit."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def summarize_pgid(folder, expected_indices):
    raw = folder / 'attack=forgery/all_results.csv'
    rows = list(csv.DictReader(raw.open()))
    assert len(rows) == len(expected_indices) * 7
    groups, safe = defaultdict(list), []
    for row in rows:
        a, b = row['message_bits_str_initial'], row['message_bits_str_recovered']
        assert len(a) == len(b) == 256 and set(a + b) <= {'0', '1'}
        ba = sum(x == y for x, y in zip(a, b)) / 256
        accepted = ba > 181 / 256
        assert ba == float(row['bit_accuracy'])
        assert row['detection_successful'] == str(accepted)
        item = dict(image_idx=int(row['image_idx']), stage=row['stage'], step=int(row['attack_step']),
                    bit_accuracy=ba, accepted=accepted)
        safe.append(item)
        groups[(item['stage'], item['step'])].append(item)
    expected_groups = {('watermarked', -1)} | {(s, t) for s in ['attacked', 'defended'] for t in [50, 100, 150]}
    assert set(groups) == expected_groups
    summary = []
    for (stage, step), group in sorted(groups.items()):
        assert sorted(r['image_idx'] for r in group) == expected_indices
        summary.append(dict(stage=stage, step=step, n=len(group), accepted=sum(r['accepted'] for r in group),
                            mean_bit_accuracy=statistics.mean(r['bit_accuracy'] for r in group)))
    return dict(raw_sha256=sha(raw), rows=safe, summary=summary, all_row_checks_passed=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--job', choices=['masker', 'pgid0', 'pgid5'], required=True)
    args = p.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'protocol.json').read_text())
    root = Path(plan['root'])
    state = run / (args.job + '_state.json')
    with state.open('x') as f:
        json.dump({'status': 'preflight', 'supervisor_pid': os.getpid()}, f)
    record = dict(job=args.job, supervisor_pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'),
                  start_time_utc=datetime.now(timezone.utc).isoformat(), status='preflight')
    try:
        for name, digest in plan['source_sha256'].items():
            assert sha(run / name) == digest, name
        if args.job == 'masker':
            gpu = plan['masker_gpu']
            python = root / 'envs/semantic_forgery_py310/bin/python'
            command = [str(python), str(run / 'revalidate_masker.py'), '--run', str(run)]
            cwd = run
        else:
            start = int(args.job[4:])
            shard = next(x for x in plan['pgid']['shards'] if x['start_index'] == start)
            gpu = shard['gpu']
            python = root / 'envs/pgid_revalidation_py310/bin/python'
            assert not (run / args.job).exists()
            for item in plan['pgid_inputs']:
                assert sha(run / item['cover']) == item['cover_sha256']
            for name, digest in plan['model_sha256'].items():
                if 'models--Manojb' in name:
                    assert sha(root / name) == digest, name
            command = [str(python), '-u', 'run.py', '--wm_type', 'GS', '--attack_type', 'forgery',
                       '--num_images', str(shard['num_images']), '--start_index', str(start),
                       '--cover_images_dir', 'cover_images/ms_coco', '--out_dir', str(run / args.job),
                       '--modelid_target', plan['pgid']['modelid'], '--modelid_attacker', plan['pgid']['modelid'],
                       '--steps', '151', '--num_inference_steps_target', '50', '--num_inference_steps_attacker', '50',
                       '--scheduler_target', 'DDIM', '--scheduler_attacker', 'DDIM', '--lr', '0.01', '--seed', '1', '--gpu', 'cuda:0']
            cwd = run / 'PGID'
        assert gpu in [7, 5, 4]
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 50 and int(info[3]) == 0, info
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=info[1], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   HF_HOME=str(root / 'cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONUNBUFFERED='1', PYTHONPATH='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PIP_INDEX_URL=plan['package_index'], TOKENIZERS_PARALLELISM='false')
        packages = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], env=env, text=True)
        (run / (args.job + '_packages.txt')).write_text(packages)
        with (run / (args.job + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(status='running', child_pid=child.pid, gpu=gpu, gpu_uuid=info[1], command=command)
            write(state, record)
            code = child.wait()
        record['returncode'] = code
        if code:
            raise RuntimeError('Child failed; inspect ' + args.job + '.log')
        if args.job == 'masker':
            result = json.loads((run / 'masker/complete.json').read_text())
            assert result['observations'] == 1000 and result['pairs'] == 4950
            record['result'] = result
        else:
            record['result'] = summarize_pgid(run / args.job, list(range(start, start + shard['num_images'])))
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['last_event_utc'] = datetime.now(timezone.utc).isoformat()
        write(state, record)


if __name__ == '__main__':
    main()
