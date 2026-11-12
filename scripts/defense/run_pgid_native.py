"""Execute the pinned, unmodified PGID entry point and audit its saved decisions."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path('/data0/liyanlei/watermark_research')
    assert run.parent == root / 'runs'
    plan = json.loads((run / 'protocol.json').read_text())
    assert sha(__file__) == plan['worker_sha256']
    out = run / f'native_{args.shard}'
    out.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    try:
        source = Path(plan['source'])
        for name, digest in plan['source_sha256'].items():
            assert sha(source / name) == digest, name
        gpu = plan['gpu_indices'][args.shard]
        assert gpu in [7, 5, 4]
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                  '--query-gpu=index,uuid,memory.used,utilization.gpu',
                  '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == gpu and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(root / 'cache/huggingface'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', PYTHONPATH='')
        reference = root / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert reference.read_text().strip() == plan['model_revision']
        shard = plan['shards'][args.shard]
        command = [sys.executable, '-u', str(source / 'run.py'),
                   '--wm_type', plan['carrier'], '--attack_type', plan['attack_type'],
                   '--num_images', str(shard['count']), '--start_index', str(shard['start']),
                   '--out_dir', str(out), '--modelid_target', plan['model_id'],
                   '--modelid_attacker', plan['model_id'], '--scheduler_target', 'DDIM',
                   '--scheduler_attacker', 'DDIM', '--num_inference_steps_target', '50',
                   '--num_inference_steps_attacker', '50', '--steps', '151',
                   '--lr', '0.01', '--seed', '1', '--gpu', 'cuda:0']
        if plan['attack_type'] == 'forgery':
            command.extend(['--cover_images_dir', plan['cover_images_dir']])
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1], command=command)
        write(out / 'state.json', state)
        packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, env=env)
        (out / 'packages.txt').write_text(packages)
        work = out / 'work'
        work.mkdir()
        with (out / 'official.log').open('x') as log:
            subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = out / ('attack=' + plan['attack_type']) / 'all_results.csv'
        rows = []
        for row in csv.DictReader(raw.open()):
            a, b = row['message_bits_str_initial'], row['message_bits_str_recovered']
            assert len(a) == len(b) == 256
            accuracy = sum(x == y for x, y in zip(a, b)) / 256
            assert accuracy == float(row['bit_accuracy'])
            accepted = accuracy > 181 / 256
            assert row['detection_successful'] == str(accepted)
            rows.append(dict(task=int(row['image_idx']), stage=row['stage'],
                             checkpoint=int(row['attack_step']), ba=accuracy, accepted=accepted))
        expected = {(i, stage, checkpoint) for i in range(shard['start'], shard['start'] + shard['count'])
                    for stage, checkpoint in [('watermarked', -1)] + [(s, c) for s in ['attacked', 'defended'] for c in [50, 100, 150]]}
        assert len(rows) == len(expected) and {(x['task'], x['stage'], x['checkpoint']) for x in rows} == expected
        artifacts = {str(p.relative_to(out)): sha(p) for p in out.rglob('*.png')}
        state.update(status='complete', rows=rows, raw_sha256=sha(raw), image_sha256=artifacts,
                     completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(out / 'state.json', {k: v for k, v in state.items() if k not in ['rows', 'image_sha256']})


if __name__ == '__main__':
    main()
