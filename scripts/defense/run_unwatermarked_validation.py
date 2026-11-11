"""Full DDIM50 natural-negative validation followed by automatic frozen scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root, args.run
    assert not any((run / n).exists() for n in ['status.json', 'complete.json', 'failed.json'])
    plan = json.loads((run / 'protocol.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(root / 'cache/huggingface'), PYTHONPATH=str(run / 'scripts'))
    try:
        assert plan['evaluation_split'] == 'unwatermarked' and len(plan['rows']) == 100
        assert len({r['id'] for r in plan['rows']}) == 100 and len(plan['identity_tasks']) == 64
        for name, digest in plan['source_sha256'].items():
            assert sha(run / 'scripts' / name) == digest
        for row in plan['rows'] + plan['bridge_rows']:
            assert sha(run / row['image']) == row['image_sha256']
            for key in ['expected_inversion', 'expected_features']:
                if key in row:
                    assert sha(run / row[key]) == row[key + '_sha256']
        devices = []
        for index in plan['execution_gpus']:
            assert index in [7, 5, 4]
            raw = subprocess.check_output(['nvidia-smi', '-i', str(index),
                '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            i, uuid, memory, util = [x.strip() for x in raw.strip().split(',')]
            assert int(i) == index and int(memory) <= 50 and int(util) == 0, 'Assigned GPU is occupied'
            devices.append((index, uuid))
        assert 1 <= len(devices) <= 3 and len(set(devices)) == len(devices)
        write(run / 'status.json', dict(stage='caching_unwatermarked', pid=os.getpid(), devices=devices))
        def worker(slot, device):
            index, uuid = device
            for shard in range(slot, 3, len(devices)):
                command = [str(root / 'envs/ddim_replay_py310/bin/python'), str(run / 'scripts/cache_geometry_calibration.py'),
                           '--run', str(run), '--root', str(root), '--shard', str(shard), '--gpu-index', str(index)]
                with (run / f'cache_{shard}.log').open('w') as log:
                    subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=uuid), stdout=log, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, slot, device) for slot, device in enumerate(devices)]
            for future in futures:
                future.result()
        write(run / 'status.json', dict(stage='scoring_unwatermarked', pid=os.getpid()))
        with (run / 'score.log').open('w') as log:
            subprocess.run([str(root / 'envs/semantic_forgery_py310/bin/python'),
                str(run / 'scripts/score_unwatermarked_validation.py'), '--run', str(run)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        scores = json.loads((run / 'scores/results.json').read_text())
        assert scores['completed'] and scores['queries'] == 100 and scores['identity_decisions'] == 6400
        write(run / 'complete.json', dict(completed=True, queries=100, identity_decisions=6400,
            protocol_sha256=sha(run / 'protocol.json'), results_sha256=sha(run / 'scores/results.json'), new_attacks_run=False))
        write(run / 'status.json', dict(stage='complete', pid=os.getpid()))
    except Exception as exc:
        write(run / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
