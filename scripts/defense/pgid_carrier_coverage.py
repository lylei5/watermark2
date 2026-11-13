"""Freeze and execute native PGID carrier coverage; upstream algorithms unchanged."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data0/liyanlei/watermark_research')


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


def prepare(run):
    run.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / 'runs/three_baselines_20261113/source_manifest.json').read_text())
    source = manifest['created_source_roots']['PGID']
    parent = ROOT / 'runs/defense_revalidation_20261111'
    inputs = json.loads((parent / 'protocol.json').read_text())['pgid_inputs']
    covers = parent / 'PGID/cover_images/ms_coco'
    actual = sorted(p.name for p in covers.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg'])[:10]
    assert actual == [Path(x['cover']).name for x in inputs]
    for x in inputs:
        assert sha(parent / x['cover']) == x['cover_sha256']
    for name, digest in source['files'].items():
        assert sha(Path(source['path']) / name) == digest
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Native TR and PRC PGID-F on10 author prompt/cover pairs per carrier',
        source=source, inputs=inputs, cover_images_dir=str(covers),
        jobs=[dict(carrier='TR', gpu=5), dict(carrier='PRC', gpu=4)],
        attack_type='forgery', attack='Unchanged released Imprint CVPR2025 implementation',
        attack_code_modified=False, actual_updates=151, checkpoints=[50, 100, 150],
        checkpoint_updates=[51, 101, 151], target_steps=50, attacker_steps=50,
        lr=.01, seed=1, start=0, count=10, attacker_scheduler='DDIM',
        target_schedulers=dict(TR='DDIM', PRC='DPM'),
        defense=dict(steps=50, s=3, gamma=.001, k=dict(TR=15, PRC=7)),
        model='Manojb/stable-diffusion-2-1-base', revision='0094d483a120f3f33dafbd187ea4aa60d10de75c',
        thresholds=dict(TR=dict(metric='p_value', operator='<', value=.053849971492007605),
                        PRC=dict(metric='PRC_detection_results', operator='native_boolean', value=None)),
        identity='Native public fixture keys/messages; original entry-point access boundary',
        independent_confirmation=False, cohort='Previously observed author10 prompt/cover development pairs',
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, carrier, plan):
    out = run / carrier
    out.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), carrier=carrier, protocol_sha256=sha(run / 'protocol.json'))
    try:
        source = Path(plan['source']['path'])
        for name, digest in plan['source']['files'].items():
            assert sha(source / name) == digest, name
        for item in plan['inputs']:
            assert sha(Path(plan['cover_images_dir']) / Path(item['cover']).name) == item['cover_sha256']
        reference = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert reference.read_text().strip() == plan['revision']
        gpu = next(j['gpu'] for j in plan['jobs'] if j['carrier'] == carrier)
        assert gpu in [5, 4]
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == gpu and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(ROOT / 'cache/huggingface'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONPATH='')
        command = [sys.executable, '-u', str(source / 'run.py'), '--wm_type', carrier,
            '--attack_type', 'forgery', '--num_images', '10', '--start_index', '0',
            '--out_dir', str(out), '--modelid_target', plan['model'], '--modelid_attacker', plan['model'],
            '--scheduler_target', plan['target_schedulers'][carrier], '--scheduler_attacker', 'DDIM',
            '--num_inference_steps_target', '50', '--num_inference_steps_attacker', '50',
            '--steps', '151', '--lr', '0.01', '--seed', '1', '--gpu', 'cuda:0',
            '--cover_images_dir', plan['cover_images_dir']]
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1], command=command,
                     started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out / 'state.json', state)
        (out / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], env=env, text=True))
        work = out / 'work'
        work.mkdir()
        with (out / 'official.log').open('x') as log:
            subprocess.run(command, env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = out / 'attack=forgery/all_results.csv'
        rows = []
        for row in csv.DictReader(raw.open()):
            if carrier == 'TR':
                value = float(row['p_value'])
                assert math.isfinite(value) and 0 <= value <= 1
                accepted = value < plan['thresholds']['TR']['value']
            else:
                assert row['PRC_detection_results'] in ['True', 'False']
                value = row['PRC_detection_results'] == 'True'
                accepted = value
            assert row['detection_successful'] == str(accepted)
            rows.append(dict(task=int(row['image_idx']), stage=row['stage'],
                checkpoint=int(row['attack_step']), metric=value, accepted=accepted))
        expected = {(i, s, c) for i in range(10) for s, c in [('watermarked', -1)] +
                    [(s, c) for s in ['attacked', 'defended'] for c in [50, 100, 150]]}
        assert len(rows) == 70 and {(x['task'], x['stage'], x['checkpoint']) for x in rows} == expected
        summary = []
        for stage, checkpoint in [('watermarked', -1)] + [(s, c) for s in ['attacked', 'defended'] for c in [50, 100, 150]]:
            selected = [r for r in rows if r['stage'] == stage and r['checkpoint'] == checkpoint]
            summary.append(dict(stage=stage, checkpoint=checkpoint, accepted=sum(r['accepted'] for r in selected), total=10))
        state.update(status='complete', rows=rows, summary=summary, raw_sha256=sha(raw),
            image_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*.png')},
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out / 'complete.json', state)
        print(json.dumps(dict(carrier=carrier, summary=summary)), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(out / 'state.json', {k: v for k, v in state.items() if k not in ['rows', 'image_sha256']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'TR', 'PRC'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256']
        execute(run, args.mode, plan)


if __name__ == '__main__':
    main()
