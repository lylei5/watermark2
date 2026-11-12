"""Freeze published PGID-R GS validation using its unchanged released experiment."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/data0/liyanlei/watermark_research')
    run = root / 'runs/pgid_native_removal_20261113'
    run.mkdir(exist_ok=False)
    manifest = json.loads((root / 'runs/three_baselines_20261113/source_manifest.json').read_text())
    official = manifest['created_source_roots']['PGID']
    prior = root / 'runs/seal_native_20261113_v2/pipeline_state.json'
    pid = json.loads(prior.read_text())['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'seal_native_20261113_v2' in command
    previous = dict(pid=pid, cmdline_sha256=hashlib.sha256(command).hexdigest())
    for name in ['run_pgid_native.py', 'run_baseline_after_exit.py', 'prepare_pgid_native.py']:
        shutil.copy2(Path(__file__).parent / name, run / name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Native PGID-R GS on10 official prompt cases; released full151 update Imprint removal, not paper-scale reproduction',
                python=str(root / 'envs/pgid_revalidation_py310/bin/python'),
                source=official['path'], source_commit=official['commit'], source_sha256=official['files'],
                carrier='GS', attack_type='removal',
                published_attack='Imprint removal, CVPR2025, as released unchanged by PGID authors',
                model_id='Manojb/stable-diffusion-2-1-base', model_revision='0094d483a120f3f33dafbd187ea4aa60d10de75c',
                steps=151, checkpoints=[50, 100, 150], actual_updates=[51, 101, 151],
                defense=dict(steps=50, stopping_timestep=10, num_skip_inversion=1, guidance_strength=.045),
                gpu_indices=[7, 5, 4], predecessors=[previous, previous, previous],
                shards=[dict(start=0, count=4), dict(start=4, count=3), dict(start=7, count=3)],
                watermark_identity='Unchanged public official fixture keys/messages; no unseen-secret security claim',
                cohort='Official first10 prompts already inspected and used in previous PGID-F reproduction',
                worker='run_pgid_native.py', worker_sha256=sha(run / 'run_pgid_native.py'),
                coordinator_sha256=sha(run / 'run_baseline_after_exit.py'))
    (run / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))))


if __name__ == '__main__':
    main()
