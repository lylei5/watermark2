"""Freeze SEAL's released semantic pipeline and final-paper decision constants."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--after-run', required=True)
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name == 'posix'
    run = (root / 'runs' / args.run_name).resolve()
    assert run.parent == root / 'runs'
    run.mkdir(exist_ok=False)
    source = root / 'runs/three_baselines_20261113/source_manifest.json'
    preflight = root / 'runs/three_baselines_20261113/seal_preflight.json'
    manifest = json.loads(source.read_text())
    official = manifest['created_source_roots']['SEAL']
    # The source manifest schema is recorded alongside the protocol.
    old = root / 'runs/phase1_trajectory_native_gs_20261113'
    cohort = json.loads((old / 'protocol.json').read_text())['sources']
    sources = []
    for item in cohort:
        group = [x for x in cohort if x['dataset'] == item['dataset']]
        nxt = group[(group.index(item) + 1) % len(group)]
        sources.append(dict(task=item['task'], dataset=item['dataset'], prompt=item['caption'],
                            null_prompt=nxt['caption'], generation_seed=202611130000 + item['task']))
    parent = (root / 'runs' / args.after_run).resolve()
    assert parent.parent == root / 'runs'
    pid = json.loads((parent / 'pipeline_state.json').read_text())['pid']
    path = Path(f'/proc/{pid}/cmdline')
    command = path.read_bytes() if path.exists() else b''
    assert not command or str(parent).encode() in command
    predecessor = dict(pid=pid, cmdline_sha256=hashlib.sha256(command).hexdigest())
    for name in ['run_seal_native.py', 'run_baseline_after_exit.py', 'prepare_seal_native.py']:
        shutil.copy2(Path(__file__).parent / name, run / name)
    shutil.copy2(preflight, run / 'environment_assets.json')
    shutil.copy2(source, run / 'source_manifest.json')
    model = root / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
    assert json.loads((model / 'scheduler/scheduler_config.json').read_text())['_class_name'] == 'PNDMScheduler'
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='100-source development validation of released SEAL semantic generation and detection; not the paper full benchmark',
                python=str(root / 'envs/seal_reproduction_py310/bin/python'),
                source=official['path'], source_sha256=official['files'],
                source_commit=official['commit'], source_manifest_sha256=sha(source),
                environment_assets_sha256=sha(preflight),
                diffusion_model=str(model),
                model_provenance='Cached Manojb SD2.1 mirror; exact stabilityai fp16 snapshot equivalence not yet certified',
                caption_model=str(root / 'models/seal/blip2-flan-t5-xl'),
                embedding_model=str(root / 'models/seal/caption-embedding'),
                expected_scheduler='PNDMScheduler',
                inversion='Official backward_ddim reversed on the loaded scheduler timesteps; empty prompt CFG1, requested50',
                generation='Official pipeline defaults: requested50, CFG7.5, FP16',
                patches=1024, bits_per_patch=7, official_projection_seed=42,
                tau=2.3, minimum_matches=12,
                decision_source='ICCV2025 final Algorithm3 and Supplement9.1; d<2.3, count>=12',
                source_paper_gap='Released key derivation does not expose the final paper secret salt',
                sources=sources, cohort_protocol_sha256=sha(old / 'protocol.json'),
                gpu_indices=[7, 5, 4], predecessors=[predecessor] * 3,
                worker='run_seal_native.py', worker_sha256=sha(run / 'run_seal_native.py'),
                coordinator_sha256=sha(run / 'run_baseline_after_exit.py'),
                attacks=0, distortions=0)
    (run / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))))


if __name__ == '__main__':
    main()
