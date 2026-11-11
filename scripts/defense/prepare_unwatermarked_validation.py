"""Freeze all 100 existing COCO negatives without changing detector parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--overrides', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root, args.run
    assert not any((run / name).exists() for name in ['protocol.json', 'status.json', 'complete.json'])
    prior_run = root / 'runs/gs_published_conditions_20261111'
    prior = json.loads((prior_run / 'protocol.json').read_text())
    keys = ['nominal_alpha', 'detector', 'baseline', 'bridge_rows', 'bridge_max_relative_l2', 'bridge_rule',
            'server_model', 'runtime_expected', 'inversion', 'pipeline_scope', 'cache_runtime_pins',
            'supplementary_ba_controls', 'identity_tasks', 'identity_root', 'execution_gpus']
    plan = {key: prior[key] for key in keys}
    run.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prior_run / 'scripts', run / 'scripts', dirs_exist_ok=True)
    for path in args.overrides.glob('*.py'):
        shutil.copy2(path, run / 'scripts' / path.name)
    if not (run / 'bridge').exists():
        (run / 'bridge').symlink_to((prior_run / 'bridge').resolve(), target_is_directory=True)
    dataset = root / 'datasets/confirmation100_relay_v11/rgb512'
    if not (run / 'images').exists():
        (run / 'images').symlink_to(dataset, target_is_directory=True)
    inputs = sorted(dataset.glob('*.png'))
    assert len(inputs) == 100
    known = json.loads((root / 'runs/semantic_codes_cpu_20261109/records.json').read_text())
    known = {r['file']: r['sha256'] for r in known if r['group'] == 'natural' and r['variant'] == 'clean'}
    assert set(known).issubset({p.name for p in inputs})
    captions = json.loads((dataset.parent / 'metadata/captions_val2017.json').read_text())
    coco_ids = {r['id'] for r in captions['images']}
    assert all(int(p.stem) in coco_ids for p in inputs)
    rows = []
    for path in inputs:
        digest = sha(path)
        if path.name in known:
            assert known[path.name] == digest
        rows.append(dict(id='natural_' + path.stem, task=int(path.stem), method='sembind_scattered',
                         label='unwatermarked_natural', split='unwatermarked', image='images/' + path.name,
                         image_sha256=digest))
    plan.update(evaluation_split='unwatermarked', rows=rows, queries=100, identity_decisions=6400,
        input_population='All100 existing COCO2017 natural RGB512 development assets',
        prior_semantic_cpu_image_hash_matches=len(known), coco_metadata_ids_verified=True,
        independent_confirmation=False, input_transforms='None beyond existing RGB512 assets',
        new_attacks_run=False, novel_or_modified_attacks=False, thresholds_changed=False,
        prior_protocol_sha256=sha(prior_run / 'protocol.json'),
        source_sha256={p.name: sha(p) for p in sorted((run / 'scripts').glob('*.py'))})
    (run / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(protocol_sha256=sha(run / 'protocol.json'), queries=100, decisions=6400)))


if __name__ == '__main__':
    main()
