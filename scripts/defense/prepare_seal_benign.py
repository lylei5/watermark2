"""Freeze all published SEAL benign conditions on the complete paired cohort."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--after-run', required=True)
    args = p.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    run = (root / 'runs' / args.run_name).resolve()
    parent = (root / 'runs' / args.after_run).resolve()
    assert run.parent == parent.parent == root / 'runs'
    run.mkdir(exist_ok=False)
    names = ['run_seal_benign.py', 'seal_paper_simhash.py', 'run_baseline_after_exit.py', 'prepare_seal_benign.py']
    for name in names:
        shutil.copy2(Path(__file__).parent / name, run / name)
    sources, parents = [], []
    for variant, name in [('released', 'seal_native_20261113_v3'), ('paper', 'seal_paper_native_20261113')]:
        directory = root / 'runs' / name
        plan = json.loads((directory / 'protocol.json').read_text())
        assert json.loads((directory / 'pipeline_state.json').read_text())['status'] == 'complete'
        parents.append(dict(run=str(directory), protocol_sha256=sha(directory / 'protocol.json'),
                            result_sha256=sha(directory / 'audited_summary.json')))
        for shard in range(3):
            result = json.loads((directory / f'native_{shard}/complete.json').read_text())
            assert result['protocol_sha256'] == parents[-1]['protocol_sha256']
            for item in result['rows']:
                decision = next(x for x in item['decisions'] if x['kind'] == 'genuine')
                image = directory / item['images']['genuine']['path']
                assert sha(image) == item['images']['genuine']['sha256']
                sources.append(dict(task=item['task'], dataset=item['dataset'], variant=variant, image=str(image),
                    image_sha256=sha(image), clean_caption=item['captions']['genuine'],
                    clean_accepted=decision['accepted'], clean_matching_patches=decision['matches']))
    assert len(sources) == 200
    tree = ast.parse((Path(plan['source']) / 'image_transformations.py').read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'distortions' for t in node.targets)]
    assert len(assignments) == 1
    conditions = ast.literal_eval(assignments[0].value)
    assert [x['name'] for x in conditions] == ['Clean', 'Rot75', 'JPEG25', 'CropScale75', 'Blur8', 'Noise0.1', 'Brightness6']
    prior = json.loads((parent / 'pipeline_state.json').read_text())
    cmdline = Path(f"/proc/{prior['pid']}/cmdline")
    command = cmdline.read_bytes() if cmdline.exists() else b''
    assert not command or str(parent).encode() in command
    predecessor = dict(pid=prior['pid'], cmdline_sha256=hashlib.sha256(command).hexdigest())
    retained = ['python', 'source', 'source_sha256', 'source_commit', 'diffusion_model', 'caption_model',
                'embedding_model', 'private_salt_file', 'private_salt_sha256']
    output = {k:plan[k] for k in retained}
    output.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='All7 released SEAL benign conditions on100 sources for each released/paper variant; unconditional denominators, not1000-source paper replication',
        sources=sorted(sources, key=lambda x:(x['task'],x['variant'])), parents=parents, conditions=conditions,
        conditions_source='Unchanged AST literal from official image_transformations.py; unchanged utils.image_distortion',
        known_code_paper_gaps=['Noise uses np.array(None).shape and uint8 arithmetic in the released call, yielding a shared offset rather than independent Gaussian pixels',
            'Blur8 is PIL GaussianBlur radius8, not a declared8x8 kernel', 'Brightness6 is ColorJitter brightness6, sampled in[0,7]'],
        local_sha256={name:sha(run / name) for name in names}, worker='run_seal_benign.py',
        worker_sha256=sha(run / 'run_seal_benign.py'), coordinator_sha256=sha(run / 'run_baseline_after_exit.py'),
        gpu_indices=[7,5,4], predecessors=[predecessor]*3, attacks=0,
        no_clean_success_filter=True, clean_replay_equality_required=True)
    (run / 'protocol.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'), sources=len(sources), conditions=len(conditions))))


if __name__ == '__main__':
    main()
