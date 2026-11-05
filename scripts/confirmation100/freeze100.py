"""Freeze the confirmation100 design and role runtimes. CPU only; no model imports."""
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU only')

ROOT = Path('/data0/liyanlei/watermark_research')
DESIGN_SRC = Path(sys.argv[1])
RUN_DIR = sys.argv[2]
PRIVATE_FOLDER = sys.argv[3]
VERSION = sys.argv[4] if len(sys.argv) > 4 else 'v1'

VERSIONS = {
    'torch': '2.5.1',
    'torchvision': '0.20.1',
    'diffusers': '0.31.0',
    'transformers': '4.46.3',
    'accelerate': '1.1.1',
    'Pillow': '11.0.0',
    'numpy': '2.2.6',
}
FILES = [
    'scripts/confirmation100/common100.py',
    'scripts/confirmation100/prepare100.py',
    'scripts/confirmation100/attack100.py',
    'scripts/confirmation100/evaluate100.py',
    'scripts/confirmation100/analyze100.py',
    'scripts/confirmation100/control100.py',
    'scripts/confirmation100/start100.py',
    'scripts/cover_pilot/contract.py',
    'scripts/cover_pilot/runtime.py',
    'scripts/parallel_execution/launcher.py',
    'scripts/parallel_execution/scheduler.py',
    'scripts/reference_reliability/private_state.py',
    'scripts/cross_generator/verify_message_scores.py',
    'third_party/semantic-forgery/run_imprint_forgery.py',
    'third_party/semantic-forgery/utils/imprint_utils.py',
    'third_party/semantic-forgery/utils/image_utils.py',
    'third_party/semantic-forgery/utils/pipe/pipe_utils.py',
    'third_party/semantic-forgery/utils/wm/gs_provider.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        raise SystemExit('Refusing to overwrite: ' + str(path))
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise SystemExit('Version mismatch: ' + name)
    code_hashes = {}
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit('Missing source: ' + relative)
        code_hashes[str(path)] = sha(path)
    design = json.loads(DESIGN_SRC.read_text(encoding='utf-8'))
    if design['resources']['run_dir'] != RUN_DIR or design['resources']['private_folder'] != PRIVATE_FOLDER:
        raise SystemExit('Design resource paths differ from freeze arguments')
    protocols = ROOT / 'protocols'
    design_path = protocols / ('confirmation100_design_' + VERSION + '.json')
    write_once(design_path, design)
    design_sha = sha(design_path)
    devices = design['devices']
    base = dict(stage='confirmation100', source_commit='e457085+confirmation100-' + VERSION,
                design_path=str(design_path), design_sha256=design_sha,
                devices=devices, versions=VERSIONS, code_hashes=code_hashes)
    attacker = dict(base, role='attacker')
    evaluator = dict(base, role='evaluator', private_folder=str(ROOT / PRIVATE_FOLDER))
    controller = dict(base, role='controller', run_dir=RUN_DIR,
                      private_folder=str(ROOT / PRIVATE_FOLDER),
                      watchdogs=dict(attack=9000, preparation=7200, evaluation=3600),
                      role_runtimes={
                          'attacker': dict(path='protocols/confirmation100_attacker_' + VERSION + '.json', sha256=''),
                          'evaluator': dict(path='protocols/confirmation100_evaluator_' + VERSION + '.json', sha256=''),
                      })
    for name, value in (('attacker', attacker), ('evaluator', evaluator)):
        path = protocols / ('confirmation100_' + name + '_' + VERSION + '.json')
        write_once(path, value)
        (protocols / ('confirmation100_' + name + '_' + VERSION + '.sha256')).write_text(sha(path) + '\n')
        if name == 'attacker':
            controller['role_runtimes']['attacker']['sha256'] = sha(path)
        else:
            controller['role_runtimes']['evaluator']['sha256'] = sha(path)
    controller_path = protocols / ('confirmation100_controller_' + VERSION + '.json')
    write_once(controller_path, controller)
    (protocols / ('confirmation100_controller_' + VERSION + '.sha256')).write_text(sha(controller_path) + '\n')
    print(json.dumps(dict(design_sha256=design_sha,
                          attacker_sha256=controller['role_runtimes']['attacker']['sha256'],
                          evaluator_sha256=controller['role_runtimes']['evaluator']['sha256'],
                          controller_sha256=sha(controller_path),
                          code_files=len(code_hashes), tasks=len(design['tasks'])), indent=2))


if __name__ == '__main__':
    main()
