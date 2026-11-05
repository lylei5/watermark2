"""Contracts for the frozen confirmation100 baseline. Importing does not initialize CUDA."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'cover_pilot'))
import contract as c
import runtime as r
sys.path.insert(0, str(BASE / 'parallel_execution'))
import launcher as launch
import scheduler as sched

ROOT = r.ROOT
POINTS = (20, 30)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tasks(d):
    return d['tasks']


def update_code(path):
    tree = ast.parse(Path(path).read_bytes())
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    body = []
    for node in loop.body:
        if isinstance(node, ast.If):
            break
        body.append(node)
    if len(body) != 5:
        raise ValueError('Official update changed')
    return compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec')


def guard(scheduler, records):
    original = scheduler.set_timesteps

    def checked(*a, **kw):
        result = original(*a, **kw)
        ts = scheduler.timesteps.tolist()
        if ts != list(range(1, 982, 20)):
            raise ValueError('Unexpected actual inverse timestep array')
        records.append(dict(timesteps=ts, endpoint_alpha_cumprod=float(scheduler.alphas_cumprod[ts[-1]])))
        return result
    scheduler.set_timesteps = checked


def design(cfg):
    path = Path(cfg['design_path'])
    if c.sha(path) != cfg['design_sha256']:
        raise ValueError('Design changed')
    d = c.read(path)
    if d['schema'] != 1 or d['stage'] != 'confirmation100_baseline':
        raise ValueError('Wrong design')
    n = len(d['tasks'])
    if not 20 <= n <= 100 or [t['id'] for t in d['tasks']] != list(range(1, n + 1)):
        raise ValueError('Wrong task inventory')
    return d


def gate(runtime, out, role):
    cfg = r.verify_runtime(Path(runtime))
    if cfg['role'] != role:
        raise ValueError('Wrong role: expected ' + role)
    d = design(cfg)
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES')
    matches = [x for x in cfg['devices'] if x['uuid'] == uuid]
    if len(matches) != 1:
        raise ValueError('Unauthorized GPU UUID')
    dev = sched.Device(**matches[0])
    expected = dict(CUDA_VISIBLE_DEVICES=dev.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if dev.index == 6 or any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong environment')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    idle = sched.verify_idle(dev, launch.sample, rows.append)
    c.write(out / 'device_preimport.json', dict(device=dev.__dict__, idle=idle, samples=rows))
    if not idle:
        raise RuntimeError('Child device busy; preserve no retry')
    return cfg, d


def by_id(d):
    return {t['id']: t for t in d['tasks']}


def validate_job(path, cfg, d):
    path = Path(path)
    job = c.read(path)
    c.keys(job, ['schema', 'task_id', 'seed', 'reference', 'cover', 'proxy', 'native'])
    t = by_id(d).get(job['task_id'])
    if job['schema'] != 1 or t is None:
        raise ValueError('Unknown task')
    if job['seed'] != t['attack_seed'] or job['proxy'] != d['proxy'] or job['native'] != d['native']:
        raise ValueError('Frozen job changed')
    for kind in ('reference', 'cover'):
        if job[kind] != c.png_record(path.parent / (kind + '.png')):
            raise ValueError('Input changed')
    return job


def candidates(root, count=100):
    root = Path(root)
    rows = []
    for sid in range(1, count + 1):
        out = root / ('attack_' + str(sid))
        job = root / 'attack_inputs' / str(sid) / 'job.json'
        frozen = c.read(out / 'candidates_frozen.json')
        if frozen['job_sha256'] != c.sha(job) or not c.read(out / 'execution.json')['outcome']['completed']:
            raise ValueError('Unsealed job')
        for n in POINTS:
            record = c.png_record(out / ('update_' + str(n) + '.png'))
            if frozen['images'][str(n)] != record:
                raise ValueError('PNG changed')
            rows.append(dict(task_id=sid, actual_update=n,
                             path='attack_' + str(sid) + '/update_' + str(n) + '.png',
                             image=record))
    return rows


