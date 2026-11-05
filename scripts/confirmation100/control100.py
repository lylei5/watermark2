import os

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU supervisor required')

import concurrent.futures
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from common100 import c, r, ROOT, HERE, launch, sched, design, candidates, tasks, validate_job


def status(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class Runner(launch.ProcessRunner):
    def __init__(self, path):
        super().__init__(ROOT, path, '', {}, {})
        self.commands = {}

    def command(self, job, device, out):
        entry, args = self.commands[job.name]
        return [sys.executable, str(HERE / entry)] + [str(a) for a in args] + [str(out)]


def run(runtime):
    launch.check_host(ROOT)
    cfg = r.verify_runtime(Path(runtime))
    d = design(cfg)
    root = ROOT / cfg['run_dir']
    private = ROOT / cfg['private_folder']
    devices = [sched.Device(**x) for x in cfg['devices']]
    if len(devices) != 5 or len({x.index for x in devices}) != 5:
        raise ValueError('Expected five distinct devices')
    if any(x.index == 6 for x in devices):
        raise ValueError('GPU 6 excluded')
    attacker_runtime = ROOT / cfg['role_runtimes']['attacker']['path']
    evaluator_runtime = ROOT / cfg['role_runtimes']['evaluator']['path']
    for role in ('attacker', 'evaluator'):
        info = cfg['role_runtimes'][role]
        if c.sha(ROOT / info['path']) != info['sha256']:
            raise ValueError('Role runtime changed: ' + role)
    watchdogs = cfg.get('watchdogs', {})
    attack_watchdog = int(watchdogs.get('attack', 9000))
    prepare_watchdog = int(watchdogs.get('preparation', 7200))
    evaluation_watchdog = int(watchdogs.get('evaluation', 3600))
    runner = Runner(runtime)
    finished = []
    stop = threading.Event()

    def idle_wait(device, label, done):
        while True:
            if stop.is_set() or done.is_set():
                return False
            samples = []
            if sched.verify_idle(device, launch.sample, samples.append):
                return True
            status(root / ('gpu' + str(device.index) + '_waiting.json'),
                   dict(state='waiting_resource', task=label, checked_at_utc=sched.now(), samples=samples))
            time.sleep(30)

    def child(name, entry, args, role, device, marker, phase, watchdog, limit):
        rolepath = ROOT / cfg['role_runtimes'][role]['path']
        if c.sha(rolepath) != cfg['role_runtimes'][role]['sha256']:
            raise ValueError('Role runtime changed before child')
        out = root / name
        out.mkdir()
        c.write(out / 'claim.json', dict(task=name, gpu_uuid=device.uuid, started_at_utc=sched.now()))
        runner.commands[name] = (entry, args)
        runner.jobs[name] = dict(watchdog_seconds=watchdog, output_limit_bytes=limit, completion_marker=marker)
        runner.phases[name] = 'attack' if phase == 'native_attack' else phase
        env = dict(CUDA_VISIBLE_DEVICES=device.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                   HF_HOME=str(ROOT / 'cache/huggingface'), PYTHONDONTWRITEBYTECODE='1',
                   TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        status(root / ('gpu' + str(device.index) + '_status.json'),
               dict(state='running', task=name, started_at_utc=sched.now()))
        start = sched.now()
        outcome = runner(sched.Job(name, c.sha(runtime)), device, out, env)
        c.write(out / 'execution.json', dict(started_at_utc=start, outcome=outcome,
                                             gpu_index=device.index, gpu_uuid=device.uuid))
        c.write(out / 'execution_seal.json', dict(sha256=c.sha(out / 'execution.json')))
        if not outcome['completed']:
            stop.set()
            raise RuntimeError('Launched task failed; no retry: ' + name)
        finished.append(name)
        print('CHILD_COMPLETED', name, flush=True)

    with sched.project_lock(ROOT / 'runs/.imprint_delivery.lock'):
        if root.exists() or private.exists():
            raise ValueError('Existing run; no repeated launch')
        used = sum(launch.tree_bytes(ROOT / n) for n in ('runs', 'reports', 'scripts', 'protocols')) + \
            sum(p.stat().st_size for p in ROOT.glob('*.zip'))
        if used + 2 * 1024 ** 3 > 10737418240 or shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
            raise ValueError('Storage preflight failed')
        root.mkdir()
        private.mkdir(mode=0o700)
        c.write(root / 'launch_intent.json', dict(source_commit=cfg['source_commit'],
                                                  runtime_sha256=c.sha(runtime),
                                                  controller_pid=os.getpid(),
                                                  started_at_utc=sched.now(),
                                                  max_children=5, dispatch='dynamic_work_queue',
                                                  storage_before_bytes=used))
        c.write(root / 'design.json', d)
        try:
            status(root / 'scheduler_status.json', dict(state='preparation', completed=finished))
            child('prepare', 'prepare100.py', [evaluator_runtime, private], 'evaluator', devices[0],
                  'COMPLETED_NATIVE_PREPARE', 'preparation', prepare_watchdog, 1024 ** 3)
            inputs = root / 'attack_inputs'
            inputs.mkdir()
            for task in tasks(d):
                sid = task['id']
                folder = inputs / str(sid)
                folder.mkdir()
                for kind in ('reference', 'cover'):
                    with (folder / (kind + '.png')).open('xb') as f:
                        f.write((root / 'prepare' / str(sid) / (kind + '.png')).read_bytes())
                job = dict(schema=1, task_id=sid, seed=task['attack_seed'],
                           reference=c.png_record(folder / 'reference.png'),
                           cover=c.png_record(folder / 'cover.png'),
                           proxy=d['proxy'], native=d['native'])
                c.write(folder / 'job.json', job)
                validate_job(folder / 'job.json', r.verify_runtime(attacker_runtime), d)
            c.write(root / 'attack_inputs_sealed.json',
                    {p.relative_to(inputs).as_posix(): c.sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()})
            ids = [task['id'] for task in tasks(d)]
            status(root / 'scheduler_status.json', dict(state='attacking', completed=finished,
                                                        pending=len(ids), dispatch='dynamic_work_queue'))
            pending = iter(ids)
            attack_lock = threading.Lock()
            attack_started = [0]
            attack_done = threading.Event()

            def attack_worker(device):
                while True:
                    if stop.is_set() or attack_done.is_set():
                        return
                    if not idle_wait(device, 'attack_queue', attack_done):
                        return
                    with attack_lock:
                        sid = next(pending, None)
                        if sid is not None:
                            attack_started[0] += 1
                            if attack_started[0] == len(ids):
                                attack_done.set()
                    if sid is None:
                        attack_done.set()
                        return
                    child('attack_' + str(sid), 'attack100.py', [attacker_runtime, inputs / str(sid) / 'job.json'],
                          'attacker', device, 'COMPLETED_NATIVE_ATTACK', 'native_attack',
                          attack_watchdog, 256 * 1024 ** 2)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(attack_worker, device) for device in devices]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if attack_started[0] != len(ids):
                raise RuntimeError('Work queue left unstarted tasks: ' + str(len(ids) - attack_started[0]))
            c.write(root / 'candidates_sealed.json', dict(rows=candidates(root, len(ids)), sealed_at_utc=sched.now(),
                                                          target_scores_seen=0))
            status(root / 'scheduler_status.json', dict(state='evaluating', completed=finished))

            shards = iter(range(5))
            eval_lock = threading.Lock()
            eval_done = threading.Event()

            def eval_worker(device):
                while True:
                    if stop.is_set() or eval_done.is_set():
                        return
                    if not idle_wait(device, 'evaluation_queue', eval_done):
                        return
                    with eval_lock:
                        k = next(shards, None)
                    if k is None:
                        eval_done.set()
                        return
                    child('evaluate_' + str(k), 'evaluate100.py', [evaluator_runtime, root, private, k, 5],
                          'evaluator', device, 'COMPLETED_NATIVE_EVAL_SHARD', 'evaluation',
                          evaluation_watchdog, 512 * 1024 ** 2)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(eval_worker, device) for device in devices]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if len(list(root.glob('evaluate_*/summary.json'))) != 5:
                raise RuntimeError('Evaluation shards incomplete')
            status(root / 'scheduler_status.json', dict(state='analyzing', completed=finished))
            from analyze100 import analyze
            analyze(root, private)
            c.write(root / 'results_sealed.json',
                    dict(analysis_sha256=c.sha(root / 'analysis.json'),
                         executions={p.parent.name: c.sha(p) for p in root.glob('*/execution.json')},
                         finished_at_utc=sched.now()))
            status(root / 'scheduler_status.json', dict(state='completed', completed=finished))
        except BaseException as exc:
            stop.set()
            status(root / 'scheduler_status.json', dict(state='stopped', completed=finished, reason=str(exc),
                                                        retry_allowed=False, checked_at_utc=sched.now()))
            raise


if __name__ == '__main__':
    run(Path(sys.argv[1]))

