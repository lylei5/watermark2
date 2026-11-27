"""Stage 6 Zero-Collision Event-Driven Coordinator (strictly excludes GPU 6; max 3 concurrent GPUs).

Checks idle GPUs dynamically before EVERY job launch. Uses POSIX kernel-blocking os.waitpid(-1, 0)
(zero busy-waiting while workers run). If no allowed GPUs are currently idle at startup,
exits immediately with rc=2 (no polling loop!).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage6_20261126'
PY_MAIN = '/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python'
PY_SEAL = '/data0/liyanlei/watermark_research/envs/seal_reproduction_py310/bin/python'
ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)
MAX_CONCURRENT = 3


def write_state(state: dict):
    p = RUN / 'coordinator_state.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def get_idle_gpus(exclude: set) -> list:
    idle = []
    for idx in ALLOWED_GPUS:
        if idx == 6 or idx in exclude:
            continue
        info = subprocess.check_output(
            ['nvidia-smi', '-i', str(idx), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
        ).strip().split(', ')
        busy = subprocess.check_output(
            ['nvidia-smi', '-i', str(idx), '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True,
        ).strip()
        if int(info[0]) < 100 and int(info[1]) == 0 and not busy:
            idle.append(idx)
    return idle


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    # 1. CPU-only prepares first
    for cmd_desc, cmd in [
        ('s6a_prepare', [PY_MAIN, '-u', str(RUN / 's6a_coco1k_fid.py'), 'prepare']),
        ('s6b_prepare', [PY_MAIN, '-u', str(RUN / 's6b_imprint10_suite.py'), 'prepare']),
    ]:
        rc = subprocess.run(cmd).returncode
        assert rc == 0, f'{cmd_desc} failed rc={rc}'

    state_p = RUN / 'coordinator_state.json'
    state = json.loads(state_p.read_bytes()) if state_p.exists() else {}

    # Ordered task graph: (name, deps, py_bin, script, base_args, needs_gpu)
    tasks = [
        # Priority 1: COCO-1K FID (~1.9h on 3 GPUs)
        ('coco1k_s0', [], PY_MAIN, 's6a_coco1k_fid.py', ['run_shard', '--shard', '0', '--num-shards', '3'], True),
        ('coco1k_s1', [], PY_MAIN, 's6a_coco1k_fid.py', ['run_shard', '--shard', '1', '--num-shards', '3'], True),
        ('coco1k_s2', [], PY_MAIN, 's6a_coco1k_fid.py', ['run_shard', '--shard', '2', '--num-shards', '3'], True),
        ('coco1k_fid', ['coco1k_s0', 'coco1k_s1', 'coco1k_s2'], PY_MAIN, 's6a_coco1k_fid.py', ['finalize_fid'], True),

        # Priority 2: Same-Prompt Imprint N=10 (6 new cases each -> merges with Stage 5 0..3)
        ('same_sembind', [], PY_MAIN, 's6b_imprint10_suite.py', ['same_sembind'], True),
        ('same_hyb', [], PY_MAIN, 's6b_imprint10_suite.py', ['same_hyb'], True),
        ('same_sscd64', [], PY_MAIN, 's6b_imprint10_suite.py', ['same_sscd64'], True),

        # Priority 3: Cross-Prompt Imprint N=10 (all 6 SD2.1 methods)
        ('cross_sscd64_p0', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_sscd64_p0'], True),
        ('cross_sscd64_p1', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_sscd64_p1'], True),
        ('cross_hyb', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_hyb'], True),
        ('cross_native_pgid', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_native_pgid'], True),
        ('cross_sembind', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_sembind'], True),
        ('cross_seal_gen', [], PY_MAIN, 's6b_imprint10_suite.py', ['cross_seal_gen'], True),
        ('cross_seal_verify', ['cross_seal_gen'], PY_SEAL, 's6b_seal_verify.py', [], True),

        # Final CPU merge + 0-mismatch audit
        ('imprint10_merge', [
            'same_sembind', 'same_hyb', 'same_sscd64',
            'cross_sscd64_p0', 'cross_sscd64_p1', 'cross_hyb',
            'cross_native_pgid', 'cross_sembind', 'cross_seal_verify',
        ], PY_MAIN, 's6b_imprint10_suite.py', ['merge_all'], False),
        ('audit_stage6', ['coco1k_fid', 'imprint10_merge'], PY_MAIN, 's6_audit_and_viz.py', [], False),
    ]

    for name, *_ in tasks:
        if state.get(name, 'pending') != 'complete':
            state[name] = 'pending'
    write_state(state)

    # Check if at least one allowed GPU is free before starting
    gpu_tasks_pending = any(needs_gpu and state[name] != 'complete' for name, _, _, _, _, needs_gpu in tasks)
    if gpu_tasks_pending and not get_idle_gpus(set()):
        print('[S6 Coordinator] No allowed GPUs (0,1,2,3,4,5,7) currently idle. Exiting cleanly (rc=2) without polling.', flush=True)
        sys.exit(2)

    active = {}  # pid -> (name, proc, gpu_or_None, log_fp)

    def launch_ready():
        in_use_gpus = {v[2] for v in active.values() if v[2] is not None}
        idle_gpus = get_idle_gpus(in_use_gpus)
        for name, deps, py_bin, script, base_args, needs_gpu in tasks:
            if state[name] != 'pending':
                continue
            if any(state[d] != 'complete' for d in deps):
                continue
            if needs_gpu:
                if len(in_use_gpus) >= MAX_CONCURRENT or not idle_gpus:
                    continue
                gpu = idle_gpus.pop(0)
                in_use_gpus.add(gpu)
                cmd = [py_bin, '-u', str(RUN / script), *base_args, '--gpu', str(gpu)]
                log_p = RUN / f'log_{name}_gpu{gpu}.log'
            else:
                gpu = None
                cmd = [py_bin, '-u', str(RUN / script), *base_args]
                log_p = RUN / f'log_{name}_cpu.log'

            log_fp = log_p.open('a', encoding='utf-8')
            proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
            active[proc.pid] = (name, proc, gpu, log_fp)
            state[name] = f'running_gpu{gpu}' if gpu is not None else 'running_cpu'
            write_state(state)
            print(f'[{time.strftime("%H:%M:%S")}] LAUNCH {name} pid={proc.pid} gpu={gpu}', flush=True)

    launch_ready()
    while active:
        pid, status = os.waitpid(-1, 0)  # Kernel blocks until a child exits (zero CPU polling)
        if pid not in active:
            continue
        name, proc, gpu, log_fp = active.pop(pid)
        rc = os.waitstatus_to_exitcode(status)
        proc.returncode = rc
        log_fp.close()
        state[name] = 'complete' if rc == 0 else f'failed_rc{rc}'
        write_state(state)
        print(f'[{time.strftime("%H:%M:%S")}] EXIT {name} rc={rc} gpu={gpu}', flush=True)
        launch_ready()

    failed = [k for k, v in state.items() if str(v).startswith('failed')]
    pending = [k for k, v in state.items() if v == 'pending']
    if failed:
        print(f'[S6 Coordinator] FINISHED WITH FAILURES: {failed}', flush=True)
        sys.exit(1)
    if pending:
        print(f'[S6 Coordinator] Stopped early (no idle GPU for remaining tasks): {pending}', flush=True)
        sys.exit(2)
    print('[S6 Coordinator] ALL STAGE 6 TASKS COMPLETE & AUDITED (0 MISMATCH)!', flush=True)


if __name__ == '__main__':
    main()
