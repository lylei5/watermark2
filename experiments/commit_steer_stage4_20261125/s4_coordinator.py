"""Kernel process-exit (pidfd) 3-GPU coordinator for Stage 4 (zero busy-wait polling).
Watches active workers on GPU 0 (Stage 4D gen20), GPU 1 (Stage 4C PGID-F+NativeGS), and GPU 2 (Stage 4B Hyb100):
  - Lane GPU 0: wait gen20 -> run SemBind benign20 shard 0/2 -> claim attacks_null if not yet claimed
  - Lane GPU 2: wait Stage 4B -> wait gen20 -> run SemBind benign20 shard 1/2 -> claim attacks_null if not yet claimed
  - Lane GPU 1: wait Stage 4C -> wait gen20 -> claim attacks_null if not yet claimed
  - Post-join: merge SemBind benign20 shards -> run Stage 4 FID + independent 0-mismatch audit.
"""
import argparse
import ctypes
import errno
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
S4 = ROOT / 'runs/commit_steer_stage4_20261125'
PY = '/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python'
SYS_PIDFD_OPEN = 434


def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}][S4-COORD-3GPU] {msg}', flush=True)


def wait_pid_exit(pid: int, label: str):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
    if fd >= 0:
        try:
            p = select.poll()
            p.register(fd, select.POLLIN)
            log(f'Waiting via kernel pidfd(fd={fd}) on {label} (pid={pid})...')
            p.poll()
            log(f'Process-exit event received for {label} (pid={pid}).')
        finally:
            os.close(fd)
        return
    err = ctypes.get_errno()
    if err == errno.ESRCH:
        log(f'{label} (pid={pid}) already exited.')
        return
    log(f'pidfd_open errno={err}; waiting via tail --pid={pid} for {label}...')
    subprocess.run(['tail', f'--pid={pid}', '-f', '/dev/null'], check=False)
    log(f'Exit observed for {label} (pid={pid}).')


def wait_gpu_release(gpu: int, timeout: int = 90) -> bool:
    assert gpu in (0, 1, 2, 3, 4, 5, 7) and gpu != 6
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
        ).strip().split(', ')
        busy = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True,
        ).strip()
        if int(info[0]) < 100 and int(info[1]) == 0 and not busy:
            return True
        time.sleep(5)
    log(f'WARNING: GPU {gpu} not idle after {timeout}s; refusing to force launch.')
    return False


def run_step(cmd, cwd: Path, log_path: Path, label: str) -> int:
    log(f'Starting {label}: {" ".join(cmd)}')
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as lf:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.wait()
    log(f'Finished {label} with rc={rc}')
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid-gen20', type=int, required=True)
    ap.add_argument('--pid-s4c', type=int, required=True)
    ap.add_argument('--pid-s4b', type=int, required=True)
    args = ap.parse_args()

    gen20_done = threading.Event()
    attacks_lock = threading.Lock()
    attacks_claimed = {'done': False}
    state_lock = threading.Lock()
    status = dict(
        s4a='complete',
        s4b='running',
        s4c='running',
        s4d_gen20='running',
        s4d_benign_s0='pending',
        s4d_benign_s1='pending',
        s4d_attacks_null='pending',
        audit='pending',
    )

    def save_status():
        with state_lock:
            tmp = S4 / 'coordinator_state.json.tmp'
            tmp.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
            tmp.replace(S4 / 'coordinator_state.json')

    save_status()

    def try_run_attacks_null(gpu: int):
        if not (S4 / 'head2head_sembind/gen20_summary.json').exists():
            return
        with attacks_lock:
            if attacks_claimed['done']:
                return
            attacks_claimed['done'] = True
        if not wait_gpu_release(gpu):
            with attacks_lock:
                attacks_claimed['done'] = False
            return
        status['s4d_attacks_null'] = f'running_gpu{gpu}'
        save_status()
        rc = run_step(
            [PY, '-u', str(S4 / 's4d_sembind_head2head.py'), 'attacks_null', '--gpu', str(gpu)],
            S4 / 'head2head_sembind', S4 / f's4d_attacks_null_gpu{gpu}.log', f'Stage4D-SemBind-AttacksNull(GPU{gpu})'
        )
        status['s4d_attacks_null'] = 'complete' if rc == 0 else f'failed_rc_{rc}'
        save_status()

    def lane_gpu0():
        wait_pid_exit(args.pid_gen20, 'Stage4D-Gen20(GPU0)')
        status['s4d_gen20'] = 'complete' if (S4 / 'head2head_sembind/gen20_summary.json').exists() else 'failed'
        save_status()
        gen20_done.set()
        if status['s4d_gen20'] != 'complete':
            return
        if wait_gpu_release(0):
            status['s4d_benign_s0'] = 'running_gpu0'
            save_status()
            rc0 = run_step(
                [PY, '-u', str(S4 / 's4d_sembind_head2head.py'), 'benign20', '--gpu', '0', '--shard', '0', '--num-shards', '2'],
                S4 / 'head2head_sembind', S4 / 's4d_benign_s0_gpu0.log', 'Stage4D-SemBind-Benign-Shard0(GPU0)'
            )
            status['s4d_benign_s0'] = 'complete' if rc0 == 0 else f'failed_rc_{rc0}'
            save_status()
        try_run_attacks_null(0)

    def lane_gpu2():
        wait_pid_exit(args.pid_s4b, 'Stage4B-Hyb100(GPU2)')
        status['s4b'] = 'complete' if (S4 / 'multidim100_hyb/live_attacks/summary.json').exists() else 'failed'
        save_status()
        gen20_done.wait()
        if (S4 / 'head2head_sembind/gen20_summary.json').exists() and wait_gpu_release(2):
            status['s4d_benign_s1'] = 'running_gpu2'
            save_status()
            rc1 = run_step(
                [PY, '-u', str(S4 / 's4d_sembind_head2head.py'), 'benign20', '--gpu', '2', '--shard', '1', '--num-shards', '2'],
                S4 / 'head2head_sembind', S4 / 's4d_benign_s1_gpu2.log', 'Stage4D-SemBind-Benign-Shard1(GPU2)'
            )
            status['s4d_benign_s1'] = 'complete' if rc1 == 0 else f'failed_rc_{rc1}'
            save_status()
        try_run_attacks_null(2)

    def lane_gpu1():
        wait_pid_exit(args.pid_s4c, 'Stage4C-PGID-NativeGS(GPU1)')
        status['s4c'] = 'complete' if (S4 / 'head2head_pgid_native_gs/attacks_summary.json').exists() else 'failed'
        save_status()
        gen20_done.wait()
        try_run_attacks_null(1)

    threads = [threading.Thread(target=fn, daemon=False) for fn in (lane_gpu0, lane_gpu1, lane_gpu2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if (S4 / 'head2head_sembind/benign_shard0.json').exists() and (S4 / 'head2head_sembind/benign_shard1.json').exists():
        run_step(
            [PY, '-u', str(S4 / 's4d_sembind_head2head.py'), 'merge_benign'],
            S4 / 'head2head_sembind', S4 / 's4d_merge.log', 'Stage4D-SemBind-MergeBenign'
        )

    status['audit'] = 'running'
    save_status()
    rc_aud = run_step(
        [PY, '-u', str(S4 / 's4_audit_and_viz.py'), 'all'],
        S4, S4 / 's4_audit.log', 'Stage4-FID-And-Audit'
    )
    status['audit'] = 'complete' if rc_aud == 0 else f'failed_rc_{rc_aud}'
    save_status()
    log('ALL STAGE 4 3-GPU LANES FINISHED.')


if __name__ == '__main__':
    main()
