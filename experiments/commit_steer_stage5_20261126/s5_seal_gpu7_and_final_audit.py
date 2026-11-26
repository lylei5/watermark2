"""Runs SEAL attacks_gen -> attacks_verify immediately on idle GPU 7, then waits via kernel pidfd
on s5_coordinator.py (pid 39701) and executes final merge/audit so audit_stage5.json covers ALL 5A/5B/5C/5D."""
import ctypes, errno, json, os, select, subprocess, time
from pathlib import Path

S5 = Path('/data0/liyanlei/watermark_research/runs/commit_steer_stage5_20261126')
PY = '/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python'
PY_SEAL = '/data0/liyanlei/watermark_research/envs/seal_reproduction_py310/bin/python'
SYS_PIDFD_OPEN = 434

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][S5-SEAL-GPU7] {msg}", flush=True)

def update_state(key, val):
    p = S5 / 'coordinator_state.json'
    st = json.loads(p.read_bytes()) if p.exists() else {}
    st[key] = val
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(st, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)

def wait_gpu_idle(gpu=7, timeout=90):
    assert gpu != 6
    dl = time.time() + timeout
    while time.time() < dl:
        info = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True
        ).strip().split(', ')
        busy = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True
        ).strip()
        if int(info[0]) < 100 and int(info[1]) == 0 and not busy:
            return True
        time.sleep(5)
    return False

def run_step(cmd, cwd, log_path, label):
    log(f'Starting {label}: {" ".join(cmd)}')
    with log_path.open('a', encoding='utf-8') as lf:
        rc = subprocess.Popen(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT).wait()
    log(f'Finished {label} with rc={rc}')
    return rc

def wait_pid_exit(pid: int):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
    if fd >= 0:
        try:
            p = select.poll(); p.register(fd, select.POLLIN)
            log(f'Waiting via kernel pidfd(fd={fd}) on coordinator pid={pid} before final audit...')
            p.poll()
            log(f'Coordinator pid={pid} exited.')
        finally:
            os.close(fd)
        return
    if ctypes.get_errno() == errno.ESRCH:
        log(f'Coordinator pid={pid} already exited.')
        return
    subprocess.run(['tail', f'--pid={pid}', '-f', '/dev/null'], check=False)

if wait_gpu_idle(7):
    update_state('s5a_seal_attacks_gen', 'running_gpu7')
    rc1 = run_step(
        [PY, '-u', str(S5 / 's5a_seal_head2head.py'), 'attacks_gen', '--gpu', '7'],
        S5 / 'seal_head2head', S5 / 's5a_seal_attacks_gpu7.log', 'S5A-SEAL-AttacksGen(GPU7)'
    )
    update_state('s5a_seal_attacks_gen', 'complete' if rc1 == 0 else f'failed_rc_{rc1}')
    if rc1 == 0 and wait_gpu_idle(7):
        update_state('s5a_seal_attacks_verify', 'running_gpu7')
        rc2 = run_step(
            [PY_SEAL, '-u', str(S5 / 's5a_seal_head2head.py'), 'attacks_verify', '--gpu', '7'],
            S5 / 'seal_head2head', S5 / 's5a_seal_attacks_gpu7.log', 'S5A-SEAL-AttacksVerify(GPU7)'
        )
        update_state('s5a_seal_attacks_verify', 'complete' if rc2 == 0 else f'failed_rc_{rc2}')

wait_pid_exit(39701)
update_state('merge_and_audit', 'running')
rc_aud = run_step([PY, '-u', str(S5 / 's5_audit_and_viz.py')], S5, S5 / 's5_audit.log', 'S5-Final-Audit-All')
update_state('merge_and_audit', 'complete' if rc_aud == 0 else f'failed_rc_{rc_aud}')
log('ALL STAGE 5 (INCLUDING SEAL ATTACKS ON GPU 7) + FINAL AUDIT COMPLETE.')
