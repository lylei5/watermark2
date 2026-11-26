"""Stage 5 3-GPU Queue Coordinator (GPUs 3, 4, 5 only; strictly excludes GPU 6; zero busy-wait polling).
Wave 1 (parallel across GPUs 3, 4, 5):
  - GPU 3: SEAL lane (isolated venvs: seal_reproduction_py310 benign_all -> semantic_forgery_py310 attacks_gen -> seal_reproduction_py310 attacks_verify)
  - GPU 4: s5c_sembind_instance_and_removal.py instance_forgery (SemBind Same-Prompt Imprint151 + Reprompt+)
  - GPU 5: s5b_removal_and_instance_cs.py main_lane (WAVES8/Regen20 GS/sscd64/hyb + sscd64 Same-Prompt Imprint151 + Reprompt+)
Wave 2 (work-stealing queue claimed by each lane immediately upon Wave 1 completion, after wait_gpu_release):
  1. s5d_sdxl_imprint.py (SDXL 1024x1024 Live 151-step Imprint, 4 cases)
  2. s5c_sembind_instance_and_removal.py removal20 --shard 0 --num-shards 2
  3. s5c_sembind_instance_and_removal.py removal20 --shard 1 --num-shards 2
  4. s5b_removal_and_instance_cs.py same_prompt_hyb (hyb Same-Prompt Imprint151, 4 cases)
Post-join:
  - s5c_sembind_instance_and_removal.py merge_removal
  - s5_audit_and_viz.py (Same-Prompt Montage + Independent 0-Mismatch Audit)
"""
import argparse
import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
S5 = ROOT / 'runs/commit_steer_stage5_20261126'
PY = '/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python'
PY_SEAL = '/data0/liyanlei/watermark_research/envs/seal_reproduction_py310/bin/python'


def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}][S5-COORD] {msg}', flush=True)


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
    ap.add_argument('--gpus', nargs=3, type=int, default=[3, 4, 5])
    args = ap.parse_args()
    g_seal, g_sb, g_cs = args.gpus
    for g in args.gpus:
        assert g in (0, 1, 2, 3, 4, 5, 7) and g != 6, f'Forbidden GPU {g}'

    state_lock = threading.Lock()
    status = dict(
        s5a_seal_benign='pending',
        s5a_seal_attacks_gen='pending',
        s5a_seal_attacks_verify='pending',
        s5c_sembind_instance='pending',
        s5b_cs_main='pending',
        s5d_sdxl_imprint='pending',
        s5c_sembind_rem_s0='pending',
        s5c_sembind_rem_s1='pending',
        s5b_hyb_imprint='pending',
        merge_and_audit='pending',
    )

    def save_status():
        with state_lock:
            tmp = S5 / 'coordinator_state.json.tmp'
            tmp.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
            tmp.replace(S5 / 'coordinator_state.json')

    save_status()

    wave2_lock = threading.Lock()
    wave2_queue = deque([
        ('s5d_sdxl_imprint', lambda gpu: ([PY, '-u', str(S5 / 's5d_sdxl_imprint.py'), '--gpu', str(gpu)], S5 / 'sdxl_imprint', S5 / f's5d_sdxl_imprint_gpu{gpu}.log', f'S5D-SDXL-Imprint(GPU{gpu})')),
        ('s5c_sembind_rem_s0', lambda gpu: ([PY, '-u', str(S5 / 's5c_sembind_instance_and_removal.py'), 'removal20', '--gpu', str(gpu), '--shard', '0', '--num-shards', '2'], S5 / 'sembind_instance_and_removal', S5 / f's5c_rem_s0_gpu{gpu}.log', f'S5C-SemBind-Rem-S0(GPU{gpu})')),
        ('s5c_sembind_rem_s1', lambda gpu: ([PY, '-u', str(S5 / 's5c_sembind_instance_and_removal.py'), 'removal20', '--gpu', str(gpu), '--shard', '1', '--num-shards', '2'], S5 / 'sembind_instance_and_removal', S5 / f's5c_rem_s1_gpu{gpu}.log', f'S5C-SemBind-Rem-S1(GPU{gpu})')),
        ('s5b_hyb_imprint', lambda gpu: ([PY, '-u', str(S5 / 's5b_removal_and_instance_cs.py'), 'same_prompt_hyb', '--gpu', str(gpu)], S5 / 'removal_and_instance_cs', S5 / f's5b_hyb_imp_gpu{gpu}.log', f'S5B-Hyb-SamePrompt-Imprint(GPU{gpu})')),
    ])

    def drain_wave2(gpu: int):
        while True:
            with wave2_lock:
                if not wave2_queue:
                    return
                key, builder = wave2_queue.popleft()
            if not wait_gpu_release(gpu):
                with wave2_lock:
                    wave2_queue.appendleft((key, builder))
                return
            status[key] = f'running_gpu{gpu}'
            save_status()
            cmd, cwd, lpath, label = builder(gpu)
            rc = run_step(cmd, cwd, lpath, label)
            status[key] = 'complete' if rc == 0 else f'failed_rc_{rc}'
            save_status()

    def seal_lane(gpu: int):
        steps = [
            ('s5a_seal_benign', [PY_SEAL, '-u', str(S5 / 's5a_seal_head2head.py'), 'benign_all', '--gpu', str(gpu)], f'S5A-SEAL-Benign(GPU{gpu})'),
            ('s5a_seal_attacks_gen', [PY, '-u', str(S5 / 's5a_seal_head2head.py'), 'attacks_gen', '--gpu', str(gpu)], f'S5A-SEAL-AttacksGen(GPU{gpu})'),
            ('s5a_seal_attacks_verify', [PY_SEAL, '-u', str(S5 / 's5a_seal_head2head.py'), 'attacks_verify', '--gpu', str(gpu)], f'S5A-SEAL-AttacksVerify(GPU{gpu})'),
        ]
        for key, cmd, label in steps:
            if not wait_gpu_release(gpu):
                status[key] = f'skipped_gpu{gpu}_busy'
                save_status()
                return
            status[key] = f'running_gpu{gpu}'
            save_status()
            rc = run_step(cmd, S5 / 'seal_head2head', S5 / f's5a_seal_gpu{gpu}.log', label)
            status[key] = 'complete' if rc == 0 else f'failed_rc_{rc}'
            save_status()
            if rc != 0 and key == 's5a_seal_attacks_gen':
                break
        drain_wave2(gpu)

    def simple_lane(gpu: int, key: str, cmd, cwd: Path, lpath: Path, label: str):
        if wait_gpu_release(gpu):
            status[key] = f'running_gpu{gpu}'
            save_status()
            rc = run_step(cmd, cwd, lpath, label)
            status[key] = 'complete' if rc == 0 else f'failed_rc_{rc}'
            save_status()
        else:
            status[key] = f'skipped_gpu{gpu}_busy'
            save_status()
            return
        drain_wave2(gpu)

    threads = [
        threading.Thread(target=seal_lane, args=(g_seal,), daemon=False),
        threading.Thread(
            target=simple_lane,
            args=(g_sb, 's5c_sembind_instance', [PY, '-u', str(S5 / 's5c_sembind_instance_and_removal.py'), 'instance_forgery', '--gpu', str(g_sb)], S5 / 'sembind_instance_and_removal', S5 / f's5c_instance_gpu{g_sb}.log', f'S5C-SemBind-Instance(GPU{g_sb})'),
            daemon=False,
        ),
        threading.Thread(
            target=simple_lane,
            args=(g_cs, 's5b_cs_main', [PY, '-u', str(S5 / 's5b_removal_and_instance_cs.py'), 'main_lane', '--gpu', str(g_cs)], S5 / 'removal_and_instance_cs', S5 / f's5b_cs_main_gpu{g_cs}.log', f'S5B-CS-Main(GPU{g_cs})'),
            daemon=False,
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status['merge_and_audit'] = 'running'
    save_status()
    sb_dir = S5 / 'sembind_instance_and_removal'
    if (sb_dir / 'removal_shard0.json').exists() and (sb_dir / 'removal_shard1.json').exists():
        run_step(
            [PY, '-u', str(S5 / 's5c_sembind_instance_and_removal.py'), 'merge_removal'],
            sb_dir, S5 / 's5c_merge.log', 'S5C-SemBind-MergeRemoval',
        )
    rc_aud = run_step(
        [PY, '-u', str(S5 / 's5_audit_and_viz.py')],
        S5, S5 / 's5_audit.log', 'S5-Audit-And-Viz',
    )
    status['merge_and_audit'] = 'complete' if rc_aud == 0 else f'failed_rc_{rc_aud}'
    save_status()
    log('ALL STAGE 5 3-GPU LANES FINISHED.')


if __name__ == '__main__':
    main()
