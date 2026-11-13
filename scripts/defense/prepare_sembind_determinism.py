"""Freeze diagnostic inversion repeats on three inputs with observed replay variation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);p.add_argument('--after-run',required=True)
    args=p.parse_args();root=Path('/data0/liyanlei/watermark_research')
    run=(root/'runs'/args.run_name).resolve();parent=(root/'runs'/args.after_run).resolve()
    assert run.parent==parent.parent==root/'runs';run.mkdir(exist_ok=False)
    old=root/'runs/sembind_prc_native_20261113_v3';previous=json.loads((old/'protocol.json').read_text());cases=[]
    for task,kind in [(4,'prc'),(4,'prc_s'),(1,'null')]:
        rows=json.loads((old/f'native_{task%3}/complete.json').read_text())['rows']
        row=next(r for r in rows if r['task']==task and r['image_kind']==kind)
        cases.append(dict(task=task,kind=kind,image=row['image'],image_sha256=row['image_sha256'],seed=202611130000+task))
    names=['run_sembind_determinism.py','prepare_sembind_determinism.py','run_baseline_after_exit.py','sembind_official_audit.py','codeword_binding.py']
    for name in names:shutil.copy2(Path(__file__).parent/name,run/name)
    pid=json.loads((parent/'pipeline_state.json').read_text())['pid'];file=Path(f'/proc/{pid}/cmdline');command=file.read_bytes() if file.exists() else b''
    assert not command or str(parent).encode() in command
    predecessor=dict(pid=pid,cmdline_sha256=hashlib.sha256(command).hexdigest())
    plan={k:previous[k] for k in ['python','paths','official_sha256','scheduler_sha256']}
    plan.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='3 deliberately selected diagnostic inputs, each2 default and2 deterministic full50-step DPM/VAE100 inversions; no performance estimate',
        cases=cases,backend_modes=['default','deterministic'],repeats_per_mode=2,cublas_workspace_config=':4096:8',
        numeric_algorithm_unchanged=True,source_protocol_sha256=sha(old/'protocol.json'),
        local_sha256={name:sha(run/name) for name in names},worker='run_sembind_determinism.py',
        worker_sha256=sha(run/'run_sembind_determinism.py'),coordinator_sha256=sha(run/'run_baseline_after_exit.py'),
        gpu_indices=[7,5,4],predecessors=[predecessor]*3,attack_jobs=0)
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))))


if __name__=='__main__':main()
