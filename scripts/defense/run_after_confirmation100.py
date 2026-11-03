"""Wait for the confirmation100 run to finish, then run the SemBind smoke on the first idle GPU."""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/confirmation100_baseline_20261102T204500Z'
STATUS = RUN / 'scheduler_status.json'
PY = ROOT / 'envs/semantic_forgery_py310/bin/python'


def state():
    try:
        return json.loads(STATUS.read_text()).get('state')
    except Exception:  # noqa: BLE001
        return None


while state() not in ('completed', 'stopped'):
    time.sleep(60)


def idle_gpu():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 4:
            continue
        idx, uuid, mem, util = parts
        if int(idx) in (2, 3, 4, 5, 7) and int(mem) < 500 and int(util) == 0:
            return uuid
    return None


uuid = None
while uuid is None:
    uuid = idle_gpu()
    if uuid is None:
        time.sleep(30)

ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = ROOT / 'runs' / ('sembind_smoke_' + ts)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
cmd = [str(PY), str(ROOT / 'scripts/defense/sembind_smoke.py'), str(out), '30', 'cuda:0']
with open(str(out) + '.log', 'w', encoding='utf-8') as log:
    rc = subprocess.run(cmd, cwd=str(ROOT / 'scripts/defense'), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print('SEMBIND_SMOKE_RC', rc, 'OUT', out, flush=True)
