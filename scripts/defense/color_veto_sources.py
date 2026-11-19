"""Schedule the unchanged prompt-pair renderer on idle authorized devices."""
import os
import subprocess
import sys
import time
from pathlib import Path

import color_budget_sources as renderer
from color_joint_sources import ROOT, select

RUN = ROOT/'runs/color_veto_20261119'


def idle(gpu):
    values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    memory,utilization=map(int,values.strip().split(','))
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                      '--format=csv,noheader'],text=True)
    return memory<16 and utilization==0 and not processes.strip()


def schedule():
    import fcntl
    lock=(RUN/'generation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (RUN/'sources.json').exists():
        select(RUN,[ROOT/'runs/color_joint_20261119/sources.json'],'color-veto-prompt-v1','color-veto-seed-v1')
    active={}
    while not all((RUN/f'generated_{s}.json').exists() for s in range(3)) or active:
        for gpu,(proc,log,shard) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close()
                del active[gpu]
                if code and not (RUN/f'gpu_race_{shard}').exists():
                    raise RuntimeError(f'Generation failed. See generate_{shard}.log')
        busy={entry[2] for entry in active.values()}
        pending=[s for s in range(3) if s not in busy and not (RUN/f'generated_{s}.json').exists()]
        for gpu in (4,5,7):
            if pending and gpu not in active and idle(gpu):
                s=pending.pop(0)
                (RUN/f'gpu_race_{s}').unlink(missing_ok=True)
                log=(RUN/f'generate_{s}.log').open('a')
                proc=subprocess.Popen([sys.executable,__file__,str(s),str(gpu)],stdout=log,stderr=log)
                active[gpu]=(proc,log,s)
        if active or pending:
            time.sleep(20 if active else 60)


if __name__=='__main__':
    if len(sys.argv)==1:
        schedule()
    else:
        shard,gpu=map(int,sys.argv[1:])
        renderer.RUN=RUN
        if not idle(gpu):
            (RUN/f'gpu_race_{shard}').touch()
            sys.exit(2)
        renderer.generate(shard,gpu)
