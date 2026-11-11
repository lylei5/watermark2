"""Isolate the recorded development runtime, gate replay, then continue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    root=args.root;run=args.run;primary=root/'envs/semantic_forgery_py310/bin/python'
    envroot=root/'envs/ddim_replay_py310';python=envroot/'bin/python'
    def status(stage,**kw):
        (run/'runtime_setup_status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2)+'\n')
    def execute(command,name,env=None):
        with (run/name).open('w') as log:
            subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    try:
        status('creating_isolated_environment')
        assert not envroot.exists(),'Do not overwrite an existing environment'
        execute([primary,'-m','venv',envroot],'venv_setup.log')
        status('installing_recorded_runtime')
        execute([python,'-m','pip','install','--timeout','30','--retries','1','--index-url','https://download.pytorch.org/whl/cu121',
            'torch==2.1.2+cu121','torchvision==0.16.2+cu121'],'torch_install.log')
        execute([python,'-m','pip','install','--timeout','30','--retries','1','numpy==1.26.4','diffusers==0.30.3',
            'transformers==4.46.3','scipy==1.15.3','huggingface-hub==0.36.2','safetensors==0.7.0',
            'Pillow==12.2.0','cryptography==48.0.0','psutil'],'runtime_install.log')
        # Reproduce the installed development combination without upgrading torch.
        execute([python,'-m','pip','install','--no-deps','--timeout','30','--retries','1','accelerate==1.13.0'],'accelerate_install.log')
        status('checking_development_replay')
        device=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert int(device[0])==5 and int(device[2])<=50 and int(device[3])==0,'GPU5 is occupied; no automatic retry'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device[1],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_HOME=str(root/'cache/huggingface'),PYTHONPATH=str(run/'scripts'))
        execute([python,run/'scripts/bridge_ddim_runtime.py','--run',run,'--output',run/'bridge_runtime.json'],
            'bridge_runtime.log',env)
        bridge=json.loads((run/'bridge_runtime.json').read_text())
        if not bridge['passed']:raise RuntimeError('Original bridge tolerance still fails; calibration remains unscored')
        status('continuing_frozen_validation',bridge_passed=True)
        execute([primary,run/'scripts/continue_geometry_validation.py','--root',root,'--run',run,
            '--predecessor-pid','22316','--gpu-indices','5','--cache-python',python],'continuation.log',env)
        if not (run/'complete.json').exists():raise RuntimeError('Validation ended without complete results')
        status('complete')
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
