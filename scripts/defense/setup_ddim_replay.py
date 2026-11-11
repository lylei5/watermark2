"""Isolate the recorded development runtime, gate replay, then continue."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--index-url',default='https://pypi.tuna.tsinghua.edu.cn/simple/')
    args=p.parse_args()
    root=args.root;run=args.run;primary=root/'envs/semantic_forgery_py310/bin/python'
    envroot=root/'envs/ddim_replay_py310';python=envroot/'bin/python'
    lock=(run/'runtime_setup.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        assert (envroot/'pyvenv.cfg').is_file() and python.is_file()
        assert not (run/'bridge_runtime.json').exists(),'Replay already has results; preserve them'
        archive=run/'setup_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S')
        archive.mkdir(parents=True)
        for name in ['runtime_setup_status.json','runtime_setup.log','torch_install.log',
                     'runtime_install.log','accelerate_install.log','pip_install.log']:
            if (run/name).exists():shutil.copy2(run/name,archive/name)
    index=args.index_url
    install_env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL=index,PIP_EXTRA_INDEX_URL='',
                     PIP_DISABLE_PIP_VERSION_CHECK='1')
    temp=root/'cache/runtime_tmp';temp.mkdir(exist_ok=True)
    install_env['TMPDIR']=str(temp)
    install=[python,'-m','pip','install','--index-url',index,'--only-binary=:all:',
             '--timeout','30','--retries','4']
    def status(stage,**kw):
        (run/'runtime_setup_status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2)+'\n')
    def execute(command,name,env=None):
        with (run/name).open('w') as log:
            subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    try:
        status('creating_isolated_environment')
        if not args.resume:
            assert not envroot.exists(),'Use --resume for the existing isolated environment'
            execute([primary,'-m','venv',envroot],'venv_setup.log')
        (envroot/'pip.conf').write_text('[global]\nindex-url = '+index+'\nextra-index-url =\n')
        status('installing_recorded_runtime')
        execute(install+['pip==25.0.1'],'pip_install.log',install_env)
        # Local wheels are recovered from completed downloads; no alternate network index.
        wheels=root/'cache/runtime_wheels'
        cached=[wheels/'torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl',
                wheels/'torchvision-0.16.2+cu121-cp310-cp310-linux_x86_64.whl']
        if all(path.is_file() for path in cached):
            execute(install+['--no-deps']+cached,'cached_torch_install.log',install_env)
        execute(install+['torch==2.1.2+cu121','torchvision==0.16.2+cu121',
            'numpy==1.26.4','diffusers==0.30.3',
            'transformers==4.46.3','scipy==1.15.3','huggingface-hub==0.36.2','safetensors==0.7.0',
            'Pillow==12.2.0','cryptography==48.0.0','psutil'],'runtime_install.log',install_env)
        # Reproduce the installed development combination without upgrading torch.
        execute(install+['--no-deps','accelerate==1.13.0'],'accelerate_install.log',install_env)
        execute([python,'-c','import torch, torchvision, diffusers, transformers, accelerate; '
                 'print(torch.__version__, torchvision.__version__, diffusers.__version__); '
                 'assert torch.__version__ == "2.1.2+cu121"; '
                 'assert torchvision.__version__ == "0.16.2+cu121"; '
                 'assert diffusers.__version__ == "0.30.3"'],'runtime_import_check.log',
                dict(install_env,CUDA_VISIBLE_DEVICES=''))
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
