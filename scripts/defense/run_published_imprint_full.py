"""Run the unchanged CVPR2025 Imprint update on public benchmark inputs.

The released update is compiled directly from the pinned source. All 151
updates and the released checkpoint cadence are required. No detector, key,
semantic model, candidate selection, or early termination enters this process.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from PIL import Image

REVISION = 'ca68950cd7b53b8438a49580f134b2a0db28d778'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p = Path(p)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def official_update(upstream):
    source = upstream/'run_imprint_forgery.py'
    tree = ast.parse(source.read_bytes())
    loop = next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    body=[]
    for node in loop.body:
        if isinstance(node,ast.If): break
        body.append(node)
    expected=ast.parse('optim.zero_grad()\ninverted_latent = diffpipe(z0, "", guidance_scale=1.)\nl = torch.nn.functional.mse_loss(inverted_latent, zT_retrieved.detach())\nl.backward()\noptim.step()\n')
    actual=ast.Module(body=body,type_ignores=[])
    if ast.dump(actual)!=ast.dump(expected): raise ValueError('Published update differs from pinned contract')
    return compile(actual,str(source),'exec')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True,choices=range(3))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(args.protocol.read_text())
    assert plan['updates']==151 and plan['inverse_steps']==50 and plan['lr']==.01
    assert plan['max_concurrent_gpus']==3 and plan['excluded_gpu_indices']==[6]
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    info=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if int(info[0]) not in plan['worker_gpu_indices'] or int(info[2])>50 or int(info[3])!=0:
        raise RuntimeError('Worker GPU is not allowed and idle')
    current=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if current!=REVISION: raise RuntimeError('Wrong published source revision')
    for name in ['run_imprint_forgery.py','utils/imprint_utils.py','utils/image_utils.py','utils/utils.py']:
        expected=subprocess.check_output(['git','-C',str(args.upstream),'show',REVISION+':'+name])
        if hashlib.sha256(expected).hexdigest()!=sha(args.upstream/name): raise RuntimeError('Published source edited: '+name)
    update=official_update(args.upstream)
    sys.path.insert(0,str(args.upstream))
    from utils import imprint_utils as official
    from utils.image_utils import PIL_to_torch
    from utils.utils import set_random_seed
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    set_random_seed(1)
    pipe,forward,inverse=official.load_attacker_pipe(modelid=args.model,scheduler='DDIM',device='cuda:0')
    diffpipe=official.DiffPipe(pipe,scheduler=inverse,device=pipe.device)
    provenance=dict(protocol_sha256=sha(args.protocol),script_sha256=sha(__file__),upstream_revision=current,
        source_sha256={n:sha(args.upstream/n) for n in ['run_imprint_forgery.py','utils/imprint_utils.py','utils/image_utils.py','utils/utils.py']},
        gpu=dict(index=int(info[0]),uuid=uuid),torch=torch.__version__,dtype=str(pipe.unet.dtype),model=args.model,
        reference_pixel_range='[0,1] via released PIL_to_torch; preserved published entry behavior',
        cover_pixel_range='[-1,1] inside released pixel_to_latent',
        validation='All released cadence images saved; verification performed separately without feedback',
        isolation='No verifier inputs loaded; same OS account, not a sandbox boundary')
    write(args.output/f'worker_{args.shard}.json',provenance)
    jobs=[job for i,job in enumerate(plan['jobs']) if i%3==args.shard]
    for job in jobs:
        out=args.output/job['id']; out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():
            completed=json.loads((out/'complete.json').read_text())
            assert completed['updates_executed']==151 and completed['protocol_sha256']==sha(args.protocol)
            continue
        for name in ['reference','cover']:
            if sha(args.protocol.parent/job[name])!=job[name+'_sha256']: raise ValueError('Public input changed')
        set_random_seed(1)
        cover=Image.open(args.protocol.parent/job['cover']).resize((512,512)).convert('RGB')
        reference=Image.open(args.protocol.parent/job['reference']).convert('RGB')
        z0_original=official.pixel_to_latent(cover,pipe)
        z0=torch.nn.Parameter(z0_original.detach().clone())
        optim=torch.optim.Adam([z0],lr=.01)
        with torch.no_grad():
            target=official.invert_image(pipe=pipe,image_pt=PIL_to_torch([reference],dtype=torch.float32,device=pipe.device),
                                        scheduler=inverse,num_inference_steps=50)
        scope=dict(optim=optim,diffpipe=diffpipe,z0=z0,torch=torch,zT_retrieved=target)
        trace=[]; checkpoints=[]; started=time.monotonic()
        for step in range(151):
            exec(update,scope)
            loss=float(scope['l'].detach())
            if not torch.isfinite(z0).all() or not torch.isfinite(scope['l']): raise FloatingPointError('Nonfinite official run')
            trace.append(dict(update=step+1,source_step=step,loss=loss,elapsed_seconds=time.monotonic()-started))
            if step%10==0:
                candidate=official.latent_to_pil(z0,pipe)[0]
                path=out/f'update_{step+1:03}.png'; candidate.save(path)
                checkpoints.append(dict(update=step+1,path=path.name,sha256=sha(path)))
                write(out/'progress.json',dict(updates_executed=step+1,required_updates=151,trace=trace,checkpoints=checkpoints))
                print('PROGRESS',job['id'],step+1,151,round(time.monotonic()-started,1),flush=True)
        assert len(trace)==151 and checkpoints[-1]['update']==151 and len(checkpoints)==16
        write(out/'complete.json',dict(**provenance,job=job,updates_executed=151,required_updates=151,
            seconds=time.monotonic()-started,trace=trace,checkpoints=checkpoints))
        print('COMPLETE',job['id'],flush=True)
    write(args.output/f'shard_{args.shard}_complete.json',dict(completed=True,jobs=[j['id'] for j in jobs],updates_per_job=151))


if __name__=='__main__':
    main()
