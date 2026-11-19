"""Paired DDIM trunk with RGB gradients on the final pre-decoding latent."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare import ROOT,RUN,read,put,hashfile,moments


def idle_gpu(idx):
    assert idx in (4,5,7)
    info=subprocess.check_output(['nvidia-smi','-i',str(idx),'--query-gpu=uuid,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True).strip().split(', ')
    busy=subprocess.check_output(['nvidia-smi','-i',str(idx),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if int(info[1])<16 and int(info[2])==0 and not busy.strip():
        return info[0]


def optimize(pipe,z,target,config):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    count=6 if config['arm']=='joint' else 3
    pars=torch.zeros(68,device='cuda',requires_grad=True)
    adam=torch.optim.Adam([pars],lr=config.get('lr',.035)) if config['solver']=='adam' else None
    target=torch.tensor(target,device='cuda')
    weights=torch.tensor([50.]*3+[50.*config['ratio']**.5]*3,device='cuda')
    curve=[]
    best=None
    for i in range(config['steps']+1):
        bias=F.interpolate(pars[4:].reshape(1,4,4,4),size=z.shape[-2:],mode='bilinear',align_corners=False)
        adjusted=z.float()*(1+pars[:4].view(1,4,1,1))+bias
        dec=pipe.vae.decode((adjusted.to(z.dtype)/pipe.vae.config.scaling_factor),return_dict=False)[0]
        x=(dec.float()/2+.5).clamp(0,1)
        v,mu=torch.var_mean(x,dim=(0,2,3),unbiased=False)
        stats=torch.cat([mu,v])
        r=((stats-target)*weights)[:count]
        loss=r.square().sum()
        values=stats.detach().cpu().tolist()
        value=float(loss.detach())
        assert torch.isfinite(loss)
        curve.append(dict(step=i,loss=value,moments=values))
        if best is None or value<best[0]:
            pixels=x.detach()[0].permute(1,2,0).mul(255).round().byte().cpu().numpy()
            best=value,Image.fromarray(pixels),i
        err=(stats-target).detach().abs()
        if i==config['steps'] or (err[:3].max()<.009 and (count==3 or err[3:].max()<.0045)):
            break
        if adam is not None:
            adam.zero_grad()
            loss.backward()
            adam.step()
        else:
            jac=torch.stack([torch.autograd.grad(r[j],pars,retain_graph=j+1<count)[0] for j in range(count)])
            delta=-jac.T@torch.linalg.solve(jac@jac.T+.05*torch.eye(count,device='cuda'),r.detach())
            delta*=min(1.,config.get('trust',.25)/max(1e-9,float(delta.abs().max())))
            with torch.no_grad():
                pars.add_(delta)
        with torch.no_grad():
            pars.clamp_(-1.,1.)
            pars[:4].clamp_(-.5,.5)
        del dec,x,loss,r,stats,mu,v
    return best[1],curve,best[2]


def main(stage,shard,gpu):
    uuid=idle_gpu(gpu)
    if not uuid:
        sys.exit(75)
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import torch
    import numpy as np
    from diffusers import StableDiffusionPipeline,DDIMScheduler
    model=read(ROOT/'runs/unified_seal_clean_20261116/model_manifest.json')['diffusion_model']['root']
    pipe=StableDiffusionPipeline.from_pretrained(model,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    cfg=read(RUN/f'{stage}_config.json')
    items=[s for s in read(RUN/f'{stage}_sources.json') if s['id']%3==shard]
    torch.set_num_threads(2)
    with torch.no_grad():
        pipe('warmup',generator=torch.Generator(device='cuda').manual_seed(73),num_inference_steps=3)
    for s in items:
        folder=RUN/stage/f"{s['id']:03d}"
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'record.json').exists():
            continue
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.no_grad():
            z=pipe(s['prompt'],generator=torch.Generator(device='cuda').manual_seed(s['seed']),
                   num_inference_steps=50,guidance_scale=7.5,height=512,width=512,output_type='latent').images
        torch.cuda.synchronize()
        trunk=time.perf_counter()-start
        tick=time.perf_counter()
        with torch.no_grad():
            dec=pipe.vae.decode(z/pipe.vae.config.scaling_factor,return_dict=False)[0]
            base=pipe.image_processor.postprocess(dec,output_type='pil')[0]
        torch.cuda.synchronize()
        baseline_time=trunk+time.perf_counter()-tick
        base.save(folder/'baseline.png')
        out=dict(baseline=dict(moments=moments(base),seconds=baseline_time,sha256=hashfile(folder/'baseline.png')))
        configs=cfg[s['id']%len(cfg):]+cfg[:s['id']%len(cfg)]
        for c in configs:
            torch.cuda.synchronize()
            tick=time.perf_counter()
            img,curve,best=optimize(pipe,z,s['target'],c)
            torch.cuda.synchronize()
            secs=time.perf_counter()-tick
            path=folder/(c['name']+'.png')
            img.save(path)
            out[c['name']]=dict(moments=moments(img),seconds=trunk+secs,feedback_seconds=secs,best=best,
                                curve=curve,sha256=hashfile(path))
        put(folder/'record.json',dict(id=s['id'],seed=s['seed'],gpu=gpu,uuid=uuid,trunk_seconds=trunk,
            config_sha256=hashfile(RUN/f'{stage}_config.json'),latent_sha256=__import__('hashlib').sha256(
            z.detach().cpu().numpy().tobytes()).hexdigest(),arms=out))


def schedule(stage):
    import fcntl
    lock=(RUN/f'{stage}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=list(range(3))
    active={}
    while pending or active:
        for gpu,(s,p,log) in list(active.items()):
            code=p.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            if code==75:
                pending.append(s)
            elif code:
                raise RuntimeError(f'{stage} shard {s} failed, see log')
        for gpu in (4,5,7):
            if pending and gpu not in active and idle_gpu(gpu):
                s=pending.pop(0)
                log=(RUN/f'{stage}_{s}.log').open('a')
                p=subprocess.Popen([sys.executable,__file__,stage,str(s),str(gpu)],stdout=log,stderr=log)
                active[gpu]=(s,p,log)
        if pending or active:
            time.sleep(8 if active else 40)


if __name__=='__main__':
    if len(sys.argv)==2:
        schedule(sys.argv[1])
    else:
        main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
