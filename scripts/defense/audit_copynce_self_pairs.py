"""Check released CopyNCE self-pair predictions across batch layouts on249."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/capsule_source_transfer_20261115_v3'
DATA=ROOT/'runs/instance_representation_20261113_v2'


def main():
    plan=json.loads((DATA/'protocol.json').read_text());rows=json.loads((RUN/'rows.json').read_text())
    raw=subprocess.check_output(['nvidia-smi','-i','7,5,4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(x[0]):x for x in [line.split(', ') for line in raw.strip().splitlines()]}
    gpu=next(devices[i] for i in [7,5,4] if int(devices[i][2])<100 and int(devices[i][3])==0)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu[1],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    sys.path.insert(0,str(RUN));import copynce_control as cc
    torch,config,builders=cc.imports()
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model,cfg,_=cc.load_model('matching',torch,config,builders);model=model.to('cuda')
    transform=transforms.Compose([transforms.Resize((cfg.input.img_size,cfg.input.img_size),interpolation=InterpolationMode[cfg.input.augmentation.interpolation.upper()]),transforms.ToTensor(),transforms.Normalize(cfg.input.augmentation.post_process.normalize.mean,cfg.input.augmentation.post_process.normalize.std)])
    result=[]
    with torch.no_grad():
        for task in [0,1,50]:
            source=plan['sources'][task];path=Path(plan['parent'])/source['images']['bound']['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['images']['bound']['sha256']
            im=Image.open(path).convert('RGB');x=transform(im)[None].to('cuda')
            single=float(model({'queries':x,'references':x})['preds'].item())
            dense=x.expand(13,-1,-1,-1).contiguous()
            batch=float(model({'queries':dense,'references':dense})['preds'].reshape(-1)[0].item())
            old=1-next(r['copynce_matcher'] for r in rows if r['task']==task and r['kind']=='bound' and r['method']=='spread_order_a03')
            result.append(dict(task=task,single_pred=single,dense13_pred=batch,run_pred=old,max_difference=max(abs(single-old),abs(batch-old))))
    output=dict(status='complete',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu=int(gpu[0]),pairs=result,
        scope='Three exact-reference pairs, released matcher, fixed weights; no score or threshold changes.',max_difference=max(r['max_difference'] for r in result))
    path=RUN/'matcher_self_pair_audit.json';assert not path.exists();path.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
