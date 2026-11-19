"""Read PNGs and measure paired visual fidelity, text similarity and seed diversity."""
import os
import sys
import numpy as np
import torch
from PIL import Image
import lpips
from transformers import CLIPModel,CLIPProcessor
from prepare import ROOT,RUN,read,put,hashfile,moments


def tensor(img):
    return torch.from_numpy(np.array(img).transpose(2,0,1).copy()).float()[None]/127.5-1


def main(stage):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(4)
    asset=read(ROOT/'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip=CLIPModel.from_pretrained(asset['root'],local_files_only=True).eval()
    proc=CLIPProcessor.from_pretrained(asset['root'],local_files_only=True)
    lp=lpips.LPIPS(net='alex').eval()
    samples=read(RUN/f'{stage}_sources.json')
    rows=[]
    with torch.inference_mode():
        for s in samples:
            i=s['id']
            folder=RUN/stage/f'{i:03d}'
            rec=read(folder/'record.json')
            assert rec['seed']==s['seed'] and rec['config_sha256']==hashfile(RUN/f'{stage}_config.json')
            images={}
            for k,r in rec['arms'].items():
                p=folder/(k+'.png')
                assert hashfile(p)==r['sha256']
                with Image.open(p) as im:
                    images[k]=im.convert('RGB')
                assert np.allclose(moments(images[k]),r['moments'],atol=1e-10,rtol=0)
            inputs=proc(text=[s['prompt']],images=list(images.values()),return_tensors='pt',padding=True,truncation=True,max_length=77)
            res=clip(**inputs)
            for j,(k,im) in enumerate(images.items()):
                x=tensor(im)
                d=np.abs(np.array(rec['arms'][k]['moments'])-s['target'])
                row=dict(id=i,pair=s['pair'],arm=k,mu_mae=d[:3].mean(),var_mae=d[3:].mean(),
                    mean_hit=bool(max(d[:3])<=.02),joint_hit=bool(max(d[:3])<=.02 and max(d[3:])<=.01),
                    seconds=rec['arms'][k]['seconds'],feedback_seconds=rec['arms'][k].get('feedback_seconds',0),
                    lpips=float(lp(tensor(images['baseline']),x).item()) if k!='baseline' else 0.,
                    clip_text=float(res.image_embeds[j]@res.text_embeds[0]),
                    clip_image=float(res.image_embeds[j]@res.image_embeds[0]))
                if s['idx']==1:
                    with Image.open(RUN/stage/f'{i-1:03d}'/(k+'.png')) as sibling:
                        row['seed_lpips']=float(lp(tensor(sibling.convert('RGB')),x).item())
                rows.append(row)
    summary={}
    for k in images:
        group=[r for r in rows if r['arm']==k]
        summary[k]={name:float(np.mean([r[name] for r in group if name in r])) for name in
            ['mu_mae','var_mae','mean_hit','joint_hit','seconds','feedback_seconds','lpips','clip_text','clip_image','seed_lpips']}
    put(RUN/f'{stage}_quality.json',dict(rows=rows,summary=summary,clip_asset=asset['root'],lpips_backbone='alex'))


if __name__=='__main__':
    main(sys.argv[1])
