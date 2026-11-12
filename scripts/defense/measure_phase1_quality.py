"""CPU-only CLIP ViT-L/14 and paired LPIPS for completed phase1 generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    torch.set_num_threads(4)
    plan=json.loads((run/'protocol.json').read_text());records=[]
    for shard in range(3):
        done=json.loads((run/f'generate_{shard}/complete.json').read_text())
        assert done['status']=='complete' and done['protocol_sha256']==sha(run/'protocol.json')
        records.extend(done['rows'])
    records.sort(key=lambda r:r['task']);assert [r['task'] for r in records]==list(range(100))
    asset=json.loads((run.parent/'phase1_backend_audit_20261113_v2/complete.json').read_text())['model_files']['clip_vitl14']
    model_root=Path(asset['root'])
    for name,info in asset['files'].items():assert sha(model_root/name)==info['sha256']
    clip=CLIPModel.from_pretrained(model_root,local_files_only=True).eval().cpu()
    processor=CLIPProcessor.from_pretrained(model_root,local_files_only=True)
    perceptual=lpips.LPIPS(net='alex').eval().cpu()
    rows=[];started=time.monotonic()
    with torch.no_grad():
        for record in records:
            source=plan['sources'][record['task']];images=[];names=['aux','same','gs','bound']
            for name in names:
                item=record['images'][name];path=run/item['path'];assert sha(path)==item['sha256']
                images.append(Image.open(path).convert('RGB'))
            inputs=processor(text=[source['caption']],images=images,return_tensors='pt',padding=True,truncation=True,max_length=77)
            outputs=clip(**inputs)
            cos=(outputs.image_embeds@outputs.text_embeds.T).flatten().numpy()
            tensors={name:torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float().unsqueeze(0)/127.5-1 for name,im in zip(names,images)}
            metrics={f'clip_{name}':float(score) for name,score in zip(names,cos)}
            for left,right in [('aux','bound'),('aux','gs'),('aux','same'),('gs','bound')]:
                metrics[f'lpips_{left}_{right}']=float(perceptual(tensors[left],tensors[right]).item())
            rows.append(dict(task=source['task'],dataset=source['dataset'],**metrics))
    summary=[]
    for dataset in ['COCO','SDP']:
        subset=[r for r in rows if r['dataset']==dataset]
        idx=np.random.default_rng(2026111302).integers(0,len(subset),(10000,len(subset)))
        for name in metrics:
            values=np.array([r[name] for r in subset])
            summary.append(dict(dataset=dataset,metric=name,n=len(values),mean=float(values.mean()),
                ci95=np.quantile(values[idx].mean(axis=1),[.025,.975]).tolist()))
    result=dict(status='complete',protocol_sha256=sha(run/'protocol.json'),script_sha256=sha(__file__),
        clip_model=asset,clip_metric='Cosine of normalized image/text embeddings, raw scale,77-token truncation',
        lpips='AlexNet LPIPS CPU, images mapped to[-1,1]',rows=rows,summary=summary,
        uncertainty='10000 percentile bootstrap resamples by prompt; seed2026111302',
        seconds=time.monotonic()-started,gpu_used=False,
        scope='Descriptive50 images per dataset per arm; FID5000 and paper1000-prompt/10-trial quality benchmark not reproduced')
    path=run/'quality_summary.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
