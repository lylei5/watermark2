"""Select new prompts and render ordinary independent-seed pairs on 249."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_unified_sources import BLOCKED, PromptIndex, normalize, ROOT

RUN = ROOT/'runs/color_budget_20261119'
MODEL_MANIFEST = ROOT/'runs/unified_seal_clean_20261116/model_manifest.json'


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def select():
    import pyarrow.parquet as pq
    from transformers import CLIPTokenizer
    previous = read(ROOT/'runs/phase1_trajectory_native_gs_20261113/protocol.json')['sources']
    unified = read(ROOT/'runs/unified_main_sources_20261116/selection.json')
    index = PromptIndex(CLIPTokenizer.from_pretrained(
        Path(read(MODEL_MANIFEST)['diffusion_model']['root'])/'tokenizer', local_files_only=True))
    used = set()
    for s in previous + unified['sources']:
        prompt = s.get('caption') or s['prompt']
        tokens, grams, _ = index.encode(prompt)
        index.add(prompt,tokens,grams)
        used.add((s['dataset'],str(s['source_id'])))
    coco_path=ROOT/'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
    sdp_path=ROOT/'runs/phase1_contract_20261113/sdp_eval.parquet'
    for path in (coco_path,sdp_path):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==unified['source_files'][str(path)]
    coco = {}
    for row in sorted(read(coco_path)['annotations'],key=lambda r:r['id']):
        coco.setdefault(str(row['image_id']),row['caption'])
    pools={'COCO':coco,'SDP':{str(i):p for i,p in enumerate(pq.read_table(sdp_path)['Prompt'].to_pylist())}}
    sources=[]
    for dataset,pool in pools.items():
        chosen=0
        for sid in sorted(pool,key=lambda k:digest('colour-budget-1/'+dataset+'/'+k)):
            p=pool[sid]
            tokens,grams,nt=index.encode(p)
            if (dataset,sid) in used or not normalize(p) or BLOCKED.search(p) or nt>75 or index.match(p,tokens,grams):
                continue
            index.add(p,tokens,grams)
            seeds=[int(digest(f'colour-pair/{dataset}/{sid}/{j}')[:15],16) for j in range(2)]
            sources.append(dict(task=len(sources),dataset=dataset,source_id=sid,prompt=p,seeds=seeds,
                                split='calibration' if chosen<20 else 'test'))
            chosen+=1
            if chosen==50:
                break
        assert chosen==50
    assert len({v for s in sources for v in s['seeds']})==200
    with (RUN/'sources.json').open('x') as f:
        json.dump(sources,f,ensure_ascii=False,indent=2)


def generate(shard, gpu):
    assert gpu in (4,5,7)
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True).strip().split(', ')
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert int(row[1])<16 and int(row[2])==0 and not processes.strip(), (gpu,row,processes)
    os.environ['CUDA_VISIBLE_DEVICES']=row[0]
    import torch
    import diffusers
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    tick=time.monotonic()
    torch.set_num_threads(2)
    model=read(MODEL_MANIFEST)['diffusion_model']
    pipe=StableDiffusionPipeline.from_pretrained(model['root'],torch_dtype=torch.float16,local_files_only=True).to('cuda')
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    records=[]
    for s in read(RUN/'sources.json'):
        if s['task']%3!=shard:
            continue
        dest=RUN/'images'/f"{s['task']:03d}"
        dest.mkdir(parents=True,exist_ok=True)
        record=dest/'receipt.json'
        if record.exists():
            item=read(record)
            assert all(hashlib.sha256((dest/f'{i}.png').read_bytes()).hexdigest()==h for i,h in enumerate(item['sha256']))
            records.append(item)
            continue
        rng=[torch.Generator(device='cuda').manual_seed(v) for v in s['seeds']]
        with torch.inference_mode():
            result=pipe([s['prompt']]*2,generator=rng,num_inference_steps=50,guidance_scale=7.5,height=512,width=512)
        assert not any(result.nsfw_content_detected or []) and len(result.images)==2
        hashes=[]
        for i,img in enumerate(result.images):
            path=dest/f'{i}.png'
            img.save(path)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        assert len(set(hashes))==2
        item=dict(task=s['task'],sha256=hashes,seeds=s['seeds'],prompt_sha256=digest(s['prompt']))
        record.write_text(json.dumps(item))
        records.append(item)
    (RUN/f'generated_{shard}.json').write_text(json.dumps(dict(gpu=gpu,uuid=row[0],records=records,
        seconds=time.monotonic()-tick,diffusers=diffusers.__version__,torch=torch.__version__,model=model['root'],
        scheduler=dict(pipe.scheduler.config)),default=str))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['select','generate'])
    p.add_argument('--shard',type=int)
    p.add_argument('--gpu',type=int)
    args=p.parse_args()
    if args.stage=='select':
        select()
    else:
        generate(args.shard,args.gpu)
