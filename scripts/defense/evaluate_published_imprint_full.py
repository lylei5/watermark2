"""Separate full DPM verifier for every published Imprint checkpoint."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from sembind_official_audit import load_official, carrier, sha, write
from codeword_binding import sembind_mask_bits, bit_accuracy
from content_authentication import recover_groups, content_scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--identity-root',type=Path,required=True)
    p.add_argument('--chain-run',type=Path,help='Use a freshly executed chain and its pinned released source')
    p.add_argument('--shard',type=int,required=True,choices=range(3))
    args=p.parse_args()
    root,run=args.root,args.run
    plan=json.loads((run/'protocol.json').read_text())
    assert (run/'outputs'/f'shard_{args.shard}_complete.json').is_file()
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    device=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if int(device[0]) not in [7,5,4] or int(device[2])>50 or int(device[3])!=0: raise RuntimeError('Evaluator GPU not idle or allowed')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.chain_run:
        chain_paths=json.loads((args.chain_run/'paths.json').read_text())
        original=json.loads((Path(chain_paths['output'])/'protocol.json').read_text())
        official=Path(chain_paths['official'])
        scheduler=Path(chain_paths['scheduler'])
        model=chain_paths['model']
    else:
        bundle=root/'runs/sembind_audit_evidence_20261110'
        original=json.loads((bundle/'sembind_official_audit_20261110/protocol.json').read_text())
        official=bundle/'third_party/SemBind_pinned'
        scheduler=bundle/'third_party/legacy_diffusers/scheduling_dpmsolver_multistep.py'
        model=json.loads((root/'protocols/confirmation100_design_v7.json').read_text())['proxy']['snapshot']
    for rel,digest in original['source_sha256'].items():
        assert sha(official/rel.replace('\\','/'))==digest
    assert sha(scheduler)==original['scheduler_sha256']
    for rel in ['unet/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors','text_encoder/model.safetensors']:
        hashes=original['model_files_sha256']
        assert sha(Path(model)/rel)==hashes.get(rel,hashes.get(rel.replace('/','\\')))
    jobs=[j for i,j in enumerate(plan['jobs']) if i%3==args.shard]
    out=run/'evaluation'/f'shard_{args.shard}';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for job in jobs:
        complete=json.loads((run/'outputs'/job['id']/'complete.json').read_text())
        assert complete['updates_executed']==151 and len(complete['trace'])==151
        assert complete['protocol_sha256']==sha(run/'protocol.json')
        assert [c['update'] for c in complete['checkpoints']]==list(range(1,152,10))
        for kind in ['reference','cover']:
            path=run/job[kind];assert sha(path)==job[kind+'_sha256']
            records.append(dict(id=job['id']+'_'+kind,task=job['task'],method=job['method'],kind=kind,
                                image_path=str(path),image_sha256=sha(path),cover_path=str(run/job['cover'])))
        for c in complete['checkpoints']:
            path=run/'outputs'/job['id']/c['path'];assert sha(path)==c['sha256']
            records.append(dict(id=job['id']+f"_update_{c['update']:03}",task=job['task'],method=job['method'],kind='candidate',
                update=c['update'],image_path=str(path),image_sha256=sha(path),cover_path=str(run/job['cover'])))
    write(out/'protocol.json',dict(rows=records,protocol_sha256=sha(run/'protocol.json'),gpu_uuid=uuid,
        inversion=dict(steps=50,inv_order=0,decoder_steps=100,prompt='',cfg=3.,dtype='float32'),
        scope='All 16 checkpoints and both reference/cover inputs for each assigned full-budget job',
        dependencies={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers']},
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('sembind_official_audit.py'))))
    if args.chain_run:
        sys.path.insert(0,str(args.chain_run/'scripts'))
        spec=importlib.util.spec_from_file_location('model',official/'model.py')
        official_model=importlib.util.module_from_spec(spec)
        sys.modules['model']=official_model
        spec.loader.exec_module(official_model)
    else:
        sys.path.insert(0,str(root/'scripts/defense'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from semantic_code import SemanticCodeExtractor
    masker=SemanticCodeExtractor(root/'models/sembind/dinov2-giant',root/'models/sembind/sembind_masker.pth')
    for r in records:
        path=out/(r['id']+'.npz')
        code=masker.code(Image.open(r['image_path']).convert('RGB')).cpu().numpy().astype(np.uint8)
        np.savez_compressed(path,sembind_code=code)
    del masker;gc.collect();torch.cuda.empty_cache()
    inversion,pipe=load_official(SimpleNamespace(official=official,scheduler=scheduler,model=Path(model)))
    started=time.monotonic();scores=[]
    for r in records:
        path=out/(r['id']+'.npy');image=Image.open(r['image_path']).convert('RGB')
        torch.manual_seed(202611100000+r['task'])
        z=inversion.exact_inversion(image,prompt='',guidance_scale=3.,num_inference_steps=50,
            test_num_inference_steps=50,inv_order=0,decoder_inv=True,pipe=pipe)
        latent=z.detach().cpu().numpy();assert latent.shape==(1,4,64,64) and np.isfinite(latent).all()
        np.save(path,latent)
        _,cipher,message,perm=carrier(args.identity_root,r['task'])
        stream=cipher^np.tile(message,64)
        code=np.load(out/(r['id']+'.npz'))['sembind_code']
        mask=sembind_mask_bits(code[None],perm)[0] if r['method']=='sembind' else np.zeros(16384,dtype=np.uint8)
        ba=float(bit_accuracy((latent.reshape(-1)>0).astype(np.uint8)^mask^stream,message)[0])
        extra=content_scores(recover_groups(latent,cipher,perm),code) if r['method']=='sembind' and not args.chain_run else {}
        scores.append(dict(**r,ba=ba,accepted=ba>181/256,**extra,inversion_sha256=sha(path),features_sha256=sha(out/(r['id']+'.npz'))))
        write(out/'progress.json',dict(completed=len(scores),required=len(records),seconds=time.monotonic()-started))
        print('VERIFIED',r['id'],ba,len(scores),len(records),flush=True)
    del pipe;gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,str(root/'third_party/semantic-forgery'))
    import lpips
    from utils.image_utils import psnr_PIL,ssim_PIL,msssim_PIL,lpips_PIL
    quality_model=lpips.LPIPS(net='alex').eval().cpu()
    for r in scores:
        if r['kind']!='candidate': continue
        image=Image.open(r['image_path']).convert('RGB');cover=Image.open(r['cover_path']).convert('RGB')
        with torch.no_grad():
            r.update(psnr=psnr_PIL(image,cover),ssim=float(ssim_PIL(image,cover)[0]),
                     ms_ssim=msssim_PIL(image,cover),lpips=lpips_PIL(image,cover,quality_model,'cpu'))
    write(out/'results.json',dict(completed=True,rows=scores,protocol_sha256=sha(run/'protocol.json'),
        full_budget_jobs=len(jobs),checkpoint_count=16*len(jobs),seconds=time.monotonic()-started,
        scope='Fixed small development component comparison; no independent confirmation or SOTA claim'))
    print('EVALUATION_COMPLETE',args.shard,len(scores),flush=True)


if __name__=='__main__':main()
