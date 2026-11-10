"""Compare effective fp16 diffusion weights and semantic assets on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(model,semantic,source):
    torch.set_num_threads(2);result=dict(effective_weights={},configs={},semantic={})
    for name in ['unet/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors','text_encoder/model.safetensors']:
        h=hashlib.sha256()
        with safe_open(model/name,framework='pt',device='cpu') as f:
            for key in sorted(f.keys()):
                t=f.get_tensor(key).to(torch.float16).contiguous()
                h.update((key+str(tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        result['effective_weights'][name]=h.hexdigest()
    for folder in ['unet','vae','text_encoder','scheduler','tokenizer']:
        for p in sorted((model/folder).iterdir()):
            if p.suffix=='.json':
                data=json.loads(p.read_text())
                for key in ['_name_or_path','_diffusers_version','torch_dtype']:data.pop(key,None)
                result['configs'][p.relative_to(model).as_posix()]=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
            elif p.name in ['merges.txt','vocab.txt']:
                result['configs'][p.relative_to(model).as_posix()]=file_sha(p)
    for p in sorted((semantic/'dinov2-giant').iterdir()):
        if p.suffix in ['.json','.safetensors','.bin']:result['semantic'][p.relative_to(semantic).as_posix()]=file_sha(p)
    result['semantic']['sembind_masker.pth']=file_sha(semantic/'sembind_masker.pth')
    result['semantic']['model.py']=file_sha(source)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--semantic',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.model,args.semantic,args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print('AUDIT_WRITTEN',args.output,flush=True)
