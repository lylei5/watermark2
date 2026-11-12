"""Read-only official-head logits for a frozen, source-separated development manifest."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    r=a.run;plan=json.loads((r/'protocol.json').read_text());gpu=plan['gpu_indices'][a.shard]
    for name,digest in plan['source_sha256'].items():assert sha(r/name)==digest,name
    for name,digest in plan['external_sha256'].items():assert sha(name)==digest,name
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert gpu in [7,5,4] and int(info[0])==gpu and int(info[2])<50 and int(info[3])==0,info
    os.environ.update(CUDA_VISIBLE_DEVICES=info[1],HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('model',Path(plan['paths']['official'])/'model.py');m=importlib.util.module_from_spec(spec);sys.modules['model']=m;spec.loader.exec_module(m)
    sys.path.insert(0,str(Path(plan['chain_run'])/'scripts'));from sembind import SemanticMasker
    masker=SemanticMasker(Path(plan['paths']['semantic'])/'dinov2-giant',Path(plan['paths']['semantic'])/'sembind_masker.pth');captured={}
    masker.model.hash_head.register_forward_hook(lambda module,inputs,output:captured.update(logits=output.detach().cpu().numpy().ravel()))
    out=r/f'features_{a.shard}';out.mkdir(exist_ok=False);rows=[]
    for i,item in enumerate(plan['development']):
        if i%3!=a.shard:continue
        assert sha(item['image'])==item['image_sha256'] and sha(item['code'])==item['code_sha256']
        bits=masker.code(Image.open(item['image']).convert('RGB')).cpu().numpy().astype(np.uint8);logits=captured['logits']
        assert np.array_equal(bits,np.load(item['code'])['code']) and np.array_equal(bits,(logits>0).astype(np.uint8))
        f=out/f'{i:03}.npz';np.savez_compressed(f,code=bits,logits=logits)
        rows.append(dict(index=i,path=str(f.relative_to(r)),sha256=sha(f)))
    result=dict(completed=True,rows=rows,protocol_sha256=sha(r/'protocol.json'),gpu=gpu,pid=os.getpid())
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',a.shard,len(rows),flush=True)


if __name__=='__main__':main()
