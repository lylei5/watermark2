"""Compare repeated full SemBind inversion under default and deterministic backends."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,choices=range(3),required=True)
    args=p.parse_args();root=Path('/data0/liyanlei/watermark_research');run=args.run.resolve()
    assert run.parent==root/'runs'
    plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['local_sha256'].items():assert sha(run/name)==digest
    out=run/f'native_{args.shard}';out.mkdir(exist_ok=False)
    state=dict(status='preflight',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'))
    try:
        gpu=plan['gpu_indices'][args.shard];assert gpu in [7,5,4]
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert int(info[0])==gpu and int(info[2])<100 and int(info[3])==0,info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1],CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        sys.path.insert(0,str(root/'envs/phase1_prc_extra'))
        import numpy as np
        import torch
        from PIL import Image
        from sembind_official_audit import load_official
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        paths={k:Path(v) for k,v in plan['paths'].items()}
        for name,digest in plan['official_sha256'].items():assert sha(paths['official']/name)==digest
        assert sha(paths['scheduler'])==plan['scheduler_sha256']
        inversion,pipe=load_official(SimpleNamespace(**paths));pipe.set_progress_bar_config(disable=True)
        case=plan['cases'][args.shard];assert sha(case['image'])==case['image_sha256']
        image=Image.open(case['image']).convert('RGB');rows=[];arrays={}
        for mode,enabled in [('default',False),('deterministic',True)]:
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=enabled
            torch.use_deterministic_algorithms(enabled)
            values=[]
            for repeat in range(2):
                torch.manual_seed(case['seed'])
                try:
                    value=inversion.exact_inversion(image,prompt='',guidance_scale=3.,num_inference_steps=50,
                        test_num_inference_steps=50,inv_order=0,decoder_inv=True,pipe=pipe).detach().cpu().numpy()
                    assert np.isfinite(value).all()
                    arrays[f'{mode}_{repeat}']=value;values.append(value)
                except RuntimeError as error:
                    if mode!='deterministic':raise
                    rows.append(dict(mode=mode,status='unsupported',error=str(error)));break
            if len(values)==2:
                rows.append(dict(mode=mode,status='complete',exact_equal=bool(np.array_equal(*values)),
                    max_abs_difference=float(np.max(np.abs(values[0]-values[1]))),
                    sign_disagreements=int(np.sum((values[0]>0)!=(values[1]>0)))))
        file=out/'repeated_inversions.npz';np.savez_compressed(file,**arrays)
        state.update(status='complete',case=case,rows=rows,gpu=gpu,gpu_uuid=info[1],artifact_sha256=sha(file),
            deterministic_replay_passed=any(r['mode']=='deterministic' and r.get('exact_equal') for r in rows),
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out/'complete.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        (out/'state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
