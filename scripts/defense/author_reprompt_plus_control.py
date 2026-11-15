"""Published CVPR2025 Reprompt+ budget on the paired GS/TR development cohort."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import author_reprompt_control as basic

ROOT = basic.ROOT
BASE = ROOT / 'runs/author_reprompt_control_20261114_v2'
read, write, sha = basic.read, basic.write, basic.sha


def prepare(run):
    basic.author_check()
    parent = read(BASE / 'protocol.json')
    inputs = ROOT / 'runs/phase1_public_inputs_20261113/protocol.json'
    targets = read(inputs)['targets']
    refs = [r for r in parent['references'] if r['carrier'] in ['GS', 'TR']]
    assert len(refs) == 20
    for ref in refs:
        ref['target_prompts'] = [targets[ref['task'] + 10*j] for j in range(3)]
    inversions = {}
    for model in ['sd21', 'sd15']:
        done = read(BASE / model / 'complete.json')
        assert done['status'] == 'complete' and sha(BASE / model / 'rows.json') == done['rows_sha256']
        inversions[model] = [r for r in read(BASE / model / 'rows.json') if r['carrier'] in ['GS', 'TR']]
        for r in inversions[model]:
            assert sha(r['latent']) == r['latent_sha256']
    run.mkdir(exist_ok=False)
    for source in [Path(__file__), Path(basic.__file__)]:
        shutil.copy2(source, run / source.name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scripts={p.name:sha(p) for p in run.glob('*.py')}, parent_protocol_sha256=sha(BASE/'protocol.json'),
        author_commit=basic.AUTHOR_COMMIT, author_source_sha256=parent['author_source_sha256'],
        models=parent['models'], model_configs=parent['model_configs'], references=refs,
        frozen_author_inversions=inversions, input_protocol_sha256=sha(inputs),
        prompts_per_reference=3, gs_resamples_per_prompt=3, tr_resamples_per_prompt=1,
        candidates_per_proxy=120, proxies=['sd21','sd15'], seed=1,
        inverse_steps=50, forward_steps=50, forward_guidance=7.5, dtype='float32',
        inversion_policy='Reuse hash-verified author50-step inversions of identical public PNGs.',
        resampling='Call unchanged GsProvider.wiggle_latents with l=1,dtype=float32,device=cuda; no verifier key needed.',
        seed_policy='Reset author RNG to seed1 per reference; consume sequential published resampling calls across3prompts x3draws.',
        aggregation='Any accepted candidate among9GS or3TR; retain every candidate, no adaptive feedback.',
        private_inputs=False, isolation='Logical input separation under shared Unix user; no OS sandbox.',
        scope='Published Reprompt+ candidate budget on20paired development references; not100-source four-target-model paper matrix.',
        differences=['Reviewed I2P subset; target model is existingSD2.1 cohort.',
            'SD1.5 proxy is an additional paired control.',
            'PRC excluded: no new resampling rule or attack extension.'],
        paper='https://arxiv.org/html/2412.03283v3#S3.SS3', allowed_gpu_indices=[7,5,4], excluded_gpu_indices=[6])
    write(run/'protocol.json', plan)
    print({'run':str(run),'protocol_sha256':sha(run/'protocol.json'),'candidates':240}, flush=True)


def worker(run, model, gpu):
    plan=read(run/'protocol.json')
    out=run/model;out.mkdir(exist_ok=False)
    state=dict(status='preflight',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'))
    write(out/'state.json',state)
    try:
        for name,digest in plan['scripts'].items():assert sha(run/name)==digest
        basic.author_check()
        for name,digest in plan['author_source_sha256'].items():assert sha(basic.AUTHOR/name)==digest
        for name,digest in plan['model_configs'][model].items():assert sha(Path(plan['models'][model])/name)==digest
        assert gpu in plan['allowed_gpu_indices'] and gpu!=6
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert int(info[0])==gpu and int(info[2])<100 and int(info[3])==0,info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1],HF_HOME=str(ROOT/'cache/huggingface'),
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        sys.path.insert(0,str(basic.AUTHOR))
        import numpy as np
        import torch
        from utils.pipe.SD_provider import SDPipeProvider
        from utils.wm.gs_provider import GsProvider
        from utils.utils import set_random_seed
        torch.set_num_threads(4)
        provider=SDPipeProvider(pretrained_model_name_or_path=plan['models'][model],resolution=512,device=torch.device('cuda:0'),
            eager_loading=True,unet_id_or_checkpoint_dir=None,lora_checkpoint_dir=None,vae_id=None,zero_unet=False,disable_tqdm=True)
        assert provider.get_dtype()==torch.float32
        # wiggle_latents reads only these public method-context fields.
        context=SimpleNamespace(l=1,dtype=torch.float32,device=torch.device('cuda:0'))
        originals={r['id']:r for r in plan['frozen_author_inversions'][model]}
        (out/'images').mkdir();(out/'latents').mkdir();rows=[]
        state.update(status='running',gpu=gpu,gpu_uuid=info[1],started_at=time.time())
        write(out/'state.json',state)
        for ref in plan['references']:
            original=originals[ref['id']]
            assert sha(original['latent'])==original['latent_sha256']
            z=torch.from_numpy(np.load(original['latent'])['proxy_zT']).to('cuda:0')
            set_random_seed(plan['seed'])
            for prompt_index,prompt in enumerate(ref['target_prompts']):
                for repeat in range(3 if ref['carrier']=='GS' else 1):
                    begin=time.perf_counter()
                    trial=f"{ref['id']}_{model}_p{prompt_index}_r{repeat}"
                    with torch.no_grad():
                        used=GsProvider.wiggle_latents(context,z.clone()).to(dtype=provider.get_dtype()) if ref['carrier']=='GS' else z.clone()
                        assert torch.isfinite(used).all()
                        if ref['carrier']=='GS':assert torch.equal(used>=0,z>=0)
                        generated=provider.generate(prompts=[prompt['prompt']],num_inference_steps=50,guidance_scale=7.5,latents=used)
                    image=out/'images'/f'{trial}.png';latent=out/'latents'/f'{trial}.npz'
                    generated['images_PIL'][0].save(image);np.savez_compressed(latent,proxy_zT=used.cpu().numpy())
                    rows.append(dict(id=ref['id'],trial=trial,carrier=ref['carrier'],task=ref['task'],proxy=model,
                        prompt_index=prompt_index,resample_index=repeat,target_source_index=prompt['source_index'],
                        image=str(image),image_sha256=sha(image),pixels_sha256=__import__('hashlib').sha256(generated['images_PIL'][0].convert('RGB').tobytes()).hexdigest(),
                        latent=str(latent),latent_sha256=sha(latent),original_proxy_latent_sha256=original['latent_sha256'],seconds=time.perf_counter()-begin))
                    write(out/'rows.json',rows)
            print({'model':model,'reference':ref['id'],'completed_candidates':len(rows)},flush=True)
        assert len(rows)==120
        basic.author_check()
        state.update(status='complete',count=len(rows),seconds=time.time()-state['started_at'],rows_sha256=sha(out/'rows.json'))
        write(out/'complete.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(out/'state.json',state)


def launch(run):
    jobs=[]
    for model,gpu in [('sd21',7),('sd15',5)]:
        log=open(run/(model+'.log'),'xb')
        p=subprocess.Popen([str(basic.PYTHON),str(run/Path(__file__).name),'worker','--run',str(run),'--model',model,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();jobs.append(dict(model=model,gpu=gpu,pid=p.pid))
    write(run/'launch.json',dict(jobs=jobs,protocol_sha256=sha(run/'protocol.json')))
    print(jobs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','worker']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',choices=['sd21','sd15']);p.add_argument('--gpu',type=int);a=p.parse_args()
    assert a.run.resolve().parent==ROOT/'runs'
    if a.mode=='prepare':prepare(a.run)
    elif a.mode=='launch':launch(a.run)
    else:worker(a.run,a.model,a.gpu)
