"""Measure native inversion readouts for existing official enrollment views on249."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path('/data0/liyanlei/watermark_research')
BASE=ROOT/'runs/instance_coset_recovery_20261113'
CHANNEL=ROOT/'runs/carrier_content_channel_20261113'
NATIVE=ROOT/'runs/defense_revalidation_20261111'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    base=json.loads((BASE/'protocol.json').read_text())
    channel=json.loads((CHANNEL/'protocol.json').read_text())
    shutil.copy2(__file__,run/Path(__file__).name)
    plan=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Register paired content and independent inversion-channel changes for conditional key recovery',
        sources=[r for r in base['sources']if r['cohort']=='PGID_GS'],
        base=str(BASE),base_protocol_sha256=sha(BASE/'protocol.json'),
        feature_rows_sha256=sha(BASE/'feature_rows.json'),features_sha256=sha(BASE/'features.npz'),
        channel=str(CHANNEL),channel_protocol_sha256=sha(CHANNEL/'protocol.json'),
        channel_rows_sha256=sha(CHANNEL/'rows.json'),
        native_protocol_sha256=sha(NATIVE/'protocol.json'),
        distortion_file=base['distortion_file'],distortion_sha256=base['distortion_sha256'],
        conditions=base['conditions'],model_revision=channel['model_revision'],
        enrollment_queries=80,new_inversions=80,gpu=7,
        readout='32 key-free block means/RMS,4channels x2x2spatial blocks',
        calibration_sources=[0,1,2,3,4],evaluation_sources=[5,6,7,8,9],
        calibration_targets=[.90,.95],projection_repetitions=3,
        methods=['clean_margin','orbit_margin','tangent8','paired_channel','permuted_channel','content_self_fit'],
        coefficient_model='Registration-only ridge on eight independent channel-readout differences, lambda=.1*trace(Gram)/8; L1 coefficient norm clipped to2',
        registration_application='Apply the same channel-conditioned correction to enrollment views and queries before reliable clean-margin1023-bit coding',
        control='Cyclic shift by2 of eight content-response rows with channel rows fixed, same capacity and helper size',
        claim_scope='Development mechanism and actual recovery; no independent confirmation',
        attack_scope='Existing complete151-update Imprint outputs only; no new or modified attack',
        attack_generation=False,source_sha256=sha(run/Path(__file__).name))
    write(run/'protocol.json',plan)
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))),flush=True)


def execute(run,plan):
    state=dict(status='preflight',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'))
    assert not(run/'state.json').exists()
    write(run/'state.json',state)
    try:
        for path,digest in [(BASE/'protocol.json',plan['base_protocol_sha256']),
            (BASE/'feature_rows.json',plan['feature_rows_sha256']),(BASE/'features.npz',plan['features_sha256']),
            (CHANNEL/'protocol.json',plan['channel_protocol_sha256']),(CHANNEL/'rows.json',plan['channel_rows_sha256']),
            (NATIVE/'protocol.json',plan['native_protocol_sha256']),(Path(plan['distortion_file']),plan['distortion_sha256'])]:
            assert sha(path)==digest
        parent=json.loads((NATIVE/'protocol.json').read_text())
        for name,digest in parent['source_sha256'].items():assert sha(NATIVE/name)==digest
        ref=ROOT/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip()==plan['model_revision']
        info=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert int(info[0])==7 and int(info[2])<50 and int(info[3])==0,info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1],HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        import numpy as np
        import torch
        from PIL import Image
        sys.path.insert(0,str(NATIVE/'PGID'))
        from pipe import pipe_utils
        from utils import set_random_seed
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        spec=importlib.util.spec_from_file_location('registration_distortion',plan['distortion_file'])
        distort=importlib.util.module_from_spec(spec);spec.loader.exec_module(distort)
        provider=pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent['pgid']['modelid'],
            resolution=512,device='cuda:0',scheduler_name='DDIM',scheduler_inverse_name='DDIM',eager_loading=True)
        metadata=json.loads((BASE/'feature_rows.json').read_text())
        (run/'work').mkdir();(run/'latents').mkdir();os.chdir(run/'work')
        state.update(status='running',gpu=7,gpu_uuid=info[1],started_at=time.time());write(run/'state.json',state)
        rows=[]
        with torch.no_grad():
            for source in plan['sources']:
                assert sha(source['reference'])==source['reference_sha256']
                clean=Image.open(source['reference']).convert('RGB')
                views=[r for r in metadata if r['id']==source['id']and r['kind'].startswith('enroll_')]
                assert len(views)==8
                for view in views:
                    options=SimpleNamespace(**{k:None for k in plan['conditions']})
                    setattr(options,view['family'],plan['conditions'][view['family']])
                    distort.set_random_seed(view['seed'])
                    image=distort.image_distortion(clean.copy(),view['seed'],options).convert('RGB')
                    assert hashlib.sha256(image.tobytes()).hexdigest()==view['pixels_sha256']
                    set_random_seed(view['seed'])
                    z=provider.invert_images(image,num_inference_steps=50)['zT_torch']
                    path=run/'latents'/f"{source['id']}_{view['kind']}.npz"
                    np.savez_compressed(path,z=z.cpu().numpy())
                    rows.append(dict(**view,latent=str(path.relative_to(run)),latent_sha256=sha(path)))
                print('PAIRED_REGISTERED',source['id'],flush=True)
        assert len(rows)==80
        write(run/'rows.json',rows)
        state.update(status='complete',rows=len(rows),rows_sha256=sha(run/'rows.json'),seconds=time.time()-state['started_at'])
        write(run/'complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'state.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    if a.mode=='prepare':prepare(run)
    else:
        plan=json.loads((run/'protocol.json').read_text());assert sha(__file__)==plan['source_sha256'];execute(run,plan)


if __name__=='__main__':main()
