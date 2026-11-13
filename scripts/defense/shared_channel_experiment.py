"""Frozen paired-spectrum defense experiment on249; existing attacks are read-only."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
RUNS = ROOT / 'runs'
BASE = RUNS / 'instance_coset_recovery_20261113'
CHANNEL = RUNS / 'carrier_content_channel_20261113'
PAIRED = RUNS / 'paired_edit_registration_20261113'
TRANSFER = RUNS / 'recovered_carrier_transfer_20261113'
COVERAGE = RUNS / 'pgid_carrier_coverage_20261113'
NATIVE = RUNS / 'defense_revalidation_20261111'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def prepare(run):
    run.mkdir(exist_ok=False)
    bp, tp, cp = read(BASE/'protocol.json'), read(TRANSFER/'protocol.json'), read(COVERAGE/'protocol.json')
    native = read(NATIVE/'protocol.json')
    sources = {x['id']: x for x in bp['sources'] if x['cohort'] == 'PGID_GS'}
    rows = []
    for x in read(CHANNEL/'rows.json'):
        identity = f"G{x['task']:03d}"
        source = sources[identity]
        kind = f"imprint_{x['checkpoint']}" if x['kind'] == 'imprint' else x['kind']
        image = source['reference'] if x['kind'] != 'imprint' else next(
            q['path'] for q in source['attacks'] if q['checkpoint'] == x['checkpoint'])
        rows.append(dict(id=identity, carrier='GS', task=x['task'], kind=kind,
            image=image, image_sha256=sha(image), family=kind if kind in bp['conditions'] else None,
            seed=x['seed'], resize=False, pixels_sha256=x['pixels_sha256'],
            latent=str(CHANNEL/x['latent']), latent_sha256=x['latent_sha256'],
            native_metric=x['ba'], native_accepted=x['native']))
    for x in read(PAIRED/'rows.json'):
        source = sources[x['id']]
        rows.append(dict(id=x['id'], carrier='GS', task=source['task'], kind=x['kind'],
            image=source['reference'], image_sha256=source['reference_sha256'], family=x['family'],
            seed=x['seed'], resize=False, pixels_sha256=x['pixels_sha256'],
            latent=str(PAIRED/x['latent']), latent_sha256=x['latent_sha256'],
            native_metric=None, native_accepted=None))
    for x in read(TRANSFER/'controls.json'):
        carrier, task, kind = x['carrier'], x['task'], x['kind']
        if kind == 'cover':
            item = next(y for y in cp['inputs'] if y['index'] == task)
            path = Path(cp['cover_images_dir']) / Path(item['cover']).name
        elif kind.startswith('imprint_'):
            path = COVERAGE/carrier/f"attack=forgery/attacked_images_step_{kind.split('_')[1]}/image_{task:03d}.png"
        else:
            path = COVERAGE/carrier/f'attack=forgery/watermarked_images/image_{task:03d}.png'
        family = kind if kind in tp['conditions'] else None
        seed = (tp['seed_base'] + ['TR','PRC'].index(carrier)*10000 + task*100 +
                list(tp['conditions']).index(kind)) if family else None
        rows.append(dict(id=x['id'], carrier=carrier, task=task, kind=kind, image=str(path),
            image_sha256=sha(path), family=family, seed=seed, resize=kind == 'cover',
            pixels_sha256=x['pixels_sha256'], latent=str(TRANSFER/'latents'/f"{x['id']}_{kind}.npz"),
            latent_sha256=x['latent_sha256'], native_metric=x['native_metric'], native_accepted=x['native_accepted']))
    assert len(rows) == 490 and len({(x['id'],x['kind']) for x in rows}) == 490
    # Source and transform manifests are checked before any new model execution.
    for x in rows:
        assert sha(x['latent']) == x['latent_sha256']
    revision = read(PAIRED/'protocol.json')['model_revision']
    model = ROOT/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots'/revision
    vae_files = {str(p): sha(p) for p in (model/'vae').iterdir() if p.is_file()}
    scripts = ['shared_channel_experiment.py', 'shared_channel_features.py',
               'evaluate_shared_channel.py', 'instance_coset_recovery.py', 'recovered_carrier_transfer.py']
    for name in scripts:
        shutil.copy2(Path(__file__).with_name(name), run/name)
    private = ROOT/'private_eval'/run.name
    private.mkdir(exist_ok=False)
    (private/'provider.bin').write_bytes(secrets.token_bytes(32))
    prior_private = Path(read(PAIRED/'recovery_protocol.json')['private'])
    for rep in range(3):
        shutil.copy2(prior_private/f'projection_{rep}.npy', private/f'projection_{rep}.npy')
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hypothesis='Common edit transfer cancels in paired content/noise spectra while instance relation remains distinctive',
        claim_scope='Development mechanism experiment; common-filter algebra does not imply actual diffusion equivariance',
        recipes=rows, existing_queries=490, fresh_vae_encodings=500, extra_sscd_enrollment_encodings=160,
        same_prompt_controls=dict(count=10, seed_base=2026111400, prompts=native['pgid_inputs'],
            generation_steps=50, generation_cfg=7.5, generation_scheduler='DDIM',
            inversion_steps=50, native_readouts=['DDIM','DPM exact'], attack=False,
            description='Independent unwatermarked instances; no watermark latent is reused'),
        conditions=bp['conditions'], distortion_file=bp['distortion_file'], distortion_sha256=bp['distortion_sha256'],
        enrollment_families=tp['enrollment_families'], enrollment_seed_base=tp['enrollment_seed_base'],
        model_path=str(model), model_id=native['pgid']['modelid'], vae_files=vae_files, sscd=bp['model'], gpu=7,
        author_source=native['source_sha256'], coverage_protocol_sha256=sha(COVERAGE/'protocol.json'),
        paired_protocol_sha256=sha(PAIRED/'protocol.json'), transfer_protocol_sha256=sha(TRANSFER/'protocol.json'),
        fit_sources=[0,1,2,3], calibration_sources=[4,5,6], evaluation_sources=[7,8,9],
        center_fit='36 GS reference/enrollment views from four fitting sources; no evaluation negatives',
        calibration_targets=[.90,.95], projection_repetitions=3,
        methods=['shared_phase','shared_log_gain','content_phase','noise_phase','phase_concat','permuted_phase','sscd','sscd_centered'],
        key_enrollment='Reference-only reliable-coordinate1023-bit BCH code; no per-record tangent removal',
        spectral_pool='Complex phases:4x8x8 cells; log gains:4x16x8; every descriptor512D',
        native_envelope='AES-GCM full native state once per record; actual recovered128-bit key unwraps authenticated data key and blob hash',
        private=str(private), provider_sha256=sha(private/'provider.bin'),
        projection_sha256=[sha(private/f'projection_{i}.npy') for i in range(3)],
        scripts={n:sha(run/n) for n in scripts}, attack_generation=False,
        attack_scope='Existing unchanged CVPR2025 full151-update Imprint only', independent_confirmation=False)
    write(run/'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run/'protocol.json'))), flush=True)


def extract(run, plan):
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'))
    assert not (run/'extract_state.json').exists()
    write(run/'extract_state.json', state)
    try:
        for name,digest in plan['scripts'].items(): assert sha(run/name) == digest
        for name,digest in plan['author_source'].items(): assert sha(NATIVE/name) == digest
        for name,digest in plan['vae_files'].items(): assert sha(name) == digest
        ref=ROOT/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip() == Path(plan['model_path']).name
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        assert sha(plan['sscd']['path']) == plan['sscd']['sha256']
        info = subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == 7 and int(info[2]) < 100 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT/'cache/huggingface'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from diffusers import AutoencoderKL
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        vae = AutoencoderKL.from_pretrained(plan['model_path'], subfolder='vae', local_files_only=True,
                                           torch_dtype=torch.float32).to('cuda').eval()
        model = torch.jit.load(plan['sscd']['path'],map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320,320]),transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
        spec = importlib.util.spec_from_file_location('shared_distortions',plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec); spec.loader.exec_module(distort)
        (run/'work').mkdir(); (run/'same_prompt').mkdir(); os.chdir(run/'work')
        state.update(status='running',gpu=7,gpu_uuid=info[1],started_at=time.time()); write(run/'extract_state.json',state)
        rows, content, noise, sscd = [], [], [], []
        def observe(image, z, item):
            assert image.size == (512,512)
            with torch.no_grad():
                pixels = np.array(image).astype(np.float32)/127.5-1
                x = torch.from_numpy(pixels.transpose(2,0,1)).unsqueeze(0).to('cuda')
                a = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
                f = torch.nn.functional.normalize(model(transform(image).unsqueeze(0).to('cuda')),dim=1)
            content.append(a.cpu().numpy().reshape(4,64,64)); noise.append(z.reshape(4,64,64)); sscd.append(f.cpu().numpy()[0])
            rows.append(item)
        for item in plan['recipes']:
            assert sha(item['image']) == item['image_sha256'] and sha(item['latent']) == item['latent_sha256']
            image = Image.open(item['image'])
            if item['resize']: image = image.resize((512,512))
            image = image.convert('RGB')
            if item['family']:
                options = SimpleNamespace(**{k:None for k in plan['conditions']})
                setattr(options,item['family'],plan['conditions'][item['family']]); distort.set_random_seed(item['seed'])
                image = distort.image_distortion(image.copy(),item['seed'],options).convert('RGB')
            assert hashlib.sha256(image.tobytes()).hexdigest() == item['pixels_sha256']
            observe(image,np.load(item['latent'])['z'],dict(item))
        templates = {}
        for carrier in ['GS','TR','PRC']:
            for task in range(10):
                identity = f'G{task:03d}' if carrier == 'GS' else f'{carrier}_{task:03d}'
                j = next(j for j,r in enumerate(rows) if r['id'] == identity and r['kind'] == 'clean')
                vectors = [sscd[j]]
                if carrier == 'GS':
                    vectors += [sscd[k] for k,r in enumerate(rows) if r['id'] == identity and r['kind'].startswith('enroll_')]
                else:
                    clean = Image.open(rows[j]['image']).convert('RGB')
                    for f,family in enumerate(plan['enrollment_families']):
                        for repeat in range(2):
                            seed = plan['enrollment_seed_base'] + ['TR','PRC'].index(carrier)*10000 + task*100 + f*10 + repeat
                            options = SimpleNamespace(**{k:None for k in plan['conditions']})
                            setattr(options,family,plan['conditions'][family]); distort.set_random_seed(seed)
                            image = distort.image_distortion(clean.copy(),seed,options).convert('RGB')
                            with torch.no_grad():
                                vectors.append(torch.nn.functional.normalize(model(transform(image).unsqueeze(0).to('cuda')),dim=1).cpu().numpy()[0])
                assert len(vectors) == 9
                templates[identity] = np.stack(vectors)
        np.savez_compressed(run/'sscd_templates.npz',**templates)
        # Independent seed controls are ordinary generation, with no attack objective.
        sys.path.insert(0,str(NATIVE/'PGID'))
        from pipe import pipe_utils
        from utils import set_random_seed,configure_prc
        from watermarks.prc_watermark import parser as prc_parser
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=plan['model_id'],resolution=512,
            device='cuda:0',scheduler_name='DDIM',scheduler_inverse_name='DDIM',eager_loading=True,disable_tqdm=True)
        same = []
        for item in plan['same_prompt_controls']['prompts']:
            task = item['index']; seed = plan['same_prompt_controls']['seed_base'] + task; set_random_seed(seed)
            z = torch.from_numpy(np.random.default_rng(seed).standard_normal((1,4,64,64)).astype(np.float32)).to('cuda')
            with torch.no_grad():
                image = provider.generate([item['prompt']],num_inference_steps=50,guidance_scale=7.5,latents=z,return_latents=False)['images'][0].convert('RGB')
                inv = provider.invert_images(image,num_inference_steps=50)['zT_torch'].cpu().numpy()
            path = run/'same_prompt'/f'{task:03d}.png'; image.save(path)
            latent_path = run/'same_prompt'/f'{task:03d}_DDIM.npz'; np.savez_compressed(latent_path,z=inv)
            info = dict(task=task,kind='same_prompt',image=str(path),image_sha256=sha(path),family=None,seed=seed,resize=False,
                pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),latent=str(latent_path),latent_sha256=sha(latent_path),
                native_metric=None,native_accepted=None)
            observe(image,inv,dict(info,id=f'G{task:03d}',carrier='GS'))
            # The identical DDIM image/readout is also queried against the TR record.
            content.append(content[-1].copy()); noise.append(noise[-1].copy()); sscd.append(sscd[-1].copy())
            rows.append(dict(info,id=f'TR_{task:03d}',carrier='TR'))
            same.append((image,info))
        del provider; torch.cuda.empty_cache()
        cp = read(COVERAGE/'protocol.json')
        assert [x['prompt'] for x in cp['inputs']] == [x['prompt'] for x in plan['same_prompt_controls']['prompts']]
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=plan['model_id'],resolution=512,
            device='cuda:0',scheduler_name='DPM',scheduler_inverse_name='DPM',eager_loading=True,disable_tqdm=True)
        cfg = prc_parser.parse_args([]); cfg.wm_type='PRC'; cfg.scheduler_target='DPM'; configure_prc(cfg,provider)
        for image,info in same:
            set_random_seed(info['seed'])
            inv = provider.exact_inversion(images=image,num_inference_steps=50,decode_inv=True)['zT_torch'].detach().cpu().numpy()
            path = run/'same_prompt'/f"{info['task']:03d}_DPM.npz"; np.savez_compressed(path,z=inv)
            # VAE/SSCD content is identical across native inversion backends.
            j = next(j for j,r in enumerate(rows) if r['id'] == f"G{info['task']:03d}" and r['kind'] == 'same_prompt')
            content.append(content[j].copy()); sscd.append(sscd[j].copy()); noise.append(inv.reshape(4,64,64))
            rows.append(dict(info,id=f"PRC_{info['task']:03d}",carrier='PRC',latent=str(path),latent_sha256=sha(path)))
        assert len(rows) == 520
        np.savez_compressed(run/'features.npz',content=np.stack(content),noise=np.stack(noise),sscd=np.stack(sscd))
        write(run/'feature_rows.json',rows)
        state.update(status='complete',rows=len(rows),fresh_vae_encodings=500,new_generations=10,new_inversions=20,
            seconds=time.time()-state['started_at'],artifacts={n:sha(run/n)for n in ['features.npz','feature_rows.json','sscd_templates.npz']})
        write(run/'extract_complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc)); raise
    finally:
        write(run/'extract_state.json',state)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','extract']); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); run=a.run.resolve(); assert ROOT.is_dir() and run.parent == RUNS
    if a.mode == 'prepare': prepare(run)
    else:
        plan=read(run/'protocol.json'); assert sha(__file__) == plan['scripts'][Path(__file__).name]; extract(run,plan)


if __name__ == '__main__': main()
