"""Frozen strong controls and unchanged native/PGID verification of Reprompt."""
import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')
ATTACK = ROOT / 'runs/author_reprompt_control_20261114_v2'
COVERAGE = ROOT / 'runs/pgid_carrier_coverage_20261113'
SHARED = ROOT / 'runs/shared_channel_binding_20261114_v2'
COPY = ROOT / 'runs/copynce_control_20261114'
PYTHON = ROOT / 'envs/pgid_revalidation_py310/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def wait_jobs():
    fds = []
    for job in read(ATTACK / 'launch.json')['jobs']:
        if (ATTACK / job['model'] / 'complete.json').exists():
            continue
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, job['pid'], 0)
        if fd >= 0:
            command = Path(f"/proc/{job['pid']}/cmdline").read_bytes()
            assert str(ATTACK).encode() in command
            fds.append(fd)
        else:
            assert ctypes.get_errno() == 3
    for fd in fds:
        select.select([fd], [], [])
        os.close(fd)
    rows = []
    for model in ['sd21', 'sd15']:
        done = read(ATTACK / model / 'complete.json')
        assert done['status'] == 'complete' and done['count'] == 30
        assert done['protocol_sha256'] == sha(ATTACK / 'protocol.json')
        assert sha(ATTACK / model / 'rows.json') == done['rows_sha256']
        rows.extend(read(ATTACK / model / 'rows.json'))
    return rows


def prepare(run):
    run.mkdir(exist_ok=False)
    for name in ['evaluate_author_reprompt.py', 'copynce_control.py']:
        shutil.copy2(Path(__file__).with_name(name), run / name)
    plan = dict(created_at=time.time(), attack=str(ATTACK), attack_protocol_sha256=sha(ATTACK / 'protocol.json'),
        local_sha256={p.name: sha(p) for p in run.glob('*.py')},
        coverage_protocol_sha256=sha(COVERAGE / 'protocol.json'),
        copynce_protocol_sha256=sha(COPY / 'protocol.json'), copynce_summary_sha256=sha(COPY / 'summary.json'),
        shared_protocol_sha256=sha(SHARED / 'protocol.json'),
        raw_inversion='Same author validate dispatch:GS/TR DDIM; PRC exact DPM decode_inv=True.',
        defended_inversion='Unchanged PGIDPipeline.run_defense; default per-carrier k,50steps,s3,gamma0.001.',
        attack_references=30, attack_candidates=60, native_decisions=120,
        content_controls=['SSCD single/centered/9template float32/int8', 'CopyNCE descriptor', 'CopyNCE matcher'],
        calibration='Reuse prior genuine-only frozen GS30 thresholds; no fitting on Reprompt.',
        private_inputs='Only native verification subprocess reads carrier material. Content extraction has public images/model inputs.',
        attacks_changed=False, allowed_gpu_indices=[7,5,4],
        reuse=dict(run=str(ROOT/'runs/author_reprompt_evaluation_20261114'),prc_pid=30182,
            artifacts={n:sha(ROOT/'runs/author_reprompt_evaluation_20261114'/n) for n in
                ['content_rows.json','features.npz','content_complete.json','GS/rows.json']}))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def bind(gpu):
    assert gpu in [7,5,4]
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert int(info[0]) == gpu and int(info[2]) < 100 and int(info[3]) == 0, info
    os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT/'cache/huggingface'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return info[1]


def content(run, gpu):
    uuid = bind(gpu)
    begin = time.monotonic()
    import numpy as np
    from PIL import Image
    import copynce_control as control
    torch, config, builders = control.imports()
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cp = read(COPY / 'protocol.json')
    for name, digest in cp['author_files'].items():
        assert sha(control.AUTHOR/name) == digest
    refs = read(ATTACK / 'protocol.json')['references']
    queries = read(run / 'queries.json')
    meta = refs + queries
    images = []
    for row in meta:
        assert sha(row['image']) == row['image_sha256']
        im = Image.open(row['image']).convert('RGB')
        assert hashlib.sha256(im.tobytes()).hexdigest() == row['pixels_sha256']
        images.append(im)
    desc, cfg, _ = control.load_model('descriptor', torch, config, builders)
    matcher, _, _ = control.load_model('matching', torch, config, builders)
    desc, matcher = desc.cuda(), matcher.cuda()
    tr = transforms.Compose([transforms.Resize((cfg.input.img_size,cfg.input.img_size), interpolation=InterpolationMode[cfg.input.augmentation.interpolation.upper()]),
        transforms.ToTensor(), transforms.Normalize(cfg.input.augmentation.post_process.normalize.mean,cfg.input.augmentation.post_process.normalize.std)])
    tensors = torch.stack([tr(im) for im in images])
    ds=[]
    with torch.no_grad():
        for i in range(0,len(images),8):
            ds.append(torch.nn.functional.normalize(desc({'images':tensors[i:i+8].cuda()})['feat'],dim=1).cpu().numpy())
        ds=np.concatenate(ds)
        matches=[]
        indices={r['id']:i for i,r in enumerate(refs)}
        for i in range(0,len(queries),8):
            q=queries[i:i+8]
            out=matcher({'queries':tensors[30+i:30+i+len(q)].cuda(), 'references':tensors[[indices[r['id']] for r in q]].cuda()})['preds']
            matches.extend(out.cpu().numpy().reshape(-1).tolist())
    model = torch.jit.load(str(ROOT/'models/sscd/sscd_disc_mixup.torchscript.pt'),map_location='cuda').eval()
    tr = transforms.Compose([transforms.Resize([320,320]),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    features=[]
    with torch.no_grad():
        for im in images:
            features.append(torch.nn.functional.normalize(model(tr(im).unsqueeze(0).cuda()),dim=1).cpu().numpy()[0])
    features=np.stack(features)
    rows=[]
    for i,row in enumerate(queries):
        r=indices[row['id']]
        rows.append(dict(row,descriptor_distance=float(np.clip(1-ds[r]@ds[30+i],0,2)),matcher_distance=1-matches[i],
            sscd_distance=float(np.clip(1-features[r]@features[30+i],0,2))))
    write(run/'content_rows.json',rows)
    np.savez_compressed(run/'features.npz',sscd=features,descriptor=ds)
    write(run/'content_complete.json',dict(status='complete',gpu=gpu,gpu_uuid=uuid,seconds=time.monotonic()-begin,
        rows=60,images=90,artifacts={n:sha(run/n) for n in ['content_rows.json','features.npz']}))


def native(run, carrier, gpu):
    uuid=bind(gpu)
    begin=time.monotonic()
    parent=read(COVERAGE/'protocol.json')
    source=Path(parent['source']['path'])
    for name,digest in parent['source']['files'].items():
        assert sha(source/name)==digest
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from PIL import Image
    from pipe import pipe_utils
    from pgid_defense import PGIDPipeline
    from watermarks.gs_watermark import parser as gs_parser
    from watermarks.tr_watermark import parser as tr_parser
    from watermarks.prc_watermark import parser as prc_parser
    from watermarks.watermark_utils import SemanticWatermarks, validate, get_detection_threshold
    from utils import get_watermark_args,set_random_seed,configure_prc,configure_PGID
    torch.set_num_threads(4)
    cfg={'GS':gs_parser,'TR':tr_parser,'PRC':prc_parser}[carrier].parse_args([])
    cfg.wm_type,cfg.attack_type=carrier,'forgery'
    cfg.scheduler_target='DPM' if carrier=='PRC' else 'DDIM'
    cfg.prc_key_path=str(source/'watermarks/prc_key.pkl')
    set_random_seed(parent['seed'])
    kwargs=get_watermark_args(cfg,'cuda:0')
    provider=pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent['model'],resolution=512,device='cuda:0',
        scheduler_name=cfg.scheduler_target,scheduler_inverse_name=cfg.scheduler_target,eager_loading=True,disable_tqdm=True)
    configure_prc(cfg,provider); configure_PGID(cfg)
    defense=PGIDPipeline(provider,provider.scheduler.config,num_inference_steps=cfg.num_defense_steps,
        stopping_timestep=cfg.stopping_timestep,num_skip_inversion=cfg.num_skip_inversion,guidance_strength=cfg.guidance_strength)
    threshold=get_detection_threshold(carrier,parent['model'])
    out=run/carrier;out.mkdir(exist_ok=False);(out/'latents').mkdir()
    queries=read(run/'queries.json')
    csv_paths=([ROOT/f'runs/defense_revalidation_20261111/pgid{k}/attack=forgery/all_results.csv' for k in [0,5]] if carrier=='GS' else [COVERAGE/carrier/'attack=forgery/all_results.csv'])
    originals=[r for path in csv_paths for r in csv.DictReader(path.open())]
    reuse=read(run/'protocol.json')['reuse']
    rows=read(Path(reuse['run'])/'GS/rows.json') if carrier=='GS' else []
    if carrier=='GS':
        assert sha(Path(reuse['run'])/'GS/rows.json')==reuse['artifacts']['GS/rows.json']
        assert len(rows)==20
        for row in rows:assert sha(row['latent'])==row['latent_sha256']
    completed={(r['id'],r['proxy'],r['defended']) for r in rows}
    for wm,idx in SemanticWatermarks[carrier].value.generate_watermark_signals(total_num_signals=10,latent_shape=provider.get_latent_shape(),
            batch_size=1,target_start_index=0,target_end_index=10,w_seed=parent['seed'],**kwargs):
        original=next(r for r in originals if int(r['image_idx'])==idx and r['stage']=='watermarked')
        initial=original.get('message_bits_str_initial') or None
        if carrier in ['GS','PRC']:
            assert initial==''.join(format(b,'08b') for b in wm.message)
        for query in [r for r in queries if r['carrier']==carrier and r['task']==idx]:
            assert sha(query['image'])==query['image_sha256']
            image=Image.open(query['image']).convert('RGB')
            for defended in [False,True]:
                if (query['id'],query['proxy'],defended) in completed:
                    continue
                set_random_seed(parent['seed'])
                if defended:
                    z=defense.run_defense(image)['zT']
                elif carrier=='PRC':
                    z=provider.exact_inversion(images=image,num_inference_steps=50,decode_inv=True)['zT_torch'].detach()
                else:
                    with torch.no_grad():
                        z=provider.invert_images(image,num_inference_steps=50)['zT_torch']
                result=validate(image,wm,provider,50,initial,provided_zT=z)
                metric=float(result['bit_accuracy']) if carrier=='GS' else float(result['p_value']) if carrier=='TR' else bool(result['PRC_detection_results'])
                accepted=metric>181/256 if carrier=='GS' else metric<threshold if carrier=='TR' else metric
                latent=out/'latents'/f"{query['id']}_{query['proxy']}_{int(defended)}.npz"
                np.savez_compressed(latent,z=z.detach().cpu().numpy())
                rows.append(dict(id=query['id'],carrier=carrier,task=idx,proxy=query['proxy'],defended=defended,
                    pixels_sha256=query['pixels_sha256'],metric=metric,accepted=accepted,latent=str(latent),latent_sha256=sha(latent)))
                write(out/'rows.json',rows)
    assert len(rows)==40
    write(out/'complete.json',dict(status='complete',gpu=gpu,gpu_uuid=uuid,rows=40,rows_sha256=sha(out/'rows.json'),
        seconds=time.monotonic()-begin,author_threshold=threshold,used_threshold=181/256 if carrier=='GS' else threshold,
        pgid=dict(steps=cfg.num_defense_steps,k=cfg.stopping_timestep,s=cfg.num_skip_inversion,gamma=cfg.guidance_strength)))


def pipeline(run):
    state=dict(status='waiting_for_attack_exit',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'))
    write(run/'state.json',state)
    try:
        rows=wait_jobs();write(run/'queries.json',rows)
        state['status']='evaluating';write(run/'state.json',state)
        script=run/Path(__file__).name
        reuse=read(run/'protocol.json')['reuse']; previous=Path(reuse['run'])
        for name in ['content_rows.json','features.npz','content_complete.json']:
            assert sha(previous/name)==reuse['artifacts'][name]
            shutil.copy2(previous/name,run/name)
        jobs=[]
        for carrier,gpu in [('GS',7),('TR',5)]:
            log=open(run/(carrier+'.log'),'xb')
            p=subprocess.Popen([str(PYTHON),str(script),'native','--run',str(run),'--carrier',carrier,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
            log.close();jobs.append((carrier,p))
        exits={carrier:p.wait() for carrier,p in jobs}
        assert not any(exits.values()),exits
        if not (previous/'PRC/complete.json').exists():
            libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,reuse['prc_pid'],0)
            if fd>=0:
                command=Path(f"/proc/{reuse['prc_pid']}/cmdline").read_bytes();assert str(previous).encode() in command
                select.select([fd],[],[]);os.close(fd)
            else:assert ctypes.get_errno()==3
        done=read(previous/'PRC/complete.json');assert done['status']=='complete'
        assert sha(previous/'PRC/rows.json')==done['rows_sha256']
        shutil.copytree(previous/'PRC',run/'PRC')
        state.update(status='complete',native_rows=120,content_rows=60)
        write(run/'complete.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        write(run/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pipeline','content','native']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--carrier',choices=['GS','TR','PRC']);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode!='prepare':
        assert sha(__file__)==read(a.run/'protocol.json')['local_sha256'][Path(__file__).name]
    if a.mode=='prepare':prepare(a.run)
    elif a.mode=='pipeline':pipeline(a.run)
    elif a.mode=='content':content(a.run,a.gpu)
    else:native(a.run,a.carrier,a.gpu)
