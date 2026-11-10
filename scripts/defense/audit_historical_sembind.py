"""Verifier-only retrospective audit of existing historical SemBind images.

No attack code is imported or executed. These previously inspected assets are
diagnostic data, separate from the independent development/confirmation split.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from content_authentication import recover_groups, content_scores
from absolute_calibration import calibrate_absolute


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    device = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    if len(device) != 4 or device[0] == '6' or int(device[2]) > 50 or int(device[3]) != 0:
        raise RuntimeError('Selected GPU is excluded or not idle')
    write(out/'device_preimport.json', dict(index=device[0], uuid=uuid,
                                          memory_mib=int(device[2]), utilization=int(device[3])))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261110)
    start = time.monotonic()
    sys.path.insert(0, str(root/'scripts/defense'))
    from sembind import SemanticMasker
    records = []
    runs = [('historical30', 'sembind_pairs30_20261102T233720Z', range(1,31)),
            ('historical_named151', 'sembind_151step_20261103T002848Z', range(1,3))]
    for label, folder, ids in runs:
        for task in ids:
            for kind in ['reference', 'forged']:
                path = root/'runs'/folder/f'{task}_{kind}.png'
                if not path.is_file(): raise FileNotFoundError(path)
                records.append(dict(id=f'{label}_{task:02}_{kind}', run=label,
                                    task=task, kind=kind, image_path=str(path), image_sha256=sha(path)))
    write(out/'protocol.json', dict(scope='historical diagnostic only; no new attacks',
        selection='All 30 existing pairs plus both pairs in the named151 directory',
        budget='Actual launch budgets unverified; directory names are not evidence',
        generation='Historical SD2.1/DDIM project adapter',
        independent_confirmation_read=False, rows=records))
    masker = SemanticMasker(root/'models/sembind/dinov2-giant', root/'models/sembind/sembind_masker.pth')
    permutation = masker.perm.cpu().numpy().copy()
    codes = {}
    for row in records:
        codes[row['id']] = masker.code(Image.open(row['image_path']).convert('RGB')).cpu().numpy().astype(np.uint8)
    del masker
    gc.collect(); torch.cuda.empty_cache()
    sys.path.insert(0, str(root/'scripts/cover_pilot'))
    import runtime
    design = json.loads((root/'protocols/confirmation100_design_v7.json').read_text())
    provider = runtime.official_provider(design['proxy'])
    from utils.wm.gs_provider import GsProvider
    wm = GsProvider(latent_shape=(1,4,64,64), device='cuda:0', dtype=torch.float32,
                    message_width_in_bytes=32, num_replications=64, l=1, offset=0)
    # Known-identity verifier uses the historical fixed GS identity. No private
    # values are written to the report or sent to any candidate generator.
    cipherbits = (wm.get_wm_latents()['zT_torch'].detach().cpu().numpy().reshape(-1)>0).astype(np.uint8)
    rows = []
    for record in records:
        ident = record['id']; image = Image.open(record['image_path']).convert('RGB')
        with torch.no_grad():
            latent = provider.invert_images(image, num_inference_steps=50)['zT_torch']
            code = codes[ident]
            mask = (1-2*np.tile(code.astype(np.float32),16)[permutation]).reshape(1,4,64,64)
            ba = float(wm.get_accuracies(latent*torch.from_numpy(mask).to('cuda:0'))['bit_accuracies'][0])
        z = latent.detach().cpu().numpy()
        np.save(out/(ident+'.npy'), z)
        np.savez_compressed(out/(ident+'.npz'), sembind_code=code)
        scores = content_scores(recover_groups(z,cipherbits,permutation),code)
        row = dict(**record, ba=ba, accepted=ba>181/256, **scores,
                   inversion_sha256=sha(out/(ident+'.npy')), features_sha256=sha(out/(ident+'.npz')))
        if record['kind']=='forged':
            old = json.loads(Path(record['image_path']).with_name(f"{record['task']}_result.json").read_text())
            row.update(historical_ba=old['ba_unbound'], ba_difference=ba-old['ba_unbound'])
        rows.append(row)
        write(out/'progress.json',dict(completed=len(rows),total=len(records)))
        print('SCORED',len(rows),len(records),ident,ba,flush=True)
    summaries=[]; comparisons=[]
    for label, _, _ in runs:
        genuine=[r for r in rows if r['run']==label and r['kind']=='reference']
        candidates=[r for r in rows if r['run']==label and r['kind']=='forged']
        for kind, subset in [('reference',genuine),('forged',candidates)]:
            summaries.append(dict(run=label,kind=kind,n=len(subset),accepted=sum(r['accepted'] for r in subset),
                mean_ba=float(np.mean([r['ba'] for r in subset])),min_ba=min(r['ba'] for r in subset)))
        for target in [0.9,1.0]:
            for field in ['ba','raw_agreement','content_agreement','profiled_content_agreement','profile_content_lower_bound']:
                cal=calibrate_absolute([r[field] for r in genuine],[r['accepted'] for r in genuine],target)
                item=dict(run=label,score=field,**cal)
                if cal['feasible']:
                    item['candidate_n']=len(candidates)
                    item['candidate_accepted']=sum(r['accepted'] and r[field]>=cal['threshold'] for r in candidates)
                comparisons.append(item)
    result=dict(completed=True,scope='retrospective diagnostic; clean genuine references only; not confirmation',
        new_attacks_run=False,budget_verified=False,seconds=time.monotonic()-start,summary=summaries,
        comparisons=comparisons,rows=rows,script_sha256=sha(__file__),torch=torch.__version__,
        provider_sha256=sha(root/'third_party/semantic-forgery/utils/pipe/pipe_provider.py'),
        masker_sha256=sha(root/'scripts/defense/sembind.py'),model=design['proxy'],
        max_historical_ba_difference=max(abs(r['ba_difference']) for r in rows if 'ba_difference' in r))
    write(out/'results.json',result)
    print(json.dumps(dict(summary=summaries,max_historical_ba_difference=result['max_historical_ba_difference'])),flush=True)


if __name__=='__main__':
    main()
