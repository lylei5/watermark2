"""Frozen-image verification with phase-two calibration and held-out identities."""
import csv
import gzip
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ONE = HERE.parent/'rgb_moments_phase1_20261120'
TWO = HERE.parent/'rgb_moments_phase2_20261120'
sys.path[:0] = [str(TWO), str(ONE)]
from prepare import ROOT, read, hashfile
from generate import idle_gpu


def put(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def unit(x):
    return x/np.maximum(1e-12,np.linalg.norm(x,axis=-1,keepdims=True))


def largest(d):
    return np.partition(d,-4,axis=-1)[...,-4:].mean(-1)


def aligned(a,b):
    return np.maximum(0,1-(a*b).sum(-1))


class Matcher:
    def __init__(self, views, identity):
        self.g = unit(views[:,0])
        self.p = unit(views[:,1:])
        self.m = unit(views[:,1:].mean(1))
        self.projections = {}
        self.enrolled = {}
        for name in ['public64','E','key2','key3','key4','key5']:
            seed = hmac.digest(('rgb3-study-'+name).encode(),identity.encode(),'sha256')
            rng = np.random.default_rng(int.from_bytes(seed[:16],'big'))
            proj = rng.standard_normal((self.p.shape[-1],64), dtype=np.float32)
            self.projections[name] = proj
            self.enrolled[name] = unit(self.p@proj)
        rng = np.random.default_rng(20261120)
        self.perm = rng.permutation(self.p.shape[-1])
        self.sign = rng.choice([-1,1],self.p.shape[-1]).astype('float32')

    def score(self, query):
        g,p = unit(query[0]),unit(query[1:])
        d = aligned(self.p,p)
        c = largest(d)
        sim = self.p@p.T
        unordered = np.maximum(0,1-.5*(sim.max(-1).mean(-1)+sim.max(-2).mean(-1)))
        # Tail aggregation in both directions isolates spatial correspondence at fixed pooling.
        unordered_tail = .5*(largest(np.maximum(0,1-sim.max(-1))) + largest(np.maximum(0,1-sim.max(-2))))
        glob = aligned(self.g,g)
        result = dict(B=glob.min(), C=c.min(), B_full=glob[0], C_full=c[0],
                      unordered=unordered_tail.min(), unordered_avg=unordered.min(),
                      spatial_avg=d.mean(-1).min(), pool=aligned(self.m,unit(query[1:].mean(0))).min())
        result['orthogonal'] = largest(aligned(self.p[:,:,self.perm]*self.sign,p[:,self.perm]*self.sign)).min()
        for k,proj in self.projections.items():
            projected = unit(p@proj)
            result[k] = largest(aligned(self.enrolled[k],projected)).min()
            if k=='key2':
                result['wrong_key'] = largest(aligned(self.enrolled['E'],projected)).min()
        return result


def main(stage):
    from time import perf_counter
    started = perf_counter()
    for gpu in (4,5,7):
        uuid = idle_gpu(gpu)
        if uuid:
            break
    else:
        raise SystemExit('No idle permitted GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    import torch
    import transformers
    from transformers import Dinov2Model
    from torchvision.transforms import functional as F
    import evaluate as previous
    sys.path.insert(0,str(ROOT/'runs/order_capsule_confirmation_20261115'))
    from order_recovery_data import waves_module
    torch.set_num_threads(2)
    waves, geom = waves_module(), previous.official_geometry()
    model_dir = ROOT/'models/sembind/dinov2-giant'
    model = Dinov2Model.from_pretrained(model_dir, local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    cfg = read(model_dir/'preprocessor_config.json')

    def encode(images):
        arr = []
        for im in images:
            im = im.resize((224,224),Image.Resampling.BICUBIC)
            arr.append(F.normalize(F.to_tensor(im),cfg['image_mean'],cfg['image_std']))
        with torch.inference_mode():
            batch = torch.stack(arr).cuda().half()
            tokens = model(batch).last_hidden_state.float()
            assert tokens.shape[1:]==(257,1536)
            grid = tokens[:,1:].reshape(len(arr),8,2,8,2,1536).mean((2,4)).reshape(len(arr),64,1536)
            return torch.cat((tokens[:,:1],grid),1).cpu().numpy()

    def edits(im, seed):
        x = F.to_tensor(im)
        out = {'crop50':F.to_pil_image(geom['center_crop'](x,.5)),
               'resize70':F.to_pil_image(geom['resize'](x,.7))}
        for k in previous.EDITS[:3]:
            out[k] = waves.apply_distortion([im],k,strength=.4,distortion_seed=seed)[0]
        return out

    def checked(path, sha):
        assert hashfile(path)==sha
        with Image.open(path) as im:
            return im.convert('RGB')

    old = {}
    for fn in ['pilot_scores.csv.gz','test_scores.csv.gz']:
        with gzip.open(TWO/fn,'rt') as f:
            for r in csv.DictReader(f):
                if r['method']=='local':
                    old[r['split'],int(r['sample']),r['condition']]=float(r['score'])
    rows = []
    def measure(split, domain, cluster, idx, ref, imgs, old_split):
        center = F.to_pil_image(geom['center_crop'](F.to_tensor(ref),.5))
        features = encode([ref,center])
        matcher = Matcher(features, f'{domain}/{split}/{idx}')
        ref_rgb = previous.feature(ref)
        items = list(imgs.items())
        for off in range(0,len(items),4):
            chunk = items[off:off+4]
            tokens = encode([im for _,im in chunk])
            for (cond,im),token in zip(chunk,tokens):
                scores = matcher.score(token)
                scores['A'] = previous.distances(ref_rgb,previous.feature(im))['local']
                prior = old.get((old_split,idx,cond))
                if prior is not None:
                    assert abs(scores['A']-prior)<1e-9
                scores = {k:float(v) for k,v in scores.items()}
                assert np.isfinite(list(scores.values())).all() and abs(scores['C']-scores['orthogonal'])<4e-6
                rows.append(dict(split=split,domain=domain,cluster=cluster,sample=idx,condition=cond,**scores))

    frozen = read(TWO/'calibration.json')
    if stage=='test':
        assert read(HERE/'pilot_decision.json')['expand']
    for split,archive,ids in ([('cal','test',frozen['calibration_ids']),('pilot','refine',list(range(12)))]
                              if stage=='pilot' else [('test','test',frozen['test_ids'])]):
        src = read(ONE/f'{archive}_sources.json')
        for i in ids:
            s = src[i]
            def load_id(j):
                d = ONE/archive/f'{j:03d}'
                return checked(d/'joint_gn8.png', read(d/'record.json')['arms']['joint_gn8']['sha256'])
            ref = load_id(i)
            imgs = dict(clean=ref,**edits(ref,20261120+s['pair']),cross_seed=load_id(i^1),
                        donor=checked(Path(s['donor']),s['donor_sha256']))
            measure(split,s['dataset'],s['pair'],i,ref,imgs,split)

    external = [(e,TWO/'author') for e in read(TWO/'public_manifest.json') if e['group'] in ('color','replacement')]
    if stage=='test':
        external.extend((e,HERE/'additional') for e in read(HERE/'additional/manifest.json'))
    for ex,folder in external:
        domain,i = ex['group'],ex['item']
        is_pilot = i in (12,27,47,89) if domain=='replacement' else i%2==1
        if (stage=='pilot')!=is_pilot:
            continue
        imgs = {k:checked(folder/v,ex['sha256'][k]) for k,v in ex['files'].items()}
        ref = imgs.pop('ref')
        pairs = {domain+'_'+k:v for k,v in imgs.items()}
        pairs.update(edits(ref,20261120+i))
        measure(stage+'_public',domain,i,i,ref,pairs,'public')

    keys = list(rows[0].keys())[5:]
    def threshold(key, group):
        vals = sorted(r[key] for r in group if r['split']=='cal' and r['condition'] in previous.EDITS)
        assert len(vals)==120
        return vals[113]
    if stage=='pilot':
        thresholds = {k:threshold(k,rows) for k in keys if k!='wrong_key'}
        assert abs(thresholds['A']-frozen['thresholds']['local'])<1e-9
    else:
        thresholds = read(HERE/'thresholds.json')
    for r in rows:
        r['D'] = max(r['A']/thresholds['A'],r['C']/thresholds['C'])
    if stage=='pilot':
        thresholds['D'] = threshold('D',rows)
        thresholds['wrong_key'] = thresholds['E']
        put('thresholds.json',thresholds)
    records = []
    for r in rows:
        for k in [*keys,'D']:
            records.append({**{x:r[x] for x in list(r)[:5]},'method':k,'score':r[k],
                            'threshold':thresholds[k],'passed':int(r[k]<=thresholds[k])})
    with gzip.open(HERE/(stage+'_scores.csv.gz'),'wt',newline='') as f:
        w = csv.DictWriter(f,list(records[0]))
        w.writeheader()
        w.writerows(records)
    if stage=='pilot':
        def count(k,split,conds):
            return sum(r['passed'] for r in records if r['method']==k and r['split']==split and r['condition'] in conds)
        legal = count('C','pilot',previous.EDITS)
        a,c = [count(k,'pilot_public',['replacement_output']) for k in ('A','C')]
        put('pilot_decision.json',dict(legal_C=legal,legal_total=60,replacement_A=a,replacement_C=c,
                                       replacement_total=4,expand=bool(legal>=54 and a-c>=2)))
    put(stage+'_run.json',dict(gpu=gpu,uuid=uuid,seconds=perf_counter()-started,pairs=len(rows),rows=len(records),
                               torch=torch.__version__,transformers=transformers.__version__,model=str(model_dir)))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('pilot','test')
    main(sys.argv[1])
