"""Paired GPU simulation of coding mechanics, optionally using measured codes.

Only the channel errors are synthetic in the feature-cache experiment.
Neither experiment generates images or measures actual forgery success.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binom
from codeword_binding import CodewordBinding, keyed_order, sembind_mask_bits

METHODS = ['sembind_scattered', 'parity4_scattered',
           'codeword_aligned_degree1', 'codeword_aligned_parity4']
TAU = 181/256


def masks(codes, key):
    perm = keyed_order(key, b'sembind/permutation/', 16384)
    return [sembind_mask_bits(codes, perm),
            CodewordBinding(key, 4, False).mask_bits(codes),
            CodewordBinding(key, 1, True).mask_bits(codes),
            CodewordBinding(key, 4, True).mask_bits(codes)]


def evaluate(codes, pairs, e, key, seed, device):
    all_masks = [torch.from_numpy(x).to(device) for x in masks(codes, key)]
    rng = torch.Generator(device=device).manual_seed(seed)
    result = [[] for _ in METHODS]
    for off in range(0, len(pairs), 64):
        batch = pairs[off:off+64]
        a = torch.tensor([x[0] for x in batch], device=device)
        b = torch.tensor([x[1] for x in batch], device=device)
        n = len(batch)
        msg = torch.randint(0,2,(n,1,256),generator=rng,device=device,dtype=torch.uint8)
        noise = (torch.rand((n,64,256),generator=rng,device=device)<e).to(torch.uint8)
        for mi, mask in enumerate(all_masks):
            delta = (mask[a]^mask[b]).reshape(n,64,256)
            decoded = ((msg ^ noise ^ delta).sum(dim=1) > 32)
            ba = (decoded == msg[:,0]).float().mean(dim=1)
            result[mi].extend(ba.cpu().tolist())
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    start=time.monotonic()
    summary=[]
    if args.features:
        records=json.loads((args.features/'records.json').read_text())
        codes=np.load(args.features/'features.npz')['codes']
        if not (args.features/'manifest.json').exists():
            raise ValueError('Feature extraction must finish before evaluation')
        files=sorted({r['file'] for r in records if r['group']=='natural'})
        idx={(r['group'],r['file'],r['variant']):i for i,r in enumerate(records)}
        pairs=[]; labels=[]
        for split, names in [('development',files[:20]),('held_out_pilot',files[20:])]:
            for name in names:
                for variant in ['jpeg50','blur_resize','crop90']:
                    pairs.append((idx['natural',name,'clean'],idx['natural',name,variant]))
                    labels.append(split+'/'+variant)
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    pairs.append((idx['natural',a,'clean'],idx['natural',b,'clean']))
                    labels.append(split+'/cross_image')
        hist=sorted({r['file'].split('_')[0] for r in records if r['group']=='historical'},key=int)
        for sid in hist:
            pairs.append((idx['historical',sid+'_reference.png','clean'],idx['historical',sid+'_forged.png','clean']))
            labels.append('historical/ref_forged')
        (args.output/'pairs.json').write_text(json.dumps([dict(a=records[a],b=records[b],label=l) for (a,b),l in zip(pairs,labels)],indent=2))
        d=np.array([(codes[a]!=codes[b]).mean() for a,b in pairs])
        for e in [0,.1,.25,.35]:
            scores=[]
            for k in range(5):
                key=hashlib.sha256(('diagnostic-key-'+str(k)).encode()).digest()
                scores.append(evaluate(codes,pairs,e,key,20261109+k,args.device))
            scores=np.array(scores)
            np.savez_compressed(args.output/('scores_e'+str(e)+'.npz'),ba=scores,semantic_distance=d)
            for label in sorted(set(labels)):
                sel=np.array(labels)==label
                for mi,method in enumerate(METHODS):
                    a=scores[:,mi,sel]
                    summary.append(dict(e=e,label=label,method=method,n_pairs=int(sel.sum()),
                                        n_keys=5,mean_ba=float(a.mean()),acceptance=float((a>TAU).mean()),
                                        per_key_acceptance=(a>TAU).mean(axis=1).tolist()))
            print('FINISHED_CHANNEL',e,flush=True)
        distances={label:dict(n=int((np.array(labels)==label).sum()),
                              mean=float(d[np.array(labels)==label].mean()),
                              minimum=float(d[np.array(labels)==label].min()),
                              maximum=float(d[np.array(labels)==label].max())) for label in sorted(set(labels))}
    else:
        distances=None
        rng=np.random.default_rng(20261109)
        for d in [.0,.02,.05,.1,.2,.3,.4,.5]:
            a=rng.integers(0,2,(1024,1024),dtype=np.uint8)
            b=a ^ (rng.random(a.shape)<d).astype(np.uint8)
            codes=np.concatenate([a,b]); pairs=[(i,i+1024) for i in range(1024)]
            key=hashlib.sha256(b'mechanism-key').digest()
            for e in [0,.1,.25,.35]:
                ba=evaluate(codes,pairs,e,key,20261109,args.device)
                for mi,method in enumerate(METHODS):
                    summary.append(dict(semantic_flip_probability=d,e=e,method=method,n=1024,
                                        mean_ba=float(ba[mi].mean()),acceptance=float((ba[mi]>TAU).mean())))
            print('FINISHED_DISTANCE',d,flush=True)
    meta=dict(experiment='real_codes_synthetic_channel' if args.features else 'synthetic_codes_synthetic_channel',
              device=args.device,device_name=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
              torch=torch.__version__,threshold=TAU,comparison='>',strict_minimum_correct_bits=182,
              iid_null_binomial_tail=float(binom.sf(181,256,.5)),
              caveat='Binomial null is theoretical under independent fair bits; no empirical 1e-8 FPR claim.',
              distances=distances,rows=summary,seconds=time.monotonic()-start,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE',meta['seconds'],flush=True)


if __name__=='__main__':
    main()
