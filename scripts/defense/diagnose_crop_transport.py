"""CPU-only known-crop diagnostic on cached genuine DDIM inversions.

Uses the exact existing benign crop, without generating any candidate or
accessing attack scores. Known crop coordinates are oracle side information.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sembind_official_audit import carrier, sha
from codeword_binding import sembind_mask_bits


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    protocol=dict(scope='CPU oracle diagnostic; known benign crop metadata; no new attacks or defense claims',
        images='All 16 SemBind and 4 GS development references with existing clean/crop90 inversions',
        crop_box=[25,25,486,486],original_size=512,query_size=512,latent_size=64,
        mapping='Latent-cell centers interpreted as image coordinates; geometric proxy, not assumed diffusion equivariance',
        modes=['nearest','bilinear'],support='All points with complete in-bounds bilinear stencil; shared across modes',
        threshold='Conservative full-message agreement > 181/256; unsupported bits count as unknown failures',
        erasure_policy='Unsupported bits remain unknown; fixed zero-on-tie decoder plus stricter no-tie evidence reported',
        scores_used_for_transform_selection=False,script_sha256=sha(__file__))
    # Freeze mapping and all cases before computing any metric.
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    axis=torch.arange(64,dtype=torch.float32)+.5
    mapped=2*(axis*8-25)/461-1
    yy,xx=torch.meshgrid(mapped,mapped,indexing='ij')
    grid=torch.stack([xx,yy],-1)[None]
    valid=((xx>=-1+1/64)&(xx<=1-1/64)&(yy>=-1+1/64)&(yy<=1-1/64)).numpy()
    support=np.tile(valid[None],(4,1,1)).reshape(-1)
    rows=[]
    for method,n in [('sembind_scattered',16),('gs_control',4)]:
        for task in range(n):
            prefix=f'{task:03}_{method}'
            paths={kind:args.root/'inversions'/f'{prefix}_{kind}.npy' for kind in ['clean','crop90']}
            clean=np.load(paths['clean'],allow_pickle=False).reshape(1,4,64,64)
            crop=np.load(paths['crop90'],allow_pickle=False).reshape(1,4,64,64)
            _,cipher,message,perm=carrier(args.root,task);stream=cipher^np.tile(message,64)
            if method=='sembind_scattered':
                aux=np.load(args.root/'aux_codes'/f'{task:03}.npz')['sembind_code']
                query=np.load(args.root/'query_features'/f'{prefix}_crop90.npz')['sembind_code']
                source_mask=sembind_mask_bits(aux[None],perm)[0]
                query_mask=sembind_mask_bits(query[None],perm)[0]
            else:source_mask=query_mask=np.zeros(16384,dtype=np.uint8)
            expected=cipher^source_mask
            for mode in ['identity','nearest','bilinear']:
                aligned=crop if mode=='identity' else F.grid_sample(torch.from_numpy(crop),grid,mode=mode,padding_mode='zeros',align_corners=False).numpy()
                signs=(aligned.reshape(-1)>0).astype(np.uint8)
                bits=(signs^query_mask^stream).reshape(64,256)
                observed=support.reshape(64,256)
                totals=observed.sum(0);ones=(bits*observed).sum(0)
                supported=totals>0
                recovered=(ones>totals/2).astype(np.uint8)
                correct=(recovered==message)&supported
                ba=float(correct.sum()/256)
                decisive=supported&(2*ones!=totals)
                decisive_lower=float(((recovered==message)&decisive).sum()/256)
                x=aligned.reshape(-1)[support];y=clean.reshape(-1)[support]
                rows.append(dict(task=task,method=method,mode=mode,oracle_erasure_ba=ba,accepted=ba>181/256,
                    agreement_upper_bound=float((correct.sum()+(~supported).sum())/256),
                    decisive_agreement_lower_bound=decisive_lower,decisive_accepted=decisive_lower>181/256,
                    unknown_or_tied_message_bits=int((~decisive).sum()),
                    supported_bit_accuracy=float(correct.sum()/supported.sum()),unsupported_message_bits=int((~supported).sum()),
                    source_sign_error=float((signs[support]!=expected[support]).mean()),
                    clean_source_sign_error=float(((clean.reshape(-1)>0).astype(np.uint8)[support]!=expected[support]).mean()),
                    clean_inversion_sign_agreement=float((signs[support]==(y>0)).mean()),
                    clean_inversion_cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))),
                    normalized_squared_difference=float(np.sum((x-y)**2)/np.sum(y**2)),
                    observed_fraction=float(support.mean()),min_observations_per_message_bit=int(totals.min()),
                    max_observations_per_message_bit=int(totals.max()),tie_count=int(((2*ones==totals)&supported).sum()),
                    source_sha256={k:sha(v) for k,v in paths.items()}))
    summary=[]
    for method in ['sembind_scattered','gs_control']:
        for mode in ['identity','nearest','bilinear']:
            subset=[r for r in rows if r['method']==method and r['mode']==mode]
            summary.append(dict(method=method,mode=mode,n=len(subset),accepted=sum(r['accepted'] for r in subset),decisive_accepted=sum(r['decisive_accepted'] for r in subset),
                **{'mean_'+k:float(np.mean([r[k] for r in subset])) for k in ['oracle_erasure_ba','decisive_agreement_lower_bound','source_sign_error','clean_source_sign_error',
                    'clean_inversion_sign_agreement','clean_inversion_cosine','normalized_squared_difference','supported_bit_accuracy','unsupported_message_bits']}))
    result=dict(scope=protocol['scope'],observed_fraction=float(support.mean()),summary=summary,rows=rows,
                protocol_sha256=sha(args.output/'protocol.json'),gpu_used=False,confirmation_read=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,observed_fraction=result['observed_fraction']),indent=2))


if __name__=='__main__':main()
