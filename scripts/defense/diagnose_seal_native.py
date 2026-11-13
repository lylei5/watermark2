"""Separate reference drift from inversion errors in saved SEAL clean artifacts."""
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name=='posix'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    def blocks(array):
        return array.reshape(1,4,32,2,32,2).transpose(2,4,0,1,3,5).reshape(1024,16)
    def matches(first,second):
        difference=(first.astype(np.float32)-second.astype(np.float32)).astype(np.float16)
        norms=np.sqrt(np.sum(blocks(difference).astype(np.float64)**2,axis=1)).astype(np.float16)
        return norms.astype(np.float64)<2.3
    rows=[];groups=defaultdict(list);sources={}
    for variant,name in [('released','seal_native_20261113_v3'),('paper','seal_paper_native_20261113')]:
        run=root/'runs'/name
        sources[name]=dict(protocol_sha256=sha(run/'protocol.json'),result_sha256=sha(run/'audited_summary.json'))
        for shard in range(3):
            for item in json.loads((run/f'native_{shard}/complete.json').read_text())['rows']:
                file=run/item['latents']['path'];assert sha(file)==item['latents']['sha256']
                with np.load(file) as data:
                    initial,reference,recovered=[data[k] for k in ['initial','genuine_reference','recovered']]
                    equal=np.all(blocks(initial)==blocks(reference),axis=1)
                    fixed=matches(recovered,initial);actual=matches(recovered,reference)
                decision=next(x for x in item['decisions'] if x['kind']=='genuine')
                assert int(actual.sum())==decision['matches']
                row=dict(variant=variant,task=item['task'],dataset=item['dataset'],accepted=decision['accepted'],
                    matching_patches=int(actual.sum()),identical_carrier_blocks=int(equal.sum()),
                    fixed_embedding_reference_matches=int(fixed.sum()),
                    identical_and_recovered_blocks=int((equal&fixed).sum()),
                    changed_reference_accidental_matches=int((~equal&actual).sum()),
                    actual_rejected_fixed_reference_accepted=bool(not decision['accepted'] and fixed.sum()>=12),
                    fixed_reference_rejected=bool(fixed.sum()<12))
                rows.append(row)
                for dataset in [item['dataset'],'ALL']:
                    groups[(variant,dataset)].append(row)
    summaries=[]
    for key,subset in sorted(groups.items()):
        summaries.append(dict(group=list(key),n=len(subset),accepted=sum(x['accepted'] for x in subset),
            drift_sufficient_for_rejection=sum(x['actual_rejected_fixed_reference_accepted'] for x in subset),
            fixed_reference_rejected=sum(x['fixed_reference_rejected'] for x in subset),
            mean_identical_carrier_blocks=float(np.mean([x['identical_carrier_blocks'] for x in subset])),
            mean_fixed_reference_matches=float(np.mean([x['fixed_embedding_reference_matches'] for x in subset]))))
    official=root/'third_party/frozen/SEAL_92d31b3'
    sys.path.insert(0,str(official))
    from utils import image_distortion,set_random_seed
    array=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
    sample=Image.fromarray(array);noise_checks=[]
    for seed in [0,1,50]:
        set_random_seed(seed)
        output=np.asarray(image_distortion(None,sample,seed=seed,gaussian_std=.1)[1])
        offsets=(output.astype(np.int16)-array.astype(np.int16))%256
        unique=np.unique(offsets)
        assert unique.size==1
        noise_checks.append(dict(seed=seed,shape=list(output.shape),unique_mod256_offsets=unique.tolist()))
    output=dict(scope='CPU diagnosis of existing clean images; fixed embedding reference is unavailable to a normal verifier and is not a proposed detector',
        source_files=sources,rows=rows,summaries=summaries,distortion_source_sha256=sha(official/'utils.py'),
        released_noise_checks=noise_checks,script_sha256=sha(__file__),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    target=root/'runs/three_baselines_20261113/seal_clean_diagnosis.json';assert not target.exists()
    target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(summaries=summaries,noise_checks=noise_checks)))


if __name__=='__main__':
    main()
