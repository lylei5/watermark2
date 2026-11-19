"""Independent confirmation cohort for the fixed two-distance verifier."""
import argparse
import json

import color_budget_sources as base
from prepare_unified_sources import BLOCKED, PromptIndex, normalize, ROOT

PILOT = ROOT/'runs/color_budget_20261119'
RUN = ROOT/'runs/color_joint_20261119'


def select(run=RUN, extra_prior=(), order_domain='colour-joint-confirm', seed_domain='joint-confirm'):
    import pyarrow.parquet as pq
    from pathlib import Path
    from transformers import CLIPTokenizer
    prior = base.read(ROOT/'runs/phase1_trajectory_native_gs_20261113/protocol.json')['sources']
    prior += base.read(ROOT/'runs/unified_main_sources_20261116/selection.json')['sources']
    prior += base.read(PILOT/'sources.json')
    for path in extra_prior:
        prior += base.read(path)
    model = base.read(base.MODEL_MANIFEST)['diffusion_model']['root']
    index = PromptIndex(CLIPTokenizer.from_pretrained(Path(model)/'tokenizer',local_files_only=True))
    used = {(s['dataset'],str(s['source_id'])) for s in prior}
    for s in prior:
        prompt = s.get('caption') or s['prompt']
        tokens,grams,_ = index.encode(prompt)
        index.add(prompt,tokens,grams)
    coco = {}
    annotations = base.read(ROOT/'datasets/confirmation100_relay_v11/metadata/captions_val2017.json')['annotations']
    for row in sorted(annotations,key=lambda r:r['id']):
        coco.setdefault(str(row['image_id']),row['caption'])
    sdp = pq.read_table(ROOT/'runs/phase1_contract_20261113/sdp_eval.parquet')['Prompt'].to_pylist()
    out = []
    for dataset,pool in [('COCO',coco),('SDP',dict((str(i),p) for i,p in enumerate(sdp)))]:
        selected = 0
        for sid in sorted(pool,key=lambda k:base.digest(f'{order_domain}/{dataset}/{k}')):
            prompt = pool[sid]
            tok,grams,n = index.encode(prompt)
            if (dataset,sid) in used or n>75 or not normalize(prompt) or BLOCKED.search(prompt) or index.match(prompt,tok,grams):
                continue
            index.add(prompt,tok,grams)
            seeds=[int(base.digest(f'{seed_domain}/{dataset}/{sid}/{i}')[:15],16) for i in range(2)]
            out.append(dict(task=len(out),dataset=dataset,source_id=sid,prompt=prompt,seeds=seeds,split='test'))
            selected+=1
            if selected==30:
                break
        assert selected==30
    assert len({i for s in out for i in s['seeds']})==120
    with (run/'sources.json').open('x') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('select','generate'))
    p.add_argument('--shard',type=int)
    p.add_argument('--gpu',type=int)
    args=p.parse_args()
    if args.stage=='select':
        select()
    else:
        base.RUN=RUN
        base.generate(args.shard,args.gpu)
