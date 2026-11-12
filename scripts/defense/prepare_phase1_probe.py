"""Select sources and freeze a native-GS / observational trajectory experiment on249."""
import argparse
import datetime
import hashlib
import json
import re
import secrets
import shutil
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['select','freeze'],required=True)
    a=parser.parse_args()
    root=a.root.resolve()
    assert str(root)=='/data0/liyanlei/watermark_research'
    run=root/'runs/phase1_trajectory_native_gs_20261113'
    if a.stage=='select':
        sys.path.insert(0,str(root/'envs/phase1_extra'))
        import pyarrow.parquet as pq
        run.mkdir(exist_ok=True)
        assert not (run/'selection.json').exists() and not (run/'protocol.json').exists()
        previous=set(); previous_ids=set(); manifests={}
        def scan(obj):
            if isinstance(obj,dict):
                for key,value in obj.items():
                    if key in ['caption','prompt','text','source_prompt','generation_prompt'] and isinstance(value,str):
                        previous.add(' '.join(value.lower().split()))
                    if key in ['image_id','coco_id']:
                        try: previous_ids.add(int(value))
                        except (TypeError,ValueError): pass
                    if key in ['filename','file_name','url','path'] and isinstance(value,str):
                        for match in re.findall(r'(\d{12})\.(?:jpg|png)',value): previous_ids.add(int(match))
                    scan(value)
            elif isinstance(obj,list):
                for item in obj: scan(item)
        candidates=list((root/'runs').glob('*/protocol.json'))+list((root/'runs').glob('*/generation_plan.json'))
        candidates+=list((root/'datasets/confirmation100_relay_v11').glob('*manifest*.json'))
        candidates+=list((root/'private_eval').glob('*/generation_plan.json'))
        for path in sorted(set(candidates)):
            if path.stat().st_size>10_000_000: continue
            scan(json.loads(path.read_text()));manifests[str(path)]=sha(path)
        # Fixed pre-generation content filter; no images or outcome scores exist yet.
        blocked=re.compile(r'\b(nsfw|nude|naked|nudity|nudes|undressed|unclothed|no clothes|bathhouse|blood|topless|bottomless|porn\w*|sex\w*|erotic\w*|fetish\w*|breasts?|nipples?|genitals?|penis|vagina|lingerie|bikini|gore|gory|beheaded|decapitat\w*|dismember\w*|rape|hentai|loli\w*|shota\w*)\b',re.I)
        captions=root/'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
        sdp=root/'runs/phase1_contract_20261113/sdp_eval.parquet'
        coco=json.loads(captions.read_text()); by_image={}
        for item in sorted(coco['annotations'],key=lambda x:x['id']):
            by_image.setdefault(item['image_id'],item)
        table=pq.read_table(sdp).to_pylist()
        field=next(k for k in table[0] if k.lower()=='prompt')
        pools={
            'COCO':[dict(source_id=str(k),caption=v['caption'],image_id=k,caption_id=v['id']) for k,v in by_image.items()],
            'SDP':[dict(source_id=str(i),caption=r[field]) for i,r in enumerate(table)]}
        sources=[];selection={}
        for dataset,pool in pools.items():
            pool.sort(key=lambda x:hashlib.sha256(('phase1-trajectory-20261113/'+dataset+'/'+x['source_id']).encode()).hexdigest())
            selected=[];rejected=[]
            for item in pool:
                text=' '.join(item['caption'].lower().split())
                reason='content_filter' if blocked.search(text) else ('recorded_source_overlap' if text in previous or item.get('image_id') in previous_ids else None)
                if reason: rejected.append(dict(source_id=item['source_id'],reason=reason));continue
                previous.add(text)
                selected.append(dict(task=len(sources)+len(selected),dataset=dataset,**item))
                if len(selected)==50: break
            assert len(selected)==50
            sources.extend(selected)
            selection[dataset]=dict(pool_size=len(pool),selected=50,skipped=rejected)
        write(run/'selection.json',dict(sources=sources,selection=selection,
            ordering='SHA256 of fixed domain/dataset/source_id; first caption by annotation id per COCO image',
            source_files={str(captions):sha(captions),str(sdp):sha(sdp)},exclusion_manifest_sha256=manifests,
            scope='New against inspected recorded sources only; not a guarantee of disjoint model training or undocumented experiments',
            content_filter=blocked.pattern,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(sources,ensure_ascii=False,indent=2))
        return
    selection=json.loads((run/'selection.json').read_text())
    review=json.loads((run/'prompt_review.json').read_text())
    assert review['selection_sha256']==sha(run/'selection.json') and review['approved_tasks']==list(range(100))
    private=root/'private_eval/phase1_trajectory_native_gs_20261113'
    private.mkdir(mode=0o700,exist_ok=False)
    secret=dict(binding_key=secrets.token_hex(32),
        sources={str(i):dict(torch_seed=secrets.randbits(63),numpy_seed=secrets.randbits(32)) for i in range(100)})
    write(private/'seeds.json',secret);(private/'seeds.json').chmod(0o600)
    for row in selection['sources']:
        row['aux_seed']=secrets.randbits(63);row['independent_seed']=secrets.randbits(63)
    names=['prepare_phase1_probe.py','run_phase1_probe_worker.py','run_phase1_probe.py',
           'summarize_phase1_probe.py','sembind_official_audit.py','semantic_code.py','codeword_binding.py']
    for name in names: shutil.copy2(Path(__file__).parent/name,run/name)
    upstream=root/'runs/phase1_contract_20261113/gs_watermark.py'
    shutil.copy2(upstream,run/'gs_watermark.py')
    names+=['gs_watermark.py']
    paths=json.loads((root/'runs/sembind_chain_revalidation_20261111/paths.json').read_text())
    previous=json.loads((root/'runs/continuous_geometry_screen_20261113_v2/protocol.json').read_text())
    external={p:h for p,h in previous['external_sha256'].items() if
        '/official/' in p or '/models/sembind/' in p or '/cache/huggingface/' in p or p==paths['scheduler']}
    for path,digest in external.items(): assert sha(path)==digest,path
    plan=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Fresh 50 COCO and50 SDP prompts; observational trajectory and paper-spec GS-S with original GS carrier, not complete SemBind replication',
        sources=selection['sources'],selection_sha256=sha(run/'selection.json'),prompt_review_sha256=sha(run/'prompt_review.json'),
        paths=paths,hf_home=str(root/'cache/huggingface'),extra_dependencies=str(root/'envs/phase1_extra'),
        identity_root=str(private),private_seed_sha256=sha(private/'seeds.json'),
        source_sha256={name:sha(run/name) for name in names},external_sha256=external,
        gpu_indices=[7,5,4],generation=dict(steps=50,cfg=7.5,solver_order=1,dtype='float32',size=512),
        inversion=dict(steps=50,cfg=3.,prompt='',inv_order=0,decoder_inv=True,decoder_optimization_steps=100),
        gs=dict(upstream_commit='09c678fadc7545acf7be12647ddf2a5e66f6a9dc',carrier='Gaussian_Shading default OTP',
            channel_copy=1,hw_copy=8,message_bits=256,repetitions=64,user_number=1,fpr=1e-6,
            sample_dtype='Original FP16 truncSampling output cast to FP32 for released DPM pipeline'),
        binding=dict(sigma=1,expansion='np.tile(code,16)',permutation='HMAC-SHA256 sorted labels with private global binding key',
            permutation_domain='sembind/permutation/',key_scope='one private binding key; fresh OTP key/message per image'),
        working_points=[166,167,182],trajectory_steps=list(range(5,51)),
        trajectory_definition='One-based UNet evaluation; scheduler.model_outputs[-1] after step is guided x0 prediction evaluated at pre-step latent',
        controls=['aux0 vs GS-S final','aux0 vs same prompt independent seed','aux0 vs within-dataset different prompt cyclic next','JPEG50 aux0 and GS-S'],
        primary_step=35,gates=dict(p5_greater_than=.95,paired_mean_advantage_bootstrap_lower_greater_than=0.,
            require_each_dataset=True,required_controls=['aux_final','same_prompt']),
        statistics=dict(bootstrap_resamples=10000,seed=2026111301,cluster='prompt',
            primary='Step35 only; exploratory earlier steps do not select a deployment step'),
        observational_only=True,attack_jobs=0)
    write(run/'protocol.json',plan)
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'),prompts=100,attack_jobs=0)))


if __name__=='__main__':
    main()
