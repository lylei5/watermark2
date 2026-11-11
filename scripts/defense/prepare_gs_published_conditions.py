"""Invoke unchanged CVPR2024 Gaussian Shading distortion code and run.sh settings."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from PIL import Image


PARAMETERS=[('random_crop_ratio',.6),('random_drop_ratio',.8),('brightness_factor',6),
            ('jpeg_ratio',25),('gaussian_blur_r',4),('gaussian_std',.05),
            ('median_blur_k',7),('sp_prob',.05),('resize_ratio',.25)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True);args=parser.parse_args()
    run=args.run;source=args.source_run
    assert not (run/'protocol.json').exists()
    origin=json.loads((run/'upstream.json').read_text())
    assert origin['commit']=='09c678fadc7545acf7be12647ddf2a5e66f6a9dc'
    for name,digest in origin['files'].items():assert sha(run/'vendor'/name)==digest,name
    spec=importlib.util.spec_from_file_location('gs_official_distortions',run/'vendor/image_utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inherited=json.loads((source/'protocol.json').read_text())
    plan=dict(inherited);plan['rows']=[]
    (run/'queries').mkdir();(run/'transform_work').mkdir()
    os.chdir(run/'transform_work')
    for task in inherited['identity_tasks']:
        original=source/'queries'/f'{task:03}_sembind_scattered_clean.png'
        for name,value in PARAMETERS:
            options=SimpleNamespace(**{key:None for key,_ in PARAMETERS});setattr(options,name,value)
            seed=2026111100+task
            module.set_random_seed(seed)
            query=module.image_distortion(Image.open(original).convert('RGB'),seed,options)
            label='gs_'+name+'_'+str(value);ident=f'{task:03}_sembind_scattered_{label}'
            output=run/'queries'/(ident+'.png');query.save(output)
            plan['rows'].append(dict(id=ident,task=task,method='sembind_scattered',label=label,
                split='confirmation',image='queries/'+output.name,image_sha256=sha(output),
                source_clean_sha256=sha(original),seed=seed,upstream_parameter={name:value}))
    plan.update(scope='Cross-condition evaluation using the unchanged CVPR2024 Gaussian Shading distortion suite',
        confirmation_queries=576,identity_decisions=36864,wrong_identity_decisions=36288,
        upstream=origin,condition_parameters=[dict(parameter=k,value=v) for k,v in PARAMETERS],
        parent_confirmation_protocol_sha256=sha(source/'protocol.json'),
        freeze_time_utc=datetime.now(timezone.utc).isoformat(),
        confirmation_decisions_read_at_freeze=False,
        new_attacks_run=True,novel_or_modified_attacks=False,published_distortion_instances_generated=True,
        baseline_scope='Official image transformations and published settings on project SemBind references; not a full Gaussian Shading method reproduction',
        limitations=['Same 64 sources as the six-condition confirmation; report suites separately',
                     'Gaussian Shading crop is zero-padded, not crop-and-resize',
                     'No changes to upstream functions or published strengths',
                     'No improved or novel attacks; no claim of complete defense superiority'])
    names=list(inherited['source_sha256'])+['prepare_gs_published_conditions.py','after_confirmation.py']
    plan['source_sha256']={name:sha(run/'scripts'/name) for name in names}
    (run/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(queries=len(plan['rows']),protocol_sha256=sha(run/'protocol.json'),upstream=origin)))


if __name__=='__main__':main()
