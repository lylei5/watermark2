"""Same-pixel published evaluation inputs for order recovery on249."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

ROOT=Path('/data0/liyanlei/watermark_research')
PARENT=ROOT/'runs/transported_capsule_observability_20261115'
SHARED=ROOT/'runs/shared_channel_binding_20261114_v2'
DATA=ROOT/'runs/instance_representation_20261113_v2'
WAVES=ROOT/'third_party/WAVES_1477635_source'
FAMILIES=['random_crop_ratio','random_drop_ratio','gaussian_std','sp_prob']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixel_sha(image):return hashlib.sha256(image.tobytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def waves_module():
    # Bind only the three unchanged author helpers needed by distortions.py;
    # avoid importing unrelated visualization/dataset dependencies.
    images=module('waves_image_utils',WAVES/'utils/image_utils.py')
    seeds=module('waves_exp_utils',WAVES/'utils/exp_utils.py')
    shim=types.ModuleType('utils');shim.to_tensor=images.to_tensor;shim.to_pil=images.to_pil;shim.set_random_seed=seeds.set_random_seed
    previous=sys.modules.get('utils');sys.modules['utils']=shim
    try:return module('waves_distortions',WAVES/'distortions/distortions.py')
    finally:
        if previous is None:sys.modules.pop('utils',None)
        else:sys.modules['utils']=previous


def author_edit(image,family,seed,plan,distortion):
    args=SimpleNamespace(**{k:None for k in plan['conditions']})
    setattr(args,family,plan['conditions'][family]);distortion.set_random_seed(seed)
    return distortion.image_distortion(image.copy(),seed,args).convert('RGB')


def development_sources():
    from PIL import Image
    plan=read(SHARED/'protocol.json');assert sha(plan['distortion_file'])==plan['distortion_sha256']
    distortion=module('order_author_edit',plan['distortion_file']);cases=read(PARENT/'cases.json');meta=read(SHARED/'feature_rows.json')
    cache={}
    def pixels(row):
        digest=row['pixels_sha256']
        if digest not in cache:
            assert sha(row['image'])==row['image_sha256']
            image=Image.open(row['image'])
            if row.get('resize'):image=image.resize((512,512))
            image=image.convert('RGB')
            if row.get('family'):image=author_edit(image,row['family'],row['seed'],plan,distortion)
            assert pixel_sha(image)==digest;cache[digest]=image
        return cache[digest]
    for identity in sorted({c['base']['id'] for c in cases}):
        part=[c for c in cases if c['base']['id']==identity]
        reference=next(c for c in part if c['base']['kind']=='clean');clean=pixels(reference['query']);carrier=reference['base']['carrier'];task=reference['base']['task']
        templates=[clean]
        if carrier=='GS':
            templates += [pixels(r) for r in meta if r['id']==identity and r['kind'].startswith('enroll_')]
        else:
            for f,family in enumerate(FAMILIES):
                for repeat in range(2):
                    seed=plan['enrollment_seed_base']+['TR','PRC'].index(carrier)*10000+task*100+f*10+repeat
                    templates.append(author_edit(clean,family,seed,plan,distortion))
        assert len(templates)==9
        fields=['id','carrier','task','kind','query_id','split','pixels_sha256','native_metric','native_accepted','case_id','proxy','trial']
        yield identity,clean,templates,[({k:v for k,v in c['base'].items() if k in fields},pixels(c['query'])) for c in part]
        cache.clear()


def confirmation_sources():
    from PIL import Image
    data=read(DATA/'protocol.json');assert sha(data['distortion_file'])==data['distortion_sha256']
    distortion=module('order_author_edit',data['distortion_file']);waves=waves_module()
    for source in data['sources']:
        task=source['task'];identity=f'confirmation_{task:03d}';images={}
        for kind,artifact in source['images'].items():
            path=Path(data['parent'])/artifact['path'];assert sha(path)==artifact['sha256'];images[kind]=Image.open(path).convert('RGB')
        clean=images['bound'];templates=[clean]
        for f,family in enumerate(FAMILIES):
            for repeat in range(2):templates.append(author_edit(clean,family,2026611500+task*100+f*10+repeat,data,distortion))
        queries=[]
        for kind,image in images.items():
            queries.append((dict(id=identity,task=task,kind=kind,case_id=f'{identity}/{kind}',dataset=source['dataset'],split='confirmation',is_genuine=kind=='bound',is_unseen=False),image))
        for i,family in enumerate(data['conditions']):
            image=author_edit(clean,family,data['seed_base']+task*100+i,data,distortion)
            queries.append((dict(id=identity,task=task,kind=family,case_id=f'{identity}/{family}',dataset=source['dataset'],split='confirmation',is_genuine=True,is_unseen=False),image))
        for f,family in enumerate(['contrast','rotation','resizedcrop','brightness','noise']):
            for level in range(1,6):
                strength=level/5;seed=2026711500+task*100+f*5+level
                image=waves.apply_distortion([clean.copy()],family,strength=strength,distortion_seed=seed,relative_strength=True)[0].convert('RGB')
                kind=f'waves_{family}_{level}'
                queries.append((dict(id=identity,task=task,kind=kind,case_id=f'{identity}/{kind}',dataset=source['dataset'],split='confirmation',is_genuine=True,is_unseen=family in ['contrast','rotation','resizedcrop'],family=family,strength=strength,seed=seed),image))
        assert len(queries)==38 and len(templates)==9
        yield identity,clean,templates,queries


def sources(stage):return development_sources() if stage=='development' else confirmation_sources()
