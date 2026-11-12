"""Freeze public inputs for unchanged published baselines and natural-image null."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();root=a.root.resolve();assert str(root)=='/data0/liyanlei/watermark_research'
    run=root/'runs/phase1_public_inputs_20261113';run.mkdir(exist_ok=False)
    previous=root/'runs/phase1_trajectory_native_gs_20261113'
    selection=json.loads((previous/'selection.json').read_text())
    used={r['image_id'] for r in selection['sources'] if 'image_id' in r}
    def scan(obj):
        if isinstance(obj,dict):
            for key,val in obj.items():
                if key in ['image_id','coco_id','cover_image_id']:
                    try:used.add(int(val))
                    except (ValueError,TypeError):pass
                scan(val)
        elif isinstance(obj,list):
            for val in obj:scan(val)
        elif isinstance(obj,str):
            for val in re.findall(r'(\d{12})\.(?:jpg|png)',obj):used.add(int(val))
    for path in selection['exclusion_manifest_sha256']:scan(json.loads(Path(path).read_text()))
    for folder in ['originals','rgb512']:
        for path in (root/'datasets/confirmation100_relay_v11'/folder).rglob('*'):scan(path.name)
    metadata=root/'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
    coco=json.loads(metadata.read_text());licenses={x['id']:x for x in coco['licenses']}
    pool=[x for x in coco['images'] if x['id'] not in used and x['license']==4]
    pool.sort(key=lambda x:hashlib.sha256(('phase1-cover-20261113/'+str(x['id'])).encode()).hexdigest())
    covers=pool[:100];assert len(covers)==100
    i2p=root/'runs/phase1_reprompt_sources_20261113/selection.json'
    targets=json.loads(i2p.read_text())
    # Manual pre-generation review found a dismemberment-themed prompt at index3019.
    safe=[r for r in targets['sources'] if r['source_index']!=3019]
    table=list(csv.DictReader((root/'tools/i2p_benchmark.csv').open()))
    blocked=re.compile(targets['filter'],re.I)
    eligible=[]
    for i,r in enumerate(table):
        if i==3019 or float(r['nudity_percentage'])>0 or float(r['inappropriate_percentage'])>0:continue
        if any(x in r['categories'].lower() for x in ['sexual','self-harm','hate']) or blocked.search(r['prompt']):continue
        eligible.append(dict(source_index=i,prompt=r['prompt'],categories=r['categories'],
            inappropriate_percentage=float(r['inappropriate_percentage']),nudity_percentage=float(r['nudity_percentage'])))
    eligible.sort(key=lambda r:hashlib.sha256(('phase1-i2p-20261113/'+str(r['source_index'])).encode()).hexdigest())
    seen={r['prompt'] for r in safe}
    for row in eligible:
        if len(safe)==100:break
        if row['prompt'] not in seen:safe.append(row);seen.add(row['prompt'])
    assert len(safe)==100
    plan=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        generation_protocol_sha256=sha(previous/'protocol.json'),coco_metadata_sha256=sha(metadata),
        covers=[dict(task=i,image_id=x['id'],file_name=x['file_name'],license=licenses[x['license']],
            flickr_url=x.get('flickr_url'),original_coco_url=x['coco_url'],
            download_url='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+x['file_name']) for i,x in enumerate(covers)],
        targets=safe,i2p_source_sha256=sha(root/'tools/i2p_benchmark.csv'),i2p_selection_sha256=sha(i2p),
        i2p_limit=targets['scope'],manual_prompt_exclusions=[dict(source_index=3019,reason='Dismemberment-themed description; pre-generation content review')],
        cover_selection='SHA256 ranking; license4(CC BY), excludes inspected prior IDs and all current generation source IDs',
        cover_preprocessing='Pillow Image.open(raw).resize((512,512)).convert(RGB); no additional transform',
        failure_policy='Preserve fixed source IDs and failures; no replacements or automatic retries',
        maximum_download_bytes=8*1024*1024,parallel_downloads=4,attack_jobs=0,
        scope='Prepared inputs only; filtered I2P is not an unrestricted paper replication; no image redistribution to Git')
    write(run/'protocol.json',plan);print('FROZEN',sha(run/'protocol.json'),flush=True)
    print('NEW_TARGET',json.dumps(safe[-1]),flush=True)
    for name in ['originals','rgb512']:(run/name).mkdir()
    def fetch(item):
        from PIL import Image
        raw=run/'originals'/item['file_name'];image=run/'rgb512'/f"{item['task']:03}.png"
        try:
            with urllib.request.urlopen(item['download_url'],timeout=30) as response:
                data=response.read(plan['maximum_download_bytes']+1)
            assert 0<len(data)<=plan['maximum_download_bytes']
            raw.write_bytes(data)
            with Image.open(raw) as opened:
                dimensions=list(opened.size);opened.resize((512,512)).convert('RGB').save(image)
            return dict(**item,status='complete',source_dimensions=dimensions,raw_sha256=sha(raw),
                image=str(image.relative_to(run)),image_sha256=sha(image),bytes=len(data))
        except Exception as exc:return dict(**item,status='failed',error=repr(exc))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,item) for item in plan['covers']]):rows.append(future.result())
    rows.sort(key=lambda r:r['task']);ok=sum(r['status']=='complete' for r in rows)
    write(run/'acquisition.json',dict(status='complete' if ok==100 else 'partial',successful=ok,required=100,rows=rows,
        protocol_sha256=sha(run/'protocol.json'),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ACQUIRED',ok,100,flush=True)


if __name__=='__main__':main()
