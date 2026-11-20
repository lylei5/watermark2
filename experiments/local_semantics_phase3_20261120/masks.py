"""Obtain author masks for descriptive size analysis after the verifier is frozen."""
import csv
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from additional import IDS, REV, fetch


def main(folder):
    folder.mkdir(parents=True,exist_ok=True)
    selected=set([12,25,27,35,47,64,89,90]+IDS)
    receipt=folder/'mask_sizes.csv'
    cached={}
    if receipt.is_file():
        with receipt.open() as f:
            cached={int(r['item']):r for r in csv.DictReader(f)}
    def record(i):
        path=folder/f'{i}.png'
        if i in cached and path.is_file():
            raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==cached[i]['sha256']
            scene=cached[i]['scene']
        else:
            url=('https://datasets-server.huggingface.co/rows?dataset=osunlp%2FMagicBrush'
                 f'&config=default&split=dev&offset={i}&length=1')
            r=json.loads(fetch(url))['rows'][0]
            assert r['row_idx']==i
            url=r['row']['mask_img']['src']
            assert f'/--/{REV}/' in url
            scene=r['row']['img_id']
            raw=fetch(url)
            path.write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as im:
            assert im.mode=='RGBA'
            hist=im.getchannel('A').histogram()
            fraction=sum(hist[:128])/sum(hist)
        return dict(item=i,scene=scene,masked_fraction=fraction,sha256=hashlib.sha256(raw).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(record,sorted(selected)))
    assert {r['item'] for r in out}==selected and len({r['scene'] for r in out})==len(selected)
    with (folder/'mask_sizes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(out[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(out)


if __name__=='__main__':
    main(Path(sys.argv[1]))
