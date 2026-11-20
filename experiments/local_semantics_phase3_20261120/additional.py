"""Retrieve the preselected first-turn MagicBrush scenes, preserving author outputs."""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

REV = '6c254560879df7851233ea0fd2ae2a62764d0ef2'
IDS = [128,135,136,141,151,155,169,177,180,184,190,195]


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main(folder):
    folder.mkdir(parents=True, exist_ok=True)
    api = 'https://datasets-server.huggingface.co/rows?dataset=osunlp%2FMagicBrush&config=default&split=dev&offset=100&length=100'
    records = {r['row_idx']:r['row'] for r in json.loads(fetch(api))['rows']}

    def get(i):
        obj = records[i]
        assert obj['turn_index']==1
        row = dict(group='replacement',item=i,scene=obj['img_id'],instruction=obj['instruction'],
                   dataset_revision=REV, files={},sha256={})
        for role,field in [('ref','source_img'),('output','target_img')]:
            url = obj[field]['src']
            assert f'/--/{REV}/' in url
            name = f'replacement_{i}_{role}.jpg'
            path = folder/name
            if not path.is_file():
                path.write_bytes(fetch(url))
            with Image.open(path) as im:
                im.verify()
            row['files'][role] = name
            row['sha256'][role] = hashlib.sha256(path.read_bytes()).hexdigest()
        return row

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(get,IDS))
    assert len({r['scene'] for r in rows})==len(IDS)
    (folder/'manifest.json').write_bytes(json.dumps(rows,indent=2).encode())


if __name__=='__main__':
    main(Path(sys.argv[1]))
