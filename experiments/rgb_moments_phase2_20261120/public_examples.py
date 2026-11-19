"""Download fixed author outputs and record the identity of each comparison image."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


def main(folder):
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    jobs = []

    def add(group, item, files, **metadata):
        row = dict(group=group, item=item, files={}, **metadata)
        for role, url in files.items():
            name = f"{group}_{item}_{role}." + url.split('?')[0].rsplit('.', 1)[1]
            row["files"][role] = name
            jobs.append((url, name))
        entries.append(row)

    repo = 'https://raw.githubusercontent.com/'
    pitie = repo + 'frcs/colour-transfer/50397f021a92856b151984102573f9ce0de5f6de/'
    add('distribution', 1, {k: pitie + f for k, f in {
        'source':'scotland_house.png','ref':'scotland_plain.png','output':'result_IDT.png'}.items()},
        provenance=pitie)
    luan = repo + 'luanfujun/deep-photo-styletransfer/4801fa2dca2e2b52847c377f451246a39eae154a/examples/'
    for i in range(1, 9):
        add('color', i, dict(source=luan+f'input/in{i}.png', ref=luan+f'style/tar{i}.png',
                             output=luan+f'final_results/best{i}_t_1000.png'), provenance=luan)
    api = 'https://datasets-server.huggingface.co/rows?dataset=osunlp%2FMagicBrush&config=default&split=dev&offset=0&length=100'
    rows = json.load(urlopen(api,timeout=60))['rows']
    for i in (12,25,27,35,47,64,89,90):
        r = rows[i]['row']
        assert rows[i]['row_idx']==i and r['turn_index']==1
        url = r['source_img']['src']
        revision = url.split('/--/')[1]
        add('replacement', i, dict(ref=url,output=r['target_img']['src']), instruction=r['instruction'],
            scene=r['img_id'], dataset_revision=revision, split='dev')

    def download(job):
        url, name = job
        path = folder / name
        if not path.exists():
            data = urlopen(url, timeout=90).read()
            tmp = path.with_suffix('.part')
            tmp.write_bytes(data)
            with Image.open(tmp) as image:
                image.verify()
            tmp.replace(path)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes = dict(pool.map(download, jobs))
    for row in entries:
        row['sha256'] = {k:hashes[v] for k,v in row['files'].items()}
    (folder/'manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf8')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
