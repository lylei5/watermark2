"""File transport only: download fixed COCO objects without local image processing."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();initial=json.loads((a.report/'acquisition.json').read_text(encoding='utf-8'))
    assert initial['protocol_sha256']==sha(a.report/'protocol.json')
    a.output.mkdir(exist_ok=False);(a.output/'originals').mkdir()
    failed=[r for r in initial['rows'] if r['status']=='failed']
    policy=dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(a.report/'protocol.json'),
        initial_acquisition_sha256=sha(a.report/'acquisition.json'),script_sha256=sha(__file__),
        tasks=[r['task'] for r in failed],scope='One additional file-transfer attempt through the local connection; identical frozen URLs/IDs; no source replacements, image processing or model calls',
        maximum_object_bytes=8*1024*1024,parallel_transfers=4)
    (a.output/'transport_protocol.json').write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8',newline='\n')
    def download(row):
        assert re.fullmatch(r'\d{12}\.jpg',row['file_name'])
        assert row['download_url']=='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+row['file_name']
        try:
            with urllib.request.urlopen(row['download_url'],timeout=30) as response:
                data=response.read(policy['maximum_object_bytes']+1)
            assert 0<len(data)<=policy['maximum_object_bytes']
            path=a.output/'originals'/row['file_name'];path.write_bytes(data)
            return dict(task=row['task'],image_id=row['image_id'],filename=row['file_name'],status='downloaded',sha256=sha(path),bytes=len(data))
        except Exception as exc:return dict(task=row['task'],image_id=row['image_id'],filename=row['file_name'],status='failed',error=repr(exc))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download,row) for row in failed]):rows.append(future.result())
    rows.sort(key=lambda r:r['task']);success=sum(r['status']=='downloaded' for r in rows)
    (a.output/'transport_receipt.json').write_text(json.dumps(dict(rows=rows,successful=success,required=len(failed),
        transport_protocol_sha256=sha(a.output/'transport_protocol.json'),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(successful=success,required=len(failed))))


if __name__=='__main__':main()
