"""Receive and verify the official COCO test2017 archive on249."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

SIZE = 6646970404
CHUNK = 8 * 1024 * 1024
EXPECTED_ETAG = '1d4d2301bdee781394c21718428764d2-793'
ROOT = Path('/data0/liyanlei/watermark_research/datasets')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relay-port', type=int, default=18091)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT
    output.mkdir(exist_ok=True)
    chunks = output / 'chunks'
    chunks.mkdir(exist_ok=True)
    state = dict(status='downloading', pid=os.getpid(), bytes_expected=SIZE,
                 official_url='http://images.cocodataset.org/zips/test2017.zip',
                 transport_url='https://s3.amazonaws.com/images.cocodataset.org/zips/test2017.zip',
                 transport='loopback-only SSH reverse relay; no archive persisted on the PC',
                 expected_etag=EXPECTED_ETAG, chunk_bytes=CHUNK,
                 started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(output / 'state.json', state)
    def receive(index):
        start = index * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        target = chunks / f'{index:04}.bin'
        if target.exists() and target.stat().st_size == end-start+1:
            return target
        request = urllib.request.Request(f'http://127.0.0.1:{args.relay_port}/test2017.zip',
                                         headers={'Range': f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    assert response.status == 206
                    assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{SIZE}'
                    data = response.read(end-start+2)
                assert len(data) == end-start+1
                temporary = target.with_suffix('.part')
                temporary.write_bytes(data)
                temporary.replace(target)
                return target
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    try:
        count = (SIZE + CHUNK - 1) // CHUNK
        with ThreadPoolExecutor(max_workers=4) as pool:
            completed = 0
            for path in pool.map(receive, range(count)):
                completed += 1
                if completed % 50 == 0 or completed == count:
                    save(output / 'progress.json', dict(completed_chunks=completed, total_chunks=count))
                    print('CHUNKS_COMPLETE', completed, count, flush=True)
        state['status'] = 'verifying'
        save(output / 'state.json', state)
        digest = hashlib.sha256()
        md5_parts = []
        archive = output / 'test2017.zip'
        temporary_archive = output / 'test2017.zip.assembling'
        with temporary_archive.open('wb') as destination:
            for index in range(count):
                data = (chunks / f'{index:04}.bin').read_bytes()
                digest.update(data)
                md5_parts.append(hashlib.md5(data).digest())
                destination.write(data)
        etag = hashlib.md5(b''.join(md5_parts)).hexdigest() + '-' + str(count)
        assert temporary_archive.stat().st_size == SIZE
        assert etag == EXPECTED_ETAG, ('8MiB multipart ETag mismatch; do not admit this asset', etag)
        temporary_archive.replace(archive)
        extracted = 0
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.is_dir():
                    assert entry.filename == 'test2017/'
                    continue
                path = Path(entry.filename)
                assert path.parent == Path('test2017') and path.suffix.lower() == '.jpg'
                assert len(path.stem) == 12 and path.stem.isdigit()
                target = output / path
                target.parent.mkdir(exist_ok=True)
                # Reading each member checks the ZIP CRC before persistence.
                data = z.read(entry)
                target.write_bytes(data)
                extracted += 1
        state.update(status='complete', archive_sha256=digest.hexdigest(),
                     actual_etag=etag, extracted_images=extracted,
                     completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(output / 'complete.json', state)
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(output / 'state.json', state)


if __name__ == '__main__':
    main()
