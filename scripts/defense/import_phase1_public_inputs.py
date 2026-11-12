"""Verify relayed fixed source files and normalize them on server 249."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert str(run) == '/data0/liyanlei/watermark_research/runs/phase1_public_inputs_20261113'
    assert not (run / 'acquisition_complete.json').exists()
    initial = json.loads((run / 'acquisition.json').read_text())
    assert initial['protocol_sha256'] == sha(run / 'protocol.json')
    incoming = run / 'relay'
    incoming.mkdir(exist_ok=False)
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        assert len({m.name.rstrip('/') for m in members}) == len(members)
        for member in members:
            destination = (incoming / member.name).resolve()
            assert destination.is_relative_to(incoming)
            assert member.isfile() or member.isdir()
            assert member.size <= 8*1024*1024
        for member in members:
            destination = incoming / member.name
            if member.isdir():
                destination.mkdir(exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('xb') as target:
                    shutil.copyfileobj(source, target)
    receipt = json.loads((incoming / 'transport_receipt.json').read_text())
    policy = json.loads((incoming / 'transport_protocol.json').read_text())
    assert receipt['transport_protocol_sha256'] == sha(incoming / 'transport_protocol.json')
    assert policy['initial_acquisition_sha256'] == sha(run / 'acquisition.json')
    assert policy['protocol_sha256'] == sha(run / 'protocol.json')
    failed = {r['task'] for r in initial['rows'] if r['status'] == 'failed'}
    assert set(policy['tasks']) == failed
    assert {r['task'] for r in receipt['rows']} == failed
    assert len(receipt['rows']) == len(failed)
    transfers = {r['task']: r for r in receipt['rows']}
    from PIL import Image
    rows = []
    for old in initial['rows']:
        row = dict(old)
        if old['status'] == 'complete':
            assert sha(run / 'originals' / old['file_name']) == old['raw_sha256']
            assert sha(run / old['image']) == old['image_sha256']
        else:
            transfer = transfers[old['task']]
            assert transfer['filename'] == old['file_name'] and transfer['image_id'] == old['image_id']
            if transfer['status'] == 'downloaded':
                raw = incoming / 'originals' / old['file_name']
                assert sha(raw) == transfer['sha256'] and raw.stat().st_size == transfer['bytes']
                image = run / 'rgb512' / f"{old['task']:03}.png"
                assert not image.exists()
                with Image.open(raw) as opened:
                    dimensions = list(opened.size)
                    opened.resize((512, 512)).convert('RGB').save(image)
                row.pop('error')
                row.update(status='complete', source_dimensions=dimensions,
                           raw=str(raw.relative_to(run)), raw_sha256=sha(raw), bytes=raw.stat().st_size,
                           image=str(image.relative_to(run)), image_sha256=sha(image), transport='relay')
        rows.append(row)
    count = sum(r['status'] == 'complete' for r in rows)
    result = dict(status='complete' if count == 100 else 'partial', successful=count, required=100,
                  rows=rows, protocol_sha256=sha(run / 'protocol.json'),
                  initial_acquisition_sha256=sha(run / 'acquisition.json'),
                  transport_receipt_sha256=sha(incoming / 'transport_receipt.json'),
                  import_script_sha256=sha(__file__), archive_sha256=sha(args.archive),
                  completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Fixed public image inputs only; normalization on249; no attack execution')
    (run / 'acquisition_complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], successful=count, required=100)))


if __name__ == '__main__':
    main()
