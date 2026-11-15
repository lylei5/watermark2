"""Freeze disjoint COCO natural-null cohorts on249, with duplicate screening."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_image(image):
    from PIL import Image, ImageOps
    return ImageOps.fit(image.convert('RGB'), (512, 512), method=Image.Resampling.BICUBIC,
                        centering=(.5, .5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT / 'datasets'
    output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn
    archive = ROOT / 'datasets/coco_test2017_official'
    receipt = json.loads((archive / 'complete.json').read_text())
    assert receipt['status'] == 'complete' and receipt['actual_etag'] == '1d4d2301bdee781394c21718428764d2-793'
    inputs = sorted((archive / 'test2017').glob('*.jpg'))
    assert len(inputs) == receipt['extracted_images']
    metadata = ROOT / 'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
    val_ids = {int(x['id']) for x in json.loads(metadata.read_text())['images']}
    assert all(int(path.stem) not in val_ids for path in inputs), 'COCO source-ID overlap with prompt pool'
    external = sorted((Path('/data0/liyanlei/FontGuard_repro/exp_data/val2017')).glob('*.jpg'))
    old = json.loads((ROOT / 'runs/unified_runtime_probe_20261115/protocol.json').read_text())
    external += [Path(item['path']) for item in old['inputs']]
    started = time.monotonic()

    def feature(item):
        path, kind = item
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                rgb = common_image(image)
            thumbnail = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            low = dctn(thumbnail, type=2)[:8, :8]
            phash = int.from_bytes(np.packbits(low > np.median(low)).tobytes(), 'big')
            return dict(path=str(path), kind=kind, file_sha256=hashlib.sha256(data).hexdigest(),
                        pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        phash64=f'{phash:016x}', width=width, height=height,
                        source_id=path.stem if kind == 'test2017' else None)
        except Exception as exc:
            return dict(path=str(path), kind=kind, error=repr(exc))

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(feature, [(p, 'external_development') for p in external] + [(p, 'test2017') for p in inputs]))
    errors = [row for row in records if 'error' in row]
    valid = [row for row in records if 'error' not in row]
    parent = list(range(len(valid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    buckets, pixels = {}, {}
    bands = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]
    hashes = []
    near_edges = 0
    for i, row in enumerate(valid):
        code = int(row['phash64'], 16)
        hashes.append(code)
        candidates = set()
        for shift, width in bands:
            bucket = (shift, (code >> shift) & ((1 << width)-1))
            candidates.update(buckets.get(bucket, []))
            buckets.setdefault(bucket, []).append(i)
        for j in candidates:
            if (code ^ hashes[j]).bit_count() <= 4:
                union(i, j)
                near_edges += 1
        if row['pixel_sha256'] in pixels:
            union(i, pixels[row['pixel_sha256']])
        pixels[row['pixel_sha256']] = i
    groups = {}
    for i in range(len(valid)):
        groups.setdefault(find(i), []).append(i)
    representatives, excluded_external = [], 0
    for indices in groups.values():
        rows = [valid[i] for i in indices]
        candidates = [row for row in rows if row['kind'] == 'test2017']
        if any(row['kind'] == 'external_development' for row in rows):
            excluded_external += len(candidates)
            continue
        if candidates:
            row = min(candidates, key=lambda value: value['source_id'])
            representatives.append(dict(row, duplicate_cluster_size=len(candidates)))
    representatives.sort(key=lambda row: hashlib.sha256(('watermark2-null-20261115/' + row['source_id']).encode()).digest())
    protocol = dict(scope='Natural-image calibration/validation, independent of generated and attacked images',
                    worker_sha256=sha(__file__), archive_receipt_sha256=sha(archive / 'complete.json'),
                    archive_sha256=receipt['archive_sha256'], preprocessing='PIL RGB then ImageOps.fit 512x512, BICUBIC, centering(.5,.5), no EXIF transpose',
                    duplicate_policy='Exact RGB512 hashes and transitive 64-bit DCT pHash Hamming<=4; one image per cluster; clusters touching external development excluded',
                    duplicate_limit='Perceptual screening is a heuristic, not a guarantee that all related photographs are detected',
                    ordering='SHA256(watermark2-null-20261115/ + COCO source ID)',
                    original_test_images=len(inputs), external_development_images=len(external),
                    invalid_images=errors, near_duplicate_edges=near_edges, excluded_due_to_external=excluded_external,
                    independent_candidate_clusters=len(representatives),
                    metadata_exclusion_sha256=sha(metadata), split_counts={'calibration': 20000, 'validation': 20000},
                    elapsed_seconds=time.monotonic()-started)
    (output / 'all_features.json').write_text(json.dumps(records, indent=2) + '\n')
    if len(representatives) < 40000:
        protocol['status'] = 'insufficient_after_audit'
        (output / 'audit.json').write_text(json.dumps(protocol, indent=2) + '\n')
        raise RuntimeError(f'Need 40000 clusters; have {len(representatives)}. Do not fill with duplicates.')
    rows = [dict(row, index=i, split='calibration' if i < 20000 else 'validation')
            for i, row in enumerate(representatives[:40000])]
    (output / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    protocol.update(status='complete', manifest_sha256=sha(output / 'manifest.json'),
                    all_features_sha256=sha(output / 'all_features.json'), unused_clusters=len(representatives)-40000)
    (output / 'complete.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
