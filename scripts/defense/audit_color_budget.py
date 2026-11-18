"""Recheck score decisions, split membership, reference bytes and generation receipts on 249."""
import csv
import gzip
from collections import Counter

import numpy as np
from color_identity_probe import ROOT, read, save, sha


def main():
    run = ROOT/'runs/color_budget_20261119'
    sources = read(run/'sources.json')
    methods = read(run/'color_budget_20261119.json')['methods']
    evidence = read(run/'complete.json')
    for path, expected in evidence['sha256'].items():
        from pathlib import Path
        assert sha(Path(path).read_bytes()) == expected
    assert Counter((r['dataset'],r['split']) for r in sources)=={
        ('COCO','calibration'):20,('COCO','test'):30,('SDP','calibration'):20,('SDP','test'):30}
    hashes = set()
    for i,gpu in enumerate([4,5,7]):
        generated = read(run/f'generated_{i}.json')
        assert generated['gpu']==gpu
        for row in generated['records']:
            s = sources[row['task']]
            assert s['task']%3==i and s['seeds']==row['seeds'] and sha(s['prompt'].encode())==row['prompt_sha256']
            for idx, value in enumerate(row['sha256']):
                assert sha((run/'images'/f"{s['task']:03d}"/f'{idx}.png').read_bytes())==value
                hashes.add(value)
    assert len(hashes)==200
    with gzip.open(run/'scores.csv.gz','rt') as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==len({tuple(r[k] for k in ('task','seed_idx','family','strength','method')) for r in rows})==17000
    assert all(r['split']==sources[int(r['task'])]['split'] and r['dataset']==sources[int(r['task'])]['dataset'] for r in rows)
    cut = read(run/'thresholds.json')
    summary = read(run/'summary.json')['test']
    for m in methods:
        for split,n in [('calibration',80),('test',120)]:
            data = np.fromfile(run/f'{split}_{m}.bin',dtype='<u2')
            assert len(data)==n*144 and np.all(data.reshape(n,144).sum(1)==65535)
        for condition in ('pooled','contrast'):
            relevant = [r for r in rows if r['method']==m and r['family']!='clean' and
                        (condition=='pooled' or (r['family']=='contrast' and r['strength']=='0.8'))]
            cal = [r for r in relevant if r['split']=='calibration']
            test = [r for r in relevant if r['split']=='test']
            assert len(test)==(1920 if condition=='pooled' else 120)
            threshold = np.quantile([float(r['own']) for r in cal],.95,method='inverted_cdf')
            assert threshold==cut[m][condition]
            res = summary[m][condition]
            assert res['counts']['genuine_pass']==sum(float(r['own'])<=cut[m]['pooled'] for r in test)
            assert res['counts']['sibling_pass']==sum(float(r['sibling'])<=cut[m]['pooled'] for r in test)
            assert np.isclose(np.mean([float(r['gallery']) for r in test]),res['rates']['gallery'])
            q = np.quantile([float(r['own']) for r in test],.95,method='inverted_cdf')
            assert np.isclose(np.mean([float(r['sibling'])<=q for r in test]),res['descriptive_test95']['far'])
    save(run/'audit.json',dict(unique_png_hashes=len(hashes),unique_scores=len(rows),methods=methods,
        calibration_references=80,test_references=120,bytes_per_reference=288,
        verified=['receipts and PNG hashes','completed run file hashes','score keys and prompt splits',
                  'reference lengths and masses','pooled and contrast calibration thresholds',
                  'held-out genuine and sibling counts, retrieval and descriptive 95% ROC']))


if __name__=='__main__':
    main()
