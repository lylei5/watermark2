"""Server-only interval and primitive BCH dimension audit of completed diagnostics."""
import hashlib
import json
from pathlib import Path
from scipy.stats import beta


def main():
    root = Path('/data0/liyanlei/watermark_research/runs')
    assert root.is_dir()
    probe = root / 'instance_representation_20261113_v2'
    quantized = root / 'sscd_binary_fe_20261113'
    intervals = []
    for row in json.loads((probe / 'scores.json').read_text())['results']:
        count = row['groups']['evaluation/all']['same']['accepted']
        intervals.append(dict(method=row['method'], anchor=row['anchor'], k=count, n=50,
            clopper_pearson_95=[0.0 if count == 0 else float(beta.ppf(.025, count, 51-count)),
                               1.0 if count == 50 else float(beta.ppf(.975, count+1, 50-count))]))
    dimensions = []
    for row in json.loads((quantized / 'complete.json').read_text())['results']:
        n, t = row['length']-1, row['radius']
        roots = set()
        for i in range(1, 2*t+1):
            k = i % n
            while k not in roots:
                roots.add(k)
                k = (2*k) % n
        dimensions.append(dict(repetition=row['repetition'], selector=row['selector'],
            observed_bits=row['length'], bch_n=n, t=t, designed_distance=2*t+1,
            dimension=n-len(roots),
            note='Remove one observed bit: this upper-bounds remaining errors; code dimension only, decoder not executed'))
    for path, value in [(probe / 'wrong_instance_intervals.json', intervals),
                        (quantized / 'bch_dimensions.json', dimensions)]:
        path.write_text(json.dumps(value, indent=2)+'\n')
    receipt = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [probe/'scores.json', quantized/'complete.json']},
        output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [probe/'wrong_instance_intervals.json', quantized/'bch_dimensions.json']})
    (quantized / 'bounds_audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
