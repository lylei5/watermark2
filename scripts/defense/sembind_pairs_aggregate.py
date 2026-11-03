"""Aggregate SemBind pair shards."""
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
n = int(sys.argv[2])
rows = []
for p in sorted(out.glob('summary_shard_*.json')):
    rows.extend(json.loads(p.read_text(encoding='utf-8'))['rows'])
if len(rows) != n:
    raise SystemExit('expected %d rows, got %d' % (n, len(rows)))
ba_u = [r['ba_unbound'] for r in rows]
ba_b = [r['ba_bound'] for r in rows]
result = dict(n=len(rows),
              sembind_passes=sum(1 for r in rows if r['ba_unbound'] > 0.70703125),
              vanilla_passes=sum(1 for r in rows if r['ba_bound'] > 0.70703125),
              mean_ba_unbound=statistics.mean(ba_u), median_ba_unbound=statistics.median(ba_u),
              min_ba_unbound=min(ba_u), max_ba_unbound=max(ba_u),
              mean_ba_bound=statistics.mean(ba_b),
              rows=rows)
(out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
