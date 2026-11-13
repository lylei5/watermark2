"""Audit the saved repeated inversions without rerunning any image model."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--method',choices=['determinism'])
    args=p.parse_args();run=args.run.resolve();assert run.parent==Path('/data0/liyanlei/watermark_research/runs')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    rows=[];sources={}
    for shard in range(3):
        file=run/f'native_{shard}/complete.json';data=json.loads(file.read_text())
        assert data['status']=='complete' and data['protocol_sha256']==sha(run/'protocol.json')
        array_file=run/f'native_{shard}/repeated_inversions.npz';assert sha(array_file)==data['artifact_sha256']
        assert sha(data['case']['image'])==data['case']['image_sha256']
        with np.load(array_file) as arrays:
            for item in data['rows']:
                if item['status']=='complete':
                    first,second=[arrays[f"{item['mode']}_{i}"] for i in range(2)]
                    assert item['exact_equal']==bool(np.array_equal(first,second))
                    assert item['max_abs_difference']==float(np.max(np.abs(first-second)))
                    assert item['sign_disagreements']==int(np.sum((first>0)!=(second>0)))
                rows.append(dict(shard=shard,case=data['case'],**item))
        sources[str(file.relative_to(run))]=sha(file)
    result=dict(status='complete',protocol_sha256=sha(run/'protocol.json'),rows=rows,source_files=sources,
        all_deterministic_repeats_exact=all(r.get('exact_equal',False) for r in rows if r['mode']=='deterministic'),
        scope='3 selected diagnostic inputs; paired default/strict repeats withCUBLAS_WORKSPACE_CONFIG=:4096:8; no universal or author-environment equivalence claim',
        summary_worker_sha256=sha(__file__),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    file=run/'audited_summary.json';assert not file.exists();file.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
