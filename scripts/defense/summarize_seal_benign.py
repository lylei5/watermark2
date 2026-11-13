"""Independently recheck every SEAL condition, including failed genuine inputs."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--method', choices=['seal_benign'], default='seal_benign')
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == Path('/data0/liyanlei/watermark_research/runs')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    from scipy.stats import beta
    plan = json.loads((run / 'protocol.json').read_text())
    protocol_hash = sha(run / 'protocol.json')
    rows, source_files = [], {}
    for shard in range(3):
        file = run / f'native_{shard}/complete.json'
        result = json.loads(file.read_text())
        assert result['status'] == 'complete' and result['protocol_sha256'] == protocol_hash
        rows.extend(result['rows'])
        source_files[str(file.relative_to(run))] = sha(file)
    expected = {(s['variant'],s['task'],c['name']) for s in plan['sources'] for c in plan['conditions']}
    assert len(rows) == len(expected) == 1400
    assert {(r['variant'],r['task'],r['condition']) for r in rows} == expected
    groups, public = defaultdict(list), []
    maximum = 0.
    for row in rows:
        for artifact in ['image','latent']:
            assert sha(run / row[artifact]) == row[artifact+'_sha256']
        with np.load(run / row['latent']) as data:
            delta = (data['recovered'].astype(np.float32)-data['reference'].astype(np.float32)).astype(np.float16)
            blocks = delta.reshape(1,4,32,2,32,2).transpose(2,4,0,1,3,5).reshape(1024,16)
            norms = np.sqrt(np.sum(blocks.astype(np.float64)**2,axis=1)).astype(np.float16).astype(np.float64)
        assert np.allclose(norms,row['distances'],atol=.004,rtol=.001)
        maximum = max(maximum,float(np.max(np.abs(norms-row['distances']))))
        assert int(np.sum(norms<2.3)) == row['matching_patches']
        assert row['accepted'] == (row['matching_patches']>=12)
        if row['condition']=='Clean':
            assert row['matching_patches']==row['clean_matching_patches'] and row['accepted']==row['clean_accepted']
        item = {k:v for k,v in row.items() if k not in ['caption','distances']}
        public.append(item)
        for dataset in [row['dataset'],'ALL']:
            groups[(row['variant'],dataset,row['condition'])].append(item)
    summaries=[]
    for key,subset in sorted(groups.items()):
        n=len(subset); k=sum(r['accepted'] for r in subset)
        summaries.append(dict(group=list(key),n=n,accepted=k,rate=k/n,
            cp95=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],
            clean_accepted=sum(r['clean_accepted'] for r in subset),
            mean_matching_patches=float(np.mean([r['matching_patches'] for r in subset]))))
    output=dict(status='complete',protocol_sha256=protocol_hash,scope=plan['scope'],source_files=source_files,
        rows=public,summaries=summaries,artifact_checks=2800,independent_sources_per_variant=100,
        cpu_norm_max_difference=maximum,no_clean_success_filter=True,
        known_code_paper_gaps=plan['known_code_paper_gaps'],paper_full_benchmark_reproduced=False,
        summary_worker_sha256=sha(__file__),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    file=run/'audited_summary.json';assert not file.exists()
    file.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(run=run.name,summaries=[s for s in summaries if s['group'][1]=='ALL'])),flush=True)


if __name__=='__main__':
    main()
