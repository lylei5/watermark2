"""Internal verifier ablation: omit the query term from the linear decoder.

Uses authorized provider secrets inside249; produces no images or attack outputs.
This tests whether exact-key recovery itself requires image evidence. It is not
an adversary experiment and does not assume the provider secret is public.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main(run):
    assert str(run).startswith('/data0/liyanlei/watermark_research/runs/')
    os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    sys.path[:0]=[str(run),'/data0/liyanlei/watermark_research/tools/transport_recovery_dependencies']
    import numpy as np
    import transported_key_capsule as codec
    plan=json.loads((run/'protocol.json').read_text()); private=Path(plan['private'])
    provider=(private/'provider.bin').read_bytes(); rows=[]
    assert hashlib.sha256(provider).hexdigest()==plan['provider_sha256']
    entries=json.loads((run/'enrollment.json').read_text())
    for row in entries:
        filename=f"{row['id']}_{row['method']}.json" if 'method' in row else f"{row['id']}_{row['alpha']}.json"
        path=private/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['helper_sha256']
        record=json.loads(path.read_text());m=record['metadata'];h=codec.unpack(m['capsule'])
        b,_=codec.basis(provider,m['id'],m.get('basis_kind','dct')) if m.get('version',1)>=2 else codec.basis(provider,m['id'])
        a=m['alpha']*b.reshape(-1,128)
        # Identical least-squares/sign decoder with its image contribution omitted.
        estimate=np.linalg.solve(a.T@a,a.T@h.mean(axis=2).reshape(-1))
        key=np.packbits((estimate>=0).astype(np.uint8)).tobytes()
        confirmed=hashlib.sha256(b'transported/'+m['id'].encode()+key).hexdigest()==m['confirmation']
        rows.append(dict(id=m['id'],method=row.get('method',str(row['alpha'])),helper_sha256=row['helper_sha256'],exact_key_without_query=confirmed))
    groups=[]
    for method in sorted({r['method'] for r in rows}):
        part=[r for r in rows if r['method']==method]
        groups.append(dict(method=method,exact_keys=sum(r['exact_key_without_query'] for r in part),total=len(part)))
    out=dict(status='complete',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),groups=groups,rows=rows,
        query_images_read=0,attack_generation=False,native_detection=False,
        scope='Internal decoder ablation with provider secret. Final image certificates are not evaluated; no claim of secret-free forgery or final defense acceptance.')
    target=run/'query_dependence.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
