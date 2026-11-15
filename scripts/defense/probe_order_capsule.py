"""Frozen development-only probe of partial-order key recovery on249."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
DATA = ROOT/'runs/instance_representation_20261113_v2'
RUN = ROOT/'runs/order_capsule_probe_20261115'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert not RUN.exists(); RUN.mkdir()
    private = ROOT/'private_eval'/RUN.name; private.mkdir(mode=0o700)
    provider = secrets.token_bytes(32); (private/'provider.bin').write_bytes(provider)
    os.chmod(private/'provider.bin', 0o600)
    for name in ['probe_order_capsule.py', 'order_constrained_capsule.py', 'transported_key_capsule.py']:
        shutil.copy2(Path(__file__).with_name(name), RUN/name)
    plan = dict(created_at=time.time(), tasks=list(range(10)), alpha=.05,
                conditions=['bound', 'gaussian_std', 'brightness_factor', 'same'],
                scripts={p.name:sha(p) for p in RUN.glob('*.py')},
                data_sha256=sha(DATA/'protocol.json'), provider_sha256=sha(private/'provider.bin'),
                scope='Development-only fixed first10previous sources; all outputs preserved. No unseen edits here.',
                mechanism='Exact key from query-induced order cone; no residual acceptance threshold.',
                controls=['same-alpha original forward decoder', 'direct reference order residual'],
                private=str(private), attacks_generated=False)
    write(RUN/'protocol.json', plan)
    state=dict(status='running', pid=os.getpid(), protocol_sha256=sha(RUN/'protocol.json'))
    write(RUN/'state.json',state)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    sys.path[:0]=[str(RUN),str(ROOT/'tools/transport_recovery_dependencies')]
    import numpy as np
    from PIL import Image
    import transported_key_capsule as codec
    import order_constrained_capsule as order
    from scipy.optimize import minimize
    # Independent small constrained optimization verifies projection with ties.
    rng=np.random.default_rng(20261115); tests=[]
    for i in range(12):
        values=rng.normal(size=8); query=rng.integers(0,4,size=8)
        edges=[(a,b) for a in range(8) for b in range(8) if query[a]<query[b]]
        constraints=[dict(type='ineq',fun=lambda z,a=a,b=b:z[b]-z[a]) for a,b in edges]
        reference=minimize(lambda z:.5*np.sum((z-values)**2),np.zeros(8),method='SLSQP',constraints=constraints,options={'ftol':1e-12,'maxiter':500})
        assert reference.success
        error=float(np.max(np.abs(order.project_order(values,query)-reference.x))); assert error<1e-5,error
        tests.append(error)
    write(RUN/'projection_checks.json',dict(cases=12,max_difference=max(tests)))
    data=json.loads((DATA/'protocol.json').read_text())
    spec=importlib.util.spec_from_file_location('distort',data['distortion_file']);dist=importlib.util.module_from_spec(spec);spec.loader.exec_module(dist)
    assert sha(Path(data['distortion_file']))==data['distortion_sha256']
    # Author argument names are asserted, not silently substituted.
    assert set(plan['conditions'][1:3])<=set(data['conditions']),list(data['conditions'])
    (RUN/'work').mkdir();os.chdir(RUN/'work');rows=[];start=time.time()
    for source in data['sources'][:10]:
        task=source['task']; images={}
        for kind in ['bound','same']:
            artifact=source['images'][kind];path=Path(data['parent'])/artifact['path'];assert sha(path)==artifact['sha256'];images[kind]=Image.open(path).convert('RGB')
        for kind in plan['conditions'][1:3]:
            i=list(data['conditions']).index(kind);seed=data['seed_base']+task*100+i
            args=SimpleNamespace(**{k:None for k in data['conditions']});setattr(args,kind,data['conditions'][kind]);dist.set_random_seed(seed)
            images[kind]=dist.image_distortion(images['bound'].copy(),seed,args).convert('RGB')
        identity=f'order_probe_{task:03d}';payload=codec.canonical(dict(id=identity));record,key=codec.enroll(identity,images['bound'],payload,provider,.05,mode='spread_order')
        write(private/f'{identity}.json',record)
        x=codec.rgb(images['bound'],'order').reshape(-1,3)
        for kind in plan['conditions']:
            im=images[kind];tick=time.time();out,detail=order.recover(record,im,provider)
            if out is not None:assert out==(key,payload)
            old,old_detail=codec.observe(record,im,provider,'spread_order')
            if old is not None:assert old==key
            rows.append(dict(task=task,kind=kind,key_recovered=out is not None,detail=detail,
                             forward_key=old is not None,forward_detail=old_detail,
                             order_reference=order.order_residual(x,codec.rgb(im,'order').reshape(-1,3)),
                             image_sha256=hashlib.sha256(im.tobytes()).hexdigest(),seconds=time.time()-tick))
        codec._basis_cache.clear()
    write(RUN/'rows.json',rows)
    aggregate={k:dict(total=sum(r['kind']==k for r in rows),order=sum(r['kind']==k and r['key_recovered'] for r in rows),forward=sum(r['kind']==k and r['forward_key'] for r in rows)) for k in plan['conditions']}
    write(RUN/'aggregate.json',aggregate)
    state.update(status='complete',seconds=time.time()-start,observations=len(rows),aggregate=aggregate,artifacts={n:sha(RUN/n) for n in ['rows.json','aggregate.json','projection_checks.json']})
    write(RUN/'complete.json',state);write(RUN/'state.json',state);print(json.dumps(state))


if __name__=='__main__': main()
