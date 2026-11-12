"""Server unit checks of the constrained embedding and unchanged decoder."""
import argparse
import itertools
import json
import secrets
import tempfile
from pathlib import Path
import numpy as np
from anchor_binding import identity, order
from decision_region_binding import embed, decode, ARMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        for t in range(16):
            secret={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
            secret.update(amplitude_seed=100+t,projection_order_seed=200+t)
            (root/f'{t:03}.json').write_text(json.dumps(secret))
            rng=np.random.default_rng(300+t);z=rng.standard_normal((1,4,64,64)).astype(np.float32);h=rng.integers(0,2,1024,dtype=np.uint8)
            _,c,_,key=identity(root,t);target=c^np.tile(h,16)[order(key,16384)]
            outputs={a:embed(z,h,root,t,a) for a in ARMS}
            for a,out in outputs.items():
                assert decode(out,h,root,t,a)==1.
                if a!='full':assert np.array_equal(np.abs(out),np.abs(z))
            mismatch=lambda out:((out.ravel()>0)!=target).reshape(64,256).sum(0)
            assert np.all(mismatch(outputs['saturated'])==0)
            assert np.array_equal(mismatch(outputs['minimal56']),mismatch(outputs['random56']))
            assert np.all(mismatch(outputs['minimal56'])<=8)
            costs={a:float(np.sum((out.astype(float)-z)**2)) for a,out in outputs.items()}
            assert costs['minimal56']<=costs['random56']<=costs['saturated']
            changed=(outputs['minimal56'].ravel()!=z.ravel())
            # Exchange optimality: every changed cost is <= every retained wrong cost in its group.
            wrong=(z.ravel()>0)!=target
            for b in range(256):
                g=np.arange(b,16384,256);selected=g[changed[g]];retained=g[wrong[g]&~changed[g]]
                if len(selected) and len(retained):assert np.max(np.abs(z.ravel()[selected]))<=np.min(np.abs(z.ravel()[retained]))
            rows.append(dict(identity=t,distortion=costs,flips=int(changed.sum())))
    # Exhaustive reduced-coordinate sign-equivariance check, ideal uniform cipher.
    magnitudes=np.array([.1,.4,.8,1.2]);hist={s:0 for s in itertools.product([-1,1],repeat=4)}
    for s in hist:
        for target in hist:
            x=magnitudes*np.asarray(s);wrong=np.flatnonzero(np.asarray(s)!=target)
            selected=wrong[np.argsort(magnitudes[wrong])[:max(0,len(wrong)-1)]]
            out=x.copy();out[selected]*=-1;hist[tuple(np.sign(out).astype(int))]+=1
    assert len(set(hist.values()))==1
    args.output.write_text(json.dumps(dict(passed=True,identities=16,roundtrips=64,rows=rows,
        ideal_small_system_output_counts=list(hist.values()),scope='Algebraic and combinatorial tests, not image robustness or cryptographic security proof'),indent=2)+'\n')


if __name__=='__main__':main()
