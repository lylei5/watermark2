"""Server-only algebra, coupling, and marginal checks for the frozen carrier."""
import hashlib
import json
from pathlib import Path
import secrets
import tempfile
import sys
import numpy as np
from scipy.fft import dctn
from scipy.stats import norm, kstest
from quantile_binding import embed, decode
from anchor_binding import identity, order, partition


def main():
    output=Path(sys.argv[1]);samples={a:[] for a in ['quantile_coupled','quantile_independent']}
    roundtrips=0;bit_errors=0
    with tempfile.TemporaryDirectory(prefix='quantile_unit_',dir=output.parent) as temporary:
        root=Path(temporary)
        for i in range(32):
            private=dict(key=secrets.token_hex(32),nonce=secrets.token_hex(16),message=secrets.token_hex(32),
                         binding_key=secrets.token_hex(32),amplitude_seed=71000+i,independent_anchor_seed=72000+i)
            (root/f'{i:03}.json').write_text(json.dumps(private))
            rng=np.random.default_rng(73000+i);aux=rng.normal(size=(1,4,64,64)).astype(np.float32)
            code=rng.integers(0,2,1024,dtype=np.uint8);_,cipher,msg,key=identity(root,i)
            expected=cipher^np.tile(code,16)[order(key,16384)]
            low,high=partition('anchor_low')
            assert np.array_equal(np.bincount(np.arange(len(low))%256,minlength=256),np.full(256,32))
            assert np.array_equal(np.bincount(np.arange(len(high))%256,minlength=256),np.full(256,32))
            for arm in samples:
                z=embed(aux,code,root,i,arm);v=dctn(z.astype(float),axes=(-2,-1),norm='ortho').ravel()
                base=dctn(aux.astype(float),axes=(-2,-1),norm='ortho').ravel() if arm=='quantile_coupled' else np.random.default_rng(72000+i).normal(size=16384)
                assert np.isfinite(v).all()
                assert np.array_equal(v[low]>0,base[low]>0)
                assert np.allclose(np.abs(v[high]),np.abs(base[high]),atol=3e-7,rtol=1e-6)
                raw=(v>0).astype(np.uint8);raw[low]=(np.abs(v[low])>=norm.ppf(.75))
                raw=raw[np.concatenate([low,high])]
                errors=int(np.count_nonzero(raw!=expected));bit_errors+=errors
                assert errors==0,(i,arm,errors)
                independent=((raw^np.tile(code,16)[order(key,16384)]^cipher^np.tile(msg,64)).reshape(64,256).sum(0)>32)
                assert np.array_equal(independent,msg)
                assert decode(z,code,root,i,arm)==1.;roundtrips+=1
                samples[arm].append(v)
    stats={}
    for arm,parts in samples.items():
        v=np.concatenate(parts);ks=float(kstest(v,'norm').statistic)
        stats[arm]=dict(n=len(v),mean=float(v.mean()),variance=float(v.var()),ks_statistic=ks)
        assert abs(v.mean())<.01 and abs(v.var()-1)<.02 and ks<.005,stats[arm]
    result=dict(passed=True,identities=32,roundtrips=roundtrips,raw_bit_errors=bit_errors,
                magnitude_repetitions_per_bit=32,sign_repetitions_per_bit=32,
                statistics=stats,scope='Synthetic algebra and marginal sanity checks; not robustness, security, or joint-distribution proof',
                source_sha256={p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest() for p in ['test_quantile_binding.py','quantile_binding.py','anchor_binding.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
