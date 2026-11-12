"""Server-side algebraic, numerical, roundtrip and coupling checks."""
import argparse,hashlib,json,secrets,tempfile
from pathlib import Path
import numpy as np
from scipy.stats import norm
from transport_binding import conditional_transport,embed,decode,ARMS
from anchor_binding import identity,order


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(202611128000)
    z=np.linspace(-8,8,4097)
    for bit in [0,1]:
        y=conditional_transport(z,np.full(z.shape,bit))
        assert np.all(np.diff(y)>0) and np.all((y>0)==bit)
        central=np.abs(z)<5
        np.testing.assert_allclose(y[central],norm.ppf((norm.cdf(z[central])+bit)/2),rtol=2e-6,atol=1e-7)
        # Transport inverse recovers the original Gaussian quantile centrally.
        recovered=norm.ppf(2*norm.cdf(y[central])-bit)
        np.testing.assert_allclose(recovered,z[central],rtol=1e-4,atol=5e-4)
    roundtrips=0
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for t in range(16):
            secret={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
            secret.update(amplitude_seed=100+t,transport_shuffle_seed=200+t)
            (root/f'{t:03}.json').write_text(json.dumps(secret))
            z=rng.standard_normal((1,4,64,64)).astype(np.float32);h=rng.integers(0,2,1024,dtype=np.uint8)
            _,cipher,_,key=identity(root,t);target=cipher^np.tile(h,16)[order(key,16384)]
            outputs={arm:embed(z,h,root,t,arm) for arm in ARMS}
            for arm,value in outputs.items():
                assert decode(value,h,root,t,arm)==1 and np.array_equal(value.ravel()>0,target.astype(bool));roundtrips+=1
            for bit in [0,1]:
                positions=target==bit
                assert np.array_equal(np.sort(outputs['transport'].ravel()[positions]),np.sort(outputs['shuffled_transport'].ravel()[positions]))
            # Monotone assignment is the minimum square cost for this output multiset,
            # among permutations within each target orthant (rearrangement inequality).
            minimum=outputs['transport'].ravel();shuffled=outputs['shuffled_transport'].ravel()
            assert np.sum((minimum-z.ravel())**2)<=np.sum((shuffled-z.ravel())**2)+1e-6
    n=200000;z=rng.standard_normal(n);target=rng.integers(0,2,n);y=conditional_transport(z,target)
    result=dict(passed=True,roundtrips=roundtrips,central_quantile_and_inverse_checks=True,
        full_sign_match=True,within_orthant_permutation_preserves_magnitudes=True,rearrangement_cost_checks=True,
        illustrative_gaussian_sample=dict(n=n,mean=float(y.mean()),second_moment=float(np.mean(y.astype(float)**2)),mse=float(np.mean((y-z)**2))),
        limitation='Monte Carlo moments are descriptive; distribution preservation follows only under the stated ideal independent-cipher model.',
        source_sha256=hashlib.sha256(Path(__file__).with_name('transport_binding.py').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
