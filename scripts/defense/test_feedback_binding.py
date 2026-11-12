"""Server-only invariants for feedback carrier updates; no convergence claim."""
import hashlib,json,secrets,sys,tempfile
from pathlib import Path
import numpy as np
from feedback_binding import embed,decode


def main():
    output=Path(sys.argv[1]);rng=np.random.default_rng(202611126001);checks=0
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='feedback_unit_') as directory:
        root=Path(directory)
        for task in range(16):
            s=dict(key=secrets.token_hex(32),nonce=secrets.token_hex(16),message=secrets.token_hex(32),binding_key=secrets.token_hex(32),amplitude_seed=81000+task,independent_anchor_seed=82000+task)
            (root/f'{task:03}.json').write_text(json.dumps(s));h=rng.integers(0,2,1024,dtype=np.uint8);previous=None
            for arm in ['full','feedback','consensus']:
                z=embed(None,h,root,task,arm);assert decode(z,h,root,task,arm)==1.
                if previous is not None:assert np.array_equal(z,previous)
                previous=z;checks+=1
            changed=h.copy();changed[rng.choice(1024,size=137,replace=False)]^=1
            new=embed(None,changed,root,task,'feedback')
            assert np.array_equal(np.abs(new),np.abs(previous))
            assert int(np.sum((new>0)!=(previous>0)))==137*16
            assert decode(new,changed,root,task,'feedback')==1.
    result=dict(passed=True,identities=16,arm_roundtrips=checks,changed_semantic_bits=137,changed_carrier_signs=137*16,
        scope='Carrier algebra only; three-round semantic dynamics, image quality, and robustness require the queued full model experiment',
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['feedback_binding.py','anchor_binding.py','test_feedback_binding.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
