"""Server-only invariants and independent probability-space numerical audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from semantic_evidence import scores,grouped_evidence
from semantic_evidence_reference import grouped_reference


def main():
    rng=np.random.default_rng(202611125001);model=dict(gain=.8,noise_variance=.6,temperature=.7,semantic_error=.1)
    checks=0;maximum=0.
    for trial in range(24):
        perm=rng.permutation(16384);cipher=rng.integers(0,2,16384,dtype=np.uint8);message=rng.integers(0,2,256,dtype=np.uint8);h=rng.integers(0,2,1024,dtype=np.uint8)
        signs=2*(cipher^np.tile(h,16)[perm]).astype(float)-1
        z=np.abs(rng.normal(size=16384))*signs+rng.normal(scale=[0,.5,2][trial%3],size=16384)
        logits=(2*h.astype(float)-1)*rng.uniform(.01,4,1024)+rng.normal(scale=.5,size=1024);q=(logits>0).astype(np.uint8)
        actual=scores(z,logits,q,cipher,message,perm,model);reference=grouped_reference(z,logits,q,cipher,perm,model)
        for k in reference:
            error=abs(actual[k]-reference[k]);assert error<1e-6,error;maximum=max(maximum,error);checks+=1
        mask=np.tile(q,16)[perm];stream=cipher^np.tile(message,64)
        recovered=((z>0).astype(np.uint8)^mask^stream).reshape(64,256).sum(0)>32
        assert actual['ba']==float(np.mean(recovered==message))
        assert abs(grouped_evidence(np.full(16384,.5),np.zeros(1024),cipher,perm))<1e-12
        perfect=grouped_evidence(.999*signs,.2*(2*h.astype(float)-1),cipher,perm)
        assert abs(perfect-np.log(1.2))<1e-8
        opposed=grouped_evidence(.999*signs,-.2*(2*h.astype(float)-1),cipher,perm)
        assert abs(opposed-np.log(.8))<1e-8
        assert actual['group_query']<=np.log(2) and actual['group_constant']<=np.log(2)
    out=dict(passed=True,independent_score_comparisons=checks,maximum_difference=maximum,trials=24,
             invariants=['Hard BA equals direct bit recovery','Zero semantic certainty supplies zero binding evidence','Sixteen consistent carrier observations remain bounded by one semantic-bit contribution','Opposite semantic evidence has negative log ratio','Per-bit evidence never exceeds log2'],
             scope='Synthetic decoder algebra and numerical equivalence; no claim of image robustness or security',
             source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['semantic_evidence.py','semantic_evidence_reference.py','test_semantic_evidence.py']})
    Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
