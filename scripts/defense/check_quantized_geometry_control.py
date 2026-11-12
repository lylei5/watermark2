"""Verify quantized-rotation control equals a binary mask in a fixed frame."""
import argparse
import hashlib
import json
import secrets
import tempfile
from pathlib import Path
import numpy as np
from continuous_geometry_binding import geometry,feature,rotate,embed,decode
from transport_binding import conditional_transport
from anchor_binding import identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rng=np.random.default_rng(202611130400);errors=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for t in range(16):
            secret={k:secrets.token_bytes(n).hex() for k,n in [('key',32),('nonce',16),('message',32),('binding_key',32)]}
            secret['amplitude_seed']=t
            (root/f'{t:03}.json').write_text(json.dumps(secret))
            z=rng.standard_normal((1,4,64,64)).astype(np.float32)
            logits=rng.standard_normal(1024);bundle=dict(logits=logits,code=(logits>0).astype(np.uint8))
            _,cipher,message,_=identity(root,t)
            omega,beta,perm=geometry(str(root),t)
            fixed_theta=beta+np.pi/2
            mask=np.empty(16384,dtype=np.uint8)
            mask[perm]=np.repeat(np.tile(omega@feature(bundle)<=0,8),2)
            source=rotate(z,fixed_theta,perm,inverse=True).ravel()
            encoded=conditional_transport(source,cipher^mask)
            fixed_frame=rotate(encoded,fixed_theta,perm).astype(np.float32)
            quantized=embed(z,bundle,root,t,'quantized_geometry')
            error=float(np.max(np.abs(fixed_frame-quantized)));errors.append(error)
            np.testing.assert_allclose(fixed_frame,quantized,rtol=2e-6,atol=2e-7)
            # Compare decoders on an arbitrary Gaussian observation, not just an embedded point.
            observed=rng.standard_normal((1,4,64,64)).astype(np.float32)
            raw=rotate(observed,fixed_theta,perm,inverse=True).ravel()>0
            bits=raw.astype(np.uint8)^mask^cipher^np.tile(message,64)
            alternate=float(np.mean((bits.reshape(64,256).sum(0)>32)==message))
            assert alternate==decode(observed,bundle,root,t,'quantized_geometry')
    result=dict(passed=True,identities=16,max_absolute_embedding_difference=max(errors),
        arbitrary_observation_decoder_matches=16,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='The quantized control is exactly a binary semantic mask under a content-independent secret pair rotation, with identical Gaussian conditional transport; numeric checks support implementation equivalence, not new defense efficacy.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
