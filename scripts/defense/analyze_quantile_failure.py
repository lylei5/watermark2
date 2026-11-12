"""Server-only diagnostic decomposition; no detector fitting or attack changes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.fft import dctn
from scipy.special import ndtri


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    sys.path.insert(0,str(r))
    from anchor_binding import identity,order,partition
    plan=json.loads((r/'protocol.json').read_text());d=json.loads((r/'genuine_verified.json').read_text())
    keys={i:identity(plan['identity_root'],i) for i in range(6)}
    perms={(i,n):order(keys[i][3],n) for i in range(6) for n in [8192,16384]}
    rows=[]
    for x in d['rows']:
        t=x['task'];arm=x['arm'];z=np.load(r/x['latent']);q=np.load(r/x['code'])['code']
        h=np.load(r/f'genuine_{t%3}/codes/{t:03}_aux.npz')['code'];_,cipher,msg,_=keys[t]
        if arm=='full':raw=(z.ravel()>0).astype(np.uint8)
        elif arm=='anchor_low':raw=(dctn(z,axes=(-2,-1),norm='ortho').ravel()[partition(arm)[1]]>0).astype(np.uint8)
        else:
            v=dctn(z.astype(float),axes=(-2,-1),norm='ortho').ravel();lo,hi=partition('anchor_low')
            raw=np.concatenate([(np.abs(v[lo])>=ndtri(.75)),v[hi]>0]).astype(np.uint8)
        n=len(raw);perm=perms[t,n];hm=np.tile(h,n//1024)[perm];qm=np.tile(q,n//1024)[perm]
        ideal=cipher[:n]^hm;stream=cipher[:n]^np.tile(msg,n//256)
        def ba(observation,mask):
            bits=observation^mask^stream
            return float(np.mean((bits.reshape(-1,256).sum(0)>len(bits)/512)==msg))
        actual=ba(raw,qm);assert actual==x['ba']
        row=dict(task=t,arm=arm,condition=x['condition'],actual=actual,enrollment_oracle=ba(raw,hm),noiseless_channel=ba(ideal,qm),channel_ber=float(np.mean(raw!=ideal)),semantic_ber=float(np.mean(h!=q)))
        if arm.startswith('quantile_'):
            for label,sl in [('magnitude',slice(0,8192)),('sign',slice(8192,None))]:
                row[label+'_ber']=float(np.mean(raw[sl]!=ideal[sl]))
                for name,mask in [('actual',qm),('oracle',hm)]:
                    bits=(raw^mask^stream)[sl]
                    row[label+'_'+name]=float(np.mean((bits.reshape(32,256).sum(0)>16)==msg))
        rows.append(row)
    summary=[]
    for arm in plan['arms']:
        v=[x for x in rows if x['arm']==arm]
        fields=['actual','enrollment_oracle','noiseless_channel']
        if arm.startswith('quantile_'):fields+=['magnitude_actual','magnitude_oracle','sign_actual','sign_oracle']
        summary.append(dict(arm=arm,n=len(v),accepted={k:sum(x[k]>181/256 for x in v) for k in fields},mean={k:float(np.mean([x[k] for x in v])) for k in v[0] if k.endswith('_ber')}))
    result=dict(scope='Post-hoc diagnostic; oracle uses enrollment information unavailable to actual verifier; noiseless counterfactual removes inversion error, not a new detector',rows=rows,summary=summary,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((r/'genuine_verified.json').read_bytes()).hexdigest())
    (r/'failure_decomposition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
