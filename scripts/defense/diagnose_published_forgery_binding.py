"""Read-only content-code diagnostic of completed published Imprint outcomes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run
    g=json.loads((run/'genuine_verified.json').read_text());f=json.loads((run/'forgery_verified.json').read_text());plan=json.loads((run/'protocol.json').read_text())
    assert g['audit_passed'] and f['audit_passed'] and len(f['rows'])==216
    sys.path.insert(0,str(run));from anchor_binding import identity,order
    rows=[]
    for source in plan['sources']:
        t=source['task'];records={r['kind']:r for r in f['rows'] if r['task']==t and (r['kind']!='candidate' or r['update']==151)}
        codes={}
        for kind,r in records.items():
            assert sha(run/r['code'])==r['code_sha256'] and sha(run/r['image'])==r['image_sha256']
            codes[kind]=np.load(run/r['code'])['code']
        hpath=run/f'genuine_{t%3}/codes/{t:03}_aux.npz';h=np.load(hpath)['code']
        r=records['candidate'];assert sha(run/r['latent'])==r['latent_sha256']
        z=np.load(run/r['latent']).ravel();_,cipher,message,key=identity(plan['identity_root'],t);perm=order(key,16384)
        def ba(code):
            b=(z>0).astype(np.uint8)^np.tile(code,16)[perm]^cipher^np.tile(message,64)
            return float(np.mean((b.reshape(64,256).sum(0)>32)==message))
        actual=ba(codes['candidate']);assert actual==r['ba']
        rows.append(dict(task=t,caption=source['caption'],actual_ba=actual,accepted=actual>181/256,
            enrollment_oracle_ba=ba(h),aux_to_reference=float(np.mean(h!=codes['reference'])),
            aux_to_cover=float(np.mean(h!=codes['cover'])),aux_to_candidate=float(np.mean(h!=codes['candidate'])),
            reference_to_cover=float(np.mean(codes['reference']!=codes['cover'])),
            reference_to_candidate=float(np.mean(codes['reference']!=codes['candidate'])),
            cover_to_candidate=float(np.mean(codes['cover']!=codes['candidate'])),
            lpips_to_cover=r['lpips'],image=r['image'],image_sha256=r['image_sha256'],
            enrollment_code_sha256=sha(hpath)))
    result=dict(completed=True,scope='Read-only diagnostic of all twelve already-completed, unchanged151-update CVPR2025 Imprint endpoints. No attacks generated, altered or optimized.',
        rows=rows,genuine_manifest_sha256=sha(run/'genuine_verified.json'),forgery_manifest_sha256=sha(run/'forgery_verified.json'),
        script_sha256=sha(__file__),limitation='Consumed development outcomes; oracle uses unavailable enrollment code. Semantic closeness is descriptive association, not causal identification or independent defense validation.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['task','actual_ba','accepted','aux_to_cover','aux_to_candidate','reference_to_candidate','enrollment_oracle_ba']} for r in rows],indent=2))


if __name__=='__main__':main()
