"""Post-completion diagnostic of physical, intentional and binding errors.

Enrollment-code scores are oracle diagnostics, never deployed decisions.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    sys.path.insert(0,str(run));from anchor_binding import identity,order
    plan=json.loads((run/'protocol.json').read_text());g=json.loads((run/'genuine_verified.json').read_text())
    assert g['audit_passed'] and len(g['rows'])==240
    keys={t:identity(plan['identity_root'],t) for t in range(6)}
    source={x['task']:x for x in plan['sources']};refs={(r['task'],r['arm']):r for r in g['references']}
    rows=[]
    for r in g['rows']:
        t=r['task'];secret,c,m,bkey=keys[t];perm=order(bkey,16384);ref=refs[t,r['arm']]
        assert sha(run/r['latent'])==r['latent_sha256'] and sha(run/ref['initial_latent'])==ref['initial_latent_sha256']
        z=np.load(run/r['latent']).ravel();initial=np.load(run/ref['initial_latent']).ravel()
        q=np.load(run/r['code'])['code'];h=np.load(run/f'genuine_{t%3}/codes/{t:03}_aux.npz')['code']
        aux=np.random.default_rng(source[t]['aux_seed']).standard_normal((1,4,64,64)).astype(np.float32).ravel()
        target=c^np.tile(h,16)[perm];initial_sign=(initial>0).astype(np.uint8);observed=(z>0).astype(np.uint8)
        shaping=initial_sign^target;physical=initial_sign^observed;binding=np.tile(h^q,16)[perm]
        residual=observed^c^np.tile(q,16)[perm]
        assert np.array_equal(residual,shaping^physical^binding)
        def ba(mask):
            bits=observed^mask^c^np.tile(m,64)
            return float(np.mean((bits.reshape(64,256).sum(0)>32)==m))
        actual=ba(np.tile(q,16)[perm]);oracle=ba(np.tile(h,16)[perm]);assert actual==r['ba']
        classes={'reflected':(initial>0)!=(aux>0),'retained_correct':((initial>0)==(aux>0))&(shaping==0),
                 'retained_wrong':((initial>0)==(aux>0))&(shaping==1)}
        statistics={k:dict(n=int(v.sum()),physical_flips=int(physical[v].sum()),
                          mean_absolute_initial=float(np.mean(np.abs(initial[v]))) if v.any() else None)
                    for k,v in classes.items()}
        rows.append(dict(id=r['id'],task=t,arm=r['arm'],condition=r['condition'],actual_ba=actual,
            actual_accepted=actual>181/256,enrollment_oracle_ba=oracle,enrollment_oracle_accepted=oracle>181/256,
            shaping_error=float(shaping.mean()),physical_error=float(physical.mean()),
            binding_error=float(binding.mean()),residual_error=float(residual.mean()),coordinate_classes=statistics))
    summary={}
    for a in plan['arms']:
        rr=[r for r in rows if r['arm']==a]
        summary[a]=dict(actual_accepted=sum(r['actual_accepted'] for r in rr),oracle_accepted=sum(r['enrollment_oracle_accepted'] for r in rr),
            physical_error=float(np.mean([r['physical_error'] for r in rr])),
            binding_error=float(np.mean([r['binding_error'] for r in rr])),
            classes={k:dict(n=sum(r['coordinate_classes'][k]['n'] for r in rr),
                            physical_flips=sum(r['coordinate_classes'][k]['physical_flips'] for r in rr)) for k in classes})
    result=dict(completed=True,diagnostic_script_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        genuine_manifest_sha256=sha(run/'genuine_verified.json'),rows=rows,summary=summary,
        limitation='Enrollment oracle is unavailable at inference; correlated coordinate rates are descriptive, not independent trials. Coordinate classes describe signs only; full, transport and shuffled_transport may also change amplitudes.')
    (run/'channel_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
