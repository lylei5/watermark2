"""Event-scheduled clean mechanism screen with artifact and decoder audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import numpy as np
from scipy.special import ndtr, ndtri
from anchor_binding import identity, order
from continuous_geometry_binding import angles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def bundle(path):
    with np.load(path) as data:
        result={k:data[k] for k in ['code','logits']}
    assert np.array_equal(result['code'].astype(bool),result['logits']>0)
    return result


def distance(a,b):
    x=a['logits'].astype(float);y=b['logits'].astype(float)
    return float(np.linalg.norm(x/np.linalg.norm(x)-y/np.linalg.norm(y)))


def audit_rotation(z,theta,permutation,inverse):
    # Separate complex-number implementation of the real2x2 worker rotation.
    pairs=np.asarray(z,dtype=float).ravel()[permutation].reshape(-1,2)
    value=(pairs[:,0]+1j*pairs[:,1])*np.exp((-1j if inverse else 1j)*theta)
    flat=np.empty(16384)
    flat[permutation]=np.column_stack([value.real,value.imag]).ravel()
    return flat


def score(z,q,root,t,arm,keys):
    _,cipher,message,key=keys[t]
    if arm=='full':
        coordinate=z.ravel()
        mask=np.tile(q['code'],16)[order(key,16384)]
    else:
        theta,perm=angles(q,root,t,arm)
        coordinate=audit_rotation(z,theta,perm,True)
        mask=np.zeros(16384,dtype=np.uint8)
    bits=(coordinate>0).astype(np.uint8)^mask^cipher^np.tile(message,64)
    return float(np.mean((bits.reshape(64,256).sum(0)>32)==message))


def audit(run,plan):
    rows=[];references=[]
    for shard in range(3):
        done=json.loads((run/f'genuine_{shard}/complete.json').read_text())
        assert done['status']=='complete' and done['protocol_sha256']==sha(run/'protocol.json')
        rows.extend(done['rows']);references.extend(done['references'])
    expected={(s['task'],arm,'clean') for s in plan['sources'] for arm in plan['arms']}
    assert len(rows)==24 and {(r['task'],r['arm'],r['condition']) for r in rows}==expected
    assert len(references)==24 and {(r['task'],r['arm']) for r in references}=={x[:2] for x in expected}
    root=Path(plan['identity_root']);keys={t:identity(root,t) for t in range(6)}
    wrong=[]
    for r in rows:
        for field in ['image','latent','code']:
            assert sha(run/r[field])==r[field+'_sha256']
        z=np.load(run/r['latent']);q=bundle(run/r['code'])
        ba=score(z,q,root,r['task'],r['arm'],keys)
        assert ba==r['ba'] and (ba>181/256)==r['accepted']
        for t in keys:
            if t==r['task']:continue
            other=score(z,q,root,t,r['arm'],keys)
            wrong.append(dict(id=r['id'],arm=r['arm'],identity=t,ba=other,accepted=other>181/256))
    import torch
    import lpips
    from PIL import Image
    sys.path.insert(0,plan['upstream'])
    from utils.image_utils import lpips_PIL
    torch.set_num_threads(4)
    quality=lpips.LPIPS(net='alex').eval().cpu()
    for r in references:
        t=r['task'];arm=r['arm'];directory=run/f'genuine_{t%3}'
        h=bundle(directory/'codes'/f'{t:03}_aux.npz')
        q=bundle(directory/'codes'/f'{t:03}_{arm}.npz')
        assert float(np.mean(h['code']!=q['code']))==r['semantic_mismatch']
        assert sha(run/r['image'])==r['image_sha256']
        assert sha(run/r['initial_latent'])==r['initial_latent_sha256']
        initial=np.load(run/r['initial_latent']).ravel()
        item=next(s for s in plan['sources'] if s['task']==t)
        z=np.random.default_rng(item['aux_seed']).standard_normal((1,4,64,64)).astype(np.float32).ravel()
        secret,cipher,_,key=keys[t]
        if arm=='full':
            target=cipher^np.tile(h['code'],16)[order(key,16384)]
            expected=np.abs(np.random.default_rng(secret['amplitude_seed']).standard_normal(16384))*(2*target.astype(float)-1)
        else:
            theta,perm=angles(h,root,t,arm)
            source=audit_rotation(z,theta,perm,True)
            if arm=='continuous_independent':
                encoded=np.abs(np.random.default_rng(secret['amplitude_seed']).standard_normal(16384))*(2*cipher.astype(float)-1)
            else:
                encoded=ndtri((ndtr(source)+cipher)/2).astype(np.float32)
            expected=audit_rotation(encoded,theta,perm,False)
        np.testing.assert_allclose(initial,expected.astype(np.float32),rtol=3e-5,atol=5e-7)
        assert score(initial,h,root,t,arm,keys)==1
        r.update(continuous_registration_distance=distance(h,q),initial_mse=float(np.mean((initial.astype(float)-z)**2)))
        with torch.no_grad():
            measured=float(lpips_PIL(Image.open(run/r['image']).convert('RGB'),Image.open(run/r['auxiliary']).convert('RGB'),quality,'cpu'))
        assert abs(measured-r['lpips'])<1e-6
    summary=[]
    for arm in plan['arms']:
        rr=[r for r in rows if r['arm']==arm];ff=[r for r in references if r['arm']==arm]
        summary.append(dict(arm=arm,n=6,accepted=sum(r['accepted'] for r in rr),mean_ba=float(np.mean([r['ba'] for r in rr])),
            mean_distance=float(np.mean([r['continuous_registration_distance'] for r in ff])),
            mean_hamming=float(np.mean([r['semantic_mismatch'] for r in ff])),
            mean_lpips=float(np.mean([r['lpips'] for r in ff])),mean_initial_mse=float(np.mean([r['initial_mse'] for r in ff])),
            wrong_identity_accepts=sum(r['accepted'] for r in wrong if r['arm']==arm)))
    assert len(wrong)==120
    result=dict(audit_passed=True,rows=rows,references=references,wrong_identity=wrong,summary=summary,
        protocol_sha256=sha(run/'protocol.json'),
        audit_scope='Input/output hashes, independent complex rotation and direct-CDF embedding reconstruction, hard vote and threshold recomputation, CPU LPIPS; keyed phase helper shared with worker.',
        scope=plan['scope'])
    save(run/'genuine_verified.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent-pid',type=int,required=True)
    a=p.parse_args();run=a.run.resolve();plan=json.loads((run/'protocol.json').read_text())
    def state(status,**extra):
        save(run/'pipeline_state.json',dict(status=status,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    try:
        for name,digest in plan['source_sha256'].items():assert sha(run/name)==digest
        for name,digest in plan['identity_sha256'].items():assert sha(Path(plan['identity_root'])/name)==digest
        assert sha(run/'unit_checks.json')==plan['unit_checks_sha256']
        assert json.loads((run/'unit_checks.json').read_text())['passed']
        parent=Path(plan['parent_run'])/'pipeline_state.json'
        state('waiting',stage='parent_exit_event',parent_pid=a.parent_pid)
        try:
            fd=os.pidfd_open(a.parent_pid)
        except ProcessLookupError:
            assert parent.exists()
        else:
            select.select([fd],[],[]);os.close(fd)
        previous=json.loads(parent.read_text())
        assert previous['pid']==a.parent_pid and previous['status']=='complete',previous
        time.sleep(10)
        children=[]
        for i in range(3):
            with (run/f'genuine_{i}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable,str(run/'run_anchor_binding_worker.py'),'--run',str(run),'--stage','genuine','--shard',str(i)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='4')))
        state('running',stage='clean_mechanism_screen',worker_pids=[c.pid for c in children])
        codes=[c.wait() for c in children];assert not any(codes),codes
        result=audit(run,plan);s={r['arm']:r for r in result['summary']};candidate=s['continuous_transport']
        gates=dict(clean=candidate['accepted']==6 and candidate['accepted']>=max(r['accepted'] for r in s.values()),
            binding=candidate['mean_distance']<=min(s['full']['mean_distance'],s['continuous_independent']['mean_distance']),
            continuous_vs_quantized=candidate['mean_ba']>=s['quantized_geometry']['mean_ba'],
            fidelity=candidate['mean_lpips']<=s['continuous_independent']['mean_lpips']-.05,
            wrong_identity=candidate['wrong_identity_accepts']<=min(r['wrong_identity_accepts'] for r in s.values()))
        save(run/'pilot_decision.json',dict(gates=gates,advance=all(gates.values()),attack_jobs=0,scope='Clean feasibility only'))
        lines=['# Continuous semantic geometry: clean screen','','Six fresh sources;24 full50-step inversions with100-step decoder optimization.','',
            '| Arm | Clean | Mean BA | Registration distance | Preview LPIPS | Wrong identity |',
            '|---|---:|---:|---:|---:|---:|']
        for r in result['summary']:
            lines.append(f"| {r['arm']} | {r['accepted']}/6 | {r['mean_ba']:.6f} | {r['mean_distance']:.6f} | {r['mean_lpips']:.6f} | {r['wrong_identity_accepts']}/30 |")
        lines += ['',f'Frozen gates: {gates}.',f'Advance: {all(gates.values())}. No attack jobs were included.',plan['scope']]
        (run/'analysis.md').write_text('\n'.join(lines)+'\n')
        state('complete',stage='clean_screen_audited',genuine_records=24,attack_jobs=0)
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':
    main()
