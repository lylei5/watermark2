"""Wait for worker exits, independently audit saved outputs, and summarize mechanisms."""
import argparse
from collections import defaultdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def audit(run):
    # No GPU or project decoder imports: independently reconstruct the message.
    import numpy as np
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    plan = json.loads((run/'protocol.json').read_text())
    parent = Path(plan['parent_run'])
    paths = {k: Path(v) for k, v in json.loads((parent/'paths.json').read_text()).items()}
    rows = []
    for shard in range(3):
        d = json.loads((run/f'shard_{shard}/complete.json').read_text())
        assert d['status'] == 'complete' and d['protocol_sha256'] == sha(run/'protocol.json')
        assert len(d['rows']) == 48
        rows.extend(d['rows'])
    assert len(rows) == 144
    assert {(r['id'],r['setting']) for r in rows} == {(j['id'],s) for j in plan['jobs'] for s in ['blind_encoder','blind_decoder']}
    identities = {}
    for task in range(4):
        secret = json.loads((paths['root']/'private'/f'{task:03}.json').read_text())
        key, nonce, msg, bkey = [bytes.fromhex(secret[k]) for k in ['key','nonce','message','binding_key']]
        cipher = np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64),dtype=np.uint8))
        message = np.unpackbits(np.frombuffer(msg,dtype=np.uint8))
        tags = [hmac.new(bkey,b'sembind/permutation/'+j.to_bytes(4,'big'),hashlib.sha256).digest() for j in range(16384)]
        perm = np.asarray(sorted(range(16384),key=tags.__getitem__))
        identities[task] = (cipher, message, perm)
    for r in rows:
        latent = run/r['latent']
        base = latent.parent.parent
        codepath = base/'codes'/(r['id']+'.npz')
        imagepath = base/'images'/(r['id']+'.png')
        assert sha(latent) == r['latent_sha256'] and sha(codepath) == r['code_sha256'] and sha(imagepath) == r['image_sha256']
        task, method = r['task'], r['method']
        cipher, message, perm = identities[task]
        stream = cipher ^ np.tile(message,64)
        signs = (np.load(latent).reshape(-1)>0).astype(np.uint8)
        codes = {'query':np.load(codepath)['sembind_code'],
                 'auxiliary':np.load(paths['output']/'codes'/f'{task:03}_aux.npz')['sembind_code'],
                 'clean':np.load(paths['output']/'codes'/f'{task:03}_{method}.npz')['sembind_code']}
        for label, code in codes.items():
            mask = np.tile(code,16)[perm] if method=='sembind' else np.zeros(16384,dtype=np.uint8)
            ba = float((((signs ^ mask ^ stream).reshape(64,256).sum(axis=0)>32)==message).mean())
            assert ba == r[label+'_ba'] and (ba>181/256) == r[label+'_accepted']
        source_mask = np.tile(codes['auxiliary'],16)[perm] if method=='sembind' else np.zeros(16384,dtype=np.uint8)
        assert float((signs != (cipher ^ source_mask)).mean()) == r['channel_sign_error']
        assert float((codes['auxiliary']==codes['query']).mean()) == r['auxiliary_query_agreement']
        assert float((codes['clean']==codes['query']).mean()) == r['clean_query_agreement']
    groups = defaultdict(list)
    for r in rows:
        groups[(r['method'],r['setting'],r['condition'])].append(r)
    summary = []
    for (method,setting,condition),group in sorted(groups.items()):
        assert len(group)==4
        summary.append(dict(method=method,setting=setting,condition=condition,n=4,
            query_accepted=sum(r['query_accepted'] for r in group),
            enrollment_diagnostic_accepted=sum(r['auxiliary_accepted'] for r in group),
            binding_associated_rejections=sum(r['auxiliary_accepted'] and not r['query_accepted'] for r in group),
            channel_unresolved_rejections=sum(not r['auxiliary_accepted'] and not r['query_accepted'] for r in group),
            query_only_acceptances=sum(not r['auxiliary_accepted'] and r['query_accepted'] for r in group),
            mean_query_ba=sum(r['query_ba'] for r in group)/4,
            mean_channel_sign_error=sum(r['channel_sign_error'] for r in group)/4))
    result = dict(completed=True,audit_passed=True,protocol_sha256=sha(run/'protocol.json'),
        independently_recomputed_scores=144,unique_sources=4,queries=72,summary=summary,rows=rows,
        interpretation='Enrollment code is private diagnostic information, not an admissible detector. Categories describe paired score changes, not causal proof. Repeated conditions are dependent; no population FPR or SOTA claim.')
    save(run/'verified_results.json',result)
    lines = ['# Fresh paired SemBind/GS published-condition results','',
        'All 144 latent scores and artifact hashes passed independent CPU audit. Four development identities, 72 queries; repeated conditions are not independent samples.','',
        '| Method | Inversion | Accepted | Enrollment-code diagnostic |','|---|---|---:|---:|']
    for method in ['sembind','gs']:
        for setting in ['blind_encoder','blind_decoder']:
            group=[r for r in rows if r['method']==method and r['setting']==setting]
            lines.append(f"| {method} | {setting} | {sum(r['query_accepted'] for r in group)}/36 | {sum(r['auxiliary_accepted'] for r in group)}/36 |")
    lines += ['',result['interpretation'],'',
        'Next fixed validation: unchanged published Imprint baseline with 151 updates, all 16 checkpoints, full 100-step decoder optimization and 50-step DPM inversion. No threshold selection or attack modification follows these observations.']
    (run/'analysis.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    run=args.run.resolve()
    plan=json.loads((run/'protocol.json').read_text())
    assert sha(__file__)==plan['finalizer_sha256']
    processes=[]
    for i in range(3):
        command=[sys.executable,str(run/'revalidate_sembind_published.py'),'--run',str(run),'--shard',str(i)]
        with (run/f'worker_{i}.log').open('w') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        processes.append(proc)
        save(run/f'launch_{i}.json',dict(pid=proc.pid,gpu=plan['gpu_indices'][i],command=command))
    save(run/'pipeline_state.json',dict(status='running',stage='published_conditions',pid=os.getpid(),workers=[p.pid for p in processes]))
    codes=[p.wait() for p in processes]
    if any(codes):
        save(run/'pipeline_state.json',dict(status='failed',stage='published_conditions',returncodes=codes))
        raise RuntimeError(f'Incomplete workers: {codes}')
    try:
        audit(run)
        save(run/'pipeline_state.json',dict(status='complete',stage='audited_published_conditions'))
        next_plan=plan.get('next_pipeline')
        if next_plan:
            assert sha(next_plan['protocol'])==next_plan['protocol_sha256']
            assert sha(next_plan['script'])==next_plan['script_sha256']
            save(run/'pipeline_state.json',dict(status='running',stage='next_published_baseline'))
            subprocess.run([next_plan['python'],next_plan['script'],'--run',next_plan['run']],check=True)
            save(run/'pipeline_state.json',dict(status='complete',stage='published_conditions_and_baseline'))
    except Exception as exc:
        save(run/'pipeline_state.json',dict(status='failed',stage='audit_or_next_pipeline',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
