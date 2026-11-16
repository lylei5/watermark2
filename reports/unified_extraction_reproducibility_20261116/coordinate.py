from pathlib import Path
import json,subprocess,os,ctypes,select,hashlib,numpy as np
run=Path(__file__).parent
plan=json.loads((run/'protocol.json').read_text())
root=Path('/data0/liyanlei/watermark_research')
previous=json.loads((Path(plan['recovery_from'])/'pipeline_state.json').read_text())
state=dict(status='running',pid=os.getpid(),protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),existing_workers=previous['children'][:2],children=[])
def save():
    (run/'pipeline_state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
    target=Path(plan['plans'][2]['run'])
    with (target/'worker.log').open('x') as log:
        child=subprocess.Popen([str(root/'envs/semantic_forgery_py310/bin/python'),str(target/'deterministic_entry.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='4'))
    state['children']=[child.pid]
    save()
    libc=ctypes.CDLL(None,use_errno=True)
    for pid in state['existing_workers']:
        fd=libc.syscall(434,pid,0)
        if fd>=0:
            select.select([fd],[],[])
            os.close(fd)
        elif ctypes.get_errno()!=3:
            raise OSError(ctypes.get_errno(),'pidfd_open')
    state['exit_codes']=[child.wait()]
    assert state['exit_codes']==[0],state
    for item in plan['plans']:
        folder=Path(item['run'])
        assert hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()==item['protocol_sha256']
        receipt=json.loads((folder/'sembind/complete.json').read_text())
        assert receipt['status']=='complete' and len(receipt['measurements'])==8
        for row in receipt['measurements']:
            assert row['unet_calls']==50
            assert hashlib.sha256((folder/'sembind'/row['artifact']).read_bytes()).hexdigest()==row['artifact_sha256']
    results=[]
    for index in range(8):
        original=np.load(Path(plan['plans'][0]['run'])/'sembind'/f'{index:03}_sembind.npz')
        for item in plan['plans'][1:]:
            actual=np.load(Path(item['run'])/'sembind'/f'{index:03}_sembind.npz')
            delta=np.abs(original['recovered'].astype(float)-actual['recovered'].astype(float))
            results.append(dict(source=index,candidate=item['label'],maximum=float(delta.max()),mean=float(delta.mean()),sign_changes=int((np.sign(original['recovered'])!=np.sign(actual['recovered'])).sum()),allclose=bool(np.allclose(original['recovered'],actual['recovered'],atol=1e-5,rtol=1e-5)),semantic_equal=bool(np.array_equal(original['semantic_code'],actual['semantic_code']))))
    passed=all(r['allclose'] and r['semantic_equal'] for r in results)
    (run/'parity.json').write_text(json.dumps(dict(passed=passed,comparisons=results,protocol_sha256=state['protocol_sha256']),indent=2)+'\n')
    assert passed
    state['status']='complete'
except Exception as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    save()
