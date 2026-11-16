from pathlib import Path
import json,hashlib,ctypes,select,os,subprocess
run=Path(__file__).parent
root=Path('/data0/liyanlei/watermark_research')
plan=json.loads((run/'protocol.json').read_text())
state=dict(status='waiting_for_reproducibility_exit',pid=os.getpid(),target=plan['target'])
def save():
    (run/'pipeline_state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.syscall(434,plan['reference_pid'],0)
    if fd>=0:
        path=Path(f"/proc/{plan['reference_pid']}/cmdline")
        if path.exists():
            command=path.read_bytes()
            assert not command or hashlib.sha256(command).hexdigest()==plan['reference_cmdline_sha256']
        select.select([fd],[],[])
        os.close(fd)
    elif ctypes.get_errno()!=3:
        raise OSError(ctypes.get_errno(),'pidfd_open')
    reference=Path(plan['reference'])
    assert json.loads((reference/'pipeline_state.json').read_text())['status']=='complete'
    assert json.loads((reference/'parity.json').read_text())['passed']
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    target=Path(plan['target'])
    python=str(root/'envs/semantic_forgery_py310/bin/python')
    subprocess.run([python,str(root/'scripts/defense/queue_unified_null_features.py'),'--run',str(target),'--prepare','--sembind-reference',str(reference)],check=True)
    with (target/'coordinator.log').open('x') as log:
        child=subprocess.Popen([python,str(target/'queue_unified_null_features.py'),'--run',str(target)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,run=str(target),protocol_sha256=hashlib.sha256((target/'protocol.json').read_bytes()).hexdigest())
    (target/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    state.update(status='launched',coordinator=receipt)
except Exception as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    save()
