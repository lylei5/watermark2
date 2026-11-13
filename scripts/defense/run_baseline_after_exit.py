"""Launch frozen baseline shards after recorded predecessors exit, without polling."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert str(run.parent) == '/data0/liyanlei/watermark_research/runs'
    plan = json.loads((run / 'protocol.json').read_text())
    assert sha(__file__) == plan['coordinator_sha256']
    assert sha(run / plan['worker']) == plan['worker_sha256']
    assert plan['gpu_indices'] == [7, 5, 4]
    assert not (run / 'pipeline_state.json').exists()
    state = dict(status='queued', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'),
                 started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), events=[], workers=[])
    lock = threading.RLock()

    def save():
        tmp = run / 'pipeline_state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(run / 'pipeline_state.json')

    def execute(shard):
        previous = plan['predecessors'][shard]
        pid = previous['pid']
        assert os.uname().machine == 'x86_64'
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.syscall(434, pid, 0)
        if descriptor < 0:
            if ctypes.get_errno() != 3:
                raise OSError(ctypes.get_errno(), 'pidfd_open')
            descriptor = None
        if descriptor is not None:
            try:
                command = Path(f'/proc/{pid}/cmdline')
                # An empty cmdline means an already-exited zombie; the pidfd is ready.
                if command.exists():
                    actual = command.read_bytes()
                    assert not actual or hashlib.sha256(actual).hexdigest() == previous['cmdline_sha256']
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
        # Let the driver's utilization sampling window expire after context release.
        # This is one fixed settling interval, not an experiment-progress poll.
        time.sleep(15)
        with (run / f'native_{shard}.log').open('x') as log:
            child = subprocess.Popen([plan['python'], str(run / plan['worker']), '--run', str(run),
                                      '--shard', str(shard)], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, OPENBLAS_NUM_THREADS='4'))
            with lock:
                state.update(status='running')
                state['workers'].append(dict(shard=shard, pid=child.pid, gpu=plan['gpu_indices'][shard]))
                save()
                print(json.dumps(dict(event='worker_started', shard=shard, pid=child.pid)), flush=True)
            code = child.wait()
        return dict(shard=shard, pid=child.pid, exit_code=code,
                    completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())

    save()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute, shard) for shard in range(3)]
            for future in as_completed(futures):
                event = future.result()
                with lock:
                    state['events'].append(event)
                    save()
                print(json.dumps(event), flush=True)
        assert all(e['exit_code'] == 0 for e in state['events']), state['events']
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
