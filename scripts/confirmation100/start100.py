import os
import sys
import subprocess
from pathlib import Path
from common100 import c, r, ROOT, HERE, launch


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor required')
    launch.check_host(ROOT)
    runtime = ROOT / ('protocols/confirmation100_controller_' + (sys.argv[1] if len(sys.argv) > 1 else 'v1') + '.json')
    cfg = r.verify_runtime(runtime)
    out = ROOT / 'reports' / Path(cfg['run_dir']).name
    if out.exists() or (ROOT / cfg['run_dir']).exists():
        raise ValueError('Existing launch intent; no retry')
    out.mkdir()
    with (out / 'stdout.log').open('xb') as stdout, (out / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen([sys.executable, str(HERE / 'control100.py'), str(runtime)],
                                 cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                                 stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
    c.write(out / 'launch.json', dict(controller_pid=child.pid, runtime_sha256=c.sha(runtime), run_dir=cfg['run_dir']))
    print('LAUNCHED', cfg['run_dir'], child.pid)


if __name__ == '__main__':
    main()
