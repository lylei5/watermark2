"""Seal completed attack checkpoints for a truncated confirmation100 run (30 pairs)."""
import sys
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
import common100 as m  # noqa: E402


def main(run_dir, n):
    run = Path(run_dir)
    n = int(n)
    rows = []
    for sid in range(1, n + 1):
        out = run / ('attack_' + str(sid))
        job = run / 'attack_inputs' / str(sid) / 'job.json'
        frozen = m.c.read(out / 'candidates_frozen.json')
        if frozen['job_sha256'] != m.c.sha(job):
            raise ValueError('job hash mismatch ' + str(sid))
        if not m.c.read(out / 'summary.json').get('completed'):
            raise ValueError('incomplete task ' + str(sid))
        for point in m.POINTS:
            record = m.c.png_record(out / ('update_' + str(point) + '.png'))
            if frozen['images'][str(point)] != record:
                raise ValueError('checkpoint changed ' + str(sid))
            rows.append(dict(task_id=sid, actual_update=point,
                             path='attack_' + str(sid) + '/update_' + str(point) + '.png',
                             image=record))
    m.c.write(run / 'candidates_sealed_30.json', dict(rows=rows, n=n, sealed_at_utc=m.sched.now()))
    print('sealed', len(rows), 'checkpoints for', n, 'tasks')


if __name__ == '__main__':
    main(*sys.argv[1:])
