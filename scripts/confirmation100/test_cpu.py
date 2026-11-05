"""CPU contract tests for the frozen confirmation100 baseline. No CUDA import."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU only')

from common100 import c, r, ROOT, BASE, by_id, candidates, validate_job

DESIGN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'protocols/confirmation100_design_v1.json'


def expect_fail(fn, label):
    try:
        fn()
    except Exception:
        return
    raise AssertionError('Expected failure: ' + label)


def main():
    d = json.loads(DESIGN.read_text(encoding='utf-8'))
    assert d['schema'] == 1 and d['stage'] == 'confirmation100_baseline'
    tasks = d['tasks']
    n = len(tasks)
    assert 40 <= n <= 100 and [t['id'] for t in tasks] == list(range(1, n + 1))
    assert len({t['cover'] for t in tasks}) == n
    assert len({t['reference_prompt'] for t in tasks}) == n
    assert len({t['reference_seed'] for t in tasks}) == n
    assert len({t['attack_seed'] for t in tasks}) == n
    assert {x['index'] for x in d['devices']} == {2, 3, 4, 5, 7}
    for task in tasks:
        path = ROOT / task['cover']
        assert path.is_file(), path
        assert c.sha(path) == task['cover_sha256'], task['id']
    print('PASS design: %d tasks, covers hashed, 5 devices, GPU6 absent' % n)

    byid = by_id(d)
    tmp = Path(tempfile.mkdtemp())
    try:
        folder = tmp / '1'
        folder.mkdir()
        shutil.copyfile(ROOT / tasks[0]['cover'], folder / 'cover.png')
        shutil.copyfile(ROOT / tasks[0]['cover'], folder / 'reference.png')
        job = dict(schema=1, task_id=1, seed=byid[1]['attack_seed'],
                   reference=c.png_record(folder / 'reference.png'),
                   cover=c.png_record(folder / 'cover.png'),
                   proxy=d['proxy'], native=d['native'])
        c.write(folder / 'job.json', job)
        validate_job(folder / 'job.json', {}, d)
        bad = dict(job, seed=job['seed'] + 1)
        (folder / 'job.json').unlink()
        c.write(folder / 'job.json', bad)
        expect_fail(lambda: validate_job(folder / 'job.json', {}, d), 'tampered seed')
        print('PASS job validation: accepts frozen job, rejects tampered seed')

        import numpy as np
        from common100 import module
        private = module('confirmation100_test_private', BASE / 'reference_reliability/private_state.py')
        decoder = {'np': np}
        r.definitions(BASE / 'cross_generator/verify_message_scores.py', {'decode'}, decoder)
        run = ROOT / 'runs/31_native_calibration_20261024T190323Z'
        record = c.read(run / 'prepare/1/record.json')
        raw = private.load(ROOT / 'private_eval/31_native_calibration_20261024T190323Z', 1,
                           record['encryption_fingerprints'])
        inverse = np.load(run / 'evaluate/1_151/target_inverse.npy', allow_pickle=False)
        stored = c.read(run / 'evaluate/1_151/metrics.json')['bit_accuracy']
        check = decoder['decode'](inverse, raw)
        assert check == stored, (check, stored)
        print('PASS decoder regression: independent CPU decode equals stage31 stored BA', check)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
