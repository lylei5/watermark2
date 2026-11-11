"""Server-only, allowlisted storage audit and verified file cleanup."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path('/data0/liyanlei')
AUDIT = ROOT / 'watermark_research/runs/storage_audit_20261111'


def dump(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fingerprint(path):
    p = Path(path)
    assert p.is_absolute() and not p.is_symlink()
    p.resolve().relative_to(ROOT.resolve())
    s = p.stat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
    return dict(path=str(p), bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev, links=s.st_nlink)


def sha(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def file_sha(p):
    with Path(p).open('rb') as f:
        return sha(f)


def inventory(name):
    p = subprocess.run(['du', '-x', '-B1', '--max-depth=2', str(ROOT)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    (AUDIT / (name + '.tsv')).write_text(p.stdout)
    (AUDIT / (name + '_errors.txt')).write_text(p.stderr)
    (AUDIT / (name + '_filesystem.txt')).write_text(subprocess.check_output(
        ['df', '-B1', str(ROOT)], universal_newlines=True))


def verify():
    inventory('before')
    plan, checks = [], []
    for item in json.loads((AUDIT / 'incomplete_candidates.json').read_text()):
        p = Path(item['path'])
        s = p.stat()
        assert p.name.endswith('.incomplete') and s.st_size == item['bytes']
        assert abs(s.st_mtime - item['mtime']) < .001 and time.time() - s.st_mtime > 7 * 86400
        plan.append(dict(**fingerprint(p), reason='stale incomplete download'))
    archives = [
        ('natural_image_512bit_watermark_poc/datasets/div2k/DIV2K_train_HR.zip',
         'natural_image_512bit_watermark_poc/datasets/div2k', ''),
        ('FontGuard_repro/fontguard_ckpt.zip', 'FontGuard_repro/exp_data', 'fontguard_ckpt/'),
        ('watermark_research/runs/geometry_calibration_payload_20261110.zip',
         'watermark_research/runs/geometry_calibration_20261110', ''),
    ]
    for archive, extracted, prefix in archives:
        p, base = ROOT / archive, ROOT / extracted
        before = fingerprint(p)
        matches, failures = [], []
        with zipfile.ZipFile(p) as z:
            for entry in z.infolist():
                if entry.is_dir():
                    continue
                if not entry.filename.startswith(prefix):
                    failures.append(entry.filename)
                    continue
                target = base / entry.filename[len(prefix):]
                target.resolve().relative_to(base.resolve())
                if not target.is_file() or target.stat().st_size != entry.file_size:
                    failures.append(entry.filename)
                    continue
                prior = target.stat()
                with z.open(entry) as f:
                    expected = sha(f)
                actual = file_sha(target)
                after = target.stat()
                if expected != actual or (prior.st_size, prior.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    failures.append(entry.filename)
                    continue
                matches.append(dict(member=entry.filename, retained=str(target), sha256=actual,
                                    bytes=after.st_size, mtime_ns=after.st_mtime_ns))
        assert before == fingerprint(p)
        checks.append(dict(archive=str(p), matches=matches, failures=failures))
        if matches and not failures:
            plan.append(dict(**before, reason='archive fully matches retained extracted files',
                             verification_index=len(checks) - 1))
        print(json.dumps(dict(archive=archive, matched=len(matches), failures=failures)), flush=True)
    duplicates = [(ROOT / 'watermark_research/runs/benign_dev_assets.zip',
                   ROOT / 'watermark_research/runs/content_auth_20261109/benign_dev_assets.zip')]
    partial = ROOT / 'FontGuard_repro/partial/exp_data'
    for p in partial.rglob('*'):
        target = ROOT / 'FontGuard_repro/exp_data' / p.relative_to(partial)
        if p.is_file() and target.is_file():
            duplicates.append((p, target))
    for p, target in duplicates:
        before = fingerprint(p)
        if p.stat().st_size != target.stat().st_size:
            continue
        digest = file_sha(p)
        if digest == file_sha(target):
            assert before == fingerprint(p)
            plan.append(dict(**before, reason='byte-identical retained copy',
                             retained=str(target), sha256=digest,
                             retained_fingerprint=fingerprint(target)))
    dump('archive_checks.json', checks)
    dump('verified_plan.json', plan)
    print(json.dumps(dict(files=len(plan), bytes=sum(x['bytes'] for x in plan),
                          allocated_bytes=sum(x['allocated_bytes'] for x in plan))), flush=True)


def execute():
    plan = json.loads((AUDIT / 'verified_plan.json').read_text())
    checks = json.loads((AUDIT / 'archive_checks.json').read_text())
    opened = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    opened.add(str(fd.resolve(strict=True)))
                except (OSError, RuntimeError):
                    pass
        except OSError:
            pass
    for item in plan:
        assert fingerprint(item['path']) == {k: item[k] for k in fingerprint(item['path'])}
        assert str(Path(item['path']).resolve()) not in opened
        assert item['links'] == 1
        if 'retained' in item:
            assert fingerprint(item['retained']) == item['retained_fingerprint']
        if 'verification_index' in item:
            for retained in checks[item['verification_index']]['matches']:
                s = Path(retained['retained']).stat()
                assert (s.st_size, s.st_mtime_ns) == (retained['bytes'], retained['mtime_ns'])
    results = []
    for item in plan:
        try:
            Path(item['path']).unlink()
            results.append(dict(**item, status='deleted', deleted_at=time.time()))
        except OSError as e:
            results.append(dict(**item, status='failed', error=str(e)))
        dump('deletion_results.json', results)
    inventory('after')
    print(json.dumps(dict(deleted=sum(x['status'] == 'deleted' for x in results),
                          freed_allocated_bytes=sum(x['allocated_bytes'] for x in results if x['status'] == 'deleted'),
                          failures=[x for x in results if x['status'] != 'deleted'])))


if __name__ == '__main__':
    assert ROOT.is_dir() and os.name == 'posix'
    {'verify': verify, 'execute': execute}[sys.argv[1]]()
