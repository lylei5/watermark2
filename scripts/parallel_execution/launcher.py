"""Frozen-plan subprocess launcher for separately authorized future experiments."""
import argparse
import csv
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import scheduler as s

REMOTE_ROOT = Path('/data0/liyanlei/watermark_research')
LIMIT = 10 * 1024 ** 3
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected a project-relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes project')
    return path


def check_host(project):
    if sys.platform != 'linux' or project.resolve() != REMOTE_ROOT or getpass.getuser() != 'liyanlei':
        raise ValueError('Model execution requires liyanlei on the designated Linux project')
    addresses = subprocess.check_output(['hostname', '-I'], text=True, timeout=10).split()
    if '172.18.185.249' not in addresses:
        raise ValueError('Designated server address is absent')


def sample(index):
    rows = list(csv.reader(subprocess.check_output([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines(), skipinitialspace=True))
    apps = list(csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError('Ambiguous GPU identity')
    row = rows[0]
    return dict(index=int(row[0]), uuid=row[1], model=row[2], memory_used_mib=int(row[3]),
                utilization_percent=int(row[4]), compute_apps=[a for a in apps if a and a[0] == row[1]])


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def verify_pins(project, plan):
    for name, expected in plan['files'].items():
        if not re.fullmatch('[0-9a-f]{64}', expected) or sha(local(project, name)) != expected:
            raise ValueError('Pinned file changed: ' + name)
    for model in plan['models']:
        snapshot = local(project, model['snapshot'])
        if not re.fullmatch('[0-9a-f]{40}', model['revision']) or snapshot.name != model['revision']:
            raise ValueError('Model snapshot must name its exact local revision')
        if model['manifest'] not in plan['files']:
            raise ValueError('Model file manifest is not pinned')
        manifest = read(local(project, model['manifest']))
        actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
        if not actual or actual != set(manifest):
            raise ValueError('Model manifest must cover the complete local snapshot')
        for name, digest in manifest.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid model manifest path')
            # Snapshot files may link to cached sibling blobs inside this project.
            path = local(project, (Path(model['snapshot']) / name).as_posix())
            if sha(path) != digest:
                raise ValueError('Local model file changed: ' + name)
    for name, version in plan['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Package version changed: ' + name)


def load_plan(project, path, expected):
    project, path = Path(project).resolve(), Path(path).resolve()
    if not path.is_relative_to(project) or sha(path) != expected:
        raise ValueError('Plan path/hash does not match the frozen plan')
    plan = read(path)
    if plan['schema'] != 1 or not re.fullmatch(r'parallel_[A-Za-z0-9_-]+', plan['run_name']):
        raise ValueError('A separate future parallel run name is required')
    if plan['python_version'] != sys.version.split()[0] or not plan['versions']:
        raise ValueError('Freeze Python and the model package versions')
    if len(plan['methods']) < 2 or len(set(plan['methods'])) != len(plan['methods']):
        raise ValueError('Freeze the distinct compared methods')
    if not plan['models']:
        raise ValueError('Pin at least one existing local model snapshot')
    for name in ('launcher.py', 'worker.py', 'scheduler.py'):
        relative = (HERE / name).relative_to(project).as_posix()
        if relative not in plan['files']:
            raise ValueError('Launcher source must be frozen: ' + relative)
    devices = tuple(s.Device(**d) for d in plan['devices'])
    jobs, phases, groups = {}, {}, {}
    for phase in ('attack', 'evaluation'):
        current = []
        for group in plan[phase + '_groups']:
            if phase == 'attack' and sorted(j['method'] for j in group['jobs']) != sorted(plan['methods']):
                raise ValueError('Every attack task must contain each compared method exactly once')
            members = []
            for spec in group['jobs']:
                name = spec['name']
                if name in jobs:
                    raise ValueError('Duplicate job name')
                if spec['entry'] not in plan['files'] or spec['parameters'] not in plan['files']:
                    raise ValueError('Entry and parameters must be pinned')
                if not isinstance(spec['argv'], list) or not all(isinstance(a, str) for a in spec['argv']):
                    raise ValueError('Arguments must be an explicit string array')
                if '{output}' not in spec['argv'] or '{parameters}' not in spec['argv']:
                    raise ValueError('Worker arguments must include output and frozen parameters')
                if not 0 < spec['watchdog_seconds'] <= 86400 or not 0 < spec['output_limit_bytes'] <= LIMIT:
                    raise ValueError('Invalid frozen watchdog/output allowance')
                if not spec['completion_marker'] or '\n' in spec['completion_marker']:
                    raise ValueError('Expected one exact completion marker line')
                local(project, spec['entry']); local(project, spec['parameters'])
                binding = hashlib.sha256(json.dumps(dict(plan_sha256=expected, job=spec), sort_keys=True).encode()).hexdigest()
                members.append(s.Job(name, binding)); jobs[name] = spec; phases[name] = phase
            current.append(s.Group(group['name'], tuple(members)))
        groups[phase] = tuple(current)
    s.validate(devices, groups['attack'], groups['evaluation'])
    if sum(j['output_limit_bytes'] for j in jobs.values()) > LIMIT:
        raise ValueError('Declared total outputs exceed 10 GiB')
    verify_pins(project, plan)
    return plan, devices, groups, jobs, phases


def enumeration(plan, devices, groups, jobs, expected):
    return dict(plan_sha256=expected, run_name=plan['run_name'], dry_run=True,
                devices=[d.__dict__ for d in devices], maximum_concurrent_processes=len(devices),
                excluded_gpu_indices=[6], method_device_affinity=True,
                declared_max_output_bytes=sum(j['output_limit_bytes'] for j in jobs.values()),
                phases={phase: [dict(group=g.name, jobs=[j.name for j in g.jobs]) for g in gs]
                        for phase, gs in groups.items()}, model_processes_started=0)


class ProcessRunner:
    """Run one owned subprocess, record evidence, and enforce its frozen limits."""
    def __init__(self, project, plan_path, expected, jobs, phases):
        self.project, self.plan_path = Path(project), Path(plan_path)
        self.expected, self.jobs, self.phases = expected, jobs, phases

    def command(self, job, device, output):
        return [sys.executable, str(HERE / 'worker.py'), '--project', str(self.project),
                '--plan', str(self.plan_path), '--plan-sha256', self.expected, '--job', job.name,
                '--gpu-index', str(device.index), '--output', str(output)]

    def __call__(self, job, device, output, environment):
        spec = self.jobs[job.name]
        command = self.command(job, device, output)
        # No shell, no inherited PYTHONPATH injection, no access tokens in records.
        env = dict(os.environ, **environment)
        env.pop('PYTHONPATH', None)
        start = time.monotonic(); reason = None
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=self.project, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=os.name != 'nt')
            try:
                identity = dict(pid=child.pid, command=command, started_at_utc=s.now(),
                                gpu_index=device.index, gpu_uuid=device.uuid, environment=environment)
                if sys.platform == 'linux':
                    try:
                        identity['proc_start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                    except FileNotFoundError:
                        identity['proc_start_ticks'] = None  # A fast failed child may already have exited.
                s.write_once(output / 'process.json', identity)
                while child.poll() is None:
                    if time.monotonic() - start > spec['watchdog_seconds']:
                        reason = 'watchdog_expired'
                    if tree_bytes(output) > spec['output_limit_bytes']:
                        reason = 'output_allowance_exceeded'
                    if reason:
                        self.terminate_owned(child)
                        break
                    time.sleep(.1)
            except BaseException:
                self.terminate_owned(child)
                raise
            exit_code = child.wait()
        if tree_bytes(output) > spec['output_limit_bytes']:
            reason = 'output_allowance_exceeded'
        summary_path = output / 'summary.json'
        completed = (exit_code == 0 and reason is None and summary_path.is_file()
                     and read(summary_path).get('completed') is True
                     and spec['completion_marker'] in (output / 'stdout.log').read_text(encoding='utf-8', errors='replace').splitlines())
        record = dict(completed=completed, exit_code=exit_code, stop_reason=reason,
                      process_wall_seconds=time.monotonic() - start,
                      artifacts={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        if completed and self.phases[job.name] == 'attack':
            record['candidate_sha256'] = sha(output / 'candidate.png')
        return record

    @staticmethod
    def terminate_owned(child):
        # Only our child/process group can be terminated, never a busy GPU owner.
        if child.poll() is None:
            if os.name == 'nt':
                child.kill()
            else:
                os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def execute(project, path, expected, resume=False):
    project = Path(project).resolve()
    check_host(project)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervisor must have empty CUDA visibility')
    plan, devices, groups, jobs, phases = load_plan(project, path, expected)
    maximum = sum(j['output_limit_bytes'] for j in jobs.values())
    metadata_allowance = len(jobs) * 65536 + 1024 ** 2
    def preflight():
        # Checked while holding the shared experiment lock, before any new claim.
        verify_pins(project, plan)
        if sha(path) != expected:
            raise ValueError('Frozen plan changed before lock acquisition')
        used = tree_bytes(project / 'runs') + tree_bytes(project / 'reports')
        if used + maximum + metadata_allowance > LIMIT or shutil.disk_usage(project).free < maximum + metadata_allowance:
            raise ValueError('Insufficient storage under the retained 10-GiB cap')
    runner = ProcessRunner(project, path, expected, jobs, phases)
    scheduler = s.Scheduler(project, plan['run_name'], devices, groups['attack'], groups['evaluation'], runner,
        lambda device: s.verify_idle(device, sample,
            lambda record: scheduler.event(event='supervisor_idle_gate', gpu_uuid=device.uuid, detail=record)),
        resource_wait_seconds=plan['resource_wait_seconds'], resume=resume, preflight=preflight)
    return scheduler.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        plan, devices, groups, jobs, _ = load_plan(args.project, args.plan, args.plan_sha256)
        result = enumeration(plan, devices, groups, jobs, args.plan_sha256)
    else:
        result = execute(args.project, args.plan, args.plan_sha256, args.resume)
    print(json.dumps(result, indent=2))
    return 0 if args.dry_run or result['completed'] else 1


if __name__ == '__main__':
    sys.exit(main())
