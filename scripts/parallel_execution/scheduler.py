"""Future-run scheduling kernel. No model imports or implicit launch.

The project adapter supplies a runner with a second idle check before CUDA import.
Existing frozen runs and controllers are deliberately not modified by this module.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Device:
    index: int
    uuid: str
    model: str


@dataclass(frozen=True)
class Job:
    name: str
    parameters_sha256: str


@dataclass(frozen=True)
class Group:
    name: str
    jobs: tuple


def validate(devices, attack_groups, evaluation_groups):
    if not 1 <= len(devices) <= 4:
        raise ValueError("Select between one and four devices")
    if len({d.index for d in devices}) != len(devices) or len({d.uuid for d in devices}) != len(devices):
        raise ValueError("Repeated GPU identity")
    for device in devices:
        if type(device.index) is not int or device.index not in (0, 1, 2, 3, 4, 5, 7):
            raise ValueError("Unauthorized GPU index (GPU6 is excluded)")
        if not device.uuid.startswith("GPU-") or not device.model:
            raise ValueError("Missing explicit GPU UUID/model")
    if len({d.model for d in devices}) != 1:
        raise ValueError("This cost-comparison kernel requires one GPU model")
    groups = tuple(attack_groups) + tuple(evaluation_groups)
    if any(not isinstance(g, Group) for g in groups) or any(not isinstance(d, Device) for d in devices):
        raise ValueError("Expected explicit immutable device/group records")
    if not attack_groups or not evaluation_groups or any(not g.jobs for g in groups):
        raise ValueError("Both phases require nonempty groups")
    if len({g.name for g in groups}) != len(groups):
        raise ValueError("Duplicate group")
    jobs = [job for group in groups for job in group.jobs]
    if any(not isinstance(job, Job) for job in jobs):
        raise ValueError("Expected immutable job records")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job")
    for name in [g.name for g in groups] + [j.name for j in jobs]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Names must be single safe path components")
    for job in jobs:
        if not re.fullmatch(r"[0-9a-f]{64}", job.parameters_sha256):
            raise ValueError("Each job must bind its frozen parameters")


@contextmanager
def project_lock(path):
    """Shares the existing project's experiment lock; never kills an owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def verify_idle(device, sample, record, sleep=time.sleep, max_samples=15):
    """Three consecutive strict samples; identity errors fail immediately.

    Returns False for a busy card. The caller may wait without claiming a job.
    The real child must repeat this check before CUDA-capable imports.
    """
    if device.index == 6 or max_samples < 3:
        raise ValueError("Invalid idle-gate request")
    stable = 0
    for attempt in range(max_samples):
        observation = sample(device.index)
        record(dict(event="sample", attempt=attempt, observation=observation))
        if (observation["index"], observation["uuid"], observation["model"]) != (device.index, device.uuid, device.model):
            raise ValueError("GPU identity mismatch")
        if observation["compute_apps"]:
            return False
        idle = observation["memory_used_mib"] <= 50 and observation["utilization_percent"] == 0
        stable = stable + 1 if idle else 0
        if stable == 3:
            record(dict(event="idle_verified", samples=attempt + 1))
            return True
        if attempt + 1 < max_samples:
            sleep(2)
    return False


class Scheduler:
    """Run independent task groups concurrently, methods within a group serially.

    runner(job, device, output, environment) must return a JSON-safe record with
    completed=True and exit_code=0 after child exit, plus candidate_sha256 for an
    attack. Its job-specific input validation, watchdog, stdout/stderr capture,
    second idle gate and exact command recording are mandatory adapter duties.
    launcher.ProcessRunner supplies these duties for frozen future CLI workers.
    """
    def __init__(self, project, run_name, devices, attack_groups, evaluation_groups,
                 runner, idle_gate, resource_wait_seconds=300, poll_seconds=2, resume=False, preflight=None):
        validate(devices, attack_groups, evaluation_groups)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
            raise ValueError("Unsafe run name")
        if resource_wait_seconds < 0 or poll_seconds <= 0:
            raise ValueError("Invalid resource waiting policy")
        self.project = Path(project)
        self.root = self.project / "runs" / run_name
        self.devices = tuple(devices)
        self.attacks = tuple(Group(g.name, tuple(g.jobs)) for g in attack_groups)
        self.evaluations = tuple(Group(g.name, tuple(g.jobs)) for g in evaluation_groups)
        self.runner, self.idle_gate = runner, idle_gate
        self.wait_seconds, self.poll_seconds = resource_wait_seconds, poll_seconds
        self.guard = threading.Lock()
        self.available = set(range(len(devices)))
        self.stop = threading.Event()
        self.results, self.failures = {}, []
        self.candidates = {}
        self.resume = resume
        self.affinity = {}
        self.preflight = preflight

    def verify_candidates(self):
        for name, expected in self.candidates.items():
            path = self.root / name / "candidate.png"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Sealed candidate changed: " + name)

    def event(self, **data):
        with self.guard:
            with (self.root / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(dict(at_utc=now(), **data), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def acquire(self, group):
        deadline = time.monotonic() + self.wait_seconds
        while not self.stop.is_set():
            for index, device in enumerate(self.devices):
                if group.name in self.affinity and self.affinity[group.name] != device.uuid:
                    continue
                with self.guard:
                    reserved = index in self.available
                    if reserved:
                        self.available.remove(index)
                if not reserved:
                    continue
                try:
                    if self.idle_gate(device):
                        self.event(event="group_device_acquired", group=group.name, gpu_index=device.index, gpu_uuid=device.uuid)
                        return index, device
                except BaseException:
                    with self.guard:
                        self.available.add(index)
                    raise
                with self.guard:
                    self.available.add(index)
            if time.monotonic() >= deadline:
                raise RuntimeError("Resource wait expired before job reservation")
            time.sleep(self.poll_seconds)
        return None

    def group(self, group, phase):
        lease = None
        try:
            if all(job.name in self.results for job in group.jobs):
                return
            lease = self.acquire(group)
            if lease is None:
                return
            _, device = lease
            for job in group.jobs:
                if job.name in self.results:
                    continue
                if self.stop.is_set():
                    return
                # The GPU remains associated with this task across its methods.
                if not self.idle_gate(device):
                    raise RuntimeError("Assigned device became busy before job reservation")
                if phase == "evaluation":
                    self.verify_candidates()
                output = self.root / job.name
                with self.guard:
                    if self.stop.is_set():
                        return
                    output.mkdir()  # Atomic exclusive task reservation; never overwrites/retries.
                    write_once(output / "claim.json", dict(job=job.name, group=group.name, phase=phase,
                        parameters_sha256=job.parameters_sha256, gpu_index=device.index,
                        gpu_uuid=device.uuid, claimed_at_utc=now()))
                environment = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=device.uuid,
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                    PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                self.event(event="job_started", job=job.name, group=group.name, phase=phase, gpu_uuid=device.uuid)
                started = time.monotonic()
                try:
                    outcome = json.loads(json.dumps(self.runner(job, device, output, environment), allow_nan=False))
                    successful = outcome.get("completed") is True and outcome.get("exit_code") == 0
                    if phase == "attack" and successful:
                        if not re.fullmatch(r"[0-9a-f]{64}", outcome.get("candidate_sha256", "")):
                            raise ValueError("Runner omitted candidate seal")
                        candidate = output / "candidate.png"
                        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != outcome["candidate_sha256"]:
                            raise ValueError("Candidate bytes do not match runner seal")
                except BaseException as exc:
                    outcome = dict(completed=False, exit_code=None, error_type=type(exc).__name__, reason=str(exc))
                    successful = False
                record = dict(job=job.name, group=group.name, phase=phase, gpu_index=device.index,
                    gpu_uuid=device.uuid, parameters_sha256=job.parameters_sha256,
                    elapsed_seconds=time.monotonic() - started, outcome=outcome)
                write_once(output / "execution.json", record)
                write_once(output / "execution_seal.json", dict(
                    sha256=hashlib.sha256((output / "execution.json").read_bytes()).hexdigest()))
                with self.guard:
                    self.results[job.name] = record
                self.event(event="job_finished", job=job.name, completed=successful)
                if not successful:
                    raise RuntimeError("Job failed; no automatic retry: " + job.name)
        except BaseException as exc:
            with self.guard:
                self.stop.set()
                self.failures.append(dict(group=group.name, phase=phase, error_type=type(exc).__name__, reason=str(exc)))
            self.event(event="group_stopped", group=group.name, reason=str(exc))
        finally:
            if lease is not None:
                with self.guard:
                    self.available.add(lease[0])

    def phase(self, groups, phase):
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [executor.submit(self.group, group, phase) for group in groups]
            for future in futures:
                future.result()  # In-flight jobs finish; no external process is killed.

    def recover(self, specification):
        """Reuse verified successes only; a claimed uncertain/failed job blocks recovery."""
        if json.loads((self.root / "schedule.json").read_text()) != specification:
            raise ValueError("Recovery schedule differs from the frozen schedule")
        if (self.root / "results_sealed.json").exists():
            raise ValueError("Completed run is already sealed")
        for phase, groups in (("attack", self.attacks), ("evaluation", self.evaluations)):
            for group in groups:
                for job in group.jobs:
                    output = self.root / job.name
                    if not output.exists():
                        continue
                    claim = json.loads((output / "claim.json").read_text())
                    record_path = output / "execution.json"
                    if not record_path.is_file():
                        raise ValueError("Ambiguous claimed job; never relaunch: " + job.name)
                    seal = json.loads((output / "execution_seal.json").read_text())
                    if hashlib.sha256(record_path.read_bytes()).hexdigest() != seal["sha256"]:
                        raise ValueError("Recovered execution record changed")
                    record = json.loads(record_path.read_text())
                    expected = dict(job=job.name, group=group.name, phase=phase,
                                    parameters_sha256=job.parameters_sha256)
                    if any(claim.get(k) != v or record.get(k) != v for k, v in expected.items()):
                        raise ValueError("Recovery record identity mismatch")
                    if not any(d.uuid == record.get("gpu_uuid") and d.index == record.get("gpu_index")
                               for d in self.devices):
                        raise ValueError("Recovery device is not in the frozen device set")
                    if any(claim.get(k) != record.get(k) for k in ("gpu_uuid", "gpu_index")):
                        raise ValueError("Claim and execution devices differ")
                    previous = self.affinity.setdefault(group.name, record["gpu_uuid"])
                    if previous != record["gpu_uuid"]:
                        raise ValueError("Recovered methods used different devices")
                    outcome = record["outcome"]
                    if outcome.get("completed") is not True or outcome.get("exit_code") != 0:
                        raise ValueError("Failed job; never retry: " + job.name)
                    # The concrete launcher seals every produced file after the child exits.
                    for relative, digest in outcome.get("artifacts", {}).items():
                        path = (output / relative).resolve()
                        if not path.is_relative_to(output.resolve()) or not path.is_file():
                            raise ValueError("Invalid recovered artifact")
                        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                            raise ValueError("Recovered artifact changed: " + relative)
                    if phase == "attack":
                        self.candidates[job.name] = outcome["candidate_sha256"]
                    self.results[job.name] = record
        self.verify_candidates()
        seal_path = self.root / "candidates_sealed.json"
        if seal_path.exists():
            seal = json.loads(seal_path.read_text())
            expected_count = sum(len(g.jobs) for g in self.attacks)
            if len(self.candidates) != expected_count or seal["candidates"] != self.candidates:
                raise ValueError("Recovered candidate barrier differs")
        elif any(r["phase"] == "evaluation" for r in self.results.values()):
            raise ValueError("Evaluation exists without the candidate barrier")
        self.event(event="explicit_recovery", inherited_jobs=sorted(self.results))

    def run(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Supervisor must have empty CUDA visibility")
        with project_lock(self.project / "runs" / ".imprint_delivery.lock"):
            if self.preflight is not None:
                self.preflight()
            specification = dict(devices=[d.__dict__ for d in self.devices],
                attacks=[dict(name=g.name, jobs=[j.__dict__ for j in g.jobs]) for g in self.attacks],
                evaluations=[dict(name=g.name, jobs=[j.__dict__ for j in g.jobs]) for g in self.evaluations],
                max_concurrent_groups=len(self.devices), group_device_affinity=True,
                excluded_gpu_indices=[6], automatic_job_retry=False)
            if self.resume:
                self.recover(specification)
            else:
                self.root.mkdir()
                write_once(self.root / "schedule.json", specification)
            self.phase(self.attacks, "attack")
            if not self.stop.is_set():
                self.candidates = {job.name: self.results[job.name]["outcome"]["candidate_sha256"]
                                   for group in self.attacks for job in group.jobs}
                try:
                    self.verify_candidates()
                    if not (self.root / "candidates_sealed.json").exists():
                        write_once(self.root / "candidates_sealed.json", dict(sealed_at_utc=now(), candidates=self.candidates))
                    self.event(event="candidate_barrier_passed", candidates=len(self.candidates))
                    self.phase(self.evaluations, "evaluation")
                    self.verify_candidates()
                except BaseException as exc:
                    self.stop.set()
                    self.failures.append(dict(phase="candidate_barrier", error_type=type(exc).__name__, reason=str(exc)))
            expected = [job.name for group in self.attacks + self.evaluations for job in group.jobs]
            unrecorded = set(expected) - set(self.results)
            summary = dict(completed=not self.stop.is_set(), failures=self.failures,
                completed_jobs=[name for name, r in sorted(self.results.items()) if r["outcome"].get("completed") is True and r["outcome"].get("exit_code") == 0],
                failed_jobs=[name for name, r in sorted(self.results.items()) if not (r["outcome"].get("completed") is True and r["outcome"].get("exit_code") == 0)],
                unstarted_jobs=sorted(name for name in unrecorded if not (self.root / name).exists()),
                ambiguous_claimed_jobs=sorted(name for name in unrecorded if (self.root / name).exists()),
                finished_at_utc=now(), active_jobs=0)
            status = self.root / "status.json"
            if status.exists():
                number = 1
                while (self.root / f"recovery_status_{number}.json").exists():
                    number += 1
                status = self.root / f"recovery_status_{number}.json"
            write_once(status, summary)
            if summary["completed"]:
                write_once(self.root / "results_sealed.json", dict(sealed_at_utc=now(),
                    executions={name: hashlib.sha256((self.root / name / "execution.json").read_bytes()).hexdigest()
                                for name in sorted(self.results)},
                    status_file=status.name, status_sha256=hashlib.sha256(status.read_bytes()).hexdigest()))
            return summary
