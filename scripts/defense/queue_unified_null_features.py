"""Queue full SemBind and PGID GS/TR null extraction after SEAL; no scores tuned."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path('/data0/liyanlei/watermark_research')
PRIOR = ROOT/'runs/unified_seal_clean_20261116'
PROBE = ROOT/'runs/unified_runtime_probe_20261115'
DATA = ROOT/'datasets/unified_coco_nulls_20261115/manifest.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(run, sembind_reference=None):
    assert not run.exists()
    original = json.loads((PROBE/'protocol.json').read_text())
    assert sha(DATA) == '13c09dd26845ccad25d9e553888b60fb2e3283de6fd89aaee00e9508b96917a6'
    prior = json.loads((PRIOR/'pipeline_state.json').read_text())
    predecessor = dict(run=str(PRIOR), protocol_sha256=sha(PRIOR/'protocol.json'))
    if prior['status'] == 'complete':
        assert json.loads((PRIOR/'clean_results.json').read_text())['status'] == 'complete'
        predecessor.update(pid=None, cmdline_sha256=None, completed_results_sha256=sha(PRIOR/'clean_results.json'))
    else:
        command = Path(f"/proc/{prior['pid']}/cmdline").read_bytes()
        assert str(PRIOR).encode() in command and b'--coordinate' in command
        predecessor.update(pid=prior['pid'], cmdline_sha256=hashlib.sha256(command).hexdigest())
    reference_runs = dict(sembind=str(PROBE), pgid=str(PROBE))
    reproducibility = None
    if sembind_reference is not None:
        sembind_reference = sembind_reference.resolve()
        assert sembind_reference.parent == ROOT/'runs'
        assert json.loads((sembind_reference/'pipeline_state.json').read_text())['status'] == 'complete'
        parity = json.loads((sembind_reference/'parity.json').read_text())
        assert parity['passed'] and len(parity['comparisons']) == 16
        references = json.loads((sembind_reference/'protocol.json').read_text())
        reference_runs['sembind'] = next(p['run'] for p in references['plans'] if p['label'] == 'legacy')
        reproducibility = dict(run=str(sembind_reference), parity_sha256=sha(sembind_reference/'parity.json'))
    run.mkdir()
    shutil.copy2(__file__, run/Path(__file__).name)
    shutil.copy2(Path(__file__).parent/'unified_runtime_probe.py', run/'unified_runtime_probe.py')
    for name, digest in original['helper_sha256'].items():
        assert sha(PROBE/name) == digest
        shutil.copy2(PROBE/name, run/name)
    shutil.copy2(ROOT/'runs/unified_seal_nulls_20261115/prepare_coco_nulls.py', run/'prepare_coco_nulls.py')
    models = json.loads((ROOT/'runs/unified_seal_nulls_20261115/model_manifest.json').read_text())
    model_files = {str(Path(models['diffusion_model']['root'])/name): digest for name, digest in models['diffusion_model']['files'].items()}
    semantic = Path(original['methods']['sembind']['paths']['semantic'])
    for path in sorted((semantic/'dinov2-giant').rglob('*')) + [semantic/'sembind_masker.pth']:
        if path.is_file():
            model_files[str(path)] = sha(path)
    ref = ROOT/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
    assert ref.read_text().strip() == '0094d483a120f3f33dafbd187ea4aa60d10de75c'
    model_files[str(ref)] = sha(ref)
    save(run/'model_files.json', model_files)
    save(run/'template.json', original)
    inputs = json.loads(DATA.read_text())
    save(run/'inputs.json', [dict(path=x['path'], sha256=x['file_sha256'], pixel_sha256=x['pixel_sha256'],
        index=x['index'], source_id=x['source_id'], split=x['split']) for x in inputs])
    # A100-image chunk is a restart boundary, never a statistical subsample.
    jobs = [dict(name=f'{method}_{start:05}', method=method, start=start, end=start+100)
            for start in range(0, 40000, 100) for method in ['sembind', 'pgid']]
    plan = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        scope='Full40000 common natural inputs: native SemBind raw inverse+semantic code; original PGID default/F/R for GS/TR; no PRC-specific PGID claim',
        calibration_rows=20000, validation_rows=20000, inputs_sha256=sha(run/'inputs.json'),
        data_manifest_sha256=sha(DATA), jobs=jobs, total_jobs=800, expected_feature_artifacts=160000,
        predecessor=predecessor,
        source_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},
        python=str(ROOT/'envs/semantic_forgery_py310/bin/python'),
        expected_unet_calls=dict(pgid_default=50, pgid_f=298, pgid_r=200, sembind=50),
        budgets=original['budgets'],
        deterministic_backend_methods=['sembind'] if reproducibility else [],
        reproducibility=reproducibility,
        development_preflight=dict(sources=8, reference_runs=reference_runs,
                                   reference_protocol_sha256={method:sha(Path(path)/'protocol.json') for method,path in reference_runs.items()},
                                   raw_latent_atol=1e-5, raw_latent_rtol=1e-5, semantic_code_exact=True,
                                   stop_before_natural_extraction_on_failure=True),
        frozen_seed_policy='Preserve runtime-worker per-local-index seed; same100-index schedule for both methods; no outcome-driven changes',
        calibration_policy='Extract only; do not tune thresholds or view score distributions here. Scores require frozen carrier/identity interfaces later.',
        cost_note='Complete original50step/100VAE-update budget retained. Strict-backend development timing is132-137seconds/image; historical non-deterministic495GPUhour estimate is superseded. See reproducibility cost_estimate.json for scope and exclusions.',
        gpu_cap=3, excluded_gpu=6, attacks=0)
    save(run/'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run/'protocol.json'), jobs=800, expected_feature_artifacts=160000)))


def coordinate(run):
    plan = json.loads((run/'protocol.json').read_text())
    assert Path(__file__).resolve() == run/Path(__file__).name
    for name, digest in plan['source_sha256'].items():
        assert sha(run/name) == digest, name
    lock = (run/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run/'pipeline_state.json').exists()
    state = dict(status='waiting_for_seal_clean_exit', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'), completed_jobs=0, active=[])
    mutex = threading.RLock()
    stop = threading.Event()
    save(run/'pipeline_state.json', state)
    try:
        previous = plan['predecessor']
        libc = ctypes.CDLL(None, use_errno=True)
        if previous['pid'] is not None:
            descriptor = libc.syscall(434, previous['pid'], 0)
            if descriptor < 0:
                if ctypes.get_errno() != 3:
                    raise OSError(ctypes.get_errno(), 'pidfd_open')
            else:
                try:
                    path = Path(f"/proc/{previous['pid']}/cmdline")
                    if path.exists():
                        command = path.read_bytes()
                        assert not command or hashlib.sha256(command).hexdigest() == previous['cmdline_sha256']
                    select.select([descriptor], [], [])
                finally:
                    os.close(descriptor)
        prior_run = Path(previous['run'])
        assert sha(prior_run/'protocol.json') == previous['protocol_sha256']
        assert json.loads((prior_run/'pipeline_state.json').read_text())['status'] == 'complete'
        assert json.loads((prior_run/'clean_results.json').read_text())['status'] == 'complete'
        if previous.get('completed_results_sha256'):
            assert sha(prior_run/'clean_results.json') == previous['completed_results_sha256']
        for method, path in plan['development_preflight']['reference_runs'].items():
            assert sha(Path(path)/'protocol.json') == plan['development_preflight']['reference_protocol_sha256'][method]
        if plan['reproducibility']:
            check = plan['reproducibility']
            assert sha(Path(check['run'])/'parity.json') == check['parity_sha256']
        assert sha(DATA) == plan['data_manifest_sha256']
        pins = json.loads((run/'model_files.json').read_text())
        for path, digest in pins.items():
            assert sha(path) == digest, path
        time.sleep(15)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        idle = {}
        for line in inventory.splitlines():
            index, uuid, memory, util = [x.strip() for x in line.split(',')]
            if int(index) != 6 and int(memory) < 100 and int(util) == 0:
                idle[int(index)] = uuid
        available = [i for i in [7,5,4,3,2,1,0] if i in idle][:3]
        assert len(available) == 3
        save(run/'gpu_assignment.json', dict(gpu_indices=available, gpu_uuids=[idle[g] for g in available]))
        inputs = json.loads((run/'inputs.json').read_text())
        template = json.loads((run/'template.json').read_text())

        def execute(job, gpu, development=False):
            target = run/'chunks'/job['name']
            target.mkdir(parents=True, exist_ok=False)
            for name in plan['source_sha256']:
                if name.endswith('.py') and name != Path(__file__).name:
                    shutil.copy2(run/name, target/name)
            cfg = dict(template, scope='Development parity gate' if development else plan['scope'],
                       inputs=template['inputs'] if development else inputs[job['start']:job['end']],
                       worker_sha256=sha(target/'unified_runtime_probe.py'),
                       helper_sha256={p.name:sha(p) for p in target.glob('*.py') if p.name != 'unified_runtime_probe.py'},
                       gpu_indices=dict(pgid=gpu, sembind=gpu, seal=gpu),
                       expected_unet_calls=plan['expected_unet_calls'], parent_protocol_sha256=sha(run/'protocol.json'))
            cfg['deterministic_backend'] = job['method'] in plan['deterministic_backend_methods']
            if not development:
                cfg['preprocessing'] = 'common_natural_512'
            save(target/'protocol.json', cfg)
            # Device identity cannot silently change between queue assignment and launch.
            actual_uuid = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            assert actual_uuid == idle[gpu]
            time.sleep(5)
            with (target/'worker.log').open('x') as log:
                child = subprocess.Popen([plan['python'],str(target/'unified_runtime_probe.py'),'--run',str(target),'--method',job['method']],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='4'))
                with mutex:
                    state['active'].append(dict(job=job['name'],gpu=gpu,pid=child.pid))
                    save(run/'pipeline_state.json',state)
                code = child.wait()
            with mutex:
                state['active'] = [x for x in state['active'] if x['pid'] != child.pid]
                save(run/'pipeline_state.json',state)
            assert code == 0, (job['name'],code)
            result = json.loads((target/job['method']/'complete.json').read_text())
            assert result['status'] == 'complete'
            if development:
                import numpy as np
                maximum = 0.
                reference_run = Path(plan['development_preflight']['reference_runs'][job['method']])
                reference_rows = json.loads((reference_run/job['method']/'complete.json').read_text())['measurements']
                reference_artifacts = {row['artifact']:row['artifact_sha256'] for row in reference_rows}
                for row in result['measurements']:
                    assert sha(reference_run/job['method']/row['artifact']) == reference_artifacts[row['artifact']]
                    actual = np.load(target/job['method']/row['artifact'])
                    original = np.load(reference_run/job['method']/row['artifact'])
                    assert set(actual.files) == set(original.files)
                    for key in original.files:
                        if key == 'semantic_code':
                            assert np.array_equal(actual[key],original[key])
                        else:
                            maximum = max(maximum,float(np.max(np.abs(actual[key]-original[key]))))
                            assert np.allclose(actual[key],original[key],atol=1e-5,rtol=1e-5), (row['artifact'],key)
                save(target/'parity.json',dict(passed=True,artifacts=len(result['measurements']),max_absolute_difference=maximum,
                    tolerance=plan['development_preflight'],reference_protocol_sha256=sha(reference_run/'protocol.json')))
            return target

        state['status'] = 'development_parity_gate'
        save(run/'pipeline_state.json',state)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute,dict(name=f'development_{method}',method=method),gpu,True)
                       for method,gpu in zip(['sembind','pgid'],available)]
            for future in futures:
                future.result()
        tasks = queue.Queue()
        for job in plan['jobs']:
            tasks.put(job)
        state['status'] = 'extracting_full_natural_cohort'
        save(run/'pipeline_state.json',state)

        def consume(gpu):
            while not stop.is_set():
                try:
                    job = tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    execute(job,gpu)
                    with mutex:
                        state['completed_jobs'] += 1
                        save(run/'pipeline_state.json',state)
                except Exception:
                    stop.set()
                    raise
                finally:
                    tasks.task_done()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(consume,gpu) for gpu in available]
            for future in futures:
                future.result()
        assert state['completed_jobs'] == 800
        counts = Counter()
        seen = set()
        with (run/'feature_records.jsonl').open('x') as stream:
            for job in plan['jobs']:
                target = run/'chunks'/job['name']
                result = json.loads((target/job['method']/'complete.json').read_text())
                for row in result['measurements']:
                    identity=(row['arm'],row['global_source'])
                    assert identity not in seen
                    seen.add(identity)
                    assert row['protocol_sha256'] == sha(target/'protocol.json')
                    assert sha(target/job['method']/row['artifact']) == row['artifact_sha256']
                    assert row['unet_calls'] == plan['expected_unet_calls'][row['arm']]
                    counts[(row['arm'],row['split'])] += 1
                    stream.write(json.dumps(dict(row,artifact=str(target/job['method']/row['artifact'])))+'\n')
        assert len(seen) == 160000
        assert all(counts[(arm,split)] == 20000 for arm in ['sembind','pgid_default','pgid_f','pgid_r'] for split in ['calibration','validation'])
        for path,digest in pins.items():
            assert sha(path) == digest,path
        save(run/'feature_extraction_complete.json',dict(status='complete',protocol_sha256=sha(run/'protocol.json'),
            artifacts=160000,counts={f'{arm}/{split}':n for (arm,split),n in counts.items()},
            records_sha256=sha(run/'feature_records.jsonl'),thresholds_calibrated=False,attacks=0))
        state['status']='complete'
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        save(run/'pipeline_state.json',state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--sembind-reference',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    assert run.parent == ROOT/'runs'
    prepare(run,args.sembind_reference) if args.prepare else coordinate(run)
