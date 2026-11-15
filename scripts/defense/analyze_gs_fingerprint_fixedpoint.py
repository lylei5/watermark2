"""Independent CPU replay, prompt-cluster intervals, and fixed GS comparison figures."""
import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
ROOT = Path('/data0/liyanlei/watermark_research')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-radius), min(1., center+radius)]


def summary(values):
    import numpy as np
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                min=float(np.min(values)), max=float(np.max(values)), p95=float(np.quantile(values, .95)))


def main(run):
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    plan = read(run / 'protocol.json')
    complete = read(run / 'supervisor_complete.json')
    assert complete['complete'], 'Incomplete experiment must not receive a pass verdict'
    assert complete['cumulative_gpu_hours_upper_bound'] <= 3
    gpu = read(run / 'gpu_complete.json')
    rounds, finals, queries = [read(run / name) for name in ['rounds.json', 'finals.json', 'queries.json']]
    assert len(queries) == 2000 and len(finals) == 200
    assert gpu['unet_calls'] == (50 + len(rounds) + len(queries)) * 50
    tasks = [r['task'] for r in plan['sources']]
    conditions = ['clean', *plan['conditions']]
    configs = {r['name']: r for r in plan['ratios']}
    assert len({(r['task'], r['ratio'], r['condition']) for r in queries}) == 2000
    projection = np.load(run / 'public_projection.npy')
    private = np.load(run / 'private_material.npz')
    key = np.packbits(private['key'].flatten()).tobytes()
    pad = np.unpackbits(np.frombuffer(hmac.digest(key, b'gs-fingerprint-pad-v1', 'sha256'), dtype=np.uint8))
    seed = int.from_bytes(hmac.digest(key, b'gs-fingerprint-positions-v1', 'sha256')[:16], 'big')
    locations = np.random.Generator(np.random.PCG64(seed)).permutation(16384)
    tau = plan['similarity_threshold']
    audit = dict(gs_replays=0, payload_replays=0, score_replays=0, round_replays=0,
                 image_hash_checks=0, unselected_coordinate_checks=0, mismatches=0, max_score_error=0.)
    final_lookup = {(r['task'], r['ratio']): r for r in finals}

    def score(bits, value):
        y = np.einsum('i,ij->j', value.astype(np.float64), projection)
        return float(np.sum(np.where(bits, y, -y)) / np.sqrt(256*np.sum(y*y)))

    for row in rounds:
        folder = run / 'images' / str(row['task'])
        stem = folder / (row['ratio'] + '_r' + str(row['iteration']))
        data = np.load(stem.with_suffix('.npz'))
        assert sha(stem.with_suffix('.png')) == row['image_sha256']
        new_bits = data['feature'].astype(float) @ projection >= 0
        bits = data['written_bits']
        assert np.array_equal(new_bits, data['fingerprint'])
        assert np.sum(new_bits != bits) == row['flips']
        assert bool(np.array_equal(new_bits, bits)) == row['converged']
        drift = 1 - score(bits, data['feature'])
        assert abs(drift-row['drift']) < 1e-12
        base = np.load(folder / 'baseline.npz')['initial_latent']
        selected = locations[:configs[row['ratio']]['count']]
        outside = np.ones(16384, dtype=bool)
        outside[selected] = False
        assert np.array_equal(base.flatten()[outside], data['initial_latent'].flatten()[outside])
        audit['round_replays'] += 1
        audit['unselected_coordinate_checks'] += 1
        audit['image_hash_checks'] += 1
    for row in queries:
        folder = run / 'queries' / str(row['task']) / row['ratio']
        data = np.load(folder / (row['condition'] + '.npz'))
        assert sha(folder / (row['condition'] + '.png')) == row['image_sha256']
        latent = data['inverted_latent']
        assert np.isfinite(latent).all() and np.isfinite(data['feature']).all()
        unmasked = np.logical_xor(latent > 0, private['key'])
        votes = unmasked.reshape(4, 8, 8, 8, 8).sum(axis=(1, 3))
        ba = float(np.mean((votes > 32) == private['watermark'].reshape(4, 8, 8)))
        assert ba == row['gs_ba'] and (ba > 181/256) == row['gs_accept']
        assert row['unet_calls'] == 50
        audit['gs_replays'] += 1
        audit['image_hash_checks'] += 1
        if row['ratio'] == 'baseline':
            continue
        cfg = configs[row['ratio']]
        selected = latent.flatten()[locations[:cfg['count']]].astype(float)
        raw = []
        for value in selected:
            raw.append(int(value > 0))
            if cfg['width'] == 2:
                raw.append(int(abs(value) >= 0.6744897501960817))
        recovered = np.array([int(sum(raw[b::256])*2 > len(raw[b::256])) for b in range(256)], dtype=np.uint8) ^ pad
        assert np.array_equal(recovered, data['recovered_bits'])
        written = np.load(final_lookup[(row['task'], row['ratio'])]['latent'])['written_bits']
        assert np.array_equal(written, data['written_bits'])
        assert np.sum(recovered != written) == row['bit_errors']
        value = score(recovered, data['feature'])
        error = abs(value-row['similarity'])
        assert error < 1e-12
        assert (value >= tau) == row['fingerprint_accept']
        assert bool(ba > 181/256 and value >= tau) == row['joint']
        audit['max_score_error'] = max(audit['max_score_error'], error)
        audit['payload_replays'] += 1
        audit['score_replays'] += 1
    index = {(r['task'], r['ratio'], r['condition']): r for r in queries}
    resample = np.random.default_rng(2026111602).integers(0, 50, (5000, 50))

    def clustered(values):
        return np.quantile(np.asarray(values)[resample].mean(axis=1), [.025, .975]).tolist()

    result = dict(protocol_sha256=sha(run / 'protocol.json'), n_prompts=50, conditions=conditions,
                  gpu=complete, audit=audit, ratios={}, baseline={})
    for condition in conditions:
        n = sum(index[(t, 'baseline', condition)]['gs_accept'] for t in tasks)
        result['baseline'][condition] = dict(gs_accept=n, n=50, wilson95=wilson(n, 50))
    for name, config in configs.items():
        selected = [final_lookup[(t, name)] for t in tasks]
        converged = sum(r['converged'] for r in selected)
        strict = sum(r['converged'] and r['strict_drift_pass'] for r in selected)
        data = dict(coordinates=config['count'], coordinate_fraction=config['count']/16384,
                    converged=converged, converged_wilson95=wilson(converged, 50), strict_converged=strict,
                    convergence_by_round={str(k): sum(r['converged'] and r['rounds'] <= k for r in selected) for k in [1,2,3]},
                    final_flips=summary([r['flips'] for r in selected]),
                    final_drift=summary([r['drift'] for r in selected]),
                    baseline_sscd_distance=summary([r['baseline_sscd_distance'] for r in selected]),
                    drift_by_round={}, conditions={})
        two_cycles, first_flips, last_flips, flipped_margins, stable_margins = [], [], [], [], []
        for task in tasks:
            task_rounds = [r for r in rounds if r['task']==task and r['ratio']==name]
            first_flips.append(task_rounds[0]['flips'])
            last_flips.append(task_rounds[-1]['flips'])
            payloads = [np.load(run/'images'/str(task)/(name+'_r'+str(r['iteration'])+'.npz')) for r in task_rounds]
            previous = np.load(run/'images'/str(task)/'baseline.npz')['feature']
            for payload in payloads:
                margin = np.abs(previous.astype(float) @ projection)
                changed = payload['written_bits'] != payload['fingerprint']
                flipped_margins.extend(margin[changed].tolist())
                stable_margins.extend(margin[~changed].tolist())
                previous = payload['feature']
            if len(payloads)==3 and not selected[tasks.index(task)]['converged']:
                if np.array_equal(payloads[0]['fingerprint'],payloads[2]['fingerprint']):
                    two_cycles.append(task)
        data['mechanism_diagnostics'] = dict(exact_two_cycle_tasks=two_cycles,
            first_round_flips=summary(first_flips), final_round_flips=summary(last_flips),
            prompts_fewer_flips_final_than_first=int(np.sum(np.array(last_flips)<first_flips)),
            flipped_previous_projection_margins=summary(flipped_margins) if flipped_margins else None,
            stable_previous_projection_margins=summary(stable_margins),
            interpretation='Descriptive only; no selection, projection fitting, or code changes')
        for k in [1,2,3]:
            rows = [r for r in rounds if r['ratio'] == name and r['iteration'] == k]
            data['drift_by_round'][str(k)] = dict(n=len(rows), flips=summary([r['flips'] for r in rows]),
                    asymmetric_distance=summary([r['drift'] for r in rows]),
                    previous_sscd_distance=summary([r['previous_sscd_distance'] for r in rows])) if rows else dict(n=0)
        for condition in conditions:
            rows = [index[(t, name, condition)] for t in tasks]
            base = [index[(t, 'baseline', condition)] for t in tasks]
            joint = sum(r['joint'] for r in rows)
            gs = sum(r['gs_accept'] for r in rows)
            data['conditions'][condition] = dict(n=50, joint=joint, joint_wilson95=wilson(joint,50),
                gs=gs, gs_wilson95=wilson(gs,50), fingerprint=sum(r['fingerprint_accept'] for r in rows),
                ber=summary([r['bit_errors']/256 for r in rows]), zero_bit_errors=sum(r['bit_errors']==0 for r in rows),
                gs_delta=gs-sum(r['gs_accept'] for r in base),
                gs_losses=sum(b['gs_accept'] and not r['gs_accept'] for b,r in zip(base, rows)),
                gs_gains=sum(not b['gs_accept'] and r['gs_accept'] for b,r in zip(base, rows)))
        joint_by_prompt = [np.mean([index[(t,name,c)]['joint'] for c in conditions[1:]]) for t in tasks]
        gs_delta_by_prompt = [np.mean([index[(t,name,c)]['gs_accept']-index[(t,'baseline',c)]['gs_accept'] for c in conditions[1:]]) for t in tasks]
        data['nine_edits'] = dict(joint=sum(data['conditions'][c]['joint'] for c in conditions[1:]), n=450,
            joint_cluster95=clustered(joint_by_prompt), gs_delta_cluster95=clustered(gs_delta_by_prompt),
            mean_ber=float(np.mean([index[(t,name,c)]['bit_errors']/256 for t in tasks for c in conditions[1:]])),
            gs=sum(data['conditions'][c]['gs'] for c in conditions[1:]),
            convergence_required_joint=sum(index[(t,name,c)]['joint'] and final_lookup[(t,name)]['converged'] for t in tasks for c in conditions[1:]))
        data['passes'] = dict(convergence_and_drift=strict>=45,
                             joint_clean=data['conditions']['clean']['joint']>=45,
                             joint_nine_edits=data['nine_edits']['joint']>=405,
                             gs_no_drop=all(r['gs_delta']>=0 for r in data['conditions'].values()))
        data['passed'] = all(data['passes'].values())
        result['ratios'][name] = data
    result['passed'] = any(r['passed'] for r in result['ratios'].values())
    result['decision'] = 'Report result and stop; no attacks or additional variants'
    result['cpu_audit_seconds'] = time.monotonic()-started
    write(run / 'independent_audit.json', audit)
    write(run / 'results.json', result)
    print(json.dumps(dict(event='audit_complete', audit=audit, ratios={k:dict(converged=v['converged'],
        joint_clean=v['conditions']['clean']['joint'], joint_nine=v['nine_edits']['joint'], passed=v['passed']) for k,v in result['ratios'].items()})), flush=True)

    # Quality metrics remain CPU-only and run after the diffusion worker has exited.
    import torch
    import lpips
    from torchvision import transforms
    torch.set_num_threads(4)
    cached = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    assert cached.exists(), 'Missing cached LPIPS backbone; downloads are not allowed'
    metric = lpips.LPIPS(net='alex').eval().cpu()
    tensor = transforms.ToTensor()
    quality = []
    with torch.no_grad():
        for task in tasks:
            baseline = Image.open(run / 'images' / str(task) / 'baseline.png').convert('RGB')
            a = tensor(baseline).unsqueeze(0)
            for name in configs:
                image = Image.open(final_lookup[(task,name)]['image']).convert('RGB')
                b = tensor(image).unsqueeze(0)
                mse = float(torch.mean((a-b)**2))
                quality.append(dict(task=task, ratio=name, mse=mse, psnr=None if mse==0 else -10*math.log10(mse),
                                    lpips=float(metric(2*a-1,2*b-1).item())))
    write(run / 'quality_rows.json', quality)
    for name in configs:
        rows = [r for r in quality if r['ratio']==name]
        result['ratios'][name]['quality'] = dict(lpips=summary([r['lpips'] for r in rows]),
             psnr=summary([r['psnr'] for r in rows if r['psnr'] is not None]), exact_pixels=sum(r['mse']==0 for r in rows))
    write(run / 'results.json', result)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = run / 'figures'
    output.mkdir(exist_ok=True)
    arms = ['baseline', *configs]
    titles = ['Native GS', '1% (163 positions)', '3% (491 positions)', '6% (983 positions)']
    def contact(tasks_to_show, filename):
        fig, axes = plt.subplots(len(tasks_to_show), 4, figsize=(12,3*len(tasks_to_show)), layout='constrained')
        for i, task in enumerate(tasks_to_show):
            for j, arm in enumerate(arms):
                path = run/'images'/str(task)/'baseline.png' if arm=='baseline' else Path(final_lookup[(task,arm)]['image'])
                axes[i,j].imshow(Image.open(path), interpolation='nearest')
                axes[i,j].set(xticks=[],yticks=[])
                if i==0:
                    axes[i,j].set_title(titles[j],fontsize=11)
                if j==0:
                    axes[i,j].set_ylabel('Prompt '+str(task),fontsize=11)
        fig.savefig(output/filename,dpi=160)
        plt.close(fig)
    contact(plan['visual_tasks'],'preselected_comparison.png')
    worst = list(dict.fromkeys(r['task'] for r in sorted(quality,key=lambda r:r['lpips'],reverse=True)))[:3]
    contact(worst,'largest_lpips_comparison.png')
    labels = ['1%','3%','6%']
    fig, axes = plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax, key, target, title in zip(axes,['converged','clean','edits'],[90,90,90],
            ['Exact convergence within 3 rounds','Joint acceptance: clean','Joint acceptance: 9 edits']):
        for i,name in enumerate(configs):
            r = result['ratios'][name]
            if key=='converged':
                p,ci = r['converged']/50,r['converged_wilson95']
            elif key=='clean':
                p,ci = r['conditions']['clean']['joint']/50,r['conditions']['clean']['joint_wilson95']
            else:
                p,ci = r['nine_edits']['joint']/450,r['nine_edits']['joint_cluster95']
            ax.errorbar(i,p*100,yerr=[[max(0,p-ci[0])*100],[max(0,ci[1]-p)*100]],fmt='o',color='#0072B2',capsize=4)
        ax.axhline(target,color='#D55E00',linestyle='--',linewidth=1)
        ax.set(xticks=range(3),xticklabels=labels,ylim=(-3,103),title=title,ylabel='Percent')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'feasibility.png',dpi=220)
    fig.savefig(output/'feasibility.pdf')
    plt.close(fig)
    write(output/'manifest.json',dict(source_sha256=sha(run/'results.json'), script_sha256=sha(__file__),
        audience='Research mechanism-screen report; no journal format claimed',
        image_adjustments='None; original RGB pixels, equal display sizes, nearest display interpolation',
        preselected_tasks=plan['visual_tasks'], largest_lpips_tasks=worst, largest_lpips_selection='Post hoc descriptive illustration',
        intervals='Wilson95% for n50 prompt proportions; prompt-cluster percentile bootstrap5000 for450 edits',
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    write(run/'analysis_complete.json',dict(status='complete',cpu_seconds=time.monotonic()-started,gpu_jobs=0,
         quality_pairs=len(quality),script_sha256=sha(__file__),passed=result['passed']))
    print(json.dumps(dict(event='analysis_complete',passed=result['passed'],seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    assert args.run.parent.resolve()==(ROOT/'runs').resolve()
    if args.wait_pid:
        import select
        import subprocess
        try:
            descriptor=os.pidfd_open(args.wait_pid)
        except ProcessLookupError:
            descriptor=None
        except AttributeError:
            # This server Python lacks pidfd_open. Wait on process lifetime only;
            # do not inspect or poll experiment records.
            subprocess.run(['tail','--pid='+str(args.wait_pid),'-f','/dev/null'],check=True)
            descriptor=None
        if descriptor is not None:
            select.select([descriptor],[],[])
            os.close(descriptor)
    main(args.run)
