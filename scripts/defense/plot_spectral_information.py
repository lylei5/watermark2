"""Content-only information controls, with complete counts and source provenance."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def main():
    assert ROOT.is_dir()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    r = ROOT / 'runs/shared_channel_binding_20261114_v2'
    c = ROOT / 'runs/copynce_control_20261114'
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    information = read(r / 'spectral_information.json')
    plan = read(r / 'protocol.json')
    meta = read(r / 'feature_rows.json')
    copied = read(c / 'raw_rows.json')
    assert read(c / 'extract_complete.json')['artifacts']['raw_rows.json'] == sha(c / 'raw_rows.json')
    index = {(x['id'], x['kind']): i for i, x in enumerate(meta)}
    with np.load(r / 'descriptors.npz') as z:
        sscd = z['sscd']
    all_rows = list(information['rows'])
    for x in copied:
        j = index[x['id'], 'clean']
        i = index[x['query_id'], 'clean' if x['kind'] == 'wrong_record' else x['kind']]
        common = {k: x[k] for k in ['id', 'query_id', 'carrier', 'task', 'split', 'kind']}
        for method, distance in [('sscd', float(np.clip(1-sscd[j] @ sscd[i], 0, 2))),
                                  ('copynce_descriptor', x['descriptor_distance']), ('copynce_matcher', x['matcher_distance'])]:
            all_rows.append(dict(**common, method=method, distance=distance))
    methods = [('shared_phase_pooled', 'Relative phase, 512D'), ('shared_phase_full', 'Relative phase, 16384D'),
               ('content_phase_full', 'Content phase, 16384D'), ('noise_phase_full', 'Noise phase, 16384D'),
               ('sscd', 'SSCD, 512D'), ('copynce_descriptor', 'CopyNCE descriptor, 512D'), ('copynce_matcher', 'CopyNCE matcher, reference image')]
    fields = [('all_genuine', 'Legitimate retained (higher is better)', '#0072B2'),
              ('wrong_record', 'Wrong record accepted (lower is better)', '#B35A00'),
              ('imprint_150', 'Full Imprint accepted (lower is better)', '#B35A00')]
    data = []
    for method, label in methods:
        part = [x for x in all_rows if x['method'] == method]
        calibration = sorted(x['distance'] for x in part if x['split'] == 'calibration' and x['kind'] in ['clean', *plan['conditions']])
        assert len(calibration) == 30
        cutoff = calibration[28]
        for carrier in ['TR', 'PRC']:
            for kind, _, _ in fields:
                group = [x for x in part if x['carrier'] == carrier and (x['kind'] in ['clean', *plan['conditions']] if kind == 'all_genuine' else x['kind'] == kind)]
                data.append(dict(method=method, label=label, carrier=carrier, kind=kind, threshold=cutoff,
                                 accepted=sum(x['distance'] <= cutoff for x in group), total=len(group)))
    out = r / 'information_figure'
    out.mkdir(exist_ok=False)
    with (out / 'source_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.30, right=.985, top=.86, bottom=.14, wspace=.13, hspace=.35)
        y = np.arange(len(methods))
        for row, carrier in enumerate(['TR', 'PRC']):
            for col, (kind, title, color) in enumerate(fields):
                ax = axes[row, col]
                selected = [next(x for x in data if x['method'] == method and x['carrier'] == carrier and x['kind'] == kind) for method, _ in methods]
                values = [100*x['accepted']/x['total'] for x in selected]
                ax.barh(y, values, height=.60, color=color, alpha=.9, zorder=2)
                for position, value, point in zip(y, values, selected):
                    ax.text(2 if value < 35 else value-2, position, f"{point['accepted']}/{point['total']}",
                            ha='left' if value < 35 else 'right', va='center', color='black' if value < 35 else 'white', fontsize=9)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                ax.grid(axis='x', color='#dddddd', linewidth=.6, zorder=0)
                ax.tick_params(axis='y', length=0)
                ax.set_yticks(y, [label for _, label in methods])
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=12)
                else:
                    ax.set_xlabel('Observed acceptance (%)')
                if col == 0:
                    ax.text(-1.18, 1.04, carrier, transform=ax.transAxes, fontsize=13, weight='bold')
        axes[0, 0].invert_yaxis()
        fig.suptitle('Instance separation and forgery rejection are different requirements', x=.30, ha='left', y=.98, fontsize=14)
        fig.text(.30, .925, 'Content-only decisions; thresholds frozen on 30 GS calibration images at the 95% target.', fontsize=10)
        fig.text(.02, .055, '10 source images per carrier. The 90 wrong-record pairs reuse these sources; counts are not independent trials.\n'
                 '16384D controls use 32x the dimensions; the CopyNCE matcher reads the full reference. Development data; no native-watermark gate.', fontsize=9)
        fig.savefig(out / 'instance_and_forgery.png', dpi=180)
        fig.savefig(out / 'instance_and_forgery.pdf')
        plt.close(fig)
    with Image.open(out / 'instance_and_forgery.png') as im:
        metadata = dict(pixels=list(im.size), mode=im.mode, dpi=im.info.get('dpi'))
    (out / 'manifest.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),
        inputs={str(p): sha(p) for p in [r / 'spectral_information.json', r / 'descriptors.npz', c / 'raw_rows.json', r / 'protocol.json']},
        output_files={p.name: sha(p) for p in out.iterdir()}, metadata=metadata, dimensions_inches=[12.4, 7.5],
        uncertainty='Observed finite-cohort counts; no interval treating repeated record pairs as independent',
        target='General research report; publisher-specific formatting not assessed',
        software_reference='Scientific Agent Skills, https://arxiv.org/abs/2609.00065',
        alt_text='At a frozen GS95 calibration target, retaining full relative phase removes ordinary wrong-record matches but still accepts most full Imprint outputs. SSCD and the released CopyNCE matcher provide stronger forgery rejection on these TR and PRC development images.'), indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
