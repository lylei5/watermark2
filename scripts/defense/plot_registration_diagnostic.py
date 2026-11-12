"""Plot all genuine queries and final published endpoints with oracle axes."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--replicas', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    registration = json.loads(a.registration.read_text())
    replicas = json.loads(a.replicas.read_text())
    physical = {r['id']: r for r in replicas['rows'] if r['phase'] != 'calibration'}
    rows = []
    for r in registration['rows']:
        if r['phase'] != 'genuine' and not (r['kind'] == 'candidate' and r['update'] == 151):
            continue
        q = physical[r['id']]
        assert q['oracle_semantic_error'] == r['auxiliary_hamming']
        rows.append(dict(id=r['id'], task=r['task'], kind=r['kind'],
            accepted=r['original_accepted'], physical_correct=q['oracle_channel_correct'],
            auxiliary_hamming=r['auxiliary_hamming'], reference_hamming=r['reference_hamming']))
    assert len(rows) == 132
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'registration_source_data.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader(); w.writerows(rows)
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained', sharex=True, sharey=True)
        for ax, key, title in zip(axes, ['auxiliary_hamming', 'reference_hamming'],
                                  ['a  Auxiliary enrollment', 'b  Delivered-image reference']):
            for kind, color, marker, label in [('genuine', '#0072B2', 'o', 'Genuine query'),
                                               ('candidate', '#D55E00', '^', 'Final Imprint')]:
                for accepted in [True, False]:
                    rr = [r for r in rows if r['kind'] == kind and r['accepted'] == accepted]
                    ax.scatter([r['physical_correct'] for r in rr], [r[key] for r in rr],
                        marker=marker, s=25 if kind == 'genuine' else 45,
                        facecolors=color if accepted else 'none', edgecolors=color, linewidths=.8,
                        alpha=.8, label=label + (' / accepted' if accepted else ' / rejected'))
            for name, label, offset in [('008_full_00', '008 clean', (-55, 10)),
                                        ('011_full_01', '011 crop', (12, 3)),
                                        ('004_full_update_151', '004 Imprint', (9, -40))]:
                r = next(r for r in rows if r['id'] == name)
                ax.annotate(label, (r['physical_correct'], r[key]), xytext=offset,
                    textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='-', lw=.6, color='#555555'))
            ax.set(xlim=(.45, 1), ylim=(-.015, .60), title=title,
                   xlabel='Physical sign correctness (enrollment oracle)')
        axes[0].set_ylabel('Semantic Hamming rate')
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=7, frameon=False)
        fig.suptitle('Oracle diagnosis: 120 genuine queries + 12 full-budget endpoints', fontsize=10)
        fig.savefig(a.output / 'registration_diagnostic.png', dpi=300)
        fig.savefig(a.output / 'registration_diagnostic.pdf')
        plt.close(fig)
    endpoint = next(r for r in rows if r['id'] == '004_full_update_151')
    dominated = [r['id'] for r in rows if r['kind'] == 'genuine' and r['accepted']
        and r['physical_correct'] >= endpoint['physical_correct']
        and r['auxiliary_hamming'] >= endpoint['auxiliary_hamming']]
    manifest = dict(script_sha256=sha(__file__), registration_sha256=sha(a.registration),
        replica_diagnostic_sha256=sha(a.replicas), source_rows=132, sources=12,
        packages=dict(python=platform.python_version(), matplotlib=matplotlib.__version__),
        exclusions='None within 120 genuine and12 final endpoints; intermediate checkpoints are outside this figure scope.',
        uncertainty='All observations shown; conditions from the same source are dependent. No confidence intervals or fitted boundaries.',
        oracle_scope='Physical correctness uses initial target signs; right panel also uses stored delivered-reference semantic codes. Neither is an implemented defense.',
        monotone_discordance_obstruction=dict(endpoint=endpoint['id'], genuine_ids=dominated,
            scope='Only suspicion scores nondecreasing in BOTH physical correctness and auxiliary semantic mismatch. This is not an impossibility result for arbitrary detectors.'),
        figure_inches=[8, 4], png_dpi=300,
        alt_text='Oracle scatter plots compare physical recovery with semantic mismatch under auxiliary versus delivered-image registration. A clean legitimate source has more auxiliary mismatch and stronger carrier recovery than the accepted forgery; a cropped genuine query remains distant from the delivered reference.')
    (a.output / 'registration_figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(plotted=True, rows=len(rows), monotone_dominated_genuine=dominated)))


if __name__ == '__main__':
    main()
