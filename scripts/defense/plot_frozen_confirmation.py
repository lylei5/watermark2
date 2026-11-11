"""Plot complete frozen evaluations with actual test rates and source counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    assert data['completed'] and data['all_input_hashes_verified']
    args.output.mkdir(parents=True, exist_ok=True)
    rows = data['genuine_by_condition']
    groups = {r['group']: r for r in data['summary']}
    n = data['queries']
    wn = groups['wrong_identity']['n']
    sources = data['confirmation_sources']
    label_map = {'blur1_resize80': 'Blur 1 + resize 80%', 'crop90': 'Center crop 90% + resize',
                 'clean': 'Clean', 'jpeg50': 'JPEG 50', 'jpeg75': 'JPEG 75', 'noise0.01': 'Noise 0.01'}
    methods = [('default_accepted', 'Default BA', '#0072B2', 'o'),
               ('all_fit_accepted', 'All-fit geometry (primary)', '#D55E00', 'D'),
               ('split_fit_accepted', 'Split-fit geometry', '#333333', 'x')]
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'figure.dpi': 150}):
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4.8, .4 * len(rows) + 2)), layout='constrained')
        y = np.arange(len(rows))
        for i, (key, label, color, marker) in enumerate(methods):
            axes[0].scatter([100 * r[key] / r['n'] for r in rows], y + (i - 1) * .2,
                            marker=marker, color=color, s=35, label=label)
        axes[0].set(yticks=y, yticklabels=[label_map.get(r['group'], r['group']) for r in rows],
                    xlim=(-3, 105), xticks=[0, 25, 50, 75, 100], xlabel='Observed genuine acceptance (%)',
                    title=f'A  Genuine queries (n = {sources} per condition)')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=.15)
        axes[0].legend(loc='lower left', bbox_to_anchor=(0, -.38), fontsize=8, frameon=False)
        ax = axes[1]
        for i, point in enumerate(data['supplementary_ba_controls']):
            xx = 100 * point['genuine_accepted'] / n
            yy = 100 * point['wrong_identity_accepted'] / wn
            ax.scatter(xx, yy, marker='o', color='#0072B2', s=35)
            ax.annotate(f"BA cal. target {point['target_absolute_tpr']:.0%}", (xx, yy),
                        xytext=[(-95, 12), (-95, 12), (-95, 8), (-100, -20)][i],
                        textcoords='offset points', fontsize=8)
        for key, label, color, marker in methods:
            if key == 'split_fit_accepted':
                continue
            xx = 100 * groups['genuine'][key] / n
            yy = 100 * groups['wrong_identity'][key] / wn
            ax.scatter(xx, yy, marker=marker, color=color, s=60,
                       edgecolors='black', linewidths=.6,
                       label=f"{label}: {groups['genuine'][key]}/{n}; {groups['wrong_identity'][key]}/{wn}")
        ax.set(xlim=(-3, 105), ylim=(-6, 106), xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100],
               xlabel='Observed genuine acceptance (%)', ylabel='Observed wrong-identity acceptance (%)',
               title='B  Actual rates at frozen thresholds')
        ax.legend(loc='upper left', fontsize=7, frameon=False)
        ax.grid(alpha=.15)
        fig.suptitle(f'Frozen verification: {sources} sources, {n} queries', fontsize=12)
        fig.get_layout_engine().set(rect=(0, .15, 1, .72))
        fig.text(.5, .035, f'All {len(rows)} views/source retained. Correlated identity pairs; zero observed errors is not an ultra-low FPR guarantee.',
                 ha='center', fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(args.output / f'confirmation.{ext}', dpi=300)
        plt.close(fig)
    with (args.output / 'condition_counts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'figure_provenance.json').write_text(json.dumps(dict(
        summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=sources, queries=n, wrong_identity_decisions=wn, omitted_conditions=[],
        uncertainty_displayed=False, counts_only=True, publisher_format_verified=False,
        dpi=300, matplotlib_version=matplotlib.__version__), indent=2) + '\n')


if __name__ == '__main__':
    main()
