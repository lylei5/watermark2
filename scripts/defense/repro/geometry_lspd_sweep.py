"""L-SPD sensitivity sweep for the Geometry reproduction.

Diagnosis being tested
----------------------
With the paper's 16x16 grid, a grid cell holds 4x4 = 16 latent positions, each
carrying C=4 channels. Our reading takes those 16 positions as the token
neighbourhood, so the local covariance is a 4x4 SPD matrix estimated from M=16
samples in N=4 dimensions. For two INDEPENDENT token sets, the expected AIRM
distance between two such Wishart-like sample covariances is a content-free
constant. If that is what our score measures, genuine and forged recoveries
will both sit at the same value (~3.3), which is exactly what we observe
(genuine 3.284 vs forged 3.334, AUC 0.48).

This sweep varies the neighbourhood size (grid) and the regulariser to see
whether any admissible reading of Definition 4.8 reproduces the paper's
reported L-SPD AUC (0.997 for GS, proxy SD2.1 -> target SDXL, 20 updates).

A control is included: the same statistic computed between zT and an unrelated
task's zT. If L-SPD stays ~constant there too, the statistic is content-blind
rather than discriminative.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import geometry as G  # noqa: E402


def main(run_dir, out_path):
    run = Path(run_dir)
    tasks = sorted(int(d.name.split('_')[0]) for d in (run / 'evaluate').iterdir()
                   if d.is_dir() and d.name.endswith('_reference'))
    zt = {t: np.load(run / 'prepare' / str(t) / 'original_noise.npy') for t in tasks}
    inv = {}
    for t in tasks:
        for label in ('reference', '20'):
            p = run / 'evaluate' / f'{t}_{label}' / 'target_inverse.npy'
            if p.exists():
                inv[(t, label)] = np.load(p)

    out = dict(reproduction='geometry_lspd_grid_sweep', run=str(run), n_tasks=len(tasks), sweeps=[])
    for grid in (4, 8, 16, 32):
        for eps in (1e-6, 1e-2):
            gen, forg, ctrl = [], [], []
            for t in tasks:
                a = zt[t]
                if (t, 'reference') in inv:
                    gen.append(G.lspd(a, inv[(t, 'reference')], grid=grid, eps=eps))
                if (t, '20') in inv:
                    forg.append(G.lspd(a, inv[(t, '20')], grid=grid, eps=eps))
                t2 = tasks[(tasks.index(t) + 1) % len(tasks)]
                if (t2, '20') in inv:
                    ctrl.append(G.lspd(a, inv[(t2, '20')], grid=grid, eps=eps))
            row = dict(
                grid=grid,
                cell_tokens=(64 // grid) ** 2,
                eps=eps,
                auc_genuine_vs_forged=G.auc([-v for v in gen], [-v for v in forg]),
                genuine_mean=float(np.mean(gen)),
                forged_mean=float(np.mean(forg)),
                control_mean=float(np.mean(ctrl)),
                control_gap=float(abs(np.mean(gen) - np.mean(ctrl))),
            )
            out['sweeps'].append(row)
            print(json.dumps(row), flush=True)

    out['paper_reference'] = dict(lspd_auc_gs_sdxl_20updates=0.997)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
