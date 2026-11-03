"""Reproduction attempt for the Geometry defence (ICML 2026), paper's own task.

Why this file exists
--------------------
The earlier `reports/geometry_30.json` answered a DIFFERENT question. There the
negative class was "this candidate's inverse vs another task's zT", i.e. an
identity-matching task. The paper (Tab.2 / Tab.4) asks a forgery-detection
question: separate GENUINE watermarked recoveries from FORGED recoveries, both
scored against their own reference zT.

So the previous numbers (G-Cos AUC 0.9958, L-SPD 0.638) are not comparable to
the paper's reported values at all. This script rebuilds the paper's task from
data already on disk, with no GPU and no new generation.

Data used (all from the frozen baseline run)
--------------------------------------------
  prepare/<id>/original_noise.npy        zT of the pair
  evaluate/<id>_reference/target_inverse.npy    zhat_T of the genuine watermarked image
  evaluate/<id>_20/target_inverse.npy           zhat_T of the forged candidate, 20 updates
  evaluate/<id>_30/target_inverse.npy           zhat_T of the forged candidate, 30 updates
  evaluate/<id>_cover/target_inverse.npy        zhat_T of the clean cover (no watermark)

Comparison target (paper Tab.4, proxy SD2.1 -> target SDXL, GS):
  20 updates: G-Cos 1.000 / L-SPD 0.997
  50 updates: G-Cos 1.000 / L-SPD 0.995
  100 updates: G-Cos 1.000 / L-SPD 0.992

Claim boundary
--------------
This is a paper-formula reimplementation, NOT the authors' code. The authors
have not released the defence code (`geodistort-wm` README: "will be released
soon"; latest commit 72880c9). The rejection region R, the thresholds tau_s and
tau_l, the final AND rule, epsilon and the token-to-patch mapping are all
unknown. Therefore this script reports AUC ONLY. It does not produce, and must
not be reported as, an accept/reject decision.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import geometry as G  # noqa: E402

PAPER_TAB4_GS_SDXL = {20: (1.000, 0.997), 50: (1.000, 0.995), 100: (1.000, 0.992)}


def load(p):
    return np.load(p)


def main(run_dir, out_path):
    run = Path(run_dir)
    tasks = sorted(int(d.name.split('_')[0]) for d in (run / 'evaluate').iterdir()
                   if d.is_dir() and d.name.endswith('_reference'))
    rows = []
    for sid in tasks:
        zt = load(run / 'prepare' / str(sid) / 'original_noise.npy')
        rec = {}
        for label in ('reference', 'cover', '20', '30'):
            p = run / 'evaluate' / f'{sid}_{label}' / 'target_inverse.npy'
            if not p.exists():
                rec[label] = None
                continue
            z = load(p)
            rec[label] = dict(gcos=G.gcos(zt, z), lspd=G.lspd(zt, z))
        rows.append(dict(task_id=sid, **rec))

    def vals(label, key):
        return [r[label][key] for r in rows if r.get(label)]

    out = dict(
        reproduction='geometry_paper_task',
        official_code=False,
        run=str(run),
        n_tasks=len(rows),
        note='AUC only; thresholds and decision rule are not public, so no accept/reject is produced.',
        classes={},
        comparison_to_paper={},
        rows=rows,
    )

    # Paper's task: genuine (reference) vs forged (candidate), both against own zT.
    # Scoring direction: genuineness score = +G-Cos (genuine high), = -L-SPD (genuine low).
    sign = {'gcos': 1.0, 'lspd': -1.0}
    for label, steps in (('20', 20), ('30', 30)):
        for key, name in (('gcos', 'G-Cos'), ('lspd', 'L-SPD')):
            gen = vals('reference', key)
            forg = vals(label, key)
            a = G.auc([sign[key] * v for v in gen], [sign[key] * v for v in forg])
            out['classes'][f'{name}_genuine_vs_forged{label}'] = dict(
                auc=a,
                genuine_mean=float(np.mean(gen)),
                forged_mean=float(np.mean(forg)),
                n=len(gen),
            )
    for key, name in (('gcos', 'G-Cos'), ('lspd', 'L-SPD')):
        gen = vals('reference', key)
        cov = vals('cover', key)
        out['classes'][f'{name}_genuine_vs_cover'] = dict(
            auc=G.auc([sign[key] * v for v in gen], [sign[key] * v for v in cov]),
            genuine_mean=float(np.mean(gen)),
            cover_mean=float(np.mean(cov)),
            n=len(gen),
        )

    for steps, (pg, pl) in PAPER_TAB4_GS_SDXL.items():
        if steps not in (20, 30):
            continue
        mine_g = out['classes'][f'G-Cos_genuine_vs_forged{steps}']['auc']
        mine_l = out['classes'][f'L-SPD_genuine_vs_forged{steps}']['auc']
        out['comparison_to_paper'][str(steps)] = dict(
            paper_gcos=pg, paper_lspd=pl, ours_gcos=mine_g, ours_lspd=mine_l,
            delta_gcos=mine_g - pg, delta_lspd=mine_l - pl,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
