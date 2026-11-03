"""Evaluate geometry-defense scores on an existing sealed run (CPU only).

For every task we compare the reference's original target latent zT (saved at
preparation as original_noise.npy) against the target inversion zhatT of each
candidate image. The matched pair is the positive class; every other task's zT
against the same candidate is the negative class. AUC is reported; no threshold
is applied because the published deployment thresholds are still unknown.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/defense'))
import geometry as G


def main(run_dir, out_path):
    run = Path(run_dir)
    design = json.loads((run / 'design.json').read_text(encoding='utf-8'))
    tasks = [t for t in design['tasks']
             if (run / 'evaluate' / (str(t['id']) + '_' + '30')).is_dir()
             and (run / 'prepare' / str(t['id']) / 'original_noise.npy').is_file()]
    zts = {}
    for task in tasks:
        sid = task['id']
        zts[sid] = np.load(run / 'prepare' / str(sid) / 'original_noise.npy', allow_pickle=False)
    label_dirs = {}
    for task in tasks:
        sid = task['id']
        for folder in sorted((run / 'evaluate').glob(str(sid) + '_*')):
            label = folder.name[len(str(sid)) + 1:]
            label_dirs.setdefault(label, {})[sid] = np.load(folder / 'target_inverse.npy', allow_pickle=False)
    result = {'run': str(run), 'tasks': len(tasks), 'labels': {}, 'scope': 'paper-formula scores; no threshold decision'}
    for label, inverses in sorted(label_dirs.items()):
        rows = []
        pos_cos, neg_cos, pos_spd, neg_spd = [], [], [], []
        for task in tasks:
            sid = task['id']
            cand = inverses[sid]
            c = G.gcos(zts[sid], cand)
            d = G.lspd(zts[sid], cand)
            rows.append(dict(task_id=sid, gcos=c, sad=G.sad(zts[sid], cand), lspd=d))
            pos_cos.append(c)
            pos_spd.append(d)
            for other in tasks:
                oid = other['id']
                if oid == sid:
                    continue
                neg_cos.append(G.gcos(zts[oid], cand))
                neg_spd.append(G.lspd(zts[oid], cand))
        result['labels'][label] = dict(
            n=len(rows),
            gcos_mean=float(np.mean(pos_cos)), gcos_median=float(np.median(pos_cos)),
            gcos_min=float(np.min(pos_cos)), gcos_max=float(np.max(pos_cos)),
            gcos_neg_mean=float(np.mean(neg_cos)), gcos_neg_max=float(np.max(neg_cos)),
            gcos_auc=G.auc(pos_cos, neg_cos),
            sad_mean=float(np.mean([r['sad'] for r in rows])),
            lspd_mean=float(np.mean(pos_spd)), lspd_median=float(np.median(pos_spd)),
            lspd_min=float(np.min(pos_spd)), lspd_max=float(np.max(pos_spd)),
            lspd_neg_mean=float(np.mean(neg_spd)), lspd_neg_min=float(np.min(neg_spd)),
            lspd_auc=G.auc(neg_spd, pos_spd),
            rows=rows)
        print(label, 'gcos_auc %.4f' % result['labels'][label]['gcos_auc'],
              'lspd_auc %.4f' % result['labels'][label]['lspd_auc'],
              'gcos %.3f' % result['labels'][label]['gcos_mean'])
    Path(out_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
