"""CPU audit of reserved coordinates, actual changes, and native GS redundancy."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')


def main(run):
    import numpy as np
    rows = json.loads((run/'rounds.json').read_text())
    private = np.load(run/'private_material.npz')
    output = []
    for row in rows:
        folder = run/'images'/str(row['task'])
        base = np.load(folder/'baseline.npz')['initial_latent']
        z = np.load(folder/(row['ratio']+'_r'+str(row['iteration'])+'.npz'))['initial_latent']
        votes = np.logical_xor(z>0,private['key']).reshape(4,8,8,8,8).sum(axis=(1,3))
        ba = float(np.mean((votes>32)==private['watermark'].reshape(4,8,8)))
        assert ba == row['initial_gs_ba']
        output.append(dict(task=row['task'],ratio=row['ratio'],iteration=row['iteration'],native_gs_ba=ba,
            changed_coordinates=int(np.sum(z!=base)),changed_signs=int(np.sum((z>0)!=(base>0))),
            relative_l2_change=float(np.linalg.norm(z.astype(float)-base.astype(float))/np.linalg.norm(base.astype(float)))))
    summary = {}
    for name in ['p01','p03','p06']:
        selected = [r for r in output if r['ratio']==name]
        summary[name] = dict(rounds=len(selected),native_gs_ba_min=min(r['native_gs_ba'] for r in selected),
            native_gs_ba_max=max(r['native_gs_ba'] for r in selected),
            changed_coordinates_mean=float(np.mean([r['changed_coordinates'] for r in selected])),
            changed_signs_mean=float(np.mean([r['changed_signs'] for r in selected])),
            relative_l2_change_mean=float(np.mean([r['relative_l2_change'] for r in selected])))
    value = dict(device='CPU249',independent_gs_noise_replays=len(output),mismatches=0,summary=summary,rows=output,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest())
    (run/'noise_channel_audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(checks=len(output),mismatches=0,summary=summary)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    assert args.run.parent==Path('/data0/liyanlei/watermark_research/runs')
    main(args.run)
