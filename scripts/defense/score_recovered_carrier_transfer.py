"""One-shot exit-triggered audit of recovery, native, PGID and SSCD transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from recovered_carrier_transfer import ROOT,PARENT,GENUINE,wait,sha,write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parents',type=Path,required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT/'runs'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    state = dict(status='waiting_for_transfer_and_genuine_exit',pid=os.getpid(),
                 parents_sha256=sha(args.parents),source_sha256=sha(__file__))
    assert not (run/'audit_state.json').exists()
    write(run/'audit_state.json',state)
    try:
        parents = json.loads(args.parents.read_text())
        for previous in parents:
            wait(previous)
        plan = json.loads((run/'protocol.json').read_text())
        done = json.loads((run/'complete.json').read_text())
        assert done['status'] == 'complete' and done['protocol_sha256'] == sha(run/'protocol.json')
        for name,digest in done['artifacts'].items():
            assert sha(run/name) == digest
        rows = json.loads((run/'rows.json').read_text())
        controls = json.loads((run/'controls.json').read_text())
        assert len(rows) == 3360 and len(controls) == 280
        assert all(r['key_recovered'] == (r['region_distance'] <= 127) for r in rows)
        lookup = {(r['id'],r['kind']):r for r in controls}
        native,pgid,disagreements = {},{},[]
        evidence = {}
        for carrier in ['TR','PRC']:
            attack_done = json.loads((PARENT/carrier/'complete.json').read_text())
            genuine_done = json.loads((GENUINE/carrier/'complete.json').read_text())
            assert attack_done['status'] == genuine_done['status'] == 'complete'
            assert genuine_done['protocol_sha256'] == plan['genuine_protocol_sha256']
            assert sha(GENUINE/carrier/'rows.json') == genuine_done['rows_sha256']
            evidence[carrier] = dict(attack=sha(PARENT/carrier/'complete.json'),genuine=sha(GENUINE/carrier/'complete.json'))
            for r in json.loads((GENUINE/carrier/'rows.json').read_text()):
                key = (f"{carrier}_{r['task']:03d}",r['kind'])
                assert lookup[key]['pixels_sha256'] == r['pixels_sha256']
                (pgid if r['defended'] else native)[key] = r['accepted']
            for r in attack_done['rows']:
                if r['stage'] == 'watermarked':
                    continue
                key = (f"{carrier}_{r['task']:03d}",f"imprint_{r['checkpoint']}")
                assert r['stage'] in ['attacked','defended']
                (native if r['stage'] == 'attacked' else pgid)[key] = r['accepted']
        assert len(native) == len(pgid) == 280
        for r in controls:
            key = (r['id'],r['kind'])
            if r['native_accepted'] != native[key]:
                disagreements.append(dict(id=r['id'],kind=r['kind'],fresh=r['native_accepted'],prior=native[key]))
        conditions = list(plan['conditions'])
        def summarize(part,accept):
            output = []
            for carrier in ['TR','PRC']:
                for kind in ['clean','all_edits','all_genuine','cover','imprint_50','imprint_100','imprint_150',*conditions]:
                    group = [r for r in part if r['carrier'] == carrier and
                        (r['kind'] in conditions if kind == 'all_edits' else
                         r['kind'] in ['clean',*conditions] if kind == 'all_genuine' else r['kind'] == kind)]
                    assert group
                    output.append(dict(carrier=carrier,kind=kind,accepted=sum(bool(accept(r))for r in group),total=len(group)))
            return output
        summaries = [dict(method='native_fresh',summary=summarize(controls,lambda r:r['native_accepted'])),
                     dict(method='native_prior',summary=summarize(controls,lambda r:native[r['id'],r['kind']])),
                     dict(method='PGID',summary=summarize(controls,lambda r:pgid[r['id'],r['kind']]))]
        for repetition in range(3):
            for method in plan['methods']:
                part = [r for r in rows if r['repetition'] == repetition and r['method'] == method]
                summaries.append(dict(method=method,repetition=repetition,point='maximum_radius127',
                                      summary=summarize(part,lambda r:r['joint_accepted'])))
        for point in plan['operating_points']:
            item = dict(**point,point='Frozen GS calibration transferred without fitting')
            if not point['feasible']:
                item['summary'] = None
            elif point['repetition'] is None:
                item['summary'] = summarize(controls,lambda r:r['native_accepted'] and r[point['method']] <= point['cutoff'])
            else:
                part = [r for r in rows if r['method'] == point['method'] and r['repetition'] == point['repetition']]
                item['summary'] = summarize(part,lambda r:r['joint_accepted'] and r['region_distance'] <= point['cutoff'])
            summaries.append(item)
        result = dict(status='complete',summaries=summaries,native_replay_disagreements=disagreements,
            recovery_decisions=len(rows),native_inversions=len(controls),protocol_sha256=sha(run/'protocol.json'),
            evidence=evidence,source_sha256=sha(__file__),attack_generation=False,
            scope='Full-denominator transfer with frozen GS calibration; native replay differences preserved')
        assert not (run/'audited_scores.json').exists()
        write(run/'audited_scores.json',result)
        state.update(status='complete',scores_sha256=sha(run/'audited_scores.json'))
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        write(run/'audit_state.json',state)


if __name__ == '__main__':
    main()
