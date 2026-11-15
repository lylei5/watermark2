"""Post-completion audit of frozen transfer observations; no new inference."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/capsule_source_transfer_20261115_v3'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    done=read(RUN/'complete.json');plan=read(RUN/'protocol.json');assert done['status']=='complete'
    for n,h in done['artifacts'].items():assert sha(RUN/n)==h
    rows=read(RUN/'rows.json');groups=defaultdict(dict)
    for r in rows:
        assert r['method'] not in groups[r['task'],r['kind']]
        groups[r['task'],r['kind']][r['method']]=r
        assert r['returned']==(r['key_recovered'] and r['score']<=plan['thresholds'][r['method']])
        if r['key_recovered']:assert r['detail']['key_confirmed']
    assert len(groups)==1300 and len(rows)==2600
    for part in groups.values():
        assert set(part)==set(plan['methods'])
        assert len({r['pixels_sha256'] for r in part.values()})==1
        for control in plan['controls']:assert len({r[control] for r in part.values()})==1
    assert sum(r['key_recovered'] for r in rows)==done['exact_keys']
    assert sum(r['returned'] for r in rows)==done['payload_returns']
    assert done['reference_overlap']==0 and done['gpu'] in [7,5,4]
    counts=Counter(t for t,k in groups);assert len(counts)==100 and set(counts.values())=={13}
    refs=[next(iter(part.values())) for (task,kind),part in groups.items() if kind=='bound']
    matched=read(RUN/'matched.json');summary=Counter('better' if r['candidate']<r['baseline'] else 'equal' if r['candidate']==r['baseline'] else 'worse' for r in matched)
    out=dict(status='complete',source_sha256=sha(Path(__file__)),input_complete_sha256=sha(RUN/'complete.json'),
        observations=len(rows),paired_queries=len(groups),reference_ids=len(refs),distinct_reference_pixels=len({r['pixels_sha256'] for r in refs}),
        distinct_query_pixels=len({next(iter(part.values()))['pixels_sha256'] for part in groups.values()}),
        exact_keys=done['exact_keys'],payload_returns=done['payload_returns'],enrollment_verification_decodes=done['enrollment_verification_decodes'],
        matched_points=len(matched),exact_genuine_count_matches=sum(r['exact_match'] for r in matched),matched_outcomes=dict(summary),
        gpu=done['gpu'],gpu_uuid=done['gpu_uuid'],native_calls=0,attack_generation=False,sscd_feature_parity=read(RUN/'model_parity.json'))
    target=RUN/'audit.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
