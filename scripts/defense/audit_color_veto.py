"""Reconstruct both decisions with sorted lists and verify all enrollment receipts."""
from bisect import bisect_right
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

from color_identity_probe import ROOT

RUN=ROOT/'runs/color_veto_20261119'
PARENT=ROOT/'runs/color_joint_20261119'


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path,'rt') as f:
        return [r for r in csv.DictReader(f) if r['family']!='clean']


def components(r,who,null):
    a,b=[(1+bisect_right(null[k],float(r[k+'_'+who])))/(1+len(null[k])) for k in ('color','layout')]
    assert a*b==float(r['joint_'+who])
    return a*b,max(a,b)


def main():
    receipt=read(RUN/'complete.json')
    for name,h in receipt['sha256'].items():
        assert digest(RUN/name)==h
    sources=read(RUN/'sources.json')
    assert Counter(s['dataset'] for s in sources)=={'COCO':30,'SDP':30}
    past=read(ROOT/'runs/phase1_trajectory_native_gs_20261113/protocol.json')['sources']
    past+=read(ROOT/'runs/unified_main_sources_20261116/selection.json')['sources']
    past+=read(ROOT/'runs/color_budget_20261119/sources.json')+read(PARENT/'sources.json')
    assert not ({(s['dataset'],str(s['source_id'])) for s in sources}&{(s['dataset'],str(s['source_id'])) for s in past})
    seen=set()
    for shard in range(3):
        gen=read(RUN/f'generated_{shard}.json')
        assert gen['gpu'] in (4,5,7) and len(gen['records'])==20
        for rec in gen['records']:
            s=sources[rec['task']]
            assert s['task']%3==shard and s['seeds']==rec['seeds']
            assert hashlib.sha256(s['prompt'].encode()).hexdigest()==rec['prompt_sha256']
            for j,h in enumerate(rec['sha256']):
                assert digest(RUN/'images'/f"{s['task']:03d}"/f'{j}.png')==h
                seen.add(h)
    assert len(seen)==len({v for s in sources for v in s['seeds']})==120
    refs=(RUN/'test_hybrid96_48.bin').read_bytes()
    assert len(refs)==120*288 and all(sum(entry)==65535 for entry in struct.iter_unpack('<144H',refs))
    cfg=read(PARENT/'calibration.json')
    cuts=read(RUN/'cutoffs.json')
    cal=rows(PARENT/'cal_scores.csv.gz')
    v=sorted(components(r,'own',cfg['null'])[1] for r in cal)[math.ceil(1280*.98)-1]
    assert v==cuts['veto']
    rr=rows(RUN/'test_scores.csv.gz')
    assert len(rr)==len({(r['task'],r['idx'],r['family'],r['level']) for r in rr})==1920
    report=read(RUN/'summary.json')
    for method in ('product','veto'):
        def value(r,who):
            prod,worst=components(r,who,cfg['null'])
            return math.inf if method=='veto' and worst>v else prod
        cut=sorted(value(r,'own') for r in cal)[1215]
        assert cut==cuts['methods'][method]['cut']
        items=report['methods'][method]
        threshold=sorted(value(r,'own') for r in rr)[1823]
        if math.isfinite(threshold):
            assert threshold==items['matched']['cutoff']
        else:
            assert items['matched'] is None
        for k,who,sub in [('pass','own',rr),('far','sibling',rr)]+[
            (key,'own',[r for r in rr if r['level']=='0.8' and r['family']==f])
            for key,f in [('strong_blur_pass','blurring'),('strong_contrast_pass','contrast')]]:
            vals=[value(r,who) for r in sub]
            assert sum(x<=cut for x in vals)/len(vals)==items['frozen'][k]
            if items['matched']:
                assert sum(x<=threshold for x in vals)/len(vals)==items['matched']['inclusive'][k]
                frac=items['matched']['tie_fraction']
                exact=(sum(x<threshold for x in vals)+frac*sum(x==threshold for x in vals))/len(vals)
                assert math.isclose(exact,items['matched']['exact'][k],abs_tol=1e-12)
    audit=dict(unique_images=120,stored_bytes=len(refs),edited_queries=len(rr),per_prompt_directions=2,
               methods=['product','veto'],frozen_and_matched_rates_reconstructed=True,
               complete_sha256=digest(RUN/'complete.json'),audit_script_sha256=digest(Path(__file__)))
    (RUN/'audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
