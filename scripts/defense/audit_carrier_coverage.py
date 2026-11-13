"""Verify complete published TR/PRC attack artifacts and fixed SSCD scores on249."""
import csv
import json
from pathlib import Path

from recovered_carrier_transfer import ROOT,PARENT,GATE,sha,write


def main():
    assert ROOT.is_dir()
    output=PARENT/'coverage_audit.json';assert not output.exists()
    plan=json.loads((PARENT/'protocol.json').read_text())
    for name,digest in plan['source']['files'].items():assert sha(Path(plan['source']['path'])/name)==digest
    gate=json.loads((GATE/'complete.json').read_text())
    assert gate['status']=='complete'and gate['protocol_sha256']==sha(GATE/'protocol.json')
    for name,digest in gate['artifacts'].items():assert sha(GATE/name)==digest
    gate_rows=json.loads((GATE/'rows.json').read_text());assert len(gate_rows)==260
    summaries=[];images=0
    for carrier in ['TR','PRC']:
        folder=PARENT/carrier;done=json.loads((folder/'complete.json').read_text())
        assert done['status']=='complete'and done['protocol_sha256']==sha(PARENT/'protocol.json')
        assert sha(folder/'attack=forgery/all_results.csv')==done['raw_sha256']
        original=list(csv.DictReader((folder/'attack=forgery/all_results.csv').open()))
        assert len(original)==70
        expected={(i,stage,k)for i in range(10)for stage,k in [('watermarked',-1),
            *[(stage,k)for stage in ['attacked','defended']for k in [50,100,150]]]}
        assert len(done['rows'])==70
        assert {(r['task'],r['stage'],r['checkpoint'])for r in done['rows']}==expected
        for r in done['rows']:
            assert r['accepted']==(r['metric']<plan['thresholds']['TR']['value']if carrier=='TR'else r['metric'])
        for name,digest in done['image_sha256'].items():assert sha(folder/name)==digest;images+=1
        for stage,k in [('watermarked',-1),*[(stage,k)for stage in ['attacked','defended']for k in [50,100,150]]]:
            rr=[r for r in done['rows']if r['stage']==stage and r['checkpoint']==k]
            summaries.append(dict(carrier=carrier,stage=stage,checkpoint=k,accepted=sum(r['accepted']for r in rr),total=len(rr)))
    result=dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(PARENT/'protocol.json'),
        carrier_rows=140,validated_images=images,source_files=len(plan['source']['files']),actual_updates_per_source=151,
        sources_per_carrier=10,summary=summaries,sscd_rows=len(gate_rows),sscd_complete_sha256=sha(GATE/'complete.json'),
        scope='Complete original Imprint budgets; genuine/PGID and actual recovery are reported separately')
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
