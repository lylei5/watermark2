"""Descriptive TR PGID threshold tradeoffs on complete paired evaluation data."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert ROOT.is_dir()and args.output.resolve().is_relative_to(ROOT/'runs')and not args.output.exists()
    attack=ROOT/'runs/pgid_carrier_coverage_20261113'
    genuine=ROOT/'runs/pgid_carrier_genuine_20261113_v3/TR'
    done=json.loads((genuine/'complete.json').read_text());assert done['status']=='complete'
    assert sha(genuine/'rows.json')==done['rows_sha256']
    controls=json.loads((genuine/'rows.json').read_text())
    attacked=json.loads((attack/'TR/complete.json').read_text());assert attacked['status']=='complete'
    assert sha(attack/'TR/attack=forgery/all_results.csv')==attacked['raw_sha256']
    plan=json.loads((attack/'protocol.json').read_text())
    positive=[r for r in controls if r['defended']and r['kind']!='cover'];assert len(positive)==100
    values=sorted(r['metric']for r in positive)
    points=[('author',plan['thresholds']['TR']['value'],None)]
    for count in [60,89,90,91,95,97,100]:points.append(('evaluation_matched',math.nextafter(values[count-1],math.inf),count))
    rows=[]
    for kind,threshold,target in points:
        record=dict(point=kind,target_legitimate=target,threshold=threshold,operator='<',
                    legitimate=sum(r['metric']<threshold for r in positive),legitimate_total=100,
                    cover=sum(r['metric']<threshold for r in controls if r['defended']and r['kind']=='cover'),cover_total=10)
        for step in [50,100,150]:
            part=[r for r in attacked['rows']if r['stage']=='defended'and r['checkpoint']==step];assert len(part)==10
            record[f'imprint_{step}']=sum(r['metric']<threshold for r in part)
        record['imprint_total_per_checkpoint']=10;rows.append(record)
    result=dict(status='complete',source_sha256=sha(Path(__file__)),
        scope='Evaluation-matched absolute legitimate counts for descriptive tradeoff; author operating point retained separately',
        changed_detection_thresholds_in_experiments=False,
        PRC='Author boolean decision has29/100legitimate acceptance; no scalar threshold sweep is available in the collected output',
        evidence={str(p):sha(p)for p in [genuine/'complete.json',genuine/'rows.json',attack/'TR/complete.json',attack/'protocol.json']},rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
