"""Algebraic boundary checks for the defense's error/erasure codec on249."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

ROOT=Path('/data0/liyanlei/watermark_research')
assert ROOT.is_dir()
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(ROOT/'tools/transport_recovery_dependencies'))
from reedsolo import RSCodec,ReedSolomonError

codec=RSCodec(239)
rows=[]
for errors,erasures in [(0,0),(119,0),(0,239),(20,199),(119,1),(120,0),(20,200)]:
    key=secrets.token_bytes(16)
    received=bytearray(codec.encode(key))
    erased=list(range(erasures))
    for i in erased:received[i]=0
    for i in range(erasures,erasures+errors):received[i]^=1
    try:
        recovered,_,_=codec.decode(received,erase_pos=erased)
        exact=bytes(recovered)==key
    except ReedSolomonError:
        exact=False
    inside=2*errors+erasures<=239
    assert exact if inside else not exact
    rows.append(dict(errors=errors,erasures=erasures,inside_guaranteed_region=inside,exact=exact))
out=ROOT/'runs/spatial_witness_recovery_20261114_v2/codec_audit.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='complete',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rows=rows,scope='Numerical code-channel checks, not image attacks or a visual FPR experiment.'),indent=2)+'\n')
print(out.read_text())
