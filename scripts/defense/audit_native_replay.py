"""Measure PRC-S replay equality under identical frozen inputs and numeric code."""
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name=='posix'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    old=root/'runs/sembind_prc_native_20261113_v2'
    new=root/'runs/sembind_prc_native_20261113_v3'
    plans=[json.loads((p/'protocol.json').read_text()) for p in [old,new]]
    for key in ['private','private_key_sha256','binding_key_sha256','sources','local_sha256','generation','inversion']:
        assert plans[0][key]==plans[1][key],key
    comparisons=[]
    for shard in [1,2]:
        files=[p/f'native_{shard}/complete.json' for p in [old,new]]
        records=[json.loads(p.read_text()) for p in files]
        assert all(x['status']=='complete' for x in records)
        assert len(records[0]['rows'])==len(records[1]['rows'])==8
        differences=[]
        for first,second in zip(records[0]['rows'],records[1]['rows']):
            assert (first['task'],first['image_kind'],first['detector'])==(second['task'],second['image_kind'],second['detector'])
            changed={k:[v,second[k]] for k,v in first.items() if k!='image' and v!=second[k]}
            arrays={}
            with np.load(old/first['latent']) as a,np.load(new/second['latent']) as b:
                assert a.files==b.files
                for key in a.files:
                    arrays[key]=dict(exact_equal=bool(np.array_equal(a[key],b[key])),
                        max_abs_difference=float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))))
            if changed:
                differences.append(dict(task=first['task'],kind=first['image_kind'],detector=first['detector'],changed=changed,arrays=arrays))
            for row,directory in [(first,old),(second,new)]:
                assert sha(row['image'])==row['image_sha256']
                assert sha(directory/row['latent'])==row['latent_sha256']
        comparisons.append(dict(shard=shard,decisions=8,exact_equal=not differences,differences=differences,
            old_complete_sha256=sha(files[0]),new_complete_sha256=sha(files[1])))
    result=dict(status='complete',source_run=str(old),replay_run=str(new),comparisons=comparisons,
        scope='4 sources previously completed on the same GPUs,8 generated images,12 inverted-image artifacts,16 decisions; not cross-device or author-environment equivalence',
        script_sha256=sha(__file__),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path=root/'runs/three_baselines_20261113/native_replay_audit.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
