"""Sample FID with standard pool3 features and an exact low-rank covariance trace."""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from prepare import RUN,ROOT,read,put,hashfile


def model():
    sys.path.insert(0,str(RUN/'assets/lib'))
    from pytorch_fid import inception
    weights=RUN/'assets/fid.pth'
    assert hashfile(weights)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
    inception.load_state_dict_from_url=lambda *a,**k:torch.load(weights,map_location='cpu',weights_only=True)
    return inception.InceptionV3().eval()


def feature(net,paths):
    import torchvision.transforms.functional as TF
    vec=[]
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as im:
                vec.append(net(TF.to_tensor(im.convert('RGB')).unsqueeze(0))[0].flatten().numpy())
    return np.stack(vec)


def distance(a,b):
    a,b=a.astype('float64'),b.astype('float64')
    ma,mb=a.mean(0),b.mean(0)
    x,y=(a-ma)/np.sqrt(len(a)-1),(b-mb)/np.sqrt(len(b)-1)
    cross=np.linalg.svd(x@y.T,compute_uv=False).sum()
    return float(max(0,((ma-mb)**2).sum()+(x*x).sum()+(y*y).sum()-2*cross))


def main(stage):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(4)
    net=model()
    if stage=='reference':
        import hashlib
        folder=ROOT/'datasets/coco_test2017_official/test2017'
        paths=sorted(folder.glob('*.jpg'),key=lambda p:hashlib.sha256(('rgb-fid-v1/'+p.name).encode()).digest())[:256]
        assert len(paths)==256
        np.save(RUN/'assets/real_pool3.npy',feature(net,paths))
        put(RUN/'real_reference.json',[dict(name=p.name,sha256=hashfile(p)) for p in paths])
        return
    sources=read(RUN/'test_sources.json')
    real=np.load(RUN/'assets/real_pool3.npy')
    assert real.shape==(256,2048)
    arms=['baseline']+[c['name'] for c in read(RUN/'test_config.json')]
    out={}
    for arm in arms:
        paths=[RUN/'test'/f"{s['id']:03d}"/(arm+'.png') for s in sources]
        feats=feature(net,paths)
        np.save(RUN/'assets'/f'{arm}_pool3.npy',feats)
        if arm=='baseline':
            base=feats.copy()
        out[arm]=dict(vs_real_mixed=distance(feats,real),
                      vs_real_coco=distance(feats[[s['dataset']=='COCO' for s in sources]],real),
                      vs_paired_baseline=distance(feats,base))
    put(RUN/'fid.json',out)


if __name__=='__main__':
    main(sys.argv[1])
