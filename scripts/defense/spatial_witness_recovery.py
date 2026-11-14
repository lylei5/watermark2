"""Reference-side held-out spatial evidence and exact error/erasure key recovery.

Research candidate, not a proven secure fuzzy extractor. Registration and robust
hashing are existing techniques; novelty and visual separation require evidence.
"""
import base64
import hashlib
import hmac
import json
import secrets

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from reedsolo import RSCodec, ReedSolomonError

CODEC = RSCodec(239)


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':')).encode()


def pack(a):
    a=np.ascontiguousarray(a)
    return dict(dtype=a.dtype.str,shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode())


def unpack(d):
    return np.frombuffer(base64.b64decode(d['data']),dtype=d['dtype']).reshape(d['shape'])


def gray(image):
    return cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)


def anchors(image):
    g=gray(image)
    points,desc=cv2.SIFT_create(nfeatures=1800).detectAndCompute(g,None)
    if desc is None:
        return np.empty((0,3),dtype=np.float32),np.empty((0,128),dtype=np.float32)
    return np.asarray([[*p.pt,p.size] for p in points],dtype=np.float32),desc


def enroll(identity,image,payload,provider,key=None):
    key=secrets.token_bytes(16) if key is None else key
    assert len(key)==16
    rng=np.random.default_rng(int.from_bytes(secrets.token_bytes(16),'big'))
    g=cv2.GaussianBlur(gray(image).astype(np.float32),(0,0),2.)
    # Witnesses occupy the right-hand region. Reference registration anchors
    # for heldout mode have descriptor-support bounds entirely left of x224.
    centers=np.stack([rng.integers(312,480,255),rng.integers(32,480,255)],axis=1)
    pairs=[];bits=[]
    for center in centers:
        pool=center[None,None,:]+rng.integers(-12,13,(512,2,2))
        values=g[pool[:,:,1],pool[:,:,0]]
        diff=values[:,0]-values[:,1]
        valid=np.linalg.norm(pool[:,0]-pool[:,1],axis=1)>=8
        chosen=np.argsort(-np.where(valid,np.abs(diff),-1),kind='stable')[:8]
        pairs.append(pool[chosen]);bits.append(diff[chosen]>=0)
    points,desc=anchors(image)
    symbols=np.packbits(np.asarray(bits,dtype=np.uint8),axis=1).reshape(-1)
    word=np.frombuffer(bytes(CODEC.encode(key)),dtype=np.uint8)
    assert len(word)==255
    meta=dict(version=1,id=identity,codec='RS(255,16),GF256,239parity',
        pairs=pack(np.asarray(pairs,dtype='<u2')),anchors=pack(points.astype('<f4')),
        descriptors=pack(desc.astype('<f4')),offset=pack(symbols^word),
        confirmation=hashlib.sha256(b'witness/'+identity.encode()+key).hexdigest())
    nonce=secrets.token_bytes(12)
    wrapping=hmac.new(provider,b'wrap/'+identity.encode()+key,hashlib.sha256).digest()
    record=dict(metadata=meta,nonce=nonce.hex(),wrapped=base64.b64encode(AESGCM(wrapping).encrypt(nonce,payload,canonical(meta))).decode())
    record['mac']=hmac.new(provider,canonical(record),hashlib.sha256).hexdigest()
    return record,key


def recover(record,image,query_anchors,provider,mode):
    assert mode in ['identity','registered','heldout']
    checked={k:v for k,v in record.items() if k!='mac'}
    if not hmac.compare_digest(record['mac'],hmac.new(provider,canonical(checked),hashlib.sha256).hexdigest()):
        return None,dict(reason='record_authentication')
    meta=record['metadata']; h=np.eye(3);inliers=0;matches=0
    if mode!='identity':
        points,desc=unpack(meta['anchors']),unpack(meta['descriptors'])
        if mode=='heldout':
            valid=points[:,0]+np.maximum(16,6*points[:,2])<224
            points,desc=points[valid],desc[valid]
        qpoints,qdesc=query_anchors
        if len(points)<8 or len(qpoints)<8:
            return None,dict(reason='insufficient_anchors',anchors=len(points))
        knn=cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc,qdesc,k=2)
        good=[a for a,b in knn if a.distance<.75*b.distance]
        matches=len(good)
        if matches<8:
            return None,dict(reason='insufficient_matches',matches=matches)
        src=np.float32([points[m.queryIdx,:2] for m in good]);dst=np.float32([qpoints[m.trainIdx,:2] for m in good])
        cv2.setRNGSeed(0)
        h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=2000,confidence=.995)
        inliers=0 if mask is None else int(mask.sum())
        if h is None or not np.isfinite(h).all() or inliers<8:
            return None,dict(reason='registration_failure',matches=matches,inliers=inliers)
    g=cv2.warpPerspective(gray(image).astype(np.float32),h,(512,512),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    g=cv2.GaussianBlur(g,(0,0),2.)
    pairs=unpack(meta['pairs']).astype(np.int32)
    mapped=cv2.perspectiveTransform(pairs.reshape(-1,1,2).astype(np.float32),h).reshape(255,8,2,2)
    seen=((mapped>=8)&(mapped<=503)).all(axis=(1,2,3))
    erasures=np.flatnonzero(~seen).tolist()
    detail=dict(matches=matches,inliers=inliers,erasures=len(erasures),observed_symbols=int(seen.sum()))
    if len(erasures)>239:
        return None,dict(detail,reason='insufficient_witnesses')
    observed=np.packbits((g[pairs[:,:,0,1],pairs[:,:,0,0]]>=g[pairs[:,:,1,1],pairs[:,:,1,0]]).astype(np.uint8),axis=1).reshape(-1)
    received=observed^unpack(meta['offset'])
    try:
        message,corrected,errata=CODEC.decode(received.tobytes(),erase_pos=erasures)
    except ReedSolomonError:
        return None,dict(detail,reason='decoding_failure')
    key=bytes(message)
    if len(key)!=16 or not hmac.compare_digest(meta['confirmation'],hashlib.sha256(b'witness/'+meta['id'].encode()+key).hexdigest()):
        return None,dict(detail,reason='key_confirmation_failure')
    word=np.frombuffer(bytes(CODEC.encode(key)),dtype=np.uint8)
    errors=int(np.count_nonzero((word!=received)&seen))
    if 2*errors+len(erasures)>239:
        return None,dict(detail,reason='outside_code_region',symbol_errors=errors)
    wrapping=hmac.new(provider,b'wrap/'+meta['id'].encode()+key,hashlib.sha256).digest()
    payload=AESGCM(wrapping).decrypt(bytes.fromhex(record['nonce']),base64.b64decode(record['wrapped']),canonical(meta))
    return (key,payload),dict(detail,reason='recovered',symbol_errors=errors,code_budget=2*errors+len(erasures))
