"""Mandatory observable-band constraints for actual instance-key recovery.

Layered authentication and error/erasure coding have established prior art.
This prototype tests a query-observability rule without asserting novelty.
"""
import hashlib
import hmac
import secrets

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from reedsolo import RSCodec, ReedSolomonError

from spatial_witness_recovery import canonical, pack, unpack, gray

CODEC=RSCodec(239)
NAMES=['coarse','middle','fine']
MIN_OBSERVED=64
RELATIVE_MARGIN=.1
ABSOLUTE_MARGIN=.5
_last_decoding=None


def bands(image):
    g=gray(image).astype(np.float32)
    smooth=[cv2.GaussianBlur(g,(0,0),sigma) for sigma in [2.,4.,8.]]
    return [smooth[2],smooth[1]-smooth[2],smooth[0]-smooth[1]]


def measurements(band,pairs):
    return band[pairs[:,:,0,1],pairs[:,:,0,0]]-band[pairs[:,:,1,1],pairs[:,:,1,0]]


def enroll(identity,image,payload,provider,key=None,layout='multiscale'):
    assert layout=='multiscale'
    key=secrets.token_bytes(16) if key is None else key
    assert len(key)==16
    rng=np.random.default_rng(int.from_bytes(secrets.token_bytes(16),'big'))
    xx,yy=np.meshgrid(np.linspace(28,484,17),np.linspace(28,484,15))
    centers=np.round(np.stack([xx.ravel(),yy.ravel()],axis=1)).astype(int)+rng.integers(-4,5,(255,2))
    word=np.frombuffer(bytes(CODEC.encode(key)),dtype=np.uint8)
    layers=[]
    for name,band in zip(NAMES,bands(image)):
        pairs=[]
        for center in centers:
            pool=center[None,None,:]+rng.integers(-12,13,(512,2,2))
            diff=band[pool[:,0,1],pool[:,0,0]]-band[pool[:,1,1],pool[:,1,0]]
            valid=np.linalg.norm(pool[:,0]-pool[:,1],axis=1)>=4
            chosen=np.argsort(-np.where(valid,np.abs(diff),-1),kind='stable')[:8]
            pairs.append(pool[chosen])
        pairs=np.asarray(pairs,dtype='<u2')
        diff=measurements(band,pairs)
        symbols=np.packbits((diff>=0).astype(np.uint8),axis=1).reshape(-1)
        layers.append(dict(name=name,pairs=pack(pairs),reference_margin=pack(np.median(np.abs(diff),axis=1).astype('<f4')),
            offset=pack(symbols^word)))
    meta=dict(version=1,id=identity,codec='RS(255,16),GF256,239parity',bands=layers,
        sigma=[2.,4.,8.],minimum_observed=MIN_OBSERVED,relative_margin=RELATIVE_MARGIN,
        absolute_margin=ABSOLUTE_MARGIN,confirmation=hashlib.sha256(b'observable/'+identity.encode()+key).hexdigest())
    nonce=secrets.token_bytes(12)
    wrapping=hmac.new(provider,b'observable-wrap/'+identity.encode()+key,hashlib.sha256).digest()
    record=dict(metadata=meta,nonce=nonce.hex(),wrapped=__import__('base64').b64encode(AESGCM(wrapping).encrypt(nonce,payload,canonical(meta))).decode())
    record['mac']=hmac.new(provider,canonical(record),hashlib.sha256).hexdigest()
    return record,key


def decode_layer(layer,band,meta):
    pairs=unpack(layer['pairs']).astype(np.int32)
    diff=measurements(band,pairs)
    strength=np.median(np.abs(diff),axis=1)
    floor=np.maximum(meta['absolute_margin'],meta['relative_margin']*unpack(layer['reference_margin']))
    seen=strength>=floor
    detail=dict(name=layer['name'],observed_symbols=int(seen.sum()),active=bool(seen.sum()>=meta['minimum_observed']),
        median_relative_margin=float(np.median(strength/np.maximum(unpack(layer['reference_margin']),1e-6))))
    if not detail['active']:return None,dict(detail,reason='insufficient_observability')
    symbols=np.packbits((diff>=0).astype(np.uint8),axis=1).reshape(-1)
    received=symbols^unpack(layer['offset']);erasures=np.flatnonzero(~seen).tolist()
    try:
        message,_,_=CODEC.decode(received.tobytes(),erase_pos=erasures)
    except ReedSolomonError:
        return None,dict(detail,reason='decoding_failure')
    key=bytes(message)
    if len(key)!=16 or not hmac.compare_digest(meta['confirmation'],hashlib.sha256(b'observable/'+meta['id'].encode()+key).hexdigest()):
        return None,dict(detail,reason='key_confirmation_failure')
    errors=int(np.count_nonzero((np.frombuffer(bytes(CODEC.encode(key)),dtype=np.uint8)!=received)&seen))
    budget=2*errors+len(erasures)
    if budget>239:return None,dict(detail,reason='outside_code_region',symbol_errors=errors)
    return key,dict(detail,reason='recovered',symbol_errors=errors,erasures=len(erasures),code_budget=budget)


def recover(record,image,query_anchors,provider,mode):
    global _last_decoding
    # The inherited harness uses identity mode only for authentication rejection checks.
    assert mode in ['identity','coarse','any_scale','all_scales','observable_scales']
    checked={k:v for k,v in record.items() if k!='mac'}
    if not hmac.compare_digest(record['mac'],hmac.new(provider,canonical(checked),hashlib.sha256).hexdigest()):
        return None,dict(reason='record_authentication')
    meta=record['metadata']
    cache_id=(record['mac'],hashlib.sha256(image.convert('RGB').tobytes()).digest())
    reused=_last_decoding is not None and _last_decoding[0]==cache_id
    if reused:
        _,keys,details=_last_decoding
    else:
        keys=[];details=[]
        for layer,band in zip(meta['bands'],bands(image)):
            key,detail=decode_layer(layer,band,meta);keys.append(key);details.append(detail)
        # Four access rules share identical observations and decoded candidates.
        # Keep only the immediately preceding authenticated record/query in memory.
        _last_decoding=(cache_id,keys,details)
    required=([0] if mode in ['identity','coarse'] else list(range(3)) if mode=='all_scales' else
        [i for i,d in enumerate(details) if d['active']] if mode=='any_scale' else [0]+[i for i in [1,2] if details[i]['active']])
    success=any(keys[i] is not None for i in required) if mode=='any_scale' else all(keys[i] is not None for i in required)
    detail=dict(reason='recovered' if success else 'required_evidence_failure',required_bands=required,bands=details,decode_reused=reused)
    if not success:return None,detail
    recovered=[keys[i] for i in required if keys[i] is not None]
    assert recovered and all(k==recovered[0] for k in recovered)
    key=recovered[0]
    wrapping=hmac.new(provider,b'observable-wrap/'+meta['id'].encode()+key,hashlib.sha256).digest()
    payload=AESGCM(wrapping).decrypt(bytes.fromhex(record['nonce']),__import__('base64').b64decode(record['wrapped']),canonical(meta))
    return (key,payload),detail
