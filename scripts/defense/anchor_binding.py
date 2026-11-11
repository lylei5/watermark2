"""Conditional Gaussian anchor carrier; experimental defensive embedding only."""
import hashlib
import hmac
import json
from pathlib import Path
import numpy as np
from scipy.fft import dctn,idctn
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms


def identity(root,task):
    s=json.loads((Path(root)/f'{task:03}.json').read_text())
    key,nonce,message,bkey=[bytes.fromhex(s[k]) for k in ['key','nonce','message','binding_key']]
    cipher=np.unpackbits(np.frombuffer(Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(message*64),dtype=np.uint8))
    return s,cipher,np.unpackbits(np.frombuffer(message,dtype=np.uint8)),bkey


def order(key,n):
    domain=b'sembind/permutation/' if n==16384 else b'anchor/permutation/'
    tags=[hmac.new(key,domain+i.to_bytes(4,'big'),hashlib.sha256).digest() for i in range(n)]
    return np.asarray(sorted(range(n),key=tags.__getitem__))


def partition(arm):
    if arm=='anchor_random':
        low=np.sort(np.random.default_rng(2026111201).permutation(4096)[:2048])
    else:
        low=np.asarray(sorted(range(4096),key=lambda j:(j//64+j%64,j))[:2048])
    a=np.concatenate([low+4096*c for c in range(4)])
    b=np.setdiff1d(np.arange(16384),a)
    assert len(a)==len(b)==8192
    return a,b


def embed(aux_latent,code,root,task,arm):
    s,cipher,_,key=identity(root,task)
    n=16384 if arm=='full' else 8192
    amplitude=np.abs(np.random.default_rng(s['amplitude_seed']).standard_normal(16384))[:n]
    mask=np.tile(np.asarray(code,dtype=np.uint8),n//1024)[order(key,n)]
    symbols=amplitude*(2*(cipher[:n]^mask).astype(np.float64)-1)
    if arm=='full':return symbols.reshape(1,4,64,64).astype(np.float32)
    a,b=partition(arm)
    coordinates=dctn(aux_latent,axes=(-2,-1),norm='ortho').reshape(-1).astype(np.float64)
    if arm=='independent_low':
        coordinates[a]=np.random.default_rng(s['independent_anchor_seed']).standard_normal(8192)
    coordinates[b]=symbols
    return idctn(coordinates.reshape(1,4,64,64),axes=(-2,-1),norm='ortho').astype(np.float32)


def decode(latent,code,root,task,arm):
    _,cipher,message,key=identity(root,task)
    if arm=='full':coordinates=np.asarray(latent).reshape(-1)
    else:
        _,b=partition(arm)
        coordinates=dctn(latent,axes=(-2,-1),norm='ortho').reshape(-1)[b]
    n=len(coordinates);perm=order(key,n)
    mask=np.tile(np.asarray(code,dtype=np.uint8),n//1024)[perm]
    stream=cipher[:n]^np.tile(message,n//256)
    bits=(coordinates>0).astype(np.uint8)^mask^stream
    votes=bits.reshape(n//256,256).sum(0)
    recovered=(votes>(n//256)/2).astype(np.uint8)
    return float(np.mean(recovered==message))
