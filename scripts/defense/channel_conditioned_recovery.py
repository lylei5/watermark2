"""Authenticated recovery conditioned on independently measured inversion changes."""
import base64
import hashlib
import hmac

import numpy as np

from instance_coset_recovery import canonical


def encode_array(value):
    array=np.asarray(value,dtype='<f2')
    return dict(shape=list(array.shape),data=base64.b64encode(array.tobytes()).decode())


def decode_array(value):
    return np.frombuffer(base64.b64decode(value['data']),dtype='<f2').reshape(value['shape']).astype(np.float32)


def channel_readout(z):
    z=np.asarray(z,dtype=np.float64).reshape(4,64,64)
    def pool(x):return x.reshape(4,2,32,2,32).mean(axis=(2,4)).reshape(-1)
    return np.concatenate([pool(z),np.sqrt(pool(z*z))]).astype(np.float32)


def make_normalizer(reference,views,channel_reference,channel_views,method):
    delta_f=views-reference
    if method=='permuted_channel':
        delta_f=np.roll(delta_f,2,axis=0)
    if method=='content_self_fit':
        base=reference
        design=views-reference
        scale=np.ones(512,dtype=np.float32)
    else:
        base=channel_reference
        design=channel_views-channel_reference
        scale=np.maximum(np.sqrt(np.mean(design**2,axis=0)),1e-5)
    return dict(method=method,base=encode_array(base),design=encode_array(design),
                scale=encode_array(scale),content_responses=encode_array(delta_f),ridge_fraction=.1,l1_limit=2.)


def correct(query,readout,normalizer):
    if normalizer is None:return query
    base,design,scale,response=[decode_array(normalizer[k])for k in ['base','design','scale','content_responses']]
    observed=query if normalizer['method']=='content_self_fit' else readout
    design=design/scale
    delta=(observed-base)/scale
    gram=design@design.T
    penalty=max(float(np.trace(gram))/len(gram)*normalizer['ridge_fraction'],1e-8)
    coefficients=np.linalg.solve(gram+penalty*np.eye(len(gram)),design@delta)
    coefficients*=min(1.,normalizer['l1_limit']/max(float(np.abs(coefficients).sum()),1e-8))
    corrected=query-coefficients@response
    return (corrected/max(float(np.linalg.norm(corrected)),1e-12)).astype(np.float32)


class ChannelConditionedRecovery:
    def __init__(self,codec):self.codec=codec

    def enroll(self,identity,reference,views,families,h0,hviews,material,provider,method,key):
        selectors={'clean_margin':'clean_margin','orbit_margin':'orbit_margin','tangent8':'tangent8_margin'}
        normalizer=None if method in selectors else make_normalizer(reference,views,h0,hviews,method)
        f0=correct(reference,h0,normalizer)
        fv=np.stack([correct(f,h,normalizer)for f,h in zip(views,hviews)])
        inner,_=self.codec.enroll(identity,f0,fv,families,material,provider,
                                  selector=selectors.get(method,'clean_margin'),mode='fixed',key=key)
        record=dict(version=1,normalizer=normalizer,inner=inner)
        record['mac']=hmac.new(provider,b'channel-record/'+canonical(record),hashlib.sha256).hexdigest()
        return record

    def recover(self,record,query,readout,provider):
        unsigned={k:v for k,v in record.items()if k!='mac'}
        mac=hmac.new(provider,b'channel-record/'+canonical(unsigned),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac,record['mac']):return None,dict(reason='record_authentication',attempts=0)
        feature=correct(query,readout,record['normalizer'])
        return self.codec.recover(record['inner'],feature,provider)

    def controls(self,record,query,readout,reference,reference_readout):
        q=correct(query,readout,record['normalizer'])
        f=correct(reference,reference_readout,record['normalizer'])
        distance=self.codec.recovery_distance(record['inner'],q,f)
        transform=record['inner']['metadata'].get('feature_transform')
        continuous=1-self.codec.transform_feature(q,transform)@self.codec.transform_feature(f,transform)
        return distance,float(np.clip(continuous,0,2))
