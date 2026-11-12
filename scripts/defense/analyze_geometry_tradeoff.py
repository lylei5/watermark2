"""Analytic semantic-distance response of the frozen rotation mechanism.

Gaussian-channel mathematics, not image perturbations or attack development.
No parameter fit or experiment scores are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();alpha=2.;grid=np.linspace(0,np.sqrt(2),401)
    rows=[];crossings=[];checks=[]
    for sigma in [0.,.4,1.]:
        rho=1/np.sqrt(1+sigma*sigma)
        p0=.5+np.arcsin(rho)/np.pi
        if sigma==0:
            k=np.arange(1,2048,2)
            coefficients=4/(np.pi**2*k*k)
        else:
            k=np.arange(1,128,2)
            coefficients=np.array([4/np.pi**2*quad(lambda delta:np.arcsin(rho*np.cos(delta)),0,np.pi/2,
                weight='cos',wvar=int(odd),epsabs=1e-12)[0] for odd in k])
            assert np.min(coefficients)>-1e-12
            coefficients=np.maximum(coefficients,0)
            assert abs(.5+coefficients.sum()-p0)<1e-10
        def continuous(distance):
            if distance==0:return float(p0)
            return float(.5+coefficients@np.exp(-.5*(k*alpha*distance)**2))
        def quantized(distance):
            mismatch=np.arccos(np.clip(1-distance*distance/2,-1,1))/np.pi
            return float(p0-(2*p0-1)*mismatch)
        if sigma>0:
            for distance in [.01,.1,.3,.5,1.,np.sqrt(2)]:
                numerical=.5+quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*np.arcsin(rho*np.cos(alpha*distance*x))/np.pi,
                    -10,10,epsabs=1e-11,limit=300)[0]
                discrepancy=abs(numerical-continuous(distance))
                assert discrepancy<1e-8
                checks.append(dict(sigma=sigma,distance=float(distance),absolute_error=discrepancy))
        values=[]
        for distance in grid:
            smooth=continuous(distance);binary=quantized(distance)
            rows.append(dict(sigma=sigma,distance=float(distance),continuous=smooth,quantized=binary,
                             gain=smooth-binary))
            values.append(smooth-binary)
        roots=[]
        for i in range(1,len(grid)-1):
            if values[i]*values[i+1]<0:
                roots.append(float(brentq(lambda d:continuous(d)-quantized(d),grid[i],grid[i+1])))
        crossings.append(dict(sigma=sigma,nonzero_crossings=roots,
            small_distance_continuous_gain=float(continuous(.01)-quantized(.01)),
            distance_half_continuous_gain=float(continuous(.5)-quantized(.5))))
    result=dict(completed=True,alpha=alpha,rows=rows,crossings=crossings,independent_quadrature_checks=checks,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Ideal independent isotropic Gaussian measurement channel; fixed semantic displacement independent of payload/channel. Frequency-averaged sign correctness, not majority BA or image-defense results.',
        derivation='p=1/2+sum over positive odd k of a_k exp(-k^2 alpha^2 d^2/2); a_k=(4/pi^2) integral_0^(pi/2) arcsin(rho cos t) cos(kt)dt. Quantized angle changes by pi with probability arccos(1-d^2/2)/pi.',
        limitations='Positive Gaussian noise gives a local second-order penalty only. Nonlocal gains need not have the same sign; higher correctness at large semantic distance can weaken authentication. Frozen alpha unchanged, no fit or image observations used.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(alpha=alpha,crossings=crossings),indent=2))


if __name__=='__main__':main()
