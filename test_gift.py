import numpy as np
from GIFT import geometry,metric_from_source,gaussian_source

def test_flat_metric_zero_curvature():
    n=41; x=np.linspace(-2,2,n); z=np.zeros((n,n)); o=np.ones((n,n))
    _,_,R,det=geometry(o,z,o,x[1]-x[0],x[1]-x[0])
    assert np.max(np.abs(R))<1e-10 and np.min(det)>0

def test_metric_positive_definite():
    x=np.linspace(-3,3,51); X,Y=np.meshgrid(x,x)
    g=metric_from_source(*gaussian_source(X,Y,amp=3.0)); det=g[0]*g[2]-g[1]**2
    assert np.min(g[0])>0 and np.min(det)>0

def test_constant_conformal_metric_flat():
    n=31; x=np.linspace(-1,1,n); o=np.ones((n,n))*2.5; z=np.zeros((n,n))
    _,_,R,_=geometry(o,z,o,x[1]-x[0],x[1]-x[0])
    assert np.max(np.abs(R))<1e-10
