"""GIFT 2D reference solver.
Separates a constitutive source-to-metric map, derived differential geometry,
and pure versus forced covariant motion.
"""
import numpy as np

def gaussian_source(X,Y,amp=1.0,spread=1.2,center=(0.0,0.0)):
    px,py=center; r2=(X-px)**2+(Y-py)**2
    phi=amp*np.exp(-r2/(2*spread**2))
    return phi, phi*(-(Y-py)/spread), phi*(1-r2/(2*spread**2))

def metric_from_source(T00,T01,T11,kappa=0.35):
    """Reference constitutive law g=exp(kappa H); H symmetric => g is SPD."""
    a=T00+T11; b=T01; d=T00-T11
    tr2=.5*(a+d); q=.5*(a-d); s=np.sqrt(q*q+b*b)
    e=np.exp(kappa*tr2); ks=kappa*s; c=np.cosh(ks)
    fac=np.empty_like(s,dtype=np.result_type(s,float))
    mask=s>1e-12
    np.divide(np.sinh(ks),s,out=fac,where=mask)
    fac[~mask]=kappa+(kappa**3*s[~mask]**2)/6.0
    return e*(c+fac*q), e*(fac*b), e*(c-fac*q)

def inverse_metric(g00,g01,g11):
    det=g00*g11-g01*g01
    if np.any(~np.isfinite(det)) or np.any(det<=0):
        raise ValueError("Metric must be finite and positive definite")
    return g11/det,-g01/det,g00/det,det

def geometry(g00,g01,g11,dx,dy):
    """Return Levi-Civita symbols, Ricci tensor, scalar curvature, determinant."""
    gi00,gi01,gi11,det=inverse_metric(g00,g01,g11)
    def deriv(f):
        fy,fx=np.gradient(f,dy,dx,edge_order=2); return fx,fy
    g00x,g00y=deriv(g00); g01x,g01y=deriv(g01); g11x,g11y=deriv(g11)
    G=np.zeros((2,2,2)+g00.shape)
    G[0,0,0]=.5*(gi00*g00x+gi01*(2*g01x-g00y))
    G[0,0,1]=G[0,1,0]=.5*(gi00*g00y+gi01*g11x)
    G[0,1,1]=.5*(gi00*(2*g01y-g11x)+gi01*g11y)
    G[1,0,0]=.5*(gi01*g00x+gi11*(2*g01x-g00y))
    G[1,0,1]=G[1,1,0]=.5*(gi01*g00y+gi11*g11x)
    G[1,1,1]=.5*(gi01*(2*g01y-g11x)+gi11*g11y)

    dG=np.zeros((2,)+G.shape)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gx,gy=deriv(G[k,i,j]); dG[0,k,i,j]=gx; dG[1,k,i,j]=gy

    Ric=np.zeros((2,2)+g00.shape)
    for i in range(2):
        for j in range(2):
            r=np.zeros_like(g00,dtype=float)
            for k in range(2):
                r += dG[k,k,i,j]-dG[j,k,i,k]
                for l in range(2):
                    r += G[k,i,j]*G[l,k,l]-G[l,i,k]*G[k,j,l]
            Ric[i,j]=r

    # Discrete Ricci need not be exactly symmetric, so contract both off-diagonal terms.
    R=gi00*Ric[0,0]+gi01*(Ric[0,1]+Ric[1,0])+gi11*Ric[1,1]
    return G,Ric,R,det

def bilinear(field,x,y,xgrid,ygrid):
    x=np.clip(x,xgrid[0],xgrid[-1]); y=np.clip(y,ygrid[0],ygrid[-1])
    ix=max(0,min(np.searchsorted(xgrid,x)-1,len(xgrid)-2))
    iy=max(0,min(np.searchsorted(ygrid,y)-1,len(ygrid)-2))
    tx=(x-xgrid[ix])/(xgrid[ix+1]-xgrid[ix]); ty=(y-ygrid[iy])/(ygrid[iy+1]-ygrid[iy])
    return ((1-tx)*(1-ty)*field[iy,ix]+tx*(1-ty)*field[iy,ix+1]
            +(1-tx)*ty*field[iy+1,ix]+tx*ty*field[iy+1,ix+1])

def covariant_acceleration(pos,vel,G,xgrid,ygrid,force=None,damping=0.0):
    """Coordinate acceleration for geodesic (force=None,damping=0) or forced motion."""
    Gam=np.empty((2,2,2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                Gam[k,i,j]=bilinear(G[k,i,j],*pos,xgrid,ygrid)
    a=-np.einsum('kij,i,j->k',Gam,vel,vel)
    if force is not None: a=a+np.asarray(force,dtype=float)
    return a-damping*vel

if __name__=="__main__":
    import matplotlib.pyplot as plt
    x=np.linspace(-5,5,101); y=np.linspace(-5,5,101); X,Y=np.meshgrid(x,y)
    g=metric_from_source(*gaussian_source(X,Y,amp=1.5))
    _,_,R,_=geometry(*g,x[1]-x[0],y[1]-y[0])
    fig,ax=plt.subplots(); im=ax.contourf(X,Y,R,30)
    fig.colorbar(im,ax=ax,label="Scalar curvature R")
    ax.set(xlabel="Interaction coordinate I1",ylabel="Interaction coordinate I2",
           title="GIFT: scalar curvature of the interaction metric")
    plt.show()
