"""Full original margin projection; no new encoder strength or direction code."""
import numpy as np
from . import projection_margin as carrier

def correction(clean,book,message):
    projected,evidence=carrier.write(clean,book,message)
    u=projected-np.asarray(clean,dtype=np.float32)
    return u,projected,evidence

def numerical_equivalence(a,b,atol=2e-5,rtol=2e-4):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    error=float(np.max(np.abs(a-b)));tol=atol*max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b))))+rtol*float(np.max(np.abs(b)))
    return {'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),'maxabs':error,'tolerance':tol,
            'pass':bool(np.isfinite(a).all() and np.isfinite(b).all() and error<=tol),'atol':atol,'rtol':rtol}
