"""Exact finite-cylinder / finite-rectangle ray tests and safe spatial culling."""
import numpy as np
from scipy.spatial import cKDTree


def cylinder_intersection(points, directions, tower, radius, zlo, zhi):
    """Vectorized reference. Returns first positive distance and surface kind."""
    p,d = np.broadcast_arrays(np.asarray(points,float),np.asarray(directions,float))
    shape=p.shape[:-1]
    best=np.full(shape,np.inf)
    kind=np.zeros(shape,np.int32)
    if radius<=0:
        return best,kind
    x,y=p[...,0]-tower[0],p[...,1]-tower[1]
    a=d[...,0]**2+d[...,1]**2
    b=x*d[...,0]+y*d[...,1]
    disc=b*b-a*(x*x+y*y-radius*radius)
    for sign in (-1,1):
        t=(-b+sign*np.sqrt(np.maximum(0,disc)))/np.where(a>1e-20,a,1)
        z=p[...,2]+t*d[...,2]
        ok=(a>1e-20)&(disc>=0)&(t>1e-8)&(z>=zlo-1e-9)&(z<=zhi+1e-9)&(t<best)
        best=np.where(ok,t,best)
        kind=np.where(ok,1,kind)
    for z in (zlo,zhi):
        t=(z-p[...,2])/np.where(np.abs(d[...,2])>1e-15,d[...,2],1)
        xx,yy=x+t*d[...,0],y+t*d[...,1]
        ok=(np.abs(d[...,2])>1e-15)&(t>1e-8)&(xx*xx+yy*yy<=radius*radius+1e-9)&(t<best-1e-9)
        best=np.where(ok,t,best)
        kind=np.where(ok,2,kind)
    return best,kind


def rectangle_hits(points,directions,center,normal,u,v,width,height):
    delta=points-center
    denom=directions@normal
    t=-(delta@normal)/np.where(np.abs(denom)>1e-13,denom,1.)
    rel=delta+t[...,None]*directions
    return (np.abs(denom)>1e-13)&(t>1e-8)&(np.abs(rel@u)<=width/2+1e-9)&(np.abs(rel@v)<=height/2+1e-9)


def neighbor_lists(field,sun,target,v,cone,all_pairs=False):
    """Conservative candidates, accounting for complete mirror extents and cone.

    An upward ray cannot hit another mirror after exceeding the highest mirror
    edge. This gives a finite distance bound, then sphere/cone culling tightens
    it. No arbitrary nearest-K truncation and no assumed fixed neighbor radius.
    """
    n=len(field)
    # Exhaustive mode is inexpensive for unit tests and prevents a geometric
    # culling approximation from hiding a missed blocker in small designs.
    if n < 128: all_pairs=True
    if all_pairs:
        ids=np.array([j for i in range(n) for j in range(n) if i!=j],np.int64)
        ptr=np.arange(n+1,dtype=np.int64)*(n-1)
        return (ptr,ids),(ptr,ids)
    extent=np.abs(v[:,2])*field.heights/2
    top=np.max(field.centers[:,2]+extent)
    bottom=field.centers[:,2]-extent
    min_alt=np.minimum(np.arcsin(sun[2]),np.arcsin(target[:,2]))-cone
    if np.any(min_alt<=1e-5):
        # Error assumptions beyond upward-ray guarantee use exhaustive candidates.
        return neighbor_lists(field,sun,target,v,cone,all_pairs=True)
    radius=(top-bottom)/np.tan(min_alt)+field.radii+field.radii.max()+1e-6
    neighbors=cKDTree(field.centers[:,:2]).query_ball_point(field.centers[:,:2],radius)
    counts=np.array([len(a) for a in neighbors])
    ii=np.repeat(np.arange(n),counts)
    jj=np.concatenate(neighbors).astype(np.int64)
    keep=ii!=jj
    ii,jj=ii[keep],jj[keep]
    delta=field.centers[jj]-field.centers[ii]
    dist2=np.einsum('ij,ij->i',delta,delta)
    rsum=field.radii[ii]+field.radii[jj]
    envelope=rsum+(np.sqrt(dist2)+rsum)*np.sin(cone)+1e-7
    result=[]
    for direction in (np.broadcast_to(sun,delta.shape),target[ii]):
        along=np.einsum('ij,ij->i',delta,direction)
        mask=(along>=-rsum)&(dist2-along*along<=envelope*envelope)
        count=np.bincount(ii[mask],minlength=n)
        result.append((np.r_[0,np.cumsum(count)].astype(np.int64),np.ascontiguousarray(jj[mask])))
    return tuple(result)
