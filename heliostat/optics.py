"""Joint ray-energy accounting: shading, blocking and interception are correlated.

Scrambled Sobol rays integrate mirror area × finite solar disk (four dimensions).
Independent scrambles, NOT individual rays, are used for uncertainty estimates.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ctypes
import hashlib
import platform
import subprocess
import warnings
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from .model import Site,time_grid
from .geometry import neighbor_lists,cylinder_intersection,rectangle_hits


@lru_cache(maxsize=32)
def samples(count,seed):
    if count<1 or count&(count-1):
        raise ValueError("Sobol sample count must be a positive power of two")
    q=qmc.Sobol(6,scramble=True,seed=int(seed)).random_base2(int(np.log2(count)))
    return np.ascontiguousarray(q[:,:2]-.5),q[:,2:4],np.ascontiguousarray(np.clip(ndtri(q[:,4:6]),-4,4))


def solar_rays(sun,disk,radius):
    axis=np.array([0.,0.,1.]) if abs(sun[2])<.99 else np.array([1.,0.,0.])
    a=np.cross(sun,axis);a/=np.linalg.norm(a)
    b=np.cross(sun,a)
    # Uniform projected solid angle (constant radiance disk).
    sint=np.sin(radius)*np.sqrt(disk[:,0])
    phi=2*np.pi*disk[:,1]
    return np.ascontiguousarray(np.sqrt(1-sint*sint)[:,None]*sun+
                                (sint*np.cos(phi))[:,None]*a+(sint*np.sin(phi))[:,None]*b)


@lru_cache(maxsize=1)
def native_library():
    src=Path(__file__).with_name('raytrace.cpp')
    digest=hashlib.sha256(src.read_bytes()+platform.platform().encode()).hexdigest()[:16]
    folder=src.parent.parent/'build'/'native'
    folder.mkdir(parents=True,exist_ok=True)
    binary=folder/f'raytrace-{digest}.so'
    if not binary.exists():
        command=['c++','-O3','-std=c++11','-fPIC',
                 '-dynamiclib' if platform.system()=='Darwin' else '-shared',str(src),'-o',str(binary)]
        subprocess.run(command,check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str(binary))
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.trace.argtypes=[ctypes.c_int,ctypes.c_int]+[f64]*10+[i64]*4+[ctypes.c_double]*7+[f64]
    lib.trace.restype=None
    return lib


def trace_numpy(field,normal,u,v,directions,uv,gauss,sun,neighbors,site):
    """Independent vectorized Python implementation of the native ray kernel."""
    out=np.zeros((len(field),7))
    zlo=site.tower_height-site.receiver_height/2
    zhi=site.tower_height+site.receiver_height/2
    for i in range(len(field)):
        p=field.centers[i]+uv[:,0,None]*field.widths[i]*u[i]+uv[:,1,None]*field.heights[i]*v[i]
        weight=np.maximum(0,directions@normal[i])/(directions@sun)
        local=normal[i]+site.slope_error*(gauss[:,0,None]*u[i]+gauss[:,1,None]*v[i])
        local/=np.linalg.norm(local,axis=1)[:,None]
        r=2*np.einsum('ij,ij->i',local,directions)[:,None]*local-directions
        arrival,kind=cylinder_intersection(p,r,field.tower,site.receiver_radius,zlo,zhi)
        hit=kind==1
        # Test upstream visibility FROM the mirror TOWARD the Sun. Photon
        # propagation is -s, but a shadow query starts at the receiver point.
        incident=directions
        shadow=np.isfinite(cylinder_intersection(p,incident,field.tower,site.tower_radius,0,zlo)[0])
        shadow|=np.isfinite(cylinder_intersection(p,incident,field.tower,site.receiver_radius,zlo,zhi)[0])
        ptr,ids=neighbors[0]
        for j in ids[ptr[i]:ptr[i+1]]:
            shadow|=rectangle_hits(p,incident,field.centers[j],normal[j],u[j],v[j],field.widths[j],field.heights[j])
        block=cylinder_intersection(p,r,field.tower,site.tower_radius,0,zlo)[0]<arrival
        ptr,ids=neighbors[1]
        for j in ids[ptr[i]:ptr[i+1]]:
            block|=rectangle_hits(p,r,field.centers[j],normal[j],u[j],v[j],field.widths[j],field.heights[j])
        total=weight.sum();survived=weight[~shadow&~block].sum()
        accepted=weight[~shadow&~block&hit].sum()
        out[i]=[total/len(uv),survived/total if total else 0,
                accepted/survived if survived else 0,weight[~shadow].sum()/total if total else 0,
                weight[~shadow&block].sum()/total if total else 0,weight[hit].sum()/total if total else 0,
                accepted/len(uv)]
    return out


@dataclass
class Evaluation:
    # metrics time × mirror × [cos, sb, trunc, shadow, blocking_loss, isolated_trunc, accepted_cos]
    metrics: np.ndarray
    power_kw: np.ndarray
    atmospheric: np.ndarray
    area: np.ndarray
    time_indices: np.ndarray
    ray_count: int
    seed: int
    reflectivity: float

    @property
    def annual_power_kw(self):
        return float(self.power_kw.sum(axis=1).mean())

    @property
    def unit_power(self):
        return self.annual_power_kw/self.area.sum()

    @property
    def per_mirror_kw(self):
        return self.power_kw.mean(axis=0)

    def rows(self):
        optical=self.metrics[:,:,6]*self.atmospheric[None,:]*self.reflectivity
        # Field efficiencies are area weighted; Q3 must not average unlike mirrors equally.
        mean=lambda a:np.average(a,axis=1,weights=self.area)
        return np.c_[mean(optical),mean(self.metrics[:,:,0]),mean(self.metrics[:,:,1]),
                     np.full(len(optical),np.average(self.atmospheric,weights=self.area)),
                     mean(self.metrics[:,:,2]),self.power_kw.sum(axis=1)/1000,
                     self.power_kw.sum(axis=1)/self.area.sum(),mean(self.metrics[:,:,3]),
                     mean(self.metrics[:,:,4]),mean(self.metrics[:,:,5])]

    def monthly(self):
        if not np.array_equal(self.time_indices,np.arange(60)):
            raise ValueError("Monthly reporting requires all 60 prescribed times")
        return self.rows().reshape(12,5,-1).mean(axis=1)


def evaluate(field,site=Site(),rays=256,seed=2023,time_indices=None,
             interactions=True,backend='auto',all_pairs=False):
    indices=np.arange(60) if time_indices is None else np.asarray(time_indices,int)
    if len(indices)==0 or np.any((indices<0)|(indices>=60)):
        raise ValueError("Invalid time indices")
    _,suns,dnis=time_grid(site)
    uv,disk,gauss=samples(rays,seed)
    cone=site.solar_radius+2*np.arctan(4*np.sqrt(2)*site.slope_error)
    lib=None
    # Native kernel is independently checked against the NumPy oracle in tests.
    if backend!='numpy':
        try:
            lib=native_library()
        except (OSError,subprocess.CalledProcessError) as error:
            if backend=='native':raise
            warnings.warn(f'Native compiler unavailable; using NumPy: {error}')
    result=[]
    for k in indices:
        sun=suns[k]
        target,normal,u,v,distance=field.geometry(sun,site)
        directions=solar_rays(sun,disk,site.solar_radius)
        if interactions:
            neighbors=neighbor_lists(field,sun,target,v,cone,all_pairs)
        else:
            empty=(np.zeros(len(field)+1,np.int64),np.empty(0,np.int64))
            neighbors=(empty,empty)
        if lib is None:
            out=trace_numpy(field,normal,u,v,directions,uv,gauss,sun,neighbors,site)
        else:
            out=np.empty((len(field),7))
            lib.trace(len(field),rays,field.centers,field.widths,field.heights,
                      np.ascontiguousarray(normal),np.ascontiguousarray(u),np.ascontiguousarray(v),
                      directions,uv,gauss,np.ascontiguousarray(sun),*neighbors[0],*neighbors[1],
                      *field.tower,site.tower_height,site.receiver_radius,site.receiver_height,
                      site.tower_radius,site.slope_error,out)
        result.append(out)
    metrics=np.asarray(result)
    atmospheric=.99321-.0001176*distance+1.97e-8*distance**2
    if np.any(distance>1000):
        raise ValueError("Atmospheric formula used outside d <= 1000 m")
    power=dnis[indices,None]*field.area[None,:]*atmospheric[None,:]*site.reflectivity*metrics[:,:,6]
    return Evaluation(metrics,power,atmospheric,field.area.copy(),indices,rays,seed,site.reflectivity)
