"""Offline, deterministic heliostat field design. Requires numpy, openpyxl and compiled trace."""
from pathlib import Path
import numpy as np
import subprocess, csv, json, time, math

ROOT=Path(__file__).resolve().parent
EXE=ROOT/'trace.exe'
WORK=ROOT/'runs'
WORK.mkdir(exist_ok=True)

def write_design(path, a, tower):
    np.savetxt(path,a,header=f'{len(a)} {tower[0]:.9f} {tower[1]:.9f}',comments='',fmt='%.9f')

def trace(a,tower,label,samples=64,seed=20260905,alpha=.00465,mast=3.5):
    inp=WORK/(label+'.txt');out=WORK/(label+'.csv')
    write_design(inp,a,tower)
    p=subprocess.run([str(EXE),str(inp),str(out),str(samples),str(seed),str(alpha),str(mast),'8'],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    t=np.genfromtxt(out,delimiter=',',names=True)
    per=np.genfromtxt(str(out)+'.mirrors.csv',delimiter=',',names=True)
    return {'power':float(t['power_MW'].mean()),'unit':float(t['unit_kW_m2'].mean()),'area':float((a[:,2]*a[:,3]).sum()),'N':len(a),'per':per,'time':t}

def hex_field(w,h,z,ty,rot=0,stretch=1,phase=0,setback=True):
    # x-spacing can expand; triangular nearest-neighbour spacing stays >= w+5.02.
    pitch=w+5.02; dx=pitch*stretch; dy=max(pitch*.5,math.sqrt(max(0,pitch*pitch-dx*dx/4)))
    coords=[];rd=.5*math.hypot(w,h) if setback else 0.;angle=math.radians(rot)
    for row in range(-90,91):
      for col in range(-55,56):
        x=dx*(col+.5*(row%2));y=dy*(row+phase)
        xx=x*math.cos(angle)-y*math.sin(angle);yy=x*math.sin(angle)+y*math.cos(angle)+ty
        if xx*xx+yy*yy<=(350-rd)**2 and math.hypot(xx,yy-ty)>=100+rd:
          coords.append([xx,yy,w,h,z])
    return np.array(coords)

def trim(a,tower,label,samples=128,target=60.10):
    a=a.copy()
    for iteration in range(7):
      r=trace(a,tower,f'{label}_t{iteration}',samples)
      if r['power']<target: return a,r
      p=r['per']['unit_kW_m2']*a[:,2]*a[:,3]/1000
      order=np.argsort(-r['per']['unit_kW_m2'],kind='stable')
      n=int(np.searchsorted(np.cumsum(p[order]),target))+1
      if n>=len(a):return a,r
      a=a[np.sort(order[:n])]
    return a,trace(a,tower,label+'_last',samples)

def ring_field(w,h,z,ty,dr=1.0,offset=.5):
    pitch=w+5.02; rd=.5*math.hypot(w,h); coords=[]
    radius=100+rd; ir=0
    while radius<350+abs(ty):
      num=int(math.floor(PI/math.asin(pitch/(2*radius))))
      for j in range(num):
        angle=2*math.pi*(j+offset*(ir%2))/num
        x=radius*math.sin(angle);y=ty+radius*math.cos(angle)
        if x*x+y*y>(350-rd)**2:continue
        # Greedy exclusion also makes noninteger dr choices feasible.
        if coords:
          prev=np.array(coords[-2*num:])
          if np.any((prev[:,0]-x)**2+(prev[:,1]-y)**2<pitch*pitch-1e-8):continue
        coords.append([x,y,w,h,z])
      radius+=pitch*dr;ir+=1
    return np.array(coords)

PI=math.pi

def zonal_field(w,h,z,ty,span=2.,radfactor=.8,extra=0.):
    pitch=w+5.02; rd=.5*math.hypot(w,h); coords=[]
    radius=100+rd; zone_start=radius; num=int(math.floor(PI/math.asin(pitch/(2*radius))));ir=0
    while radius<350+abs(ty):
      if radius>zone_start*span:
        num=int(math.floor(PI/math.asin(pitch/(2*radius))));zone_start=radius;ir=0
        radius+=pitch*.2
      delta=PI/num
      # Same-zone alternation lets the next ring occupy angular gaps.
      angular=2*PI*(np.arange(num)+.5*(ir%2))/num
      points=np.c_[radius*np.sin(angular),ty+radius*np.cos(angular)]
      for x,y in points:
        if x*x+y*y>(350-rd)**2:continue
        if coords:
          prev=np.array(coords[-3*num:]);
          if np.any((prev[:,0]-x)**2+(prev[:,1]-y)**2<pitch*pitch-1e-8):continue
        coords.append([x,y,w,h,z])
      # Solve the inter-ring centre distance constraint for staggered points.
      ds=max(0,pitch*pitch-radius*radius*math.sin(delta)**2)
      step=max(.3*pitch,-radius+radius*math.cos(delta)+math.sqrt(ds))
      step=max(step,radfactor*h*radius/(2*(80-z)))+extra
      radius+=step+1e-5;ir+=1
    return np.array(coords)

def zonal_search():
    records=[];idx=0
    for w,h in [(6,6),(6.5,6.5),(7,7),(7.5,7.5),(8,7),(8,8)]:
      for ty in [0.,-80.,80.]:
       for span,factor in [(2.,.8),(2.,1.1),(1.4,.8),(1.4,1.1)]:
        label=f'zone{idx:03}';idx+=1;a=zonal_field(w,h,6,ty,span,factor)
        r=trace(a,(0,ty),label,64)
        if r['power']>=60.10:a,r=trim(a,(0,ty),label,64)
        d=dict(w=w,h=h,z=6.,ty=ty,span=span,factor=factor,label=label,N=r['N'],area=r['area'],power=r['power'],unit=r['unit'])
        records.append(d);write_design(WORK/(label+'_final.txt'),a,(0,ty));print(json.dumps(d),flush=True)
    with (ROOT/'zonal.json').open('w') as f:json.dump(records,f,indent=2)

def center_search():
    records=[];idx=0
    for w in [6.,6.25,6.5,6.75,7.,7.25,7.5,7.75,8.]:
      for ty in [-60.,0.,60.]:
       for rot in [0.,30.]:
        label=f'center{idx:03}';idx+=1;h=w;z=h/2+.05;a=hex_field(w,h,z,ty,rot,setback=False)
        r=trace(a,(0,ty),label,64)
        if r['power']>=60.10:a,r=trim(a,(0,ty),label,128)
        d=dict(w=w,h=h,z=z,ty=ty,rot=rot,label=label,N=r['N'],area=r['area'],power=r['power'],unit=r['unit'])
        records.append(d);write_design(WORK/(label+'_final.txt'),a,(0,ty));print(json.dumps(d),flush=True)
    with (ROOT/'center.json').open('w') as f:json.dump(records,f,indent=2)

def refine_search():
    records=[];idx=0
    for w,h in [(6.05,6.05),(6.1,6.1),(6.15,6.15),(6.2,6.2),(6.25,6.25),(6.4,6.2)]:
      for ty in [-120.,-90.,-60.,-30.]:
       for rot in [15.,30.,45.]:
        label=f'refine{idx:03}';idx+=1;z=h/2+.05;a=hex_field(w,h,z,ty,rot,setback=False)
        r=trace(a,(0,ty),label,64)
        if r['power']>=60.10:a,r=trim(a,(0,ty),label,128)
        d=dict(w=w,h=h,z=z,ty=ty,rot=rot,label=label,N=r['N'],area=r['area'],power=r['power'],unit=r['unit'])
        records.append(d);write_design(WORK/(label+'_final.txt'),a,(0,ty));print(json.dumps(d),flush=True)
    with (ROOT/'refine.json').open('w') as f:json.dump(records,f,indent=2)

def geometry_search():
    records=[];idx=0
    for w,h in [(6,6),(6.5,6.5),(7,6),(7,7),(7.5,7),(7.5,7.5),(8,7),(8,8)]:
      for ty in [0.,-60.,-120.]:
       for geometry in ['hex30','hex0s1.35','hex90s1.35','ring1','ring1.15']:
        label=f'geom{idx:03}';idx+=1
        if geometry.startswith('ring'): a=ring_field(w,h,6,ty,float(geometry[4:]))
        else:
          parts=geometry[3:].split('s');a=hex_field(w,h,6,ty,float(parts[0]),float(parts[1]) if len(parts)>1 else 1)
        r=trace(a,(0,ty),label,64)
        if r['power']>=60.10:a,r=trim(a,(0,ty),label,64)
        d=dict(w=w,h=h,z=6.,ty=ty,geometry=geometry,label=label,N=r['N'],area=r['area'],power=r['power'],unit=r['unit'])
        records.append(d);write_design(WORK/(label+'_final.txt'),a,(0,ty));print(json.dumps(d),flush=True)
    with (ROOT/'geometry.json').open('w') as f:json.dump(records,f,indent=2)

def coarse():
    records=[]; t0=time.time();idx=0
    for w in [6.,7.,8.]:
      for h in [5.,6.,7.,8.]:
       if h>w:continue
       for ty in [0.,-80.,-140.,-200.]:
        label=f'coarse{idx:03}';idx+=1;a=hex_field(w,h,6,ty)
        r=trace(a,(0,ty),label,64)
        if r['power']>=60.10:a,r=trim(a,(0,ty),label,64)
        d=dict(w=w,h=h,z=6.,ty=ty,rot=0.,stretch=1.,label=label,N=r['N'],area=r['area'],power=r['power'],unit=r['unit'])
        records.append(d);write_design(WORK/(label+'_final.txt'),a,(0,ty))
        print(json.dumps(d),flush=True)
    with (ROOT/'coarse.json').open('w') as f:json.dump(records,f,indent=2)
    print('elapsed',time.time()-t0,flush=True)

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='geometry':geometry_search()
    elif len(sys.argv)>1 and sys.argv[1]=='zonal':zonal_search()
    elif len(sys.argv)>1 and sys.argv[1]=='center':center_search()
    elif len(sys.argv)>1 and sys.argv[1]=='refine':refine_search()
    else:coarse()
