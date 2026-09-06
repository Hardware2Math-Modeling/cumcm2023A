"""Offline staggered concentric-ring search; preserves earlier designs."""
from pathlib import Path
import numpy as np, math, json, time, argparse
import solve, adaptive
from recompute import constraints
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'ring_trial'; DEST.mkdir(exist_ok=True)
solve.WORK=DEST/'runs'; solve.WORK.mkdir(exist_ok=True); adaptive.WORK=solve.WORK
records=[]

def rings(w,h,z,ty,group=5,extra=0.,phase=0.):
    """Equal counts within radial zones; half-angle alternating rings.
    Transition rings are separated by one pitch; all pair constraints audited.
    Boundary/exclusion apply to base centres, as in the triangular baseline.
    """
    pitch=w+5.02; radius=100.02; result=[]; meta=[]; ir=0
    while radius<=350+abs(ty):
        n=int(math.floor(math.pi/math.asin(pitch/(2*radius))))
        for k in range(group):
            theta=2*math.pi*(np.arange(n)+.5*(k%2)+phase)/n
            xy=np.c_[radius*np.sin(theta),ty+radius*np.cos(theta)]
            keep=np.sum(xy*xy,axis=1)<=350**2
            points=xy[keep]
            result.extend(np.c_[points,np.full((len(points),3),[w,h,z])].tolist())
            meta.append({'ring':ir,'radius':radius,'nominal_count':n,'kept':int(keep.sum()),'zone_ring':k})
            ir+=1
            if k<group-1:
                delta=math.pi/n
                step=max(pitch/2+.00001,-radius+radius*math.cos(delta)+math.sqrt(max(0,pitch*pitch-radius*radius*math.sin(delta)**2)))
            else:step=pitch
            radius+=step+extra+.00001
            if radius>350+abs(ty):break
    return np.array(result),meta

def evaluate(a,tower,label,samples=64,**params):
    start=time.time();r=solve.trace(a,tower,label,samples)
    row={k:r[k] for k in ['power','unit','area','N']};row.update(label=label,samples=samples,seconds=time.time()-start,**params)
    records.append(row);(DEST/'evaluations.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(row),flush=True);return r

def search():
    for w in [6.,6.5,7.,7.5,8.]:
      for ty in [-60.,0.,60.]:
       for group in [3,5,8]:
        a,_=rings(w,w,w/2+.05,ty,group)
        evaluate(a,(0,ty),f'coarse{len(records):03}',64,w=w,h=w,z=w/2+.05,ty=ty,group=group,phase=0.)
    (DEST/'coarse.json').write_text(json.dumps(records,indent=2))

def refine():
    global records
    records=json.loads((DEST/'evaluations.json').read_text())
    base=json.loads((DEST/'coarse.json').read_text())
    feasible=sorted([r for r in base if r['power']>=60.2],key=lambda x:-x['unit'])
    if not feasible:raise RuntimeError('No coarse feasible ring design')
    bests=feasible[:3]
    seen=set()
    for b in bests:
      for dw,dy,phase,zadd in [(0,0,0,0),(-.25,0,0,0),(.25,0,0,0),(0,-30,0,0),(0,30,0,0),(0,0,.25,0),(0,0,0,1),(0,0,0,2)]:
        w=b['w']+dw;ty=b['ty']+dy;z=min(6,w/2+.05+zadd);g=b['group']
        key=(w,ty,z,g,phase)
        if key in seen or w>8:continue
        seen.add(key);a,_=rings(w,w,z,ty,g,phase=phase)
        evaluate(a,(0,ty),f'refine{len(records):03}',128,w=w,h=w,z=z,ty=ty,group=g,phase=phase)
    candidates=sorted([r for r in records if r['power']>=60.2],key=lambda x:-x['unit'])[:5]
    options=[]
    for j,b in enumerate(candidates):
        a,_=rings(b['w'],b['w'],b['z'],b['ty'],b['group'],phase=b['phase']);tower=(0,b['ty'])
        # Remove a small number at a time, validating the full coupled field.
        r=evaluate(a,tower,f'trim{j}_start',512)
        for it in range(4):
            p=r['per']['unit_kW_m2']*a[:,2]*a[:,3]/1000
            order=np.argsort(r['per']['unit_kW_m2'],kind='stable')
            budget=max(0,r['power']-60.20)
            n=int(np.searchsorted(np.cumsum(p[order]),budget))
            if n<1:break
            keep=np.ones(len(a),bool);keep[order[:n]]=False
            test=a[keep];rr=evaluate(test,tower,f'trim{j}_{it}',512)
            if rr['power']<60.10:break
            a=test;r=rr
        if r['power']>=60.1:options.append((a,r,b))
    a,r,b=max(options,key=lambda x:x[1]['unit'])
    solve.write_design(DEST/'q2.txt',a,(0,b['ty']))
    (DEST/'q2_params.json').write_text(json.dumps(b,indent=2))
    for m in [4096,16384]:evaluate(a,(0,b['ty']),f'q2_{m}',m)
    check=constraints(a,(0,b['ty']));assert check['passed']
    (DEST/'q2_checks.json').write_text(json.dumps(check,indent=2))

def q3():
    global records
    records=json.loads((DEST/'evaluations.json').read_text())
    bpar=json.loads((DEST/'q2_params.json').read_text());tower=(0,bpar['ty'])
    # Reintroduce all candidate positions from the selected ring generator.
    a,meta=rings(bpar['w'],bpar['w'],bpar['z'],bpar['ty'],bpar['group'],phase=bpar['phase'])
    dist=np.linalg.norm(a[:,:2]-tower,axis=1);lo=max(2.,bpar['w']/2+.05)
    edges=np.linspace(dist.min()-1e-7,dist.max()+1e-7,9);band=np.clip(np.searchsorted(edges,dist,side='right')-1,0,7)
    levels=lo+(6-lo)*(np.arange(8)/7)**4;levels[-1]=6.;a[:,4]=levels[band]
    path=[]
    for sweep in range(2):
      for k in range(6,-1,-1):
        lower=lo if k==0 else levels[k-1];upper=levels[k+1]
        choices=np.unique(np.round([lower,levels[k],(lower+levels[k])/2,(levels[k]+upper)/2,upper],8))
        best=None
        for j,z in enumerate(choices):
            test=a.copy();test[band==k,4]=z
            r=evaluate(test,tower,f'height{sweep}_{k}_{j}',128)
            if best is None or r['power']>best[0]['power']:best=(r,z,test)
        levels[k]=best[1];a=best[2];path.append({'sweep':sweep,'band':k+1,'height':float(levels[k]),'power':best[0]['power']})
    options=[];r=evaluate(a,tower,'q3_height_only',4096)
    if r['power']>=60.1:options.append((a.copy(),r,'height_only'))
    allocations=[]
    for it in range(3):
        a,r=adaptive.allocation(a,tower,f'ring_sizes{it}',256,60.20,False)
        allocations.append(r);active=a[a[:,3]>=2]
        if r['power']>=60.10:options.append((active.copy(),r,f'sizes{it}'))
    if not options:raise RuntimeError('No Q3 feasible candidate')
    a,r,origin=max(options,key=lambda x:x[1]['unit'])
    solve.write_design(DEST/'q3.txt',a,tower)
    for m in [4096,16384]:evaluate(a,tower,f'q3_{m}',m)
    check=constraints(a,tower);assert check['passed']
    order=np.argsort(np.linalg.norm(a[:,:2]-tower,axis=1));check['outermost_height']=float(a[order[-1],4]);check['height_monotone']=bool(np.all(np.diff(a[order,4])>=-1e-7))
    (DEST/'q3_checks.json').write_text(json.dumps(check,indent=2))
    (DEST/'q3_optimization.json').write_text(json.dumps({'selected':origin,'levels':levels.tolist(),'edges':edges.tolist(),'path':path,'allocations':allocations,'rings':meta},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['search','refine','q3']);args=parser.parse_args()
    {'search':search,'refine':refine,'q3':q3}[args.stage]()
