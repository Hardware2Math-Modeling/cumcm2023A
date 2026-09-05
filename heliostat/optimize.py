"""Multi-fidelity, reproducible mixed discrete/continuous field optimization.

The isolated-mirror screen is only an optimistic bound/initialization. Every
retained design is scored with interacting ray tracing at all 60 prescribed times.
No claim of a global optimum is made for this nonconvex layout problem.
"""
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from .model import Field,Site,time_grid,validate_field
from .optics import evaluate
from .io import write_json,save_field


@dataclass(frozen=True)
class SearchBudget:
    global_candidates: int=256
    finalists: int=24
    refinement_trials: int=36
    q3_trials: int=48
    screening_rays: int=32
    search_rays: int=128
    refinement_rays: int=256
    # Search target has a 50 kW numerical reserve; final independent lower-bound
    # certification adds its own measured reserve and can replenish mirrors.
    target_kw: float=60050.
    seed: int=202309
    workers: int=4


def decode(p):
    p=np.clip(np.asarray(p),0,1)
    if len(p)==11:p=np.r_[p,1/3]  # backward-compatible checkpoint: four-ring bands
    width=2+6*p[2]
    height=2+(width-2)*p[3]
    zmin=max(2.,height/2+.1)
    return dict(tower=np.array([160*(p[0]-.5),-245+450*p[1]]),width=width,height=height,
                z=zmin+(6-zmin)*p[4],pitch=width+5.05+3*p[5],stretch=.8660254038+.5*p[6],
                angle=np.pi/3*p[7],phase=p[8:10],
                family=('adaptive_rings' if p[10]>2/3 else 'rings' if p[10]>1/3 else 'staggered'),
                blocking_factor=.8+.8*p[6],band_length=2+int(np.rint(6*p[11])))


def encode(tower=(0,-180),width=7,height=7,z=5,gap=.05,stretch=.9,angle=0,family='staggered'):
    zmin=max(2.,height/2+.1)
    return np.array([tower[0]/160+.5,(tower[1]+245)/450,(width-2)/6,
                     (height-2)/(width-2) if width>2 else 1,(z-zmin)/(6-zmin),
                     (gap-.05)/3,(stretch-.8660254038)/.5,angle/(np.pi/3),.5,.5,
                     .85 if family=='adaptive_rings' else .5 if family=='rings' else .15,1/3])


def candidates(p,site=Site()):
    d=decode(p)
    tower,w,h,z,s=d['tower'],d['width'],d['height'],d['z'],d['pitch']
    rad=np.hypot(w,h)/2
    if np.linalg.norm(tower)>site.field_radius-site.exclusion_radius:
        return None
    outer=site.field_radius-rad-.02
    inner=site.exclusion_radius+rad+.02
    if d['family']=='staggered':
        n=int(np.ceil(site.field_radius/(s*.866)))+3
        ix,iy=np.meshgrid(np.arange(-n,n+1),np.arange(-n,n+1))
        x=(ix+.5*(iy%2)+d['phase'][0]-.5)*s
        y=(iy+d['phase'][1]-.5)*s*d['stretch']
        xy=np.c_[x.ravel(),y.ravel()]
        a=d['angle'];xy=xy@np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    else:
        points=[]
        max_radius=outer+np.linalg.norm(tower)
        # Keep the SAME angular count within each band, alternate half-cell phase.
        # Recounting every ring destroys staggering and forces massive conflict
        # deletion. At a band transition, one full-pitch radial gap is sufficient.
        r=inner+d['phase'][1]*s
        band_length=d['band_length']
        while r<max_radius:
            count=max(3,int(np.floor(np.pi/np.arcsin(min(1,s/(2*r))))))
            for k in range(band_length):
                if r>max_radius:break
                angles=2*np.pi*(np.arange(count)+.5*(k%2)+d['phase'][0])/count+d['angle']
                points.append(tower+np.c_[r*np.cos(angles),r*np.sin(angles)])
                if d['family']=='adaptive_rings':
                    dr=max(s*.8660254038,d['blocking_factor']*h*.92*r/(2*(site.tower_height-z)))
                else:dr=s*d['stretch']
                r+=max(s,dr) if k==band_length-1 else dr
        xy=np.concatenate(points) if points else np.empty((0,2))
    keep=(np.linalg.norm(xy,axis=1)<=outer)&(np.linalg.norm(xy-tower,axis=1)>=inner)
    xy=xy[keep]
    if not len(xy):return None
    if d['family']!='staggered':
        # Different ring counts can create cross-ring collisions. Explicitly remove
        # them with a deterministic independent-set pass, never assume stagger suffices.
        tree=cKDTree(xy)
        conflicts=tree.query_ball_point(xy,s-1e-8)
        enabled=np.ones(len(xy),bool)
        order=np.lexsort((xy[:,0],-xy[:,1]))
        keep=[]
        for i in order:
            if enabled[i]:
                keep.append(i);enabled[conflicts[i]]=False
        xy=xy[keep]
    return Field(np.c_[xy,np.full(len(xy),z)],w,h,tower,'q2')


def initial_population(count,seed):
    engine=qmc.Sobol(12,scramble=True,seed=seed)
    pop=engine.random_base2(int(np.ceil(np.log2(max(2,count)))) )[:count]
    # Feasible-focused samples plus an unrestricted fraction for smaller mirrors.
    pop[:count*3//4,2]=.45+.55*pop[:count*3//4,2]
    pop[:count*3//4,3]=.6+.4*pop[:count*3//4,3]
    pop[:count*3//4,1]*=.55
    pop[:count*3//4,5]*=.45
    pop[:count*3//4,6]*=.55
    warm=[]
    for w in (5.,5.5,6.,6.5,7.,7.5,8.):
        for ty in (0.,-60.,-160.):
            for family in ('staggered','rings','adaptive_rings'):
                p=encode((0,ty),w,w,w/2+.1,stretch=.866026,family=family)
                p[9]=0.
                if family=='adaptive_rings':p[6]=.25  # blocking factor 1.0: dense central bands
                warm.append(p)
    for w in (6.,6.25,6.5,6.75,7.,7.25,7.5,7.75,8.):
        for phase in (0.,.3,.6,.9):
            p=encode((0,0),w,w,w/2+.1,stretch=.866026,family='rings')
            p[9]=phase;warm.append(p)
    warm=np.array(warm)
    return np.vstack([warm,pop])


def screen_task(args):
    p,site,budget=args
    field=candidates(p,site)
    if field is None:return None
    # Cheap rigorous bound: eta_cos, eta_sb, eta_trunc, eta_at <= 1.
    max_density=time_grid(site)[2].mean()*site.reflectivity
    if field.area.sum()*max_density<budget.target_kw:return None
    e=evaluate(field,site,rays=budget.screening_rays,seed=budget.seed,interactions=False)
    order=np.argsort(-e.per_mirror_kw/field.area)
    cum=np.cumsum(e.per_mirror_kw[order])
    if cum[-1]<budget.target_kw:return None
    n=int(np.searchsorted(cum,budget.target_kw)+1)
    unit=cum[n-1]/field.area[order[:n]].sum()
    actual=evaluate(field,site,rays=budget.screening_rays,seed=budget.seed)
    capacity=actual.annual_power_kw
    merit=unit*min(1.,capacity/budget.target_kw)**8
    return {'parameters':p.tolist(),'optimistic_unit':float(unit),'screen_merit':float(merit),'count':n,
            'coarse_interacting_capacity_kw':capacity,'optimistic_capacity_kw':float(cum[-1])}


def trim_field(field,site,target,rays,seed,evaluation=None,passes=5):
    """Recompute interactions after deletion; removed mirrors can no longer block.

    Removing mirrors whose OWN power <= surplus is conservative: every remaining
    ray can only gain visibility. QMC sample identity is preserved across subsets.
    """
    e=evaluation or evaluate(field,site,rays=rays,seed=seed)
    for _ in range(passes):
        if e.annual_power_kw<target:return field,e
        order=np.argsort(-e.per_mirror_kw/field.area,kind='stable')
        cumulative=np.cumsum(e.per_mirror_kw[order])
        n=int(np.searchsorted(cumulative,target)+1)
        if n>=len(field):break
        field=field.subset(order[:n])
        e=evaluate(field,site,rays=rays,seed=seed)
    return field,e


def select_design(p,site,budget,rays=None):
    rays=rays or budget.search_rays
    field=candidates(p,site)
    if field is None:return None
    e=evaluate(field,site,rays=rays,seed=budget.seed)
    if e.annual_power_kw<budget.target_kw:
        # Some crowded fields gain net energy when low-value blockers are removed.
        isolated=evaluate(field,site,rays=rays,seed=budget.seed,interactions=False)
        order=np.argsort(-isolated.per_mirror_kw/field.area)
        cum=np.cumsum(isolated.per_mirror_kw[order])
        if cum[-1]<budget.target_kw:return None
        found=None
        for factor in (1.04,1.10):
            n=min(len(field),int(np.searchsorted(cum,budget.target_kw*factor)+1))
            sub=field.subset(order[:n]);es=evaluate(sub,site,rays=rays,seed=budget.seed)
            if es.annual_power_kw>=budget.target_kw:
                found=(sub,es);break
        if found is None:return None
        field,e=found
    field,e=trim_field(field,site,budget.target_kw,rays,budget.seed,e)
    check=validate_field(field,site,uniform=True,whole_mirror=True)
    if not check['valid']:raise AssertionError(check)
    return field,e


def design_task(args):
    p,site,budget=args
    result=select_design(p,site,budget)
    if result is None:return None
    field,e=result
    return {'parameters':p,'field':field,'unit':e.unit_power,'power_kw':e.annual_power_kw}


def capacity_task(args):
    p,site,budget=args
    field=candidates(p,site)
    if field is None:return None
    e=evaluate(field,site,rays=budget.search_rays,seed=budget.seed)
    return {'parameters':p.tolist(),'capacity_kw':e.annual_power_kw,'unit':e.unit_power}


def feasibility_search(screened,site,budget,outdir,progress):
    """Phase I: evolve near-feasible complete layouts instead of discarding them."""
    top=sorted(screened,key=lambda a:a['coarse_interacting_capacity_kw'],reverse=True)[:16]
    population=[np.r_[x['parameters'],1/3] if len(x['parameters'])==11 else np.array(x['parameters']) for x in top]
    rng=np.random.default_rng(budget.seed+301)
    archive=[]
    steps=np.array([.05,.045,.07,.03,.18,.018,.05,.2,.3,.17,0.,.2])
    with ProcessPoolExecutor(max_workers=budget.workers) as pool:
        for generation in range(6):
            batch=population[:8] if generation==0 else []
            while len(batch)<32:
                parent=population[rng.integers(min(8,len(population)))].copy()
                mask=rng.random(12)<.3;mask[rng.integers(12)]=True
                child=parent+mask*rng.normal(size=12)*steps*(.8**generation)
                child[8:10]%=1
                batch.append(np.clip(child,0,1))
            results=[x for x in pool.map(capacity_task,[(p,site,budget) for p in batch]) if x is not None]
            archive.extend(results)
            ranked=sorted(archive,key=lambda x:x['capacity_kw'],reverse=True)
            population=[np.array(x['parameters']) for x in ranked[:16]]
            progress(f"Q2 feasibility generation {generation+1}/6: best full-field power {ranked[0]['capacity_kw']/1000:.5f} MW",flush=True)
            write_json(archive,outdir/'q2_feasibility_history.json')
    ranked=sorted(archive,key=lambda x:x['capacity_kw'],reverse=True)
    return [np.array(x['parameters']) for x in ranked[:16]]


def diverse_finalists(screened,count):
    ranked=sorted(screened,key=lambda a:a['screen_merit'],reverse=True)
    selected=[]
    # Keep true top candidates plus distinct tower/size/layout basins.
    for item in ranked:
        p=np.array(item['parameters'])
        if not selected or len(selected)<max(3,count//4) or all(
            np.linalg.norm((p-np.array(other['parameters']))[[1,2,3,5,6,10]])>.17 for other in selected):
            selected.append(item)
        if len(selected)>=count:break
    for item in ranked:
        if len(selected)>=count:break
        if not any(item is x for x in selected):selected.append(item)
    return selected


def optimize_q2(outdir,site=Site(),budget=SearchBudget(),progress=print,resume=False):
    start=perf_counter();history=[]
    population=initial_population(budget.global_candidates,budget.seed)
    progress(f'Q2: isolated upper-bound screening of {len(population)} parameter sets',flush=True)
    from .optics import native_library
    native_library()  # compile once before starting workers
    tasks=[(p,site,budget) for p in population]
    screened=[]
    if resume and (outdir/'q2_screening.json').exists():
        import json
        screened=json.loads((outdir/'q2_screening.json').read_text())
        progress(f'Resumed {len(screened)} screening records; full optical scoring is rerun',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=budget.workers) as pool:
            for k,result in enumerate(pool.map(screen_task,tasks,chunksize=1),1):
                if result is not None:screened.append(result)
                if k%32==0:progress(f'  screened {k}/{len(tasks)}, potentially feasible {len(screened)}',flush=True)
    if not screened:raise RuntimeError('No potentially feasible layouts; expand screening budget')
    write_json(screened,outdir/'q2_screening.json')
    finalists=diverse_finalists(screened,budget.finalists)
    tasks=[(np.array(x['parameters']),site,budget) for x in finalists]
    scored=[]
    progress(f'Q2: interacting 60-time ray tracing of {len(tasks)} finalists',flush=True)
    with ProcessPoolExecutor(max_workers=budget.workers) as pool:
        for k,result in enumerate(pool.map(design_task,tasks,chunksize=1),1):
            if result is not None:
                scored.append(result)
                history.append(dict(stage='full_screen',trial=k,unit=result['unit'],power_kw=result['power_kw'],
                                    parameters=result['parameters'],count=len(result['field'])))
                progress(f"  candidate {k}: {result['unit']:.6f} kW/m², {result['power_kw']/1000:.4f} MW",flush=True)
            else:progress(f'  candidate {k}: rejected by full optical model',flush=True)
    if len(scored)<4:
        progress('Q2: promoting near-feasible layouts to a dedicated capacity-restoration search',flush=True)
        rescued=feasibility_search(screened,site,budget,outdir,progress)
        with ProcessPoolExecutor(max_workers=budget.workers) as pool:
            for result in pool.map(design_task,[(p,site,budget) for p in rescued]):
                if result:
                    scored.append(result)
                    history.append(dict(stage='feasibility_restoration',unit=result['unit'],power_kw=result['power_kw'],
                                        parameters=result['parameters'],count=len(result['field'])))
    if not scored:raise RuntimeError('No feasible design after capacity restoration; increase budget/layout freedom')
    scored.sort(key=lambda a:a['unit'],reverse=True)
    # Promote multiple basins to finer fidelity before selecting the incumbent.
    fine=[]
    for item in scored[:min(6,len(scored))]:
        result=select_design(item['parameters'],site,budget,budget.refinement_rays)
        if result:
            field,e=result;fine.append(dict(parameters=item['parameters'],field=field,unit=e.unit_power,power_kw=e.annual_power_kw))
    if not fine:raise RuntimeError('No feasible design after fidelity promotion')
    fine.sort(key=lambda a:a['unit'],reverse=True)
    best=fine[0]
    best['parameters']=np.r_[best['parameters'],1/3] if len(best['parameters'])==11 else np.array(best['parameters'])
    rng=np.random.default_rng(budget.seed+1)
    progress(f"Q2: fine incumbent {best['unit']:.6f}; starting {budget.refinement_trials} layout trials",flush=True)
    # Differential mutations between elite basins plus coordinate pattern moves.
    for k in range(budget.refinement_trials):
        p=np.array(best['parameters']).copy()
        if k%4==0 and len(fine)>=3:
            a,b=rng.choice(len(fine),2,replace=False)
            trial=p+.35*(np.array(fine[a]['parameters'])-np.array(fine[b]['parameters']))
            cross=rng.random(len(p))<.65;cross[rng.integers(len(p))]=True
            p=np.where(cross,trial,p)
        else:
            dims=[1,2,3,4,5,6,7,8,9,0,11]
            dim=dims[k%len(dims)]
            step=(.10 if k<budget.refinement_trials//2 else .035)
            p[dim]+=step*(1 if rng.random()<.5 else -1)
        p=np.clip(p,0,1)
        result=select_design(p,site,budget,budget.refinement_rays)
        if result:
            field,e=result
            improved=e.unit_power>best['unit']+1e-8
            history.append(dict(stage='refine',trial=k,unit=e.unit_power,power_kw=e.annual_power_kw,
                                parameters=p,count=len(field),accepted=improved))
            if improved:
                best=dict(parameters=p,field=field,unit=e.unit_power,power_kw=e.annual_power_kw)
                save_field(field,outdir/'q2_checkpoint.npz')
        if k%4==0 or (result and improved):
            progress(f"  refinement {k+1}/{budget.refinement_trials}: best {best['unit']:.6f}",flush=True)
    field,e=trim_field(best['field'],site,budget.target_kw,budget.refinement_rays,budget.seed)
    field.name='q2'
    save_field(field,outdir/'q2.npz')
    write_json(history,outdir/'q2_search_history.json')
    write_json({'parameters':best['parameters'],'decoded':decode(best['parameters']),
                'budget':budget,'seconds':perf_counter()-start,'unit':e.unit_power,
                'power_kw':e.annual_power_kw,'constraints':validate_field(field,site,True,True)},outdir/'q2_search.json')
    return field,e


def q3_groups(field):
    relative=field.centers[:,:2]-field.tower
    radius=np.linalg.norm(relative,axis=1)
    edges=np.quantile(radius,[1/3,2/3])
    radial=np.searchsorted(edges,radius)
    # East-west symmetry is useful but not imposed on individual coordinates.
    side=(np.abs(relative[:,0])>np.abs(relative[:,1])*.7).astype(int)
    return radial*2+side


def replenish(field,e,pool,pool_density,site,budget,rays=None):
    """Restore capacity with explicitly spaced unused candidate sites, then retrace."""
    rays=rays or budget.refinement_rays
    if e.annual_power_kw>=budget.target_kw:return field,e
    if pool is None:return field,e
    for _ in range(3):
        tree=cKDTree(field.centers[:,:2])
        near=tree.query_ball_point(pool.centers[:,:2],float(max(pool.widths.max(),field.widths.max())+5))
        allowed=[]
        for i,js in enumerate(near):
            if not js:
                allowed.append(i);continue
            distance=np.linalg.norm(field.centers[js,:2]-pool.centers[i,:2],axis=1)
            if np.all(distance>=np.maximum(field.widths[js],pool.widths[i])+5+1e-7):allowed.append(i)
        if not allowed:break
        order=np.array(allowed)[np.argsort(-pool_density[allowed])]
        # Expected per-mirror power is only a count proposal, never a feasibility certificate.
        proposed=np.cumsum(pool_density[order]*pool.area[order]*.9)
        count=min(len(order),int(np.searchsorted(proposed,budget.target_kw-e.annual_power_kw+15)+1))
        add=pool.subset(order[:count])
        trial=Field(np.vstack([field.centers,add.centers]),np.r_[field.widths,add.widths],
                    np.r_[field.heights,add.heights],field.tower,field.name)
        if not validate_field(trial,site,whole_mirror=True)['valid']:break
        field=trial;e=evaluate(field,site,rays=rays,seed=budget.seed)
        if e.annual_power_kw>=budget.target_kw:break
    return field,e


def optimize_q3(q2,outdir,site=Site(),budget=SearchBudget(),progress=print):
    """Actual heterogeneous dimensions/heights, insertion, and position search."""
    start=perf_counter()
    field=q2.copy(name='q3')
    best=evaluate(field,site,rays=budget.refinement_rays,seed=budget.seed)
    groups=q3_groups(field)
    history=[]
    rng=np.random.default_rng(budget.seed+30)
    import json
    config=json.loads((outdir/'q2_search.json').read_text())
    pool=candidates(config['parameters'],site)
    pool_upper=evaluate(pool,site,rays=budget.refinement_rays,seed=budget.seed,interactions=False)
    pool_density=pool_upper.per_mirror_kw/pool.area
    # Uniform Q2 is the feasible incumbent; all accepted changes improve its ratio.
    for k in range(budget.q3_trials):
        g=k%6
        mask=groups==g
        if not mask.any():continue
        widths=field.widths.copy();heights=field.heights.copy();centers=field.centers.copy()
        kind=(k//6)%5
        step=.35 if k<budget.q3_trials*2//3 else .15
        if kind==0:heights[mask]-=step
        elif kind==1:widths[mask]-=step
        elif kind==2:centers[mask,2]+=(-.45 if (k//30)%2==0 else .45)
        elif kind==3:
            # Improve a group's area allocation with unequal widths/heights.
            heights[mask]+=step*(1 if rng.random()<.5 else -1)
            widths[mask]+=step*(1 if rng.random()<.5 else -1)
        else:
            centers[mask,2]+=.4
            heights[mask]-=step/2
        widths=np.clip(widths,2,8);heights=np.minimum(widths,np.clip(heights,2,8))
        centers[:,2]=np.clip(np.maximum(centers[:,2],heights/2+.1),2,6)
        trial=Field(centers,widths,heights,field.tower,'q3')
        if not validate_field(trial,site,whole_mirror=True)['valid']:continue
        e=evaluate(trial,site,rays=budget.refinement_rays,seed=budget.seed)
        # Small groups can free area but lose capacity. Reinvest only part of the
        # saved area in high-efficiency mirrors/groups with spare dimension headroom.
        if e.annual_power_kw<budget.target_kw and e.unit_power>best.unit_power:
            donors=np.where(~mask & (heights<np.minimum(widths,8)-.05))[0]
            if len(donors):
                ranking=donors[np.argsort(-e.per_mirror_kw[donors]/trial.area[donors])]
                amount=min(len(ranking),max(20,int((budget.target_kw-e.annual_power_kw)/(.15*widths[ranking].mean()*e.unit_power))+1))
                heights[ranking[:amount]]=np.minimum(widths[ranking[:amount]],heights[ranking[:amount]]+.15)
                centers[:,2]=np.maximum(centers[:,2],heights/2+.1)
                trial=Field(centers,widths,heights,field.tower,'q3')
                if validate_field(trial,site,whole_mirror=True)['valid']:
                    e=evaluate(trial,site,rays=budget.refinement_rays,seed=budget.seed)
        if e.annual_power_kw<budget.target_kw and e.unit_power>best.unit_power:
            trial,e=replenish(trial,e,pool,pool_density,site,budget)
        if e.annual_power_kw>=budget.target_kw:
            trial,e=trim_field(trial,site,budget.target_kw,budget.refinement_rays,budget.seed,e,passes=2)
        accepted=e.annual_power_kw>=budget.target_kw and e.unit_power>best.unit_power+1e-8
        history.append(dict(trial=k,group=g,kind=kind,unit=e.unit_power,power_kw=e.annual_power_kw,accepted=accepted))
        if accepted:
            field,best=trial,e
            groups=q3_groups(field)
            save_field(field,outdir/'q3_checkpoint.npz')
        if k%6==0 or accepted:
            progress(f'Q3 group trial {k+1}/{budget.q3_trials}: best {best.unit_power:.6f}, accepted={accepted}',flush=True)
    # Discrete exchange search on actual interacting losses, not static ranking.
    for k in range(8):
        distance=cKDTree(field.centers[:,:2]).query(pool.centers[:,:2])[0]
        available=np.where(distance>.1)[0]
        if not len(available):break
        source=np.argsort(best.per_mirror_kw/field.area)[k%min(8,len(field))]
        destination=available[np.argsort(-pool_density[available])[k%min(4,len(available))]]
        centers=field.centers.copy();centers[source,:2]=pool.centers[destination,:2]
        trial=field.copy(centers=centers)
        if not validate_field(trial,site,whole_mirror=True)['valid']:continue
        e=evaluate(trial,site,rays=budget.refinement_rays,seed=budget.seed)
        accepted=e.annual_power_kw>=budget.target_kw and e.unit_power>best.unit_power
        history.append(dict(stage='position_exchange',trial=k,unit=e.unit_power,power_kw=e.annual_power_kw,accepted=accepted))
        if accepted:field,best=trial,e
    # A small tower pattern search is permitted in Q3 too.
    for delta in ([0,-4],[0,4],[-3,0],[3,0]):
        trial=field.copy(tower=field.tower+delta)
        if not validate_field(trial,site,whole_mirror=True)['valid']:continue
        e=evaluate(trial,site,rays=budget.refinement_rays,seed=budget.seed)
        accepted=e.annual_power_kw>=budget.target_kw and e.unit_power>best.unit_power
        history.append(dict(stage='tower_pattern',delta=delta,unit=e.unit_power,power_kw=e.annual_power_kw,accepted=accepted))
        if accepted:field,best=trial,e
    # Individual mirror refinement near the power boundary, genuinely heterogeneous.
    # Use full geometric reevaluation; area scaling alone is not a valid optical model.
    for k in range(12):
        density=best.per_mirror_kw/field.area
        eligible=np.where(field.heights>2.15)[0]
        if not len(eligible):break
        i=eligible[np.argsort(density[eligible])[k%min(len(eligible),24)]]
        step=min(.3,max(.025,(best.annual_power_kw-budget.target_kw)/(field.widths[i]*max(density[i],.1))))
        h=field.heights.copy();h[i]=max(2,h[i]-step)
        trial=field.copy(heights=h)
        e=evaluate(trial,site,rays=budget.refinement_rays,seed=budget.seed)
        accepted=e.annual_power_kw>=budget.target_kw and e.unit_power>best.unit_power
        history.append(dict(stage='individual',trial=k,mirror=int(i),unit=e.unit_power,power_kw=e.annual_power_kw,accepted=accepted))
        if accepted:field,best=trial,e
    field,best=trim_field(field,site,budget.target_kw,budget.refinement_rays,budget.seed,best)
    field.name='q3'
    save_field(field,outdir/'q3.npz')
    write_json(history,outdir/'q3_search_history.json')
    write_json({'seconds':perf_counter()-start,'unit':best.unit_power,'power_kw':best.annual_power_kw,
                'groups':6,'distinct_dimensions':len(np.unique(np.c_[field.widths,field.heights],axis=0)),
                'distinct_heights':len(np.unique(field.centers[:,2])),
                'constraints':validate_field(field,site,whole_mirror=True)},outdir/'q3_search.json')
    return field,best
