"""Independent scrambles, fidelity convergence, constraints and sensitivity audit."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import platform
import sys
import hashlib
import numpy as np
from scipy.stats import t as student_t
from .model import Field,Site,validate_field
from .optics import evaluate,Evaluation
from .optimize import candidates,replenish
from .io import (ROOT,PROBLEM,load_attachment,load_field,save_field,write_design,
                 write_json,write_metrics,write_instantaneous,input_hashes)


def aggregate(results):
    """Aggregate ENERGY numerators/denominators, not conditional ratios."""
    first=results[0]
    raw=np.stack([e.metrics for e in results])
    cosine=raw[:,:,:,0].mean(axis=0)
    survive=(raw[:,:,:,0]*raw[:,:,:,1]).mean(axis=0)
    accepted=raw[:,:,:,6].mean(axis=0)
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a),where=b>0)
    metrics=np.stack([cosine,divide(survive,cosine),divide(accepted,survive),
                      divide((raw[:,:,:,0]*raw[:,:,:,3]).mean(axis=0),cosine),
                      divide((raw[:,:,:,0]*raw[:,:,:,4]).mean(axis=0),cosine),
                      divide((raw[:,:,:,0]*raw[:,:,:,5]).mean(axis=0),cosine),accepted],axis=-1)
    return Evaluation(metrics,np.mean([e.power_kw for e in results],axis=0),first.atmospheric,
                      first.area,first.time_indices,sum(e.ray_count for e in results),-1,first.reflectivity)


def uncertainty(powers):
    x=np.array(powers,float)
    if len(x)<2:raise ValueError('At least two independent scrambles are required')
    se=float(x.std(ddof=1)/np.sqrt(len(x)))
    mean=float(x.mean())
    return {'mean_kw':mean,'replicate_kw':x.tolist(),'standard_error_kw':se,
            'lower_95_one_sided_kw':mean-float(student_t.ppf(.95,len(x)-1))*se,
            'two_sided_95_halfwidth_kw':float(student_t.ppf(.975,len(x)-1))*se,
            'interpretation':'Approximate uncertainty of randomized quadrature only; excludes physical-model error.'}


def evaluation_task(args):
    field,site,rays,seed=args
    # Content-addressed cache makes independent verification resumable without
    # ever reusing a result for different geometry, assumptions or optical code.
    from dataclasses import asdict
    digest=hashlib.sha256()
    for a in (field.centers,field.widths,field.heights,field.tower):digest.update(a.tobytes())
    digest.update(json.dumps(asdict(site),sort_keys=True).encode())
    digest.update(f'{rays}:{seed}'.encode())
    for filename in ('model.py','optics.py','geometry.py','raytrace.cpp'):
        digest.update((ROOT/'heliostat'/filename).read_bytes())
    folder=ROOT/'build'/'evaluation_cache';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest.hexdigest()+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            return Evaluation(data['metrics'],data['power_kw'],data['atmospheric'],field.area,
                              np.arange(60),rays,seed,site.reflectivity)
    result=evaluate(field,site,rays=rays,seed=seed)
    import os
    temporary=path.with_name(path.stem+f'.{os.getpid()}.npz')
    np.savez_compressed(temporary,metrics=result.metrics,power_kw=result.power_kw,atmospheric=result.atmospheric)
    temporary.replace(path)
    return result


def replicated(field,site,rays,seeds,workers):
    with ProcessPoolExecutor(max_workers=min(workers,len(seeds))) as pool:
        values=list(pool.map(evaluation_task,[(field,site,rays,s) for s in seeds]))
    return aggregate(values),uncertainty([e.annual_power_kw for e in values])


def certified(field,pool,site,budget,rays,seeds,progress=print):
    for attempt in range(4):
        progress(f'Verifying {field.name}: {len(field)} mirrors × 60 times × {rays} rays × {len(seeds)} scrambles',flush=True)
        mean,stats=replicated(field,site,rays,seeds,budget.workers)
        if stats['lower_95_one_sided_kw']>=60000:
            return field,mean,stats
        progress(f"  {field.name}: lower estimate {stats['lower_95_one_sided_kw']/1000:.5f} MW; repairing capacity",flush=True)
        if pool is None:raise RuntimeError('Cannot restore rated power without unused candidate sites')
        upper=evaluate(pool,site,rays=512,seed=budget.seed,interactions=False)
        # Include the measured quadrature discrepancy in the search-side reserve.
        reserve=max(150.,60000-stats['lower_95_one_sided_kw']+100.)
        proposal=replace(budget,target_kw=mean.annual_power_kw+reserve)
        trial,trial_e=replenish(field,mean,pool,upper.per_mirror_kw/pool.area,site,proposal,rays=512)
        if len(trial)==len(field):raise RuntimeError('Capacity repair exhausted geometrically valid candidates')
        field=trial
    raise RuntimeError('Capacity still uncertified after four explicit repairs')


def legacy_field():
    # Reconstruct the old algorithm's fixed layout deterministically; never use
    # the newly optimized objective to pick its mirrors.
    x=np.arange(-350,350.001,13.01)
    xx,yy=np.meshgrid(x,x)
    xy=np.c_[xx.ravel(),yy.ravel()]
    radius=np.linalg.norm(xy,axis=1)
    xy=xy[(radius>=100)&(radius<=350)]
    f=Field(np.c_[xy,np.full(len(xy),6.)],8,8,[0,0],'legacy_q2')
    from .model import time_grid
    _,suns,dni=time_grid()
    target=suns[0]  # overwritten below; distance is time invariant
    powers=[]
    for sun,irradiance in zip(suns,dni):
        target,_,_,_,distance=f.geometry(sun)
        cosine=np.sqrt((1+target@sun)/2)
        atmospheric=.99321-.0001176*distance+1.97e-8*distance**2
        powers.append(irradiance*64*cosine*atmospheric*.95*.90*.92)
    p=np.mean(powers,axis=0);order=np.argsort(-p)
    n=np.searchsorted(np.cumsum(p[order]),60000)+1
    return f.subset(order[:n])


def verify_all(outdir,site,budget,rays=4096,replicates=4,progress=print):
    if replicates<2:raise ValueError('--replicates must be >= 2')
    seeds=[budget.seed+10000+997*k for k in range(replicates)]
    fields={'q1':load_attachment(),'q2':load_field(outdir/'q2.npz'),'q3':load_field(outdir/'q3.npz')}
    config=json.loads((outdir/'q2_search.json').read_text())
    pool=candidates(config['parameters'],site)
    evaluations={};audit={};convergence=[]
    for name,field in fields.items():
        if name=='q1':
            progress('Verifying fixed attachment Q1',flush=True)
            e,stats=replicated(field,site,rays,seeds,budget.workers)
        else:
            field,e,stats=certified(field,pool,site,budget,rays,seeds,progress)
        constraints=validate_field(field,site,uniform=name=='q2',whole_mirror=name!='q1')
        if not constraints['valid']:raise AssertionError(constraints)
        fields[name]=field;evaluations[name]=e
        audit[name]={'uncertainty':stats,'constraints':constraints,'power_MW':e.annual_power_kw/1000,
                     'unit_power_kW_m2':e.unit_power,'area_weighted_efficiencies':e.rows().mean(axis=0),
                     'tower_xy':field.tower,'width_range_m':[field.widths.min(),field.widths.max()],
                     'height_range_m':[field.heights.min(),field.heights.max()],
                     'installation_range_m':[field.centers[:,2].min(),field.centers[:,2].max()]}
        save_field(field,outdir/f'{name}.npz')
        for count in sorted(set([128,512,min(2048,rays),rays])):
            # Same independent seed, nested Sobol rules. Final uncertainty uses all scrambles.
            progress(f'  {name} convergence: {count} rays',flush=True)
            sample=evaluation_task((field,site,count,seeds[0]))
            convergence.append({'problem':name,'rays':count,'seed':seeds[0],
                                'power_MW':sample.annual_power_kw/1000,'unit_power':sample.unit_power,
                                'difference_from_final_mean_percent':100*(sample.annual_power_kw/e.annual_power_kw-1)})
        progress(f'  {name}: {e.annual_power_kw/1000:.6f} MW; {e.unit_power:.6f} kW/m²',flush=True)
        write_json(audit,outdir/'verification_progress.json')
    # Never assert that Q3 improves Q2 merely because it did at search fidelity.
    difference=np.array(audit['q3']['uncertainty']['replicate_kw'])/fields['q3'].area.sum()-np.array(audit['q2']['uncertainty']['replicate_kw'])/fields['q2'].area.sum()
    audit['q3_vs_q2_paired_unit_difference']=uncertainty(difference)
    audit['q3_vs_q2_paired_unit_difference']['units']='kW/m² (generic uncertainty keys retain _kw suffix)'
    audit['q3_improves_q2']=bool(evaluations['q3'].unit_power>=evaluations['q2'].unit_power)
    if not audit['q3_improves_q2']:
        write_json(audit,outdir/'verification_failed.json')
        raise RuntimeError('Q3 loses its improvement at independent fidelity; rerun/polish Q3, not publish a false improvement')
    sensitivity=[]
    scenarios=[('shaft_radius_0m',replace(site,tower_radius=0)),
               ('shaft_radius_3m',replace(site,tower_radius=3)),
               ('solar_radius_4.2mrad',replace(site,solar_radius=.0042)),
               ('solar_radius_5.1mrad',replace(site,solar_radius=.0051)),
               ('slope_error_0.5mrad',replace(site,slope_error=.0005)),
               ('slope_error_1.0mrad',replace(site,slope_error=.001))]
    srays=min(rays,1024)
    for name in ('q1','q2','q3'):
        baseline=evaluation_task((fields[name],site,srays,seeds[0]))
        with ProcessPoolExecutor(max_workers=budget.workers) as workers:
            results=workers.map(evaluation_task,[(fields[name],s,srays,seeds[0]) for _,s in scenarios])
            for (label,settings),result in zip(scenarios,results):
                sensitivity.append({'problem':name,'scenario':label,'power_MW':result.annual_power_kw/1000,
                                    'paired_change_percent':100*(result.annual_power_kw/baseline.annual_power_kw-1),
                                    'rays':srays,'seed':seeds[0]})
        progress(f'Sensitivity completed for {name}',flush=True)
    legacy=legacy_field();legacy_eval=evaluate(legacy,site,rays=min(rays,2048),seed=seeds[0])
    audit['legacy_q2_under_same_optical_model']={'count':len(legacy),'power_MW':legacy_eval.annual_power_kw/1000,
                                               'unit_power_kW_m2':legacy_eval.unit_power}
    for name in ('q2','q3'):write_design(fields[name],outdir/f'result{name[-1]}.xlsx')
    write_metrics(evaluations,outdir/'metrics.csv')
    write_instantaneous(evaluations,outdir/'instantaneous.csv')
    for name,e in evaluations.items():
        np.savez_compressed(outdir/f'{name}_optics.npz',metrics=e.metrics,power_kw=e.power_kw,
                            atmospheric=e.atmospheric,area=e.area,time_indices=e.time_indices)
    audit['sampling']={'rays_per_mirror_time_per_scramble':rays,'scrambles':replicates,'seeds':seeds,
                       'total_primary_rays':int(sum(len(f) for f in fields.values())*60*rays*replicates)}
    audit['assumptions']=site
    audit['input_sha256']=input_hashes()
    audit['environment']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}
    sources=[ROOT/'solve_mirror_field.py',*sorted((ROOT/'heliostat').glob('*.py')),ROOT/'heliostat'/'raytrace.cpp']
    audit['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(audit,outdir/'verification.json')
    write_json(convergence,outdir/'convergence.json')
    write_json(sensitivity,outdir/'sensitivity.json')
    from .plots import create_plots
    create_plots(fields,evaluations,convergence,outdir,site)
    return audit
