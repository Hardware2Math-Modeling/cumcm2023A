"""Reproduce the paper experiment from the repository's documented warm start.

Run: /opt/miniconda3/envs/tf2/bin/python final/reproduce.py
Physical formulas are unchanged; search refinements are in heliostat.optimize/refine.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
import hashlib
import json
import time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from heliostat.model import Site, validate_field
from heliostat.optics import evaluate
from heliostat.optimize import (SearchBudget, candidates, decode, select_design,
                                optimize_q3, trim_field, tower_phase_search)
from heliostat.verification import (replicated, certified, evaluation_task,
                                    uncertainty, legacy_field)
from heliostat.io import (load_attachment, load_field, save_field, write_json,
                          write_metrics, write_instantaneous, input_hashes, HEADERS)

OUT = Path(__file__).resolve().parent / 'data'
SEEDS = [1915721, 1916718, 1917715, 1918712]
WARM = np.array([.5, .5449981933853429, .8483680877431757, 1., 0., 0.,
                 .0000011923999998497692, .1122756431127115, .6,
                 .014215868393180123, .5, .3978557525038477])


def q2_search(site, budget):
    """Multi-start tower/phase search followed by local refinement for Q2."""
    result = select_design(WARM, site, budget, 512)
    if result is None:
        raise RuntimeError('Warm-start capacity check failed')
    field, best = result
    p = WARM.copy()
    history = [dict(stage='warm_start', parameters=p, power_kw=best.annual_power_kw,
                    unit=best.unit_power, count=len(field), accepted=True)]
    print('Q2 warm:', best.annual_power_kw, best.unit_power, len(field), flush=True)
    p,field,best,restarts=tower_phase_search(p,site,budget)
    history.extend(restarts)
    # Paired coordinate moves, decreasing scale. All trials use all 60 times.
    for scale in (.02, .007):
        for dim in (1, 2, 3, 4, 6, 7, 8, 9):
            batch=[]
            for sign in (-1, 1):
                q=p.copy();q[dim]=np.clip(q[dim]+sign*scale,0,1)
                if np.array_equal(q,p):continue
                batch.append(q)
            for q in batch:
                trial=select_design(q,site,budget,256)
                if trial is None:
                    history.append(dict(parameters=q,stage='coordinate',accepted=False,
                                        reason='insufficient_capacity'))
                    continue
                f,e=trial
                # Compare promising moves at the same higher quadrature fidelity.
                if e.unit_power>best.unit_power:
                    fine=select_design(q,site,budget,512)
                    if fine is None:continue
                    f,e=fine
                accepted=e.unit_power>best.unit_power and e.annual_power_kw>=budget.target_kw
                history.append(dict(stage='coordinate',parameters=q,power_kw=e.annual_power_kw,
                                    unit=e.unit_power,count=len(f),accepted=accepted))
                if accepted:p,field,best=q,f,e
            print('Q2 refinement:',scale,dim,best.annual_power_kw,best.unit_power,flush=True)
            write_json(history,OUT/'q2_search_history.json')
    field.name='q2'
    save_field(field,OUT/'q2.npz')
    write_json(dict(parameters=p,decoded=decode(p),budget=budget,power_kw=best.annual_power_kw,
                    unit=best.unit_power,warm_start=WARM),OUT/'q2_search.json')
    return field


def verify(site,budget):
    fields={'q1':load_attachment(), 'q2':load_field(OUT/'q2.npz'),
            'q3':load_field(OUT/'q3.npz')}
    p=json.loads((OUT/'q2_search.json').read_text())['parameters']
    pool=candidates(p,site)
    audit={};evals={}
    for name,f in fields.items():
        print('Final independent verification:',name,flush=True)
        if name=='q1':e,stats=replicated(f,site,4096,SEEDS,4)
        else:f,e,stats=certified(f,pool,site,budget,4096,SEEDS)
        check=validate_field(f,site,uniform=name=='q2',whole_mirror=name!='q1')
        assert check['valid'],check
        fields[name]=f;evals[name]=e
        save_field(f,OUT/f'{name}.npz')
        np.savez_compressed(OUT/f'{name}_optics.npz',metrics=e.metrics,power_kw=e.power_kw,
                            atmospheric=e.atmospheric,area=e.area,time_indices=e.time_indices)
        audit[name]=dict(uncertainty=stats,constraints=check,power_MW=e.annual_power_kw/1000,
                         design_sha256=hashlib.sha256((OUT/f'{name}.npz').read_bytes()).hexdigest(),
                         unit_power_kW_m2=e.unit_power,annual=e.rows().mean(axis=0),
                         monthly=e.monthly(),tower_xy=f.tower,
                         width_range_m=[f.widths.min(),f.widths.max()],
                         height_range_m=[f.heights.min(),f.heights.max()],
                         installation_range_m=[f.centers[:,2].min(),f.centers[:,2].max()])
        print(name,audit[name]['power_MW'],e.unit_power,stats['lower_95_one_sided_kw'],flush=True)
        write_json(audit,OUT/'verification_progress.json')
        if name!='q1':
            with (OUT/f'result{name[-1]}.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.writer(stream);writer.writerow(HEADERS)
                for i,(c,w,h) in enumerate(zip(f.centers,f.widths,f.heights),1):
                    writer.writerow([*f.tower,i,w,h,*c])
    dif=np.array(audit['q3']['uncertainty']['replicate_kw'])/fields['q3'].area.sum()-np.array(audit['q2']['uncertainty']['replicate_kw'])/fields['q2'].area.sum()
    audit['paired_q3_minus_q2']=uncertainty(dif)
    audit['paired_q3_minus_q2']['units']='kW/m2'
    audit['q3_improves_q2']=bool(evals['q3'].unit_power>evals['q2'].unit_power)
    previous=ROOT/'final/audit/before_revision/q3.npz'
    if previous.exists():
        old=load_field(previous)
        old_e,old_stats=replicated(old,site,4096,SEEDS,budget.workers)
        paired=np.array(audit['q3']['uncertainty']['replicate_kw'])/fields['q3'].area.sum()-np.array(old_stats['replicate_kw'])/old.area.sum()
        audit['revision_q3_comparison']=dict(old_power_MW=old_e.annual_power_kw/1000,
            old_unit_power=old_e.unit_power,old_area_m2=float(old.area.sum()),old_uncertainty=old_stats,
            paired_unit_gain=uncertainty(paired),relative_gain_percent=100*(evals['q3'].unit_power/old_e.unit_power-1))
        print('Q3 revision comparison:',audit['revision_q3_comparison']['relative_gain_percent'],flush=True)
    audit['assumptions']=site
    audit['sampling']=dict(rays=4096,replicates=4,seeds=SEEDS)
    audit['input_sha256']=input_hashes()
    sources=[*sorted((ROOT/'heliostat').glob('*.py')),ROOT/'heliostat/raytrace.cpp',Path(__file__),ROOT/'final/reoptimize.py']
    audit['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(audit,OUT/'verification.json')
    write_metrics(evals,OUT/'metrics.csv');write_instantaneous(evals,OUT/'instantaneous.csv')
    assert audit['q3_improves_q2'],'Independent Q3 improvement did not survive'
    conv=[]
    for name,f in fields.items():
        tasks=[(f,site,n,SEEDS[0]) for n in (128,512,1024,2048,4096)]
        with ProcessPoolExecutor(max_workers=4) as executor:
            for task,e in zip(tasks,executor.map(evaluation_task,tasks)):
                conv.append(dict(problem=name,rays=task[2],power_MW=e.annual_power_kw/1000,
                                 relative_error_percent=100*(e.annual_power_kw/evals[name].annual_power_kw-1)))
        print('Convergence completed:',name,flush=True)
    write_json(conv,OUT/'convergence.json')
    scenarios=[('shaft_0m',replace(site,tower_radius=0)),('shaft_3m',replace(site,tower_radius=3)),
               ('sun_4.2mrad',replace(site,solar_radius=.0042)),('sun_5.1mrad',replace(site,solar_radius=.0051)),
               ('slope_0.5mrad',replace(site,slope_error=.0005)),('slope_1mrad',replace(site,slope_error=.001))]
    sensitivity=[]
    for name,f in fields.items():
        base=evaluation_task((f,site,1024,SEEDS[0]))
        with ProcessPoolExecutor(max_workers=4) as executor:
            results=executor.map(evaluation_task,[(f,s,1024,SEEDS[0]) for _,s in scenarios])
            for (label,s),e in zip(scenarios,results):
                sensitivity.append(dict(problem=name,scenario=label,power_MW=e.annual_power_kw/1000,
                                        paired_change_percent=100*(e.annual_power_kw/base.annual_power_kw-1)))
        print('Sensitivity completed:',name,flush=True)
    write_json(sensitivity,OUT/'sensitivity.json')
    legacy=legacy_field();le=evaluation_task((legacy,site,2048,SEEDS[0]))
    write_json(dict(count=len(legacy),power_MW=le.annual_power_kw/1000,unit=le.unit_power),OUT/'legacy_comparison.json')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=input_hashes();start=time.time();site=Site()
    budget=SearchBudget(target_kw=60150.,refinement_rays=512,q3_trials=48,seed=202309,workers=4)
    if not (OUT/'q2.npz').exists():q2_search(site,budget)
    if not (OUT/'q3.npz').exists():optimize_q3(load_field(OUT/'q2.npz'),OUT,site,budget)
    verify(site,budget)
    assert before==input_hashes()
    write_json(dict(seconds=time.time()-start,input_unchanged=True,budget=budget),OUT/'run.json')


if __name__=='__main__':main()
