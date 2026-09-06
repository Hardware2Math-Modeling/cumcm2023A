"""Auditable continuation search using complete, bidirectional neighbourhoods."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .model import validate_field
from .optimize import candidates, q3_groups, replenish, trim_field
from .optics import evaluate
from .verification import evaluation_task


def score_move(args):
    field,site,budget=args
    return evaluation_task((field,site,budget.refinement_rays,budget.seed))


def translated_field(field,delta,site):
    """Move mirror coordinates with the tower, then remove boundary violations."""
    centers=field.centers.copy();centers[:,:2]+=delta
    trial=field.copy(centers=centers,tower=field.tower+delta)
    radius=np.hypot(trial.widths,trial.heights)/2
    return trial.subset(np.flatnonzero(np.linalg.norm(centers[:,:2],axis=1)+radius<=site.field_radius))


def grouped_moves(field,step):
    groups=q3_groups(field)
    for group in np.unique(groups):
        mask=groups==group
        for variable in ('width','height','z'):
            for sign in (-1,1):
                c=field.centers.copy();w=field.widths.copy();h=field.heights.copy()
                if variable=='width':w[mask]=np.clip(w[mask]+sign*step,2,8)
                elif variable=='height':h[mask]=np.clip(h[mask]+sign*step,2,8)
                else:c[mask,2]=np.clip(c[mask,2]+sign*step,2,6)
                # Permit increased mirror height to raise its supporting centre.
                if variable=='height':c[:,2]=np.maximum(c[:,2],h/2+.1)
                yield dict(group=int(group),variable=variable,delta=sign*step),field.copy(centers=c,widths=w,heights=h)


def same_field(a,b):
    return all(np.array_equal(x,y) for x,y in zip((a.centers,a.widths,a.heights,a.tower),
                                                (b.centers,b.widths,b.heights,b.tower)))


def neighbourhood(field,best,moves,site,budget,stage,executor,history):
    """Score all neighbours against the same incumbent; filter feasibility first."""
    rows=[];pending=[]
    for metadata,trial in moves:
        row=dict(stage=stage,**metadata,accepted=False,rays=budget.refinement_rays)
        history.append(row)
        if same_field(field,trial):row['reason']='unchanged';continue
        check=validate_field(trial,site,whole_mirror=True)
        if not check['valid']:row.update(reason='geometry',violations=check['violations']);continue
        rows.append((row,trial));pending.append((trial,site,budget))
    winner=None
    for (row,trial),e in zip(rows,executor.map(score_move,pending)):
        row.update(unit=e.unit_power,power_kw=e.annual_power_kw,count=len(trial))
        if e.annual_power_kw<budget.target_kw:row['reason']='insufficient_capacity'
        elif e.unit_power<=best.unit_power+1e-8:row['reason']='no_improvement'
        elif winner is None or e.unit_power>winner[2].unit_power:winner=row,trial,e
    if winner is not None:
        winner[0]['accepted']=True
        return winner[1],winner[2],True
    return field,best,False


def standardize_dimensions(field,best,site,budget,history,tolerance=2e-4,parameters=None):
    """Compare six regional standard mirror sizes with a stated epsilon tolerance."""
    groups=q3_groups(field);w=field.widths.copy();h=field.heights.copy()
    for g in np.unique(groups):
        mask=groups==g
        sizes,counts=np.unique(np.c_[w[mask],h[mask]],axis=0,return_counts=True)
        w[mask],h[mask]=sizes[np.argmax(counts)]
    c=field.centers.copy();c[:,2]=np.maximum(c[:,2],h/2+.1)
    trial=field.copy(centers=c,widths=w,heights=h)
    row=dict(stage='regional_standardization',relative_tolerance=tolerance,accepted=False)
    history.append(row)
    check=validate_field(trial,site,whole_mirror=True)
    if not check['valid']:row['reason']='geometry';return field,best
    e=score_move((trial,site,budget))
    if e.annual_power_kw<budget.target_kw and e.unit_power>=best.unit_power*(1-tolerance):
        # Recover a small reserve deficit with regional size increments before
        # introducing extra low-yield mirrors. This retains a short size catalogue.
        restores=[]
        for group in np.unique(groups):
            mask=groups==group
            for increment in (.005,.01,.025):
                if np.any(trial.heights[mask]+increment>trial.widths[mask]):continue
                heights=trial.heights.copy();heights[mask]+=increment
                centers=trial.centers.copy();centers[:,2]=np.maximum(centers[:,2],heights/2+.1)
                restored=trial.copy(centers=centers,heights=heights)
                if validate_field(restored,site,whole_mirror=True)['valid']:
                    restores.append((int(group),increment,restored))
        winner=None;rows=[]
        with ProcessPoolExecutor(max_workers=budget.workers) as executor:
            for (group,increment,f),er in zip(restores,executor.map(score_move,[(f,site,budget) for _,_,f in restores])):
                rows.append(dict(group=group,increment=increment,unit=er.unit_power,power_kw=er.annual_power_kw))
                if er.annual_power_kw>=budget.target_kw and (winner is None or er.unit_power>winner[1].unit_power):
                    winner=f,er,group,increment
        row['capacity_restoration_trials']=rows
        if winner is not None:
            trial,e,group,increment=winner;w,h=trial.widths,trial.heights
            row['regional_capacity_restoration']=dict(group=group,increment=increment)
    if e.annual_power_kw<budget.target_kw and e.unit_power>=best.unit_power*(1-tolerance) and parameters is not None:
        # Standardizing a few mirrors can spend a small part of the reserve.
        # Restore it explicitly instead of rejecting an otherwise useful design.
        q=np.array(parameters).copy();q[0]=field.tower[0]/160+.5;q[1]=(field.tower[1]+245)/450
        pool=candidates(q,site)
        if pool is not None:
            upper=evaluate(pool,site,rays=budget.refinement_rays,seed=budget.seed,interactions=False)
            trial,e=replenish(trial,e,pool,upper.per_mirror_kw/pool.area,site,budget)
            row['added_mirrors']=len(trial)-len(field)
            w,h=trial.widths,trial.heights
    row.update(unit=e.unit_power,power_kw=e.annual_power_kw,
               before_types=len(np.unique(np.c_[field.widths,field.heights],axis=0)),
               after_types=len(np.unique(np.c_[w,h],axis=0)),
               relative_loss=1-e.unit_power/best.unit_power)
    row['accepted']=bool(e.annual_power_kw>=budget.target_kw and e.unit_power>=best.unit_power*(1-tolerance)
                         and row['after_types']<row['before_types'])
    return (trial,e) if row['accepted'] else (field,best)


def polish_q3(field,parameters,site,budget,progress=print,checkpoint=None,max_sweeps=2):
    """Complete group sweeps, coupled tower relocation, then regional standardization.

    A sweep always covers all 6 groups x 3 variables x 2 directions. Step size
    changes only between sweeps. Reaching max_sweeps is reported as a budget stop,
    never as convergence. Every optical comparison uses identical rays and seed.
    """
    history=[];best=score_move((field,site,budget));statuses=[]
    with ProcessPoolExecutor(max_workers=budget.workers) as executor:
        for step in (.15,.05):
            for sweep in range(max_sweeps):
                moves=[(dict(**m,step=step,sweep=sweep),f) for m,f in grouped_moves(field,step)]
                field,best,improved=neighbourhood(field,best,moves,site,budget,'group_pattern',executor,history)
                progress(f'Q3 complete sweep: step={step}, sweep={sweep+1}, improved={improved}, J={best.unit_power:.6f}',flush=True)
                if checkpoint:checkpoint(field,history)
                if not improved:break
            statuses.append(dict(step=step,sweeps=sweep+1,stop='budget' if improved else 'no_improvement'))
        moves=[]
        for delta in ([0,-4],[0,4],[-3,0],[3,0]):
            trial=translated_field(field,np.array(delta),site)
            q=np.array(parameters).copy();q[0]=trial.tower[0]/160+.5;q[1]=(trial.tower[1]+245)/450
            pool=candidates(q,site)
            if pool is not None:
                e=score_move((trial,site,budget))
                if e.annual_power_kw<budget.target_kw:
                    upper=evaluate(pool,site,rays=budget.refinement_rays,seed=budget.seed,interactions=False)
                    trial,e=replenish(trial,e,pool,upper.per_mirror_kw/pool.area,site,budget)
            moves.append((dict(delta=delta),trial))
        field,best,_=neighbourhood(field,best,moves,site,budget,'coupled_tower',executor,history)
    field,best=trim_field(field,site,budget.target_kw,budget.refinement_rays,budget.seed,best)
    field,best=standardize_dimensions(field,best,site,budget,history,parameters=parameters)
    history.append(dict(stage='stop',group_sweeps=statuses,global_optimality_claim=False))
    if checkpoint:checkpoint(field,history)
    return field,best,history
