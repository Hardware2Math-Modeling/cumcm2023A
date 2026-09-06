"""Bounded audit of the existing paper designs; does not replace them."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from heliostat.io import load_field,save_field,write_json
from heliostat.model import Site,validate_field
from heliostat.optimize import candidates
from heliostat.verification import evaluation_task

OUT=ROOT/'tmp/search_diagnostic'


def score(task):
    label,f,rays,seed=task
    c=validate_field(f,whole_mirror=True)
    if not c['valid']:return dict(label=label,valid=False,constraints=c)
    e=evaluation_task((f,Site(),rays,seed))
    return dict(label=label,valid=True,count=len(f),area=float(f.area.sum()),
                power_MW=e.annual_power_kw/1000,unit=e.unit_power,rays=rays,seed=seed,
                efficiencies=e.rows().mean(axis=0),constraints=c)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    f=load_field(ROOT/'final/data/q3.npz');r=np.linalg.norm(f.centers[:,:2]-f.tower,axis=1)
    low=float(f.centers[:,2].min());elevated=f.centers[:,2]>low+.01
    designs={'q3_baseline':f}
    for height in [5.,5.5,6.]:
        cc=f.centers.copy();cc[elevated,2]=height
        designs[f'outer_z_{height}']=f.copy(centers=cc)
    t=(r-r.min())/(r.max()-r.min())
    for power in [.5,1.,2.]:
        for edge in [4.4951042632295275,6.]:
            cc=f.centers.copy();cc[:,2]=low+(edge-low)*t**power
            designs[f'smooth_p{power}_edge{edge:.3f}']=f.copy(centers=cc)
    for step in [-.15,.15]:
        hh=f.heights.copy();hh[elevated]=np.minimum(f.widths[elevated],hh[elevated]+step)
        designs[f'outer_mirror_height_{step:+.2f}']=f.copy(heights=hh)
    # Reverse only the twelve final individual perturbations, using the checkpoint
    # written at the last accepted group move and matching centers by identity.
    group=load_field(ROOT/'final/data/q3_checkpoint.npz')
    from scipy.spatial import cKDTree
    dist,ids=cKDTree(group.centers[:,:2]).query(f.centers[:,:2])
    assert dist.max()<1e-8
    designs['without_individual_shrink']=f.copy(heights=group.heights[ids])
    tasks=[]
    for label,field in designs.items():
        save_field(field,OUT/f'{label}.npz')
        tasks.append((label,field,1024,915721))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=[]
        for result in pool.map(score,tasks):
            results.append(result);print('HEIGHT',result['label'],result.get('power_MW'),result.get('unit'),flush=True)
            write_json(results,OUT/'height_screen.json')
    p=np.array(json.loads((ROOT/'final/data/q2_search.json').read_text())['parameters'])
    tasks=[]
    for y in [-160.,-120.,-80.,-40.,0.,40.,80.]:
        q=p.copy();q[1]=(y+245)/450
        field=candidates(q)
        tasks.append((f'tower_y_{y:+.0f}',field,256,915721))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=[]
        for result in pool.map(score,tasks):
            results.append(result);print('TOWER',result['label'],result.get('power_MW'),result.get('unit'),flush=True)
            write_json(results,OUT/'tower_fixed_family_screen.json')


if __name__=='__main__':main()
