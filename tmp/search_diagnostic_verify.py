"""Independent paired checks of selected audit perturbations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from concurrent.futures import ProcessPoolExecutor
from heliostat.io import load_field,write_json
from heliostat.model import Site
from heliostat.verification import evaluation_task,uncertainty
import numpy as np


if __name__=='__main__':
    out=ROOT/'tmp/search_diagnostic'
    labels=['q3_baseline','outer_z_6.0','without_individual_shrink']
    seeds=[916718,917715,918712]
    rows=[];tasks=[]
    for label in labels:
        f=load_field(out/f'{label}.npz')
        tasks.extend((f,Site(),4096,s) for s in seeds)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index,e in enumerate(pool.map(evaluation_task,tasks)):
            row=dict(label=labels[index//len(seeds)],rays=4096,seed=seeds[index%len(seeds)],
                     power_MW=e.annual_power_kw/1000,unit=e.unit_power)
            rows.append(row);print(row,flush=True)
            write_json(rows,out/'paired_verification_rows.json')
    base=np.array([x['power_MW']*1000 for x in rows if x['label']==labels[0]])
    summary={}
    for label in labels:
        pw=np.array([x['power_MW']*1000 for x in rows if x['label']==label])
        summary[label]=dict(power=uncertainty(pw),paired_difference_kw=uncertainty(pw-base))
        print(label,summary[label],flush=True)
    write_json(summary,out/'paired_verification.json')
