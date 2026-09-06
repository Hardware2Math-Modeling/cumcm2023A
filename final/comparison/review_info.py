"""Read-only comparison of info designs with our final fields.

Outputs go here; neither source designs nor either paper is overwritten.
1024-ray comparisons diagnose model/layout differences, not final certification.
"""
from pathlib import Path
import sys,json,subprocess,hashlib,tempfile
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from heliostat.model import Field,Site,validate_field
from heliostat.io import load_field,write_json
from heliostat.verification import evaluation_task

OUT=Path(__file__).resolve().parent
RAYS=1024;SEED=1915721


def read_other(q):
    p=ROOT/f'info/data/q{q}.txt';header=p.read_text().splitlines()[0].split()
    a=np.loadtxt(p,skiprows=1)
    assert len(a)==int(header[0])
    return Field(a[:,[0,1,4]],a[:,2],a[:,3],np.array(header[1:3],float),f'info_q{q}')


def details(f):
    site=validate_field(f,whole_mirror=True)
    centers=validate_field(f,whole_mirror=False)
    radius=np.linalg.norm(f.centers[:,:2]-f.tower,axis=1)
    outer=np.linalg.norm(f.centers[:,:2],axis=1)+f.radii>350+1e-8
    inner=radius-f.radii<100-1e-8
    sizes,counts=np.unique(np.c_[f.widths,f.heights],axis=0,return_counts=True)
    return dict(centers=centers,whole_mirror=site,tower=f.tower,
                outer_violations=int(outer.sum()),inner_violations=int(inner.sum()),
                strict_violations=int((outer|inner).sum()),
                radial_rings=len(np.unique(np.round(radius,5))),
                dimension_types=sizes,dimension_counts=counts,
                installation_levels=np.unique(f.centers[:,2],return_counts=True),
                radius_max=float(radius.max()))


def main():
    paths=[*sorted((ROOT/'info').rglob('*')),ROOT/'final/paper.pdf',ROOT/'final/data/q2.npz',ROOT/'final/data/q3.npz']
    protected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    fields={f'{owner}_q{q}':read_other(q) if owner=='info' else load_field(ROOT/f'final/data/q{q}.npz')
            for owner in ('info','ours') for q in (1,2,3)}
    geometry={name:details(f) for name,f in fields.items()}
    write_json(geometry,OUT/'geometry.json')
    tasks=[];labels=[]
    for name,f in fields.items():
        for mast in ((2.,3.5) if name.startswith('info') else (3.5,)):
            labels.append(dict(design=name,evaluator='ours',mast_radius=mast,variant='raw'))
            tasks.append((f,replace(Site(),tower_radius=mast),RAYS,SEED))
        if name.startswith('info') and name[-1]!='1':
            keep=(np.linalg.norm(f.centers[:,:2],axis=1)+f.radii<=350+1e-8)&(np.linalg.norm(f.centers[:,:2]-f.tower,axis=1)-f.radii>=100-1e-8)
            labels.append(dict(design=name,evaluator='ours',mast_radius=2.,variant='strict_clip'))
            tasks.append((f.subset(np.flatnonzero(keep)),Site(),RAYS,SEED))
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as executor:
        for label,task,e in zip(labels,tasks,executor.map(evaluation_task,tasks)):
            row=dict(**label,rays=RAYS,seed=SEED,count=len(task[0]),area=float(task[0].area.sum()),
                     power_MW=e.annual_power_kw/1000,unit=e.unit_power,annual=e.rows().mean(axis=0))
            rows.append(row);print(row['design'],row['variant'],row['mast_radius'],row['power_MW'],row['unit'],flush=True)
            write_json(rows,OUT/'cross_evaluation.json')
    # Compile readable source into a temporary native executable (the supplied
    # .exe files target Windows); no changes to the imported info directory.
    with tempfile.TemporaryDirectory(prefix='heliostat-info-') as directory:
        tmp=Path(directory);exe=tmp/'trace'
        subprocess.run(['c++','-O3','-std=c++17',str(ROOT/'info/trace.cpp'),'-o',str(exe)],check=True)
        for name,f in fields.items():
            inp=tmp/f'{name}.txt';out=tmp/f'{name}.csv'
            np.savetxt(inp,np.c_[f.centers[:,:2],f.widths,f.heights,f.centers[:,2]],
                       header=f'{len(f)} {f.tower[0]:.17g} {f.tower[1]:.17g}',comments='',fmt='%.17g')
            subprocess.run([str(exe),str(inp),str(out),str(RAYS),'20260905','.00465','3.5','1'],check=True)
            t=np.genfromtxt(out,delimiter=',',names=True)
            row=dict(design=name,evaluator='info_source',mast_radius=3.5,variant='raw',rays=RAYS,seed=20260905,
                     count=len(f),area=float(f.area.sum()),power_MW=float(t['power_MW'].mean()),unit=float(t['unit_kW_m2'].mean()))
            rows.append(row);print(row,flush=True);write_json(rows,OUT/'cross_evaluation.json')
    for path,digest in protected.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    write_json(dict(input_sha256=protected,rays=RAYS,unchanged=True,
                    purpose='Same evaluator and assumptions comparison; single scramble, not certification'),OUT/'provenance.json')


if __name__=='__main__':main()
