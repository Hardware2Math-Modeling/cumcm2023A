"""Additional numerical checks on info's frozen designs; no optimization."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess,shutil,time,sys
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=HERE/'data'
SEED=20260905

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    source=ROOT/'info/trace.cpp';cache=ROOT/'build/paper_trace';cache.mkdir(parents=True,exist_ok=True)
    exe=cache/('trace_'+sha(source)[:12])
    if not exe.exists():subprocess.run(['c++','-O3','-std=c++17',str(source),'-o',str(exe)],check=True)
    for q in (1,2,3):
        shutil.copy2(ROOT/f'info/data/q{q}.txt',DATA/f'q{q}.txt')
        for tail in ('_times.csv','_times.csv.mirrors.csv','_trace.json'):
            shutil.copy2(ROOT/f'final/info_figures/data/q{q}{tail}',DATA/f'q{q}{tail}')
    expected=json.loads((ROOT/'info/data/expected.json').read_text())
    tasks=[]
    for q in (1,2,3):
        for n in (1024,4096):tasks.append((q,n,SEED,.00465,3.5,'convergence'))
    for q in (2,3):
        for seed in (20260916,20261913,20262910,20263907):
            tasks.append((q,4096,seed,.00465,3.5,'independent'))
        for alpha,mast,label in ((.0042,3.5,'sun_4.2'),(.0051,3.5,'sun_5.1'),(.00465,0.,'shaft_0'),(.00465,2.,'shaft_2')):
            tasks.append((q,1024,SEED,alpha,mast,label))
    def one(task):
        q,n,seed,alpha,mast,label=task
        tag=f'q{q}_{label}_{n}_{seed}'
        out=DATA/(tag+'.csv');meta=DATA/(tag+'.json')
        key=dict(q=q,rays=n,seed=seed,alpha=alpha,mast=mast,input_sha256=sha(DATA/f'q{q}.txt'),source_sha256=sha(source))
        reuse=meta.exists() and out.exists() and json.loads(meta.read_text()).get('settings')==key
        if not reuse:
            print('Computing',tag,flush=True);start=time.monotonic()
            subprocess.run([str(exe),str(DATA/f'q{q}.txt'),str(out),str(n),str(seed),str(alpha),str(mast),'1'],check=True,capture_output=True)
            print('Finished',tag,round(time.monotonic()-start,1),'s',flush=True)
        a=np.loadtxt(DATA/f'q{q}.txt',skiprows=1)
        values=np.genfromtxt(out,delimiter=',',names=True)
        assert len(values)==60
        power=float(values['power_MW'].mean());unit=float(values['unit_kW_m2'].mean())
        assert abs(power*1000/np.prod(a[:,2:4],axis=1).sum()-unit)<1e-10
        row=dict(settings=key,label=label,power_MW=power,unit_kW_m2=unit,
                 difference_from_16384_percent=100*(power/expected[str(q)]['power_MW']-1))
        write(meta,row);return row
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(one,tasks):
            rows.append(row);write(DATA/'verification_runs.json',rows)
    stats={}
    for q in (2,3):
        x=np.array([r['power_MW'] for r in rows if r['settings']['q']==q and r['label']=='independent'])
        se=x.std(ddof=1)/np.sqrt(len(x))
        stats[str(q)]=dict(mean_MW=float(x.mean()),replicates_MW=x.tolist(),se_MW=float(se),
                          lower95_MW=float(x.mean()-t.ppf(.95,len(x)-1)*se))
    sensitivity=[]
    for row in rows:
        if row['label'] in ('convergence','independent'):continue
        q=row['settings']['q'];base=next(r for r in rows if r['settings']['q']==q and r['label']=='convergence' and r['settings']['rays']==1024)
        sensitivity.append(dict(q=q,scenario=row['label'],power_MW=row['power_MW'],
                                paired_change_percent=100*(row['power_MW']/base['power_MW']-1)))
    write(DATA/'verification_summary.json',dict(independent=stats,sensitivity=sensitivity,
            inference='Four fresh random shifts at 4096 rays. Student-t bound estimates numerical integration uncertainty only.',
            design_unchanged=all(sha(DATA/f'q{q}.txt')==sha(ROOT/f'info/data/q{q}.txt') for q in (1,2,3))))
    print('All verification completed',flush=True)

if __name__=='__main__':main()
