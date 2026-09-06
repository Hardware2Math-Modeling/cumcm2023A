"""Q3: per-mirror discrete height allocation with neighbours frozen in probes,
followed by full coupled ray tracing. No external data or network calls."""
from pathlib import Path
import numpy as np,subprocess,json
from solve import ROOT,WORK,hex_field,trace,write_design
PROBE=ROOT/'trace_probe.exe'

def allocation(a,tower,label,samples=128,target=60.15,probe_heights=True):
    w=float(a[0,2]); hs=sorted(set([h for h in [2.,3.,4.,4.5,5.,5.5,6.,6.25,6.5,6.75,7.,7.5,8.] if h<=w]+[w]))
    design=WORK/(label+'_input.txt');write_design(design,a,tower)
    energies=[np.zeros(len(a))];areas=[np.zeros(len(a))];options=[(0.,0.)]
    zchoices=[0] if not probe_heights else [0,-1,6.]
    for h in hs:
      for z in zchoices:
        out=WORK/f'{label}_h{h:g}_z{z:g}.csv'
        p=subprocess.run([str(PROBE),str(design),str(out),str(samples),'20260905','.00465','3.5','8',str(h),str(z)],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr)
        per=np.genfromtxt(str(out)+'.mirrors.csv',delimiter=',',names=True)
        energies.append(per['unit_kW_m2']*w*h);areas.append(np.full(len(a),w*h));options.append((h,z))
    P=np.array(energies);A=np.array(areas)
    def choose(lam):
      idx=np.argmax(P-lam*A,axis=0);return idx,float(P[idx,np.arange(len(a))].sum()/1000)
    low=0.;high=1.5;idx,capacity=choose(low)
    if capacity<target:
      print(label,'insufficient probe capacity',capacity,flush=True)
    else:
      for _ in range(45):
        mid=(low+high)/2;ix,p=choose(mid)
        if p>=target:low=mid;idx=ix
        else:high=mid
    out=a.copy()
    for i,j in enumerate(idx):
      h,z=options[j];out[i,3]=h
      if z<0:out[i,4]=max(2.,h/2+.05)
      elif z>0:out[i,4]=max(z,h/2+.05)
    active=out[:,3]>=2
    r=trace(out[active],tower,label+'_coupled',512)
    write_design(WORK/(label+'_final.txt'),out[active],tower)
    record={'label':label,'N':r['N'],'area':r['area'],'power':r['power'],'unit':r['unit'],'probe_predicted_MW':float(P[idx,np.arange(len(a))].sum()/1000),'lambda':low,'w':w,'tower':list(tower),'heights':{str(h):int(np.sum(out[:,3]==h)) for h in np.unique(out[:,3])}}
    print(json.dumps(record),flush=True)
    return out,record

def run():
    records=[];best=None
    for i,w in enumerate([6.25,6.5,6.75,7.]):
      tower=(0,-60);a=hex_field(w,w,w/2+.05,-60,30,setback=False)
      b,r=allocation(a,tower,f'adapt{i}_0',128,60.2,False);records.append(r)
      b,r=allocation(b,tower,f'adapt{i}_1',128,60.2,False);records.append(r)
      if r['power']>=60 and (best is None or r['unit']>best[1]['unit']):best=(b,r)
    if best is not None:
      b,r=best
      b,r=allocation(b,r['tower'],'adapt_height',128,60.2,True);records.append(r)
    (ROOT/'adaptive.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':run()
