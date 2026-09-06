"""Reproduce the delivered designs offline, using the same geometric ray tracer.
Usage: python recompute.py --samples 16384 --question all
Dependencies: numpy, openpyxl. No network calls or package installation.
"""
from pathlib import Path
import argparse,csv,json,os,subprocess
import numpy as np
from openpyxl import Workbook

ROOT=Path(__file__).resolve().parent
def constraints(a,tower):
    n=len(a);xy=a[:,:2];width=a[:,2];height=a[:,3];z=a[:,4]
    minimum=float('inf');margin=float('inf')
    for start in range(0,n,256):
      ii=np.arange(start,min(start+256,n));d=np.sqrt(np.sum((xy[ii,None,:]-xy[None,:,:])**2,axis=2));d[np.arange(len(ii)),ii]=np.inf
      minimum=min(minimum,float(d.min()));margin=min(margin,float((d-np.maximum(width[ii,None],width[None,:])-5).min()))
    rad=np.linalg.norm(xy,axis=1);rt=np.linalg.norm(xy-np.array(tower),axis=1)
    ans={'N':n,'area_m2':float(np.sum(width*height)),'minimum_spacing_m':minimum,'spacing_surplus_m':margin,'maximum_field_radius_m':float(rad.max()),'minimum_tower_radius_m':float(rt.min()),'minimum_ground_clearance_m':float((z-height/2).min()),'width_min_m':float(width.min()),'width_max_m':float(width.max()),'height_min_m':float(height.min()),'height_max_m':float(height.max()),'installation_min_m':float(z.min()),'installation_max_m':float(z.max()),'maximum_receiver_distance_m':float(np.linalg.norm(np.c_[xy-np.array(tower),z-80],axis=1).max())}
    ans['passed']=bool(np.all(width>=height)&np.all(width<=8)&np.all(height>=2)&np.all(z>=2)&np.all(z<=6)&(rad.max()<=350+1e-8)&(rt.min()>=100-1e-8)&(margin>0)&np.all(z-height/2>0)&(np.linalg.norm(tower)<=350))
    return ans

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=16384);p.add_argument('--question',choices=['all','1','2','3'],default='all');p.add_argument('--seed',type=int,default=20260905);args=p.parse_args()
    exe=ROOT/('trace.exe' if os.name=='nt' else 'trace')
    if not exe.exists():
      subprocess.run(['g++','-O3','-std=c++17','-fopenmp',str(ROOT/'trace.cpp'),'-o',str(exe)],check=True)
    dest=ROOT/'recomputed';dest.mkdir(exist_ok=True);wb=Workbook();summary=wb.active;summary.title='annual';summary.append(['question','N','area_m2','optical','cosine','shadow_block','intercept','power_MW','unit_kW_m2']);validation={}
    for q in ([1,2,3] if args.question=='all' else [int(args.question)]):
      inp=ROOT/'data'/f'q{q}.txt';a=np.loadtxt(inp,skiprows=1);header=inp.read_text().splitlines()[0].split();tower=list(map(float,header[1:3]));check=constraints(a,tower);assert check['passed'];validation[q]=check
      out=dest/f'q{q}.csv';subprocess.run([str(exe),str(inp),str(out),str(args.samples),str(args.seed),'.00465','3.5','8'],check=True)
      t=np.genfromtxt(out,delimiter=',',names=True);assert len(t)==60
      summary.append([q,len(a),check['area_m2']]+[float(t[n].mean()) for n in ['optical','cosine','shadow_block','intercept','power_MW','unit_kW_m2']])
      sh=wb.create_sheet(f'q{q}_monthly');sh.append(['month','optical','cosine','shadow_block','intercept','unit_kW_m2','power_MW'])
      for m in range(1,13):sh.append([m]+[float(t[t['month']==m][n].mean()) for n in ['optical','cosine','shadow_block','intercept','unit_kW_m2','power_MW']])
      validation[q]['rated_power_passed']=bool(q==1 or t['power_MW'].mean()>=60)
      if q>1:assert validation[q]['rated_power_passed']
      if args.samples==16384 and args.seed==20260905:
        expected=json.loads((ROOT/'data/expected.json').read_text())[str(q)]
        validation[q]['absolute_power_difference_MW']=abs(float(t['power_MW'].mean())-expected['power_MW'])
        assert validation[q]['absolute_power_difference_MW']<1e-7
    wb.save(dest/'recomputed.xlsx');(dest/'checks.json').write_text(json.dumps(validation,indent=2));print('Saved:',dest)

if __name__=='__main__':main()
