"""Build the new paper's figures and numeric tables from info's frozen results."""
from pathlib import Path
import os,json,shutil,importlib.util,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/paper_matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
DATA=HERE/'data';TABLES=HERE/'tables';FIGS=HERE/'figures';CODE=HERE/'code'
for folder in (DATA,TABLES,FIGS,CODE):folder.mkdir(parents=True,exist_ok=True)

def table(name,caption,headers,rows,align=None,note=None):
    align=align or 'l'+'r'*(len(headers)-1)
    lines=[r'\begin{table}[H]\centering\small',r'\caption{'+caption+r'}\label{tab:'+name+'}',
           r'\begin{tabular}{'+align+r'}\toprule',' & '.join(headers)+r'\\\midrule']
    lines+=[' & '.join(str(v) for v in row)+r'\\' for row in rows]
    lines+=[r'\bottomrule\end{tabular}']
    if note:lines+=[r'\par\smallskip\begin{minipage}{.94\linewidth}\footnotesize '+note+r'\end{minipage}']
    lines+=[r'\end{table}']
    (TABLES/(name+'.tex')).write_text('\n'.join(lines)+'\n')

def main():
    for p in (ROOT/'info').glob('*.py'):shutil.copy2(p,CODE/p.name)
    shutil.copy2(ROOT/'info/trace.cpp',CODE/'trace.cpp')
    shutil.copy2(ROOT/'final/info_figures/build_figures.py',CODE/'figure_tools.py')
    for q in (1,2,3):
        for tail in ('.txt',):shutil.copy2(ROOT/f'info/data/q{q}{tail}',DATA/f'q{q}{tail}')
        for tail in ('_times.csv','_times.csv.mirrors.csv','_trace.json'):
            shutil.copy2(ROOT/f'final/info_figures/data/q{q}{tail}',DATA/f'q{q}{tail}')
    expected=json.loads((ROOT/'info/data/expected.json').read_text())
    spec=importlib.util.spec_from_file_location('figure_tools',CODE/'figure_tools.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.HERE=FIGS;m.DATA=DATA;m.INFO=ROOT/'info';m.ROOT=ROOT;m.OUTPUTS=[]
    m.COLORS=['#124c8c','#e27800','#b52240']
    original=m.mirrors
    def high_contrast(ax,a,tower,values,norm,cmap=m.THERMAL_CMAP,scale=m.SCALE):
        if not isinstance(cmap,str):cmap='turbo';norm=Normalize(.35,.75)
        elif cmap=='cividis':cmap='turbo'
        elif cmap=='viridis_r':cmap='plasma_r'
        return original(ax,a,tower,values,norm,cmap,scale)
    m.mirrors=high_contrast
    entries={};macros={};monthly={}
    for q in (1,2,3):
        a,tower=m.load_design(q)
        times=np.genfromtxt(DATA/f'q{q}_times.csv',delimiter=',',names=True)
        per=np.genfromtxt(DATA/f'q{q}_times.csv.mirrors.csv',delimiter=',',names=True)
        record=json.loads((DATA/f'q{q}_trace.json').read_text())
        entries[q]=(a,tower,times,per,record)
        for key,val in [('P',record['power_MW']),('J',record['unit_kW_m2']),('Eta',expected[str(q)]['optical'])]:
            macros[f'Q{["","one","two","three"][q]}{key}']=f'{val:.6f}'
        monthly[q]=np.array([[times[k][times['month']==mon].mean() for k in
            ['optical','cosine','shadow_block','intercept','power_MW','unit_kW_m2']] for mon in range(1,13)])
        rows=[[f'{i} 月',*[f'{v:.5f}' for v in row[:4]],f'{row[4]:.4f}',f'{row[5]:.5f}'] for i,row in enumerate(monthly[q],1)]
        table(f'q{q}_monthly',f'问题{["","一","二","三"][q]}逐月平均效率与热功率',
            ['月份',r'$\bar\eta$',r'$\bar\eta_{\rm cos}$',r'$\bar\eta_{\rm sb}$',r'$\bar\eta_{\rm tr}$',r'$\bar P$/MW',r'$J$/(kW/m$^2$)'],rows,
            note='各时刻先按镜面面积加权，再对当月五个规定时刻取平均；全场功率先逐镜加总。')
    gain=100*(expected['3']['unit_kW_m2']/expected['2']['unit_kW_m2']-1)
    saving=100*(1-expected['3']['area_m2']/expected['2']['area_m2'])
    macros.update(Gain=f'{gain:.3f}',Saving=f'{saving:.3f}',AreaSaved=f'{expected["2"]["area_m2"]-expected["3"]["area_m2"]:.4f}')
    (TABLES/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    table('annual','三问年平均计算结果',['问题','镜数',r'面积/m$^2$','光学效率','功率/MW',r'单位功率/(kW/m$^2$)'],
          [[f'问题{["","一","二","三"][q]}',expected[str(q)]['N'],f'{expected[str(q)]["area_m2"]:.2f}',f'{expected[str(q)]["optical"]:.5f}',f'{expected[str(q)]["power_MW"]:.4f}',f'{expected[str(q)]["unit_kW_m2"]:.5f}'] for q in (1,2,3)])
    a=entries[3][0];tower=entries[3][1];radius=np.linalg.norm(a[:,:2]-tower,axis=1)
    edges=np.linspace(radius.min()-1e-7,radius.max()+1e-7,9);bands=np.clip(np.searchsorted(edges,radius,side='right')-1,0,7)
    table('q3_bands','第三问八个径向区间的安装高度',['区间','距塔半径范围/m','安装高度/m','镜数'],
          [[k+1,f'{edges[k]:.3f}--{edges[k+1]:.3f}',f'{np.unique(a[bands==k,4]).item():.8f}',int((bands==k).sum())] for k in range(8)])
    sizes,counts=np.unique(a[:,3],return_counts=True)
    table('q3_sizes','第三问的离散镜面规格',['宽度/m','镜面高度/m','单镜面积/m$^2$','镜数','数量占比'],
          [[5.75,f'{h:g}',f'{5.75*h:.4f}',n,f'{100*n/len(a):.2f}\\%'] for h,n in zip(sizes,counts)])
    geometry=json.loads((ROOT/'final/comparison/geometry.json').read_text())
    checks=[('镜间距超出下限','minimum_spacing_margin_m'),('旋转离地裕量','minimum_ground_clearance_m'),('镜心场界裕量','minimum_site_margin_m'),('镜心禁区裕量','minimum_exclusion_margin_m')]
    table('constraints','按镜心定义检查的几何约束裕量（m）',['检查项目','问题一','问题二','问题三'],
          [[label,*[f'{geometry[f"info_q{q}"]["centers"][key]:.6f}' for q in (1,2,3)]] for label,key in checks])
    # Seasonal and instantaneous charts from the same 16384-ray run.
    m.build(entries)
    fig,ax=plt.subplots(figsize=(7.7,3.65),layout='constrained')
    heat=entries[1][2]['power_MW'].reshape(12,5).T
    im=ax.imshow(heat,origin='lower',aspect='auto',cmap='turbo',extent=(.5,12.5,.5,5.5))
    ax.set(xticks=range(1,13),yticks=range(1,6),yticklabels=['9:00','10:30','12:00','13:30','15:00'],xlabel='月份',ylabel='当地太阳时')
    fig.colorbar(im,ax=ax,label='第一问全场热功率 / MW');m.save(fig,'q1_time_heatmap')
    if (DATA/'verification_summary.json').exists():
        summary=json.loads((DATA/'verification_summary.json').read_text());runs=json.loads((DATA/'verification_runs.json').read_text())
        table('convergence','不同光线数下的年均热功率（MW）',['方案','1024 条','4096 条','16384 条','4096 至 16384 差值/kW'],
              [[f'问题{["","一","二","三"][q]}',*[f'{next(r["power_MW"] for r in runs if r["label"]=="convergence" and r["settings"]["q"]==q and r["settings"]["rays"]==n):.6f}' for n in (1024,4096)],f'{expected[str(q)]["power_MW"]:.6f}',f'{1000*abs(next(r["power_MW"] for r in runs if r["label"]=="convergence" and r["settings"]["q"]==q and r["settings"]["rays"]==4096)-expected[str(q)]["power_MW"]):.4f}'] for q in (1,2,3)])
        table('independent','四组新随机平移下的数值复核',['方案','均值/MW','标准误/kW','单侧95\\%下界/MW'],
              [[f'问题{["","一","二","三"][q]}',f'{summary["independent"][str(q)]["mean_MW"]:.6f}',f'{1000*summary["independent"][str(q)]["se_MW"]:.4f}',f'{summary["independent"][str(q)]["lower95_MW"]:.6f}'] for q in (2,3)],
              note='每组每镜每时刻 4096 条光线；仅反映随机化数值积分误差，不含物理模型误差。')
        labels={'sun_4.2':'太阳半角 4.2 mrad','sun_5.1':'太阳半角 5.1 mrad','shaft_0':'塔身半径 0 m','shaft_2':'塔身半径 2 m'}
        table('sensitivity','物理参数扰动引起的年均功率变化',['参数情景','问题二功率/MW','变化','问题三功率/MW','变化'],
              [[label,*[v for q in (2,3) for r in summary['sensitivity'] if r['q']==q and r['scenario']==key for v in (f'{r["power_MW"]:.5f}',f'{r["paired_change_percent"]:+.4f}\\%')]] for key,label in labels.items()],
              note='固定最终设计；各情景与基准均使用 1024 条光线和同一随机种子。')
        fig,axs=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained')
        for q,c in zip((1,2,3),m.COLORS):
            sub=[r for r in runs if r['label']=='convergence' and r['settings']['q']==q]
            axs[0].plot([r['settings']['rays'] for r in sub]+[16384],[r['difference_from_16384_percent'] for r in sub]+[0],marker='s',label=f'问题{["","一","二","三"][q]}',color=c)
        axs[0].set(xscale='log',xlabel='每镜每时刻光线数',ylabel='相对 16384 条结果的偏差 / %',xticks=[1024,4096,16384],xticklabels=['1024','4096','16384'])
        axs[0].legend();axs[0].grid(alpha=.2)
        x=np.arange(4)
        for q,offset,c in ((2,-.18,m.COLORS[0]),(3,.18,m.COLORS[2])):
            values=[next(r['paired_change_percent'] for r in summary['sensitivity'] if r['q']==q and r['scenario']==key) for key in labels]
            axs[1].bar(x+offset,values,.34,label=f'问题{["","一","二","三"][q]}',color=c)
        axs[1].set(xticks=x,xticklabels=['太阳4.2','太阳5.1','塔身0','塔身2'],ylabel='年均功率相对变化 / %')
        axs[1].axhline(0,c='gray',lw=.7);axs[1].legend();m.save(fig,'verification')
    metadata=dict(source='info frozen ring designs',figure_scale=1.2,reference_pose='March 21 12:00',
                  gain_q3_vs_q2_percent=gain,area_saving_percent=saving,
                  inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'info/trace.cpp',*[ROOT/f'info/data/q{q}.txt' for q in (1,2,3)]]})
    (DATA/'paper_sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print('Paper assets generated',flush=True)

if __name__=='__main__':main()
