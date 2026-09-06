"""Generate every paper number, table and scientific figure from verified data."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'build'/'paper_matplotlib'))
import sys
sys.path.insert(0,str(HERE.parent))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from heliostat.io import load_field

DATA=HERE/'data';TABLES=HERE/'tables';FIGS=HERE/'figures'
TABLES.mkdir(exist_ok=True);FIGS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Songti SC','Arial'],'font.size':10,
                     'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.65,'lines.linewidth':1.6,
                     'xtick.direction':'out','ytick.direction':'out',
                     'savefig.facecolor':'white'})
COLORS=['#315e79','#ba722a','#377e64']


def read(name):return json.loads((DATA/name).read_text())


def table(filename,caption,label,headers,rows,align=None,note=None,size='small'):
    align=align or ('l'+'r'*(len(headers)-1))
    placement='H' if filename.endswith(('_monthly.tex','_design.tex')) else 'htbp'
    text=[r'\begin{table}['+placement+']','\\centering\\'+size,
          r'\caption{'+caption+r'}\label{tab:'+label+'}',
          r'\begin{tabular}{'+align+'}',r'\toprule',
          ' & '.join(headers)+r'\\\midrule']
    text.extend(' & '.join(map(str,row))+r'\\' for row in rows)
    text += [r'\bottomrule',r'\end{tabular}']
    if note:text += [r'\par\vspace{3pt}\begin{minipage}{.95\linewidth}\footnotesize '+note+r'\end{minipage}']
    text += [r'\end{table}']
    (TABLES/filename).write_text('\n'.join(text)+'\n')


def save(fig,name):
    fig.savefig(FIGS/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
    fig.savefig(FIGS/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def field_axes(ax,f):
    ax.add_patch(Circle((0,0),350,fill=False,lw=.8,color='#404040'))
    ax.add_patch(Circle(f.tower,100,fill=False,lw=.7,ls='--',color='#777777'))
    ax.scatter(*f.tower,marker='+',s=90,c='black',linewidths=1.4,zorder=5)
    ax.set(xlim=(-365,365),ylim=(-365,365),xlabel='东向坐标 x / m',ylabel='北向坐标 y / m')
    ax.set_aspect('equal');ax.set_xticks([-300,-150,0,150,300]);ax.set_yticks([-300,-150,0,150,300])


def main():
    audit=read('verification.json');conv=read('convergence.json');sens=read('sensitivity.json')
    legacy=read('legacy_comparison.json')
    fields={key:load_field(DATA/f'{key}.npz') for key in ['q1','q2','q3']}
    density={key:np.load(DATA/f'{key}_optics.npz')['power_kw'].mean(axis=0)/f.area for key,f in fields.items()}
    annual={key:np.array(audit[key]['annual']) for key in fields}
    monthly={key:np.array(audit[key]['monthly']) for key in fields}
    macros={}
    for key,prefix in [('q1','Qone'),('q2','Qtwo'),('q3','Qthree')]:
        a=annual[key];f=fields[key]
        for suffix,value,digits in [('P',a[5],4),('J',a[6],5),('Opt',a[0],5),('Cos',a[1],5),
                                    ('SB',a[2],5),('Trunc',a[4],5),('At',a[3],5),
                                    ('Width',f.widths[0],4),('Height',f.heights[0],4),('Z',f.centers[0,2],4)]:
            macros[prefix+suffix]=f'{value:.{digits}f}'
        macros[prefix+'Count']=str(len(f))
        rows=[]
        for m,row in enumerate(monthly[key],1):
            rows.append([f'{m} 月 21 日',*[f'{row[i]:.5f}' for i in [0,1,2,4]],f'{row[5]:.4f}',f'{row[6]:.5f}'])
        table(f'{key}_monthly.tex',f'问题{dict(q1="一",q2="二",q3="三")[key]}每月 21 日平均光学效率及输出热功率',key+'month',
              ['日期',r'$\mean\eta$',r'$\mean\eta_{\rm cos}$',r'$\mean\eta_{\rm sb}$',r'$\mean\eta_{\rm trunc}$',r'$\mean P$/MW',r'$J$/($\unitp$)'],rows,
              note='各项效率为五个规定时刻的面积加权均值，效率无量纲；功率先逐镜求和再对时刻平均。')
    f2,f3=fields['q2'],fields['q3']
    macros['QthreeTypes']=str(len(np.unique(np.c_[f3.widths,f3.heights],axis=0)))
    macros['QthreeMinTypeCount']=str(np.unique(np.c_[f3.widths,f3.heights],axis=0,return_counts=True)[1].min())
    macros['QthreeZTypes']=str(len(np.unique(f3.centers[:,2])))
    macros['GainJ']=f'{100*(annual["q3"][6]/annual["q2"][6]-1):.3f}'
    macros['AreaSaving']=f'{100*(1-f3.area.sum()/f2.area.sum()):.3f}'
    if 'revision_q3_comparison' in audit:
        revision=audit['revision_q3_comparison']
        macros['RevisionOldP']=f'{revision["old_power_MW"]:.4f}'
        macros['RevisionOldJ']=f'{revision["old_unit_power"]:.5f}'
        macros['RevisionGain']=f'{revision["relative_gain_percent"]:.3f}'
        macros['RevisionLow']=f'{revision["paired_unit_gain"]["lower_95_one_sided_kw"]:.6f}'
    macros['PairedDiff']=f'{audit["paired_q3_minus_q2"]["mean_kw"]:.6f}'
    macros['PairedLow']=f'{audit["paired_q3_minus_q2"]["lower_95_one_sided_kw"]:.6f}'
    macros['LegacyP']=f'{legacy["power_MW"]:.4f}'
    macros['ConvMax']=f'{max(abs(x["relative_error_percent"]) for x in conv if x["rays"]==4096):.4f}'
    (TABLES/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    for key,f in [('q2',f2),('q3',f3)]:
        rows=[['吸收塔位置',f'$({f.tower[0]:.4f},\,{f.tower[1]:.4f})$','m']]
        if key=='q2':rows += [['镜面尺寸（宽 $\\times$ 高）',f'${f.widths[0]:.4f}\\times{f.heights[0]:.4f}$','m $\\times$ m'],
                              ['统一安装高度',f'{f.centers[0,2]:.4f}','m']]
        else:
            for label,arr in [('镜面宽度范围',f.widths),('镜面高度范围',f.heights),('安装高度范围',f.centers[:,2])]:
                rows.append([label,f'{arr.min():.4f} 至 {arr.max():.4f}','m'])
        rows += [['定日镜数目',len(f),'面'],['总镜面面积',f'{f.area.sum():.4f}',r'm$^2$'],
                 ['年平均输出热功率',f'{annual[key][5]:.4f}','MW'],
                 ['单位面积年平均热功率',f'{annual[key][6]:.5f}',r'kW/m$^2$']]
        table(f'{key}_design.tex','统一镜型设计参数' if key=='q2' else '异构镜型设计参数与范围',key+'design',
              ['参数','结果','单位'],rows,'lrl',note='坐标均相对题目场地圆心；逐镜完整精度参数见随文数据。')
    rows=[[label,*[f'{annual[key][i]:.5f}' for i in [0,1,2,4]],f'{annual[key][5]:.4f}',f'{annual[key][6]:.5f}']
          for key,label in [('q1','问题一'),('q2','问题二'),('q3','问题三')]]
    table('annual.tex','三问年平均光学效率及输出热功率','annual',
          ['方案',r'$\mean\eta$',r'$\mean\eta_{\rm cos}$',r'$\mean\eta_{\rm sb}$',r'$\mean\eta_{\rm trunc}$',r'$\mean P$/MW',r'$J$/($\unitp$)'],rows)
    checks=[('最小镜间距余量','minimum_spacing_margin_m'),('最小旋转离地间隙','minimum_ground_clearance_m'),
            ('最小场界余量','minimum_site_margin_m'),('最小禁区余量','minimum_exclusion_margin_m')]
    rows=[[label,f'{audit["q2"]["constraints"][key]:.6f}',f'{audit["q3"]["constraints"][key]:.6f}'] for label,key in checks]
    table('constraints.tex','最终新设计的几何约束裕量（m）','constraints',['检查项目','问题二','问题三'],rows)
    rows=[]
    for key,label in [('q1','问题一'),('q2','问题二'),('q3','问题三')]:
        u=audit[key]['uncertainty']
        rows.append([label,f'{u["mean_kw"]/1000:.5f}',f'{u["standard_error_kw"]:.4f}',
                     f'{u["lower_95_one_sided_kw"]/1000:.5f}','定场计算' if key=='q1' else '通过'])
    table('uncertainty.tex','四组独立随机化的年均功率复核','uncertainty',
          ['方案','均值/MW','标准误/kW','单侧 95\% 下界/MW','额定值检查'],rows,
          note='每组每镜每时刻 4096 条光线；区间仅表示随机化数值积分的不确定性。')
    rows=[]
    for key,label in [('q1','问题一'),('q2','问题二'),('q3','问题三')]:
        lookup={x['rays']:x['power_MW'] for x in conv if x['problem']==key}
        rows.append([label,*[f'{lookup[n]:.4f}' for n in [128,512,1024,2048,4096]],f'{annual[key][5]:.4f}'])
    table('convergence.tex','不同光线数下的年均热功率（MW）','convergence',
          ['方案','128','512','1024','2048','4096','最终均值'],rows)
    scenario_names=[('shaft_0m','无塔身（半径 0 m）'),('shaft_3m','塔身半径 3 m'),
                    ('sun_4.2mrad','太阳视半径 4.2 mrad'),('sun_5.1mrad','太阳视半径 5.1 mrad'),
                    ('slope_0.5mrad','斜率误差 0.5 mrad'),('slope_1mrad','斜率误差 1.0 mrad')]
    rows=[]
    for scenario,label in scenario_names:
        row=[label]
        for key in fields:
            val=next(x['paired_change_percent'] for x in sens if x['problem']==key and x['scenario']==scenario)
            row.append(f'{val:+.3f}\\%')
        rows.append(row)
    table('sensitivity.tex','物理假设变化引起的年均热功率相对变化','sensitivity',
          ['相对基准的扰动','问题一','问题二','问题三'],rows,
          note='固定最终布局；所有情景与各自基准均采用同种子的 1024 条光线。')

    fig,axs=plt.subplots(1,2,figsize=(8.3,3.6),layout='constrained')
    f=fields['q1'];im=axs[0].scatter(*f.centers[:,:2].T,c=density['q1'],s=3,cmap='viridis',rasterized=True)
    field_axes(axs[0],f);axs[0].set_title('(a) 附件镜场单位面积年均热功率')
    fig.colorbar(im,ax=axs[0],shrink=.8,pad=.02,label='kW/m²')
    m=np.arange(1,13)
    for idx,label,color in [(1,'余弦效率',COLORS[0]),(2,'阴影遮挡效率',COLORS[1]),(4,'截断效率',COLORS[2])]:
        axs[1].plot(m,monthly['q1'][:,idx],marker='o',ms=3,label=label,c=color)
    axs[1].set(title='(b) 逐月平均分项效率',xlabel='月份',ylabel='效率',xticks=[1,3,6,9,12],ylim=(.65,1.01))
    axs[1].legend(loc='lower center',frameon=False,fontsize=9);axs[1].grid(alpha=.2)
    save(fig,'q1_analysis')

    fig,axs=plt.subplots(1,2,figsize=(8.3,3.75),layout='constrained')
    norm=Normalize(min(density['q2'].min(),density['q3'].min()),max(density['q2'].max(),density['q3'].max()))
    for ax,key,title in zip(axs,['q2','q3'],['(a) 统一镜型','(b) 异构镜型']):
        f=fields[key];im=ax.scatter(*f.centers[:,:2].T,c=density[key],s=3,cmap='viridis',norm=norm,rasterized=True)
        field_axes(ax,f);ax.set_title(f'{title}：{len(f)} 面')
    fig.colorbar(im,ax=axs,shrink=.82,pad=.015,label='单位面积年均热功率 / (kW/m²)')
    save(fig,'designs')

    fig,axs=plt.subplots(1,2,figsize=(8.3,3.7),layout='constrained')
    for ax,values,title,label,cmap in [(axs[0],f3.area,'(a) 镜面面积分布','镜面面积 / m²','viridis'),
                                      (axs[1],f3.centers[:,2],'(b) 安装高度分布','安装高度 / m','cividis')]:
        im=ax.scatter(*f3.centers[:,:2].T,c=values,s=3,cmap=cmap,rasterized=True)
        field_axes(ax,f3);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8,pad=.02,label=label)
    save(fig,'heterogeneity')

    fig,axs=plt.subplots(1,2,figsize=(8.3,3.15),layout='constrained')
    for key,label,color in zip(fields,['问题一','问题二','问题三'],COLORS):
        axs[0].plot(m,monthly[key][:,5],marker='s' if key=='q2' else 'o',
                    ms=3.5 if key=='q2' else 3,label=label,color=color,
                    linestyle='--' if key=='q2' else '-',zorder=4 if key=='q2' else 3)
    axs[0].axhline(60,color='#777777',lw=.9,ls='--');axs[0].text(1,60.7,'60 MW 年均额定值',fontsize=8,color='#666666')
    axs[0].set(xlabel='月份',ylabel='月均热功率 / MW',xticks=[1,3,6,9,12],title='(a) 全场月均热功率')
    axs[0].legend(frameon=False,loc='lower center',fontsize=9);axs[0].grid(alpha=.2)
    for key,label,color in zip(['q2','q3'],['统一镜型','异构镜型'],COLORS[1:]):
        axs[1].plot(m,monthly[key][:,6],marker='o',ms=3,label=label,c=color)
    axs[1].set(xlabel='月份',ylabel='单位面积月均热功率 / (kW/m²)',xticks=[1,3,6,9,12],title='(b) 容量达标方案的面积效率')
    axs[1].legend(frameon=False,fontsize=9);axs[1].grid(alpha=.2)
    save(fig,'monthly_power')

    fig,ax=plt.subplots(figsize=(7.4,2.65),layout='constrained')
    for key,label,color in zip(fields,['问题一','问题二','问题三'],COLORS):
        sub=[x for x in conv if x['problem']==key]
        ax.plot([x['rays'] for x in sub],[x['relative_error_percent'] for x in sub],marker='o',ms=4,label=label,c=color)
    ax.axhline(0,color='#888888',lw=.8);ax.set_xscale('log',base=2)
    ax.set_xticks([128,512,1024,2048,4096],['128','512','1024','2048','4096'])
    ax.set(xlabel='每镜每时刻光线数 K',ylabel='相对最终均值偏差 / %')
    ax.grid(alpha=.2);ax.legend(frameon=False,ncol=3)
    save(fig,'convergence')
    print('Generated verified numbers, 10 result tables and 5 vector figures.')


if __name__=='__main__':main()
