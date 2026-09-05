"""Standalone, reproducible diagnostic figures; no paper generation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def create_plots(fields,evaluations,convergence,outdir,site):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.3),constrained_layout=True)
    for ax,(name,f) in zip(axes,fields.items()):
        e=evaluations[name]
        color=e.per_mirror_kw/f.area
        sc=ax.scatter(f.centers[:,0],f.centers[:,1],c=color,s=3,cmap='viridis',vmin=.15,vmax=.85,rasterized=True)
        ax.add_patch(Circle((0,0),site.field_radius,fill=False,color='#778899'))
        ax.add_patch(Circle(f.tower,site.exclusion_radius,fill=False,color='#B8573A',linestyle='--'))
        ax.plot(*f.tower,'+',color='#B8573A',markersize=12)
        ax.set(xlim=(-360,360),ylim=(-360,360),aspect='equal',xlabel='East (m)',ylabel='North (m)',
               title=f'{name.upper()}: {len(f):,} mirrors\n{e.annual_power_kw/1000:.3f} MW | {e.unit_power:.4f} kW/m²')
    fig.colorbar(sc,ax=axes,shrink=.72,label='Annual thermal output per mirror area (kW/m²)')
    fig.savefig(outdir/'layouts.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,e in evaluations.items():
        m=e.monthly()
        axes[0].plot(np.arange(1,13),m[:,5],marker='o',label=name.upper())
        rows=[r for r in convergence if r['problem']==name]
        axes[1].plot([r['rays'] for r in rows],[r['difference_from_final_mean_percent'] for r in rows],marker='o',label=name.upper())
    axes[0].axhline(60,color='gray',linestyle='--',label='60 MW annual target')
    axes[0].set(xlabel='Month',ylabel='Mean thermal power (MW)',xticks=range(1,13))
    axes[1].set(xlabel='Sobol rays per mirror and time',ylabel='Difference from final mean (%)',xscale='log')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(outdir/'power_and_convergence.png',dpi=180);plt.close(fig)
    f=fields['q3'];radius=np.linalg.norm(f.centers[:,:2]-f.tower,axis=1)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,value,title in zip(axes,[f.widths,f.heights,f.centers[:,2]],['Mirror width (m)','Mirror height (m)','Installation height (m)']):
        ax.scatter(radius,value,s=4,alpha=.45)
        ax.set(xlabel='Distance from tower (m)',ylabel=title)
        ax.grid(alpha=.2)
    fig.savefig(outdir/'q3_dimensions.png',dpi=180);plt.close(fig)
