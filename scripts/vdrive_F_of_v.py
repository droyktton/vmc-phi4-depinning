#!/usr/bin/env python3
"""v against the mean drive <F> on a linear scale, for every spring H0 of
stage vd and for the fixed-force points, and the force deficit
h_c - <F> at the slowest imposed v against L_m = sqrt(0.47/H0), for each L.
Reads the same tree as vdrive_beta.py (<root>/ff_L*/h*/, <root>/vd_L*_H*/v*/),
second half of each run, seeds averaged with errors from their spread.

  python3 scripts/vdrive_F_of_v.py data/vdrive_D1.0 --hc 0.0663 --outdir results/vdrive_D1.0_vd
"""
import argparse
import glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--hc',type=float,required=True); p.add_argument('--outdir',required=True)
args=p.parse_args(); R=args.root; hc=args.hc; Path(args.outdir).mkdir(parents=True,exist_ok=True)
def stats(vals): vals=np.array(vals); return vals.mean(), (vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0)
def half(f): a=np.loadtxt(f); return a[len(a)//2:]
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
# fixed force
for L,mk,c in [(1024,'s','k'),(2048,'D','0.5')]:
    P=[]
    for hd in glob.glob(f'{R}/ff_L{L}/h*'):
        vs=[np.polyfit(a[:,0],a[:,1],1)[0] for a in map(half,glob.glob(hd+'/vdyn_h*_seed*.dat')) if len(a)>10]
        if vs: P.append((float(hd.split('/h')[-1]),*stats(vs)))
    P=np.array(sorted(P)); P=P[P[:,0]<0.075]
    ax[0].errorbar(P[:,0],P[:,1],yerr=P[:,2],fmt=mk,mfc='none',color=c,label=f'fuerza fija, L={L}',zorder=3)
cols={'3e-05':'#9ecae1','1e-05':'#4292c6','3e-06':'#08306b'}; Lm={'3e-05':125,'1e-05':217,'3e-06':396}
low=[]
for H in ['3e-05','1e-05','3e-06']:
    for L,mk in [(1024,'o'),(2048,'^')]:
        d=f'{R}/vd_L{L}_H{H}'
        if not glob.glob(d): continue
        P=[]
        for vd in glob.glob(d+'/v*'):
            Fs=[a[:,3].mean() for a in map(half,glob.glob(vd+'/vdyn_*.dat'))]
            P.append((float(vd.split('/v')[-1]),*stats(Fs),len(Fs)))
        P=np.array(sorted(P))
        ax[0].errorbar(P[:,1],P[:,0],xerr=P[:,2],fmt=mk+'-',color=cols[H],ms=5,lw=1,label=f'v impuesta, L={L}, H0={H} (L_m={Lm[H]})')
        low.append((L,Lm[H],P[0,1],P[0,2],P[1,1],P[1,2],P[0,0]))
ax[0].axvline(hc,color='r',ls='--',lw=1); ax[0].text(hc,0.061,' h_c (bisección)',color='r',fontsize=8)
ax[0].set_xlabel('fuerza media  ⟨F⟩  (o h a fuerza fija)'); ax[0].set_ylabel('v'); ax[0].set_ylim(0,0.065)
ax[0].set_title('v(F) en escala lineal'); ax[0].legend(fontsize=7,loc='lower right')
low=np.array(low)
for L,mk,ls in [(1024,'o','-'),(2048,'^','--')]:
    a=low[low[:,0]==L]; a=a[np.argsort(a[:,1])]
    ax[1].errorbar(a[:,1],hc-a[:,2],yerr=a[:,3],fmt=mk+ls,color='#08519c',mfc='none' if L==2048 else None,label=f'L={L}, v={a[0,6]:.2g}')
    p=np.polyfit(np.log(a[:,1]),np.log(hc-a[:,2]),1); print(L,'slope',p[0])
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('L_m = sqrt(0.47/H0)'); ax[1].set_ylabel('h_c − ⟨F⟩')
ax[1].set_title('déficit de fuerza a la v más baja'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{args.outdir}/F_of_v.png',dpi=130); print(f'wrote {args.outdir}/F_of_v.png')

