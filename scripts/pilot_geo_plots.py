#!/usr/bin/env python3
"""Avalanche geometry of the quasi-static pilot (main_vd.cu columns ell, height, n_mv, depth),
from data/pilot_geo/<D>_<H>/vd_seed1.dat into results/pilot_qs/:
  geo_S_vs_ell.png      S against the lateral extent ell: d_f from S ~ ell^d_f
  geo_height_vs_ell.png height of the avalanche against ell: anisotropy height ~ ell^chi
  geo_PS_overhang.png   P(S) of the avalanches that raise the number of multivalued columns
                        (dn_mv > 0) and of the rest
  geo_fits.txt          the fitted exponents

  python3 scripts/pilot_geo_plots.py [datadir]   (from the repository root)
"""
import glob, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
B=str(Path(__file__).resolve().parents[1]); OUT=B+'/results/pilot_qs'
DATA=sys.argv[1] if len(sys.argv)>1 else B+'/data/pilot_geo'
Path(OUT).mkdir(parents=True, exist_ok=True)
C={'0.3':'#2a78d6','1.0':'#eb6834','1.5':'#1baf7a'}
LS={'3e-5':'-','1e-5':'--'}; MK={'3e-5':'o','1e-5':'s'}; LM={'3e-5':125,'1e-5':217}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.9','font.size':9})
SMIN=10        # below this, single-site flips dominate

def runs():
    for D in ['0.3','1.0','1.5']:
        for H in ['3e-5','1e-5']:
            p=sorted(glob.glob(f'{DATA}/D{D}_H{H}*/vd_seed1.dat'))
            p=[q for q in p if 'escaped' not in q] or p
            if p: yield D,H,p[0]
def load(p, transient=0.1):
    # complete rows only (the last one may be half-written in a running job)
    a=np.array([l.split() for l in open(p) if l[0]!='#' and len(l.split())==12],float); a=a[int(len(a)*transient):]
    dn=np.diff(a[:,10],prepend=a[0,10])          # change of multivalued columns over the step
    return a, dn
def binned(x,y,nb=14):
    e=np.logspace(np.log10(x.min()),np.log10(x.max()*1.0001),nb+1); i=np.digitize(x,e)-1
    xm=np.array([np.exp(np.log(x[i==k]).mean()) for k in range(nb) if (i==k).sum()>=3])
    ym=np.array([np.exp(np.log(y[i==k]).mean()) for k in range(nb) if (i==k).sum()>=3])
    return xm,ym
def slope(x,y,lo,hi):
    m=(x>=lo)&(x<=hi)
    if m.sum()<5: return np.nan,np.nan
    c,cov=np.polyfit(np.log(x[m]),np.log(y[m]),1,cov=True); return c[0],np.sqrt(cov[0,0])

R=[(D,H,p)+load(p) for D,H,p in runs()]
fits=['# Delta H0 L_m n(S>=10) d_f(S~ell) chi(height~ell) frac_dn>0 tau_mle(dn>0) tau_mle(dn<=0)  [fits over ell in 8..L_m]']

for name,yc,ylab,key in [('geo_S_vs_ell',4,'S (sitios)','d_f'),('geo_height_vs_ell',8,'altura (filas)','χ')]:
    fig,ax=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for k,D in enumerate(['0.3','1.0','1.5']):
        for (D2,H,p,a,dn) in R:
            if D2!=D: continue
            m=(a[:,4]>=SMIN)&(a[:,7]>0); x=a[m,7]; y=a[m,yc]
            ax[k].plot(x,y,MK[H],ms=2,alpha=0.25,color=C[D],mec='none')
            xb,yb=binned(x,y); ax[k].plot(xb,yb,LS[H],lw=2,color=C[D],label=f'L_m={LM[H]}')
            s,e=slope(x,y,8,LM[H]); ax[k].text(0.03,0.92 if H=='3e-5' else 0.84,f'L_m={LM[H]}: {key}={s:.2f}±{e:.2f}',transform=ax[k].transAxes,fontsize=8)
        xr=np.array([8,1024])
        if yc==4:
            ax[k].plot(xr,2*xr**(91/48),':',color='0.4',lw=1,label='ℓ^{91/48} (perc.)'); ax[k].plot(xr,0.3*xr**2.25,'-.',color='0.6',lw=1,label='ℓ^{1+ζ}, ζ=1.25')
        else:
            ax[k].plot(xr,xr,':',color='0.4',lw=1,label='ℓ (isótropa)'); ax[k].plot(xr,0.5*xr**1.25,'-.',color='0.6',lw=1,label='ℓ^{1.25} (qEW)')
        ax[k].set(xscale='log',yscale='log',xlabel='ℓ (columnas con flips)',title=f'Δ={D}'); ax[k].legend(fontsize=7,loc='lower right')
    ax[0].set_ylabel(ylab); fig.suptitle(f'Geometría de avalanchas, S≥{SMIN}, L=1024, 1 seed; ajuste en 8≤ℓ≤L_m',fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png',dpi=130); plt.close()

def tau_mle(S,smin,smax):
    S=S[(S>=smin)&(S<=smax)]
    if len(S)<20: return np.nan,len(S)
    t=np.linspace(0.3,2.5,881); ls=np.log(S).sum()
    Z=[np.sum(np.arange(smin,smax+1,dtype=float)**-x) for x in t]
    ll=-t*ls-len(S)*np.log(Z); return t[np.argmax(ll)],len(S)

fig,ax=plt.subplots(1,3,figsize=(12,4),sharey=True)
for (D,H,p,a,dn) in R:
    k=['0.3','1.0','1.5'].index(D); S=a[:,4]; m=S>=SMIN
    Sm=(S[S>0]**2).mean()/(2*S[S>0].mean())
    e=np.logspace(1,np.log10(S.max()+1),22); w=np.diff(e); xc=np.sqrt(e[1:]*e[:-1])
    row=[D,H,LM[H],m.sum()]
    for sel,lab,mk in [(m&(dn>0),'Δn_mv>0','^'),(m&(dn<=0),'Δn_mv≤0','v')]:
        h,_=np.histogram(S[sel],e); ok=h>0
        ax[k].plot(xc[ok],h[ok]/w[ok]/m.sum(),LS[H]+mk,ms=4,color=C[D],alpha=1 if lab[-2]=='>' else 0.5,label=f'L_m={LM[H]}, {lab} ({sel.sum()})')
    s1,_=slope(a[m,7],S[m],8,LM[H]); s2,_=slope(a[m,7],a[m,8],8,LM[H])
    t1,_=tau_mle(S[m&(dn>0)],SMIN,int(Sm)); t2,_=tau_mle(S[m&(dn<=0)],SMIN,int(Sm))
    fits.append(f'{D} {H} {LM[H]} {m.sum()} {s1:.3f} {s2:.3f} {(m&(dn>0)).sum()/max(m.sum(),1):.3f} {t1:.2f} {t2:.2f}')
for k,D in enumerate(['0.3','1.0','1.5']):
    xr=np.array([10,1e5]); ax[k].plot(xr,0.3*xr**-1.11,':',color='0.4',lw=1,label='S^{-1.11}')
    ax[k].set(xscale='log',yscale='log',xlabel='S',title=f'Δ={D}'); ax[k].legend(fontsize=6.5,loc='lower left')
ax[0].set_ylabel('P(S) (normalizada al total S≥10)')
fig.suptitle('P(S) separando avalanchas que crean columnas multivaluadas (Δn_mv>0) del resto',fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/geo_PS_overhang.png',dpi=130); plt.close()
open(f'{OUT}/geo_fits.txt','w').write('\n'.join(fits)+'\n'); print('\n'.join(fits))
