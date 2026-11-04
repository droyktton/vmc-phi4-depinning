#!/usr/bin/env python3
"""Avalanche statistics of main_vb.cu bubbles (many seeds), the analysis of
Durin et al. (PRB 110, L020405 (2024)) on the T = 0 quasi-static bubble.

Reads <datadir>/D<Delta>/s<seed>/{vb,vbaval}_seed<seed>.dat. Drops step 0
(the relaxation of the initial disk), the last, cut step (converged = -1),
and the clusters with Sdown > 0. Per Delta:
  - F - sigma/2R (threshold of the curved wall minus the Laplace pressure);
  - P(S), P(ell) (log-binned, S >= SMIN);
  - S against ell, with two lateral lengths: ell_PCA = a*sqrt(lambda_+)
    (principal axes of the cluster's covariance, a fixed by S = pi a^2
    sqrt(lambda_+ lambda_-), Durin et al. S4) and ell_theta = R * angular
    extent; and the radial extent dr against ell (anisotropy).
Writes bubble_aval.png and bubble_aval.txt into --out.

  python3 scripts/bubble_aval.py data/vb_stats --out results/vb_stats
"""
import argparse, glob, re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('datadir'); ap.add_argument('--out',default='.')
ap.add_argument('--smin',type=float,default=10); ap.add_argument('--lmin',type=float,default=8)
ap.add_argument('--H0',type=float,default=3e-5)
a=ap.parse_args(); Path(a.out).mkdir(parents=True,exist_ok=True)
SIGMA=2*np.sqrt(2)/3; LM=np.sqrt(SIGMA/(2*a.H0))
C=['#2a78d6','#eb6834','#1baf7a','#8a5cc2']

def load(D):
    av=[]; Fs=[]; nseed=0
    for p in sorted(glob.glob(f'{a.datadir}/D{D}/s*/vb_seed*.dat')):
        d=np.loadtxt(p,ndmin=2)
        if len(d)<3: continue
        q=p.replace('/vb_seed','/vbaval_seed')
        c=np.loadtxt(q,ndmin=2)
        ok=set(d[(d[:,0]>0)&(d[:,6]==1),0].astype(int))
        c=c[np.isin(c[:,0].astype(int),list(ok))&(c[:,2]==0)]
        dd=d[(d[:,0]>0)&(d[:,6]==1)]
        Fs.append(dd[:,3]-SIGMA/(2*dd[:,2])); av.append(c); nseed+=1
    if not av: return None
    return np.vstack(av), np.concatenate(Fs), nseed

def lengths(c):
    cxx,cyy,cxy=c[:,5],c[:,6],c[:,7]
    tr=cxx+cyy; det=cxx*cyy-cxy**2; disc=np.sqrt(np.maximum(tr**2/4-det,0))
    lp=tr/2+disc; lm=np.maximum(tr/2-disc,1/12)   # 1/12: variance of a single row
    S=c[:,1]; aa=np.sqrt(S/(np.pi*np.sqrt(lp*lm))); a0=np.median(aa)
    return a0*np.sqrt(lp), c[:,12], c[:,13], a0
def logbin(x,lo,nb=20):
    x=x[x>=lo]; e=np.logspace(np.log10(lo),np.log10(x.max()*1.0001),nb+1); h,_=np.histogram(x,e)
    w=np.diff(e); xc=np.sqrt(e[1:]*e[:-1]); ok=h>0; return xc[ok],(h/w/len(x))[ok]
def fit(x,y,lo,hi):
    m=(x>=lo)&(x<=hi)&(y>0)
    if m.sum()<8: return np.nan,np.nan
    c,cov=np.polyfit(np.log(x[m]),np.log(y[m]),1,cov=True); return c[0],np.sqrt(cov[0,0])
def binned(x,y,nb=14):
    e=np.logspace(np.log10(x.min()),np.log10(x.max()*1.0001),nb+1); i=np.digitize(x,e)-1
    k=[j for j in range(nb) if (i==j).sum()>=3]
    return np.array([np.exp(np.log(x[i==j]).mean()) for j in k]),np.array([np.exp(np.log(y[i==j]).mean()) for j in k])
def tau_mle(S,smin,smax):
    S=S[(S>=smin)&(S<=smax)]
    if len(S)<30: return np.nan
    t=np.linspace(0.5,2.5,801); Z=[np.sum(np.arange(smin,smax+1,dtype=float)**-x) for x in t]
    return t[np.argmax(-t*np.log(S).sum()-len(S)*np.log(Z))]

Ds=sorted({re.search(r'/D([0-9.]+)$',p).group(1) for p in glob.glob(f'{a.datadir}/D*')},key=float)
fig,ax=plt.subplots(2,3,figsize=(15,8.5)); ax=ax.ravel()
lines=[f'# H0={a.H0} L_m={LM:.0f}; clusters with S>={a.smin}; fits over {a.lmin}<=ell<=L_m',
       '# Delta seeds n(S>=smin) <F-sigma/2R> sd  tau_MLE(smin..S_m)  1+zeta(ell_PCA)  1+zeta(ell_theta)  chi(dr~ell_theta)  a_PCA  max S']
for k,D in enumerate(Ds):
    r=load(D)
    if r is None: continue
    c,F,ns=r; S=c[:,1]; m=S>=a.smin; cm=c[m]
    lp,lt,dr,a0=lengths(cm); Sm=cm[:,1]
    Smax_cut=(S[S>0]**2).mean()/(2*S[S>0].mean())
    x,y=logbin(Sm,a.smin); ax[0].loglog(x,y,'o-',ms=3,color=C[k],label=f'Δ={D} ({m.sum()} aval., {ns} seeds)')
    x,y=logbin(lp,a.lmin); ax[1].loglog(x,y,'o-',ms=3,color=C[k],label=f'Δ={D}, ℓ_PCA')
    x,y=logbin(lt,a.lmin); ax[1].loglog(x,y,'s--',ms=3,color=C[k],alpha=0.6,label=f'Δ={D}, ℓ_θ')
    ax[2+k].loglog(lp,Sm,'.',ms=2,alpha=0.2,color=C[k]); xb,yb=binned(lp,Sm); ax[2+k].loglog(xb,yb,'-',lw=2,color=C[k],label='ℓ_PCA (Durin)')
    xb,yb=binned(lt,Sm); ax[2+k].loglog(xb,yb,'--',lw=2,color='0.3',label='ℓ_θ = R Δθ')
    s1,e1=fit(lp,Sm,a.lmin,LM); s2,e2=fit(lt,Sm,a.lmin,LM); s3,e3=fit(lt,dr,a.lmin,LM)
    xr=np.array([a.lmin,LM*3])
    ax[2+k].loglog(xr,0.5*xr**2.25,':',color='0.5',lw=1,label='ℓ^{2.25} (qEW)'); ax[2+k].loglog(xr,2*xr**1.63,'-.',color='0.5',lw=1,label='ℓ^{1.63} (qKPZ, Durin)')
    ax[2+k].axvline(LM,color='0.8',lw=1); ax[2+k].set(title=f'Δ={D}: 1+ζ = {s1:.2f}±{e1:.2f} (PCA), {s2:.2f}±{e2:.2f} (ℓ_θ)',xlabel='ℓ',ylabel='S'); ax[2+k].legend(fontsize=7)
    xb,yb=binned(lt,dr); ax[4].loglog(xb,yb,'o-',ms=3,color=C[k],label=f'Δ={D}: χ={s3:.2f}±{e3:.2f}')
    ax[5].hist(F,bins=60,histtype='step',color=C[k],density=True,label=f'Δ={D}: {F.mean():.4f}±{F.std():.4f}')
    t=tau_mle(Sm,int(a.smin),int(max(Smax_cut,a.smin+10)))
    lines.append(f'{D} {ns} {m.sum()} {F.mean():.5f} {F.std():.5f} {t:.2f} {s1:.3f}±{e1:.3f} {s2:.3f}±{e2:.3f} {s3:.3f}±{e3:.3f} {a0:.3f} {S.max():.0f}')
xr=np.array([a.smin,1e5]); ax[0].loglog(xr,0.3*xr**-1.11,':',color='0.4',lw=1,label='S^{-1.11} (qEW)'); ax[0].loglog(xr,0.5*xr**-1.26,'-.',color='0.6',lw=1,label='S^{-1.26} (qKPZ)')
ax[0].set(xlabel='S',ylabel='P(S)'); ax[0].legend(fontsize=7)
ax[1].set(xlabel='ℓ',ylabel='P(ℓ)'); ax[1].legend(fontsize=7)
xr=np.array([a.lmin,LM*3]); ax[4].loglog(xr,xr,':',color='0.4',lw=1,label='ℓ (isótropa)'); ax[4].loglog(xr,0.3*xr**1.25,'-.',color='0.6',lw=1,label='ℓ^{1.25} (qEW)')
ax[4].set(xlabel='ℓ_θ',ylabel='extensión radial dr',title='anisotropía'); ax[4].legend(fontsize=7)
ax[5].set(xlabel='F − σ/2R',title='umbral de la burbuja'); ax[5].legend(fontsize=7)
for x in ax: x.spines[['top','right']].set_visible(False)
fig.suptitle(f'Burbuja cuasiestática (main_vb), H0={a.H0}, L_m={LM:.0f}',fontsize=11); fig.tight_layout()
fig.savefig(f'{a.out}/bubble_aval.png',dpi=120); open(f'{a.out}/bubble_aval.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
