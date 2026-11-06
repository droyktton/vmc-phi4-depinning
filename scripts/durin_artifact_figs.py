import numpy as np, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
"""Figures for section 8 of the Durin reanalysis artifact (our phi^4 bubbles only, vb_big data).
Usage: python3 scripts/durin_artifact_figs.py [outdir]   (run from repo root; default outdir results/vb_big/)"""
import sys, os
OUT=os.path.join(sys.argv[1] if len(sys.argv)>1 else 'results/vb_big','')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
C={'0.3':'#2a5b86','1.0':'#b5651d'}; NEVER=2**31-1; LM=217
# --- fig 1: arrival maps
fig,ax=plt.subplots(1,2,figsize=(10,5))
for k,D in enumerate(['0.3','1.0']):
    f=np.fromfile(f'data/vb_big/D{D}/s1/vbflip_seed1.bin',dtype=np.int32).reshape(2048,2048)
    d=np.loadtxt(f'data/vb_big/D{D}/s1/vb_seed1.dat'); tmax=int(d[d[:,6]==1,0].max())
    g=f[::2,::2].astype(float); g[(g>tmax)]=np.nan; g[g<0]=0
    m=np.ma.masked_invalid(g)
    ax[k].imshow(m,cmap='viridis',origin='lower',interpolation='nearest'); ax[k].set_xticks([]); ax[k].set_yticks([])
    for t in np.linspace(0,tmax,7)[1:].astype(int):
        ax[k].contour(((f<=t))[::2,::2].astype(float),levels=[0.5],colors='w',linewidths=0.6)
    ax[k].set_title(f'Δ = {D}: arrival step (1 bubble, {tmax} steps)',fontsize=10)
    for s in ax[k].spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(OUT+'fig_maps.png',dpi=90)
# --- fig 2: S vs ell, P(S), S(q)
def clusters(D):
    cs=[]
    for p in glob.glob(f'data/vb_big/D{D}/s*/vbaval_seed*.dat'):
        d=np.loadtxt(p.replace('vbaval','vb'),ndmin=2); ok=d[(d[:,0]>0)&(d[:,6]==1),0]
        c=np.loadtxt(p,ndmin=2); cs.append(c[np.isin(c[:,0],ok)&(c[:,2]==0)])
    c=np.vstack(cs); S=c[:,1]; cxx,cyy,cxy=c[:,5],c[:,6],c[:,7]; tr=cxx+cyy; disc=np.sqrt(np.maximum(tr**2/4-(cxx*cyy-cxy**2),0))
    lp=np.maximum(tr/2+disc,1e-12); lm=np.maximum(tr/2-disc,1/12); return S,np.sqrt(S/(np.pi*np.sqrt(lp*lm)))*np.sqrt(lp)
def wall_sq(f,t,L):
    yy,xx=np.nonzero(f<=t); dx=xx-L/2+0.5; dy=yy-L/2+0.5; r=np.hypot(dx,dy); R=np.sqrt(len(r)/np.pi); k=r>0.5*R
    N=int(np.pi*R); b=((np.arctan2(dy[k],dx[k])+np.pi)/(2*np.pi)*N).astype(int)%N
    Rt=np.zeros(N); np.maximum.at(Rt,b,r[k]+0.5); e=Rt==0
    if e.any(): j=np.arange(N); Rt[e]=np.interp(j[e],j[~e],Rt[~e],period=N)
    h=np.fft.rfft(Rt-Rt.mean())/N; return np.arange(1,len(h))/R,(2*np.pi*R*np.abs(h)**2)[1:]
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
ax[0].axhspan(4,400,color='0.9',zorder=0); ax[0].text(1.3,30,'paper window\nS = 4–400',fontsize=8,color='0.35')
for D in ['0.3','1.0']:
    S,l=clusters(D); m=S>=1
    e=np.logspace(0,np.log10(l.max()),26); i=np.digitize(l,e); xb=[];yb=[]
    for j in range(1,len(e)):
        k=i==j
        if k.sum()>=5: xb.append(np.exp(np.log(l[k]).mean())); yb.append(np.exp(np.log(S[k]).mean()))
    ax[0].loglog(xb,yb,'o-',ms=3.5,lw=1.5,color=C[D],label=f'Δ = {D}')
    x=np.logspace(1,np.log10(S.max()),22); h,_=np.histogram(S[S>=10],x); xc=np.sqrt(x[1:]*x[:-1]); ok=h>0
    ax[1].loglog(xc[ok],(h/np.diff(x)/(S>=10).sum())[ok],'o-',ms=3.5,lw=1.5,color=C[D],label=f'Δ = {D}')
    sq=[]
    for p in glob.glob(f'data/vb_big/D{D}/s*/vbflip_seed*.bin'):
        d=np.loadtxt(p.replace('vbflip_','vb_').replace('.bin','.dat'),ndmin=2); j=np.nonzero((d[:,6]==1)&(d[:,2]>=550))[0]
        if len(j): sq.append(wall_sq(np.fromfile(p,dtype=np.int32).reshape(2048,2048),int(d[j[0],0]),2048))
    n=min(len(s[1]) for s in sq); Sq=np.mean([s[1][:n] for s in sq],0); q=sq[0][0][:n]
    e=np.logspace(np.log10(q[0]),np.log10(q[-1]),30); i=np.digitize(q,e)
    xb=[np.exp(np.log(q[i==j]).mean()) for j in range(1,30) if (i==j).sum()]; yb=[np.exp(np.log(Sq[i==j]).mean()) for j in range(1,30) if (i==j).sum()]
    ax[2].loglog(xb,yb,'o-',ms=3.5,lw=1.5,color=C[D],label=f'Δ = {D} (R ≈ 550, {len(sq)} bubbles)')
xr=np.array([3,600]); ax[0].loglog(xr,0.9*xr**2.25,':',color='0.3',lw=1,label='ℓ^2.25 (qEW)'); ax[0].loglog(xr,2.2*xr**1.63,'--',color='0.3',lw=1,label='ℓ^1.63 (paper)')
ax[0].axvline(LM,color='0.75',lw=1); ax[0].text(LM*1.05,15,'L_m',fontsize=8,color='0.4')
ax[0].set(xlabel='ℓ (PCA, lattice units)',ylabel='⟨S⟩',title='Cluster shape: S vs ℓ',ylim=(5,1e6)); ax[0].legend(fontsize=8,frameon=False)
xr=np.array([10,2e5]); ax[1].loglog(xr,0.3*xr**-1.11,':',color='0.3',lw=1,label='S^-1.11 (qEW)'); ax[1].loglog(xr,0.5*xr**-1.26,'--',color='0.3',lw=1,label='S^-1.26 (qKPZ)')
ax[1].set(xlabel='S',ylabel='P(S)',title='Avalanche sizes (S ≥ 10)'); ax[1].legend(fontsize=8,frameon=False)
xr=np.array([2*np.pi/LM,0.5]); ax[2].loglog(xr,0.012*xr**-3.5,':',color='0.3',lw=1,label='q^-3.5 (ζ = 1.25)'); ax[2].loglog(xr,0.25*xr**-2.26,'--',color='0.3',lw=1,label='q^-2.26 (ζ = 0.63)')
ax[2].axvline(2*np.pi/LM,color='0.75',lw=1); ax[2].set(xlabel='q = n/R',ylabel='S(q)',title='Wall structure factor of R(θ)'); ax[2].legend(fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(OUT+'fig_stats.png',dpi=90)
