#!/usr/bin/env python3
"""Space-time clustering and wall structure factor of main_vb.cu bubbles,
from the flip map vbflip_seed<seed>.bin (int32, L*L: step at which each
site last became phi>0; -1 inside the initial disk; INT_MAX outside).

1. Clustering as in Durin et al. (PRB 110, L020405 (2024)): two flipped
   sites belong to the same cluster if they are within ds (Euclidean,
   lattice units) and flipped within dt steps of each other. dt = 0,
   ds = 1 is the quasi-static avalanche (one connected cluster of one
   step, as in vbaval); dt > 0 joins avalanches that follow each other
   in place, like the frame events of the experiment. Per (dt, ds):
   P(S), and S against ell_PCA = a sqrt(lambda_+) (Durin et al. S4).
2. S(q) of the wall rho(theta) = R(theta) - <R>, R(theta) the outermost
   bubble site per angular bin (bin ~ 2 sites at the wall; an empty bin is interpolated), at the steps
   where the mean radius crosses --radii; S(q) = 2 pi R |rho_n|^2 with
   q = n/R (the planar L |h_q|^2), averaged over seeds. A self-affine
   wall has S(q) ~ q^-(1+2 zeta).

Steps 0 (relaxation of the initial disk) and the last, cut one are left
out (their sites count as inside / outside).

  python3 scripts/bubble_cluster.py data/vb_flip --out results/vb_flip --H0 3e-5
"""
import argparse, glob, re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ap=argparse.ArgumentParser(); ap.add_argument('datadir'); ap.add_argument('--out',default='.')
ap.add_argument('--H0',type=float,default=3e-5); ap.add_argument('--smin',type=float,default=10)
ap.add_argument('--lmin',type=float,default=8)
ap.add_argument('--clusterings',default='0:1,0:2,1:2,3:2,10:2,30:2',help='dt:ds pairs')
ap.add_argument('--radii',default='200,300,400')
a=ap.parse_args(); Path(a.out).mkdir(parents=True,exist_ok=True)
SIGMA=2*np.sqrt(2)/3; LM=np.sqrt(SIGMA/(2*a.H0)); NEVER=2**31-1
CL=[tuple(int(v) for v in p.split(':')) for p in a.clusterings.split(',')]
RAD=[float(r) for r in a.radii.split(',')]
C=['#2a78d6','#eb6834','#1baf7a','#8a5cc2','#d6a02a','#c2416b','#555555']

def runs(D):
    for p in sorted(glob.glob(f'{a.datadir}/D{D}/s*/vbflip_seed*.bin')):
        d=np.loadtxt(re.sub(r'vbflip_(seed\d+)\.bin',r'vb_\1.dat',p),ndmin=2)
        f=np.fromfile(p,dtype=np.int32); L=int(round(np.sqrt(f.size)))
        yield f.reshape(L,L), d

def clusters(f,tmax,dt,ds):
    """labels of the sites flipped in 1..tmax; returns per-cluster S, l+, l-, t0, t1"""
    valid=(f>=1)&(f<=tmax); n=int(valid.sum())
    ids=np.full(f.shape,-1,np.int64); ids[valid]=np.arange(n)
    offs=[(dx,dy) for dy in range(0,ds+1) for dx in range(-ds,ds+1) if dx*dx+dy*dy<=ds*ds and (dy>0 or dx>0)]
    r=[];c=[]
    for dx,dy in offs:
        g=np.roll(f,(-dy,-dx),axis=(0,1)); gi=np.roll(ids,(-dy,-dx),axis=(0,1))
        m=valid&(gi>=0)&(np.abs(f.astype(np.int64)-g)<=dt)
        r.append(ids[m]); c.append(gi[m])
    r=np.concatenate(r); c=np.concatenate(c)
    nc,lab=connected_components(coo_matrix((np.ones(len(r),np.int8),(r,c)),shape=(n,n)),directed=False)
    y,x=np.nonzero(valid); x=x.astype(float); y=y.astype(float); t=f[valid]  # bubble centred: no unwrapping
    S=np.bincount(lab,minlength=nc).astype(float)
    mx=np.bincount(lab,x,nc)/S; my=np.bincount(lab,y,nc)/S
    cxx=np.bincount(lab,x*x,nc)/S-mx**2; cyy=np.bincount(lab,y*y,nc)/S-my**2; cxy=np.bincount(lab,x*y,nc)/S-mx*my
    tr=cxx+cyy; disc=np.sqrt(np.maximum(tr**2/4-(cxx*cyy-cxy**2),0))
    lp=np.maximum(tr/2+disc,1/12); lm=np.maximum(tr/2-disc,1/12)
    t0=np.full(nc,NEVER); np.minimum.at(t0,lab,t); t1=np.zeros(nc,int); np.maximum.at(t1,lab,t)
    return S,lp,lm,t0,t1

def wall_sq(f,t,L):
    """S(q) of the outermost wall of {f <= t}"""
    yy,xx=np.nonzero(f<=t); dx=xx-L/2+0.5; dy=yy-L/2+0.5
    r=np.hypot(dx,dy); A=len(r); R=np.sqrt(A/np.pi); k=r>0.5*R
    N=int(np.pi*R); b=((np.arctan2(dy[k],dx[k])+np.pi)/(2*np.pi)*N).astype(int)%N  # bins ~2 sites wide at the wall
    Rt=np.zeros(N); np.maximum.at(Rt,b,r[k]+0.5)
    e=Rt==0                                                                           # a ray that misses every site
    if e.any(): j=np.arange(N); Rt[e]=np.interp(j[e],j[~e],Rt[~e],period=N)
    rho=Rt-Rt.mean(); h=np.fft.rfft(rho)/N
    n=np.arange(len(h)); return n[1:]/R, (2*np.pi*R*np.abs(h)**2)[1:], R, rho.std()

def logbin(x,lo,nb=20):
    x=x[x>=lo]; e=np.logspace(np.log10(lo),np.log10(x.max()*1.0001),nb+1); h,_=np.histogram(x,e)
    w=np.diff(e); xc=np.sqrt(e[1:]*e[:-1]); ok=h>0; return xc[ok],(h/w/len(x))[ok]
def binned(x,y,nb=14):
    e=np.logspace(np.log10(x.min()),np.log10(x.max()*1.0001),nb+1); i=np.digitize(x,e)-1
    k=[j for j in range(nb) if (i==j).sum()>=3]
    return np.array([np.exp(np.log(x[i==j]).mean()) for j in k]),np.array([np.exp(np.log(y[i==j]).mean()) for j in k])
def fit(x,y,lo,hi):
    m=(x>=lo)&(x<=hi)&(y>0)
    if m.sum()<8: return np.nan,np.nan
    c,cov=np.polyfit(np.log(x[m]),np.log(y[m]),1,cov=True); return c[0],np.sqrt(cov[0,0])

Ds=sorted({re.search(r'/D([0-9.]+)$',p).group(1) for p in glob.glob(f'{a.datadir}/D*')},key=float)
lines=[f'# H0={a.H0} L_m={LM:.0f}; clusters with S>={a.smin}; S~ell^(1+zeta) over {a.lmin}<=ell<=L_m',
       '# Delta dt ds n(S>=smin) <steps spanned> 1+zeta(ell_PCA) slope_P(S)(smin..S_m) max S']
fig,ax=plt.subplots(len(Ds),3,figsize=(15,4.6*len(Ds)),squeeze=False)
for i,D in enumerate(Ds):
    res={cl:[] for cl in CL}; sq={R:[] for R in RAD}; ns=0
    for f,d in runs(D):
        ok=d[:,6]==1; tmax=int(d[ok,0].max()); L=f.shape[0]; ns+=1
        for cl in CL: res[cl].append(np.column_stack(clusters(f,tmax,*cl)))
        for R in RAD:
            j=np.nonzero(ok&(d[:,2]>=R))[0]
            if len(j): sq[R].append(wall_sq(f,int(d[j[0],0]),L))
    for k,cl in enumerate(CL):
        z=np.vstack(res[cl]); z=z[z[:,0]>=a.smin]; S=z[:,0]
        aa=np.sqrt(S/(np.pi*np.sqrt(z[:,1]*z[:,2]))); lp=np.median(aa)*np.sqrt(z[:,1])
        s1,e1=fit(lp,S,a.lmin,LM); dur=(z[:,4]-z[:,3]+1).mean()
        x,y=logbin(S,a.smin); Sm=np.sqrt((S**2).mean()); tp,_=fit(x,y,a.smin,Sm)
        lab=f'Δt={cl[0]}, Δs={cl[1]}'
        ax[i,0].loglog(x,y,'o-',ms=3,color=C[k],label=f'{lab}: {len(S)} cl., pend. {tp:.2f}')
        xb,yb=binned(lp,S); ax[i,1].loglog(xb,yb,'o-',ms=3,color=C[k],label=f'{lab}: 1+ζ={s1:.2f}±{e1:.2f}')
        lines.append(f'{D} {cl[0]} {cl[1]} {len(S)} {dur:.1f} {s1:.3f}±{e1:.3f} {tp:.2f} {S.max():.0f}')
    xr=np.array([a.smin,1e5]); ax[i,0].loglog(xr,0.3*xr**-1.11,':',color='0.4',lw=1,label='S^{-1.11} (qEW)'); ax[i,0].loglog(xr,0.5*xr**-1.26,'-.',color='0.6',lw=1,label='S^{-1.26} (qKPZ)')
    ax[i,0].set(xlabel='S',ylabel='P(S)',title=f'Δ={D}, {ns} burbujas'); ax[i,0].legend(fontsize=7)
    xr=np.array([a.lmin,3*LM]); ax[i,1].loglog(xr,0.5*xr**2.25,':',color='0.5',lw=1,label='ℓ^{2.25} (qEW)'); ax[i,1].loglog(xr,2*xr**1.63,'-.',color='0.5',lw=1,label='ℓ^{1.63} (Durin)')
    ax[i,1].axvline(LM,color='0.8',lw=1); ax[i,1].set(xlabel='ℓ_PCA',ylabel='S',title=f'Δ={D}: clustering espaciotemporal'); ax[i,1].legend(fontsize=7)
    for k,R in enumerate(RAD):
        if not sq[R]: continue
        q=sq[R][0][0]; nmin=min(len(s[0]) for s in sq[R]); Rm=np.mean([s[2] for s in sq[R]]); w=np.mean([s[3] for s in sq[R]])
        Sq=np.mean([s[1][:nmin] for s in sq[R]],axis=0); q=np.arange(1,nmin+1)/Rm
        xb,yb=binned(q,Sq,nb=30); sl,el=fit(xb,yb,2*np.pi/LM,2*np.pi/(2*a.lmin))
        ax[i,2].loglog(xb,yb,'o-',ms=3,color=C[k],label=f'R≈{Rm:.0f} (w={w:.1f}): pend. {sl:.2f}±{el:.2f} → ζ={(-sl-1)/2:.2f}')
        lines.append(f'# S(q) Delta={D} R={Rm:.0f} seeds={len(sq[R])} width={w:.2f} slope(2pi/L_m..2pi/{2*a.lmin:.0f})={sl:.3f}±{el:.3f} zeta={(-sl-1)/2:.3f}')
    xr=np.array([2*np.pi/LM,1.0]); ax[i,2].loglog(xr,1e-2*xr**-3.5,':',color='0.5',lw=1,label='q^{-3.5} (qEW, ζ=1.25)'); ax[i,2].loglog(xr,1e-1*xr**-2.26,'-.',color='0.5',lw=1,label='q^{-2.26} (ζ=0.63)')
    ax[i,2].axvline(2*np.pi/LM,color='0.8',lw=1); ax[i,2].set(xlabel='q',ylabel='S(q)',title=f'Δ={D}: S(q) de R(θ)'); ax[i,2].legend(fontsize=7)
for x in ax.ravel(): x.spines[['top','right']].set_visible(False)
fig.suptitle(f'Burbuja cuasiestática: clustering espaciotemporal y S(q), H0={a.H0}, L_m={LM:.0f}',fontsize=11); fig.tight_layout()
fig.savefig(f'{a.out}/bubble_cluster.png',dpi=120); open(f'{a.out}/bubble_cluster.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
