#!/usr/bin/env python3
"""Figures of the quasi-static pilot of 2026-10-29 (docs/plans/qs_delta_sweep.md),
from data/pilot_qs (vader, L = 512) and data/pilot_qs_L1024 (cluster, L = 1024)
into results/pilot_qs/:
  pilot_snapshots.png     metastable configurations, L = 512, Delta = 0.3, 1.0, 1.5
  pilot_drive.png         F(W), L = 1024, H0 = 3e-5 and 1e-5
  pilot_avalanches.png    P(S) per H0, and the cutoff S_m = <S^2>/2<S> against L_m
  pilot_correlator.png    connected correlator C(w)/C(0) of F
  pilot_L512_vs_L1024.png P(S), <F>, sd(F) at equal L_m for L = 512 and 1024
  compare_dW.png          P(S) for dW = 0.5, 0.1, 0.02 over the same stretch of W

  python3 scripts/pilot_qs_plots.py   (from the repository root)
"""
import glob, re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
B=str(Path(__file__).resolve().parents[1]); OUT=B+'/results/pilot_qs'
Path(OUT).mkdir(parents=True, exist_ok=True)
C={'0.3':'#2a78d6','1.0':'#eb6834','1.5':'#1baf7a'}
LS={'3e-5':'-','1e-5':'--'}; LM={'3e-5':125,'1e-5':217}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'0.9','font.size':9})
def load(p, transient=0.1):
    a=np.loadtxt(p); return a[int(len(a)*transient):]
def L1(D,H,dW): return f'{B}/data/pilot_qs_L1024/D{D}_H{H}_dW{dW}/vd_seed1.dat'
runs=[(D,H) for D in ['0.3','1.0','1.5'] for H in ['3e-5','1e-5'] if glob.glob(L1(D,H,'0.5'))]

# 1. drive F(W)
fig,ax=plt.subplots(3,1,figsize=(10,7.5),sharex=True)
for k,D in enumerate(['0.3','1.0','1.5']):
    for H in ['3e-5','1e-5']:
        p=L1(D,H,'0.5')
        if not glob.glob(p): continue
        a=load(p,0); ax[k].plot(a[:,1],a[:,3],LS[H],lw=0.8,color=C[D],alpha=1 if H=='3e-5' else 0.6,label=f'H0={H} (L_m={LM[H]})')
    ax[k].set_ylabel('F'); ax[k].set_title(f'Δ={D}',loc='left',fontsize=9); ax[k].legend(fontsize=7,loc='upper right')
ax[-1].set_xlabel('W (filas)'); fig.suptitle('Fuerza de pinning F(W), cuasiestático, L=1024, dW=0.5, 1 seed',fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'/pilot_drive.png',dpi=130); plt.close()

# 2. correlator
fig,ax=plt.subplots(1,1,figsize=(7,4.2)); ax=[ax]
for D,H in runs:
    a=load(L1(D,H,'0.1')) if glob.glob(L1(D,H,'0.1')) else load(L1(D,H,'0.5'))
    dW=a[1,1]-a[0,1]; f=a[:,3]-a[:,3].mean(); n=len(f); ml=int(300/dW)
    Cw=np.array([np.mean(f[:n-k]*f[k:]) for k in range(0,ml,max(1,int(1/dW)))]); w=np.arange(len(Cw))*max(1,int(1/dW))*dW
    ax[0].plot(w,Cw/Cw[0],LS[H],color=C[D],label=f'Δ={D}, L_m={LM[H]}')
ax[0].axhline(0,color='0.5',lw=0.6); ax[0].set_xlabel('w (filas)'); ax[0].set_ylabel('C(w)/C(0)'); ax[0].set_title('correlador conectado de F, normalizado')
for x in ax: x.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'/pilot_correlator.png',dpi=130); plt.close()

# 3. avalanches
def PS(S,lo=10):
    S=S[S>=lo]; bins=np.logspace(1,np.log10(S.max())+0.01,18); h,e=np.histogram(S,bins=bins); c=np.sqrt(e[1:]*e[:-1]); d=h/np.diff(e)/len(S); return c[h>0],d[h>0],len(S)
fig,ax=plt.subplots(1,3,figsize=(14,4.2))
Sm={}
for D,H in runs:
    p=L1(D,H,'0.1') if (H=='3e-5' and glob.glob(L1(D,H,'0.1'))) else L1(D,H,'0.5')
    S=load(p)[1:,4]; c,d,n=PS(S); Sb=S[S>=10]; Sm[(D,H)]=np.mean(Sb**2)/2/np.mean(Sb)
    ax[0 if H=='3e-5' else 1].loglog(c,d,'o'+LS[H],ms=4,color=C[D],label=f'Δ={D} ({n} aval.)')
for k,H in enumerate(['3e-5','1e-5']):
    s=np.logspace(1,5); ax[k].loglog(s,0.3*s**-1.11,':',color='0.4',lw=1,label='S^-1.11 (qEW)')
    ax[k].set_xlabel('S (sitios)'); ax[k].set_ylabel('P(S), S≥10'); ax[k].set_title(f'H0={H}, L_m={LM[H]}'); ax[k].legend(fontsize=7)
for D in ['0.3','1.5']:
    x=[LM['3e-5'],LM['1e-5']]; y=[Sm[(D,'3e-5')],Sm[(D,'1e-5')]]; sl=np.log(y[1]/y[0])/np.log(x[1]/x[0])
    ax[2].loglog(x,y,'o-',color=C[D],label=f'Δ={D}: pendiente {sl:.2f} → ζ≈{sl-1:.2f}')
ax[2].loglog([LM['3e-5']],[Sm[('1.0','3e-5')]],'o',color=C['1.0'],label='Δ=1.0 (solo L_m=125)')
ax[2].set_xlabel('L_m'); ax[2].set_ylabel('S_m = ⟨S²⟩/2⟨S⟩'); ax[2].set_title('cutoff vs L_m: S_m ~ L_m^(1+ζ)'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'/pilot_avalanches.png',dpi=130); plt.close()

# 4. L=512 vs L=1024 at L_m=125
fig,ax=plt.subplots(1,2,figsize=(11,4))
for k,D in enumerate(['0.3','1.5']):
    for L,p,mk in [(512,f'{B}/data/pilot_qs/D{D}_dW0.1/vd_seed1.dat','s'),(1024,L1(D,'3e-5','0.1'),'o')]:
        a=load(p); S=a[1:,4]; c,d,n=PS(S); rows=a[-1,1]-a[0,1]
        ax[k].loglog(c,d,mk+'-',ms=4,color=C[D],mfc='none' if L==512 else C[D],label=f'L={L}: ⟨F⟩={a[:,3].mean():.5f}, sd={a[:,3].std():.5f}, {n/rows:.2f} aval./fila')
    ax[k].set_title(f'Δ={D}, L_m=125, dW=0.1'); ax[k].set_xlabel('S (sitios)'); ax[k].set_ylabel('P(S), S≥10'); ax[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'/pilot_L512_vs_L1024.png',dpi=130); plt.close()

# 5. snapshots (L=512)
fig,ax=plt.subplots(1,3,figsize=(13,3.4))
for k,(D,Ly) in enumerate([('0.3',512),('1.0',1024),('1.5',1024)]):
    fs=sorted(glob.glob(f'{B}/data/pilot_qs/D{D}_dW0.5/vdfield_step*_seed1.bin'),key=lambda s:int(re.search(r'step(\d+)',s).group(1)))
    phi=np.fromfile(fs[-1],dtype=np.float32).reshape(Ly,512); u=int(np.count_nonzero(phi>0)/512)
    lo,hi=max(0,u-160),min(Ly,u+160)
    ax[k].imshow(phi[lo:hi],origin='lower',cmap='RdBu_r',vmin=-1.2,vmax=1.2,extent=[0,512,lo,hi],aspect='equal',interpolation='nearest'); ax[k].grid(False)
    st=re.search(r'step(\d+)',fs[-1]).group(1)
    ax[k].set_title(f'Δ={D}, paso {st} (L=512)'); ax[k].set_xlabel('x'); ax[k].set_ylabel('y (caja)')
fig.suptitle('Configuraciones metaestables (φ: rojo > 0, azul < 0), ±160 filas alrededor de la pared',fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'/pilot_snapshots.png',dpi=130); plt.close()

# 6. dW comparison over a common stretch of W
R=B+'/data/pilot_qs_L1024'
# compare_dW
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
cols_dW={'0.5':'#9ecae1','0.1':'#4292c6','0.02':'#08306b'}
for k,D in enumerate(['0.3','1.5']):
    for dW in ['0.5','0.1','0.02']:
        a=np.loadtxt(f'{R}/D{D}_H3e-5_dW{dW}/vd_seed1.dat')
        sel=(a[:,1]>=560)&(a[:,1]<=1510); b=a[sel]
        S=b[1:,4]; Sb=S[S>=10]; F=b[:,3]; nc=np.sum(b[:,6]!=1)
        print(f'D{D} dW{dW:5s} W {b[0,1]:.0f}-{b[-1,1]:.0f}: steps={len(b)} <F>={F.mean():.5f} sd={F.std():.5f} nS>=10={len(Sb)} <S>={Sb.mean():.0f} S2/2S={np.mean(Sb**2)/2/np.mean(Sb):.0f} Smax={S.max():.0f} notconv={nc} sweeps/step={b[:,5].mean():.0f}')
        bins=np.logspace(1,5.5,20); h,e=np.histogram(Sb,bins=bins); c=np.sqrt(e[1:]*e[:-1])
        dens=h/np.diff(e)/len(Sb)
        ax[k].loglog(c[h>0],dens[h>0],'o-',ms=4,color=cols_dW[dW],label=f'dW={dW} ({len(Sb)})')
    ax[k].set_title(f'Δ={D}, L=1024, L_m=125, W∈[560,1510]'); ax[k].set_xlabel('S (sitios)'); ax[k].set_ylabel('P(S), S≥10'); ax[k].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT+'/compare_dW.png',dpi=130); plt.close()
