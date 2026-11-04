#!/usr/bin/env python3
"""Snapshots of a main_vb.cu bubble run: the last dumped field, with the
wall (phi = 0 contour) of every earlier dump on top, coloured by step,
and the drive F against 1/R next to the Laplace line sigma/2R.

  python3 scripts/vb_snapshots.py <rundir> [--seed 1] [--out bubble.png] [--hd h]
"""
import argparse, glob, re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap=argparse.ArgumentParser(); ap.add_argument('rundir'); ap.add_argument('--seed',type=int,default=1)
ap.add_argument('--out',default=None); ap.add_argument('--hd',type=float,default=None,help='planar threshold, for the line h_d + sigma/2R')
a=ap.parse_args()
SIGMA=2*np.sqrt(2)/3
hdr=open(f'{a.rundir}/vb_seed{a.seed}.dat').readline()
L=int(re.search(r'L=(\d+)',hdr).group(1)); D=re.search(r'AMPDIS=(\S+)',hdr).group(1)
dumps=sorted(glob.glob(f'{a.rundir}/vbfield_step*_seed{a.seed}.bin'),key=lambda p:int(re.search(r'step(\d+)_',p).group(1)))
steps=[int(re.search(r'step(\d+)_',p).group(1)) for p in dumps]
fig,ax=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={'width_ratios':[1.1,1]})
last=np.fromfile(dumps[-1],np.float32).reshape(L,L)
ax[0].imshow(last,origin='lower',cmap='Greys',vmin=-1.2,vmax=1.2,interpolation='nearest')
cm=plt.get_cmap('viridis')
for k,p in enumerate(dumps):
    f=np.fromfile(p,np.float32).reshape(L,L)
    ax[0].contour(f,levels=[0],colors=[cm(k/max(len(dumps)-1,1))],linewidths=0.7)
ax[0].set(title=f'Δ={D}, L={L}: φ at step {steps[-1]}, walls at steps {steps[0]}–{steps[-1]}',xticks=[],yticks=[])
sm=plt.cm.ScalarMappable(cmap=cm,norm=plt.Normalize(steps[0],steps[-1])); fig.colorbar(sm,ax=ax[0],label='step',shrink=0.8)
d=np.loadtxt(f'{a.rundir}/vb_seed{a.seed}.dat'); d=d[d[:,6]>=0]
R,F=d[:,2],d[:,3]
ax[1].plot(1/R,F,'.',ms=3,label='F (spring)')
x=np.linspace(0,1/R.min()*1.05,50); ax[1].plot(x,SIGMA/2*x,'k--',lw=1,label='σ/2R')
if a.hd is not None: ax[1].plot(x,a.hd+SIGMA/2*x,'r:',lw=1,label=f'h_d + σ/2R (h_d={a.hd})')
ax[1].set(xlabel='1/R',ylabel='F',title=f'R from {R.min():.0f} to {R.max():.0f}, {len(d)} steps'); ax[1].legend()
ax[1].spines[['top','right']].set_visible(False)
fig.tight_layout(); out=a.out or f'{a.rundir}/bubble_seed{a.seed}.png'; fig.savefig(out,dpi=110); print(out)
