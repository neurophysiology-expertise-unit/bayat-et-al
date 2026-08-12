"""Test 2 REDONE (const-I0): a construction where the recruited fraction genuinely FALLS with
alpha while sigma_eff RISES. Excitability drive is inverted: gdrive = gamma_base*(OFFSET - alpha)
so higher alpha => less drive => fewer units past Hopf (f_det falls), but sigma_eff=sigma(1+4a)
still rises. If A_act rises (follows the noise), f_det was a correlate; if A_act falls (follows
f_det), Hopf recruitment is causal. f_det computed INSTANTANEOUSLY (fraction with trace(a)>=0),
valid for non-monotonic drive.
Run: python phase3_confound.py"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from phase2_coupling import ALPHAS
from pathlib import Path
OFFSET = 1.2

@njit(fastmath=True, cache=False)
def sweep(seed, avals, steps, nx, ny, sigma):
    np.random.seed(seed); na = avals.shape[0]; n = nx*ny
    gb = np.random.uniform(0.05,0.34,(nx,ny))*(1.0+2.0*np.random.standard_normal((nx,ny)))
    I0b = np.random.uniform(0.01,0.15,(nx,ny)); tb = np.random.uniform(0.5,1.1,(nx,ny))
    D0 = np.random.uniform(0.05,0.5,(nx,ny)); kap = np.random.uniform(1.0,4.0,(nx,ny))
    C = np.random.uniform(-0.1,0.3,(nx,ny)); h = np.random.uniform(0.4,1.2,(nx,ny))
    t0 = int(0.3*steps); sq = DT**0.5; af = np.zeros((na,n))
    for idx in range(na):
        a = avals[idx]
        I0 = 0.05 + I0b; tau = 10.0/((1.0+0.8*a)*tb); De = D0/(1.0+(kap*a)**4)
        th = THETA_BASE + 0.7*a; se = sigma*(1.0+4.0*a)
        gd = gb*(OFFSET - a)                     # inverted drive: falls with alpha
        cc = np.zeros((nx,ny)); cnt = 0
        for t in range(steps):
            noise = se*NOISE_MULT*np.random.standard_normal((nx,ny))
            Ca = 0.5*(1.0+np.tanh(ETA*(C-th))); diff = De*laplacian(Ca)
            dC = C-(C**3)/3.0-h+I0+gd+diff; dh = (C+A_FHN-B_FHN*h)/tau
            C = C+DT*dC+sq*noise; h = h+DT*dh; C = np.minimum(np.maximum(C,-4.0),4.0)
            if t>=t0: cc += Ca; cnt += 1
        af[idx,:] = (cc/cnt).reshape(n)
    return af, gb.reshape(n), (0.05+I0b).reshape(n), tb.reshape(n)

@njit(parallel=True, fastmath=True, cache=False)
def ens(seeds, avals, steps, nx, ny, sigma):
    ns=seeds.shape[0]; na=avals.shape[0]; n=nx*ny
    AF=np.zeros((ns,na,n)); G=np.zeros((ns,n)); I=np.zeros((ns,n)); T=np.zeros((ns,n))
    for i in prange(ns):
        af,g,i0c,tb = sweep(seeds[i],avals,steps,nx,ny,sigma); AF[i]=af; G[i]=g; I[i]=i0c; T[i]=tb
    return AF,G,I,T

def Cstar(drive):
    q = -3.0*(drive - A_FHN/B_FHN); D=(q/2.0)**2+(0.75/3.0)**3; s=np.sqrt(D)
    return np.cbrt(-q/2.0+s)+np.cbrt(-q/2.0-s)

def main():
    L=32; ns=40; T=200.0; steps=int(T/DT); seeds=np.arange(11,11+ns); sig=SIGMA_EM_PREDICTED
    t0=time.time()
    AF,G,I0c,TB = ens(seeds,ALPHAS,steps,L,L,sig)
    print(f"confound run ({time.time()-t0:.0f}s)\n")
    AA = AF.mean((0,2)); se = sig*(1.0+4.0*ALPHAS)
    # instantaneous f_det: fraction with trace(a)>=0, drive = i0c + gamma_base*(OFFSET-a)
    fdet=np.zeros(len(ALPHAS))
    for j,a in enumerate(ALPHAS):
        drive = I0c + G*(OFFSET-a); th=10.0/((1.0+0.8*a)*TB)
        tr = (1.0-Cstar(drive)**2) - B_FHN/th
        fdet[j] = np.mean(tr>=0)
    print("TEST 2 (inverted drive): f_det should FALL with alpha, sigma_eff RISES")
    print(f"  {'alpha':>6} {'f_det':>8} {'A_act':>8} {'sigma_eff':>10}")
    for a,fd,aa,s in zip(ALPHAS,fdet,AA,se):
        print(f"  {a:6.3f} {fd:8.4f} {aa:8.4f} {s:10.4f}")
    print(f"\n  f_det 0->1.11: {fdet[-1]-fdet[0]:+.4f}   A_act: {AA[-1]-AA[0]:+.4f}   "
          f"sigma_eff: {se[-1]-se[0]:+.4f}")
    kf = int(fdet.argmax()); ka = int(AA.argmax())
    print(f"  f_det peaks @a={ALPHAS[kf]:.3f}; A_act peaks @a={ALPHAS[ka]:.3f}")
    if fdet[-1] < fdet[kf]-1e-6 and AA[-1] > AA[0]:
        print("  -> f_det FALLS at high alpha while A_act RISES => A_act follows NOISE, f_det a correlate")
    else:
        print("  -> inspect: see whether A_act tracks f_det or sigma_eff")

if __name__ == "__main__":
    main()
