#!/usr/bin/env python3
# Exact Sod shock-tube solution and comparison with the solver output.
import numpy as np
from scipy.optimize import brentq

gamma = 1.4
# left / right states (rho, u, p)
rhoL, uL, pL = 1.0, 0.0, 1.0
rhoR, uR, pR = 0.125, 0.0, 0.1
x0 = 0.5      # initial discontinuity
t  = 0.20

aL = np.sqrt(gamma*pL/rhoL)
aR = np.sqrt(gamma*pR/rhoR)

def f(p, pk, rhok, ak):
    if p > pk:  # shock
        A = 2.0/((gamma+1.0)*rhok)
        B = (gamma-1.0)/(gamma+1.0)*pk
        return (p-pk)*np.sqrt(A/(p+B))
    else:       # rarefaction
        return 2.0*ak/(gamma-1.0)*((p/pk)**((gamma-1.0)/(2.0*gamma)) - 1.0)

def fprime_total(p):
    return f(p, pL, rhoL, aL) + f(p, pR, rhoR, aR) + (uR-uL)

pstar = brentq(fprime_total, 1e-8, 10.0)
ustar = 0.5*(uL+uR) + 0.5*(f(pstar,pR,rhoR,aR) - f(pstar,pL,rhoL,aL))

# densities in the star region
if pstar > pL:
    rhoLs = rhoL*((pstar/pL+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*pstar/pL+1))
else:
    rhoLs = rhoL*(pstar/pL)**(1.0/gamma)
if pstar > pR:
    rhoRs = rhoR*((pstar/pR+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*pstar/pR+1))
else:
    rhoRs = rhoR*(pstar/pR)**(1.0/gamma)

def sample(x):
    s = (x-x0)/t
    if s <= ustar:        # left of contact
        if pstar > pL:    # left shock
            SL = uL - aL*np.sqrt((gamma+1)/(2*gamma)*pstar/pL + (gamma-1)/(2*gamma))
            if s <= SL: return rhoL, uL, pL
            return rhoLs, ustar, pstar
        else:             # left rarefaction
            aLs = aL*(pstar/pL)**((gamma-1)/(2*gamma))
            SHL = uL - aL
            STL = ustar - aLs
            if s <= SHL: return rhoL, uL, pL
            if s >= STL: return rhoLs, ustar, pstar
            u = 2/(gamma+1)*(aL + (gamma-1)/2*uL + s)
            a = 2/(gamma+1)*(aL + (gamma-1)/2*(uL - s))
            rho = rhoL*(a/aL)**(2/(gamma-1))
            p = pL*(a/aL)**(2*gamma/(gamma-1))
            return rho, u, p
    else:                 # right of contact
        if pstar > pR:    # right shock
            SR = uR + aR*np.sqrt((gamma+1)/(2*gamma)*pstar/pR + (gamma-1)/(2*gamma))
            if s >= SR: return rhoR, uR, pR
            return rhoRs, ustar, pstar
        else:             # right rarefaction
            aRs = aR*(pstar/pR)**((gamma-1)/(2*gamma))
            SHR = uR + aR
            STR = ustar + aRs
            if s >= SHR: return rhoR, uR, pR
            if s <= STR: return rhoRs, ustar, pstar
            u = 2/(gamma+1)*(-aR + (gamma-1)/2*uR + s)
            a = 2/(gamma+1)*(aR - (gamma-1)/2*(uR - s))
            rho = rhoR*(a/aR)**(2/(gamma-1))
            p = pR*(a/aR)**(2*gamma/(gamma-1))
            return rho, u, p

print(f"exact star state:  p* = {pstar:.6f},  u* = {ustar:.6f},  rhoL* = {rhoLs:.6f},  rhoR* = {rhoRs:.6f}")

data = np.loadtxt("output/sod_profile.dat")
x, rho, u, p = data[:,0], data[:,1], data[:,2], data[:,3]

ex = np.array([sample(xi) for xi in x])
rho_e, u_e, p_e = ex[:,0], ex[:,1], ex[:,2]

def l1(a, b): return np.mean(np.abs(a-b))
print(f"L1 errors:  rho = {l1(rho,rho_e):.4e}   u = {l1(u,u_e):.4e}   p = {l1(p,p_e):.4e}")
print(f"Linf errors: rho = {np.max(np.abs(rho-rho_e)):.4e}  u = {np.max(np.abs(u-u_e)):.4e}  p = {np.max(np.abs(p-p_e)):.4e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xf = np.linspace(x.min(), x.max(), 2000)
    exf = np.array([sample(xi) for xi in xf])
    fig, ax = plt.subplots(1, 3, figsize=(15,4.2))
    for k,(num,lab) in enumerate([(rho,"density"),(u,"velocity"),(p,"pressure")]):
        ax[k].plot(xf, exf[:,k], 'k-', lw=1.5, label="exact")
        ax[k].plot(x, num, 'r.', ms=4, label="wave3d (3D, pseudo-2D)")
        ax[k].set_title(f"Sod {lab}, t={t}")
        ax[k].set_xlabel("x"); ax[k].legend(); ax[k].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/sod_validation.png", dpi=110)
    print("wrote output/sod_validation.png")
except Exception as e:
    print("plot skipped:", e)
