#!/usr/bin/env python3
"""Transient channel decay: EXACT validation of the viscous Brinkman no-slip wall.

Fluid at rest-forcing-free between two no-slip walls, uniform u=U at t=0:
  u(y,t) = sum_n 4U/((2n+1)pi) sin((2n+1)pi(y-y0)/(2h)) exp(-nu ((2n+1)pi/(2h))^2 t)
Periodic in x AND y (the y wrap joins solid to solid), so there is no inflow,
no outflow, no blockage and no freestream ambiguity -- every discrepancy is the
wall treatment or the viscous operator.
"""
import sys, numpy as np
fn, YLO, YHI, NU, U, T = sys.argv[1], 0.1, 0.4, 1e-3, 1.0, float(sys.argv[2])
h = 0.5*(YHI-YLO)
def exact(y, t, N=200):
    s = np.zeros_like(y)
    for n in range(N):
        k = (2*n+1)*np.pi/(2*h)
        s += 4*U/((2*n+1)*np.pi)*np.sin(k*(y-YLO))*np.exp(-NU*k*k*t)
    return s
d = np.loadtxt(fn); x, y, u = d[:,0], d[:,1], d[:,2]
m = (y > YLO) & (y < YHI)
yy, uu = y[m], u[m]
# average over x at each y (solution is x-uniform)
yb = np.unique(np.round(yy, 7))
prof = np.array([uu[np.abs(yy-t0) < 1e-9].mean() for t0 in yb])
ex = exact(yb, T)
err = prof - ex
print(f"  centreline: num {np.interp(0.25, yb, prof):.5f}   exact {np.interp(0.25, yb, ex):.5f}")
print(f"  L2 error over the channel = {np.sqrt(np.mean(err**2)):.4e}   Linf = {np.max(np.abs(err)):.4e}")
print(f"  relative L2 = {np.sqrt(np.mean(err**2))/np.sqrt(np.mean(ex**2)):.4%}")
print(f"\n  {'y':>8}{'num':>10}{'exact':>10}{'diff':>11}")
for t0 in (0.105, 0.115, 0.13, 0.16, 0.20, 0.25, 0.30, 0.34, 0.37, 0.385, 0.395):
    a, b = np.interp(t0, yb, prof), np.interp(t0, yb, ex)
    print(f"  {t0:>8.3f}{a:>10.5f}{b:>10.5f}{a-b:>11.2e}")
# wall shear -> the quantity the no-slip wall actually has to get right
sel = (yb > YLO) & (yb < YLO+0.25*h)
gn = np.polyfit(yb[sel], prof[sel], 1)[0]
ge = np.polyfit(yb[sel], exact(yb[sel], T), 1)[0]
print(f"\n  wall shear du/dy: num {gn:.4f}   exact {ge:.4f}   err {100*(gn-ge)/ge:+.1f}%")
