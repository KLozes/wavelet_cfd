#!/usr/bin/env python3
"""Cf from a flat-plate dump, using the total stress at the FIRST cells above
the wall (the inner layer is a constant-stress layer, so (mu+mu_t) du/dy there
~= tau_w).  A linear fit over a tall window is NOT valid here: y+ of the first
cell is ~11, so the viscous sublayer is unresolved and a wall-gradient fit that
reaches into the wake returns nonsense (measured: negative Cf).
Calibrate against the solver's own uTau-based Cf before trusting it.
"""
import sys, numpy as np
fn, YW, NU, U, NC = sys.argv[1], 0.02, 1e-6, 1.0, int(sys.argv[2]) if len(sys.argv)>2 else 3
d = np.loadtxt(fn); x, y, u, mt = d[:,0], d[:,1], d[:,2], d[:,6]
print(f"  {'x':>6}{'Re_x':>10}{'Cf':>10}{'Cf corr':>10}{'err%':>8}{'mu_t/mu':>10}")
for xs in (0.4, 0.7, 1.0, 1.3):
    m = np.abs(x-xs) < 0.006
    if m.sum() < 8: continue
    yy, uu, mm = y[m], u[m], mt[m]
    # The window spans several x and several AMR levels, so many cells share a y.
    # Average per unique y first -- otherwise the "first NC cells" can all sit at
    # the SAME height and the gradient divides by zero.
    yb = np.unique(np.round(yy, 9))
    uu = np.array([u[m][np.abs(y[m]-t) < 1e-9].mean() for t in yb])
    mm = np.array([mt[m][np.abs(y[m]-t) < 1e-9].mean() for t in yb])
    yy = yb
    k = np.where(yy > YW)[0][:NC]
    if len(k) < 2: continue
    dudy = (uu[k[-1]] - uu[k[0]])/(yy[k[-1]] - yy[k[0]])
    mue  = NU + np.mean(mm[k])
    Cf, Rex = 2*mue*dudy/U**2, U*xs/NU
    corr = 0.0592*Rex**-0.2
    print(f"  {xs:>6.2f}{Rex:>10.3e}{Cf:>10.5f}{corr:>10.5f}{100*(Cf-corr)/corr:>8.1f}{np.mean(mm[k])/NU:>10.1f}")
