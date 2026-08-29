#!/usr/bin/env python3
"""Is the shock offset driven by the trailing-edge Kutta deficit?

Shock foot: interpolate where the upper-surface Cp crosses the SONIC value Cp*,
which is sub-sample-precise -- unlike a max-gradient estimate, which is quantised
to the ~64-point surface sampling and cannot resolve a 0.025c shift.
Kutta indicator: the TE pressure jump Cp_upper - Cp_lower, which must -> 0 for a
correctly enforced Kutta condition at a sharp trailing edge.
"""
import sys, numpy as np
import os
MINF, GAM = float(os.environ.get("MINF", 0.73)), 1.4
cpStar = (2/(GAM*MINF**2))*(((2 + (GAM-1)*MINF**2)/(GAM+1))**(GAM/(GAM-1)) - 1)

def probe(p):
    d = np.loadtxt(p)
    x, cp, side = d[:,0], d[:,2], d[:,5]
    x = x - x.min()                                  # LE at 0
    c = x.max()
    up, lo = side > 0, side < 0
    xu, cu = x[up]/c, cp[up]; o = np.argsort(xu); xu, cu = xu[o], cu[o]
    xl, cl = x[lo]/c, cp[lo]; o = np.argsort(xl); xl, cl = xl[o], cl[o]
    # shock foot: LAST upward crossing of Cp* on the upper surface
    xs = np.nan
    for i in range(len(xu)-1):
        if cu[i] < cpStar <= cu[i+1] and xu[i] > 0.2:
            t = (cpStar - cu[i])/(cu[i+1] - cu[i]); xs = xu[i] + t*(xu[i+1] - xu[i])
    # TE pressure jump (Kutta): last 2% of chord on each surface
    tu = cu[xu > 0.98].mean() if (xu > 0.98).any() else np.nan
    tl = cl[xl > 0.98].mean() if (xl > 0.98).any() else np.nan
    return xs, tu - tl, cu.min()

print(f"  sonic Cp* = {cpStar:.4f}")
print(f"{'case':<12}{'shock foot':>12}{'offset':>9}{'TE dCp (Kutta)':>16}{'min Cp':>9}")
base = None
for f in sys.argv[1:]:
    xs, dte, mn = probe(f)
    if base is None: base = xs
    nm = f.split('/')[-1].replace('.dat','')
    print(f"{nm:<12}{xs:>12.4f}{xs-base:>9.4f}{dte:>16.4f}{mn:>9.3f}")
