#!/usr/bin/env python3
"""Upper/lower asymmetry at alpha=0.  For a symmetric section the exact solution
is mirror-symmetric, so ANY upper-lower difference is pure discretisation error."""
import sys, os, numpy as np
MINF, GAM = float(os.environ.get("MINF", 0.8)), 1.4
cpS = (2/(GAM*MINF**2))*(((2+(GAM-1)*MINF**2)/(GAM+1))**(GAM/(GAM-1))-1)
def foot(x, cp):
    o = np.argsort(x); x, cp = x[o], cp[o]; r = np.nan
    for i in range(len(x)-1):
        if cp[i] < cpS <= cp[i+1] and x[i] > 0.2:
            t = (cpS-cp[i])/(cp[i+1]-cp[i]); r = x[i]+t*(x[i+1]-x[i])
    return r
print(f"{'case':<10}{'shock up':>10}{'shock lo':>10}{'|u-l|':>9}{'L2(Cp_u-Cp_l)':>15}{'max|dCp|':>10}")
for f in sys.argv[1:]:
    d = np.loadtxt(f); x = d[:,0]-d[:,0].min(); c = x.max(); x /= c
    cp, s = d[:,2], d[:,5]
    xu, cu = x[s>0], cp[s>0]; xl, cl = x[s<0], cp[s<0]
    fu, fl_ = foot(xu, cu), foot(xl, cl)
    g = np.linspace(0.02, 0.98, 300)
    du = np.interp(g, *(lambda a,b: (a[np.argsort(a)], b[np.argsort(a)]))(xu, cu))
    dl = np.interp(g, *(lambda a,b: (a[np.argsort(a)], b[np.argsort(a)]))(xl, cl))
    nm = os.path.basename(f).replace('.dat','')
    print(f"{nm:<10}{fu:>10.4f}{fl_:>10.4f}{abs(fu-fl_):>9.4f}"
          f"{np.sqrt(np.mean((du-dl)**2)):>15.5f}{np.max(np.abs(du-dl)):>10.4f}")
