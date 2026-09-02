#!/usr/bin/env python3
"""Total fluid mass on the annulus vs time from the numbered svortex dumps --
is the density error a DRIFT of the mean (a conservation leak at the cut
boundary) or a shape error?   usage: svortex_mass_drift.py dir [dir ...]"""
import sys, glob, numpy as np
RI, RO = 1.0, 1.384; GAM = 1.4
def exact_rho(r, M=0.2):
    # isentropic vortex, rho_i = 1 at r = RI, u_theta = M c_i RI/r
    return (1 + 0.5*(GAM-1)*M*M*(1 - (RI/r)**2))**(1/(GAM-1))
s = np.linspace(-0.5, 0.5, 9)[:-1] + 1/16; SX, SY = np.meshgrid(s, s)
for d in sys.argv[1:]:
    files = sorted(glob.glob(f"{d}/svortex_field_*.dat"))
    print(f"\n{d.split('/')[-1]}:  t   mass err (rel)   mean rho err (interior)   L2 rho (interior)")
    for f in files[1::3] + [files[-1]]:
        a = np.loadtxt(f); a = a[a[:, 9] > 0]
        xs = np.unique(np.round(a[:, 0], 8)); h = np.min(np.diff(xs))
        r = np.hypot(a[:, 0], a[:, 1])
        al = np.array([np.mean(((rr := np.hypot(x + SX*h, y + SY*h)) > RI) & (rr < RO)) for x, y in zip(a[:, 0], a[:, 1])])
        mass = np.sum(a[:, 2]*al)*h*h
        # exact mass: integrate exact_rho over the annulus
        rr = np.linspace(RI, RO, 4001); mex = 2*np.pi*np.trapz(exact_rho(rr)*rr, rr)
        inner = (r - RI > 4*h) & (RO - r > 4*h)
        e = a[inner, 2] - exact_rho(r[inner])
        t = 2*int(f[-6:-4])
        print(f"   {t:4d}   {(mass-mex)/mex:+.3e}        {e.mean():+.3e}                {np.sqrt((e**2).mean()):.3e}")
