#!/usr/bin/env python3
"""List the cells carrying the largest |dq/dt| on the annulus, with their cut
geometry: signed wall distance in cells, azimuth, and the fluid fraction alpha
of the cell (recomputed analytically by sub-sampling the square).  Tells
whether the residual max sits on sliver cells, on R-Cells, on a particular
azimuth, or is spread.   usage: svortex_top_resid.py field.dat [N]"""
import sys, numpy as np
d = np.loadtxt(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RI, RO = 1.0, 1.384
d = d[d[:, 9] > 0]
xs = np.unique(np.round(d[:, 0], 8)); h = np.min(np.diff(xs))
rr = np.hypot(d[:, 0], d[:, 1]); dw = np.minimum(rr - RI, RO - rr)/h
th = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
o = np.argsort(-d[:, 9])
s = np.linspace(-0.5, 0.5, 9)[:-1] + 1/16; SX, SY = np.meshgrid(s, s)
def alpha(x, y):
    r = np.hypot(x + SX*h, y + SY*h); return np.mean((r > RI) & (r < RO))
print(f"h = {h:.5f}   cells {len(d)}   RMS {np.sqrt((d[:,9]**2).mean()):.3e}")
print(f"{'rank':>4} {'|dq/dt|':>10} {'x':>8} {'y':>8} {'theta':>7} {'dwall/h':>8} {'alpha':>6}  {'rho':>8} {'p':>8}")
for k, i in enumerate(o[:N]):
    print(f"{k+1:4d} {d[i,9]:10.3e} {d[i,0]:8.4f} {d[i,1]:8.4f} {th[i]:7.1f} {dw[i]:8.2f} "
          f"{alpha(d[i,0], d[i,1]):6.2f}  {d[i,2]:8.5f} {d[i,5]:8.5f}")
# how concentrated, and where, is the top 1% of cells?
n1 = max(1, len(d)//100); top = o[:n1]
print(f"\ntop 1% ({n1} cells) carry {100*(d[top,9]**2).sum()/(d[:,9]**2).sum():.0f}% of sum(r^2);"
      f"  {100*np.mean(dw[top] < 0):.0f}% R-Cells, {100*np.mean((dw[top] >= 0) & (dw[top] < 1.5)):.0f}% first fluid row,"
      f" {100*np.mean(dw[top] >= 1.5):.0f}% further out")
a_top = np.array([alpha(d[i,0], d[i,1]) for i in top])
print(f"  alpha of the top 1%: median {np.median(a_top):.2f}, 10th pct {np.percentile(a_top,10):.2f}, share with alpha<0.25: {100*np.mean(a_top<0.25):.0f}%")
