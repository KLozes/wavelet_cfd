#!/usr/bin/env python3
"""Where the steady residual actually lives — spatial map of per-cell |dq/dt|.

  usage: plot_residual_field.py field.dat out.png "title"

Reads a wave3d writeIbField dump produced with --residevery (column 10 = the
per-cell residual from the last sample) and --fieldall 1 (all leaves, not just
the finest band).  Companion to plot_residual.py, which shows the same quantity
reduced to one number per iteration.

Colour: magnitude over many decades -> perceptually-uniform, lightness-monotonic
sequential ramp on a log scale (magma; dark = converged, bright = active).  Not
a rainbow, and no diverging pair -- there is no meaningful midpoint in |dq/dt|.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.colors import LogNorm

src, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "residual field"
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

d = np.loadtxt(src)
if d.shape[1] < 10:
    sys.exit("no residual column -- rerun the solver with --residevery N")
# Keep only cells the residual is DEFINED on.  Coarse/fine GHOST cells and cells
# inside the body are written as exactly 0; plotting them paints black rings
# along every AMR interface that look like structure but are just "not a DOF".
d = d[d[:, 9] > 0]

fig, ax = plt.subplots(1, 2, figsize=(14.2, 4.9))
windows = [(-1.00, 1.40, -0.80, 0.80, "full field"),
           (-0.60, 0.70, -0.28, 0.28, "zoom on the immersed boundary")]

pos = d[:, 9][d[:, 9] > 0]
vmin = max(pos.min(), pos.max()*1e-5) if len(pos) else 1e-12
vmax = pos.max() if len(pos) else 1.0
norm = LogNorm(vmin=vmin, vmax=vmax)

for a, (XL, XR, YB, YT, tag) in zip(ax, windows):
    k = (d[:, 0] > XL-.1) & (d[:, 0] < XR+.1) & (d[:, 1] > YB-.1) & (d[:, 1] < YT+.1)
    w = d[k]
    tri = mtri.Triangulation(w[:, 0], w[:, 1])
    cen = np.column_stack([w[:, 0][tri.triangles].mean(1),
                           w[:, 1][tri.triangles].mean(1)])
    tri.set_mask(poly.contains_points(cen))
    v = np.clip(w[:, 9], vmin, vmax)
    f = a.tripcolor(tri, v, norm=norm, cmap="magma", shading="gouraud")
    cb = plt.colorbar(f, ax=a, fraction=0.046, pad=0.02)
    cb.set_label(r"per-cell $|\Delta q/\Delta t|$", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    a.fill(gx, gy, color="0.75", zorder=5)
    a.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "k-", lw=0.9, zorder=6)
    a.set_xlim(XL, XR); a.set_ylim(YB, YT); a.set_aspect("equal")
    a.set_title(tag, fontsize=10)
    a.set_xlabel("x/c", fontsize=9); a.set_ylabel("y/c", fontsize=9)
    a.tick_params(labelsize=8)

fig.suptitle(title, fontsize=11.5)
plt.tight_layout()
plt.savefig(out, dpi=140)
print("wrote", out)

# how much of the total residual sits within N cells of the wall?
r = d[:, 9]
tot = (r**2).sum()
print(f"  cells {len(d)}   max |dq/dt| = {r.max():.3e}   RMS = {np.sqrt((r**2).mean()):.3e}")
order = np.argsort(-r)
for frac in (0.5, 0.9):
    n = np.searchsorted(np.cumsum(r[order]**2)/tot, frac) + 1
    print(f"  {frac*100:.0f}% of sum(r^2) comes from the top {n} cells "
          f"({100*n/len(d):.2f}% of the field)")
