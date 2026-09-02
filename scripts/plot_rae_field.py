#!/usr/bin/env python3
"""RAE 2822 flow field from a wave3d writeIbField dump.

  usage: plot_rae_field.py field.dat out.png "title"

Dump columns: x/c y/c rho u v p mach cp fluid, origin at the body's bbox centre.
The SECTION carries the angle of attack (Main.cu rotates it by -aoa), so the
geometry overlay gets the same rotation before centring.  Points are scattered
AMR cell centres, NOT a grid -- triangulate, never reshape.

Colour: Mach is a magnitude -> perceptually-uniform sequential ramp (viridis;
CVD-safe, not a rainbow).  Cp is a polarity about 0 -> diverging pair with a
neutral midpoint (RdBu_r), midpoint pinned to Cp = 0 so the neutral band means
"freestream pressure" rather than "middle of the data".
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm

src, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "RAE 2822"
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

XL, XR, YB, YT = -1.00, 1.40, -0.80, 0.80   # body spans x/c -0.5..0.5 (centred); room for the wake
d = np.loadtxt(src)
k = (d[:, 0] > XL-.1) & (d[:, 0] < XR+.1) & (d[:, 1] > YB-.1) & (d[:, 1] < YT+.1)
d = d[k]
tri = mtri.Triangulation(d[:, 0], d[:, 1])
cen = np.column_stack([d[:, 0][tri.triangles].mean(1), d[:, 1][tri.triangles].mean(1)])
tri.set_mask(poly.contains_points(cen))       # drop triangles inside the section
mach, cp = d[:, 6], d[:, 7]

fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.0))
panels = [
    ("Mach number", mach, np.linspace(0.0, 1.35, 40), "viridis", None),
    (r"$C_p$",      cp,   np.linspace(-1.30, 1.10, 40), "RdBu_r",
     TwoSlopeNorm(vmin=-1.30, vcenter=0.0, vmax=1.10)),
]
for a, (name, v, lv, cm, nrm) in zip(ax, panels):
    f = a.tricontourf(tri, v, levels=lv, cmap=cm, norm=nrm, extend="both")
    if nrm is None:                            # sonic line only on the Mach panel
        a.tricontour(tri, v, levels=[1.0], colors="w", linewidths=1.7)
        a.plot([], [], "w-", lw=1.7, label="sonic line, $M=1$")
        a.legend(loc="lower right", fontsize=8, framealpha=0.9)
    cb = plt.colorbar(f, ax=a, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    a.set_title(name, fontsize=11)
    a.fill(gx, gy, color="0.22", zorder=5)
    a.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "k-", lw=0.9, zorder=6)
    a.set_xlim(XL, XR); a.set_ylim(YB, YT); a.set_aspect("equal")
    a.set_xlabel("x/c", fontsize=9); a.set_ylabel("y/c", fontsize=9)
    a.tick_params(labelsize=8)

fig.suptitle(title, fontsize=12)
plt.tight_layout()
plt.savefig(out, dpi=140)
print("wrote", out)
print(f"  cells plotted {len(d)}   Mach max {mach.max():.3f}   "
      f"Cp range [{cp.min():+.3f}, {cp.max():+.3f}]")
sup = mach > 1.0
print(f"  supersonic cells {sup.sum()} ({100*sup.mean():.1f}% of the window)")
