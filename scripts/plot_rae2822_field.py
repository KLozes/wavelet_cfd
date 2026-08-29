#!/usr/bin/env python3
"""Plot the RAE 2822 flow field and surface pressure (--case 15).

Reads output/rae2822_field.dat and output/rae2822_surface.dat.  Solid cells
(fluid = 0) are masked so the body is a hole rather than contoured ghost data.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

fld  = sys.argv[1] if len(sys.argv) > 1 else "output/rae2822_field.dat"
surf = sys.argv[2] if len(sys.argv) > 2 else "output/rae2822_surface.dat"
geom = sys.argv[3] if len(sys.argv) > 3 else "geom/rae2822.dat"
out  = sys.argv[4] if len(sys.argv) > 4 else "output/rae2822_field.png"

d = np.loadtxt(fld)
x, y, rho, u, v, p, mach, cp, fl = (d[:, i] for i in range(9))
g = np.loadtxt(geom)
# geometry in the same body-centred, chord-normalised frame as the field
gx = (g[:, 0] - 0.5*(g[:, 0].min() + g[:, 0].max()))
gy = (g[:, 1] - 0.5*(g[:, 1].min() + g[:, 1].max()))
gx, gy = np.append(gx, gx[0]), np.append(gy, gy[0])

fluid = fl > 0.5
tri = Triangulation(x[fluid], y[fluid])
# drop triangles whose centroid falls inside the body (the hole)
cxs = x[fluid][tri.triangles].mean(axis=1)
cys = y[fluid][tri.triangles].mean(axis=1)
from matplotlib.path import Path
inside = Path(np.column_stack([gx, gy])).contains_points(np.column_stack([cxs, cys]))
big = (np.hypot(x[fluid][tri.triangles][:, 0] - x[fluid][tri.triangles][:, 1],
                y[fluid][tri.triangles][:, 0] - y[fluid][tri.triangles][:, 1]) > 0.12)
tri.set_mask(inside | big)

fig = plt.figure(figsize=(11, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

for ax, val, label, cmap in ((fig.add_subplot(gs[0, 0]), cp, r"$C_p$", "RdBu_r"),
                             (fig.add_subplot(gs[0, 1]), mach, "Mach", "viridis")):
    lim = np.nanpercentile(np.abs(val[fluid]), 99)
    kw = dict(cmap=cmap)
    if label == r"$C_p$":
        kw.update(vmin=-lim, vmax=lim)
    tc = ax.tripcolor(tri, val[fluid], shading="gouraud", **kw)
    ax.plot(gx, gy, "-", color="k", lw=1.1)
    ax.fill(gx, gy, color="white", zorder=3)
    ax.plot(gx, gy, "-", color="k", lw=1.1, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.2); ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(label)
    fig.colorbar(tc, ax=ax, shrink=0.82)

ax = fig.add_subplot(gs[1, :])
s = np.loadtxt(surf)
xc, yn, scp = s[:, 0], s[:, 1], s[:, 2]
ok = np.isfinite(scp)
side = s[:, 5] if s.shape[1] > 5 else np.where(yn >= 0, 1, -1)
up, lo = ok & (side > 0), ok & (side < 0)
o = np.argsort(xc[up]); ax.plot(xc[up][o], scp[up][o], "-o", ms=3, lw=1.0,
                                color="#c0392b", label="upper")
o = np.argsort(xc[lo]); ax.plot(xc[lo][o], scp[lo][o], "-o", ms=3, lw=1.0,
                                color="#2471a3", label="lower")
ax.axhline(0.0, lw=0.6, color="0.7")
ax.invert_yaxis(); ax.set_xlabel("x/c"); ax.set_ylabel(r"$C_p$")
ax.legend(frameon=False); ax.grid(alpha=0.25)
ax.set_title("surface pressure, sampled along the level-set normal at 0.5h")


import os
TITLE = os.environ.get("RAE_TITLE",
    r"RAE 2822, immersed level-set body — EULER (no viscosity, no wall model)")
fig.suptitle(TITLE, y=0.995)
fig.tight_layout()
fig.savefig(out, dpi=135)
print("wrote", out)
print("  Cp range on the surface: [%+.3f, %+.3f]" % (np.nanmin(scp[ok]), np.nanmax(scp[ok])))
print("  Mach range in the field: [%.3f, %.3f]" % (mach[fluid].min(), mach[fluid].max()))
