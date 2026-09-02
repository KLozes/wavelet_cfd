#!/usr/bin/env python3
"""Side-by-side per-cell |dq/dt| maps on a SHARED log scale.

  usage: plot_residual_field_compare.py out.png "title" f1.dat "label 1" f2.dat "label 2" [...]

Each field file is a writeIbField dump made with --residevery N (column 10 =
per-cell residual from the last sample) and --fieldall 1 (all leaves).  A shared
colour normalisation is the whole point -- per-panel autoscaling would make two
very different residual levels look identical.

Colour: magnitude over decades -> perceptually-uniform, lightness-monotonic
sequential ramp (magma).  Cells where the residual is undefined (coarse/fine
GHOST cells, cells inside the body) are written as 0 and dropped, otherwise they
paint black rings along every AMR interface that look like structure.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.colors import LogNorm

out, title, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
pairs = [(rest[i + 1], rest[i]) for i in range(0, len(rest), 2)]
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

data = []
for lab, f in pairs:
    d = np.loadtxt(f)
    if d.shape[1] < 10:
        sys.exit(f"{f}: no residual column -- rerun with --residevery N")
    data.append((lab, d[d[:, 9] > 0]))

allv = np.concatenate([d[:, 9] for _, d in data])
norm = LogNorm(vmin=max(allv.min(), allv.max()*1e-4), vmax=allv.max())

XL, XR, YB, YT = -0.72, 0.85, -0.35, 0.35
fig, ax = plt.subplots(1, len(data), figsize=(7.3*len(data), 4.2), squeeze=False)
for a, (lab, d) in zip(ax[0], data):
    k = (d[:, 0] > XL-.1) & (d[:, 0] < XR+.1) & (d[:, 1] > YB-.1) & (d[:, 1] < YT+.1)
    w = d[k]
    tri = mtri.Triangulation(w[:, 0], w[:, 1])
    cen = np.column_stack([w[:, 0][tri.triangles].mean(1), w[:, 1][tri.triangles].mean(1)])
    tri.set_mask(poly.contains_points(cen))
    f = a.tripcolor(tri, np.clip(w[:, 9], norm.vmin, norm.vmax), norm=norm,
                    cmap="magma", shading="gouraud")
    cb = plt.colorbar(f, ax=a, fraction=0.040, pad=0.02)
    cb.set_label(r"$|\Delta q/\Delta t|$", fontsize=8); cb.ax.tick_params(labelsize=8)
    a.fill(gx, gy, color="0.75", zorder=5)
    a.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "k-", lw=0.9, zorder=6)
    a.set_xlim(XL, XR); a.set_ylim(YB, YT); a.set_aspect("equal")
    a.set_title(f"{lab}   (RMS {np.sqrt((d[:, 9]**2).mean()):.2e}, "
                f"max {d[:, 9].max():.2e})", fontsize=10)
    a.set_xlabel("x/c", fontsize=9); a.set_ylabel("y/c", fontsize=9)
    a.tick_params(labelsize=8)

fig.suptitle(title, fontsize=11.5)
plt.tight_layout(); plt.savefig(out, dpi=150)
print("wrote", out)
for lab, d in data:
    r = d[:, 9]; o = np.argsort(-r)
    n = np.searchsorted(np.cumsum(r[o]**2)/(r**2).sum(), 0.5) + 1
    print(f"  {lab:<26s} RMS {np.sqrt((r**2).mean()):.3e}  max {r.max():.3e}  "
          f"50% of sum(r^2) in top {n} cells ({100*n/len(d):.2f}%)")
