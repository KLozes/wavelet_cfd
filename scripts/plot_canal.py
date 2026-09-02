#!/usr/bin/env python3
"""Canal-with-bump (case 18) field picture + floor Cp.

  usage: plot_canal.py field.dat cp.dat y0 out.png "title"

field.dat is a writeIbField dump (x/c, y/c relative to the bump-disc centre
(1.5, y0 - 1.2)); cp.dat is output/canal_cp.dat.  Top: Mach number with the
sonic line and the immersed geometry; middle: pressure; bottom: floor Cp in
the paper's frame (bump on [0,1]) plus mdot(x).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

fld, cpf, y0, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
title = sys.argv[5] if len(sys.argv) > 5 else "canal with 10% bump"
cx, cy = 1.5, y0 - 1.2
d = np.loadtxt(fld)
x, y = d[:, 0] + cx, d[:, 1] + cy
live = d[:, 8] > 0.5
d, x, y = d[live], x[live], y[live]

# geometry: floor with circular-arc bump, ceiling
xb = np.linspace(1, 2, 200); yb = cy + np.sqrt(1.3**2 - (xb - cx)**2)
fx = np.concatenate([[0], xb, [3]]); fy = np.concatenate([[y0], yb, [y0]])

fig, ax = plt.subplots(3, 1, figsize=(11, 9.6), gridspec_kw={"height_ratios": [1.6, 1.6, 1.3]})
tri = mtri.Triangulation(x, y)
cen = np.column_stack([x[tri.triangles].mean(1), y[tri.triangles].mean(1)])
# drop triangles that span the body (their centroid is below the floor curve) or the ceiling
mask = (cen[:, 1] < np.interp(cen[:, 0], fx, fy)) | (cen[:, 1] > y0 + 1)
tri.set_mask(mask)
for a, col, lab, cmap in ((ax[0], 6, "Mach", "viridis"), (ax[1], 5, "p / p0", "cividis")):
    f = a.tripcolor(tri, d[:, col], cmap=cmap, shading="gouraud")
    if col == 6:
        try: a.tricontour(tri, d[:, col], levels=[1.0], colors="w", linewidths=1.0)
        except Exception: pass
    plt.colorbar(f, ax=a, fraction=0.025, pad=0.01).set_label(lab, fontsize=9)
    a.fill_between(fx, -0.1, fy, color="0.75", zorder=5)
    a.fill_between([0, 3], [y0+1, y0+1], [y0+1.2, y0+1.2], color="0.75", zorder=5)
    a.plot(fx, fy, "k-", lw=0.8, zorder=6); a.plot([0, 3], [y0+1, y0+1], "k-", lw=0.8, zorder=6)
    a.set_xlim(0, 3); a.set_ylim(0, y0 + 1.05); a.set_aspect("equal")
    a.set_ylabel("y", fontsize=9); a.tick_params(labelsize=8)
ax[0].set_title(title, fontsize=11)

c = np.loadtxt(cpf)
a = ax[2]
a.plot(c[:, 0] - 1.0, c[:, 1], "g.-", ms=3, lw=0.8, label="floor Cp (lowest live cell)")
a.set_ylim(-2.0, 1.2); a.invert_yaxis()
a.set_xlabel("x (paper frame: bump on [0, 1])", fontsize=9); a.set_ylabel(r"$C_p$", fontsize=9)
a.axvspan(0, 1, color="0.92", zorder=0)
a2 = a.twinx()
a2.plot(c[:, 0] - 1.0, c[:, 2], "r-", lw=0.8, alpha=0.7, label=r"$\dot m(x)$")
a2.set_ylabel(r"$\dot m(x)$", fontsize=9, color="r"); a2.tick_params(axis="y", colors="r", labelsize=8)
a.legend(loc="lower left", fontsize=8); a2.legend(loc="lower right", fontsize=8)
a.set_xlim(-1, 2); a.tick_params(labelsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=140)
print("wrote", out)
m = d[:, 6]
print(f"  Mach: min {m.min():.3f} max {m.max():.3f}   supersonic cells {np.sum(m>1)}")
print(f"  mdot(x): min {c[:,2].min():.5f} max {c[:,2].max():.5f}")
