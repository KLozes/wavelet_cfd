#!/usr/bin/env python3
"""RAE 2822 case 9: Mach field, inviscid vs wall-modelled RANS, side by side.

   usage: plot_rae_field_compare.py out.png field1.dat label1 field2.dat label2 ...
Solid cells (fluid = 0) are masked so the body is a hole, not contoured ghosts.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

out = sys.argv[1]
pairs = [(sys.argv[i], sys.argv[i+1]) for i in range(2, len(sys.argv), 2)]
n = len(pairs)
fig, axes = plt.subplots(1, n, figsize=(6.4*n, 5.0), squeeze=False)
lv = np.linspace(0.30, 1.25, 39)

for ax, (f, lab) in zip(axes[0], pairs):
    d = np.loadtxt(f)
    x, y, mach, fl = d[:, 0], d[:, 1], d[:, 6], d[:, 8]
    m = (fl > 0.5) & (np.abs(x) < 1.1) & (np.abs(y) < 0.6)
    xs, ys, ms = x[m], y[m], mach[m]
    tri = Triangulation(xs, ys)
    t = tri.triangles
    # mask triangles that bridge the body (long edges) -- the body is a hole
    e = np.hypot(xs[t][:, [0, 1, 2]] - xs[t][:, [1, 2, 0]],
                 ys[t][:, [0, 1, 2]] - ys[t][:, [1, 2, 0]]).max(axis=1)
    tri.set_mask(e > 0.045)
    cf = ax.tricontourf(tri, ms, levels=lv, cmap="turbo", extend="both")
    ax.tricontour(tri, ms, levels=[1.0], colors="w", linewidths=1.8)
    ax.set_aspect("equal"); ax.set_xlim(-0.75, 0.85); ax.set_ylim(-0.35, 0.35)
    ax.set_title(f"{lab}\nwhite line = sonic (M=1)", fontsize=10)
    ax.set_xlabel("x/c (origin at body centre)")
    fig.colorbar(cf, ax=ax, shrink=0.82, label="Mach")
axes[0][0].set_ylabel("y/c")
fig.suptitle("RAE 2822 case 9: M=0.729, $\\alpha$=2.31$^\\circ$, Re=6.5e6 — nLvls 7", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
