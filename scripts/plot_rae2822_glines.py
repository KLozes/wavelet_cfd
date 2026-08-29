#!/usr/bin/env python3
"""Ghost-fill image lines: ghost centre -> foot point -> sample point at s*=2h.

Reads output/rae2822_glines.dat (writeIbGhostLines) and the surface dump for
the outline.  Interior ghosts (phi>0) vs intersecting cells (phi<=0, centre in
the fluid) are distinguished -- the two populations the fill treats by one rule."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

g = np.loadtxt("output/rae2822_glines.dat")
xg, yg, phi, h, lvl, xs, ys, nx, ny, xi, yi, dS, itx = (g[:, i] for i in range(13))
itx = itx.astype(bool)
s = np.loadtxt("output/rae2822_surface.dat")
gx, gy = np.append(s[:, 3], s[0, 3]), np.append(s[:, 4], s[0, 4])
cx, cy = 0.5*(gx.min()+gx.max()), 0.5*(gy.min()+gy.max())
chord = gx.max() - gx.min()

lines = np.stack([np.column_stack([xg, yg]), np.column_stack([xs, ys]),
                  np.column_stack([xi, yi])], axis=1)

# grid cells from the mask dump (finest level), coloured fluid / ghost-band / deep
msk = np.loadtxt("output/rae2822_mask.dat")
mx, my, mh, mlvl, mphi, mibm = msk[:,0], msk[:,1], msk[:,2], msk[:,3].astype(int), msk[:,4], msk[:,5].astype(int)
mfin = mlvl == mlvl.max()

def panel(ax, half, ctr, title, lw=0.6, ms=2.0):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    px, py = ctr
    mm = mfin & (np.abs(mx-px) < half+0.01) & (np.abs(my-py) < half+0.01)
    pats, cols = [], []
    for xx, yy, hh, fb, ph in zip(mx[mm], my[mm], mh[mm], mibm[mm], mphi[mm]):
        pats.append(Rectangle((xx-hh/2, yy-hh/2), hh, hh))
        # fluid: white; filled ghost band (phi <= 2.5h): pale purple; deep body: darker
        cols.append("#ffffff" if fb else ("#e8dff5" if ph <= 2.5*hh else "#d3cade"))
    pc = PatchCollection(pats, facecolor=cols, edgecolor="0.8", linewidth=0.3, zorder=1)
    ax.add_collection(pc)
    m = (np.abs(xg-px) < half) & (np.abs(yg-py) < half)
    ax.plot(gx, gy, "-", color="crimson", lw=1.6, zorder=6, label="surface")
    for sel, col, lab in ((m & ~itx, "0.45", "interior ghost (phi>0)"),
                          (m & itx, "#8e44ad", "intersecting (centre in fluid)")):
        ax.add_collection(LineCollection(lines[sel], colors=col, linewidths=lw, zorder=4))
        ax.plot(xg[sel], yg[sel], "s", ms=ms+0.8, color=col, zorder=5, label=lab)
    ax.plot(xs[m], ys[m], "o", ms=ms, color="#117a65", zorder=7, label="foot point")
    ax.plot(xi[m], yi[m], "^", ms=ms+0.5, color="#1f4e79", zorder=7, label="sample at s*=2h")
    ax.set_xlim(px-half, px+half); ax.set_ylim(py-half, py+half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
panel(axes[0], 0.58*chord, (cx, cy), "all %d filled ghosts (finest level)" % len(xg))
iLE = int(np.argmin(gx))
panel(axes[1], 0.05*chord, (gx[iLE], gy[iLE]),
      "leading edge: normals fan around the 1-cell nose", lw=0.9, ms=3.5)
iTE = int(np.argmax(gx))
panel(axes[2], 0.06*chord, (gx[iTE]-0.04*chord, gy[iTE]),
      "aft/TE: sub-cell thin -- opposite-side lines meet", lw=0.9, ms=3.5)
axes[2].legend(loc="lower left", fontsize=7.5, framealpha=0.95)
fig.suptitle("Ghost-fill image lines: cell centre ■ -> foot point ● -> sample ▲ at a FIXED 2h standoff "
             "(%d ghosts, %d intersecting)" % (len(xg), itx.sum()), y=0.99)
fig.savefig("output/rae2822_glines.png", dpi=140, bbox_inches="tight")
print("wrote output/rae2822_glines.png")
hf = h.min()
print("  standoff check: sample wall-distance dS: min %.2fh  median %.2fh  max %.2fh"
      % (dS.min()/hf, np.median(dS)/hf, dS.max()/hf))
