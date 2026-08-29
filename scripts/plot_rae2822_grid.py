#!/usr/bin/env python3
"""Plot the AMR block structure around the RAE 2822 (--case 15).

Reads output/rae2822_grid.dat (writeGridBlocks) and geom/rae2822.dat.
Three panels: the whole 24-chord domain, a body-scale view, and a nose zoom
with individual CELLS drawn so the resolution at the leading edge is visible.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

grid = sys.argv[1] if len(sys.argv) > 1 else "output/rae2822_grid.dat"
# NOTE: geom/rae2822.dat is in UNIT-CHORD coordinates; the blocks are in DOMAIN
# coordinates.  The surface dump carries the section in domain coordinates
# (columns 4,5), already scaled, rotated by -aoa and translated -- use it.
surf = sys.argv[2] if len(sys.argv) > 2 else "output/rae2822_surface.dat"
out  = sys.argv[3] if len(sys.argv) > 3 else "output/rae2822_grid.png"
BS = 4   # blockSize

d = np.loadtxt(grid)
x0, y0, side, lvl, inr = d[:,0], d[:,1], d[:,2], d[:,3].astype(int), d[:,4].astype(int)
keep = inr > 0
x0, y0, side, lvl = x0[keep], y0[keep], side[keep], lvl[keep]

sd = np.loadtxt(surf)
gx, gy = np.append(sd[:,3], sd[0,3]), np.append(sd[:,4], sd[0,4])   # domain coords
cx, cy = 0.5*(gx.min()+gx.max()), 0.5*(gy.min()+gy.max())
chord = gx.max() - gx.min()

nl = lvl.max() + 1
cmap = plt.get_cmap("viridis", nl)

def panel(ax, half, cells=False, title="", ctr=None):
    px, py = ctr if ctr is not None else (cx, cy)
    sel = (x0 + side > px-half) & (x0 < px+half) & (y0 + side > py-half) & (y0 < py+half)
    pats, cols = [], []
    for xx, yy, ss, ll in zip(x0[sel], y0[sel], side[sel], lvl[sel]):
        if cells:                       # draw individual cells
            h = ss/BS
            for a in range(BS):
                for b in range(BS):
                    pats.append(Rectangle((xx+a*h, yy+b*h), h, h)); cols.append(ll)
        else:
            pats.append(Rectangle((xx, yy), ss, ss)); cols.append(ll)
    pc = PatchCollection(pats, edgecolor="0.35", linewidth=0.25, alpha=0.95)
    pc.set_array(np.array(cols)); pc.set_cmap(cmap); pc.set_clim(-0.5, nl-0.5)
    ax.add_collection(pc)
    ax.plot(gx, gy, "-", color="crimson", lw=1.6, zorder=5)
    ax.set_xlim(px-half, px+half); ax.set_ylim(py-half, py+half)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return pc

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
panel(axes[0], 12.0*chord, False, "full domain (24 chords)\nblocks, coloured by level")
panel(axes[1], 0.9*chord,  False, "body scale\nblocks")
iLE = int(np.argmin(gx))                     # leading edge in domain coords
pc = panel(axes[2], 0.06*chord, True,
           "leading-edge zoom\nindividual CELLS (LE radius $\\approx$ 1 cell)",
           ctr=(gx[iLE], gy[iLE]))
cb = fig.colorbar(pc, ax=axes, shrink=0.8, ticks=range(nl), pad=0.015)
cb.set_label("refinement level")
cnt = " ".join("L%d=%d" % (l, (lvl == l).sum()) for l in range(nl))
import os
fig.suptitle(os.environ.get("RAE_GRID_TITLE",
             "RAE 2822: wavelet AMR grid, nLvls = %d  (%d blocks: %s)"
             % (nl, len(lvl), cnt)), y=0.99)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
hf = side.min()/BS
print("  finest cell = %.5f = %.5f chord   ->  %.0f cells per chord" % (hf, hf/chord, chord/hf))
print("  RAE 2822 LE radius = 0.0082c = %.2f finest cells" % (0.0082*chord/hf))
