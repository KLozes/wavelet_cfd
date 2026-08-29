#!/usr/bin/env python3
"""Show the cached level set (F_PHI) and the immersed-boundary tag (F_IBM).

Reads output/rae2822_mask.dat (writeIbMask): x y h lvl phi ibmCached fluidAnalytic.
Panel 1: phi on the finest level, with the phi = 0 contour = the true surface.
Panel 2: the F_IBM tag the solver actually keys wall faces on (fluid/solid).
Panel 3: nose zoom of the tag, with the section outline, showing the sub-cell
         standoff d_FC that the level set provides at each wall face.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

msk  = sys.argv[1] if len(sys.argv) > 1 else "output/rae2822_mask.dat"
surf = sys.argv[2] if len(sys.argv) > 2 else "output/rae2822_surface.dat"
out  = sys.argv[3] if len(sys.argv) > 3 else "output/rae2822_mask.png"

d = np.loadtxt(msk)
x, y, h, lvl, phi, ibmC, ibmA = (d[:, i] for i in range(7))
lvl = lvl.astype(int); ibmC = ibmC.astype(int); ibmA = ibmA.astype(int)
lf = lvl.max()
fine = lvl == lf

sd = np.loadtxt(surf)
gx, gy = np.append(sd[:, 3], sd[0, 3]), np.append(sd[:, 4], sd[0, 4])
cx, cy = 0.5*(gx.min()+gx.max()), 0.5*(gy.min()+gy.max())
chord = gx.max() - gx.min()
hf = h[fine].min()

def cells(ax, sel, val, cmap, norm=None, half=0.62, ctr=None, ec="none", lw=0.0):
    px, py = ctr if ctr is not None else (cx, cy)
    m = sel & (np.abs(x-px) < half*chord) & (np.abs(y-py) < half*chord)
    pats = [Rectangle((xx-hh/2, yy-hh/2), hh, hh) for xx, yy, hh in zip(x[m], y[m], h[m])]
    pc = PatchCollection(pats, cmap=cmap, edgecolor=ec, linewidth=lw)
    pc.set_array(val[m])
    if norm: pc.set_clim(*norm)
    ax.add_collection(pc)
    ax.plot(gx, gy, "-", color="crimson", lw=1.4, zorder=5)
    ax.set_xlim(px-half*chord, px+half*chord); ax.set_ylim(py-half*chord, py+half*chord)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return pc

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

# --- phi, finest level; positive INSIDE the body
lim = 0.25*chord
pc = cells(axes[0], fine, phi/chord, "RdBu_r", (-lim/chord, lim/chord))
axes[0].set_title(r"cached level set $\varphi$ / c   (positive INSIDE)"
                  "\nred contour = polyline surface")
fig.colorbar(pc, ax=axes[0], shrink=0.78)

# --- F_IBM tag
cm = ListedColormap(["#8e44ad", "#dfe6e9"])     # 0 = solid/non-fluid, 1 = fluid
pc2 = cells(axes[1], fine, ibmC.astype(float), cm, (0, 1))
axes[1].set_title("F_IBM tag (finest level)\npurple = non-fluid (no solution), grey = fluid")

# --- nose zoom, cells outlined
iLE = int(np.argmin(gx))
pc3 = cells(axes[2], fine, ibmC.astype(float), cm, (0, 1),
            half=0.06, ctr=(gx[iLE], gy[iLE]), ec="0.45", lw=0.4)
axes[2].set_title("leading-edge zoom\ntag is per-cell; the WALL standoff is sub-cell")

nsolid = int((ibmC[fine] == 0).sum())
fig.suptitle("RAE 2822: cached geometry.  finest cell = %.4f c (%.0f/chord);  "
             "%d of %d finest cells tagged non-fluid;  cached vs analytic mismatches: %d"
             % (hf/chord, chord/hf, nsolid, int(fine.sum()), int((ibmC != ibmA).sum())),
             y=0.99)
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
print("  phi range on finest level: [%.4f, %.4f] chords" % (phi[fine].min()/chord, phi[fine].max()/chord))
print("  |phi| < h/2 (cells the surface passes through): %d" % int((np.abs(phi[fine]) < h[fine]/2).sum()))
print("  cached-vs-analytic mismatches: %d" % int((ibmC != ibmA).sum()))
