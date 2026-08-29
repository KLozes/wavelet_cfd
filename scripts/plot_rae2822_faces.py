#!/usr/bin/env python3
"""Show the immersed wall-model construction at every modelled face.

Reads output/rae2822_faces.dat (writeIbWallFaces):
  xf yf dir h lvl dFcOverH xs ys nx ny xip yip modelled
Draws the grid-aligned wall FACE, the foot point on the true surface, the
normal line, and the image point where the wall function is evaluated.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

fac  = sys.argv[1] if len(sys.argv) > 1 else "output/rae2822_faces.dat"
surf = sys.argv[2] if len(sys.argv) > 2 else "output/rae2822_surface.dat"
out  = sys.argv[3] if len(sys.argv) > 3 else "output/rae2822_faces.png"

d = np.loadtxt(fac)
xf, yf, dr, h, lvl, dfc, xs, ys, nx, ny, xi, yi, mod = (d[:, i] for i in range(13))
dr = dr.astype(int)

sd = np.loadtxt(surf)
gx, gy = np.append(sd[:, 3], sd[0, 3]), np.append(sd[:, 4], sd[0, 4])
cx, cy = 0.5*(gx.min()+gx.max()), 0.5*(gy.min()+gy.max())
chord = gx.max() - gx.min()

# the face is a grid-aligned segment of length h, perpendicular to its direction
fseg = []
for a, b, dd, hh in zip(xf, yf, dr, h):
    if dd == 1:  fseg.append([(a-hh/2, b), (a+hh/2, b)])     # y-face: spans x
    else:        fseg.append([(a, b-hh/2), (a, b+hh/2)])     # x-face: spans y
fseg = np.array(fseg)
nseg = np.stack([np.column_stack([xs, ys]), np.column_stack([xi, yi])], axis=1)

LEAF = lvl.max()   # only LEAF faces act: parent cells are restricted from their
                   # children every stage, so their wall fluxes are discarded.
def panel(ax, half, ctr, marks, title, leafOnly=True):
    px, py = ctr
    m = (np.abs(xf-px) < half) & (np.abs(yf-py) < half)
    if leafOnly: m &= (lvl == LEAF)
    ax.plot(gx, gy, "-", color="crimson", lw=1.8, zorder=6, label="true surface")
    lc = LineCollection(fseg[m], cmap="plasma", linewidths=3.0, zorder=4)
    lc.set_array(dfc[m]); lc.set_clim(0.0, 1.1)
    ax.add_collection(lc)
    ax.add_collection(LineCollection(nseg[m], colors="0.35", linewidths=0.7,
                                     zorder=5, label="normal"))
    if marks:
        ax.plot(xs[m], ys[m], "o", ms=marks, color="#117a65", zorder=7,
                label="foot point on surface")
        ax.plot(xi[m], yi[m], "s", ms=marks, color="#1f4e79", zorder=7,
                label=r"image point ($d_{IP}=3h$)")
    ax.set_xlim(px-half, px+half); ax.set_ylim(py-half, py+half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return lc

fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
lc = panel(axes[0], 0.60*chord, (cx, cy), 2.0,
           "%d LEAF wall faces (of %d over all levels)\n"
           "face colour = $d_{FC}/h$ (sub-cell standoff)"
           % ((lvl == lvl.max()).sum(), len(xf)))
iLE = int(np.argmin(gx))
panel(axes[1], 0.055*chord, (gx[iLE], gy[iLE]), 6.0,
      "leading-edge zoom: face, foot point, normal, image point")
axes[1].legend(loc="lower right", fontsize=8, framealpha=0.92)
cb = fig.colorbar(lc, ax=axes, shrink=0.8, pad=0.015)
cb.set_label(r"$d_{FC}/h$   (0.1 = floor, ~1 = a full cell)")

fig.suptitle("RAE 2822 immersed wall model: the FACE is grid-aligned, but the wall "
             "standoff and normal are taken from the level set", y=0.98)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
print("  faces: %d  (%d y-faces, %d x-faces)" % (len(xf), (dr==1).sum(), (dr==0).sum()))
print("  d_FC/h: min %.3f  median %.3f  max %.3f   (%d at the 0.1 floor)"
      % (dfc.min(), np.median(dfc), dfc.max(), (dfc <= 0.1001).sum()))
print("  image-point standoff above the face: min %.3f h"
      % (np.hypot(xi-xf, yi-yf)/h).min())
