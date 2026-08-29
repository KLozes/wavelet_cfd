#!/usr/bin/env python3
"""Zoom on the shock region: grid cells coloured by level, with the Cp field's
shock visible via overlaid Mach contour cells and the surface."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

g = np.loadtxt("output/rae2822_grid.dat")
x0, y0, side, lvl, inr = g[:,0], g[:,1], g[:,2], g[:,3].astype(int), g[:,4].astype(int)
keep = inr > 0
x0, y0, side, lvl = x0[keep], y0[keep], side[keep], lvl[keep]

s = np.loadtxt("output/rae2822_surface.dat")
gx, gy = np.append(s[:,3], s[0,3]), np.append(s[:,4], s[0,4])
chord = gx.max() - gx.min()
xLE = gx.min()

f = np.loadtxt("output/rae2822_field.dat")
fx, fy, mach, fl = f[:,0], f[:,1], f[:,6], f[:,8]
cx = 0.5*(gx.min()+gx.max()); cy = 0.5*(gy.min()+gy.max())

# window around the shock: x/c 0.35..0.85, y from surface to +0.45c
wx0, wx1 = xLE+0.35*chord, xLE+0.85*chord
wy0, wy1 = cy-0.06*chord, cy+0.42*chord
BS = 4
nl = lvl.max()+1
cmap = plt.get_cmap("viridis", nl)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.2))

# panel 1: blocks coloured by level
m = (x0+side > wx0) & (x0 < wx1) & (y0+side > wy0) & (y0 < wy1)
pats = [Rectangle((xx,yy),ss,ss) for xx,yy,ss in zip(x0[m],y0[m],side[m])]
pc = PatchCollection(pats, edgecolor="0.25", linewidth=0.35)
pc.set_array(lvl[m]); pc.set_cmap(cmap); pc.set_clim(-0.5, nl-0.5)
a1.add_collection(pc)
a1.plot(gx, gy, "-", color="crimson", lw=2.0, zorder=5)
a1.set_title("blocks by refinement level")
cb = fig.colorbar(pc, ax=a1, shrink=0.8, ticks=range(nl)); cb.set_label("level")

# panel 2: finest-level Mach field (shows the shock) + block edges of fine levels
mm = (fl > 0.5) & (fx*chord+cx > wx0-cx+0*chord) if False else (fl > 0.5)
FX, FY = fx*chord+cx, fy*chord+cy      # field dump is body-centred /chord
sel = mm & (FX > wx0) & (FX < wx1) & (FY > wy0) & (FY < wy1)
sc = a2.scatter(FX[sel], FY[sel], c=mach[sel], s=6, cmap="RdYlBu_r", vmin=0.6, vmax=1.25, marker="s")
m2 = m & (lvl >= nl-2)
for xx, yy, ss in zip(x0[m2], y0[m2], side[m2]):
    a2.add_patch(Rectangle((xx,yy), ss, ss, fill=False, edgecolor="0.3", lw=0.4))
a2.plot(gx, gy, "-", color="crimson", lw=2.0, zorder=5)
a2.set_title("finest-level Mach (shock) + fine-block outlines")
cb2 = fig.colorbar(sc, ax=a2, shrink=0.8); cb2.set_label("Mach")

for a in (a1, a2):
    a.set_xlim(wx0, wx1); a.set_ylim(wy0, wy1)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

fig.suptitle("RAE 2822 transonic, nLvls 7: the grid at the shock (x/c 0.35-0.85, up to 0.42c above)", y=0.98)
fig.savefig("output/shock_grid.png", dpi=140, bbox_inches="tight")
print("wrote output/shock_grid.png")
