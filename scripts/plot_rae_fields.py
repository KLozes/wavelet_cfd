#!/usr/bin/env python3
"""RAE 2822 field comparison: sharp IB vs Brinkman.

Field dump columns: x/c y/c rho u v p mach cp fluid, origin at the body's bbox
centre, and the SECTION carries the angle of attack (Main.cu rotates it by -aoa),
so the overlay has to apply the same rotation before centring.
Points are scattered AMR cells, not a grid -- triangulate, do not reshape.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path

S, AOA = sys.argv[1], 2.31
g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min()+gx.max()); gy -= 0.5*(gy.min()+gy.max())
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

XL, XR, YB, YT = -1.30, 1.45, -1.00, 1.00
def load(p):
    d = np.loadtxt(p)
    k = (d[:,0]>XL-.1)&(d[:,0]<XR+.1)&(d[:,1]>YB-.1)&(d[:,1]<YT+.1)
    d = d[k]
    t = mtri.Triangulation(d[:,0], d[:,1])
    # drop triangles whose centroid is inside the section
    c = np.column_stack([d[:,0][t.triangles].mean(1), d[:,1][t.triangles].mean(1)])
    t.set_mask(poly.contains_points(c))
    return t, d[:,6], d[:,7]              # mach, cp

ts, ms, cs = load(f"{S}/fld_sharp.dat")
tb, mb, cb = load(f"{S}/fld_b0.5.dat")

gxx, gyy = np.meshgrid(np.linspace(XL,XR,520), np.linspace(YB,YT,380))
def interp(t, v):
    z = mtri.LinearTriInterpolator(t, v)(gxx, gyy)
    return np.ma.masked_invalid(z)

fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.0))
rows = [("Mach", ts, ms, tb, mb, np.linspace(0.25, 1.32, 32), "viridis", 0.12),
        (r"$C_p$", ts, cs, tb, cb, np.linspace(-1.25, 0.9, 32), "coolwarm", 0.25)]
for r, (name, t1, v1, t2, v2, lv, cm, dmax) in enumerate(rows):
    for c, (t, v, lab) in enumerate(((t1, v1, "sharp IB"), (t2, v2, r"Brinkman $\delta=0.5h$"))):
        a = ax[r][c]
        f = a.tricontourf(t, v, levels=lv, cmap=cm, extend="both")
        if name == "Mach":
            a.tricontour(t, v, levels=[1.0], colors="w", linewidths=1.6)
        a.set_title(f"{name} — {lab}", fontsize=11)
        plt.colorbar(f, ax=a, fraction=0.046)
    d = interp(t2, v2) - interp(t1, v1)
    a = ax[r][2]
    f = a.pcolormesh(gxx, gyy, d, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="auto")
    a.set_title(f"{name} — Brinkman minus sharp", fontsize=11)
    plt.colorbar(f, ax=a, fraction=0.046)
    for c in range(3):
        ax[r][c].fill(gx, gy, color="0.25", zorder=5)
        ax[r][c].plot(np.append(gx,gx[0]), np.append(gy,gy[0]), "k-", lw=.8, zorder=6)
        ax[r][c].set_xlim(XL, XR); ax[r][c].set_ylim(YB, YT); ax[r][c].set_aspect("equal")
fig.suptitle(r"RAE 2822, M=0.73, $\alpha$=2.31$^\circ$, nLvls 7 "
             "(white line = sonic)", fontsize=12)
plt.tight_layout(); plt.savefig("output/rae_fields.png", dpi=135)
print("-> output/rae_fields.png")
for nm, a, b in (("Mach", ms, mb), ("Cp", cs, cb)):
    print(f"  {nm}: sharp max {a.max():.4f}   brink max {b.max():.4f}")
