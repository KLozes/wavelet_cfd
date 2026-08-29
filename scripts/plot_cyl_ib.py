#!/usr/bin/env python3
"""Inviscid cylinder, immersed slip wall via ghost states + Riemann fluxes.
Compares surface Cp against the potential-flow exact 1 - 4 sin^2(theta)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path

d = np.loadtxt("output/cyl_field.dat")
x, y, rho, u, v, p, mach, cp, fl = (d[:, i] for i in range(9))
R = 0.05          # ibRadius (default), body centred at origin of the dump frame
fluid = fl > 0.5

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
tri = Triangulation(x[fluid], y[fluid])
cxs = x[fluid][tri.triangles].mean(axis=1); cys = y[fluid][tri.triangles].mean(axis=1)
tri.set_mask(np.hypot(cxs, cys) < R)
th = np.linspace(0, 2*np.pi, 200)

for ax, val, lab, cmap in ((axes[0], cp, r"$C_p$", "RdBu_r"),
                           (axes[1], mach, "Mach", "viridis")):
    kw = dict(cmap=cmap)
    if lab == r"$C_p$": kw.update(vmin=-3, vmax=1)
    t = ax.tripcolor(tri, val[fluid], shading="gouraud", **kw)
    ax.fill(R*np.cos(th), R*np.sin(th), color="white", zorder=3)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=1.2, zorder=4)
    ax.set_aspect("equal"); ax.set_title(lab)
    ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.2, 0.2)
    fig.colorbar(t, ax=ax, shrink=0.85)

# surface Cp vs theta from the first fluid ring (d < 1.5 cells of the surface)
h = 1.0/(32*4*2**3)          # nblocks 32, blockSize 4, nLvls 4
r = np.hypot(x[fluid], y[fluid])
ring = (r > R) & (r < R + 1.5*h)
theta = np.degrees(np.arctan2(y[fluid][ring], x[fluid][ring]))
o = np.argsort(theta)
axes[2].plot(theta[o], cp[fluid][ring][o], "o", ms=3, color="#c0392b",
             label="computed (first fluid ring)")
te = np.linspace(-180, 180, 400)
axes[2].plot(te, 1 - 4*np.sin(np.radians(te))**2, "k-", lw=1.2,
             label=r"potential flow $1-4\sin^2\theta$")
axes[2].set_xlabel(r"$\theta$ (deg, 0 = downstream)"); axes[2].set_ylabel(r"$C_p$")
axes[2].legend(frameon=False, fontsize=8); axes[2].grid(alpha=0.25)
axes[2].set_title("surface pressure vs exact")

fig.suptitle("Inviscid cylinder, M=0.2 — immersed slip wall as ghost states + ordinary Riemann fluxes (--ibrecon)", y=1.0)
fig.tight_layout()
fig.savefig("output/cyl_ib.png", dpi=140, bbox_inches="tight")
print("wrote output/cyl_ib.png")
m = np.isfinite(cp[fluid][ring])
ex = 1 - 4*np.sin(np.radians(theta))**2
print("  surface Cp rms error vs potential flow: %.4f" % np.sqrt(np.mean((cp[fluid][ring][m]-ex[m])**2)))
print("  Mach max %.3f (exact 2*Minf = 0.41 at the shoulder)" % mach[fluid].max())
