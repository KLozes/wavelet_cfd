#!/usr/bin/env python3
# 3-D reconstructed boundary of the WING (assets/wing.stl) from the Saye surface
# rule on a per-cell degree-p fit of the oracle SDF.  Colored by |phi| = distance
# from the TRUE surface: smooth surfaces stay dark, the sharp TRAILING EDGE (a
# crease) lights up -- and brighter at higher p (the polynomial oscillates at the
# kink).  Left = deg-1 (planar, robust), right = deg-2 (parabolic).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ps = [1, 2]
data = {p: np.loadtxt(f"output/wing_recon3d_p{p}.txt") for p in ps}
vmax = np.percentile(np.concatenate([d[:,3] for d in data.values()]), 90)

# ---- Fig 1: a 3-D spanwise section (y in [440,560]) ----
fig = plt.figure(figsize=(15, 6.6))
for col, p in enumerate(ps):
    d = data[p]; sel = (d[:,1] > 440) & (d[:,1] < 560)
    P, dev = d[sel, :3], d[sel, 3]
    ax = fig.add_subplot(1, 2, col+1, projection="3d")
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=dev, cmap="inferno", vmin=0, vmax=vmax,
                    s=5, alpha=0.85, linewidths=0)
    r = P.max(0)-P.min(0); ax.set_box_aspect(r/r.max())
    ax.set_xlim(P[:,0].min(),P[:,0].max()); ax.set_ylim(P[:,1].min(),P[:,1].max()); ax.set_zlim(P[:,2].min(),P[:,2].max())
    ax.view_init(elev=24, azim=-74)
    ax.set_title(f"deg-{p} reconstruction  (spanwise section)\nwhole-wing dev p95 = {np.percentile(d[:,3],95):.2f}", fontsize=11)
    ax.set_xlabel("x (chord)"); ax.set_ylabel("y (span)"); ax.set_zlabel("z")
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
cb.set_label("|φ| = distance from the TRUE wing surface")
fig.suptitle("Wing 3-D reconstructed boundary — the sharp trailing edge lights up "
             "(and brighter at deg-2: the parabola oscillates at the kink)", fontsize=12)
fig.savefig("/tmp/wing3d.png", dpi=118, bbox_inches="tight")
print("wrote /tmp/wing3d.png")

# ---- Fig 2: a mid-span airfoil SLICE (x-z), 2-D, zoomed on the trailing edge ----
fig2, axes = plt.subplots(1, 2, figsize=(15, 5.2))
for col, p in enumerate(ps):
    d = data[p]; sel = (d[:,1] > 497) & (d[:,1] < 503)
    x, z, dev = d[sel,0], d[sel,2], d[sel,3]
    ax = axes[col]
    sc = ax.scatter(x, z, c=dev, cmap="inferno", vmin=0, vmax=vmax, s=14, linewidths=0)
    ax.set_aspect("equal"); ax.set_title(f"deg-{p}: mid-span airfoil slice", fontsize=12)
    ax.set_xlabel("x (chord)"); ax.set_ylabel("z (thickness)")
    ax.grid(alpha=0.25)
fig2.colorbar(sc, ax=list(axes), shrink=0.8, pad=0.02, label="|φ| dist from true surface")
fig2.suptitle("Mid-span airfoil section of the reconstructed boundary — trailing edge (right) is the crease; "
              "deg-2 lights up more there", fontsize=12)
fig2.savefig("/tmp/wing_section.png", dpi=120, bbox_inches="tight")
print("wrote /tmp/wing_section.png")
