#!/usr/bin/env python3
# 3-D reconstructed boundary of the ROTOR 1 blade (bank geometry: blade + platform
# + fillet + tip gap, cylindrical), from the Saye surface rule on a per-cell
# degree-p fit of the composite level set.  Colored by |phi| ~ distance from the
# true surface: smooth surfaces (airfoil, platform, fillet) stay dark, the creases
# -- sharp trailing edge, tip-gap edge, platform box edges -- light up.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = "bank_v98d_ROTOR_1"
ps = [1, 2]
data = {p: np.loadtxt(f"output/{tag}_recon3d_p{p}.txt") for p in ps}
vmax = 0.006

def frame(ax, P):
    ctr = (P.max(0)+P.min(0))/2; r = (P.max(0)-P.min(0)).max()/2
    for s,c in zip("xyz", ctr): getattr(ax,f"set_{s}lim")(c-r, c+r)
    ax.set_box_aspect((1,1,1))

fig = plt.figure(figsize=(15.5, 7.6))
for col, p in enumerate(ps):
    d = data[p]; P, dev = d[:, :3], d[:, 3]
    ax = fig.add_subplot(1, 2, col+1, projection="3d")
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=dev, cmap="inferno", vmin=0, vmax=vmax,
                    s=1.6, alpha=0.7, linewidths=0)
    frame(ax, P); ax.view_init(elev=20, azim=-58)
    ax.set_title(f"deg-{p} reconstruction\n{len(P)} pts   dev p95={np.percentile(dev,95):.4f}  max={dev.max():.4f}", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (axial)")
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
cb.set_label("|φ| at the reconstructed point  ~  distance from the true surface")
fig.suptitle("ROTOR 1 blade — 3-D reconstructed boundary (blade + platform + fillet + tip gap).  "
             "Smooth surfaces dark; creases (trailing edge, tip, platform edges) light up.", fontsize=12)
fig.savefig("/tmp/blade3d.png", dpi=120, bbox_inches="tight")
print("wrote /tmp/blade3d.png")

# a zoom near the tip / trailing-edge region (high r, aft z) for p=2
fig2 = plt.figure(figsize=(8.4, 7.6))
d = data[2]; P, dev = d[:, :3], d[:, 3]
# pick the aft-tip corner: largest z and outer radius (r = sqrt(x^2+y^2))
rr = np.hypot(P[:,0], P[:,1])
sel = (P[:,2] > np.percentile(P[:,2], 70)) & (rr > np.percentile(rr, 70))
Pz, dz = P[sel], dev[sel]
ax = fig2.add_subplot(111, projection="3d")
ax.scatter(Pz[:,0], Pz[:,1], Pz[:,2], c=dz, cmap="inferno", vmin=0, vmax=vmax, s=6, alpha=0.85, linewidths=0)
frame(ax, Pz); ax.view_init(elev=22, azim=-58)
ax.set_title("deg-2, aft-tip zoom: the tip-gap edge + trailing edge (creases) glow", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
fig2.savefig("/tmp/blade3d_zoom.png", dpi=122, bbox_inches="tight")
print("wrote /tmp/blade3d_zoom.png")
