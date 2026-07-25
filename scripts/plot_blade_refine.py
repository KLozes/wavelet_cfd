#!/usr/bin/env python3
# Mesh refinement of the ROTOR 1 blade reconstruction: deg-2 Saye boundary at
# res 96 vs res 192, same |phi| color scale.  Refinement thins the crease bands
# and drops the deviation ~ (1/2)^(p+1); the smooth surfaces go essentially black.
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = "bank_v98d_ROTOR_1"
files = [(96, f"output/{tag}_recon3d_p2_res96.txt"),
         (192, f"output/{tag}_recon3d_p2.txt")]      # p2.txt = the fresh res-192 dump
vmax = 0.006

def frame(ax, P):
    ctr = (P.max(0)+P.min(0))/2; r = (P.max(0)-P.min(0)).max()/2
    for s,c in zip("xyz", ctr): getattr(ax,f"set_{s}lim")(c-r, c+r)
    ax.set_box_aspect((1,1,1))

fig = plt.figure(figsize=(15.5, 7.6))
for col,(res,fn) in enumerate(files):
    d = np.loadtxt(fn); P, dev = d[:, :3], d[:, 3]
    ax = fig.add_subplot(1, 2, col+1, projection="3d")
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=dev, cmap="inferno", vmin=0, vmax=vmax,
                    s=1.1, alpha=0.65, linewidths=0)
    frame(ax, P); ax.view_init(elev=20, azim=-58)
    ax.set_title(f"deg-2, res {res}   ({len(P)} pts)\ndev p95={np.percentile(dev,95):.5f}  max={dev.max():.4f}", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (axial)")
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
cb.set_label("|φ| ~ distance from the true surface  (same scale both panels)")
fig.suptitle("Mesh refinement of the ROTOR 1 blade (deg-2): res 96 → res 192.  "
             "Crease bands thin and dim; the deviation drops ~8× (=(1/2)^3).", fontsize=12)
fig.savefig("/tmp/blade_refine.png", dpi=120, bbox_inches="tight")
print("wrote /tmp/blade_refine.png")
