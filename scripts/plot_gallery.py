#!/usr/bin/env python3
# Gallery of CSG INTERSECTION geometries: single-polynomial vs multi-polynomial
# (CSG-aware) Saye reconstructed 3-D boundaries.  Points colored by distance from
# the TRUE boundary -> single-poly lights up along every crease; CSG-aware stays
# dark (on the surface).  Left col = single, right col = CSG-aware, per geometry.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEOM = [
    ("lens",      "two spheres  (1 ridge circle)"),
    ("band",      "sphere ∩ slab  (2 rings + 2 disks)"),
    ("wedge",     "sphere ∩ dihedral  (2 rings + edge)"),
    ("corner",    "sphere ∩ 3 planes  (rings + edges + corner)"),
    ("roundcube", "cube ∩ sphere  (rounded edges/corners)"),
    ("trisphere", "three spheres  (3 ridges)"),
]
VIEW = {"lens":(24,-52),"band":(18,-60),"wedge":(24,-58),"corner":(26,-48),
        "roundcube":(26,-52),"trisphere":(30,-52)}

fig = plt.figure(figsize=(9.6, 4.4*len(GEOM)))
for gi,(name,desc) in enumerate(GEOM):
    S = np.loadtxt(f"/tmp/surf_{name}_single.txt")
    M = np.loadtxt(f"/tmp/surf_{name}_multi.txt")
    Ps, ds = S[:, :3], S[:, 3]
    Pm, dm = M[:, :3], M[:, 3]
    vmax = max(np.percentile(np.concatenate([ds, dm]), 99.0), 1e-6)
    ev, az = VIEW[name]
    for col,(P,d,lab,cnt) in enumerate([
            (Ps, ds, "SINGLE polynomial (fit max)", len(Ps)),
            (Pm, dm, "MULTI polynomial (CSG-aware)", len(Pm))]):
        ax = fig.add_subplot(len(GEOM), 2, 2*gi+col+1, projection="3d")
        sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=d, cmap="inferno", vmin=0, vmax=vmax,
                        s=3.2, alpha=0.85, linewidths=0)
        lim = np.abs(P).max()*1.02
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=ev, azim=az)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f"{lab}\n{cnt} pts   max off-surf {d.max():.3f}", fontsize=9.5)
        if col == 0:
            ax.text2D(-0.16, 0.5, f"{name}\n{desc}", transform=ax.transAxes,
                      fontsize=10, rotation=90, va="center", ha="center", fontweight="bold")

fig.suptitle("CSG intersection boundaries: single-poly (fit max) lights up at every crease;  "
             "CSG-aware (multi-branch) stays on the true surface", fontsize=13, y=0.997)
fig.tight_layout(rect=[0.02,0,1,0.99])
fig.savefig("/tmp/surf_gallery.png", dpi=108, bbox_inches="tight")
print("wrote /tmp/surf_gallery.png")

# a second, branch-colored view of the CSG-aware boundaries (clean sharp pieces)
fig2 = plt.figure(figsize=(14.5, 9))
cmap = plt.get_cmap("tab10")
for gi,(name,desc) in enumerate(GEOM):
    M = np.loadtxt(f"/tmp/surf_{name}_multi.txt")
    P, tag = M[:, :3], M[:, 4].astype(int)
    ev, az = VIEW[name]
    ax = fig2.add_subplot(2, 3, gi+1, projection="3d")
    ax.scatter(P[:,0], P[:,1], P[:,2], c=[cmap(t%10) for t in tag], s=3.2, alpha=0.85, linewidths=0)
    lim = np.abs(P).max()*1.02
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=ev, azim=az)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_title(f"{name}: {desc}", fontsize=10)
fig2.suptitle("CSG-aware reconstructed boundaries, colored by branch — each smooth piece "
              "meets its neighbors along a sharp crease", fontsize=13)
fig2.tight_layout(rect=[0,0,1,0.96])
fig2.savefig("/tmp/surf_gallery_branch.png", dpi=112, bbox_inches="tight")
print("wrote /tmp/surf_gallery_branch.png")
