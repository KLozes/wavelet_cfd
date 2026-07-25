#!/usr/bin/env python3
# 3-D reconstructed boundary of a LENS = two spheres (R=0.72) separated along the
# (1,1,1) DIAGONAL so the crease ridge cuts through cell interiors (an axis-aligned
# crease would land on grid faces -> captured exactly even by single-poly).
# The boundary is two spherical caps meeting at a SHARP ridge circle (a crease).
# Left  = single-polynomial Saye (fit max: rounds/oscillates, ~2x spurious points).
# Right = multi-polynomial CSG-aware Saye (two smooth branches, sharp ridge).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, sep = 0.72, 0.5
c = sep/np.sqrt(3.0)
rr = np.sqrt(R*R - sep*sep)                    # ridge radius
u = np.array([1,-1,0])/np.sqrt(2); v = np.array([1,1,-2])/np.sqrt(6)   # basis of plane x+y+z=0

Ps = np.loadtxt("/tmp/surf_single.txt")[:, :3]
Pm4 = np.loadtxt("/tmp/surf_multi.txt")
Pm, tag = Pm4[:, :3], Pm4[:, 3]

def dist_to_true(P):
    x, y, z = P[:,0], P[:,1], P[:,2]
    pA = (x-c)**2 + (y-c)**2 + (z-c)**2 - R*R
    pB = (x+c)**2 + (y+c)**2 + (z+c)**2 - R*R
    comp = np.maximum(pA, pB)
    gA = 2*np.sqrt((x-c)**2+(y-c)**2+(z-c)**2); gB = 2*np.sqrt((x+c)**2+(y+c)**2+(z+c)**2)
    return np.abs(comp)/np.maximum(np.where(pA >= pB, gA, gB), 1e-9)

es, em = dist_to_true(Ps), dist_to_true(Pm)
vmax = max(np.percentile(es, 99.5), 1e-6)
th = np.linspace(0, 2*np.pi, 240)
ring = rr*(np.outer(np.cos(th), u) + np.outer(np.sin(th), v))

def setup(ax):
    ax.plot(ring[:,0], ring[:,1], ring[:,2], color="#00b0ff", lw=2.6, zorder=10, label="true crease ridge")
    ax.set_xlim(-0.7,0.7); ax.set_ylim(-0.7,0.7); ax.set_zlim(-0.7,0.7)
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=26, azim=-52)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

fig = plt.figure(figsize=(15, 7.6))
for k, (P, e, ttl) in enumerate([
        (Ps, es, f"SINGLE polynomial (fit max)\n{len(Ps)} pts (~2x, spurious)   max dist {es.max():.4f}"),
        (Pm, em, f"MULTI polynomial — CSG-aware\n{len(Pm)} pts   max dist {em.max():.4f}")]):
    ax = fig.add_subplot(1, 2, k+1, projection="3d")
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=e, cmap="inferno", vmin=0, vmax=vmax,
                    s=5, alpha=0.85, linewidths=0)
    setup(ax); ax.set_title(ttl, fontsize=12); ax.legend(loc="upper left", fontsize=9)
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
cb.set_label("distance of reconstructed point from the TRUE boundary")
fig.suptitle("3-D reconstructed boundary at a sharp crease (lens = two spheres, off-grid ridge).  "
             "Single-poly oscillates/rounds at the ridge; CSG-aware keeps it exact.", fontsize=12)
fig.savefig("/tmp/surf3d.png", dpi=115, bbox_inches="tight")
print("wrote /tmp/surf3d.png  max dist single %.4f  multi %.4f" % (es.max(), em.max()))

fig2 = plt.figure(figsize=(7.8, 7.4))
ax = fig2.add_subplot(111, projection="3d")
ax.scatter(Pm[:,0], Pm[:,1], Pm[:,2], c=np.where(tag<0.5, "#1f77b4", "#ff7f0e"), s=5, alpha=0.85, linewidths=0)
setup(ax); ax.get_legend() and ax.get_legend().remove()
ax.plot(ring[:,0], ring[:,1], ring[:,2], color="k", lw=2.4, zorder=10)
ax.set_title("CSG-aware boundary: the two spherical caps (blue / orange)\n"
             "meet cleanly along the sharp crease ridge (black)", fontsize=11)
fig2.savefig("/tmp/surf3d_multi.png", dpi=115, bbox_inches="tight")
print("wrote /tmp/surf3d_multi.png")
