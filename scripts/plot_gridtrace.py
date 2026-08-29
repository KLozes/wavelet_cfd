#!/usr/bin/env python3
"""The INITIAL build cascade: grid adaptation driven purely by GEOMETRY.

The IC is uniform freestream, so every block added here comes from the
level-set refinement criterion (cut blocks + one halo ring), not from the flow.
"""
import glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

files = sorted(glob.glob("output/gridtrace_*.dat"),
               key=lambda f: int(f.split("_")[1].split(".")[0]))
s = np.loadtxt("output/rae2822_surface.dat")
gx, gy = np.append(s[:,3], s[0,3]), np.append(s[:,4], s[0,4])
cx, cy = 0.5*(gx.min()+gx.max()), 0.5*(gy.min()+gy.max())
chord = gx.max()-gx.min()
nl = len(files)
cmap = plt.get_cmap("viridis", nl)

ncol = 4; nrow = (nl+ncol-1)//ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 3.9*nrow))
axes = np.atleast_1d(axes).ravel()
import os
half = float(os.environ.get("TRACE_HALF", "1.1"))*chord
CELLS = os.environ.get("TRACE_CELLS", "0") == "1"
BS = 4
for k, f in enumerate(files):
    d = np.loadtxt(f); ax = axes[k]
    x0,y0,side,lvl,inr = d[:,0],d[:,1],d[:,2],d[:,3].astype(int),d[:,4].astype(int)
    m = inr>0
    if CELLS:   # draw individual CELLS, not blocks
        pats, cols = [], []
        for a,b,c,L in zip(x0[m],y0[m],side[m],lvl[m]):
            h = c/BS
            for i in range(BS):
                for j in range(BS):
                    pats.append(Rectangle((a+i*h,b+j*h),h,h)); cols.append(L)
        pc=PatchCollection(pats, edgecolor="0.55", linewidth=0.15)
        pc.set_array(np.array(cols))
    else:
        pats=[Rectangle((a,b),c,c) for a,b,c in zip(x0[m],y0[m],side[m])]
        pc=PatchCollection(pats, edgecolor="0.4", linewidth=0.3)
        pc.set_array(lvl[m])          # (was missing -> everything drew one colour)
    pc.set_cmap(cmap); pc.set_clim(-0.5,nl-0.5)
    ax.add_collection(pc)
    ax.plot(gx,gy,"-",color="crimson",lw=1.6,zorder=5)
    zx = float(os.environ.get("TRACE_CX", "nan"))
    px = gx.min() if os.environ.get("TRACE_AT")=="LE" else (
         gx.max() if os.environ.get("TRACE_AT")=="TE" else cx)
    py = gy[np.argmin(gx)] if os.environ.get("TRACE_AT")=="LE" else (
         gy[np.argmax(gx)] if os.environ.get("TRACE_AT")=="TE" else cy)
    ax.set_xlim(px-half,px+half); ax.set_ylim(py-half,py+half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("pass %d: max level %d  (h = c/%d)" % (k, lvl[m].max(), 4*2**lvl[m].max()), fontsize=9.5)
for k in range(nl, len(axes)): axes[k].axis("off")
fig.suptitle(os.environ.get("TRACE_TITLE",
    "RAE 2822: INITIAL grid built from GEOMETRY alone (uniform freestream IC)"), y=0.995)
fig.tight_layout()
cb = fig.colorbar(pc, ax=axes.tolist(), shrink=0.55, ticks=range(nl), pad=0.01)
cb.set_label("refinement level")
fig.savefig("output/gridtrace.png", dpi=135, bbox_inches="tight")
print("wrote output/gridtrace.png")
for f in files:
    d=np.loadtxt(f); m=d[:,4].astype(int)>0
    print("  %s: %4d blocks" % (f.split('/')[-1], m.sum()))
