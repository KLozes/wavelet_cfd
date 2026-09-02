#!/usr/bin/env python3
"""Zoom on the blade LEADING EDGE and colour the interface quadrature by its
actual error.

phi is a signed distance, so |phi| at a surface quadrature point IS that point's
displacement from the true surface -- the picture is therefore a measurement, not
an impression.  Cut cells are drawn behind, volume points on top.

usage: python3 scripts/plot_le_zoom.py
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+",
                default=["blade_sy1", "blade_sy4", "blade_ln4", "blade_ln8"])
ap.add_argument("--labs", nargs="+",
                default=["Saye  s=1  (old default)", "Saye  s=4",
                         "piecewise-linear  s=4", "piecewise-linear  s=8"])
ap.add_argument("--rc", type=float, default=4.85, help="slab radius (physical x)")
ap.add_argument("--slab", type=float, default=0.25, help="slab half-width in cells")
ap.add_argument("--yc", type=float, default=0.140)
ap.add_argument("--zc", type=float, default=6.600)
ap.add_argument("--w", type=float, default=0.165, help="half-window")
ap.add_argument("--out", default="output/le_zoom.png")
args = ap.parse_args()

H = 0.16464                                   # cell size of the res-24 blade grid
half = args.slab * H

def load_surf(tag):
    """stream the file, keep only the slab -- the s=8 dump is 144 MB"""
    xs, ys, zs, ps = [], [], [], []
    with open("output/%s_sayesurf.csv" % tag) as fh:
        fh.readline()
        for ln in fh:
            i = ln.find(","); x = float(ln[:i])
            if abs(x - args.rc) >= half: continue
            r = ln[i+1:].split(",")
            ys.append(float(r[0])); zs.append(float(r[1])); ps.append(abs(float(r[2])))
            xs.append(x)
    return np.array(ys), np.array(zs), np.array(ps)

def load_cells(tag):
    import csv
    with open("output/%s_cells.csv" % tag) as fh: rows = list(csv.DictReader(fh))
    C = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    xs = np.stack([C["x%d" % i] for i in range(8)], 1)
    ys = np.stack([C["y%d" % i] for i in range(8)], 1)
    zs = np.stack([C["z%d" % i] for i in range(8)], 1)
    return (C["cut"].astype(int), xs.min(1), xs.max(1),
            ys.min(1), ys.max(1), zs.min(1), zs.max(1))

def load_vol(tag):
    import csv
    with open("output/%s_sayevol.csv" % tag) as fh: rows = list(csv.DictReader(fh))
    V = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    m = np.abs(V["x"] - args.rc) < half
    return V["y"][m], V["z"][m]

data = [(t, l, load_surf(t), load_cells(t), load_vol(t))
        for t, l in zip(args.tags, args.labs)]
lo = min(p[p > 0].min() for _, _, (_, _, p), _, _ in data)
hi = max(p.max() for _, _, (_, _, p), _, _ in data)
norm = LogNorm(vmin=max(lo, hi*1e-4), vmax=hi)

fig, axes = plt.subplots(1, len(data), figsize=(4.3*len(data), 5.6))
for a, (tag, lab, (sy, sz, sp), cel, (vy, vz)) in zip(np.atleast_1d(axes), data):
    cut, x0, x1, y0, y1, z0, z1 = cel
    box = (x0 <= args.rc+half) & (x1 >= args.rc-half) \
        & (y1 > args.yc-args.w) & (y0 < args.yc+args.w) \
        & (z1 > args.zc-args.w) & (z0 < args.zc+args.w)
    for c, fc, ec in [(0, "#eef1f5", "#c3ccd6"), (1, "#fff3e0", "#e8a33d")]:
        sel = box & (cut == c)
        a.add_collection(PatchCollection(
            [Rectangle((y0[i], z0[i]), y1[i]-y0[i], z1[i]-z0[i]) for i in np.where(sel)[0]],
            facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=1))
    a.scatter(vy, vz, s=13, c="#2b6cb0", zorder=2, label="volume quadrature")
    sc = a.scatter(sy, sz, s=7, c=np.maximum(sp, norm.vmin), cmap="inferno_r",
                   norm=norm, zorder=3, label="interface quadrature")
    rms = np.sqrt((sp**2).mean()) if len(sp) else float("nan")
    a.set_title("%s\nrms |$\\phi$| = %.2e   (%.2f%% of h)"
                % (lab, rms, 100*rms/H), fontsize=10.5, pad=7)
    a.set_aspect("equal")
    a.set_xlim(args.yc-args.w, args.yc+args.w)
    a.set_ylim(args.zc-args.w, args.zc+args.w)
    a.set_xlabel("y"); a.set_ylabel("z")
    a.legend(loc="lower left", fontsize=8, framealpha=0.95)
cb = fig.colorbar(sc, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.01)
cb.set_label("|$\\phi$| at the quadrature point  =  distance from the true surface")
fig.suptitle("Blade LEADING EDGE, mid-span section (r = %.2f), res 24, p=2 -- "
             "cut cells and the interface quadrature, coloured by its own error"
             % args.rc, fontsize=12)
fig.savefig(args.out, dpi=145, bbox_inches="tight")
print("wrote", args.out)
