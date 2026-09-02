#!/usr/bin/env python3
"""Show the cut cells and the quadrature the IGA solver actually integrates on.

Reads the CUT_DUMP=1 CSVs (<tag>_cells / _sayevol / _sayesurf) for two runs and
puts them side by side, so the effect of the CUT_GEOMSUB sub-cell level set is
visible directly: the SURFACE rule gets s^2 times denser (a finer, more accurate
interface), while the VOLUME rule stays at the same ~m points per cell because
NNLS prunes it straight back down -- Potter's point, made as a picture.

usage: python3 scripts/plot_cutcells.py --a output/sphere_gs1 --b output/sphere_gs4
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="output/sphere_gs1")
ap.add_argument("--b", default="output/sphere_gs4")
ap.add_argument("--la", default="CUT_GEOMSUB=1  (one level-set fit per element)")
ap.add_argument("--lb", default="CUT_GEOMSUB=4  (4$^3$ sub-cell level set)")
ap.add_argument("--out", default="output/cutcells.png")
ap.add_argument("--slab", type=float, default=0.6, help="slab half-thickness in cells")
ap.add_argument("--zc", type=float, default=None, help="slab centre in z (default: body centroid)")
ap.add_argument("--circle", action="store_true", help="overlay the mean-radius circle (sphere only)")
ap.add_argument("--title", default="sphere, res 16, p=2")
ap.add_argument("--zoom", type=float, default=0.72, help="zoom centre as a fraction of the body radius")
ap.add_argument("--proj", default="xy", choices=["xy", "yz", "zx"],
                help="plane to draw; the remaining axis is the one sliced")
args = ap.parse_args()

def rd(p):
    with open(p) as fh: rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def load(tag):
    C = rd(tag + "_cells.csv")
    xs = np.stack([C["x%d" % i] for i in range(8)], 1)
    ys = np.stack([C["y%d" % i] for i in range(8)], 1)
    zs = np.stack([C["z%d" % i] for i in range(8)], 1)
    cells = dict(cut=C["cut"].astype(int),
                 x0=xs.min(1), x1=xs.max(1), y0=ys.min(1), y1=ys.max(1),
                 z0=zs.min(1), z1=zs.max(1))
    return cells, rd(tag + "_sayevol.csv"), rd(tag + "_sayesurf.csv")

# --proj picks which two axes are drawn; the third is the one the slab cuts along.
# Everything below is written in (x, y, z) = (horizontal, vertical, sliced), so the
# permutation is applied once here rather than threaded through the plotting.
PERM = {"xy": ("x", "y", "z"), "yz": ("y", "z", "x"), "zx": ("z", "x", "y")}[args.proj]
def perm(cells, vol, surf):
    u, v, w = PERM
    c = dict(cut=cells["cut"])
    for new, old in (("x", u), ("y", v), ("z", w)):
        c[new + "0"], c[new + "1"] = cells[old + "0"], cells[old + "1"]
    rn = lambda d: {new: d[old] for new, old in (("x", u), ("y", v), ("z", w)) if old in d} | \
                   ({"w": d["w"]} if "w" in d else {})
    return c, rn(vol), rn(surf)

cA, vA, sA = perm(*load(args.a))
cB, vB, sB = perm(*load(args.b))

# equator of the body: centre from the surface points, slab one cell thick
ctr = np.array([sA["x"].mean(), sA["y"].mean(), sA["z"].mean()])
h = np.median(cA["x1"] - cA["x0"])
zc = ctr[2] if args.zc is None else args.zc
half = args.slab * h
R = np.sqrt(((np.stack([sA["x"], sA["y"], sA["z"]], 1) - ctr) ** 2).sum(1)).mean()

fig, ax = plt.subplots(1, 2, figsize=(15.0, 7.6))
for k, (cc, vv, ss, lab) in enumerate([(cA, vA, sA, args.la), (cB, vB, sB, args.lb)]):
    a = ax[k]
    inslab = (cc["z0"] <= zc + half) & (cc["z1"] >= zc - half)
    for cut, fc, ec, lw in [(0, "#e9edf2", "#b9c4d0", 0.4), (1, "#ffe0b8", "#e08a1e", 0.9)]:
        sel = inslab & (cc["cut"] == cut)
        pc = PatchCollection(
            [Rectangle((cc["x0"][i], cc["y0"][i]), cc["x1"][i] - cc["x0"][i],
                       cc["y1"][i] - cc["y0"][i]) for i in np.where(sel)[0]],
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
        a.add_collection(pc)
    mv = np.abs(vv["z"] - zc) < half
    a.scatter(vv["x"][mv], vv["y"][mv], s=5.0, c="#1f5fa8", zorder=3,
              label="pruned volume pts (%d/cell)" % round(len(vv["x"]) / max((cc["cut"] == 1).sum(), 1)))
    ms = np.abs(ss["z"] - zc) < half
    a.scatter(ss["x"][ms], ss["y"][ms], s=2.0, c="#c81e3c", zorder=4,
              label="surface pts (%d/cell)" % round(len(ss["x"]) / max((cc["cut"] == 1).sum(), 1)))
    if args.circle:
        th = np.linspace(0, 2 * np.pi, 720)
        a.plot(ctr[0] + R * np.cos(th), ctr[1] + R * np.sin(th), "-", color="#0b7a53",
               lw=1.0, zorder=5, label="mean-radius circle")
    a.set_aspect("equal"); a.set_title(lab, fontsize=12)
    sl = inslab & (cc["cut"] == 1)
    if sl.any():
        a.set_xlim(cc["x0"][sl].min() - h, cc["x1"][sl].max() + h)
        a.set_ylim(cc["y0"][sl].min() - h, cc["y1"][sl].max() + h)
    a.set_xlabel(PERM[0]); a.set_ylabel(PERM[1])
    a.legend(loc="upper right", fontsize=8, framealpha=0.95)

fig.suptitle("Cut cells and their quadrature, one-cell slab (%s)\n" % args.title +
             "orange = cut cells, grey = interior; the volume rule is pruned to the "
             "same count in both, only the interface gets finer", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(args.out, dpi=140)
print("wrote", args.out)

# ---- zoom: a few cut cells, so the sub-cell structure is legible ---------------
fig2, ax2 = plt.subplots(1, 2, figsize=(15.0, 7.6))
msz = np.abs(sA["z"] - zc) < half
zx, zy = (sA["x"][msz].mean(), sA["y"][msz].mean()) if msz.any() else (ctr[0], ctr[1])
if args.circle: zx, zy = ctr[0] + R * args.zoom, ctr[1] + R * args.zoom
w = 3.2 * h
for k, (cc, vv, ss, lab) in enumerate([(cA, vA, sA, args.la), (cB, vB, sB, args.lb)]):
    a = ax2[k]
    inslab = (cc["z0"] <= zc + half) & (cc["z1"] >= zc - half)
    box = inslab & (cc["x1"] > zx - w) & (cc["x0"] < zx + w) & (cc["y1"] > zy - w) & (cc["y0"] < zy + w)
    for cut, fc, ec in [(0, "#e9edf2", "#b9c4d0"), (1, "#ffe0b8", "#e08a1e")]:
        sel = box & (cc["cut"] == cut)
        a.add_collection(PatchCollection(
            [Rectangle((cc["x0"][i], cc["y0"][i]), cc["x1"][i] - cc["x0"][i],
                       cc["y1"][i] - cc["y0"][i]) for i in np.where(sel)[0]],
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=1))
    mv = np.abs(vv["z"] - zc) < half
    a.scatter(vv["x"][mv], vv["y"][mv], s=34, c="#1f5fa8", zorder=3, label="volume quadrature")
    ms = np.abs(ss["z"] - zc) < half
    a.scatter(ss["x"][ms], ss["y"][ms], s=12, c="#c81e3c", zorder=4, label="interface quadrature")
    if args.circle:
        th = np.linspace(0, 2 * np.pi, 2000)
        a.plot(ctr[0] + R * np.cos(th), ctr[1] + R * np.sin(th), "-", color="#0b7a53", lw=1.2, zorder=5)
    a.set_aspect("equal"); a.set_xlim(zx - w, zx + w); a.set_ylim(zy - w, zy + w)
    a.set_title(lab, fontsize=12); a.set_xlabel(PERM[0]); a.set_ylabel(PERM[1])
    a.legend(loc="upper right", fontsize=9)
fig2.suptitle("Zoom: the same cut cells at cell scale (%s)" % args.title, fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(args.out.replace(".png", "_zoom.png"), dpi=140)
print("wrote", args.out.replace(".png", "_zoom.png"))
