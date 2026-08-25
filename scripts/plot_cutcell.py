#!/usr/bin/env python3
"""Draw ONE cut cell from the solver's own quadrature, and VERIFY it.

Every check below compares the solver's rules against a value computed
independently in this script from the analytic circle -- so the plot is not
just a picture, it is a falsifiable claim about the geometry the solver is
integrating over.

usage:  python3 scripts/plot_cutcell.py --elem 1
"""
import argparse, csv, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

def rd(p):
    with open(p) as fh: rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="output")
ap.add_argument("--elem", type=int, default=1)
ap.add_argument("--cx", type=float, default=1.5)
ap.add_argument("--cy", type=float, default=2.0)
ap.add_argument("--R",  type=float, default=0.5)
ap.add_argument("--out", default="output/cut_cell.png")
a = ap.parse_args()

geom = rd(os.path.join(a.dir, "cut_geom.csv"))
wall = rd(os.path.join(a.dir, "cut_wall.csv"))
vol  = rd(os.path.join(a.dir, "cut_vol.csv"))
fine = rd(os.path.join(a.dir, "cut_fine.csv")) if os.path.exists(
       os.path.join(a.dir, "cut_fine.csv")) else None

g = {k: v[geom["elem"] == a.elem][0] for k, v in geom.items()}
x0, y0, hx, hy = g["x0"], g["y0"], g["hx"], g["hy"]
sel = lambda d: {k: v[d["elem"] == a.elem] for k, v in d.items()}
w, v = sel(wall), sel(vol)
f = sel(fine) if fine is not None else None

# ---- INDEPENDENT reference: exact fluid area of this cell -------------------
# fluid = cell minus disc.  Integrate the disc's y-extent clipped to the cell.
def disc_chord(x):
    d = a.R*a.R - (x - a.cx)**2
    if d <= 0: return 0.0
    s = math.sqrt(d)
    lo, hi = max(y0, a.cy - s), min(y0 + hy, a.cy + s)
    return max(0.0, hi - lo)
disc_area, _ = quad(disc_chord, max(x0, a.cx - a.R), min(x0 + hx, a.cx + a.R),
                    limit=400, epsabs=1e-14, epsrel=1e-14)
exact_fluid = hx*hy - disc_area
# the solver reports a REFERENCE-cell fraction; z drops out (z-invariant body)
solver_fluid = g["volfrac"]*hx*hy

# exact wall arc length inside this cell: angles where the circle is in the cell
th = np.linspace(0, 2*math.pi, 2000001)
xx, yy = a.cx + a.R*np.cos(th), a.cy + a.R*np.sin(th)
inside = (xx >= x0-1e-12) & (xx <= x0+hx+1e-12) & (yy >= y0-1e-12) & (yy <= y0+hy+1e-12)
exact_arc = inside.sum()/len(th) * 2*math.pi*a.R
solver_arc = g["wallarea"]*hx          # reference measure -> physical length

rw = np.hypot(w["x"] - a.cx, w["y"] - a.cy)
rv = np.hypot(v["x"] - a.cx, v["y"] - a.cy)

print(f"cut element {a.elem}   cell x[{x0:.4f},{x0+hx:.4f}] y[{y0:.4f},{y0+hy:.4f}]")
print(f"  wall points ON the circle      max| |r|-R | = {np.abs(rw-a.R).max():.3e}")
print(f"  volume points IN the fluid     min( |r|-R ) = {(rv-a.R).min():+.3e}   (must be > 0)")
print(f"  volume points inside the cell  {int(((v['x']>=x0-1e-12)&(v['x']<=x0+hx+1e-12)&(v['y']>=y0-1e-12)&(v['y']<=y0+hy+1e-12)).all())} (1 = yes)")
print(f"  fluid area   solver {solver_fluid:.10f}   exact {exact_fluid:.10f}   rel {abs(solver_fluid-exact_fluid)/exact_fluid:.3e}")
print(f"  wall arc     solver {solver_arc:.10f}   exact {exact_arc:.10f}   rel {abs(solver_arc-exact_arc)/exact_arc:.3e}")
print(f"  quadrature weights sum to the area (that IS the fluid-area check above); {len(w['x'])} wall pts, {len(v['x'])} volume pts")

fig, ax = plt.subplots(figsize=(7.2, 7.2))
if f is not None and len(f["x"]):
    ax.scatter(f["x"], f["y"], c=f["rho"], s=26, marker="s", cmap="viridis",
               linewidths=0, alpha=0.95, label="polynomial, fluid side (cut_fine)")
tt = np.linspace(0, 2*math.pi, 4000)
ax.plot(a.cx + a.R*np.cos(tt), a.cy + a.R*np.sin(tt), "-", color="crimson", lw=1.6,
        label="analytic circle (independent)")
ax.add_patch(plt.Rectangle((x0, y0), hx, hy, fill=False, ec="k", lw=1.6, label="cell"))
ax.plot(v["x"], v["y"], "o", ms=3.5, mfc="none", mec="#1f77b4", label="Saye volume points")
ax.plot(w["x"], w["y"], ".", ms=5, color="k", label="Saye wall points")
ax.set_xlim(x0 - 0.12*hx, x0 + 1.12*hx); ax.set_ylim(y0 - 0.12*hy, y0 + 1.12*hy)
ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_title(f"cut element {a.elem}:  fluid area {solver_fluid:.8f} vs exact {exact_fluid:.8f}"
             f"\nwall pts off the circle by {np.abs(rw-a.R).max():.1e}", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.savefig(a.out, dpi=150, bbox_inches="tight")
print("wrote", a.out)
