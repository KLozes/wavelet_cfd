#!/usr/bin/env python3
"""Does the phi = 1/2 contour lie on the body surface?

  usage: plot_brink_halfcontour.py out.png h delta/h

Colour is phi itself on a LINEAR 0..1 scale with a diverging map, so white is
phi = 1/2 and the two halves of the band get equal weight.  A log scale is the
wrong tool twice over here: it gave 95% of its range to phi < 1/2 (making the
ramp look as if it sat inside the surface), and it inflated a band that is in
truth razor thin -- at delta = h/8, phi runs 0.1 -> 0.9 across 0.27 of a cell.

The white midline is phi = 1/2.  The measurement panel takes the contour
matplotlib extracts from the phi field and evaluates the EXACT signed distance
at each of its vertices: if the half-contour is the surface, that distance is
zero to within the contouring grid.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

out = sys.argv[1]
h   = float(sys.argv[2])
dh  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.125
d   = dh*h
EPS = 1e-6
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
P = np.column_stack([gx, gy])
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

def sdf(q):
    a = P; b = np.roll(P, -1, axis=0); e = b - a; L2 = (e*e).sum(1)
    w = q[:, None, :] - a[None, :, :]
    t = np.clip((w*e).sum(2)/L2, 0, 1)
    dd = w - t[:, :, None]*e
    dist = np.sqrt((dd*dd).sum(2)).min(1)
    return np.where(poly.contains_points(q), -dist, dist)

def phi_of(S):  return EPS + (1 - EPS)/(1 + np.exp(-2*S/d))

fig = plt.figure(figsize=(13.6, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.36, wspace=.28)

def panel(ax, XL, XR, YB, YT, n, title, cells=False):
    X, Y = np.meshgrid(np.linspace(XL, XR, n), np.linspace(YB, YT, int(n*(YT-YB)/(XR-XL))))
    S = sdf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    im = ax.pcolormesh(X, Y, phi_of(S), cmap="RdBu_r", vmin=0.0, vmax=1.0, shading="gouraud")
    # the contour matplotlib finds in the phi field itself
    cs = ax.contour(X, Y, phi_of(S), levels=[0.5], colors="k", linewidths=2.6)
    ax.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "-", color="#00D6A0",
            lw=1.1, dashes=(5, 4))
    if cells:
        for xv in np.arange(np.ceil(XL/h)*h, XR, h): ax.axvline(xv, color="k", lw=.35, alpha=.2)
        for yv in np.arange(np.ceil(YB/h)*h, YT, h): ax.axhline(yv, color="k", lw=.35, alpha=.2)
    ax.set_aspect("equal"); ax.set_xlim(XL, XR); ax.set_ylim(YB, YT)
    ax.set_title(title, fontsize=10.5); ax.tick_params(labelsize=8)
    return im, cs

axA = fig.add_subplot(gs[0, :])
im, csA = panel(axA, 0.10, 0.52, -0.055, 0.045, 900,
                f"φ, LINEAR 0→1 (white = ½),   δ = {dh}h — on a linear scale the smeared band is barely a third of a cell wide")
axA.set_xlabel("x/c  (body-centred)", fontsize=9); axA.set_ylabel("y/c", fontsize=9)
cb = plt.colorbar(im, ax=axA, fraction=.020, pad=.012)
cb.set_label("φ    (½ = the wall)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)

axB = fig.add_subplot(gs[1, 0])
x0, i0 = 0.35, np.argmin(np.abs(gx - 0.35) + (gy < 0)*10)
y0 = gy[i0]
panel(axB, x0-3*h, x0+3*h, y0-3*h, y0+3*h, 500,
      f"zoom: 6 cells across the upper surface at x/c = {x0:+.2f}", cells=True)
axB.set_xlabel("x/c", fontsize=9); axB.set_ylabel("y/c", fontsize=9)

# ---- phi along the wall normal, linear axes --------------------------------
axC = fig.add_subplot(gs[1, 1])
tang = P[i0+1] - P[i0-1]; nrm = np.array([tang[1], -tang[0]]); nrm /= np.hypot(*nrm)
if sdf(np.array([[gx[i0] + 0.01*nrm[0], gy[i0] + 0.01*nrm[1]]]))[0] < 0: nrm = -nrm
tt = np.linspace(-2*h, 2*h, 1200)
Q = np.column_stack([gx[i0] + tt*nrm[0], gy[i0] + tt*nrm[1]])
sc = sdf(Q); pc = phi_of(sc)
axC.plot(tt/h, pc, lw=2.2, color="#0E6B5B")
axC.axhline(0.5, color="#A5433A", lw=1, ls="--"); axC.axvline(0.0, color="#A5433A", lw=1, ls="--")
axC.plot([0], [phi_of(np.array([0.0]))[0]], "o", ms=7, mfc="none", mec="#A5433A", mew=2)
axC.annotate("φ(0) = ½ + ε/2", xy=(0, 0.5), xytext=(0.45, 0.62), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#A5433A"))
axC.set_xlabel("distance along the wall normal,  s/h", fontsize=9)
axC.set_ylabel("φ", fontsize=9); axC.set_xlim(-2, 2); axC.set_ylim(-0.03, 1.03)
axC.set_title(f"φ across the wall at x/c = {x0:+.2f}", fontsize=10)
axC.grid(alpha=.28); axC.tick_params(labelsize=8)

# ---- residual, LINEAR ------------------------------------------------------
axD = fig.add_subplot(gs[1, 2])
V = np.vstack([seg for seg in csA.allsegs[0] if len(seg) > 1])
dv = sdf(V); gspc = (0.52-0.10)/900/h
axD.plot(V[:, 0], np.abs(dv)/h, ".", ms=2.4, color="#0E6B5B")
axD.axhline(gspc, color="#A5433A", lw=1.3, ls="--",
            label=f"contouring grid, {gspc:.3f} h")
axD.set_xlabel("x/c  (body-centred)", fontsize=9)
axD.set_ylabel("|distance to the surface|  /  h", fontsize=9)
axD.set_title("every vertex of the extracted φ=½ contour", fontsize=10)
axD.set_ylim(0, 1.35*gspc); axD.grid(alpha=.28)
axD.legend(fontsize=8.5, loc="upper right"); axD.tick_params(labelsize=8)

fig.suptitle("The φ = ½ contour IS the body surface — solid line: contour extracted from the φ field; "
             "dashed: the geometry file", fontsize=12)
plt.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
print(f"  contour vertices: {len(V)}")
print(f"  |s| at those vertices:  median {np.median(np.abs(dv)):.3e} c = {np.median(np.abs(dv))/h:.2e} h")
print(f"                          max    {np.abs(dv).max():.3e} c = {np.abs(dv).max()/h:.2e} h")
print(f"  contouring grid spacing:       {(0.52-0.10)/900:.3e} c = {(0.52-0.10)/900/h:.3f} h")
