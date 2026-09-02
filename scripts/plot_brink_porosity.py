#!/usr/bin/env python3
"""Brinkman volume fraction phi near the RAE 2822 trailing edge.

  usage: plot_brink_porosity.py out.png h [delta/h ...]

phi is a pure function of the geometry and the band width -- phi = eps +
(1-eps) sigmoid(2 s / delta) with s the exact signed distance to the section --
so this needs no solver run.  The point of the picture is that "solid" means
phi -> eps: wherever the section is thinner than the band, the body never gets
there and cannot hold circulation, no matter how well-balanced the scheme is.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1]
h   = float(sys.argv[2])
dels= [float(a) for a in sys.argv[3:]] or [1.5, 0.125]
EPS = 1e-6
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
P = np.column_stack([gx, gy])

def sdf(q):
    """signed distance, POSITIVE OUTSIDE (in the fluid), exact for the polyline"""
    a = P; b = np.roll(P, -1, axis=0)
    e = b - a; L2 = (e*e).sum(1)
    w = q[:, None, :] - a[None, :, :]
    t = np.clip((w*e).sum(2)/L2, 0, 1)
    d = w - t[:, :, None]*e
    dist = np.sqrt((d*d).sum(2)).min(1)
    cross = ((a[None,:,1] > q[:,None,1]) != (b[None,:,1] > q[:,None,1]))
    xint = a[None,:,0] + (q[:,None,1]-a[None,:,1])/(b[None,:,1]-a[None,:,1]+1e-300)*(b[None,:,0]-a[None,:,0])
    inside = (np.logical_and(cross, q[:,None,0] < xint).sum(1) % 2).astype(bool)
    return np.where(inside, -dist, dist)

XL, XR = 0.28, 0.56          # body is centred: trailing edge sits at x ~ +0.5
YB, YT = -0.075, 0.055
nx, ny = 620, 300
X, Y = np.meshgrid(np.linspace(XL, XR, nx), np.linspace(YB, YT, ny))
S = sdf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

fig = plt.figure(figsize=(13.6, 7.4))
gs = fig.add_gridspec(2, len(dels), height_ratios=[1.35, 1], hspace=.34, wspace=.30)
norm = LogNorm(vmin=EPS, vmax=1.0)

for c, dh in enumerate(dels):
    d = dh*h
    phi = EPS + (1-EPS)/(1+np.exp(-2*S/d))
    ax = fig.add_subplot(gs[0, c])
    im = ax.pcolormesh(X, Y, phi, norm=norm, cmap="magma", shading="gouraud")
    ax.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "-", color="#54CFB4", lw=1.4)
    for xv in np.arange(np.ceil(XL/h)*h, XR, h):
        ax.axvline(xv, color="k", lw=.3, alpha=.16)
    for yv in np.arange(np.ceil(YB/h)*h, YT, h):
        ax.axhline(yv, color="k", lw=.3, alpha=.16)
    ax.set_aspect("equal"); ax.set_xlim(XL, XR); ax.set_ylim(YB, YT)
    ax.set_title(f"δ = {dh}h" + ("   (old default)" if dh >= 1.0 else "   (what worked)"), fontsize=11)
    ax.set_xlabel("x/c  (body-centred; TE at +0.5)", fontsize=9)
    if c == 0: ax.set_ylabel("y/c", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=.040, pad=.025)
    cb.set_label("φ  (ε = 10⁻⁶ is solid)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)

# ---- min phi through the section, vs x ------------------------------------
ax2 = fig.add_subplot(gs[1, :])
xs = np.linspace(XL, 0.502, 400)
for dh in dels:
    d = dh*h; mn = []
    for xi in xs:
        yy = np.linspace(-0.08, 0.06, 900)
        s = sdf(np.column_stack([np.full_like(yy, xi), yy]))
        mn.append((EPS + (1-EPS)/(1+np.exp(-2*s/d))).min())
    ax2.semilogy(xs, mn, lw=1.9, label=f"δ = {dh}h")
ax2.axhline(EPS, color="k", ls=":", lw=1)
ax2.text(XL+.004, EPS*1.5, "ε — fully solid", fontsize=8.5, color="k")
ax2.axhline(0.01, color="#A5433A", ls="--", lw=1)
ax2.text(XL+.004, .013, "1% open", fontsize=8.5, color="#A5433A")
ax2.set_xlim(XL, 0.505); ax2.set_ylim(EPS*.5, 1)
ax2.set_xlabel("x/c  (body-centred)", fontsize=9.5)
ax2.set_ylabel("min φ through the section", fontsize=9.5)
ax2.grid(alpha=.25); ax2.legend(fontsize=9, loc="lower right")
ax2.set_title("how solid the section actually gets, approaching the trailing edge", fontsize=10.5)
ax2.tick_params(labelsize=8.5)

fig.suptitle(f"RAE 2822 trailing edge — Brinkman volume fraction, h = {h:.5f} c (nlvls 6)", fontsize=12.5)
plt.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
for dh in dels:
    d = dh*h
    for xi in (0.40, 0.46, 0.49, 0.50):
        yy = np.linspace(-0.08, 0.06, 1200)
        s = sdf(np.column_stack([np.full_like(yy, xi), yy]))
        print(f"  delta={dh:5}h  x/c={xi:+.3f}  min phi = {(EPS+(1-EPS)/(1+np.exp(-2*s/d))).min():.3e}")
