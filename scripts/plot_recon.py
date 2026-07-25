#!/usr/bin/env python3
# Plot the true level set zero contour vs the per-cell degree-p polynomial
# reconstruction that the Qp cut quadrature fits, on constant-radius airfoil
# slices.  Overlaying the two zero contours shows where the smooth polynomial
# fit matches the geometry and where it oscillates -- the creases.
#
#   ./wavefem_dp --bank ... --cyl --recon 2 ...   # writes output/<tag>_recon.bin
#   python3 scripts/plot_recon.py output/<tag>_recon.bin
import sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

path = sys.argv[1] if len(sys.argv) > 1 else "output/bank_v98d_ROTOR_1_recon.bin"
f = open(path, "rb")
nsl, pr = struct.unpack("ii", f.read(8))

# ---- load slices --------------------------------------------------------------
slices = []
for sl in range(nsl):
    NA, NB = struct.unpack("ii", f.read(8))
    q1hi, q2hi, h, physR, prd = struct.unpack("ddddd", f.read(40))
    true = np.frombuffer(f.read(NA*NB*4), dtype=np.float32).reshape(NB, NA).astype(np.float64)
    recon = np.frombuffer(f.read(NA*NB*4), dtype=np.float32).reshape(NB, NA).astype(np.float64)
    T = true.T; R = recon.T                 # -> [arc, z]
    arc = np.linspace(0, q1hi, NA); z = np.linspace(0, q2hi, NB)
    slices.append(dict(T=T, R=R, arc=arc, z=z, h=h, physR=physR, q1hi=q1hi, q2hi=q2hi))

spanlab = ["30", "55", "80"] if nsl == 3 else [str(i) for i in range(nsl)]

def creased_cells(T, arc, z, h, q1hi, q2hi):
    # returns (cj, ck, kind) for creased cells, kind in {"cut","interior","exterior"}
    gA, gZ = np.gradient(T, arc, z)
    ncj = int(round(q1hi/h)); nck = int(round(q2hi/h)); out = []
    for cj in range(ncj):
        am = (arc >= cj*h) & (arc < (cj+1)*h)
        if not am.any(): continue
        for ck in range(nck):
            zm = (z >= ck*h) & (z < (ck+1)*h)
            if not zm.any(): continue
            sub = np.ix_(am, zm); Tc = T[sub]; near = np.abs(Tc) < 0.6*h
            if near.sum() < 4: continue
            ang = np.arctan2(gA[sub][near], gZ[sub][near])
            spread = np.arccos(np.clip((np.cos(ang).mean()**2 + np.sin(ang).mean()**2)**0.5, 0, 1))
            if np.degrees(spread) <= 26: continue
            kind = "cut" if (Tc.min() < 0 < Tc.max()) else ("interior" if Tc.max() <= 0 else "exterior")
            out.append((cj, ck, kind))
    return out

def draw(ax, S, xlim=None, ylim=None):
    T, R, arc, z, h = S["T"], S["R"], S["arc"], S["z"], S["h"]
    q1hi, q2hi = S["q1hi"], S["q2hi"]
    Z, A = np.meshgrid(z, arc)
    ax.contourf(Z, A, T, levels=[-1e9, 0], colors=["#cfe3f7"], alpha=0.7)
    for gc in np.arange(0, q2hi+h, h): ax.axvline(gc, color="0.85", lw=0.5, zorder=1)
    for gc in np.arange(0, q1hi+h, h): ax.axhline(gc, color="0.85", lw=0.5, zorder=1)
    kcol = {"cut": "#ff7f0e", "interior": "#f7e463", "exterior": "#d9d9d9"}
    for cj, ck, kind in creased_cells(T, arc, z, h, q1hi, q2hi):
        ax.add_patch(plt.Rectangle((ck*h, cj*h), h, h, color=kcol[kind], alpha=0.85, zorder=1.5, ec="none"))
    ax.contour(Z, A, T, levels=[0], colors="k", linewidths=2.0, zorder=4)
    ncj = int(round(q1hi/h)); nck = int(round(q2hi/h))
    for cj in range(ncj):
        ai = np.where((arc >= cj*h) & (arc <= (cj+1)*h))[0]
        if len(ai) < 2: continue
        for ck in range(nck):
            zi = np.where((z >= ck*h) & (z <= (ck+1)*h))[0]
            if len(zi) < 2: continue
            sub = R[np.ix_(ai, zi)]
            if sub.min() > 0 or sub.max() < 0: continue
            ax.contour(Z[np.ix_(ai, zi)], A[np.ix_(ai, zi)], sub, levels=[0],
                       colors="#d62728", linewidths=1.5, zorder=5)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_aspect("equal")

LEG = [Line2D([0],[0], color="k", lw=2, label="true level set  φ=0"),
       Line2D([0],[0], color="#d62728", lw=1.5, label=f"per-cell deg-{pr} reconstruction  φ=0"),
       Patch(facecolor="#ff7f0e", label="creased CUT cell (interface kink — hurts Saye)"),
       Patch(facecolor="#f7e463", label="creased INTERIOR cell (medial axis — uncut, harmless)"),
       Patch(facecolor="#cfe3f7", label="blade material (φ<0)")]

# ---- figure 1: full airfoil sections -----------------------------------------
fig, axes = plt.subplots(nsl, 1, figsize=(13, 4.1*nsl))
if nsl == 1: axes = [axes]
for sl, S in enumerate(slices):
    ax = axes[sl]; T, arc, z, h = S["T"], S["arc"], S["z"], S["h"]
    inside = T < 0
    if inside.any():
        ii, jj = np.where(inside)
        a0, a1, z0, z1 = arc[ii.min()], arc[ii.max()], z[jj.min()], z[jj.max()]
        draw(ax, S, xlim=(z0-0.06*(z1-z0)-2*h, z1+0.06*(z1-z0)+2*h),
                    ylim=(a0-0.10*(a1-a0)-2*h, a1+0.10*(a1-a0)+2*h))
    else: draw(ax, S)
    ax.set_title(f"r = {S['physR']:.3f}  ({spanlab[sl]}% span)   h = {h:.4f}", fontsize=11)
    ax.set_xlabel("z  (axial / chord)"); ax.set_ylabel("arc  (thickness)")
axes[0].legend(handles=LEG, loc="upper right", fontsize=9, framealpha=0.95)
fig.suptitle(f"Blade level set vs degree-{pr} per-cell reconstruction", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.98])
out1 = path.replace(".bin", ".png"); fig.savefig(out1, dpi=115); print("wrote", out1)

# ---- figure 2: trailing-edge zooms (the sharpest crease) ---------------------
fig2, axes2 = plt.subplots(1, nsl, figsize=(5.2*nsl, 5.0))
if nsl == 1: axes2 = [axes2]
for sl, S in enumerate(slices):
    ax = axes2[sl]; T, arc, z, h = S["T"], S["arc"], S["z"], S["h"]
    inside = T < 0
    ii, jj = np.where(inside)
    zTE = z[jj.max()]                                   # trailing edge = max chord
    aTE = arc[ii[jj.argmax()]]
    w = 3.2*h
    draw(ax, S, xlim=(zTE-2*w, zTE+0.5*w), ylim=(aTE-w, aTE+w))
    ax.set_title(f"{spanlab[sl]}% span — trailing edge", fontsize=11)
    ax.set_xlabel("z");
    if sl == 0: ax.set_ylabel("arc")
axes2[0].legend(handles=LEG[:2], loc="upper left", fontsize=8, framealpha=0.95)
fig2.suptitle(f"Trailing-edge crease: the degree-{pr} parabola (red) cannot follow the "
              f"sharp black corner — it rounds/overshoots", fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.96])
out2 = path.replace(".bin", "_te.png"); fig2.savefig(out2, dpi=125); print("wrote", out2)
