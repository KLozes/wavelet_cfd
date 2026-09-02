#!/usr/bin/env python3
"""Is the residual ripple pattern reflecting off the AMR coarse/fine interfaces?

  usage: plot_residual_amr.py field.dat grid.dat cx cy chord out.png "title"

Overlays the level boundaries from a writeGridBlocks dump on the per-cell
|dq/dt| map, and reports the ripple wavelength in units of the LOCAL cell size.
Two things settle the question:
  * do the ripple crests line up with the interfaces (reflection), or ignore
    them (a wave radiating from the wall that merely crosses them)?
  * is the wavelength ~2h (a grid-scale / odd-even mode, i.e. a discrete
    artifact) or many h (a resolved acoustic wave)?
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.colors import LogNorm

fld, grd = sys.argv[1], sys.argv[2]
cx, cy, chord = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
out = sys.argv[6]
title = sys.argv[7] if len(sys.argv) > 7 else "residual vs AMR interfaces"
AOA = 2.31

g = np.loadtxt("geom/rae2822.dat")
ca, sa = np.cos(np.radians(AOA)), np.sin(np.radians(AOA))
gx, gy = g[:, 0]*ca + g[:, 1]*sa, -g[:, 0]*sa + g[:, 1]*ca
gx -= 0.5*(gx.min() + gx.max()); gy -= 0.5*(gy.min() + gy.max())
poly = Path(np.column_stack([np.append(gx, gx[0]), np.append(gy, gy[0])]))

d = np.loadtxt(fld)
d = d[d[:, 9] > 0]                              # live DOFs only

# grid blocks -> body-centred chord units
gb = np.loadtxt(grd)
bx = (gb[:, 0] - cx)/chord
by = (gb[:, 1] - cy)/chord
bs = gb[:, 2]/chord
blvl = gb[:, 3].astype(int)
LMAX = blvl.max()

XL, XR, YB, YT = -0.85, 1.00, -0.45, 0.45
pos = d[:, 9][d[:, 9] > 0]
norm = LogNorm(vmin=max(pos.min(), pos.max()*1e-4), vmax=pos.max())

fig, ax = plt.subplots(1, 2, figsize=(14.6, 4.4))
for a, overlay in zip(ax, (False, True)):
    k = (d[:, 0] > XL-.1) & (d[:, 0] < XR+.1) & (d[:, 1] > YB-.1) & (d[:, 1] < YT+.1)
    w = d[k]
    tri = mtri.Triangulation(w[:, 0], w[:, 1])
    cen = np.column_stack([w[:, 0][tri.triangles].mean(1), w[:, 1][tri.triangles].mean(1)])
    tri.set_mask(poly.contains_points(cen))
    f = a.tripcolor(tri, np.clip(w[:, 9], norm.vmin, norm.vmax),
                    norm=norm, cmap="magma", shading="gouraud")
    if overlay:
        # one cyan outline per block, per level: interfaces are where the block
        # size changes.  Cyan reads clearly on magma at every level of the ramp.
        m = (bx + bs > XL) & (bx < XR) & (by + bs > YB) & (by < YT)
        for x0, y0, s in zip(bx[m], by[m], bs[m]):
            a.add_patch(plt.Rectangle((x0, y0), s, s, fill=False,
                                      ec="#22d3ee", lw=0.45, alpha=0.85, zorder=4))
        a.set_title("with AMR block / level boundaries", fontsize=10)
    else:
        a.set_title("residual only", fontsize=10)
    cb = plt.colorbar(f, ax=a, fraction=0.040, pad=0.02)
    cb.set_label(r"$|\Delta q/\Delta t|$", fontsize=8); cb.ax.tick_params(labelsize=8)
    a.fill(gx, gy, color="0.75", zorder=5)
    a.plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "k-", lw=0.9, zorder=6)
    a.set_xlim(XL, XR); a.set_ylim(YB, YT); a.set_aspect("equal")
    a.set_xlabel("x/c", fontsize=9); a.set_ylabel("y/c", fontsize=9)
    a.tick_params(labelsize=8)

fig.suptitle(title, fontsize=11.5)
plt.tight_layout(); plt.savefig(out, dpi=150)
print("wrote", out)

# --- ripple wavelength along a horizontal cut ahead of the LE ---------------
for ycut in (0.12, -0.12):
    sel = np.abs(d[:, 1] - ycut) < 0.008
    if sel.sum() < 40:
        continue
    x = d[sel, 0]; r = np.log10(d[sel, 9]); o = np.argsort(x)
    x, r = x[o], r[o]
    xs = np.linspace(x.min(), x.max(), 800)
    rs = np.interp(xs, x, r); rs -= rs.mean()
    fft = np.abs(np.fft.rfft(rs*np.hanning(len(rs))))
    fr = np.fft.rfftfreq(len(xs), xs[1]-xs[0])
    kpk = fr[1:][np.argmax(fft[1:])]
    lam = 1.0/kpk                                   # chords
    print(f"  y/c={ycut:+.2f}: dominant ripple wavelength {lam:.4f} c "
          f"= {lam/0.0078:.1f} finest cells (h/c=0.0078)")
