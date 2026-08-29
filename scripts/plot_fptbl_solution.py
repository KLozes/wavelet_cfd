#!/usr/bin/env python3
"""Flat-plate turbulent boundary layer: field, wall-unit profile and skin friction.

   usage: python3 scripts/plot_fptbl_solution.py [outdir] [plateX0]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

out = sys.argv[1] if len(sys.argv) > 1 else "output"
x0  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
# Immersed case (testCase 14): the wall is a level set at y = wallY, not the
# domain bottom.  The solver writes the profile against the RAW y coordinate, so
# without this every y+ is offset by wallY*uTau/nu (842 wall units at the FPTBL
# conditions) and the cells INSIDE the body -- which still hold their initial
# freestream -- masquerade as a near-wall profile sitting flat at u+ = u_inf/uTau.
wallY = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
KAPPA, BSTAR = 0.41, 0.09

# ---- the wall function, Eq. (35) with the exact residues (see KtauSst.h) -----
def uplus(yp):
    y = np.maximum(yp, 0.0); s = np.sqrt(y)
    return (-6.6501958897564286*np.arctan(-0.53910202540295604*s + 1.2396760212086757)
            -4.0953337526867921*np.arctan( 0.33111991742187391*s + 0.2888463815305457)
            -0.18730207210813268*np.log(y - 4.5990404887908563*s + 8.7285826072687183)
            +4.0077031531887295 *np.log(y + 1.7446632856128177*s + 9.8816880908378995)
            -2.7627533816733845 *np.log(s + 2.8543772031780388)
            +1.2067490125650551)

fld  = np.loadtxt(os.path.join(out, "fptbl_field.dat"))
prof = np.loadtxt(os.path.join(out, "fptbl_prof.dat"))
with open(os.path.join(out, "fptbl_prof.dat")) as f:
    hdr = f.readline()
uTau = float(hdr.split("uTau=")[1].split()[0])
nu   = float(hdr.split("nu=")[1].split()[0])

x, y, u = fld[:, 0], fld[:, 1], fld[:, 2]
if wallY != 0.0: y = y - wallY
muT     = fld[:, 6]
o = np.lexsort((x, y)); x, y, u, muT = x[o], y[o], u[o], muT[o]
xs, ys = np.unique(x), np.unique(y)
U   = u.reshape(len(ys), len(xs))
MUT = muT.reshape(len(ys), len(xs))

d, yp, uu, up, kk, tt, mt = (prof[:, i] for i in range(7))
if wallY != 0.0:
    d = d - wallY
    # The solver's profile writer walks the RAW y grid, so rows at and below the
    # wall are masked cells still holding the untouched initial state (k = kInf,
    # tau = tauInf).  A d > 0 test is not enough: the at-wall row survives on a
    # float residual.  Cut on the physical signature instead -- frozen cells sit
    # at kInf, three decades below the turbulent band -- and keep everything from
    # the first genuinely turbulent sample upward.
    o0 = np.argsort(d)
    kTurb = kk[o0] > 10.0*np.nanmin(kk)
    iFirst = int(np.argmax(kTurb)) if kTurb.any() else 0
    sel = o0[iFirst:]
    d, uu, kk, tt, mt = d[sel], uu[sel], kk[sel], tt[sel], mt[sel]
    yp = d*uTau/nu                           # recompute: the solver used raw y
    up = uu/uTau
s = np.argsort(d); d, yp, uu, up, kk, mt = d[s], yp[s], uu[s], up[s], kk[s], mt[s]
# delta99 at this station, so the profile panels show the LAYER, not the freestream
uinfP = np.nanmax(uu)
iEdge = int(np.argmax(uu >= 0.99*uinfP))
ypEdge = yp[iEdge] if iEdge > 0 else yp.max()
keep = yp <= 2.0*ypEdge

fig = plt.figure(figsize=(11.5, 6.9))
gs  = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.25],
                       left=0.06, right=0.95, top=0.90, bottom=0.08,
                       hspace=0.45, wspace=0.28)

# ---- streamwise velocity, y stretched so the layer is visible ---------------
ax = fig.add_subplot(gs[0, :])
pm = ax.pcolormesh(xs, ys, U, cmap="viridis", norm=Normalize(0, np.nanmax(U)), shading="auto")
ax.axhline(0, color="k", lw=2)
ax.plot([x0, xs.max()], [0, 0], color="crimson", lw=3, solid_capstyle="butt",
        label="modelled wall")
# boundary-layer edge: where u first reaches 0.99 u_inf
uinf = np.nanmax(U)
edge = np.array([ys[np.argmax(U[:, i] >= 0.99*uinf)] for i in range(len(xs))])
ax.plot(xs, edge, "w--", lw=1.4, label=r"$\delta_{99}$")
ax.axvline(x0 + 0.97, color="w", ls=":", lw=1.2)
ax.text(x0 + 0.97, ys.max()*0.86, " x/L=0.97", color="w", fontsize=8)
ax.set_xlabel("x"); ax.set_ylabel("y")
# the layer is ~1% of the box height, so zoom to a few delta or it is invisible
# only the developed part of the layer sets the zoom: upstream of the leading
# edge (and, for the immersed case, below the wall) "edge" is meaningless
eOK = edge[(xs >= x0) & (edge > 0)]
eTop = 4.0*np.nanmax(eOK) if eOK.size else ys.max()
eBot = 6.0*np.nanmin(eOK) if eOK.size else ys.max()
ax.set_ylim(0, min(ys.max(), max(eTop, eBot)))
# ---- mesh overlay -----------------------------------------------------------
# Horizontal lines are TRUE cell faces: the layer is resolved across ~10 of them
# and, for the immersed case, they show which rows the UTCart mask removes.  The
# x faces are ~1300 cells across the box, so they are drawn on a stride or the
# panel goes solid black -- the stride is stated in the legend, not implied.
hy = float(np.min(np.diff(ys))) if len(ys) > 1 else 0.0
hx = float(np.min(np.diff(xs))) if len(xs) > 1 else 0.0
yTop = ax.get_ylim()[1]
if hy > 0:
    yf = np.arange(np.floor(ys.min()/hy)*hy - 0.5*hy, yTop + hy, hy)
    for v in yf[(yf >= ax.get_ylim()[0]) & (yf <= yTop)]:
        ax.axhline(v, color="k", lw=0.35, alpha=0.30, zorder=2)
xStride = max(1, int(round(len(xs)/60.0)))
if hx > 0:
    for v in xs[::xStride] - 0.5*hx:
        ax.axvline(v, color="k", lw=0.35, alpha=0.22, zorder=2)
ax.plot([], [], color="k", lw=0.35, alpha=0.45,
        label=f"cells: every y face, every {xStride}th x face")
if wallY != 0.0:
    # the masked band: wall up to the first fluid cell face (UTCart marks every
    # intersecting cell non-fluid, so this whole strip is removed from the solve)
    ax.axhspan(0.0, hy, color="crimson", alpha=0.13, zorder=1,
               label="masked band (intersecting cells)")
ax.set_title("streamwise velocity — the layer grows from the leading edge", fontsize=10)
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.80)
fig.colorbar(pm, ax=ax, pad=0.01, label="u")

# ---- u+ vs y+ ---------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
yy = np.logspace(-1, np.log10(max(3e3, yp.max())), 400)
ax.semilogx(yy, uplus(yy), "k-", lw=1.4, label="wall function, Eq. (35)")
ax.semilogx(yy, np.log(yy)/KAPPA + 5.2199, "k:", lw=1.0, label=r"log law $\ln y^+/\kappa+5.22$")
m = (yp > 0) & keep
ax.semilogx(yp[m], up[m], "o", ms=4.5, color="crimson", label="wave3d_dp (first cell = wall model)")
ax.axvspan(1, yp[m].min(), color="0.85", zorder=0)
ax.text(1.3, 1.0, "modelled\n(not resolved)", fontsize=7, color="0.35")
ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$u^+$")
ax.set_xlim(1, 2.2*ypEdge); ax.set_ylim(0, max(28, np.nanmax(up[m])*1.12))
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"velocity profile at x/L = 0.97  ($u_\\tau$ = {uTau:.5f})", fontsize=10)

# ---- turbulence profiles ----------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ax.semilogx(yp[m], kk[m]/uTau**2, "o-", ms=4, lw=1.2, color="tab:blue",
            label=r"$\tilde k/u_\tau^2$")
ax.axhline(1/np.sqrt(BSTAR), color="tab:blue", ls="--", lw=1,
           label=r"log-layer $1/\sqrt{\beta^*}$")
ax2 = ax.twinx()
ax2.semilogx(yp[m], mt[m]/nu, "s-", ms=4, lw=1.2, color="tab:orange",
             label=r"$\mu_t/\mu$")
ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$\tilde k/u_\tau^2$", color="tab:blue")
ax2.set_ylabel(r"$\mu_t/\mu$", color="tab:orange")
ax.set_xlim(1, 2.2*ypEdge); ax.grid(alpha=0.3)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
ax.set_title("turbulence profiles at the same station", fontsize=10)

fig.suptitle(r"Flat-plate turbulent boundary layer, $Re_L=5\times10^6$, $M=0.2$ — "
             r"$\tilde k$–$\tilde\tau$ SST with the algebraic wall model", fontsize=12, y=0.975)
png = os.path.join(out, "fptbl_solution.png")
fig.savefig(png, dpi=140)
# constrained_layout reserves room the wide field panel does not use; crop it
try:
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bb = ImageChops.difference(im, bg).getbbox()
    if bb:
        pad = 12
        im.crop((max(bb[0]-pad,0), max(bb[1]-pad,0),
                 min(bb[2]+pad, im.width), min(bb[3]+pad, im.height))).save(png)
except Exception:
    pass
print("wrote", os.path.join(out, "fptbl_solution.png"))
print(f"u_tau = {uTau:.6f}   Cf = {2*uTau**2:.6f}   "
      f"first cell y+ = {yp[yp>0].min():.0f}   delta99+ = {ypEdge:.0f}")
