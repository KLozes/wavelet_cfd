#!/usr/bin/env python3
"""
Fitted (degree-q) level set vs the TRUE level set, on a constant-z slice.

phi_fit is the per-cell degree-q polynomial fitted to the (q+1)^3 GLL samples of
the true SDF -- i.e. exactly the geometry a Q_q solver "sees".  Contouring both
at 0 shows what the fitted geometry keeps and what it loses.  Written for the
ROTOR 1 root fillet, where R_fillet is sub-cell at practical resolutions.

  usage: plot_fitslice.py output/<tag>_fitslice.txt [out.png]
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fn = sys.argv[1] if len(sys.argv) > 1 else "output/bank_v98d_ROTOR_1_fitslice.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "output/fitslice.png"

hdr = {}
with open(fn) as f:
    line = f.readline()
    for tok in line.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            hdr[k] = v
z    = float(hdr.get("z", 0.0))
res  = int(hdr.get("res", 0))
h    = float(hdr.get("h", 0.0))
deg  = int(hdr.get("deg", 2))

d = np.loadtxt(fn, comments="#")
x, y, pt, pf = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
n = int(round(np.sqrt(len(x))))
X  = x.reshape(n, n); Y = y.reshape(n, n)
PT = pt.reshape(n, n); PF = pf.reshape(n, n)

# hub circle (platform top) -- the fillet lives where the blade meets it
R_HUB, R_FIL = 3.6374, 0.05
R = np.hypot(X, Y)

# junction centroid (for the mid zoom)
band = (np.abs(PT) < 0.5*h) & (np.abs(R - R_HUB) < 6*h)
xc, yc = (X[band].mean(), Y[band].mean()) if band.sum() else (X.mean(), Y.mean())
# an ACTUAL point on the fillet arc, for the tight close-up: on the true surface
# and within ~1.5 R of the platform-top circle.  Take the one with the most
# negative y so we sit on a single corner rather than straddling two.
# the blade/platform fillet: the CONCAVE corner where the blade surface meets the
# platform-top circle.  Override with argv[3],argv[4] if the auto-pick misses.
fil = (np.abs(PT) < 0.75*(X[0,1]-X[0,0])) & (np.abs(R - R_HUB) < 3*R_FIL) & (X > R_HUB - 0.05)
if fil.sum() > 0:
    idx = np.argmin(np.hypot(X[fil]-3.62, Y[fil]+0.50)); xf, yf = X[fil][idx], Y[fil][idx]
else:
    xf, yf = xc, yc
if len(sys.argv) > 4: xf, yf = float(sys.argv[3]), float(sys.argv[4])

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))

def draw(a, half, title, show_grid, cx=None, cy=None):
    cx = xc if cx is None else cx; cy = yc if cy is None else cy
    a.contour(X, Y, PT, [0.0], colors="k",  linewidths=2.0)
    a.contour(X, Y, PF, [0.0], colors="C3", linewidths=1.4, linestyles="--")
    th = np.linspace(0, 2*np.pi, 2000)
    a.plot(R_HUB*np.cos(th), R_HUB*np.sin(th), color="C0", lw=0.8, alpha=0.55)
    if show_grid:
        lo0 = float(hdr["lo"]) if "lo" in hdr else X.min()
        g0 = np.floor((cx-half - X.min())/h)*h + X.min()
        for gx in np.arange(g0, cx+half+h, h): a.axvline(gx, color="0.8", lw=0.7, zorder=0)
        g0 = np.floor((cy-half - Y.min())/h)*h + Y.min()
        for gy in np.arange(g0, cy+half+h, h): a.axhline(gy, color="0.8", lw=0.7, zorder=0)
    a.set_xlim(cx-half, cx+half); a.set_ylim(cy-half, cy+half)
    a.set_aspect("equal"); a.set_title(title, fontsize=10)
    a.set_xlabel("x"); a.set_ylabel("y")

draw(ax[0], 1.2,  f"slice z = {z:.4f}   (overview)", False)
draw(ax[1], 6*h,  f"junction zoom   h = {h:.4f}  (12 cells across)", True)
draw(ax[2], 2.4*R_FIL, f"FILLET close-up  (grey = cell edges)\nR_fil = {R_FIL} = {R_FIL/h:.2f} cells",
     True, xf, yf)
# fillet-band deviation, quoted on the panel
_ds = X[0,1]-X[0,0]
_on = (np.abs(PT) < 0.75*_ds) & (np.abs(R-R_HUB) < 2*R_FIL)
if _on.sum():
    _d = np.abs(PT[_on] - PF[_on])   # separation between the zero sets
    ax[2].text(0.03, 0.03,
        f"fillet-band separation $|\\phi-\\phi_h|$\nmax {_d.max():.2e} = {_d.max()/h:.2f} h = {100*_d.max()/R_FIL:.0f}% of $R_{{fil}}$"
        f"\nmean {_d.mean():.2e} = {_d.mean()/h:.3f} h",
        transform=ax[2].transAxes, fontsize=8.5, va="bottom",
        bbox=dict(fc="white", ec="0.7", alpha=0.9))

from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color="k",  lw=2.0, label="true level set  $\\phi=0$"),
                    Line2D([], [], color="C3", lw=1.4, ls="--",
                           label=f"fitted degree-{deg} level set  $\\phi_h=0$"),
                    Line2D([], [], color="C0", lw=0.8, label="hub circle (platform top)")],
           loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle(f"ROTOR 1 root fillet: true vs fitted degree-{deg} zero contour  "
             f"(res {res}, h = {h:.4f}, fillet R = {R_FIL} = {R_FIL/h:.2f} cells)",
             fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(out, dpi=155)
print("wrote", out)

# quantitative: how far apart are the two zero sets near the junction?
sel = (np.abs(X-xc) < 3*h) & (np.abs(Y-yc) < 3*h) & (np.abs(PT) < 0.75*(X[0,1]-X[0,0]))
if sel.sum() > 0:
    dev = np.abs(PT[sel] - PF[sel])
    print(f"near junction (+/-3h): max|phi_true - phi_fit| = {dev.max():.4e} "
          f"= {dev.max()/h:.3f} h ;  mean = {dev.mean():.4e}")
