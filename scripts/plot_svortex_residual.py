#!/usr/bin/env python3
"""Where the steady residual lives on the supersonic-vortex annulus (case 16).

  usage: plot_svortex_residual.py out.png "title" fieldA.dat "label A" [fieldB.dat "label B"]

Reads writeIbField dumps (output/svortex_field_NN.dat, column 10 = per-cell
|dq/dt| from the last --residevery sample).  Three views answer three
different questions:
  1. map of |dq/dt| on the annulus (log, magma) -- band at the wall, or
     distributed?  Localised to particular cut cells, or azimuthally uniform?
  2. RMS residual vs distance from the NEAREST wall in cells -- how wide is
     the wall band and how much does it stick up above the interior floor?
  3. RMS residual vs azimuth -- a grid-aligned cut (theta = 0, 90, ...) vs a
     diagonal one (45, 135, ...) tells you whether the cut geometry drives it;
     a smooth low-mode pattern that DIFFERS between two dumps is a travelling
     wave, not a fixed wall defect.
With two files the map shows both on a SHARED colour scale and the profiles
overlay.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out, title, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
pairs = [(rest[i + 1], rest[i]) for i in range(0, len(rest), 2)]
RI, RO = 1.0, 1.384

data = []
for lab, f in pairs:
    d = np.loadtxt(f)
    if d.shape[1] < 10:
        sys.exit(f"{f}: no residual column -- rerun with --residevery N")
    d = d[d[:, 9] > 0]                       # live DOFs only (R-Cells included)
    data.append((lab, d))

# cell size from the dump: smallest x spacing
xs = np.unique(np.round(data[0][1][:, 0], 8)); h = np.min(np.diff(xs))
allv = np.concatenate([d[:, 9] for _, d in data])
norm = LogNorm(vmin=max(allv.min(), allv.max()*1e-4), vmax=allv.max())

n = len(data)
fig = plt.figure(figsize=(6.2*n + 6.4, 5.6))
gs = fig.add_gridspec(2, n + 1, width_ratios=[1]*n + [1.05])
for i, (lab, d) in enumerate(data):
    a = fig.add_subplot(gs[:, i])
    f = a.scatter(d[:, 0], d[:, 1], c=np.clip(d[:, 9], norm.vmin, norm.vmax), s=(h*115)**2*1.05,
                  marker="s", norm=norm, cmap="magma", linewidths=0)
    th = np.linspace(0, 2*np.pi, 400)
    for r in (RI, RO):
        a.plot(r*np.cos(th), r*np.sin(th), "-", color="#22d3ee", lw=0.7, alpha=0.9)
    a.set_aspect("equal"); a.set_xlim(-RO*1.03, RO*1.03); a.set_ylim(-RO*1.03, RO*1.03)
    r = d[:, 9]
    a.set_title(f"{lab}\nRMS {np.sqrt((r**2).mean()):.2e}   max {r.max():.2e}", fontsize=10)
    a.set_xlabel("x", fontsize=9); a.set_ylabel("y", fontsize=9); a.tick_params(labelsize=8)
    cb = plt.colorbar(f, ax=a, fraction=0.046, pad=0.02)
    cb.set_label(r"per-cell $|\Delta q/\Delta t|$", fontsize=8); cb.ax.tick_params(labelsize=8)

# --- profiles ---------------------------------------------------------------
ar = fig.add_subplot(gs[0, n]); at = fig.add_subplot(gs[1, n])
cols = ["#e45756", "#4c78a8", "#54a24b"]
for (lab, d), c in zip(data, cols):
    rr = np.hypot(d[:, 0], d[:, 1]); res = d[:, 9]
    # signed distance to the nearest wall, in cells (negative = centre in solid = R-Cell)
    dw = np.minimum(rr - RI, RO - rr)/h
    bins = np.arange(-1.0, np.ceil(dw.max()) + 1, 1.0)
    idx = np.digitize(dw, bins)
    prof = np.array([np.sqrt((res[idx == k]**2).mean()) if (idx == k).any() else np.nan
                     for k in range(1, len(bins))])
    ar.semilogy(0.5*(bins[1:] + bins[:-1]), prof, "o-", ms=3, lw=1.1, color=c, label=lab)
    # which wall?  split the first 3 cells inner vs outer
    inner = (rr - RI < 3*h); outer = (RO - rr < 3*h)
    print(f"  {lab:<28s} RMS all {np.sqrt((res**2).mean()):.3e}   inner-wall band(3h) "
          f"{np.sqrt((res[inner]**2).mean()):.3e}   outer-wall band(3h) "
          f"{np.sqrt((res[outer]**2).mean()):.3e}   interior(>4h) "
          f"{np.sqrt((res[dw > 4]**2).mean()):.3e}")
    # azimuthal profile, 72 bins of 5 degrees, interior and wall band separately
    thd = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    tb = np.arange(0, 361, 5); ti = np.digitize(thd, tb)
    for sel, ls, tag in ((dw > 4, "-", "interior"), (dw <= 1.5, "--", "wall band")):
        pa = np.array([np.sqrt((res[sel & (ti == k)]**2).mean()) if (sel & (ti == k)).any()
                       else np.nan for k in range(1, len(tb))])
        at.semilogy(0.5*(tb[1:] + tb[:-1]), pa, ls, lw=1.1, color=c, label=f"{lab} {tag}")
    o = np.argsort(-res); k50 = np.searchsorted(np.cumsum(res[o]**2)/(res**2).sum(), 0.5) + 1
    print(f"  {'':28s} 50% of sum(r^2) in top {k50} cells ({100*k50/len(d):.1f}%);  "
          f"of those, {100*np.mean(dw[o[:k50]] <= 1.5):.0f}% are within 1.5h of a wall")
ar.set_xlabel("distance to nearest wall  [cells]  (<0 : R-Cell)", fontsize=9)
ar.set_ylabel("RMS |dq/dt|", fontsize=9); ar.grid(alpha=.3); ar.legend(fontsize=7); ar.tick_params(labelsize=8)
at.set_xlabel("azimuth  [deg]", fontsize=9); at.set_ylabel("RMS |dq/dt|", fontsize=9)
at.set_xticks(range(0, 361, 45)); at.grid(alpha=.3); at.legend(fontsize=6.5, ncol=2); at.tick_params(labelsize=8)
fig.suptitle(title, fontsize=11.5)
plt.tight_layout(); plt.savefig(out, dpi=140)
print("wrote", out)
