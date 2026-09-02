#!/usr/bin/env python3
"""RAE 2822 case 9: computed surface Cp vs the AGARD AR-138 experiment.

  usage: plot_rae_cp_exp.py out.png exp.dat surf1.dat label1 [surf2.dat label2 ...]

Experiment file : x/c  Cp  side          (geom/rae2822_case9_exp.dat)
Solver dump     : x/c  yn/c  Cp  xSurf  ySurf  side   (writeIbSurface)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, expf, args = sys.argv[1], sys.argv[2], sys.argv[3:]
G, M = 1.4, 0.729
cpStar = 2/(G*M*M)*(((2 + (G-1)*M*M)/(G+1))**(G/(G-1)) - 1)
grid = np.linspace(0.0, 1.0, 400)


def split(x, cp, side):
    """-> (x,cp) sorted, for upper and lower."""
    o = []
    for s in (1, -1):
        m = side == s
        k = np.argsort(x[m])
        o.append((x[m][k], cp[m][k]))
    return o


def cn_shock(xu, cu, xl, cl):
    """normal-force coefficient and upper-surface shock location."""
    iu = np.interp(grid, xu, cu)
    il = np.interp(grid, xl, cl)
    CN = np.trapezoid(il - iu, grid)
    w = (grid > 0.15) & (grid < 0.95)           # ignore the LE peak
    d = np.diff(iu)[w[:-1]]
    xs = grid[w][:-1][np.argmax(d)] if len(d) else np.nan
    return CN, xs


e = np.loadtxt(expf)
(exu, ecu), (exl, ecl) = split(e[:, 0], e[:, 1], e[:, 2])
eCN, eXs = cn_shock(exu, ecu, exl, ecl)

fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.plot(exu, ecu, "o", ms=4.5, mfc="none", mec="k", mew=1.0, label="experiment (upper)")
ax.plot(exl, ecl, "s", ms=4.5, mfc="none", mec="0.45", mew=1.0, label="experiment (lower)")

cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
rows = [("experiment (AGARD AR-138)", eCN, 0.803, eXs)]
for k in range(0, len(args), 2):
    d = np.loadtxt(args[k]); lab = args[k+1]; c = cols[(k//2) % len(cols)]
    (xu, cu), (xl, cl) = split(d[:, 0], d[:, 2], d[:, 5])
    ax.plot(xu, cu, "-",  lw=1.6, color=c, label=f"{lab} (upper)")
    ax.plot(xl, cl, "--", lw=1.6, color=c, label=f"{lab} (lower)")
    CN, xs = cn_shock(xu, cu, xl, cl)
    rows.append((lab, CN, None, xs))

ax.axhline(cpStar, color="gray", ls=":", lw=1.2)
ax.text(0.99, cpStar, f"sonic $C_p^*$ = {cpStar:.2f} ", va="bottom", ha="right",
        fontsize=8, color="gray")
ax.axvline(eXs, color="gray", ls=":", lw=1.0)
ax.annotate(f"experiment shock $\\approx${eXs:.2f}c", xy=(eXs, -1.32),
            xytext=(0.20, -1.32), fontsize=8, color="gray", va="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

txt = "\n".join(f"{n:<26s} $C_N$={v:+.3f}" + (f" (table {t:.3f})" if t else "")
                + (f"   shock {s:.2f}c" if s == s else "")
                for n, v, t, s in rows)
ax.text(0.015, 0.03, txt, transform=ax.transAxes, fontsize=8, family="monospace",
        va="bottom", bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.4"))

ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02); ax.set_ylim(1.35, -1.6)
ax.set_xlabel("x/c"); ax.set_ylabel("$C_p$")
ax.set_title("RAE 2822 case 9  —  M$_\\infty$=0.729, $\\alpha$=2.31$^\\circ$, Re=6.5$\\times$10$^6$\n"
             "immersed wall-modelled RANS (SA) vs Cook, McDonald & Firmin", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.93)
fig.tight_layout(); fig.savefig(out, dpi=140)
print("wrote", out)
for n, v, t, s in rows:
    print(f"  {n:<28s} CN = {v:+.4f}" + (f"   [table {t:.3f}]" if t else "")
          + (f"   shock x/c = {s:.3f}" if s == s else ""))
