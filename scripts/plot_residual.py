#!/usr/bin/env python3
"""Steady-state residual history from a wave3d --residevery run.

  usage: plot_residual.py out.png run1.log "label 1" [run2.log "label 2" ...]

Parses the solver's [resid] lines:
  [resid] iter  N  R = ...  R/R0 = ...  R(>Hh from wall) = ...  max = ... at ... h

R is ||dq/dt|| over ALL live fluid cells (the convergence criterion); the
R(>Hh) column is the same norm with cells near the immersed body excluded, and
is a DIAGNOSTIC for localising a stall -- plotted dashed, never as the headline.
"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"\[resid\]\s+iter\s+(\d+)\s+R = ([0-9.eE+-]+)\s+R/R0 = ([0-9.eE+-]+)"
    r"\s+R\(>([0-9.]+)h from wall\) = ([0-9.eE+-]+)\s+max = ([0-9.eE+-]+)")

# validated with dataviz/scripts/validate_palette.js (light, categorical):
# adjacent CVD dE 21.1 protan / 33.8 tritan, normal 31.7 -- all checks PASS.
COLS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]


def read(path):
    it, r, rf, mx, h = [], [], [], [], None
    for ln in open(path):
        m = LINE.search(ln)
        if not m:
            continue
        it.append(int(m.group(1))); r.append(float(m.group(2)))
        h = m.group(4);             rf.append(float(m.group(5)))
        mx.append(float(m.group(6)))
    return np.array(it), np.array(r), np.array(rf), np.array(mx), h


out, args = sys.argv[1], sys.argv[2:]
fig, ax = plt.subplots(figsize=(9.0, 5.4))
band, hh = [], None

for k in range(0, len(args), 2):
    it, r, rf, mx, h = read(args[k])
    lab, c = args[k + 1], COLS[(k // 2) % len(COLS)]
    hh = h or hh
    good = it > 0                                  # iter 0 has no dq yet
    ax.plot(it[good], r[good], "-", lw=2.0, color=c, label=f"{lab} — all fluid cells")
    ax.plot(it[good], rf[good], "--", lw=1.4, color=c, alpha=0.75,
            label=f"{lab} — excluding <{h}h from wall")
    # direct label at the right end (<=4 series: label them, don't rely on colour)
    ax.annotate(lab, xy=(it[good][-1], r[good][-1]), xytext=(6, 0),
                textcoords="offset points", color=c, fontsize=9,
                fontweight="bold", va="center")
    tail = r[it > it.max() * 0.6]
    if len(tail):
        band.append((lab, c, tail.min(), tail.max()))

# the plateau each configuration settles into
msg = "   ".join(f"{l}: {lo:.1e}–{hi:.1e}" for l, _, lo, hi in band)
ax.text(0.015, 0.03, "non-decaying band (last 40% of the march)\n  " + msg,
        transform=ax.transAxes, fontsize=8.5, family="monospace", va="bottom",
        bbox=dict(fc="white", ec="0.75", alpha=0.93, boxstyle="round,pad=0.4"))

ax.set_yscale("log")
ax.set_xlim(right=ax.get_xlim()[1]*1.10)      # headroom for the direct labels
ax.set_xlabel("iteration")
ax.set_ylabel(r"residual   $\|\Delta q/\Delta t\|_{\rm RMS}$")
ax.set_title("RAE 2822 steady convergence — immersed-boundary ghosts vs ghost-free\n"
             "solid = every fluid cell (the criterion);  dashed = far field only "
             f"(>{hh}h from wall, diagnostic)", fontsize=10.5)
ax.grid(alpha=0.25, which="both", lw=0.6)          # recessive grid
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=8, loc="upper right", framealpha=0.93)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
for l, _, lo, hi in band:
    print(f"  {l:<22s} plateau {lo:.3e} .. {hi:.3e}")
