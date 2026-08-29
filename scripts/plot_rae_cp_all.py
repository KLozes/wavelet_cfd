#!/usr/bin/env python3
"""RAE 2822 case 9 surface Cp: inviscid vs wall-modelled RANS variants.

   usage: plot_rae_cp_all.py out.png surf.dat label [surf.dat label ...]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
args = sys.argv[2:]
g, M = 1.4, 0.729
cpStar = 2/(g*M*M)*(((2 + (g-1)*M*M)/(g+1))**(g/(g-1)) - 1)

fig, ax = plt.subplots(figsize=(8.0, 5.2))
cols = ["#111111", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
for k in range(0, len(args), 2):
    d = np.loadtxt(args[k]); lab = args[k+1]; c = cols[(k//2) % len(cols)]
    x, cp, side = d[:, 0], d[:, 2], d[:, 5]
    gg = np.linspace(0, 1, 300)
    cu = cl = None
    for s, ls, mk in ((1, "-", "o"), (-1, "--", None)):
        m = side == s
        o = np.argsort(x[m])
        ax.plot(x[m][o], cp[m][o], ls, lw=1.5, color=c, ms=2.5,
                marker=mk, label=f"{lab} ({'upper' if s > 0 else 'lower'})")
        if s > 0: cu = np.interp(gg, x[m][o], cp[m][o])
        else:     cl = np.interp(gg, x[m][o], cp[m][o])
    Cl = np.trapezoid(cl - cu, gg)
    ax.plot([], [], " ", label=f"     $C_l$ = {Cl:.3f}")

ax.axhline(cpStar, color="gray", ls=":", lw=1.2)
ax.text(0.985, cpStar, f"sonic $C_p^*$ = {cpStar:.2f} ", va="bottom", ha="right",
        fontsize=8, color="gray")
ax.axvline(0.55, color="gray", ls=":", lw=1.0)
ax.annotate("experiment shock ~0.55c", xy=(0.55, -1.30), xytext=(0.66, -1.42),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.invert_yaxis()
ax.set_xlabel("x/c"); ax.set_ylabel("$C_p$"); ax.set_xlim(-0.02, 1.02)
ax.set_ylim(1.25, -1.55)
ax.set_title("RAE 2822 case 9 (M=0.729, $\\alpha$=2.31$^\\circ$, Re=6.5e6), nLvls 7\n"
             "experiment: $C_l$ = 0.803, shock ~0.55c", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=2, loc="lower center", framealpha=0.92)
fig.tight_layout(); fig.savefig(out, dpi=140)
print("wrote", out)
