#!/usr/bin/env python3
"""RAE 2822 case 9 surface Cp: our immersed wall-modeled RANS vs resolution.

   usage: python3 scripts/plot_rae_cp_compare.py out.png surf1.dat label1 [surf2.dat label2 ...]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
args = sys.argv[2:]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for k in range(0, len(args), 2):
    d = np.loadtxt(args[k]); lab = args[k+1]
    x, cp, side = d[:, 0], d[:, 2], d[:, 5]
    for s, ls in ((1, "-"), (-1, "--")):
        m = side == s
        o = np.argsort(x[m])
        ax.plot(x[m][o], cp[m][o], ls, lw=1.3, color=cols[(k//2) % 4],
                label=f"{lab} {'upper' if s > 0 else 'lower'}")
# sonic line for M_inf = 0.729
g, M = 1.4, 0.729
cpStar = 2/(g*M*M)*(((2 + (g-1)*M*M)/(g+1))**(g/(g-1)) - 1)
ax.axhline(cpStar, color="k", ls=":", lw=1.0, label=f"sonic $C_p^*$ = {cpStar:.2f}")
ax.axvline(0.55, color="gray", ls=":", lw=1.0)
ax.text(0.56, ax.get_ylim()[0]+0.1, "experiment shock ~0.55c", fontsize=7, color="gray")
ax.invert_yaxis()
ax.set_xlabel("x/c"); ax.set_ylabel("$C_p$")
ax.set_title("RAE 2822 case 9 (M=0.729, $\\alpha$=2.31$^\\circ$, Re=6.5e6)\nimmersed wall-modeled RANS (SA, --ibwm 4)")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=140)
print("wrote", out)
