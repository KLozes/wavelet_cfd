#!/usr/bin/env python3
"""Skin friction along the flat plate against the TMR reference value.

   usage: python3 scripts/plot_fptbl.py [output/fptbl_cf.dat] [plateX0]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "output/fptbl_cf.dat"
x0   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
d    = np.loadtxt(path)
s, cf = d[:, 0] - x0, d[:, 1]          # distance from the leading edge
o = np.argsort(s); s, cf = s[o], cf[o]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(s, cf, lw=1.4, label="wave3d_dp  k~-tau~ SST + wall model")
ax.axhline(0.0027, ls="--", c="k", lw=1.0, label="TMR reference at x/L = 0.97")
ax.plot([0.97], [0.0027], "ko", ms=5)
i = int(np.argmin(np.abs(s - 0.97)))
ax.plot([s[i]], [cf[i]], "ro", ms=5,
        label=f"ours at x/L=0.97: {cf[i]:.5f} ({100*(cf[i]-0.0027)/0.0027:+.1f}%)")
ax.set_xlabel("distance from the leading edge, x/L")
ax.set_ylabel(r"$C_f$")
ax.set_ylim(0, max(0.006, 1.2*np.nanmax(cf[s > 0.05])))
ax.set_xlim(0, s.max())
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("output/fptbl_cf.png", dpi=140)
print(f"Cf at x/L=0.97 = {cf[i]:.6f}   ({100*(cf[i]-0.0027)/0.0027:+.1f}% vs TMR 0.0027)")
print("wrote output/fptbl_cf.png")
