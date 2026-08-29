#!/usr/bin/env python3
"""Plot the RAE 2822 surface pressure (--case 15) against the section geometry.

Reads output/rae2822_surface.dat (written by CompressibleSolver::writeIbSurface)
and geom/rae2822.dat.  Cp is plotted with the axis inverted, the usual
convention, so suction is up.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

surf = sys.argv[1] if len(sys.argv) > 1 else "output/rae2822_surface.dat"
geom = sys.argv[2] if len(sys.argv) > 2 else "geom/rae2822.dat"
out  = sys.argv[3] if len(sys.argv) > 3 else "output/rae2822_cp.png"

d = np.loadtxt(surf)
xc, yn, cp = d[:, 0], d[:, 1], d[:, 2]
ok = np.isfinite(cp) & (xc > -0.05) & (xc < 1.05)
side = d[:, 5] if d.shape[1] > 5 else np.where(yn >= 0, 1, -1)
up, lo = ok & (side > 0), ok & (side < 0)
g = np.loadtxt(geom)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
a1.plot(xc[up], cp[up], "o", ms=3.5, color="#c0392b", label="upper surface")
a1.plot(xc[lo], cp[lo], "o", ms=3.5, color="#2471a3", label="lower surface")
a1.axhline(0.0, lw=0.6, color="0.7")
a1.invert_yaxis()
a1.set_ylabel(r"$C_p$")
a1.legend(frameon=False)
a1.grid(alpha=0.25)
a1.set_title("RAE 2822, immersed level-set body, M = 0.2, "
             r"$\alpha = 2.31^\circ$ (slip wall)")

a2.plot(np.append(g[:, 0], g[0, 0]), np.append(g[:, 1], g[0, 1]), "-", lw=1.2, color="0.25")
a2.set_aspect("equal")
a2.set_xlabel("x/c")
a2.set_ylabel("y/c")
a2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
print("  points: %d upper, %d lower" % (up.sum(), lo.sum()))
print("  Cp max = %+.3f (stagnation, ideal +1)   Cp min = %+.3f (suction peak)"
      % (np.nanmax(cp[ok]), np.nanmin(cp[ok])))
