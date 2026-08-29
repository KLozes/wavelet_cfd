#!/usr/bin/env python3
"""Cp on the RAE 2822 (M=0.73, alpha=2.31) : sharp IB vs Brinkman at several delta.

Surface dump columns: x/c  yn/c  Cp  xSurf  ySurf  side   (side +1 upper, -1 lower).
Classification is by the OUTWARD NORMAL's chord-normal component, so nose and tail
points of the cambered section are labelled correctly -- do not re-derive it from y.
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
CASES = [("sharp.dat", "sharp IB", "k", 2.0, "-"),
         ("b1.5.dat",  r"Brinkman $\delta=1.5h$",  "#c44e52", 1.3, "-"),
         ("b0.75.dat", r"Brinkman $\delta=0.75h$", "#dd8452", 1.3, "-"),
         ("b0.5.dat",  r"Brinkman $\delta=0.5h$",  "#937860", 1.3, "-"),
         ("b0.35.dat", r"Brinkman $\delta=0.35h$", "#55a868", 1.3, "-"),
         ("b0.15.dat", r"Brinkman $\delta=0.15h$", "#4c72b0", 1.3, "-")]

def load(p):
    d = np.loadtxt(p)
    out = {}
    for s, key in ((1, "up"), (-1, "lo")):
        m = d[:, 5] == s
        x, cp = d[m, 0], d[m, 2]
        o = np.argsort(x)
        out[key] = (x[o], cp[o])
    return out

def metrics(c):
    x, cp = c["up"]
    k = (x > 0.05) & (x < 0.95)
    if k.sum() < 5: return dict(mincp=np.nan, xs=np.nan, grad=np.nan, cn=np.nan)
    i = np.argmin(cp[k]); mincp = cp[k][i]
    # shock: steepest positive dCp/dx on the upper surface aft of the suction peak
    xu, cu = x[k], cp[k]
    g = np.gradient(cu, xu)
    aft = xu > xu[i]
    xs, grad = (np.nan, np.nan)
    if aft.sum() > 3:
        j = np.argmax(g[aft]); xs = xu[aft][j]; grad = g[aft][j]
    # normal force from the Cp loop (lower minus upper)
    xl, cl = c["lo"]
    xg = np.linspace(0.01, 0.99, 400)
    cn = np.trapezoid(np.interp(xg, xl, cl) - np.interp(xg, x, cp), xg)
    return dict(mincp=mincp, xs=xs, grad=grad, cn=cn)

have = [(f, l, col, lw, ls) for f, l, col, lw, ls in CASES if os.path.exists(f"{S}/{f}")]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2),
                       gridspec_kw=dict(width_ratios=[1.55, 1]))
rows = []
for f, lab, col, lw, ls in have:
    c = load(f"{S}/{f}")
    for key in ("up", "lo"):
        x, cp = c[key]
        ax[0].plot(x, cp, ls, color=col, lw=lw, label=lab if key == "up" else None)
    m = metrics(c); m["lab"] = lab; rows.append(m)
    ax[1].plot(*c["up"], ls, color=col, lw=lw, label=lab)

ax[0].invert_yaxis(); ax[0].set_xlabel("x/c"); ax[0].set_ylabel("$C_p$")
ax[0].set_title("RAE 2822, M=0.73, "+r"$\alpha$=2.31$^\circ$, nLvls 7 (upper and lower)")
ax[0].legend(fontsize=9, loc="lower right"); ax[0].grid(alpha=.3)
ax[1].invert_yaxis(); ax[1].set_xlim(0.35, 0.75); ax[1].set_xlabel("x/c")
ax[1].set_title("shock region, upper surface"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("output/rae_cp_delta.png", dpi=140)

print(f"{'config':<26}{'min Cp':>9}{'x_shock':>9}{'dCp/dx':>9}{'C_n':>8}")
for r in rows:
    print(f"{r['lab']:<26}{r['mincp']:>9.3f}{r['xs']:>9.3f}{r['grad']:>9.1f}{r['cn']:>8.3f}")
print("\n-> output/rae_cp_delta.png")
