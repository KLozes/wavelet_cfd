#!/usr/bin/env python3
"""
Quantify how far the fitted degree-q zero contour sits from the TRUE one.

Separation between the two zero sets is |phi_true - phi_fit| evaluated at the
SAME point (both are ~signed distances since |grad phi| ~ 1, so the difference
is the normal offset between the surfaces).

DO NOT use |phi_fit| alone on "surface" samples: those samples are only within
the picking tolerance of the true surface, so |phi_fit| there is dominated by
that tolerance, not by the surface separation.  That mistake inflated the fillet
figure by ~5x (19% of R_fil instead of the true 3.8%).

  usage: fitslice_quant.py <fitslice.txt> [R_fillet]
"""
import sys, numpy as np

fn = sys.argv[1]
RFIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

hdr = {}
with open(fn) as f:
    for tok in f.readline().lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1); hdr[k] = v
h   = float(hdr["h"]); res = int(hdr["res"]); deg = int(hdr["deg"])

d = np.loadtxt(fn, comments="#")
x, y, pt, pf = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
n = int(round(np.sqrt(len(x))))
X, Y, PT, PF = (a.reshape(n, n) for a in (x, y, pt, pf))
ds = X[0, 1] - X[0, 0]                      # fine sample spacing
R_HUB = 3.6374
R = np.hypot(X, Y)

# points essentially ON the true surface
on  = np.abs(PT) < 0.75*ds
SEP = np.abs(PT - PF)                       # <- surface-to-surface separation

print(f"file {fn}")
print(f"  res {res}   h = {h:.5f}   fine spacing = {ds:.5f}   fit degree {deg}")
print(f"  fillet R = {RFIL} = {RFIL/h:.3f} cells")
print(f"  samples on the true surface: {on.sum()}   (metric = |phi_true - phi_fit|)")

def rep(mask, label):
    if mask.sum() == 0:
        print(f"  {label:<26s} (no samples)"); return
    v = SEP[mask]
    print(f"  {label:<26s} n={v.size:6d}  max {v.max():.4e} ({v.max()/h:6.3f} h, "
          f"{100*v.max()/RFIL:6.1f}% of R_fil)   mean {v.mean():.4e} ({v.mean()/h:.3f} h)")

rep(on, "whole slice")
# the fillet arc: on the true surface AND within ~2 R_fil of the platform-top circle
fil = on & (np.abs(R - R_HUB) < 2.0*RFIL)
rep(fil, "fillet band (|r-r_hub|<2R)")
far = on & (np.abs(R - R_HUB) > 8.0*RFIL)
rep(far, "away from the fillet")
