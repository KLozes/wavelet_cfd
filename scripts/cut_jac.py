#!/usr/bin/env python3
"""Eigen-analysis of a cut element's own Jacobian (dgcutjac_test output).

Three questions:
  1. max Re lambda(A) -- does the ISOLATED element reproduce the measured
     +13/time?  If not, the instability is in the coupling and no element-local
     operator (damping, filter, hyper-viscosity) can reach it.
  2. Where does the unstable eigenvector live, by TOTAL DEGREE?  A degree-N
     eigenvector can be damped by a modal filter; a degree-0/1 one cannot.
  3. What sigma_0 would a degree-indexed damping need, and what does it cost
     the low modes?  A - Sigma swept over (sigma0, s, d0).
"""
import sys, glob, os
import numpy as np

def load(path):
    deg = e2 = None
    rows = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('#deg:'): deg = np.array([int(t) for t in ln[5:].split()])
            elif ln.startswith('#e2:'): e2 = np.array([int(t) for t in ln[4:].split()])
            elif ln.startswith('#'):    continue
            else: rows.append([float(t) for t in ln.strip().split(',')])
    return np.array(rows), deg, e2

def report(path, lam_h):
    A, deg, e2 = load(path)
    nu = A.shape[0]; nb = len(deg); nq = nu // nb
    dmode = np.repeat(deg, nq)            # unknown index -> total degree
    w, V = np.linalg.eig(A)
    k = int(np.argmax(w.real))
    print(f"\n=== {os.path.basename(path)}   {nu}x{nu}, nb={nb} ===")
    print(f"  max Re lambda = {w.real.max():+.4f}   (lambda/h = {lam_h:.1f}, "
          f"measured solver growth 13.2)")
    print(f"  spectral radius = {np.abs(w).max():.1f}")
    top = np.argsort(-w.real)[:6]
    print("  leading eigenvalues:  " + "  ".join(f"{w[t].real:+.3f}{w[t].imag:+.3f}i" for t in top))
    v = V[:, k]
    E = np.abs(v)**2
    print("  unstable eigenvector energy by total degree:")
    for d in range(deg.max()+1):
        f = E[dmode == d].sum()/E.sum()
        print(f"     deg {d}: {100*f:6.2f}%   {'#'*int(60*f)}")
    # which conserved variable
    Eq = np.array([E[q::nq].sum() for q in range(nq)])/E.sum()
    print("  ... by field (rho, mx, my, mz, E): " +
          " ".join(f"{100*x:.1f}%" for x in Eq))
    if w.real.max() <= 0:
        print("  >> the ISOLATED element is STABLE: the growth is in the COUPLING.")
        return
    # sigma sweep
    N = deg.max()
    print("  A - Sigma sweep   (sigma_d = s0*((d-d0)+/(N-d0))^(2s))")
    print("      d0  s   sigma0 | maxRe   sigma(1)  sigma(2)")
    for d0 in (0, 1, 2):
        for s in (1, 2, 3):
            for s0 in (5, 10, 20, 50, 100, 200):
                q = np.clip((dmode - d0)/max(N - d0, 1), 0, None)**(2*s)
                Anew = A - np.diag(s0*q)
                mr = np.linalg.eigvals(Anew).real.max()
                if mr <= 0 or s0 == 200:
                    sd = lambda d: s0*(max(d-d0,0)/max(N-d0,1))**(2*s)
                    print(f"      {d0}   {s}  {s0:5d} | {mr:+7.3f}  {sd(1):8.3f} {sd(2):8.3f}"
                          + ("   <-- stabilised" if mr <= 0 else "   (never)"))
                    break

if __name__ == "__main__":
    pats = sys.argv[1:] or sorted(glob.glob("cutjac_*.csv"))
    for p in pats:
        report(p, 16.0)
