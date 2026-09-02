#!/usr/bin/env python3
"""Convergence-rate table for the quarter-annulus vortex (Ndiaye et al. Tables 10/11).

  usage: svq_rates.py prefix        # reads <prefix>N{16,32,...}.log

Parses the 'paper table' block wave3d prints at the end of a case-16 run and
fits  log(err) = rate*log(h) + c  by least squares over ALL grids (the paper's
fit is over its five grids), per region W / I / B and per norm.  Also prints
the pairwise rates so the trend toward the asymptotic rate is visible.
"""
import sys, glob, re
import numpy as np

pre = sys.argv[1]
runs = {}
for f in glob.glob(pre + "N*.log"):
    N = int(re.search(r"N(\d+)\.log$", f).group(1))
    tab = {}
    for line in open(f):
        m = re.match(r"\s+([WIB]) \((all|uncut|cut)\)\s+(\d+)\s+(.*)", line)
        if m:
            tab[m.group(1)] = np.array(m.group(4).split(), float)
    res = [l for l in open(f) if l.startswith("[resid]")]
    rr = float(re.search(r"R/R0 = ([0-9.e+-]+)", res[-1]).group(1)) if res else np.nan
    if len(tab) == 3:
        runs[N] = (tab, rr)
Ns = sorted(runs)
cols = ["L1(rho)", "L2(rho)", "Linf(rho)", "L1(p)", "L2(p)", "Linf(p)"]
print(f"{'N':>5} {'R/R0':>9}  " + "  ".join(f"{c:>11}" for c in cols))
for reg in "WIB":
    print(f"-- region {reg}")
    for N in Ns:
        t, rr = runs[N]
        print(f"{N:5d} {rr:9.1e}  " + "  ".join(f"{v:11.4e}" for v in t[reg]))
    h = np.log(1.0/np.array(Ns, float))
    E = np.log(np.array([runs[N][0][reg] for N in Ns]))
    fit = [np.polyfit(h, E[:, j], 1)[0] for j in range(6)]
    print(f"{'fit':>5} {'':>9}  " + "  ".join(f"{r:11.2f}" for r in fit))
    for a, b in zip(Ns[:-1], Ns[1:]):
        pr = np.log(runs[a][0][reg]/runs[b][0][reg])/np.log(b/a)
        print(f"{a:>3}->{b:<3} {'':>7}  " + "  ".join(f"{r:11.2f}" for r in pr))
