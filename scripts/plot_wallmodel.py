#!/usr/bin/env python3
"""Wall-model comparison on the Brinkman path vs the grid-aligned reference.

Four panels: (a) route comparison at x=1.0; (b) traction route at four stations
(development); (c) semi-log inner scaling at x=1.0; (d) Cf vs x.
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
YW = 4.5*(1.5/(64*8))/4; h = (1.5/(64*8))/4; NU = 1e-6
KAP, B = 0.41, 5.2
dlt = 0.5*h; Lp = np.pi*dlt

def prof(fn, xs, yw):
    d = np.loadtxt(fn); x, y, u = d[:,0], d[:,1], d[:,2]
    m = np.abs(x-xs) < 0.006
    yb = np.unique(np.round(y[m], 9))
    uu = np.array([u[m][np.abs(y[m]-t) < 1e-9].mean() for t in yb])
    k = yb >= yw
    return yb[k]-yw, uu[k]

def utau_wf(u, d, nu):
    ut = max(np.sqrt(nu*u/d), 1e-12)
    for _ in range(80):
        yp = d*ut/nu
        up = yp if yp < 11.0 else np.log(max(yp,1e-12))/KAP + B
        g  = up + (0 if yp < 11.0 else 1.0/KAP)
        ut = max(ut - (ut*up - u)/max(g,1e-12), 1e-12)
    return ut

CASES = [(f"{S}/ga.dat",         0.0, "case 13 (grid-aligned ref)", "k",       2.2, "-"),
         (f"{S}/wm/noslip.dat",  YW,  "Brinkman no-slip",           "#999999", 1.4, "-"),
         (f"{S}/wm/s1.dat",      YW,  "slip 1: velocity target",    "#c44e52", 1.4, "-"),
         (f"{S}/wm/s2.dat",      YW,  "slip 2: permeability",       "#4c72b0", 1.4, "-"),
         (f"{S}/wm/final3.dat",  YW,  "slip 3: traction, wmmatch 3, converged", "#55a868", 1.9, "-")]

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

# (a) all routes at x=1.0
for fn, yw, lab, c, lw, ls in CASES:
    d, u = prof(fn, 1.0, yw)
    ax[0][0].plot(u, d/1e-3, ls, color=c, lw=lw, label=lab)
ax[0][0].set_xlim(-0.05, 1.15); ax[0][0].set_ylim(0, 20)
ax[0][0].set_xlabel("$u/U_\\infty$"); ax[0][0].set_ylabel("wall distance  $d\\times10^{3}$")
ax[0][0].set_title("(a) wall-model routes at x=1.0  (turbulence active)")
ax[0][0].legend(fontsize=8.5, loc="lower right"); ax[0][0].grid(alpha=.3)

# (b) traction vs ref at four stations
cols = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
for xs, c in zip((0.4, 0.7, 1.0, 1.3), cols):
    d, u = prof(f"{S}/ga.dat", xs, 0.0)
    ax[0][1].plot(u, d/1e-3, "-", color=c, lw=2.0, label=f"ref x={xs}")
    d, u = prof(f"{S}/wm/final3.dat", xs, YW)
    ax[0][1].plot(u, d/1e-3, "--", color=c, lw=1.4)
ax[0][1].set_xlim(-0.05, 1.15); ax[0][1].set_ylim(0, 20)
ax[0][1].set_xlabel("$u/U_\\infty$"); ax[0][1].set_ylabel("wall distance  $d\\times10^{3}$")
ax[0][1].set_title("(b) development: reference (solid) vs traction (dashed)")
ax[0][1].legend(fontsize=8.5, loc="lower right"); ax[0][1].grid(alpha=.3)

# (c) inner scaling at x=1.0
d, u = prof(f"{S}/ga.dat", 1.0, 0.0)
ut_ref = utau_wf(np.interp(Lp, d, u), Lp, NU)
for fn, yw, lab, c, lw, ls in (CASES[0], CASES[4]):
    d, u = prof(fn, 1.0, yw)
    ut = utau_wf(np.interp(3*h, d, u), 3*h, NU)
    yp = d*ut/NU; k = (yp > 1) & (yp < 3000)
    ax[1][0].semilogx(yp[k], u[k]/ut, ls, color=c, lw=lw, label=lab+f"  ($u_\\tau$={ut:.4f})")
ypl = np.logspace(0, 3.5, 100)
ax[1][0].semilogx(ypl[ypl<12], ypl[ypl<12], ":", color="0.5", lw=1)
ax[1][0].semilogx(ypl[ypl>8], np.log(ypl[ypl>8])/KAP+B, ":", color="0.5", lw=1,
                  label="$u^+=y^+$;  log law")
ax[1][0].set_xlabel("$y^+$"); ax[1][0].set_ylabel("$u^+$")
ax[1][0].set_title("(c) inner scaling at x=1.0")
ax[1][0].legend(fontsize=8.5, loc="upper left"); ax[1][0].grid(alpha=.3, which="both")

# (d) Cf vs x
ref = np.loadtxt(f"{S}/ga_cf.dat")
# the file holds EVERY output interval appended; keep only the final snapshot
starts = np.where(np.diff(ref[:,0]) < 0)[0] + 1
if len(starts): ref = ref[starts[-1]:]
k = (ref[:,0] > 0.15) & (ref[:,0] < 1.45)
ax[1][1].plot(ref[k,0], ref[k,1], "k-", lw=2.0, label="case 13 (solver $u_\\tau$)")
xs_ = np.linspace(0.2, 1.45, 60)
ax[1][1].plot(xs_, 0.0592*(1e6*xs_)**-0.2, "--", color="0.5", lw=1.3,
              label="$0.0592\\,Re_x^{-1/5}$")
xt, cft = [], []
for xs in np.linspace(0.25, 1.4, 24):
    d, u = prof(f"{S}/wm/final3.dat", xs, YW)
    if len(d) < 4: continue
    ut = utau_wf(abs(np.interp(3*h, d, u)), 3*h, NU)
    xt.append(xs); cft.append(2*ut*ut)
ax[1][1].plot(xt, cft, "-", color="#55a868", lw=1.9,
              label="traction, wmmatch 3, tEnd 4 (offline WF)")
# d95 growth inset: the development test
axi = ax[1][1].inset_axes([0.52, 0.52, 0.45, 0.42])
for fn_, yw_, c_ in ((f"{S}/ga.dat", 0.0, "k"), (f"{S}/wm/final3.dat", YW, "#55a868")):
    xs_l, dl_ = [], []
    for xq in np.linspace(0.3, 1.4, 16):
        dq, uq = prof(fn_, xq, yw_)
        if len(dq) < 4: continue
        ue = np.median(uq[dq > 0.05]); iq = np.argmax(uq >= 0.95*ue)
        if iq > 0: xs_l.append(xq); dl_.append(np.interp(0.95*ue, uq[:iq+1], dq[:iq+1]))
    axi.plot(xs_l, np.array(dl_)*1e3, color=c_, lw=1.5)
axi.set_title(r"$\delta_{95}\times10^3$: layer growth", fontsize=8)
axi.tick_params(labelsize=7); axi.grid(alpha=.3)
ax[1][1].set_xlabel("x"); ax[1][1].set_ylabel("$C_f$"); ax[1][1].set_ylim(0, 0.008)
ax[1][1].set_title("(d) skin friction")
ax[1][1].legend(fontsize=8.5); ax[1][1].grid(alpha=.3)

fig.suptitle("Brinkman wall models vs grid-aligned reference — immersed flat plate, "
             "$Re=10^6$, $M=0.2$, $k$–$\\tau$ SST, $y^+_1\\!\\approx\\!11$", fontsize=12)
plt.tight_layout(); plt.savefig("output/wallmodel_cmp.png", dpi=140)
print("-> output/wallmodel_cmp.png")
