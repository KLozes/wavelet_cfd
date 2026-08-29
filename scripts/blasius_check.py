#!/usr/bin/env python3
"""Validate the viscous Brinkman no-slip wall against the exact Blasius solution.

Blasius: f''' + f f''/2 = 0, f(0)=f'(0)=0, f'(inf)=1, with u/U = f'(eta) and
eta = (y - y_wall) sqrt(U / (nu (x - x_LE))).  Wall shear gives Cf = 0.664/sqrt(Re_x).
Solved here by RK4 shooting on f''(0) -- no table, no external data.
"""
import sys, numpy as np

def blasius(eta_max=10.0, n=20000):
    def rhs(F):
        f, g, h = F           # f, f', f''
        return np.array([g, h, -0.5*f*h])
    def shoot(s):
        F = np.array([0.0, 0.0, s]); de = eta_max/n
        out = [F.copy()]
        for _ in range(n):
            k1 = rhs(F); k2 = rhs(F+de/2*k1); k3 = rhs(F+de/2*k2); k4 = rhs(F+de*k3)
            F = F + de/6*(k1+2*k2+2*k3+k4); out.append(F.copy())
        return np.array(out)
    lo, hi = 0.1, 1.0
    for _ in range(80):                     # bisect on f'(inf) = 1
        mid = 0.5*(lo+hi)
        if shoot(mid)[-1][1] < 1.0: lo = mid
        else: hi = mid
    s = 0.5*(lo+hi); sol = shoot(s)
    return np.linspace(0, eta_max, n+1), sol[:,1], s

eta, fp, fpp0 = blasius()
print(f"  Blasius f''(0) = {fpp0:.6f}  (exact 0.332057)")

# Wall spans the whole domain and the inflow is uniform, so the boundary
# layer originates at the inflow plane: x_LE = 0.  Using ibtype 4 with a
# raised plate instead makes a FORWARD-FACING STEP (measured: flow already
# down to 0.75 upstream of the LE, separated jet at u=1.5 downstream).
fn, XLE, YW, NU, U = sys.argv[1], 0.0, 0.05, 1e-4, 1.0
d = np.loadtxt(fn); x, y, u = d[:,0], d[:,1], d[:,2]
print(f"\n{'x':>6}{'Re_x':>10}{'d99 num':>10}{'d99 exact':>11}{'err%':>7}"
      f"{'Cf num':>10}{'Cf exact':>10}{'err%':>7}")
for xs in (0.7, 1.2, 1.7):
    m = np.abs(x-xs) < 0.006
    if m.sum() < 10: continue
    yy, uu = y[m], u[m]; o = np.argsort(yy); yy, uu = yy[o], uu[o]
    k = yy >= YW
    yy, uu = yy[k]-YW, uu[k]
    Rex = U*(xs-XLE)/NU
    d99e = 5.0*(xs-XLE)/np.sqrt(Rex)
    # freestream from a band just ABOVE the layer, not a fixed height
    Ue = np.median(uu[(yy > 2*d99e) & (yy < 4*d99e)])
    i = np.argmax(uu >= 0.99*Ue)
    d99 = np.interp(0.99*Ue, uu[:i+1], yy[:i+1]) if i > 0 else np.nan
    # wall shear from the first few points above the wall
    sel = yy < 0.25*d99e
    dudy = np.polyfit(yy[sel], uu[sel], 1)[0] if sel.sum() >= 3 else np.nan
    Cf, Cfe = 2*NU*dudy/U**2, 0.664/np.sqrt(Rex)
    print(f"{xs:>6.2f}{Rex:>10.0f}{d99:>10.4f}{d99e:>11.4f}{100*(d99-d99e)/d99e:>7.1f}"
          f"{Cf:>10.5f}{Cfe:>10.5f}{100*(Cf-Cfe)/Cfe:>7.1f}")
print("\n  profile vs Blasius at x=1.2 (eta, u/U num, u/U exact):")
m = np.abs(x-1.2) < 0.006
yy, uu = y[m], u[m]; o = np.argsort(yy); yy, uu = yy[o], uu[o]
k = yy >= YW; yy, uu = yy[k]-YW, uu[k]
d99e = 5.0*1.2/np.sqrt(U*1.2/NU)
Ue = np.median(uu[(yy > 2*d99e) & (yy < 4*d99e)])
et = yy*np.sqrt(U/(NU*(1.2-XLE)))
for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
    print(f"    eta={t:>4.1f}   num {np.interp(t, et, uu)/Ue:>6.3f}   exact {np.interp(t, eta, fp):>6.3f}")
