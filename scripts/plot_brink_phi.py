#!/usr/bin/env python3
"""Plot the Brinkman volume fraction phi(r) exactly as dgBrinkPhi computes it,
to verify the object edge is smooth but compact (smeared over 1-2 finest cells)."""
import numpy as np
import matplotlib.pyplot as plt

# case-9 params (must match DgMain.cu / dgBrinkPhi)
R   = 0.5
ibX, ibY = 1.5, 2.0
eps = 1e-4
hF  = 6.0/24/8            # finest cell: domain 6, nElemX 24, nLvls 4 -> /8
brinkDelta = 1.0          # finest-cell units (--ibbrinkdelta default)
d   = brinkDelta*hF       # physical half-width

def phi(r):
    tt = (r - (R - d))/(2.0*d)
    out = np.where(tt <= 0, eps,
          np.where(tt >= 1, 1.0,
                   eps + (1-eps)*tt*tt*(3-2*tt)))
    return out

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- (a) radial profile, finest cells marked -------------------------------
r = np.linspace(R-4*hF, R+4*hF, 2000)
ax0.plot(r, phi(r), lw=2.2, color='C0')
# finest-cell gridlines around the surface
for k in range(-4, 5):
    xg = R + k*hF
    ax0.axvline(xg, color='0.85', lw=0.8, zorder=0)
ax0.axvspan(R-d, R+d, color='C1', alpha=0.15, label=f'transition 2d = {2*d:.4f} = 2 cells')
ax0.axvline(R, color='k', ls='--', lw=1.2, label='true wall  r=R=0.5')
ax0.set_xlabel('r  (distance from cylinder center)')
ax0.set_ylabel(r'$\phi$  (fluid volume fraction)')
ax0.set_title(f'phi(r): smooth but compact\nhalf-width d = {brinkDelta} finest cell = {d:.5f},  h_finest = {hF:.5f}')
ax0.set_ylim(-0.05, 1.08)
ax0.legend(loc='center left', fontsize=9)
ax0.grid(alpha=0.15)

# ---- (b) 2D map near the cylinder, finest grid overlaid --------------------
pad = 0.28
xs = np.linspace(ibX-R-pad, ibX+R+pad, 600)
ys = np.linspace(ibY-R-pad, ibY+R+pad, 600)
XX, YY = np.meshgrid(xs, ys)
RR = np.sqrt((XX-ibX)**2 + (YY-ibY)**2)
PHI = phi(RR)
im = ax1.pcolormesh(XX, YY, PHI, shading='auto', cmap='viridis', vmin=0, vmax=1)
# finest grid lines
gx = np.arange(np.floor((ibX-R-pad)/hF), np.ceil((ibX+R+pad)/hF)+1)*hF
gy = np.arange(np.floor((ibY-R-pad)/hF), np.ceil((ibY+R+pad)/hF)+1)*hF
for xg in gx: ax1.axvline(xg, color='w', lw=0.25, alpha=0.35)
for yg in gy: ax1.axhline(yg, color='w', lw=0.25, alpha=0.35)
th = np.linspace(0, 2*np.pi, 400)
ax1.plot(ibX+R*np.cos(th), ibY+R*np.sin(th), 'r--', lw=1.4, label='true wall')
ax1.set_aspect('equal'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title('phi field with finest grid (h=0.03125) overlaid')
ax1.legend(loc='upper right', fontsize=9)
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04); cb.set_label(r'$\phi$')

fig.tight_layout()
out = 'scripts/brink_phi.png'
fig.savefig(out, dpi=130)
print('wrote', out)
