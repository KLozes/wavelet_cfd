#!/usr/bin/env python3
"""Plot the raw SDF slices dumped by `wavefem --slice`.

Shows the composite level set (blade + platform + fillet + tip gap + sector)
directly, so the CSG algebra can be inspected rather than inferred from where
the cut cells happened to land.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'output/bank_v98d_ROTOR_1_slice.bin'
out  = sys.argv[2] if len(sys.argv) > 2 else '/tmp/slice.png'
with open(path, 'rb') as f:
    npl, NA, NB, _ = np.fromfile(f, dtype=np.int32, count=4)
    fig, axes = plt.subplots(1, npl, figsize=(5.6*npl, 5.8))
    if npl == 1: axes = [axes]
    for p in range(npl):
        kind, x0, x1, y0, y1, hcell, ucell, vcell = np.fromfile(f, dtype=np.float64, count=8)
        nl = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        name = f.read(nl).decode()
        phi  = np.fromfile(f, dtype=np.float32, count=NA*NB).reshape(NB, NA)
        gcel = np.fromfile(f, dtype=np.float32, count=NA*NB).reshape(NB, NA)
        ax = axes[p]
        ext = [x0, x1, y0, y1]
        lim = max(1e-6, 0.35*np.percentile(np.abs(phi), 90))
        ax.imshow(phi, origin='lower', extent=ext, aspect='auto',
                  cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.contour(np.linspace(x0, x1, NA), np.linspace(y0, y1, NB), phi,
                   levels=[0.0], colors='k', linewidths=1.6)
        # the solver's background cells.  gcel is the distance to the nearest
        # cell face as a FRACTION of a cell, so the contour level is the pixel
        # size measured in cells -- that keeps the lines one pixel wide on every
        # panel regardless of the axis units.
        du = (x1 - x0)/(NA - 1)/ucell
        dv = (y1 - y0)/(NB - 1)/vcell
        ax.contour(np.linspace(x0, x1, NA), np.linspace(y0, y1, NB), gcel,
                   levels=[0.7*max(du, dv)], colors='0.30', linewidths=0.4, alpha=0.8)
        ax.set_xlabel('z (axial)' if kind == 0 else "theta' (rad)")
        ax.set_ylabel('r')
        ax.set_title(name, fontsize=10)
        if kind == 1:                      # mark the two periodic (cyclic) faces
            ax.axvline(x0, color='m', lw=1.4, ls='--')
            ax.axvline(x1, color='m', lw=1.4, ls='--')
plt.tight_layout()
plt.savefig(out, dpi=110)
print('wrote', out)
