#!/usr/bin/env python3
"""Schlieren (|grad rho|) beside the AMR grid to see whether wake structure is
being missed by the refinement indicator."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

n = sys.argv[1] if len(sys.argv) > 1 else "00013"
rho  = mpimg.imread(f"output/image00_{n}.png").astype(float)
grid = mpimg.imread(f"output/grid_{n}.png").astype(float)
if rho.ndim == 3:  rho  = rho[...,0]
if grid.ndim == 3: grid = grid[...,0]

# schlieren = |grad rho|, log-scaled to reveal weak wake gradients
gy, gx = np.gradient(rho)
sch = np.sqrt(gx*gx + gy*gy)
sch = np.log1p(sch / (sch.max()+1e-12) * 500)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
ax[0].imshow(rho, cmap="turbo", origin="upper");  ax[0].set_title(f"density  (frame {n})")
ax[1].imshow(sch, cmap="inferno", origin="upper"); ax[1].set_title("schlieren |grad rho| (log) -- what SHOULD refine")
ax[2].imshow(grid, cmap="gray", origin="upper");   ax[2].set_title("AMR grid -- what IS refined")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("scripts/wake_check.png", dpi=110)
print("wrote scripts/wake_check.png")
