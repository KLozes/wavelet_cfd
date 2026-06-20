#!/usr/bin/env python3
# Render the 2D circular Sod explosion: density field + adaptive grid (block levels).
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

frame = sys.argv[1] if len(sys.argv) > 1 else "00022"

dens = np.array(Image.open(f"output/image00_{frame}.png")).astype(float)
grid = np.array(Image.open(f"output/grid_{frame}.png")).astype(float)

# normalize density 0..1 for display (it was written normalized per-frame already)
dens /= dens.max() + 1e-12

# grid image encodes (level+1); internal block lines are 0. recover the level map.
g = grid / (grid.max() + 1e-12)

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
im0 = ax[0].imshow(dens, origin="lower", cmap="inferno")
ax[0].set_title("density  (2D circular Sod explosion, pseudo-2D)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(g, origin="lower", cmap="viridis")
ax[1].set_title("adaptive grid: refinement level + block boundaries")
ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

plt.tight_layout()
out = f"output/circle_{frame}.png"
plt.savefig(out, dpi=110)
print("wrote", out, " density shape", dens.shape, " grid shape", grid.shape)
