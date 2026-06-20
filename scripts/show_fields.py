#!/usr/bin/env python3
# Multi-panel view of the final circular-Sod state:
# density, total energy, z-momentum (should be ~0), pressure, and the AMR grid.
import sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def load(path):
    a = np.array(Image.open(path)).astype(float)
    return np.flipud(a)  # origin lower

frame = sys.argv[1] if len(sys.argv) > 1 else \
        re.search(r'_(\d+)\.png$', sorted(glob.glob("output/image00_*.png"))[-1]).group(1)

panels = [
    ("output/image00_%s.png" % frame, "density (Rho)",        "inferno"),
    ("output/image04_%s.png" % frame, "total energy (RhoE)",  "inferno"),
    ("output/pressure_final.png",     "pressure",             "inferno"),
    ("output/image03_%s.png" % frame, "z-momentum (RhoW)\n(roundoff, per-frame normalized)", "seismic"),
    ("output/grid_%s.png"   % frame,  "AMR refinement level", "viridis"),
]

fig, ax = plt.subplots(1, len(panels), figsize=(4.2*len(panels), 4.4))
for k,(path,title,cmap) in enumerate(panels):
    img = load(path)
    img = img/ (img.max()+1e-12)
    im = ax[k].imshow(img, origin="lower", cmap=cmap)
    ax[k].set_title(title, fontsize=10)
    ax[k].set_xticks([]); ax[k].set_yticks([])
plt.tight_layout()
out = "output/fields_%s.png" % frame
plt.savefig(out, dpi=110)
print("wrote", out)
