#!/usr/bin/env python3
# Assemble the circular-Sod frames into a side-by-side (density | adaptive grid) animated gif.
import glob, re, sys
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def frame_id(path):
    return re.search(r'_(\d+)\.png$', path).group(1)

dens_files = sorted(glob.glob("output/image00_*.png"), key=frame_id)
ids = [frame_id(f) for f in dens_files]
grid_files = [f"output/grid_{i}.png" for i in ids]
print(f"{len(ids)} frames")

# global grid max for consistent level coloring across frames
gmax = 1.0
for gf in grid_files:
    gmax = max(gmax, float(np.array(Image.open(gf)).max()))

inferno = cm.get_cmap("inferno")
viridis = cm.get_cmap("viridis")

def colorize(arr, cmap, vmax):
    a = np.clip(arr.astype(float) / (vmax + 1e-12), 0, 1)
    rgb = (cmap(a)[:, :, :3] * 255).astype(np.uint8)
    return rgb

frames = []
for df, gf in zip(dens_files, grid_files):
    d = np.array(Image.open(df))
    g = np.array(Image.open(gf))
    d = np.flipud(d); g = np.flipud(g)          # origin lower
    dc = colorize(d, inferno, d.max())            # density: per-frame scale
    gc = colorize(g, viridis, gmax)               # grid: fixed level scale
    sep = np.full((dc.shape[0], 6, 3), 255, np.uint8)
    combo = np.concatenate([dc, sep, gc], axis=1)
    frames.append(Image.fromarray(combo))

out = "output/circle_evolution.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
print("wrote", out, "size", frames[0].size)
