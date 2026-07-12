#!/usr/bin/env python3
"""Colorize the solver's grayscale PNG frames with a matplotlib colormap.

Reads every grayscale .png in a directory (the solver writes 16-bit grayscale
via png++), maps intensity through the chosen matplotlib colormap, and writes
RGB PNGs.  Any matplotlib colormap name works (viridis, plasma, inferno,
magma, turbo, jet, coolwarm, RdBu_r, ...).

Usage:
    python3 scripts/colorize.py <directory> <colormap> [--out DIR] [--suffix S]

    --out DIR    output directory (default: <directory>/<colormap>/)
    --suffix S   output filename suffix before .png (default: none)

Examples:
    python3 scripts/colorize.py output viridis     -> output/viridis/*.png
    python3 scripts/colorize.py output turbo       -> output/turbo/*.png
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg


def get_cmap(name):
    try:                                   # matplotlib >= 3.6
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError, TypeError):
        try:                               # older matplotlib
            import matplotlib.cm as cm
            return cm.get_cmap(name)
        except ValueError:
            return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing grayscale .png frames")
    ap.add_argument("colormap", help="matplotlib colormap name (e.g. viridis, turbo)")
    ap.add_argument("--out", default=None, help="output directory (default: <directory>/<colormap>/)")
    ap.add_argument("--suffix", default="", help="output filename suffix (default: none)")
    args = ap.parse_args()

    cmap = get_cmap(args.colormap)
    if cmap is None:
        sys.exit(f"unknown colormap '{args.colormap}' -- see matplotlib.colormaps "
                 f"(e.g. viridis plasma inferno magma turbo jet coolwarm)")

    suffix = args.suffix
    outdir = args.out or os.path.join(args.directory, args.colormap)
    os.makedirs(outdir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.directory) if f.lower().endswith(".png"))
    done = skipped = 0
    for name in files:
        if suffix and name.endswith(suffix + ".png"):
            skipped += 1                    # already a colorized output
            continue
        src = os.path.join(args.directory, name)
        img = mpimg.imread(src)
        if img.ndim != 2:
            skipped += 1                    # RGB(A) input -- leave alone
            continue
        dst = os.path.join(outdir, name[:-4] + suffix + ".png")
        # imread returns floats in [0,1] for gray PNGs; the solver frames are
        # already normalized to full range, so map [0,1] straight through
        mpimg.imsave(dst, img, cmap=cmap, vmin=0.0, vmax=1.0)
        done += 1
        print(f"{name} -> {os.path.relpath(dst)}")
    print(f"colorized {done} file(s), skipped {skipped}")


if __name__ == "__main__":
    main()
