#!/usr/bin/env python3
"""Plot a cut-cell DG solution honestly.

Three things every existing plot of a --cutcell run got wrong, and what this
script does instead:

  1. The body was INVISIBLE.  buildCutElems relabels cut blocks IB_FLUID and
     sets ibOn = 0, so the paint kernel's SDF branch never fired; solid blocks
     kept the frozen initial condition and rendered as pristine freestream.
     The solver now writes kPaintVoid there and paintField reserves PNG value 0
     for it, so anything reading 0 knows there is no solution at that pixel.

  2. Absolute values were UNRECOVERABLE.  paintField rescales by each frame's
     own min/max and the PNG recorded neither, so no two frames shared a scale.
     output/paint_scale.csv now carries min/max/domain per frame; this script
     inverts it, and --clim locks one scale across a sequence.

  3. A cut element's solution was never sampled.  Its state is a modal
     polynomial supported on the fluid side only; the tensor nodes the image
     interpolates include points buried in the solid.  The solver now also
     writes cut_{geom,wall,vol}.csv -- the modal solution evaluated at the Saye
     volume points (inside the fluid by construction) and at the Saye surface
     points (on the true wall) -- and this script draws those directly.

usage:
    python3 scripts/plot_cut.py                       # latest frame, density
    python3 scripts/plot_cut.py --field 04 --frame 7
    python3 scripts/plot_cut.py --clim 0.8 1.3 --out output/cut_frame.png
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

FIELD_NAME = {"00": "rho", "01": "rho u", "02": "rho v", "03": "rho w", "04": "rho E"}


def read_scale(path):
    """file -> dict of the frame's paint metadata."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["file"]] = {
                "min": float(row["min"]), "max": float(row["max"]),
                "nvoid": int(row["nvoid"]),
                "nx": int(row["nx"]), "ny": int(row["ny"]),
                "domx": float(row.get("domx", 0)) or None,
                "domy": float(row.get("domy", 0)) or None,
            }
    return out


def load_frame(png, meta):
    """16-bit greyscale -> physical values, with 0 decoded as 'no data'.

    paintField maps live data to [1, 65535] and reserves 0, so the mask is
    exact rather than a guess at a magic value.
    """
    a = np.array(Image.open(png)).astype(np.float64)
    void = a <= 0.0
    if meta is None:
        val = a / 65535.0            # no sidecar: normalised units, say so
        return val, void, None
    lo, hi = meta["min"], meta["max"]
    val = lo + (a - 1.0) / 65534.0 * (hi - lo)
    return val, void, (lo, hi)


def load_cut(stem):
    def rd(suffix):
        p = f"{stem}_{suffix}.csv"
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None
    return rd("geom"), rd("wall"), rd("vol"), rd("fine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output")
    ap.add_argument("--field", default="00", help="00 rho, 01 rho u, ... 04 rho E")
    ap.add_argument("--frame", type=int, default=-1, help="-1 = latest")
    ap.add_argument("--clim", nargs=2, type=float, default=None,
                    help="lock the colour range (use across a sequence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.dir, f"image{args.field}_*.png")))
    if not frames:
        sys.exit(f"no image{args.field}_*.png in {args.dir}")
    png = frames[args.frame]

    scale = read_scale(os.path.join(args.dir, "paint_scale.csv"))
    meta = scale.get(png)
    val, void, rng = load_frame(png, meta)
    ny, nx = val.shape
    domx = (meta or {}).get("domx") or float(nx)
    domy = (meta or {}).get("domy") or float(ny)

    geom, wall, vol, fine = load_cut(os.path.join(args.dir, "cut"))
    have_cut = wall is not None and len(wall.get("x", [])) > 0

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 1.0])

    # ---- panel 1: the field, body masked, true wall drawn over it ----------
    ax = fig.add_subplot(gs[0, 0])
    finite = val[~void]
    if args.clim:
        vmin, vmax = args.clim
    elif finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    shown = np.ma.masked_where(void, val)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#2b2b2b")          # the body: explicitly "no solution here"
    im = ax.imshow(shown, origin="lower", extent=[0, domx, 0, domy],
                   cmap=cmap, norm=Normalize(vmin, vmax), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    if have_cut:
        ax.plot(wall["x"], wall["y"], ".", ms=1.6, color="white", alpha=0.9,
                label="Saye wall points")
    if geom is not None:
        for x0, y0, hx, hy in zip(geom["x0"], geom["y0"], geom["hx"], geom["hy"]):
            ax.add_patch(plt.Rectangle((x0, y0), hx, hy, fill=False,
                                       ec="#ff5555", lw=0.7, alpha=0.85))
    ax.set_title(f"{os.path.basename(png)}  [{FIELD_NAME.get(args.field, args.field)}]"
                 + (f"   range {rng[0]:.4g}..{rng[1]:.4g}" if rng else "  (normalised)"))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if have_cut:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    # ---- panel 2: the wall, which a cut run never produced before ----------
    ax2 = fig.add_subplot(gs[0, 1])
    if have_cut:
        cx = 0.5 * (wall["x"].min() + wall["x"].max())
        cy = 0.5 * (wall["y"].min() + wall["y"].max())
        th = np.degrees(np.arctan2(wall["y"] - cy, wall["x"] - cx))
        o = np.argsort(th)
        ax2.plot(th[o], wall["cp"][o], ".", ms=2.5, color="#1f77b4")
        ax2.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax2.set_xlabel("theta [deg]  (0 = +x, downstream)")
        ax2.set_ylabel("Cp")
        ax2.set_title("wall Cp at the Saye surface points")
        # a P0 cut run gives ONE constant per element, so a staircase here is
        # the discretization being honest, not a plotting artifact
        ax2.grid(alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "no cut_wall.csv\n(run with --cutcell 1)",
                 ha="center", va="center")
        ax2.set_axis_off()

    # ---- panel 3: the cut band at its OWN resolution -----------------------
    # The raster gives a cut element only blockSize pixels per axis, so the band
    # reads as blocks no matter what the polynomial does.  cut_fine.csv is the
    # element's own polynomial on a dense grid, fluid side only.
    ax3 = fig.add_subplot(gs[0, 2])
    src = fine if (fine is not None and len(fine.get("x", [])) > 0) else vol
    if src is not None and len(src.get("x", [])) > 0:
        sc = ax3.scatter(src["x"], src["y"], c=src["rho"], s=(3 if src is fine else 7),
                         marker="s", cmap="viridis", vmin=vmin, vmax=vmax,
                         linewidths=0)
        fig.colorbar(sc, ax=ax3, fraction=0.04, pad=0.01)
        if have_cut:
            # scatter, NOT a line: the wall points are ordered by element, so a
            # connected line draws chords across the body between elements
            ax3.plot(wall["x"], wall["y"], ".", ms=1.0, color="k", alpha=0.7)
        ax3.set_aspect("equal")
        ax3.set_title(("rho on the cut band, from each element's OWN\npolynomial "
                       "(fluid side only)") if src is fine else
                      "rho at the Saye VOLUME points", fontsize=9)
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
    else:
        ax3.text(0.5, 0.5, "no cut_fine.csv / cut_vol.csv", ha="center", va="center")
        ax3.set_axis_off()

    out = args.out or os.path.join(args.dir, f"cut_{args.field}_"
                                             f"{os.path.basename(png).split('_')[-1]}")
    fig.savefig(out, dpi=150)
    nv = int(void.sum())
    print(f"wrote {out}   void pixels {nv} ({100.0*nv/void.size:.1f}%)"
          + (f"   value range {rng[0]:.6g} .. {rng[1]:.6g}" if rng else ""))


if __name__ == "__main__":
    main()
