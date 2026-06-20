#!/usr/bin/env python3
"""Plot filled signed-distance contours from a narrowbandSDF .vtk file.

Reads the legacy VTK STRUCTURED_POINTS file written by ./narrowbandSDF and draws
center-plane (or chosen) slices as smooth filled contour bands with a bold black
zero-level contour marking the surface -- the style of SDF figures like the
piggy-bank cutaway. The far-field (cells clamped to +band) is masked so only the
narrowband shell and the resolved interior are drawn.

usage:
    python3 sdf/plot_slices.py [out/foo_sdf.vtk] [--axis xy|xz|yz|all]
                               [--index N] [--levels N] [--cmap NAME]

Defaults to output/wing_sdf.vtk and all three center slices.
"""

import argparse
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_vtk(path):
    """Return (vol[nz,ny,nx], origin(3), spacing) from a legacy VTK
    STRUCTURED_POINTS file with big-endian float SCALARS."""
    with open(path, "rb") as f:
        data = f.read()
    marker = b"LOOKUP_TABLE default\n"
    head_end = data.find(marker) + len(marker)
    head = data[:head_end].decode("ascii", "replace")
    dims = origin = spacing = None
    for line in head.splitlines():
        t = line.split()
        if line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in t[1:4])
        elif line.startswith("ORIGIN"):
            origin = tuple(float(x) for x in t[1:4])
        elif line.startswith("SPACING"):
            spacing = float(t[1])
    nx, ny, nz = dims
    n = nx * ny * nz
    vol = np.frombuffer(data[head_end:head_end + 4 * n], dtype=">f4").astype(np.float32)
    return vol.reshape(nz, ny, nx), np.array(origin), spacing


def extract_slice(vol, origin, dx, axis, index):
    """Return (Z, xc, yc, xlabel, ylabel) for an axis-aligned slice.
    axis: 'xy' (const z), 'xz' (const y), 'yz' (const x)."""
    nz, ny, nx = vol.shape
    if axis == "xy":
        idx = index if index is not None else nz // 2
        Z = vol[idx, :, :]                       # (ny, nx)
        xc = origin[0] + np.arange(nx) * dx
        yc = origin[1] + np.arange(ny) * dx
        return Z, xc, yc, "x", "y"
    if axis == "xz":
        idx = index if index is not None else ny // 2
        Z = vol[:, idx, :]                       # (nz, nx)
        xc = origin[0] + np.arange(nx) * dx
        yc = origin[2] + np.arange(nz) * dx
        return Z, xc, yc, "x", "z"
    if axis == "yz":
        idx = index if index is not None else nx // 2
        Z = vol[:, :, idx]                       # (nz, ny)
        xc = origin[1] + np.arange(ny) * dx
        yc = origin[2] + np.arange(nz) * dx
        return Z, xc, yc, "y", "z"
    raise ValueError(f"unknown axis '{axis}'")


def reconstruct_full(Z, dx):
    """Reconstruct a full signed distance field over the slice.

    The solver fills every cell of an active block with its true signed distance,
    which can exceed the narrowband half-width `band`; only cells *outside* the
    active blocks are clamped to a single far-field constant (`+band`). So the
    value to replace is that background constant -- everything else is a resolved
    distance and is kept as-is. The background is then filled with a Euclidean
    distance transform so the whole domain carries evenly spaced contour bands.

    NB: the background is the most common *positive* value (the `+band` constant),
    NOT `nanmax(Z)`: with the full active-block fill `nanmax` is now a genuine
    beyond-band distance, and using it would misclassify the outermost resolved
    cells as enclosed far field and flip them to spurious negatives."""
    from scipy import ndimage
    pos = Z[Z > 0]
    if pos.size:
        vals, counts = np.unique(pos, return_counts=True)
        bg = float(vals[np.argmax(counts)])      # far-field constant = +band
        far = Z == bg
    else:
        far = np.zeros_like(Z, dtype=bool)        # slice fully resolved, nothing to fill
    lbl, _ = ndimage.label(far)                  # separate exterior/interior far field
    edge = np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]))
    edge = set(edge.tolist()) - {0}
    exterior_far = np.isin(lbl, list(edge)) if edge else np.zeros_like(far)
    solid = (Z < 0) | (far & ~exterior_far)      # inside = negative cells + enclosed far field
    din = ndimage.distance_transform_edt(solid)
    dout = ndimage.distance_transform_edt(~solid)
    edt = np.where(solid, -(din - 0.5), dout - 0.5) * dx
    # keep every resolved cell (the solver's true distance, beyond-band included);
    # the integer EDT only fills the clamped background far field.
    known = ~far
    return np.where(known, Z, edt)


def plot_slice(ax, Z, xc, yc, dx, n_inside, show_grid, fill, clip, supersample, vmax=None):
    """Topographic iso-distance bands in the style of signed_heat.py: evenly
    spaced stripes two alternating tones per side (light/deep blue inside,
    light/rose outside), 0 on a band boundary, with a bold zero contour, a lime
    surface outline and a faint grid."""
    F = reconstruct_full(Z, dx) if fill else Z

    # cell-grid edges from the *original* grid (drawn before any supersampling)
    xe = xc[0] - dx / 2 + np.arange(len(xc) + 1) * dx
    ye = yc[0] - dx / 2 + np.arange(len(yc) + 1) * dx

    # sub-cell sampling: smoothly interpolate the field onto a finer grid so the
    # contour bands are no longer stair-stepped at cell boundaries
    if supersample > 1:
        from scipy.ndimage import zoom
        F = zoom(F, supersample, order=3)
        xc = np.linspace(xc[0], xc[-1], F.shape[1])
        yc = np.linspace(yc[0], yc[-1], F.shape[0])

    vmin = float(np.nanmin(F))
    vhi = vmax if vmax is not None else float(np.percentile(F, clip))

    # evenly spaced bands; band width set by the interior depth (~n_inside
    # stripes inside), 0 forced onto a band edge so the lightest tone hugs the
    # surface (after Feng & Crane 2024, Fig. 1).
    dband = abs(vmin) / n_inside
    neg = np.arange(0.0, vmin - dband, -dband)[::-1]   # [.., -2d, -d, 0]
    pos = np.arange(dband, vhi + dband, dband)         # [d, 2d, ..]
    levels = np.concatenate([neg, pos])
    reds = ["#F7D2D2", "#E06666"]                      # light pink / rose
    blues = ["#BBD6F2", "#2E6FB0"]                     # light / deep blue
    colors = []
    for a, b in zip(levels[:-1], levels[1:]):
        mid = 0.5 * (a + b)
        k = int(abs(mid) / dband)                      # 0 = nearest surface
        colors.append((blues if mid < 0 else reds)[k % 2])
    cf = ax.contourf(xc, yc, F, levels=levels, colors=colors, extend="max")

    if show_grid:                                      # true cell grid
        ax.vlines(xe, ye[0], ye[-1], colors="0.45", lw=0.25, alpha=0.5, zorder=1.5)
        ax.hlines(ye, xe[0], xe[-1], colors="0.45", lw=0.25, alpha=0.5, zorder=1.5)

    # the surface: bold black zero contour with a lime outline on top
    ax.contour(xc, yc, F, levels=[0.0], colors="k", linewidths=1.8)
    ax.contour(xc, yc, F, levels=[0.0], colors="lime", linewidths=0.8)
    ax.set_aspect("equal")
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", nargs="?", default="output/wing_sdf.vtk")
    ap.add_argument("--axis", default="all", choices=["xy", "xz", "yz", "all"])
    ap.add_argument("--index", type=int, default=None, help="slice index (default: center)")
    ap.add_argument("--levels", type=int, default=6, help="iso-distance stripes inside the body")
    ap.add_argument("--clip", type=float, default=99.0,
                    help="color-scale percentile of the distance (default 99)")
    ap.add_argument("--supersample", type=int, default=4,
                    help="sub-cell upsampling factor for smooth bands (default 4; 1 = off)")
    ap.add_argument("--no-grid", dest="grid", action="store_false", help="hide cell grid")
    ap.add_argument("--no-fill", dest="fill", action="store_false",
                    help="plot the raw narrowband instead of a reconstructed full field")
    ap.add_argument("--vmax", type=float, default=None, help="clip the outer range to vmax")
    ap.add_argument("--out", default=None, help="output png (default: alongside the vtk)")
    args = ap.parse_args()

    vol, origin, dx = read_vtk(args.vtk)
    print(f"{args.vtk}: dims {vol.shape[::-1]} (x,y,z)  dx={dx:.4g}  "
          f"range [{vol.min():.4g}, {vol.max():.4g}]")

    axes = ["xy", "xz", "yz"] if args.axis == "all" else [args.axis]

    # extract first so the figure can be sized to each slice's true aspect ratio
    # (a long thin airfoil in a square axes gets letterboxed and the grid mushes)
    data = [extract_slice(vol, origin, dx, axis, args.index) for axis in axes]
    base = 4.0                                            # axes height, inches
    widths = [base * float(np.clip((xc[-1] - xc[0]) / (yc[-1] - yc[0]), 0.3, 8.0))
              for _, xc, yc, _, _ in data]
    fig, axs = plt.subplots(1, len(axes), squeeze=False,
                            figsize=(sum(widths) + 1.4, base),
                            gridspec_kw={"width_ratios": widths, "wspace": 0.15})
    cf = None
    for a, axis, (Z, xc, yc, xl, yl) in zip(axs[0], axes, data):
        cf = plot_slice(a, Z, xc, yc, dx, args.levels, args.grid, args.fill,
                        args.clip, args.supersample, args.vmax)
        a.set_title(f"{axis} slice")
        a.set_xlabel(xl); a.set_ylabel(yl)

    fig.colorbar(cf, ax=axs[0].tolist(), fraction=0.025, pad=0.01,
                 label="signed distance")
    base_name = args.vtk.rsplit(".", 1)[0]
    out = args.out or f"{base_name}_contours.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
