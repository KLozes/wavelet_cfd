#!/usr/bin/env python3
"""Plot SDF cross sections from a sparse wavesdf VTK (UNSTRUCTURED_GRID).

Reads the legacy-VTK unstructured grid written by ./wavesdf (one voxel hexahedron
per active narrowband cell, with the signed distance as cell data) and draws
axis-aligned cross sections as diverging filled contours (blue inside / red
outside) with a bold green zero level set marking the surface -- the style of the
wing figure.  Only the narrowband carries data; the far field (no cell) is left
blank.

By default only the narrowband shell is drawn (far field left blank).  Pass
--fill to reconstruct the full field beyond the band from the band's sign, or
--style topo for the topographic iso-distance bands (which implies --fill).

usage:
    python3 scripts/plot_sdf.py [output/foo_sdf.vtk]
            [--planes xy,xz]        # any of xy (planform), xz (airfoil), yz (section)
            [--at z=6.2,y=250]      # slice coords per const-axis (default: middle layer)
            [--style diverging|topo] [--fill] [--levels 20] [--out foo.png]

Defaults to output/sym_wing3_sdf.vtk and the planform + airfoil planes.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def open_blocks(path):
    """Memory-map the points / connectivity / sdf blocks of the legacy BINARY
    UNSTRUCTURED_GRID in place, *without* reading them into RAM.  Returns big-endian
    memmaps (pts[nPts,3], conn[nC,9], sdf[nC]); the OS pages in only the bytes that
    are actually touched, so a multi-GB file costs almost no resident memory until a
    slice gathers from it."""
    with open(path, "rb") as f:
        rd = f.readline                                     # raw bytes, no decode
        assert rd().startswith(b"# vtk"), "not a legacy VTK file"
        rd()                                                # comment
        assert rd().strip() == b"BINARY", "expected BINARY"
        assert b"UNSTRUCTURED_GRID" in rd(), "expected UNSTRUCTURED_GRID"
        t = rd().split(); assert t[0] == b"POINTS"; nPts = int(t[1])
        pts_off = f.tell()
        f.seek(nPts*3*4, 1); rd()                           # skip points + newline
        t = rd().split(); assert t[0] == b"CELLS"; nC = int(t[1]); sz = int(t[2])
        conn_off = f.tell()
        f.seek(sz*4, 1); rd()                               # skip connectivity + newline
        rd()                                                # CELL_TYPES line
        f.seek(nC*4, 1); rd()                               # skip cell types + newline
        rd()                                                # CELL_DATA
        rd(); rd()                                          # SCALARS, LOOKUP_TABLE
        sdf_off = f.tell()
    pts  = np.memmap(path, dtype=">f4", mode="r", offset=pts_off,  shape=(nPts, 3))
    conn = np.memmap(path, dtype=">i4", mode="r", offset=conn_off, shape=(nC, 9))
    sdf  = np.memmap(path, dtype=">f4", mode="r", offset=sdf_off,  shape=(nC,))
    return pts, conn, sdf


# plane -> (constant axis, horizontal axis, vertical axis, nice name, h label, v label)
PLANES = {
    "xy": (2, 0, 1, "planform", "x (chord)", "y (span)"),
    "xz": (1, 0, 2, "airfoil",  "x (chord)", "z (thickness)"),
    "yz": (0, 1, 2, "section",  "y (span)",  "z (thickness)"),
}


def reconstruct_full(G, dx):
    """Fill the far field (NaN, beyond the band) of a slice with a signed
    Euclidean distance reconstructed from the narrowband, so the whole slice
    carries the field (like a full SDF) instead of a band-only shell.  The band's
    own resolved distances are kept; only the missing far field is synthesized.

    Far regions touching the slice border are exterior (+), enclosed far regions
    are interior (-); the sign of the band cells fixes which is which."""
    from scipy import ndimage
    far = np.isnan(G)
    if not far.any():
        return G
    lbl, _ = ndimage.label(far)                            # connected far regions
    edge = set(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]).tolist()) - {0}
    exterior = np.isin(lbl, list(edge)) if edge else np.zeros_like(far)
    solid = (G < 0) | (far & ~exterior)                    # interior = neg band + enclosed far
    din = ndimage.distance_transform_edt(solid)
    dout = ndimage.distance_transform_edt(~solid)
    edt = np.where(solid, -(din - 0.5), dout - 0.5) * dx
    return np.where(far, edt, G)


def extract_slice(pts, conn, sdf, plane, coord=None):
    """Reconstruct a regular 2D array for one cross section, materializing only the
    cells that fall in the chosen layer (NaN outside the band).

    Cells sit on a uniform grid and node 0 of each voxel hexahedron is its min
    corner, so the cell center is corner0 + dx/2.  We read just the const-axis
    coordinate of every cell's corner0 to pick the layer, then gather full coords
    and sdf for that (small) subset -- the 64M-cell field never lands in RAM."""
    ca, ha, va, *_ = PLANES[plane]
    node0 = np.asarray(conn[:, 1])                          # min-corner point id per cell
    zc = np.asarray(pts[node0, ca], np.float64)            # const-axis coord of each cell
    layers = np.unique(zc)
    sel = layers[np.argmin(np.abs(layers - coord))] if coord is not None \
          else layers[len(layers)//2]                      # default: middle layer
    idx = np.flatnonzero(zc == sel)                        # cells in this slice (small)
    p0 = node0[idx]
    H = np.asarray(pts[p0, ha], np.float64)
    V = np.asarray(pts[p0, va], np.float64)
    S = np.asarray(sdf[idx], np.float64)
    dx = np.min(np.diff(np.unique(H)))                     # uniform cell size
    H += 0.5*dx; V += 0.5*dx; sel += 0.5*dx                # corner0 -> cell center
    ih = np.round((H - H.min())/dx).astype(int)
    iv = np.round((V - V.min())/dx).astype(int)
    G = np.full((iv.max()+1, ih.max()+1), np.nan)
    G[iv, ih] = S
    xs = H.min() + dx*np.arange(G.shape[1])
    ys = V.min() + dx*np.arange(G.shape[0])
    return xs, ys, G, sel


def _decorate(ax, plane, sel):
    ca, _, _, name, hl, vl = PLANES[plane]
    ax.set_aspect("equal")
    ax.set_title(f"{name}: SDF on {hl[0]}-{vl[0]} at {'xyz'[ca]}={sel:.1f}")
    ax.set_xlabel(hl); ax.set_ylabel(vl)


def plot_diverging(ax, xs, ys, G, plane, sel, nlev):
    """Diverging blue(inside)/red(outside) filled contours with a green zero set."""
    vmax = float(np.nanmax(np.abs(G)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, nlev + 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cf = ax.contourf(xs, ys, G, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(xs, ys, G, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    if np.nanmin(G) < 0 < np.nanmax(G):                    # surface present in slice
        ax.contour(xs, ys, G, levels=[0.0], colors="#00f5a0", linewidths=0.8)
    _decorate(ax, plane, sel)
    return cf


def plot_topo(ax, xs, ys, F, plane, sel, n_inside=6, supersample=3):
    """Evenly spaced iso-distance bands (two alternating tones per side, 0 on a
    band edge) -- the topographic 'signed-heat' style (after Feng & Crane 2024).
    Needs a full field (use with reconstruction); NaNs are treated as far field."""
    if supersample > 1:
        from scipy.ndimage import zoom
        F = zoom(np.nan_to_num(F, nan=float(np.nanmax(F))), supersample, order=1)
        xs = np.linspace(xs[0], xs[-1], F.shape[1])
        ys = np.linspace(ys[0], ys[-1], F.shape[0])
    vmin = float(np.nanmin(F)); vhi = float(np.nanpercentile(F, 99))
    dband = (abs(vmin) / n_inside) if vmin < 0 else (vhi / n_inside if vhi > 0 else 1.0)
    neg = np.arange(0.0, vmin - dband, -dband)[::-1]       # [.., -2d, -d, 0]
    pos = np.arange(dband, vhi + dband, dband)             # [d, 2d, ..]
    levels = np.concatenate([neg, pos])
    reds = ["#F7D2D2", "#E06666"]; blues = ["#BBD6F2", "#2E6FB0"]
    colors = []
    for a, b in zip(levels[:-1], levels[1:]):
        mid = 0.5 * (a + b); k = int(abs(mid) / dband)     # 0 = nearest surface
        colors.append((blues if mid < 0 else reds)[k % 2])
    cf = ax.contourf(xs, ys, F, levels=levels, colors=colors, extend="max")
    ax.contour(xs, ys, F, levels=[0.0], colors="k", linewidths=1.6)
    ax.contour(xs, ys, F, levels=[0.0], colors="#00f5a0", linewidths=0.9)
    _decorate(ax, plane, sel)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", nargs="?", default="output/sym_wing3_sdf.vtk")
    ap.add_argument("--planes", default="xy,xz",
                    help="comma list of xy (planform), xz (airfoil), yz (section)")
    ap.add_argument("--at", default="",
                    help="comma list of const-axis coords, e.g. z=6.2,y=250 (default: middle)")
    ap.add_argument("--style", choices=["diverging", "topo"], default="diverging",
                    help="diverging blue/red contours (default) or topographic iso-distance bands")
    ap.add_argument("--fill", action="store_true",
                    help="reconstruct the full field beyond the band (fills the far field)")
    ap.add_argument("--levels", type=int, default=20, help="contour bands (diverging style)")
    ap.add_argument("--out", default=None, help="output png (default: alongside the vtk)")
    ap.add_argument("--dpi", default="grid",
                    help="output resolution: an integer, or 'grid' (default) to match the "
                         "cell grid -- one output pixel per cell so nothing is downsampled")
    args = ap.parse_args()

    pts, conn, sdf = open_blocks(args.vtk)
    print(f"{args.vtk}: {len(sdf)} active cells  "
          f"sdf range [{float(sdf.min()):.4g}, {float(sdf.max()):.4g}]")

    coords = {}
    for tok in filter(None, args.at.split(",")):
        ax_name, val = tok.split("="); coords["xyz".index(ax_name.strip())] = float(val)

    planes = [p.strip() for p in args.planes.split(",") if p.strip()]
    fill = args.fill or args.style == "topo"               # topo needs a full field
    data = []
    for p in planes:
        xs, ys, G, sel = extract_slice(pts, conn, sdf, p, coords.get(PLANES[p][0]))
        if fill:
            G = reconstruct_full(G, xs[1] - xs[0])
        data.append((xs, ys, G, sel))

    plt.style.use("dark_background")
    base = 4.5                                              # axes height (inches)
    widths = [base * float(np.clip((xs[-1]-xs[0]) / (ys[-1]-ys[0]), 0.25, 8.0))
              for xs, ys, _, _ in data]
    fig, axs = plt.subplots(1, len(planes), squeeze=False,
                            figsize=(sum(widths) + 1.6*len(planes), base),
                            gridspec_kw={"width_ratios": widths, "wspace": 0.35})
    for a, p, (xs, ys, G, sel) in zip(axs[0], planes, data):
        if args.style == "topo":
            cf = plot_topo(a, xs, ys, G, p, sel)
        else:
            cf = plot_diverging(a, xs, ys, G, p, sel, args.levels)
        fig.colorbar(cf, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle("narrowband SDF cross sections  (green = zero level set)", y=1.02)

    if args.dpi == "grid":                                  # one output pixel per cell
        dpi = int(np.ceil(max(max(G.shape[1]/w, G.shape[0]/base)
                              for (_, _, G, _), w in zip(data, widths))))
    else:
        dpi = int(args.dpi)

    out = args.out or args.vtk.rsplit(".", 1)[0] + "_xsec.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"wrote {out} at {dpi} dpi")


if __name__ == "__main__":
    main()
