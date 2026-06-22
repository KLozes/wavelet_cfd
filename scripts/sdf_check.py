#!/usr/bin/env python3
"""Validate a wavesdf sparse VTK against the analytic sphere SDF.

Reads the legacy-VTK BINARY UNSTRUCTURED_GRID written by ./wavesdf (one voxel
hexahedron per active narrowband cell, signed distance as cell data) and compares
each cell-center value against the exact distance to a sphere (default R=0.30,
center=(.5,.5,.5) in world coords -- assets/sphere.stl has bbox [0.2,0.8]^3, so
those hold regardless of the --margin used to build it).

    python3 scripts/sdf_check.py [output/sphere_sdf.vtk] [band_cutoff]
            [--R 0.30] [--center 0.5 0.5 0.5]

band_cutoff (optional): also report a row restricted to |sdf| < cutoff, i.e.
strictly inside the band -- excludes the beyond-band shell that whole-block fill
leaves at |sdf| up to ~band+blockDiag (and the sign noise right at the surface).
"""

import argparse
import numpy as np


def read_unstructured(path):
    """Return (cellCenters[nC,3], sdf[nC], dx) from a legacy VTK BINARY
    UNSTRUCTURED_GRID of hexahedra with big-endian data.

    Memory-maps the points / connectivity / sdf blocks in place rather than
    reading them into RAM (the strategy plot_sdf.open_blocks uses): the header is
    parsed only to find each block's byte offset, then np.memmap lets the OS page
    in just the bytes actually touched, so a multi-GB field costs almost no
    resident memory.  Cells are uniform voxels, so the center is the min corner
    (node 0) + dx/2 -- this touches only pts[node0] ([nC,3]) instead of gathering
    all 8 corners per cell ([nC,8,3]); the cell-types block is never read."""
    with open(path, "rb") as f:
        rd = f.readline                                     # raw bytes, no decode
        assert rd().startswith(b"# vtk"), "not a legacy VTK file"
        rd()                                                # comment
        assert rd().strip() == b"BINARY", "expected BINARY"
        assert b"UNSTRUCTURED_GRID" in rd(), "expected UNSTRUCTURED_GRID (sparse output)"
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
    dx = float(abs(pts[conn[0, 2], 0] - pts[conn[0, 1], 0]))   # corner spacing = cell size
    node0 = np.asarray(conn[:, 1])                          # min-corner point id per cell
    cen = np.asarray(pts[node0], np.float64) + 0.5*dx       # uniform voxel center
    return cen, np.asarray(sdf, np.float64), dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", nargs="?", default="output/sphere_sdf.vtk")
    ap.add_argument("cutoff", nargs="?", type=float, default=None,
                    help="also report cells strictly inside the band (|sdf| < cutoff)")
    ap.add_argument("--R", type=float, default=0.30)
    ap.add_argument("--center", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    args = ap.parse_args()

    cen, sdf, dx = read_unstructured(args.vtk)
    C = np.array(args.center)
    exact = np.linalg.norm(cen - C, axis=1) - args.R

    def report(tag, mask):
        if not mask.any():
            print(f"[{tag}] no cells"); return
        e = np.abs(sdf[mask] - exact[mask])
        se = int(np.sum(np.sign(sdf[mask]) != np.sign(exact[mask])))
        print(f"[{tag}] cells={int(mask.sum())}  "
              f"L1={e.mean():.4e} ({e.mean()/dx:.3f} cells)  "
              f"Linf={e.max():.4e} ({e.max()/dx:.3f} cells)  signErr={se}")

    print(f"{args.vtk}: {len(sdf)} active cells  dx={dx:.4g}  "
          f"sphere R={args.R} center={tuple(args.center)}")
    report("all active cells     ", np.ones(len(sdf), bool))
    if args.cutoff is not None:
        report(f"in band |sdf|<{args.cutoff:g}", np.abs(sdf) < args.cutoff)


if __name__ == "__main__":
    main()
