#!/usr/bin/env python3
"""
Build a watertight STL of a single blade from an IDS aero bank file, including
a TIP CLEARANCE GAP and a ROOT FILLET.

The bank file stores 13 MASTER airfoil sections per blade row on streamlines
running hub -> casing, as closed contours in (z, r, t) with t = r*theta
(parse_bank.py registers them into the global meridional frame).  This script
lofts those sections into a closed solid:

  TIP GAP     the span fraction whose mean clearance equals the requested gap
              is found by bisection, and the tip section's radius is then
              snapped onto r = r_casing(z) - gap so the clearance is constant
              along the chord rather than only correct on average.  The bank's
              airfoil stack usually stops a little inboard of the casing, so the
              bisection is allowed past f = 1: the loft is extrapolated linearly
              in span, carrying the section's own trend (chord shrinking, stagger
              rolling) instead of stretching the last section radially.

  ROOT FILLET a rolling-ball blend of radius R between the blade flank and the
              hub.  At height s above the hub the fillet stands off the blade
              surface by

                  w(s) = R - sqrt(R^2 - (R - s)^2),      0 <= s <= R,

              (w(R) = 0 at the tangency, w(0) = R at the foot) which is applied
              as an OUTWARD offset of the section in its own (meridional,
              tangential) plane.  Outward offsets never fold on a convex
              stretch, so only the concave (pressure-side) curvature bounds R;
              the script detects a fold exactly, by counting section edges that
              reverse direction under the offset.

  ROOT/TIP    the hub section is snapped onto r = r_hub(z) and the tip section
              onto the clearance surface, so the solid is closed by flat caps
              that lie exactly on the flow-path walls.

The result is a closed, outward-oriented triangle soup written as binary STL,
suitable for the CutFEM solver (./wavefem), which needs a watertight surface
for its inside/outside ray test.

Examples
--------
  # list the rows in the bank
  python3 scripts/blade_stl.py --list

  # rotor 1 with a 0.5% span tip gap and a 3% span root fillet
  python3 scripts/blade_stl.py --row "ROTOR 1" --gap-frac 0.005 --fillet-frac 0.03

  # explicit dimensions (bank length units), finer surface
  python3 scripts/blade_stl.py --row "ROTOR 4" --gap 0.02 --fillet 0.06 \
      --refine 2 --nspan 80 --out assets/rotor4.stl

  # every row, default clearances
  python3 scripts/blade_stl.py --all
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_bank as pb


# ---------------------------------------------------------------------------
#  section helpers
# ---------------------------------------------------------------------------

def wall_r(wall, z):
    """Radius of a flow-path wall (hub or casing) at axial position z."""
    return np.interp(z, wall[:, 0], wall[:, 1])


def interp_contour(row, sp, f):
    """Blade contour at span fraction f, LINEARLY EXTRAPOLATING beyond the
    outermost MASTER section.

    parse_bank._interp_contour clamps (np.interp), which matters here: the bank's
    airfoil stack typically stops a little inboard of the casing, so building a
    tip at a prescribed clearance means continuing the loft outward.  Doing that
    by extrapolating (z, r, t) carries the section's own trend -- chord shrinking,
    stagger rolling -- whereas simply stretching the last section radially would
    keep a hub-like planform out at the tip.
    """
    secs = sorted(row["sections"], key=lambda s: s["index"])
    Z = np.array([s["zg"] for s in secs])
    R = np.array([s["r"] for s in secs])
    T = np.array([s["t"] for s in secs])

    def col(M):
        if 0.0 <= f <= 1.0:
            return np.array([np.interp(f, sp, M[:, i]) for i in range(M.shape[1])])
        j0, j1 = (-2, -1) if f > 1.0 else (0, 1)     # end pair to extrapolate on
        w = (f - sp[j0])/(sp[j1] - sp[j0])
        return M[j0] + w*(M[j1] - M[j0])

    return {"z": col(Z), "r": col(R), "t": col(T),
            "le_idx": secs[0]["le_idx"], "te_idx": secs[0]["te_idx"],
            "n_edge": row["n_edge"], "n_surface": row["n_surface"]}


def section_loop(cont, refine=1):
    """Closed section polygon, resampled from the bank's own contour structure.

    A MASTER contour is laid out  surface | edge-arc | surface | edge-arc  with
    n_surface and n_edge points each (80 and 27 here), i.e. the blunt leading and
    trailing edges get a quarter of the points over a few percent of the
    perimeter.  Resampling each of those four segments separately -- by arc
    length, keeping the raw proportions -- preserves that.

    This matters more than it looks: projecting onto the LE->TE chord direction
    instead (the natural thing for a blade-to-blade mesh) collapses each edge arc
    onto a single point, turning the nose and tail into knife edges.  The loft
    would still be watertight, but the fillet offset folds there, and a stress
    analysis would see a false singularity at the trailing edge.
    """
    z, r, t = cont["z"], cont["r"], cont["t"]
    ns, ne = cont["n_surface"], cont["n_edge"]
    n = len(z)
    P = np.column_stack([z, r, t])

    # cumulative arc length around the closed loop
    d = np.roll(P, -1, axis=0) - P
    seg = np.linalg.norm(d, axis=1)
    Lc = np.concatenate([[0.0], np.cumsum(seg)])          # n+1 knots, Lc[-1] = perimeter
    Pc = np.vstack([P, P[0]])

    bounds = [0, ns, ns + ne, 2*ns + ne, n]               # segment start indices
    counts = [ns, ne, ns, ne]
    out = []
    for k in range(4):
        m = max(2, int(round(refine*counts[k])))
        a, b = Lc[bounds[k]], Lc[bounds[k+1]]
        u = np.linspace(a, b, m, endpoint=False)          # exclude the shared join
        out.append(np.column_stack([np.interp(u, Lc, Pc[:, j]) for j in range(3)]))
    return np.vstack(out)


def loop_ehat(loop, cont, refine):
    """Unit LE->TE direction in the (z, r) meridional plane, for the section's
    own in-plane frame."""
    ns, ne = cont["n_surface"], cont["n_edge"]
    m0 = int(round(refine*ns)) + int(round(refine*ne))//2               # LE arc mid
    m1 = 2*int(round(refine*ns)) + int(round(refine*ne)) \
         + int(round(refine*ne))//2                                     # TE arc mid
    m1 = min(m1, len(loop) - 1)
    d = np.array([loop[m1, 0] - loop[m0, 0], loop[m1, 1] - loop[m0, 1]])
    nrm = np.hypot(*d)
    return d/nrm if nrm > 0 else np.array([1.0, 0.0])


def outward_offset(loop, w, ehat):
    """Offset a closed section outward by w (scalar or per-vertex), in the
    section's own plane.

    The section is flattened to (m, t) with m the projection onto the LE->TE
    meridional direction `ehat` = (e_z, e_r); the 2-D outward normal there is
    mapped back to (z, r, t).  Vertex normals are the angle-independent average
    of the two adjacent edge normals, which keeps convex corners (the LE and TE
    arcs) smooth.
    """
    w = np.broadcast_to(np.asarray(w, float), (len(loop),))
    if not np.any(w > 0):
        return loop.copy()
    z, r, t = loop[:, 0], loop[:, 1], loop[:, 2]
    m = z*ehat[0] + r*ehat[1]
    P = np.column_stack([m, t])
    n = len(P)

    e = np.roll(P, -1, axis=0) - P                       # edge vectors
    L = np.hypot(e[:, 0], e[:, 1])
    L[L < 1e-14] = 1e-14
    en = np.column_stack([e[:, 1]/L, -e[:, 0]/L])        # right-hand edge normal
    vn = en + np.roll(en, 1, axis=0)                     # vertex normal
    vl = np.hypot(vn[:, 0], vn[:, 1])
    vl[vl < 1e-14] = 1e-14
    vn /= vl[:, None]

    # Orient outward.  en = (e_y, -e_x) is the right-hand normal, which points
    # OUT of a counter-clockwise loop (check on the unit square: bottom edge
    # e = (1,0) -> en = (0,-1)), so only a clockwise loop needs the flip.
    area2 = np.sum(P[:, 0]*np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1)*P[:, 1])
    if area2 < 0:
        vn = -vn

    dm, dt = w*vn[:, 0], w*vn[:, 1]
    out = loop.copy()
    out[:, 0] = z + dm*ehat[0]
    out[:, 1] = r + dm*ehat[1]
    out[:, 2] = t + dt
    return out


def fold_count(loop, off, ehat):
    """Number of edges that REVERSED direction under the offset.

    This is the exact local test for a folded (self-overlapping) blend, and it
    is what actually matters -- a curvature bound computed on the resampled
    contour mostly measures resampling wiggle at the clustered LE/TE, where
    consecutive points are ~1e-3 chord apart.
    """
    def flat(P):
        return np.column_stack([P[:, 0]*ehat[0] + P[:, 1]*ehat[1], P[:, 2]])
    e0 = np.roll(flat(loop), -1, axis=0) - flat(loop)
    e1 = np.roll(flat(off), -1, axis=0) - flat(off)
    d = np.einsum('ij,ij->i', e0, e1)
    n0 = np.hypot(*e0.T)
    return int(np.sum(d[n0 > 1e-12] < 0))


def to_cartesian(loop):
    """(z, r, t=r*theta) -> (x=axial, y, z)."""
    z, r, t = loop[:, 0], loop[:, 1], loop[:, 2]
    th = t/np.where(np.abs(r) < 1e-12, 1e-12, r)
    return np.column_stack([z, r*np.cos(th), r*np.sin(th)])


# ---------------------------------------------------------------------------
#  blade construction
# ---------------------------------------------------------------------------

def ear_clip(P):
    """Triangulate a simple polygon (N,2) by ear clipping.

    Returns index triangles wound consistently with the INPUT ordering, which is
    what lets the caller close the lofted tube: the flank quads leave the hub
    loop traversed one way and the tip loop the other, so the two caps must be
    wound oppositely for the surface to be consistently oriented.
    """
    n = len(P)
    idx = list(range(n))
    area2 = float(np.sum(P[:, 0]*np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1)*P[:, 1]))
    flip = area2 < 0
    if flip:
        idx.reverse()

    def cross(o, a, b):
        return ((P[a][0]-P[o][0])*(P[b][1]-P[o][1])
                - (P[a][1]-P[o][1])*(P[b][0]-P[o][0]))

    def inside(a, b, c, p):
        d1 = cross(a, b, p); d2 = cross(b, c, p); d3 = cross(c, a, p)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    tris, guard = [], 0
    while len(idx) > 3 and guard < 4*n:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            a, b, c = idx[i-1], idx[i], idx[(i+1) % len(idx)]
            if cross(a, b, c) <= 0:                 # reflex (working CCW)
                continue
            if any(inside(a, b, c, q) for q in idx if q not in (a, b, c)):
                continue
            tris.append((a, b, c))
            idx.pop(i)
            clipped = True
            break
        if not clipped:                             # numerically stuck: fan out
            for i in range(1, len(idx)-1):
                tris.append((idx[0], idx[i], idx[i+1]))
            idx = []
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    if flip:
        tris = [(c, b, a) for (a, b, c) in tris]
    return tris


def build_blade(row, hub, cas, gap, fillet, refine, nspan, nfillet, verbose=True):
    """Loft one blade row's MASTER sections into a closed surface grid.

    The stack is trimmed POINTWISE, not by span station.  The MASTER sections
    sit on streamlines, which cut the hub obliquely -- on this compressor the
    height above the hub varies by 8% of span around a single "hub" section, with
    part of the root buried 0.18 below the wall.  Picking one span fraction and
    snapping it flat therefore leaves the neighbouring station still buried and
    the first slab of the loft folded under itself (measured: it destroyed more
    volume than the fillet added).

    So each contour index k gets its own trim fractions: f0[k] where the height
    above the hub crosses zero, f1[k] where the clearance below the casing
    crosses `gap`.  Sampling sigma in [0,1] between them traces the SAME flank
    surface -- only the curves we sample along change -- but the end stations are
    now the true blade/hub intersection and the true clearance surface, so both
    caps are exact and the fillet can be evaluated per POINT from that point's
    own height.

    Returns (LOOP, hub_tris, tip_tris): LOOP is (NS, NP, 3) Cartesian section
    polygons hub-first, and the cap index triangulations.
    """
    sp = pb._span_param(row)

    def loop_at(f):
        c = interp_contour(row, sp, f)
        return section_loop(c, refine), c

    lp0, c0 = loop_at(0.0)
    NP = len(lp0)

    # ---- stack the sections on an f-grid wide enough to bracket both trims ---
    f_lo, f_hi = 0.0, 1.0
    while f_lo > -1.0:
        lp, _ = loop_at(f_lo)
        if np.all(lp[:, 1] - wall_r(hub, lp[:, 0]) < 0):
            break
        f_lo -= 0.05
    while f_hi < 2.0:
        lp, _ = loop_at(f_hi)
        if np.all(wall_r(cas, lp[:, 0]) - lp[:, 1] < gap):
            break
        f_hi += 0.05

    M = 241
    fg = np.linspace(f_lo, f_hi, M)
    Z = np.zeros((M, NP)); R = np.zeros((M, NP)); T = np.zeros((M, NP))
    for j, f in enumerate(fg):
        lp, _ = loop_at(f)
        Z[j], R[j], T[j] = lp[:, 0], lp[:, 1], lp[:, 2]
    H = R - wall_r(hub, Z)                    # height above the hub
    C = wall_r(cas, Z) - R                    # clearance below the casing

    # ---- per-point trim fractions ------------------------------------------
    f0 = np.zeros(NP); f1 = np.zeros(NP)
    for k in range(NP):
        f0[k] = np.interp(0.0, np.maximum.accumulate(H[:, k]), fg)
        f1[k] = np.interp(-gap, np.maximum.accumulate(-C[:, k]), fg) if gap > 0 else fg[-1]
    f1 = np.maximum(f1, f0 + 1e-6)

    span = float(np.mean(np.array([np.interp(f1[k], fg, H[:, k]) for k in range(NP)])))

    # ---- station distribution: cluster where the fillet lives ---------------
    if fillet > 0:
        sf = min(fillet/max(span, 1e-9), 0.9)
        u = np.sin(0.5*np.pi*np.linspace(0.0, 1.0, nfillet))
        sig = np.concatenate([sf*u, np.linspace(sf, 1.0, nspan + 1)[1:]])
    else:
        sig = np.linspace(0.0, 1.0, nspan)

    NS = len(sig)
    LOOP = np.zeros((NS, NP, 3))
    flat = np.zeros((NS, NP, 2))
    nfold = 0

    for i, sg in enumerate(sig):
        fk = f0 + sg*(f1 - f0)                            # per-point span fraction
        loop = np.column_stack([
            np.array([np.interp(fk[k], fg, Z[:, k]) for k in range(NP)]),
            np.array([np.interp(fk[k], fg, R[:, k]) for k in range(NP)]),
            np.array([np.interp(fk[k], fg, T[:, k]) for k in range(NP)])])

        # exact end trims (kill interpolation drift at the caps)
        if i == 0:
            loop[:, 1] = wall_r(hub, loop[:, 0])
        elif i == NS - 1 and gap > 0:
            loop[:, 1] = wall_r(cas, loop[:, 0]) - gap

        ehat = loop_ehat(loop, c0, refine)

        # Rolling-ball fillet, evaluated at each POINT's own height above the
        # hub.  The offset is computed in the section's (m, t) chart, but
        # m = z*e_z + r*e_r has a radial component, so offsetting along it would
        # move the point in SPAN -- a fillet only ever grows along the wall, so
        # every point is put back at the height it started from.
        if fillet > 0:
            hpt = loop[:, 1] - wall_r(hub, loop[:, 0])
            w = fillet - np.sqrt(np.maximum(0.0, fillet**2 - (fillet - np.minimum(hpt, fillet))**2))
            w = np.where(hpt < fillet, w, 0.0)
            if np.any(w > 0):
                off = outward_offset(loop, w, ehat)
                off[:, 1] = wall_r(hub, off[:, 0]) + hpt
                nfold += fold_count(loop, off, ehat)
                loop = off

        LOOP[i] = to_cartesian(loop)
        flat[i] = np.column_stack([loop[:, 0]*ehat[0] + loop[:, 1]*ehat[1], loop[:, 2]])

    # caps: the flanks leave the hub loop traversed forwards and the tip loop
    # backwards, so the hub cap is wound the other way (see ear_clip).
    tip_tris = ear_clip(flat[-1])
    hub_tris = [(c, b, a) for (a, b, c) in ear_clip(flat[0])]

    if verbose:
        print(f"    exposed span      {span:.4f}   (trim f = {f0.min():.3f}..{f0.max():.3f}"
              f" at the hub, {f1.min():.3f}..{f1.max():.3f} at the tip)")
        print(f"    tip gap           {gap:.5f}  (span fraction {gap/span*100:.2f}%)")
        print(f"    root fillet       {fillet:.5f} (span fraction {fillet/span*100:.2f}%)")
        print(f"    sections          {NS} span x {NP} contour"
              + (f"  ({nfillet} in the fillet)" if fillet > 0 else ""))
        if nfold:
            print(f"    WARNING: the fillet offset folded {nfold} section edges -- the"
                  f" radius exceeds the\n             local concave curvature."
                  f"  Reduce --fillet.")
    return LOOP, hub_tris, tip_tris


def surface_triangles(LOOP, hub_tris, tip_tris):
    """Close the lofted sections into a watertight, outward-oriented triangle
    list: quad strips between consecutive span stations, plus the two caps."""
    NS, NP, _ = LOOP.shape
    tris = []
    for i in range(NS - 1):
        for k in range(NP):
            k2 = (k + 1) % NP
            a, b, c, d = LOOP[i, k], LOOP[i, k2], LOOP[i+1, k2], LOOP[i+1, k]
            tris.append((a, b, c))
            tris.append((a, c, d))
    for (x, y, z) in hub_tris:
        tris.append((LOOP[0, x], LOOP[0, y], LOOP[0, z]))
    for (x, y, z) in tip_tris:
        tris.append((LOOP[-1, x], LOOP[-1, y], LOOP[-1, z]))

    T = np.array(tris)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    T = T[np.linalg.norm(n, axis=1) > 1e-14]          # drop degenerate slivers

    vol = np.sum(np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])))/6.0
    if vol < 0:
        T = T[:, ::-1, :]
    return T


# ---------------------------------------------------------------------------
#  checks and output
# ---------------------------------------------------------------------------

def check_watertight(T, tol=1e-9):
    """Every edge must be shared by exactly two triangles.  The solver's
    inside/outside test is a ray-cast parity count, which silently gives wrong
    signs on an open surface -- so this is worth asserting, not assuming."""
    q = np.round(T.reshape(-1, 3)/tol).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    edges = np.vstack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    bad = int(np.sum(counts != 2))
    return bad, len(np.unique(idx))


def mesh_props(T):
    v = np.sum(np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])))/6.0
    a = 0.5*np.sum(np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1))
    return abs(v), a


def write_stl(path, T, name="blade"):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n/np.where(ln[:, None] < 1e-30, 1.0, ln[:, None])
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(T)))
        rec = np.zeros((len(T), 12), dtype="<f4")
        rec[:, 0:3] = n
        rec[:, 3:6] = T[:, 0]
        rec[:, 6:9] = T[:, 1]
        rec[:, 9:12] = T[:, 2]
        buf = np.zeros(len(T), dtype=[("d", "<f4", 12), ("a", "<u2")])
        buf["d"] = rec
        fh.write(buf.tobytes())


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bank", nargs="?", default=os.path.join(root, "assets/bank_v98d.txt"))
    ap.add_argument("--row", default="ROTOR 1", help='blade row label, e.g. "ROTOR 1"')
    ap.add_argument("--all", action="store_true", help="write every row")
    ap.add_argument("--list", action="store_true", help="list the rows and exit")
    ap.add_argument("--gap", type=float, default=None, help="tip clearance (bank units)")
    ap.add_argument("--gap-frac", type=float, default=0.005,
                    help="tip clearance as a fraction of span (default 0.5%%)")
    ap.add_argument("--fillet", type=float, default=None, help="root fillet radius")
    ap.add_argument("--fillet-frac", type=float, default=0.02,
                    help="root fillet as a fraction of span (default 2%%)")
    ap.add_argument("--refine", type=float, default=1.0,
                    help="contour point density, relative to the bank's own "
                         "214-point sections (default 1 = exactly the design points)")
    ap.add_argument("--nspan", type=int, default=60, help="span stations above the fillet")
    ap.add_argument("--nfillet", type=int, default=12, help="span stations in the fillet")
    ap.add_argument("--out", default=None, help="output STL (default assets/<row>.stl)")
    ap.add_argument("--outdir", default=os.path.join(root, "assets"))
    args = ap.parse_args()

    stations, rows = pb.parse(args.bank)
    if args.list:
        print(f"{args.bank}: {len(rows)} blade rows")
        hub, cas = pb._walls(stations)
        for r in rows:
            le = r["le_locus"]
            print(f"  {r['label']:<9} blades={int(r['nblades']):>3}  sections={len(r['sections']):>2}"
                  f"  LE r = {le[0,1]:.3f} .. {le[-1,1]:.3f}"
                  f"  z = {le[:,0].mean():.3f}")
        return

    hub, cas = pb._walls(stations)
    targets = rows if args.all else [r for r in rows if r["label"] == args.row]
    if not targets:
        print(f"no row named '{args.row}'; use --list")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    for row in targets:
        print(f"{row['label']}  ({int(row['nblades'])} blades)")
        # span, for the fractional gap / fillet options
        sp = pb._span_param(row)
        def h(f):
            lp = section_loop(interp_contour(row, sp, f), 0.25)
            return float(np.mean(lp[:, 1] - wall_r(hub, lp[:, 0])))
        # exposed span: the root normally starts below the hub line, so measure
        # from where the blade meets the wall (matches build_blade)
        f0 = 0.0
        if h(0.0) < 0.0:
            a, b = 0.0, 1.0
            for _ in range(50):
                m = 0.5*(a + b)
                a, b = (m, b) if h(m) < 0 else (a, m)
            f0 = 0.5*(a + b)
        span = h(1.0) - h(f0)
        gap = args.gap if args.gap is not None else args.gap_frac*span
        fil = args.fillet if args.fillet is not None else args.fillet_frac*span

        LOOP, hub_tris, tip_tris = build_blade(row, hub, cas, gap, fil,
                                               args.refine, args.nspan, args.nfillet)
        T = surface_triangles(LOOP, hub_tris, tip_tris)
        bad, nv = check_watertight(T)
        vol, area = mesh_props(T)
        print(f"    triangles         {len(T)}  ({nv} unique vertices)")
        print(f"    watertight        {'YES' if bad == 0 else f'NO ({bad} unpaired edges)'}")
        print(f"    volume / area     {vol:.6g} / {area:.6g}")

        name = args.out if (args.out and not args.all) else \
            os.path.join(args.outdir, row["label"].lower().replace(" ", "") + ".stl")
        write_stl(name, T, row["label"])
        print(f"    wrote             {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
