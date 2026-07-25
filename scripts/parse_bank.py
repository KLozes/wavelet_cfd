#!/usr/bin/env python3
"""
Parse an IDS aero "bank" file (e.g. bank_v98d.txt) and provide the shared
parsing / plotting / geometry utilities used by the mesh generators (hmesh.py,
omesh.py):

  * parse() the bank file into stations + blade rows (with each MASTER airfoil
    registered into the global meridional frame),
  * plot the meridional flow path and the airfoil cross-sections,
  * the meridional flow-path grid (flowpath_grid), node-spacing laws and the
    per-passage blade-to-blade block geometry (sector_block_at) that the H- and
    O-mesh builders consume.

Run directly to parse a bank file and write the flow-path + cross-section plots.

File layout (reverse-engineered from bank_v98d.txt)
--------------------------------------------------
* A handful of "global" throughflow stations, then 10 blade rows delimited by
  `BEGIN  <VANE|ROTOR>  <idx>  <nsect>` ... and a final `END`.
* Each station is introduced by a header line whose first token is one of
  FREE / INSI / VANE / ROTOR followed by  z_hub r_hub z_tip r_tip ...
  and contains labelled 13-value blocks (AXIAL VEL., RADIUS, Z, ...), one
  value per streamline (13 streamlines: 0,.05,.1,.2..1).
* Each blade row also carries 13 `MASTER` sections (one per streamline).
  A MASTER block is a closed airfoil contour stored as three concatenated
  arrays of equal length:  Z (axial), R (radius), T (tangential, R*theta).

Usage:  python3 parse_bank.py [bank_file]
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm

# matches every floating-point token, including E-notation values that are
# concatenated without separating whitespace, e.g. "-8.60E-02-8.91E-02"
FLOAT = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")

STATION_KEYS = {"FREE", "INSI", "VANE", "ROTOR"}


def is_label(line):
    s = line.lstrip()
    return bool(s) and s[0].isalpha()


def floats(line):
    return [float(x) for x in FLOAT.findall(line)]


def read_block(lines, i):
    """Read consecutive pure-data (float) lines starting at index i.
    Returns (numpy array of all floats, index of next non-data line)."""
    vals = []
    j = i
    while j < len(lines):
        if is_label(lines[j]):
            break
        if lines[j].strip():
            vals.extend(floats(lines[j]))
        j += 1
    return np.asarray(vals), j


def parse(path):
    with open(path) as fh:
        lines = fh.readlines()

    stations = []      # dicts: row, kind, R, Z, SL  (13-value arrays)
    rows = []          # dicts: label, type, index, nblades, sections[]
    cur_row = None
    cur_st = None
    n_edge, n_surface = 27, 80     # MASTER arc/surface counts (NO. COORDS)

    i = 0
    while i < len(lines):
        line = lines[i]
        if not is_label(line):
            i += 1
            continue

        s = line.strip()
        tok = s.split()
        head = tok[0]

        if head == "BEGIN":
            cur_row = {"label": f"{tok[1]} {tok[2]}", "type": tok[1],
                       "index": int(tok[2]), "nblades": None, "sections": []}
            rows.append(cur_row)
            cur_st = None
            i += 1
            continue

        if head == "END":
            break

        if head == "MASTER":
            arr, j = read_block(lines, i + 1)
            n = len(arr) // 3
            sec = {"index": int(tok[1]),
                   "z": arr[0:n], "r": arr[n:2 * n], "t": arr[2 * n:3 * n]}
            if cur_row is not None:
                cur_row["sections"].append(sec)
            i = j
            continue

        if head == "NO." and len(tok) > 1 and tok[1] == "BLADES":
            if cur_row is not None and cur_row["nblades"] is None:
                cur_row["nblades"] = float(tok[2])
            i += 1
            continue

        if head == "NO." and len(tok) > 1 and tok[1] == "COORDS":
            n_edge, n_surface = int(tok[2]), int(tok[3])
            i += 1
            continue

        # station header (FREE / INSI V / INSI R / VANE n / ROTOR n) -- has its
        # own coordinates on the same line, so it is a station, not a data block
        if head in STATION_KEYS and len(floats(s)) >= 4:
            hv = floats(s)
            cur_st = {"row": cur_row["label"] if cur_row else "GLOBAL",
                      "kind": head, "header": hv,
                      "R": None, "Z": None, "SL": None}
            stations.append(cur_st)
            i += 1
            continue

        # labelled 13-value data block belonging to the current station
        arr, j = read_block(lines, i + 1)
        if cur_st is not None and arr.size:
            if head == "RADIUS":
                cur_st["R"] = arr
            elif head == "Z":
                cur_st["Z"] = arr
            elif head == "STREAMLINE":
                cur_st["SL"] = arr
        i = j

    register(stations, rows, n_edge, n_surface)
    return stations, rows


def blade_le_te(stations, row):
    """The leading- and trailing-edge quasi-orthogonal stations of a row.

    LE is the row's first (FREE) station; TE is its VANE/ROTOR station.  Both
    carry full per-streamline R/Z in the file (the TE QO's flow data is written
    at the end of the blade-row block, after the MASTER airfoils)."""
    rs = sorted((s for s in stations
                 if s["row"] == row["label"] and s["R"] is not None),
                key=lambda s: np.mean(s["Z"]))
    if not rs:
        return None, None
    le = rs[0]
    te = next((s for s in rs if s["kind"] in ("VANE", "ROTOR")), None)
    return le, te


def register(stations, rows, n_edge, n_surface):
    """Move every MASTER airfoil section into the global meridional frame.

    A MASTER contour is stored as 2*n_surface + 2*n_edge points per coordinate,
    laid out  surface | edge-arc | surface | edge-arc.  The mid node of each arc
    is the true leading / trailing edge; one arc sits upstream (LE), one
    downstream (TE).  We align the LE arc to the row's FREE leading-edge QO
    station, which (the blade chord matching the LE->TE QO spacing) drops the TE
    arc onto the VANE/ROTOR trailing-edge QO station.
    """
    a = n_surface + n_edge // 2                 # mid of first edge arc
    b = 2 * n_surface + n_edge + n_edge // 2    # mid of second edge arc
    for row in rows:
        row["n_edge"], row["n_surface"] = n_edge, n_surface
        le = blade_le_te(stations, row)[0]
        if le is None:
            continue
        secs = sorted(row["sections"], key=lambda s: s["index"])
        le_i, te_i = (a, b) if secs[0]["z"][a] <= secs[0]["z"][b] else (b, a)
        dz = float(np.median([le["Z"][s["index"] - 1] - s["z"][le_i]
                              for s in secs]))
        for s in secs:
            s["zg"] = s["z"] + dz
            s["rg"] = s["r"]
            s["le_idx"], s["te_idx"] = le_i, te_i
        row["le_locus"] = np.array([(s["zg"][le_i], s["rg"][le_i]) for s in secs])
        row["te_locus"] = np.array([(s["zg"][te_i], s["rg"][te_i]) for s in secs])


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def row_color(rtype):
    return "#c0392b" if rtype == "ROTOR" else "#2471a3"   # rotor red, vane blue


def short_name(row):
    return ("R" if row["type"] == "ROTOR" else "V") + str(row["index"])


def trace_edge(streamline, p_from, p_to):
    """Polyline along a streamline (sz, sr) between two endpoints, keeping the
    interior streamline points so the edge follows the flow path exactly."""
    sz, sr = streamline
    lo, hi = sorted((p_from[0], p_to[0]))
    m = (sz > lo) & (sz < hi)
    pts = sorted(zip(sz[m], sr[m]))
    if p_from[0] > p_to[0]:
        pts = pts[::-1]
    return [(float(z), float(r)) for z, r in pts]


def on_line(streamline, z):
    """Radius of a streamline at axial position z (linear interpolation)."""
    sz, sr = streamline
    return float(np.interp(z, sz, sr))


def blade_polygon(row, hub_sl, cas_sl):
    """Meridional blade footprint.  The leading- and trailing-edge sides come
    from the blade sections; the hub and tip sides trace the hub and casing
    flow-path lines exactly.  The four blade corners are snapped onto the
    endwalls so the hub/tip edges follow the flow path with no overshoot."""
    le = [tuple(p) for p in row["le_locus"]]
    te = [tuple(p) for p in row["te_locus"]]
    le[0] = (le[0][0], on_line(hub_sl, le[0][0]))      # hub-LE  -> hub line
    te[0] = (te[0][0], on_line(hub_sl, te[0][0]))      # hub-TE  -> hub line
    le[-1] = (le[-1][0], on_line(cas_sl, le[-1][0]))   # tip-LE  -> casing line
    te[-1] = (te[-1][0], on_line(cas_sl, te[-1][0]))   # tip-TE  -> casing line
    tip = trace_edge(cas_sl, le[-1], te[-1])   # tip-LE -> tip-TE along casing
    hub = trace_edge(hub_sl, te[0], le[0])     # hub-TE -> hub-LE along hub
    return np.array(le + tip + te[::-1] + hub)


N_BLADE = 50          # streamwise cells across each blade
N_UPSTREAM = 8        # streamwise cells, upstream gap section (mix-plane -> LE)
N_DOWNSTREAM = 8      # streamwise cells, downstream gap section (TE -> mix-plane)
N_SPAN = 20           # spanwise nodes, hub -> casing
N_SECTOR = 8       # cells across the sector (theta direction)

SPACING = "chebyshev"     # node distribution: "uniform" or "cosine" (== "chebyshev");
#                        cosine family = full cosine between two walls, half
#                        cosine for the one-sided gap / theta sections


_COSINE = ("cosine", "chebyshev", "cos")     # cosine-family spacings


def _span_nodes(n, spacing="uniform"):
    """Node fractions in [0, 1], clustered toward *both* ends (walls) for the
    cosine family ('cosine'/'chebyshev', a full cosine 0.5(1-cos pi t))."""
    if spacing in _COSINE:
        return 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / (n - 1)))
    return np.linspace(0.0, 1.0, n)


def _one_sided(n, spacing, wall):
    """Node fractions in [0, 1] clustered toward a single wall for the cosine
    family -- a *half cosine* (cos over [0, pi/2]) -- expanding to the opposite
    (periodic / mixing-plane) end.  wall='lo' clusters at 0, 'hi' at 1."""
    if spacing not in _COSINE:
        return np.linspace(0.0, 1.0, n)
    j = np.arange(n)
    if wall == "lo":
        return 1.0 - np.cos(0.5 * np.pi * j / (n - 1))
    return np.sin(0.5 * np.pi * j / (n - 1))


def _walls(stations):
    """Hub and casing wall curves (z, r), from the station endpoints."""
    full = [s for s in stations if s["R"] is not None and s["Z"] is not None]
    hub = np.array(sorted((s["Z"][0], s["R"][0]) for s in full))
    cas = np.array(sorted((s["Z"][-1], s["R"][-1]) for s in full))
    return hub, cas


def _span_fractions(stations):
    """Spanwise streamline fractions (the LE/TE node distribution)."""
    for s in stations:
        if s.get("SL") is not None:
            return np.asarray(s["SL"], float)
    return np.linspace(0.0, 1.0, 13)


def _edge_nodes(locus, hub, cas, u):
    """LE/TE node curve resampled uniformly along the hub->tip direction.

    Parametrising by the projection onto the hub-to-tip vector (rather than by
    radius) keeps the spacing well-behaved for any quasi-orthogonal
    orientation, including the near-radial spans of centrifugal stages."""
    e = np.asarray(locus, float).copy()
    z_h, z_t = e[0, 0], e[-1, 0]
    e[0] = [z_h, np.interp(z_h, hub[:, 0], hub[:, 1])]     # hub on wall
    e[-1] = [z_t, np.interp(z_t, cas[:, 0], cas[:, 1])]    # tip on casing
    d = e[-1] - e[0]
    t = (e - e[0]) @ d / (d @ d)            # projection onto hub->tip direction
    t = np.maximum.accumulate(t)            # guard monotonicity (hooked hubs)
    t = (t - t[0]) / (t[-1] - t[0])
    return np.column_stack([np.interp(u, t, e[:, 0]),
                            np.interp(u, t, e[:, 1])])


def _wall_qo(z0, z1, hub, cas, frac):
    """Straight inlet/outlet QO node curve between the two wall endpoints."""
    p0 = np.array([z0, np.interp(z0, hub[:, 0], hub[:, 1])])
    p1 = np.array([z1, np.interp(z1, cas[:, 0], cas[:, 1])])
    return p0 + np.outer(frac, p1 - p0)


def _tfi(A, B, s, hub, cas, frac):
    """Transfinite interpolation between boundary node-curves A and B,
    conforming to the hub/casing walls.  s gives the streamwise node fractions
    (incl. both ends); returns the leading columns (the B column is left to the
    caller)."""
    P00, P01, P10, P11 = A[0], A[-1], B[0], B[-1]
    cols = []
    for u in s[:-1]:
        zh = P00[0] + u * (P10[0] - P00[0])
        zc = P01[0] + u * (P11[0] - P01[0])
        Bo = np.array([zh, np.interp(zh, hub[:, 0], hub[:, 1])])
        To = np.array([zc, np.interp(zc, cas[:, 0], cas[:, 1])])
        v = frac[:, None]
        corner = ((1 - u) * (1 - v) * P00 + (1 - u) * v * P01
                  + u * (1 - v) * P10 + u * v * P11)
        cols.append((1 - u) * A + u * B + (1 - v) * Bo + v * To - corner)
    return cols


def _ordered_blades(rows):
    return sorted(rows, key=lambda r: float(np.mean(r["le_locus"][:, 0])))


def streamwise_boundaries(rows, stations):
    """Ordered streamwise boundaries along the flow path -- inlet, then each
    blade's LE and TE, the mid-gap mixing plane between consecutive blades, and
    the outlet -- as (z, r) loci, plus the (n_cells, code) of every segment.

    Shared by the meridional flow-path mesh and the blade-to-blade sector mesh
    so both use the same axial discretisation.  code: 0 gap, 1 vane, 2 rotor."""
    blades = _ordered_blades(rows)
    cols = sorted((s for s in stations if s["R"] is not None),
                  key=lambda c: np.mean(c["Z"]))
    inlet = np.column_stack([cols[0]["Z"], cols[0]["R"]])
    outlet = np.column_stack([cols[-1]["Z"], cols[-1]["R"]])

    bnds = [{"locus": inlet, "role": "inlet"}]
    for i, row in enumerate(blades):
        bnds.append({"locus": row["le_locus"], "role": "LE", "row": row})
        bnds.append({"locus": row["te_locus"], "role": "TE", "row": row})
        if i < len(blades) - 1:
            mix = 0.5 * (row["te_locus"] + blades[i + 1]["le_locus"])
            bnds.append({"locus": mix, "role": "MIX"})
    bnds.append({"locus": outlet, "role": "outlet"})

    # (n_cells, code, streamwise mode): 'blade' clusters at both LE & TE walls,
    # 'hi'/'lo' cluster one-sided at the blade wall (LE/TE), expanding to the
    # mixing plane.
    segs = []
    for A, B in zip(bnds, bnds[1:]):
        if A["role"] == "LE" and B["role"] == "TE":
            segs.append((N_BLADE, 2 if A["row"]["type"] == "ROTOR" else 1, "blade"))
        elif B["role"] == "LE":                      # inlet/mix -> LE (wall at end)
            segs.append((N_UPSTREAM, 0, "hi"))
        else:                                        # TE -> mix/outlet (wall at start)
            segs.append((N_DOWNSTREAM, 0, "lo"))
    return bnds, segs


def flowpath_grid(rows, stations, n_span=N_SPAN, spacing=SPACING):
    """Build the meridional flow-path node grid Zg, Rg of shape (n_span, n_col),
    its per-cell type, and the streamwise column range owned by each blade
    (its upstream gap + blade + downstream gap, mixing-plane to mixing-plane).
    The sector meshes reuse this grid so their (z, r) ride the same meshlines."""
    hub, cas = _walls(stations)
    u = _span_nodes(n_span, spacing)
    bnds, segs = streamwise_boundaries(rows, stations)
    edges = [_edge_nodes(b["locus"], hub, cas, u) for b in bnds]

    columns, ctype = [], []
    for A, B, (n, code, smode) in zip(edges, edges[1:], segs):
        s = (_span_nodes(n + 1, spacing) if smode == "blade"
             else _one_sided(n + 1, spacing, smode))
        columns.extend(_tfi(A, B, s, hub, cas, u))
        ctype.extend([code] * n)
    columns.append(edges[-1])

    Zg = np.array([c[:, 0] for c in columns]).T     # (n_span, n_col)
    Rg = np.array([c[:, 1] for c in columns]).T

    bcol = [0]                                       # column index of each bnd
    for n, _, _ in segs:
        bcol.append(bcol[-1] + n)
    le_bi = {b["row"]["label"]: i for i, b in enumerate(bnds) if b.get("role") == "LE"}
    te_bi = {b["row"]["label"]: i for i, b in enumerate(bnds) if b.get("role") == "TE"}
    # (c0, c1, le_col, te_col): column slice for the blade + its local LE/TE cols
    blade_range = {}
    for lbl in le_bi:
        c0 = bcol[le_bi[lbl] - 1]
        c1 = bcol[te_bi[lbl] + 1] + 1
        blade_range[lbl] = (c0, c1, bcol[le_bi[lbl]] - c0, bcol[te_bi[lbl]] - c0)
    return dict(Zg=Zg, Rg=Rg, ctype=np.array(ctype), span=u,
                blade_range=blade_range, hub=hub, cas=cas)


def plot_mesh(ax, stations, rows, n_span=N_SPAN, spacing=SPACING):
    """Structured H-mesh bounded by the hub & casing walls (see flowpath_grid)."""
    hub, cas = _walls(stations)
    g = flowpath_grid(rows, stations, n_span, spacing)
    Zg, Rg, ctype = g["Zg"], g["Rg"], g["ctype"]
    ns, nL = Zg.shape
    C = np.tile(ctype, (ns - 1, 1))

    cmap = ListedColormap(["white", row_color("VANE"), row_color("ROTOR")])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(Zg, Rg, C, cmap=cmap, norm=norm, alpha=0.55,
                  edgecolors="0.25", linewidth=0.35, zorder=1)
    ax.plot(hub[:, 0], hub[:, 1], color="0.0", lw=2.2, zorder=3)
    ax.plot(cas[:, 0], cas[:, 1], color="0.0", lw=2.2, zorder=3)
    ax.set_xlabel("axial  Z")
    ax.set_ylabel("radius  R")
    ax.set_title(f"Flow-path mesh  —  blade={N_BLADE}, up={N_UPSTREAM}, "
                 f"down={N_DOWNSTREAM}, span={ns}  ({(nL - 1) * (ns - 1)} cells)")
    ax.set_aspect("equal")


def mixing_planes(stations, rows, n_span=N_SPAN, spacing=SPACING):
    """Mid-gap QO node curves at every rotor<->vane interface."""
    hub, cas = _walls(stations)
    u = _span_nodes(n_span, spacing)
    out = []
    for A, B in zip(_ordered_blades(rows), _ordered_blades(rows)[1:]):
        if A["type"] == B["type"]:
            continue
        te = _edge_nodes(A["te_locus"], hub, cas, u)
        le = _edge_nodes(B["le_locus"], hub, cas, u)
        mid = 0.5 * (te + le)
        out.append((mid[:, 0], mid[:, 1]))
    return out


def draw_mixing_planes(axes, mps):
    for z, r in mps:
        for ax in axes:
            ax.plot(z, r, color="0.1", lw=1.3, ls="--", dashes=(5, 3),
                    zorder=5, solid_capstyle="round")


def plot_flowpath(stations, rows, fname):
    full = [st for st in stations if st["R"] is not None and st["Z"] is not None]
    nsl = max(len(st["R"]) for st in full)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=True)

    # streamline k = streamline-k point of every station, ordered along the path
    sl = []
    for k in range(nsl):
        pts = sorted((full[q]["Z"][k], full[q]["R"][k]) for q in range(len(full)))
        sl.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))

    for k in range(nsl):
        ax.plot(sl[k][0], sl[k][1], color="0.45", lw=0.7, zorder=1)

    # endwalls: hub (k=0) and casing (k=last) drawn bold
    for k in (0, nsl - 1):
        ax.plot(sl[k][0], sl[k][1], color="0.0", lw=2.2, zorder=3)

    # blade rows: one filled polygon patch per row, with a horizontal
    # two-line block label (name over blade count)
    for row in rows:
        col = row_color(row["type"])
        poly = blade_polygon(row, sl[0], sl[nsl - 1])
        ax.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=0.55,
                             edgecolor=col, lw=1.0, zorder=4,
                             joinstyle="round"))
        cz, cr = poly[:, 0].mean(), poly[:, 1].mean()
        txt = short_name(row) + (f"\n{int(row['nblades'])}"
                                 if row["nblades"] else "")
        ax.text(cz, cr, txt, ha="center", va="center", fontsize=8.5,
                family="monospace", weight="bold", color="white", zorder=6,
                linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none",
                          alpha=0.9))

    ax.set_ylabel("radius  R")
    ax.set_title("Meridional flow path  —  HiPR V98d compressor")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(handles=[Line2D([], [], color="0.15", lw=2, label="endwalls"),
                       Line2D([], [], color="0.75", lw=1, label="streamlines"),
                       Line2D([], [], color=row_color("ROTOR"), lw=6, alpha=.4,
                              label="rotor"),
                       Line2D([], [], color=row_color("VANE"), lw=6, alpha=.4,
                              label="stator/vane")],
              loc="lower right", fontsize=8, framealpha=0.9)

    plot_mesh(ax2, stations, rows)

    draw_mixing_planes([ax, ax2], mixing_planes(stations, rows))

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print(f"wrote {fname}")


def meridional_coord(z, r, le_idx, te_idx):
    """Chordwise meridional coordinate: the projection of every contour point
    onto the leading-edge -> trailing-edge direction in the (z, r) plane.

    Unlike an arc-length fold this stays monotonic and smooth through the blunt
    leading/trailing-edge arcs, so the blade-to-blade outline closes cleanly
    without LE/TE hooks while still accounting for the radius change."""
    le = np.array([z[le_idx], r[le_idx]])
    te = np.array([z[te_idx], r[te_idx]])
    d = te - le
    return ((z - le[0]) * d[0] + (r - le[1]) * d[1]) / np.hypot(*d)


def plot_cross_sections(rows, fname):
    ncol = 5
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).ravel()

    span_lab = ["hub", "mid", "tip"]
    span_col = ["#2471a3", "#27ae60", "#c0392b"]

    for ax, row in zip(axes, rows):
        secs = row["sections"]
        picks = sorted({0, len(secs) // 2, len(secs) - 1})
        for p, lab, col in zip(picks, span_lab, span_col):
            sec = secs[p]
            li, ti = sec["le_idx"], sec["te_idx"]
            m = meridional_coord(sec["z"], sec["r"], li, ti)
            t = sec["t"]
            mc = np.append(m, m[0])
            tc = np.append(t, t[0])
            ax.fill(mc, tc, color=col, alpha=0.12, lw=0)
            ax.plot(mc, tc, color=col, lw=1.3,
                    label=f"{lab} (sec {sec['index']})")
            # the LE / TE nodes that define the chord (projection) direction
            ax.plot(m[li], t[li], "o", mfc=col, mec="k", mew=0.4, ms=3, zorder=5)
            ax.plot(m[ti], t[ti], "s", mfc=col, mec="k", mew=0.4, ms=3, zorder=5)
        ax.set_title(row["label"].replace(" ", "") +
                     (f"   {int(row['nblades'])} blades"
                      if row["nblades"] else ""),
                     fontsize=10, color=row_color(row["type"]), weight="bold")
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.4)
        ax.tick_params(labelsize=8)
        h, lbls = ax.get_legend_handles_labels()
        h += [Line2D([], [], ls="none", marker="o", mfc="0.7", mec="k", ms=3),
              Line2D([], [], ls="none", marker="s", mfc="0.7", mec="k", ms=3)]
        lbls += ["LE node", "TE node"]
        ax.legend(h, lbls, fontsize=7, loc="best")

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle("Airfoil cross-sections  —  blade-to-blade view "
                 "(meridional distance  m   vs   tangential  Rθ)", fontsize=13)
    fig.supxlabel("meridional distance  m", fontsize=10)
    fig.supylabel("tangential  Rθ", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, dpi=300)
    print(f"wrote {fname}")


# ---------------------------------------------------------------------------
# blade-to-blade (theta - axial) sector mesh
# ---------------------------------------------------------------------------



def _blade_surfaces(sec):
    """Upper (suction) and lower (pressure) blade surfaces as dicts of arrays
    (m, z, r, t) from LE to TE, plus chord and LE/TE point dicts."""
    li, ti = sec["le_idx"], sec["te_idx"]
    m = meridional_coord(sec["z"], sec["r"], li, ti)
    n = len(m)
    fwd = np.array([(li + k) % n for k in range((ti - li) % n + 1)])
    bwd = np.array([(li - k) % n for k in range((li - ti) % n + 1)])

    def pack(idx):
        return {"m": m[idx], "z": sec["z"][idx], "r": sec["r"][idx],
                "t": sec["t"][idx]}

    A, B = pack(fwd), pack(bwd)
    upper, lower = (A, B) if A["t"].mean() > B["t"].mean() else (B, A)
    le = {k: float(sec[k][li]) for k in ("z", "r", "t")}
    te = {k: float(sec[k][ti]) for k in ("z", "r", "t")}
    return upper, lower, float(m[ti]), le, te


def _control(s):
    """Thomas-Middlecoff-style 1-D control function -s''/s' for a desired node
    distribution s (computational index -> fraction); zero at the ends.  Adding
    c * r_xi to the Laplacian makes the elliptic solve hold that spacing."""
    s = np.asarray(s, float)
    c = np.zeros_like(s)
    if len(s) > 2:
        d1 = 0.5 * (s[2:] - s[:-2])
        d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]
        c[1:-1] = -d2 / np.where(np.abs(d1) < 1e-12, 1e-12, d1)
    return c


def sector_block_at(z, r, sec, n_blades, le_col, te_col,
                    n_sector=N_SECTOR, spacing=SPACING):
    """Two periodic H-blocks for one blade passage at one span node.

    The streamwise (z, r) come from the flow-path grid (radius rides the
    meshline through gaps and blade).  The blade is parametrised by chord
    fraction anchored on the grid's LE/TE columns, so its thickness closes
    exactly at those columns.  The passage spans one pitch (2*pi/N): each block
    runs from its blade surface to the periodic boundary at camber +/- pitch/2
    (one pitch apart), and the camber/dividing line continues tangent into the
    gaps."""
    upper, lower, chord_c, le, te = _blade_surfaces(sec)
    p_le = np.array([z[le_col], r[le_col]])
    p_te = np.array([z[te_col], r[te_col]])
    dhat = (p_te - p_le) / np.hypot(*(p_te - p_le))
    m = (z - p_le[0]) * dhat[0] + (r - p_le[1]) * dhat[1]   # chordwise position
    chord = m[te_col]
    frac = m / chord                                        # 0 at LE, 1 at TE col
    blade = (frac >= 0.0) & (frac <= 1.0)

    # surfaces vs their own chord fraction -> evaluated at the column fractions
    ou, ol = np.argsort(upper["m"]), np.argsort(lower["m"])
    suf, tuf = upper["m"][ou] / chord_c, upper["t"][ou]
    slf, tlf = lower["m"][ol] / chord_c, lower["t"][ol]
    fc = np.clip(frac, 0.0, 1.0)
    ts, tp = np.interp(fc, suf, tuf), np.interp(fc, slf, tlf)
    half_th = np.where(blade, 0.5 * (ts - tp) / r, 0.0)     # 0 at LE & TE columns

    # camber theta over the blade (fine in fraction) for stable end slopes
    ff = np.linspace(0.0, 1.0, 41)
    ob = np.argsort(frac[blade])
    rb = np.interp(ff, frac[blade][ob], r[blade][ob])
    thcf = 0.5 * (np.interp(ff, suf, tuf) + np.interp(ff, slf, tlf)) / rb
    k = max(1, int(0.12 * len(ff)))
    s_le = (thcf[k] - thcf[0]) / (ff[k] * chord)            # d(theta)/dm at LE
    s_te = (thcf[-1] - thcf[-1 - k]) / ((1.0 - ff[-1 - k]) * chord)

    # camber on the blade, tangent extension into the gaps
    th_div = np.interp(fc, ff, thcf)
    th_div = np.where(frac < 0.0, thcf[0] + s_le * m,
                      np.where(frac > 1.0, thcf[-1] + s_te * (m - chord), th_div))
    th_s, th_p = th_div + half_th, th_div - half_th        # suction / pressure
    half = np.pi / n_blades                                # half pitch
    sj = _one_sided(n_sector + 1, spacing, "lo")           # cluster at surface
    ns1 = n_sector + 1

    blocks = {}
    for name, surf_th, peri in (("upper", th_s, th_div + half),
                                ("lower", th_p, th_div - half)):
        theta = surf_th[None, :] + sj[:, None] * (peri - surf_th)[None, :]
        blocks[name] = {"m": np.tile(m, (ns1, 1)), "z": np.tile(z, (ns1, 1)),
                        "r": np.tile(r, (ns1, 1)), "theta": theta}
    return dict(blocks=blocks, pitch=2.0 * np.pi / n_blades, chord=chord,
                m_in=float(m[0]), m_out=float(m[-1]), blade=blade,
                le_col=le_col, te_col=te_col)


def _span_param(row):
    """Spanwise fraction of each MASTER section, matching the flow-path mesh's
    _edge_nodes parametrisation (projection onto the LE hub->tip direction)."""
    le = row["le_locus"]
    d = le[-1] - le[0]
    t = (le - le[0]) @ d / (d @ d)
    t = np.maximum.accumulate(t)
    return (t - t[0]) / (t[-1] - t[0])


def _interp_contour(row, sp, f):
    """Blade contour (z, r, t) interpolated onto span fraction f from the
    MASTER sections (parametrised by sp).  Uses the globalised axial coordinate
    zg so it shares the frame of the mixing-plane / LE-TE loci."""
    secs = sorted(row["sections"], key=lambda s: s["index"])
    Z = np.array([s["zg"] for s in secs])
    R = np.array([s["r"] for s in secs])
    T = np.array([s["t"] for s in secs])
    interp = lambda M: np.array([np.interp(f, sp, M[:, i]) for i in range(M.shape[1])])
    return {"z": interp(Z), "r": interp(R), "t": interp(T),
            "le_idx": secs[0]["le_idx"], "te_idx": secs[0]["te_idx"]}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bank_v98d.txt"
    stations, rows = parse(path)

    print(f"parsed {len(stations)} stations and {len(rows)} blade rows:")
    for row in rows:
        nb = row["nblades"]
        print(f"  {row['label']:<9} sections={len(row['sections']):>2} "
              f"blades={nb}")

    plot_flowpath(stations, rows, "flowpath.png")
    plot_cross_sections(rows, "airfoils.png")


if __name__ == "__main__":
    main()
