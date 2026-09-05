#ifndef CUT_CLIP_H
#define CUT_CLIP_H
// ---------------------------------------------------------------------------
//  Host-only 2-D cell clipper: the FLUID region of an axis-aligned cell against
//  a closed polyline body, computed from the segments themselves.  No level set
//  and no polynomial fit: the result is exact for the polyline, whatever its
//  thickness (including a zero-thickness plate), and a cell crossed by more than
//  one wall comes back as more than one fluid loop -- that is the split-cell
//  detector.
//
//  Construction (the classical cut-cell walk, Berger / Aftosmis):
//    1. every body segment is clipped to the CLOSED cell box (Liang-Barsky);
//       consecutive clipped pieces joined at a vertex inside the box form a
//       CHAIN with one entry and one exit point on the cell boundary.  A vertex
//       exactly on a face does not break a chain, and a segment lying along a
//       face stays in its chain -- both degeneracies fall out of the closed box.
//    2. chains are oriented FLUID ON THE LEFT.
//    3. loops: leave a chain at its exit, walk the cell boundary counter-
//       clockwise (cell interior on the left) to the nearest chain entry, follow
//       that chain, ... until back at the start.  Each loop is one connected
//       fluid piece; the boundary walks are exactly the open parts of the faces.
//    4. a chain with no entry/exit (whole body inside one cell) is a hole.
//  Everything is accumulated edge by edge: shoelace area, first moments, per-
//  face open length + tangential first moment, the wall vector sum(A n) and the
//  wall centroid.  Doubles throughout; the geometry is preprocessing.
// ---------------------------------------------------------------------------
#include <cmath>

struct ClipLoop {
  double area, mx, my;          // area and first moments (centroid = m/area)
  double faceLen[4];            // open length on 0=low-x 1=high-x 2=low-y 3=high-y
  double faceMom[4];            // tangential first moment of the open part (physical coord)
  double wallVx, wallVy;        // sum over wall edges of the OUTWARD (from fluid) A n
  double wallLen, wallMx, wallMy;
  int    nIv[4];                // open INTERVALS per face (tangential coord, ascending)
  double iv[4][4][2];           // [face][interval][lo,hi]; the face-segment pairing needs them
  double intLen, intMx, intMy;  // INTERNAL face: the extension of a slit tip to the boundary
  double intNx, intNy;          // its outward normal times length (from this piece into the other)
  // the loop POLYGON, counter-clockwise (fluid on the left): vertex v starts
  // edge v -> v+1 (closed), ek = 0 wall edge, 1 open cell-boundary edge,
  // 2 internal (slit-tip extension).  The P1 cut elements build their volume
  // rule and wall quadrature from it; vOvf = capacity hit (polygon incomplete).
  int    nv;  bool vOvf;
  double vx[160], vy[160];  signed char ek[160];
};

struct ClipResult {
  int  nLoop;                   // 0 with nHole 0 => the body does not cross this cell
  ClipLoop loop[8];
  int  nHole;  double holeArea, holeMx, holeMy;
  bool overflow, bad;           // capacity hit / topology error (loop failed to close)
};

namespace cutclip {

static const int MAXCH = 256, MAXV = 65536;   // host-only static buffers (~1 MB)

// cell-boundary parameter s in [0,P): bottom (0..dx), right, top, left.
struct Box {
  double x0, y0, x1, y1, dx, dy, P;
  double sOf(double x, double y, int face) const {
    // face: 0 = x0, 1 = x1, 2 = y0, 3 = y1 (the Liang-Barsky plane order)
    double s;
    if      (face == 2) s = x - x0;                    // bottom
    else if (face == 1) s = dx + (y - y0);             // right
    else if (face == 3) s = dx + dy + (x1 - x);        // top
    else                s = 2*dx + dy + (y1 - y);      // left
    if (s >= P) s -= P; if (s < 0) s = 0;
    return s;
  }
  int nearestFace(double x, double y) const {
    double d[4] = { fabs(x-x0), fabs(x-x1), fabs(y-y0), fabs(y-y1) };
    int f = 0; for (int k = 1; k < 4; k++) if (d[k] < d[f]) f = k;
    return f;
  }
  void point(double s, double &x, double &y) const {
    if (s < dx)               { x = x0 + s;             y = y0; }
    else if (s < dx + dy)     { x = x1;                 y = y0 + (s - dx); }
    else if (s < 2*dx + dy)   { x = x1 - (s - dx - dy); y = y1; }
    else                      { x = x0;                 y = y1 - (s - 2*dx - dy); }
  }
  int faceAt(double s) const {              // walk-face index of parameter s (start of a piece)
    if (s < dx) return 2; if (s < dx + dy) return 1; if (s < 2*dx + dy) return 3; return 0;
  }
};

struct Chain { int v0, nv; double sIn, sOut; bool closed; };

// Liang-Barsky against the CLOSED box.  Returns false if the segment misses the
// box or only touches it at a point.  faceIn/faceOut = clipping plane (0..3) or
// -1 when that end is a vertex inside the box.
inline bool clipSeg(const Box &B, double ax, double ay, double bx, double by,
                    double &t0, double &t1, int &faceIn, int &faceOut) {
  const double dx = bx - ax, dy = by - ay;
  const double p[4] = { -dx, dx, -dy, dy };
  const double q[4] = { ax - B.x0, B.x1 - ax, ay - B.y0, B.y1 - ay };
  t0 = 0; t1 = 1; faceIn = -1; faceOut = -1;
  for (int k = 0; k < 4; k++) {
    if (p[k] == 0) { if (q[k] < 0) return false; continue; }
    const double t = q[k]/p[k];
    if (p[k] < 0) { if (t > t0) { t0 = t; faceIn = k; } }
    else          { if (t < t1) { t1 = t; faceOut = k; } }
  }
  if (t0 >= t1) return false;
  if (t0 <= 0) { t0 = 0; faceIn = -1; }
  if (t1 >= 1) { t1 = 1; faceOut = -1; }
  return true;
}

inline void addEdge(ClipLoop &L, double xa, double ya, double xb, double yb) {
  const double cr = xa*yb - xb*ya;
  L.area += 0.5*cr;
  L.mx   += (xa + xb)*cr/6.0;
  L.my   += (ya + yb)*cr/6.0;
}
// record edge a -> b (its start vertex and kind) in the loop polygon
inline void pushEdge(ClipLoop &L, double xa, double ya, int kind) {
  if (L.nv >= 160) { L.vOvf = true; return; }
  L.vx[L.nv] = xa; L.vy[L.nv] = ya; L.ek[L.nv] = (signed char)kind; L.nv++;
}

// walk the boundary CCW from s by length d, accumulating faces + shoelace
inline void walk(const Box &B, ClipLoop &L, double s, double d) {
  // corners over TWO laps: a walk may wrap past s = P and still cross corners
  const double corner[8] = { B.dx, B.dx + B.dy, 2*B.dx + B.dy, B.P,
                             B.P + B.dx, B.P + B.dx + B.dy, B.P + 2*B.dx + B.dy, 2*B.P };
  double xa, ya; B.point(s, xa, ya);
  double sc = s; const double sEnd = s + d;
  while (true) {
    // next corner strictly beyond sc
    double sn = sEnd;
    for (int k = 0; k < 8; k++) if (corner[k] > sc && corner[k] < sn) sn = corner[k];
    const double len = sn - sc;
    if (len > 0) {
      double sMod = sc; if (sMod >= B.P) sMod -= B.P;
      const int wf = B.faceAt(sMod);                 // 2 bottom, 1 right, 3 top, 0 left
      const int f  = (wf == 0) ? 0 : (wf == 1) ? 1 : (wf == 2) ? 2 : 3;
      double xb, yb; double sb = sn; if (sb >= B.P) sb -= B.P;
      // sb == 0 after wrap means the low-left corner: point(0) gives it
      B.point(sb, xb, yb);
      L.faceLen[f] += len;
      const double tmid = (f <= 1) ? 0.5*(ya + yb) : 0.5*(xa + xb);
      L.faceMom[f] += len*tmid;
      if (L.nIv[f] < 4) {
        const double ta = (f <= 1) ? ya : xa, tb = (f <= 1) ? yb : xb;
        L.iv[f][L.nIv[f]][0] = fmin(ta, tb); L.iv[f][L.nIv[f]][1] = fmax(ta, tb);
        L.nIv[f]++;
      }
      addEdge(L, xa, ya, xb, yb);
      pushEdge(L, xa, ya, 1);
      xa = xb; ya = yb;
    }
    sc = sn;
    if (sc >= sEnd) break;
  }
}

// seg: 2*n doubles, closed polyline (last -> first implied).
// fluidLeftForward: true if walking the polyline forward keeps the FLUID on the
//   left (CW body with solid inside, or CCW loop bounding the fluid).
inline void clipCell(const Box &B, const double *seg, int n, bool fluidLeftForward,
                     ClipResult &R) {
  R.nLoop = 0; R.nHole = 0; R.holeArea = R.holeMx = R.holeMy = 0;
  R.overflow = false; R.bad = false;
  static double vx[MAXV], vy[MAXV];
  static bool   eop[MAXV];         // edge v -> v+1 is an OPEN internal face (slit-tip extension)
  static Chain ch[MAXCH];
  int nv = 0, nch = 0;
  bool pendingSpur = false; double spurEx = 0, spurEy = 0; int spurFace = -1;
  for (int v = 0; v < MAXV; v++) eop[v] = false;
  // A slit TIP inside the cell (the polyline reverses at a vertex) is extended
  // along its tangent to the cell boundary: the two sides of a zero-thickness
  // body then become separate pieces, and the extension is an OPEN face between
  // them (a flux face, not a wall) -- the user's "aperture with its split self".
  auto rayToBox = [&](double px0, double py0, double ddx, double ddy, double &ex, double &ey, int &face) -> bool {
    double sBest = 1e300; face = -1;
    if (ddx > 0) { const double t = (B.x1 - px0)/ddx; if (t > 0 && t < sBest) { sBest = t; face = 1; } }
    if (ddx < 0) { const double t = (B.x0 - px0)/ddx; if (t > 0 && t < sBest) { sBest = t; face = 0; } }
    if (ddy > 0) { const double t = (B.y1 - py0)/ddy; if (t > 0 && t < sBest) { sBest = t; face = 3; } }
    if (ddy < 0) { const double t = (B.y0 - py0)/ddy; if (t > 0 && t < sBest) { sBest = t; face = 2; } }
    if (face < 0) return false;
    ex = px0 + sBest*ddx; ey = py0 + sBest*ddy;
    if (face == 0) ex = B.x0; if (face == 1) ex = B.x1; if (face == 2) ey = B.y0; if (face == 3) ey = B.y1;
    return true;
  };

  // find a segment that starts a chain (rejected, or entering from outside)
  int start = -1;
  for (int e = 0; e < n; e++) {
    const int f = (e + 1 == n) ? 0 : e + 1;
    double t0, t1; int fi, fo;
    if (!clipSeg(B, seg[2*e], seg[2*e+1], seg[2*f], seg[2*f+1], t0, t1, fi, fo) || t0 > 0)
      { start = e; break; }
  }
  if (start < 0) {
    // every segment inside and continuous: the whole loop sits in this cell.
    // Solid inside the loop -> a hole in an otherwise full cell.  Fluid inside
    // the loop (a duct smaller than a cell) -> the loop IS the fluid piece.
    ClipLoop H = ClipLoop();
    for (int e = 0; e < n; e++) {
      const int f = (e + 1 == n) ? 0 : e + 1;
      addEdge(H, seg[2*e], seg[2*e+1], seg[2*f], seg[2*f+1]);
    }
    const bool ccw = H.area > 0;
    const bool fluidInside = (ccw == fluidLeftForward);
    if (!fluidInside) {
      R.nHole = 1; R.holeArea = fabs(H.area); R.holeMx = H.mx; R.holeMy = H.my;
      if (H.area < 0) { R.holeMx = -R.holeMx; R.holeMy = -R.holeMy; }
      return;
    }
    ClipLoop &L = R.loop[0]; L = ClipLoop();
    for (int e = 0; e < n; e++) {                 // traverse with the fluid on the left
      const int ea = ccw ? e : ((e + 1 == n) ? 0 : e + 1);
      const int eb = ccw ? ((e + 1 == n) ? 0 : e + 1) : e;
      const double xa = seg[2*ea], ya = seg[2*ea+1], xb = seg[2*eb], yb = seg[2*eb+1];
      addEdge(L, xa, ya, xb, yb);
      pushEdge(L, xa, ya, 0);
      const double ex = xb - xa, ey = yb - ya, len = sqrt(ex*ex + ey*ey);
      L.wallVx += ey; L.wallVy += -ex; L.wallLen += len;
      L.wallMx += 0.5*(xa + xb)*len; L.wallMy += 0.5*(ya + yb)*len;
    }
    R.nLoop = 1;
    return;
  }

  // ---- chains, in polyline order -------------------------------------------
  bool open = false;
  for (int q = 0; q < n; q++) {
    const int e = (start + q) % n, f = (e + 1 == n) ? 0 : e + 1;
    const double ax = seg[2*e], ay = seg[2*e+1], bx = seg[2*f], by = seg[2*f+1];
    if (ax == bx && ay == by) continue;                      // duplicate point
    double t0, t1; int fi, fo;
    const bool ok = clipSeg(B, ax, ay, bx, by, t0, t1, fi, fo);
    if (!ok) { open = false; continue; }
    // A piece shorter than roundoff (a vertex one ulp inside a face, then the
    // polyline leaves) is a duplicate point, not a chain: skipping it leaves the
    // chain state alone -- an entering sliver is replaced by the next piece
    // starting at its inside vertex, an exiting one by the provisional exit.
    if ((t1 - t0)*sqrt((bx-ax)*(bx-ax) + (by-ay)*(by-ay)) <= 1e-12*(B.dx + B.dy)) continue;
    if (t0 > 0 || !open) {                                   // new chain
      if (nch >= MAXCH || nv + 3 > MAXV) { R.overflow = true; return; }
      Chain &C = ch[nch++]; C.v0 = nv; C.nv = 0; C.closed = false;
      if (pendingSpur && t0 == 0) {
        // the return leg of a slit tip: start on the boundary at the spur's end
        // and come back along the OPEN extension to the tip
        vx[nv] = spurEx; vy[nv] = spurEy; eop[nv] = true; nv++; C.nv++;
        C.sIn = B.sOf(spurEx, spurEy, spurFace);
        pendingSpur = false;
      } else {
        C.sIn = B.sOf(ax + t0*(bx - ax), ay + t0*(by - ay), fi >= 0 ? fi : B.nearestFace(ax + t0*(bx - ax), ay + t0*(by - ay)));
      }
      const double px = ax + t0*(bx - ax), py = ay + t0*(by - ay);
      vx[nv] = px; vy[nv] = py; nv++; C.nv++;
      open = true;
    }
    pendingSpur = false;
    Chain &C = ch[nch-1];
    if (nv + 2 > MAXV) { R.overflow = true; return; }
    const double qx = ax + t1*(bx - ax), qy = ay + t1*(by - ay);
    vx[nv] = qx; vy[nv] = qy; nv++; C.nv++;
    if (t1 < 1) {                                            // exits here
      C.sOut = B.sOf(qx, qy, fo);
      open = false;
    } else {
      const int nf = B.nearestFace(qx, qy);
      C.sOut = B.sOf(qx, qy, nf);
      // A vertex lying ON the boundary ends the chain: the polyline may turn
      // back there (a slit tip on a face -- then the two sides are separate
      // pieces) or pass through it (then the tracing continues into the next
      // chain at the coincident point, see the twin test in the loop walk).
      const double dOn = (nf == 0) ? fabs(qx - B.x0) : (nf == 1) ? fabs(qx - B.x1)
                       : (nf == 2) ? fabs(qy - B.y0) : fabs(qy - B.y1);
      if (dOn <= 1e-12*(B.dx + B.dy)) open = false;
      else {
        // slit TIP inside the cell?  (next non-degenerate segment anti-parallel)
        int f2 = f, g = (f2 + 1 == n) ? 0 : f2 + 1; int guard = 0;
        while (guard++ < n && seg[2*g] == seg[2*f2] && seg[2*g+1] == seg[2*f2+1]) { f2 = g; g = (g + 1 == n) ? 0 : g + 1; }
        const double dx0 = bx - ax, dy0 = by - ay, l0 = sqrt(dx0*dx0 + dy0*dy0);
        const double dx1 = seg[2*g] - seg[2*f2], dy1 = seg[2*g+1] - seg[2*f2+1], l1 = sqrt(dx1*dx1 + dy1*dy1);
        if (l0 > 0 && l1 > 0 && (dx0*dx1 + dy0*dy1)/(l0*l1) < -(1.0 - 1e-9)) {
          double ex, ey; int ef;
          if (rayToBox(qx, qy, dx0/l0, dy0/l0, ex, ey, ef)) {
            eop[nv-1] = true;                                 // edge tip -> E is OPEN
            vx[nv] = ex; vy[nv] = ey; nv++; C.nv++;
            C.sOut = B.sOf(ex, ey, ef);
            open = false;
            pendingSpur = true; spurEx = ex; spurEy = ey; spurFace = ef;
          }
        }
      }
    }
  }
  if (nch == 0) return;

  // ---- orientation: fluid on the left ---------------------------------------
  if (!fluidLeftForward) {
    for (int c = 0; c < nch; c++) {
      Chain &C = ch[c];
      for (int a = C.v0, b = C.v0 + C.nv - 1; a < b; a++, b--) {
        double t = vx[a]; vx[a] = vx[b]; vx[b] = t;
        t = vy[a]; vy[a] = vy[b]; vy[b] = t;
      }
      // edge flags: edge (a,a+1) becomes edge (n-2-a, n-1-a); re-key them
      { static bool tmp[MAXV];
        for (int a = 0; a + 1 < C.nv; a++) tmp[a] = eop[C.v0 + a];
        for (int a = 0; a + 1 < C.nv; a++) eop[C.v0 + (C.nv - 2 - a)] = tmp[a]; }
      const double t = C.sIn; C.sIn = C.sOut; C.sOut = t;
    }
  }

  // ---- loops ------------------------------------------------------------------
  static bool used[MAXCH];
  for (int c = 0; c < nch; c++) used[c] = false;
  for (int c0 = 0; c0 < nch; c0++) {
    if (used[c0]) continue;
    if (R.nLoop >= 8) { R.overflow = true; return; }
    ClipLoop &L = R.loop[R.nLoop];
    L = ClipLoop();
    int cur = c0; int guard = 0;
    while (true) {
      used[cur] = true;
      const Chain &C = ch[cur];
      for (int v = C.v0; v + 1 < C.v0 + C.nv; v++) {
        addEdge(L, vx[v], vy[v], vx[v+1], vy[v+1]);
        pushEdge(L, vx[v], vy[v], eop[v] ? 2 : 0);
        const double ex = vx[v+1] - vx[v], ey = vy[v+1] - vy[v];
        const double len = sqrt(ex*ex + ey*ey);
        if (eop[v]) {                                        // open extension: an internal face
          L.intNx += ey; L.intNy += -ex;
          L.intLen += len;
          L.intMx  += 0.5*(vx[v] + vx[v+1])*len;
          L.intMy  += 0.5*(vy[v] + vy[v+1])*len;
          continue;
        }
        L.wallVx += ey; L.wallVy += -ex;                     // outward normal of a CCW loop edge
        L.wallLen += len;
        L.wallMx  += 0.5*(vx[v] + vx[v+1])*len;
        L.wallMy  += 0.5*(vy[v] + vy[v+1])*len;
      }
      // nearest entry counter-clockwise from this exit
      int best = -1; double bd = 1e300;
      // direction of this chain's LAST edge (for the twin test below)
      double ex0 = 0, ey0 = 0;
      { const int a = C.v0 + C.nv - 2, b = C.v0 + C.nv - 1;
        ex0 = vx[b] - vx[a]; ey0 = vy[b] - vy[a];
        const double l = sqrt(ex0*ex0 + ey0*ey0); if (l > 0) { ex0 /= l; ey0 /= l; } }
      for (int c = 0; c < nch; c++) {
        double d = ch[c].sIn - C.sOut;
        if (d < 0) d += B.P;
        if (d >= B.P) d -= B.P;
        if (d < 1e-10*B.P) {
          // A chain entering exactly where this one exits: either the polyline
          // passes through a boundary vertex (chains are cut there) and the loop
          // simply continues into it, or it is the REVERSE TWIN of a zero-
          // thickness wall (anti-parallel first edge) -- turning into that would
          // close an empty loop and lose the fluid on both sides, so the
          // continuation is the boundary walk all the way round instead.  A slit
          // tip inside the cell is its own twin: whole boundary, one loop.
          const Chain &D = ch[c];
          double ex1 = vx[D.v0+1] - vx[D.v0], ey1 = vy[D.v0+1] - vy[D.v0];
          const double l = sqrt(ex1*ex1 + ey1*ey1); if (l > 0) { ex1 /= l; ey1 /= l; }
          const bool twin = (ex0*ex1 + ey0*ey1) < -(1.0 - 1e-9);
          if (twin) d = B.P; else d = 0;
        }
        if (d < bd) { bd = d; best = c; }
      }
      walk(B, L, C.sOut, bd);
      if (best == c0) break;
      if (used[best] || ++guard > MAXCH) { R.bad = true; break; }
      cur = best;
    }
    if (L.area < 0) R.bad = true;
    R.nLoop++;
  }
}

} // namespace cutclip
#endif
