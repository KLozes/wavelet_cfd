#ifndef FEM_BLADE_GEOM_H
#define FEM_BLADE_GEOM_H

//
// Build the FULL-SPAN blade solid from a bank file's MASTER sections.
//
// Deliberately no trimming here: the loft runs from below the hub to above the
// casing and is capped at both ends, so it is a closed watertight solid whose
// signed distance is meaningful everywhere.  The hub cut, the tip clearance and
// the root fillet are then done as SDF algebra against the platform and shroud
// (see BladeSdf.cuh) rather than by cutting the surface -- which is what makes
// the fillet an exact rolling-ball blend along the true blade/platform
// intersection curve instead of a per-span-station offset.
//
// Frame: the machine axis is +Z.  X = r cos(theta), Y = r sin(theta),
// Z = z_axial, with theta = t/r from the bank's tangential coordinate t.
//

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "Bank.h"
#include "Vec3f.cuh"
#include "Features.h"

namespace blade {

// spanwise fraction of each MASTER section: the projection of the LE locus onto
// its own hub->tip direction (matches parse_bank._span_param, so the sections
// keep the same parametrisation the mesh generators use)
inline std::vector<double> spanParam(const bank::Row &row) {
  size_t n = row.leZ.size();
  std::vector<double> t(n, 0.0);
  if (n < 2) return t;
  double dz = row.leZ[n-1] - row.leZ[0], dr = row.leR[n-1] - row.leR[0];
  double dd = dz*dz + dr*dr;
  for (size_t i = 0; i < n; i++)
    t[i] = ((row.leZ[i] - row.leZ[0])*dz + (row.leR[i] - row.leR[0])*dr)/dd;
  for (size_t i = 1; i < n; i++) t[i] = std::max(t[i], t[i-1]);   // monotone
  double t0 = t[0], t1 = t[n-1];
  for (size_t i = 0; i < n; i++) t[i] = (t[i] - t0)/(t1 - t0);
  return t;
}

struct Contour {
  std::vector<double> z, r, t;
  int nEdge = 27, nSurface = 80;
};

// Blade contour at span fraction f, LINEARLY EXTRAPOLATING outside [0,1] so the
// loft can be continued past the outermost section (the airfoil stack normally
// stops a little inboard of the casing, and a little below the hub).
//
// The extrapolation base pair is chosen ~`extBase` of span apart, NOT the
// adjacent sections: the bank clusters streamlines near the walls (0, .05, .1,
// .2, ...), so extrapolating on the end pair multiplies the step between two
// sections 5% of span apart by a large factor and the section balloons -- which
// showed up as a spurious solid band right across the pitch.
inline Contour interpContour(const bank::Row &row, const std::vector<double> &sp,
                             double f, double extBase = 0.25) {
  const std::vector<bank::Section> &S = row.sections;
  size_t np = S[0].z.size(), ns = S.size();
  Contour c;
  c.nEdge = row.nEdge; c.nSurface = row.nSurface;
  c.z.resize(np); c.r.resize(np); c.t.resize(np);

  auto col = [&](std::vector<double> &out, int which) {
    if (f >= 0.0 && f <= 1.0) {
      size_t j = 1;
      while (j < ns - 1 && sp[j] < f) j++;
      double w = (f - sp[j-1])/(sp[j] - sp[j-1]);
      for (size_t i = 0; i < np; i++) {
        const bank::Section &A = S[j-1], &Bb = S[j];
        double a = which == 0 ? A.zg[i] : which == 1 ? A.r[i] : A.t[i];
        double b = which == 0 ? Bb.zg[i] : which == 1 ? Bb.r[i] : Bb.t[i];
        out[i] = a + w*(b - a);
      }
    } else {
      size_t j0, j1;
      if (f > 1.0) {                       // outboard: last section + one ~extBase in
        j1 = ns - 1; j0 = ns - 2;
        while (j0 > 0 && sp[j1] - sp[j0] < extBase) j0--;
      } else {                             // inboard: first section + one ~extBase out
        j0 = 0; j1 = 1;
        while (j1 + 1 < ns && sp[j1] - sp[j0] < extBase) j1++;
      }
      double w = (f - sp[j0])/(sp[j1] - sp[j0]);
      for (size_t i = 0; i < np; i++) {
        const bank::Section &A = S[j0], &Bb = S[j1];
        double a = which == 0 ? A.zg[i] : which == 1 ? A.r[i] : A.t[i];
        double b = which == 0 ? Bb.zg[i] : which == 1 ? Bb.r[i] : Bb.t[i];
        out[i] = a + w*(b - a);
      }
    }
  };
  col(c.z, 0); col(c.r, 1); col(c.t, 2);
  return c;
}

//
// Camber line of a blade SECTION, sampled at nff chord fractions 0..1.
//
// Faithful port of parse_bank._blade_surfaces + the camber computation that
// omesh.py uses (and which the passage centreline / periodic faces are built
// on).  The camber is the MIDPOINT of the two blade surfaces at each chord
// fraction -- 0.5*(theta_suction + theta_pressure) -- NOT the mean of all
// contour vertices: the midpoint is free of the blunt LE/TE arc artefacts that
// corrupt a vertex mean (they show up as a spurious dip in theta right at the
// leading edge).  Chord fraction is the meridional coordinate: the projection
// of each contour point onto the LE->TE direction in the (z, r) plane.
//
// Returns theta_c[k], and the camber's z_c[k], r_c[k] (taken from one surface;
// the two barely differ at a fixed chord fraction), for k in [0, nff).
//
inline void camberLine(const bank::Row &row, const std::vector<double> &sp,
                       double fspan, int nff,
                       std::vector<double> &thC, std::vector<double> &zC,
                       std::vector<double> &rC) {
  Contour c = interpContour(row, sp, fspan);
  int n = (int)c.z.size();
  int li = row.sections[0].leIdx, ti = row.sections[0].teIdx;
  // meridional coordinate: projection onto the LE->TE direction
  double dz = c.z[ti] - c.z[li], dr = c.r[ti] - c.r[li];
  double dd = std::hypot(dz, dr);
  dz /= dd; dr /= dd;
  std::vector<double> m(n);
  for (int i = 0; i < n; i++) m[i] = (c.z[i]-c.z[li])*dz + (c.r[i]-c.r[li])*dr;
  double chord = m[ti];

  // one surface: the monotone index path LE->TE (step +1 or -1), returned as
  // arrays sorted by chord fraction so they can be interpolated
  struct Surf { std::vector<double> m, t, z, r; };
  auto buildSurf = [&](int step) {
    int cnt = (step > 0 ? ((ti - li) % n + n) % n : ((li - ti) % n + n) % n) + 1;
    std::vector<int> idx(cnt);
    for (int k = 0; k < cnt; k++) idx[k] = ((li + step*k) % n + n) % n;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return m[a] < m[b]; });
    Surf s;
    for (int id : idx) {
      s.m.push_back(m[id]/chord); s.t.push_back(c.t[id]);
      s.z.push_back(c.z[id]);     s.r.push_back(c.r[id]);
    }
    return s;
  };
  Surf A = buildSurf(+1), Bs = buildSurf(-1);

  auto interp = [](const std::vector<double> &X, const std::vector<double> &Y, double x){
    if (x <= X.front()) return Y.front();
    if (x >= X.back())  return Y.back();
    size_t i = (size_t)(std::upper_bound(X.begin(), X.end(), x) - X.begin());
    double w = (x - X[i-1])/(X[i] - X[i-1]);
    return Y[i-1] + w*(Y[i] - Y[i-1]);
  };

  thC.resize(nff); zC.resize(nff); rC.resize(nff);
  for (int k = 0; k < nff; k++) {
    double ff = (double)k/(nff-1);
    double rk = interp(A.m, A.r, ff);
    double ta = interp(A.m, A.t, ff), tb = interp(Bs.m, Bs.t, ff);
    thC[k] = 0.5*(ta + tb)/std::max(1e-9, rk);   // camber theta = midpoint t / r
    zC[k]  = interp(A.m, A.z, ff);
    rC[k]  = rk;
  }
}

//
// Closed section polygon, resampled from the bank's own contour structure.
//
// The contour is  surface | edge-arc | surface | edge-arc.  Resampling each of
// those four segments separately -- by arc length, keeping the raw proportions
// -- preserves the blunt leading and trailing edges.  Projecting onto the
// LE->TE chord direction instead (the natural thing for a blade-to-blade mesh)
// collapses each edge arc onto a single point and turns the nose and tail into
// knife edges, which a stress analysis would read as a false singularity.
//
inline void sectionLoop(const Contour &c, double refine,
                        std::vector<double> &oz, std::vector<double> &orr,
                        std::vector<double> &ot) {
  size_t n = c.z.size();
  std::vector<double> Lc(n + 1, 0.0);
  for (size_t i = 0; i < n; i++) {
    size_t j = (i + 1) % n;
    double dz = c.z[j]-c.z[i], dr = c.r[j]-c.r[i], dt = c.t[j]-c.t[i];
    Lc[i+1] = Lc[i] + std::sqrt(dz*dz + dr*dr + dt*dt);
  }
  auto sample = [&](const std::vector<double> &V, double u) {
    size_t i = (size_t)(std::upper_bound(Lc.begin(), Lc.end(), u) - Lc.begin());
    if (i == 0) i = 1;
    if (i > n) i = n;
    double w = (Lc[i] - Lc[i-1] > 0) ? (u - Lc[i-1])/(Lc[i] - Lc[i-1]) : 0.0;
    return V[i-1] + w*(V[i % n] - V[i-1]);
  };

  int ns = c.nSurface, ne = c.nEdge;
  int bounds[5] = {0, ns, ns + ne, 2*ns + ne, (int)n};
  int counts[4] = {ns, ne, ns, ne};
  oz.clear(); orr.clear(); ot.clear();
  for (int k = 0; k < 4; k++) {
    int m = std::max(2, (int)std::lround(refine*counts[k]));
    double a = Lc[(size_t)bounds[k]], b = Lc[(size_t)bounds[k+1]];
    for (int q = 0; q < m; q++) {
      double u = a + (b - a)*q/(double)m;              // exclude the shared join
      oz.push_back(sample(c.z, u));
      orr.push_back(sample(c.r, u));
      ot.push_back(sample(c.t, u));
    }
  }
}

// ---------------------------------------------------------------------------
//  ear clipping (cap triangulation)
// ---------------------------------------------------------------------------
//
// Returns index triangles wound consistently with the INPUT ordering.  That is
// what lets the caller close the lofted tube: the flank quads leave the bottom
// loop traversed one way and the top loop the other, so the two caps must be
// wound oppositely for the surface to be consistently oriented.
//
inline std::vector<std::array<int,3>> earClip(const std::vector<double> &X,
                                              const std::vector<double> &Y) {
  int n = (int)X.size();
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  double area2 = 0;
  for (int i = 0; i < n; i++) {
    int j = (i+1) % n;
    area2 += X[i]*Y[j] - X[j]*Y[i];
  }
  bool flip = area2 < 0;
  if (flip) std::reverse(idx.begin(), idx.end());

  auto cross = [&](int o, int a, int b) {
    return (X[a]-X[o])*(Y[b]-Y[o]) - (Y[a]-Y[o])*(X[b]-X[o]);
  };
  auto inside = [&](int a, int b, int c, int p) {
    double d1 = cross(a,b,p), d2 = cross(b,c,p), d3 = cross(c,a,p);
    bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(neg && pos);
  };

  std::vector<std::array<int,3>> tris;
  int guard = 0;
  while ((int)idx.size() > 3 && guard < 4*n) {
    guard++;
    bool clipped = false;
    for (int i = 0; i < (int)idx.size(); i++) {
      int a = idx[(i - 1 + idx.size()) % idx.size()];
      int b = idx[i];
      int c = idx[(i + 1) % idx.size()];
      if (cross(a, b, c) <= 0) continue;                     // reflex
      bool bad = false;
      for (int q : idx) {
        if (q == a || q == b || q == c) continue;
        if (inside(a, b, c, q)) { bad = true; break; }
      }
      if (bad) continue;
      tris.push_back({a, b, c});
      idx.erase(idx.begin() + i);
      clipped = true;
      break;
    }
    if (!clipped) {                                          // stuck: fan out
      for (int i = 1; i + 1 < (int)idx.size(); i++)
        tris.push_back({idx[0], idx[i], idx[i+1]});
      idx.clear();
      break;
    }
  }
  if (idx.size() == 3) tris.push_back({idx[0], idx[1], idx[2]});
  if (flip) for (auto &t : tris) std::swap(t[0], t[2]);
  return tris;
}

// ---------------------------------------------------------------------------
//  the full-span blade solid
// ---------------------------------------------------------------------------

inline float3 toCart(double z, double r, double t) {
  double th = (std::fabs(r) > 1e-12) ? t/r : 0.0;
  return make_float3((float)(r*std::cos(th)), (float)(r*std::sin(th)), (float)z);
}

//
// Loft the row's sections over [fLo, fHi] into a closed triangle mesh.
// nSpan span stations, `refine` scales the contour point count relative to the
// bank's own (1 = exactly the design points).
//
inline void buildBladeMesh(const bank::Row &row, double fLo, double fHi,
                           int nSpan, double refine,
                           std::vector<StlTri> &tris) {
  std::vector<double> sp = spanParam(row);
  std::vector<std::vector<float3>> P(nSpan);
  std::vector<double> mz, mr, mt;

  double ehz = 0, ehr = 0;
  for (int i = 0; i < nSpan; i++) {
    double f = fLo + (fHi - fLo)*i/(double)(nSpan - 1);
    Contour c = interpContour(row, sp, f);
    sectionLoop(c, refine, mz, mr, mt);
    P[i].resize(mz.size());
    for (size_t k = 0; k < mz.size(); k++) P[i][k] = toCart(mz[k], mr[k], mt[k]);
    if (i == 0) {
      int ns = std::max(2, (int)std::lround(refine*c.nSurface));
      int ne = std::max(2, (int)std::lround(refine*c.nEdge));
      int a = ns + ne/2, b = 2*ns + ne + ne/2;
      b = std::min(b, (int)mz.size() - 1);
      ehz = mz[(size_t)b] - mz[(size_t)a];
      ehr = mr[(size_t)b] - mr[(size_t)a];
      double nn = std::hypot(ehz, ehr);
      if (nn > 0) { ehz /= nn; ehr /= nn; }
    }
  }
  size_t NP = P[0].size();

  // caps, triangulated in the section's own (m, t) chart
  std::vector<double> capX(NP), capY(NP);
  auto capIdx = [&](int station) {
    double f = fLo + (fHi - fLo)*station/(double)(nSpan - 1);
    Contour c = interpContour(row, sp, f);
    sectionLoop(c, refine, mz, mr, mt);
    for (size_t k = 0; k < NP; k++) {
      capX[k] = mz[k]*ehz + mr[k]*ehr;
      capY[k] = mt[k];
    }
    return earClip(capX, capY);
  };
  std::vector<std::array<int,3>> loTris = capIdx(0);
  std::vector<std::array<int,3>> hiTris = capIdx(nSpan - 1);

  tris.clear();
  tris.reserve(2*(size_t)(nSpan-1)*NP + loTris.size() + hiTris.size());
  auto push = [&](float3 a, float3 b, float3 c) {
    StlTri T; T.v[0] = a; T.v[1] = b; T.v[2] = c;
    T.n = normalize(cross(b - a, c - a));
    tris.push_back(T);
  };
  for (int i = 0; i + 1 < nSpan; i++)
    for (size_t k = 0; k < NP; k++) {
      size_t k2 = (k + 1) % NP;
      push(P[i][k],  P[i][k2],   P[i+1][k2]);
      push(P[i][k],  P[i+1][k2], P[i+1][k]);
    }
  // the flanks leave the bottom loop traversed forwards, so its cap is wound
  // the other way
  for (auto &t : loTris) push(P[0][(size_t)t[2]], P[0][(size_t)t[1]], P[0][(size_t)t[0]]);
  for (auto &t : hiTris) push(P[nSpan-1][(size_t)t[0]], P[nSpan-1][(size_t)t[1]],
                              P[nSpan-1][(size_t)t[2]]);

  // orient outward (positive enclosed volume)
  double vol = 0;
  for (const StlTri &t : tris)
    vol += (double)dot(t.v[0], cross(t.v[1], t.v[2]))/6.0;
  if (vol < 0)
    for (StlTri &t : tris) { std::swap(t.v[1], t.v[2]); t.n = t.n*(-1.0f); }
}

//
// Watertightness: every edge must be shared by exactly two triangles.
//
// Worth asserting rather than assuming.  The signed distance takes its SIGN
// from a ray-cast parity count, which is silently meaningless on an open or
// double-covered surface: a handful of points get the wrong sign, the level set
// picks up isolated islands and holes, and the failure looks like a geometry
// bug much further downstream.
inline int countOpenEdges(const std::vector<StlTri> &tris, int *nDegen = nullptr) {
  double diag = 0;
  float3 lo = make_float3(1e30f,1e30f,1e30f), hi = make_float3(-1e30f,-1e30f,-1e30f);
  for (const StlTri &t : tris)
    for (int k = 0; k < 3; k++) { lo = fmin3(lo, t.v[k]); hi = fmax3(hi, t.v[k]); }
  diag = (double)norm(hi - lo);
  double q = diag*1e-7;                       // weld tolerance
  typedef unsigned long long u64k;
  std::unordered_map<u64k,int> vid;
  vid.reserve(tris.size()*3);
  auto key = [&](float3 v) {
    long long a = llround(v.x/q), b = llround(v.y/q), c = llround(v.z/q);
    return (u64k)((a*73856093LL) ^ (b*19349663LL) ^ (c*83492791LL));
  };
  std::unordered_map<u64k,int> ecnt;
  ecnt.reserve(tris.size()*3);
  int nd = 0;
  for (const StlTri &t : tris) {
    if (norm(cross(t.v[1]-t.v[0], t.v[2]-t.v[0])) < 1e-16f) { nd++; continue; }
    int id[3];
    for (int k = 0; k < 3; k++) {
      u64k kk = key(t.v[k]);
      auto it = vid.find(kk);
      if (it == vid.end()) { int n = (int)vid.size(); vid[kk] = n; id[k] = n; }
      else id[k] = it->second;
    }
    for (int k = 0; k < 3; k++) {
      int a = id[k], b = id[(k+1)%3];
      u64k e = (a < b) ? ((u64k)a<<32 | (unsigned)b) : ((u64k)b<<32 | (unsigned)a);
      ecnt[e]++;
    }
  }
  if (nDegen) *nDegen = nd;
  int bad = 0;
  for (auto &kv : ecnt) if (kv.second != 2) bad++;
  return bad;
}

// enclosed volume / area of a closed triangle soup (diagnostics)
inline void meshProps(const std::vector<StlTri> &tris, double &vol, double &area) {
  vol = 0; area = 0;
  for (const StlTri &t : tris) {
    vol += (double)dot(t.v[0], cross(t.v[1], t.v[2]))/6.0;
    area += 0.5*(double)norm(cross(t.v[1] - t.v[0], t.v[2] - t.v[0]));
  }
  vol = std::fabs(vol);
}

}  // namespace blade

#endif
