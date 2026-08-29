#ifndef FEM_SAYEQUAD_H
#define FEM_SAYEQUAD_H

//
// High-order quadrature for an implicitly-defined cut cell, after
//   R. Saye, "High-order quadrature on multi-dimensional domains defined by
//   implicit functions" (the Bernstein/poly recursion, Algoim ImplicitPoly-
//   Quadrature; JCP 448:110720, 2021).
//
// Input: ONE level-set polynomial phi (PolyND, monomial basis) on the reference
// cell [0,1]^3.  In the solver phi is the Qp Lagrange interpolant of the nodal
// signed-distance values ("level set at the solution points") converted to the
// monomial basis -- so the geometry order is O(h^{p+1}), matched to a Qp field.
//
// Output:
//   sayeVolume  -> nodes+weights for  int_{phi<0} f            over the cell
//   sayeSurface -> nodes+weights+normals for  int_{phi=0} g dS over the cell
//
// Method (height-function recursion, dimension d -> d-1):
//   1. find a "height" axis k along which phi is monotone over the box
//      (grad_k keeps one sign); if none exists, bisect the widest axis (a cut
//      cell with a crease/high curvature) and recurse -- bounded depth.
//   2. build the base problem on the face perpendicular to k, whose splitters
//      are the two face restrictions phi|_{x_k=lo}, phi|_{x_k=hi}; recurse.
//   3. for each base node, partition the height column [lo_k,hi_k] at the roots
//      of phi and lay a Gauss rule on the phi<0 pieces (volume), or evaluate at
//      the roots with the co-area weight |grad phi|/|d phi/dx_k| (surface).
//
// The interface is captured through the EXACT roots of the polynomial phi, so
// there is no piecewise-planar geometry cap -- the only errors are phi's
// approximation of the true level set (O(h^{p+1})) and the smooth base Gauss
// rule (spectral).  This replaces the marching-tetrahedra O(h^2) rule.
//
// This header is the correctness reference: recursion + arena, host+device
// callable.  Buffers are fixed-capacity (no allocation) so it ports to the GPU.
//

#include "Poly.h"

// -------- output node ------------------------------------------------------
struct SayeNode {
  real x[3];      // point in the reference cell [0,1]^3
  real w;         // weight
  real n[3];      // outward-ish normal grad/|grad| (surface only; else unused)
};

struct SayeSet {
  SayeNode *p;
  i32 n, cap;
  bool ovf;
  __host__ __device__ void add(const SayeNode &q) {
    if (n < cap) p[n++] = q; else ovf = true;
  }
};

// -------- scratch arena (stack-discipline) ---------------------------------
struct SayeArena {
  SayeNode *buf;
  i32 cap, top;
  __host__ __device__ i32 mark() const { return top; }
  __host__ __device__ void release(i32 m) { top = m; }
  __host__ __device__ SayeSet span(i32 n) {
    SayeSet s; s.p = buf + top; s.n = 0; s.cap = n; s.ovf = false;
    top += n; if (top > cap) { s.cap = 0; s.ovf = true; }  // arena exhausted
    return s;
  }
};

// -------- Gauss-Legendre on [0,1] ------------------------------------------
// Computed by Newton iteration on the Legendre three-term recurrence, in
// DOUBLE regardless of `real`, so the rule is accurate to machine precision.
//
// WHAT THIS REPLACED, and why it mattered: the previous version was a set of
// literal tables truncated to TEN significant digits --
//     W[10] = {0.0333356722, 0.0747256746, ...}
// whose weights sum to 1 + 4.0e-10 instead of 1.  A tensor FACE rule squares
// that: a fully-fluid cut face measured its own area as 1 + 8.0e-10, which is
// EXACTLY the defect measured on the case-9 cylinder (dgcutjac_test).  Every
// cut-cell geometric quantity inherited it: face areas, the fitted volume
// rule, the discrete divergence theorem the volume weights are corrected
// against (GCL residual floored at 4e-10), and hence free-stream preservation
// (|R~|_inf floored at 1e-8).  Raising cfg.ng could not help -- the floor is
// the table, not the order -- which is why ng = 10 and ng = 16 gave
// bit-identical answers.
//
// The 10-point cap is unchanged (GaussRule is a by-value struct that appears
// in host-side recursion; growing it grows every frame).  n > GAUSS_MAX still
// clamps -- that is an order limit, no longer an accuracy one.
static constexpr i32 GAUSS_MAX = 10;
struct GaussRule { i32 n; real x[GAUSS_MAX]; real w[GAUSS_MAX]; };
__host__ __device__ inline GaussRule gaussLegendre(i32 n) {
  GaussRule g;
  if (n < 1) n = 1;
  if (n > GAUSS_MAX) n = GAUSS_MAX;
  g.n = n;
  const double PI_G = 3.14159265358979323846;
  for (i32 i = 0; i < n; i++) {
    // Chebyshev/Tricomi start, then Newton on P_n
    double t = cos(PI_G*((double)i + 0.75)/((double)n + 0.5));
    double p0 = 1.0, p1 = 0.0, dp = 1.0;
    for (i32 it = 0; it < 100; it++) {
      p0 = 1.0; p1 = 0.0;
      for (i32 k = 0; k < n; k++) {            // p0 -> P_n(t), p1 -> P_{n-1}(t)
        const double p2 = p1; p1 = p0;
        p0 = ((2.0*(double)k + 1.0)*t*p1 - (double)k*p2)/((double)k + 1.0);
      }
      dp = (double)n*(t*p0 - p1)/(t*t - 1.0);
      const double dt = -p0/dp;
      t += dt;
      if (fabs(dt) <= 1e-16*(fabs(t) + 1.0)) break;
    }
    const double wt = 2.0/((1.0 - t*t)*dp*dp);
    // [-1,1] -> [0,1]; the Newton roots come out DESCENDING in t, so fill from
    // the far end to keep x ascending (callers assume an ordered rule)
    g.x[n-1-i] = (real)(0.5*(1.0 + t));
    g.w[n-1-i] = (real)(0.5*wt);
  }
  return g;
}

// -------- configuration ----------------------------------------------------
struct SayeCfg {
  i32  ng;         // Gauss points per height column / base direction
  i32  maxDepth;   // subdivision cap before low-order fallback
  real gradTol;    // |grad_k| below this -> axis not usable as height dir
  __host__ __device__ static SayeCfg def() {
    // maxDepth 10.  It WAS 6, on the reasoning that a well-resolved cut cell
    // needs 0-2 subdivisions and anything deeper is an under-resolved cell the
    // solver would h-refine anyway -- so the bound "is never the accuracy-
    // limiting factor".  That was wrong, and measurably so, because the two
    // recursions do DIFFERENT things when they hit the cap and neither is a
    // graceful degradation:
    //   * arrangementRule (volume and face rules) calls fallbackTensor, which
    //     tensors the WHOLE box and ignores the interface -- it over-counts;
    //   * sayeSurfaceRec (the wall rule) just returns, dropping the piece --
    //     it under-counts.
    // So the cap does not lower the order, it breaks the closed-surface
    // identity CLOSED INT n dS = 0 that free-stream preservation rests on, in
    // opposite directions on the two sides of the same identity.  No volume-
    // weight correction can repair that (it is exactly what CutElemOps calls
    // bndIncons), and raising the Gauss order cannot either -- which is the
    // signature that had been read as a structural defect of the scheme.
    // MEASURED on the dgcutelem_test cells at N=3, depth 6 -> 10:
    //   interior bubble   wall area 0.580 -> 1.131, i.e. 4 pi (0.3)^2 EXACTLY:
    //                     at depth 6 HALF THE SPHERE was missing.  GCL 3.4e-03
    //                     -> 1.1e-11.
    //   near-face tangency  volume 0.40704 -> 0.41489, GCL 1.7e-03 -> 3.4e-10.
    //   worst GCL over all cells 3.4e-03 -> 3.9e-10, 2 geometry-limited -> 0.
    // Deeper is NOT better: at 14 the interior bubble fails to build outright
    // (the recursion outruns the arena), so 10 is the measured setting, not an
    // "as deep as possible" one.  CUT_MAXDEPTH overrides it in the DG build.
    SayeCfg c; c.ng = 5; c.maxDepth = 10; c.gradTol = (real)1e-9; return c;
  }
};

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------

// number/list of active axes (act[a] == true)
__host__ __device__ inline i32 activeAxes(const bool act[3], i32 ax[3]) {
  i32 n = 0;
  for (i32 a = 0; a < 3; a++) if (act[a]) ax[n++] = a;
  return n;
}

// is psi sign-definite over the box (no zero crossing)?  sampled test.
__host__ __device__ inline bool signDefinite(const PolyND &psi,
                                             const real lo[3], const real hi[3],
                                             const bool act[3]) {
  const i32 S = 3;
  bool sawPos = false, sawNeg = false;
  i32 ax[3]; i32 na = activeAxes(act, ax);
  i32 tot = 1; for (i32 a = 0; a < na; a++) tot *= S;
  for (i32 t = 0; t < tot; t++) {
    real x[3] = { lo[0], lo[1], lo[2] };
    i32 q = t;
    for (i32 a = 0; a < na; a++) {
      i32 idx = q % S; q /= S;
      real f = (S==1)?(real)0.5:(real)idx/(S-1);
      x[ax[a]] = lo[ax[a]] + f*(hi[ax[a]]-lo[ax[a]]);
    }
    real v = psi.eval(x);
    if (v > 0) sawPos = true; else if (v < 0) sawNeg = true;
    if (sawPos && sawNeg) return false;
  }
  return true;
}

// choose a height axis among the active ones: grad along it must keep one sign
// over the box, with |grad| >= gradTol at the samples.  returns -1 if none.
__host__ __device__ inline i32 heightDir(const PolyND *psis, i32 npsi,
                                         const real lo[3], const real hi[3],
                                         const bool act[3], real gradTol) {
  const i32 S = 3;
  i32 ax[3]; i32 na = activeAxes(act, ax);
  i32 best = -1; real bestScore = -1;
  for (i32 c = 0; c < na; c++) {
    i32 k = ax[c];
    bool ok = true; real minAbs = 1e30;
    for (i32 m = 0; m < npsi && ok; m++) {
      PolyND dk = psis[m].partial(k);
      bool sawPos = false, sawNeg = false;
      i32 tot = 1; for (i32 a = 0; a < na; a++) tot *= S;
      for (i32 t = 0; t < tot; t++) {
        real x[3] = { lo[0], lo[1], lo[2] };
        i32 q = t;
        for (i32 a = 0; a < na; a++) {
          i32 idx = q % S; q /= S;
          real f = (real)idx/(S-1);
          x[ax[a]] = lo[ax[a]] + f*(hi[ax[a]]-lo[ax[a]]);
        }
        real g = dk.eval(x);
        if (g > 0) sawPos = true; else if (g < 0) sawNeg = true;
        real ag = fabs(g); if (ag < minAbs) minAbs = ag;
        if (sawPos && sawNeg) { ok = false; break; }
      }
    }
    if (ok && minAbs >= gradTol && minAbs > bestScore) { bestScore = minAbs; best = k; }
  }
  return best;
}

// collect the sorted interior roots along axis k, at fixed other-axis coords xf,
// over (a,b), from all psis; returns count (capped at 8).
__host__ __device__ inline i32 columnRoots(const PolyND *psis, i32 npsi, i32 k,
                                           const real xf[3], real a, real b,
                                           real out[8]) {
  real all[8]; i32 n = 0;
  for (i32 m = 0; m < npsi; m++) {
    Poly1 p1 = psis[m].line(k, xf);
    real r[PMAXRT]; i32 nr = poly1Roots(p1, a, b, r);
    for (i32 i = 0; i < nr; i++) if (n < 8) all[n++] = r[i];
  }
  // insertion sort + dedup
  for (i32 i = 1; i < n; i++) {
    real v = all[i]; i32 j = i-1;
    while (j >= 0 && all[j] > v) { all[j+1] = all[j]; j--; }
    all[j+1] = v;
  }
  i32 m2 = 0;
  for (i32 i = 0; i < n; i++)
    if (m2 == 0 || all[i] - out[m2-1] > (real)1e-10) out[m2++] = all[i];
  return m2;
}

// low-order fallback: plain tensor Gauss over the active box (used only when the
// subdivision cap is hit -- keeps the rule bounded and flags via ovf).
__host__ __device__ inline void fallbackTensor(const real lo[3], const real hi[3],
                                               const bool act[3], i32 ng,
                                               SayeSet *out) {
  GaussRule g = gaussLegendre(ng);
  i32 ax[3]; i32 na = activeAxes(act, ax);
  i32 tot = 1; for (i32 a = 0; a < na; a++) tot *= g.n;
  for (i32 t = 0; t < tot; t++) {
    SayeNode nd; nd.x[0]=lo[0]; nd.x[1]=lo[1]; nd.x[2]=lo[2];
    real w = 1; i32 q = t;
    for (i32 a = 0; a < na; a++) {
      i32 idx = q % g.n; q /= g.n; i32 kk = ax[a];
      nd.x[kk] = lo[kk] + g.x[idx]*(hi[kk]-lo[kk]);
      w *= g.w[idx]*(hi[kk]-lo[kk]);
    }
    nd.w = w; out->add(nd);
  }
}

// ---------------------------------------------------------------------------
//  arrangement rule: a quadrature over the active box that resolves the sign
//  arrangement of {psis}.  Every Gauss node lands strictly inside one sign
//  cell, so applying it to (indicator * f) integrates to high order.  Used for
//  the base problems AND, at the top level, for the volume (filter phi<0).
// ---------------------------------------------------------------------------
__host__ __device__ inline void arrangementRule(const PolyND *psis, i32 npsi,
                                                real lo[3], real hi[3], bool act[3],
                                                SayeSet *out, SayeArena *ar,
                                                const SayeCfg &cfg, i32 depth) {
  i32 ax[3]; i32 na = activeAxes(act, ax);
  GaussRule g = gaussLegendre(cfg.ng);

  if (na == 1) {                                   // ---- 1-D base case ----
    i32 k = ax[0];
    real xf[3] = { lo[0], lo[1], lo[2] };          // other coords are fixed
    real rt[8]; i32 nr = columnRoots(psis, npsi, k, xf, lo[k], hi[k], rt);
    real bd[10]; i32 nb = 0; bd[nb++] = lo[k];
    for (i32 i = 0; i < nr; i++) bd[nb++] = rt[i];
    bd[nb++] = hi[k];
    for (i32 s = 0; s + 1 < nb; s++) {
      real a = bd[s], b = bd[s+1], len = b - a;
      if (len <= 0) continue;
      for (i32 i = 0; i < g.n; i++) {
        SayeNode nd; nd.x[0]=lo[0]; nd.x[1]=lo[1]; nd.x[2]=lo[2];
        nd.x[k] = a + g.x[i]*len; nd.w = g.w[i]*len;
        out->add(nd);
      }
    }
    return;
  }

  // ---- reduction: pick a height axis or subdivide ----
  i32 k = heightDir(psis, npsi, lo, hi, act, cfg.gradTol);
  if (k < 0) {
    if (depth >= cfg.maxDepth) { fallbackTensor(lo, hi, act, cfg.ng, out); return; }
    // bisect the widest active axis
    i32 wax = ax[0]; real wid = hi[ax[0]]-lo[ax[0]];
    for (i32 c = 1; c < na; c++) if (hi[ax[c]]-lo[ax[c]] > wid) { wid = hi[ax[c]]-lo[ax[c]]; wax = ax[c]; }
    real mid = (real)0.5*(lo[wax]+hi[wax]);
    real slo[3]={lo[0],lo[1],lo[2]}, shi[3]={hi[0],hi[1],hi[2]};
    shi[wax] = mid; arrangementRule(psis, npsi, slo, shi, act, out, ar, cfg, depth+1);
    slo[wax] = mid; shi[wax] = hi[wax];
    arrangementRule(psis, npsi, slo, shi, act, out, ar, cfg, depth+1);
    return;
  }

  // build base splitters = face restrictions, dropping sign-definite ones
  bool bact[3] = { act[0], act[1], act[2] }; bact[k] = false;
  PolyND bpsi[16]; i32 nbp = 0;
  for (i32 m = 0; m < npsi && nbp+2 <= 16; m++) {
    PolyND a = psis[m].subst(k, lo[k]);
    PolyND b = psis[m].subst(k, hi[k]);
    if (!signDefinite(a, lo, hi, bact)) bpsi[nbp++] = a;
    if (!signDefinite(b, lo, hi, bact)) bpsi[nbp++] = b;
  }

  i32 m0 = ar->mark();
  SayeSet base = ar->span(4096);
  if (nbp == 0) {
    // no base splitters: the whole face contributes; a single tensor Gauss base
    GaussRule gg = gaussLegendre(cfg.ng);
    i32 bax[3]; i32 nba = activeAxes(bact, bax);
    i32 tot = 1; for (i32 a = 0; a < nba; a++) tot *= gg.n;
    for (i32 t = 0; t < tot; t++) {
      SayeNode nd; nd.x[0]=lo[0]; nd.x[1]=lo[1]; nd.x[2]=lo[2];
      real w = 1; i32 q = t;
      for (i32 a = 0; a < nba; a++) {
        i32 idx = q % gg.n; q /= gg.n; i32 kk = bax[a];
        nd.x[kk] = lo[kk] + gg.x[idx]*(hi[kk]-lo[kk]); w *= gg.w[idx]*(hi[kk]-lo[kk]);
      }
      nd.w = w; base.add(nd);
    }
  } else {
    arrangementRule(bpsi, nbp, lo, hi, bact, &base, ar, cfg, depth);
  }
  if (base.ovf) out->ovf = true;

  // for each base node, integrate the height column
  for (i32 ib = 0; ib < base.n; ib++) {
    real xf[3] = { base.p[ib].x[0], base.p[ib].x[1], base.p[ib].x[2] };
    real rt[8]; i32 nr = columnRoots(psis, npsi, k, xf, lo[k], hi[k], rt);
    real bd[10]; i32 nb = 0; bd[nb++] = lo[k];
    for (i32 i = 0; i < nr; i++) bd[nb++] = rt[i];
    bd[nb++] = hi[k];
    for (i32 s = 0; s + 1 < nb; s++) {
      real a = bd[s], b = bd[s+1], len = b - a;
      if (len <= 0) continue;
      for (i32 i = 0; i < g.n; i++) {
        SayeNode nd; nd.x[0]=xf[0]; nd.x[1]=xf[1]; nd.x[2]=xf[2];
        nd.x[k] = a + g.x[i]*len; nd.w = base.p[ib].w * g.w[i]*len;
        out->add(nd);
      }
    }
  }
  ar->release(m0);
}

// ---------------------------------------------------------------------------
//  public: volume rule for {phi < 0} on [0,1]^3
// ---------------------------------------------------------------------------
__host__ __device__ inline void sayeVolume(const PolyND &phi, SayeSet *out,
                                          SayeArena *ar,
                                          const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,0}, hi[3] = {1,1,1}; bool act[3] = {true,true,true};
  i32 m0 = ar->mark();
  // collector takes half the free arena; the recursion's base scratch nests on
  // the other half (stack-discipline mark/release keeps it disjoint from `all`).
  i32 half = (ar->cap - ar->top) / 2;
  SayeSet all = ar->span(half);
  arrangementRule(&phi, 1, lo, hi, act, &all, ar, cfg, 0);
  if (all.ovf) out->ovf = true;
  for (i32 i = 0; i < all.n; i++)
    if (phi.eval(all.p[i].x) < 0) out->add(all.p[i]);
  ar->release(m0);
}

// ---------------------------------------------------------------------------
//  public: surface rule for {phi = 0} on [0,1]^3  (nodes carry unit normal)
// ---------------------------------------------------------------------------
__host__ __device__ inline void sayeSurfaceRec(const PolyND &phi,
                                              real lo[3], real hi[3], bool act[3],
                                              SayeSet *out, SayeArena *ar,
                                              const SayeCfg &cfg, i32 depth) {
  i32 k = heightDir(&phi, 1, lo, hi, act, cfg.gradTol);
  if (k < 0) {
    if (depth >= cfg.maxDepth) return;  // give up on this sliver piece
    i32 ax[3]; i32 na = activeAxes(act, ax);
    i32 wax = ax[0]; real wid = hi[ax[0]]-lo[ax[0]];
    for (i32 c = 1; c < na; c++) if (hi[ax[c]]-lo[ax[c]] > wid) { wid = hi[ax[c]]-lo[ax[c]]; wax = ax[c]; }
    real mid = (real)0.5*(lo[wax]+hi[wax]);
    real slo[3]={lo[0],lo[1],lo[2]}, shi[3]={hi[0],hi[1],hi[2]};
    shi[wax]=mid; sayeSurfaceRec(phi, slo, shi, act, out, ar, cfg, depth+1);
    slo[wax]=mid; shi[wax]=hi[wax]; sayeSurfaceRec(phi, slo, shi, act, out, ar, cfg, depth+1);
    return;
  }

  bool bact[3] = { act[0], act[1], act[2] }; bact[k] = false;
  PolyND bpsi[2]; i32 nbp = 0;
  { PolyND a = phi.subst(k, lo[k]); if (!signDefinite(a, lo, hi, bact)) bpsi[nbp++]=a; }
  { PolyND b = phi.subst(k, hi[k]); if (!signDefinite(b, lo, hi, bact)) bpsi[nbp++]=b; }

  i32 m0 = ar->mark();
  SayeSet base = ar->span(4096);
  if (nbp == 0) {
    GaussRule gg = gaussLegendre(cfg.ng);
    i32 bax[3]; i32 nba = activeAxes(bact, bax);
    i32 tot = 1; for (i32 a = 0; a < nba; a++) tot *= gg.n;
    for (i32 t = 0; t < tot; t++) {
      SayeNode nd; nd.x[0]=lo[0]; nd.x[1]=lo[1]; nd.x[2]=lo[2];
      real w = 1; i32 q = t;
      for (i32 a = 0; a < nba; a++) {
        i32 idx = q % gg.n; q /= gg.n; i32 kk = bax[a];
        nd.x[kk] = lo[kk] + gg.x[idx]*(hi[kk]-lo[kk]); w *= gg.w[idx]*(hi[kk]-lo[kk]);
      }
      nd.w = w; base.add(nd);
    }
  } else {
    arrangementRule(bpsi, nbp, lo, hi, bact, &base, ar, cfg, depth);
  }
  if (base.ovf) out->ovf = true;

  for (i32 ib = 0; ib < base.n; ib++) {
    real xf[3] = { base.p[ib].x[0], base.p[ib].x[1], base.p[ib].x[2] };
    Poly1 p1 = phi.line(k, xf);
    real rt[PMAXRT]; i32 nr = poly1Roots(p1, lo[k], hi[k], rt);
    for (i32 i = 0; i < nr; i++) {
      real full[3] = { xf[0], xf[1], xf[2] }; full[k] = rt[i];
      real gr[3]; phi.grad(full, gr);
      real gk = fabs(gr[k]);
      if (gk < cfg.gradTol) continue;
      real gmag = sqrt(gr[0]*gr[0]+gr[1]*gr[1]+gr[2]*gr[2]);
      SayeNode nd;
      nd.x[0]=full[0]; nd.x[1]=full[1]; nd.x[2]=full[2];
      nd.w = base.p[ib].w * gmag / gk;
      real inv = gmag>0 ? 1/gmag : 0;
      nd.n[0]=gr[0]*inv; nd.n[1]=gr[1]*inv; nd.n[2]=gr[2]*inv;
      out->add(nd);
    }
  }
  ar->release(m0);
}

__host__ __device__ inline void sayeSurface(const PolyND &phi, SayeSet *out,
                                           SayeArena *ar,
                                           const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,0}, hi[3] = {1,1,1}; bool act[3] = {true,true,true};
  sayeSurfaceRec(phi, lo, hi, act, out, ar, cfg, 0);
}

// ---------------------------------------------------------------------------
//  public: rule for the part of a CELL FACE inside {phi < 0}
//
//  d = face axis (0,1,2), side = 0 (x_d = 0) or 1 (x_d = 1).  Weights are the
//  2-D area measure on that face; nodes carry the fixed coordinate so they can
//  be evaluated with the same 3-D basis as the volume rule.
//
//  Needed by DISCONTINUOUS methods and not by continuous ones, which is why it
//  did not exist before: a cut-cell DG integrates a numerical flux over the
//  fluid part of each cell face, whereas the continuous CutFEM only ever needed
//  {phi<0} volumes, the {phi=0} wall, and full (uncut) faces for ghost penalty.
//
//  The arrangement recursion already supports it -- deactivating axis d and
//  pinning lo[d]=hi[d] reduces it to the 2-D case -- so this is a wrapper, not
//  new quadrature machinery.
// ---------------------------------------------------------------------------
__host__ __device__ inline void sayeFace(const PolyND &phi, i32 d, i32 side,
                                         SayeSet *out, SayeArena *ar,
                                         const SayeCfg &cfg = SayeCfg::def()) {
  real v = side ? (real)1 : (real)0;
  real lo[3] = {0,0,0}, hi[3] = {1,1,1};
  bool act[3] = {true,true,true};
  lo[d] = hi[d] = v; act[d] = false;
  i32 m0 = ar->mark();
  i32 half = (ar->cap - ar->top) / 2;
  SayeSet all = ar->span(half);
  arrangementRule(&phi, 1, lo, hi, act, &all, ar, cfg, 0);
  if (all.ovf) out->ovf = true;
  for (i32 i = 0; i < all.n; i++)
    if (phi.eval(all.p[i].x) < 0) out->add(all.p[i]);
  ar->release(m0);
}

// ---------------------------------------------------------------------------
//  PSEUDO-2D (EXTRUDED) RULES.
//
//  When the geometry is z-invariant -- a cylinder, an extruded profile -- the
//  cut region is (2-D shape) x (full z), and the right rule is the tensor
//  product of a 2-D/1-D rule in the slice with a Gauss rule in z.  Building it
//  that way is not just cheaper than the 3-D recursion; it is the only way to
//  get the rule EXACTLY symmetric in z, which the 3-D fit does not give:
//  measured on a cylinder, the fitted 3-D volume rule is moment-symmetric
//  (sum w (z-1/2) ~ 4e-16) but its individual weights differ from their
//  z-mirrors by up to 4.2e-03, and the Euler flux -- rational in the modal
//  coefficients, so outside the moment space -- aliases that asymmetry into
//  spurious z-odd modes.
//
//  These three deactivate axes through the same act[] mask the recursion
//  already carries, so no new machinery is involved.
// ---------------------------------------------------------------------------

// region {phi < 0} in the plane z = z0; weights are the 2-D area measure
__host__ __device__ inline void sayeSlice2D(const PolyND &phi, real z0, SayeSet *out,
                                            SayeArena *ar, const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,z0}, hi[3] = {1,1,z0}; bool act[3] = {true,true,false};
  i32 m0 = ar->mark();
  i32 half = (ar->cap - ar->top) / 2;
  SayeSet all = ar->span(half);
  arrangementRule(&phi, 1, lo, hi, act, &all, ar, cfg, 0);
  if (all.ovf) out->ovf = true;
  for (i32 i = 0; i < all.n; i++)
    if (phi.eval(all.p[i].x) < 0) out->add(all.p[i]);
  ar->release(m0);
}

// interval {phi < 0} on the line {x_d = v, z = z0}, d in {0,1}; 1-D length measure
__host__ __device__ inline void sayeEdge1D(const PolyND &phi, i32 d, real v, real z0,
                                           SayeSet *out, SayeArena *ar,
                                           const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,z0}, hi[3] = {1,1,z0}; bool act[3] = {true,true,false};
  lo[d] = hi[d] = v; act[d] = false;
  i32 m0 = ar->mark();
  i32 half = (ar->cap - ar->top) / 2;
  SayeSet all = ar->span(half);
  arrangementRule(&phi, 1, lo, hi, act, &all, ar, cfg, 0);
  if (all.ovf) out->ovf = true;
  for (i32 i = 0; i < all.n; i++)
    if (phi.eval(all.p[i].x) < 0) out->add(all.p[i]);
  ar->release(m0);
}

// curve {phi = 0} in the plane z = z0, with in-plane normals (n_z = 0 for a
// z-invariant phi, since the normal comes from the 3-D gradient)
__host__ __device__ inline void sayeCurve2D(const PolyND &phi, real z0, SayeSet *out,
                                            SayeArena *ar, const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,z0}, hi[3] = {1,1,z0}; bool act[3] = {true,true,false};
  sayeSurfaceRec(phi, lo, hi, act, out, ar, cfg, 0);
}

// ---------------------------------------------------------------------------
//  CSG-aware multi-polynomial rules.
//
//  A crease is where phi = max_m(phi_m) (or min) of several SMOOTH branches has
//  a kink -- fitting ONE polynomial to that max oscillates.  Instead, fit a
//  smooth polynomial to EACH branch and integrate the region the signs define,
//  with every surface smooth (no oscillation).  This is what Saye's method is
//  built for: the arrangement recursion already carries a LIST of polynomials.
//
//  sayeVolumeMulti  : the INTERSECTION region  {phi_m < 0 for all m}   (= max<0)
//  sayeSurfaceMulti : its boundary  U_m { phi_m = 0 and phi_k<0, k!=m }, each
//                     piece smooth, meeting at the crease edge; the node normal
//                     is that branch's grad phi_m / |grad phi_m|.
// ---------------------------------------------------------------------------
__host__ __device__ inline void sayeVolumeMulti(const PolyND *phis, i32 npsi,
                                               SayeSet *out, SayeArena *ar,
                                               const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,0}, hi[3] = {1,1,1}; bool act[3] = {true,true,true};
  i32 m0 = ar->mark();
  i32 half = (ar->cap - ar->top) / 2;
  SayeSet all = ar->span(half);
  arrangementRule(phis, npsi, lo, hi, act, &all, ar, cfg, 0);   // resolves ALL branches
  if (all.ovf) out->ovf = true;
  for (i32 i = 0; i < all.n; i++) {
    bool in = true;
    for (i32 m = 0; m < npsi; m++) if (phis[m].eval(all.p[i].x) >= 0) { in = false; break; }
    if (in) out->add(all.p[i]);
  }
  ar->release(m0);
}

__host__ __device__ inline void sayeSurfaceMulti(const PolyND *phis, i32 npsi,
                                                SayeSet *out, SayeArena *ar,
                                                const SayeCfg &cfg = SayeCfg::def()) {
  real lo[3] = {0,0,0}, hi[3] = {1,1,1}; bool act[3] = {true,true,true};
  for (i32 m = 0; m < npsi; m++) {
    i32 mk = ar->mark();
    i32 half = (ar->cap - ar->top) / 2;
    SayeSet sm = ar->span(half);
    sayeSurfaceRec(phis[m], lo, hi, act, &sm, ar, cfg, 0);   // {phi_m = 0}
    if (sm.ovf) out->ovf = true;
    for (i32 i = 0; i < sm.n; i++) {                          // keep the part inside the others
      bool in = true;
      for (i32 k = 0; k < npsi; k++) {
        if (k == m) continue;
        if (phis[k].eval(sm.p[i].x) > 0) { in = false; break; }
      }
      if (in) out->add(sm.p[i]);
    }
    ar->release(mk);
  }
}

#endif
