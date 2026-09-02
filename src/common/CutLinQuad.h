#ifndef CUT_LIN_QUAD_H
#define CUT_LIN_QUAD_H

// ---------------------------------------------------------------------------
//  Cut-cell quadrature from a PIECEWISE-LINEAR level set on a sampled lattice.
//
//  The alternative to Saye's height-function recursion.  The element [0,1]^3 is
//  cut by a lattice of (s+1)^3 level-set samples taken FROM THE ORACLE -- not
//  interpolated from the element corners, which is what separates this from
//  CutQuad.cuh's cqOneTet: that one sub-divides a trilinear interpolant of 8
//  element values, so its sub-cells carry no geometric information the element
//  did not already have.  Here every sub-corner is a real oracle evaluation, so
//  the reconstructed interface converges as O((h/s)^2).
//
//  Each sub-cube is Kuhn-split into 6 tets; phi is LINEAR on each tet, so the
//  cut is a plane and the marching-tet cases below are EXACT for the model.
//  Two consequences that the Saye path does not have:
//
//    * the purity test is exact, not a Lipschitz bound.  A tet's vertices are
//      cube corners, so if all 8 corners of a sub-cube share a sign then no tet
//      of it can contain a zero -- interior sub-cubes take a plain tensor Gauss
//      rule and exterior ones are skipped, with no possibility of clipping a
//      feature the model actually represents.
//    * the domain is a POLYHEDRON and the interface triangles are exactly its
//      faces, so the discrete divergence theorem holds to machine precision:
//      sum_Gamma w n = 0 exactly.  Saye's rule has no such guarantee -- its
//      recursion drops surface pieces at the depth cap (documented in SayeCfg)
//      and under-counts, which is invisible in |Gamma_h| but not in this sum.
//
//  Cost: (s+1)^3 oracle calls per cut element against s^3*(gq+1)^3 for a
//  per-sub-cell Saye fit -- 125 vs 1728 at s=4, gq=2.
//
//  Quadrature is collapsed-Duffy Gauss, so any degree is available with no rule
//  tables: on a tet the map (u,v,w) -> barycentric has Jacobian (1-u)^2(1-v),
//  raising the integrand degree by 3, so n = p+2 points/axis is exact for the
//  degree-2p moment target; on a triangle the Jacobian is (1-u) and n = p+1.
//
//  Emits SayeNode so every consumer downstream (NNLS compression, the element
//  kernels, the GPU pools) is unchanged.  Weights are REFERENCE measure on the
//  unit cube, matching the Saye convention.
// ---------------------------------------------------------------------------

#include <vector>
#include "Util.cuh"
#include "SayeQuad.h"     // SayeNode, GaussRule, gaussLegendre

// Kuhn decomposition of the unit cube into 6 tets sharing the 0--7 diagonal.
static inline void clKuhn(i32 t, i32 v[4]) {
  static const i32 K[6][4] = {{0,1,3,7}, {0,1,5,7}, {0,4,5,7},
                              {0,4,6,7}, {0,2,6,7}, {0,2,3,7}};
  for (i32 i = 0; i < 4; i++) v[i] = K[t][i];
}

static inline double clTetVol(const double a[3], const double b[3],
                              const double c[3], const double d[3]) {
  double u[3], v[3], w[3];
  for (i32 i = 0; i < 3; i++) { u[i]=b[i]-a[i]; v[i]=c[i]-a[i]; w[i]=d[i]-a[i]; }
  return fabs(u[0]*(v[1]*w[2]-v[2]*w[1])
            - u[1]*(v[0]*w[2]-v[2]*w[0])
            + u[2]*(v[0]*w[1]-v[1]*w[0]))/6.0;
}

// zero crossing of the linear interpolant along the edge a--b
static inline void clEdge(const double pa[3], double fa,
                          const double pb[3], double fb, double out[3]) {
  double t = fa/(fa-fb);
  if (!(t > 0)) t = 0; if (t > 1) t = 1;
  for (i32 d = 0; d < 3; d++) out[d] = pa[d] + t*(pb[d]-pa[d]);
}

// ---- collapsed-Duffy rules -------------------------------------------------
// tet: weights sum to the tet volume
static inline void clAddTet(std::vector<SayeNode> &out, const GaussRule &g,
                            const double p0[3], const double p1[3],
                            const double p2[3], const double p3[3]) {
  const double V = clTetVol(p0,p1,p2,p3);
  if (!(V > 0)) return;
  const double s6V = 6.0*V;
  for (i32 i = 0; i < g.n; i++) for (i32 j = 0; j < g.n; j++) for (i32 k = 0; k < g.n; k++) {
    const double u=(double)g.x[i], v=(double)g.x[j], w=(double)g.x[k];
    const double L1=u, L2=v*(1.0-u), L3=w*(1.0-u)*(1.0-v), L0=1.0-L1-L2-L3;
    SayeNode nd{};
    for (i32 d = 0; d < 3; d++)
      nd.x[d] = (real)(L0*p0[d] + L1*p1[d] + L2*p2[d] + L3*p3[d]);
    nd.w = (real)((double)g.w[i]*(double)g.w[j]*(double)g.w[k]
                  * (1.0-u)*(1.0-u)*(1.0-v) * s6V);
    out.push_back(nd);
  }
}

// triangle: weights sum to the triangle area, normal oriented AWAY from xin
// (the inside centroid), i.e. outward from {phi < 0}
static inline void clAddTri(std::vector<SayeNode> &out, const GaussRule &g,
                            const double q0[3], const double q1[3],
                            const double q2[3], const double xin[3]) {
  double e1[3], e2[3], nz[3];
  for (i32 d = 0; d < 3; d++) { e1[d]=q1[d]-q0[d]; e2[d]=q2[d]-q0[d]; }
  nz[0]=e1[1]*e2[2]-e1[2]*e2[1];
  nz[1]=e1[2]*e2[0]-e1[0]*e2[2];
  nz[2]=e1[0]*e2[1]-e1[1]*e2[0];
  const double nl = sqrt(nz[0]*nz[0]+nz[1]*nz[1]+nz[2]*nz[2]);
  if (!(nl > 0)) return;
  const double A = 0.5*nl;
  for (i32 d = 0; d < 3; d++) nz[d] /= nl;
  double dot = 0;                                  // point it away from the solid
  for (i32 d = 0; d < 3; d++) dot += nz[d]*((q0[d]+q1[d]+q2[d])/3.0 - xin[d]);
  if (dot < 0) for (i32 d = 0; d < 3; d++) nz[d] = -nz[d];
  const double s2A = 2.0*A;
  for (i32 i = 0; i < g.n; i++) for (i32 j = 0; j < g.n; j++) {
    const double u=(double)g.x[i], v=(double)g.x[j];
    const double L1=u, L2=v*(1.0-u), L0=1.0-L1-L2;
    SayeNode nd{};
    for (i32 d = 0; d < 3; d++) {
      nd.x[d] = (real)(L0*q0[d] + L1*q1[d] + L2*q2[d]);
      nd.n[d] = (real)nz[d];
    }
    nd.w = (real)((double)g.w[i]*(double)g.w[j]*(1.0-u)*s2A);
    out.push_back(nd);
  }
}

// ---- marching tetrahedra ---------------------------------------------------
// Same case topology as CutQuad.cuh's cqMarchTet (verified there for p=1); the
// difference is arbitrary-degree output and std::vector storage.
static inline void clMarchTet(std::vector<SayeNode> &vol, std::vector<SayeNode> &srf,
                              const GaussRule &gv, const GaussRule &gs,
                              const double p[4][3], const double f[4]) {
  i32 in[4], out[4], nin=0, nout=0;
  for (i32 v = 0; v < 4; v++) { if (f[v] < 0) in[nin++]=v; else out[nout++]=v; }
  if (nin == 0) return;
  if (nin == 4) { clAddTet(vol, gv, p[0], p[1], p[2], p[3]); return; }

  double xin[3] = {0,0,0};                          // inside centroid: orients n
  for (i32 i = 0; i < nin; i++)
    for (i32 d = 0; d < 3; d++) xin[d] += p[in[i]][d]/nin;

  if (nin == 1) {
    const i32 a = in[0]; double q[3][3];
    for (i32 i = 0; i < 3; i++) clEdge(p[a], f[a], p[out[i]], f[out[i]], q[i]);
    clAddTet(vol, gv, p[a], q[0], q[1], q[2]);
    clAddTri(srf, gs, q[0], q[1], q[2], xin);
    return;
  }
  if (nin == 3) {
    const i32 b = out[0]; double q[3][3];
    for (i32 i = 0; i < 3; i++) clEdge(p[in[i]], f[in[i]], p[b], f[b], q[i]);
    const double *A0=p[in[0]], *A1=p[in[1]], *A2=p[in[2]];
    clAddTet(vol, gv, A0, A1, A2, q[0]);
    clAddTet(vol, gv, A1, A2, q[0], q[1]);
    clAddTet(vol, gv, A2, q[0], q[1], q[2]);
    clAddTri(srf, gs, q[0], q[1], q[2], xin);
    return;
  }
  // nin == 2: inside region is a wedge, interface is a quad
  const i32 A=in[0], Bv=in[1], C=out[0], D=out[1];
  double qAC[3], qAD[3], qBC[3], qBD[3];
  clEdge(p[A],  f[A],  p[C], f[C], qAC);
  clEdge(p[A],  f[A],  p[D], f[D], qAD);
  clEdge(p[Bv], f[Bv], p[C], f[C], qBC);
  clEdge(p[Bv], f[Bv], p[D], f[D], qBD);
  clAddTet(vol, gv, p[A], qAC, qAD, p[Bv]);
  clAddTet(vol, gv, qAC,  qAD, p[Bv], qBC);
  clAddTet(vol, gv, qAD,  p[Bv], qBC, qBD);
  clAddTri(srf, gs, qAC, qBC, qBD, xin);
  clAddTri(srf, gs, qAC, qBD, qAD, xin);
}

// ---------------------------------------------------------------------------
//  Full rule for one element from an (s+1)^3 lattice of level-set samples.
//  phiLat is indexed  i + (s+1)*(j + (s+1)*k),  i/j/k = 0..s, in the element's
//  reference cube.  degTarget is the polynomial degree the rule must integrate
//  exactly (2p for the stiffness moment space).
// ---------------------------------------------------------------------------
static inline void cutLinRule(const real *phiLat, i32 s, i32 degTarget,
                              std::vector<SayeNode> &vol,
                              std::vector<SayeNode> &srf) {
  const i32 L = s+1;
  const double ds = 1.0/(double)s;
  // exactness: tet Duffy raises degree by 3, triangle by 1
  const GaussRule gv = gaussLegendre((degTarget+4)/2);
  const GaussRule gs = gaussLegendre((degTarget+2)/2);
  const GaussRule gi = gaussLegendre(degTarget/2 + 1);   // interior sub-cube
  auto at=[&](i32 i,i32 j,i32 k)->double{ return (double)phiLat[i + L*(j + L*k)]; };

  for (i32 sk = 0; sk < s; sk++) for (i32 sj = 0; sj < s; sj++) for (i32 si = 0; si < s; si++) {
    double cf[8]; double cp[8][3];
    i32 nneg = 0;
    for (i32 n = 0; n < 8; n++) {
      const i32 di=(n&1), dj=((n>>1)&1), dk=((n>>2)&1);
      cp[n][0]=(si+di)*ds; cp[n][1]=(sj+dj)*ds; cp[n][2]=(sk+dk)*ds;
      cf[n]=at(si+di, sj+dj, sk+dk);
      if (cf[n] < 0) nneg++;
    }
    if (nneg == 0) continue;                       // wholly outside (exact)
    if (nneg == 8) {                               // wholly inside (exact)
      const double wv = ds*ds*ds;
      for (i32 i = 0; i < gi.n; i++) for (i32 j = 0; j < gi.n; j++) for (i32 k = 0; k < gi.n; k++) {
        SayeNode nd{};
        nd.x[0]=(real)((si+(double)gi.x[i])*ds);
        nd.x[1]=(real)((sj+(double)gi.x[j])*ds);
        nd.x[2]=(real)((sk+(double)gi.x[k])*ds);
        nd.w=(real)((double)gi.w[i]*(double)gi.w[j]*(double)gi.w[k]*wv);
        vol.push_back(nd);
      }
      continue;
    }
    for (i32 t = 0; t < 6; t++) {                  // cut: Kuhn split, march
      i32 v[4]; clKuhn(t, v);
      double tp[4][3], tf[4];
      for (i32 i = 0; i < 4; i++) {
        for (i32 d = 0; d < 3; d++) tp[i][d]=cp[v[i]][d];
        tf[i]=cf[v[i]];
      }
      clMarchTet(vol, srf, gv, gs, tp, tf);
    }
  }
}

// ---------------------------------------------------------------------------
//  Exact inside-outside test for the piecewise-linear model, and Potter's
//  Algorithm 4.2 candidate grid built on top of it.
//
//  A point's sub-cube is found by scaling; within it, the Kuhn tet is the one
//  whose Freudenthal path follows the DESCENDING order of the local coordinates
//  (the 6 tets of clKuhn are exactly the 6 orderings), and the interpolant is
//  linear along that path.  So this returns precisely the phi that clMarchTet
//  cut against -- the candidates are inside the same polyhedron the moments
//  were computed on, which is what stride-subsampling the emitted rule failed
//  to guarantee (it aliases against the structured Gauss blocks: measured 18%
//  worse L2 on the sphere at s=4).
// ---------------------------------------------------------------------------
static inline double cutLinPhi(const real *phiLat, i32 s, const double x[3]) {
  const i32 L = s+1;
  i32 c[3]; double u[3];
  for (i32 d = 0; d < 3; d++) {
    double t = x[d]*(double)s;
    if (t < 0) t = 0; if (t > s) t = (double)s;
    i32 i = (i32)t; if (i > s-1) i = s-1;
    c[d] = i; u[d] = t - (double)i;
  }
  i32 o[3] = {0,1,2};                       // sort the local coords DESCENDING
  if (u[o[0]] < u[o[1]]) { i32 t=o[0]; o[0]=o[1]; o[1]=t; }
  if (u[o[1]] < u[o[2]]) { i32 t=o[1]; o[1]=o[2]; o[2]=t; }
  if (u[o[0]] < u[o[1]]) { i32 t=o[0]; o[0]=o[1]; o[1]=t; }
  i32 g[3] = {c[0], c[1], c[2]};
  double f[4];
  f[0] = (double)phiLat[g[0] + L*(g[1] + L*g[2])];
  for (i32 k = 0; k < 3; k++) {
    g[o[k]] += 1;
    f[k+1] = (double)phiLat[g[0] + L*(g[1] + L*g[2])];
  }
  return f[0] + (f[1]-f[0])*u[o[0]] + (f[2]-f[1])*u[o[1]] + (f[3]-f[2])*u[o[2]];
}

// Uniform gN^3 grid of candidate nodes, kept where the piecewise-linear level
// set is < thresh (Potter Sec. 4.8 wants a slightly NEGATIVE threshold so the
// initial rule is not Lobatto; thresh = 0 keeps the plain interior test).
static inline void cutLinCandidates(const real *phiLat, i32 s, i32 gN,
                                    double thresh, std::vector<SayeNode> &cand) {
  cand.clear();
  for (i32 k = 0; k < gN; k++) for (i32 j = 0; j < gN; j++) for (i32 i = 0; i < gN; i++) {
    const double x[3] = { (i+0.5)/gN, (j+0.5)/gN, (k+0.5)/gN };
    if (cutLinPhi(phiLat, s, x) >= thresh) continue;
    SayeNode nd{};
    nd.x[0]=(real)x[0]; nd.x[1]=(real)x[1]; nd.x[2]=(real)x[2];
    cand.push_back(nd);
  }
}

#endif
