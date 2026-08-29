#ifndef CUT_QUAD_H
#define CUT_QUAD_H

//
// Cut-cell quadrature by marching tetrahedra.
//
// The paper integrates over K \cap Omega with a boundary-representation rule
// (Sec 3.2).  Here the geometry arrives as a level set sampled at the FE nodes,
// so the natural equivalent is:
//
//   1. optionally split the element into cutSub^3 sub-cubes (phi at sub-corners
//      from the element's trilinear interpolant),
//   2. Kuhn-split each sub-cube into 6 tetrahedra sharing the 0-7 diagonal,
//   3. on each tet, interpolate phi LINEARLY and apply marching tetrahedra:
//      the {phi < 0} part is a tet, a wedge (2 in / 2 out) or a wedge (3 in),
//      and the interface is one triangle or two.
//
// The interface is therefore piecewise planar -- an O(h^2) geometry error,
// exactly matched to the p=1 elements used here (Thm 4.2 gives O(h^2) in L^2).
//
// All coordinates are REFERENCE coordinates in the unit cube [0,1]^3.  The
// element map is a uniform scaling by h, so volume weights carry h^3, surface
// weights h^2, and normals are unchanged.
//

#include "Util.cuh"
#include "CutFem.cuh"

// capacity: cutSub <= 2 -> 48 tets; a tet emits at most 3 sub-tets (4 points
// each) and 2 interface triangles (3 points each).
static constexpr i32 FEM_MAXSUB = 2;
static constexpr i32 FEM_MAXTET = 6*FEM_MAXSUB*FEM_MAXSUB*FEM_MAXSUB;   // 48
static constexpr i32 FEM_MAXQV  = FEM_MAXTET*3*4;                       // 576
static constexpr i32 FEM_MAXQS  = FEM_MAXTET*2*3;                       // 288

// Kuhn decomposition of the unit cube into 6 tets sharing the 0--7 diagonal.
// Each is one monotone bit-flip path from corner 0 to corner 7.
__host__ __device__ inline void kuhnTet(i32 t, i32 v[4]) {
  const i32 K[6][4] = {{0,1,3,7}, {0,1,5,7}, {0,4,5,7},
                       {0,4,6,7}, {0,2,6,7}, {0,2,3,7}};
  for (i32 i = 0; i < 4; i++) v[i] = K[t][i];
}

// 4-point degree-2 tetrahedron rule (barycentric), weights V/4 each
static constexpr real TET4_A = (real)0.5854101966249685;
static constexpr real TET4_B = (real)0.1381966011250105;

__host__ __device__ inline real tetVolume(const real p0[3], const real p1[3],
                                          const real p2[3], const real p3[3]) {
  real a[3], b[3], c[3];
  for (i32 d = 0; d < 3; d++) { a[d] = p1[d]-p0[d]; b[d] = p2[d]-p0[d]; c[d] = p3[d]-p0[d]; }
  real det = a[0]*(b[1]*c[2]-b[2]*c[1])
           - a[1]*(b[0]*c[2]-b[2]*c[0])
           + a[2]*(b[0]*c[1]-b[1]*c[0]);
  return fabs(det)/6;
}

//
// Quadrature buffers filled by cutQuadrature().  Volume points carry the
// reference position and a weight that already includes h^3; surface points
// additionally carry the outward unit normal and a weight including h^2.
//
struct CutQuadBuf {
  i32  nv, ns;
  real *qv;   // [4*nv]  xi, eta, zeta, w
  real *qs;   // [7*ns]  xi, eta, zeta, w, nx, ny, nz
  i32  maxv, maxs;
  // optional: raw interface TRIANGLES (9 reals each) for surface output.
  // Left null by the device kernels, which only need quadrature points.
  real *tri = nullptr;
  i32   nt = 0, maxt = 0;
};

__host__ __device__ inline void cqAddTet(CutQuadBuf &B, real h3,
                                         const real p0[3], const real p1[3],
                                         const real p2[3], const real p3[3]) {
  real V = tetVolume(p0, p1, p2, p3);
  if (V <= 0) return;
  real w = V*h3/4;
  const real *P[4] = {p0, p1, p2, p3};
  for (i32 q = 0; q < 4; q++) {
    if (B.nv >= B.maxv) return;
    real *o = B.qv + 4*B.nv;
    for (i32 d = 0; d < 3; d++) {
      real s = 0;
      for (i32 v = 0; v < 4; v++) s += (v == q ? TET4_A : TET4_B) * P[v][d];
      o[d] = s;
    }
    o[3] = w;
    B.nv++;
  }
}

// One interface triangle.  `xout` is a point strictly on the phi > 0 side, used
// to orient the normal outward from Omega.
__host__ __device__ inline void cqAddTri(CutQuadBuf &B, real h2,
                                         const real p0[3], const real p1[3],
                                         const real p2[3], const real xout[3]) {
  real e1[3], e2[3], n[3];
  for (i32 d = 0; d < 3; d++) { e1[d] = p1[d]-p0[d]; e2[d] = p2[d]-p0[d]; }
  n[0] = e1[1]*e2[2] - e1[2]*e2[1];
  n[1] = e1[2]*e2[0] - e1[0]*e2[2];
  n[2] = e1[0]*e2[1] - e1[1]*e2[0];
  real nn = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (nn <= (real)1e-24) return;
  real A = nn/2;
  for (i32 d = 0; d < 3; d++) n[d] /= nn;
  // orient outward: from the triangle centroid toward the outside vertex
  real ctr[3], dot = 0;
  for (i32 d = 0; d < 3; d++) {
    ctr[d] = (p0[d]+p1[d]+p2[d])/3;
    dot += n[d]*(xout[d]-ctr[d]);
  }
  if (dot < 0) for (i32 d = 0; d < 3; d++) n[d] = -n[d];

  if (B.tri && B.nt < B.maxt) {           // surface-output path (host only)
    real *o = B.tri + 9*B.nt;
    for (i32 d = 0; d < 3; d++) { o[d] = p0[d]; o[3+d] = p1[d]; o[6+d] = p2[d]; }
    B.nt++;
  }

  // 3-point degree-2 rule: edge midpoints, weight A/3
  const real MID[3][3] = {{(real)0.5,(real)0.5,0}, {0,(real)0.5,(real)0.5}, {(real)0.5,0,(real)0.5}};
  const real *P[3] = {p0, p1, p2};
  for (i32 q = 0; q < 3; q++) {
    if (B.ns >= B.maxs) return;
    real *o = B.qs + 7*B.ns;
    for (i32 d = 0; d < 3; d++)
      o[d] = MID[q][0]*P[0][d] + MID[q][1]*P[1][d] + MID[q][2]*P[2][d];
    o[3] = A*h2/3;
    o[4] = n[0]; o[5] = n[1]; o[6] = n[2];
    B.ns++;
  }
}

// linear interpolation of the zero crossing on edge (a,b)
__host__ __device__ inline void cqEdge(const real pa[3], real fa,
                                       const real pb[3], real fb, real out[3]) {
  real t = fa / (fa - fb);
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  for (i32 d = 0; d < 3; d++) out[d] = pa[d] + t*(pb[d]-pa[d]);
}

//
// Marching tetrahedra on one tet.  p[4] are reference-space vertices, f[4] the
// (linearly interpolated) level-set values.  Emits the {f < 0} sub-volume and
// the interface triangles.
//
__host__ __device__ inline void cqMarchTet(CutQuadBuf &B, real h3, real h2,
                                           const real p[4][3], const real f[4]) {
  i32 in[4], out[4], nin = 0, nout = 0;
  for (i32 v = 0; v < 4; v++) {
    if (f[v] < 0) in[nin++] = v; else out[nout++] = v;
  }
  if (nin == 0) return;
  if (nin == 4) { cqAddTet(B, h3, p[0], p[1], p[2], p[3]); return; }

  // centroid of the outside vertices: orients the interface normal
  real xout[3] = {0,0,0};
  for (i32 i = 0; i < nout; i++)
    for (i32 d = 0; d < 3; d++) xout[d] += p[out[i]][d]/nout;

  if (nin == 1) {
    i32 a = in[0];
    real q[3][3];
    for (i32 i = 0; i < 3; i++) cqEdge(p[a], f[a], p[out[i]], f[out[i]], q[i]);
    cqAddTet(B, h3, p[a], q[0], q[1], q[2]);
    cqAddTri(B, h2, q[0], q[1], q[2], xout);
    return;
  }

  if (nin == 3) {
    i32 b = out[0];
    real q[3][3];
    for (i32 i = 0; i < 3; i++) cqEdge(p[in[i]], f[in[i]], p[b], f[b], q[i]);
    // wedge (in0,in1,in2 | q0,q1,q2) -> 3 tets
    const real *A0 = p[in[0]], *A1 = p[in[1]], *A2 = p[in[2]];
    cqAddTet(B, h3, A0, A1, A2, q[0]);
    cqAddTet(B, h3, A1, A2, q[0], q[1]);
    cqAddTet(B, h3, A2, q[0], q[1], q[2]);
    cqAddTri(B, h2, q[0], q[1], q[2], xout);
    return;
  }

  // nin == 2: the inside region is a wedge with triangles
  //   (A, qAC, qAD) and (B, qBC, qBD)
  i32 A = in[0], Bv = in[1], C = out[0], D = out[1];
  real qAC[3], qAD[3], qBC[3], qBD[3];
  cqEdge(p[A],  f[A],  p[C], f[C], qAC);
  cqEdge(p[A],  f[A],  p[D], f[D], qAD);
  cqEdge(p[Bv], f[Bv], p[C], f[C], qBC);
  cqEdge(p[Bv], f[Bv], p[D], f[D], qBD);
  cqAddTet(B, h3, p[A], qAC, qAD, p[Bv]);
  cqAddTet(B, h3, qAC,  qAD, p[Bv], qBC);
  cqAddTet(B, h3, qAD,  p[Bv], qBC, qBD);
  // interface quad, cycle qAC - qBC - qBD - qAD
  cqAddTri(B, h2, qAC, qBC, qBD, xout);
  cqAddTri(B, h2, qAC, qBD, qAD, xout);
}

//
// ONE tetrahedron of the cut rule: `t` runs over 6*sub^3 (sub-cube major, Kuhn
// tet minor).  Appending to B rather than owning it lets a CUDA block share
// the work across threads, each with a small private buffer.
//   phiN  the element's 8 nodal level-set values
//   sub   sub-cube count per direction (1 or 2)
//   h     element size
//
__host__ __device__ inline void cqOneTet(CutQuadBuf &B, const real phiN[8],
                                         i32 sub, real h, i32 t) {
  i32 sc = t/6, kt = t%6;
  i32 si = sc % sub, sj = (sc/sub) % sub, sk = sc/(sub*sub);
  real ds = (real)1/sub;
  real cp[8][3], cf[8];
  for (i32 n = 0; n < 8; n++) {
    cp[n][0] = (si + (n&1))*ds;
    cp[n][1] = (sj + ((n>>1)&1))*ds;
    cp[n][2] = (sk + ((n>>2)&1))*ds;
    cf[n] = q1Interp(phiN, cp[n][0], cp[n][1], cp[n][2]);
  }
  i32 v[4]; kuhnTet(kt, v);
  real tp[4][3], tf[4];
  for (i32 i = 0; i < 4; i++) {
    for (i32 d = 0; d < 3; d++) tp[i][d] = cp[v[i]][d];
    tf[i] = cf[v[i]];
  }
  // REFERENCE measure: weights are volumes/areas on the unit cube.  The
  // physical measure is applied by the element kernel through det(J) and
  // Nanson's formula, which is what lets the same rule serve a cubic Cartesian
  // element and a curved (r, theta, z) brick.
  (void)h;
  cqMarchTet(B, (real)1, (real)1, tp, tf);
}

// full cut-cell rule for one element (host convenience: surface extraction)
__host__ __device__ inline void cutQuadrature(CutQuadBuf &B, const real phiN[8],
                                              i32 sub, real h) {
  B.nv = B.ns = B.nt = 0;
  for (i32 t = 0; t < 6*sub*sub*sub; t++) cqOneTet(B, phiN, sub, h, t);
}

// 2x2x2 Gauss on the reference cube (exact for the Q1 stiffness integrand)
static constexpr real GAUSS2 = (real)0.2113248654051871;   // (1 - 1/sqrt(3))/2

__host__ __device__ inline void gauss2Point(i32 q, real x[3], real &w) {
  const real g[2] = {GAUSS2, 1-GAUSS2};
  x[0] = g[q&1]; x[1] = g[(q>>1)&1]; x[2] = g[(q>>2)&1];
  w = (real)0.125;                                   // reference-cube weight
}

#endif
