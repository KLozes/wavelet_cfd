#ifndef CUT_FEM_H
#define CUT_FEM_H

//
// Shared definitions for the CutFEM linear-elasticity solver (wavefem):
// Q1 shape functions, the level set that carves Omega out of the background
// grid, and the problem data (material, body force, boundary tagging, and the
// manufactured solution used for the convergence study).
//
// Method: Hansbo, Larson & Larsson, "Cut Finite Element Methods for Linear
// Elasticity Problems" (arXiv:1703.04377).  See CutFemSolver.cuh for how the
// paper's pieces map onto this grid.
//

#include "Util.cuh"
#include "Vec3f.cuh"
#include "Bvh.h"
#include "BvhQuery.h"
#include "BladeSdf.cuh"

// ---------------------------------------------------------------------------
//  Q1 trilinear hexahedron on the reference cube [0,1]^3
// ---------------------------------------------------------------------------
//
// Corner numbering n = a + 2b + 4c with (a,b,c) the x/y/z bits, matching the
// grid's (i,j,k) cell-corner convention.  A physical element is the cube
// [x0, x0+h]^3, so the map is a uniform scaling: gradients pick up 1/h,
// volumes h^3, areas h^2, and normals are unchanged.
//

__host__ __device__ inline void q1Shape(real xi, real et, real ze, real N[8]) {
  real sx[2] = {1-xi, xi}, sy[2] = {1-et, et}, sz[2] = {1-ze, ze};
  for (i32 n = 0; n < 8; n++)
    N[n] = sx[n&1] * sy[(n>>1)&1] * sz[(n>>2)&1];
}

// gradients in PHYSICAL coordinates (element size h)
__host__ __device__ inline void q1Grad(real xi, real et, real ze, real h, real G[8][3]) {
  real sx[2] = {1-xi, xi}, sy[2] = {1-et, et}, sz[2] = {1-ze, ze};
  real dd[2] = {-1, 1};
  real ih = 1/h;
  for (i32 n = 0; n < 8; n++) {
    i32 a = n&1, b = (n>>1)&1, c = (n>>2)&1;
    G[n][0] = dd[a] * sy[b] * sz[c] * ih;
    G[n][1] = sx[a] * dd[b] * sz[c] * ih;
    G[n][2] = sx[a] * sy[b] * dd[c] * ih;
  }
}

//
// Isoparametric Q1: physical gradients and det(J) from the element's 8 PHYSICAL
// corners.  On a cubic element this reduces exactly to q1Grad with J = h*I; on
// the cylindrical grid the element is a curved (r, theta, z) brick and the
// Jacobian carries the metric, so the elasticity operator itself stays plain
// Cartesian -- no curvilinear strain-displacement relations to derive.
//
__host__ __device__ inline bool q1GradIso(const real C[8][3], real xi, real et, real ze,
                                          real G[8][3], real &detJ, real Jinv[3][3]) {
  real sx[2] = {1-xi, xi}, sy[2] = {1-et, et}, sz[2] = {1-ze, ze};
  real dd[2] = {-1, 1};
  real dN[8][3];
  for (i32 n = 0; n < 8; n++) {
    i32 a = n&1, b = (n>>1)&1, c = (n>>2)&1;
    dN[n][0] = dd[a]*sy[b]*sz[c];
    dN[n][1] = sx[a]*dd[b]*sz[c];
    dN[n][2] = sx[a]*sy[b]*dd[c];
  }
  real J[3][3] = {{0,0,0},{0,0,0},{0,0,0}};      // J[i][j] = dx_i/dxi_j
  for (i32 n = 0; n < 8; n++)
    for (i32 i = 0; i < 3; i++)
      for (i32 j = 0; j < 3; j++) J[i][j] += C[n][i]*dN[n][j];

  real c00 = J[1][1]*J[2][2] - J[1][2]*J[2][1];
  real c01 = J[1][2]*J[2][0] - J[1][0]*J[2][2];
  real c02 = J[1][0]*J[2][1] - J[1][1]*J[2][0];
  detJ = J[0][0]*c00 + J[0][1]*c01 + J[0][2]*c02;
  if (!(fabs(detJ) > (real)1e-30)) { detJ = 0; return false; }
  real id = 1/detJ;
  Jinv[0][0] = c00*id;
  Jinv[1][0] = c01*id;
  Jinv[2][0] = c02*id;
  Jinv[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2])*id;
  Jinv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0])*id;
  Jinv[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1])*id;
  Jinv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1])*id;
  Jinv[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2])*id;
  Jinv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0])*id;
  // grad_x N = J^-T grad_xi N
  for (i32 n = 0; n < 8; n++)
    for (i32 i = 0; i < 3; i++) {
      real g = 0;
      for (i32 j = 0; j < 3; j++) g += Jinv[j][i]*dN[n][j];
      G[n][i] = g;
    }
  return true;
}

// physical position of a reference point, from the element's corners
__host__ __device__ inline void q1PosIso(const real C[8][3], real xi, real et, real ze,
                                         real X[3]) {
  real N[8]; q1Shape(xi, et, ze, N);
  X[0] = X[1] = X[2] = 0;
  for (i32 n = 0; n < 8; n++)
    for (i32 i = 0; i < 3; i++) X[i] += N[n]*C[n][i];
}

// trilinear interpolation of 8 nodal values
__host__ __device__ inline real q1Interp(const real f[8], real xi, real et, real ze) {
  real N[8]; q1Shape(xi, et, ze, N);
  real s = 0;
  for (i32 n = 0; n < 8; n++) s += N[n]*f[n];
  return s;
}

// ---------------------------------------------------------------------------
//  Level set:  Omega = { phi < 0 }
// ---------------------------------------------------------------------------
//
// BladeSdf (BladeSdf.cuh) is the one geometry type: with the platform, fillet,
// tip gap and sector cut all switched off it degenerates to the plain BVH
// signed distance of whatever closed surface it was given, which is exactly the
// STL case.
//
// The oracle is only ever called at GRID NODES; every cut computation
// downstream uses the trilinear interpolant of those nodal values, which is
// what makes the discrete domain Omega_h well defined and single valued.
//
typedef BladeSdf LevelSet;

// ---------------------------------------------------------------------------
//  Problem data
// ---------------------------------------------------------------------------
//
// CASE_MMS   manufactured solution (convergence study).  The whole boundary is
//            Dirichlet, u is trigonometric, and f = -div sigma(u) is closed
//            form -- none of which needs the DOMAIN to be analytic, so this
//            runs on the STL body directly.
// CASE_LOAD  engineering load case: clamped where x < clampX (u = 0), traction
//            elsewhere, gravity body load.
//
// CASE_MMS_CYL: a manufactured solution COMPATIBLE with the cyclic pitch tie
// u(theta+pitch)=R(pitch)u(theta).  The Cartesian trig CASE_MMS is NOT (a
// rotation of the domain does not rotate that u into itself), so it cannot
// converge on the periodic sector -- its error floors at O(1) at the seam.  A
// field whose cylindrical components are theta-independent IS compatible:
//   u = (A x, A y, sin(kz z))  (uniform radial expansion + an axial wave),
// for which f = -div sigma(u) = (0, 0, (2mu+lam) kz^2 sin(kz z)).
enum FemCase { CASE_MMS = 0, CASE_LOAD = 1, CASE_MMS_CYL = 2 };
static constexpr real MMSCYL_A = (real)0.1;   // radial-expansion amplitude

struct FemProblem {
  i32  caseId;
  real mu, lam, rho;
  real kw[3];       // MMS wave numbers (k1,k2,k3)
  real gvec[3];     // body force per unit volume (CASE_LOAD)
  real trac[3];     // uniform Neumann traction, applied where x >= tracX0
  real tracX0;
  real clampX;      // Dirichlet part of dOmega (bcMode 0): x < clampX
  i32  bcMode;      // 0 = axial plane, 1 = platform underside (bladed sector)
  // bcMode 1 clamps the platform underside, r = r_hub(z) - platThick.  That
  // surface FOLLOWS THE SLOPING HUB -- over one blade row the hub radius moves
  // far more than a mesh cell, so a constant clamp radius would tag a band
  // across the middle of the solid instead of the attachment face.  The hub
  // table is carried here so the test is the real surface.
  const real *hubTab;
  i32  nWall;
  real wallZ0, wallDz, platThick, clampTol;
  real omega;       // shaft speed (rad/s): adds the centrifugal body force
                    // rho*omega^2*r outward, which is THE blade load case

  // --- manufactured displacement --------------------------------------------
  //   u1 = sin(k1 x) cos(k2 y) cos(k3 z)
  //   u2 = cos(k1 x) sin(k2 y) cos(k3 z)
  //   u3 = cos(k1 x) cos(k2 y) sin(k3 z)
  // so  div u = (k1+k2+k3) cos cos cos,  lap u_i = -(k1^2+k2^2+k3^2) u_i,  and
  //   -div sigma(u) = -mu lap u - (mu+lam) grad(div u)
  //                 = [mu K^2 + (mu+lam) S k_i] u_i   (componentwise).
  __host__ __device__ void exactU(real x, real y, real z, real u[3]) const {
    if (caseId == CASE_MMS_CYL) {
      u[0] = MMSCYL_A*x; u[1] = MMSCYL_A*y; u[2] = sin(kw[2]*z);
      return;
    }
    real s1 = sin(kw[0]*x), c1 = cos(kw[0]*x);
    real s2 = sin(kw[1]*y), c2 = cos(kw[1]*y);
    real s3 = sin(kw[2]*z), c3 = cos(kw[2]*z);
    u[0] = s1*c2*c3;
    u[1] = c1*s2*c3;
    u[2] = c1*c2*s3;
  }

  // gu[i][j] = du_i/dx_j
  __host__ __device__ void exactGradU(real x, real y, real z, real gu[3][3]) const {
    if (caseId == CASE_MMS_CYL) {
      for (i32 i=0;i<3;i++) for (i32 j=0;j<3;j++) gu[i][j]=0;
      gu[0][0]=MMSCYL_A; gu[1][1]=MMSCYL_A; gu[2][2]=kw[2]*cos(kw[2]*z);
      return;
    }
    real k1 = kw[0], k2 = kw[1], k3 = kw[2];
    real s1 = sin(k1*x), c1 = cos(k1*x);
    real s2 = sin(k2*y), c2 = cos(k2*y);
    real s3 = sin(k3*z), c3 = cos(k3*z);
    gu[0][0] =  k1*c1*c2*c3;  gu[0][1] = -k2*s1*s2*c3;  gu[0][2] = -k3*s1*c2*s3;
    gu[1][0] = -k1*s1*s2*c3;  gu[1][1] =  k2*c1*c2*c3;  gu[1][2] = -k3*c1*s2*s3;
    gu[2][0] = -k1*s1*c2*s3;  gu[2][1] = -k2*c1*s2*s3;  gu[2][2] =  k3*c1*c2*c3;
  }

  __host__ __device__ void bodyForce(real x, real y, real z, real f[3]) const {
    if (caseId == CASE_MMS_CYL) {
      f[0] = 0; f[1] = 0;
      f[2] = (2*mu + lam)*kw[2]*kw[2]*sin(kw[2]*z);
      return;
    }
    if (caseId != CASE_MMS) {
      f[0] = gvec[0]; f[1] = gvec[1]; f[2] = gvec[2];
      if (omega > 0) {                       // centrifugal: rho * omega^2 * r
        real w2 = rho*omega*omega;
        f[0] += w2*x; f[1] += w2*y;
      }
      return;
    }
    real u[3]; exactU(x, y, z, u);
    real K2 = kw[0]*kw[0] + kw[1]*kw[1] + kw[2]*kw[2];
    real S  = kw[0] + kw[1] + kw[2];
    for (i32 i = 0; i < 3; i++)
      f[i] = (mu*K2 + (mu+lam)*S*kw[i]) * u[i];
  }

  // sigma(u).n  for a given displacement gradient
  __host__ __device__ static void traction(const real gu[3][3], real mu_, real lam_,
                                           const real n[3], real g[3]) {
    real div = gu[0][0] + gu[1][1] + gu[2][2];
    for (i32 i = 0; i < 3; i++) {
      real s = lam_*div*n[i];
      for (i32 j = 0; j < 3; j++) s += mu_*(gu[i][j] + gu[j][i])*n[j];
      g[i] = s;
    }
  }

  __host__ __device__ bool isDirichlet(real x, real y, real z) const {
    if (caseId == CASE_MMS || caseId == CASE_MMS_CYL) return true;   // pure displacement
    if (bcMode == 1) {
      real u = (z - wallZ0)/wallDz;
      i32 i = (i32)floor(u);
      if (i < 0) i = 0;
      if (i > nWall-2) i = nWall-2;
      real rh = hubTab[i] + (u - i)*(hubTab[i+1] - hubTab[i]);
      return sqrt(x*x + y*y) < rh - platThick + clampTol;
    }
    return x < clampX;
  }

  __host__ __device__ void dirichletData(real x, real y, real z, real g[3]) const {
    if (caseId == CASE_MMS || caseId == CASE_MMS_CYL) { exactU(x, y, z, g); return; }
    g[0] = g[1] = g[2] = 0;                       // clamped
  }

  __host__ __device__ void neumannData(real x, real y, real z,
                                       const real n[3], real g[3]) const {
    if (caseId == CASE_MMS || caseId == CASE_MMS_CYL) {
      real gu[3][3]; exactGradU(x, y, z, gu);
      traction(gu, mu, lam, n, g);
      return;
    }
    if (x >= tracX0) { g[0] = trac[0]; g[1] = trac[1]; g[2] = trac[2]; }
    else             { g[0] = g[1] = g[2] = 0; }
  }
};

// von Mises stress from a displacement gradient
__host__ __device__ inline real vonMises(const real gu[3][3], real mu, real lam) {
  real div = gu[0][0] + gu[1][1] + gu[2][2];
  real s[3][3];
  for (i32 i = 0; i < 3; i++)
    for (i32 j = 0; j < 3; j++)
      s[i][j] = mu*(gu[i][j] + gu[j][i]) + (i==j ? lam*div : (real)0);
  real tr = (s[0][0] + s[1][1] + s[2][2])/3;
  real d = 0;
  for (i32 i = 0; i < 3; i++)
    for (i32 j = 0; j < 3; j++) {
      real dij = s[i][j] - (i==j ? tr : (real)0);
      d += dij*dij;
    }
  return sqrt((real)1.5*d);
}

#endif
