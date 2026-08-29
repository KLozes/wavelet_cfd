#ifndef FEM_SBMSHIFT_H
#define FEM_SBMSHIFT_H

//
// Shifted Boundary Method (SBM) shift operator, for the Gap-SBM path (CutFemSbm).
//
// SBM imposes boundary conditions on a SURROGATE boundary (full mesh faces ~1 cell
// inside the true boundary) rather than cutting cells.  The link to the true
// boundary is the TAYLOR SHIFT: for a field w defined on the surrogate cell,
//
//   S_h w(x~) = w + grad w . d + 1/2 d^T (grad^2 w) d + ... + (1/p!)(grad^p w)[d..d]
//             ~ w(x~ + d) = w(x_true),
//
// truncated at total order p (matching the Q_p field) -> O(h^{p+1}) consistency,
// where d = x_true - x~ is the distance vector from the oracle.
//
// Because the field is a tensor Q_p polynomial, S_h w is assembled per basis
// function:  S_h phi_a(x~) = sum_{|alpha|<=p} (1/alpha!) dref^alpha
//                            * l_i^{(ax)}(x~_x) l_j^{(ay)}(x~_y) l_k^{(az)}(x~_z)
// with a=(i,j,k) and dref = d / h (Cartesian: physical deriv = ref deriv / h).  So
// the SBM boundary term reuses the SAME nodal dofs, weighted by S_h phi_a instead
// of the plain trace phi_a.  This header provides:
//   * sbmDerivMatrix: the inverse Vandermonde (monomial coeffs of the 1-D basis)
//   * deriv1: the m-th derivative of every 1-D basis function at an arbitrary x
//   * sbmShiftAll: S_h phi_a for every basis a at a surrogate reference point
// Geometry-agnostic (the caller supplies x~ in the owning cell's ref coords and
// the reference shift vector dref); host+device.
//

#include "LagrangeBasis.h"
#include "PolyFit.h"    // vandermondeInv

// monomial coefficients of the 1-D Lagrange basis: Vm[k][a] = coeff of x^k in l_a.
// (l_a interpolates e_a, so its coeffs are column a of the inverse Vandermonde.)
__host__ __device__ inline void sbmDerivMatrix(const LagrangeBasis &B, real Vm[QN_MAX][QN_MAX]) {
  vandermondeInv(B.p, B.t, Vm);   // Vm[k][a] = (V^{-1})[k][a]
}

// out[a] = l_a^{(m)}(x)  for all a  (m-th derivative of each 1-D basis at x)
__host__ __device__ inline void deriv1(const LagrangeBasis &B, const real Vm[QN_MAX][QN_MAX],
                                       real x, i32 m, real out[QN_MAX]) {
  i32 n = B.n, p = B.p;
  for (i32 a = 0; a < n; a++) {
    real s = 0, xp = 1;                       // xp = x^{k-m}
    for (i32 k = m; k <= p; k++) {
      real ff = 1;                            // falling factorial k!/(k-m)!
      for (i32 q = 0; q < m; q++) ff *= (real)(k - q);
      s += Vm[k][a] * ff * xp;
      xp *= x;
    }
    out[a] = s;
  }
}

// S_h phi_a for every basis a=(i,j,k) at surrogate ref point xr[3] with reference
// shift dref[3] (= physical distance vector / h).  out has (p+1)^3 entries.
// Truncated Taylor at total order p (|alpha| <= p) -> O(h^{p+1}).
__host__ __device__ inline void sbmShiftAll(const LagrangeBasis &B, const real Vm[QN_MAX][QN_MAX],
                                            const real xr[3], const real dref[3],
                                            real *out) {
  i32 n = B.n, p = B.p;
  real Lx[QN_MAX][QN_MAX], Ly[QN_MAX][QN_MAX], Lz[QN_MAX][QN_MAX];   // [m][a], m=0..p
  for (i32 m = 0; m <= p; m++) {
    deriv1(B, Vm, xr[0], m, Lx[m]);
    deriv1(B, Vm, xr[1], m, Ly[m]);
    deriv1(B, Vm, xr[2], m, Lz[m]);
  }
  real dpx[QN_MAX], dpy[QN_MAX], dpz[QN_MAX];   // dref^m / m!
  dpx[0]=dpy[0]=dpz[0]=1;
  for (i32 m = 1; m <= p; m++) {
    dpx[m] = dpx[m-1]*dref[0]/m;
    dpy[m] = dpy[m-1]*dref[1]/m;
    dpz[m] = dpz[m-1]*dref[2]/m;
  }
  for (i32 a = 0; a < n*n*n; a++) out[a] = 0;
  for (i32 mx = 0; mx <= p; mx++)
  for (i32 my = 0; my <= p-mx; my++)
  for (i32 mz = 0; mz <= p-mx-my; mz++) {
    real cx[QN_MAX], cy[QN_MAX], cz[QN_MAX];
    for (i32 i = 0; i < n; i++) cx[i] = dpx[mx]*Lx[mx][i];
    for (i32 j = 0; j < n; j++) cy[j] = dpy[my]*Ly[my][j];
    for (i32 k = 0; k < n; k++) cz[k] = dpz[mz]*Lz[mz][k];
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++)
      out[i + n*(j + n*k)] += cx[i]*cy[j]*cz[k];
  }
}

// distance vector d and true unit normal nu at a point where the level set is phi
// with gradient g[3] (one Newton step to {phi=0}; exact for a true SDF).  Returns
// the (approximate) distance |d|.  Caller supplies phi, g from the oracle.
__host__ __device__ inline real sbmDistNormal(real phi, const real g[3],
                                              real d[3], real nu[3]) {
  real g2 = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
  real inv = (g2 > (real)1e-30) ? (real)1/g2 : 0;
  for (i32 i = 0; i < 3; i++) d[i] = -phi*g[i]*inv;          // Newton step to phi=0
  real gm = sqrt(g2), invm = gm>0 ? (real)1/gm : 0;
  for (i32 i = 0; i < 3; i++) nu[i] = g[i]*invm;
  return sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
}

#endif
