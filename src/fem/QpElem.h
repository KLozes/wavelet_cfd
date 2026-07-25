#ifndef FEM_QPELEM_H
#define FEM_QPELEM_H

//
// Qp element bulk elasticity operator (matrix-free), the M1 core.
//   a_e(u,v) = int_Omega_e  sigma(u) : eps(v) dx,
//   sigma = 2 mu eps + lambda tr(eps) I,   eps = sym(grad u).
//
// Isoparametric cube of size h: grad_phys = (1/h) grad_ref, dx = h^3 dxi, which
// collapse to a single factor h.  So
//   y_a[i] = h * sum_q w_q * sum_j sigma_ref[i][j] (grad_ref phi_a)[j]
// with sigma_ref built from the REFERENCE gradients.
//
// Two quadrature sources, same core:
//   * uncut element -> tensor GLL rule (sum-factorizable; here direct)
//   * cut element   -> Saye volume rule on {phi<0}  (SayeQuad.h)
//
// Cylindrical / general Jacobian is a later refinement (M4); this is the
// Cartesian constant-Jacobian operator used to pass the M1 MMS gate.
//

#include "QpBasis.h"
#include "SayeQuad.h"

// action of the element bulk stiffness on nodal displacement u -> y, given an
// explicit quadrature (points in [0,1]^3, weights).  u,y packed [3*a + i].
__host__ __device__ inline void qpElemCore(const QpBasis &B, real mu, real lam,
                                           real h, const real (*pts)[3],
                                           const real *w, i32 npts,
                                           const real *u, real *y) {
  i32 ndof = B.n*B.n*B.n;
  for (i32 a = 0; a < 3*ndof; a++) y[a] = 0;
  real gb[3*QN_MAX*QN_MAX*QN_MAX];
  for (i32 q = 0; q < npts; q++) {
    real x[3] = { pts[q][0], pts[q][1], pts[q][2] };
    B.allGradRef(x, gb);
    // gradU[i][j] = d u_i / d xi_j
    real gradU[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (i32 a = 0; a < ndof; a++)
      for (i32 i = 0; i < 3; i++) {
        real ui = u[3*a+i];
        gradU[i][0] += ui*gb[3*a+0];
        gradU[i][1] += ui*gb[3*a+1];
        gradU[i][2] += ui*gb[3*a+2];
      }
    real eps[3][3];
    for (i32 i = 0; i < 3; i++) for (i32 j = 0; j < 3; j++)
      eps[i][j] = (real)0.5*(gradU[i][j] + gradU[j][i]);
    real tr = eps[0][0]+eps[1][1]+eps[2][2];
    real sig[3][3];
    for (i32 i = 0; i < 3; i++) for (i32 j = 0; j < 3; j++)
      sig[i][j] = 2*mu*eps[i][j] + (i==j ? lam*tr : (real)0);
    real wq = w[q];
    for (i32 a = 0; a < ndof; a++)
      for (i32 i = 0; i < 3; i++)
        y[3*a+i] += wq*(sig[i][0]*gb[3*a+0] + sig[i][1]*gb[3*a+1] + sig[i][2]*gb[3*a+2]);
  }
  for (i32 a = 0; a < 3*ndof; a++) y[a] *= h;
}

// uncut element: tensor GLL quadrature (weights wq_i wq_j wq_k at nodes)
__host__ __device__ inline void qpElemUncut(const QpBasis &B, real mu, real lam,
                                            real h, const real *u, real *y) {
  i32 n = B.n, npts = n*n*n;
  real pts[QN_MAX*QN_MAX*QN_MAX][3], w[QN_MAX*QN_MAX*QN_MAX];
  i32 q = 0;
  for (i32 k = 0; k < n; k++)
  for (i32 j = 0; j < n; j++)
  for (i32 i = 0; i < n; i++) {
    pts[q][0]=B.t[i]; pts[q][1]=B.t[j]; pts[q][2]=B.t[k];
    w[q] = B.wq[i]*B.wq[j]*B.wq[k]; q++;
  }
  qpElemCore(B, mu, lam, h, pts, w, npts, u, y);
}

// action of the element bulk stiffness using a Saye node list directly
// (points+weights carried in SayeNode.x / .w).  Same core, strided input.
__host__ __device__ inline void qpElemCoreSaye(const QpBasis &B, real mu, real lam,
                                               real h, const SayeNode *nodes,
                                               i32 npts, const real *u, real *y) {
  i32 ndof = B.n*B.n*B.n;
  for (i32 a = 0; a < 3*ndof; a++) y[a] = 0;
  real gb[3*QN_MAX*QN_MAX*QN_MAX];
  for (i32 q = 0; q < npts; q++) {
    real x[3] = { nodes[q].x[0], nodes[q].x[1], nodes[q].x[2] };
    B.allGradRef(x, gb);
    real gradU[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (i32 a = 0; a < ndof; a++)
      for (i32 i = 0; i < 3; i++) {
        real ui = u[3*a+i];
        gradU[i][0] += ui*gb[3*a+0];
        gradU[i][1] += ui*gb[3*a+1];
        gradU[i][2] += ui*gb[3*a+2];
      }
    real eps[3][3];
    for (i32 i = 0; i < 3; i++) for (i32 j = 0; j < 3; j++)
      eps[i][j] = (real)0.5*(gradU[i][j] + gradU[j][i]);
    real tr = eps[0][0]+eps[1][1]+eps[2][2];
    real sig[3][3];
    for (i32 i = 0; i < 3; i++) for (i32 j = 0; j < 3; j++)
      sig[i][j] = 2*mu*eps[i][j] + (i==j ? lam*tr : (real)0);
    real wq = nodes[q].w;
    for (i32 a = 0; a < ndof; a++)
      for (i32 i = 0; i < 3; i++)
        y[3*a+i] += wq*(sig[i][0]*gb[3*a+0] + sig[i][1]*gb[3*a+1] + sig[i][2]*gb[3*a+2]);
  }
  for (i32 a = 0; a < 3*ndof; a++) y[a] *= h;
}

// cut element: Saye volume rule for {phi<0}; phi = degree-p fit of nodal values
__host__ __device__ inline void qpElemCut(const QpBasis &B, real mu, real lam,
                                          real h, const PolyND &phi,
                                          const real *u, real *y,
                                          SayeNode *arenaBuf, i32 arenaCap,
                                          SayeNode *outBuf, i32 outCap) {
  SayeArena ar; ar.buf = arenaBuf; ar.cap = arenaCap; ar.top = 0;
  SayeSet out; out.p = outBuf; out.n = 0; out.cap = outCap; out.ovf = false;
  sayeVolume(phi, &out, &ar, SayeCfg::def());
  qpElemCoreSaye(B, mu, lam, h, out.p, out.n, u, y);
}

#endif
