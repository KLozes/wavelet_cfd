#ifndef FEM_POLYFIT_H
#define FEM_POLYFIT_H

//
// Fit a tensor monomial PolyND (degree p per axis) to level-set values sampled
// at the (p+1)^3 tensor Gauss-Lobatto-Legendre nodes of a cell -- i.e. build the
// Qp Lagrange interpolant of "the level set at the solution points" and hand it
// to the Saye recursion in the monomial basis it wants.
//
// GLL nodes are the Qp solution nodes, so in the solver phi is already stored
// there; this converts nodal -> monomial once per cut cell.  Small (p+1)x(p+1)
// Vandermonde inverse, precomputed per p, applied as a tensor contraction.
//

#include "Poly.h"

// GLL nodes on [0,1] for p = 1..3 (matches the Qp basis)
__host__ __device__ inline void gllNodes(i32 p, real t[PNC]) {
  if (p == 1) { t[0]=0; t[1]=1; }
  else if (p == 2) { t[0]=0; t[1]=(real)0.5; t[2]=1; }
  else { t[0]=0; t[1]=(real)0.2763932023; t[2]=(real)0.7236067977; t[3]=1; }
}

// invert the (p+1)x(p+1) Vandermonde  V[i][j] = t_i^j  ->  Vinv (row-major)
__host__ __device__ inline void vandermondeInv(i32 p, const real t[PNC],
                                              real Vinv[PNC][PNC]) {
  i32 n = p+1;
  real A[PNC][2*PNC];
  for (i32 i = 0; i < n; i++) {
    real tp = 1;
    for (i32 j = 0; j < n; j++) { A[i][j] = tp; tp *= t[i]; }
    for (i32 j = 0; j < n; j++) A[i][n+j] = (i==j) ? 1 : 0;
  }
  // Gauss-Jordan with partial pivoting
  for (i32 c = 0; c < n; c++) {
    i32 piv = c; real best = fabs(A[c][c]);
    for (i32 r = c+1; r < n; r++) if (fabs(A[r][c]) > best) { best = fabs(A[r][c]); piv = r; }
    if (piv != c) for (i32 j = 0; j < 2*n; j++) { real tmp=A[c][j]; A[c][j]=A[piv][j]; A[piv][j]=tmp; }
    real d = A[c][c];
    for (i32 j = 0; j < 2*n; j++) A[c][j] /= d;
    for (i32 r = 0; r < n; r++) if (r != c) {
      real f = A[r][c];
      for (i32 j = 0; j < 2*n; j++) A[r][j] -= f*A[c][j];
    }
  }
  for (i32 i = 0; i < n; i++) for (i32 j = 0; j < n; j++) Vinv[i][j] = A[i][n+j];
}

// build a degree-p PolyND on [0,1]^3 from nodal values v[i + (p+1)*(j + (p+1)*k)]
// at the tensor GLL nodes (i,j,k index x,y,z).  Returns monomial coeffs.
__host__ __device__ inline PolyND fitPoly3(i32 p, const real *v) {
  real t[PNC]; gllNodes(p, t);
  real Vi[PNC][PNC]; vandermondeInv(p, t, Vi);
  i32 n = p+1;
  PolyND poly; poly.zero(3);
  poly.deg[0] = p; poly.deg[1] = p; poly.deg[2] = p;
  // c[a][b][cc] = sum_{i,j,k} Vi[a][i] Vi[b][j] Vi[cc][k] v[i,j,k]
  for (i32 a = 0; a < n; a++)
  for (i32 b = 0; b < n; b++)
  for (i32 cc = 0; cc < n; cc++) {
    real s = 0;
    for (i32 i = 0; i < n; i++)
    for (i32 j = 0; j < n; j++)
    for (i32 k = 0; k < n; k++)
      s += Vi[a][i]*Vi[b][j]*Vi[cc][k]*v[i + n*(j + n*k)];
    poly.at(a,b,cc) = s;
  }
  return poly;
}

#endif
