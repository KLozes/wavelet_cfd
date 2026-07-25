#ifndef FEM_POLY_H
#define FEM_POLY_H

//
// Tensor-product polynomials on a box, for the Saye high-order cut quadrature
// (SayeQuad.cuh).  The level set inside a cell is the Qp Lagrange interpolant of
// the nodal values; here it is carried in the MONOMIAL power basis per axis,
// degree <= PDEG, so that the operations Saye needs are all closed form:
//
//   * restrict to a face  x_a = c        -> a (d-1)-dim polynomial
//   * slice to a line at fixed coords     -> a 1-D polynomial (for root finding)
//   * partial derivative d/dx_a           -> for the monotone-direction test
//   * evaluate, gradient
//
// PDEG = 3 covers p = 2 (quadratic geometry) and p = 3 (cubic).  Restriction and
// slicing never raise the degree, so the whole recursion stays within PDEG.
//
// __host__ __device__ throughout: the same routine builds/tests on the host and
// runs per cut cell on the GPU.
//

#include "Util.cuh"

static constexpr i32 PDEG   = 3;            // max polynomial degree per axis
static constexpr i32 PNC    = PDEG + 1;     // coefficients per axis
static constexpr i32 PMAXRT = PDEG;         // max real roots of a 1-D poly

// ---------------------------------------------------------------------------
//  1-D polynomial  p(t) = sum_{i=0}^{PDEG} c[i] t^i
// ---------------------------------------------------------------------------
struct Poly1 {
  real c[PNC];
  i32  deg;

  __host__ __device__ real eval(real t) const {
    real s = 0;
    for (i32 i = deg; i >= 0; i--) s = s*t + c[i];
    return s;
  }
  __host__ __device__ Poly1 deriv() const {
    Poly1 d; d.deg = deg > 0 ? deg-1 : 0;
    for (i32 i = 0; i < PNC; i++) d.c[i] = 0;
    for (i32 i = 1; i <= deg; i++) d.c[i-1] = i*c[i];
    return d;
  }
  __host__ __device__ i32 realDeg(real tol = (real)1e-12) const {
    i32 d = 0;
    for (i32 i = 0; i <= deg; i++) if (fabs(c[i]) > tol) d = i;
    return d;
  }
};

// real roots of p in the OPEN interval (a,b), returned sorted; count returned.
// Closed form for deg <= 2; for deg 3 a robust hybrid (find one root by bracket+
// Newton, deflate to a quadratic).  Robust to leading-coefficient degeneracy.
__host__ __device__ inline i32 poly1Roots(const Poly1 &p, real a, real b,
                                          real out[PMAXRT]) {
  i32 d = p.realDeg();
  i32 n = 0;
  if (d == 0) return 0;

  if (d == 1) {
    real r = -p.c[0]/p.c[1];
    if (r > a && r < b) out[n++] = r;
    return n;
  }
  if (d == 2) {
    real A = p.c[2], B = p.c[1], C = p.c[0];
    real disc = B*B - 4*A*C;
    if (disc < 0) return 0;
    real sq = sqrt(disc);
    // numerically stable quadratic roots
    real q = -(real)0.5*(B + (B >= 0 ? sq : -sq));
    real r1 = q/A, r2 = (fabs(q) > 0 ? C/q : r1);
    real lo = fmin(r1, r2), hi = fmax(r1, r2);
    if (lo > a && lo < b) out[n++] = lo;
    if (hi > a && hi < b && hi != lo) out[n++] = hi;
    return n;
  }

  // deg 3: sample the sign on a fine bracket, Newton-polish each sign change.
  // (a cut cell's 1-D slice is well separated; a handful of brackets suffices.)
  const i32 NB = 16;
  real prev_t = a, prev_f = p.eval(a);
  for (i32 i = 1; i <= NB; i++) {
    real t = a + (b - a)*i/NB, f = p.eval(t);
    if (prev_f == 0 && prev_t > a) { if (n < PMAXRT) out[n++] = prev_t; }
    else if (prev_f*f < 0) {
      // bisection + Newton on [prev_t, t]
      real lo = prev_t, hi = t, flo = prev_f;
      real r = (real)0.5*(lo+hi);
      Poly1 dp = p.deriv();
      for (i32 it = 0; it < 40; it++) {
        real fr = p.eval(r), dr = dp.eval(r);
        if (fabs(dr) > (real)1e-14) {
          real rn = r - fr/dr;
          if (rn > lo && rn < hi) { r = rn; }
          else { if (flo*fr < 0) hi = r; else { lo = r; flo = fr; } r = (real)0.5*(lo+hi); }
        } else { if (flo*fr < 0) hi = r; else { lo = r; flo = fr; } r = (real)0.5*(lo+hi); }
        if (fabs(fr) < (real)1e-13) break;
      }
      if (r > a && r < b && n < PMAXRT) out[n++] = r;
    }
    prev_t = t; prev_f = f;
  }
  return n;
}

// ---------------------------------------------------------------------------
//  d-dim tensor monomial polynomial on the reference box [0,1]^d
//    p(x) = sum_{i,j,k} c[i][j][k] x^i y^j z^k     (unused axes have deg 0)
//  stored flat: coeff(i,j,k) = c[i + PNC*(j + PNC*k)]
// ---------------------------------------------------------------------------
struct PolyND {
  real c[PNC*PNC*PNC];
  i32  dim;            // 1, 2, or 3 active axes (the leading ones)
  i32  deg[3];         // degree per active axis

  __host__ __device__ void zero(i32 d) {
    dim = d;
    for (i32 i = 0; i < PNC*PNC*PNC; i++) c[i] = 0;
    for (i32 a = 0; a < 3; a++) deg[a] = 0;
  }
  __host__ __device__ real &at(i32 i, i32 j, i32 k) { return c[i + PNC*(j + PNC*k)]; }
  __host__ __device__ real  at(i32 i, i32 j, i32 k) const { return c[i + PNC*(j + PNC*k)]; }

  __host__ __device__ real eval(const real x[3]) const {
    // Horner per axis; unused axes contribute only the i=0 slab
    real px[PNC] = {1,0,0,0}, py[PNC] = {1,0,0,0}, pz[PNC] = {1,0,0,0};
    for (i32 i = 1; i < PNC; i++) px[i] = px[i-1]*(dim>0 ? x[0] : 0);
    for (i32 j = 1; j < PNC; j++) py[j] = py[j-1]*(dim>1 ? x[1] : 0);
    for (i32 k = 1; k < PNC; k++) pz[k] = pz[k-1]*(dim>2 ? x[2] : 0);
    real s = 0;
    for (i32 k = 0; k <= (dim>2?deg[2]:0); k++)
    for (i32 j = 0; j <= (dim>1?deg[1]:0); j++)
    for (i32 i = 0; i <= deg[0]; i++)
      s += at(i,j,k)*px[i]*py[j]*pz[k];
    return s;
  }

  // slice to a 1-D polynomial along active axis `ax`, with the OTHER active axes
  // fixed at xf[] (xf indexed by axis; the ax entry ignored)
  __host__ __device__ Poly1 line(i32 ax, const real xf[3]) const {
    Poly1 p; p.deg = deg[ax];
    for (i32 i = 0; i < PNC; i++) p.c[i] = 0;
    real po[3][PNC];
    for (i32 a = 0; a < 3; a++) {
      po[a][0] = 1;
      for (i32 i = 1; i < PNC; i++) po[a][i] = po[a][i-1]*((a<dim) ? xf[a] : 0);
    }
    for (i32 k = 0; k <= (dim>2?deg[2]:0); k++)
    for (i32 j = 0; j <= (dim>1?deg[1]:0); j++)
    for (i32 i = 0; i <= deg[0]; i++) {
      // the coefficient of x_ax^e where e is the power on axis ax
      i32 e = (ax==0)?i:(ax==1)?j:k;
      real term = at(i,j,k);
      if (ax!=0) term *= po[0][i];
      if (ax!=1) term *= po[1][j];
      if (ax!=2) term *= po[2][k];
      p.c[e] += term;
    }
    return p;
  }

  // restrict to the face  x_ax = val  -> a (dim-1) polynomial whose active axes
  // are the remaining ones, re-indexed to the leading slots
  __host__ __device__ PolyND restrict_(i32 ax, real val) const {
    PolyND r; r.zero(dim-1);
    real pw[PNC]; pw[0] = 1;
    for (i32 i = 1; i < PNC; i++) pw[i] = pw[i-1]*val;
    // map remaining axes (a != ax) to new leading axes 0..dim-2 in order
    i32 map[3], m = 0;
    for (i32 a = 0; a < dim; a++) if (a != ax) map[m++] = a;
    for (i32 a = 0; a < r.dim; a++) r.deg[a] = deg[map[a]];
    for (i32 k = 0; k <= (dim>2?deg[2]:0); k++)
    for (i32 j = 0; j <= (dim>1?deg[1]:0); j++)
    for (i32 i = 0; i <= deg[0]; i++) {
      i32 idx[3] = {i,j,k};
      real w = pw[idx[ax]];
      i32 ni = (r.dim>0) ? idx[map[0]] : 0;
      i32 nj = (r.dim>1) ? idx[map[1]] : 0;
      r.at(ni, nj, 0) += at(i,j,k)*w;
    }
    return r;
  }

  // substitute axis `ax` = val, KEEPING 3-D indexing (that axis collapses to
  // degree 0, its contribution folded into power 0).  `dim` is unchanged; the
  // caller tracks which axes are still "active" via a mask.  This is the face
  // restriction the Saye recursion needs, without re-indexing axes.
  __host__ __device__ PolyND subst(i32 ax, real val) const {
    PolyND r; r.zero(dim);
    for (i32 a = 0; a < 3; a++) r.deg[a] = deg[a];
    r.deg[ax] = 0;
    real pw[PNC]; pw[0] = 1;
    for (i32 i = 1; i < PNC; i++) pw[i] = pw[i-1]*val;
    for (i32 k = 0; k <= deg[2]; k++)
    for (i32 j = 0; j <= deg[1]; j++)
    for (i32 i = 0; i <= deg[0]; i++) {
      i32 idx[3] = {i,j,k};
      i32 ni=i, nj=j, nk=k;
      if (ax==0) ni=0; else if (ax==1) nj=0; else nk=0;
      r.at(ni,nj,nk) += at(i,j,k)*pw[idx[ax]];
    }
    return r;
  }

  // partial derivative d/dx_ax (same dim, degree drops by one on ax)
  __host__ __device__ PolyND partial(i32 ax) const {
    PolyND d; d.zero(dim);
    for (i32 a = 0; a < 3; a++) d.deg[a] = deg[a];
    d.deg[ax] = deg[ax] > 0 ? deg[ax]-1 : 0;
    for (i32 k = 0; k <= (dim>2?deg[2]:0); k++)
    for (i32 j = 0; j <= (dim>1?deg[1]:0); j++)
    for (i32 i = 0; i <= deg[0]; i++) {
      i32 idx[3] = {i,j,k};
      i32 e = idx[ax];
      if (e == 0) continue;
      i32 di=i, dj=j, dk=k;
      if (ax==0) di--; else if (ax==1) dj--; else dk--;
      d.at(di,dj,dk) += e*at(i,j,k);
    }
    return d;
  }

  // gradient at x (all three axes; collapsed axes give 0)
  __host__ __device__ void grad(const real x[3], real g[3]) const {
    for (i32 a = 0; a < 3; a++) g[a] = partial(a).eval(x);
  }
};

#endif
