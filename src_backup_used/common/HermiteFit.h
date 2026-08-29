#ifndef FEM_HERMITEFIT_H
#define FEM_HERMITEFIT_H

//
// HERMITE least-squares fit of the level set.
//
// fitPoly3 (PolyFit.h) interpolates phi ALONE at the (p+1)^3 GLL nodes, so the
// fitted zero-contour is only as good as a degree-p value fit: the surface
// displacement error is O(h^{p+1})/|grad phi|.  On the blade at res48 that put
// the p2-fitted upper face 0.01006 outward (6.6% of a cell), inflating the local
// thickness 5.4% -- and since bending stress ~ M c / I ~ 1/t^2, that biased the
// peak stress ~11% LOW.  The lower face was exact, so it is a one-sided
// reconstruction error, not noise.
//
// The SDF oracle already computes grad(phi) on its way to the pseudonormal sign
// (BladeSdf::phiGrad), so the normal is free.  Matching it directly pins both the
// LOCATION and the ORIENTATION of the zero contour, which is exactly what the
// value-only fit gets wrong.
//
// A full tensor-product Hermite would need mixed derivatives (phi_xy, phi_xyz)
// that the oracle does not provide, so this is an overdetermined LEAST-SQUARES
// fit instead: m^3 nodes x 4 data each (phi, phi_xi0, phi_xi1, phi_xi2) against
// (q+1)^3 monomial coefficients, with 4m^3 > (q+1)^3.
//
// The design matrix depends only on (q, m) in REFERENCE coordinates -- never on
// the cell -- so it is factored once and each cell costs one back-substitution.
//
// Degree is capped by PDEG (Poly.h): PDEG=4 lets p2 fit degree 4 (error
// O(h^3) -> O(h^5)) at no cost elsewhere.  Going beyond needs a PDEG bump, which
// grows every PolyND from PNC^3 reals and adds Saye roots -- do that separately.
//

#include "Poly.h"
#include <vector>
#include <cmath>

struct HermiteFit {
  i32 q = 0, m = 0, nc = 0, nr = 0;
  std::vector<double> At;    // (nc x nr) transpose of the design matrix
  std::vector<double> L;     // (nc x nc) Cholesky factor of A^T A
  std::vector<double> t;     // m Chebyshev-Lobatto nodes on [0,1]

  bool ok() const { return nc > 0; }

  // Chebyshev-Lobatto nodes: well conditioned and include the endpoints, so the
  // fit still sees the cell faces (where neighbouring cells must agree).
  static double node(i32 i, i32 mm) {
    if (mm == 1) return 0.5;
    return 0.5 * (1.0 - std::cos(M_PI * (double)i / (double)(mm - 1)));
  }

  // Smallest m with 4m^3 >= 1.3 (q+1)^3 -- 30% oversampling for conditioning.
  static i32 autoM(i32 qq) {
    double need = 1.3 * std::pow((double)(qq + 1), 3.0) / 4.0;
    i32 mm = 2; while ((double)mm * mm * mm < need) mm++;
    return mm;
  }

  // gw: weight on the gradient rows.  There are 3x as many gradient equations
  // as value equations, so gw=1 lets the normal direction dominate the fit and
  // degrade phi near the contour -- which is the only place phi accuracy matters.
  double gw = 1.0;

  void init(i32 qq, i32 mm, double gwt = 1.0) {
    q = qq; m = mm; gw = gwt;
    nc = (q + 1) * (q + 1) * (q + 1);
    nr = 4 * m * m * m;
    t.resize(m);
    for (i32 i = 0; i < m; i++) t[i] = node(i, m);

    // A: rows = [value at each node, then d/dxi0, d/dxi1, d/dxi2]
    std::vector<double> A((size_t)nr * nc, 0.0);
    auto pw = [](double x, i32 e) { double r = 1; for (i32 i = 0; i < e; i++) r *= x; return r; };
    i32 nn = m * m * m;
    for (i32 k = 0; k < m; k++)
    for (i32 j = 0; j < m; j++)
    for (i32 i = 0; i < m; i++) {
      i32 nod = i + m * (j + m * k);
      double x = t[i], y = t[j], z = t[k];
      for (i32 c = 0; c <= q; c++)
      for (i32 b = 0; b <= q; b++)
      for (i32 a = 0; a <= q; a++) {
        i32 col = a + (q + 1) * (b + (q + 1) * c);
        double xa = pw(x, a), yb = pw(y, b), zc = pw(z, c);
        A[(size_t)nod * nc + col]              = xa * yb * zc;
        A[(size_t)(nn + nod) * nc + col]       = gw * (a ? a * pw(x, a - 1) : 0.0) * yb * zc;
        A[(size_t)(2 * nn + nod) * nc + col]   = gw * xa * (b ? b * pw(y, b - 1) : 0.0) * zc;
        A[(size_t)(3 * nn + nod) * nc + col]   = gw * xa * yb * (c ? c * pw(z, c - 1) : 0.0);
      }
    }

    At.assign((size_t)nc * nr, 0.0);
    for (i32 r = 0; r < nr; r++)
      for (i32 c = 0; c < nc; c++) At[(size_t)c * nr + r] = A[(size_t)r * nc + c];

    // normal equations A^T A, Cholesky (tiny Tikhonov guard for safety)
    std::vector<double> N((size_t)nc * nc, 0.0);
    for (i32 a = 0; a < nc; a++)
      for (i32 b = 0; b <= a; b++) {
        double s = 0;
        for (i32 r = 0; r < nr; r++) s += At[(size_t)a * nr + r] * At[(size_t)b * nr + r];
        N[(size_t)a * nc + b] = N[(size_t)b * nc + a] = s;
      }
    double tr = 0; for (i32 a = 0; a < nc; a++) tr += N[(size_t)a * nc + a];
    for (i32 a = 0; a < nc; a++) N[(size_t)a * nc + a] += 1e-13 * tr / nc;

    L.assign((size_t)nc * nc, 0.0);
    for (i32 a = 0; a < nc; a++)
      for (i32 b = 0; b <= a; b++) {
        double s = N[(size_t)a * nc + b];
        for (i32 c = 0; c < b; c++) s -= L[(size_t)a * nc + c] * L[(size_t)b * nc + c];
        if (a == b) L[(size_t)a * nc + b] = std::sqrt(s > 0 ? s : 0.0);
        else        L[(size_t)a * nc + b] = s / L[(size_t)b * nc + b];
      }
  }

  // v[m^3] = phi, g[3*m^3] = d phi / d xi_d  (REFERENCE frame: multiply the
  // computational-frame gradient by h, since x = (cell + xi) * h).
  PolyND apply(const real *v, const real *g) const {
    i32 nn = m * m * m;
    std::vector<double> d(nr), y(nc), x(nc);
    for (i32 i = 0; i < nn; i++) {
      d[i] = v[i];
      for (i32 dd = 0; dd < 3; dd++) d[(size_t)(dd + 1) * nn + i] = gw * g[(size_t)3 * i + dd];
    }
    for (i32 a = 0; a < nc; a++) {
      double s = 0;
      for (i32 r = 0; r < nr; r++) s += At[(size_t)a * nr + r] * d[r];
      y[a] = s;
    }
    for (i32 a = 0; a < nc; a++) {                       // forward solve
      double s = y[a];
      for (i32 c = 0; c < a; c++) s -= L[(size_t)a * nc + c] * x[c];
      x[a] = s / L[(size_t)a * nc + a];
    }
    for (i32 a = nc - 1; a >= 0; a--) {                  // back solve
      double s = x[a];
      for (i32 c = a + 1; c < nc; c++) s -= L[(size_t)c * nc + a] * x[c];
      x[a] = s / L[(size_t)a * nc + a];
    }
    PolyND poly; poly.zero(3);
    poly.deg[0] = poly.deg[1] = poly.deg[2] = q;
    for (i32 c = 0; c <= q; c++)
      for (i32 b = 0; b <= q; b++)
        for (i32 a = 0; a <= q; a++)
          poly.at(a, b, c) = (real)x[a + (q + 1) * (b + (q + 1) * c)];
    return poly;
  }
};

#endif
