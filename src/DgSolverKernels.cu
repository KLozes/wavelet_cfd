#include <cstdio>
#include <cmath>
#include "DgSolverKernels.cuh"

//
// All kernels for the wavedg3d multi-resolution DGSEM solver.
//
// node indexing within an element (matches GET_CELL_INDICES):
//   nd = i + j*NNODE + k*NNODE^2,  i = x-node (fastest), k = z-node (slowest)
//
// face-node flattening used by the trace/mortar helpers:
//   fn = c1 + NNODE*c2 with (t1,t2) the two tangential axes of the face:
//   dir 0 (x-face): t1=y, t2=z;  dir 1: t1=x, t2=z;  dir 2: t1=x, t2=y
//

/* ════════════════════════════════════════════════════════════════════════
 * Reference-element operators (constant memory) + host setup / self-test
 * ════════════════════════════════════════════════════════════════════════ */

__constant__ real c_D   [NNODE][NNODE];  // differentiation matrix D_ik = l_k'(xi_i)
__constant__ real c_w   [NNODE];         // GLL weights (sum = 2)
__constant__ real c_winv[NNODE];         // 1/w_i (surface lift)
__constant__ real c_xi  [NNODE];         // LGL node coordinates on [-1,1]
__constant__ real c_I [2][NNODE][NNODE]; // half-interval interpolation I[s][a][b] = l_b(xi_s(a)),
                                         //   xi_0(a) = (xi_a-1)/2, xi_1(a) = (xi_a+1)/2
__constant__ real c_R [2][NNODE][NNODE]; // EXACT-L2 restriction (tensor 1D): coarse = sum_s R[s] . child_s.
                                         // Exact-L2 (not the GLL-discrete adjoint) so that R(I(u)) == u
                                         // exactly for nodal polynomials -- the MRA detail of locally
                                         // smooth data then vanishes to roundoff (cancellation property),
                                         // and merges still conserve the GLL cell mean exactly because
                                         // GLL-4 integrates the degree-3 nodal polynomials exactly.
__constant__ real c_IbFilt[NNODE][NNODE]; // IB donor top-mode projection (see hIbFilt)
__constant__ real c_Vinv[NNODE][NNODE]; // nodal -> orthonormal-Legendre modal transform (inverse
                                        // Vandermonde) for the Persson-Peraire smoothness sensor

// LGL nodes / GLL weights for p = 1..3 (from dgsem_lobatto_2d.cu)
static const double lgl_xi_tab[3][4] = {
    /* p=1 */ {-1.0, 1.0},
    /* p=2 */ {-1.0, 0.0, 1.0},
    /* p=3 */ {-1.0,-0.4472135954999579, 0.4472135954999579, 1.0},
};
static const double gll_w_tab[3][4] = {
    /* p=1 */ {1.0, 1.0},
    /* p=2 */ {0.33333333333333333, 1.33333333333333333, 0.33333333333333333},
    /* p=3 */ {0.16666666666666667, 0.83333333333333333,
               0.83333333333333333, 0.16666666666666667},
};

// 8-point Gauss-Legendre on [-1,1] (exact to degree 15) for the L2 projections
static const double gs8_x[8] = {
  -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
   0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363};
static const double gs8_w[8] = {
   0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
   0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763};

// barycentric differentiation matrix D_ik = l_k'(x_i) (dgsem build_D, verbatim)
static void dgBuildD(int n, const double *x, double D[NNODE][NNODE]) {
  double b[NNODE];
  for (int kk = 0; kk < n; kk++) {
    b[kk] = 1.0;
    for (int m = 0; m < n; m++)
      if (m != kk) b[kk] /= (x[kk] - x[m]);
  }
  for (int i = 0; i < n; i++) {
    double diag = 0.0;
    for (int kk = 0; kk < n; kk++) {
      if (kk != i) {
        D[i][kk] = (b[kk] / b[i]) / (x[i] - x[kk]);
        diag    -= D[i][kk];
      }
    }
    D[i][i] = diag;
  }
}

// Lagrange basis l_b evaluated at x (nodes xi[0..n-1])
static double dgLagrange(int n, const double *xi, int b, double x) {
  double v = 1.0;
  for (int m = 0; m < n; m++)
    if (m != b) v *= (x - xi[m]) / (xi[b] - xi[m]);
  return v;
}

// 4x4 linear solve (Gaussian elimination with partial pivoting), for M_c^-1
static void dgSolve(int n, double A[NNODE][NNODE], double B[NNODE][NNODE]) {
  // solves A X = B in place: B <- A^-1 B
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c+1; r < n; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (p != c) {
      for (int m = 0; m < n; m++) { double t=A[c][m]; A[c][m]=A[p][m]; A[p][m]=t; }
      for (int m = 0; m < n; m++) { double t=B[c][m]; B[c][m]=B[p][m]; B[p][m]=t; }
    }
    double d = A[c][c];
    for (int m = 0; m < n; m++) { A[c][m] /= d; B[c][m] /= d; }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      double f = A[r][c];
      if (f == 0.0) continue;
      for (int m = 0; m < n; m++) { A[r][m] -= f*A[c][m]; B[r][m] -= f*B[c][m]; }
    }
  }
}

static double hD[NNODE][NNODE], hI[2][NNODE][NNODE], hR[2][NNODE][NNODE];
static double hVinv[NNODE][NNODE];
static double hIbFilt[NNODE][NNODE];   // V diag(1,..,1,0) V^-1: zero the top mode
static double hXi[NNODE], hW[NNODE];

// orthonormal Legendre P^_m(x) = P_m(x) * sqrt((2m+1)/2), m = 0..3
static double dgLegendreON(int m, double x) {
  double P = (m == 0) ? 1.0
           : (m == 1) ? x
           : (m == 2) ? 0.5*(3.0*x*x - 1.0)
                      : 0.5*(5.0*x*x*x - 3.0*x);
  return P * sqrt((2.0*m + 1.0)/2.0);
}

static void dgBuildOperators(void) {
  const double *xi = lgl_xi_tab[dgOrder-1];
  const double *w  = gll_w_tab[dgOrder-1];
  for (int i = 0; i < NNODE; i++) { hXi[i] = xi[i]; hW[i] = w[i]; }

  dgBuildD(NNODE, xi, hD);

  // half-interval interpolation: I[s][a][b] = l_b(xi mapped into half s)
  for (int s = 0; s < 2; s++)
    for (int a = 0; a < NNODE; a++) {
      double x = (s == 0) ? 0.5*(xi[a] - 1.0) : 0.5*(xi[a] + 1.0);
      for (int b = 0; b < NNODE; b++) hI[s][a][b] = dgLagrange(NNODE, xi, b, x);
    }

  // exact-L2 restriction: R[s] = M_c^-1 B[s],
  //   M_c[i][j] = int_-1^1 l_i l_j dx,
  //   B[s][i][a] = (1/2) int_-1^1 l_i((y + 2s - 1)/2) l_a(y) dy   (child s frame)
  for (int s = 0; s < 2; s++) {
    double M[NNODE][NNODE], B[NNODE][NNODE];
    for (int i = 0; i < NNODE; i++)
      for (int j = 0; j < NNODE; j++) { M[i][j] = 0.0; B[i][j] = 0.0; }
    for (int q = 0; q < 8; q++) {
      double y = gs8_x[q], gw = gs8_w[q];
      double xp = 0.5*(y + 2.0*s - 1.0);     // parent-frame coordinate of the child point
      double lc[NNODE], lf[NNODE], lcp[NNODE];
      for (int m = 0; m < NNODE; m++) {
        lc[m]  = dgLagrange(NNODE, xi, m, y);    // coarse basis at parent point y (for M)
        lcp[m] = dgLagrange(NNODE, xi, m, xp);   // coarse basis at child-mapped point
        lf[m]  = dgLagrange(NNODE, xi, m, y);    // child basis in its own frame
      }
      for (int i = 0; i < NNODE; i++) {
        for (int j = 0; j < NNODE; j++) {
          if (s == 0) M[i][j] += gw * lc[i] * lc[j];   // build M once (s==0 pass)
          B[i][j] += 0.5 * gw * lcp[i] * lf[j];
        }
      }
    }
    if (s == 1) {   // M was only accumulated on the s==0 pass; rebuild for the solve
      for (int q = 0; q < 8; q++) {
        double y = gs8_x[q], gw = gs8_w[q];
        double lc[NNODE];
        for (int m = 0; m < NNODE; m++) lc[m] = dgLagrange(NNODE, xi, m, y);
        for (int i = 0; i < NNODE; i++)
          for (int j = 0; j < NNODE; j++) M[i][j] += gw * lc[i] * lc[j];
      }
    }
    dgSolve(NNODE, M, B);
    for (int i = 0; i < NNODE; i++)
      for (int a = 0; a < NNODE; a++) hR[s][i][a] = B[i][a];
  }

  // modal transform for the Persson-Peraire sensor: hVinv = V^-1 with
  // V[i][m] = P^_m(xi_i) (orthonormal Legendre at the LGL nodes)
  {
    double V[NNODE][NNODE], Id[NNODE][NNODE];
    for (int i = 0; i < NNODE; i++)
      for (int m = 0; m < NNODE; m++) {
        V[i][m] = dgLegendreON(m, xi[i]);
        Id[i][m] = (i == m) ? 1.0 : 0.0;
      }
    dgSolve(NNODE, V, Id);          // Id <- V^-1
    for (int m = 0; m < NNODE; m++)
      for (int i = 0; i < NNODE; i++) hVinv[m][i] = Id[m][i];
    // IB donor filter: project out the top Legendre mode (V diag V^-1).
    // The image-point evaluation feeding the ghost fill reads the donor
    // through this filter -- the top mode is the only structure that fits
    // between a wall face and its image point, and it closes the
    // ghost<->fluid feedback loop with reflection gain > 1.
    // (dgSolve destroyed V in place -- rebuild it.)
    for (int i = 0; i < NNODE; i++)
      for (int m = 0; m < NNODE; m++) V[i][m] = dgLegendreON(m, xi[i]);
    for (int i = 0; i < NNODE; i++)
      for (int j = 0; j < NNODE; j++) {
        double v = 0;
        for (int m = 0; m < NNODE-1; m++) v += V[i][m]*Id[m][j];
        hIbFilt[i][j] = v;
      }
  }
}

void dgUploadOperators(void) {
  dgBuildOperators();
  real D[NNODE][NNODE], I2[2][NNODE][NNODE], R2[2][NNODE][NNODE];
  real xi[NNODE], w[NNODE], winv[NNODE];
  for (int i = 0; i < NNODE; i++) {
    xi[i] = (real)hXi[i]; w[i] = (real)hW[i]; winv[i] = (real)(1.0/hW[i]);
    for (int j = 0; j < NNODE; j++) D[i][j] = (real)hD[i][j];
  }
  for (int s = 0; s < 2; s++)
    for (int i = 0; i < NNODE; i++)
      for (int j = 0; j < NNODE; j++) {
        I2[s][i][j] = (real)hI[s][i][j];
        R2[s][i][j] = (real)hR[s][i][j];
      }
  cudaMemcpyToSymbol(c_D,    D,    sizeof(D));
  cudaMemcpyToSymbol(c_w,    w,    sizeof(w));
  cudaMemcpyToSymbol(c_winv, winv, sizeof(winv));
  cudaMemcpyToSymbol(c_xi,   xi,   sizeof(xi));
  cudaMemcpyToSymbol(c_I,    I2,   sizeof(I2));
  cudaMemcpyToSymbol(c_R,    R2,   sizeof(R2));
  real Vi[NNODE][NNODE];
  for (int m = 0; m < NNODE; m++)
    for (int i = 0; i < NNODE; i++) Vi[m][i] = (real)hVinv[m][i];
  cudaMemcpyToSymbol(c_Vinv, Vi, sizeof(Vi));
  real Fl[NNODE][NNODE];
  for (int m = 0; m < NNODE; m++)
    for (int i = 0; i < NNODE; i++) Fl[m][i] = (real)hIbFilt[m][i];
  cudaMemcpyToSymbol(c_IbFilt, Fl, sizeof(Fl));
}

// host access to the reference-element weights/nodes (diagnostic integrals)
void dgGetHostOps(double *w, double *xi) {
  dgBuildOperators();
  for (int i = 0; i < NNODE; i++) { w[i] = hW[i]; xi[i] = hXi[i]; }
}

// host double-precision mirror of dgIbHermite for the selftest
static double dgIbHermiteHost(int dirichlet, double bc,
    double F, double hF1, double hF2, double sigma, int order) {
  if (order <= 1)
    return dirichlet ? (bc + (F - bc)*sigma) : (F + bc*(sigma - 1.0));
  double b0, b1, b2, b3 = 0.0;
  if (dirichlet) {
    double A = F - bc;
    b0 = bc;
    if (order >= 3) { b3 = 0.5*hF2 - hF1 + A; b2 = hF1 - A - 2.0*b3; }
    else            { b2 = hF1 - A; }
    b1 = A - b2 - b3;
  } else {
    b1 = bc;
    if (order >= 3) { b3 = (bc + hF2 - hF1)/3.0; b2 = 0.5*(hF2 - 6.0*b3); }
    else            { b2 = 0.5*(hF1 - bc); }
    b0 = F - bc - b2 - b3;
  }
  return b0 + sigma*(b1 + sigma*(b2 + sigma*b3));
}

bool dgOperatorSelfTest(void) {
  dgBuildOperators();
  bool ok = true;
  auto expect = [&](const char *name, double v, double tol) {
    if (fabs(v) > tol) { printf("[selftest] FAIL %s: |%.3e| > %.1e\n", name, v, tol); ok = false; }
  };
  // D rows sum to zero (derivative of constants)
  for (int i = 0; i < NNODE; i++) {
    double s = 0; for (int j = 0; j < NNODE; j++) s += hD[i][j];
    expect("D row sum", s, 1e-12);
  }
  // I reproduces constants and degree-p monomials
  for (int s = 0; s < 2; s++)
    for (int a = 0; a < NNODE; a++)
      for (int m = 0; m <= dgOrder; m++) {
        double x = (s==0) ? 0.5*(hXi[a]-1.0) : 0.5*(hXi[a]+1.0);
        double v = 0;
        for (int b = 0; b < NNODE; b++) v += hI[s][a][b] * pow(hXi[b], m);
        expect("I monomial", v - pow(x, m), 1e-12);
      }
  // R o P = identity: restriction of the injected coarse monomial recovers it
  for (int m = 0; m <= dgOrder; m++) {
    for (int i = 0; i < NNODE; i++) {
      double v = 0;
      for (int s = 0; s < 2; s++)
        for (int a = 0; a < NNODE; a++) {
          double xp = (s==0) ? 0.5*(hXi[a]-1.0) : 0.5*(hXi[a]+1.0);
          v += hR[s][i][a] * pow(xp, m);
        }
      expect("RoP monomial", v - pow(hXi[i], m), 1e-12);
    }
  }
  // conservation adjoint: sum_i w_i R[s][i][a] = w_a/2 (merge preserves the GLL mean)
  for (int s = 0; s < 2; s++)
    for (int a = 0; a < NNODE; a++) {
      double v = 0;
      for (int i = 0; i < NNODE; i++) v += hW[i]*hR[s][i][a];
      expect("R mean adjoint", v - 0.5*hW[a], 1e-12);
    }
  // mortar lift of a constant is exact: sum_{s,a} (w_a/2) I[s][a][j] = w_j
  for (int j = 0; j < NNODE; j++) {
    double v = 0;
    for (int s = 0; s < 2; s++)
      for (int a = 0; a < NNODE; a++) v += 0.5*hW[a]*hI[s][a][j];
    expect("mortar constant", v - hW[j], 1e-12);
  }
  // IB wall-normal Hermite closed forms: reconstruct manufactured polynomials
  // exactly (order 3 <-> cubics, order 2 <-> quadratics, order 1 <-> linears),
  // both BC types, across the whole evaluation range sigma in [-1, 1]
  for (int ord = 1; ord <= 3; ord++) {
    double c[4] = {0.37, -1.21, 0.83, 0.59};        // f = c0 + c1 s + c2 s^2 + c3 s^3
    for (int m = ord + 1; m < 4; m++) c[m] = 0.0;   // degree matches the order
    double F  = c[0] + c[1] + c[2] + c[3];          // f(1)
    double F1 = c[1] + 2*c[2] + 3*c[3];             // f'(1)
    double F2 = 2*c[2] + 6*c[3];                    // f''(1)
    for (int bc = 0; bc < 2; bc++) {                // 0 Neumann, 1 Dirichlet
      double dat = bc ? c[0] : c[1];                // f(0) or f'(0)
      for (double sg = -1.0; sg <= 1.001; sg += 0.25) {
        double v = dgIbHermiteHost(bc, dat, F, F1, F2, sg, ord);
        double e = c[0] + sg*(c[1] + sg*(c[2] + sg*c[3]));
        expect("IB hermite", v - e, 1e-12);
      }
    }
  }
  // physical-scaling sweep: reconstruct f(s) = 2 + 3 s^2 (Neumann, g0 = 0)
  // through varying image distances d -- the normalized forms must be
  // d-invariant (this is the fp32-conditioning property)
  for (double d = 0.05; d < 3.01; d *= 2.0) {
    double F  = 2.0 + 3.0*d*d;                      // f(d)
    double hF1 = (6.0*d)*d;                         // f'(d) * d
    double hF2 = 6.0*d*d;                           // f''(d) * d^2
    for (double sg = -1.0; sg <= 1.001; sg += 0.5) {
      double v = dgIbHermiteHost(0, 0.0, F, hF1, hF2, sg, 3);
      double e = 2.0 + 3.0*(sg*d)*(sg*d);
      expect("IB hermite scale", v - e, 1e-11);
    }
  }

  if (ok) printf("[selftest] all operator identities pass (p=%d)\n", dgOrder);
  return ok;
}

/* ════════════════════════════════════════════════════════════════════════
 * Device helpers: state conversions and fluxes (dgsem ports, 3D / 5 vars)
 * ════════════════════════════════════════════════════════════════════════ */

#define DG_EPSF ((real)1e-13)
#define DG_UMAX ((real)1e5)

__device__ __forceinline__ real dgSoundSpeed(real p, real rho) {
  real cs2 = dgGam * p / rho;
  return sqrt(fmax(cs2, (real)1e-14));
}

__device__ __forceinline__ void dgP2C(const real W[5], real U[5]) {
  U[0] = W[0];
  U[1] = W[0]*W[1];
  U[2] = W[0]*W[2];
  U[3] = W[0]*W[3];
  U[4] = W[4]/(dgGam - (real)1.0) + (real)0.5*W[0]*(W[1]*W[1] + W[2]*W[2] + W[3]*W[3]);
}

__device__ __forceinline__ real dgPressureFromCons(const real U[5]) {
  real rho = fmax(U[0], DG_EPSF);
  real ke  = (real)0.5*(U[1]*U[1] + U[2]*U[2] + U[3]*U[3])/rho;
  return (dgGam - (real)1.0)*(U[4] - ke);
}

__device__ __forceinline__ void dgSanitizeCons(real U[5]) {
  U[0] = fmax(U[0], DG_EPSF);
  real p = dgPressureFromCons(U);
  if (p < DG_EPSF)
    U[4] = DG_EPSF/(dgGam-(real)1.0) + (real)0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/U[0];
}

__device__ __forceinline__ void dgSanitizePrim(real W[5]) {
  W[0] = fmax(W[0], DG_EPSF);
  W[4] = fmax(W[4], DG_EPSF);
  if (W[0] < (real)1e-5) { W[1] = 0; W[2] = 0; W[3] = 0; return; }
  W[1] = fmax(fmin(W[1], DG_UMAX), -DG_UMAX);
  W[2] = fmax(fmin(W[2], DG_UMAX), -DG_UMAX);
  W[3] = fmax(fmin(W[3], DG_UMAX), -DG_UMAX);
}

__device__ __forceinline__ void dgConsToPrimSane(const real U[5], real W[5]) {
  real Us[5] = {U[0], U[1], U[2], U[3], U[4]};
  dgSanitizeCons(Us);
  W[0] = Us[0];
  W[1] = Us[1]/Us[0];
  W[2] = Us[2]/Us[0];
  W[3] = Us[3]/Us[0];
  W[4] = (dgGam-(real)1.0)*(Us[4] - (real)0.5*(Us[1]*Us[1]+Us[2]*Us[2]+Us[3]*Us[3])/Us[0]);
  dgSanitizePrim(W);
}

// physical Euler flux along axis dir from primitives
__device__ __forceinline__ void dgEulerFluxAxis(const real W[5], i32 dir, real F[5]) {
  real un = W[1+dir];
  real E  = W[4]/(dgGam-(real)1.0) + (real)0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
  F[0] = W[0]*un;
  F[1] = W[0]*un*W[1];
  F[2] = W[0]*un*W[2];
  F[3] = W[0]*un*W[3];
  F[1+dir] += W[4];
  F[4] = (E + W[4])*un;
}

__device__ __forceinline__ real dgLogMean(real aL, real aR) {
  real d  = aL/aR;
  real f  = (d-(real)1.0)/(d+(real)1.0);
  real u2 = f*f;
  real FF = (u2 < (real)1e-4)
          ? ((real)1.0 + u2*((real)(1.0/3.0) + u2*((real)(1.0/5.0) + u2*(real)(1.0/7.0))))
          : (log(d)/((real)2.0*f));
  return (aL+aR)/((real)2.0*FF);
}

// Chandrashekar entropy-conservative two-point flux along axis dir (3D / 5 vars)
__device__ __forceinline__ void dgEcFluxAxis(const real WL[5], const real WR[5],
                                             i32 dir, real F[5]) {
  real bL = (real)0.5*WL[0]/WL[4],  bR = (real)0.5*WR[0]/WR[4];
  real r_ln = dgLogMean(WL[0], WR[0]);
  real b_ln = dgLogMean(bL, bR);
  real r_av = (real)0.5*(WL[0]+WR[0]);
  real b_av = (real)0.5*(bL+bR);
  real u_av = (real)0.5*(WL[1]+WR[1]);
  real v_av = (real)0.5*(WL[2]+WR[2]);
  real w_av = (real)0.5*(WL[3]+WR[3]);
  real p_hat = (real)0.5*r_av/b_av;
  real vel2  = (real)0.5*(WL[1]*WL[1]+WL[2]*WL[2]+WL[3]*WL[3])
             + (real)0.5*(WR[1]*WR[1]+WR[2]*WR[2]+WR[3]*WR[3]);
  real e_int = (real)0.5*((real)1.0/((dgGam-(real)1.0)*b_ln) - vel2);
  real un_av = (dir == 0) ? u_av : ((dir == 1) ? v_av : w_av);
  real f1 = r_ln*un_av;
  F[0] = f1;
  F[1] = f1*u_av;
  F[2] = f1*v_av;
  F[3] = f1*w_av;
  F[1+dir] += p_hat;
  F[4] = f1*e_int + F[1]*u_av + F[2]*v_av + F[3]*w_av;
}

// HLLC flux along axis dir from primitives (Toro star states, dgsem port)
__device__ void dgHllcAxis(const real WL[5], const real WR[5], i32 dir, real F[5]) {
  const i32 n  = 1 + dir;                    // normal velocity slot
  const i32 t1 = 1 + ((dir+1) % 3);          // tangential slots
  const i32 t2 = 1 + ((dir+2) % 3);

  real rL = WL[0], pL = WL[4], rR = WR[0], pR = WR[4];
  real unL = WL[n],  unR = WR[n];
  real cL = dgSoundSpeed(pL, rL), cR = dgSoundSpeed(pR, rR);
  real EL = pL/(dgGam-(real)1.0) + (real)0.5*rL*(WL[1]*WL[1]+WL[2]*WL[2]+WL[3]*WL[3]);
  real ER = pR/(dgGam-(real)1.0) + (real)0.5*rR*(WR[1]*WR[1]+WR[2]*WR[2]+WR[3]*WR[3]);

  real SL = fmin(unL - cL, unR - cR);
  real SR = fmax(unL + cL, unR + cR);

  if (SL >= (real)0.0) {
    F[0] = rL*unL;
    F[1] = rL*unL*WL[1]; F[2] = rL*unL*WL[2]; F[3] = rL*unL*WL[3];
    F[n] += pL;
    F[4] = (EL + pL)*unL;
    return;
  }
  if (SR <= (real)0.0) {
    F[0] = rR*unR;
    F[1] = rR*unR*WR[1]; F[2] = rR*unR*WR[2]; F[3] = rR*unR*WR[3];
    F[n] += pR;
    F[4] = (ER + pR)*unR;
    return;
  }

  real dL = rL*(SL - unL), dR = rR*(SR - unR);
  real Ss = (pR - pL + dL*unL - dR*unR)/(dL - dR);

  bool left = (Ss >= (real)0.0);
  const real *WK = left ? WL : WR;
  real rK = WK[0], pK = WK[4], unK = WK[n];
  real EK = left ? EL : ER;
  real SK = left ? SL : SR;
  real fact = rK*(SK - unK)/(SK - Ss);

  // star conservative state (tangential velocities carried through)
  real Us0 = fact;
  real UsN = fact*Ss;
  real UsT1 = fact*WK[t1];
  real UsT2 = fact*WK[t2];
  real UsE = fact*(EK/rK + (Ss - unK)*(Ss + pK/(rK*(SK - unK))));

  // K-side flux and conservative state
  real FK0 = rK*unK;
  real FKN = rK*unK*unK + pK;
  real FKT1 = rK*unK*WK[t1];
  real FKT2 = rK*unK*WK[t2];
  real FKE = (EK + pK)*unK;

  real UK0 = rK, UKN = rK*unK, UKT1 = rK*WK[t1], UKT2 = rK*WK[t2], UKE = EK;

  F[0] = FK0 + SK*(Us0 - UK0);
  F[n] = FKN + SK*(UsN - UKN);
  F[t1] = FKT1 + SK*(UsT1 - UKT1);
  F[t2] = FKT2 + SK*(UsT2 - UKT2);
  F[4] = FKE + SK*(UsE - UKE);
}

/* ════════════════════════════════════════════════════════════════════════
 * Geometry helpers
 * ════════════════════════════════════════════════════════════════════════ */

// physical element widths per direction at a level (element = one block)
__device__ __forceinline__ void dgElemSize(DgSolver &grid, i32 lvl, real h[3]) {
  h[0] = grid.getDx(lvl)*blockSize;
  h[1] = grid.getDy(lvl)*blockSize;
  h[2] = grid.getDz(lvl)*blockSize;   // pseudo2D: the (never refined) full z extent
}

// physical position of LGL node i along axis "dir" of element (lvl, eb)
__device__ __forceinline__ real dgNodePos(real hDir, i32 eb, i32 i) {
  return (eb + (c_xi[i] + (real)1.0)*(real)0.5) * hDir;
}

// element node index from face coordinates: face-normal axis dir with normal
// index nrm, tangential indices (a=t1, b=t2)
__device__ __forceinline__ i32 dgFaceNode(i32 dir, i32 nrm, i32 a, i32 b) {
  if (dir == 0) return nrm + a*NNODE + b*NNODE*NNODE;
  if (dir == 1) return a + nrm*NNODE + b*NNODE*NNODE;
  return a + b*NNODE + nrm*NNODE*NNODE;
}

/* ════════════════════════════════════════════════════════════════════════
 * Initial conditions (analytic, primitive -> conservative at LGL nodes)
 * ════════════════════════════════════════════════════════════════════════ */

__device__ void dgVortexExact(real x, real y, real u0, real v0, real cx, real cy,
                              real W[5]) {
  const real eps = 5.0;
  real dx = x - cx, dy = y - cy;
  real r2 = dx*dx + dy*dy;
  real f  = eps/((real)2.0*PI) * exp((real)0.5*((real)1.0 - r2));
  W[1] = u0 - f*dy;
  W[2] = v0 + f*dx;
  W[3] = 0.0;
  real dT = -(dgGam-(real)1.0)*eps*eps/((real)8.0*dgGam*PI*PI) * exp((real)1.0 - r2);
  real T  = fmax((real)1.0 + dT, (real)1e-6);
  W[0] = pow(T, (real)1.0/(dgGam-(real)1.0));
  W[4] = pow(T, dgGam/(dgGam-(real)1.0));
}

// double Mach reflection: Ms=10 shock at 60 deg, foot at (x0, 0)
__device__ __forceinline__ void dgDmrPost(real W[5]) {
  W[0] = 8.0;  W[1] = 7.144709581221619;  W[2] = -4.125;  W[3] = 0.0;  W[4] = 116.5;
}
__device__ __forceinline__ void dgDmrPre(real W[5]) {
  W[0] = 1.4;  W[1] = 0.0;  W[2] = 0.0;  W[3] = 0.0;  W[4] = 1.0;
}
__device__ __forceinline__ bool dgDmrPostSide(real x, real y, real t, real x0) {
  // shock line x_s(y,t) = x0 + (y + 20 t)/sqrt(3)
  return x < x0 + (y + (real)20.0*t)*(real)0.5773502691896258;
}

__device__ void dgEvalIC(DgSolver &grid, real x, real y, real z, i32 lvl, real W[5]) {
  real cx = grid.domainSize[0]*(real)0.5;
  real cy = grid.domainSize[1]*(real)0.5;
  real cz = grid.domainSize[2]*(real)0.5;
  // Interface smoothing width, FIXED at the finest element size (not the eval
  // level's).  Fixed => the interface stays the same physical width at every
  // level, so the significant-detail band tightens to a few finest elements
  // instead of staying ~1-element-wide at every level (a wide band).  Sizing it
  // to the FINEST element keeps the jump resolved there (a sub-cell step Gibbs-
  // oscillates at p=3 and, with no AV at t=0 since v=0, blows up on step 1).
  real hF[3]; dgElemSize(grid, grid.nLvls-1, hF);
  real delta = grid.icDelta * hF[0];

  switch (grid.icType) {
    case 0: {  // x-aligned Sod, interface at x = cx
      real phi = (real)0.5*((real)1.0 + tanh((cx - x)/delta));
      W[0] = (real)0.125 + ((real)1.0 - (real)0.125)*phi;
      W[1] = W[2] = W[3] = 0.0;
      W[4] = (real)0.1 + ((real)1.0 - (real)0.1)*phi;
    } break;
    case 1: {  // circular (pseudo2D) / spherical Sod blast, dgsem strengths
      real dx = x-cx, dy = y-cy, dz = grid.pseudo2D ? (real)0.0 : (z-cz);
      real r = sqrt(dx*dx + dy*dy + dz*dz);
      real phi = (real)0.5*((real)1.0 + tanh(((real)0.25*grid.domainSize[0] - r)/delta));
      W[0] = (real)0.125 + ((real)11.0 - (real)0.125)*phi;
      W[1] = W[2] = W[3] = 0.0;
      W[4] = (real)0.1 + ((real)10.0 - (real)0.1)*phi;
    } break;
    case 2:    // isentropic vortex (pseudo2D)
      dgVortexExact(x, y, grid.vortexU0, grid.vortexU0, cx, cy, W);
      break;
    case 3:    // uniform free stream (nonzero velocity: exercises all flux paths)
      W[0] = 1.0; W[1] = 0.3; W[2] = 0.2; W[3] = grid.pseudo2D ? (real)0.0 : (real)0.1;
      W[4] = 1.0;
      break;
    case 4:    // double Mach reflection initial state
      if (dgDmrPostSide(x, y, (real)0.0, grid.dmrShockPos)) dgDmrPost(W);
      else dgDmrPre(W);
      break;
    case 5: {  // Gaussian density pulse advecting diagonally
      real dx = x-cx, dy = y-cy, dz = grid.pseudo2D ? (real)0.0 : (z-cz);
      real s2 = (real)0.01*grid.domainSize[0]*grid.domainSize[0];
      W[0] = (real)1.0 + (real)0.5*exp(-(dx*dx+dy*dy+dz*dz)/s2);
      W[1] = 0.5; W[2] = 0.3; W[3] = grid.pseudo2D ? (real)0.0 : (real)0.2;
      W[4] = 1.0;
    } break;
    case 6: {  // Gresho vortex (Gresho & Chan): exact centrifugal balance,
               // rho = 1, peak u_phi = 1 at r = 0.2, p0 = greshoP0 sets Mach
      real dx = x-cx, dy = y-cy;
      real r  = sqrt(dx*dx + dy*dy);
      real wang, pr, p0 = grid.greshoP0;
      if (r < (real)0.2)      { wang = (real)5.0;
                                pr = p0 + (real)12.5*r*r; }
      else if (r < (real)0.4) { wang = (real)2.0/r - (real)5.0;
                                pr = p0 + (real)12.5*r*r + (real)4.0*log((real)5.0*r)
                                   - (real)20.0*r + (real)4.0; }
      else                    { wang = (real)0.0;
                                pr = p0 - (real)2.0 + (real)4.0*log((real)2.0); }
      W[0] = 1.0;
      W[1] = -wang*dy;
      W[2] =  wang*dx;
      W[3] = 0.0;
      W[4] = pr;
    } break;
    case 7:   // uniform freestream at Mach machInf (a_inf = 1); the immersed
              // solid gets it too -- ghost fills overwrite, dead is never read
      W[0] = (real)1.0; W[1] = grid.machInf; W[2] = W[3] = (real)0.0;
      W[4] = (real)1.0/dgGam;
      break;
    default:
      W[0] = 1.0; W[1] = W[2] = W[3] = 0.0; W[4] = 1.0;
  }
}

__global__ void dgSetICKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    real h[3]; dgElemSize(grid, lvl, h);
    real x = dgNodePos(h[0], ib, i);
    real y = dgNodePos(h[1], jb, j);
    real z = dgNodePos(h[2], kb, k);

    real W[5], U[5];
    dgEvalIC(grid, x, y, z, lvl, W);
    dgP2C(W, U);
    for (i32 q = 0; q < 5; q++) grid.getField(D_RHO + q)[cIdx] = U[q];
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Trace helpers for the nonconforming (Nitsche/mortar) face coupling
 * ════════════════════════════════════════════════════════════════════════ */

// interpolate a coarse face-trace (Wcf, face-node flattened, SANITIZED prims)
// to the fine point (fa, fb) of subface (s1, s2).  zIdent: the t2 axis is the
// pseudo2D z direction (conforming node-to-node, no interpolation).
// Both sides of a nonconforming face call THIS function with identical inputs,
// making the twice-computed numerical fluxes bitwise identical.
__device__ __forceinline__ void dgTraceAt(const real Wcf[5][NNODE*NNODE],
                                          i32 s1, i32 s2, i32 fa, i32 fb,
                                          bool zIdent, real out[5]) {
  for (i32 q = 0; q < 5; q++) {
    real acc = 0.0;
    if (zIdent) {
      for (i32 c1 = 0; c1 < NNODE; c1++)
        acc += c_I[s1][fa][c1]*Wcf[q][c1 + NNODE*fb];
    } else {
      for (i32 c2 = 0; c2 < NNODE; c2++) {
        real t2c = c_I[s2][fb][c2];
        for (i32 c1 = 0; c1 < NNODE; c1++)
          acc += c_I[s1][fa][c1]*t2c*Wcf[q][c1 + NNODE*c2];
      }
    }
    out[q] = acc;
  }
}

// gather an element's face trace as SANITIZED primitives into face-node layout
// (from global conservative fields)
__device__ __forceinline__ void dgGatherFacePrims(DgSolver &grid, i32 eIdx,
                                                  i32 dir, i32 nrm,
                                                  real Wcf[5][NNODE*NNODE]) {
  for (i32 c2 = 0; c2 < NNODE; c2++)
    for (i32 c1 = 0; c1 < NNODE; c1++) {
      i32 nd = dgFaceNode(dir, nrm, c1, c2);
      real U[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[eIdx*blockSizeTot + nd];
      real W[5];
      dgConsToPrimSane(U, W);
      for (i32 q = 0; q < 5; q++) Wcf[q][c1 + NNODE*c2] = W[q];
    }
}

// weak boundary ghost state at a face quadrature point
__device__ __forceinline__ void dgBcState(DgSolver &grid, const real Win[5],
                                          i32 dir, i32 side,
                                          real x, real y, real t, real Wg[5]) {
  switch (grid.bcType) {
    case 0:   // slip wall: mirror the normal velocity
      for (i32 q = 0; q < 5; q++) Wg[q] = Win[q];
      Wg[1+dir] = -Win[1+dir];
      break;
    case 4:   // double Mach reflection
      if (dir == 0 && side == 0) { dgDmrPost(Wg); }                       // left inflow
      else if (dir == 0 && side == 1) { for (i32 q=0;q<5;q++) Wg[q]=Win[q]; }  // right outflow
      else if (dir == 1 && side == 0) {                                   // bottom
        if (x < grid.dmrShockPos) dgDmrPost(Wg);
        else { for (i32 q=0;q<5;q++) Wg[q]=Win[q]; Wg[2] = -Win[2]; }     // reflecting wall
      }
      else if (dir == 1 && side == 1) {                                   // top: moving shock
        if (dgDmrPostSide(x, y, t, grid.dmrShockPos)) dgDmrPost(Wg); else dgDmrPre(Wg);
      }
      else { for (i32 q=0;q<5;q++) Wg[q]=Win[q]; }                        // z: transmissive
      break;
    case 5:   // supersonic freestream: x-lo Dirichlet inflow (all
              // characteristics incoming), everything else transmissive
      if (dir == 0 && side == 0) {
        Wg[0] = (real)1.0; Wg[1] = grid.machInf; Wg[2] = (real)0.0;
        Wg[3] = (real)0.0; Wg[4] = (real)1.0/dgGam;   // a_inf = 1, u = M
      } else {
        for (i32 q = 0; q < 5; q++) Wg[q] = Win[q];
      }
      break;
    case 3:   // transmissive (zero gradient)
    default:
      for (i32 q = 0; q < 5; q++) Wg[q] = Win[q];
      break;
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Per-element AV viscosity nu_e -> D_SCRATCH[elem node 0], computed BEFORE
 * each RHS stage so the face lift can apply an interface jump penalty (the
 * element-local AV Laplacian carries no dissipation ACROSS faces -- a shock
 * straddling a coarse/fine interface then blows up; this is the BR2-lite
 * coupling and the seam where full NS/BR2 viscosity slots in later).
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void dgAvNuKernel(DgSolver &grid) {
  __shared__ real sV [DG_EPB][2][blockSizeTot];   // Ducros: u,v | Persson: rho, modal
  __shared__ real sRed[DG_EPB][2][blockSizeTot];  // theta|energy / lambda reduce

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    const bool active = (loc != kEmpty);
    i32 lvl = 0, ib = 0, jb = 0, kb = 0;
    if (active) grid.decode(loc, lvl, ib, jb, kb);
    real h[3] = {1,1,1};
    if (active) dgElemSize(grid, lvl, h);

    // sensorType: 0 = Ducros only, 1 = Persson only, 2 = MAX of both (default).
    // The two have complementary blind spots -- Ducros misses divergence-free
    // oscillation layers (nlvls-6 wall reflection); Persson (rho-modal) misses
    // discontinuities sitting BETWEEN elements and corruption living in the
    // momentum field while rho stays smooth (nlvls-3 blast, t=0.015).
    const bool doDucros  = (grid.sensorType != 1);
    const bool doPersson = (grid.sensorType >= 1);

    real lamNode = 0, c2 = 0, rhoNode = 0;
    if (active) {
      real U[5], W[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      c2 = dgGam*W[4]/fmax(W[0], (real)1e-12);
      lamNode = fabs(W[1]) + fabs(W[2]) + fabs(W[3]) + sqrt(fmax(c2, (real)1e-14));
      rhoNode = W[0];   // Persson senses DENSITY (pressure gets sanitizer-floored
                        // to a constant in near-vacuum zones -> sensor blindness)
      sV[ell][0][nd] = W[1];
      sV[ell][1][nd] = W[2];
    }
    sRed[ell][1][nd] = lamNode;
    __syncthreads();

    real thD = 0.0;
    if (doDucros) {
      // ── Ducros: compression rate vs acoustic rate, per node -> element max ──
      real thetaNode = 0;
      if (active) {
        real jacx = (real)2.0/h[0], jacy = (real)2.0/h[1];
        real lenp = h[0]/(real)(2*dgOrder+1);
        i32 ndX0 = j*NNODE + k*NNODE*NNODE, ndY0 = i + k*NNODE*NNODE;
        real du = 0, dv = 0;
        for (i32 m = 0; m < NNODE; m++) {
          du += c_D[i][m]*sV[ell][0][ndX0 + m];
          dv += c_D[j][m]*sV[ell][1][ndY0 + m*NNODE];
        }
        real divu = jacx*du + jacy*dv;
        if (!grid.pseudo2D) {
          real dw = 0;
          for (i32 m = 0; m < NNODE; m++) {
            real U[5], W[5];
            i32 nz = i + j*NNODE + m*NNODE*NNODE;
            for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)bIdx*blockSizeTot + nz];
            dgConsToPrimSane(U, W);
            dw += c_D[k][m]*W[3];
          }
          divu += ((real)2.0/h[2])*dw;
        }
        divu = fmin(divu, (real)0.0);
        real du2 = divu*divu;
        thetaNode = du2/(du2 + grid.avKsensor*c2/(lenp*lenp) + (real)1e-30);
      }
      sRed[ell][0][nd] = thetaNode;
      __syncthreads();
      for (i32 m = 0; m < blockSizeTot; m++) thD = fmax(thD, sRed[ell][0][m]);
      __syncthreads();   // sRed[0] reused by the Persson pass
    }

    if (doPersson) {
      // ── Persson-Peraire: energy fraction of the highest Legendre modes ──
      sV[ell][0][nd] = rhoNode;
      __syncthreads();
      real a = 0;
      if (active) {
        for (i32 c = 0; c < NNODE; c++) {
          real vc = c_Vinv[k][c];
          for (i32 b = 0; b < NNODE; b++) {
            real vb = c_Vinv[j][b]*vc;
            for (i32 aa = 0; aa < NNODE; aa++)
              a += c_Vinv[i][aa]*vb*sV[ell][0][aa + b*NNODE + c*NNODE*NNODE];
          }
        }
      }
      sRed[ell][0][nd] = a*a;   // top-mode set = max(i,j,k)==p
      __syncthreads();
    }

    if (active && nd == 0) {
      real lam = 0;
      for (i32 m = 0; m < blockSizeTot; m++) lam = fmax(lam, sRed[ell][1][m]);

      real th = thD;
      if (doPersson) {
        real total = 0, top = 0;
        for (i32 m = 0; m < blockSizeTot; m++) {
          i32 mi = m % NNODE, mj = (m/NNODE) % NNODE, mk = m/(NNODE*NNODE);
          total += sRed[ell][0][m];
          if (mi == NNODE-1 || mj == NNODE-1 || mk == NNODE-1)
            top += sRed[ell][0][m];
        }
        real S = top/fmax(total, (real)1e-30);
        real s = log10(fmax(S, (real)1e-30));
        real s0 = grid.ppS0, kap = grid.ppKappa;
        real thP = (s < s0-kap) ? (real)0.0
                 : (s > s0+kap) ? (real)1.0
                 : (real)0.5*((real)1.0 + sin((real)0.5*(real)PI*(s - s0)/kap));
        th = fmax(th, thP);
        // fluctuation modal energy (total minus the mean mode): the
        // indicator-1 amplitude floor
        grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 2] = total - sRed[ell][0][0];
      }
      // face jump penalty scale: sensor-gated wavespeed theta_e * lambda_e
      // (Rusanov-type, bounded by the physical wavespeed => explicit-dt stable;
      // an IP-style (p+1)^2/h scaling blows up through the 1/w0 lift factor).
      // IB ghost elements publish an UNGATED fraction ibPen of the Rusanov
      // scale lambda_e, so fluid<->ghost faces always carry a jump penalty:
      // the ghost-fill <-> fluid feedback loop has a wall-reflection mode
      // slightly above unity that HLLC cannot damp near u = 0 (measured
      // rest-state e-folding ~0.06 with zero published here), while the FULL
      // lambda exceeds the explicit-dt penalty bound through the 1/w0 lift
      // (blowup at t~1).  theta (slab +1) stays 0 -- ghosts never gate
      // volume AV or the fill's donor-order fallback.
      const bool ibFluid = (grid.ibClassList[bIdx] == IB_FLUID);
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot]     = grid.avOn ? (ibFluid ? th*lam : grid.ibPen*lam) : (real)0.0;
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1] = ibFluid ? th : (real)0.0;
    }
    __syncthreads();
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * The RHS kernel: volume (EC flux differencing) + faces (conforming /
 * mortar / weak BC) + Ducros-sensor element-local artificial viscosity
 * ════════════════════════════════════════════════════════════════════════ */

// interface jump penalty (BR2-lite): add the interior-penalty diffusive flux
// -sigma (U_R - U_L) to the face flux.  Both sides compute the identical value
// from the identical canonical trace pair, so conservation is untouched.  This
// is the ONLY cross-face dissipation the AV has -- the volume AV Laplacian is
// element-local, and without this a shock straddling a nonconforming interface
// feeds the jump unstably.
__device__ __forceinline__ void dgJumpPenalty(const real WL[5], const real WR[5],
                                              real sigma, real fs[5]) {
  real UL[5], UR[5];
  dgP2C(WL, UL);
  dgP2C(WR, UR);
  for (i32 q = 0; q < 5; q++) fs[q] -= sigma*(UR[q] - UL[q]);
}

// penalty coefficient for a face: a Rusanov scale gated by the LARGER of the
// elements' modal sensors and the face's OWN relative density jump,
//   sigma = avPen * avCav * 0.5 * max( max(sensL, sensR),  theta_jump * lam_face )
//   theta_jump = jr/(jr + 0.1),  jr = |rho_R - rho_L| / rho_avg
// The modal (Persson) sensor is blind to a discontinuity that sits BETWEEN
// elements (zero intra-element content, theta = 0 exactly at the jump), which
// blew up the nlvls-3 blast at t = 0.015 -- the jump self-gate sees precisely
// what the penalty damps, independent of element alignment.  Still bounded by
// the physical wavespeed => explicit-dt stable like HLLC itself; exactly zero
// on jump-free faces (free-stream preservation untouched).
__device__ __forceinline__ real dgPenaltySigma(DgSolver &grid, real sL, real sR,
                                               const real WL[5], const real WR[5]) {
  real lamF = fmax(fabs(WL[1])+fabs(WL[2])+fabs(WL[3]) + dgSoundSpeed(WL[4], WL[0]),
                   fabs(WR[1])+fabs(WR[2])+fabs(WR[3]) + dgSoundSpeed(WR[4], WR[0]));
  real jr = fabs(WR[0] - WL[0]) / fmax((real)0.5*(WL[0]+WR[0]), (real)1e-12);
  real thJ = jr/(jr + (real)0.1);
  return grid.avPen * grid.avCav * (real)0.5 * fmax(fmax(sL, sR), thJ*lamF);
}

// face lift for one face of one element, executed by that face's node threads.
// sWe: this element's sanitized primitives (shared).  Adds into R[5].
__device__ void dgFaceLift(DgSolver &grid, const real (*sWe)[blockSizeTot],
                           i32 bIdx, i32 lvl, i32 ib, i32 jb, i32 kb,
                           i32 dir, i32 side, i32 a, i32 b,
                           const real h[3], real t, real R[5]) {
  const i32 faceSlot[3][2] = {{12,14},{10,16},{4,22}};

  const i32  nrm    = side ? (NNODE-1) : 0;         // my face-normal node index
  const i32  myNd   = dgFaceNode(dir, nrm, a, b);
  const real jacDir = (real)2.0/h[dir];
  const real sgn    = side ? (real)-1.0 : (real)1.0;
  const bool zIdent = (grid.pseudo2D != 0) && (dir != 2);  // t2 axis is unrefined z

  real Wme[5];
  for (i32 q = 0; q < 5; q++) Wme[q] = sWe[q][myNd];
  real fOwn[5];
  dgEulerFluxAxis(Wme, dir, fOwn);
  const real nuOwn = grid.avOn ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot] : (real)0.0;

  // ── resolve the face topology ────────────────────────────────────────
  i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
  i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
  i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);

  i32 nSame = grid.nbrIdxList[27*bIdx + faceSlot[dir][side]];

  if (nSame == bEmpty && grid.isExteriorBlock(lvl, nib, njb, nkb)) {
    if (grid.bcType == 2) {
      // periodic: wrap and fall through to the wrapped same/coarse/fine dispatch
      grid.wrapBlockPeriodic(lvl, nib, njb, nkb);
      nSame = grid.getBlockIdx(grid.encode(lvl, nib, njb, nkb));
    } else {
      // weak BC: ghost state at my face node, conforming-style lift
      // physical position of this face point (for space/time-dependent BCs)
      real xs[3];
      xs[dir] = (ib*(dir==0)+jb*(dir==1)+kb*(dir==2) + (side ? 1 : 0)) * h[dir];
      i32 t1ax = (dir==0) ? 1 : 0;
      i32 t2ax = (dir==2) ? 1 : 2;
      i32 t1bb = (dir==0) ? jb : ib;
      i32 t2bb = (dir==2) ? jb : kb;
      xs[t1ax] = dgNodePos(h[t1ax], t1bb, a);
      xs[t2ax] = dgNodePos(h[t2ax], t2bb, b);
      real Wg[5];
      dgBcState(grid, Wme, dir, side, xs[0], xs[1], t, Wg);
      real fs[5];
      if (side) dgHllcAxis(Wme, Wg, dir, fs);
      else      dgHllcAxis(Wg, Wme, dir, fs);
      if (grid.avOn) {   // ghost shares my sensor; the jump self-gate sees the rest
        real sig = side ? dgPenaltySigma(grid, nuOwn, nuOwn, Wme, Wg)
                        : dgPenaltySigma(grid, nuOwn, nuOwn, Wg, Wme);
        if (side) dgJumpPenalty(Wme, Wg, sig, fs);
        else      dgJumpPenalty(Wg, Wme, sig, fs);
      }
      for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs[q] - fOwn[q]);
      return;
    }
  }

  const i32 nbrNrm = side ? 0 : (NNODE-1);   // facing face of the neighbor

  if (nSame != bEmpty) {
    // ── conforming: node-to-node HLLC, canonical +axis argument order ──
    real U[5], Wn[5];
    i32 nd = dgFaceNode(dir, nbrNrm, a, b);
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[nSame*blockSizeTot + nd];
    dgConsToPrimSane(U, Wn);
    real fs[5];
    if (side) dgHllcAxis(Wme, Wn, dir, fs);
    else      dgHllcAxis(Wn, Wme, dir, fs);
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)nSame*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, Wme, Wn)
                      : dgPenaltySigma(grid, nuOwn, nuN, Wn, Wme);
      if (side) dgJumpPenalty(Wme, Wn, sig, fs);
      else      dgJumpPenalty(Wn, Wme, sig, fs);
    }
    for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs[q] - fOwn[q]);
    return;
  }

  // tangential block indices (mine and the missing neighbor's -- they agree)
  const i32 t1b = (dir==0) ? njb : nib;
  const i32 t2b = (dir==2) ? njb : nkb;

  // ── coarser neighbor: I am the fine side ────────────────────────────
  i32 cIdxN = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                           grid.pseudo2D ? nkb : (nkb>>1)));
  if (lvl > 0 && cIdxN != bEmpty) {
    const i32 s1 = t1b & 1;
    const i32 s2 = zIdent ? 0 : (t2b & 1);
    real Wcf[5][NNODE*NNODE];
    dgGatherFacePrims(grid, cIdxN, dir, nbrNrm, Wcf);
    real Wc[5];
    dgTraceAt(Wcf, s1, s2, a, b, zIdent, Wc);
    dgSanitizePrim(Wc);   // Lagrange interp of the coarse trace can overshoot to
                          // negative rho/p between nodes at a strong shock; the
                          // coarse (mortar) side sanitizes the identical value,
                          // so the twice-computed flux stays bitwise-consistent
    real fs[5];
    if (side) dgHllcAxis(Wme, Wc, dir, fs);
    else      dgHllcAxis(Wc, Wme, dir, fs);
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)cIdxN*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, Wme, Wc)
                      : dgPenaltySigma(grid, nuOwn, nuN, Wc, Wme);
      if (side) dgJumpPenalty(Wme, Wc, sig, fs);
      else      dgJumpPenalty(Wc, Wme, sig, fs);
    }
    for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs[q] - fOwn[q]);
    return;
  }

  // ── finer neighbors: I am the coarse (mortar) side ──────────────────
  // F*_c = exact-L2 projection of the mortar flux back to my face polynomial:
  //   F*_c(a,b) = sum over subfaces (s1,s2) and fine points (fa,fb) of
  //     R[s1][a][fa] * (R[s2][b][fb] | delta_z) * f*
  // with f* the SAME pointwise HLLC values the fine sides compute.  The exact
  // projection (R o I = Id) passes resolved fluxes through unchanged -- the
  // GLL-discrete transpose is NOT a projection and its aliasing perturbation
  // proved linearly UNSTABLE (free-stream test grows exponentially).  R's
  // mean-adjoint identity (sum_a w_a R[s][a][fa] = w_fa/2) keeps the exchange
  // discretely conservative.
  {
    // my own face trace in face-node layout (sanitized prims from shared)
    real Wcf[5][NNODE*NNODE];
    for (i32 c2 = 0; c2 < NNODE; c2++)
      for (i32 c1 = 0; c1 < NNODE; c1++) {
        i32 nd = dgFaceNode(dir, nrm, c1, c2);
        for (i32 q = 0; q < 5; q++) Wcf[q][c1 + NNODE*c2] = sWe[q][nd];
      }

    real Fs[5] = {0,0,0,0,0};
    const i32 s2max = zIdent ? 1 : 2;
    for (i32 s2 = 0; s2 < s2max; s2++)
      for (i32 s1 = 0; s1 < 2; s1++) {
        // the child element behind subface (s1,s2)
        i32 cib, cjb, ckb;
        if (dir == 0) { cib = 2*nib + (side ? 0 : 1); cjb = 2*t1b + s1;
                        ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
        else if (dir == 1) { cjb = 2*njb + (side ? 0 : 1); cib = 2*t1b + s1;
                             ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
        else { ckb = 2*nkb + (side ? 0 : 1); cib = 2*t1b + s1; cjb = 2*t2b + s2; }
        i32 fIdx = grid.getBlockIdx(grid.encode(lvl+1, cib, cjb, ckb));
        if (fIdx == bEmpty) continue;   // grading guarantees existence; guard anyway

        for (i32 fb = (zIdent ? b : 0); fb < (zIdent ? b+1 : NNODE); fb++) {
          real wtB = zIdent ? (real)1.0 : c_R[s2][b][fb];
          for (i32 fa = 0; fa < NNODE; fa++) {
            // my trace interpolated to the fine point (identical to what the
            // fine side computes as its coarse-neighbor trace, sanitized the
            // same way so the pointwise flux matches bitwise)
            real To[5];
            dgTraceAt(Wcf, s1, s2, fa, fb, zIdent, To);
            dgSanitizePrim(To);
            // the fine element's own face node
            real U[5], Wf[5];
            i32 nd = dgFaceNode(dir, nbrNrm, fa, fb);
            for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[fIdx*blockSizeTot + nd];
            dgConsToPrimSane(U, Wf);
            real fs[5];
            if (side) dgHllcAxis(To, Wf, dir, fs);
            else      dgHllcAxis(Wf, To, dir, fs);
            if (grid.avOn) {
              real nuF = grid.getField(D_SCRATCH)[(u64)fIdx*blockSizeTot];
              real sig = side ? dgPenaltySigma(grid, nuOwn, nuF, To, Wf)
                              : dgPenaltySigma(grid, nuOwn, nuF, Wf, To);
              if (side) dgJumpPenalty(To, Wf, sig, fs);
              else      dgJumpPenalty(Wf, To, sig, fs);
            }
            real coef = c_R[s1][a][fa] * wtB;
            for (i32 q = 0; q < 5; q++) Fs[q] += coef*fs[q];
          }
        }
      }
    for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(Fs[q] - fOwn[q]);
  }
}

__global__ void dgRhsKernel(DgSolver &grid, real t) {
  __shared__ real sW [DG_EPB][5][blockSizeTot];   // sanitized primitives
  __shared__ real sGx[DG_EPB][5][blockSizeTot];   // AV gradient banks
  __shared__ real sGy[DG_EPB][5][blockSizeTot];
  __shared__ real sGz[DG_EPB][5][blockSizeTot];
  __shared__ real sRed[DG_EPB][2][blockSizeTot];  // theta / lambda reductions

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    // IB ghost/dead elements are never evolved (their nodal values are set by
    // the wall reconstruction); they still provide face traces to neighbors
    const bool active = (loc != kEmpty) && (grid.ibClassList[bIdx] == IB_FLUID);
    i32 lvl = 0, ib = 0, jb = 0, kb = 0;
    if (active) grid.decode(loc, lvl, ib, jb, kb);

    real h[3] = {1,1,1};
    if (active) dgElemSize(grid, lvl, h);
    const real jacx = (real)2.0/h[0], jacy = (real)2.0/h[1], jacz = (real)2.0/h[2];

    // ── phase 1: load, sanitize, primitives -> shared ──────────────────
    if (active) {
      real U[5], W[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      for (i32 q = 0; q < 5; q++) sW[ell][q][nd] = W[q];
    }
    __syncthreads();

    real R[5] = {0,0,0,0,0};
    real lamNode = 0.0, thetaNode = 0.0;

    if (active) {
      real Wi[5];
      for (i32 q = 0; q < 5; q++) Wi[q] = sW[ell][q][nd];

      // ── phase 2: volume term ──────────────────────────────────────────
      const i32 ndX0 = j*NNODE + k*NNODE*NNODE;   // line bases
      const i32 ndY0 = i + k*NNODE*NNODE;
      const i32 ndZ0 = i + j*NNODE;
      if (grid.ecVolume) {
        real ax[5] = {0,0,0,0,0}, ay[5] = {0,0,0,0,0}, az[5] = {0,0,0,0,0};
        for (i32 m = 0; m < NNODE; m++) {
          real Wm[5], Fs[5];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndX0 + m];
          dgEcFluxAxis(Wi, Wm, 0, Fs);
          for (i32 q = 0; q < 5; q++) ax[q] += c_D[i][m]*Fs[q];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
          dgEcFluxAxis(Wi, Wm, 1, Fs);
          for (i32 q = 0; q < 5; q++) ay[q] += c_D[j][m]*Fs[q];
          if (!grid.pseudo2D) {
            for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
            dgEcFluxAxis(Wi, Wm, 2, Fs);
            for (i32 q = 0; q < 5; q++) az[q] += c_D[k][m]*Fs[q];
          }
        }
        for (i32 q = 0; q < 5; q++)
          R[q] = -(real)2.0*(jacx*ax[q] + jacy*ay[q] + jacz*az[q]);
      } else {
        for (i32 m = 0; m < NNODE; m++) {
          real Wm[5], Fs[5];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndX0 + m];
          dgEulerFluxAxis(Wm, 0, Fs);
          for (i32 q = 0; q < 5; q++) R[q] -= jacx*c_D[i][m]*Fs[q];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
          dgEulerFluxAxis(Wm, 1, Fs);
          for (i32 q = 0; q < 5; q++) R[q] -= jacy*c_D[j][m]*Fs[q];
          if (!grid.pseudo2D) {
            for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
            dgEulerFluxAxis(Wm, 2, Fs);
            for (i32 q = 0; q < 5; q++) R[q] -= jacz*c_D[k][m]*Fs[q];
          }
        }
      }

      // ── phase 3: face lifts (boundary-node threads only) ─────────────
      if (i == 0)        dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 0, 0, j, k, h, t, R);
      if (i == NNODE-1)  dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 0, 1, j, k, h, t, R);
      if (j == 0)        dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 1, 0, i, k, h, t, R);
      if (j == NNODE-1)  dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 1, 1, i, k, h, t, R);
      if (!grid.pseudo2D) {
        if (k == 0)       dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 2, 0, i, j, h, t, R);
        if (k == NNODE-1) dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 2, 1, i, j, h, t, R);
      }

      // ── phase 3.5: wave speed (the element sensor theta_e comes from
      //    dgAvNuKernel via the SCRATCH slab -- Ducros or Persson-Peraire) ──
      real c = dgSoundSpeed(Wi[4], Wi[0]);
      lamNode = fabs(Wi[1]) + fabs(Wi[2]) + fabs(Wi[3]) + c;
    }

    // ── per-element reductions: lam_e (theta_e read from the sensor slab) ──
    sRed[ell][0][nd] = thetaNode;
    sRed[ell][1][nd] = lamNode;
    __syncthreads();
    real lam_e = 0;
    for (i32 m = 0; m < blockSizeTot; m++)
      lam_e = fmax(lam_e, sRed[ell][1][m]);
    real theta_e = (active && grid.avOn)
                 ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1] : (real)0.0;

    // ── phase 4: element-local artificial viscosity (two-pass) ────────
    if (grid.avOn) {
      real lenp = h[0]/(real)(2*dgOrder+1);
      real nu_e = grid.avCav * theta_e * lenp * lam_e;
      if (active) {
        const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
        const i32 ndY0 = i + k*NNODE*NNODE;
        const i32 ndZ0 = i + j*NNODE;
        real gx[5] = {0,0,0,0,0}, gy[5] = {0,0,0,0,0}, gz[5] = {0,0,0,0,0};
        for (i32 m = 0; m < NNODE; m++) {
          real Wm[5], Um[5];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndX0 + m];
          dgP2C(Wm, Um);
          for (i32 q = 0; q < 5; q++) gx[q] += c_D[i][m]*Um[q];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
          dgP2C(Wm, Um);
          for (i32 q = 0; q < 5; q++) gy[q] += c_D[j][m]*Um[q];
          if (!grid.pseudo2D) {
            for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
            dgP2C(Wm, Um);
            for (i32 q = 0; q < 5; q++) gz[q] += c_D[k][m]*Um[q];
          }
        }
        for (i32 q = 0; q < 5; q++) {
          sGx[ell][q][nd] = nu_e*jacx*gx[q];
          sGy[ell][q][nd] = nu_e*jacy*gy[q];
          sGz[ell][q][nd] = grid.pseudo2D ? (real)0.0 : nu_e*jacz*gz[q];
        }
      }
      __syncthreads();
      if (active) {
        const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
        const i32 ndY0 = i + k*NNODE*NNODE;
        const i32 ndZ0 = i + j*NNODE;
        for (i32 q = 0; q < 5; q++) {
          real sx = 0, sy = 0, sz = 0;
          for (i32 m = 0; m < NNODE; m++) {
            sx += c_w[m]*c_D[m][i]*sGx[ell][q][ndX0 + m];
            sy += c_w[m]*c_D[m][j]*sGy[ell][q][ndY0 + m*NNODE];
            if (!grid.pseudo2D) sz += c_w[m]*c_D[m][k]*sGz[ell][q][ndZ0 + m*NNODE*NNODE];
          }
          R[q] -= jacx*c_winv[i]*sx + jacy*c_winv[j]*sy
                + (grid.pseudo2D ? (real)0.0 : jacz*c_winv[k]*sz);
        }
      }
    }

    // ── write RHS + the per-node dt bound ──────────────────────────────
    if (active) {
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHS+q)[bIdx*blockSizeTot + nd] = R[q];
      real hmin = fmin(h[0], grid.pseudo2D ? h[0] : fmin(h[1], h[2]));
      grid.getField(D_LAM)[bIdx*blockSizeTot + nd] =
          hmin/(fmax(lam_e, (real)1e-10)*(real)NNODE);
    }
    __syncthreads();   // shared reused next grid-stride iteration
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Time stepping: SSP-RK3 (Shu-Osher) + Zhang-Shu positivity limiter
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void dgCopyQ0Kernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_Q0+q)[cIdx] = grid.getField(D_RHO+q)[cIdx];
  }
}

__global__ void dgRk3StageKernel(DgSolver &grid, i32 stage, real dt) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;   // ghosts hold their fill
    real U[5];
    for (i32 q = 0; q < 5; q++) {
      real q0 = grid.getField(D_Q0+q)[cIdx];
      real qc = grid.getField(D_RHO+q)[cIdx];
      real L  = grid.getField(D_RHS+q)[cIdx];
      if (stage == 0)      U[q] = q0 + dt*L;
      else if (stage == 1) U[q] = (real)0.75*q0 + (real)0.25*(qc + dt*L);
      else                 U[q] = (real)(1.0/3.0)*q0 + (real)(2.0/3.0)*(qc + dt*L);
    }
    dgSanitizeCons(U);
    for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
  }
}

__global__ void dgPositivityKernel(DgSolver &grid) {
  const real eps_rho = 1e-12, eps_p = 1e-12;
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;   // ghost fills are
    // non-conservative by design: never Zhang-Shu-limit them
    real *F[5];
    for (i32 q = 0; q < 5; q++) F[q] = grid.getField(D_RHO+q) + (u64)bIdx*blockSizeTot;

    // GLL cell mean: (1/8) sum w_i w_j w_k U (z weights are uniform copies in pseudo2D)
    real Ubar[5] = {0,0,0,0,0};
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      real wijk = (real)0.125*c_w[i]*c_w[j]*c_w[k];
      for (i32 q = 0; q < 5; q++) Ubar[q] += wijk*F[q][nd];
    }
    Ubar[0] = fmax(Ubar[0], eps_rho);
    {
      real p = dgPressureFromCons(Ubar);
      if (p < eps_p)
        Ubar[4] = eps_p/(dgGam-(real)1.0)
                + (real)0.5*(Ubar[1]*Ubar[1]+Ubar[2]*Ubar[2]+Ubar[3]*Ubar[3])/Ubar[0];
    }

    // theta1: density
    real rhoMin = (real)1e30;
    for (i32 nd = 0; nd < blockSizeTot; nd++) rhoMin = fmin(rhoMin, F[0][nd]);
    real tRho = 1.0;
    if (rhoMin < eps_rho)
      tRho = (Ubar[0]-eps_rho)/fmax(Ubar[0]-rhoMin, (real)1e-30);
    tRho = fmax((real)0.0, fmin((real)1.0, tRho));

    // theta2: pressure (bisection per offending node)
    real theta = tRho;
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      real U[5], Ud[5];
      for (i32 q = 0; q < 5; q++) U[q] = F[q][nd];
      for (i32 q = 0; q < 5; q++) Ud[q] = Ubar[q] + tRho*(U[q]-Ubar[q]);
      if (dgPressureFromCons(Ud) < eps_p) {
        real lo = 0, hi = tRho;
        for (i32 it = 0; it < 20; it++) {
          real tm = (real)0.5*(lo+hi);
          real Um[5];
          for (i32 q = 0; q < 5; q++) Um[q] = Ubar[q] + tm*(U[q]-Ubar[q]);
          if (dgPressureFromCons(Um) >= eps_p) lo = tm; else hi = tm;
        }
        theta = fmin(theta, lo);
      }
    }

    if (theta < (real)1.0) {
      for (i32 nd = 0; nd < blockSizeTot; nd++)
        for (i32 q = 0; q < 5; q++)
          F[q][nd] = Ubar[q] + theta*(F[q][nd]-Ubar[q]);
    }
  }
}

__global__ void dgLamKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || grid.ibClassList[bIdx] != IB_FLUID) {
      grid.getField(D_LAM)[cIdx] = 1e30;   // IB ghost/dead never bound dt
      continue;
    }
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    real U[5], W[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[cIdx];
    dgConsToPrimSane(U, W);
    real lam = fabs(W[1]) + fabs(W[2]) + fabs(W[3]) + dgSoundSpeed(W[4], W[0]);
    real hmin = fmin(h[0], grid.pseudo2D ? h[0] : fmin(h[1], h[2]));
    grid.getField(D_LAM)[cIdx] = hmin/(fmax(lam, (real)1e-10)*(real)NNODE);
  }
}

__global__ void dgSortFieldDataKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    i32 bIdxOld = grid.bIdxList[bIdx];
    i32 cIdxOld = bIdxOld*blockSizeTot + cIdx % blockSizeTot;
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_RHO+q)[cIdx] = grid.getField(D_Q0+q)[cIdxOld];
    grid.bFlagsList[bIdxOld] = DELETE;
  }
}

__global__ void dgSnapshotQ0Kernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_Q0+q)[cIdx] = grid.getField(D_RHO+q)[cIdx];
  }
}

__global__ void dgPressureToScratchKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    real U[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[cIdx];
    grid.getField(D_SCRATCH)[cIdx] = dgPressureFromCons(U);
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Image build: evaluate the DG polynomial at the uniform pixel centers
 * (Lagrange interpolation from the LGL solution nodes), not nearest-node fill
 * ════════════════════════════════════════════════════════════════════════ */

// Lagrange basis l_a(x) on the LGL nodes (constant memory c_xi)
__device__ __forceinline__ real dgBasisAt(i32 a, real x) {
  real v = 1.0;
  for (i32 m = 0; m < NNODE; m++)
    if (m != a) v *= (x - c_xi[m])/(c_xi[a] - c_xi[m]);
  return v;
}

__global__ void dgComputeImageDataKernel(DgSolver &grid, i32 f) {
  real *U = (f >= 0) ? grid.getField(f) : nullptr;
  const real zmid = (real)0.5*grid.domainSize[2];

  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (!grid.isInteriorBlock(lvl, ib, jb, kb)) continue;

    // this element must intersect the mid-z slice
    real h[3]; dgElemSize(grid, lvl, h);
    real zeta = 0.0;
    if (!grid.pseudo2D) {
      real z0 = kb*h[2];
      if (zmid < z0 || zmid >= z0 + h[2]) continue;
      zeta = (real)2.0*(zmid - z0)/h[2] - (real)1.0;
    }

    i32 nPix = powi(2, grid.nLvls-1-lvl);   // pixels per DG cell
    i32 span = blockSize*nPix;              // pixels per element per axis

    // contract the z-direction once: plane[a][b] = sum_c l_c(zeta) U[a,b,c]
    real Lz[NNODE], plane[NNODE][NNODE];
    if (f >= 0) {
      for (i32 c = 0; c < NNODE; c++) Lz[c] = dgBasisAt(c, zeta);
      for (i32 b = 0; b < NNODE; b++)
        for (i32 a = 0; a < NNODE; a++) {
          real s = 0.0;
          for (i32 c = 0; c < NNODE; c++)
            s += Lz[c]*U[(u64)bIdx*blockSizeTot + a + b*NNODE + c*NNODE*NNODE];
          plane[a][b] = s;
        }
    }

    for (i32 py = 0; py < span; py++) {
      real eta = (real)2.0*(py + (real)0.5)/span - (real)1.0;
      real Ly[NNODE];
      for (i32 b = 0; b < NNODE; b++) Ly[b] = dgBasisAt(b, eta);
      i32 jPxl = jb*span + py;
      if (jPxl < 0 || jPxl >= grid.imageSizeX[1]) continue;

      for (i32 px = 0; px < span; px++) {
        i32 iPxl = ib*span + px;
        if (iPxl < 0 || iPxl >= grid.imageSizeX[0]) continue;
        real val;
        if (f >= 0) {
          real xi = (real)2.0*(px + (real)0.5)/span - (real)1.0;
          real acc = 0.0;
          for (i32 a = 0; a < NNODE; a++) {
            real Lxa = dgBasisAt(a, xi);
            for (i32 b = 0; b < NNODE; b++) acc += Lxa*Ly[b]*plane[a][b];
          }
          val = acc;
        } else {
          // grid view: clean 1-pixel element boundaries, brightness by level
          // (coarse = darker, fine = brighter).  Drawing the left/bottom edge
          // of every element tiles into full gridlines (the shared edge is
          // owned by the +side element); finest elements are only blockSize
          // pixels wide, so a full border ring would read as solid -- a single
          // edge line stays thin at every level.
          val = (px == 0 || py == 0) ? (real)(lvl+1) : (real)0.0;
        }
        grid.imageDataX[(u64)jPxl*grid.imageSizeX[0] + iPxl] = val;
      }
    }
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * MRA indicator: transient restriction on the octet anchor, detail norms,
 * significance votes (wavelet-free MRA, leaf-only)
 * ════════════════════════════════════════════════════════════════════════ */

// atomicMax for nonnegative reals via ordered integer bits
__device__ __forceinline__ void dgAtomicMaxPos(real *addr, real v) {
#ifdef USE_DOUBLE
  atomicMax((unsigned long long*)addr, (unsigned long long)__double_as_longlong(v));
#else
  atomicMax((int*)addr, __float_as_int(v));
#endif
}

__global__ void dgScalesKernel(DgSolver &grid) {
  // register accumulation + block tree-reduction: 6 atomics per CUDA block.
  // (The naive per-node version issued ~1.5M atomicAdds onto SIX addresses per
  // adaptation; same-address atomics serialize, costing ~13 ms per adapt --
  // it dominated the entire solver at 75% of runtime.)
  __shared__ real sRed[6][cudaBlockSize];
  const i32 tid = threadIdx.x;
  const bool sum = (grid.scaleMode == 0);
  real part[6] = {0,0,0,0,0,0};

  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;   // solid values must not
    // pollute the global detail normalization
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (sum) {
      real h[3]; dgElemSize(grid, lvl, h);
      real volw = (h[0]*(real)0.5*c_w[i]) * (h[1]*(real)0.5*c_w[j]) * (h[2]*(real)0.5*c_w[k]);
      for (i32 q = 0; q < 5; q++)
        part[q] += volw*grid.getField(D_RHO+q)[cIdx];
      part[5] += volw;
    } else {
      for (i32 q = 0; q < 5; q++)
        part[q] = fmax(part[q], fabs(grid.getField(D_RHO+q)[cIdx]));
    }
  }

  for (i32 q = 0; q < 6; q++) sRed[q][tid] = part[q];
  __syncthreads();
  for (i32 s = cudaBlockSize/2; s > 0; s >>= 1) {
    if (tid < s)
      for (i32 q = 0; q < 6; q++)
        sRed[q][tid] = sum ? (sRed[q][tid] + sRed[q][tid+s])
                           : fmax(sRed[q][tid], sRed[q][tid+s]);
    __syncthreads();
  }
  if (tid == 0) {
    for (i32 q = 0; q < 6; q++) {
      if (sum) atomicAdd(&grid.globalScale[q], sRed[q][0]);
      else if (q < 5) dgAtomicMaxPos(&grid.globalScale[q], sRed[q][0]);
    }
  }
}

// single-thread epilogue: derive the indicator scales c_i from the reduced
// sums/maxima entirely on device -- the host never touches globalScale/cScale
// during a run (managed-memory page migration cost ~ms per adapt otherwise)
__global__ void dgFinalizeScalesKernel(DgSolver &grid) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (grid.scaleMode == 0) {
    real vol = fmax(grid.globalScale[5], (real)1e-30);
    for (i32 q = 0; q < 5; q++)
      grid.cScale[q] = fmax((real)1.0, fabs(grid.globalScale[q]/vol));
  } else {
    for (i32 q = 0; q < 5; q++)
      grid.cScale[q] = fmax((real)1e-12, grid.globalScale[q]);
  }
}

// is (lvl, ib,jb,kb) the anchor (even-parity member) of its sibling octet?
__device__ __forceinline__ bool dgIsAnchor(DgSolver &grid, i32 ib, i32 jb, i32 kb) {
  bool zEven = grid.pseudo2D ? true : ((kb & 1) == 0);
  return ((ib & 1) == 0) && ((jb & 1) == 0) && zEven;
}

// gather the sibling octet of the block at (lvl, aib, ajb, akb) (anchor coords);
// returns false if any sibling is missing at this level (refined deeper)
__device__ bool dgGatherOctet(DgSolver &grid, i32 lvl, i32 aib, i32 ajb, i32 akb,
                              i32 sIdx[8]) {
  i32 dkMax = grid.pseudo2D ? 0 : 1;
  bool ok = true;
  for (i32 dk = 0; dk <= 1; dk++)
    for (i32 dj = 0; dj <= 1; dj++)
      for (i32 di = 0; di <= 1; di++) {
        i32 s = di + 2*dj + 4*dk;
        if (dk > dkMax) { sIdx[s] = sIdx[s-4]; continue; }   // pseudo2D: alias z pair
        i32 idx = grid.getBlockIdx(grid.encode(lvl, aib+di, ajb+dj, akb+dk));
        sIdx[s] = idx;
        if (idx == bEmpty) ok = false;
      }
  return ok;
}

// exact-L2 restriction of the octet's nodal data to virtual-parent node (i,j,k)
__device__ real dgRestrictNode(DgSolver &grid, const i32 sIdx[8], i32 q,
                               i32 i, i32 j, i32 k) {
  real *F = grid.getField(D_RHO+q);
  real acc = 0.0;
  if (grid.pseudo2D) {
    for (i32 s2 = 0; s2 < 2; s2++)
      for (i32 s1 = 0; s1 < 2; s1++) {
        const real *Fc = F + (u64)sIdx[s1+2*s2]*blockSizeTot;
        for (i32 b = 0; b < NNODE; b++) {
          real rb = c_R[s2][j][b];
          for (i32 a = 0; a < NNODE; a++)
            acc += c_R[s1][i][a]*rb*Fc[a + b*NNODE + k*NNODE*NNODE];
        }
      }
  } else {
    for (i32 s3 = 0; s3 < 2; s3++)
      for (i32 s2 = 0; s2 < 2; s2++)
        for (i32 s1 = 0; s1 < 2; s1++) {
          const real *Fc = F + (u64)sIdx[s1+2*s2+4*s3]*blockSizeTot;
          for (i32 c = 0; c < NNODE; c++) {
            real rc = c_R[s3][k][c];
            for (i32 b = 0; b < NNODE; b++) {
              real rb = c_R[s2][j][b]*rc;
              for (i32 a = 0; a < NNODE; a++)
                acc += c_R[s1][i][a]*rb*Fc[a + b*NNODE + c*NNODE*NNODE];
            }
          }
        }
  }
  return acc;
}

// prolongation of virtual-parent data (uL, node-flattened) to the child node
// (i,j,k) of the child with parities (s1,s2,s3)
__device__ __forceinline__ real dgProlongNode(DgSolver &grid, const real *uL,
                                              i32 s1, i32 s2, i32 s3,
                                              i32 i, i32 j, i32 k) {
  real acc = 0.0;
  if (grid.pseudo2D) {
    for (i32 b = 0; b < NNODE; b++) {
      real tb = c_I[s2][j][b];
      for (i32 a = 0; a < NNODE; a++)
        acc += c_I[s1][i][a]*tb*uL[a + b*NNODE + k*NNODE*NNODE];
    }
  } else {
    for (i32 c = 0; c < NNODE; c++) {
      real tc = c_I[s3][k][c];
      for (i32 b = 0; b < NNODE; b++) {
        real tb = c_I[s2][j][b]*tc;
        for (i32 a = 0; a < NNODE; a++)
          acc += c_I[s1][i][a]*tb*uL[a + b*NNODE + c*NNODE*NNODE];
      }
    }
  }
  return acc;
}

// phase A: anchors of complete octets restrict to the virtual parent (Q0 bank
// of the anchor), zero the detail accumulator (LAM slab), set snapValid=1
__global__ void dgRestrictToAnchorKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (!dgIsAnchor(grid, ib, jb, kb)) continue;

    i32 sIdx[8];
    if (!dgGatherOctet(grid, lvl, ib, jb, kb, sIdx)) continue;
    {   // octets touching the immersed body carry no valid detail: any non-
        // fluid member leaves snapValid 0 (members then self-KEEP -> the octet
        // can never merge across the wall)
      i32 nS = grid.pseudo2D ? 4 : 8;
      bool allFluid = true;
      for (i32 sm = 0; sm < nS; sm++)
        if (grid.ibClassList[sIdx[sm]] != IB_FLUID) { allFluid = false; break; }
      if (!allFluid) continue;
    }

    for (i32 q = 0; q < 5; q++)
      grid.getField(D_Q0+q)[cIdx] = dgRestrictNode(grid, sIdx, q, i, j, k);
    grid.getField(D_LAM)[cIdx] = 0.0;
    if (cIdx % blockSizeTot == 0) grid.snapValidList[bIdx] = 1;
  }
}

// phase B: every member of a complete octet accumulates its GLL detail norm
// into the anchor's LAM slab (nodes 0..4 hold the 5 per-variable norms^2)
__global__ void dgDetailNormKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 aib = ib & ~1, ajb = jb & ~1, akb = grid.pseudo2D ? kb : (kb & ~1);
    i32 aIdx = grid.getBlockIdx(grid.encode(lvl, aib, ajb, akb));
    if (aIdx == bEmpty || !grid.snapValidList[aIdx]) continue;

    i32 s1 = ib & 1, s2 = jb & 1, s3 = grid.pseudo2D ? 0 : (kb & 1);
    real frac  = grid.pseudo2D ? (real)0.25 : (real)0.125;
    real wnode = (real)0.125*c_w[i]*c_w[j]*c_w[k];

    for (i32 q = 0; q < 5; q++) {
      const real *uLq = grid.getField(D_Q0+q) + (u64)aIdx*blockSizeTot;   // anchor's virtual-parent bank
      real P = dgProlongNode(grid, uLq, s1, s2, s3, i, j, k);
      real d = grid.getField(D_RHO+q)[cIdx] - P;
      atomicAdd(&grid.getField(D_LAM)[(u64)aIdx*blockSizeTot + q], frac*wnode*d*d);
    }
  }
}

// phase C: anchors threshold the octet significance and vote; incomplete-octet
// members and level-0 leaves self-KEEP
__global__ void dgVoteKernel(DgSolver &grid, real epsL, i32 allowRefine) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl == 0) atomicMax(&grid.bFlagsList[bIdx], KEEP);   // base level never coarsens

    i32 ibc = grid.ibClassList[bIdx];
    if (ibc == IB_DEAD) continue;      // stays DELETE: solid interior cascades to base
    if (ibc == IB_GHOST) {             // never coarsen a ghost (band pins it finest)
      atomicMax(&grid.bFlagsList[bIdx], KEEP);
      continue;
    }

    i32 aib = ib & ~1, ajb = jb & ~1, akb = grid.pseudo2D ? kb : (kb & ~1);
    i32 aIdx = grid.getBlockIdx(grid.encode(lvl, aib, ajb, akb));
    bool complete = (aIdx != bEmpty) && grid.snapValidList[aIdx];

    if (!complete) {
      // a sibling is refined deeper (the tree condition testifies significance),
      // or the octet extends outside the base grid: keep this leaf at its level
      // -- do NOT force it finer (that "octet completion" is not in the paper and
      // cascades to near-full refinement; the paper keeps the family as a graded
      // tree via the ancestor condition instead).
      atomicMax(&grid.bFlagsList[bIdx], KEEP);
      continue;
    }
    if (aIdx != bIdx) continue;   // the anchor votes for the octet

    real sig = 0.0;
    for (i32 q = 0; q < 5; q++) {
      real n2 = grid.getField(D_LAM)[(u64)bIdx*blockSizeTot + q];
      sig = fmax(sig, sqrt(fmax(n2, (real)0.0))/grid.cScale[q]);
    }
    // detail lives on the virtual parent at level lvl-1: eps = epsL * 2^(lvl-nLvls)
    real eps = epsL/(real)powi(2, grid.nLvls - lvl);

    // SINGLE threshold (refineFac*eps, refineFac default 1): a significant octet
    // refines toward the finest level; a smooth feature's detail decays as h^p
    // and drops below the threshold within a few levels, stopping its
    // refinement, while a shock's non-decaying detail keeps it refined to the
    // finest level -- so it never straddles a coarse/fine face.
    // allowRefine == 0: coarsen-guard mode for the alternative indicators --
    // the octet detail IS the information a merge would destroy (projection
    // residual), so it vetoes coarsening (KEEP) while the alternate indicator
    // supplies the REFINE votes.  Without this guard, jump/entropy indicators
    // merge small-but-real signal (they cannot see the loss a merge causes),
    // and the tail-ring churn pins the vortex error at ~3e-4.
    real thr = grid.refineFac*eps;
    i32 vote = (sig > thr) ? ((allowRefine && lvl < grid.nLvls-1) ? REFINE : KEEP) : DELETE;

    if (vote > DELETE) {
      i32 sIdx[8];
      dgGatherOctet(grid, lvl, aib, ajb, akb, sIdx);
      i32 nS = grid.pseudo2D ? 4 : 8;
      for (i32 s = 0; s < nS; s++)
        if (sIdx[s] != bEmpty) atomicMax(&grid.bFlagsList[sIdx[s]], vote);
    }
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Alternative adaptation indicators (--indicator 1/2/3).  Each produces a
 * per-element vote; everything downstream (neighbor rule, buffer, grading,
 * merge, spawn) is indicator-agnostic.
 * ════════════════════════════════════════════════════════════════════════ */

// indicator 1: vote directly on the element smoothness sensor theta_e
// (computed by dgAvNuKernel into SCRATCH[1]) with hysteresis thresholds
__global__ void dgSensorVoteKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;   // IB classes are handled
    // by the MRA vote gate + band vote
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (lvl == 0) atomicMax(&grid.bFlagsList[bIdx], KEEP);

    real th = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1];
    if (grid.sensorType == 1) {   // amplitude floor (Persson theta is scale-free)
      real fluct = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 2];
      real floor2 = (real)1e-12*grid.cScale[0]*grid.cScale[0];
      if (fluct < floor2) continue;   // quiescent: leave at DELETE (coarsenable)
    }
    i32 vote = DELETE;
    if (th > grid.ppRefine && lvl < grid.nLvls-1) vote = REFINE;
    else if (th > grid.ppCoarsen) vote = KEEP;
    if (vote > DELETE) atomicMax(&grid.bFlagsList[bIdx], vote);
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Immersed boundary: ghost-element Hermite reconstruction (cylinder SDF).
 * Cut elements (+ the first fully-solid layer facing fluid) are GHOSTs: never
 * evolved; every RK stage each of their nodes is refilled from a degree-3
 * polynomial along the local wall normal through image point -> node ->
 * boundary point, with the donor DG polynomial's value/1st/2nd normal
 * derivatives at the image point and the slip-wall BC at the boundary point.
 * ════════════════════════════════════════════════════════════════════════ */

// signed distance to the cylinder (axis along z): positive = fluid
__device__ __forceinline__ real dgIbPhi(DgSolver &grid, real x, real y) {
  real dx = x - grid.ibX, dy = y - grid.ibY;
  return sqrt(dx*dx + dy*dy) - grid.ibR;
}

// exact SDF range over an axis-aligned box (circle-to-box distance bounds)
__device__ __forceinline__ void dgIbPhiRangeBox(DgSolver &grid,
    real x0, real x1, real y0, real y1, real &phiMin, real &phiMax) {
  real cx = grid.ibX, cy = grid.ibY;
  real dxlo = fmax((real)0.0, fmax(x0 - cx, cx - x1));
  real dylo = fmax((real)0.0, fmax(y0 - cy, cy - y1));
  real dxhi = fmax(fabs(x0 - cx), fabs(x1 - cx));
  real dyhi = fmax(fabs(y0 - cy), fabs(y1 - cy));
  phiMin = sqrt(dxlo*dxlo + dylo*dylo) - grid.ibR;
  phiMax = sqrt(dxhi*dxhi + dyhi*dyhi) - grid.ibR;
}

// pass 1: geometric class from the element box's exact SDF range
__global__ void dgIbClassifyGeomKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) { grid.ibClassList[bIdx] = IB_FLUID; continue; }
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    real phiMin, phiMax;
    dgIbPhiRangeBox(grid, ib*h[0], (ib+1)*h[0], jb*h[1], (jb+1)*h[1], phiMin, phiMax);
    // an element is GHOST not only when cut but whenever its box comes
    // within ibGraze*h of the wall: a fully-fluid "grazing sliver" element
    // has evolved nodes arbitrarily close to the wall, coupled to
    // reconstructed traces on its wall-side faces -- a marginally unstable
    // stiff mode concentrated at its near-wall corner node (measured: rest
    // state grows at e-folding ~0.06 seeded by fp32 roundoff, worst node at
    // phi = 0.25h).  Reconstructing those slivers removes the mode at the
    // cost of a slightly thicker non-conservative layer.
    // ghost = element whose CENTER is inside the solid (the FV-IBM rule).
    // Classifying every cut element as ghost slaves genuinely-fluid nodes to
    // the reconstruction and makes the wall under-reflective: wall-normal
    // momentum is absorbed, not returned (measured at rest: an element-scale
    // alternating ring mode, e-folding ~0.06, immune to face-penalty/image-
    // distance/order/filter variations).  Center-in-fluid cut elements EVOLVE;
    // their solid-corner nodes are governed by the neighboring ghost fills
    // through the face coupling + Zhang-Shu.  ibGraze > 0 optionally widens
    // the ghost margin (kept as a knob; default 0 = pure center rule).
    // ibCut 1 (default, the design rule): every cut element is a ghost --
    // the wall lives inside the ghost layer and the fluid feels it through
    // the reconstructed traces (center-rule evolution lets flow stream
    // THROUGH the wall: measured stagnation Cp 0.02 instead of 1.75 at M=3).
    // ibCut 0: FV-IBM center rule (kept for A/B).
    real graze = grid.ibGraze*h[0];
    bool solidish;
    if (grid.ibCut) {
      solidish = (phiMin < graze);            // cut or grazing -> ghost
    } else {
      real cxE = (ib + (real)0.5)*h[0], cyE = (jb + (real)0.5)*h[1];
      solidish = (dgIbPhi(grid, cxE, cyE) < graze);
    }
    grid.ibClassList[bIdx] = !solidish ? ((phiMax <= (real)0.0) ? IB_DEAD : IB_FLUID)
                           : ((phiMax <= (real)0.0) ? IB_DEAD : IB_GHOST);
  }
}

// pass 2: a fully-solid element with any FLUID face neighbor (same level,
// coarser cover, or finer children -- the face-topology dispatch) becomes a
// GHOST, so no fluid face ever resolves to an unfilled DEAD element.  In-place
// and race-free: only rewrites DEAD -> GHOST and only reads "== IB_FLUID",
// which pass 1 fixed and this pass never changes.
__global__ void dgIbPromoteKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || grid.ibClassList[bIdx] != IB_DEAD) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    bool touchesFluid = false;
    for (i32 dir = 0; dir < (grid.pseudo2D ? 2 : 3) && !touchesFluid; dir++)
      for (i32 side = 0; side < 2 && !touchesFluid; side++) {
        i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
        i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
        i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);
        if (grid.isExteriorBlock(lvl, nib, njb, nkb)) continue;
        i32 nIdx = grid.getBlockIdx(grid.encode(lvl, nib, njb, nkb));
        if (nIdx != bEmpty) {
          touchesFluid = (grid.ibClassList[nIdx] == IB_FLUID);
          continue;
        }
        if (lvl > 0) {   // coarser cover
          nIdx = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                              grid.pseudo2D ? nkb : (nkb>>1)));
          if (nIdx != bEmpty) {
            touchesFluid = (grid.ibClassList[nIdx] == IB_FLUID);
            continue;
          }
        }
        if (lvl < grid.nLvls-1) {   // finer children covering the face
          const i32 t1b = (dir==0) ? njb : nib;
          const i32 t2b = (dir==2) ? njb : nkb;
          const i32 s2max = (grid.pseudo2D && dir != 2) ? 1 : 2;
          for (i32 s2 = 0; s2 < s2max && !touchesFluid; s2++)
            for (i32 s1 = 0; s1 < 2 && !touchesFluid; s1++) {
              i32 cib, cjb, ckb;
              if (dir == 0) { cib = 2*nib + (side ? 0 : 1); cjb = 2*t1b + s1;
                              ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else if (dir == 1) { cjb = 2*njb + (side ? 0 : 1); cib = 2*t1b + s1;
                                   ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else { ckb = 2*nkb + (side ? 0 : 1); cib = 2*t1b + s1; cjb = 2*t2b + s2; }
              i32 cIdxN = grid.getBlockIdx(grid.encode(lvl+1, cib, cjb, ckb));
              if (cIdxN != bEmpty)
                touchesFluid = (grid.ibClassList[cIdxN] == IB_FLUID);
            }
        }
      }
    if (touchesFluid) grid.ibClassList[bIdx] = IB_GHOST;
  }
}

// force the finest level where the element box intersects the band
// |phi| < ibBand * h_finest around the surface: donors and their neighbors
// stay finest-level, and every octet with a cut member holds a KEEP member,
// making a ghost/fluid mixed merge structurally impossible.
__global__ void dgIbBandVoteKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3];  dgElemSize(grid, lvl, h);
    real hF[3]; dgElemSize(grid, grid.nLvls-1, hF);
    real band = grid.ibBand*hF[0];
    real phiMin, phiMax;
    dgIbPhiRangeBox(grid, ib*h[0], (ib+1)*h[0], jb*h[1], (jb+1)*h[1], phiMin, phiMax);
    if (phiMin < band && phiMax > -band)
      atomicMax(&grid.bFlagsList[bIdx], (lvl < grid.nLvls-1) ? REFINE : KEEP);
    // no vote outside the band: the flow indicator owns the rest of the grid
  }
}

// leaf element containing the physical point (leaves tile the domain exactly
// once; query the hash finest -> coarsest with the corpse-safe lookup)
__device__ i32 dgIbLocateLeaf(DgSolver &grid, real x, real y, real z,
                              i32 &lvlOut, i32 &ibOut, i32 &jbOut, i32 &kbOut) {
  for (i32 lvl = grid.nLvls-1; lvl >= 0; lvl--) {
    real h[3]; dgElemSize(grid, lvl, h);
    i32 ib = (i32)floor(x/h[0]);
    i32 jb = (i32)floor(y/h[1]);
    i32 kb = grid.pseudo2D ? 0 : (i32)floor(z/h[2]);
    if (grid.isExteriorBlock(lvl, ib, jb, kb)) continue;
    i32 idx = grid.getBlockIdx(grid.encode(lvl, ib, jb, kb));
    if (idx != bEmpty) { lvlOut = lvl; ibOut = ib; jbOut = jb; kbOut = kb; return idx; }
  }
  return bEmpty;
}

// first and second derivatives of the Lagrange basis l_a at x (c_xi nodes)
__device__ real dgBasisD1At(i32 a, real x) {
  real acc = 0.0;
  for (i32 m = 0; m < NNODE; m++) {
    if (m == a) continue;
    real v = (real)1.0/(c_xi[a] - c_xi[m]);
    for (i32 n = 0; n < NNODE; n++)
      if (n != a && n != m) v *= (x - c_xi[n])/(c_xi[a] - c_xi[n]);
    acc += v;
  }
  return acc;
}
__device__ real dgBasisD2At(i32 a, real x) {
  real acc = 0.0;
  for (i32 m = 0; m < NNODE; m++) {
    if (m == a) continue;
    for (i32 n = 0; n < NNODE; n++) {
      if (n == a || n == m) continue;
      real v = (real)1.0/((c_xi[a] - c_xi[m])*(c_xi[a] - c_xi[n]));
      for (i32 r = 0; r < NNODE; r++)
        if (r != a && r != m && r != n) v *= (x - c_xi[r])/(c_xi[a] - c_xi[r]);
      acc += v;   // ordered (m,n) pairs: exactly d2/dx2 of the product
    }
  }
  return acc;
}

// evaluate the donor element's polynomial at local coords zeta[3]: value F,
// first F1 and second F2 NORMAL-directional derivatives, of the sanitized
// primitives rotated into the wall frame (rho, v_n, v_t1, v_t2, p).
// Rotation is constant per ghost node, so rotate-then-interpolate is exact.
__device__ void dgIbDonorEval(DgSolver &grid, i32 dIdx, const real hd[3],
                              const real zeta[3], const real n[3],
                              real F[5], real F1[5], real F2[5]) {
  real Lx[NNODE], Dx[NNODE], Cx[NNODE];
  real Ly[NNODE], Dy[NNODE], Cy[NNODE], Lz[NNODE];
  for (i32 a = 0; a < NNODE; a++) {
    Lx[a] = dgBasisAt(a, zeta[0]); Dx[a] = dgBasisD1At(a, zeta[0]); Cx[a] = dgBasisD2At(a, zeta[0]);
    Ly[a] = dgBasisAt(a, zeta[1]); Dy[a] = dgBasisD1At(a, zeta[1]); Cy[a] = dgBasisD2At(a, zeta[1]);
    Lz[a] = grid.pseudo2D ? (real)0.0 : dgBasisAt(a, zeta[2]);
  }
  if (grid.ibFilt) {   // read the donor through the top-mode projection:
    // w~ = Filt^T w per axis (derivatives of the filtered polynomial too)
    real t0[NNODE], t1[NNODE], t2[NNODE];
    for (i32 b = 0; b < NNODE; b++) {
      real a0 = 0, a1 = 0, a2 = 0;
      for (i32 a = 0; a < NNODE; a++) {
        a0 += Lx[a]*c_IbFilt[a][b]; a1 += Dx[a]*c_IbFilt[a][b]; a2 += Cx[a]*c_IbFilt[a][b];
      }
      t0[b] = a0; t1[b] = a1; t2[b] = a2;
    }
    for (i32 b = 0; b < NNODE; b++) { Lx[b] = t0[b]; Dx[b] = t1[b]; Cx[b] = t2[b]; }
    for (i32 b = 0; b < NNODE; b++) {
      real a0 = 0, a1 = 0, a2 = 0;
      for (i32 a = 0; a < NNODE; a++) {
        a0 += Ly[a]*c_IbFilt[a][b]; a1 += Dy[a]*c_IbFilt[a][b]; a2 += Cy[a]*c_IbFilt[a][b];
      }
      t0[b] = a0; t1[b] = a1; t2[b] = a2;
    }
    for (i32 b = 0; b < NNODE; b++) { Ly[b] = t0[b]; Dy[b] = t1[b]; Cy[b] = t2[b]; }
  }
  const real gx = (real)2.0/hd[0], gy = (real)2.0/hd[1];
  for (i32 q = 0; q < 5; q++) { F[q] = F1[q] = F2[q] = 0.0; }

  // pseudo2D: the z direction carries a single (constant-in-z) node layer at
  // k = 0..NNODE-1 all holding the same 2D state; use the k=0 plane directly
  const i32 kmax = grid.pseudo2D ? 1 : NNODE;
  for (i32 c = 0; c < kmax; c++)
    for (i32 b = 0; b < NNODE; b++)
      for (i32 a = 0; a < NNODE; a++) {
        real lz = grid.pseudo2D ? (real)1.0 : Lz[c];
        real wv = Lx[a]*Ly[b]*lz;
        real w1 = (n[0]*gx*Dx[a]*Ly[b] + n[1]*gy*Lx[a]*Dy[b])*lz;
        real w2 = (n[0]*n[0]*gx*gx*Cx[a]*Ly[b] + n[1]*n[1]*gy*gy*Lx[a]*Cy[b]
                   + (real)2.0*n[0]*n[1]*gx*gy*Dx[a]*Dy[b])*lz;
        i32 nd = a + NNODE*(b + NNODE*c);
        real U[5], Wp[5];
        for (i32 q = 0; q < 5; q++)
          U[q] = grid.getField(D_RHO+q)[(u64)dIdx*blockSizeTot + nd];
        dgConsToPrimSane(U, Wp);
        real pw[5] = { Wp[0],
                       Wp[1]*n[0] + Wp[2]*n[1],       // v_n
                      -Wp[1]*n[1] + Wp[2]*n[0],       // v_t1
                       Wp[3],                          // v_t2 (= w; cylinder axis)
                       Wp[4] };
        for (i32 q = 0; q < 5; q++) {
          F [q] += wv*pw[q];
          F1[q] += w1*pw[q];
          F2[q] += w2*pw[q];
        }
      }
}

// wall-normal Hermite in the normalized coordinate sigma = s/d_i (wall at 0,
// image at 1): f = b0 + b1 s + b2 s^2 + b3 s^3.  Conditions: value/1st/2nd
// derivative at sigma=1 (F, hF1 = F1*d_i, hF2 = F2*d_i^2) plus the wall BC at
// sigma=0 -- Dirichlet f(0)=bc, or Neumann f'(0)=bc (bc pre-scaled by d_i).
// All coefficients are O(field): no 1/d^k term survives (fp32-safe).
__device__ __forceinline__ real dgIbHermite(i32 dirichlet, real bc,
    real F, real hF1, real hF2, real sigma, i32 order) {
  // The ghost element holds the MIRROR WORLD about the wall: every node --
  // including the fluid-side nodes of cut elements, which are exactly the
  // face nodes its fluid neighbor reads -- carries the BC-reflected state,
  //   Dirichlet f0:  ghost = 2 f0 - f(|sigma|)     (odd about the wall value)
  //   Neumann  g0:   ghost = f(|sigma|) - 2|sigma| hg0   (even, slope-corrected)
  // with the Hermite polynomial always evaluated INSIDE its data interval
  // [0,1] (|sigma| <= 1 by the mirror-with-floor rule).  Two measured failure
  // modes forced this form: (a) filling fluid-side face nodes with the plain
  // interpolant f(sigma) makes the wall TRANSPARENT (p_stag 0.80 vs pitot 8.6
  // at M=3 -- HLLC transmits into the non-conservative ghost instead of
  // reflecting; pinned-freestream ghosts give exactly p_inf, so the faces do
  // couple); (b) raw polynomial extrapolation at sigma < 0 is the order-3
  // rest-state instability (e-folding ~0.06) and gives +9-instead-of-+3
  // startup jets at order 2.
  real a = fabs(sigma), v;
  if (order <= 1)
    v = dirichlet ? (bc + (F - bc)*a) : (F + bc*(a - (real)1.0));
  else {
    real b0, b1, b2, b3 = 0.0;
    if (dirichlet) {
      real A = F - bc;
      b0 = bc;
      if (order >= 3) { b3 = (real)0.5*hF2 - hF1 + A; b2 = hF1 - A - (real)2.0*b3; }
      else            { b2 = hF1 - A; }
      b1 = A - b2 - b3;
    } else {
      b1 = bc;
      if (order >= 3) { b3 = (bc + hF2 - hF1)*(real)(1.0/3.0); b2 = (real)0.5*(hF2 - (real)6.0*b3); }
      else            { b2 = (real)0.5*(hF1 - bc); }
      b0 = F - bc - b2 - b3;
    }
    v = b0 + a*(b1 + a*(b2 + a*b3));
  }
  return dirichlet ? ((real)2.0*bc - v) : (v - (real)2.0*a*bc);
}

// the ghost fill: one thread per ghost node
__global__ void dgIbFillKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.ibClassList[bIdx] != IB_GHOST) continue;
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    GET_CELL_INDICES
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    real x = dgNodePos(h[0], ib, i);
    real y = dgNodePos(h[1], jb, j);
    real z = dgNodePos(h[2], kb, k);

    real dxc = x - grid.ibX, dyc = y - grid.ibY;
    real r  = fmax(sqrt(dxc*dxc + dyc*dyc), (real)1e-12*grid.ibR);
    real n[3] = { dxc/r, dyc/r, (real)0.0 };  // outward wall normal
    real sg = r - grid.ibR;                   // signed distance (< 0 inside)
    real xb = x - sg*n[0], yb = y - sg*n[1];  // boundary foot

    // image point: mirror with a floor, pushed out until its leaf is FLUID
    real di = fmax(fabs(sg), grid.ibImageFac*h[0]);
    i32 dIdx = bEmpty, dl = 0, dib = 0, djb = 0, dkb = 0;
    for (i32 t = 0; t < 8; t++) {
      i32 idx = dgIbLocateLeaf(grid, xb + di*n[0], yb + di*n[1], z, dl, dib, djb, dkb);
      if (idx != bEmpty && grid.ibClassList[idx] == IB_FLUID) { dIdx = idx; break; }
      di += (real)0.5*h[0];
    }
    if (dIdx == bEmpty) {   // geometrically (near-)impossible for a convex body:
      atomicAdd(&grid.ibCnt[IB_CNT_NODONOR], 1);   // keep previous values, count it
      continue;
    }
    real xi = xb + di*n[0], yi = yb + di*n[1];

    real hd[3]; dgElemSize(grid, dl, hd);
    real zeta[3] = { (real)2.0*(xi/hd[0] - dib) - (real)1.0,
                     (real)2.0*(yi/hd[1] - djb) - (real)1.0,
                     grid.pseudo2D ? (real)0.0
                                   : (real)2.0*(z/hd[2] - dkb) - (real)1.0 };

    i32 ord = grid.ibOrder;
    if (ord == 0) {   // diagnostic mode: pin ghosts to the freestream state
      real Wc[5] = { (real)1.0, grid.machInf, (real)0.0, (real)0.0, (real)1.0/dgGam };
      real Uc[5];
      dgP2C(Wc, Uc);
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = Uc[q];
      continue;
    }
    if (ord > 2 &&
        grid.getField(D_SCRATCH)[(u64)dIdx*blockSizeTot + 1] > grid.ibShockTheta) {
      ord = 2;   // shocked donor: its 2nd derivatives are oscillation, drop them
      atomicAdd(&grid.ibCnt[IB_CNT_FALLBACK], 1);
    }

    real F[5], F1[5], F2[5];
    dgIbDonorEval(grid, dIdx, hd, zeta, n, F, F1, F2);

    real sigma = sg/di;   // in [-1, 1] by the mirror-with-floor construction
    real d2 = di*di;
    // curvature-consistent wall pressure gradient dp/dn = rho v_t^2 / R
    real g0p = grid.ibCurv ? F[0]*(F[2]*F[2] + F[3]*F[3])/grid.ibR : (real)0.0;

    // diagnostic sub-modes (--ibord): 4 = order 1 with v_n hard-zeroed;
    // 5 = order 1 with rho/vt/p pinned to freestream (v_n reconstructed);
    // used to bisect the reconstruction feedback channel by channel
    i32 dbgMode = ord;
    if (ord >= 4) ord = 1;
    real Wf[5];
    for (i32 pass = 0; pass < 2; pass++) {
      Wf[0] = dgIbHermite(0, (real)0.0, F[0], F1[0]*di, F2[0]*d2, sigma, ord); // rho: dn = 0
      Wf[1] = dgIbHermite(1, (real)0.0, F[1], F1[1]*di, F2[1]*d2, sigma, ord); // v_n = 0
      Wf[2] = dgIbHermite(0, (real)0.0, F[2], F1[2]*di, F2[2]*d2, sigma, ord); // v_t1: dn = 0
      Wf[3] = dgIbHermite(0, (real)0.0, F[3], F1[3]*di, F2[3]*d2, sigma, ord); // v_t2: dn = 0
      Wf[4] = dgIbHermite(0, g0p*di,    F[4], F1[4]*di, F2[4]*d2, sigma, ord); // p: curvature
      if (dbgMode == 4) Wf[1] = (real)0.0;
      if (dbgMode == 5) { Wf[0] = (real)1.0; Wf[2] = Wf[3] = (real)0.0; Wf[4] = (real)1.0/dgGam; }
      if (Wf[0] > DG_EPSF && Wf[4] > DG_EPSF) break;
      ord = 1;   // inadmissible: retry linear (monotone between BC and image)
      atomicAdd(&grid.ibCnt[IB_CNT_RETRY1], 1);
    }

    real W[5] = { Wf[0],
                  Wf[1]*n[0] - Wf[2]*n[1],
                  Wf[1]*n[1] + Wf[2]*n[0],
                  Wf[3],
                  Wf[4] };
    dgSanitizePrim(W);
    real U[5];
    dgP2C(W, U);
    for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
  }
}

// --debug audit: (i) no FLUID face may resolve to a DEAD element (the
// classification invariant); (ii) no sub-finest octet may mix GHOST and FLUID
// members (the band guarantee that protects the merge machinery)
__global__ void dgIbCheckKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || grid.ibClassList[bIdx] != IB_FLUID) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    for (i32 dir = 0; dir < (grid.pseudo2D ? 2 : 3); dir++)
      for (i32 side = 0; side < 2; side++) {
        i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
        i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
        i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);
        if (grid.isExteriorBlock(lvl, nib, njb, nkb)) continue;
        i32 nIdx = grid.getBlockIdx(grid.encode(lvl, nib, njb, nkb));
        if (nIdx == bEmpty && lvl > 0)
          nIdx = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                              grid.pseudo2D ? nkb : (nkb>>1)));
        if (nIdx != bEmpty && grid.ibClassList[nIdx] == IB_DEAD) {
          printf("[ibclass] FLUID lvl %d elem (%d,%d,%d) faces DEAD (dir %d side %d)\n",
                 lvl, ib, jb, kb, dir, side);
          atomicAdd(grid.dbgCnt, 1);
        }
        // finer children: any DEAD child covering my face is a violation too
        if (nIdx == bEmpty && lvl < grid.nLvls-1) {
          const i32 t1b = (dir==0) ? njb : nib;
          const i32 t2b = (dir==2) ? njb : nkb;
          const i32 s2max = (grid.pseudo2D && dir != 2) ? 1 : 2;
          for (i32 s2 = 0; s2 < s2max; s2++)
            for (i32 s1 = 0; s1 < 2; s1++) {
              i32 cib, cjb, ckb;
              if (dir == 0) { cib = 2*nib + (side ? 0 : 1); cjb = 2*t1b + s1;
                              ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else if (dir == 1) { cjb = 2*njb + (side ? 0 : 1); cib = 2*t1b + s1;
                                   ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else { ckb = 2*nkb + (side ? 0 : 1); cib = 2*t1b + s1; cjb = 2*t2b + s2; }
              i32 cIdxN = grid.getBlockIdx(grid.encode(lvl+1, cib, cjb, ckb));
              if (cIdxN != bEmpty && grid.ibClassList[cIdxN] == IB_DEAD) {
                printf("[ibclass] FLUID lvl %d elem (%d,%d,%d) faces DEAD child\n",
                       lvl, ib, jb, kb);
                atomicAdd(grid.dbgCnt, 1);
              }
            }
        }
      }

    // (ii) sub-finest octet mixing GHOST with my FLUID
    if (lvl > 0 && lvl < grid.nLvls-1) {
      i32 aib = ib & ~1, ajb = jb & ~1, akb = grid.pseudo2D ? kb : (kb & ~1);
      i32 sIdx[8];
      if (dgGatherOctet(grid, lvl, aib, ajb, akb, sIdx)) {
        i32 nS = grid.pseudo2D ? 4 : 8;
        for (i32 sm = 0; sm < nS; sm++)
          if (grid.ibClassList[sIdx[sm]] == IB_GHOST) {
            printf("[ibclass] sub-finest octet mixes GHOST+FLUID at lvl %d (%d,%d,%d)\n",
                   lvl, aib, ajb, akb);
            atomicAdd(grid.dbgCnt, 1);
            break;
          }
      }
    }
  }
}

// sample [x, p, rho] at nS points along the stagnation line y = ibY, from the
// inflow to the cylinder nose (bow-shock standoff extraction)
__global__ void dgIbStagLineKernel(DgSolver &grid, i32 nS, real *out) {
  i32 t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= nS) return;
  real xNose = grid.ibX - grid.ibR;
  real x = (t + (real)0.5)*xNose/(real)nS;
  real y = grid.ibY;
  real z = (real)0.5*grid.domainSize[2];
  out[3*t] = x; out[3*t+1] = -1.0; out[3*t+2] = 0.0;

  i32 dl, dib, djb, dkb;
  i32 idx = dgIbLocateLeaf(grid, x, y, z, dl, dib, djb, dkb);
  if (idx == bEmpty || grid.ibClassList[idx] != IB_FLUID) return;
  real hd[3]; dgElemSize(grid, dl, hd);
  real zeta[3] = { (real)2.0*(x/hd[0] - dib) - (real)1.0,
                   (real)2.0*(y/hd[1] - djb) - (real)1.0,
                   grid.pseudo2D ? (real)0.0
                                 : (real)2.0*(z/hd[2] - dkb) - (real)1.0 };
  real n[3] = { (real)1.0, (real)0.0, (real)0.0 };   // any unit vector: value only
  real F[5], F1[5], F2[5];
  dgIbDonorEval(grid, idx, hd, zeta, n, F, F1, F2);
  out[3*t+1] = F[4];
  out[3*t+2] = F[0];
}

// debug paint: stage the class map into SCRATCH (0 fluid / 1 ghost / 2 dead)
__global__ void dgIbClassToScratchKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    grid.getField(D_SCRATCH)[cIdx] = (real)grid.ibClassList[bIdx];
  }
}

// sample p and |v_t| at nTheta points on the circle r = ibR + off (one thread
// per sample; out rows: [theta, p, vt]).  Used for Cp(theta), p_stag, Cd.
__global__ void dgIbSurfaceKernel(DgSolver &grid, i32 nTheta, real off, real *out) {
  i32 t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= nTheta) return;
  real th = (real)2.0*(real)PI*(t + (real)0.5)/(real)nTheta;
  real n[3] = { cos(th), sin(th), (real)0.0 };
  real x = grid.ibX + (grid.ibR + off)*n[0];
  real y = grid.ibY + (grid.ibR + off)*n[1];
  real z = (real)0.5*grid.domainSize[2];
  out[3*t] = th; out[3*t+1] = -1.0; out[3*t+2] = 0.0;

  i32 dl, dib, djb, dkb;
  i32 idx = dgIbLocateLeaf(grid, x, y, z, dl, dib, djb, dkb);
  if (idx == bEmpty) return;
  real hd[3]; dgElemSize(grid, dl, hd);
  real zeta[3] = { (real)2.0*(x/hd[0] - dib) - (real)1.0,
                   (real)2.0*(y/hd[1] - djb) - (real)1.0,
                   grid.pseudo2D ? (real)0.0
                                 : (real)2.0*(z/hd[2] - dkb) - (real)1.0 };
  real F[5], F1[5], F2[5];
  dgIbDonorEval(grid, idx, hd, zeta, n, F, F1, F2);
  out[3*t+1] = F[4];                              // pressure
  out[3*t+2] = sqrt(F[2]*F[2] + F[3]*F[3]);       // tangential speed
}


// static-grid vote overrides: a fixed target level from geometry
__global__ void dgStaticVoteKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    real h[3]; dgElemSize(grid, lvl, h);
    real cx = grid.domainSize[0]*(real)0.5, cy = grid.domainSize[1]*(real)0.5;
    real cz = grid.domainSize[2]*(real)0.5;
    real x = (ib + (real)0.5)*h[0], y = (jb + (real)0.5)*h[1], z = (kb + (real)0.5)*h[2];

    i32 target = 0;
    switch (grid.staticGrid) {
      case 1: {   // fine sphere about the center
        real dx = x-cx, dy = y-cy, dz = grid.pseudo2D ? (real)0.0 : (z-cz);
        target = (sqrt(dx*dx+dy*dy+dz*dz) < grid.refineRadius) ? grid.nLvls-1 : 0;
      } break;
      case 2: target = grid.nLvls-1; break;   // forced uniform fine
      case 3: target = 0; break;              // forced collapse to base
      case 4:                                  // planar x-band
        target = (fabs(x-cx) < grid.refineRadius) ? grid.nLvls-1 : 0;
        break;
    }
    i32 vote = (lvl < target) ? REFINE : ((lvl == target) ? KEEP : DELETE);
    atomicMax(&grid.bFlagsList[bIdx], vote);
    if (lvl == 0) atomicMax(&grid.bFlagsList[bIdx], KEEP);
  }
}

// shock-driven refinement: any element whose Ducros compression sensor exceeds
// `thresh` is forced to REFINE, so shocks climb to the finest level (one level
// per adaptation) and never sit on a coarse element adjacent to a finer one.
// Same sensor the RHS uses for artificial viscosity; recomputed here on the
// current solution (the RHS value is transient).  Shared-memory per-element,
// EPB elements per CUDA block (launched with DG_EPB*blockSizeTot threads).
__global__ void dgShockRefineKernel(DgSolver &grid, real thresh) {
  __shared__ real sV [DG_EPB][3][blockSizeTot];   // u, v, w
  __shared__ real sC2[DG_EPB][blockSizeTot];      // sound speed^2
  __shared__ real sTh[DG_EPB][blockSizeTot];      // per-node sensor

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    const bool active = (loc != kEmpty);
    i32 lvl = 0, ib = 0, jb = 0, kb = 0;
    if (active) grid.decode(loc, lvl, ib, jb, kb);
    real h[3] = {1,1,1};
    if (active) dgElemSize(grid, lvl, h);

    if (active) {
      real U[5], W[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      sV[ell][0][nd] = W[1]; sV[ell][1][nd] = W[2]; sV[ell][2][nd] = W[3];
      sC2[ell][nd] = dgGam*W[4]/fmax(W[0], (real)1e-12);
    }
    __syncthreads();

    real th = 0.0;
    if (active) {
      real jacx = (real)2.0/h[0], jacy = (real)2.0/h[1], jacz = (real)2.0/h[2];
      real lenp = h[0]/(real)(2*dgOrder+1);
      i32 ndX0 = j*NNODE + k*NNODE*NNODE, ndY0 = i + k*NNODE*NNODE, ndZ0 = i + j*NNODE;
      real du = 0, dv = 0, dw = 0;
      for (i32 m = 0; m < NNODE; m++) {
        du += c_D[i][m]*sV[ell][0][ndX0 + m];
        dv += c_D[j][m]*sV[ell][1][ndY0 + m*NNODE];
        if (!grid.pseudo2D) dw += c_D[k][m]*sV[ell][2][ndZ0 + m*NNODE*NNODE];
      }
      real divu = fmin(jacx*du + jacy*dv + jacz*dw, (real)0.0);
      real du2 = divu*divu;
      th = du2/(du2 + grid.avKsensor*sC2[ell][nd]/(lenp*lenp) + (real)1e-30);
    }
    sTh[ell][nd] = th;
    __syncthreads();

    if (active && nd == 0 && lvl < grid.nLvls-1 &&
        grid.ibClassList[bIdx] == IB_FLUID) {
      real thmax = 0;
      for (i32 m = 0; m < blockSizeTot; m++) thmax = fmax(thmax, sTh[ell][m]);
      if (thmax > thresh) atomicMax(&grid.bFlagsList[bIdx], REFINE);
    }
    __syncthreads();
  }
}

// Harten rule 2: neighbors of significant leaves stay
__global__ void dgNeighborRuleKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    if (grid.bFlagsList[bIdx] < KEEP) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 dkLim = grid.pseudo2D ? 0 : 1;
    for (i32 dk = -dkLim; dk <= dkLim; dk++)
      for (i32 dj = -1; dj <= 1; dj++)
        for (i32 di = -1; di <= 1; di++) {
          if (di == 0 && dj == 0 && dk == 0) continue;
          i32 ni = ib+di, nj = jb+dj, nk = kb+dk;
          if (grid.periodic) grid.wrapBlockPeriodic(lvl, ni, nj, nk);
          if (grid.isExteriorBlock(lvl, ni, nj, nk)) continue;
          i32 nIdx = grid.getBlockIdx(grid.encode(lvl, ni, nj, nk));
          if (nIdx != bEmpty) { atomicMax(&grid.bFlagsList[nIdx], KEEP); continue; }
          if (lvl > 0) {   // covered by a coarser leaf: keep it too
            i32 cIdxN = grid.getBlockIdx(grid.encode(lvl-1, ni>>1, nj>>1,
                                                     grid.pseudo2D ? nk : (nk>>1)));
            if (cIdxN != bEmpty) atomicMax(&grid.bFlagsList[cIdxN], KEEP);
          }
        }
  }
}

// snapshot the current REFINE set into snapValidList (the anchor-complete
// markers there were already consumed by dgVoteKernel and are rewritten by the
// merge phase).  Reading the snapshot -- not the live flags -- bounds the
// buffer to exactly ONE ring: without it a freshly-buffered neighbor would
// re-propagate and refinement would spread across the whole domain.
__global__ void dgSnapshotRefineKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    grid.snapValidList[bIdx] =
        (loc != kEmpty && grid.bFlagsList[bIdx] == REFINE) ? 1 : 0;
  }
}

// every same-level neighbor of a (snapshot) REFINE element is itself flagged
// REFINE, giving the fine region a one-element buffer at the fine level.  A
// coarser leaf covering a missing same-level slot is left to the grading pass,
// which already promotes the coarser neighbor of a REFINE element to REFINE.
__global__ void dgRefineBufferKernel(DgSolver &grid, real epsL) {
  // A feature occupies only part of a refined octet and advects toward the side
  // it sits on.  So instead of the full 26-neighbor ring, each refined child
  // buffers only toward the half(s) of ITSELF where the detail actually lives.
  // The detail is recomputed here exactly as the indicator did it -- d = u_child
  // - prolong(anchor restriction) -- and binned on the - / + half of each axis:
  //   a child whose own detail is insignificant buffers nothing; otherwise one
  //   dominant side -> that face (+ shared corners); both sides -> feature
  //   crosses, extend both; an axis with negligible detail -> don't extend.
  const real biasRatio = 2.0;    // side dominant if its energy > biasRatio * other
  const real featFrac  = 0.20;   // axis carries the detail if its energy > featFrac*maxAxis
  DG_BLOCK_LOOP(bIdx) {
    if (!grid.snapValidList[bIdx]) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);
    if (lvl >= grid.nLvls-1) continue;   // finest leaves are never voted REFINE

    i32 lo[3], hi[3];
    if (grid.refineBuffer >= 2) {              // mode 2: full 26-neighbor ring
      for (i32 a = 0; a < 3; a++) {
        lo[a] = (a == 2 && grid.pseudo2D) ? 0 : -1;
        hi[a] = (a == 2 && grid.pseudo2D) ? 0 :  1;
      }
    } else if (grid.indicator != 0) {
      // non-MRA indicators: the anchor Q0 restriction is not populated (it
      // holds the stale RK register), so derive the direction from the
      // density-gradient energy on the -/+ half of each axis instead
      const real *R = grid.getField(D_RHO) + (u64)bIdx*blockSizeTot;
      real eLo[3] = {0,0,0}, eHi[3] = {0,0,0};
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);
        real gx = 0, gy = 0, gz = 0;
        for (i32 m = 0; m < NNODE; m++) {
          gx += c_D[i][m]*R[m + j*NNODE + k*NNODE*NNODE];
          gy += c_D[j][m]*R[i + m*NNODE + k*NNODE*NNODE];
          if (!grid.pseudo2D) gz += c_D[k][m]*R[i + j*NNODE + m*NNODE*NNODE];
        }
        (i < NNODE/2 ? eLo[0] : eHi[0]) += gx*gx;
        (j < NNODE/2 ? eLo[1] : eHi[1]) += gy*gy;
        if (!grid.pseudo2D) (k < NNODE/2 ? eLo[2] : eHi[2]) += gz*gz;
      }
      real maxAxis = 1e-30;
      for (i32 a = 0; a < 3; a++) maxAxis = fmax(maxAxis, eLo[a] + eHi[a]);
      for (i32 a = 0; a < 3; a++) {
        if (a == 2 && grid.pseudo2D)                    { lo[a] = 0;  hi[a] = 0; }
        else if (eLo[a] + eHi[a] < featFrac*maxAxis)    { lo[a] = 0;  hi[a] = 0; }
        else if (eHi[a] > biasRatio*eLo[a])             { lo[a] = 0;  hi[a] = 1; }
        else if (eLo[a] > biasRatio*eHi[a])             { lo[a] = -1; hi[a] = 0; }
        else                                            { lo[a] = -1; hi[a] = 1; }
      }
    } else {
      // recompute this child's detail from its octet anchor's restriction (Q0)
      i32 aib = ib & ~1, ajb = jb & ~1, akb = grid.pseudo2D ? kb : (kb & ~1);
      i32 aIdx = grid.getBlockIdx(grid.encode(lvl, aib, ajb, akb));
      if (aIdx == bEmpty) continue;   // a vote-REFINE child always has its anchor
      i32 s1 = ib & 1, s2 = jb & 1, s3 = grid.pseudo2D ? 0 : (kb & 1);

      real eLo[3] = {0,0,0}, eHi[3] = {0,0,0}, wSum = 0;
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);
        real d2 = 0;
        for (i32 q = 0; q < 5; q++) {
          const real *uLq = grid.getField(D_Q0+q) + (u64)aIdx*blockSizeTot;
          real P  = dgProlongNode(grid, uLq, s1, s2, s3, i, j, k);
          real dn = (grid.getField(D_RHO+q)[(u64)bIdx*blockSizeTot + nd] - P)/grid.cScale[q];
          d2 += dn*dn;
        }
        wSum += (real)0.125*c_w[i]*c_w[j]*c_w[k]*d2;   // GLL-weighted, for the gate
        (i < NNODE/2 ? eLo[0] : eHi[0]) += d2;
        (j < NNODE/2 ? eLo[1] : eHi[1]) += d2;
        if (!grid.pseudo2D) (k < NNODE/2 ? eLo[2] : eHi[2]) += d2;
      }

      // this child carries no significant detail of its own -> don't buffer
      real eps = epsL/(real)powi(2, grid.nLvls - lvl);
      if (sqrt(fmax(wSum, (real)0.0)) <= grid.refineFac*eps) continue;

      real maxAxis = 1e-30;
      for (i32 a = 0; a < 3; a++) maxAxis = fmax(maxAxis, eLo[a] + eHi[a]);
      for (i32 a = 0; a < 3; a++) {
        if (a == 2 && grid.pseudo2D) {
          lo[a] = 0; hi[a] = 0;
        } else if (eLo[a] + eHi[a] < featFrac*maxAxis) {
          lo[a] = 0; hi[a] = 0;                        // no detail on this axis
        } else if (eHi[a] > biasRatio*eLo[a]) {
          lo[a] = 0; hi[a] = 1;                        // detail on the + half
        } else if (eLo[a] > biasRatio*eHi[a]) {
          lo[a] = -1; hi[a] = 0;                       // detail on the - half
        } else {
          lo[a] = -1; hi[a] = 1;                       // detail crosses the element
        }
      }
    }

    for (i32 dk = lo[2]; dk <= hi[2]; dk++)
      for (i32 dj = lo[1]; dj <= hi[1]; dj++)
        for (i32 di = lo[0]; di <= hi[0]; di++) {
          if (di == 0 && dj == 0 && dk == 0) continue;
          i32 ni = ib+di, nj = jb+dj, nk = kb+dk;
          if (grid.periodic) grid.wrapBlockPeriodic(lvl, ni, nj, nk);
          if (grid.isExteriorBlock(lvl, ni, nj, nk)) continue;
          i32 nIdx = grid.getBlockIdx(grid.encode(lvl, ni, nj, nk));
          if (nIdx != bEmpty) atomicMax(&grid.bFlagsList[nIdx], REFINE);
        }
  }
}

// one pass of 2:1 grading on TARGET levels (R => lvl+1, K => lvl, C => lvl-1):
// raise neighbors whose targets would differ by more than one level.
__global__ void dgEnforceGradingKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 flag = grid.bFlagsList[bIdx];
    if (flag < KEEP) continue;   // DELETE targets lvl-1: never forces anyone
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 dkLim = grid.pseudo2D ? 0 : 1;
    for (i32 dk = -dkLim; dk <= dkLim; dk++)
      for (i32 dj = -1; dj <= 1; dj++)
        for (i32 di = -1; di <= 1; di++) {
          if (di == 0 && dj == 0 && dk == 0) continue;
          i32 ni = ib+di, nj = jb+dj, nk = kb+dk;
          if (grid.periodic) grid.wrapBlockPeriodic(lvl, ni, nj, nk);
          if (grid.isExteriorBlock(lvl, ni, nj, nk)) continue;

          i32 nIdx = grid.getBlockIdx(grid.encode(lvl, ni, nj, nk));
          if (nIdx != bEmpty) {
            // same-level neighbor: REFINE (target lvl+1) forbids its DELETE (lvl-1)
            if (flag == REFINE) {
              i32 old = atomicMax(&grid.bFlagsList[nIdx], KEEP);
              if (old < KEEP) atomicAdd(grid.chgCnt, 1);
            }
            continue;
          }
          if (lvl == 0) continue;
          i32 cIdxN = grid.getBlockIdx(grid.encode(lvl-1, ni>>1, nj>>1,
                                                   grid.pseudo2D ? nk : (nk>>1)));
          if (cIdxN == bEmpty) continue;   // finer neighbors: their rules handle us
          if (flag == REFINE) {
            // my target lvl+1 vs coarser leaf: needs target >= lvl  => REFINE it
            i32 old = atomicMax(&grid.bFlagsList[cIdxN], REFINE);
            if (old < REFINE) atomicAdd(grid.chgCnt, 1);
          } else {   // KEEP: my target lvl vs coarser: needs target >= lvl-1 => KEEP it
            i32 old = atomicMax(&grid.bFlagsList[cIdxN], KEEP);
            if (old < KEEP) atomicAdd(grid.chgCnt, 1);
          }
        }
  }
}

// merge phase 1: octet mergeability verdict from a consistent flag snapshot
__global__ void dgMergeVerdictKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    grid.snapValidList[bIdx] = 0;
    if (grid.bFlagsList[bIdx] != DELETE) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (lvl == 0) continue;

    i32 sIdx[8];
    bool ok = dgGatherOctet(grid, lvl, ib & ~1, jb & ~1,
                            grid.pseudo2D ? kb : (kb & ~1), sIdx);
    if (ok) {
      i32 nS = grid.pseudo2D ? 4 : 8;
      for (i32 s = 0; s < nS; s++)
        if (grid.bFlagsList[sIdx[s]] != DELETE) { ok = false; break; }
    }
    grid.snapValidList[bIdx] = ok ? 1 : 0;
  }
}

// merge phase 2: non-mergeable DELETE leaves must stay.  Each promotion RAISES
// the leaf's target level, which can newly violate 2:1 grading against a
// neighboring octet's pending merge -- the host loops [grade -> verdict ->
// apply] to a joint fixpoint, counting promotions in chgCnt.
__global__ void dgMergeApplyKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    if (grid.bFlagsList[bIdx] == DELETE && !grid.snapValidList[bIdx]) {
      grid.bFlagsList[bIdx] = KEEP;
      atomicAdd(grid.chgCnt, 1);
    }
  }
}

// spawn: refine children of REFINE leaves; merge parents of mergeable octets
__global__ void dgSpawnKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    i32 flag = grid.bFlagsList[bIdx];

    if (flag == REFINE && lvl < grid.nLvls-1) {
      i32 dkMax = grid.pseudo2D ? 0 : 1;
      for (i32 dk = 0; dk <= dkMax; dk++)
        for (i32 dj = 0; dj <= 1; dj++)
          for (i32 di = 0; di <= 1; di++)
            grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj,
                               grid.pseudo2D ? kb : 2*kb+dk);
    }
    else if (flag == DELETE && grid.snapValidList[bIdx] && lvl > 0 &&
             dgIsAnchor(grid, ib, jb, kb)) {
      grid.activateBlock(lvl-1, ib>>1, jb>>1, grid.pseudo2D ? kb : (kb>>1));
    }
  }
}

// fill NEW refine-children by exact tensor injection from their REFINE parent
__global__ void dgProlongChildrenKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || grid.bFlagsList[bIdx] != NEW) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (lvl == 0) continue;

    i32 pIdx = grid.getBlockIdx(grid.encode(lvl-1, ib>>1, jb>>1,
                                            grid.pseudo2D ? kb : (kb>>1)));
    if (pIdx == bEmpty || grid.bFlagsList[pIdx] != REFINE) continue;

    i32 s1 = ib & 1, s2 = jb & 1, s3 = grid.pseudo2D ? 0 : (kb & 1);
    for (i32 q = 0; q < 5; q++) {
      const real *Fp = grid.getField(D_RHO+q) + (u64)pIdx*blockSizeTot;
      grid.getField(D_RHO+q)[cIdx] = dgProlongNode(grid, Fp, s1, s2, s3, i, j, k);
    }
  }
}

__global__ void dgDemoteRefinedKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] != kEmpty && grid.bFlagsList[bIdx] == REFINE)
      grid.bFlagsList[bIdx] = DELETE;
  }
}

// fill NEW merge-parents by exact-L2 restriction of their (DELETE) octet
__global__ void dgRestrictParentsKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || grid.bFlagsList[bIdx] != NEW) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (lvl >= grid.nLvls-1) continue;

    i32 sIdx[8];
    if (!dgGatherOctet(grid, lvl+1, 2*ib, 2*jb, grid.pseudo2D ? kb : 2*kb, sIdx))
      continue;
    i32 nS = grid.pseudo2D ? 4 : 8;
    bool merge = true;
    for (i32 s = 0; s < nS; s++)
      if (grid.bFlagsList[sIdx[s]] != DELETE) { merge = false; break; }
    if (!merge) continue;

    for (i32 q = 0; q < 5; q++)
      grid.getField(D_RHO+q)[cIdx] = dgRestrictNode(grid, sIdx, q, i, j, k);
  }
}

// --debug: every face must resolve to exactly one of {same-level neighbor,
// coarser cover, complete set of fine children, domain exterior} -- a hole
// here means the RHS silently drops a face flux (Fs = 0) and blows up
__global__ void dgCheckFaceTopologyKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    for (i32 dir = 0; dir < (grid.pseudo2D ? 2 : 3); dir++)
      for (i32 side = 0; side < 2; side++) {
        i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
        i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
        i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);
        if (grid.isExteriorBlock(lvl, nib, njb, nkb)) {
          if (grid.bcType != 2) continue;          // weak BC face
          grid.wrapBlockPeriodic(lvl, nib, njb, nkb);
        }
        if (grid.getBlockIdx(grid.encode(lvl, nib, njb, nkb)) != bEmpty) continue;
        if (lvl > 0 &&
            grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                         grid.pseudo2D ? nkb : (nkb>>1))) != bEmpty)
          continue;
        // must be fully covered by children
        bool ok = (lvl < grid.nLvls-1);
        if (ok) {
          const i32 t1b = (dir==0) ? njb : nib;
          const i32 t2b = (dir==2) ? njb : nkb;
          const i32 s2max = (grid.pseudo2D && dir != 2) ? 1 : 2;
          for (i32 s2 = 0; s2 < s2max && ok; s2++)
            for (i32 s1 = 0; s1 < 2 && ok; s1++) {
              i32 cib, cjb, ckb;
              if (dir == 0) { cib = 2*nib + (side ? 0 : 1); cjb = 2*t1b + s1;
                              ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else if (dir == 1) { cjb = 2*njb + (side ? 0 : 1); cib = 2*t1b + s1;
                                   ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
              else { ckb = 2*nkb + (side ? 0 : 1); cib = 2*t1b + s1; cjb = 2*t2b + s2; }
              if (grid.getBlockIdx(grid.encode(lvl+1, cib, cjb, ckb)) == bEmpty) ok = false;
            }
        }
        if (!ok) {
          printf("[facetopo] HOLE: lvl %d elem (%d,%d,%d) face dir %d side %d has no neighbor\n",
                 lvl, ib, jb, kb, dir, side);
          atomicAdd(grid.dbgCnt, 1);
        }
      }
  }
}

// --debug: every leaf must have no live ancestor (leaves tile the domain once)
__global__ void dgCheckLeafCoverKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    i32 pi = ib, pj = jb, pk = kb;
    for (i32 l = lvl-1; l >= 0; l--) {
      pi >>= 1; pj >>= 1; if (!grid.pseudo2D) pk >>= 1;
      if (grid.getBlockIdx(grid.encode(l, pi, pj, pk)) != bEmpty) {
        printf("[leafcover] VIOLATION: leaf (%d: %d,%d,%d) has live ancestor at lvl %d\n",
               lvl, ib, jb, kb, l);
        atomicAdd(grid.dbgCnt, 1);
      }
    }
  }
}
