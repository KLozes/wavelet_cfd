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
// ── Gauss-Legendre flux-reconstruction operators (only meaningful when the
//    node set is Gauss; on Lobatto c_tL/c_tR are e_0/e_{N-1} and the g' vectors
//    reduce to the boundary 1/w lift) ──────────────────────────────────────
__constant__ real c_tL [NNODE];   // l_i(-1): interpolate a nodal field to the LEFT  face
__constant__ real c_tR [NNODE];   // l_i(+1): interpolate a nodal field to the RIGHT face
__constant__ real c_gpL[NNODE];   // g_L'(xi_i): left  correction-function derivative (FR lift)
__constant__ real c_gpR[NNODE];   // g_R'(xi_i): right correction-function derivative (FR lift)
__constant__ real c_ibLXi[NNODE]; // FRIB image-line node set: ALWAYS Lobatto (wall at
                                  // xi_0 = -1 exactly), independent of the element node
                                  // set -- on Gauss solution points the wall is not an
                                  // element node and the phi_i(-1)=delta_i0 wall-solve
                                  // shortcut only holds on the line's OWN LGL basis.
__constant__ real c_ibLD0[NNODE]; // that Lobatto line basis' derivative row at xi=-1
__constant__ real c_dpPhi[NNODE]; // orthonormal top Legendre mode P^_p(xi_i): the dual-
                                  // pairing upwind operator (D+ - D-) = -tau*phi*(H phi)^T
                                  // (arXiv 2411.06629 A.4).  sum_i w_i phi_i = 0 => the
                                  // volume upwind term is exactly conservative (selftest)

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

// Gauss-Legendre nodes / weights for p = 1..3 (NNODE = 2..4).  Interior points
// (none coincide with +-1), so every face trace is an interpolation -- the
// defining feature of Gauss flux reconstruction vs collocated Lobatto DGSEM.
static const double gs_xi_tab[3][4] = {
    /* p=1 */ {-0.5773502691896257,  0.5773502691896257},
    /* p=2 */ {-0.7745966692414834,  0.0,                0.7745966692414834},
    /* p=3 */ {-0.8611363115940526, -0.3399810435848563,
                0.3399810435848563,  0.8611363115940526},
};
static const double gs_w_tab[3][4] = {
    /* p=1 */ {1.0, 1.0},
    /* p=2 */ {0.5555555555555556, 0.8888888888888888, 0.5555555555555556},
    /* p=3 */ {0.3478548451374538, 0.6521451548625461,
               0.6521451548625461, 0.3478548451374538},
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
static double hTL[NNODE], hTR[NNODE];        // face interpolation l_i(-+1)
static double hgpL[NNODE], hgpR[NNODE];      // FR correction derivatives g_{L,R}'(xi_i)
static double hDpPhi[NNODE];                 // DP-SBP top orthonormal Legendre mode

// Legendre L_k and L_k' at x (Bonnet recurrences) -- selftest support for the
// Lobatto FR/DG equivalence identity
static void dgLegendreAt(int k, double x, double &L, double &Lp) {
  double L0 = 1.0, L1 = x, D0 = 0.0, D1 = 1.0;
  if (k <= 0) { L = L0; Lp = D0; return; }
  for (int j = 1; j < k; j++) {
    double L2 = ((2.0*j + 1.0)*x*L1 - j*L0)/(j + 1.0);
    double D2 = D0 + (2.0*j + 1.0)*L1;
    L0 = L1; L1 = L2; D0 = D1; D1 = D2;
  }
  L = L1; Lp = D1;
}

// Huynh left correction function g_HU (FRIB.pdf Eq 10; k = NNODE SPs):
//   g_HU(x) = ((-1)^k/2) [ ((k-1) L_k + k L_{k-2})/(2k-1) - L_{k-1} ]
// with g(-1) = 1, g(+1) = 0 (exact by L_j(+-1) = (+-1)^j).  Value and
// derivative from the recurrences.
static void dgHuynhG(double x, double &g, double &gp) {
  const int k = NNODE;
  double Lk, Lkp, Lm1, Lm1p, Lm2, Lm2p;
  dgLegendreAt(k,   x, Lk,  Lkp);
  dgLegendreAt(k-1, x, Lm1, Lm1p);
  dgLegendreAt(k-2, x, Lm2, Lm2p);
  double s = (k % 2) ? -0.5 : 0.5;   // ((-1)^k)/2
  g  = s*(((k - 1.0)*Lk  + k*Lm2 )/(2.0*k - 1.0) - Lm1);
  gp = s*(((k - 1.0)*Lkp + k*Lm2p)/(2.0*k - 1.0) - Lm1p);
}

// orthonormal Legendre P^_m(x) = P_m(x) * sqrt((2m+1)/2), m = 0..3
static double dgLegendreON(int m, double x) {
  double P = (m == 0) ? 1.0
           : (m == 1) ? x
           : (m == 2) ? 0.5*(3.0*x*x - 1.0)
                      : 0.5*(5.0*x*x*x - 3.0*x);
  return P * sqrt((2.0*m + 1.0)/2.0);
}

// host entropy variables v = dU/du for U = -rho s/(gam-1), s = ln p - gam ln rho
// (selftest round-trip; the device twins dgEntVars/dgEntVarsToPrim drive the
// Gauss FR surface).  Primitive I/O.
static void dgEntVarsHost(const double W[5], double v[5]) {
  double rho = W[0], p = W[4], q2 = W[1]*W[1]+W[2]*W[2]+W[3]*W[3];
  double s = log(p) - (double)dgGam*log(rho);
  v[0] = ((double)dgGam - s)/((double)dgGam-1.0) - rho*q2/(2.0*p);
  v[1] = rho*W[1]/p; v[2] = rho*W[2]/p; v[3] = rho*W[3]/p; v[4] = -rho/p;
}
static void dgEntVarsToPrimHost(const double v[5], double W[5]) {
  double g1 = (double)dgGam-1.0;
  double vv2 = v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
  double s = (double)dgGam - g1*(v[0] - vv2/(2.0*v[4]));
  double rho = pow(-v[4]*exp(s), -1.0/g1);
  double p = -rho/v[4];
  W[0]=rho; W[1]=v[1]/(-v[4]); W[2]=v[2]/(-v[4]); W[3]=v[3]/(-v[4]); W[4]=p;
}

static void dgBuildOperators(i32 gauss, i32 frType) {
  const double *xi = gauss ? gs_xi_tab[dgOrder-1] : lgl_xi_tab[dgOrder-1];
  const double *w  = gauss ? gs_w_tab [dgOrder-1] : gll_w_tab [dgOrder-1];
  for (int i = 0; i < NNODE; i++) { hXi[i] = xi[i]; hW[i] = w[i]; }

  dgBuildD(NNODE, xi, hD);

  // ── flux-reconstruction operators ─────────────────────────────────────
  // face interpolation vectors l_i(-+1): on Lobatto these are e_0 / e_{N-1}
  // (boundary nodes sit on the faces); on Gauss they are dense (extrapolation).
  for (int i = 0; i < NNODE; i++) {
    hTL[i] = dgLagrange(NNODE, xi, i, -1.0);
    hTR[i] = dgLagrange(NNODE, xi, i, +1.0);
  }
  // correction-function derivatives g_{L,R}'(xi_i).  g_DG (Radau) reproduces the
  // nodal DG lift M^-1 E^T B: g_L' = -l_i(-1)/w_i, g_R' = l_i(+1)/w_i.  g_HU is
  // Huynh's g2 (left correction from dgHuynhG; the right correction mirrors it,
  // g_R(x) = g_HU(-x) -> g_R'(x) = -g_HU'(-x)).  On Lobatto BOTH collapse to the
  // boundary 1/w0 lift (g_HU' vanishes at interior nodes) -- the FR/DGSEM
  // equivalence proven in the selftest.
  for (int i = 0; i < NNODE; i++) {
    if (frType == 1) {          // g_HU (Huynh)
      double gL, gpL, gR, gpR;
      dgHuynhG( xi[i], gL, gpL);
      dgHuynhG(-xi[i], gR, gpR);
      hgpL[i] =  gpL;
      hgpR[i] = -gpR;
    } else {                    // g_DG (Radau) == nodal DG
      hgpL[i] = -hTL[i]/w[i];
      hgpR[i] =  hTR[i]/w[i];
    }
  }

  // dual-pairing upwind SBP mode (arXiv 2411.06629): the rank-1 dissipation
  // A = H(D+ - D-) = -tau*(H phi)(H phi)^T with phi the top ORTHONORMAL
  // Legendre mode at the nodes.  A is symmetric negative semi-definite (A.4),
  // Q+ + Q-^T = B is untouched (A.3, A symmetric), D+- stay degree p-1 exact
  // (A.2: the quadrature projection of any lower mode vanishes), and
  // sum_i w_i phi_i = 0 makes the volume upwind term exactly conservative.
  for (int i = 0; i < NNODE; i++) hDpPhi[i] = dgLegendreON(dgOrder, xi[i]);
  // discrete-normalize: sum w phi^2 = 1 under THIS node set's quadrature (on
  // Lobatto the analytic normalization is off by the 2p-degree quadrature
  // error), so the NSFR residual filter R - sigma*phi*(sum w phi R) removes
  // EXACTLY the fraction sigma of the top mode.
  {
    double n2 = 0;
    for (int i = 0; i < NNODE; i++) n2 += w[i]*hDpPhi[i]*hDpPhi[i];
    for (int i = 0; i < NNODE; i++) hDpPhi[i] /= sqrt(n2);
  }

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

void dgUploadOperators(i32 gauss, i32 frType) {
  dgBuildOperators(gauss, frType);
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
  real tL[NNODE], tR[NNODE], gpL[NNODE], gpR[NNODE];
  for (int i = 0; i < NNODE; i++) {
    tL[i]=(real)hTL[i]; tR[i]=(real)hTR[i]; gpL[i]=(real)hgpL[i]; gpR[i]=(real)hgpR[i];
  }
  cudaMemcpyToSymbol(c_tL,  tL,  sizeof(tL));
  cudaMemcpyToSymbol(c_tR,  tR,  sizeof(tR));
  cudaMemcpyToSymbol(c_gpL, gpL, sizeof(gpL));
  cudaMemcpyToSymbol(c_gpR, gpR, sizeof(gpR));
  real dpPhi[NNODE];
  for (int i = 0; i < NNODE; i++) dpPhi[i] = (real)hDpPhi[i];
  cudaMemcpyToSymbol(c_dpPhi, dpPhi, sizeof(dpPhi));
  // FRIB image-line basis: ALWAYS the Lobatto nodes (wall = node 0 at -1),
  // regardless of the element node set (--gauss).  D0 = l_m'(-1) on that basis.
  {
    const double *lx = lgl_xi_tab[dgOrder-1];
    real ibxi[NNODE], ibd0[NNODE];
    for (int m = 0; m < NNODE; m++) {
      double v = 0;
      for (int l = 0; l < NNODE; l++) {
        if (l == m) continue;
        double t = 1.0;
        for (int k2 = 0; k2 < NNODE; k2++)
          if (k2 != m && k2 != l) t *= (-1.0 - lx[k2])/(lx[m] - lx[k2]);
        v += t/(lx[m] - lx[l]);
      }
      ibxi[m] = (real)lx[m]; ibd0[m] = (real)v;
    }
    cudaMemcpyToSymbol(c_ibLXi, ibxi, sizeof(ibxi));
    cudaMemcpyToSymbol(c_ibLD0, ibd0, sizeof(ibd0));
  }
}

// host access to the reference-element weights/nodes (diagnostic integrals)
void dgGetHostOps(double *w, double *xi, i32 gauss) {
  dgBuildOperators(gauss, 1);
  for (int i = 0; i < NNODE; i++) { w[i] = hW[i]; xi[i] = hXi[i]; }
}

bool dgOperatorSelfTest(i32 gauss, i32 frType) {
  dgBuildOperators(gauss, frType);
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
  // ── flux-reconstruction operator identities (any node set) ────────────
  // face interpolation reproduces monomials up to degree p: sum_i tR[i] xi_i^m
  // = 1, sum_i tL[i] xi_i^m = (-1)^m.
  for (int m = 0; m <= dgOrder; m++) {
    double vR = 0, vL = 0;
    for (int i = 0; i < NNODE; i++) { vR += hTR[i]*pow(hXi[i], m); vL += hTL[i]*pow(hXi[i], m); }
    expect("tR monomial", vR - 1.0, 1e-11);
    expect("tL monomial", vL - ((m % 2) ? -1.0 : 1.0), 1e-11);
  }
  // correction conservation: sum_i w_i g_R'(xi_i) = g_R(1)-g_R(-1) = 1,
  // sum_i w_i g_L'(xi_i) = g_L(1)-g_L(-1) = -1 (exact under the node quadrature
  // for g' of degree <= 2N-1; holds for BOTH g_DG and g_HU, Lobatto or Gauss).
  {
    double sR = 0, sL = 0;
    for (int i = 0; i < NNODE; i++) { sR += hW[i]*hgpR[i]; sL += hW[i]*hgpL[i]; }
    expect("gR' conservation", sR - 1.0, 1e-11);
    expect("gL' conservation", sL + 1.0, 1e-11);
  }
  // entropy variables round-trip: u -> v(u) -> u exactly (the Gauss FR surface
  // interpolates entropy variables and converts back; this is that inverse).
  {
    double W[5] = {1.7, 0.3, -0.6, 0.2, 2.4}, v[5], W2[5];
    dgEntVarsHost(W, v); dgEntVarsToPrimHost(v, W2);
    for (int q = 0; q < 5; q++) expect("entvar roundtrip", W2[q] - W[q], 1e-10);
  }
  // dual-pairing SBP mode: sum_i w_i phi_i = 0 (the volume upwind term is
  // exactly conservative) and phi kills every lower mode under the quadrature
  // (degree p-1 exactness of D+-: sum_i w_i phi_i xi_i^m = 0 for m < p).
  for (int m = 0; m < dgOrder; m++) {
    double s = 0;
    for (int i = 0; i < NNODE; i++) s += hW[i]*hDpPhi[i]*pow(hXi[i], m);
    expect("dpPhi orthogonality", s, 1e-11);
  }
  { // discrete normalization: the NSFR filter removes EXACTLY sigma of the top mode
    double n2 = 0;
    for (int i = 0; i < NNODE; i++) n2 += hW[i]*hDpPhi[i]*hDpPhi[i];
    expect("dpPhi norm", n2 - 1.0, 1e-12);
  }
  { // CH_RA flux consistency: F(W,W) = exact Euler flux
    double W[5] = {1.3, 0.7, -0.4, 0.2, 2.1};
    // host-side evaluation of eq 24-26 at WL=WR=W vs exact flux (x-dir)
    double rho=W[0],u=W[1],v=W[2],w2=W[3],pp=W[4];
    double h = pp/(rho*(1.4-1.0)) + 0.5*(u*u+v*v+w2*w2) + 2.0*pp/rho;
    double FE = rho*u*h - u*pp;
    double E  = pp/0.4 + 0.5*rho*(u*u+v*v+w2*w2);
    expect("CH_RA consistency", FE - (E+pp)*u, 1e-12);
  }

  // The Lobatto FR/DG equivalence (default node set): Huynh's g_HU (docs/
  // FRIB.pdf Eq 10) satisfies g(-1)=1, g(+1)=0, and on the LGL nodes its
  // derivative VANISHES at every interior node and equals -1/w0 at the boundary
  // -- so the FR correction distributed by -g_HU' is exactly the boundary 1/w
  // lift, and FR-g_HU on Lobatto IS nodal DGSEM (Gauss breaks this: interior
  // g_HU' != 0, so the correction genuinely distributes to every node).
  if (!gauss) {
    double g, gp;
    dgHuynhG(-1.0, g, gp);
    expect("g_HU(-1)-1", g - 1.0, 1e-12);
    expect("g_HU'(-1)+1/w0", gp + 1.0/hW[0], 1e-11);
    dgHuynhG(+1.0, g, gp);
    expect("g_HU(+1)", g, 1e-12);
    expect("g_HU'(+1)", gp, 1e-11);
    for (int i = 1; i < NNODE-1; i++) {
      dgHuynhG(hXi[i], g, gp);
      expect("g_HU'(xi interior)", gp, 1e-11);
    }
  }

  // FRIB image-line wall solve (docs/FRIB.pdf Eq 18/19, LGL simplification):
  // manufactured line data u_t(xi) with the exact wall condition must be
  // reproduced by the solved wall value.  u_t(xi) = a + b(1+xi) satisfies
  // du_t/ds(0) = -u_t(0)/R  <=>  (2/dIL) b = -a/R; solve for u_t(-1) = a
  // from the sampled u_t(xi_m), m >= 1, and compare.
  for (double dIL = 0.1; dIL < 1.01; dIL *= 2.0) {   // line basis is ALWAYS Lobatto
    const double R = 0.5;
    const double *lx = lgl_xi_tab[dgOrder-1];   // the line's OWN Lobatto nodes
    double a = 1.7, b = -a*dIL/(2.0*R);
    double D0[NNODE];
    for (int m = 0; m < NNODE; m++) {   // phi_m'(-1) on the LINE (LGL) nodes
      double v = 0;
      for (int l = 0; l < NNODE; l++) {
        if (l == m) continue;
        double t = 1.0;
        for (int k2 = 0; k2 < NNODE; k2++)
          if (k2 != m && k2 != l) t *= (-1.0 - lx[k2])/(lx[m] - lx[k2]);
        v += t/(lx[m] - lx[l]);
      }
      D0[m] = v;
    }
    double sU = 0;
    for (int m = 1; m < NNODE; m++) sU += D0[m]*(a + b*(1.0 + lx[m]));
    double ut0 = -sU/(D0[0] + 0.5*dIL/R);
    expect("FRIB wall u_t", ut0 - a, 1e-11);
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

// entropy variables v = dU/du for the entropy pair U = -rho s/(gam-1),
// s = ln p - gam ln rho (same U as dgEntropyU).  On Gauss nodes the FR surface
// interpolates THESE to the faces (not the conservative/primitive state) and
// converts back -- the entropy projection that makes the split-form volume +
// generalized-SBP surface discretely entropy conservative (Chan JCP 2018).
__device__ __forceinline__ void dgEntVars(const real W[5], real v[5]) {
  real rho = fmax(W[0], DG_EPSF), p = fmax(W[4], DG_EPSF);
  real q2 = W[1]*W[1]+W[2]*W[2]+W[3]*W[3];
  real s = log(p) - dgGam*log(rho);
  v[0] = (dgGam - s)/(dgGam-(real)1.0) - rho*q2/((real)2.0*p);
  v[1] = rho*W[1]/p; v[2] = rho*W[2]/p; v[3] = rho*W[3]/p; v[4] = -rho/p;
}
// inverse: entropy variables -> sanitized primitives
__device__ __forceinline__ void dgEntVarsToPrim(const real v[5], real W[5]) {
  real g1 = dgGam-(real)1.0;
  real v5 = fmin(v[4], -DG_EPSF);              // v5 = -rho/p < 0
  real vv2 = v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
  real s = dgGam - g1*(v[0] - vv2/((real)2.0*v5));
  real rho = pow(-v5*exp(s), -(real)1.0/g1);
  real p = -rho/v5;
  W[0]=rho; W[1]=v[1]/(-v5); W[2]=v[2]/(-v5); W[3]=v[3]/(-v5); W[4]=p;
  dgSanitizePrim(W);
}

// two-point Rusanov (local Lax-Friedrichs) flux in conservative variables --
// the robust entropy-stable LOW-ORDER flux for the subcell-FV blending
// (Hennemann/Gassner, docs/subcellFV.pdf): maximally dissipative, positivity-
// friendly, provably entropy dissipative for a convex entropy.
__device__ __forceinline__ void dgRusanovAxis(const real WL[5], const real WR[5],
                                              i32 dir, real F[5]) {
  real UL[5], UR[5], FL[5], FR[5];
  dgP2C(WL, UL); dgP2C(WR, UR);
  dgEulerFluxAxis(WL, dir, FL);
  dgEulerFluxAxis(WR, dir, FR);
  real lam = fmax(fabs(WL[1+dir]) + dgSoundSpeed(WL[4], WL[0]),
                  fabs(WR[1+dir]) + dgSoundSpeed(WR[4], WR[0]));
  for (i32 q = 0; q < 5; q++)
    F[q] = (real)0.5*(FL[q] + FR[q]) - (real)0.5*lam*(UR[q] - UL[q]);
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

// Chandrashekar entropy-conservative two-point flux along axis dir (3D / 5 vars).
// chRa = true: the Ranocha pressure-equilibrium fix (CH_RA, arXiv 2507.09131
// Eq 24-26): EC + KEP + PEP -- arithmetic-mean pressure p1 = {{p}}, enthalpy
// h = 1/(p2(g-1)) + 1/2 sum(2{{vi}}^2 - {{vi^2}}) + 2 p1/rho_ln with
// p2 = (rho/p)^ln, energy flux rho_ln un h - {{un p}}.  Consistent (WL=WR
// recovers the exact flux) and pressure-equilibrium-preserving, which the
// NSFR paper ties to the positivity CFL of the two-point flux.
__device__ __forceinline__ void dgEcFluxAxis(const real WL[5], const real WR[5],
                                             i32 dir, real F[5], bool chRa = false) {
  if (chRa) {
    real r_ln = dgLogMean(WL[0], WR[0]);
    real u_av = (real)0.5*(WL[1]+WR[1]);
    real v_av = (real)0.5*(WL[2]+WR[2]);
    real w_av = (real)0.5*(WL[3]+WR[3]);
    real p1   = (real)0.5*(WL[4]+WR[4]);
    real p2   = dgLogMean(WL[0]/WL[4], WR[0]/WR[4]);
    real k2   = (real)2.0*u_av*u_av - (real)0.5*(WL[1]*WL[1]+WR[1]*WR[1])
              + (real)2.0*v_av*v_av - (real)0.5*(WL[2]*WL[2]+WR[2]*WR[2])
              + (real)2.0*w_av*w_av - (real)0.5*(WL[3]*WL[3]+WR[3]*WR[3]);
    real h    = (real)1.0/(p2*(dgGam-(real)1.0)) + (real)0.5*k2 + (real)2.0*p1/r_ln;
    real un_av = (dir == 0) ? u_av : ((dir == 1) ? v_av : w_av);
    real f1 = r_ln*un_av;
    F[0] = f1;
    F[1] = f1*u_av;
    F[2] = f1*v_av;
    F[3] = f1*w_av;
    F[1+dir] += p1;
    F[4] = f1*h - (real)0.5*(WL[1+dir]*WL[4] + WR[1+dir]*WR[4]);
    return;
  }
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

// Roe flux along axis dir from primitives (standard Roe averages + Harten
// entropy fix) -- the face dissipation the NSFR paper (arXiv 2507.09131) pairs
// with the CH_RA two-point flux.  f = (fL+fR)/2 - (1/2) sum_k alpha_k|lam_k| r_k.
__device__ void dgRoeAxis(const real WL[5], const real WR[5], i32 dir, real F[5]) {
  const i32 n = 1+dir, t1 = 1+((dir+1)%3), t2 = 1+((dir+2)%3);
  real rL = WL[0], rR = WR[0], pL = WL[4], pR = WR[4];
  real sL = sqrt(rL), sR = sqrt(rR), si = (real)1.0/(sL+sR);
  real u  = (sL*WL[n] + sR*WR[n])*si;          // Roe-avg normal velocity
  real v1 = (sL*WL[t1] + sR*WR[t1])*si;        // tangential
  real v2 = (sL*WL[t2] + sR*WR[t2])*si;
  real q2L = WL[1]*WL[1]+WL[2]*WL[2]+WL[3]*WL[3];
  real q2R = WR[1]*WR[1]+WR[2]*WR[2]+WR[3]*WR[3];
  real HL = (pL*dgGam/(dgGam-(real)1.0) + (real)0.5*rL*q2L)/rL;
  real HR = (pR*dgGam/(dgGam-(real)1.0) + (real)0.5*rR*q2R)/rR;
  real H  = (sL*HL + sR*HR)*si;
  real q2 = u*u + v1*v1 + v2*v2;
  real a2 = (dgGam-(real)1.0)*fmax(H - (real)0.5*q2, DG_EPSF);
  real a  = sqrt(a2);
  // wave strengths (Toro ch. 11)
  real dr = rR-rL, du = WR[n]-WL[n], dv1 = WR[t1]-WL[t1], dv2 = WR[t2]-WL[t2],
       dp = pR-pL;
  real rt = sL*sR;                             // Roe-average density
  real w3 = dr - dp/a2;                        // entropy wave strength
  real w1 = (dp - rt*a*du)/((real)2.0*a2);     // u - a acoustic
  real w5 = (dp + rt*a*du)/((real)2.0*a2);     // u + a acoustic
  // Harten entropy fix on the acoustic eigenvalues
  const real dfix = (real)0.1*a;
  auto efix = [&](real lam) {
    real al = fabs(lam);
    return (al < dfix) ? (real)0.5*(lam*lam/dfix + dfix) : al;
  };
  real l1 = efix(u - a), l3 = fabs(u), l5 = efix(u + a);
  // dissipation, assembled in (rho, un, ut1, ut2, E) wave components
  real D0 = w1*l1 + w3*l3 + w5*l5;
  real Dn = w1*l1*(u-a) + w3*l3*u + w5*l5*(u+a);
  real Dt1 = (w1*l1 + w3*l3 + w5*l5)*v1 + l3*rt*dv1;
  real Dt2 = (w1*l1 + w3*l3 + w5*l5)*v2 + l3*rt*dv2;
  real DE = w1*l1*(H-u*a) + w3*l3*(real)0.5*q2 + w5*l5*(H+u*a)
          + l3*rt*(v1*dv1 + v2*dv2);
  real FL[5], FR[5];
  dgEulerFluxAxis(WL, dir, FL);
  dgEulerFluxAxis(WR, dir, FR);
  F[0]  = (real)0.5*(FL[0]+FR[0])   - (real)0.5*D0;
  F[n]  = (real)0.5*(FL[n]+FR[n])   - (real)0.5*Dn;
  F[t1] = (real)0.5*(FL[t1]+FR[t1]) - (real)0.5*Dt1;
  F[t2] = (real)0.5*(FL[t2]+FR[t2]) - (real)0.5*Dt2;
  F[4]  = (real)0.5*(FL[4]+FR[4])   - (real)0.5*DE;
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

// ── immersed-boundary geometry helpers (used by classify/fill/RHS/RK) ──────
// signed distance to the cylinder (axis along z): positive = fluid
__device__ __forceinline__ real dgIbPhi(DgSolver &grid, real x, real y) {
  real dx = x - grid.ibX, dy = y - grid.ibY;
  return sqrt(dx*dx + dy*dy) - grid.ibR;
}

// live = the element integrates the DG RHS: fluid, or (--ibevolve) a CUT
// element whose fluid-side nodes evolve.  IB_CUT stays NON-fluid for donor
// sampling / MRA details / metrics -- only the evolution machinery widens.
__device__ __forceinline__ bool dgIbLive(DgSolver &grid, i32 bIdx) {
  i32 c = grid.ibClassList[bIdx];
  return c == IB_FLUID || c == IB_CUT;
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

// Free the blocks adaptLeaves flagged DELETE: release the slot and zero every
// node of every field.  This is the common deleteDataKernel minus its
// START_CELL_LOOP, whose pseudo2D guard (added for the FV solver's collapsed-z
// mode) visits only the k == 0 plane -- a DG element carries live data on all
// NNODE z-layers, so that guard would leave stale nodes in a freed slot.

__global__ void dgDeleteDataKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bFlagsList[bIdx] != DELETE) continue;
    if (cIdx % blockSizeTot == 0) {
      grid.bLocList[bIdx] = kEmpty;
      grid.bIdxList[bIdx] = bEmpty;
      atomicAdd(&(grid.nBlocks), -1);
    }
    grid.cFlagsList[cIdx] = 0;
    for (i32 f = 0; f < grid.nFields; f++) grid.getField(f)[cIdx] = 0;
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
    case 5:   // freestream tunnel: y-lo/hi SLIP WALLS (a wind-tunnel; the
              // transmissive y/corner treatment is the measured M=0
              // domain-corner mode and misbehaves subsonically), x-lo inflow,
              // x-hi outflow, z transmissive.
              //
              // THE x BOUNDARIES ARE CHARACTERISTIC-COUNTED, and they have to
              // be: this case was written for the Mach-3 blunt body, where all
              // five characteristics enter at x-lo and all five leave at x-hi,
              // so "impose everything / extrapolate everything" is correct.
              // SUBSONICALLY BOTH ENDS ARE THEN WRONG, by exactly one
              // characteristic each -- u-c runs upstream, so it LEAVES at the
              // inflow (5 conditions imposed where 4 are admissible: the
              // boundary reflects) and ENTERS at the outflow (0 imposed where 1
              // is needed: nothing pins the pressure level, and the whole field
              // is free to drift off p_inf).
              // MEASURED at M=0.2 on the cylinder, before this branch existed:
              // a near-UNIFORM Cp offset of about -0.68 -- front stagnation
              // +0.30 against the exact +1.01, shoulder -3.64 against -3.00 --
              // identical at h = 0.25 / 0.125 / 0.0625 (grid-independent, so no
              // refinement could touch it) and unchanged by the wall treatment.
              // The free-stream gate stayed clean at 9.2e-09 throughout, which
              // is the tell: a uniform state is the exact solution the Dirichlet
              // inflow imposes, so it is the one state these BCs get right.
              //
              // --subbc 1 SELECTS AN EXPERIMENTAL SUBSONIC TREATMENT, and it is
              // OFF BY DEFAULT because it is NOT VALIDATED.  It imposes the free
              // stream's rho and velocity at inflow (extrapolating p) and a back
              // pressure at outflow.  The characteristic COUNT is then right, and
              // it fixes the front stagnation point -- Cp +0.297 -> +1.059 against
              // the exact +1.010 at h = 0.25.  But it is worse where it matters
              // more: the mid-wake far field goes Cp +0.035 -> +1.578, mass
              // accumulates (dM/dt +7.2e-03) because a rigidly imposed inflow
              // mass flux against a pinned back pressure leaves nothing to
              // balance it during the transient, and the run DIES at t = 28.
              // A correct version imposes STAGNATION conditions at inflow (h0, s,
              // flow angle -- letting u adjust) and relaxes the back pressure
              // rather than pinning it.  Kept, gated and documented so the
              // diagnosis is not lost; the default reproduces the original
              // supersonic-tunnel behaviour exactly.
      if (dir == 0 && side == 0) {
        Wg[0] = (real)1.0; Wg[1] = grid.machInf; Wg[2] = (real)0.0;
        Wg[3] = (real)0.0;
        Wg[4] = (grid.subBc && grid.machInf < (real)1.0)
                  ? Win[4]                    // subsonic: p is the outgoing one
                  : (real)1.0/dgGam;            // a_inf = 1, u = M
      } else if (dir == 0 && side == 1 && grid.subBc && grid.machInf < (real)1.0) {
        // SUBSONIC OUTFLOW, characteristic form.  Imposing p = p_inf while
        // extrapolating rho and u is NOT the same thing and is badly
        // reflecting: that state violates the two invariants that are leaving
        // (entropy, and J+ = u + 2c/(gam-1)), so the mismatch radiates back in.
        // MEASURED: rear stagnation Cp +2.13 against the exact +1.01, i.e. an
        // over-pressure larger than stagnation, and the residual ROSE.
        // Carry the outgoing invariants and let only p be set:
        //     p_g   = p_inf
        //     rho_g = rho + (p_g - p)/c^2        (entropy)
        //     u_g   = u   + (p - p_g)/(rho c)    (J+, linearised)
        const real pB  = (real)1.0/dgGam;
        const real c   = dgSoundSpeed(Win[4], Win[0]);
        const real rc  = Win[0]*c;
        Wg[0] = fmax(Win[0] + (pB - Win[4])/(c*c), DG_EPSF);
        Wg[1] = Win[1] + (Win[4] - pB)/fmax(rc, DG_EPSF);
        Wg[2] = Win[2]; Wg[3] = Win[3];
        Wg[4] = pB;
      } else if (dir == 1) {
        for (i32 q = 0; q < 5; q++) Wg[q] = Win[q];
        Wg[2] = -Win[2];                               // slip wall
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
  __shared__ real sPer[DG_EPB][3];                // Persson: [0]=thP shock(rho,p),
                                                  // [1]=rho fluct, [2]=thP speed

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    // A MODAL cut block's slots hold coefficients, so every nodal sensor in
    // this kernel (Ducros divergence, Persson modal decay) would be reading a
    // coefficient vector as a field.  Cut elements run their own cut-aware
    // sensors inside dgRhsCutKernel -- which is the whole reason those exist,
    // since the tensor-node sensors were measured blind to sub-cell trouble.
    const bool modalCut = (grid.cutOn && grid.cutModal && grid.blkCut
                           && bIdx < grid.hashTable.nKeys && grid.blkCut[bIdx] >= 0);
    const bool active = (loc != kEmpty) && !modalCut;
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

    real lamNode = 0, c2 = 0, rhoNode = 0, pNode = 0, velNode = 0;
    if (active) {
      real U[5], W[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      c2 = dgGam*W[4]/fmax(W[0], (real)1e-12);
      lamNode = fabs(W[1]) + fabs(W[2]) + fabs(W[3]) + sqrt(fmax(c2, (real)1e-14));
      rhoNode = W[0];   // Persson senses DENSITY, PRESSURE, and SPEED (max):
      pNode   = W[4];   // density stays sensitive in near-vacuum (p is
                        // sanitizer-floored there), pressure catches shocks
                        // density misses, and |u| catches the wake SHEAR
                        // layers (a velocity feature, weak in rho/p) so they
                        // drive refinement (user request 2026-07-13)
      velNode = sqrt(W[1]*W[1] + W[2]*W[2] + W[3]*W[3]);
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
      // ── Persson-Peraire modal indicator on DENSITY, PRESSURE, SPEED ──
      // s = log10( top-mode modal energy / total ), a scale-free ratio.  The
      // SHOCK sensor (rho,p -> the FV blend alpha + AV) and the REFINE sensor
      // (that PLUS speed |u|) are kept SEPARATE: velocity shear (wake layers)
      // must drive REFINEMENT but NOT alpha -- blending a smooth shear toward
      // first-order FV would SMEAR it, not sharpen it (user call 2026-07-13).
      // Three passes reuse the sV[0]/sRed[0] banks (barrier between).
      real sShock = -30.0, sVel = -30.0;
      for (i32 pass = 0; pass < 3; pass++) {
        sV[ell][0][nd] = (pass == 0) ? rhoNode : (pass == 1) ? pNode : velNode;
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
        if (active && nd == 0) {
          real total = 0, top = 0;
          for (i32 m = 0; m < blockSizeTot; m++) {
            i32 mi = m % NNODE, mj = (m/NNODE) % NNODE, mk = m/(NNODE*NNODE);
            total += sRed[ell][0][m];
            if (mi == NNODE-1 || mj == NNODE-1 || mk == NNODE-1)
              top += sRed[ell][0][m];
          }
          real s = log10(fmax(top/fmax(total, (real)1e-30), (real)1e-30));
          if (pass < 2) sShock = fmax(sShock, s);   // rho, p -> shock sensor
          else          sVel   = s;                 // |u| -> refine only
          if (pass == 0)   // amplitude floor keys on the DENSITY fluctuation
            sPer[ell][1] = total - sRed[ell][0][0];
        }
        __syncthreads();
      }
      if (active && nd == 0) {
        real s0 = grid.ppS0, kap = grid.ppKappa;
        sPer[ell][0] = (sShock < s0-kap) ? (real)0.0 : (sShock > s0+kap) ? (real)1.0
                     : (real)0.5*((real)1.0 + sin((real)0.5*(real)PI*(sShock - s0)/kap));
        sPer[ell][2] = (sVel   < s0-kap) ? (real)0.0 : (sVel   > s0+kap) ? (real)1.0
                     : (real)0.5*((real)1.0 + sin((real)0.5*(real)PI*(sVel   - s0)/kap));
      }
      __syncthreads();
    }

    if (active && nd == 0) {
      real lam = 0;
      for (i32 m = 0; m < blockSizeTot; m++) lam = fmax(lam, sRed[ell][1][m]);

      // th = SHOCK sensor (Ducros + Persson rho,p): drives alpha, AV, face
      // penalty (slot 1).  thRef = th plus the velocity/shear sensor: drives
      // REFINEMENT only (slot 5, read by dgSensorVoteKernel).
      real th = thD, thRef = thD;
      real fluct = (real)1e30;
      if (doPersson) {
        th    = fmax(th, sPer[ell][0]);
        thRef = fmax(fmax(thRef, sPer[ell][0]), sPer[ell][2]);
        fluct = sPer[ell][1];
        grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 2] = fluct;
      }
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 5] = thRef;
      // slot 6 = the SUBCELL-FV blend factor alpha (shock sensor theta only),
      // stored per element so BOTH sides of a face read the same two alphas
      // (max -> symmetric -> conservative face blend).  Shared by the volume
      // FV (dgRhsKernel) and the Rusanov FACE-flux blend (dgFaceLift).
      // NO amplitude floor here: the floor belongs on REFINEMENT (don't refine
      // low-amplitude noise), NOT on stabilization -- the low-density M=3 rear
      // has a real shock/expansion (high theta) but small density fluctuation,
      // and flooring alpha there left it unstabilized -> near-vacuum undershoot
      // -> dt collapse (high-res blowup).  fluct is unused now; kept for slot 2.
      (void)fluct;
      real alphaE = (real)0.0;
      if (grid.subFv && dgIbLive(grid, bIdx)) {
        // subThr deadband (relax the FV gate once NSFR carries the mild-
        // ringing regime): theta <= subThr stays PURE high-order + filter;
        // above it alpha rescales so a SATURATED sensor still reaches
        // min(subMax, 1) -- the theta = 1 constant-extrapolation requirement
        // (Gauss traces) is preserved by construction.
        real thA = (th - grid.subThr)/fmax((real)1.0 - grid.subThr, DG_EPSF);
        alphaE = fmin(grid.subMax, fmax(thA, (real)0.0));
        if (alphaE < (real)1e-4) alphaE = (real)0.0;
      }
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] = alphaE;
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
      const bool ibFluid = dgIbLive(grid, bIdx);   // evolving IB_CUT elements
      // publish a REAL sensor like fluid (their volume runs the same blended
      // RHS); only pure ghosts publish theta = 0 + the ibPen scale.
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

// physical entropy pair for the ES limiter (Liu/Guo/Jiang/Sun, docs/
// EntropyStableDG.pdf): U = -rho s/(gam-1), s = ln p - gam ln rho; the
// entropy flux is u U (entropy advects)
__device__ __forceinline__ real dgEntropyU(const real W[5]) {
  real s = log(fmax(W[4], DG_EPSF)) - dgGam*log(fmax(W[0], DG_EPSF));
  return -W[0]*s/(dgGam - (real)1.0);
}

// proper numerical entropy flux (paper Eq 3.9, LF form) along +dir:
//   F^ = 1/2 (un_L U_L + un_R U_R) - 1/2 alpha (U_R - U_L)
__device__ __forceinline__ real dgEntFluxLF(const real WL[5], const real WR[5],
                                            i32 dir) {
  real UL = dgEntropyU(WL), UR = dgEntropyU(WR);
  real al = fabs(WL[1+dir]) + dgSoundSpeed(WL[4], WL[0]);
  real ar = fabs(WR[1+dir]) + dgSoundSpeed(WR[4], WR[0]);
  return (real)0.5*(WL[1+dir]*UL + WR[1+dir]*UR)
       - (real)0.5*fmax(al, ar)*(UR - UL);
}

// dual-pairing SBP INTERFACE upwind flux (arXiv 2411.06629 Eq 17/22, the
// alpha (B_I + B_n) g surface half of the method).  The paper's interface flux
// is the FLUX-SPLITTING upwind flux -- central average PLUS the Gamma-scaled
// entropy-variable jump dissipation -- used INSTEAD of a Riemann solver:
//   f* = (f(WL) + f(WR))/2 - (dpFace/2) * gam~_q * (g_q^R - g_q^L).
// (An earlier ADDITIVE variant on top of HLLC double-dissipated and, through
// the 1/w0 face lift, detonated a front node in ONE step -- the interface term
// REPLACES the Riemann flux in this framework.)  gam~ = (gam-1)*gamma from
// SCRATCH slots 8..12 (face value = max of both sides -> symmetric ->
// conservative), pairing directly with dgEntVars.
__device__ __forceinline__ void dgDpJumpPenalty(DgSolver &grid, i32 myIdx,
    i32 nbrIdx, const real WL[5], const real WR[5], i32 dir, real fs[5]) {
  real vL[5], vR[5], fL[5], fR[5];
  dgEntVars(WL, vL);
  dgEntVars(WR, vR);
  dgEulerFluxAxis(WL, dir, fL);
  dgEulerFluxAxis(WR, dir, fR);
  for (i32 q = 0; q < 5; q++) {
    real gml = grid.getField(D_SCRATCH)[(u64)myIdx*blockSizeTot + 8 + q];
    real gmr = (nbrIdx >= 0)
             ? grid.getField(D_SCRATCH)[(u64)nbrIdx*blockSizeTot + 8 + q] : gml;
    fs[q] = (real)0.5*(fL[q] + fR[q])
          - (real)0.5*grid.dpFace*fmax(gml, gmr)*(vR[q] - vL[q]);
  }
}

// interface flux for the subcell-FV hybrid: HLLC blended toward Rusanov by
// the face factor af = max(alpha_own, alpha_nbr) (both sides read the same
// pair -> symmetric -> conservative).  In a TROUBLED cell the low-order FV
// wants its ELEMENT-FACE flux to be the robust vacuum-safe Rusanov, not just
// the interior subcell fluxes -- HLLC's intermediate-wave structure breaks
// down at the near-vacuum M=3 rear (measured: high-res blowup there with
// HLLC faces, every stabilizer).  af is passed with the SAME (WL,WR) axis
// order the caller used, so both the average and the dissipation match.
__device__ __forceinline__ void dgIfaceFlux(DgSolver &grid, const real WL[5],
                                            const real WR[5],
                                            i32 dir, real af, real fs[5]) {
  if (grid.rusFace == 2) { dgRoeAxis(WL, WR, dir, fs); return; }  // NSFR pairing
  dgHllcAxis(WL, WR, dir, fs);
  if (af > (real)0.0) {
    real fr[5];
    dgRusanovAxis(WL, WR, dir, fr);
    for (i32 q = 0; q < 5; q++) fs[q] = ((real)1.0-af)*fs[q] + af*fr[q];
  }
}

// forward declarations (definitions later in this TU)
__device__ real dgBasisAt(i32 a, real x);
__device__ real dgIbLineBasisAt(i32 a, real x);
__device__ void dgIbFluxTrace(DgSolver &grid, const real (*sWe)[blockSizeTot],
    i32 bIdx, i32 lvl, i32 ib, i32 jb, const real h[3],
    const real xs[3], real nx, real ny, real Wg[5]);

// face lift for one face of one element, executed by that face's node threads.
// sWe: this element's sanitized primitives (shared).  Adds into R[5]; when
// the ES limiter is on it also accumulates this face node's share of the
// outward proper-entropy-flux integral (1/V) closed-surface sum into *entAcc
// (shared, atomic) -- the mean-entropy bound of the limiter.
// NB the boundary 1/w lift IS the flux-reconstruction correction for this
// node set: on Lobatto points Huynh's g_HU derivative vanishes at every
// interior node and equals -1/w0 at the boundary (FR-g_HU == nodal DGSEM;
// proven in the selftest) -- a correction-function knob would be vacuous.
__device__ void dgFaceLift(DgSolver &grid, const real (*sWe)[blockSizeTot],
                           i32 bIdx, i32 lvl, i32 ib, i32 jb, i32 kb,
                           i32 dir, i32 side, i32 a, i32 b,
                           const real h[3], real t, real R[5], real *entAcc) {
  const i32 faceSlot[3][2] = {{12,14},{10,16},{4,22}};

  const i32  nrm    = side ? (NNODE-1) : 0;         // my face-normal node index
  const i32  myNd   = dgFaceNode(dir, nrm, a, b);
  const real jacDir = (real)2.0/h[dir];
  const real sgn    = side ? (real)-1.0 : (real)1.0;
  const bool zIdent = (grid.pseudo2D != 0) && (dir != 2);  // t2 axis is unrefined z
  // outward face-quadrature weight of this node's entropy-flux share:
  // (w_a w_b / 4) * (face area / V) = (w_a w_b / 4) / h[dir], outward sign
  const real entW = (side ? (real)1.0 : (real)-1.0)
                  * c_w[a]*c_w[b]*(real)0.25*(jacDir*(real)0.5);

  real Wme[5];
  for (i32 q = 0; q < 5; q++) Wme[q] = sWe[q][myNd];
  real fOwn[5];
  dgEulerFluxAxis(Wme, dir, fOwn);
  const real nuOwn = grid.avOn ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot] : (real)0.0;
  // face-flux Rusanov fraction.  --rusface: full Rusanov everywhere (aOwn=1 ->
  // dgIfaceFlux returns pure Rusanov).  Else MOOD keeps HLLC faces (aOwn=0) so
  // a flagged cell's FV redo stays LOCAL (unchanged traces); only the non-MOOD
  // subcell-FV path blends the face toward Rusanov by the sensor alpha.
  const real aOwn  = (grid.rusFace == 1) ? (real)1.0
                   : ((grid.subFv && !grid.mood) ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0);

  // ── resolve the face topology ────────────────────────────────────────
  i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
  i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
  i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);

  i32 nSame = grid.nbrIdxList[27*bIdx + faceSlot[dir][side]];
  // A cut neighbour OWNS the shared face rule and deposits our share of the
  // flux itself (dgRhsCutKernel); computing it here would double count.
  if (grid.cutOn && nSame != bEmpty && grid.blkCut && grid.blkCut[nSame] >= 0) return;

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
      if (side) dgIfaceFlux(grid, Wme, Wg, dir, aOwn, fs);   // BC ghost: my own alpha
      else      dgIfaceFlux(grid, Wg, Wme, dir, aOwn, fs);
      if (grid.avOn) {   // ghost shares my sensor; the jump self-gate sees the rest
        real sig = side ? dgPenaltySigma(grid, nuOwn, nuOwn, Wme, Wg)
                        : dgPenaltySigma(grid, nuOwn, nuOwn, Wg, Wme);
        if (side) dgJumpPenalty(Wme, Wg, sig, fs);
        else      dgJumpPenalty(Wg, Wme, sig, fs);
      }
      if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, -1, Wme, Wg, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, -1, Wg, Wme, dir, fs);
      }
      if (grid.esLim)
        atomicAdd(entAcc, entW*(side ? dgEntFluxLF(Wme, Wg, dir)
                                     : dgEntFluxLF(Wg, Wme, dir)));
      for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs[q] - fOwn[q]);
      return;
    }
  }

  // ── SBM surrogate wall: this face abuts an INACTIVE (cut/solid) element ──
  // No ghost trace is read.  The wall is imposed as a reflective flux built
  // from MY interior trace mirrored about the TRUE (radial) wall normal at this
  // surrogate face node -- an impermeable slip wall sitting ~1 cell out in the
  // fluid, so the flow stagnates against it and the bow-shock standoff forms
  // without any reconstruction/piston.  (Zeroth-order shift: the face-node
  // trace is used directly; the gradient shift p_wall = p + grad p . d is TODO.)
  if (grid.ibSbm && !(grid.pseudo2D && dir == 2)) {
    i32 nCls = -1;
    if (nSame != bEmpty) nCls = grid.ibClassList[nSame];
    else if (lvl > 0) {
      i32 cI = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                            grid.pseudo2D ? nkb : (nkb>>1)));
      if (cI != bEmpty) nCls = grid.ibClassList[cI];
    }
    // wall ONLY if we POSITIVELY found an inactive (cut/solid) same-level or
    // coarse-cover neighbor.  nCls == -1 means "no same/coarse neighbor" -- that
    // is the coarse side of a fine interface (finer FLUID neighbors exist), NOT
    // a wall; fall through to the finer-neighbor mortar branch.  (The near-wall
    // band is finest-level, so surrogate faces are same-level -- inactive
    // neighbours are always real blocks, never holes, under mark-inactive.)
    bool wallFace = (nCls != -1) && (nCls != IB_FLUID);
    if (wallFace) {
      // ── SBM slip wall: HLLC against the TANGENTIAL wall state ──────────
      // The ghost is the interior with the TRUE-normal velocity removed
      // (u.n~ = 0), NOT mirrored -- so the axis HLLC lets the near-wall flow
      // DECELERATE onto the wall (the SBM flow-through slip condition at the
      // true wall) instead of hard-reflecting AT the surrogate face (a hard
      // wall flattens the nose over a cell -> +18% standoff).  The HLLC keeps
      // the upwind dissipation the central pressure/penalty flux lacked.
      real xs[3];
      xs[dir]  = (ib*(dir==0)+jb*(dir==1)+kb*(dir==2) + (side ? 1 : 0)) * h[dir];
      i32 t1ax = (dir==0) ? 1 : 0, t2ax = (dir==2) ? 1 : 2;
      i32 t1bb = (dir==0) ? jb : ib, t2bb = (dir==2) ? jb : kb;
      xs[t1ax] = dgNodePos(h[t1ax], t1bb, a);
      xs[t2ax] = dgNodePos(h[t2ax], t2bb, b);
      real cxr = xs[0] - grid.ibX, cyr = xs[1] - grid.ibY;
      real rr  = fmax(sqrt(cxr*cxr + cyr*cyr), (real)1e-30);
      real nx  = cxr/rr, ny = cyr/rr;               // true outward wall normal
      // ── LIMIT the DG basis feeding the wall, using ONLY FLUID DATA ──────
      // A --ibcut 0 cut cell carries nodes INSIDE the solid (r < R) whose data
      // is non-physical; the wall reconstruction / boundary shift must use only
      // the FLUID nodes (r >= R) of the cell (user's call).  The cell mean is
      // taken over fluid nodes only; then the p=2 face trace is Zhang-Shu-
      // limited toward THAT mean so it cannot undershoot rho,p to vacuum (the
      // solid-node contamination would otherwise poison the mean and the wall).
      real Wbar[5] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};
      real wf = (real)0.0;
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i=nd%NNODE, j=(nd/NNODE)%NNODE, k=nd/(NNODE*NNODE);
        real xn = dgNodePos(h[0], ib, i), yn = dgNodePos(h[1], jb, j);
        real dxn = xn - grid.ibX, dyn = yn - grid.ibY;
        if (dxn*dxn + dyn*dyn < grid.ibR*grid.ibR) continue;   // skip SOLID nodes
        real wijk = (real)0.125*c_w[i]*c_w[j]*c_w[k];
        for (i32 q = 0; q < 5; q++) Wbar[q] += wijk*sWe[q][nd];
        wf += wijk;
      }
      if (wf > (real)0.0) { for (i32 q = 0; q < 5; q++) Wbar[q] /= wf; }
      else                { for (i32 q = 0; q < 5; q++) Wbar[q] = Wme[q]; }
      // if THIS face node is inside the solid, its trace is non-physical -- use
      // the fluid mean instead of the trace as the wall reconstruction.
      real rf2 = (xs[0]-grid.ibX)*(xs[0]-grid.ibX) + (xs[1]-grid.ibY)*(xs[1]-grid.ibY);
      real Wface[5];
      if (rf2 < grid.ibR*grid.ibR) { for (i32 q=0;q<5;q++) Wface[q] = Wbar[q]; }
      else                         { for (i32 q=0;q<5;q++) Wface[q] = Wme[q]; }
      real th = (real)1.0;
      real fr0 = (real)0.2*Wbar[0], fr4 = (real)0.2*Wbar[4];
      if (Wface[0] < fr0) th = fmin(th, (Wbar[0]-fr0)/fmax(Wbar[0]-Wface[0], DG_EPSF));
      if (Wface[4] < fr4) th = fmin(th, (Wbar[4]-fr4)/fmax(Wbar[4]-Wface[4], DG_EPSF));
      th = fmax(th, (real)0.0);
      real Wl[5];
      for (i32 q = 0; q < 5; q++) Wl[q] = Wbar[q] + th*(Wface[q]-Wbar[q]);
      if (grid.ibSbm == 2) {
        // ── Option A: LOCATION-shift flow-through ────────────────────────
        // Impose u.n~ = 0 at the TRUE wall (d = (R-r) n~ inward toward the
        // cylinder) instead of at the surrogate face: extrapolate the velocity
        // inward with the element gradient, remove its normal component there,
        // and HLLC the interior against that true-wall tangential state.  Moves
        // the effective slip wall onto the circle (inward) -> smaller standoff.
        real gcU[3], gcV[3];
        {
          real dU=0,t1U=0,t2U=0, dV=0,t1V=0,t2V=0;
          for (i32 m = 0; m < NNODE; m++) {
            i32 nA=dgFaceNode(dir,m,a,b), nB=dgFaceNode(dir,nrm,m,b), nC=dgFaceNode(dir,nrm,a,m);
            dU+=c_D[nrm][m]*sWe[1][nA]; t1U+=c_D[a][m]*sWe[1][nB]; t2U+=c_D[b][m]*sWe[1][nC];
            dV+=c_D[nrm][m]*sWe[2][nA]; t1V+=c_D[a][m]*sWe[2][nB]; t2V+=c_D[b][m]*sWe[2][nC];
          }
          gcU[dir]=jacDir*dU; gcU[t1ax]=((real)2.0/h[t1ax])*t1U; gcU[t2ax]=((real)2.0/h[t2ax])*t2U;
          gcV[dir]=jacDir*dV; gcV[t1ax]=((real)2.0/h[t1ax])*t1V; gcV[t2ax]=((real)2.0/h[t2ax])*t2V;
        }
        real dseg = grid.ibR - rr;                    // inward to the true wall
        real uG = Wl[1] + (gcU[0]*nx + gcU[1]*ny)*dseg;
        real vG = Wl[2] + (gcV[0]*nx + gcV[1]*ny)*dseg;
        real unG = uG*nx + vG*ny;                     // normal velocity AT the true wall
        real W2[5] = { Wl[0], uG - unG*nx, vG - unG*ny, Wl[3], Wl[4] };  // tangential there
        real fs[5];
        if (side) dgIfaceFlux(grid, Wl, W2, dir, aOwn, fs);
        else      dgIfaceFlux(grid, W2, Wl, dir, aOwn, fs);
        for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs[q] - fOwn[q]);
        return;
      }
      if (grid.ibSbm == 3) {
        // ghost-free FRIB wall flux (see dgIbFluxTrace)
        real WgF[5];
        dgIbFluxTrace(grid, sWe, bIdx, lvl, ib, jb, h, xs, nx, ny, WgF);
        real fs3[5];
        if (side) dgIfaceFlux(grid, Wl, WgF, dir, aOwn, fs3);
        else      dgIfaceFlux(grid, WgF, Wl, dir, aOwn, fs3);
        for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(fs3[q] - fOwn[q]);
        return;
      }
      // HARD reflective wall (best standoff for a SOLID body): zero mass/energy
      // flux, momentum = wall star pressure p* in the FACE (dir) direction, p*
      // resolved in the true normal (u_n = u.n~) from the LIMITED trace Wl --
      // reflected-shock (piston) for inflow, rarefaction for outflow.  The
      // flow-through slip form is softer and stands the shock off FURTHER
      // (+26% vs +18%), so it is worse for a solid cylinder.
      real un   = Wl[1]*nx + Wl[2]*ny;              // limited normal velocity
      real rho  = fmax(Wl[0], DG_EPSF), pI = fmax(Wl[4], DG_EPSF);
      real aI   = sqrt(dgGam*pI/rho);
      real pstar;
      if (un <= (real)0.0) {
        real A  = (real)2.0/((dgGam+(real)1.0)*rho);
        real Bc = (dgGam-(real)1.0)/(dgGam+(real)1.0)*pI;
        real m2 = un*un;
        real bq = (real)2.0*pI + m2/A, cq = pI*pI - m2*Bc/A;
        pstar = (real)0.5*(bq + sqrt(fmax(bq*bq - (real)4.0*cq, (real)0.0)));
      } else {
        real base = (real)1.0 - (dgGam-(real)1.0)*(real)0.5*un/aI;
        pstar = base > (real)0.0
              ? pI*pow(base, (real)2.0*dgGam/(dgGam-(real)1.0)) : (real)0.0;
      }
      // curvature (centripetal) correction, FLUID-only u_t: flow curving around
      // the convex wall lowers the wall pressure by rho u_t^2/R over the near-
      // wall region (dp/dn = -rho u_t^2/R, the FRIB curvature term the flat
      // reflected p* misses).  u_t is from the fluid-only limited state Wl.
      real q2  = Wl[1]*Wl[1] + Wl[2]*Wl[2] + Wl[3]*Wl[3];
      real ut2 = fmax(q2 - un*un, (real)0.0);
      pstar -= grid.ibSbmCurv * rho * ut2 / fmax(grid.ibR, DG_EPSF) * h[0];
      real fs[5] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};
      fs[1+dir] = fmax(pstar, DG_EPSF);
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
    real afN = fmax(aOwn, grid.subFv ? grid.getField(D_SCRATCH)[(u64)nSame*blockSizeTot + 6] : (real)0.0);
    // ── binary-recovery interface flux (--recov, p=2 / NNODE==3 only): the
    // central flux is evaluated at the quintic trace that L2-matches BOTH
    // elements' P2 solutions on the union (Van Leer recovery; interface
    // weights [-3,12,23,23,12,-3]/64 on the 6 line dofs), and the Rusanov
    // jump dissipation of the raw traces is retained at fraction recovK --
    // interface dissipation drops from the O(h^3) trace jump to a tunable
    // sliver of it while the central part gains two orders of face accuracy.
    // Fluid-fluid, shock-blend-quiet faces only; everything else falls back.
    if (grid.recov && NNODE == 3 && afN <= (real)0.0
        && grid.ibClassList[bIdx] == IB_FLUID
        && grid.ibClassList[nSame] == IB_FLUID) {
      const real rw[6] = {(real)(-3.0/64.0), (real)(12.0/64.0), (real)(23.0/64.0),
                          (real)( 23.0/64.0), (real)(12.0/64.0), (real)(-3.0/64.0)};
      real WR[5];
      for (i32 q = 0; q < 5; q++) WR[q] = (real)0.0;
      for (i32 k = 0; k < NNODE; k++) {
        // my line node k (normal index) from shared prims; neighbor's from global
        i32 ndM = dgFaceNode(dir, k, a, b);
        real Un[5], Wnb[5];
        for (i32 q = 0; q < 5; q++) Un[q] = grid.getField(D_RHO+q)[nSame*blockSizeTot + ndM];
        dgConsToPrimSane(Un, Wnb);
        // +dir-ordered 6-line: [upstream elem k=0..2 | downstream elem k=0..2]
        i32 jMe = side ? k : (3 + k);       // side=1: neighbor is at +dir
        i32 jNb = side ? (3 + k) : k;
        for (i32 q = 0; q < 5; q++) {
          WR[q] += rw[jMe] * sWe[q][ndM];
          WR[q] += rw[jNb] * Wnb[q];
        }
      }
      dgSanitizePrim(WR);
      real fc[5], fL[5], fR[5], fr[5];
      dgEulerFluxAxis(WR, dir, fc);
      const real *WLs = side ? Wme : Wn;
      const real *WRs = side ? Wn  : Wme;
      dgRusanovAxis(WLs, WRs, dir, fr);
      dgEulerFluxAxis(WLs, dir, fL);
      dgEulerFluxAxis(WRs, dir, fR);
      for (i32 q = 0; q < 5; q++)
        fs[q] = fc[q] + grid.recovK*(fr[q] - (real)0.5*(fL[q] + fR[q]));
    } else {
    if (side) dgIfaceFlux(grid, Wme, Wn, dir, afN, fs);
    else      dgIfaceFlux(grid, Wn, Wme, dir, afN, fs);
    }
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)nSame*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, Wme, Wn)
                      : dgPenaltySigma(grid, nuOwn, nuN, Wn, Wme);
      if (side) dgJumpPenalty(Wme, Wn, sig, fs);
      else      dgJumpPenalty(Wn, Wme, sig, fs);
    }
    if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, nSame, Wme, Wn, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, nSame, Wn, Wme, dir, fs);
      }
    if (grid.esLim)
      atomicAdd(entAcc, entW*(side ? dgEntFluxLF(Wme, Wn, dir)
                                   : dgEntFluxLF(Wn, Wme, dir)));
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
    real afC = fmax(aOwn, grid.subFv ? grid.getField(D_SCRATCH)[(u64)cIdxN*blockSizeTot + 6] : (real)0.0);
    if (side) dgIfaceFlux(grid, Wme, Wc, dir, afC, fs);
    else      dgIfaceFlux(grid, Wc, Wme, dir, afC, fs);
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)cIdxN*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, Wme, Wc)
                      : dgPenaltySigma(grid, nuOwn, nuN, Wc, Wme);
      if (side) dgJumpPenalty(Wme, Wc, sig, fs);
      else      dgJumpPenalty(Wc, Wme, sig, fs);
    }
    if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, cIdxN, Wme, Wc, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, cIdxN, Wc, Wme, dir, fs);
      }
    if (grid.esLim)
      atomicAdd(entAcc, entW*(side ? dgEntFluxLF(Wme, Wc, dir)
                                   : dgEntFluxLF(Wc, Wme, dir)));
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
    real entF = (real)0.0;   // projected proper entropy flux (same R weights)
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
            real afF = fmax(aOwn, grid.subFv ? grid.getField(D_SCRATCH)[(u64)fIdx*blockSizeTot + 6] : (real)0.0);
            if (side) dgIfaceFlux(grid, To, Wf, dir, afF, fs);
            else      dgIfaceFlux(grid, Wf, To, dir, afF, fs);
            if (grid.avOn) {
              real nuF = grid.getField(D_SCRATCH)[(u64)fIdx*blockSizeTot];
              real sig = side ? dgPenaltySigma(grid, nuOwn, nuF, To, Wf)
                              : dgPenaltySigma(grid, nuOwn, nuF, Wf, To);
              if (side) dgJumpPenalty(To, Wf, sig, fs);
              else      dgJumpPenalty(Wf, To, sig, fs);
            }
            if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
              if (side) dgDpJumpPenalty(grid, bIdx, fIdx, To, Wf, dir, fs);
              else      dgDpJumpPenalty(grid, bIdx, fIdx, Wf, To, dir, fs);
            }
            real coef = c_R[s1][a][fa] * wtB;
            for (i32 q = 0; q < 5; q++) Fs[q] += coef*fs[q];
            if (grid.esLim)
              entF += coef*(side ? dgEntFluxLF(To, Wf, dir)
                                 : dgEntFluxLF(Wf, To, dir));
          }
        }
      }
    if (grid.esLim) atomicAdd(entAcc, entW*entF);
    for (i32 q = 0; q < 5; q++) R[q] += sgn*jacDir*c_winv[nrm]*(Fs[q] - fOwn[q]);
  }
}

// Boundary mass-flux diagnostic: integrate the numerical mass flux rho*u.n
// through each DOMAIN boundary (bnd[0]=x-lo, 1=x-hi, 2=y-lo, 3=y-hi), signed
// OUTWARD (positive = leaving the domain).  Uses the SAME weak-BC HLLC flux
// the scheme applies, so the sum over boundaries equals -d/dt(fluid mass)
// exactly IF the interior+IB are conservative -- any residual is the IB
// ghost-fill non-conservation.  bnd must be pre-zeroed [4].
__global__ void dgBoundaryMassFluxKernel(DgSolver &grid, real *bnd) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    if (grid.ibOn && grid.ibClassList[bIdx] != IB_FLUID) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (!grid.isInteriorBlock(lvl, ib, jb, kb)) continue;
    real h[3]; dgElemSize(grid, lvl, h);
    i32 nx = grid.baseGridSize[0]/blockSize*powi(2, lvl);
    i32 ny = grid.baseGridSize[1]/blockSize*powi(2, lvl);
    for (i32 face = 0; face < 4; face++) {
      i32 dir = face/2, side = face%2;
      bool onB = (dir == 0) ? (side ? ib == nx-1 : ib == 0)
                            : (side ? jb == ny-1 : jb == 0);
      if (!onB) continue;
      i32 nrm = side ? (NNODE-1) : 0;
      i32 t1ax = (dir == 0) ? 1 : 0;      // x-face spans (y,z); y-face spans (x,z)
      i32 t1bb = (dir == 0) ? jb : ib;
      real acc = 0;
      for (i32 b = 0; b < NNODE; b++)
        for (i32 a = 0; a < NNODE; a++) {
          i32 nd = dgFaceNode(dir, nrm, a, b);
          real U[5], Wme[5];
          for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[bIdx*blockSizeTot + nd];
          dgConsToPrimSane(U, Wme);
          real xs[3];
          xs[dir]  = ((dir==0?ib:jb) + (side?1:0))*h[dir];
          xs[t1ax] = dgNodePos(h[t1ax], t1bb, a);
          xs[2]    = dgNodePos(h[2], kb, b);
          real Wg[5];
          dgBcState(grid, Wme, dir, side, xs[0], xs[1], grid.simT, Wg);
          real fs[5];
          if (side) dgHllcAxis(Wme, Wg, dir, fs);
          else      dgHllcAxis(Wg, Wme, dir, fs);
          real wq = c_w[a]*(h[t1ax]*(real)0.5) * c_w[b]*(h[2]*(real)0.5);
          acc += wq*fs[0];
        }
      real sgn = side ? (real)1.0 : (real)-1.0;
      atomicAdd(&bnd[face], sgn*acc);
    }
  }
}

// pressure-tight volume penalization (Reiss 2021, docs/pressureTIghtBrinkman.pdf):
// phi = eps inside the object, 1 in the fluid, smoothstep over [R-delta, R+delta].
// Returns phi and writes grad phi (radial) into gp.
__device__ __forceinline__ real dgBrinkPhi(DgSolver &grid, real x, real y, real gp[2]) {
  real dx = x - grid.ibX, dy = y - grid.ibY;
  real r  = sqrt(dx*dx + dy*dy);
  // FULL width of the smooth transition = ibBrinkDelta finest elements, so the
  // object edge is smeared over exactly ibBrinkDelta cells (default 2 -> a
  // compact 2-element interface); d is the half-width used by the smoothstep.
  real hF[3]; dgElemSize(grid, grid.nLvls-1, hF);
  real d  = (real)0.5*grid.ibBrinkDelta*hF[0], eps = grid.ibBrinkEps, R = grid.ibR;
  real tt = (r - (R - d))/((real)2.0*d);
  if (tt <= (real)0.0) { gp[0]=gp[1]=(real)0.0; return eps; }
  if (tt >= (real)1.0) { gp[0]=gp[1]=(real)0.0; return (real)1.0; }
  real phi    = eps + ((real)1.0-eps)*tt*tt*((real)3.0-(real)2.0*tt);
  real dphidr = ((real)1.0-eps)*(real)6.0*tt*((real)1.0-tt)/((real)2.0*d);
  real ir = (real)1.0/fmax(r, (real)1e-12);
  gp[0] = dphidr*dx*ir; gp[1] = dphidr*dy*ir;
  return phi;
}

__global__ void dgRhsKernel(DgSolver &grid, real t) {
  __shared__ real sW [DG_EPB][5][blockSizeTot];   // sanitized primitives
  __shared__ real sGx[DG_EPB][5][blockSizeTot];   // AV gradient banks
  __shared__ real sGy[DG_EPB][5][blockSizeTot];
  __shared__ real sGz[DG_EPB][5][blockSizeTot];
  __shared__ real sRed[DG_EPB][2][blockSizeTot];  // theta / lambda reductions
  __shared__ real sEnt[DG_EPB];                   // ES limiter: outward proper-
                                                  // entropy-flux integral (1/V)
  __shared__ real sEntQ[DG_EPB][blockSizeTot];    // ES limiter: per-node GLL-
                                                  // weighted entropy (quadrature
                                                  // cell entropy of the input)

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    // IB ghost/dead elements are never evolved (their nodal values are set by
    // the wall reconstruction); they still provide face traces to neighbors
    const bool active = (loc != kEmpty) && dgIbLive(grid, bIdx);
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
      if (grid.esLim) {
        // ES limiter: stash this node's GLL-weighted entropy; nd 0 reduces
        // the QUADRATURE cell entropy of the stage input after the sync.  It
        // must be the quadrature entropy, not U(mean): by Jensen the
        // quadrature entropy exceeds the mean's by the intra-cell variance,
        // and a mean-based bound clips smooth flow (measured: 58x vortex L2
        // regression).
        sEntQ[ell][nd] = (real)0.125*c_w[i]*c_w[j]*c_w[k]*dgEntropyU(W);
        if (nd == 0) sEnt[ell] = (real)0.0;
      }
    }
    __syncthreads();
    if (active && grid.esLim && nd == 0) {
      real E0 = (real)0.0;
      for (i32 m = 0; m < blockSizeTot; m++) E0 += sEntQ[ell][m];
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 3] = E0;
    }

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
          dgEcFluxAxis(Wi, Wm, 0, Fs, grid.ecVolume == 2);
          for (i32 q = 0; q < 5; q++) ax[q] += c_D[i][m]*Fs[q];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
          dgEcFluxAxis(Wi, Wm, 1, Fs, grid.ecVolume == 2);
          for (i32 q = 0; q < 5; q++) ay[q] += c_D[j][m]*Fs[q];
          if (!grid.pseudo2D) {
            for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
            dgEcFluxAxis(Wi, Wm, 2, Fs, grid.ecVolume == 2);
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

      // ── phase 2b: SUBCELL-FV BLENDING (Hennemann/Gassner, docs/
      //    subcellFV.pdf).  In a troubled element, blend the high-order DG
      //    VOLUME with a first-order FV volume on the LGL subgrid:
      //    R_vol = (1-a) R^DG_vol + a R^FV_vol,  a = min(subMax, theta_e).
      //    The FV volume differences two-point Rusanov fluxes between adjacent
      //    LGL nodes; at the boundary sub-interface the flux is the node's own
      //    physical flux f(u) (paper Eq 18 f_0/f_N) -- the element-FACE
      //    correction stays in the UNBLENDED surface term (phase 3), so the
      //    blend is purely volume (paper Eq 20).  This is the direct
      //    node-local form: constant-per-element a makes the flux-blend a
      //    residual-blend, no subcell-flux reconstruction needed. ───────────
      // The FV blend is the POSITIVITY-PRESERVING stabilizer and stays on at
      // EVERY level a cell is troubled (a finest-only gate was tried and
      // injected +7.7% mass: it left forming shocks running pure DG during
      // the refine lag, their cell means went negative, and the Zhang-Shu
      // mean-floor clamped them up -- non-conservatively).  The sensor-driven
      // refinement (adapt step) runs IN PARALLEL: a troubled cell is both
      // FV-stabilized now AND refined toward finest, so real features (the
      // wake) get resolved while positivity is maintained throughout.
      // Amplitude floor: a scale-free-high theta on a low-amplitude cell is
      // not real trouble -- do not blend.
      // alpha from slot 6 (dgAvNuKernel: min(subMax, shock theta) with the
      // amplitude floor) -- the SAME factor the Rusanov face-flux blend uses
      real alpha = (grid.subFv && active)
                 ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0;
      if (alpha > (real)0.0) {
        real Rfv[5] = {0,0,0,0,0};
        real fL[5], fR[5], Wm[5];
        // x subcell line
        if (i == 0) dgEulerFluxAxis(Wi, 0, fL);
        else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndX0+(i-1)]; dgRusanovAxis(Wm, Wi, 0, fL); }
        if (i == NNODE-1) dgEulerFluxAxis(Wi, 0, fR);
        else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndX0+(i+1)]; dgRusanovAxis(Wi, Wm, 0, fR); }
        for (i32 q=0;q<5;q++) Rfv[q] -= jacx*c_winv[i]*(fR[q]-fL[q]);
        // y subcell line
        if (j == 0) dgEulerFluxAxis(Wi, 1, fL);
        else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndY0+(j-1)*NNODE]; dgRusanovAxis(Wm, Wi, 1, fL); }
        if (j == NNODE-1) dgEulerFluxAxis(Wi, 1, fR);
        else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndY0+(j+1)*NNODE]; dgRusanovAxis(Wi, Wm, 1, fR); }
        for (i32 q=0;q<5;q++) Rfv[q] -= jacy*c_winv[j]*(fR[q]-fL[q]);
        // z subcell line
        if (!grid.pseudo2D) {
          if (k == 0) dgEulerFluxAxis(Wi, 2, fL);
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndZ0+(k-1)*NNODE*NNODE]; dgRusanovAxis(Wm, Wi, 2, fL); }
          if (k == NNODE-1) dgEulerFluxAxis(Wi, 2, fR);
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndZ0+(k+1)*NNODE*NNODE]; dgRusanovAxis(Wi, Wm, 2, fR); }
          for (i32 q=0;q<5;q++) Rfv[q] -= jacz*c_winv[k]*(fR[q]-fL[q]);
        }
        for (i32 q=0;q<5;q++) R[q] = ((real)1.0-alpha)*R[q] + alpha*Rfv[q];
      }

      // ── phase 2c: dual-pairing SBP volume upwinding (arXiv 2411.06629
      //    Eq 22): R += (1/2) Gamma (D+ - D-) g per direction, with
      //    (D+ - D-)g = -tau*phi*(sum_m w_m phi_m g_m) the rank-1 top-mode
      //    damping and g the entropy variables.  Entropy-dissipative
      //    (g^T H (D+-D-) g = -tau (sum w phi g)^2), exactly conservative
      //    (sum w phi = 0), O(h^p) small on smooth data -- the paper's
      //    intrinsic shock stabilizer, needing no AV/subcell-FV. ────────────
      if (grid.dpSbp > (real)0.0) {
        real gt[5];
        for (i32 q = 0; q < 5; q++)
          gt[q] = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 8 + q];
        real Sx[5] = {0,0,0,0,0}, Sy[5] = {0,0,0,0,0}, Sz[5] = {0,0,0,0,0};
        for (i32 m = 0; m < NNODE; m++) {
          real Wm[5], vm[5];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndX0 + m];
          dgEntVars(Wm, vm);
          for (i32 q = 0; q < 5; q++) Sx[q] += c_w[m]*c_dpPhi[m]*vm[q];
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
          dgEntVars(Wm, vm);
          for (i32 q = 0; q < 5; q++) Sy[q] += c_w[m]*c_dpPhi[m]*vm[q];
          if (!grid.pseudo2D) {
            for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
            dgEntVars(Wm, vm);
            for (i32 q = 0; q < 5; q++) Sz[q] += c_w[m]*c_dpPhi[m]*vm[q];
          }
        }
        for (i32 q = 0; q < 5; q++)
          R[q] -= grid.dpSbp*gt[q]*(c_dpPhi[i]*Sx[q]/h[0] + c_dpPhi[j]*Sy[q]/h[1]
                 + (grid.pseudo2D ? (real)0.0 : c_dpPhi[k]*Sz[q]/h[2]));
      }

      // ── phase 3: face lifts (boundary-node threads only) ─────────────
      if (i == 0)        dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 0, 0, j, k, h, t, R, &sEnt[ell]);
      if (i == NNODE-1)  dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 0, 1, j, k, h, t, R, &sEnt[ell]);
      if (j == 0)        dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 1, 0, i, k, h, t, R, &sEnt[ell]);
      if (j == NNODE-1)  dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 1, 1, i, k, h, t, R, &sEnt[ell]);
      if (!grid.pseudo2D) {
        if (k == 0)       dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 2, 0, i, j, h, t, R, &sEnt[ell]);
        if (k == NNODE-1) dgFaceLift(grid, sW[ell], bIdx, lvl, ib, jb, kb, 2, 1, i, j, h, t, R, &sEnt[ell]);
      }

      // ── volume-penalization IB (Reiss 2021, pressureTIghtBrinkman.pdf).
      //    Two bounded, non-stiff mechanisms are added to standard Euler:
      //  (a) the flux-form momentum source p*grad(phi) in the smeared edge.
      //      grad(phi) points OUT of the body, so p*grad(phi) is a wall
      //      reaction pushing fluid away -- maximal at the stagnation point
      //      where p peaks (this is what an earlier -rho u^2 grad(phi)/phi
      //      source could NOT do: it vanished at u=0 and the nose leaked).
      //  (b) Darcy drag -chi*(rho u) in the SOLID INTERIOR ONLY (the phi==eps
      //      plateau, grad(phi)=0): it freezes the plug so the supersonic
      //      stream cannot advect through.  chi is the CFL-stable rate
      //      lam*NNODE/h times ibBrinkRate ("as big as the timestep permits");
      //      the matching kinetic energy is removed so nothing piles up. ─────
      if (grid.ibBrink) {
        real xn = dgNodePos(h[0], ib, i), yn = dgNodePos(h[1], jb, j);
        real gp[2]; real phi = dgBrinkPhi(grid, xn, yn, gp);
        R[1] += Wi[4]*gp[0];                              // p d(phi)/dx
        R[2] += Wi[4]*gp[1];                              // p d(phi)/dy
        if (phi <= grid.ibBrinkEps) {                     // deep solid: Darcy drag
          real cS   = dgSoundSpeed(Wi[4], Wi[0]);
          real lamL = fabs(Wi[1]) + fabs(Wi[2]) + fabs(Wi[3]) + cS;
          real hmn  = fmin(h[0], grid.pseudo2D ? h[0] : fmin(h[1], h[2]));
          real chi  = grid.ibBrinkRate*lamL*(real)NNODE/hmn;
          real U0[5]; dgP2C(Wi, U0);
          R[1] -= chi*U0[1]; R[2] -= chi*U0[2]; R[3] -= chi*U0[3];
          R[4] -= chi*(U0[1]*Wi[1] + U0[2]*Wi[2] + U0[3]*Wi[3]);  // kinetic drain
        }
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
    if (active && grid.esLim && nd == 0)   // face atomics complete at the sync
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 4] = sEnt[ell];
    real lam_e = 0;
    for (i32 m = 0; m < blockSizeTot; m++)
      lam_e = fmax(lam_e, sRed[ell][1][m]);
    real theta_e = (active && (grid.avOn || grid.bulkC > (real)0.0))
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

    // ── phase 4b: sensor-gated BULK (dilatation) viscosity.  beta = bulkC *
    //    theta_e * (h/N) * lam_e * rho (the AV magnitude, dilatation-only):
    //    R[1+m] += d/dx_m [beta divu], R[4] += d/dx_m [beta divu u_m] through
    //    the same weak-divergence operator as the AV.  Contacts and shear are
    //    untouched; energy production = -int beta (divu)^2 <= 0. ─────────────
    if (grid.bulkC > (real)0.0) {
      __syncthreads();   // sGx reuse: AV phase 4 is done with it
      if (active) {
        const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
        const i32 ndY0 = i + k*NNODE*NNODE;
        const i32 ndZ0 = i + j*NNODE;
        real du = 0, dv = 0, dw = 0;
        for (i32 m = 0; m < NNODE; m++) {
          du += c_D[i][m]*sW[ell][1][ndX0 + m];
          dv += c_D[j][m]*sW[ell][2][ndY0 + m*NNODE];
          if (!grid.pseudo2D) dw += c_D[k][m]*sW[ell][3][ndZ0 + m*NNODE*NNODE];
        }
        real divu = jacx*du + jacy*dv + (grid.pseudo2D ? (real)0.0 : jacz*dw);
        real lenp = h[0]/(real)(2*dgOrder+1);   // the AV length scale
        real beta = grid.bulkC * theta_e * lenp * lam_e * sW[ell][0][nd];
        sGx[ell][0][nd] = beta*divu;   // the staged dilatational flux scalar
      }
      __syncthreads();
      if (active) {
        const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
        const i32 ndY0 = i + k*NNODE*NNODE;
        const i32 ndZ0 = i + j*NNODE;
        real sxm = 0, sxe = 0, sym = 0, sye = 0, szm = 0, sze = 0;
        for (i32 m = 0; m < NNODE; m++) {
          real bx = sGx[ell][0][ndX0 + m];
          sxm += c_w[m]*c_D[m][i]*bx;
          sxe += c_w[m]*c_D[m][i]*bx*sW[ell][1][ndX0 + m];
          real by = sGx[ell][0][ndY0 + m*NNODE];
          sym += c_w[m]*c_D[m][j]*by;
          sye += c_w[m]*c_D[m][j]*by*sW[ell][2][ndY0 + m*NNODE];
          if (!grid.pseudo2D) {
            real bz = sGx[ell][0][ndZ0 + m*NNODE*NNODE];
            szm += c_w[m]*c_D[m][k]*bz;
            sze += c_w[m]*c_D[m][k]*bz*sW[ell][3][ndZ0 + m*NNODE*NNODE];
          }
        }
        R[1] -= jacx*c_winv[i]*sxm;
        R[2] -= jacy*c_winv[j]*sym;
        R[4] -= jacx*c_winv[i]*sxe + jacy*c_winv[j]*sye;
        if (!grid.pseudo2D) {
          R[3] -= jacz*c_winv[k]*szm;
          R[4] -= jacz*c_winv[k]*sze;
        }
      }
    }

    // ── phase 5: NSFR residual filter (arXiv 2507.09131).  The ESFR K_m
    //    correction is rank-1 per line, so (M+K)^-1 M reduces to removing the
    //    fraction sigma of the residual's top Legendre mode per direction
    //    (dimension-split; sum w phi = 0 keeps it exactly conservative).
    //    Linear and state-independent -- the paper's shock recipe is this
    //    filter + the positivity limiter, nothing else. ────────────────────
    if (grid.nsfr > (real)0.0) {
      // gate by (1 - alpha): the filter belongs to the HIGH-ORDER scheme only.
      // Filtering the blended residual corrupts the top mode of the subcell-FV
      // fallback exactly where positivity depends on it (measured: 5-level
      // blast blew at t=0.70 with the unconditional filter, completes gated).
      const real sigE = grid.nsfr*((real)1.0 - (grid.subFv && active
                      ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0));
      const i32 nd3 = grid.pseudo2D ? 2 : 3;
      for (i32 d3 = 0; d3 < nd3; d3++) {
        __syncthreads();
        if (active) for (i32 q = 0; q < 5; q++) sGx[ell][q][nd] = R[q];
        __syncthreads();
        if (active) {
          const i32 idx  = (d3==0) ? i : ((d3==1) ? j : k);
          const i32 base = (d3==0) ? (j*NNODE + k*NNODE*NNODE)
                         : ((d3==1) ? (i + k*NNODE*NNODE) : (i + j*NNODE));
          const i32 str  = (d3==0) ? 1 : ((d3==1) ? NNODE : NNODE*NNODE);
          for (i32 q = 0; q < 5; q++) {
            real S = 0;
            for (i32 m = 0; m < NNODE; m++)
              S += c_w[m]*c_dpPhi[m]*sGx[ell][q][base + m*str];
            R[q] -= sigE*c_dpPhi[idx]*S;
          }
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
 * GAUSS-LEGENDRE FLUX RECONSTRUCTION RHS  (--gauss)
 * ════════════════════════════════════════════════════════════════════════
 * Solution points are Gauss-Legendre (interior, none on the faces), so unlike
 * the collocated-Lobatto dgRhsKernel:
 *   - the VOLUME is the same EC flux-differencing (2 sum_m D_im f_ec), valid on
 *     any diagonal-norm SBP operator -- on Gauss Q+Q^T = tR tR^T - tL tL^T
 *     (generalized SBP);
 *   - every FACE trace is an INTERPOLATION of the ENTROPY VARIABLES to +-1
 *     (Chan JCP 2018 entropy projection), converted back to primitives -- this
 *     is what keeps the split-form scheme discretely entropy conservative;
 *   - the interface correction distributes to ALL nodes on the normal line via
 *     the correction-function derivative g'(xi_i) (c_gpL/c_gpR), not to a single
 *     boundary node.  sum_i w_i g_R' = 1 (selftest) => cell-mean conservation
 *     for BOTH g_DG and g_HU.
 * Subcell-FV / MOOD: the FV subcells span [-1,1] by cumulative Gauss weights, so
 * their two OUTER faces are the element interfaces -- and the flux there is the
 * SHARED interface flux f* (identical on both elements), NOT an extrapolated
 * nodal flux (the Gauss nodes never reach the face).  That is what keeps the FV
 * blend conservative across a nonconforming alpha jump ("update the fluxes to
 * neighbours because the FV doesn't extrapolate").
 * SCOPE (v1, uniform/conforming mesh): same-level faces, weak BC, periodic.
 * Nonconforming coarse/fine (mortar) faces are NOT yet ported -- guarded below.
 */

// interpolate a shared entropy-variable line to a face (weights t = c_tL/c_tR)
// and convert to primitives -- MY side's entropy-projected face trace.  NOTE
// the Gauss faces (+-1) lie OUTSIDE the node span, so this is an extrapolation:
// a smooth curved profile legitimately overshoots the nodal range here, so NO
// nodal clamp is applied (clamping wrecks smooth accuracy).  Robustness at
// shocks comes from the constant-extrapolation trace blend in dgGaussFaceFlux
// (a troubled cell presents its nearest-node state), not from clamping.
__device__ __forceinline__ void dgGaussMyTrace(const real (*sV)[blockSizeTot],
    const real (*sW)[blockSizeTot], i32 dir, i32 a, i32 b, const real *tvec,
    i32 nn, real W[5]) {
  real v[5] = {0,0,0,0,0};
  real rlo=(real)1e30, rhi=(real)0.0, plo=(real)1e30, phi=(real)0.0;
  for (i32 m = 0; m < NNODE; m++) {
    i32 nd = dgFaceNode(dir, m, a, b);
    for (i32 q = 0; q < 5; q++) v[q] += tvec[m]*sV[q][nd];
    rlo = fmin(rlo, sW[0][nd]); rhi = fmax(rhi, sW[0][nd]);
    plo = fmin(plo, sW[4][nd]); phi = fmax(phi, sW[4][nd]);
  }
  dgEntVarsToPrim(v, W);
  // RELATIVE-bounds fallback (near-vacuum guard): the exp inverse map is
  // hypersensitive at rho,p ~ 1e-5 -- a tiny wiggle in the interpolated s
  // swings the trace density by orders of magnitude (measured: the M=3 rear
  // stall).  A trace OUTSIDE [1/2, 2]x the line's nodal range is non-physical
  // extrapolation ring, not resolution: present the nearest node instead.
  // Smooth extrapolation overshoot is a few % -- never triggers (vortex
  // regression-exact); unlike a hard clamp this changes nothing else.
  if (W[0] > (real)2.0*rhi || W[0] < (real)0.5*rlo ||
      W[4] > (real)2.0*phi || W[4] < (real)0.5*plo) {
    i32 nd = dgFaceNode(dir, nn, a, b);
    for (i32 q = 0; q < 5; q++) W[q] = sW[q][nd];
  }
}

// a NEIGHBOUR block's entropy-projected face trace: read its normal line from
// global, cons->prim->entropy vars, project with tvec (the neighbour's face
// facing me: tL if its -1 side, tR if its +1 side).  Bitwise identical to what
// the neighbour computes for the shared face -> f* matches on both sides.
__device__ __forceinline__ void dgGaussNbrTrace(DgSolver &grid, i32 nbrIdx,
    i32 dir, i32 a, i32 b, const real *tvec, i32 nn, real W[5]) {
  real v[5] = {0,0,0,0,0};
  real rlo=(real)1e30, rhi=(real)0.0, plo=(real)1e30, phi=(real)0.0;
  real Wnn[5];
  for (i32 m = 0; m < NNODE; m++) {
    i32 nd = dgFaceNode(dir, m, a, b);
    real U[5], Wm[5], vm[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)nbrIdx*blockSizeTot + nd];
    dgConsToPrimSane(U, Wm);
    dgEntVars(Wm, vm);
    for (i32 q = 0; q < 5; q++) v[q] += tvec[m]*vm[q];
    rlo = fmin(rlo, Wm[0]); rhi = fmax(rhi, Wm[0]);
    plo = fmin(plo, Wm[4]); phi = fmax(phi, Wm[4]);
    if (m == nn) for (i32 q = 0; q < 5; q++) Wnn[q] = Wm[q];
  }
  dgEntVarsToPrim(v, W);
  if (W[0] > (real)2.0*rhi || W[0] < (real)0.5*rlo ||
      W[4] > (real)2.0*phi || W[4] < (real)0.5*plo) {
    for (i32 q = 0; q < 5; q++) W[q] = Wnn[q];   // near-vacuum guard (see MyTrace)
  }
}

// the flux-differencing volume's BOUNDARY flux, the quantity the FR correction
// must be referenced against for EXACT conservation.  The split-form volume
// cell-mean telescopes (generalized SBP: Q+Q^T = tR tR^T - tL tL^T) to
//   Gbnd_R - Gbnd_L,  Gbnd_{L/R} = sum_{a,b} t_{L/R}[a] t_{L/R}[b] f_S(u_a,u_b).
// Using ftil = Gbnd here (NOT f(projected state)) makes R_surf = -jac g'(f*-ftil)
// conservative for BOTH g_DG and g_HU, since sum_i w_i g' = +-1.  Consistent at
// a uniform state (Gbnd = f(u)) so free-stream is preserved exactly.
__device__ __forceinline__ void dgGaussBndFlux(const real (*sW)[blockSizeTot],
    i32 dir, i32 a, i32 b, bool chRa, real fL[5], real fR[5]) {
  for (i32 q = 0; q < 5; q++) { fL[q] = (real)0.0; fR[q] = (real)0.0; }
  for (i32 p = 0; p < NNODE; p++)
    for (i32 r = 0; r < NNODE; r++) {
      real Wp[5], Wr[5], F[5];
      i32 ndp = dgFaceNode(dir, p, a, b), ndr = dgFaceNode(dir, r, a, b);
      for (i32 q = 0; q < 5; q++) { Wp[q] = sW[q][ndp]; Wr[q] = sW[q][ndr]; }
      dgEcFluxAxis(Wp, Wr, dir, F, chRa);
      for (i32 q = 0; q < 5; q++) {
        fL[q] += c_tL[p]*c_tL[r]*F[q];
        fR[q] += c_tR[p]*c_tR[r]*F[q];
      }
    }
}

// gather a block's FACE TRACE ARRAYS for the Gauss mortar: at each tangential
// node pair (c1,c2), the PRIMITIVES of the entropy-projected face trace (tvec
// along the normal, converted per tangential node -- the same construction the
// block itself uses, so the values are bitwise identical to its own trace) and
// the nearest-node-plane primitives (the constant-extrapolation trace of a
// troubled cell).  Tangential interpolation to mortar points then happens in
// PRIMITIVES via dgTraceAt + sanitize, matching the validated Lobatto mortar:
// interpolating ENTROPY VARIABLES tangentially ACROSS a shock front and exp-
// mapping back amplifies the interpolation overshoot into vacuum/detonation
// states (measured: adaptive Sod blowup t=0.01), while primitive interpolation
// overshoot is linear and the sanitize + subcell blend contain it.
__device__ void dgGaussGatherFace(DgSolver &grid, i32 eIdx, i32 dir,
    const real *tvec, i32 nrm, real Vcf[5][NNODE*NNODE], real Pcf[5][NNODE*NNODE]) {
  for (i32 c2 = 0; c2 < NNODE; c2++)
    for (i32 c1 = 0; c1 < NNODE; c1++) {
      real v[5] = {0,0,0,0,0};
      real rlo=(real)1e30, rhi=(real)0.0, plo=(real)1e30, phi=(real)0.0;
      for (i32 m = 0; m < NNODE; m++) {
        i32 nd = dgFaceNode(dir, m, c1, c2);
        real U[5], Wm[5], vm[5];
        for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)eIdx*blockSizeTot + nd];
        dgConsToPrimSane(U, Wm);
        dgEntVars(Wm, vm);
        for (i32 q = 0; q < 5; q++) v[q] += tvec[m]*vm[q];
        rlo = fmin(rlo, Wm[0]); rhi = fmax(rhi, Wm[0]);
        plo = fmin(plo, Wm[4]); phi = fmax(phi, Wm[4]);
        if (m == nrm) for (i32 q = 0; q < 5; q++) Pcf[q][c1 + NNODE*c2] = Wm[q];
      }
      real Wt[5];
      dgEntVarsToPrim(v, Wt);
      // near-vacuum/overflow guard (same as dgGaussMyTrace): the exp inverse
      // can blow up to inf on line data the sanitizer floors -- fall back to
      // the nearest-node plane value for this tangential position.
      if (!(Wt[0] < (real)2.0*rhi) || Wt[0] < (real)0.5*rlo ||
          !(Wt[4] < (real)2.0*phi) || Wt[4] < (real)0.5*plo)
        for (i32 q = 0; q < 5; q++) Wt[q] = Pcf[q][c1 + NNODE*c2];
      for (i32 q = 0; q < 5; q++) Vcf[q][c1 + NNODE*c2] = Wt[q];
    }
}


// ── GHOST-FREE FRIB WALL FLUX (--ibsbm 3): the FRIB image-line solve done
//    entirely from THE ELEMENT OWNING THE WALL FACE (its degree-p polynomial
//    + the analytic levelset), evaluated AT the face point as the ghost
//    trace.  No ghost fill, no donor locate/march (the measured stair-streak
//    source), no nodonor path -- and image distances are FORCED small, the
//    memory-endorsed stable regime.  Line: wall at xi=-1 (levelset foot
//    through the face point), dIL = 2 sgF + h so the first interior sample
//    sits mid-element; samples = MY polynomial (tensor Lagrange on sWe,
//    MUSCL-limited); primitive (p,rho) wall solve with centripetal dp/dn and
//    isentropic drho/dn; piston star / LO ladder kept as TRACE choices.
__device__ void dgIbFluxTrace(DgSolver &grid, const real (*sWe)[blockSizeTot],
    i32 bIdx, i32 lvl, i32 ib, i32 jb, const real h[3],
    const real xs[3], real nx, real ny, real Wg[5]) {
  real r   = fmax(sqrt((xs[0]-grid.ibX)*(xs[0]-grid.ibX)
                     + (xs[1]-grid.ibY)*(xs[1]-grid.ibY)), (real)1e-30);
  real sgF = r - grid.ibR;                       // face point to wall (>0 outside)
  real xw  = grid.ibX + grid.ibR*nx, yw = grid.ibY + grid.ibR*ny;
  real dIL = (real)2.0*fmax(sgF, (real)0.0) + h[0];
  real tx = -ny, ty = nx;                        // wall tangent
  real Un[NNODE], Ut[NNODE], Wt[NNODE], Pn[NNODE], Rn[NNODE];
  real F1[5] = {0,0,0,0,0};                      // innermost sample
  for (i32 m = 1; m < NNODE; m++) {
    real sm = (c_ibLXi[m] + (real)1.0)*(real)0.5*dIL;
    real xm = xw + sm*nx, ym = yw + sm*ny;
    real zx = (real)2.0*(xm/h[0] - ib) - (real)1.0;   // MY element coords
    real zy = (real)2.0*(ym/h[1] - jb) - (real)1.0;   // (mild extrapolation ok)
    real W[5] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};
    for (i32 aa = 0; aa < NNODE; aa++) {
      real La = dgBasisAt(aa, zx);
      for (i32 bb = 0; bb < NNODE; bb++) {
        real w2 = La*dgBasisAt(bb, zy);
        i32 nd = aa + bb*NNODE;                  // pseudo-2D plane (k = 0)
        for (i32 q = 0; q < 5; q++) W[q] += w2*sWe[q][nd];
      }
    }
    dgSanitizePrim(W);
    if (m == 1) for (i32 q = 0; q < 5; q++) F1[q] = W[q];
    Un[m] = W[1]*nx + W[2]*ny;
    Ut[m] = W[1]*tx + W[2]*ty;
    Wt[m] = W[3];
    Pn[m] = W[4];
    Rn[m] = W[0];
  }
  real un1 = F1[1]*nx + F1[2]*ny;
  real a1  = sqrt(dgGam*fmax(F1[4], DG_EPSF)/fmax(F1[0], DG_EPSF));
  real th  = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1];
  bool gated = (fabs(un1) > (real)0.3*a1) || (th > grid.ibShockTheta);
  if (gated && un1 < (real)0.0) {
    // arriving shock: exact wall-Riemann (piston) star TRACE -- the
    // reflection BC (a smooth transpiration trace transmits; measured -88%)
    real rhoI = fmax(F1[0], DG_EPSF), pIr = F1[4];
    real pI = fmax(pIr, DG_EPSF), m2 = un1*un1;
    real A  = (real)2.0/((dgGam+(real)1.0)*rhoI);
    real Bc = (dgGam-(real)1.0)/(dgGam+(real)1.0)*pI;
    real bq = (real)2.0*pI + m2/A, cq = pI*pI - m2*Bc/A;
    real ps = (real)0.5*(bq + sqrt(fmax(bq*bq - (real)4.0*cq, (real)0.0)));
    real pCap = (pIr > (real)0.0) ? pI : (real)0.5*rhoI*m2;
    ps = fmin(ps, (real)50.0*pCap);
    real g = (dgGam-(real)1.0)/(dgGam+(real)1.0), pr = ps/pI;
    real rs = rhoI*(pr + g)/(g*pr + (real)1.0);
    real ut1 = F1[1]*tx + F1[2]*ty;
    Wg[0] = rs; Wg[1] = ut1*tx; Wg[2] = ut1*ty; Wg[3] = F1[3]; Wg[4] = ps;
    dgSanitizePrim(Wg);
    return;
  }
  if (gated) {
    // outflow-gated: LO trace (u_n linear in wall distance, rest copied)
    real s1 = (c_ibLXi[1] + (real)1.0)*(real)0.5*dIL;
    real unF = un1*(fmax(sgF, (real)0.0)/fmax(s1, DG_EPSF));
    real ut1 = F1[1]*tx + F1[2]*ty;
    Wg[0] = F1[0]; Wg[1] = unF*nx + ut1*tx; Wg[2] = unF*ny + ut1*ty;
    Wg[3] = F1[3]; Wg[4] = F1[4];
    dgSanitizePrim(Wg);
    return;
  }
  // HO: primitive wall solve (Neumann: centripetal dp, isentropic drho)
  const real *D0 = c_ibLD0;
  real sU=0, sW2=0, sP=0, sR=0;
  for (i32 m = 1; m < NNODE; m++) {
    sU += D0[m]*Ut[m]; sW2 += D0[m]*Wt[m];
    sP += D0[m]*Pn[m]; sR  += D0[m]*Rn[m];
  }
  Un[0] = (real)0.0;
  real ut1c = Ut[1];
  real gp = grid.ibCurv
          ? (real)0.5*dIL*Rn[1]*ut1c*ut1c/fmax(grid.ibR, DG_EPSF) : (real)0.0;
  real a2 = dgGam*fmax(Pn[1], DG_EPSF)/fmax(Rn[1], DG_EPSF);
  Pn[0] = (gp    - sP)/D0[0];
  Rn[0] = (gp/a2 - sR)/D0[0];
  real dnm = D0[0] + (grid.ibCurv ? (real)0.5*dIL/grid.ibR : (real)0.0);
  Ut[0] = -sU/dnm;
  Wt[0] = -sW2/D0[0];
  real xiF = fmin(fmax((real)2.0*sgF/dIL - (real)1.0, (real)-1.35), (real)1.0);
  real un=0, ut=0, wt=0, pw=0, rw=0;
  for (i32 m = 0; m < NNODE; m++) {
    real ph = dgIbLineBasisAt(m, xiF);
    un += ph*Un[m]; ut += ph*Ut[m]; wt += ph*Wt[m];
    pw += ph*Pn[m]; rw += ph*Rn[m];
  }
  if (grid.ibLimit) {   // MUSCL: no new extremum beyond wall+samples
    real lo, hi;
    lo=Un[0]; hi=Un[0]; for (i32 m=1;m<NNODE;m++){lo=fmin(lo,Un[m]);hi=fmax(hi,Un[m]);} un=fmin(fmax(un,lo),hi);
    lo=Ut[0]; hi=Ut[0]; for (i32 m=1;m<NNODE;m++){lo=fmin(lo,Ut[m]);hi=fmax(hi,Ut[m]);} ut=fmin(fmax(ut,lo),hi);
    lo=Wt[0]; hi=Wt[0]; for (i32 m=1;m<NNODE;m++){lo=fmin(lo,Wt[m]);hi=fmax(hi,Wt[m]);} wt=fmin(fmax(wt,lo),hi);
    lo=Pn[0]; hi=Pn[0]; for (i32 m=1;m<NNODE;m++){lo=fmin(lo,Pn[m]);hi=fmax(hi,Pn[m]);} pw=fmin(fmax(pw,lo),hi);
    lo=Rn[0]; hi=Rn[0]; for (i32 m=1;m<NNODE;m++){lo=fmin(lo,Rn[m]);hi=fmax(hi,Rn[m]);} rw=fmin(fmax(rw,lo),hi);
  }
  if (pw > (real)1e-3*Pn[1] && rw > (real)1e-3*Rn[1]) {
    Wg[0] = rw; Wg[1] = un*nx + ut*tx; Wg[2] = un*ny + ut*ty;
    Wg[3] = wt; Wg[4] = pw;
    dgSanitizePrim(Wg);
    return;
  }
  // inadmissible: LO
  {
    real s1 = (c_ibLXi[1] + (real)1.0)*(real)0.5*dIL;
    real unF = un1*(fmax(sgF, (real)0.0)/fmax(s1, DG_EPSF));
    real ut1 = F1[1]*tx + F1[2]*ty;
    Wg[0] = F1[0]; Wg[1] = unF*nx + ut1*tx; Wg[2] = unF*ny + ut1*ty;
    Wg[3] = F1[3]; Wg[4] = F1[4];
    dgSanitizePrim(Wg);
  }
}

// fluid-only cell mean of an arbitrary block (GSBM cut-cell trace: solid-
// filled nodes are excluded and the GLL weights renormalised -- the Lobatto
// ibcut-0 "fluid nodes only" rule, applied to the Gauss traces)
__device__ void dgGaussFluidMean(DgSolver &grid, i32 idx, i32 blvl,
    i32 bib, i32 bjb, const real hb[3], real Wbar[5]) {
  real acc[5] = {0,0,0,0,0}, wf = (real)0.0;
  for (i32 nd = 0; nd < blockSizeTot; nd++) {
    i32 ii=nd%NNODE, jj=(nd/NNODE)%NNODE, kk=nd/(NNODE*NNODE);
    real xn = dgNodePos(hb[0], bib, ii), yn = dgNodePos(hb[1], bjb, jj);
    real dxn = xn - grid.ibX, dyn = yn - grid.ibY;
    if (dxn*dxn + dyn*dyn < grid.ibR*grid.ibR) continue;
    real U[5], W[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)idx*blockSizeTot+nd];
    dgConsToPrimSane(U, W);
    real wijk = (real)0.125*c_w[ii]*c_w[jj]*c_w[kk];
    for (i32 q = 0; q < 5; q++) acc[q] += wijk*W[q];
    wf += wijk;
  }
  if (wf > (real)0.0) for (i32 q = 0; q < 5; q++) Wbar[q] = acc[q]/wf;
  else {   // fully-solid (shouldn't be active): fall back to raw mean node 0
    real U[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)idx*blockSizeTot];
    dgConsToPrimSane(U, Wbar);
  }
}

// resolve one face's SHARED interface flux f* for the Gauss FR surface.
//   Wp  = MY high-order entropy-projected face trace
//   Wnn = MY nearest-node value (CONSTANT extrapolation of the boundary subcell)
// A TROUBLED cell (subcell-FV blend alpha>0, or MOOD-flagged) presents the
// constant-extrapolated boundary-subcell state at the face, not the high-order
// trace -- "the FV doesn't extrapolate".  The face state is blended
//   Wface = (1-af) Wp + af Wnn,   af = max(alpha_me, alpha_nbr),
// with af taken as the MAX of both sides so BOTH elements build the identical
// blended traces and hence the identical f* (single-valued interface flux ->
// conservative even across an alpha jump; this is "update the fluxes to the
// neighbour").  fs = f*; WmeB returns MY blended trace (for the FR correction
// f* - f(WmeB) and the FV boundary subcell flux).
__device__ void dgGaussFaceFlux(DgSolver &grid,
    const real (*sVe)[blockSizeTot], const real (*sWe)[blockSizeTot],
    i32 bIdx, i32 lvl, i32 ib, i32 jb, i32 kb, i32 dir, i32 side, i32 a, i32 b,
    const real Wp[5], const real Wnn[5], const real h[3], real t,
    real fs[5], real WmeB[5]) {
  const i32 faceSlot[3][2] = {{12,14},{10,16},{4,22}};
  const real aOwn  = grid.subFv ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0;
  const real nuOwn = grid.avOn  ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot] : (real)0.0;

  i32 nib = ib + ((dir==0) ? (side ? 1 : -1) : 0);
  i32 njb = jb + ((dir==1) ? (side ? 1 : -1) : 0);
  i32 nkb = kb + ((dir==2) ? (side ? 1 : -1) : 0);
  i32 nSame = grid.nbrIdxList[27*bIdx + faceSlot[dir][side]];
  // A cut neighbour OWNS the shared face rule and deposits our share of the
  // flux itself (dgRhsCutKernel), so computing it here would double count.
  if (grid.cutOn && nSame != bEmpty && grid.blkCut && grid.blkCut[nSame] >= 0) return;
  const real *tNbr = side ? c_tL : c_tR;   // neighbour's facing side: my +side -> its -1 -> tL
  const i32 nnNbr = side ? 0 : (NNODE-1);  // neighbour's nearest node to the shared face

  if (nSame == bEmpty && grid.isExteriorBlock(lvl, nib, njb, nkb)) {
    if (grid.bcType == 2) {
      grid.wrapBlockPeriodic(lvl, nib, njb, nkb);
      nSame = grid.getBlockIdx(grid.encode(lvl, nib, njb, nkb));
    } else {
      // weak BC: no neighbour alpha -- blend by my own alpha, ghost from Wface
      real af = (grid.rusFace == 1) ? (real)1.0 : aOwn;
      for (i32 q = 0; q < 5; q++) WmeB[q] = ((real)1.0-af)*Wp[q] + af*Wnn[q];
      real xs[3];
      xs[dir] = (ib*(dir==0)+jb*(dir==1)+kb*(dir==2) + (side ? 1 : 0)) * h[dir];
      i32 t1ax = (dir==0) ? 1 : 0, t2ax = (dir==2) ? 1 : 2;
      i32 t1bb = (dir==0) ? jb : ib, t2bb = (dir==2) ? jb : kb;
      xs[t1ax] = dgNodePos(h[t1ax], t1bb, a);
      xs[t2ax] = dgNodePos(h[t2ax], t2bb, b);
      real Wg[5];
      dgBcState(grid, WmeB, dir, side, xs[0], xs[1], t, Wg);
      if (side) dgIfaceFlux(grid, WmeB, Wg, dir, af, fs);
      else      dgIfaceFlux(grid, Wg, WmeB, dir, af, fs);
      if (grid.avOn) {
        real sig = side ? dgPenaltySigma(grid, nuOwn, nuOwn, WmeB, Wg)
                        : dgPenaltySigma(grid, nuOwn, nuOwn, Wg, WmeB);
        if (side) dgJumpPenalty(WmeB, Wg, sig, fs);
        else      dgJumpPenalty(Wg, WmeB, sig, fs);
      }
      if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, -1, WmeB, Wg, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, -1, Wg, WmeB, dir, fs);
      }
      return;
    }
  }

  // ── SBM surrogate wall on GAUSS points (port of the dgFaceLift branch) ────
  // This face abuts an INACTIVE (cut/solid) element: no ghost data is read --
  // the wall is a pure star-pressure flux built from MY OWN limited trace.
  // Ghost-free by construction, so there is no reconstruction contract for the
  // Gauss extrapolation to break (the FRIB slam ladder's failure).  The trace
  // is the entropy-projected Wp, Zhang-Shu-limited toward the FLUID-ONLY cell
  // mean (solid nodes of --ibcut 0 cut cells carry non-physical fill data).
  if (grid.ibSbm && !(grid.pseudo2D && dir == 2)) {
    i32 nCls = -1;
    if (nSame != bEmpty) nCls = grid.ibClassList[nSame];
    else if (lvl > 0) {
      i32 cI = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                            grid.pseudo2D ? nkb : (nkb>>1)));
      if (cI != bEmpty) nCls = grid.ibClassList[cI];
    }
    bool wallFace = (nCls != -1) && (nCls != IB_FLUID);
    if (wallFace) {
      real xs[3];
      xs[dir]  = (ib*(dir==0)+jb*(dir==1)+kb*(dir==2) + (side ? 1 : 0)) * h[dir];
      i32 t1ax = (dir==0) ? 1 : 0, t2ax = (dir==2) ? 1 : 2;
      i32 t1bb = (dir==0) ? jb : ib, t2bb = (dir==2) ? jb : kb;
      xs[t1ax] = dgNodePos(h[t1ax], t1bb, a);
      xs[t2ax] = dgNodePos(h[t2ax], t2bb, b);
      real cxr = xs[0] - grid.ibX, cyr = xs[1] - grid.ibY;
      real rr  = fmax(sqrt(cxr*cxr + cyr*cyr), (real)1e-30);
      real nx  = cxr/rr, ny = cyr/rr;               // true outward wall normal
      // fluid-only cell mean (skip solid nodes r < R)
      real Wbar[5] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};
      real wf = (real)0.0;
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 ii=nd%NNODE, jj=(nd/NNODE)%NNODE, kk=nd/(NNODE*NNODE);
        real xn = dgNodePos(h[0], ib, ii), yn = dgNodePos(h[1], jb, jj);
        real dxn = xn - grid.ibX, dyn = yn - grid.ibY;
        if (dxn*dxn + dyn*dyn < grid.ibR*grid.ibR) continue;
        real wijk = (real)0.125*c_w[ii]*c_w[jj]*c_w[kk];
        for (i32 q = 0; q < 5; q++) Wbar[q] += wijk*sWe[q][nd];
        wf += wijk;
      }
      if (wf > (real)0.0) { for (i32 q = 0; q < 5; q++) Wbar[q] /= wf; }
      else                { for (i32 q = 0; q < 5; q++) Wbar[q] = Wp[q]; }
      // face state: my projected trace, or the fluid mean if the face point
      // itself is inside the solid; Zhang-Shu-limit toward the fluid mean
      real rf2 = (xs[0]-grid.ibX)*(xs[0]-grid.ibX) + (xs[1]-grid.ibY)*(xs[1]-grid.ibY);
      real Wface[5];
      // nearest-node, NOT the projection: a --ibcut 0 cut cell's projected
      // trace extrapolates THROUGH its solid-filled nodes (rest-gate amplifier)
      if (rf2 < grid.ibR*grid.ibR) { for (i32 q=0;q<5;q++) Wface[q] = Wbar[q]; }
      else                         { for (i32 q=0;q<5;q++) Wface[q] = Wnn[q]; }
      real th = (real)1.0;
      real fr0 = (real)0.2*Wbar[0], fr4 = (real)0.2*Wbar[4];
      if (Wface[0] < fr0) th = fmin(th, (Wbar[0]-fr0)/fmax(Wbar[0]-Wface[0], DG_EPSF));
      if (Wface[4] < fr4) th = fmin(th, (Wbar[4]-fr4)/fmax(Wbar[4]-Wface[4], DG_EPSF));
      th = fmax(th, (real)0.0);
      real Wl[5];
      for (i32 q = 0; q < 5; q++) Wl[q] = Wbar[q] + th*(Wface[q]-Wbar[q]);
      if (grid.ibSbm == 3) {
        // ghost-free FRIB wall flux: the image-line trace from MY polynomial
        real WgF[5];
        dgIbFluxTrace(grid, sWe, bIdx, lvl, ib, jb, h, xs, nx, ny, WgF);
        real fsw3[5];
        if (side) dgIfaceFlux(grid, Wl, WgF, dir, aOwn, fsw3);
        else      dgIfaceFlux(grid, WgF, Wl, dir, aOwn, fsw3);
        for (i32 q = 0; q < 5; q++) { fs[q] = fsw3[q]; WmeB[q] = Wp[q]; }
        return;
      }
      // GAUSS wall flux: axis-mirror HLLC, NOT the Lobatto pure-pressure flux.
      // fs = p*.e_dir has ZERO dissipation in the mass/tangential/energy rows;
      // Lobatto tolerates that because the correction is confined to the face
      // node by the 1/w0 lift, but on Gauss the g'-distributed correction
      // feeds interior nodes and the undamped reflection loop has gain > 1
      // (measured: rest gate 0.13-0.64 across every stabilizer combo).  The
      // mirror HLLC keeps the exact zero dir-mass-flux AND upwinds every row.
      real Wg[5] = { Wl[0], Wl[1], Wl[2], Wl[3], Wl[4] };
      Wg[1+dir] = -Wg[1+dir];
      // curvature (centripetal) pressure shift: the ghost presents the TRUE-
      // WALL pressure p_w = p - rho u_t^2/R * (r - R) (normal momentum balance
      // dp/dn = rho u_t^2/R integrated over THIS point's own distance).  u_t
      // from the limited trace; u_t = 0 at rest/stagnation -> exact no-op
      // there (never touches the nose standoff, per the Lobatto law).
      if (grid.ibSbmCurv > (real)0.0) {
        real unT = Wl[1]*nx + Wl[2]*ny;
        real q2t = Wl[1]*Wl[1] + Wl[2]*Wl[2] + Wl[3]*Wl[3];
        real ut2 = fmax(q2t - unT*unT, (real)0.0);
        real dpw = grid.ibSbmCurv * Wl[0] * ut2 / fmax(grid.ibR, DG_EPSF)
                 * fmax(rr - grid.ibR, (real)0.0);
        Wg[4] = fmax(Wg[4] - dpw, (real)1e-3*Wl[4]);   // floor: never vacuum ghost
      } else if (grid.ibSbmCurv < (real)0.0) {
        // A/B variant: TAYLOR-gradient pressure shift p_w = p - (r-R)(n~.grad p)
        // from the sampled element gradient (vs the centripetal closure above)
        const i32 nrmB = side ? (NNODE-1) : 0;
        real dP=0, t1P=0;
        for (i32 m = 0; m < NNODE; m++) {
          dP  += c_D[nrmB][m]*sWe[4][dgFaceNode(dir, m, a, b)];
          t1P += c_D[a][m]  *sWe[4][dgFaceNode(dir, nrmB, m, b)];
        }
        i32 t1x = (dir==0) ? 1 : 0;
        real gpx = (dir==0) ? ((real)2.0/h[0])*dP  : ((real)2.0/h[t1x])*t1P;
        real gpy = (dir==0) ? ((real)2.0/h[t1x])*t1P : ((real)2.0/h[1])*dP;
        real dpw = -grid.ibSbmCurv * (rr - grid.ibR) * (gpx*nx + gpy*ny);
        Wg[4] = fmax(Wg[4] - dpw, (real)1e-3*Wl[4]);
      }
      if (grid.ibSbmCurv != (real)0.0) {   // limit BOTH pressure variants to the
        real pLo=(real)1e30, pHi=(real)-1e30;          // line's nodal range
        for (i32 m = 0; m < NNODE; m++) {
          i32 nA = dgFaceNode(dir, m, a, b);
          pLo = fmin(pLo, sWe[4][nA]); pHi = fmax(pHi, sWe[4][nA]);
        }
        Wg[4] = fmin(fmax(Wg[4], fmax((real)0.5*pLo, (real)1e-3*Wl[4])), pHi);
      }
      real fsw[5];
      if (side) dgIfaceFlux(grid, Wl, Wg, dir, aOwn, fsw);
      else      dgIfaceFlux(grid, Wg, Wl, dir, aOwn, fsw);
      // ── the SHIFT (what makes this SBM rather than a staircase wall):
      // enforce u.n~ = 0 AT THE TRUE WALL, Taylor-shifted to the surrogate
      // face: u_n,true ~ (u + (d.grad)u).n~ with d = -(r-R) n~ (inward).
      // Imposed as a Nitsche-type JUMP PENALTY against the n~-shifted-mirror
      // state: its mass row is IDENTICALLY zero (same rho), so the axis-face
      // mass consistency that a hard n~-mirror violates (the measured Lobatto
      // leak/blowup) is preserved, and the base mirror-HLLC keeps its exact
      // zero dir-mass-flux and upwind dissipation.  At the nose (n~ ~ e_dir)
      // the base flux already enforces it and the penalty is ~0; at OBLIQUE
      // shoulder faces it supplies the true-normal condition the staircase
      // misses -- the standoff/bluntness driver.  u = 0 => exact no-op (rest
      // gate untouched).
      if (grid.ibSbmPen > (real)0.0) {
        // element velocity gradients at this face node's line (as ibSbm==2)
        real gcU[3] = {0,0,0}, gcV[3] = {0,0,0};
        const i32 nrmA = side ? (NNODE-1) : 0;
        {
          real dU=0,t1U=0,t2U=0, dV=0,t1V=0,t2V=0;
          for (i32 m = 0; m < NNODE; m++) {
            i32 nA=dgFaceNode(dir,m,a,b), nB=dgFaceNode(dir,nrmA,m,b), nC=dgFaceNode(dir,nrmA,a,m);
            dU+=c_D[nrmA][m]*sWe[1][nA]; t1U+=c_D[a][m]*sWe[1][nB]; t2U+=c_D[b][m]*sWe[1][nC];
            dV+=c_D[nrmA][m]*sWe[2][nA]; t1V+=c_D[a][m]*sWe[2][nB]; t2V+=c_D[b][m]*sWe[2][nC];
          }
          i32 t1x = (dir==0) ? 1 : 0, t2x = (dir==2) ? 1 : 2;
          gcU[dir]=((real)2.0/h[dir])*dU; gcU[t1x]=((real)2.0/h[t1x])*t1U; gcU[t2x]=((real)2.0/h[t2x])*t2U;
          gcV[dir]=((real)2.0/h[dir])*dV; gcV[t1x]=((real)2.0/h[t1x])*t1V; gcV[t2x]=((real)2.0/h[t2x])*t2V;
        }
        real dseg = grid.ibR - rr;                     // inward to the true wall
        real uS = Wl[1] + (gcU[0]*nx + gcU[1]*ny)*dseg;
        real vS = Wl[2] + (gcV[0]*nx + gcV[1]*ny)*dseg;
        if (grid.ibShift2) {
          // 2nd-order Taylor: + 1/2 dseg^2 (n~.grad)^2 u from the element
          // polynomial's Hessian at this face node (A/B vs the measured FRIB
          // noise-amplification lesson).
          const i32 t1x = (dir==0) ? 1 : 0;
          real ndir = (dir==0) ? nx : ny;             // n~ along the face axis
          real ntan = (dir==0) ? ny : nx;             // n~ along t1
          real jn = (real)2.0/h[dir], jt = (real)2.0/h[t1x];
          real Hnn_u=0, Htt_u=0, Hnt_u=0, Hnn_v=0, Htt_v=0, Hnt_v=0;
          for (i32 m = 0; m < NNODE; m++) {
            real d2n = 0, d2t = 0;                    // D^2 rows via nested D
            for (i32 l = 0; l < NNODE; l++) {
              d2n += c_D[nrmA][l]*c_D[l][m];
              d2t += c_D[a][l]*c_D[l][m];
            }
            i32 nA = dgFaceNode(dir, m, a, b);        // normal line
            i32 nB = dgFaceNode(dir, nrmA, m, b);     // t1 line
            Hnn_u += d2n*sWe[1][nA]; Htt_u += d2t*sWe[1][nB];
            Hnn_v += d2n*sWe[2][nA]; Htt_v += d2t*sWe[2][nB];
            for (i32 l = 0; l < NNODE; l++) {         // cross term
              i32 nC = dgFaceNode(dir, l, m, b);
              Hnt_u += c_D[nrmA][l]*c_D[a][m]*sWe[1][nC];
              Hnt_v += c_D[nrmA][l]*c_D[a][m]*sWe[2][nC];
            }
          }
          real d2u = ndir*ndir*jn*jn*Hnn_u + (real)2.0*ndir*ntan*jn*jt*Hnt_u
                   + ntan*ntan*jt*jt*Htt_u;
          real d2v = ndir*ndir*jn*jn*Hnn_v + (real)2.0*ndir*ntan*jn*jt*Hnt_v
                   + ntan*ntan*jt*jt*Htt_v;
          uS += (real)0.5*dseg*dseg*d2u;
          vS += (real)0.5*dseg*dseg*d2v;
        }
        // MUSCL-style limiter on the SHIFT (the FRIB --iblimit rule): the
        // wall-extrapolated velocity may not create a NEW extremum beyond the
        // normal line's nodal range -- 2nd-order/gradient noise gets clipped
        // instead of detonating, while in-range corrections pass untouched.
        {
          real uLo=(real)1e30,uHi=(real)-1e30,vLo=(real)1e30,vHi=(real)-1e30;
          for (i32 m = 0; m < NNODE; m++) {
            i32 nA = dgFaceNode(dir, m, a, b);
            uLo=fmin(uLo,sWe[1][nA]); uHi=fmax(uHi,sWe[1][nA]);
            vLo=fmin(vLo,sWe[2][nA]); vHi=fmax(vHi,sWe[2][nA]);
          }
          uS = fmin(fmax(uS, uLo), uHi);
          vS = fmin(fmax(vS, vLo), vHi);
        }
        real unS = uS*nx + vS*ny;                      // shifted true-normal velocity
        real WgN[5] = { Wl[0], Wl[1] - (real)2.0*unS*nx,
                               Wl[2] - (real)2.0*unS*ny, Wl[3], Wl[4] };
        real lamW = fabs(Wl[1]) + fabs(Wl[2]) + fabs(Wl[3])
                  + dgSoundSpeed(Wl[4], Wl[0]);
        real sig = grid.ibSbmPen * (real)0.5 * lamW;
        if (side) dgJumpPenalty(Wl, WgN, sig, fsw);    // mass row exactly 0
        else      dgJumpPenalty(WgN, Wl, sig, fsw);
      }
      for (i32 q = 0; q < 5; q++) { fs[q] = fsw[q]; WmeB[q] = Wp[q]; }
      return;
    }
  }

  if (nSame != bEmpty) {
    real Wnp[5], Wnnn[5];                 // neighbour projected + nearest-node
    { i32 ndn = dgFaceNode(dir, nnNbr, a, b); real U[5];
      for (i32 q=0;q<5;q++) U[q]=grid.getField(D_RHO+q)[(u64)nSame*blockSizeTot+ndn];
      dgConsToPrimSane(U, Wnnn); }
    // SBM cut cells (class IB_FLUID but phiMin < 0): entropy-projected traces
    // extrapolate through their solid-filled nodes -- present nearest-node
    // instead (both sides symmetric: each side applies the same rule to its
    // own data, so the shared f* stays single-valued).
    real WpEff[5], WnnEff[5];
    for (i32 q = 0; q < 5; q++) { WpEff[q] = Wp[q]; WnnEff[q] = Wnn[q]; }
    bool nbrCutNN = false;
    if (grid.ibSbm) {
      real pmn, pmx;
      dgIbPhiRangeBox(grid, ib*h[0], (ib+1)*h[0], jb*h[1], (jb+1)*h[1], pmn, pmx);
      if (pmn < (real)0.0)
        for (i32 q = 0; q < 5; q++) WpEff[q] = Wnn[q];   // cut: nearest-node trace
      real x0 = nib*h[0], y0 = njb*h[1];
      dgIbPhiRangeBox(grid, x0, x0+h[0], y0, y0+h[1], pmn, pmx);
      nbrCutNN = (pmn < (real)0.0);
    }
    // --ibevolve: an IB_CUT element COMPUTES its own face fluxes (unlike a
    // pure ghost), so any face touching a cut element must see the IDENTICAL
    // state pair from both sides or f* stops being single-valued
    // (conservation).  Rule: either side IB_CUT -> BOTH states nearest-node
    // (the entropy-projected extrapolation through mixed evolved/filled cut
    // nodes is exactly the trace the slam fix distrusts anyway).
    if (grid.ibEvolve && (grid.ibClassList[bIdx] == IB_CUT ||
                          grid.ibClassList[nSame] == IB_CUT)) {
      for (i32 q = 0; q < 5; q++) WpEff[q] = Wnn[q];
      nbrCutNN = true;
    }
    if (grid.ibOn && grid.ibClassList[nSame] != IB_FLUID) {
      // IB GHOST neighbour: present its NEAREST-NODE state as the trace.  The
      // FRIB fill constructs the wall state the face Riemann must see; on
      // Lobatto the boundary node IS that state, but the entropy-projected
      // extrapolation through a ghost's interior Gauss nodes re-rings the
      // piston/star data at the impulsive start (measured: M=3 iter-2
      // detonation).  Constant extrapolation costs O(0.1h) wall offset and
      // keeps the contract.
      for (i32 q = 0; q < 5; q++) Wnp[q] = Wnnn[q];
    } else if (nbrCutNN) {
      for (i32 q = 0; q < 5; q++) Wnp[q] = Wnnn[q];
    } else
    dgGaussNbrTrace(grid, nSame, dir, a, b, tNbr, nnNbr, Wnp);
    real aN = grid.subFv ? grid.getField(D_SCRATCH)[(u64)nSame*blockSizeTot + 6] : (real)0.0;
    real af = fmax(aOwn, aN);                    // symmetric -> identical on both sides
    real WnB[5];
    for (i32 q = 0; q < 5; q++) {
      WmeB[q] = ((real)1.0-af)*WpEff[q] + af*WnnEff[q];
      WnB[q]  = ((real)1.0-af)*Wnp[q]   + af*Wnnn[q];
    }
    real afFl = (grid.rusFace == 1) ? (real)1.0 : af;   // troubled -> Rusanov interface flux
    if (side) dgIfaceFlux(grid, WmeB, WnB, dir, afFl, fs);
    else      dgIfaceFlux(grid, WnB, WmeB, dir, afFl, fs);
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)nSame*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, WmeB, WnB)
                      : dgPenaltySigma(grid, nuOwn, nuN, WnB, WmeB);
      if (side) dgJumpPenalty(WmeB, WnB, sig, fs);
      else      dgJumpPenalty(WnB, WmeB, sig, fs);
    }
    if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, nSame, WmeB, WnB, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, nSame, WnB, WmeB, dir, fs);
      }
    return;
  }

  // ── nonconforming 2:1 faces (Gauss mortar) ──────────────────────────────
  // Same protocol as the Lobatto dgFaceLift mortar, with the face traces built
  // by ENTROPY-VARIABLE normal extrapolation first: the coarse side's trace at
  // a fine point is the tangential c_I interpolation of its entropy face array
  // (identical on both sides -> pointwise f* matches bitwise); the coarse face
  // flux is the exact-L2 c_R projection of the pointwise mortar fluxes (mean-
  // adjoint identity holds on the Gauss weights -> discretely conservative).
  // Troubled-cell constant extrapolation rides along: each side's nearest-node
  // -plane array is tangentially interpolated the same way and blended by
  // af = max(alpha_coarse, alpha_fine) -- symmetric, single-valued f*.
  const i32 t1b = (dir==0) ? njb : nib;
  const i32 t2b = (dir==2) ? njb : nkb;
  const bool zIdent = (grid.pseudo2D != 0) && (dir != 2);
  const real *tvecMe = side ? c_tR : c_tL;    // MY trace projection toward this face
  const i32  nnMe    = side ? (NNODE-1) : 0;  // my nearest-node plane

  // ── coarser neighbour: I am the fine side ────────────────────────────────
  i32 cIdxN = (lvl > 0) ? grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                           grid.pseudo2D ? nkb : (nkb>>1)))
                        : bEmpty;
  if (cIdxN != bEmpty) {
    const i32 s1 = t1b & 1;
    const i32 s2 = zIdent ? 0 : (t2b & 1);
    real Vcf[5][NNODE*NNODE], Pcf[5][NNODE*NNODE];
    dgGaussGatherFace(grid, cIdxN, dir, tNbr, nnNbr, Vcf, Pcf);
    real WcH[5], WcN[5];
    dgTraceAt(Vcf, s1, s2, a, b, zIdent, WcH);   // primitive tangential interp
    dgSanitizePrim(WcH);                         // (can overshoot at shocks)
    dgTraceAt(Pcf, s1, s2, a, b, zIdent, WcN);
    dgSanitizePrim(WcN);
    real aN = grid.subFv ? grid.getField(D_SCRATCH)[(u64)cIdxN*blockSizeTot + 6] : (real)0.0;
    real af = fmax(aOwn, aN);
    real Wc[5];
    for (i32 q = 0; q < 5; q++) {
      WmeB[q] = ((real)1.0-af)*Wp[q]  + af*Wnn[q];
      Wc[q]   = ((real)1.0-af)*WcH[q] + af*WcN[q];
    }
    real afFl = (grid.rusFace == 1) ? (real)1.0 : af;
    if (side) dgIfaceFlux(grid, WmeB, Wc, dir, afFl, fs);
    else      dgIfaceFlux(grid, Wc, WmeB, dir, afFl, fs);
    if (grid.avOn) {
      real nuN = grid.getField(D_SCRATCH)[(u64)cIdxN*blockSizeTot];
      real sig = side ? dgPenaltySigma(grid, nuOwn, nuN, WmeB, Wc)
                      : dgPenaltySigma(grid, nuOwn, nuN, Wc, WmeB);
      if (side) dgJumpPenalty(WmeB, Wc, sig, fs);
      else      dgJumpPenalty(Wc, WmeB, sig, fs);
    }
    if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
        if (side) dgDpJumpPenalty(grid, bIdx, cIdxN, WmeB, Wc, dir, fs);
        else      dgDpJumpPenalty(grid, bIdx, cIdxN, Wc, WmeB, dir, fs);
      }
    return;
  }

  // ── finer neighbours: I am the coarse (mortar) side ──────────────────────
  {
    // my own face arrays from shared -- the SAME construction the fine sides
    // gather from global (dgConsToPrimSane -> dgEntVars -> tvec), so the
    // pointwise traces match bitwise.
    real Vcf[5][NNODE*NNODE], Pcf[5][NNODE*NNODE];
    for (i32 c2 = 0; c2 < NNODE; c2++)
      for (i32 c1 = 0; c1 < NNODE; c1++) {
        real v[5] = {0,0,0,0,0};
        for (i32 m = 0; m < NNODE; m++) {
          i32 nd = dgFaceNode(dir, m, c1, c2);
          for (i32 q = 0; q < 5; q++) v[q] += tvecMe[m]*sVe[q][nd];
        }
        real Wt[5];
        dgEntVarsToPrim(v, Wt);                  // prim face trace (as the gather)
        i32 ndn = dgFaceNode(dir, nnMe, c1, c2);
        real rlo=(real)1e30, rhi=(real)0.0, plo=(real)1e30, phi=(real)0.0;
        for (i32 m = 0; m < NNODE; m++) {
          i32 nd = dgFaceNode(dir, m, c1, c2);
          rlo=fmin(rlo,sWe[0][nd]); rhi=fmax(rhi,sWe[0][nd]);
          plo=fmin(plo,sWe[4][nd]); phi=fmax(phi,sWe[4][nd]);
        }
        if (!(Wt[0] < (real)2.0*rhi) || Wt[0] < (real)0.5*rlo ||
            !(Wt[4] < (real)2.0*phi) || Wt[4] < (real)0.5*plo)
          for (i32 q = 0; q < 5; q++) Wt[q] = sWe[q][ndn];   // overflow guard
        for (i32 q = 0; q < 5; q++) {
          Vcf[q][c1 + NNODE*c2] = Wt[q];
          Pcf[q][c1 + NNODE*c2] = sWe[q][ndn];
        }
      }

    real Fs[5] = {0,0,0,0,0};
    const i32 s2max = zIdent ? 1 : 2;
    for (i32 s2 = 0; s2 < s2max; s2++)
      for (i32 s1 = 0; s1 < 2; s1++) {
        i32 cib, cjb, ckb;
        if (dir == 0)      { cib = 2*nib + (side ? 0 : 1); cjb = 2*t1b + s1;
                             ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
        else if (dir == 1) { cjb = 2*njb + (side ? 0 : 1); cib = 2*t1b + s1;
                             ckb = grid.pseudo2D ? nkb : 2*t2b + s2; }
        else               { ckb = 2*nkb + (side ? 0 : 1); cib = 2*t1b + s1;
                             cjb = 2*t2b + s2; }
        i32 fIdx = grid.getBlockIdx(grid.encode(lvl+1, cib, cjb, ckb));
        if (fIdx == bEmpty) continue;   // grading guarantees existence; guard anyway

        real aF = grid.subFv ? grid.getField(D_SCRATCH)[(u64)fIdx*blockSizeTot + 6] : (real)0.0;
        real af = fmax(aOwn, aF);
        real afFl = (grid.rusFace == 1) ? (real)1.0 : af;
        real nuF = grid.avOn ? grid.getField(D_SCRATCH)[(u64)fIdx*blockSizeTot] : (real)0.0;

        for (i32 fb = (zIdent ? b : 0); fb < (zIdent ? b+1 : NNODE); fb++) {
          real wtB = zIdent ? (real)1.0 : c_R[s2][b][fb];
          for (i32 fa = 0; fa < NNODE; fa++) {
            // my trace at the fine point (H-O + nearest-node, blended)
            real ToH[5], ToN[5], To[5];
            dgTraceAt(Vcf, s1, s2, fa, fb, zIdent, ToH);
            dgSanitizePrim(ToH);
            dgTraceAt(Pcf, s1, s2, fa, fb, zIdent, ToN);
            dgSanitizePrim(ToN);
            // the fine element's own face trace (H-O projected + nearest node)
            real WfH[5], WfN[5], Wf[5];
            dgGaussNbrTrace(grid, fIdx, dir, fa, fb, tNbr, nnNbr, WfH);
            { i32 ndn = dgFaceNode(dir, nnNbr, fa, fb); real U[5];
              for (i32 q=0;q<5;q++) U[q]=grid.getField(D_RHO+q)[(u64)fIdx*blockSizeTot+ndn];
              dgConsToPrimSane(U, WfN); }
            for (i32 q = 0; q < 5; q++) {
              To[q] = ((real)1.0-af)*ToH[q] + af*ToN[q];
              Wf[q] = ((real)1.0-af)*WfH[q] + af*WfN[q];
            }
            real fsp[5];
            if (side) dgIfaceFlux(grid, To, Wf, dir, afFl, fsp);
            else      dgIfaceFlux(grid, Wf, To, dir, afFl, fsp);
            if (grid.avOn) {
              real sig = side ? dgPenaltySigma(grid, nuOwn, nuF, To, Wf)
                              : dgPenaltySigma(grid, nuOwn, nuF, Wf, To);
              if (side) dgJumpPenalty(To, Wf, sig, fsp);
              else      dgJumpPenalty(Wf, To, sig, fsp);
            }
            if (grid.dpSbp > (real)0.0 && grid.dpFace > (real)0.0) {
              if (side) dgDpJumpPenalty(grid, bIdx, fIdx, To, Wf, dir, fsp);
              else      dgDpJumpPenalty(grid, bIdx, fIdx, Wf, To, dir, fsp);
            }
            if (grid.dbgChecks >= 3) {
              bool bad = false;
              for (i32 q = 0; q < 5; q++) bad = bad || !isfinite(fsp[q]);
              if (bad && atomicCAS(grid.nanCnt, 0, 2) == 0)
                printf("[nanprobe-mortar] dir %d side %d sub(%d,%d) fp(%d,%d) "
                       "To=%.3e,%.3e,%.3e,%.3e,%.3e Wf=%.3e,%.3e,%.3e,%.3e,%.3e af=%.2f\n",
                       dir, side, s1, s2, fa, fb,
                       (double)To[0],(double)To[1],(double)To[2],(double)To[3],(double)To[4],
                       (double)Wf[0],(double)Wf[1],(double)Wf[2],(double)Wf[3],(double)Wf[4],
                       (double)af);
            }
            real coef = c_R[s1][a][fa] * wtB;
            for (i32 q = 0; q < 5; q++) Fs[q] += coef*fsp[q];
          }
        }
      }
    for (i32 q = 0; q < 5; q++) { fs[q] = Fs[q]; WmeB[q] = Wp[q]; }
  }
}

__global__ void dgRhsGaussKernel(DgSolver &grid, real t) {
  __shared__ real sW  [DG_EPB][5][blockSizeTot];   // sanitized primitives
  __shared__ real sV  [DG_EPB][5][blockSizeTot];   // entropy variables (face projection)
  __shared__ real sLam[DG_EPB][blockSizeTot];      // per-node wave speed (dt reduce)

  const i32 ell = threadIdx.x / blockSizeTot;
  const i32 nd  = threadIdx.x % blockSizeTot;
  const i32 i = nd % NNODE, j = (nd/NNODE) % NNODE, k = nd/(NNODE*NNODE);

  for (i32 base = blockIdx.x*DG_EPB; base < grid.hashTable.nKeys; base += gridDim.x*DG_EPB) {
    const i32 bIdx = base + ell;
    u64 loc = (bIdx < grid.hashTable.nKeys) ? grid.bLocList[bIdx] : kEmpty;
    const bool active = (loc != kEmpty) && dgIbLive(grid, bIdx);
    i32 lvl = 0, ib = 0, jb = 0, kb = 0;
    if (active) grid.decode(loc, lvl, ib, jb, kb);
    real h[3] = {1,1,1};
    if (active) dgElemSize(grid, lvl, h);
    const real jacx = (real)2.0/h[0], jacy = (real)2.0/h[1], jacz = (real)2.0/h[2];

    // ── phase 1: load, sanitize, primitives + entropy variables -> shared ──
    if (active) {
      real U[5], W[5], v[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      dgEntVars(W, v);
      for (i32 q = 0; q < 5; q++) { sW[ell][q][nd] = W[q]; sV[ell][q][nd] = v[q]; }
    }
    __syncthreads();

    real R[5] = {0,0,0,0,0};
    real lamNode = 0.0;

    // first-NaN probe (--debug 3): phase-tagged, first-wins, prints once.
    // 0 = state entering the RHS, 1 = volume+DP, 2/3/4 = x/y/z faces (+FV),
    // 5 = bulk, 6 = NSFR filter.
    #define DG_NANPROBE(PH) do { \
      if (grid.dbgChecks >= 3 && active) { \
        bool bad = false; \
        for (i32 q = 0; q < 5; q++) bad = bad || !isfinite(R[q]); \
        if (bad && atomicCAS(grid.nanCnt, 0, 1) == 0) { \
          i32 lvv, ibb, jbb, kbb; grid.decode(grid.bLocList[bIdx], lvv, ibb, jbb, kbb); \
          printf("[nanprobe] iter %d phase %d lvl %d elem(%d,%d) node %d " \
                 "R=%.3e,%.3e,%.3e,%.3e,%.3e U=%.3e,%.3e,%.3e,%.3e,%.3e\n", \
                 grid.iter, (PH), lvv, ibb, jbb, nd, \
                 (double)R[0],(double)R[1],(double)R[2],(double)R[3],(double)R[4], \
                 (double)grid.getField(D_RHO)[bIdx*blockSizeTot+nd], \
                 (double)grid.getField(D_RHOU)[bIdx*blockSizeTot+nd], \
                 (double)grid.getField(D_RHOV)[bIdx*blockSizeTot+nd], \
                 (double)grid.getField(D_RHOW)[bIdx*blockSizeTot+nd], \
                 (double)grid.getField(D_RHOE)[bIdx*blockSizeTot+nd]); \
        } \
      } \
    } while (0)
    if (grid.dbgChecks >= 3 && active) {
      bool badU = false;
      for (i32 q = 0; q < 5; q++)
        badU = badU || !isfinite(grid.getField(D_RHO+q)[bIdx*blockSizeTot+nd]);
      if (badU && atomicCAS(grid.nanCnt, 0, 1) == 0) {
        i32 lvv, ibb, jbb, kbb; grid.decode(grid.bLocList[bIdx], lvv, ibb, jbb, kbb);
        printf("[nanprobe] iter %d phase 0 (STATE) lvl %d elem(%d,%d) node %d cls %d\n",
               grid.iter, lvv, ibb, jbb, nd, grid.ibClassList[bIdx]);
      }
    }

    if (active) {
      real Wi[5];
      for (i32 q = 0; q < 5; q++) Wi[q] = sW[ell][q][nd];
      const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
      const i32 ndY0 = i + k*NNODE*NNODE;
      const i32 ndZ0 = i + j*NNODE;

      // ── volume: EC flux differencing (generalized-SBP D) ────────────────
      real ax[5]={0,0,0,0,0}, ay[5]={0,0,0,0,0}, az[5]={0,0,0,0,0};
      for (i32 m = 0; m < NNODE; m++) {
        real Wm[5], Fs[5];
        for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndX0 + m];
        dgEcFluxAxis(Wi, Wm, 0, Fs, grid.ecVolume == 2);
        for (i32 q = 0; q < 5; q++) ax[q] += c_D[i][m]*Fs[q];
        for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndY0 + m*NNODE];
        dgEcFluxAxis(Wi, Wm, 1, Fs, grid.ecVolume == 2);
        for (i32 q = 0; q < 5; q++) ay[q] += c_D[j][m]*Fs[q];
        if (!grid.pseudo2D) {
          for (i32 q = 0; q < 5; q++) Wm[q] = sW[ell][q][ndZ0 + m*NNODE*NNODE];
          dgEcFluxAxis(Wi, Wm, 2, Fs, grid.ecVolume == 2);
          for (i32 q = 0; q < 5; q++) az[q] += c_D[k][m]*Fs[q];
        }
      }
      for (i32 q = 0; q < 5; q++)
        R[q] = -(real)2.0*(jacx*ax[q] + jacy*ay[q] + jacz*az[q]);

      // ── dual-pairing SBP volume upwinding (see dgRhsKernel phase 2c;
      //    identical construction -- sV already holds the entropy vars) ─────
      if (grid.dpSbp > (real)0.0) {
        real gt[5];
        for (i32 q = 0; q < 5; q++)
          gt[q] = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 8 + q];
        for (i32 q = 0; q < 5; q++) {
          real Sx = 0, Sy = 0, Sz = 0;
          for (i32 m = 0; m < NNODE; m++) {
            real wp = c_w[m]*c_dpPhi[m];
            Sx += wp*sV[ell][q][ndX0 + m];
            Sy += wp*sV[ell][q][ndY0 + m*NNODE];
            if (!grid.pseudo2D) Sz += wp*sV[ell][q][ndZ0 + m*NNODE*NNODE];
          }
          R[q] -= grid.dpSbp*gt[q]*(c_dpPhi[i]*Sx/h[0] + c_dpPhi[j]*Sy/h[1]
                 + (grid.pseudo2D ? (real)0.0 : c_dpPhi[k]*Sz/h[2]));
        }
      }

      DG_NANPROBE(1);

      // ── surface (FR correction, all nodes) + subcell-FV volume ──────────
      const real alpha = grid.subFv
                       ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0;
      real Rfv[5] = {0,0,0,0,0};
      real Wm[5];

      // x faces (tangential j,k; correction weights gpL[i]/gpR[i])
      {
        real WmeL[5], WmeR[5], LnnL[5], LnnR[5], WbL[5], WbR[5];
        real ftL[5], ftR[5], fL[5], fR[5];
        dgGaussMyTrace(sV[ell], sW[ell], 0, j, k, c_tL, 0, WmeL);
        dgGaussMyTrace(sV[ell], sW[ell], 0, j, k, c_tR, NNODE-1, WmeR);
        for (i32 q=0;q<5;q++){ LnnL[q]=sW[ell][q][ndX0+0]; LnnR[q]=sW[ell][q][ndX0+(NNODE-1)]; }
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 0, 0, j, k, WmeL, LnnL, h, t, fL, WbL);
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 0, 1, j, k, WmeR, LnnR, h, t, fR, WbR);
        dgGaussBndFlux(sW[ell], 0, j, k, grid.ecVolume == 2, ftL, ftR);
        for (i32 q = 0; q < 5; q++)
          R[q] += -jacx*(c_gpL[i]*(fL[q]-ftL[q]) + c_gpR[i]*(fR[q]-ftR[q]));
        if (alpha > (real)0.0) {
          real fLs[5], fRs[5];
          if (i == 0) { for (i32 q=0;q<5;q++) fLs[q]=fL[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndX0+(i-1)]; dgRusanovAxis(Wm, Wi, 0, fLs); }
          if (i == NNODE-1) { for (i32 q=0;q<5;q++) fRs[q]=fR[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndX0+(i+1)]; dgRusanovAxis(Wi, Wm, 0, fRs); }
          for (i32 q=0;q<5;q++) Rfv[q] -= jacx*c_winv[i]*(fRs[q]-fLs[q]);
        }
      }
      DG_NANPROBE(2);
      // y faces (tangential i,k; weights gpL[j]/gpR[j])
      {
        real WmeL[5], WmeR[5], LnnL[5], LnnR[5], WbL[5], WbR[5];
        real ftL[5], ftR[5], fL[5], fR[5];
        dgGaussMyTrace(sV[ell], sW[ell], 1, i, k, c_tL, 0, WmeL);
        dgGaussMyTrace(sV[ell], sW[ell], 1, i, k, c_tR, NNODE-1, WmeR);
        for (i32 q=0;q<5;q++){ LnnL[q]=sW[ell][q][ndY0+0]; LnnR[q]=sW[ell][q][ndY0+(NNODE-1)*NNODE]; }
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 1, 0, i, k, WmeL, LnnL, h, t, fL, WbL);
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 1, 1, i, k, WmeR, LnnR, h, t, fR, WbR);
        dgGaussBndFlux(sW[ell], 1, i, k, grid.ecVolume == 2, ftL, ftR);
        for (i32 q = 0; q < 5; q++)
          R[q] += -jacy*(c_gpL[j]*(fL[q]-ftL[q]) + c_gpR[j]*(fR[q]-ftR[q]));
        if (alpha > (real)0.0) {
          real fLs[5], fRs[5];
          if (j == 0) { for (i32 q=0;q<5;q++) fLs[q]=fL[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndY0+(j-1)*NNODE]; dgRusanovAxis(Wm, Wi, 1, fLs); }
          if (j == NNODE-1) { for (i32 q=0;q<5;q++) fRs[q]=fR[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndY0+(j+1)*NNODE]; dgRusanovAxis(Wi, Wm, 1, fRs); }
          for (i32 q=0;q<5;q++) Rfv[q] -= jacy*c_winv[j]*(fRs[q]-fLs[q]);
        }
      }
      DG_NANPROBE(3);
      // z faces (tangential i,j; weights gpL[k]/gpR[k]) -- 3D only
      if (!grid.pseudo2D) {
        real WmeL[5], WmeR[5], LnnL[5], LnnR[5], WbL[5], WbR[5];
        real ftL[5], ftR[5], fL[5], fR[5];
        dgGaussMyTrace(sV[ell], sW[ell], 2, i, j, c_tL, 0, WmeL);
        dgGaussMyTrace(sV[ell], sW[ell], 2, i, j, c_tR, NNODE-1, WmeR);
        for (i32 q=0;q<5;q++){ LnnL[q]=sW[ell][q][ndZ0+0]; LnnR[q]=sW[ell][q][ndZ0+(NNODE-1)*NNODE*NNODE]; }
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 2, 0, i, j, WmeL, LnnL, h, t, fL, WbL);
        dgGaussFaceFlux(grid, sV[ell], sW[ell], bIdx, lvl, ib, jb, kb, 2, 1, i, j, WmeR, LnnR, h, t, fR, WbR);
        dgGaussBndFlux(sW[ell], 2, i, j, grid.ecVolume == 2, ftL, ftR);
        for (i32 q = 0; q < 5; q++)
          R[q] += -jacz*(c_gpL[k]*(fL[q]-ftL[q]) + c_gpR[k]*(fR[q]-ftR[q]));
        if (alpha > (real)0.0) {
          real fLs[5], fRs[5];
          if (k == 0) { for (i32 q=0;q<5;q++) fLs[q]=fL[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndZ0+(k-1)*NNODE*NNODE]; dgRusanovAxis(Wm, Wi, 2, fLs); }
          if (k == NNODE-1) { for (i32 q=0;q<5;q++) fRs[q]=fR[q]; }
          else { for (i32 q=0;q<5;q++) Wm[q]=sW[ell][q][ndZ0+(k+1)*NNODE*NNODE]; dgRusanovAxis(Wi, Wm, 2, fRs); }
          for (i32 q=0;q<5;q++) Rfv[q] -= jacz*c_winv[k]*(fRs[q]-fLs[q]);
        }
      }

      if (alpha > (real)0.0)
        for (i32 q = 0; q < 5; q++) R[q] = ((real)1.0-alpha)*R[q] + alpha*Rfv[q];
      DG_NANPROBE(4);

      real c = dgSoundSpeed(Wi[4], Wi[0]);
      lamNode = fabs(Wi[1]) + fabs(Wi[2]) + fabs(Wi[3]) + c;
    }

    sLam[ell][nd] = lamNode;
    __syncthreads();
    real lam_e = 0;
    for (i32 m = 0; m < blockSizeTot; m++) lam_e = fmax(lam_e, sLam[ell][m]);

    // ── sensor-gated BULK (dilatation) viscosity (see dgRhsKernel phase 4b;
    //    identical construction, sV reused as the staging slab -- the surface
    //    and DP phases are done with it) ──────────────────────────────────────
    if (grid.bulkC > (real)0.0) {
      const i32 ndX0 = j*NNODE + k*NNODE*NNODE;
      const i32 ndY0 = i + k*NNODE*NNODE;
      const i32 ndZ0 = i + j*NNODE;
      real theta_e = active
                   ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1] : (real)0.0;
      if (active) {
        real du = 0, dv = 0, dw = 0;
        for (i32 m = 0; m < NNODE; m++) {
          du += c_D[i][m]*sW[ell][1][ndX0 + m];
          dv += c_D[j][m]*sW[ell][2][ndY0 + m*NNODE];
          if (!grid.pseudo2D) dw += c_D[k][m]*sW[ell][3][ndZ0 + m*NNODE*NNODE];
        }
        real divu = jacx*du + jacy*dv + (grid.pseudo2D ? (real)0.0 : jacz*dw);
        real lenp = h[0]/(real)(2*dgOrder+1);   // the AV length scale
        real beta = grid.bulkC * theta_e * lenp * lam_e * sW[ell][0][nd];
        sV[ell][0][nd] = beta*divu;
      }
      __syncthreads();
      if (active) {
        real sxm = 0, sxe = 0, sym = 0, sye = 0, szm = 0, sze = 0;
        for (i32 m = 0; m < NNODE; m++) {
          real bx = sV[ell][0][ndX0 + m];
          sxm += c_w[m]*c_D[m][i]*bx;
          sxe += c_w[m]*c_D[m][i]*bx*sW[ell][1][ndX0 + m];
          real by = sV[ell][0][ndY0 + m*NNODE];
          sym += c_w[m]*c_D[m][j]*by;
          sye += c_w[m]*c_D[m][j]*by*sW[ell][2][ndY0 + m*NNODE];
          if (!grid.pseudo2D) {
            real bz = sV[ell][0][ndZ0 + m*NNODE*NNODE];
            szm += c_w[m]*c_D[m][k]*bz;
            sze += c_w[m]*c_D[m][k]*bz*sW[ell][3][ndZ0 + m*NNODE*NNODE];
          }
        }
        R[1] -= jacx*c_winv[i]*sxm;
        R[2] -= jacy*c_winv[j]*sym;
        R[4] -= jacx*c_winv[i]*sxe + jacy*c_winv[j]*sye;
        if (!grid.pseudo2D) {
          R[3] -= jacz*c_winv[k]*szm;
          R[4] -= jacz*c_winv[k]*sze;
        }
      }
    }

    DG_NANPROBE(5);
    // ── NSFR residual filter (see dgRhsKernel phase 5; sV reused as staging) ──
    if (grid.nsfr > (real)0.0) {
      // gate by (1 - alpha): the filter belongs to the HIGH-ORDER scheme only.
      // Filtering the blended residual corrupts the top mode of the subcell-FV
      // fallback exactly where positivity depends on it (measured: 5-level
      // blast blew at t=0.70 with the unconditional filter, completes gated).
      const real sigE = grid.nsfr*((real)1.0 - (grid.subFv && active
                      ? grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] : (real)0.0));
      const i32 nd3 = grid.pseudo2D ? 2 : 3;
      for (i32 d3 = 0; d3 < nd3; d3++) {
        __syncthreads();
        if (active) for (i32 q = 0; q < 5; q++) sV[ell][q][nd] = R[q];
        __syncthreads();
        if (active) {
          const i32 idx  = (d3==0) ? i : ((d3==1) ? j : k);
          const i32 base = (d3==0) ? (j*NNODE + k*NNODE*NNODE)
                         : ((d3==1) ? (i + k*NNODE*NNODE) : (i + j*NNODE));
          const i32 str  = (d3==0) ? 1 : ((d3==1) ? NNODE : NNODE*NNODE);
          for (i32 q = 0; q < 5; q++) {
            real S = 0;
            for (i32 m = 0; m < NNODE; m++)
              S += c_w[m]*c_dpPhi[m]*sV[ell][q][base + m*str];
            R[q] -= sigE*c_dpPhi[idx]*S;
          }
        }
      }
    }

    DG_NANPROBE(6);
    #undef DG_NANPROBE
    if (active) {
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHS+q)[bIdx*blockSizeTot + nd] = R[q];
      real hmin = fmin(h[0], grid.pseudo2D ? h[0] : fmin(h[1], h[2]));
      grid.getField(D_LAM)[bIdx*blockSizeTot + nd] =
          hmin/(fmax(lam_e, (real)1e-10)*(real)NNODE);
    }
    __syncthreads();
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
    i32 cls = grid.ibClassList[bIdx];
    if (cls != IB_FLUID && cls != IB_CUT) continue;   // ghosts hold their fill
    if (cls == IB_CUT) {
      // evolving cut element (--ibevolve): only the FLUID-side nodes (phi > 0,
      // split EXACTLY at 0 -- a sliver margin measured TRANSPARENT: mixed
      // constructions along one face defeat the reflection) integrate; the
      // solid-side nodes keep the FRIB fill applied after this stage.
      GET_CELL_INDICES
      i32 lvl, ib, jb, kb;
      grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);
      real h[3]; dgElemSize(grid, lvl, h);
      if (dgIbPhi(grid, dgNodePos(h[0], ib, i), dgNodePos(h[1], jb, j))
          <= (real)0.0) continue;
    }
    real U[5];
    for (i32 q = 0; q < 5; q++) {
      real q0 = grid.getField(D_Q0+q)[cIdx];
      real qc = grid.getField(D_RHO+q)[cIdx];
      real L  = grid.getField(D_RHS+q)[cIdx];
      if (stage == 0)      U[q] = q0 + dt*L;
      else if (stage == 1) U[q] = (real)0.75*q0 + (real)0.25*(qc + dt*L);
      else                 U[q] = (real)(1.0/3.0)*q0 + (real)(2.0/3.0)*(qc + dt*L);
    }
    // dgSanitizeCons floors rho and p.  On a MODAL cut block these slots are
    // coefficients, not a state -- clamping one would silently corrupt the
    // element (and slot 0 aside, they are not even sign-definite).
    if (!(grid.cutOn && grid.cutModal && grid.blkCut && grid.blkCut[bIdx] >= 0))
      dgSanitizeCons(U);
    for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
  }
}

// MOOD: reset the per-element blend factor alpha (slot 6) to 0 -> the pure
// DG attempt.  Detection then raises it to 1 (first-order FV) where needed.
__global__ void dgMoodResetKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] = (real)0.0;
  }
}

// MOOD detection: form the candidate RK update from (Q0, RHO, RHS) WITHOUT
// committing it (RHO stays the stage-start), and flag the element for the
// first-order FV redo (slot 6 = 1) if ANY node is non-finite or has density/
// pressure below the relative floors.  A cell already at FV (alpha==1) is not
// re-flagged -- that is the bottom of the DG->FV cascade.  Reads only local
// (own-element) DOF; the redo stays local because HLLC faces are unchanged.
__global__ void dgMoodDetectKernel(DgSolver &grid, i32 stage, real dt) {
  const real rhoLo = grid.moodRho * grid.cScale[0];           // undershoot floor
  const real pLo   = grid.moodP   * ((real)1.0/dgGam);        // freestream p ref
  const real rhoHi = (real)100.0  * grid.cScale[0];           // OVERshoot cap: a
  const real pHi   = (real)1000.0 * ((real)1.0/dgGam);        // real M=3 shock is
  // rho~4x, p~10x -- 100x/1000x safely catches Gibbs overshoots the positivity
  // floor misses (measured: rho->1e11 at a forming shock), fully local (no
  // neighbour reads), so the FV redo stays local.
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    const i32 cls = grid.ibClassList[bIdx];
    if (cls != IB_FLUID && cls != IB_CUT) continue;
    if (grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] > (real)0.5) continue;  // already FV
    i32 lvlC = 0, ibC = 0, jbC = 0, kbC = 0;
    real hC[3] = {1,1,1};
    if (cls == IB_CUT) {   // test only the EVOLVED (fluid-side) nodes: the
      // solid-side candidate update is meaningless (the fill overwrites it)
      grid.decode(grid.bLocList[bIdx], lvlC, ibC, jbC, kbC);
      dgElemSize(grid, lvlC, hC);
    }
    bool bad = false;
    for (i32 nd = 0; nd < blockSizeTot && !bad; nd++) {
      if (cls == IB_CUT) {
        i32 i = nd % NNODE, j = (nd/NNODE)%NNODE;
        if (dgIbPhi(grid, dgNodePos(hC[0], ibC, i), dgNodePos(hC[1], jbC, j))
            <= (real)0.0) continue;
      }
      i32 c = bIdx*blockSizeTot + nd;
      real U[5];
      for (i32 q = 0; q < 5; q++) {
        real q0 = grid.getField(D_Q0+q)[c];
        real qc = grid.getField(D_RHO+q)[c];
        real L  = grid.getField(D_RHS+q)[c];
        if (stage == 0)      U[q] = q0 + dt*L;
        else if (stage == 1) U[q] = (real)0.75*q0 + (real)0.25*(qc + dt*L);
        else                 U[q] = (real)(1.0/3.0)*q0 + (real)(2.0/3.0)*(qc + dt*L);
      }
      real p = dgPressureFromCons(U);
      if (!isfinite(U[0]) || !isfinite(U[4]) || !isfinite(p)
          || U[0] < rhoLo || U[0] > rhoHi || p < pLo || p > pHi)
        bad = true;
    }
    if (bad) grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 6] = (real)1.0;
  }
}

// dual-pairing SBP upwind parameters (arXiv 2411.06629 Appendix C): per element,
//   gamma_1 = max_x lam/(d2eta/drho2),  gamma_(1+m) = max_x lam/(d2eta/dm_m2),
//   gamma_5 = max_x 2 M^2 lam / ((1+M^2) d2eta/dE2),
// lam = |u| + c, M^2 = |u|^2 rho/(gam p), eta = -rho ln(p/rho^gam) the
// THERMODYNAMIC entropy.  Diagonal Hessian entries (derived from the entropy
// variables g = d eta/dU):
//   d2eta/drho2 = gam/rho + (gam-1)^2 rho q^4/(4 p^2),   q^2 = |u|^2
//   d2eta/dm_m2 = (gam-1)/p * (1 + (gam-1) rho u_m^2 / p)
//   d2eta/dE2   = (gam-1)^2 rho / p^2.
// Gamma must be CONSTANT per element (the conservation proof pairs
// (D+-D-)^T 1 against Gamma g).  Stored as gamma~_q = (gam-1)*gamma_q in
// SCRATCH slots 8..12 so the RHS can pair them directly with dgEntVars (our
// entropy vars are eta/(gam-1); the product Gamma*g is scale-invariant).
__global__ void dgDpGammaKernel(DgSolver &grid) {
  const real g1 = dgGam - (real)1.0;
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    if (!dgIbLive(grid, bIdx)) continue;
    real gam[5] = {0,0,0,0,0};
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      real U[5], W[5];
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)bIdx*blockSizeTot + nd];
      dgConsToPrimSane(U, W);
      real rho = W[0], p = W[4];
      real q2  = W[1]*W[1] + W[2]*W[2] + W[3]*W[3];
      real c   = dgSoundSpeed(p, rho);
      real lam = sqrt(q2) + c;
      real Hrho = dgGam/rho + g1*g1*rho*q2*q2/((real)4.0*p*p);
      gam[0] = fmax(gam[0], lam/Hrho);
      for (i32 m = 0; m < 3; m++) {
        real Hm = g1/p*((real)1.0 + g1*rho*W[1+m]*W[1+m]/p);
        gam[1+m] = fmax(gam[1+m], lam/Hm);
      }
      real HE = g1*g1*rho/(p*p);
      real M2 = q2*rho/(dgGam*p);
      gam[4] = fmax(gam[4], (real)2.0*M2*lam/(((real)1.0+M2)*HE));
    }
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 8 + q] = g1*gam[q];
  }
}

__global__ void dgPositivityKernel(DgSolver &grid) {
  const real eps_rho = 1e-12, eps_p = 1e-12;
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    // CUT elements: Zhang-Shu is meaningless there -- its cell mean is the
    // full-tensor GLL mean, which on a cut element mixes solid-side extension
    // values into the "conserved" mean, and rescaling toward a garbage mean
    // INFLATES good nodes (measured: 1e33 momentum in one stage).  State
    // redistribution is the cut elements' stabilizer.
    if (grid.cutOn && grid.blkCut && grid.blkCut[bIdx] >= 0) continue;
    if (!dgIbLive(grid, bIdx)) continue;   // ghost fills are non-conservative
    // by design: never Zhang-Shu-limit them.  Evolving IB_CUT elements ARE
    // limited (their fluid-side nodes integrate; the fill re-writes the
    // solid side right after, so scaling those toward the mean is harmless).
    real *F[5];
    for (i32 q = 0; q < 5; q++) F[q] = grid.getField(D_RHO+q) + (u64)bIdx*blockSizeTot;

    // GLL cell mean: (1/8) sum w_i w_j w_k U (z weights are uniform copies in pseudo2D)
    real Ubar[5] = {0,0,0,0,0};
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      real wijk = (real)0.125*c_w[i]*c_w[j]*c_w[k];
      for (i32 q = 0; q < 5; q++) Ubar[q] += wijk*F[q][nd];
    }
    // NB: flooring the cell MEAN is the ONLY non-conservative step in the
    // solver (node scaling toward the mean below is conservative).  Count it
    // (ibCnt[3]) to attribute any mass/energy drift.
    if (Ubar[0] < eps_rho) atomicAdd(&grid.ibCnt[3], 1);
    Ubar[0] = fmax(Ubar[0], eps_rho);
    {
      real p = dgPressureFromCons(Ubar);
      if (p < eps_p) {
        atomicAdd(&grid.ibCnt[3], 1);
        Ubar[4] = eps_p/(dgGam-(real)1.0)
                + (real)0.5*(Ubar[1]*Ubar[1]+Ubar[2]*Ubar[2]+Ubar[3]*Ubar[3])/Ubar[0];
      }
    }

    const real epsR = eps_rho, epsP = eps_p;

    // --ibevolve IB_CUT: the min/bisection sets take EVOLVED (phi > 0) nodes
    // only -- a near-vacuum solid-side FILL node would drive theta -> 0 and
    // flatten the whole evolved band to its mean every stage (measured: the
    // p3 M=3 shoulder mass drain).  Solid nodes are re-filled right after
    // this kernel; the final scaling still moves them, harmlessly.
    const bool isCut = (grid.ibClassList[bIdx] == IB_CUT);
    bool ndOk[blockSizeTot];
    if (isCut) {
      i32 lvlC, ibC, jbC, kbC;
      grid.decode(grid.bLocList[bIdx], lvlC, ibC, jbC, kbC);
      real hC[3]; dgElemSize(grid, lvlC, hC);
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i = nd % NNODE, j = (nd/NNODE)%NNODE;
        ndOk[nd] = dgIbPhi(grid, dgNodePos(hC[0], ibC, i),
                                 dgNodePos(hC[1], jbC, j)) > (real)0.0;
      }
    } else for (i32 nd = 0; nd < blockSizeTot; nd++) ndOk[nd] = true;

    // theta1: density.  On GAUSS solution points also bound the FACE TRACES
    // (tL/tR extrapolations along every line): the RHS reads those, and they
    // can be negative while every node is positive -- the multi-node-set
    // limiter modification of arXiv 2507.09131 (their PPL checks the
    // face-reaching quadrature sets for exactly this reason).  Scaling toward
    // the mean bounds traces too (they are linear in the nodal values).
    real rhoMin = (real)1e30;
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      if (ndOk[nd]) rhoMin = fmin(rhoMin, F[0][nd]);
    if (grid.gauss && !isCut) {   // cut-element traces mix solid fills; their
      // faces present nearest-node states anyway (the symmetric-NN rule)
      for (i32 d = 0; d < (grid.pseudo2D ? 2 : 3); d++)
        for (i32 b = 0; b < NNODE; b++)
          for (i32 a = 0; a < NNODE; a++) {
            real rL = 0, rR = 0;
            for (i32 m = 0; m < NNODE; m++) {
              real v = F[0][dgFaceNode(d, m, a, b)];
              rL += c_tL[m]*v; rR += c_tR[m]*v;
            }
            rhoMin = fmin(rhoMin, fmin(rL, rR));
          }
    }
    real tRho = 1.0;
    if (rhoMin < epsR)
      tRho = (Ubar[0]-epsR)/fmax(Ubar[0]-rhoMin, (real)1e-30);
    tRho = fmax((real)0.0, fmin((real)1.0, tRho));

    // theta2: pressure (bisection per offending candidate).  Candidates =
    // solution nodes, plus (GAUSS) the face traces of the CONSERVATIVE state
    // along every line -- the states the face fluxes are actually built from.
    real theta = tRho;
    auto pBisect = [&](const real U[5]) {
      real Ud[5];
      for (i32 q = 0; q < 5; q++) Ud[q] = Ubar[q] + tRho*(U[q]-Ubar[q]);
      if (dgPressureFromCons(Ud) < epsP) {
        real lo = 0, hi = tRho;
        for (i32 it = 0; it < 20; it++) {
          real tm = (real)0.5*(lo+hi);
          real Um[5];
          for (i32 q = 0; q < 5; q++) Um[q] = Ubar[q] + tm*(U[q]-Ubar[q]);
          if (dgPressureFromCons(Um) >= epsP) lo = tm; else hi = tm;
        }
        theta = fmin(theta, lo);
      }
    };
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      if (!ndOk[nd]) continue;
      real U[5];
      for (i32 q = 0; q < 5; q++) U[q] = F[q][nd];
      pBisect(U);
    }
    if (grid.gauss && !isCut) {
      for (i32 d = 0; d < (grid.pseudo2D ? 2 : 3); d++)
        for (i32 b = 0; b < NNODE; b++)
          for (i32 a = 0; a < NNODE; a++) {
            real UL[5] = {0,0,0,0,0}, UR[5] = {0,0,0,0,0};
            for (i32 m = 0; m < NNODE; m++) {
              i32 nd = dgFaceNode(d, m, a, b);
              for (i32 q = 0; q < 5; q++) {
                UL[q] += c_tL[m]*F[q][nd];
                UR[q] += c_tR[m]*F[q][nd];
              }
            }
            pBisect(UL);
            pBisect(UR);
          }
    }

    if (theta < (real)1.0) {
      for (i32 nd = 0; nd < blockSizeTot; nd++)
        for (i32 q = 0; q < 5; q++)
          F[q][nd] = Ubar[q] + theta*(F[q][nd]-Ubar[q]);
    }
  }
}

// fully-discrete entropy-stable limiter (Liu/Guo/Jiang/Sun, docs/
// EntropyStableDG.pdf): after each RK stage, the candidate QUADRATURE cell
// entropy is bounded by the stage input's quadrature cell entropy minus dt
// times the outward proper-entropy-flux integral (slots 3/4, accumulated by
// the RHS with the paper's Eq-3.9 LF entropy flux at the SAME traces the
// face fluxes used) -- the discrete d/dt int U <= -surf F^ statement -- and
// enforced by a Zhang-Shu-type conservative scaling toward the cell mean
// (scaling toward the mean cannot increase cell entropy by convexity, so
// theta is found by bisection).  With Shu-Osher stage combinations and HLLC
// (not LF) mass/momentum fluxes the bound is a SOFT stabilizer, not the
// paper's theorem -- the relative slack absorbs the mismatch; a candidate
// whose MEAN already violates (pure flux-form mismatch) is flattened.
__global__ void dgEntropyLimitKernel(DgSolver &grid, real dt) {
  DG_BLOCK_LOOP(bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) continue;
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;
    // --eslim 1: limit EVERYWHERE (the paper's robust form).  Needed for
    // p=3 M=3: the rear crisis builds in sub-threshold NEIGHBORS before
    // flooding the flagged element -- a sensor-gated run died at t=1.09
    // with the dying element itself at theta 0.865.  Costs smooth accuracy
    // (bound mismatch accumulates: 13x M=0.3 entropy, 7% vortex).
    // --eslim 2: sensor-gated (only shock-flagged elements limited) --
    // smooth cost eliminated (vortex bit-recovered), but does NOT hold the
    // p=3 M=3 rear.  With --av 0 the sensor slab is stale and mode 2 limits
    // everywhere -- moot anyway: limiter-only M=3 dies at the startup slam
    // (iter ~20, both orders; the limiter caps CELL entropy and cannot see
    // single-node vacuum spikes -- AV interface damping stays required).
    if (grid.esLim == 2 && grid.avOn &&
        grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1] <= grid.ibShockTheta)
      continue;
    real *F[5];
    for (i32 q = 0; q < 5; q++) F[q] = grid.getField(D_RHO+q) + (u64)bIdx*blockSizeTot;
    real bound = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 3]
               - dt*grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 4];
    bound += (real)1e-4*fmax((real)1.0, fabs(bound));

    real Ubar[5] = {0,0,0,0,0};
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      real wijk = (real)0.125*c_w[i]*c_w[j]*c_w[k];
      for (i32 q = 0; q < 5; q++) Ubar[q] += wijk*F[q][nd];
    }
    auto cellEnt = [&](real th) {
      real E = (real)0.0;
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
        real wijk = (real)0.125*c_w[i]*c_w[j]*c_w[k];
        real U[5], W[5];
        for (i32 q = 0; q < 5; q++) U[q] = Ubar[q] + th*(F[q][nd] - Ubar[q]);
        dgConsToPrimSane(U, W);
        E += wijk*dgEntropyU(W);
      }
      return E;
    };
    if (cellEnt((real)1.0) <= bound) continue;
    real th = (real)0.0;
    if (cellEnt((real)0.0) <= bound) {
      real lo = (real)0.0, hi = (real)1.0;
      for (i32 it = 0; it < 16; it++) {
        real tm = (real)0.5*(lo + hi);
        if (cellEnt(tm) <= bound) lo = tm; else hi = tm;
      }
      th = lo;
    }
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++)
        F[q][nd] = Ubar[q] + th*(F[q][nd] - Ubar[q]);
  }
}

// ---- cut-element basis helpers ------------------------------------------
// Hoisted above their first user: dgLamKernel needs to evaluate a MODAL cut
// element's polynomial to get a meaningful wave speed.

static constexpr i32 CUT_NBMAX = 20;    // total-degree P^3 in 3-D


// Zhang-Shu-limit a cut element's trace state toward its fluid mean (sC[0] IS
// the mean: psi_0 = 1).  A modal trace can overshoot near under-resolved
// features; every consumer of a trace (faces, wall, mortar) limits first --
// the same protection the FRIB wall path gives its traces.
__device__ __forceinline__ void dgCutLimitTrace(const real *sC, real U[5],
                                                real psi0t) {
  real Ub[5];
  for (i32 q = 0; q < 5; q++) Ub[q] = sC[q]*psi0t;  // mean = c~_0 psi~_0, psi~_0 = 1/L00
  real rb = fmax(Ub[0], DG_EPSF), pb = dgPressureFromCons(Ub);
  real th = (real)1.0, rf = (real)0.2*rb, pf = (real)0.2*fmax(pb, DG_EPSF);
  real rhoM = fmax(U[0], DG_EPSF), pM = dgPressureFromCons(U);
  if (rhoM < rf) th = fmin(th, (rb-rf)/fmax(rb-rhoM, DG_EPSF));
  if (pM   < pf) th = fmin(th, (pb-pf)/fmax(pb-pM,   DG_EPSF));
  th = fmax(th, (real)0.0);
  for (i32 q = 0; q < 5; q++) U[q] = Ub[q] + th*(U[q]-Ub[q]);
  dgSanitizeCons(U);
}

// modal basis value / gradient at a reference point, from the stored centroid
// and scale.  Mirrors CutBasis so the device and the host builder agree.
__device__ __forceinline__ void dgCutPsi(const real cen[4], const real xr[3],
                                         i32 nb, real *psi, real *dpsi) {
  real u[3];
  for (i32 d = 0; d < 3; d++) u[d] = (xr[d] - cen[d])/cen[3];
  i32 m = 0;
  for (i32 deg = 0; deg <= dgOrder && m < nb; deg++)
  for (i32 i = deg; i >= 0 && m < nb; i--)
  for (i32 j = deg-i; j >= 0 && m < nb; j--) {
    i32 e[3] = { i, j, deg-i-j };
    real v = 1;
    for (i32 d = 0; d < 3; d++) for (i32 a = 0; a < e[d]; a++) v *= u[d];
    if (psi) psi[m] = v;
    if (dpsi) for (i32 d = 0; d < 3; d++) {
      if (e[d] == 0) { dpsi[3*m+d] = 0; continue; }
      real t = (real)e[d]/cen[3];
      for (i32 a = 0; a < 3; a++) {
        i32 pw = (a == d) ? e[a]-1 : e[a];
        for (i32 q = 0; q < pw; q++) t *= u[a];
      }
      dpsi[3*m+d] = t;
    }
    m++;
  }
}

// ORTHONORMAL basis evaluation: monomials psi, then the forward solve
// L psi~ = psi (and per gradient component).  In this frame the element mass
// is exactly I -- no dense inverse anywhere, and the round trip
// nodal -> modal -> nodal is an orthogonal projection.  The sliver conditioning
// still exists, but it enters through ONE backward-stable triangular solve
// instead of a stored explicit inverse.
__device__ __forceinline__ void dgCutSolveL(const real *Lc, i32 nb, real *v) {
  for (i32 i = 0; i < nb; i++) {
    real t = v[i];
    for (i32 j = 0; j < i; j++) t -= Lc[(size_t)i*CUT_NBMAX+j]*v[j];
    v[i] = t/Lc[(size_t)i*CUT_NBMAX+i];
  }
}
__device__ __forceinline__ void dgCutPsiO(const real cen[4], const real xr[3],
                                          i32 nb, const real *Lc,
                                          real *psi, real *dpsi) {
  dgCutPsi(cen, xr, nb, psi, dpsi);
  if (psi) dgCutSolveL(Lc, nb, psi);
  if (dpsi) {
    real col[CUT_NBMAX];
    for (i32 d = 0; d < 3; d++) {
      for (i32 m = 0; m < nb; m++) col[m] = dpsi[3*m+d];
      dgCutSolveL(Lc, nb, col);
      for (i32 m = 0; m < nb; m++) dpsi[3*m+d] = col[m];
    }
  }
}

__global__ void dgLamKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || !dgIbLive(grid, bIdx)) {
      grid.getField(D_LAM)[cIdx] = 1e30;   // IB ghost/dead never bound dt
      continue;
    }
    if (grid.ibClassList[bIdx] == IB_CUT) {
      // only the EVOLVED (phi > 0) nodes bound dt: a solid-side node is a
      // FILL OUTPUT, overwritten every stage -- a near-vacuum fill there
      // (c ~ 1e6) would collapse dt for a node the integrator never owns
      // (measured: p3 M=3 dt crawl at the shoulder, iter 134)
      GET_CELL_INDICES
      i32 lvlC, ibC, jbC, kbC;
      grid.decode(loc, lvlC, ibC, jbC, kbC);
      real hC[3]; dgElemSize(grid, lvlC, hC);
      if (dgIbPhi(grid, dgNodePos(hC[0], ibC, i), dgNodePos(hC[1], jbC, j))
          <= (real)0.0) {
        grid.getField(D_LAM)[cIdx] = 1e30;
        continue;
      }
    }
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    real U[5], W[5];
    const i32 cm = (grid.cutOn && grid.cutModal && grid.blkCut) ? grid.blkCut[bIdx] : -1;
    if (cm >= 0) {
      // MODAL cut block: these slots are COEFFICIENTS.  Reading them as a state
      // gives a meaningless wave speed -- and since dt is a MINIMUM over cells,
      // one nonsense value throttles the whole run (measured: dt 2.7e-05 instead
      // of 7.8e-04, a 30x collapse that looks exactly like a stability problem
      // and is not one).  Evaluate the element's polynomial at its OWN
      // quadrature points instead: one point per thread, the rest bound nothing.
      const i32 nd = (i32)(cIdx - (size_t)bIdx*blockSizeTot);
      const i32 q0 = grid.cutVolOff[cm], nqE = grid.cutVolOff[cm+1] - q0;
      if (nd >= nqE) { grid.getField(D_LAM)[cIdx] = 1e30; continue; }
      const real *cen = grid.cutCen + 4*cm;
      const real *Lc  = grid.cutLc + (size_t)cm*CUT_NBMAX*CUT_NBMAX;
      const i32   nbE = grid.cutNbOf[cm];
      real psi[CUT_NBMAX];
      dgCutPsiO(cen, grid.cutVolP[q0 + nd].x, nbE, Lc, psi, nullptr);
      for (i32 q = 0; q < 5; q++) { real v = 0;
        for (i32 m = 0; m < nbE; m++)
          v += grid.getField(D_RHO+q)[(size_t)bIdx*blockSizeTot + m]*psi[m];
        U[q] = v; }
    } else {
      for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[cIdx];
    }
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

// stage the Brinkman volume fraction phi(x) into SCRATCH for a paint pass
__global__ void dgBrinkPhiToScratchKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    i32 nd = cIdx % blockSizeTot;
    i32 in = nd % NNODE, jn = (nd/NNODE) % NNODE;
    real gp[2];
    grid.getField(D_SCRATCH)[cIdx] =
        dgBrinkPhi(grid, dgNodePos(h[0], ib, in), dgNodePos(h[1], jb, jn), gp);
  }
}

/* ════════════════════════════════════════════════════════════════════════
 * Image build: evaluate the solution at the uniform pixel centers by
 * piecewise-linear interpolation between the LGL solution nodes (hat basis;
 * the full Lagrange polynomial Gibbs-rings at shocks), not nearest-node fill
 * ════════════════════════════════════════════════════════════════════════ */

// Lagrange basis l_a(x) on the LGL nodes (constant memory c_xi)
__device__ real dgBasisAt(i32 a, real x) {
  real v = 1.0;
  for (i32 m = 0; m < NNODE; m++)
    if (m != a) v *= (x - c_xi[m])/(c_xi[a] - c_xi[m]);
  return v;
}

// Lagrange basis of the FRIB image line (its OWN Lobatto node set -- valid on
// any element node set; see c_ibLXi)
__device__ real dgIbLineBasisAt(i32 a, real x) {
  real v = (real)1.0;
  for (i32 m = 0; m < NNODE; m++)
    if (m != a) v *= (x - c_ibLXi[m])/(c_ibLXi[a] - c_ibLXi[m]);
  return v;
}

// piecewise-linear "hat" basis on the LGL nodes: paints monotone between
// nodes.  The full Lagrange polynomial rings at shocks (Gibbs), and the
// per-element overshoot pattern reads as blocky staircases in the image.
// TRUE Lagrange basis at an arbitrary reference coordinate.  The paint used to
// use the piecewise-linear hat below, which draws a p=3 element as a linear
// ramp between its 4 nodes -- so the shape of the solution INSIDE an element
// was never rendered, and at 4 pixels per element that reads as flat blocks.
// Evaluating the actual polynomial gives each pixel the value the solution
// really has there.  (dgLag1 is defined further down for the cut path; this is
// the same formula, local to the paint so the file order is untouched.)
__device__ __forceinline__ real dgLagAt(i32 a, real x) {
  real t = 1;
  for (i32 b = 0; b < NNODE; b++) if (b != a) t *= (x - c_xi[b])/(c_xi[a] - c_xi[b]);
  return t;
}

__device__ __forceinline__ real dgHatAt(i32 a, real x) {
  if (x <= c_xi[a]) {
    if (a == 0) return 1.0;
    real t = (x - c_xi[a-1])/(c_xi[a] - c_xi[a-1]);
    return t > (real)0.0 ? t : (real)0.0;
  }
  if (a == NNODE-1) return 1.0;
  real t = (c_xi[a+1] - x)/(c_xi[a+1] - c_xi[a]);
  return t > (real)0.0 ? t : (real)0.0;
}

__device__ i32 dgIbLocateLeaf(DgSolver &grid, real x, real y, real z,
                              i32 &lvl, i32 &ib, i32 &jb, i32 &kb);   // fwd (defined with the IB fill)

__global__ void dgComputeImageDataKernel(DgSolver &grid, i32 f) {
  real *U = (f >= 0) ? grid.getField(f) : nullptr;
  const real zmid = (real)0.5*grid.domainSize[2];

  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (!grid.isInteriorBlock(lvl, ib, jb, kb)) continue;
    // field views paint FLUID elements only: ghost/cut elements hold the
    // wall-reconstruction node data, not the flow field -- painting it draws
    // an element-granular collar at the wall that reads as a solution
    // artifact.  (Grid view f<0 still shows every element.)
    if (f >= 0 && grid.ibOn && grid.ibClassList[bIdx] != IB_FLUID) {
      // SDF-aware wall paint (kills the staircase VISUAL): a pixel of an
      // inactive element inside the TRUE circle stays 0 (the body renders as
      // a clean disc); a pixel in the staircase GAP (inactive element, phi>0)
      // samples the field at its wall-normal pushed-out point from the
      // nearest ACTIVE leaf -- the smooth near-wall flow, not element zeros.
      real h[3]; dgElemSize(grid, lvl, h);
      i32 nPix = powi(2, grid.nLvls-1-lvl);
      i32 span = blockSize*nPix;
      for (i32 py = 0; py < span; py++) {
        i32 jPxl = jb*span + py;
        if (jPxl < 0 || jPxl >= grid.imageSizeX[1]) continue;
        real y = (jb + (py + (real)0.5)/span)*h[1];
        for (i32 px = 0; px < span; px++) {
          i32 iPxl = ib*span + px;
          if (iPxl < 0 || iPxl >= grid.imageSizeX[0]) continue;
          real x = (ib + (px + (real)0.5)/span)*h[0];
          real phi = dgIbPhi(grid, x, y);
          if (phi < (real)0.0) continue;              // true body: leave 0
          // --ibevolve: an IB_CUT element's fluid-side pixels hold REAL
          // evolved data -- paint the element's own polynomial (shows the
          // evolved band directly); only solid-side pixels stay body-0.
          if (grid.ibClassList[bIdx] == IB_CUT) {
            real zx = (real)2.0*(px + (real)0.5)/span - (real)1.0;
            real zy = (real)2.0*(py + (real)0.5)/span - (real)1.0;
            real acc = (real)0.0;
            for (i32 aa = 0; aa < NNODE; aa++)
              for (i32 bb = 0; bb < NNODE; bb++)
                acc += dgLagAt(aa, zx)*dgLagAt(bb, zy)
                     * U[(u64)bIdx*blockSizeTot + aa + bb*NNODE];
            grid.imageDataX[(u64)jPxl*grid.imageSizeX[0] + iPxl] = acc;
            continue;
          }
          real dxc = x - grid.ibX, dyc = y - grid.ibY;
          real rr = fmax(sqrt(dxc*dxc + dyc*dyc), (real)1e-30);
          real nx = dxc/rr, ny = dyc/rr;
          real val = (real)0.0;
          real s = phi + (real)0.35*h[0];
          for (i32 t2 = 0; t2 < 6; t2++) {            // outward march to fluid
            real xs = grid.ibX + (grid.ibR + s)*nx;
            real ys = grid.ibY + (grid.ibR + s)*ny;
            i32 ml=0, mib=0, mjb=0, mkb=0;
            i32 idx = dgIbLocateLeaf(grid, xs, ys, (real)0.5*grid.domainSize[2],
                                     ml, mib, mjb, mkb);
            if (idx != bEmpty && grid.ibClassList[idx] == IB_FLUID) {
              real hm[3]; dgElemSize(grid, ml, hm);
              real zx = (real)2.0*(xs/hm[0] - mib) - (real)1.0;
              real zy = (real)2.0*(ys/hm[1] - mjb) - (real)1.0;
              real acc = (real)0.0;
              for (i32 aa = 0; aa < NNODE; aa++)
                for (i32 bb = 0; bb < NNODE; bb++)
                  acc += dgLagAt(aa, zx)*dgLagAt(bb, zy)
                       * U[(u64)idx*blockSizeTot + aa + bb*NNODE];
              val = acc;
              break;
            }
            s += (real)0.35*h[0];
          }
          grid.imageDataX[(u64)jPxl*grid.imageSizeX[0] + iPxl] = val;
        }
      }
      continue;
    }

    // ---- CUT MODE MASK ---------------------------------------------------
    // buildCutElems relabels every cut block IB_FLUID and sets ibOn = 0
    // (DgCutBuild.cu:208-209), so the SDF-aware branch above is never taken in
    // a --cutcell run and both of these paint values that are not the solution:
    //   * a DEAD block holds the FROZEN analytic IC forever (dgSetICKernel has
    //     no class gate and is re-run after the cut build), so the body renders
    //     as pristine freestream -- the obstacle is INVISIBLE;
    //   * a cut element's solid-side tensor nodes hold the unconstrained
    //     extension of a polynomial supported on {phi>0} only.
    // blkCut is the only cut marker, and it is null until buildCutElems runs
    // (the bootstrap paints happen before that), hence the null guard.
    const bool cutMask = (f >= 0 && grid.cutOn && grid.blkCut && grid.blkCut[bIdx] >= 0);
    if (f >= 0 && grid.cutOn && grid.ibClassList[bIdx] == IB_DEAD) {
      real hD[3]; dgElemSize(grid, lvl, hD);
      i32 spanD = blockSize*powi(2, grid.nLvls-1-lvl);
      for (i32 py = 0; py < spanD; py++) {
        i32 jP = jb*spanD + py;
        if (jP < 0 || jP >= grid.imageSizeX[1]) continue;
        for (i32 px = 0; px < spanD; px++) {
          i32 iP = ib*spanD + px;
          if (iP < 0 || iP >= grid.imageSizeX[0]) continue;
          grid.imageDataX[(u64)jP*grid.imageSizeX[0] + iP] = (real)kPaintVoid;
        }
      }
      continue;
    }

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
      for (i32 c = 0; c < NNODE; c++) Lz[c] = dgLagAt(c, zeta);
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
      for (i32 b = 0; b < NNODE; b++) Ly[b] = dgLagAt(b, eta);
      i32 jPxl = jb*span + py;
      if (jPxl < 0 || jPxl >= grid.imageSizeX[1]) continue;

      for (i32 px = 0; px < span; px++) {
        i32 iPxl = ib*span + px;
        if (iPxl < 0 || iPxl >= grid.imageSizeX[0]) continue;
        if (cutMask) {   // solid side of a cut element: not a solution value
          real xw = (ib + (px + (real)0.5)/span)*h[0];
          real yw = (jb + (py + (real)0.5)/span)*h[1];
          // the tangency guard shifts the level set by grid.cutEps (shrinking
          // the body); test the SAME surface the cut rules were built on
          if (dgIbPhi(grid, xw, yw) < -grid.cutEps) {
            grid.imageDataX[(u64)jPxl*grid.imageSizeX[0] + iPxl] = (real)kPaintVoid;
            continue;
          }
        }
        real val;
        if (f >= 0) {
          real xi = (real)2.0*(px + (real)0.5)/span - (real)1.0;
          real acc = 0.0;
          for (i32 a = 0; a < NNODE; a++) {
            real Lxa = dgLagAt(a, xi);
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

// ... and the min.  Same trick, same precondition: NONNEGATIVE reals only,
// whose IEEE bit patterns are monotone in the value.  Both helpers write the
// SAME width as `real`, so the slot stays a plain real and an ordinary typed
// read gets the answer back -- which is the bug this replaced: the cut kernel
// used the fp64 form unconditionally, so an fp32 build issued an 8-byte atomic
// on a 4-byte-aligned float slot (memcheck: "Invalid __shared__ read of size 8
// bytes ... misaligned"), killing every non-ES cut run at the first RHS.
__device__ __forceinline__ void dgAtomicMinPos(real *addr, real v) {
#ifdef USE_DOUBLE
  atomicMin((unsigned long long*)addr, (unsigned long long)__double_as_longlong(v));
#else
  atomicMin((int*)addr, __float_as_int(v));
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
    if (ibc == IB_GHOST || ibc == IB_CUT) {   // never coarsen a ghost or an
      atomicMax(&grid.bFlagsList[bIdx], KEEP); // evolving cut element (band
      continue;                                // pins them finest)
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

    // slot 5 = the REFINE sensor (shock + velocity/shear); slot 1 (alpha/AV)
    // deliberately excludes velocity so shear refines but is not FV-smeared
    real th = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 5];
    // amplitude floor (ALWAYS: the Persson theta is a scale-free modal-energy
    // RATIO and fires on roundoff-level wiggle in near-constant regions -- the
    // low-amplitude wake would otherwise refine on noise).  Refine only where
    // the fluctuation modal energy is real signal.
    if (grid.sensorType >= 1) {   // slot 2 (fluct) valid only with Persson
      real fluct = grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 2];
      real fl = grid.subFloor*grid.cScale[0];
      if (fluct < fl*fl) continue;   // quiescent: leave at DELETE (coarsenable)
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

// pass 1: geometric class from the element box's exact SDF range
__global__ void dgIbClassifyGeomKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) { grid.ibClassList[bIdx] = IB_FLUID; continue; }
    // volume penalization: the object is the phi field, not a class -- every
    // element is solved (no ghosts, no cut cells).
    if (grid.ibBrink) { grid.ibClassList[bIdx] = IB_FLUID; continue; }
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
    // --ibevolve: a genuinely CUT ghost (has fluid-side nodes) joins the
    // discretization as IB_CUT -- fluid-side nodes evolve, solid nodes keep
    // the FRIB fill.  Fully-solid elements stay DEAD (-> promoted GHOST).
    grid.ibClassList[bIdx] = !solidish ? ((phiMax <= (real)0.0) ? IB_DEAD : IB_FLUID)
                           : ((phiMax <= (real)0.0) ? IB_DEAD
                              : (grid.ibEvolve ? IB_CUT : IB_GHOST));
  }
}

// pass 2: a fully-solid element with any LIVE face neighbor (fluid, or an
// evolving IB_CUT element under --ibevolve; same level, coarser cover, or
// finer children -- the face-topology dispatch) becomes a GHOST, so no live
// face ever resolves to an unfilled DEAD element.  In-place and race-free:
// only rewrites DEAD -> GHOST and only reads fluid/cut classes, which pass 1
// fixed and this pass never changes.
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
          touchesFluid = dgIbLive(grid, nIdx);
          continue;
        }
        if (lvl > 0) {   // coarser cover
          nIdx = grid.getBlockIdx(grid.encode(lvl-1, nib>>1, njb>>1,
                                              grid.pseudo2D ? nkb : (nkb>>1)));
          if (nIdx != bEmpty) {
            touchesFluid = dgIbLive(grid, nIdx);
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
                touchesFluid = dgIbLive(grid, cIdxN);
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

// the ghost fill: one thread per ghost node
// hard two-sided clamp on the filled ghost states: NO ghost node may present
// density/pressure outside [moodRho,100]*cScale / [moodP,1000]*pinf to the
// wall face flux.  A safety net for the extreme high-res FRIB reconstruction
// (any path) that would otherwise feed an unbounded state into the adjacent
// fluid cell.  Only garbage (>100x) is touched; physical states pass through.
__global__ void dgIbGhostClampKernel(DgSolver &grid) {
  const real rLo = grid.moodRho*grid.cScale[0], rHi = (real)100.0*grid.cScale[0];
  const real pLo = grid.moodP/dgGam,            pHi = (real)1000.0/dgGam;
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.ibClassList[bIdx] != IB_GHOST) continue;
    if (grid.bLocList[bIdx] == kEmpty) continue;
    real U[5], W[5];
    for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[cIdx];
    dgConsToPrimSane(U, W);
    real rc = fmin(fmax(W[0], rLo), rHi), pc = fmin(fmax(W[4], pLo), pHi);
    if (rc != W[0] || pc != W[4]) {
      W[0] = rc; W[4] = pc;
      dgP2C(W, U);
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
    }
  }
}

// SBM cut-cell SOLID-NODE ghost fill (user call): a --ibcut 0 cut cell carries
// nodes INSIDE the solid (r<R) whose evolved data is non-physical.  Instead FILL
// them as ghost nodes -- reflect the node across the wall to an image point in
// the fluid and BILINEARLY interpolate the fluid state there (dgHatAt, piecewise
// linear -> no high-order Gibbs overshoot, the thing that made the FRIB high-
// order fill blow up), then apply the slip-wall reflection u_n -> -u_n.  Gives
// the cut cell a physical near-wall extension so its FLUID nodes' volume
// derivative sees a real state, not garbage.  Fluid nodes (r>=R) are untouched.
__global__ void dgIbSolidFillKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.ibClassList[bIdx] != IB_FLUID) continue;
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty) continue;
    GET_CELL_INDICES
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    real x = dgNodePos(h[0], ib, i), y = dgNodePos(h[1], jb, j);
    real dxc = x - grid.ibX, dyc = y - grid.ibY;
    real r  = sqrt(dxc*dxc + dyc*dyc);
    if (r >= grid.ibR) continue;                   // fluid node: leave it to evolve
    real nx = dxc/fmax(r, (real)1e-12), ny = dyc/fmax(r, (real)1e-12);
    real ri = (real)2.0*grid.ibR - r;              // reflected image radius (> R)
    real xi = grid.ibX + ri*nx, yi = grid.ibY + ri*ny;
    real z  = dgNodePos(h[2], kb, k);
    i32 dl, dib, djb, dkb;
    i32 lidx = dgIbLocateLeaf(grid, xi, yi, z, dl, dib, djb, dkb);
    if (lidx == bEmpty || grid.ibClassList[lidx] != IB_FLUID) continue;   // keep previous
    real hd[3]; dgElemSize(grid, dl, hd);
    real zeta[3] = { (real)2.0*(xi/hd[0]-dib)-(real)1.0,
                     (real)2.0*(yi/hd[1]-djb)-(real)1.0,
                     grid.pseudo2D ? (real)0.0 : (real)2.0*(z/hd[2]-dkb)-(real)1.0 };
    real W[5] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};
    real wsum = (real)0.0;
    i32 cmax = grid.pseudo2D ? 1 : NNODE;
    for (i32 c = 0; c < cmax; c++)
      for (i32 b = 0; b < NNODE; b++)
        for (i32 a = 0; a < NNODE; a++) {
          // FLUID DONOR NODES ONLY (renormalised): a donor's own solid-filled
          // nodes must never feed the fill -- on Gauss the clamped hat gives an
          // edge-gap image point 100% weight on the nearest node, which on the
          // wall side IS a solid node: a filled->filled copy loop around the
          // ring that never touches evolved fluid (the ibcut-0 rest-gate
          // instability, finally localized).
          real wv = dgHatAt(a, zeta[0])*dgHatAt(b, zeta[1])
                  * (grid.pseudo2D ? (real)1.0 : dgHatAt(c, zeta[2]));
          i32 nd = a + NNODE*(b + NNODE*c);
          real U[5], Wp[5];
          for (i32 q = 0; q < 5; q++) U[q] = grid.getField(D_RHO+q)[(u64)lidx*blockSizeTot + nd];
          dgConsToPrimSane(U, Wp);
          for (i32 q = 0; q < 5; q++) W[q] += wv*Wp[q];
          wsum += wv;
        }
    if (wsum < (real)0.05) continue;                // no meaningful fluid support: keep previous
    for (i32 q = 0; q < 5; q++) W[q] /= wsum;
    real un = W[1]*nx + W[2]*ny;                    // slip-wall reflection
    real Wg[5] = { W[0], W[1]-(real)2.0*un*nx, W[2]-(real)2.0*un*ny, W[3], W[4] };
    dgSanitizePrim(Wg);
    real Ug[5]; dgP2C(Wg, Ug);
    for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = Ug[q];
  }
}

__global__ void dgIbFillKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    const i32 cls = grid.ibClassList[bIdx];
    if (cls != IB_GHOST && cls != IB_CUT) continue;
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
    // --ibevolve: the fluid-side nodes of a CUT element EVOLVE -- fill only
    // its solid-side nodes (exact phi = 0 split, complement of the RK mask).
    // The image line below samples only NON-CUT fluid donors (the march
    // skips IB_CUT like any non-fluid class), so no partially-filled cut
    // data ever enters a reconstruction.
    if (cls == IB_CUT && sg > (real)0.0) continue;
    real xb = x - sg*n[0], yb = y - sg*n[1];  // boundary foot

    // FRIB HO-i/c (Funada & Imamura 2023, docs/FRIB.pdf) -- THE wall method.
    // (The mirror-world Hermite fill and the --ibevolve per-node hybrid were
    // REMOVED at user direction 2026-07-13; the project notes keep their
    // measured matrix and reimplementation keys.)
    // Reconstruct (u_n, H, S, u_t) along the wall-normal image line through
    // this node: wall at xi=-1, fluid end at xi=+1, length dIL sized so the
    // FIRST INTERIOR LGL position clears the cut layer (paper Eq 22).  The
    // outer image points SAMPLE the fluid-cell polynomials; the wall point
    // (exactly xi=-1 = c_xi[0], so phi_i(-1) = delta_i0) is SOLVED from
    //   u_n(-1) = 0,  dH/dxi(-1) = 0,  dS/dxi(-1) = 0,
    //   du_t/ds(0) = -u_t(0)/R   (outward s; potential-flow sign, --ibcurv)
    // and the node value is the line polynomial at its own xi -- NO mirror:
    // the face Riemann carries the consistent transpiration of the true
    // sub-face wall.  Deep nodes evaluate the near-wall xi = -1.35 state
    // (mixing constructions along one fluid-facing face is the measured
    // coherence-killer).
    // Fallback ladder: a shocked donor (H and S are interpolated as SMOOTH
    // invariants -- across an unsteady shock both jump; the raw
    // reconstruction died at iter 5 of the M=3 impulsive start), an
    // unsampleable image point, or an inadmissible state drops the node to
    // the paper's LO method (single image point, Eq 16: u_n scaled linearly
    // in wall distance, rho/p/u_t copied -- first-order, bounded); with no
    // donor at all the node keeps its previous value (counted).
    // ── SINGLE-IP FRIB (--ibsingle): one donor element, IP at MAXIMUM DEPTH
    //    along the wall-normal ray inside it (92% of the ray's box exit).
    //    Variant 1: per-field LINEAR lines (wall BC <-> IP state).
    //    Variant 2: per-field QUADRATIC Hermite (wall BC + IP value + IP
    //    normal slope) -- the mirror-era measured-optimal order, primitives.
    //    One clean donor: no multi-element seams, no march flips, and Eq 22's
    //    lower limit dissolves (samples are inside a guaranteed-uncut cell).
    if (grid.ibSingle > 0) {
      // find the FIRST fluid element along the ray, then push the IP deep
      i32 ml = 0, mib = 0, mjb = 0, mkb = 0, mIdx = bEmpty;
      real tFind = (real)0.35*h[0];
      for (i32 t = 0; t < 8; t++) {
        i32 idx = dgIbLocateLeaf(grid, xb + tFind*n[0], yb + tFind*n[1], z,
                                 ml, mib, mjb, mkb);
        if (idx != bEmpty && grid.ibClassList[idx] == IB_FLUID) { mIdx = idx; break; }
        tFind += (real)0.25*h[0];
      }
      if (mIdx == bEmpty) { atomicAdd(&grid.ibCnt[IB_CNT_NODONOR], 1); continue; }
      real hm[3]; dgElemSize(grid, ml, hm);
      // ray-box exit distance from the wall foot (xb,yb) along n
      real tExit = (real)1e30;
      if (fabs(n[0]) > (real)1e-12) {
        real b0 = mib*hm[0], b1 = (mib+1)*hm[0];
        real tc = ((n[0] > 0 ? b1 : b0) - xb)/n[0];
        if (tc > (real)0.0) tExit = fmin(tExit, tc);
      }
      if (fabs(n[1]) > (real)1e-12) {
        real b0 = mjb*hm[1], b1 = (mjb+1)*hm[1];
        real tc = ((n[1] > 0 ? b1 : b0) - yb)/n[1];
        if (tc > (real)0.0) tExit = fmin(tExit, tc);
      }
      // ibSingle 3/4 = same as 1/2 but FIXED depth d = 1.2h (smooth in node
      // position -- the max-depth box exit is DISCONTINUOUS as the normal
      // sweeps donor corners, an element-granular roughness source)
      real d = (grid.ibSingle >= 3) ? (real)1.2*h[0]
             : fmax((real)0.92*tExit, tFind);          // IP depth from the wall
      real xI = xb + d*n[0], yI = yb + d*n[1];
      real zetam[3] = { (real)2.0*(xI/hm[0] - mib) - (real)1.0,
                        (real)2.0*(yI/hm[1] - mjb) - (real)1.0,
                        grid.pseudo2D ? (real)0.0
                                      : (real)2.0*(z/hm[2] - mkb) - (real)1.0 };
      real F[5], F1[5], F2[5];
      dgIbDonorEval(grid, mIdx, hm, zetam, n, F, F1, F2);
      // gates (same ladder): arriving shock -> piston star; outflow -> LO
      real aI = sqrt(dgGam*fmax(F[4], DG_EPSF)/fmax(F[0], DG_EPSF));
      real unI = F[1]*n[0] + F[2]*n[1];
      real thD = grid.getField(D_SCRATCH)[(u64)mIdx*blockSizeTot + 1];
      bool gated = (fabs(unI) > (real)0.3*aI) || (thD > grid.ibShockTheta);
      real tx = -n[1], ty = n[0];
      real utI = F[1]*tx + F[2]*ty;
      if (gated) {
        atomicAdd(&grid.ibCnt[IB_CNT_FALLBACK], 1);
        if (unI < (real)0.0) {   // piston star
          real rhoI = fmax(F[0], DG_EPSF), pIr = F[4];
          real pI = fmax(pIr, DG_EPSF), m2 = unI*unI;
          real A  = (real)2.0/((dgGam+(real)1.0)*rhoI);
          real Bc = (dgGam-(real)1.0)/(dgGam+(real)1.0)*pI;
          real bq = (real)2.0*pI + m2/A, cq = pI*pI - m2*Bc/A;
          real ps = (real)0.5*(bq + sqrt(fmax(bq*bq - (real)4.0*cq, (real)0.0)));
          real pCap = (pIr > (real)0.0) ? pI : (real)0.5*rhoI*m2;
          ps = fmin(ps, (real)50.0*pCap);
          real g = (dgGam-(real)1.0)/(dgGam+(real)1.0), pr = ps/pI;
          real rs = rhoI*(pr + g)/(g*pr + (real)1.0);
          real W[5] = { rs, utI*tx, utI*ty, F[3], ps };
          dgSanitizePrim(W);
          real U[5]; dgP2C(W, U);
          for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
          continue;
        }
        real sEv = fmax(sg, (real)0.0);
        real W[5] = { F[0], unI*(sEv/d)*n[0] + utI*tx,
                            unI*(sEv/d)*n[1] + utI*ty, F[3], F[4] };
        dgSanitizePrim(W);
        real U[5]; dgP2C(W, U);
        for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
        continue;
      }
      // wall values with primitive BCs (curvature-gated)
      real utW, pW, rhoW;
      real curvOn = grid.ibCurv ? (real)1.0 : (real)0.0;
      real fac = (real)1.0 - curvOn*d/fmax(grid.ibR, DG_EPSF);
      utW = (fac > (real)0.2) ? utI/fac : utI;         // du_t/ds(0) = -u_t/R
      real gp = curvOn*F[0]*utW*utW/fmax(grid.ibR, DG_EPSF);
      real a2 = dgGam*fmax(F[4], DG_EPSF)/fmax(F[0], DG_EPSF);
      real sEv = fmax(sg, (real)-0.175*d);             // eval depth (clamped)
      real un, ut, pv, rv, wv;
      if (grid.ibSingle == 1 || grid.ibSingle == 3) {
        // linear per-field lines
        un = unI*(sEv/d);
        ut = utW + (utI - utW)*(sEv/d);
        pv = (F[4] - gp*d) + gp*sEv;                   // p_w + gp*s
        rv = (F[0] - gp/a2*d) + gp/a2*sEv;
        wv = F[3];
      } else {
        // quadratic Hermite: wall BC + IP value + IP normal slope
        real unp = F1[1]*n[0] + F1[2]*n[1];
        real utp = F1[1]*tx + F1[2]*ty;
        real ppn = F1[4], rpn = F1[0];
        // u_n: a=0; b d + c d^2 = unI; b + 2 c d = unp
        real cN = (unp*d - unI)/(d*d), bN = (real)2.0*unI/d - unp;
        un = bN*sEv + cN*sEv*sEv;
        // u_t: b = -a/R (curv); a(1 - d/R) + c d^2 = utI; -a/R + 2 c d = utp
        real R  = fmax(grid.ibR, DG_EPSF);
        real A11 = (real)1.0 - curvOn*d/R, A12 = d*d;
        real A21 = -curvOn/R,              A22 = (real)2.0*d;
        real det = A11*A22 - A12*A21;
        real aT, cT;
        if (fabs(det) > (real)1e-12) {
          aT = ( utI*A22 - A12*utp)/det;
          cT = ( A11*utp - A21*utI)/det;
        } else { aT = utI; cT = (real)0.0; }
        real bT = -curvOn*aT/R;
        ut = aT + bT*sEv + cT*sEv*sEv;
        // p: b = gp; c = (p' - b)/(2d); a = pI - b d - c d^2
        real cP = (ppn - gp)/((real)2.0*d);
        real aP = F[4] - gp*d - cP*d*d;
        pv = aP + gp*sEv + cP*sEv*sEv;
        real gr = gp/a2;
        real cR = (rpn - gr)/((real)2.0*d);
        real aR = F[0] - gr*d - cR*d*d;
        rv = aR + gr*sEv + cR*sEv*sEv;
        // w: linear value+slope
        wv = F[3] + F1[3]*(sEv - d);
        if (grid.ibLimit) {   // no new extremum beyond {wall, IP}
          real lo, hi;
          lo=fmin((real)0.0,unI); hi=fmax((real)0.0,unI); un=fmin(fmax(un,lo),hi);
          lo=fmin(aT,utI);        hi=fmax(aT,utI);        ut=fmin(fmax(ut,lo),hi);
          lo=fmin(aP,F[4]);       hi=fmax(aP,F[4]);       pv=fmin(fmax(pv,lo),hi);
          lo=fmin(aR,F[0]);       hi=fmax(aR,F[0]);       rv=fmin(fmax(rv,lo),hi);
        }
      }
      if (pv > (real)1e-3*F[4] && rv > (real)1e-3*F[0]) {
        real W[5] = { rv, un*n[0] + ut*tx, un*n[1] + ut*ty, wv, pv };
        dgSanitizePrim(W);
        real U[5]; dgP2C(W, U);
        for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
      } else {
        atomicAdd(&grid.ibCnt[IB_CNT_RETRY1], 1);
        real sEv2 = fmax(sg, (real)0.0);
        real W[5] = { F[0], unI*(sEv2/d)*n[0] + utI*tx,
                            unI*(sEv2/d)*n[1] + utI*ty, F[3], F[4] };
        dgSanitizePrim(W);
        real U[5]; dgP2C(W, U);
        for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
      }
      continue;
    }

    // image-line length: paper Eq 22 (3h at p=2 / 5.5h at p=3, sized so the
    // first interior line node clears the cut layer).  --ibdil overrides (in
    // units of h) -- the ibevolve line-shortening experiments: with the cut
    // element evolving, how much of the long line is still load-bearing?
    real dIL = ((grid.ibDil > (real)0.0) ? grid.ibDil
              : ((NNODE >= 4) ? (real)5.5 : (real)3.0))*h[0];
    real xiN = fmax((real)2.0*sg/dIL - (real)1.0, (real)-1.35);
    real Un[NNODE], Ut[NNODE], Wt[NNODE], Hn[NNODE], Sn[NNODE];
    real rho1 = 0, p1 = 0;
    real ipS = (real)1.0, ipF[5] = {0,0,0,0,0};   // first sampled IP (fallbacks)
    bool haveIp = false, shocked = false;
    {
      {
        bool ok = true;
        for (i32 m = 1; m < NNODE; m++) {
          real sm = (c_ibLXi[m] + (real)1.0)*(real)0.5*dIL;   // LINE node (Lobatto)
          // push-out march (mirror-fill style): an IP landing in a non-FLUID
          // leaf steps outward in fine 0.25h increments -- an instant
          // per-node fallback flips neighbouring nodes between schemes as
          // leaves re-adapt (element-granular trace jumps, the wall
          // stair-step source).  The marched sample is used AT the basis
          // position (O(0.25h) smooth-field inconsistency, continuous in
          // space/time -- preferable to a scheme flip).
          i32 ml = 0, mib = 0, mjb = 0, mkb = 0;
          i32 mIdx = bEmpty;
          for (i32 t = 0; t < 6; t++) {
            i32 idx = dgIbLocateLeaf(grid, xb + sm*n[0], yb + sm*n[1], z,
                                     ml, mib, mjb, mkb);
            if (idx != bEmpty && grid.ibClassList[idx] == IB_FLUID) { mIdx = idx; break; }
            sm += (real)0.25*h[0];
          }
          if (mIdx == bEmpty) { ok = false; break; }
          real hm[3]; dgElemSize(grid, ml, hm);
          real xm = xb + sm*n[0], ym = yb + sm*n[1];
          real zetam[3] = { (real)2.0*(xm/hm[0] - mib) - (real)1.0,
                            (real)2.0*(ym/hm[1] - mjb) - (real)1.0,
                            grid.pseudo2D ? (real)0.0
                                          : (real)2.0*(z/hm[2] - mkb) - (real)1.0 };
          real F[5], F1[5], F2[5];
          dgIbDonorEval(grid, mIdx, hm, zetam, n, F, F1, F2);
          if (!haveIp) {   // keep the innermost sample for the fallbacks
            haveIp = true; ipS = sm;
            for (i32 q = 0; q < 5; q++) ipF[q] = F[q];
            // normal-Mach gate, sensor-INDEPENDENT and SYMMETRIC: |u_n| >
            // 0.3 a at the image point violates the smooth-wall assumption
            // in either direction -- inflow is a forming wall shock (star
            // fallback), outflow at the M=3 startup rear band swings the
            // u_n line polynomial 0 -> 3 and the |u|^2 back-conversion
            // overshoots H (LO fallback).  The theta sensor alone LAGS the
            // impulsive start by a few steps, and raw FRIB inside that
            // window is an fp knife edge (identical-digit iter-2 blowups
            // that recompiles tip either way).  In settled flow -- subsonic
            // OR behind the bow shock -- near-wall u_n is small and FRIB
            // stays engaged.
            real aI = sqrt(dgGam*fmax(F[4], DG_EPSF)/fmax(F[0], DG_EPSF));
            if (fabs(F[1]) > (real)0.3*aI) {
              ok = false;
              shocked = true;
              atomicAdd(&grid.ibCnt[IB_CNT_FALLBACK], 1);
              break;
            }
          }
          // shock gate AFTER sampling (the shocked-node fallback reuses the
          // sample): H and S are interpolated as SMOOTH invariants -- both
          // jump across an unsteady shock (raw FRIB died at M=3 iter 5).
          // Behind the settled bow shock the wall band is smooth subsonic
          // and the full reconstruction re-engages.
          if (grid.getField(D_SCRATCH)[(u64)mIdx*blockSizeTot + 1]
              > grid.ibShockTheta) {
            ok = false;
            shocked = true;
            atomicAdd(&grid.ibCnt[IB_CNT_FALLBACK], 1);
            break;
          }
          real q2 = F[1]*F[1] + F[2]*F[2] + F[3]*F[3];
          Un[m] = F[1];
          Ut[m] = F[2];
          Wt[m] = F[3];
          if (grid.ibRecon == 1) {          // PRIMITIVE line: p, rho directly
            Hn[m] = F[4];                    // (Hn slot carries p)
            Sn[m] = F[0];                    // (Sn slot carries rho)
          } else {
            Hn[m] = dgGam/(dgGam - (real)1.0)*F[4]/fmax(F[0], DG_EPSF) + (real)0.5*q2;
            Sn[m] = F[4]/pow(fmax(F[0], DG_EPSF), dgGam);
          }
          if (m == 1) { rho1 = F[0]; p1 = F[4]; }
        }
        if (grid.ibHO == 2) {
          // ── PAPER WALL MODEL (Qi, Wang, Zhu, Tian, Zhao 2024, Eq 19) ──────
          // Pure form: EVERY ghost node (smooth, shocked, expansion) takes this
          // robust reconstruction -- NO piston, NO LO.  Survives the full
          // high-res M=3 impulsive start, but UNDER-REFLECTS the bow shock
          // (measured standoff -97%, p_stag -88%): the linear-u_n + centripetal
          // pressure cannot present the strong reflected-shock state, which is
          // exactly what the piston provides.  Re-introducing the piston for
          // shocked nodes recovers the standoff but reactivates its gain-loop
          // blowup at the turning shoulder -- the open design problem.
          // ONE near-wall image point + an ALGEBRAIC curvature correction,
          // evaluated at the CLAMPED near-wall trace.  Deliberately NOT the
          // deep image-line polynomial and NOT the H/S smooth-invariant
          // reconstruction.  Robust by construction: u_n is linear-to-zero at
          // the wall and clamped to |u_n,image| (no deep extrapolation
          // overshoot -> no velocity blowup); the wall pressure rises by the
          // centripetal balance dP/dn = -rho u_t^2/R from a POSITIVE
          // (Gibbs-floored) image pressure -> no vacuum ghost; rho follows
          // isentropically -> stays positive.  This handles the smooth and
          // EXPANSION nodes; SHOCKED COMPRESSION nodes fall through to the
          // (vacuum-cap-fixed) piston below, which alone presents the strong
          // reflected-shock state the M=3 standoff needs -- pure paper mode
          // under-reflects it by ~90%.  keep-previous if no donor at all.
          if (haveIp) {
            real rI  = fmax(ipF[0], DG_EPSF);
            // floor the sampled image pressure to a dynamic-pressure scale: a
            // startup DG Gibbs dip drives it <= 0, and the whole point of this
            // model is to never turn that into a non-physical ghost.
            real PI  = fmax(ipF[4], (real)1e-3*(real)0.5*rI*ipF[1]*ipF[1]);
            PI = fmax(PI, DG_EPSF);
            real unI = ipF[1], utI = ipF[2], wtI = ipF[3];
            real sgEff = fmax(sg, -ipS);              // clamp to <= 1 image spacing
            real un = unI*(sgEff/ipS);                // linear to 0 at wall, reflected
            real dn = ipS - sgEff;                     // wall-normal span, in (ipS, 2 ipS]
            real Pn = PI + rI*utI*utI/fmax(grid.ibR, DG_EPSF)*dn;  // centripetal rise
            real rhon = rI*pow(Pn/PI, (real)1.0/dgGam);            // isentropic density
            real W[5] = { rhon, un*n[0] - utI*n[1], un*n[1] + utI*n[0], wtI, Pn };
            dgSanitizePrim(W);
            real U[5];
            dgP2C(W, U);
            for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
            atomicAdd(&grid.ibCnt[IB_CNT_RETRY1], 1);
            continue;
          }
          atomicAdd(&grid.ibCnt[IB_CNT_NODONOR], 1);   // no donor: keep previous
          continue;
        }
        if (ok) {
          real un, ut, wt, Hh, Ss;
          if (grid.ibHO == 1) {
            const real *D0 = c_ibLD0;   // line-basis derivative row at the wall
            real sH = 0, sS = 0, sU = 0, sW = 0;
            for (i32 m = 1; m < NNODE; m++) {
              sH += D0[m]*Hn[m]; sS += D0[m]*Sn[m];
              sU += D0[m]*Ut[m]; sW += D0[m]*Wt[m];
            }
            Un[0] = (real)0.0;
            if (grid.ibRecon == 1) {
              // primitive wall BCs (in line coords, d/dxi = (dIL/2) d/dn):
              //   dp/dn   = rho u_t^2 / R  (centripetal; ibCurv gates it)
              //   drho/dn = dp/dn / a^2    (linearized isentropy -- no powers)
              // coefficients from the innermost sample (first-order lag).
              real ut1 = Ut[1];
              real gp  = grid.ibCurv
                       ? (real)0.5*dIL * rho1*ut1*ut1/fmax(grid.ibR, DG_EPSF)
                       : (real)0.0;
              real a2  = dgGam*fmax(p1, DG_EPSF)/fmax(rho1, DG_EPSF);
              Hn[0] = (gp      - sH)/D0[0];          // p_wall
              Sn[0] = (gp/a2   - sS)/D0[0];          // rho_wall
            } else {
              Hn[0] = -sH/D0[0];
              Sn[0] = -sS/D0[0];
            }
            Wt[0] = -sW/D0[0];
            real dnm = D0[0] + (grid.ibCurv ? (real)0.5*dIL/grid.ibR : (real)0.0);
            Ut[0] = -sU/dnm;
            un = 0; ut = 0; wt = 0; Hh = 0; Ss = 0;
            for (i32 m = 0; m < NNODE; m++) {
              real ph = dgIbLineBasisAt(m, xiN);   // LINE basis, not the element's
              un += ph*Un[m]; ut += ph*Ut[m]; wt += ph*Wt[m];
              Hh += ph*Hn[m]; Ss += ph*Sn[m];
            }
            // MUSCL monotonicity limiter: clamp each reconstructed field to the
            // range of the WALL value (m=0) + the sampled image points (m>=1),
            // so the high-order polynomial cannot create a NEW extremum -- the
            // ring in H/S that detonates the near-vacuum/high-res wall.  H,S are
            // smooth invariants so no new extremum is physical; the stagnation
            // pressure rise SURVIVES (it comes from |u| -> 0 at the wall, where
            // un=0 is in the data, not from H overshooting).
            if (grid.ibLimit) {
              real unL=Un[0],unH=Un[0],utL=Ut[0],utH=Ut[0],wtL=Wt[0],wtH=Wt[0];
              real HL=Hn[0],HH=Hn[0],SL=Sn[0],SH=Sn[0];
              for (i32 m = 1; m < NNODE; m++) {
                unL=fmin(unL,Un[m]); unH=fmax(unH,Un[m]);
                utL=fmin(utL,Ut[m]); utH=fmax(utH,Ut[m]);
                wtL=fmin(wtL,Wt[m]); wtH=fmax(wtH,Wt[m]);
                HL =fmin(HL, Hn[m]); HH =fmax(HH, Hn[m]);
                SL =fmin(SL, Sn[m]); SH =fmax(SH, Sn[m]);
              }
              un=fmin(fmax(un,unL),unH); ut=fmin(fmax(ut,utL),utH); wt=fmin(fmax(wt,wtL),wtH);
              Hh=fmin(fmax(Hh,HL),HH);   Ss=fmin(fmax(Ss,SL),SH);
            }
          } else {
            // FIRST ORDER (--ibho 0): the interpolated image point + the wall
            // boundary value, NO SLOPE.  H,S held CONSTANT at the image value
            // (dH/dn = dS/dn = 0 with no slope term -> constant); u_n LINEAR
            // from 0 at the wall (sg = 0) to the image value, via the geometric
            // wall-distance ratio sg/ipS; u_t,w_t = image (no curvature slope).
            // rho,p fall out of (H,S,|u|) below, so |u| -> 0 at the wall still
            // lifts the stagnation pressure -- but a 2-point line CANNOT ring,
            // so the near-vacuum/high-res wall stays bounded.  ipF/ipS = the
            // innermost fluid image point (the same sample the LO fallback uses).
            un = ipF[1]*(sg/ipS);
            ut = ipF[2]; wt = ipF[3];
            real q2i = ipF[1]*ipF[1] + ipF[2]*ipF[2] + ipF[3]*ipF[3];
            Hh = dgGam/(dgGam - (real)1.0)*ipF[4]/fmax(ipF[0], DG_EPSF) + (real)0.5*q2i;
            Ss = ipF[4]/pow(fmax(ipF[0], DG_EPSF), dgGam);
          }
          real q2n = un*un + ut*ut + wt*wt;
          real Tst, SsEff;
          if (grid.ibRecon == 1) {
            // direct primitives: no H - q^2/2 cancellation, no power law.
            real pw = Hh, rw = Ss;
            if (pw > (real)1e-3*p1 && rw > (real)1e-3*rho1 &&
                pw < (real)1000.0/dgGam && rw < (real)100.0*grid.cScale[0]) {
              real W[5] = { rw, un*n[0] - ut*n[1], un*n[1] + ut*n[0], wt, pw };
              dgSanitizePrim(W);
              real U[5];
              dgP2C(W, U);
              for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
              continue;
            }
            Tst = (real)-1.0; SsEff = (real)-1.0;   // inadmissible -> fallback ladder
          } else {
            Tst = (dgGam - (real)1.0)/dgGam*(Hh - (real)0.5*q2n);   // p/rho
            SsEff = Ss;
          }
          (void)SsEff;
          if (grid.ibDbg && grid.iter>=14 && grid.iter<=21 && lvl==4 && ib==106 && jb==213)
            printf("[fill rec] ijk(%d,%d,%d) xiN=%.2f Tst=%.3e Ss=%.3e Hh=%.3e q2n=%.3e un=%.3e ipF4=%.3e\n",
                   i,j,k,(double)xiN,(double)Tst,(double)Ss,(double)Hh,(double)q2n,(double)un,(double)ipF[4]);
          if (Tst > (real)0.0 && Ss > (real)0.0) {
            real rho = pow(Tst/fmax(Ss, DG_EPSF), (real)1.0/(dgGam - (real)1.0));
            real p   = Tst*rho;
            // a-posteriori-limited FRIB: the HO reconstruction is accepted
            // only if TWO-SIDED admissible (the MOOD bounds).  The lower
            // bound is positivity; the UPPER bound (100x rho, 1000x p) rejects
            // the extreme high-res reconstruction that feeds a bad wall state
            // into the fluid through the face flux (the high-res nose/rear
            // blowup no fluid-side limiter could reach) -- it drops to the
            // bounded LO fallback instead ("low order FRIB when troubled").
            real rHi = (real)100.0*grid.cScale[0], pHi = (real)1000.0/dgGam;
            if (rho > (real)1e-3*rho1 && p > (real)1e-3*p1 && rho < rHi && p < pHi) {
              real W[5] = { rho,
                            un*n[0] - ut*n[1],
                            un*n[1] + ut*n[0],
                            wt,
                            p };
              if (grid.ibDbg && grid.iter>=14 && grid.iter<=21 && lvl==4 && ib==106 && jb==213)
                printf("[fill HO ] ijk(%d,%d,%d) xiN=%.2f p=%.3e rho=%.3e Hh=%.3e q2n=%.3e\n",
                       i,j,k,(double)xiN,(double)p,(double)rho,(double)Hh,(double)q2n);
              dgSanitizePrim(W);
              real U[5];
              dgP2C(W, U);
              for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
              continue;
            }
          }
        }
      }
    }

    // shocked COMPRESSION node: EXACT WALL-RIEMANN (piston) trace.  The FRIB
    // trace is a transpiration wall -- it presents no jump to the face
    // Riemann solver, so an arriving shock transmits instead of reflecting
    // (measured: standoff -89% with a plain LO fallback).  Here the node
    // presents the exact post-reflection star state of the arriving
    // image-point flow: reflected shock, p* from the Rankine-Hugoniot piston
    // relation (closed-form positive quadratic root), rho* from the shock
    // density ratio; u_n = 0, tangential velocity passes through (contact).
    // Exact across shocks, per node, no smooth-invariant interpolation; it
    // degrades continuously into the smooth-wall state as u_n -> 0-.
    // COMPRESSION ONLY (u_n < 0): the expansion analogue (reflected
    // rarefaction) prescribes near-vacuum at the M=3 startup rear band
    // (p* = p (1-0.2 M)^7 ~ 1e-3 p) and detonates by iter 2 -- outflow
    // regions transpire through the LO fallback instead.
    if (grid.ibPiston && shocked && haveIp && ipF[1] < (real)0.0) {
      real rhoI = fmax(ipF[0], DG_EPSF), unI = ipF[1];
      real pIraw = ipF[4];
      real pI = fmax(pIraw, DG_EPSF);
      real m2 = unI*unI;
      real A  = (real)2.0/((dgGam + (real)1.0)*rhoI);
      real B  = (dgGam - (real)1.0)/(dgGam + (real)1.0)*pI;
      real b  = (real)2.0*pI + m2/A;
      real c  = pI*pI - m2*B/A;
      real ps = (real)0.5*(b + sqrt(fmax(b*b - (real)4.0*c, (real)0.0)));
      // cap the reflection at 50x the image pressure: the legitimate M=3
      // head-on reflection is ps/pI ~ 17; unbounded, a transient |u_n| spike
      // at the p=3 rear recompression feeds a gain>1 loop (star p -> face
      // flux -> larger u_n at the image -> larger star: explosion at t=1.09,
      // elem (64,61), rho 3e8) that p=2's dissipation damps.  BUT a startup DG
      // Gibbs dip drives the sampled image pressure <= 0 at the high-res nose;
      // fmax(.,DG_EPSF) then makes 50*pI ~ 1e-11 and the cap collapses the star
      // to a VACUUM wall ghost, which drains the near-wall fluid to a detonation
      // (cell 105,213, t~0.003).  When the image pressure is non-physical, cap
      // against the incoming DYNAMIC pressure (1/2 rho u_n^2) -- the physical
      // scale of the piston reflection -- so the ghost is a real stagnation
      // wall, not vacuum.  Healthy (pIraw > 0) behaviour is byte-identical.
      real pCap = (pIraw > (real)0.0) ? pI : (real)0.5*rhoI*m2;
      ps = fmin(ps, (real)50.0*pCap);
      real g  = (dgGam - (real)1.0)/(dgGam + (real)1.0);
      real pr = ps/pI;
      real rs = rhoI*(pr + g)/(g*pr + (real)1.0);
      real W[5] = { rs,
                    -ipF[2]*n[1],          // u_n = 0; tangential passes through
                     ipF[2]*n[0],
                     ipF[3],
                     ps };
      if (grid.ibDbg && grid.iter>=14 && grid.iter<=21 && lvl==4 && ib==106 && jb==213)
        printf("[fill STAR] ijk(%d,%d,%d) ps=%.3e rs=%.3e (ipF: rho=%.3e un=%.3e p=%.3e)\n",
               i,j,k,(double)ps,(double)rs,(double)ipF[0],(double)ipF[1],(double)ipF[4]);
      dgSanitizePrim(W);
      real U[5];
      dgP2C(W, U);
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
      continue;
    }

    // LO fallback (paper Eq 16): single image point, u_n scaled linearly in
    // the signed wall distance (solid nodes get the sign flip for free),
    // rho/p/u_t copied.  First-order and bounded by a real sampled state.
    if (haveIp) {
      atomicAdd(&grid.ibCnt[IB_CNT_RETRY1], 1);
      real un = (sg/ipS)*ipF[1];
      real W[5] = { ipF[0],
                    un*n[0] - ipF[2]*n[1],
                    un*n[1] + ipF[2]*n[0],
                    ipF[3],
                    ipF[4] };
      if (grid.ibDbg && grid.iter>=14 && grid.iter<=21 && lvl==4 && ib==106 && jb==213)
        printf("[fill LO  ] ijk(%d,%d,%d) sg=%.3e ipS=%.3e un=%.3e (ipF: rho=%.3e p=%.3e) shocked=%d\n",
               i,j,k,(double)sg,(double)ipS,(double)un,(double)ipF[0],(double)ipF[4],(int)shocked);
      dgSanitizePrim(W);
      real U[5];
      dgP2C(W, U);
      for (i32 q = 0; q < 5; q++) grid.getField(D_RHO+q)[cIdx] = U[q];
      continue;
    }
    if (grid.ibDbg && lvl==4 && ib==106 && jb==213)
      printf("[fill KEEP] ijk(%d,%d,%d) haveIp=%d shocked=%d  (keeping rho=%.3e rhoE=%.3e)\n",
             i,j,k,(int)haveIp,(int)shocked,
             (double)grid.getField(D_RHO)[cIdx], (double)grid.getField(D_RHOE)[cIdx]);
    atomicAdd(&grid.ibCnt[IB_CNT_NODONOR], 1);   // no donor at all: keep previous values
  }
}

// --debug audit: (i) no FLUID face may resolve to a DEAD element (the
// classification invariant); (ii) no sub-finest octet may mix GHOST and FLUID
// members (the band guarantee that protects the merge machinery)
__global__ void dgIbCheckKernel(DgSolver &grid) {
  DG_BLOCK_LOOP(bIdx) {
    u64 loc = grid.bLocList[bIdx];
    if (loc == kEmpty || !dgIbLive(grid, bIdx)) continue;   // live faces
    // (fluid + evolving cut) must never resolve to an unfilled DEAD element
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

// troubled-element indicator paint: copy the per-element blend/sensor value
// (theta_e in SCRATCH slot 1, filled by dgAvNuKernel) to every node of D_LAM
// (a DIFFERENT array -- reading SCRATCH while writing SCRATCH would race the
// node-1 slot).  Under subFv this is the FV blend factor a = min(subMax,
// theta); otherwise the raw sensor.  IB ghost/dead elements paint 0.
__global__ void dgTroubledToScratchKernel(DgSolver &grid) {
  DG_CELL_LOOP(cIdx, bIdx) {
    if (grid.bLocList[bIdx] == kEmpty) { grid.getField(D_LAM)[cIdx] = (real)0.0; continue; }
    real th = (grid.ibOn && !dgIbLive(grid, bIdx))
            ? (real)0.0 : grid.getField(D_SCRATCH)[(u64)bIdx*blockSizeTot + 1];
    grid.getField(D_LAM)[cIdx] = grid.subFv ? fmin(grid.subMax, th) : th;
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

// ===========================================================================
//  CUT-CELL DG
//
//  One CUDA block per cut element.  A cut element has no collocation, no
//  diagonal mass and no tensor structure, so none of the machinery above
//  applies: it carries a total-degree P^N MODAL basis with a dense inverse mass
//  matrix, built once by cutElemBuild().
//
//  Its STATE still lives in the ordinary nodal fieldData, so RK, positivity,
//  MOOD, output and state redistribution need no cut branch.  This kernel goes
//    nodal -> modal -> operators -> dense M^-1 -> nodal
//  and writes an ordinary nodal RHS.
//
//  FACE COUPLING.  A cut face's quadrature points are not the tensor face
//  nodes, so both sides must use the SAME rule.  The cut element owns it and
//  computes BOTH sides of the flux here, depositing the neighbour's share by
//  atomicAdd; dgRhsKernel skips any face whose neighbour is cut.  Conservative
//  by construction, and it keeps the Cartesian fast path almost untouched.
// ===========================================================================

// 1-D Lagrange basis at an arbitrary point of [-1,1] (NNODE = 4: the direct
// product form is cheaper than carrying barycentric weights)
__device__ __forceinline__ void dgLag1(real x, real L[NNODE]) {
  for (i32 a = 0; a < NNODE; a++) {
    real t = 1;
    for (i32 b = 0; b < NNODE; b++) if (b != a) t *= (x - c_xi[b])/(c_xi[a] - c_xi[b]);
    L[a] = t;
  }
}
// tensor Lagrange value at a REFERENCE point of [0,1]^3 (Saye coordinates)
__device__ __forceinline__ void dgLag3(const real xr[3], real *phi) {
  real Lx[NNODE], Ly[NNODE], Lz[NNODE];
  dgLag1((real)2.0*xr[0]-(real)1.0, Lx);
  dgLag1((real)2.0*xr[1]-(real)1.0, Ly);
  dgLag1((real)2.0*xr[2]-(real)1.0, Lz);
  for (i32 k = 0; k < NNODE; k++)
  for (i32 j = 0; j < NNODE; j++)
  for (i32 i = 0; i < NNODE; i++)
    phi[i + NNODE*(j + NNODE*k)] = Lx[i]*Ly[j]*Lz[k];
}


// ---------------------------------------------------------------------------
//  NODAL -> MODAL for every cut element, run once after the initial condition.
//  Identical arithmetic to the projection dgRhsCutKernel used to do every
//  evaluation; the difference is that the result now STAYS in the field.
//  Slots [nb, blockSizeTot) are zeroed: RK sees zero state and zero RHS there,
//  so they stay zero and cost nothing.
// ---------------------------------------------------------------------------
/* ════════════════════════════════════════════════════════════════════════
 *  STATE REDISTRIBUTION ON THE DEVICE
 *
 *  A faithful port of srdApply (common/StateRedistribution.h) + the gather /
 *  write-back that applySrd wrapped around it.  Same algebra, same order of
 *  operations, in double throughout -- so an fp32 solver still redistributes
 *  in double, exactly as the host path did.
 *
 *    kernel 1  gather : nodal state of every SRD element -> grid.srdU.  A
 *                       MODAL cut block is evaluated through its own basis at
 *                       the tensor nodes (its slots hold coefficients).
 *    kernel 2  project: one block per NON-TRIVIAL neighbourhood k.  Builds
 *                       rhs_m = SUM_j SUM_q w_j w_q hv_j psi_m(X_q) u_j(x_q)
 *                       and Cholesky-solves it -> grid.srdCoef[k].
 *    kernel 3  scatter: one block per element i.  Averages every projection
 *                       that touched i over its own tensor nodes, then writes
 *                       back -- projecting onto the cut basis over the fluid
 *                       rule for a modal cut block.
 * ════════════════════════════════════════════════════════════════════════ */

// total-degree monomials about a neighbourhood centroid (SrdBasis::eval)
__device__ __forceinline__ void dgSrdPsi(const double *bas, i32 N,
                                         const double X[3], double *psi) {
  const double sc = bas[3];
  const double u = (X[0]-bas[0])/sc, v = (X[1]-bas[1])/sc, w = (X[2]-bas[2])/sc;
  i32 m = 0;
  for (i32 d = 0; d <= N; d++)
    for (i32 i = d; i >= 0; i--)
      for (i32 j = d-i; j >= 0; j--) {
        const i32 k = d-i-j;
        double t = 1;
        for (i32 a = 0; a < i; a++) t *= u;
        for (i32 a = 0; a < j; a++) t *= v;
        for (i32 a = 0; a < k; a++) t *= w;
        psi[m++] = t;
      }
}

// tensor Lagrange values at a reference point in [0,1]^3 (double; dgLag3 is real)
__device__ __forceinline__ void dgSrdLag1(double x, double *L) {
  for (i32 a = 0; a < NNODE; a++) {
    double t = 1;
    const double xa = (double)c_xi[a];
    for (i32 b = 0; b < NNODE; b++)
      if (b != a) t *= (x - (double)c_xi[b])/(xa - (double)c_xi[b]);
    L[a] = t;
  }
}

__global__ void dgSrdGatherKernel(DgSolver &grid) {
  const i32 e = blockIdx.x;
  if (e >= grid.srdNE) return;
  const i32 b = grid.srdBlk[e];
  const i32 c = (grid.cutModal && grid.blkCut) ? grid.blkCut[b] : -1;
  double *ue = grid.srdU + (size_t)e*blockSizeTot*5;
  if (c < 0) {
    for (i32 nd = threadIdx.x; nd < blockSizeTot; nd += blockDim.x)
      for (i32 q = 0; q < 5; q++)
        ue[(size_t)nd*5+q] = (double)grid.getField(D_RHO+q)[(size_t)b*blockSizeTot+nd];
    return;
  }
  const i32 nb = grid.cutNbOf[c];
  const real *cen = grid.cutCen + 4*c;
  const real *Lc  = grid.cutLc  + (size_t)c*CUT_NBMAX*CUT_NBMAX;
  for (i32 nd = threadIdx.x; nd < blockSizeTot; nd += blockDim.x) {
    const i32 i2 = nd%NNODE, j2 = (nd/NNODE)%NNODE, k2 = nd/(NNODE*NNODE);
    real xr[3] = { (real)(0.5*((double)c_xi[i2]+1.0)),
                   (real)(0.5*((double)c_xi[j2]+1.0)),
                   (real)(0.5*((double)c_xi[k2]+1.0)) };
    real psi[CUT_NBMAX];
    dgCutPsiO(cen, xr, nb, Lc, psi, nullptr);
    for (i32 q = 0; q < 5; q++) {
      double v = 0;
      for (i32 m = 0; m < nb; m++)
        v += (double)grid.getField(D_RHO+q)[(size_t)b*blockSizeTot+m]*(double)psi[m];
      ue[(size_t)nd*5+q] = v;
    }
  }
}

__global__ void dgSrdProjectKernel(DgSolver &grid) {
  extern __shared__ double sh[];
  const i32 k = blockIdx.x;
  if (k >= grid.srdNE || grid.srdTriv[k]) return;
  const i32 nb = grid.srdNb;
  double *rhs = sh;                       // [nb*5]
  for (i32 t = threadIdx.x; t < nb*5; t += blockDim.x) rhs[t] = 0.0;
  __syncthreads();

  const double *bas = grid.srdBas + 4*k;
  for (i32 mi = grid.srdMOff[k]; mi < grid.srdMOff[k+1]; mi++) {
    const i32 j = grid.srdM[mi];
    const double wj = 1.0/(double)grid.srdCcnt[j];
    const double hv = grid.srdH[3*j]*grid.srdH[3*j+1]*grid.srdH[3*j+2];
    const double *uj = grid.srdU + (size_t)j*blockSizeTot*5;
    const i32 q0 = grid.srdQOff[j], q1 = grid.srdQOff[j+1];
    for (i32 q = q0 + threadIdx.x; q < q1; q += blockDim.x) {
      const SayeNode &sn = grid.srdQ[q];
      const double xr[3] = { (double)sn.x[0], (double)sn.x[1], (double)sn.x[2] };
      double X[3], Lx[NNODE], Ly[NNODE], Lz[NNODE], psi[64];
      for (i32 d = 0; d < 3; d++) X[d] = grid.srdX0[3*j+d] + grid.srdH[3*j+d]*xr[d];
      dgSrdLag1(2.0*xr[0]-1.0, Lx); dgSrdLag1(2.0*xr[1]-1.0, Ly); dgSrdLag1(2.0*xr[2]-1.0, Lz);
      dgSrdPsi(bas, grid.srdDeg, X, psi);
      const double wq = wj*(double)sn.w*hv;
      for (i32 cc = 0; cc < 5; cc++) {
        double uq = 0;
        for (i32 kk = 0; kk < NNODE; kk++)
          for (i32 jj = 0; jj < NNODE; jj++) {
            const double lyz = Ly[jj]*Lz[kk];
            for (i32 ii = 0; ii < NNODE; ii++)
              uq += Lx[ii]*lyz*uj[(size_t)(ii + NNODE*(jj + NNODE*kk))*5+cc];
          }
        const double wu = wq*uq;
        for (i32 m = 0; m < nb; m++) atomicAdd(&rhs[m*5+cc], wu*psi[m]);
      }
    }
  }
  __syncthreads();

  // one Cholesky solve per component against the stored factor
  if (threadIdx.x < 5) {
    const i32 cc = threadIdx.x;
    const double *L = grid.srdChol + (size_t)k*nb*nb;
    double *out = grid.srdCoef + (size_t)k*nb*5;
    double y[64];
    for (i32 i = 0; i < nb; i++) {
      double t = rhs[i*5+cc];
      for (i32 q = 0; q < i; q++) t -= L[(size_t)i*nb+q]*y[q];
      y[i] = t/L[(size_t)i*nb+i];
    }
    for (i32 i = nb-1; i >= 0; i--) {
      double t = y[i];
      for (i32 q = i+1; q < nb; q++) t -= L[(size_t)q*nb+i]*out[q*5+cc];
      out[i*5+cc] = t/L[(size_t)i*nb+i];
    }
  }
}

__global__ void dgSrdScatterKernel(DgSolver &grid) {
  extern __shared__ double sh2[];
  const i32 i = blockIdx.x;
  if (i >= grid.srdNE) return;
  const i32 nb = grid.srdNb;
  double *su = sh2;                          // [blockSizeTot*5]
  const double inv = 1.0/(double)grid.srdCcnt[i];
  const double *ui = grid.srdU + (size_t)i*blockSizeTot*5;

  for (i32 t = threadIdx.x; t < blockSizeTot*5; t += blockDim.x) su[t] = 0.0;
  __syncthreads();
  for (i32 nd = threadIdx.x; nd < blockSizeTot; nd += blockDim.x) {
    const i32 i2 = nd%NNODE, j2 = (nd/NNODE)%NNODE, k2 = nd/(NNODE*NNODE);
    const double X[3] = { grid.srdX0[3*i+0] + grid.srdH[3*i+0]*0.5*((double)c_xi[i2]+1.0),
                          grid.srdX0[3*i+1] + grid.srdH[3*i+1]*0.5*((double)c_xi[j2]+1.0),
                          grid.srdX0[3*i+2] + grid.srdH[3*i+2]*0.5*((double)c_xi[k2]+1.0) };
    double acc[5] = {0,0,0,0,0};
    for (i32 ci = grid.srdCOff[i]; ci < grid.srdCOff[i+1]; ci++) {
      const i32 j = grid.srdC[ci];
      if (grid.srdTriv[j]) {
        for (i32 q = 0; q < 5; q++) acc[q] += inv*ui[(size_t)nd*5+q];
        continue;
      }
      double psi[64];
      dgSrdPsi(grid.srdBas + 4*j, grid.srdDeg, X, psi);
      const double *cf = grid.srdCoef + (size_t)j*nb*5;
      for (i32 q = 0; q < 5; q++) {
        double t = 0;
        for (i32 m = 0; m < nb; m++) t += cf[m*5+q]*psi[m];
        acc[q] += inv*t;
      }
    }
    for (i32 q = 0; q < 5; q++) su[(size_t)nd*5+q] = acc[q];
  }
  __syncthreads();

  // ---- write back ---------------------------------------------------------
  const i32 b = grid.srdBlk[i];
  const i32 c = (grid.cutModal && grid.blkCut) ? grid.blkCut[b] : -1;
  if (c < 0) {
    for (i32 nd = threadIdx.x; nd < blockSizeTot; nd += blockDim.x)
      for (i32 q = 0; q < 5; q++)
        grid.getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] = (real)su[(size_t)nd*5+q];
    return;
  }
  // MODAL cut block: project the redistributed nodal state back onto the
  // element's own basis over its fluid rule (the host path's final step).
  double *cm = sh2 + (size_t)blockSizeTot*5;          // [CUT_NBMAX*5]
  const i32 nbc = grid.cutNbOf[c];
  for (i32 t = threadIdx.x; t < nbc*5; t += blockDim.x) cm[t] = 0.0;
  __syncthreads();
  const real *cen = grid.cutCen + 4*c;
  const real *Lc  = grid.cutLc  + (size_t)c*CUT_NBMAX*CUT_NBMAX;
  for (i32 g = grid.cutVolOff[c] + threadIdx.x; g < grid.cutVolOff[c+1]; g += blockDim.x) {
    const SayeNode &sn = grid.cutVolP[g];
    real psir[CUT_NBMAX];
    dgCutPsiO(cen, sn.x, nbc, Lc, psir, nullptr);
    double Lx[NNODE], Ly[NNODE], Lz[NNODE];
    dgSrdLag1(2.0*(double)sn.x[0]-1.0, Lx);
    dgSrdLag1(2.0*(double)sn.x[1]-1.0, Ly);
    dgSrdLag1(2.0*(double)sn.x[2]-1.0, Lz);
    for (i32 q = 0; q < 5; q++) {
      double uq = 0;
      for (i32 kk = 0; kk < NNODE; kk++)
        for (i32 jj = 0; jj < NNODE; jj++) {
          const double lyz = Ly[jj]*Lz[kk];
          for (i32 ii = 0; ii < NNODE; ii++)
            uq += Lx[ii]*lyz*su[(size_t)(ii + NNODE*(jj + NNODE*kk))*5+q];
        }
      const double wu = (double)sn.w*uq;
      for (i32 m = 0; m < nbc; m++) atomicAdd(&cm[m*5+q], wu*(double)psir[m]);
    }
  }
  __syncthreads();
  for (i32 nd = threadIdx.x; nd < blockSizeTot; nd += blockDim.x)
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] =
          (nd < nbc) ? (real)cm[nd*5+q] : (real)0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  ZHANG-SHU POSITIVITY LIMITER ON CUT ELEMENTS
 *
 *  dgPositivityKernel SKIPS cut elements, and the reason recorded for that is
 *  sound as far as it goes: "its cell mean is the full-tensor GLL mean, which
 *  mixes solid-side extension values, and rescaling toward a garbage mean
 *  INFLATES good nodes (measured 1e33)".  That objection is specific to the
 *  NODAL representation.  Under --cutmodal the obstacle is gone:
 *
 *    * the mean over the FLUID REGION is exactly c~_0 * psi~_0 = c~_0 / L00,
 *      because psi~_0 is the constant 1/L00 -- no solid-side values enter it;
 *    * the admissible set is evaluated on the element's OWN quadrature (volume,
 *      wall and cut faces), i.e. only where the polynomial is a solution;
 *    * and the deviation scaling U := Ubar + theta (U - Ubar) is, in this
 *      basis, simply "multiply modes 1.. by theta and leave c~_0 alone" -- so
 *      it is EXACTLY conservative by construction and costs no nodal round trip.
 *
 *  Two stages, as Zhang & Shu: density first, then pressure by bisection along
 *  the segment from the mean to the state (p is not linear in theta).
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void dgCutPositivityKernel(DgSolver &grid) {
  extern __shared__ real shp2[];
  const i32 c = blockIdx.x;
  if (c >= grid.nCutElem) return;
  const i32 b  = grid.cutBlk[c];
  const i32 nb = grid.cutNbOf[c];
  if (nb <= 1) return;                       // P0: the mean IS the solution
  const real *cen = grid.cutCen + 4*c;
  const real *Lc  = grid.cutLc  + (size_t)c*CUT_NBMAX*CUT_NBMAX;

  real *sC = shp2;                           // [nb*5] coefficients
  real *sR = sC + (size_t)CUT_NBMAX*5;       // [2] reduction slots
  for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x)
    sC[i] = grid.getField(D_RHO + (i%5))[(size_t)b*blockSizeTot + i/5];
  if (threadIdx.x == 0) { sR[0] = (real)1.0; sR[1] = (real)1.0; }
  __syncthreads();

  // fluid-region mean (psi~_0 = 1/L00 is constant over the element)
  real Ub[5];
  for (i32 q = 0; q < 5; q++) Ub[q] = sC[0*5+q]*((real)1.0/Lc[0]);
  const real rhoB = Ub[0];
  const real pB   = dgPressureFromCons(Ub);
  if (!(rhoB > (real)0) || !(pB > (real)0)) return;   // mean inadmissible: nothing to scale toward
  const real epsR = fmin((real)1e-13, rhoB);
  const real epsP = fmin((real)1e-13, pB);

  // the element's own admissible-set points: volume rule + wall + cut faces
  const i32 v0 = grid.cutVolOff[c],  v1 = grid.cutVolOff[c+1];
  const i32 w0 = grid.cutWalOff[c],  w1 = grid.cutWalOff[c+1];
  const i32 f0 = grid.cutFacOff[6*c], f1 = grid.cutFacOff[6*c+6];
  const i32 nTot = (v1-v0) + (w1-w0) + (f1-f0);

  // ---- stage 1: density ---------------------------------------------------
  for (i32 t = threadIdx.x; t < nTot; t += blockDim.x) {
    const SayeNode *sn = (t < v1-v0) ? &grid.cutVolP[v0+t]
                       : (t < (v1-v0)+(w1-w0)) ? &grid.cutWalP[w0+t-(v1-v0)]
                       : &grid.cutFacP[f0+t-(v1-v0)-(w1-w0)];
    real psi[CUT_NBMAX];
    dgCutPsiO(cen, sn->x, nb, Lc, psi, nullptr);
    real r = 0;
    for (i32 m = 0; m < nb; m++) r += sC[m*5+0]*psi[m];
    if (r < epsR) {
      const real th = (rhoB - r > (real)0) ? (rhoB - epsR)/(rhoB - r) : (real)0;
      dgAtomicMinPos(&sR[0], fmax(fmin(th, (real)1.0), (real)0.0));
    }
  }
  __syncthreads();
  const real th1 = sR[0];
  if (th1 < (real)1.0)
    for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x)
      if (i/5 > 0) sC[i] *= th1;              // modes 1.. only: the mean is kept
  __syncthreads();

  // ---- stage 2: pressure, bisection in theta ------------------------------
  for (i32 t = threadIdx.x; t < nTot; t += blockDim.x) {
    const SayeNode *sn = (t < v1-v0) ? &grid.cutVolP[v0+t]
                       : (t < (v1-v0)+(w1-w0)) ? &grid.cutWalP[w0+t-(v1-v0)]
                       : &grid.cutFacP[f0+t-(v1-v0)-(w1-w0)];
    real psi[CUT_NBMAX];
    dgCutPsiO(cen, sn->x, nb, Lc, psi, nullptr);
    real U[5];
    for (i32 q = 0; q < 5; q++) { real v = 0;
      for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; U[q] = v; }
    if (dgPressureFromCons(U) >= epsP) continue;
    real lo = (real)0, hi = (real)1;          // p(lo) >= epsP by the mean check
    for (i32 it = 0; it < 24; it++) {
      const real mid = (real)0.5*(lo+hi);
      real Um[5];
      for (i32 q = 0; q < 5; q++) Um[q] = Ub[q] + mid*(U[q]-Ub[q]);
      if (dgPressureFromCons(Um) >= epsP && Um[0] >= epsR) lo = mid; else hi = mid;
    }
    dgAtomicMinPos(&sR[1], lo);
  }
  __syncthreads();
  const real th2 = sR[1];
  for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) {
    real v = sC[i];
    if (i/5 > 0 && th2 < (real)1.0) v *= th2;
    grid.getField(D_RHO + (i%5))[(size_t)b*blockSizeTot + i/5] = v;
  }
}

__global__ void dgCutToModalKernel(DgSolver &grid) {
  extern __shared__ real sh[];
  real *sU = sh;                          // [5][blockSizeTot] nodal state
  real *sC = sU + 5*blockSizeTot;         // [nb][5] coefficients

  for (i32 c = blockIdx.x; c < grid.nCutElem; c += gridDim.x) {
    const i32 b = grid.cutBlk[c];
    const real *cen = grid.cutCen + 4*c;
    const real *Lc = grid.cutLc + (size_t)c*CUT_NBMAX*CUT_NBMAX;
    const i32 nb = grid.cutNbOf[c];

    for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x)
      for (i32 q = 0; q < 5; q++)
        sU[q*blockSizeTot + i] = grid.getField(D_RHO+q)[(size_t)b*blockSizeTot + i];
    for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) sC[i] = 0;
    __syncthreads();

    for (i32 g = threadIdx.x; g < grid.cutVolOff[c+1]-grid.cutVolOff[c]; g += blockDim.x) {
      const SayeNode &s = grid.cutVolP[grid.cutVolOff[c] + g];
      real psi[CUT_NBMAX], phi[blockSizeTot];
      dgCutPsiO(cen, s.x, nb, Lc, psi, nullptr);
      dgLag3(s.x, phi);
      for (i32 q = 0; q < 5; q++) {
        real uq = 0;
        for (i32 a = 0; a < blockSizeTot; a++) uq += sU[q*blockSizeTot+a]*phi[a];
        for (i32 m = 0; m < nb; m++) atomicAdd(&sC[m*5+q], s.w*psi[m]*uq);
      }
    }
    __syncthreads();
    for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x)
      for (i32 q = 0; q < 5; q++)
        grid.getField(D_RHO+q)[(size_t)b*blockSizeTot + i] =
            (i < nb) ? sC[i*5+q] : (real)0;
    __syncthreads();
  }
}

__global__ void dgRhsCutKernel(DgSolver &grid, real t) {
  extern __shared__ real sh[];
  real *sU = sh;                       // [5][blockSizeTot] nodal state
  real *sR = sU + 5*blockSizeTot;      // [nb][5] modal residual
  real *sC = sR + 5*CUT_NBMAX;         // [nb][5] modal coefficients
  real *sLam = sC + 5*CUT_NBMAX;       // [3] wave-speed reduce (AV), min(rho) reduce,
                                       // and [2] the cut-aware modal-decay verdict

  for (i32 c = blockIdx.x; c < grid.nCutElem; c += gridDim.x) {
    const i32 b = grid.cutBlk[c];
    i32 lvl, ib, jb, kb; grid.decode(grid.bLocList[b], lvl, ib, jb, kb);
    real h[3]; dgElemSize(grid, lvl, h);
    const real *cen = grid.cutCen + 4*c;
    const real *Lc = grid.cutLc + (size_t)c*CUT_NBMAX*CUT_NBMAX;
    i32 nb = grid.cutNbOf[c];              // == the full modal count: a cut
                                           // element carries an UNCUT element's
                                           // order (see CutElem.h)
    // The CUT-MOOD order drop that used to live here -- a troubled cut element
    // evaluating its RHS at P0 -- is gone with the rest of the degree-reduction
    // machinery.  Dropping a cut element to first order whenever a sensor fires
    // makes the wall first-order exactly where the flow is interesting, and it
    // silently decides the order of accuracy per element per timestep, which no
    // convergence study can survive.  The modal-decay sensor is KEPT and still
    // published (it is a useful diagnostic and drives the limiter), but it no
    // longer changes the order of the evaluation.
    const i32 nbFull = nb;

    for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x)
      for (i32 q = 0; q < 5; q++)
        sU[q*blockSizeTot + i] = grid.getField(D_RHO+q)[(size_t)b*blockSizeTot + i];
    for (i32 i = threadIdx.x; i < nbFull*5; i += blockDim.x) { sR[i] = 0; sC[i] = 0; }
    __syncthreads();
    if (grid.cutModal) {
      // MODAL RESIDENT: the coefficients ARE the state.  No projection, and --
      // the point of the whole exercise -- no evaluation of psi~ at the tensor
      // nodes, where it is 449x larger than anywhere in the element's support.
      for (i32 i = threadIdx.x; i < nbFull*5; i += blockDim.x) {
        const i32 m = i/5, q = i - 5*m;
        sC[i] = sU[q*blockSizeTot + m];
      }
      __syncthreads();
    }

    // ---- nodal -> modal, ORTHONORMAL: c~ = SUM_q w_q psi~(x_q) u(x_q) ------
    // The mass is exactly I in this frame, so the projection IS the weighted
    // sum -- no solve, no stored inverse, and the round trip is an orthogonal
    // projection.  FULL basis: the state keeps its modes even when the flux
    // evaluation below is order-dropped.
    if (!grid.cutModal)
    for (i32 g = threadIdx.x; g < grid.cutVolOff[c+1]-grid.cutVolOff[c]; g += blockDim.x) {
      const SayeNode &s = grid.cutVolP[grid.cutVolOff[c] + g];
      real psi[CUT_NBMAX], phi[blockSizeTot];
      dgCutPsiO(cen, s.x, nbFull, Lc, psi, nullptr);
      dgLag3(s.x, phi);
      for (i32 q = 0; q < 5; q++) {
        real uq = 0;
        for (i32 a = 0; a < blockSizeTot; a++) uq += sU[q*blockSizeTot+a]*phi[a];
        for (i32 m = 0; m < nbFull; m++) atomicAdd(&sC[m*5+q], s.w*psi[m]*uq);
      }
    }
    __syncthreads();
    // ---- CUT-AWARE TROUBLE SENSOR 1: MODAL DECAY over the fluid region -----
    // Orthonormal frame: energies are PLAIN COEFFICIENT SUMS, and the
    // degree-major Cholesky NESTS, so the first k coefficients span the
    // low-degree space exactly.  eta = 1 - sum_{m<k} c~^2 / sum c~^2.
    // Sensed on DENSITY (Persson's choice).
    if (threadIdx.x == 0) {
      real trouble = 0, thP = 0;
      const i32 k = grid.cutNbLo[c];
      if (nbFull > 1 && k > 0) {
        real num = 0, low = 0;
        for (i32 m = 0; m < nbFull; m++) { real cm = sC[m*5+0]; num += cm*cm;
          if (m < k) low += cm*cm; }
        if (num > (real)1e-30) {
          real eta = (real)1.0 - low/num;
          if (eta > grid.cutEta) trouble = 1;
          // PERSSON & PERAIRE's RAMP on the same quantity.  eta is the top-mode
          // energy FRACTION; his indicator is its log10, and his sensor is a
          // raised-sine ramp over [s0-kappa, s0+kappa] rather than a step -- a
          // binary gate chatters on and off between stages and puts a
          // discontinuity into the dissipation.  Same s0/kappa the Cartesian
          // elements use (--pps0 / --ppkappa), so the wall band and the mesh
          // around it are gated by one sensor with one calibration.
          const real sP = log10(fmax(eta, (real)1e-30));
          const real s0 = grid.ppS0, kap = grid.ppKappa;
          thP = (sP < s0-kap) ? (real)0.0 : (sP > s0+kap) ? (real)1.0
              : (real)0.5*((real)1.0 + sin((real)0.5*(real)PI*(sP - s0)/kap));
        }
      }
      sLam[2] = trouble;                    // sLam[0] = AV reduce, sLam[1] = min(rho)
      sLam[3] = thP;                        // continuous Persson ramp (filter gate)
    }
    __syncthreads();
    // (sLam[2] is the modal-decay verdict; it no longer drops the order)
    __syncthreads();
    for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) sR[i] = 0;
    if (threadIdx.x == 0) { sLam[0] = 0; }
    __syncthreads();

    // ---- ARTIFICIAL VISCOSITY, element-local LDG (mirrors the Cartesian
    //      two-pass BR1 term at DgSolverKernels:2076): nu_e = avCav * theta *
    //      (h/(2p+1)) * lambda_e, with theta from the SAME Ducros/Persson
    //      sensor slot the Cartesian elements publish (dgAvNuKernel runs on
    //      every active block, cut ones included -- their nodal values are the
    //      modal extension, which is exactly what the sensor should look at).
    //      The modal gradient is ANALYTIC, so the viscous volume term is one
    //      extra dot product per quadrature point; face dissipation is already
    //      present via the Rusanov jump term.
    real nuE = 0;
    if (grid.avOn) {
      if (threadIdx.x == 0) sLam[1] = (real)INFINITY;  // min(rho) reduce identity
      // (was 1e300, which an fp32 build silently overflows to the same +inf)
      __syncthreads();
      for (i32 g = threadIdx.x; g < grid.cutVolOff[c+1]-grid.cutVolOff[c]; g += blockDim.x) {
        const SayeNode &s = grid.cutVolP[grid.cutVolOff[c] + g];
        real psi[CUT_NBMAX];
        dgCutPsiO(cen, s.x, nb, Lc, psi, nullptr);
        real U[5];
        for (i32 q = 0; q < 5; q++) { real v = 0;
          for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; U[q] = v; }
        dgSanitizeCons(U);
        real rho = fmax(U[0], DG_EPSF);
        real p = dgPressureFromCons(U);
        real lam = (fabs(U[1])+fabs(U[2])+fabs(U[3]))/rho + dgSoundSpeed(p, rho);
        dgAtomicMaxPos(&sLam[0], lam);
        // min(rho): atomicMin on the raw bits -- POSITIVE reals order like
        // integers, so this is exact (rho is sanitize-floored > 0).
        dgAtomicMinPos(&sLam[1], fmax(U[0], DG_EPSF));
      }
      __syncthreads();
      // THE SENSOR IS THIS ELEMENT'S OWN, not the Cartesian slot.  The comment
      // above used to claim dgAvNuKernel publishes theta for cut blocks too --
      // it does not: it sets `active = false` for a MODAL cut block
      // (DgSolverKernels.cu:1105-1107) and EVERY D_SCRATCH write in that kernel
      // is inside the `active` branch, so slots 0 and 1 were never written for
      // a cut element and this read returned whatever the allocation held.
      // Artificial viscosity has therefore never actually been active on a cut
      // cell, in either direction: nuE was 0 here and the face penalty sigma
      // read the same dead slot.  The correct sensor is already computed a few
      // lines above -- the modal-decay verdict over the FLUID region, which is
      // the whole reason the cut-aware sensors exist.
      real theta = sLam[2];
      // CUT-AWARE TROUBLE SENSOR 2: positivity margin.  The decay sensor (and
      // the tensor-node theta) can miss imminent vacuum in the wedge; if the
      // modal solution's density anywhere in the FLUID region is below 10% of
      // the mean, run this evaluation with the sensor forced full on.
      real c0rho = fmax(sC[0]/Lc[0], DG_EPSF);   // mean rho = c~_0 psi~_0
      if (sLam[1] < (real)0.1*c0rho)
        theta = (real)1.0;
      real lenp  = h[0]/(real)(2*dgOrder+1);
      const real lamE = sLam[0];
      nuE = grid.avCav * theta * lenp * lamE;
      // publish for anyone reading THIS block's slots -- the neighbour side of
      // the conforming-face penalty, dgTroubledToScratchKernel, the refine vote
      if (threadIdx.x == 0) {
        grid.getField(D_SCRATCH)[(u64)b*blockSizeTot]     = theta*lamE;
        grid.getField(D_SCRATCH)[(u64)b*blockSizeTot + 1] = theta;
        grid.getField(D_SCRATCH)[(u64)b*blockSizeTot + 5] = theta;
        grid.getField(D_SCRATCH)[(u64)b*blockSizeTot + 6] = (real)0;   // no subcell FV here
      }
    }
    __syncthreads();

    // ---- volume:  + INT F_d(u) d(psi_m)/dx_d dV ----------------------------
    if (grid.cutDbgMask & 1)
    for (i32 g = threadIdx.x; g < grid.cutVolOff[c+1]-grid.cutVolOff[c]; g += blockDim.x) {
      const SayeNode &s = grid.cutVolP[grid.cutVolOff[c] + g];
      real psi[CUT_NBMAX], dpsi[3*CUT_NBMAX];
      dgCutPsiO(cen, s.x, nb, Lc, psi, dpsi);
      real U[5];
      for (i32 q = 0; q < 5; q++) { real v = 0;
        for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; U[q] = v; }
      dgSanitizeCons(U);
      real W[5]; W[0]=fmax(U[0],DG_EPSF);
      W[1]=U[1]/W[0]; W[2]=U[2]/W[0]; W[3]=U[3]/W[0]; W[4]=dgPressureFromCons(U);
      // ALL of the modal RHS is kept in REFERENCE measure: the projection above
      // used reference weights against the reference-mass M^-1, and dividing
      // both R and M by hx*hy*hz cancels.  So the volume term carries only the
      // 1/h_d of the physical gradient; wall and faces below follow suit.
      for (i32 d = 0; d < 3; d++) {
        real F[5]; dgEulerFluxAxis(W, d, F);
        real jac = (real)1.0/h[d];
        for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++)
          atomicAdd(&sR[m*5+q], s.w*F[q]*dpsi[3*m+d]*jac);
      }
      // AV:  R_m -= INT nu grad(U) . grad(psi_m) dV   (reference measure; the
      // physical gradients carry 1/h_d each).  Conservative-variable Laplacian,
      // like the Cartesian term.
      if (nuE > (real)0.0) {
        real gU[5][3];
        for (i32 q = 0; q < 5; q++) for (i32 d = 0; d < 3; d++) {
          real v = 0;
          for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*dpsi[3*m+d];
          gU[q][d] = v/h[d];
        }
        for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++) {
          real acc = 0;
          for (i32 d = 0; d < 3; d++) acc += gU[q][d]*dpsi[3*m+d]/h[d];
          atomicAdd(&sR[m*5+q], -s.w*nuE*acc);
        }
      }
    }
    // ---- wall:  - INT (F_wall.n) psi_m dS ---------------------------------
    if (grid.cutDbgMask & 4)
    for (i32 g = threadIdx.x; g < grid.cutWalOff[c+1]-grid.cutWalOff[c]; g += blockDim.x) {
      const SayeNode &s = grid.cutWalP[grid.cutWalOff[c] + g];
      real psi[CUT_NBMAX];
      dgCutPsiO(cen, s.x, nb, Lc, psi, nullptr);
      real U[5];
      for (i32 q = 0; q < 5; q++) { real v = 0;
        for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; U[q] = v; }
      dgCutLimitTrace(sC, U, (real)1.0/Lc[0]);
      real W[5]; W[0]=fmax(U[0],DG_EPSF);
      W[1]=U[1]/W[0]; W[2]=U[2]/W[0]; W[3]=U[3]/W[0]; W[4]=dgPressureFromCons(U);
      // NANSON: the reference normal is not the physical one unless the cell is
      // a cube, and in pseudo-2D h_z differs from h_x by orders of magnitude.
      // For the diagonal metric J = diag(h),  n~ = (n_ref,d / h_d),
      // dS_phys = |n~| * hx hy hz * dS_ref,  n_phys = n~/|n~|.
      real nt[3] = { s.n[0]/h[0], s.n[1]/h[1], s.n[2]/h[2] };
      real nm = sqrt(nt[0]*nt[0] + nt[1]*nt[1] + nt[2]*nt[2]);
      if (nm <= (real)0) continue;
      real np[3] = { nt[0]/nm, nt[1]/nm, nt[2]/nm };
      real dS = s.w*nm;                     // reference measure (see volume term)
      real Fw[5];
      if (grid.cutFsp) {                    // TRANSPARENT wall (free-stream gate):
        real Fx[5], Fy[5], Fz[5];           // the exact F.n of the trace state
        dgEulerFluxAxis(W, 0, Fx); dgEulerFluxAxis(W, 1, Fy); dgEulerFluxAxis(W, 2, Fz);
        for (i32 q = 0; q < 5; q++) Fw[q] = Fx[q]*np[0] + Fy[q]*np[1] + Fz[q]*np[2];
      } else if (grid.cutWallRiem) {
        // SOLID WALL AS A RIEMANN PROBLEM against the MIRROR state
        //     u_m = u - 2 (u.n) n,   rho_m = rho,   p_m = p
        // A Rusanov flux between the trace and its mirror collapses to
        //     F* = [ 0, (p + rho un^2 + lambda rho un) n, 0 ]
        // because the mass and energy fluxes cancel EXACTLY against the mirror
        // and the dissipation only survives in the normal momentum.  So the
        // wall is exactly conservative in mass and energy by construction, and
        // -- the point -- un now carries a restoring term.
        //
        // The pressure-only flux below has none: it imposes p n whatever the
        // normal velocity does, so no-penetration is never actually enforced,
        // the wall leaks, and nothing pulls the wall pressure back.  Measured
        // at M=0.3: Cp -15.9 at the stagnation point (theory +1.02) with wall
        // pressure going NEGATIVE (-0.32 against p_inf = 0.714).
        const real un  = W[1]*np[0] + W[2]*np[1] + W[3]*np[2];
        const real lam = fabs(un) + dgSoundSpeed(W[4], W[0]);
        const real pw  = W[4] + W[0]*un*un + lam*W[0]*un;
        Fw[0] = 0; Fw[1] = pw*np[0]; Fw[2] = pw*np[1]; Fw[3] = pw*np[2]; Fw[4] = 0;
      } else {                              // solid wall: pressure only (legacy)
        Fw[0] = 0; Fw[1] = W[4]*np[0]; Fw[2] = W[4]*np[1]; Fw[3] = W[4]*np[2]; Fw[4] = 0;
      }
      for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++)
        atomicAdd(&sR[m*5+q], -dS*Fw[q]*psi[m]);
    }
    // ---- cut faces:  - CLOSED INT (F*.n) psi_m dS -------------------------
    // The cut element OWNS the rule and computes both sides, depositing the
    // neighbour's share directly; dgRhsKernel skips faces whose neighbour is
    // cut, so nothing is double counted.
    if (grid.cutDbgMask & 2)
    for (i32 f = 0; f < 6; f++) {
      // ---- CONFORMING FACE (cut <-> Cartesian, full fluid face) ------------
      // NOT a mortar, and the distinction is not pedantic: a mortar exists to
      // couple NONCONFORMING faces, where the two sides carry different face
      // node sets and the flux must be L2-projected between them (dgFaceLift's
      // coarse/fine branch, ~line 1757 -- that is the only mortar in this
      // solver, and it belongs to AMR).  A cut element and its Cartesian
      // neighbour are ALWAYS the same level and share the whole face, so the
      // two sides have the IDENTICAL tensor face nodes.  One flux per shared
      // node, lifted natively on both sides.  There is nothing to project.
      // ONE flux per shared GLL face node, used by BOTH sides: my modal weak
      // integral over the tensor GLL face rule, and their native pointwise
      // sgn*jacDir*winv lift.  That is exactly the conforming-DG coupling, so
      // the neighbour keeps its SBP stability structure and the interface flux
      // is single-valued (conservative in the native discrete sense).  The
      // quadrature-lifted deposit that predated this broke that structure and
      // was the proven supersonic instability vector (mask-7 bisection).
      {
        const i32 dM = f/2, sideM = f%2;
        i32 oM[3] = {1,1,1}; oM[dM] = sideM ? 2 : 0;
        i32 nbM = grid.nbrIdxList[27*b + oM[0] + 3*oM[1] + 9*oM[2]];
        if (nbM == bEmpty) nbM = -1;
        const bool fullFace = fabs(grid.cutFacA[6*c+f] - (real)1.0) < (real)1e-6;
        // STARVATION COUNTER: a NON-cut neighbour across a PARTIAL face gets
        // nothing.  dgRhsKernel skips any face whose neighbour is cut, so that
        // neighbour's only source for this face is the deposit below -- and
        // that path requires the face to be fully fluid.  A face that is
        // partial with a non-cut neighbour is therefore a HOLE in the coupling.
        if (grid.cutDbg && threadIdx.x == 0 && nbM >= 0 && grid.blkCut[nbM] < 0
            && !fullFace && grid.cutFacOff[6*c+f+1] > grid.cutFacOff[6*c+f])
          atomicAdd(&grid.cutDbg[0], 1);
        if (grid.cutDbg && threadIdx.x == 0 && nbM >= 0 && grid.blkCut[nbM] < 0 && fullFace)
          atomicAdd(&grid.cutDbg[1], 1);
        if ((grid.cutDbgMask & 8) && fullFace && nbM >= 0 && grid.blkCut[nbM] < 0) {
          const real sgM  = sideM ? (real)1.0 : (real)-1.0;
          const real jacD = (real)2.0/h[dM];
          const i32  nrmN = sideM ? 0 : (NNODE-1);
          const real sgnN = sideM ? (real)1.0 : (real)-1.0;
          const i32  t1ax = (dM==0) ? 1 : 0, t2ax = (dM==2) ? 1 : 2;
          for (i32 fn = threadIdx.x; fn < NNODE*NNODE; fn += blockDim.x) {
            const i32 fa = fn % NNODE, fb = fn / NNODE;
            real xr[3]; xr[dM] = sideM ? (real)1.0 : (real)0.0;
            xr[t1ax] = (real)0.5*(c_xi[fa]+(real)1.0);
            xr[t2ax] = (real)0.5*(c_xi[fb]+(real)1.0);
            real psi[CUT_NBMAX];
            dgCutPsiO(cen, xr, nb, Lc, psi, nullptr);
            real Um[5];
            for (i32 q = 0; q < 5; q++) { real v = 0;
              for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; Um[q] = v; }
            dgCutLimitTrace(sC, Um, (real)1.0/Lc[0]);
            real Wm[5]; Wm[0]=fmax(Um[0],DG_EPSF);
            Wm[1]=Um[1]/Wm[0]; Wm[2]=Um[2]/Wm[0]; Wm[3]=Um[3]/Wm[0];
            Wm[4]=dgPressureFromCons(Um);
            i32 nodeIdx;
            { i32 ci3[3]; ci3[dM] = nrmN; ci3[t1ax] = fa; ci3[t2ax] = fb;
              nodeIdx = ci3[0] + NNODE*(ci3[1] + NNODE*ci3[2]); }
            real Uo[5], Wo[5];
            for (i32 q = 0; q < 5; q++)
              Uo[q] = grid.getField(D_RHO+q)[(size_t)nbM*blockSizeTot + nodeIdx];
            dgSanitizeCons(Uo);
            Wo[0]=fmax(Uo[0],DG_EPSF);
            Wo[1]=Uo[1]/Wo[0]; Wo[2]=Uo[2]/Wo[0]; Wo[3]=Uo[3]/Wo[0];
            Wo[4]=dgPressureFromCons(Uo);
            real fs[5];
            // SAME FLUX AS THE REST OF THE MESH.  This was hardcoded Rusanov while
            // every Cartesian face runs dgIfaceFlux (HLLC by default), so the wall
            // band carried a dissipation JUMP: Rusanov damps with lambda = |u.n| + c,
            // which at M = 0.2 is ~5x the convective scale (10x at M = 0.1, 20x at
            // M = 0.05), and it smears the contact and shear waves HLLC resolves --
            // right where the wall pressure is sampled.  --cutflux 0 restores it.
            if (grid.cutFlux) {
              if (sideM) dgIfaceFlux(grid, Wm, Wo, dM, (real)0, fs);
              else       dgIfaceFlux(grid, Wo, Wm, dM, (real)0, fs);
            } else {
              if (sideM) dgRusanovAxis(Wm, Wo, dM, fs); else dgRusanovAxis(Wo, Wm, dM, fs);
            }
            if (grid.avOn) {
              real nuMe = nuE;      // locally computed; the slot was never written
              real nuNb = grid.getField(D_SCRATCH)[(u64)nbM*blockSizeTot];
              real sig = sideM ? dgPenaltySigma(grid, nuMe, nuNb, Wm, Wo)
                               : dgPenaltySigma(grid, nuNb, nuMe, Wo, Wm);
              real Umc[5], Uoc[5]; dgP2C(Wm, Umc); dgP2C(Wo, Uoc);
              if (sideM) for (i32 q=0;q<5;q++) fs[q] -= sig*(Uoc[q]-Umc[q]);
              else       for (i32 q=0;q<5;q++) fs[q] -= sig*(Umc[q]-Uoc[q]);
            }
            // my side: weak face term over the GLL face rule (reference measure)
            const real w2d = (real)0.25*c_w[fa]*c_w[fb]/h[dM];
            for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++)
              atomicAdd(&sR[m*5+q], -sgM*w2d*fs[q]*psi[m]);
            // their side: the native pointwise lift of the SAME flux
            real fOwn[5]; dgEulerFluxAxis(Wo, dM, fOwn);
            for (i32 q = 0; q < 5; q++)
              atomicAdd(&grid.getField(D_RHS+q)[(size_t)nbM*blockSizeTot + nodeIdx],
                        sgnN*jacD*c_winv[nrmN]*(fs[q] - fOwn[q]));
          }
          continue;                        // this face is fully handled
        }
      }
      const i32 d = f/2, side = f%2;
      const i32 f0 = grid.cutFacOff[6*c+f], fn = grid.cutFacOff[6*c+f+1]-f0;
      if (fn == 0) continue;
      // same-level face neighbour.  The wall band is force-refined to the
      // finest level with several cells of margin, so a cut element's face
      // neighbours are always same-level -- no mortar case here.
      i32 o[3] = {1,1,1}; o[d] = side ? 2 : 0;
      i32 nbIdx = grid.nbrIdxList[27*b + o[0] + 3*o[1] + 9*o[2]];
      if (nbIdx == bEmpty) nbIdx = -1;
      const real sg = side ? (real)1.0 : (real)-1.0;
      const real dSf = (real)1.0/h[d];             // reference measure (see volume term)
      for (i32 g = threadIdx.x; g < fn; g += blockDim.x) {
        const SayeNode &s = grid.cutFacP[f0 + g];
        real psi[CUT_NBMAX], phi[blockSizeTot];
        dgCutPsiO(cen, s.x, nb, Lc, psi, nullptr);
        real Um[5];
        for (i32 q = 0; q < 5; q++) { real v = 0;
          for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m]; Um[q] = v; }
        dgCutLimitTrace(sC, Um, (real)1.0/Lc[0]);
        real Wm[5]; Wm[0]=fmax(Um[0],DG_EPSF);
        Wm[1]=Um[1]/Wm[0]; Wm[2]=Um[2]/Wm[0]; Wm[3]=Um[3]/Wm[0]; Wm[4]=dgPressureFromCons(Um);
        // neighbour trace at the SAME physical point: its reference coordinate
        // is ours with the face-normal coordinate flipped
        real Wo[5];
        if (nbIdx >= 0) {
          real xn[3] = { s.x[0], s.x[1], s.x[2] };
          xn[d] = (real)1.0 - xn[d];
          real Uo[5];
          const i32 nbc = (grid.cutModal && grid.blkCut) ? grid.blkCut[nbIdx] : -1;
          if (nbc >= 0) {
            // MODAL cut neighbour: evaluate ITS polynomial in ITS OWN basis.
            // The nodal path below would read its coefficients as if they were
            // node values.  This is also the honest trace: the neighbour's
            // tensor interpolant and its polynomial are different functions
            // once the element is degree-reduced or the state leaves the space.
            const real *cenN = grid.cutCen + 4*nbc;
            const real *LcN  = grid.cutLc + (size_t)nbc*CUT_NBMAX*CUT_NBMAX;
            const i32   nbN  = grid.cutNbOf[nbc];
            real psiN[CUT_NBMAX];
            dgCutPsiO(cenN, xn, nbN, LcN, psiN, nullptr);
            for (i32 q = 0; q < 5; q++) { real v = 0;
              for (i32 m = 0; m < nbN; m++)
                v += grid.getField(D_RHO+q)[(size_t)nbIdx*blockSizeTot + m]*psiN[m];
              Uo[q] = v; }
          } else {
          dgLag3(xn, phi);
          for (i32 q = 0; q < 5; q++) { real v = 0;
            for (i32 a = 0; a < blockSizeTot; a++)
              v += grid.getField(D_RHO+q)[(size_t)nbIdx*blockSizeTot + a]*phi[a];
            Uo[q] = v; }
          }
          dgSanitizeCons(Uo);
          Wo[0]=fmax(Uo[0],DG_EPSF);
          Wo[1]=Uo[1]/Wo[0]; Wo[2]=Uo[2]/Wo[0]; Wo[3]=Uo[3]/Wo[0]; Wo[4]=dgPressureFromCons(Uo);
        } else {
          for (i32 q = 0; q < 5; q++) Wo[q] = Wm[q];      // outflow / copy
        }
        real Fs[5];
        if (grid.cutFlux) {                    // see the note at the deposit above
          if (side) dgIfaceFlux(grid, Wm, Wo, d, (real)0, Fs);
          else      dgIfaceFlux(grid, Wo, Wm, d, (real)0, Fs);
        } else {
          if (side) dgRusanovAxis(Wm, Wo, d, Fs); else dgRusanovAxis(Wo, Wm, d, Fs);
        }
        // AV jump penalty, exactly as the Cartesian face path applies it: the
        // deposit is the only face flux these two elements exchange, so it must
        // carry the same dissipation the native path would.
        if (grid.avOn) {
          real nuMe = nuE;          // locally computed; the slot was never written
          real nuNb = (nbIdx >= 0) ? grid.getField(D_SCRATCH)[(u64)nbIdx*blockSizeTot] : nuMe;
          real sig = side ? dgPenaltySigma(grid, nuMe, nuNb, Wm, Wo)
                          : dgPenaltySigma(grid, nuNb, nuMe, Wo, Wm);
          real Umc[5], Uoc[5]; dgP2C(Wm, Umc); dgP2C(Wo, Uoc);
          if (side) for (i32 q=0;q<5;q++) Fs[q] -= sig*(Uoc[q]-Umc[q]);
          else      for (i32 q=0;q<5;q++) Fs[q] -= sig*(Umc[q]-Uoc[q]);
        }
        for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++)
          atomicAdd(&sR[m*5+q], -s.w*sg*Fs[q]*psi[m]*dSf);
        // (neighbour share handled node-pointwise below, outside this loop)
      }
    }
    __syncthreads();

    // ---- HIGH-ORDER MODAL HYPER-VISCOSITY (--cuthv tau) --------------------
    // Every dissipation this solver has is SENSOR-GATED: the cut element's own
    // LDG AV is nu = avCav * theta * (h/(2p+1)) * lambda with theta from the
    // Ducros/Persson shock sensor, so on smooth data theta = 0 and a cut
    // element receives NO dissipation at all.  The instability we are chasing
    // lives on a UNIFORM state (measured growth rate 13.2/time, e-fold 0.076,
    // independent of quadrature accuracy and of dt) -- precisely where every
    // gated mechanism is switched off.
    //
    // This is the ungated, high-order complement: damp mode m by
    //     sigma_m = tau * (lambda/h) * (deg(m)/N)^(2s)
    // which is zero for the constant mode, negligible for the low modes, and
    // O(tau*lambda/h) on the top mode -- the same shape as the dual-pairing
    // SBP hyper-viscosity (Hew, Duru et al., JCP 523 (2025) 113624), written
    // for a modal element whose mass matrix is the identity.  lambda/h is the
    // natural rate here and is ~13 for M=3 on h=0.25, i.e. the same order as
    // the growth rate it has to beat.
    if (grid.cutHv != (real)0) {
      real lamE = 0;
      { real Wm[5], Um[5];
        for (i32 q = 0; q < 5; q++) Um[q] = sC[0*5+q]*((real)1.0/Lc[0]);
        Wm[0]=fmax(Um[0],DG_EPSF); Wm[1]=Um[1]/Wm[0]; Wm[2]=Um[2]/Wm[0];
        Wm[3]=Um[3]/Wm[0]; Wm[4]=dgPressureFromCons(Um);
        lamE = fabs(Wm[1])+fabs(Wm[2])+fabs(Wm[3])+dgSoundSpeed(Wm[4],Wm[0]); }
      const real hmin = fmin(h[0], grid.pseudo2D ? h[0] : fmin(h[1],h[2]));
      // --cuthvgate 1: scale the filter by the Persson ramp, so it is strong
      // where the element's modal decay says it is troubled and OFF where the
      // solution is smooth.  Ungated (0) it damps the top modes everywhere,
      // which is what the original comment meant by "UNGATED".
      const real gate = grid.cutHvGate ? sLam[3] : (real)1.0;
      const real rate = gate*fabs(grid.cutHv)*lamE/fmax(hmin,(real)1e-30);
      if (rate > (real)0)
      for (i32 i = threadIdx.x; i < nb; i += blockDim.x) {
        // total degree of mode i, degree-major ordering (matches dgCutPsi)
        i32 dg = 0, m = 0;
        for (i32 d = 0; d <= dgOrder && m <= i; d++) {
          for (i32 a = d; a >= 0 && m <= i; a--)
            for (i32 b = d-a; b >= 0 && m <= i; b--) { if (m == i) dg = d; m++; }
        }
        // DIAGNOSTIC (cutHvMean): damp the MEAN too.  Not conservative -- this
        // exists to locate the unstable mode, not to be used.
        if (dg == 0 && !grid.cutHvMean) continue;
        real f;
        if (grid.cutHv < (real)0) { f = (real)1.0; }  // DIAGNOSTIC: flat damping
        else { f = (real)dg/(real)dgOrder; f = f*f; f = f*f; }   // (deg/N)^4
        for (i32 q = 0; q < 5; q++) sR[i*5+q] -= rate*f*sC[i*5+q];
      }
      __syncthreads();
    }

    // ---- PSEUDO-2D: kill the z-dependent modes of the residual --------------
    //  In a pseudo-2D run the geometry is a cylinder and the exact solution is
    //  EXACTLY z-invariant, so every mode with a z exponent > 0 is error.  It is
    //  nonetheless generated, and the mechanism is worth recording: the fitted
    //  volume rule is MOMENT-symmetric in z (measured: sum w (z-1/2) ~ 4e-16)
    //  but NOT POINTWISE symmetric -- individual weights differ from their
    //  z-mirrors by up to 4.2e-03, ~1% of the total.  That is invisible to a
    //  polynomial integrand of degree <= 2N, which only sees the moments, but
    //  the Euler flux is rational in the coefficients, so psi_m F(u) is not in
    //  that space and the asymmetry ALIASES into z-odd modes.  Measured wall Cp
    //  z-spread: 7e-09 on a uniform state (exact, F constant), 1.5e-03 at
    //  t = 0.1 with flow, 6.7e-02 by t = 5 at M = 0.5.
    //  Projecting them out is exact for this configuration, and conservative:
    //  mode 0 has z-exponent 0 and is untouched.
    if (grid.cutZ2d && grid.pseudo2D) {
      for (i32 i = threadIdx.x; i < nb; i += blockDim.x) {
        i32 kz = 0, m = 0;                       // z exponent of mode i
        for (i32 d = 0; d <= dgOrder && m <= i; d++)
          for (i32 a = d; a >= 0 && m <= i; a--)
            for (i32 b2 = d-a; b2 >= 0 && m <= i; b2--) { if (m == i) kz = d-a-b2; m++; }
        if (kz > 0) for (i32 q = 0; q < 5; q++) sR[i*5+q] = (real)0;
      }
      __syncthreads();
    }

    // ---- dc~/dt = R~ (identity mass!), then evaluate at the nodes ----------
    for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) sC[i] = sR[i];
    __syncthreads();
    if (grid.cutModal) {
      // the RHS is a coefficient vector too -- store it as one
      for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x)
        for (i32 q = 0; q < 5; q++)
          grid.getField(D_RHS+q)[(size_t)b*blockSizeTot + i] =
              (i < nb) ? sC[i*5+q] : (real)0;
    } else
    for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x) {
      i32 ii = i % NNODE, jj = (i/NNODE) % NNODE, kk = i/(NNODE*NNODE);
      real xr[3] = { (real)0.5*(c_xi[ii]+(real)1.0),
                     (real)0.5*(c_xi[jj]+(real)1.0),
                     (real)0.5*(c_xi[kk]+(real)1.0) };
      real psi[CUT_NBMAX];
      dgCutPsiO(cen, xr, nb, Lc, psi, nullptr);
      for (i32 q = 0; q < 5; q++) { real v = 0;
        for (i32 m = 0; m < nb; m++) v += sC[m*5+q]*psi[m];
        grid.getField(D_RHS+q)[(size_t)b*blockSizeTot + i] = v; }
    }
    __syncthreads();
  }
}

/* ════════════════════════════════════════════════════════════════════════
 *  ENTROPY-STABLE CUT RHS  (--cutes)   Taylor & Chan, arXiv:2412.13002
 *
 *  The operators are built on the host by DgSolver::buildCutEs (DgCutEs.cu);
 *  see that file's header for why the surface rule must BE the runtime
 *  interface rule.  Everything below is in the REFERENCE cell, so each
 *  direction-d contribution carries exactly one factor 1/h_d -- the same
 *  convention dgRhsCutKernel uses, and the same one that makes the wall term
 *  a Nanson transform without writing one: with n~_d = n_d/h_d,
 *      SUM_d (1/h_d) w_f n_d f*_d  =  w_f |n~| (f*.n_phys).
 * ════════════════════════════════════════════════════════════════════════ */

// v = ( (gam-s)/(gam-1) - rho|u|^2/(2p), rho u/p, -rho/p ), eta = -rho s/(gam-1)
__device__ __forceinline__ void dgEsEntVars(const real W[5], real v[5]) {
  const real rho = fmax(W[0], DG_EPSF), p = fmax(W[4], DG_EPSF);
  const real s  = log(p) - dgGam*log(rho);
  const real q2 = W[1]*W[1] + W[2]*W[2] + W[3]*W[3];
  v[0] = (dgGam - s)/(dgGam - (real)1.0) - rho*q2/((real)2.0*p);
  v[1] = rho*W[1]/p; v[2] = rho*W[2]/p; v[3] = rho*W[3]/p;
  v[4] = -rho/p;
}
// EXACT inverse of dgEsEntVars, transcribed from the validated host reference
// (DgEsCutTest.cu:95-101).  Two errors lived here and both were invisible in the
// CLOSED element -- the operator telescopes in whatever states it is handed, so a
// consistently-wrong round trip still gave dM/M0 = 0 exactly.  They only showed
// once a cut face was coupled to a Cartesian neighbour, whose RAW nodal state is
// the true one: the spurious jump between the round-tripped trace and the true
// neighbour state fed the Rusanov dissipation and broke free stream.
//   1. v[4] is NEGATIVE.  The host forms v[0] - vv2/(2*v5) with v5 = v[4] < 0,
//      i.e. it ADDS rho|u|^2/(2p) back; using vn = -v[4] and subtracting removed
//      it twice.
//   2. rho = (-v5 exp(s))^(-1/(gam-1)).  The expression that was here belongs to
//      a different entropy normalisation and differs by (gam-1)^(1/(gam-1))/vn.
__device__ __forceinline__ void dgEsEntVarsToPrim(const real v[5], real W[5]) {
  const real g1  = dgGam - (real)1.0;
  const real v5  = fmin(v[4], -DG_EPSF);            // strictly negative
  const real vv2 = v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
  const real s   = dgGam - g1*(v[0] - vv2/((real)2.0*v5));
  const real rho = pow(-v5*exp(s), -(real)1.0/g1);
  W[0] = fmax(rho, DG_EPSF);
  W[1] = v[1]/(-v5); W[2] = v[2]/(-v5); W[3] = v[3]/(-v5);
  W[4] = fmax(-W[0]/v5, DG_EPSF);
}
// numerically-stable logarithmic mean (Ismail & Roe, Appendix B)
__device__ __forceinline__ real dgEsLogMean(real a, real b) {
  const real z = a/b, f = (z - (real)1.0)/(z + (real)1.0), u = f*f;
  const real F = (u < (real)1e-2)
      ? ((real)1.0 + u/(real)3.0 + u*u/(real)5.0 + u*u*u/(real)7.0)
      : (log(z)/((real)2.0*f));
  return (a + b)/((real)2.0*F);
}
// Chandrashekar entropy-conservative two-point flux along axis d
__device__ __forceinline__ void dgEsEcFlux(const real WL[5], const real WR[5],
                                           i32 d, real F[5]) {
  const real bL = WL[0]/((real)2.0*WL[4]), bR = WR[0]/((real)2.0*WR[4]);
  const real rl = dgEsLogMean(WL[0], WR[0]);
  const real bl = dgEsLogMean(bL, bR);
  const real ra = (real)0.5*(WL[0] + WR[0]);
  const real ba = (real)0.5*(bL + bR);
  const real ua = (real)0.5*(WL[1]+WR[1]), va = (real)0.5*(WL[2]+WR[2]),
             wa = (real)0.5*(WL[3]+WR[3]);
  const real pa = ra/((real)2.0*ba);
  const real un = (d == 0) ? ua : ((d == 1) ? va : wa);
  F[0] = rl*un;
  F[1] = F[0]*ua; F[2] = F[0]*va; F[3] = F[0]*wa;
  F[1+d] += pa;
  const real h = (real)1.0/((real)2.0*bl*(dgGam - (real)1.0))
               - (real)0.5*(WL[1]*WR[1] + WL[2]*WR[2] + WL[3]*WR[3])
               + ua*ua + va*va + wa*wa;
  F[4] = F[0]*h + pa*un;
}

// PASS 1: the entropy projection, published to global.
//   vtil = Pq v(u_q),  Pq = Vq^T W   (the orthonormal mass is exactly I)
// It has to be a separate pass because a cut element reading a CUT neighbour's
// interface trace must see the neighbour's ENTROPY-PROJECTED state -- the same
// one the neighbour uses on its own side of that face -- or the two sides
// evaluate different fluxes and the face stops conserving.  A Cartesian
// neighbour has no entropy projection and is read raw, which is the ordinary
// treatment at an ES/non-ES interface and is still single-valued because this
// kernel computes the flux and deposits the neighbour's share itself.
__global__ void dgEsProjectKernel(DgSolver &grid) {
  extern __shared__ real shp[];
  const i32 c = blockIdx.x;
  if (c >= grid.nCutElem) return;
  const i32 b  = grid.cutBlk[c];
  const i32 nb = grid.cutNbOf[c];
  const i32 q0 = grid.esQOff[c], nq = grid.esQOff[c+1] - q0;
  real *sC  = shp;
  real *sVt = shp + (size_t)CUT_NBMAX*5;
  for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) {
    sC[i]  = grid.getField(D_RHO + (i%5))[(size_t)b*blockSizeTot + i/5];
    sVt[i] = 0;
  }
  __syncthreads();
  for (i32 i = threadIdx.x; i < nq; i += blockDim.x) {
    const real *Vq = grid.esVq + (size_t)(q0+i)*CUT_NBMAX_H;
    real U[5];
    for (i32 q = 0; q < 5; q++) { real t = 0;
      for (i32 m = 0; m < nb; m++) t += sC[m*5+q]*Vq[m]; U[q] = t; }
    dgSanitizeCons(U);
    real W[5]; W[0] = fmax(U[0], DG_EPSF);
    W[1] = U[1]/W[0]; W[2] = U[2]/W[0]; W[3] = U[3]/W[0];
    W[4] = dgPressureFromCons(U);
    real v[5]; dgEsEntVars(W, v);
    const real w = grid.esWq[q0+i];
    for (i32 m = 0; m < nb; m++)
      for (i32 q = 0; q < 5; q++) atomicAdd(&sVt[m*5+q], w*Vq[m]*v[q]);
  }
  __syncthreads();
  for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x)
    grid.esVtil[(size_t)c*CUT_NBMAX_H*5 + i] = sVt[i];
}

// ES_* bisection switches (DgMain reads the env into grid.esDbg):
//   bit 0  ES_CLOSED     : f*_a = the trace's own flux at CUT FACES too, and no
//                          neighbour deposit -- restores the configuration that
//                          measured dM/M0 = 1.5e-16 before the coupling went in
//   bit 1  ES_NODEPOSIT  : real interface flux, but do not deposit the
//                          neighbour's share (breaks conservation on purpose)
//   bit 2  ES_NOMETRIC   : drop every 1/h_d factor (pure reference measure, the
//                          host operator's convention)
__global__ void dgRhsCutEsKernel(DgSolver &grid) {
  extern __shared__ real sh[];
  const bool esClosed = (grid.esDbg & 1) != 0;
  const bool esNoDep  = (grid.esDbg & 2) != 0;
  const bool esNoMet  = (grid.esDbg & 4) != 0;
  const i32 c = blockIdx.x;
  if (c >= grid.nCutElem) return;
  const i32 b  = grid.cutBlk[c];
  const i32 nb = grid.cutNbOf[c];
  const i32 q0 = grid.esQOff[c], nq = grid.esQOff[c+1] - q0;
  const i32 f0 = grid.esFOff[c], nf = grid.esFOff[c+1] - f0;
  const size_t tq2 = (size_t)grid.esQ2Off[grid.nCutElem];

  i32 lvl, ib, jb, kb; grid.decode(grid.bLocList[b], lvl, ib, jb, kb);
  real h[3]; dgElemSize(grid, lvl, h);

  real *sWt  = sh;
  real *sVt  = sWt + (size_t)(nq+nf)*5;
  real *sMdu = sVt + (size_t)CUT_NBMAX*5;
  real *sC   = sMdu + (size_t)CUT_NBMAX*5;

  for (i32 i = threadIdx.x; i < nb*5; i += blockDim.x) {
    sC[i]  = grid.getField(D_RHO + (i%5))[(size_t)b*blockSizeTot + i/5];
    sVt[i] = grid.esVtil[(size_t)c*CUT_NBMAX_H*5 + i];   // published by pass 1
    sMdu[i] = 0;
  }
  __syncthreads();

  // ---- utilde at every hybridized point ------------------------------------
  for (i32 i = threadIdx.x; i < nq+nf; i += blockDim.x) {
    const real *V = (i < nq) ? (grid.esVq + (size_t)(q0+i)*CUT_NBMAX_H)
                             : (grid.esVf + (size_t)(f0+i-nq)*CUT_NBMAX_H);
    real v[5];
    for (i32 q = 0; q < 5; q++) { real t = 0;
      for (i32 m = 0; m < nb; m++) t += sVt[m*5+q]*V[m]; v[q] = t; }
    dgEsEntVarsToPrim(v, &sWt[(size_t)i*5]);
  }
  __syncthreads();

  const real *Qb = grid.esQ + grid.esQ2Off[c];
  const real *Eb = grid.esEmat + grid.esEOff[c];

  // ---- volume ---------------------------------------------------------------
  for (i32 i = threadIdx.x; i < nq; i += blockDim.x) {
    real r[5] = {0,0,0,0,0}, F[5];
    for (i32 d = 0; d < 3; d++) {
      const real *Qd = Qb + (size_t)d*tq2;
      const real jac = esNoMet ? (real)1.0 : (real)1.0/h[d];
      for (i32 j = 0; j < nq; j++) {
        const real a = Qd[(size_t)i*nq+j] - Qd[(size_t)j*nq+i];
        if (a == (real)0) continue;
        dgEsEcFlux(&sWt[(size_t)i*5], &sWt[(size_t)j*5], d, F);
        for (i32 q = 0; q < 5; q++) r[q] += jac*a*F[q];
      }
      for (i32 a2 = 0; a2 < nf; a2++) {
        const real Bd = grid.esWf[f0+a2]*grid.esNrm[3*(size_t)(f0+a2)+d];
        if (Bd == (real)0) continue;
        const real a = Eb[(size_t)a2*nq+i]*Bd;
        if (a == (real)0) continue;
        dgEsEcFlux(&sWt[(size_t)i*5], &sWt[(size_t)(nq+a2)*5], d, F);
        for (i32 q = 0; q < 5; q++) r[q] += jac*a*F[q];
      }
    }
    const real *Vq = grid.esVq + (size_t)(q0+i)*CUT_NBMAX_H;
    for (i32 m = 0; m < nb; m++)
      for (i32 q = 0; q < 5; q++) atomicAdd(&sMdu[m*5+q], -Vq[m]*r[q]);
  }

  // ---- surface: the interface flux, and the neighbour's share ---------------
  for (i32 a2 = threadIdx.x; a2 < nf; a2 += blockDim.x) {
    real r[5] = {0,0,0,0,0}, F[5];
    const real *Wa = &sWt[(size_t)(nq+a2)*5];
    const i32 own = grid.esOwner[f0+a2];

    // ---- the OTHER state, and the flux f*_d ---------------------------------
    real Fst[3][5];
    for (i32 d = 0; d < 3; d++) for (i32 q = 0; q < 5; q++) Fst[d][q] = 0;
    i32 nbIdx = -1, nbNode = -1;
    real Wo[5];
    if (own == 6) {
      // SOLID WALL as a Riemann problem against the mirror state.  Expressed in
      // the hybridized form the wall flux must be the f*_d whose normal
      // contraction is [0, p_w n, 0], i.e. f*_d = [0, p_w e_d, 0].
      real nt[3] = { grid.esNrm[3*(size_t)(f0+a2)+0]/h[0],
                     grid.esNrm[3*(size_t)(f0+a2)+1]/h[1],
                     grid.esNrm[3*(size_t)(f0+a2)+2]/h[2] };
      real nm = sqrt(nt[0]*nt[0] + nt[1]*nt[1] + nt[2]*nt[2]);
      if (nm > (real)0) {
        const real np[3] = { nt[0]/nm, nt[1]/nm, nt[2]/nm };
        const real un = Wa[1]*np[0] + Wa[2]*np[1] + Wa[3]*np[2];
        real pw;
        if (grid.cutFsp) {                        // transparent (free-stream gate)
          for (i32 d = 0; d < 3; d++) dgEulerFluxAxis(Wa, d, Fst[d]);
          pw = (real)0;
        } else {
          pw = grid.cutWallRiem
             ? Wa[4] + Wa[0]*un*un + (fabs(un) + dgSoundSpeed(Wa[4], Wa[0]))*Wa[0]*un
             : Wa[4];
          for (i32 d = 0; d < 3; d++) { for (i32 q = 0; q < 5; q++) Fst[d][q] = 0;
                                        Fst[d][1+d] = pw; }
        }
      }
    } else if (esClosed) {
      const i32 d = own/2;
      dgEulerFluxAxis(Wa, d, Fst[d]);          // ES_CLOSED: own flux, no exterior
    } else {
      const i32 d = own/2, side = own%2;
      i32 o[3] = {1,1,1}; o[d] = side ? 2 : 0;
      nbIdx = grid.nbrIdxList[27*b + o[0] + 3*o[1] + 9*o[2]];
      if (nbIdx == bEmpty) nbIdx = -1;
      nbNode = grid.esNode[f0+a2];
      bool have = false;
      if (nbIdx >= 0 && nbNode >= 0 && grid.blkCut[nbIdx] < 0) {
        real Uo[5];
        for (i32 q = 0; q < 5; q++)
          Uo[q] = grid.getField(D_RHO+q)[(size_t)nbIdx*blockSizeTot + nbNode];
        dgSanitizeCons(Uo);
        Wo[0] = fmax(Uo[0], DG_EPSF);
        Wo[1] = Uo[1]/Wo[0]; Wo[2] = Uo[2]/Wo[0]; Wo[3] = Uo[3]/Wo[0];
        Wo[4] = dgPressureFromCons(Uo);
        have = true;
      } else if (nbIdx >= 0 && grid.blkCut[nbIdx] >= 0) {
        // cut neighbour: evaluate ITS polynomial at the mirrored reference point
        const i32 cn = grid.blkCut[nbIdx];
        const i32 nbn = grid.cutNbOf[cn];
        const real *cenN = grid.cutCen + 4*cn;
        const real *LcN  = grid.cutLc  + (size_t)cn*CUT_NBMAX*CUT_NBMAX;
        real xr[3] = { grid.esXf[3*(size_t)(f0+a2)+0],
                       grid.esXf[3*(size_t)(f0+a2)+1],
                       grid.esXf[3*(size_t)(f0+a2)+2] };
        xr[d] = (real)1.0 - xr[d];
        real psiN[CUT_NBMAX];
        dgCutPsiO(cenN, xr, nbn, LcN, psiN, nullptr);
        // its ENTROPY-PROJECTED trace, so both sides of this face see the same
        // pair and the single-valued flux really is single-valued
        real vo[5];
        for (i32 q = 0; q < 5; q++) { real t = 0;
          for (i32 m = 0; m < nbn; m++)
            t += grid.esVtil[(size_t)cn*CUT_NBMAX_H*5 + m*5 + q]*psiN[m];
          vo[q] = t; }
        dgEsEntVarsToPrim(vo, Wo);
        have = true;
      }
      if (!have) for (i32 q = 0; q < 5; q++) Wo[q] = Wa[q];   // outflow
      real fs[5];
      if (side) dgRusanovAxis(Wa, Wo, d, fs); else dgRusanovAxis(Wo, Wa, d, fs);
      for (i32 q = 0; q < 5; q++) Fst[d][q] = fs[q];

      // the NEIGHBOUR's share of the SAME single-valued flux, lifted with ITS
      // native operator -- this is what keeps the face conservative
      if (!esNoDep && nbIdx >= 0 && nbNode >= 0 && grid.blkCut[nbIdx] < 0) {
        const real sgnN = side ? (real)1.0 : (real)-1.0;
        const i32  nrmN = side ? 0 : (NNODE-1);
        real fOwn[5]; dgEulerFluxAxis(Wo, d, fOwn);
        const real lift = sgnN*((real)2.0/h[d])*c_winv[nrmN];
        for (i32 q = 0; q < 5; q++)
          atomicAdd(&grid.getField(D_RHS+q)[(size_t)nbIdx*blockSizeTot + nbNode],
                    lift*(fs[q] - fOwn[q]));
      }
    }

    // ---- r_a = SUM_d B_d ( f*_d - SUM_i E[a][i] fEC(a,i) ) / h_d ------------
    for (i32 d = 0; d < 3; d++) {
      const real Bd = grid.esWf[f0+a2]*grid.esNrm[3*(size_t)(f0+a2)+d];
      if (Bd == (real)0) continue;
      real acc[5] = {0,0,0,0,0};
      for (i32 i = 0; i < nq; i++) {
        const real e = Eb[(size_t)a2*nq+i];
        if (e == (real)0) continue;
        dgEsEcFlux(Wa, &sWt[(size_t)i*5], d, F);
        for (i32 q = 0; q < 5; q++) acc[q] += e*F[q];
      }
      const real jacS = esNoMet ? (real)1.0 : (real)1.0/h[d];
      for (i32 q = 0; q < 5; q++) r[q] += (Bd*jacS)*(Fst[d][q] - acc[q]);
    }
    const real *Vf = grid.esVf + (size_t)(f0+a2)*CUT_NBMAX_H;
    for (i32 m = 0; m < nb; m++)
      for (i32 q = 0; q < 5; q++) atomicAdd(&sMdu[m*5+q], -Vf[m]*r[q]);
  }
  __syncthreads();

  // ---- dc~/dt = M^-1 (M du/dt) with M = I ---------------------------------
  for (i32 i = threadIdx.x; i < blockSizeTot; i += blockDim.x)
    for (i32 q = 0; q < 5; q++)
      grid.getField(D_RHS+q)[(size_t)b*blockSizeTot + i] =
          (i < nb) ? sMdu[i*5+q] : (real)0;
}
