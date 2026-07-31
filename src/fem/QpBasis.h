#ifndef FEM_QPBASIS_H
#define FEM_QPBASIS_H

//
// Qp tensor-product Lagrange basis on the reference cell [0,1]^3, GLL nodes.
// Foundation of the matrix-free higher-order operator (M1):
//
//   * tensor GLL quadrature (nodes + weights)              -- uncut elements
//   * 1-D differentiation matrix D[i][a] = l_a'(t_i)        -- sum factorization
//   * barycentric eval / gradient at ARBITRARY points       -- Saye cut points
//     (the cut quadrature points are not tensor nodes, so the operator must
//      evaluate every basis function and its gradient off-node)
//
// Coordinate-agnostic: everything is on [0,1]^3; the Jacobian (h*I in Cartesian,
// the isoparametric Q_p geometry map in cylindrical) is applied by the operator.
//
// General p via barycentric weights; specialized nodes for p=1..3 match PolyFit.
//

#include "Util.cuh"

static constexpr i32 QP_MAX = 4;          // max supported order (p4: 5 nodes THROUGH THICKNESS in one
                                          // element -- p>=4 is where shear locking in thin structures
                                          // essentially vanishes, Szabo-Babuska p-version result)
static constexpr i32 QN_MAX = QP_MAX + 1; // nodes per axis

struct QpBasis {
  i32  p, n;                 // order, nodes/axis (= p+1)
  real t[QN_MAX];            // GLL nodes on [0,1]
  real wq[QN_MAX];           // GLL quadrature weights on [0,1]
  real bw[QN_MAX];           // barycentric weights
  real D[QN_MAX][QN_MAX];    // D[i][a] = l_a'(t_i)

  __host__ __device__ void init(i32 order) {
    p = order; n = p+1;
    // GLL nodes on [0,1]
    if (p == 1) { t[0]=0; t[1]=1; }
    else if (p == 2) { t[0]=0; t[1]=(real)0.5; t[2]=1; }
    else if (p == 3) { t[0]=0; t[1]=(real)0.2763932023; t[2]=(real)0.7236067977; t[3]=1; }
    // p=4: 5-pt GLL, [-1,1] nodes {-1,-sqrt(3/7),0,sqrt(3/7),1} mapped to [0,1]
    else { t[0]=0; t[1]=(real)0.1726731646; t[2]=(real)0.5; t[3]=(real)0.8273268354; t[4]=1; }
    // GLL quadrature weights on [0,1]  (= 1/2 * the [-1,1] weights)
    if (p == 1) { wq[0]=(real)0.5; wq[1]=(real)0.5; }
    else if (p == 2) { wq[0]=(real)(1.0/6); wq[1]=(real)(4.0/6); wq[2]=(real)(1.0/6); }
    else if (p == 3) { wq[0]=(real)(1.0/12); wq[1]=(real)(5.0/12); wq[2]=(real)(5.0/12); wq[3]=(real)(1.0/12); }
    // p=4: [-1,1] weights {1/10, 49/90, 32/45, 49/90, 1/10}, halved for [0,1]
    else { wq[0]=(real)(1.0/20); wq[1]=(real)(49.0/180); wq[2]=(real)(16.0/45);
           wq[3]=(real)(49.0/180); wq[4]=(real)(1.0/20); }
    // barycentric weights  bw[j] = 1 / prod_{k!=j} (t_j - t_k)
    for (i32 j = 0; j < n; j++) {
      real prod = 1;
      for (i32 k = 0; k < n; k++) if (k != j) prod *= (t[j] - t[k]);
      bw[j] = 1/prod;
    }
    // differentiation matrix  D[i][a] = l_a'(t_i)
    for (i32 i = 0; i < n; i++)
    for (i32 a = 0; a < n; a++) {
      if (i != a) D[i][a] = (bw[a]/bw[i]) / (t[i] - t[a]);
    }
    for (i32 i = 0; i < n; i++) {
      real s = 0;
      for (i32 a = 0; a < n; a++) if (a != i) s += D[i][a];
      D[i][i] = -s;                       // rows sum to zero
    }
  }

  // 1-D Lagrange basis values L[a] = l_a(x) at an arbitrary x (barycentric)
  __host__ __device__ void basis1(real x, real L[QN_MAX]) const {
    // exact-at-node guard
    for (i32 a = 0; a < n; a++) if (x == t[a]) {
      for (i32 b = 0; b < n; b++) L[b] = (b==a)?(real)1:(real)0;
      return;
    }
    real sum = 0;
    for (i32 a = 0; a < n; a++) { L[a] = bw[a]/(x - t[a]); sum += L[a]; }
    for (i32 a = 0; a < n; a++) L[a] /= sum;
  }

  // 1-D basis values AND derivatives at an arbitrary x
  __host__ __device__ void basis1d(real x, real L[QN_MAX], real dL[QN_MAX]) const {
    bool onNode = false; i32 na = -1;
    for (i32 a = 0; a < n; a++) if (x == t[a]) { onNode = true; na = a; }
    if (onNode) {
      for (i32 b = 0; b < n; b++) { L[b] = (b==na)?(real)1:(real)0; dL[b] = D[na][b]; }
      return;
    }
    real sum = 0, dsum = 0;
    for (i32 a = 0; a < n; a++) {
      real d = bw[a]/(x - t[a]);
      L[a] = d; sum += d;
      dsum += -bw[a]/((x-t[a])*(x-t[a]));
    }
    // l_a(x) = (bw_a/(x-t_a)) / sum ;  derivative via quotient rule
    for (i32 a = 0; a < n; a++) {
      real num = bw[a]/(x - t[a]);
      real dnum = -bw[a]/((x-t[a])*(x-t[a]));
      dL[a] = (dnum*sum - num*dsum)/(sum*sum);
      L[a]  = num/sum;
    }
  }

  // scalar field value at reference point x[3], nodal values u[n*n*n] (i fastest)
  __host__ __device__ real eval(const real x[3], const real *u) const {
    real Lx[QN_MAX], Ly[QN_MAX], Lz[QN_MAX];
    basis1(x[0], Lx); basis1(x[1], Ly); basis1(x[2], Lz);
    real s = 0;
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++)
      s += u[i + n*(j + n*k)]*Lx[i]*Ly[j]*Lz[k];
    return s;
  }

  // reference-cell gradient of a scalar field at x[3]
  __host__ __device__ void gradRef(const real x[3], const real *u, real g[3]) const {
    real Lx[QN_MAX], Ly[QN_MAX], Lz[QN_MAX], dx[QN_MAX], dy[QN_MAX], dz[QN_MAX];
    basis1d(x[0], Lx, dx); basis1d(x[1], Ly, dy); basis1d(x[2], Lz, dz);
    g[0]=g[1]=g[2]=0;
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++) {
      real c = u[i + n*(j + n*k)];
      g[0] += c*dx[i]*Ly[j]*Lz[k];
      g[1] += c*Lx[i]*dy[j]*Lz[k];
      g[2] += c*Lx[i]*Ly[j]*dz[k];
    }
  }

  // reference-cell gradients of ALL (p+1)^3 basis functions at x[3], packed
  // gb[3*a + d] = d/dx_d phi_a(x).  Used per Saye cut point (no sum factorization).
  __host__ __device__ void allGradRef(const real x[3], real *gb) const {
    real Lx[QN_MAX], Ly[QN_MAX], Lz[QN_MAX], dx[QN_MAX], dy[QN_MAX], dz[QN_MAX];
    basis1d(x[0], Lx, dx); basis1d(x[1], Ly, dy); basis1d(x[2], Lz, dz);
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++) {
      i32 a = i + n*(j + n*k);
      gb[3*a+0] = dx[i]*Ly[j]*Lz[k];
      gb[3*a+1] = Lx[i]*dy[j]*Lz[k];
      gb[3*a+2] = Lx[i]*Ly[j]*dz[k];
    }
  }

  // all (p+1)^3 basis VALUES at x[3], vb[a] = phi_a(x)  (Nitsche/Saye surface)
  __host__ __device__ void allVal(const real x[3], real *vb) const {
    real Lx[QN_MAX], Ly[QN_MAX], Lz[QN_MAX];
    basis1(x[0], Lx); basis1(x[1], Ly); basis1(x[2], Lz);
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++)
      vb[i + n*(j + n*k)] = Lx[i]*Ly[j]*Lz[k];
  }
};

#endif
