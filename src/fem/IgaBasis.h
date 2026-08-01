#pragma once
// ---------------------------------------------------------------------------
//  Uniform C^{p-1} B-spline basis for IMMERSED IGA (Finite Cell Method style).
//
//  The background grid is already uniform, so the knot vector is uniform with
//  unit spacing and there is no need to store it: on any knot span exactly p+1
//  basis functions are nonzero, and their values depend only on the local
//  parameter xi in [0,1].  That keeps the per-element interface identical to
//  a nodal basis (p+1 functions per axis at a reference point), which is
//  what lets the Saye / NNLS / metric / Nitsche machinery carry over unchanged.
//
//  The PAYOFF over C^0 Q_p is dof count: C^0 needs p*nCells+1 nodes per axis,
//  C^{p-1} splines need nCells+p -- about ONE dof per cell per axis instead of
//  p, i.e. ~p^3 fewer dofs in 3-D (~8x at p=2).  Plus far cleaner discrete
//  spectra, which is the reason to want this for the modal problem.
//
//  NOTE what does NOT transfer: "exact CAD geometry".  We are immersed, so the
//  geometry comes from the level-set fit + Saye rules, never from the basis.
//
//  THIS IS THE SOLUTION BASIS.  The C^0 Lagrange Q_p path it replaced has been
//  removed: at p2 splines matched its accuracy on ~6x fewer dofs and ~5x fewer
//  CG iterations on the blade, and at p3 Q_p did not converge there at all.
//  LagrangeBasis.h retains the C^0 collocation machinery (GLL nodes, barycentric
//  weights, the differentiation matrix) that the shifted-boundary paths need.
// ---------------------------------------------------------------------------
#include "Util.cuh"

static constexpr i32 BS_PMAX = 4;
static constexpr i32 BS_NMAX = BS_PMAX + 1;

struct IgaBasis {
  i32 p, n;                       // n = p+1 nonzero functions per axis per span

  // GEOMETRY sampling nodes.  The level-set fit (fitPoly3) samples phi at these
  // and fits a degree-p polynomial, which is what the Saye rules cut against.
  // That is a GEOMETRY operation, independent of the solution basis -- it stays
  // GLL for exactly the reason it always was: a unisolvent, well-conditioned set.
  real t[BS_NMAX], wq[BS_NMAX];
  // VOLUME quadrature: n-point Gauss, exact to degree 2n-1 = 2p+1 and so exact
  // for the stiffness integrand.  Splines have NO interpolation nodes, so the
  // GLL collocation the old C^0 path used (points == nodes) is unavailable.
  real qx[BS_NMAX], qw[BS_NMAX];

  __host__ __device__ void init(i32 order) {
    p = order; n = p + 1;
    if (p == 1) { t[0]=0; t[1]=1; }
    else if (p == 2) { t[0]=0; t[1]=(real)0.5; t[2]=1; }
    else if (p == 3) { t[0]=0; t[1]=(real)0.2763932023; t[2]=(real)0.7236067977; t[3]=1; }
    else { t[0]=0; t[1]=(real)0.1726731646; t[2]=(real)0.5; t[3]=(real)0.8273268354; t[4]=1; }
    if (p == 1) { wq[0]=(real)0.5; wq[1]=(real)0.5; }
    else if (p == 2) { wq[0]=(real)(1.0/6); wq[1]=(real)(4.0/6); wq[2]=(real)(1.0/6); }
    else if (p == 3) { wq[0]=(real)(1.0/12); wq[1]=(real)(5.0/12); wq[2]=(real)(5.0/12); wq[3]=(real)(1.0/12); }
    else { wq[0]=(real)(1.0/20); wq[1]=(real)(49.0/180); wq[2]=(real)(16.0/45);
           wq[3]=(real)(49.0/180); wq[4]=(real)(1.0/20); }
    if (n == 2)      { qx[0]=(real)0.2113248654; qx[1]=(real)0.7886751346;
                       qw[0]=(real)0.5;          qw[1]=(real)0.5; }
    else if (n == 3) { qx[0]=(real)0.1127016654; qx[1]=(real)0.5; qx[2]=(real)0.8872983346;
                       qw[0]=(real)(5.0/18);     qw[1]=(real)(4.0/9); qw[2]=(real)(5.0/18); }
    else if (n == 4) { qx[0]=(real)0.0694318442; qx[1]=(real)0.3300094782;
                       qx[2]=(real)0.6699905218; qx[3]=(real)0.9305681558;
                       qw[0]=(real)0.1739274226; qw[1]=(real)0.3260725774;
                       qw[2]=(real)0.3260725774; qw[3]=(real)0.1739274226; }
    else             { qx[0]=(real)0.0469100770; qx[1]=(real)0.2307653449; qx[2]=(real)0.5;
                       qx[3]=(real)0.7692346551; qx[4]=(real)0.9530899230;
                       qw[0]=(real)0.1184634425; qw[1]=(real)0.2393143352; qw[2]=(real)0.2844444444;
                       qw[3]=(real)0.2393143352; qw[4]=(real)0.1184634425; }
  }

  // Cox-de Boor (Piegl & Tiller A2.2) specialised to unit-spacing knots.
  // With u = i + xi on span i:  left[j] = xi + j - 1,  right[j] = j - xi.
  __host__ __device__ static void evalDeg(i32 q, real xi, real *N) {
    N[0] = (real)1;
    for (i32 j = 1; j <= q; j++) {
      real saved = 0;
      for (i32 r = 0; r < j; r++) {
        real right = (real)(r + 1) - xi;      // right[r+1]
        real left  = xi + (real)(j - r) - 1;  // left[j-r]
        real tmp   = N[r] / (right + left);   // (right+left) == j, never 0
        N[r]   = saved + right * tmp;
        saved  = left * tmp;
      }
      N[j] = saved;
    }
  }

  __host__ __device__ void val(real xi, real *N) const { evalDeg(p, xi, N); }

  // d/dxi.  Unit knot spacing => N'_{j,p} = N_{j,p-1} - N_{j+1,p-1}, and the
  // degree-(p-1) functions on this span are indexed one higher, so the p+1
  // derivatives are the differences of the p lower-degree values.
  __host__ __device__ void der(real xi, real *dN) const {
    real M[BS_NMAX];
    evalDeg(p - 1, xi, M);
    for (i32 k = 0; k <= p; k++) {
      real a = (k >= 1)  ? M[k - 1] : (real)0;
      real b = (k <= p - 1) ? M[k]  : (real)0;
      dN[k] = a - b;
    }
  }

  // 1-D basis values / values+derivatives at an arbitrary xi.
  __host__ __device__ void basis1(real x, real L[BS_NMAX]) const { val(x, L); }
  __host__ __device__ void basis1d(real x, real L[BS_NMAX], real dL[BS_NMAX]) const {
    val(x, L); der(x, dL); }

  // l-th derivative trace of the 1-D basis at a face.  At C^{p-1} the traces of
  // order l<p are CONTINUOUS across a face, so their ghost-penalty jump vanishes
  // identically -- only l==p survives, where the p-th derivative of a degree-p
  // B-spline is the piecewise constant (-1)^(p-k) C(p,k), the same at both faces.
  __host__ __device__ real dlFace(i32 l, i32 k) const {
    if (l != p) return (real)0;
    real c = 1;                                  // C(p,k)
    for (i32 i = 0; i < k; i++) c = c*(real)(p-i)/(real)(i+1);
    return ((p-k) & 1) ? -c : c;
  }

  // Greville abscissa of local function k, measured from the span's left knot.
  // Needed for the linear-precision check and, later, for control-point BCs.
  __host__ __device__ real greville(i32 k) const { return (real)k - (real)(p - 1) * (real)0.5; }

  // ---- tensor-product evaluators ----
  // Same local ordering a = i + n*(j + n*k) and same gb[3*a+d] gradient layout,
  // so qpElemCoreSaye / cutCellK / cutCylK / the Nitsche loops need only a basis
  // dispatch, not new math.  Gradients are w.r.t. the REFERENCE cell [0,1]^3;
  // the physical metric is applied by the caller exactly as for Qp.
  __host__ __device__ void allVal(const real x[3], real *vb) const {
    real Nx[BS_NMAX], Ny[BS_NMAX], Nz[BS_NMAX];
    val(x[0], Nx); val(x[1], Ny); val(x[2], Nz);
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++) vb[i + n*(j + n*k)] = Nx[i]*Ny[j]*Nz[k];
  }

  __host__ __device__ void allGradRef(const real x[3], real *gb) const {
    real Nx[BS_NMAX], Ny[BS_NMAX], Nz[BS_NMAX];
    real Dx[BS_NMAX], Dy[BS_NMAX], Dz[BS_NMAX];
    val(x[0], Nx); val(x[1], Ny); val(x[2], Nz);
    der(x[0], Dx); der(x[1], Dy); der(x[2], Dz);
    for (i32 k = 0; k < n; k++)
    for (i32 j = 0; j < n; j++)
    for (i32 i = 0; i < n; i++) {
      i32 a = i + n*(j + n*k);
      gb[3*a  ] = Dx[i]*Ny[j]*Nz[k];
      gb[3*a+1] = Nx[i]*Dy[j]*Nz[k];
      gb[3*a+2] = Nx[i]*Ny[j]*Dz[k];
    }
  }
};
