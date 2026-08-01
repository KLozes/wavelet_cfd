#pragma once
// ---------------------------------------------------------------------------
//  Uniform C^{p-1} B-spline basis for IMMERSED IGA (Finite Cell Method style).
//
//  The background grid is already uniform, so the knot vector is uniform with
//  unit spacing and there is no need to store it: on any knot span exactly p+1
//  basis functions are nonzero, and their values depend only on the local
//  parameter xi in [0,1].  That keeps the per-element interface identical to
//  QpBasis (p+1 functions per axis, evaluated at a reference point), which is
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
//  COEXISTENCE: this is ADDITIVE.  The C^0 Qp path (QpBasis) stays the default
//  and is untouched -- it is the validated one (sphere MMS order ~3.7, direct ==
//  iterative to 7 digits, GPU bit-identical to host).  IGA is to be opt-in, and
//  the two must remain switchable so results can be cross-checked against each
//  other; the deliberate interface match with QpBasis (p+1 functions per axis at
//  a reference point) is what makes a runtime switch possible rather than a fork.
//  Intended gate: --basis fem|iga, defaulting to fem.
// ---------------------------------------------------------------------------
#include "Util.cuh"

static constexpr i32 BS_PMAX = 4;
static constexpr i32 BS_NMAX = BS_PMAX + 1;

struct BsplineBasis {
  i32 p, n;                       // n = p+1 nonzero functions per axis per span

  __host__ __device__ void init(i32 order) { p = order; n = p + 1; }

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

  // Greville abscissa of local function k, measured from the span's left knot.
  // Needed for the linear-precision check and, later, for control-point BCs.
  __host__ __device__ real greville(i32 k) const { return (real)k - (real)(p - 1) * (real)0.5; }
};
