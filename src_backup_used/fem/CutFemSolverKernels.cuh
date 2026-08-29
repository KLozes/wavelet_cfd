#ifndef CUT_FEM_SOLVER_KERNELS_H
#define CUT_FEM_SOLVER_KERNELS_H

#include "CutFemSolver.cuh"
#include "CutQuad.cuh"

// threads per CUDA block for the one-block-per-element kernels
static constexpr i32 FEM_EPB = 64;

void femSetKref(const real *K576);   // reference (h=1) Q1 elasticity matrix
void femInitFaceMass(void);          // 4x4 bilinear face mass matrix

// ---- mesh ---------------------------------------------------------------
__global__ void femBlockActiveKernel(CutFemSolver &S);   // flag empty blocks DELETE
__global__ void femPruneKernel(CutFemSolver &S);         // retire DELETE blocks
__global__ void femPhiKernel(CutFemSolver &S);           // nodal 1-jet (phi + normal) per block
__global__ void femSliverKernel(CutFemSolver &S);        // per-cut-cell fraction + crease flag

// ---- setup --------------------------------------------------------------
__global__ void femCutElemKernel(CutFemSolver &S);   // cut 24x24 + load + geometry
__global__ void femFullLoadKernel(CutFemSolver &S);  // interior element load
__global__ void femCutLoadKernel(CutFemSolver &S);   // scatter cut loads
__global__ void femDiagElemKernel(CutFemSolver &S);
__global__ void femDiagFaceKernel(CutFemSolver &S);

// ---- cyclic constraint  u_node = P u_real  ------------------------------
__global__ void femProlongKernel(CutFemSolver &S, const real *x);   // xn = P x
__global__ void femRestrictKernel(CutFemSolver &S, real *y);        // y  = P^T yn
__global__ void femDiagRestrictKernel(CutFemSolver &S, real *d);    // diag(P^T D P)

// ---- operator (node space) ----------------------------------------------
__global__ void femElemApplyKernel(CutFemSolver &S, const real *x, real *y);
__global__ void femFaceApplyKernel(CutFemSolver &S, const real *x, real *y);

// ---- diagnostics --------------------------------------------------------
__global__ void femErrorKernel(CutFemSolver &S);
// sample the level set (and the physical coordinates) on a structured grid
__global__ void femIsoSampleKernel(BladeSdf ls, real d0, real d1, real d2,
                                   i32 N0, i32 N1, i32 N2, float *xyz, float *phi);

// ---- dense vector helpers (length n) ------------------------------------
__global__ void femSetKernel(real *x, real v, i32 n);
__global__ void femDotKernel(const real *a, const real *b, i32 n, double *out);
__global__ void femAxpyKernel(real *y, const real *x, real a, i32 n);   // y += a x
__global__ void femXpayKernel(real *y, const real *x, real a, i32 n);   // y  = x + a y
__global__ void femJacobiKernel(real *z, const real *r, const real *d, i32 n);

#endif
