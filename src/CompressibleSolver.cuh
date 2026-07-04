#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

static constexpr real gam = 1.4;

//
// 3D compressible Euler solver.  Two discretizations share the same data layout,
// selected by `scheme`:
//   scheme 0 : finite volume    (HLLC flux + TVD reconstruction + TVD-RK3)
//   scheme 1 : RT0/P0 DG        (ported from ../fvStuff/rt_dg_euler_2d.cu)
//              density ρ, energy E : P0 (cell average)
//              momentum ρu         : RT0 (cell-average mode + per-axis slope mode)
//
// The RT0 slope DOFs are stored as the *physical* momentum gradients
//   Gx = ∂(ρu)/∂x,  Gy = ∂(ρv)/∂y,  Gz = ∂(ρw)/∂z          (modal mxs = (dx/2)·Gx)
// so that, being level-independent smooth fields, they ride the existing
// interpolating-wavelet AMR machinery unchanged.  The FV scheme leaves them 0.
//
// field layout (nFields = 25).  fields 0-4 alternate between conservative and
// primitive storage in place (see conservative/primitive conversion kernels):
//
//   0 : Rho  | Rho
//   1 : RhoU | U          (momentum cell-average mxa)
//   2 : RhoV | V          (mya)
//   3 : RhoW | W          (mza)
//   4 : RhoE | P
//   5 : Gx               RT0 x-momentum slope  (∂(ρu)/∂x)
//   6 : Gy               RT0 y-momentum slope  (∂(ρv)/∂y)
//   7 : Gz               RT0 z-momentum slope  (∂(ρw)/∂z)
//   8..15  : Old{0..7}                        (RK3 substep storage)
//   16..23 : Rhs{0..7}                        (right hand side accumulator)
//   24     : DeltaT / MagRhoU / pressure       (scratch, reused)
//
enum CompressibleField {
  F_RHO = 0, F_RHOU = 1, F_RHOV = 2, F_RHOW = 3, F_RHOE = 4,
  F_GX  = 5, F_GY  = 6, F_GZ  = 7,
  F_OLD = 8,       // Old{0..7} occupy  8..15
  F_RHS = 16,      // Rhs{0..7} occupy 16..23
  F_SCRATCH = 24,
  F_FLUX = 25      // Flux{rho,rhou,rhov,rhow,rhoE} at 25..29 — one lower-face
                   // conserved-flux vector per cell, reused for each dimension
                   // sweep (used only on the refluxing RHS path)
};
static constexpr i32 NEVOLVE = 8;                 // evolved DOFs (fields 0..7)
static constexpr i32 NCONS   = 5;                 // conserved vars carried by the flux array
static constexpr i32 nCompressibleFields = 30;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;
  real maxRho;
  real maxMagRhoU;
  real maxRhoE;
  real maxMagGrad;      // scale for thresholding the RT0 slope DOFs (Gx,Gy,Gz)
  real waveletThresh;

  i32 scheme;           // 0 = finite volume (HLLC+TVD), 1 = RT0/P0 DG
  real vortexAdvect;    // isentropic-vortex IC advection velocity (u0=v0)
  real greshoP0;        // Gresho-vortex background pressure = 1/(gam*Ma^2) (sets Mach)
  i32 staticGrid;       // 1 = fixed refinement (no dynamic wavelet adaptation)
  real refineRadius;    // static-grid: outer radius of the level-1 refinement shell (about the domain centre)
  i32 reflux;           // 1 = conservative flux correction at coarse/fine interfaces (per-dim flux-array RHS)
  i32 basisGhost;       // 1 = fill coarse/fine ghost cells from the RT0 (momentum) / P0 (rho,E) basis instead of DD

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;

  i32 immerserdBcType;
  i32 bcType;
  i32 icType;

  i32 iter;

  CompressibleSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nCompressibleFields) {
      cfl = .5;
      waveletThresh = .005;
      iter = 0;
      immerserdBcType = 0;
      bcType = 0;
      icType = 0;
      scheme = 0;
      maxMagGrad = 1.0;
      vortexAdvect = 0.0;
      greshoP0 = 0.0;
      staticGrid = 0;
      refineRadius = 0.4;
      reflux = 0;
      basisGhost = 2;   // default coarse/fine ghost fill: 0=DD, 1=RT0/P0, 2=monotone trilinear

      tGrid = 0.0;
      tSolver = 0.0;
      tOutput = 0.0;
      tTotal = 0.0;
  }

  void initialize(void);
  real step(real dt);
  void sortFieldData(void);
  void setInitialConditions(void);
  void setBoundaryConditions(void);
  void conservativeToPrimitive(void);
  void primitiveToConservative(void);
  void forwardWaveletTransform(void);
  void inverseWaveletTransform(void);

  void computeDeltaT(void);
  void computeRightHandSide(void);
  void updateFields(i32 stage);

  void restrictFields();
  void interpolateFields();

  void writeLineProfile(const char *fileName); // 1D profile dump for validation
  void printDiagnostics(void);                  // AMR-boundary spike / pseudo-2D diagnostics
  void computeVortexError(void);                // L2 error vs the exact stationary isentropic vortex
  void computeGreshoError(void);                // L2 velocity error + KE retention vs the exact Gresho vortex
  void paintPressure(const char *fileName);     // render the pressure field to a png

  __device__ Vec5 prim2cons(Vec5 prim);
  __device__ Vec5 cons2prim(Vec5 cons);
  __device__ real pressureRT(real rho, real mxa, real mya, real mza, real E);
  __device__ real lim(real &r);
  __device__ real tvdRec(real &ul, real &uc, real &ur);
  __device__ Vec5 hllcFlux(Vec5 qL, Vec5 qR, Vec3 normal);

  __device__ real getBoundaryLevelSet(Vec3 pos);
  __device__ real calcIbMask(real phi);

};

#endif
