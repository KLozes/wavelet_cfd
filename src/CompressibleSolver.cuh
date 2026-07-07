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
// field layout (nFields = 17).  fields 0-4 alternate between conservative and
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
//   8..15  : shared scratch bank (see below)
//   16     : DeltaT / MagRhoU / pressure       (scratch, reused)
//
// Time stepping is low-storage (Williamson 2N) RK3, which needs only q plus
// ONE accumulator bank, so the former separate Old/Rhs banks are a single
// aliased bank (F_OLD == F_RHS == 8) whose uses are temporally disjoint:
//   - during the RK stages: the LSRK accumulator S (the RHS kernels
//     ACCUMULATE L into it; updateFields does q += B dt S, then S *= A_next)
//   - between steps (adaptation): the wavelet-transform reference snapshot
//     and the block-sort double-buffer
//   - during a CTU-Hancock step (mdFlux==2): the half-step predicted
//     primitives (the fused single-stage corrector updates q in place and
//     never touches the bank)
// Any 3-stage 3rd-order explicit RK shares the linear stability polynomial,
// so all measured CFL limits are unchanged vs the previous Shu-Osher SSP-RK3;
// only the formal SSP property is given up.
//
// Only fields 0..7 (NEVOLVE) are sorted, restricted, interpolated and wavelet-
// transformed; 8..16 are transient.  The multiD corner-flux path computes its
// corner tensors on the fly (no flux storage fields).
//
enum CompressibleField {
  F_RHO = 0, F_RHOU = 1, F_RHOV = 2, F_RHOW = 3, F_RHOE = 4,
  F_GX  = 5, F_GY  = 6, F_GZ  = 7,
  F_OLD = 8,       // shared bank 8..15 (snapshot / sort buffer / Hancock)
  F_RHS = 8,       // alias: the LSRK accumulator during the RK stages
  F_SCRATCH = 16
};
static constexpr i32 NEVOLVE = 8;                 // evolved DOFs (fields 0..7)
static constexpr i32 nCompressibleFields = 17;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;
  // wavelet-detail normalization: domain maxima of the 4 field scales
  // {|rho|, |momentum|, |rhoE|, max|grad|}, reduced device-side each adaptation.
  // (Local / neighbourhood normalization was tested and is Pareto-dominated by
  // global-with-tighter-threshold on single-feature flows; it over-refines
  // weak-feature regions.)
  real *globalScale;    // [4]  domain max of the 4 scales
  real waveletThresh;

  i32 scheme;           // 0 = finite volume (HLLC+TVD), 1 = RT0/P0 DG
  i32 recon;            // face reconstruction of rho/p/tangential (and FV normal) velocity:
                        // 0 = smooth TVD limiter, 1 = ROUND (default), 2 = LD-ROUND,
                        // 3 = unlimited 3rd-order parabola (kappa=1/3; smooth tests only)
                        // (ROUND/LD-ROUND: Huang, Deng, Matar & Ying, JCP 555 (2026), Eqs. 4.1/4.2)
                        // ROUND: 6-7x lower smooth-wave error than TVD, cleaner low-Mach,
                        // shocks stay spike-free (soft ~1% non-TVD overshoots by design)
  i32 rt0Face;          // RT0 normal-velocity face state (scheme==1 only):
                        // 0 = linear modal (default), 1 = c=1/6 biased parabola
                        // (4th-order face average; see parabolicFace)
  i32 mdFlux;           // 1 = genuinely multidimensional Osher-type corner flux
                        // (Gaburro, Ricchiuto & Dumbser, arXiv:2506.00207, Eq. 23)
                        // with FIRST-ORDER corner states: FV = P0 cell averages,
                        // RT0 = P0 rho,E + RT0 modal momentum at the corner.
                        // pseudo-2D only; recon/rt0Face/reflux do not apply.
  real vortexAdvect;    // isentropic-vortex IC advection velocity (u0=v0)
  real greshoP0;        // Gresho-vortex background pressure = 1/(gam*Ma^2) (sets Mach)
  i32 staticGrid;       // 1 = fixed refinement (no dynamic wavelet adaptation)
  real refineRadius;    // static-grid: outer radius of the level-1 refinement shell (about the domain centre)

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;
  long tForwardUs;      // profiling: time in forwardWaveletTransform (the 6 max reductions)
  long tSortUs;         // profiling: time in sortBlocks (hash rebuild + Morton sort)

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
      recon = 1;
      rt0Face = 0;
      mdFlux = 0;
      vortexAdvect = 0.0;
      greshoP0 = 0.0;
      staticGrid = 0;
      refineRadius = 0.4;

      tGrid = 0.0;
      tSolver = 0.0;
      tOutput = 0.0;
      tTotal = 0.0;
      tForwardUs = 0;
      tSortUs = 0;
  }

  void initialize(void);
  real step(real dt);
  void sortFieldData(void);
  void setInitialConditions(void);
  void setBoundaryConditions(i32 fOff = 0);   // fOff selects the state bank (0 = live fields)
  void conservativeToPrimitive(void);
  void primitiveToConservative(void);
  void forwardWaveletTransform(void);
  void inverseWaveletTransform(void);

  void computeDeltaT(void);
  void computeRightHandSide(void);
  void updateFields(i32 stage);
  void zeroAccumulator(void);   // zero the shared bank before LSRK stage 1
#ifdef USE_MGPU
  void haloExchange(i32 fOff, i32 nf);   // fill partition-boundary ghost blocks from owners
  void rebuildGhosts(void);              // recreate the 2-ring ghost layer from neighbors' blocks
#endif

  void restrictFields();
  void interpolateFields();

  void writeLineProfile(const char *fileName); // 1D profile dump for validation
  void computeAcousticReflection(const char *fileName); // acoustic wave reflection at coarse/fine interface
  void computeAcousticL2Error(void);            // L2 velocity error for the periodic sine wave (order study)
  void printDiagnostics(void);                  // AMR-boundary spike / pseudo-2D diagnostics
  void computeVortexError(void);                // L2 error vs the exact stationary isentropic vortex
  void computeGreshoError(void);                // L2 velocity error + KE retention vs the exact Gresho vortex
  void totalConserved(double &mass, double &momx, double &energy); // domain totals of the conserved variables
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
