#ifndef DG_SOLVER_KERNELS_H
#define DG_SOLVER_KERNELS_H

#include "DgSolver.cuh"

// host-side reference-element operator setup (builds in double, uploads to the
// constant-memory symbols owned by DgSolverKernels.cu) and its self-check
void dgUploadOperators(i32 gauss, i32 frType);
bool dgOperatorSelfTest(i32 gauss, i32 frType);
void dgGetHostOps(double *w, double *xi, i32 gauss);

// RHS kernel launch geometry: EPB elements of blockSizeTot nodes per CUDA
// block.  The in-kernel working set is sW[EPB][5][64] + the AV gradient banks
// sGx/y/z[EPB][5][64]; fp64 halves EPB to stay inside sm_75's shared memory.
#ifdef USE_DOUBLE
static constexpr i32 DG_EPB = 2;
#else
static constexpr i32 DG_EPB = 4;
#endif

// grid-stride loops where `continue` is safe (the shared START_*_LOOP macros
// are while-loops whose increment a `continue` would skip)
#define DG_BLOCK_LOOP(bIdx) \
  for (i32 bIdx = threadIdx.x + blockIdx.x*blockDim.x; \
       bIdx < grid.hashTable.nKeys; bIdx += gridDim.x*blockDim.x)

#define DG_CELL_LOOP(cIdx, bIdx) \
  for (i32 cIdx = threadIdx.x + blockIdx.x*blockDim.x, bIdx = cIdx/blockSizeTot; \
       bIdx < grid.hashTable.nKeys; \
       cIdx += gridDim.x*blockDim.x, bIdx = cIdx/blockSizeTot)

__global__ void dgSetICKernel(DgSolver &grid);
__global__ void dgRhsKernel(DgSolver &grid, real t);
__global__ void dgRhsGaussKernel(DgSolver &grid, real t);   // Gauss-Legendre flux reconstruction (--gauss)
__global__ void dgAvNuKernel(DgSolver &grid);   // per-element AV nu -> D_SCRATCH (face jump penalty)
__global__ void dgRk3StageKernel(DgSolver &grid, i32 stage, real dt);
__global__ void dgCopyQ0Kernel(DgSolver &grid);
__global__ void dgPositivityKernel(DgSolver &grid);
__global__ void dgEntropyLimitKernel(DgSolver &grid, real dt);  // ES limiter (docs/EntropyStableDG.pdf)
__global__ void dgIbGhostClampKernel(DgSolver &grid);          // clamp ghost states to bounds
__global__ void dgMoodResetKernel(DgSolver &grid);              // MOOD: alpha -> 0 (DG attempt)
__global__ void dgMoodDetectKernel(DgSolver &grid, i32 stage, real dt);  // MOOD: flag failed cells
__global__ void dgDpGammaKernel(DgSolver &grid);   // DP-SBP upwind parameters -> SCRATCH 8..12
__global__ void dgLamKernel(DgSolver &grid);
__global__ void dgSnapshotQ0Kernel(DgSolver &grid);
__global__ void dgSortFieldDataKernel(DgSolver &grid);
__global__ void dgPressureToScratchKernel(DgSolver &grid);
__global__ void dgBrinkPhiToScratchKernel(DgSolver &grid);        // stage phi(x) for a paint
__global__ void dgComputeImageDataKernel(DgSolver &grid, i32 f);   // LGL -> uniform-pixel interp

// MRA indicator (leaf-only, transient restriction on the octet anchor)
__global__ void dgScalesKernel(DgSolver &grid);
__global__ void dgFinalizeScalesKernel(DgSolver &grid);  // device-side c_i (no managed page ping-pong)
__global__ void dgRestrictToAnchorKernel(DgSolver &grid);   // Q0 bank (anchor) := virtual-parent nodal values
__global__ void dgDetailNormKernel(DgSolver &grid);         // LAM slab (anchor, nodes 0..4) += GLL detail norms
__global__ void dgVoteKernel(DgSolver &grid, real epsL, i32 allowRefine);  // MRA octet votes (allowRefine=0: coarsen-guard only)
__global__ void dgSensorVoteKernel(DgSolver &grid);          // indicator 1: theta_e hysteresis vote

// immersed boundary (ghost-element Hermite reconstruction)
__global__ void dgIbClassifyGeomKernel(DgSolver &grid);      // pass 1: SDF box range -> class
__global__ void dgIbPromoteKernel(DgSolver &grid);           // pass 2: DEAD facing FLUID -> GHOST
__global__ void dgIbBandVoteKernel(DgSolver &grid);          // pin |phi| < band to the finest level
__global__ void dgIbFillKernel(DgSolver &grid);              // wall-normal Hermite ghost fill
__global__ void dgIbSolidFillKernel(DgSolver &grid);        // SBM cut-cell solid-node bilinear fill
__global__ void dgIbCheckKernel(DgSolver &grid);             // --debug classification invariants
__global__ void dgIbClassToScratchKernel(DgSolver &grid);    // debug paint staging
__global__ void dgTroubledToScratchKernel(DgSolver &grid);   // troubled-element indicator paint
__global__ void dgBoundaryMassFluxKernel(DgSolver &grid, real *bnd);  // domain boundary mass flux
__global__ void dgIbSurfaceKernel(DgSolver &grid, i32 nTheta, real off, real *out);
__global__ void dgIbStagLineKernel(DgSolver &grid, i32 nS, real *out);
__global__ void dgStaticVoteKernel(DgSolver &grid);         // staticGrid target-level vote overrides
__global__ void dgNeighborRuleKernel(DgSolver &grid);       // Harten rule 2
__global__ void dgShockRefineKernel(DgSolver &grid, real thresh);  // shock sensor -> REFINE to finest
__global__ void dgSnapshotRefineKernel(DgSolver &grid);     // snapshot REFINE set (bounds the buffer to one ring)
__global__ void dgRefineBufferKernel(DgSolver &grid, real epsL);       // REFINE the neighbor ring of every REFINE element
__global__ void dgEnforceGradingKernel(DgSolver &grid);     // 2:1 fixpoint pass on target levels
__global__ void dgMergeVerdictKernel(DgSolver &grid);       // phase 1: octet mergeability -> snapValidList
__global__ void dgMergeApplyKernel(DgSolver &grid);         // phase 2: non-mergeable DELETE -> KEEP
__global__ void dgSpawnKernel(DgSolver &grid);              // create refine children / merge parents
__global__ void dgProlongChildrenKernel(DgSolver &grid);    // fill NEW children from REFINE parents
__global__ void dgDemoteRefinedKernel(DgSolver &grid);      // REFINE -> DELETE after prolongation
__global__ void dgRestrictParentsKernel(DgSolver &grid);    // fill NEW merge parents from their octets
__global__ void dgCheckLeafCoverKernel(DgSolver &grid);     // --debug: leaves tile the domain exactly once
__global__ void dgCheckFaceTopologyKernel(DgSolver &grid);  // --debug: every face resolves (no dropped fluxes)

#endif
