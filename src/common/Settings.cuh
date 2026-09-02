#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
// Overridable per build: the DG solver requires blockSize == p+1, so a p=2
// build compiles with -DDG_ORDER=2 -DBLOCK_SIZE=3 (FV solvers keep 4).
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif
static constexpr int blockSize = BLOCK_SIZE;

// Block extent in z.  A pseudo-2D run collapses z entirely: the solver zeroes
// z-gradients, skips the z-flux and never evolves z-momentum, so the blockSize
// z-layers of a cubic block hold IDENTICAL data and are integrated redundantly.
// Building with -DCOLLAPSE_2D makes a block blockSize x blockSize x 1, which is
// the only way to recover the memory as well as the arithmetic (blockSizeTot is
// constexpr and sizes every allocation, stride and index).
// Undefined -> blockSizeZ == blockSize, so every use is a strict no-op in 3-D.
#ifdef COLLAPSE_2D
static constexpr int blockSizeZ = 1;
#else
static constexpr int blockSizeZ = blockSize;
#endif

// "no data at this pixel" for the image path.  A cut element's tensor nodes
// include points buried INSIDE the solid, where the solution is an
// unconstrained polynomial extension, and a DEAD block holds the frozen
// analytic IC forever -- painting either draws values that are not the
// solution.  paintField excludes VOID pixels from the autoscale (they would
// otherwise set the colour range) and maps them to 0, which is reserved.
static constexpr double kPaintVoid = -1e30;
static constexpr int haloSize = 2;
static constexpr int cudaBlockSize = 256;
static constexpr int cudaGridSize = 1000;
// max total number of cells across all blocks.  fieldData scales as
// nFields * blockSizeTot * (nCellsMax/blockSizeTot) * sizeof(real), so keep this
// modest for the 4 GB GTX 1650 (8M cells -> ~0.5 GB at the Euler solver's 16
// fields).  Overridable per build via -DNCELLS_MAX: the narrowband SDF stores
// only 2 fields/block, so wavesdf is compiled with a much larger cap (see the
// Makefile) to fit fine-resolution narrowbands.
#ifndef NCELLS_MAX
#define NCELLS_MAX 8000000
#endif
static constexpr int nCellsMax = NCELLS_MAX;

// Solver precision: DOUBLE by default (user's call, 2026-08-30); -DUSE_SINGLE
// opts back into float.  Measured on the RAE steady march: in float the residual
// ||dq/dt|| floors at ~5e-3 and stops, because dq cannot resolve below eps*|q|
// and the 1/dt in the norm amplifies that round-off (worst in the finest cells,
// which have the smallest dt).  The identical case in double runs 7 orders down
// to ~1.7e-6.  Every "convergence stall" chased before this -- adaptation, the
// limiter, ghosts vs ghost-free vs Brinkman, JFNK -- was that float floor.
// -DUSE_DOUBLE is still accepted so existing *_dp targets keep working.
#ifdef USE_SINGLE
typedef float real;
#else
typedef double real;
#endif

#endif
