#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
static constexpr int blockSize = 4;
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

// solver precision: float by default; -DUSE_DOUBLE builds the wave3d_dp binary
// (used for convergence studies where float roundoff floors the error)
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

#endif
