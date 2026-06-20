#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
static constexpr int blockSize = 4;
static constexpr int haloSize = 2;
static constexpr int cudaBlockSize = 256;
static constexpr int cudaGridSize = 1000;
// max total number of cells across all blocks.  fieldData scales as
// nFields * blockSizeTot * (nCellsMax/blockSizeTot) * sizeof(real), so keep this
// modest for the 4 GB GTX 1650 (8M cells -> ~0.5 GB of field storage).
static constexpr int nCellsMax = 8000000;

typedef float real;

#endif
