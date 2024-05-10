#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
static constexpr int blockSize = 4;
static constexpr int haloSize = 2;
static constexpr int cudaBlockSize = 512;
static constexpr int nCellsMax = 30000000;

typedef float dataType;

#endif
