#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings

typedef float dataType;
static constexpr uint nDim = 2;
static constexpr uint blockSize = 4;
static constexpr uint haloSize = 1;

static constexpr uint cudaBlockSize = 256;
static constexpr uint nBlocksMax = 100000;

#endif
