#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
static constexpr uint nDim = 2;
static constexpr uint blockSize = 4;
static constexpr uint haloSize = 1;
static constexpr uint cudaBlockSize = 256;
static constexpr uint nBlocksMax = 4000000;

typedef float dataType;

#endif
