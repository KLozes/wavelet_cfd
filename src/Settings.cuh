#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings
static constexpr int blockSize = 4;
static constexpr int haloSize = 2;
static constexpr int cudaBlockSize = 256;
static constexpr int nBlocksMax = 1000000;

typedef float dataType;

#endif
