#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings

typedef float dataType;
typedef unsigned int u32;
typedef unsigned long long int u64;

static constexpr u32 nDim = 2;
static constexpr u32 blockSize = 4;
static constexpr u32 haloSize = 1;

static constexpr u32 cudaBlockSize = 256;
static constexpr u32 nBlocksMax = 100000;

#endif
