#ifndef SETTINGS_H
#define SETTINGS_H

// compile time simulation settings

typedef float dataType;
typedef int i32;
typedef unsigned int u32;
typedef long long int i64;
typedef unsigned long long int u64;

static constexpr u32 nDim = 2;
static constexpr u32 blockSize = 4;
static constexpr u32 haloSize = 1;

static constexpr u32 cudaBlockSize = 256;
static constexpr u32 nBlocksMax = 100000;

#endif
