#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include "Settings.cuh"
#include "Util.cuh"
#include "HashTable.cuh"
/*
** A multilevel sparse grid data structure
*/

class MultiLevelSparseGrid : public Managed {
public:

  HashTable *hashTable;

  u32 baseGridSize[2] = {1,1};
  u32 nLvls;
  u32 nFields;

  u32 blockCounter;
  u32 nBlocks;

  u64 *locList; // block morton codes
  u32 *idxList; // block memory indices

  u32 *nbrIdxList; // cell neighbor indeces
  dataType *fieldData; // flow field data

  MultiLevelSparseGrid(u32 *baseGridSize_, u32 nLvls_, u32 nFields_);

  ~MultiLevelSparseGrid(void);

  void initGrid(void);
  void sortBlocks(void);
  virtual void sortFieldArray(void){};

  __host__ __device__ u32 &getBaseIdx(u32 i, u32 j);

  __device__ void activateBlock(u32 lvl, u32 i, u32 j);
  __device__ void deactivateBlock(u32 lvl, u32 i, u32 j);
  __device__ void deactivateBlock(u32 idx);

  __device__ void getDijk(u32 n, u32 &di, u32 &dj);

  __host__ __device__ u64 split(u32 a);
  __host__ __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j);

  __host__ __device__ u32 compact(u64 w);
  __host__ __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j);

  void resetBlockCounter(void);

};

/*
#define START_CELL_LOOP \
  u32 bIdx = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  u32 lvl, ib, jb; \
  grid.mortonDecode(grid.blockLoc[bIdx].loc, lvl, ib, jb); \
  u32 index = threadIdx.x % bSize; \
  u32 i = index % blockSize; \
  u32 j = index / blockSize; \
  while (bIdx < grid.nBlocks) {
*/

#define START_CELL_LOOP \
  u32 bIdx = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  u32 idx = threadIdx.x % blockSizeTot; \
  while (bIdx < grid.nBlocks) {

#define END_CELL_LOOP bIdx += gridDim.x; __syncthreads();}

#define START_BLOCK_LOOP \
  u32 bIdx = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIdx < grid.nBlocks) {

#define END_BLOCK_LOOP bIdx += gridDim.x;}

#define START_DYNAMIC_BLOCK_LOOP \
  __shared__ u32 startIndex; \
  __shared__ u32 endIndex; \
  while (grid.blockCounter < grid.nBlocks) { \
    if (threadIdx.x == 0) { \
      startIndex = atomicAdd(&(grid.blockCounter), blockDim.x); \
      endIndex = atomicMin(&(grid.blockCounter), grid.nBlocks); \
    } \
    __syncthreads(); \
    u32 bIdx = startIndex + threadIdx.x; \
    if ( bIdx < endIndex ) {

#define END_DYNAMIC_BLOCK_LOOP __syncthreads(); }}

#endif
