#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "Settings.cuh"
#include "Util.cuh"

/*
** A multilevel sparse grid data structure
*/
//struct Index {
//  u32 ind[nDim];
//};

class MultiLevelSparseGrid : public Managed {
public:

  u32 baseGridSize[nDim];
  u32 baseGridSizeB[nDim];
  u32 nLvls;
  u32 nFields;

  u32 blockCounter;
  i32 nBlocks;

  u64 *hashKeyList; // hash keys are block morton codes
  u32 *hashValueList; // hash values are block memory indices

  u64 *blockLocList; // block morton codes
  u32 *blockIdxList; // block memory indices

  typedef Array<dataType, blockSize> fieldData;
  fieldData *fieldDataList;

  typedef Array<u32, blockSize+2*haloSize> nbrData;
  nbrData *nbrDataList;

  MultiLevelSparseGrid(u32 *baseGridSize_, u32 nLvls_, u32 nFields_) {

    for(int d=0; d<nDim; d++) {
      baseGridSize[d] = baseGridSize_[d];
      baseGridSizeB[d] = baseGridSize[d]/blockSize;
    }
    nLvls = nLvls_;
    nFields = nFields_;

    assert(isPowerOf2(blockSize));
    nBlocks = 0; // the empty block at index 0

    cudaMallocManaged(&hashKeyList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&hashValueList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&blockLocList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&blockIdxList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&fieldDataList, nBlocksMax*nFields*sizeof(fieldData));
    cudaMallocManaged(&nbrDataList, nBlocksMax*sizeof(nbrData));

    cudaMemset(&hashKeyList, 1, nBlocksMax*sizeof(u64));
    cudaMemset(&hashValueList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&blockLocList, 1, nBlocksMax*sizeof(u64));
    cudaMemset(&blockIdxList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&fieldDataList, 0, nBlocksMax*nFields*sizeof(fieldData));
    cudaMemset(&nbrDataList, 0, nBlocksMax*sizeof(nbrData));

    cudaDeviceSynchronize();
  }

  ~MultiLevelSparseGrid(void)
  {
    cudaDeviceSynchronize();
    cudaFree(hashKeyList);
    cudaFree(hashValueList);
    cudaFree(blockLocList);
    cudaFree(blockIdxList);
    cudaFree(fieldDataList);
    cudaFree(nbrDataList);
  }

  void initGrid(void);
  void sortBlocks(void);
  void sortHashTable(void);
  virtual void sortFields(void){};

  __device__ u32 getBlockIndex(u32 lvl, u32 i, u32 j=0, u32 k=0);

  __device__ void activateBlock(u32 lvl, u32 i, u32 j=0, u32 k=0);
  __device__ void deactivateBlock(u32 lvl, u32 i, u32 j=0, u32 k=0);

  __device__ void getDijk(u32 n, u32 &di, u32 &dj);

  __device__ u64 hash(u64 x);
  __device__ u32 hashInsert(u64 key);
  __device__ u32 hashDelete(u64 key);
  __device__ u32 hashGetValue(u64 key);
  __device__ u32 hashSetValue(u64 key, u32 value);

  __device__ u64 split(u32 a);
  __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j);
  __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j, u32 k);

  __device__ u32 compact(u64 w);
  __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j);
  __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j, u32 &k);

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
  u32 index = threadIdx.x % blockSizeTot; \
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
