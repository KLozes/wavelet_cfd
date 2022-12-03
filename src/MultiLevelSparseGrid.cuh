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

  u32 baseGridSize[3] = {1,1,1};
  u32 baseGridSizeB[3] = {1,1,1};
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
    nBlocks = 0;

    cudaMallocManaged(&hashKeyList, nBlocksMaxPow2*sizeof(u64));
    cudaMallocManaged(&hashValueList, nBlocksMaxPow2*sizeof(u32));
    cudaMallocManaged(&blockLocList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&blockIdxList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&fieldDataList, nBlocksMax*nFields*sizeof(fieldData));
    cudaMallocManaged(&nbrDataList, nBlocksMax*sizeof(nbrData));

    cudaMemset(&hashKeyList, 1, nBlocksMaxPow2*sizeof(u64));
    cudaMemset(&hashValueList, 0, nBlocksMaxPow2*sizeof(u32));
    cudaMemset(&blockLocList, 1, nBlocksMax*sizeof(u64));
    cudaMemset(&blockIdxList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&fieldDataList, 0, nBlocksMax*nFields*sizeof(fieldData));
    cudaMemset(&nbrDataList, 0, nBlocksMax*sizeof(nbrData));

    // initialize the hashtable keys and value to bEmpty!
    for(u32 idx = 0; idx < nBlocksMax; idx++) {
      hashKeyList[idx] = kEmpty;
      hashValueList[idx] = bEmpty;
      blockLocList[idx] = kEmpty;
      blockIdxList[idx] = bEmpty;
    }

    // initialize the blocks of the base grid level
    for (u32 k=0; k<baseGridSizeB[2]; k++){
      for (u32 j=0; j<baseGridSizeB[1]; j++){
        for (u32 i=0; i<baseGridSizeB[0]; i++){
          activateBlock(0, i, j, k);
        }
      }
    }

    for(u32 idx = 0; idx < nBlocks; idx++) {
      printf("blockIdx = %d\n", blockIdxList[idx]);
      printf("blockLoc = %llu\n", blockLocList[idx]);
    }

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

  __host__ __device__ u32 getBlockIndex(u32 lvl, u32 i, u32 j=0, u32 k=0);

  __host__ __device__ void activateBlock(u32 lvl, u32 i, u32 j=0, u32 k=0);
  __host__ __device__ void deactivateBlock(u32 idx);

  __host__ __device__ void getDijk(u32 n, u32 &di, u32 &dj);

  __host__ __device__ u64 hash(u64 x);
  __host__ __device__ u32 hashInsertKeyValue(u64 key, u32 value=0);
  __host__ __device__ u32 hashDelete(u64 key);
  __host__ __device__ u32 hashGetValue(u64 key);
  __host__ __device__ u32 hashSetValue(u64 key, u32 value);

  __host__ __device__ u64 split(u32 a);
  __host__ __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j);
  __host__ __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j, u32 k);

  __host__ __device__ u32 compact(u64 w);
  __host__ __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j);
  __host__ __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j, u32 &k);

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
