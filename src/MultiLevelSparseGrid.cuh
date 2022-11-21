#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "Settings.cuh"
#include "Util.cuh"

/*
** A multilevel sparse grid data structure
*/

struct Index {
  u32 ind[nDim];
};

class MultiLevelSparseGrid : public Managed {
public:

  u32 baseGridSize[nDim];
  u32 nLvls;
  u32 nLvlsMax;
  u32 nBlocks;
  u32 nFields;

  u64 *hashKeyList; // hash keys are block morton codes
  u32 *hashValueList; // hash values are block memory indices

  u32 blockCounter;
  u64 *blockLocList; // block morton codes
  u32 *blockIndexList; // block memory indices

  typedef Array<dataType, blockSize> fieldData;
  fieldData *fieldDataList;

  typedef Array<u32, blockSize+2*haloSize> nbrData;
  nbrData *nbrDataList;

public:
  MultiLevelSparseGrid(u32 baseSize_[nDim], u32 nLvlsMax_, u32 nFields_)
  {
    for(int d=0; d<nDim; d++){
      baseGridSize[d] = baseSize_[d];
    }
    nLvlsMax = nLvlsMax_;
    nFields = nFields_;

    assert(isPowerOf2(blockSize));

    nBlocks = 1;
    for(int d=0; d<nDim; d++) {
      assert(baseGridSize[d]%blockSize == 0);
      nBlocks *= baseGridSize[d]/blockSize;
    }

    cudaMallocManaged(&hashKeyList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&hashValueList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&blockLocList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&blockIndexList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&fieldDataList, nBlocksMax*nFields*sizeof(fieldData));
    cudaMallocManaged(&nbrDataList, nBlocksMax*sizeof(nbrDataList));

    cudaMemset(&blockLocList, 0, nBlocksMax*sizeof(u64));
    cudaMemset(&hashValueList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&blockLocList, 0, nBlocksMax*sizeof(u64));
    cudaMemset(&blockIndexList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&fieldDataList, 0, nBlocksMax*nFields*sizeof(fieldData));
    cudaMemset(&nbrDataList, 0, nBlocksMax*sizeof(nbrDataList));

    cudaDeviceSynchronize();
  }

  __host__ ~MultiLevelSparseGrid(void)
  {
    cudaDeviceSynchronize();
    cudaFree(blockLocList);
    cudaFree(fieldDataList);
  }

  void initGrid(void);

  void sortBlocks(void);

  __host__ __device__ u32& getBaseBlockIndex(int i, int j);

  __device__ u32 getBlockIndex(u32 lvl, u32 i, u32 j);

  __device__ void activateBlock(u32 lvl, u32 i, u32 j);

  __device__ void deactivateBlock(u32 bIndex);

  __device__ void deactivateBlock(u32 lvl, u32 i, u32 j);

  __device__ void getDijk(u32 n, u32 &di, u32 &dj);

  __device__ dataType& getFieldValue(u32 fIndex, u32 bIndex, u32 i, u32 j);

  __device__ dataType& getParentFieldValue(u32 fIndex, u32 bIndex, u32 ib, u32 jb, u32 i, u32 j);


  __device__ u64 hash(u64 x);
  __device__ void hashInsert(u64 key, u32 value);
  __device__ void hashDelete(u64 key);
  __device__ u32 hashGetValue(u64 key);

  __device__ u64 split(u32 a);
  __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j);
  __device__ u64 mortonEncode(u64 lvl, u32 i, u32 j, u32 k);

  __device__ u32 compact(u64 w);
  __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j);
  __device__ void mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j, u32 &k);

  virtual void sortFields(void){};

  void resetBlockCounter(void);

};

/*
#define START_CELL_LOOP \
  u32 bIndex = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  u32 lvl, ib, jb; \
  grid.mortonDecode(grid.blockLoc[bIndex].loc, lvl, ib, jb); \
  u32 index = threadIdx.x % bSize; \
  u32 i = index % blockSize; \
  u32 j = index / blockSize; \
  while (bIndex < grid.nBlocks) {
*/


#define START_CELL_LOOP \
  u32 bIndex = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  u32 index = threadIdx.x % blockSizeTot; \
  while (bIndex < grid.nBlocks) {

#define END_CELL_LOOP bIndex += gridDim.x; __syncthreads();}

#define START_BLOCK_LOOP \
  u32 bIndex = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIndex < grid.nBlocks) {

#define END_BLOCK_LOOP bIndex += gridDim.x;}

#define START_DYNAMIC_BLOCK_LOOP \
  __shared__ u32 startIndex; \
  __shared__ u32 endIndex; \
  while (grid.blockCounter < grid.nBlocks) { \
    if (threadIdx.x == 0) { \
      startIndex = atomicAdd(&(grid.blockCounter), blockDim.x); \
      endIndex = atomicMin(&(grid.blockCounter), grid.nBlocks); \
    } \
    __syncthreads(); \
    u32 bIndex = startIndex + threadIdx.x; \
    if ( bIndex < endIndex ) {
#define END_DYNAMIC_BLOCK_LOOP __syncthreads(); }}

#endif
