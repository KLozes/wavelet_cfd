#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include <thrust/sort.h>
#include <algorithm>
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

  u64 *locList; // block morton codes
  u32 *idxList; // block memory indices
  u32 *baseIdxArray; // base block memory index array

  u32 *prntList; // block parent indices
  chldArray *chldArrayList; // block child indices

  fieldArray *fieldArrayList; // flow field data arrays
  nbrArray *nbrArrayList; // cell neighbor index arrays

  MultiLevelSparseGrid(u32 *baseGridSize_, u32 nLvls_, u32 nFields_) {

    u32 nBaseBlocks = 1;
    for(int d=0; d<nDim; d++) {
      baseGridSize[d] = baseGridSize_[d];
      baseGridSizeB[d] = baseGridSize[d]/blockSize;
      nBaseBlocks *= baseGridSizeB[d];
    }
    nLvls = nLvls_;
    nFields = nFields_;

    assert(isPowerOf2(blockSize));
    nBlocks = 0;

    cudaMallocManaged(&locList, nBlocksMax*sizeof(u64));
    cudaMallocManaged(&idxList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&baseIdxArray, max(1024, nBaseBlocks)*sizeof(u32));
    cudaMallocManaged(&prntList, nBlocksMax*sizeof(u32));
    cudaMallocManaged(&chldArrayList, nBlocksMax*sizeof(chldArray));
    cudaMallocManaged(&fieldArrayList, nBlocksMax*nFields*sizeof(fieldArray));
    cudaMallocManaged(&nbrArrayList, nBlocksMax*sizeof(nbrArray));

    cudaMemset(&locList, 1, nBlocksMax*sizeof(u64));
    cudaMemset(&idxList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&baseIdxArray, 0, max(1024, nBaseBlocks)*sizeof(u32));
    cudaMemset(&prntList, 0, nBlocksMax*sizeof(u32));
    cudaMemset(&chldArrayList, 0, nBlocksMax*sizeof(chldArray));
    cudaMemset(&fieldArrayList, 0, nBlocksMax*nFields*sizeof(fieldArray));
    cudaMemset(&nbrArrayList, 0, nBlocksMax*sizeof(nbrArray));

    for(u32 idx = 0; idx < nBlocksMax; idx++) {
      locList[idx] = kEmpty;
      idxList[idx] = bEmpty;
    }


    // fill the locList with base grid blocks in cartesian order
    for (u32 k = 0; k < baseGridSizeB[2]; k++) {
      for (u32 j = 0; j < baseGridSizeB[1]; j++) {
        for (u32 i = 0; i < baseGridSizeB[0]; i++) {
          locList[nBlocks] = mortonEncode(0, i, j, k);
          nBlocks++;
        }
      }
    }

    // sort the locList
    thrust::sort(thrust::host, locList, locList+nBlocks);

    // fill the baseIdxArray and idxList
    for(u32 idx = 0; idx < nBlocks; idx++) {
      idxList[idx] = idx;
      u32 lvl, i, j, k;
      mortonDecode(locList[idx], lvl, i, j, k);
      getBaseIdx(i,j,k) = idx;
    }

    cudaDeviceSynchronize();
  }

  ~MultiLevelSparseGrid(void)
  {
    cudaDeviceSynchronize();
    cudaFree(locList);
    cudaFree(idxList);
    cudaFree(prntList);
    cudaFree(chldArrayList);
    cudaFree(fieldArrayList);
    cudaFree(nbrArrayList);
  }

  void initGrid(void);
  void sortBlocks(void);
  virtual void sortfieldArray(void){};

  __host__ __device__ u32 &getBaseIdx(u32 i, u32 j=0, u32 k=0);

  __device__ void activateBlock(u32 lvl, u32 i, u32 j=0, u32 k=0);
  __device__ void deactivateBlock(u32 lvl, u32 i, u32 j=0, u32 k=0);
  __device__ void deactivateBlock(u32 idx);

  __device__ void getDijk(u32 n, u32 &di, u32 &dj);

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
