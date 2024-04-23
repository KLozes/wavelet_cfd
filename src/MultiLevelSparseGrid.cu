#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

#include <thrust/sort.h>
#include <algorithm>
#include <thrust/execution_policy.h>

MultiLevelSparseGrid::MultiLevelSparseGrid(u32 *baseGridSize_, u32 nLvls_, u32 nFields_) {

  u32 nBaseBlocks = 1;
  for(int d=0; d<2; d++) {
    baseGridSize[d] = baseGridSize_[d];
    nBaseBlocks *= baseGridSize[d];
  }
  nLvls = nLvls_;
  nFields = nFields_;

  assert(isPowerOf2(blockSize));

  hashTable = new HashTable();
  cudaMallocManaged(&locList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&idxList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&nbrIdxList, blockHaloSizeTot*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&fieldData, nFields*blockSizeTot*nBlocksMax*sizeof(dataType));

  cudaMemset(&locList, 0x11111111, nBlocksMax*sizeof(u64));
  cudaMemset(&idxList, 0, nBlocksMax*sizeof(u32));
  cudaMemset(&nbrIdxList, 0, blockHaloSizeTot*nBlocksMax*sizeof(u32));
  cudaMemset(&fieldData, 0, nFields*blockSizeTot*nBlocksMax*sizeof(dataType));

  // fill the locList with base grid blocks
  nBlocks = 0;
  for (u32 j = 0; j < baseGridSize[1]; j++) {
    for (u32 i = 0; i < baseGridSize[0]; i++) {
      locList[nBlocks] = mortonEncode(0, i, j);
      nBlocks++;
    }
  }

  // sort the locList
  thrust::sort(thrust::host, locList, locList+nBlocks);

  cudaDeviceSynchronize();
}

MultiLevelSparseGrid::~MultiLevelSparseGrid(void) {
  cudaDeviceSynchronize();
  cudaFree(locList);
  cudaFree(idxList);
  cudaFree(nbrIdxList);
  cudaFree(fieldData);
}


void MultiLevelSparseGrid::sortBlocks(void) {
  thrust::sort_by_key(thrust::device, locList, locList+nBlocks, idxList);
  sortFieldArray();
  //updateIndices1<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}


__device__ void MultiLevelSparseGrid::activateBlock(u32 lvl, u32 i, u32 j) {
  u64 loc = mortonEncode(lvl, i, j);
  u32 idx = hashTable->insert(loc);
  locList[idx] = loc;
  idxList[idx] = idx;
}

//__device__ void MultiLevelSparseGrid::deactivateBlock(u32 lvl, u32 i, u32 j) {
//  u64 loc = mortonEncode(lvl, i, j);
//  u32 idx = hashDelete(loc);
//  locList[idx] = kEmpty;
//  idxList[idx] = bEmpty;
//}


// seperate bits from a given integer 3 positions apart
__host__ __device__ u64 MultiLevelSparseGrid::split(u32 a) {
  u64 x = (u64)a & ((1<<20)-1); // we only look at the first 20 bits
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

// encode ijk indices and resolution level into morton code
__host__ __device__ u64 MultiLevelSparseGrid::mortonEncode(u64 lvl, u32 i, u32 j) {
  u64 morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1;
  return morton;
}

// compact separated bits into into an integer
__host__ __device__ u32 MultiLevelSparseGrid::compact(u64 w) {
  w &=                  0x1249249249249249;
  w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
  w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
  w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
  w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
  w = (w ^ (w >> 32)) & 0x00000000001fffff;
  return (u32)w;
}

// decode morton code into ij idx and resolution level
__host__ __device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j) {
  lvl = u32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
}

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1, 1>>>(*this);
}
*/
