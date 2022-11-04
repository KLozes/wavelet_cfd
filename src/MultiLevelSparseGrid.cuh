#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "Settings.cuh"
#include "Util.cuh"

/*
** A multilevel sparse grid data structure
*/

enum Flags : uint {
  NONE = 0,
  REFINE = powi(2, 0),
  ACTIVE = powi(2, 1),
  GHOST = powi(2, 2)
};

// struct containing block and cell connectivity data
struct Block
{
  uint64_t loc;
  uint flags;
  uint index;
  uint parent;
  Array<uint, 2> children;
  Array<uint, blockSize+2*haloSize> neighbors;
  __host__ __device__ void operator=(Block block) {
    loc = block.loc;
    flags = block.flags;
    index = block.index;
    parent = block.parent;
    children = block.children;

  }

  __host__ __device__ bool hasChildren() {
    bool check = false;
    for (uint i=0; i<4; i++) {
      check += ((children(i) == bEmpty) ? 0 : 1);
    }
    return check;
  }

};

class MultiLevelSparseGrid : public Managed
{
public:

  uint baseGridSize[nDim];
  uint nLvls;
  uint nLvlsMax;
  uint nBlocks;
  uint nFields;

  uint64_t hashKeyList; // hash keys are block morton codes
  uint hashValueList; // hash values are block memory indices

  uint blockCounter;
  uint64_t *blockLocList;
  uint *blockIndexList
  uint *baseBlockIndexArray;
  Block *blockList;
  Block *blockListOld;

  typedef Array<dataType, blockSize> fieldData;
  fieldData *fieldDataList;

  typedef Array<uint, blockSize+2*haloSize> nbrData;
  nbrData *nbrDataList;

public:
  MultiLevelSparseGrid(uint baseSize_[nDim], uint nLvlsMax_, uint nFields_)
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

    cudaMallocManaged(&blockLocList, nBlocksMax*sizeof(uint64_t));
    cudaMallocManaged(&blockList, nBlocksMax*sizeof(Block));
    cudaMallocManaged(&blockListOld, nBlocksMax*sizeof(Block));
    cudaMallocManaged(&baseBlockIndexArray, nBlocksMax*sizeof(uint));
    cudaMallocManaged(&fieldDataList, nBlocksMax*nFields*sizeof(fieldData));

    cudaMemset(&blockLocList, 0, nBlocksMax*sizeof(uint64_t));
    cudaMemset(&blockList, 0, nBlocksMax*sizeof(Block));
    cudaMemset(&blockListOld, 0, nBlocksMax*sizeof(Block));
    cudaMemset(&baseBlockIndexArray, 0, nBlocksMax*sizeof(uint));
    cudaMemset(&fieldDataList, 0, nBlocksMax*nFields*sizeof(fieldData));

    cudaDeviceSynchronize();
  }

  __host__ ~MultiLevelSparseGrid(void)
  {
    cudaDeviceSynchronize();
    cudaFree(blockLocList);
    cudaFree(blockList);
    cudaFree(fieldDataList);
  }

  void initGrid(void);

  void sortBlocks(void);

  __host__ __device__ uint& getBaseBlockIndex(int i, int j);

  __device__ uint getBlockIndex(uint lvl, uint i, uint j);

  __device__ void activateBlock(uint lvl, uint i, uint j);

  __device__ void deactivateBlock(uint bIndex);

  __device__ void deactivateBlock(uint lvl, uint i, uint j);

  __device__ void getDijk(uint n, uint &di, uint &dj);

  __device__ dataType& getFieldValue(uint fIndex, uint bIndex, uint i, uint j);

  __device__ dataType& getParentFieldValue(uint fIndex, uint bIndex, uint ib, uint jb, uint i, uint j);


  __device__ uint64_t hash(uint64_t x);
  __device__ void hashInsert(uint64_t key, uint32_t value);
  __device__ void hashDelete(uint64_t key, uint value);
  __device__ uint hasGetValue(uint64_t key);

  __host__ __device__ uint64_t split(uint a)
  __host__ __device__ uint64_t mortonEncode(uint64_t lvl, uint i, uint j);
  __host__ __device__ uint64_t mortonEncode(uint64_t lvl, uint i, uint j, uint k);

  __host__ __device__ uint compact(uint64_t w);
  __host__ __device__ void mortonDecode(uint64_t morton, uint &lvl, uint &i, uint &j);
  __host__ __device__ void mortonDecode(uint64_t morton, uint &lvl, uint &i, uint &j, uint &k);

  virtual void sortFields(void){};

  void resetBlockCounter(void);

};

/*
#define START_CELL_LOOP \
  uint bIndex = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  uint lvl, ib, jb; \
  grid.mortonDecode(grid.blockList[bIndex].loc, lvl, ib, jb); \
  uint index = threadIdx.x % bSize; \
  uint i = index % blockSize; \
  uint j = index / blockSize; \
  while (bIndex < grid.nBlocks) {
*/

#define START_CELL_LOOP \
  uint bIndex = blockIdx.x * nBlocksPerCudaBlock + threadIdx.x / blockSizeTot; \
  uint index = threadIdx.x % blockSizeTot; \
  while (bIndex < grid.nBlocks) {

#define END_CELL_LOOP bIndex += gridDim.x; __syncthreads();}

#define START_BLOCK_LOOP \
  uint bIndex = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIndex < grid.nBlocks) {

#define END_BLOCK_LOOP bIndex += gridDim.x;}

#define START_DYNAMIC_BLOCK_LOOP \
  __shared__ uint startIndex; \
  __shared__ uint endIndex; \
  while (grid.blockCounter < grid.nBlocks) { \
    if (threadIdx.x == 0) { \
      startIndex = atomicAdd(&(grid.blockCounter), blockDim.x); \
      endIndex = atomicMin(&(grid.blockCounter), grid.nBlocks); \
    } \
    __syncthreads(); \
    uint bIndex = startIndex + threadIdx.x; \
    if ( bIndex < endIndex ) {
#define END_DYNAMIC_BLOCK_LOOP __syncthreads(); }}

#endif
