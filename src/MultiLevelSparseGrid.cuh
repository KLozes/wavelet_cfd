#ifndef MULTILEVEL_SPARSE_GRID_H
#define MULTILEVEL_SPARSE_GRID_H

#include "Settings.cuh"
#include "Util.cuh"
#include "HashTable.cuh"

/*
** A multilevel sparse grid data structure
*/
enum BLOCK_FLAGS {
  DELETE = 0,
  KEEP = 1 << 0,
  NEW = 1 << 1,
  REFINE = 1 << 2,
};

enum CELL_FLAGS {
  GHOST  = 0,
  ACTIVE = 1 << 0,
};

class MultiLevelSparseGrid : public Managed {
public:

  HashTable hashTable;

  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {1,1};
  u32 nLvls;
  u32 nFields;
  u32 imageSize[2] = {1,1};

  u32 blockCounter;
  u32 imageCounter;
  u32 nBlocks;

  u64 *bLocList; // block morton codes
  u32 *bIdxList; // block memory indices

  u32 *nbrIdxList;     // cell neighbor indeces
  u32 *prntIdxList;    // block parent indices
  u32 *bFlagsList;     // block Flags
  u32 *cFlagsList;     // cell Flags

  dataType *fieldData; // flow field data
  dataType *imageData; // output image data

  MultiLevelSparseGrid(dataType *domainSize, u32 *baseGridSize_, u32 nLvls_, u32 nFields_);

  ~MultiLevelSparseGrid(void);

  void initializeBaseGrid(void);
  
  void adaptGrid(void);
  void sortBlocks(void);
  virtual void sortFieldData(void) = 0;

  __host__ __device__ i32 getSize(i32 lvl);
  __host__ __device__ dataType getDx(i32 lvl);
  __host__ __device__ dataType getDy(i32 lvl);
  __host__ __device__ void getCellPos(i32 lvl, i32 ib, i32 jb, i32 i, i32 j, dataType *pos);
  __host__ __device__ u32 getNbrIdx(u32 bIdx, i32 i, i32 j);
  __host__ __device__ bool isInteriorBlock(i32 lvl, i32 i, i32 j);
  __host__ __device__ bool isExteriorBlock(i32 lvl, i32 i, i32 j);

  __host__ __device__ dataType *getField(u32 f);

  __host__ __device__ void activateBlock(i32 lvl, i32 i, i32 j);
  
  __host__ __device__ u64 split(u32 a);
  __host__ __device__ u64 mortonEncode(i32 lvl, i32 i, i32 j);

  __host__ __device__ u32 compact(u64 w);
  __host__ __device__ void mortonDecode(u64 morton, i32 &lvl, i32 &i, i32 &j);

  void paint(void);
  virtual void computeImageData(i32 f); 

};

#define START_CELL_LOOP \
  u32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  u32 bIdx = cIdx / blockSizeTot; \
  u32 idx = cIdx % blockSizeTot; \
  i32 i = idx % blockSize; \
  i32 j = idx / blockSize; \
  while (bIdx < grid.nBlocks) {
#define END_CELL_LOOP cIdx += gridDim.x*blockDim.x; \
  bIdx = cIdx / blockSizeTot;  \
  __syncthreads();}

#define START_HALO_CELL_LOOP \
  u32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  u32 bIdx = cIdx / blockHaloSizeTot; \
  u32 idx = cIdx % blockHaloSizeTot; \
  i32 i = idx % blockHaloSize; \
  i32 j = idx / blockHaloSize; \
  while (bIdx < grid.nBlocks) {
#define END_HALO_CELL_LOOP cIdx += gridDim.x*blockDim.x; \
  bIdx = cIdx / blockHaloSizeTot;  \
  __syncthreads();}

#define START_BLOCK_LOOP \
  u32 bIdx = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIdx < grid.nBlocks) {
#define END_BLOCK_LOOP bIdx += gridDim.x*blockDim.x; __syncthreads();}

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
