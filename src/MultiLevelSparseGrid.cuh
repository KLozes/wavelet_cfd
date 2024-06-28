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
  NEW = 1,
  KEEP = 2,
  REFINE = 3,
};

enum CELL_FLAGS {
  GHOST  = 0,
  PARENT = 1,
  ACTIVE = 2,
};

class MultiLevelSparseGrid : public Managed {
public:

  HashTable hashTable;

  real domainSize[2] = {1.0, 1.0};
  i32 baseGridSize[2] = {1,1};
  i32 nLvls;
  i32 nFields;
  i32 imageSize[2] = {1,1};

  u32 blockCounter;
  u32 imageCounter;
  u32 nBlocks;
  u32 nBlocksPrev;

  u64 *bLocList; // block morton codes
  u32 *bIdxList; // block memory indices

  u32 *nbrIdxList;     // cell neighbor indeces
  u32 *prntIdxList;    // block parent indices
  u32 *chldIdxList;    // block child indices
  u32 *prntIdxListOld; // old block parent indices
  u32 *chldIdxListOld; // old block child indices

  u32 *bFlagsList;     // block Flags
  u32 *cFlagsList;     // cell Flags

  real *fieldData; // flow field data
  real *imageData; // output image data
  real *pixelCountData; // number of 


  int lock;

  MultiLevelSparseGrid(real *domainSize, u32 *baseGridSize_, u32 nLvls_, u32 nFields_);

  ~MultiLevelSparseGrid(void);

  void initializeBaseGrid(void);
  
  void adaptGrid(void);
  void sortBlocks(void);
  virtual void sortFieldData(void) = 0;

  __device__ i32 getSize(i32 lvl);
  __device__ real getDx(i32 lvl);
  __device__ real getDy(i32 lvl);
  __device__ void getCellPos(i32 lvl, i32 ib, i32 jb, i32 i, i32 j, real *pos);
  __device__ u32 getNbrIdx(u32 bIdx, i32 i, i32 j);
  __host__ __device__ bool isInteriorBlock(i32 lvl, i32 i, i32 j);
  __host__ __device__ bool isExteriorBlock(i32 lvl, i32 i, i32 j);

  __host__ __device__ real *getField(u32 f);

  __device__ void activateBlock(i32 lvl, i32 i, i32 j);
__device__ u32 getBlockIdx(i32 lvl, i32 i, i32 j);

  
  //__device__ u64 split(u32 a);
  __host__ __device__ u64 encode(i32 lvl, i32 i, i32 j);

  //__device__ u32 compact(u64 w);
  __host__ __device__ void decode(u64 morton, i32 &lvl, i32 &i, i32 &j);

  void paint(void);
  virtual void computeImageData(i32 f); 

};

#define START_CELL_LOOP \
  u32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  u32 bIdx = cIdx / blockSizeTot; \
  while (bIdx < grid.nBlocks) { \
    u32 idx = cIdx % blockSizeTot; \
    i32 i = idx % blockSize; \
    i32 j = idx / blockSize;
#define END_CELL_LOOP cIdx += gridDim.x*blockDim.x; \
    bIdx = cIdx / blockSizeTot; }

#define START_BLOCK_LOOP \
  u32 bIdx = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIdx < grid.nBlocks) {
#define END_BLOCK_LOOP bIdx += gridDim.x*blockDim.x;}

#endif