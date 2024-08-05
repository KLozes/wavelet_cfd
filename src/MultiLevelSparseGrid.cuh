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

  real domainSize[3] = {1.0, 1.0};
  i32 baseGridSize[3] = {1,1};
  i32 nLvls;
  i32 nFields;
  i32 imageSize[3] = {1,1};

  i32 imageCounter;
  i32 nBlocks;

  u64 *bLocList; // block morton codes
  u32 *bIdxList; // block memory indices

  u32 *nbrIdxList;     // cell neighbor indeces
  u32 *prntIdxList;    // block parent indices
  u32 *chldIdxList;    // block child indices
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
  __device__ real getDz(i32 lvl);
  __device__ Vec3 getCellPos(i32 lvl, i32 ib, i32 jb, i32 kb, i32 i, i32 j, i32 k);
  __device__ u32 getNbrIdx(u32 bIdx, i32 i, i32 j, i32 k);
  __host__ __device__ bool isInteriorBlock(i32 lvl, i32 i, i32 j, i32 k);
  __host__ __device__ bool isExteriorBlock(i32 lvl, i32 i, i32 j, i32 k);

  __host__ __device__ real *getField(u32 f);

  __device__ void activateBlock(i32 lvl, i32 i, i32 j, i32 k);
  
  __host__ __device__ u64 encode(i32 lvl, i32 i, i32 j, i32 k);
  __host__ __device__ void decode(u64 morton, i32 &lvl, i32 &i, i32 &j, i32 &k);

  void paint(void);
  virtual void computeImageData(i32 f); 

};

#define START_CELL_LOOP \
  u32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  u32 bIdx = cIdx / blockSizeTot; \
  while (bIdx < grid.hashTable.nKeys) {
#define END_CELL_LOOP cIdx += gridDim.x*blockDim.x; \
    bIdx = cIdx / blockSizeTot; }

#define GET_CELL_INDICES \
  u32 idx = cIdx % blockSizeTot; \
  i32 i = idx % blockSize; \
  i32 j = (idx / blockSize) % blockSize; \
  i32 k = idx / blockSize / blockSize;

#define START_BLOCK_LOOP \
  u32 bIdx = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIdx < grid.hashTable.nKeys) {
#define END_BLOCK_LOOP bIdx += gridDim.x*blockDim.x;}

#endif