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

  real domainSize[3] = {1.0, 1.0, 1.0};
  i32 baseGridSize[3] = {1, 1, 1};
  i32 nLvls;
  i32 nFields;
  // lean mode (the narrowband SDF): skip the flow-solver-only per-block arrays
  // (cFlagsList, nbrIdxList, prntIdxList) and the slice buffer (imageDataX), and
  // skip the matching kernels in sortBlocks.  Cuts per-block memory ~3x.
  bool lean = false;
  // pseudo2D: collapse z to a single (un-refined, un-fluxed) block.  The
  // z-direction carries blockSize uniform cells that never refine, the
  // z-momentum is never updated, and no z fluxes/boundary blocks are created.
  i32 pseudo2D = 0;
  // periodic: wrap exterior ghost blocks to the opposite interior edge (the
  // self/center neighbor slot is remapped in sortBlocks so it survives re-sorts).
  i32 periodic = 0;
  i32 imageSizeX[2] = {1,1};
  i32 imageSizeY[2] = {1,1};
  i32 imageSizeZ[2] = {1,1};

  i32 imageCounter;
  i32 nBlocks;

  u64 *bLocList;        // block location codes
  i32 *bIdxList;        // block memory indices

  i32 *nbrIdxList;      // block neighbor indeces
  i32 *prntIdxList;     // block parent indices
  i32 *chldIdxList;     // block child indices
  i32 *bFlagsList;      // block Flags
  i32 *cFlagsList;      // cell Flags

  real *fieldData;      // flow field data
  real *imageDataX;     // output image data
  real *imageDataY;     // output image data
  real *imageDataZ;     // output image data
  i32  *imageSampleX;   // number of 
  i32  *imageSampleY;
  i32  *imageSampleZ;

  MultiLevelSparseGrid(real *domainSize, i32 *baseGridSize_, i32 nLvls_, i32 nFields_, bool lean_ = false);

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
  __device__ i32 getNbrIdx(i32 bIdx, i32 i, i32 j, i32 k);
  __host__ __device__ bool isInteriorBlock(i32 lvl, i32 i, i32 j, i32 k);
  __host__ __device__ bool isExteriorBlock(i32 lvl, i32 i, i32 j, i32 k);

  __host__ __device__ real *getField(i32 f);

  __device__ void activateBlock(i32 lvl, i32 i, i32 j, i32 k);
  
  __host__ __device__ u64 encode(i32 lvl, i32 i, i32 j, i32 k);
  __host__ __device__ void decode(u64 loc, i32 &lvl, i32 &i, i32 &j, i32 &k);

  void paint(void);
  void paintField(i32 f, const char *fileName);  // render one field (or grid, f=-1) to a png
  virtual void computeImageData(i32 f);

};

#define START_CELL_LOOP \
  i32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  i32 bIdx = cIdx / blockSizeTot; \
  while (bIdx < grid.hashTable.nKeys) {
#define END_CELL_LOOP cIdx += gridDim.x*blockDim.x; \
    bIdx = cIdx / blockSizeTot; }

#define GET_CELL_INDICES \
  i32 idx = cIdx % blockSizeTot; \
  i32 i = idx % blockSize; \
  i32 j = (idx / blockSize) % blockSize; \
  i32 k = idx / blockSize / blockSize;

#define START_BLOCK_LOOP \
  i32 bIdx = threadIdx.x + blockIdx.x * blockDim.x; \
  while (bIdx < grid.hashTable.nKeys) {
#define END_BLOCK_LOOP bIdx += gridDim.x*blockDim.x;}

#endif