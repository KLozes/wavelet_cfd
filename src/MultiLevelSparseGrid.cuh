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
  // periodic: treat the domain as a torus during refinement -- grading,
  // reconstruction and boundary-ghost activation wrap across the seam so the two
  // opposite edges stay refined to matching, graded levels, and every periodic
  // ghost block has a same-level interior image to be filled from.
  i32 periodic = 0;
  i32 imageSizeX[2] = {1,1};
  i32 imageSizeY[2] = {1,1};
  i32 imageSizeZ[2] = {1,1};

  i32 imageCounter;
  i32 nBlocks;

#ifdef USE_MGPU
  // 3D coarse-grid domain decomposition: the base (level-0) grid is split into a
  // p[0]*p[1]*p[2] box layout; a block is owned by the PE that owns its level-0
  // ancestor.  A partition boundary is filled like a periodic/wall ghost, but
  // from a neighbor PE via a comm::neighborExchange message instead of a local copy.
  struct Partition {
    i32 p[3];          // process-grid dims (px,py,pz)
    i32 c[3];          // this PE's coords within the process grid
    i32 rank;          // this PE's id
    i32 b0[3], b1[3];  // owned base-block box [b0,b1) per axis (level-0 units)
    i32 nb[3];         // coarse (level-0) grid dims: baseGridSize/blockSize per axis
  } part;
  // The ONLY full-(coarse-)domain array a PE keeps: one owning rank per level-0
  // base block, replicated on every PE (tiny: nb0*nb1*nb2 int32s ~ a few KB).  It
  // is a PE's whole map of "which rank holds any block / its halo source"; the
  // per-block data itself is only this PE's subdomain + a ghost halo.  ownerPE
  // indexes it, so an arbitrary / load-balanced partition is just a rewrite of
  // this array (the box split is one particular fill).
  i32 *ownerBase;
  // neighbor PEs (the <=26 adjacent process-grid cells) for the directory/halo
  // exchanges -- a fixed set (no rebalancing), computed once in initPartition.
  i32  nNbr;        // number of neighbor PEs
  i32 *nbrRank;     // [nNbr]  neighbor PE ranks
  i32 *nbrOf;       // [size]  rank -> neighbor slot (or -1)
  void initPartition(void);                                            // set `part` + fill ownerBase
  __host__ __device__ i32  ownerPE(i32 lvl, i32 ib, i32 jb, i32 kb);   // rank owning a block (ownerBase lookup)
  __host__ __device__ bool isOwnedBlock(i32 lvl, i32 ib, i32 jb, i32 kb);
#endif

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
  __device__ void wrapBlockPeriodic(i32 lvl, i32 &i, i32 &j, i32 &k); // wrap block index into the interior range (torus)
  
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