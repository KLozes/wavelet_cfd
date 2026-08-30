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

#ifdef USE_MGPU
// partition mode: 0 = regular box split (1-D x-strips), 1 = Z-curve (Morton)
// weight-balanced cut (default).  Set from the CLI before solver construction.
extern i32 g_partMode;
#endif

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
  // leafMode (the DG solver): the grid is a leaf-only PARTITION of the domain --
  // no exterior boundary blocks, no parent blocks, no ghost cells.  BCs are
  // imposed weakly inside the face kernels, so initializeBaseGrid skips the
  // exterior ring and flagActiveCells marks every interior cell ACTIVE (the
  // GHOST/PARENT machinery is meaningless without overlapping levels).
  i32 leafMode = 0;
  // memory-layout sort order: 0 = location code (level-major, row-major k,j,i --
  // x-neighbors adjacent, y/z a row/plane apart); 1 = level-major space-filling
  // curve (Hilbert in pseudo2D, Morton in 3D -- neighbors in ALL directions and
  // sibling octets land close/contiguous in memory).  Pure locality choice:
  // hash/nbr/prnt bindings are rebuilt order-agnostically after every sort.
  i32 sortCurve = 0;
  u64 *sortKeyList = nullptr;   // [nBlocksMax] scratch keys (lazily allocated)
  // periodic: treat the domain as a torus during refinement -- grading,
  // reconstruction and boundary-ghost activation wrap across the seam so the two
  // opposite edges stay refined to matching, graded levels, and every periodic
  // ghost block has a same-level interior image to be filled from.
  i32 periodic = 0;
  i32 imageSizeX[2] = {1,1};
  i32 imageSizeY[2] = {1,1};
  i32 imageSizeZ[2] = {1,1};

  i32 imageCounter;
#ifdef USE_MGPU
  i32 partImgCounter = 0;   // partition-plot frame counter
#endif
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
  // NEED lists: support targets the grading/reconstruction closure required but
  // could not create locally (owned-target rule).  Recorded at the skip point,
  // sent to the owning rank each exchangeStructure; the owner creates them as
  // owned blocks ("adopt") and they return as our ghosts in the same exchange.
  i32  needSlot = 0;         // per-neighbor capacity (sized with dirSlot)
  i32 *needCnt  = nullptr;   // [nNbr]  needs recorded for each neighbor
  u64 *needLoc  = nullptr;   // [nNbr*needSlot]
  void initPartition(void);                                            // set `part` + fill ownerBase
  void partitionByWeight(const double *w, i32 *dst = nullptr);   // Morton-cut owner fill (w nullptr = uniform; dst nullptr = ownerBase)
  void derivePartition(void);                // bbox + neighbor set + capacity check from ownerBase
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
  i32 *snapValidList;   // 1 if this block held a valid F_OLD snapshot at copyToOld, else 0 (new/imported)
  i32 *ibClassList;     // immersed-body class per block (IB_FLUID=0 / IB_DEAD=2, solid);
                        // pure geometry -- recomputed after every sort, never permuted
  i32  dbgChecks = 0;   // runtime debug: topology/data-integrity assert kernels (--debug)
  i32 *dbgCnt = nullptr;      // [1] managed violation counter for the check kernels
  i32 *createdCnt = nullptr;  // [1] managed count of blocks CREATED (not touched) since last reset

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
  void resetGrid(void);   // wipe blocks + fields for a from-scratch rebuild
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
  // pseudo2D: refill the k>0 layers from k=0.  fOnly < 0 does every field;
  // pass a single field index before reducing that field (a guarded kernel
  // leaves k>0 stale, and a stale zero wins a min).
  void broadcastZ(i32 fOnly = -1);

  __device__ void activateBlock(i32 lvl, i32 i, i32 j, i32 k);
  __device__ i32  getBlockIdx(u64 loc);   // validated loc->slot (corpse-safe getValue; bEmpty if stale)
  __device__ void wrapBlockPeriodic(i32 lvl, i32 &i, i32 &j, i32 &k); // wrap block index into the interior range (torus)
  
  __host__ __device__ u64 encode(i32 lvl, i32 i, i32 j, i32 k);
  __host__ __device__ void decode(u64 loc, i32 &lvl, i32 &i, i32 &j, i32 &k);

  void paint(void);
  void paintField(i32 f, const char *fileName);  // render one field (or grid, f=-1) to a png
#ifdef USE_MGPU
  void paintPartition(void);   // render the rank-ownership map (rank 0 writes output/partition_*.png)
  void paintRankGrid(void);    // debug: per-rank owned+ghost block view (each rank writes rankgrid_r*.png)
#endif
  virtual void computeImageData(i32 f);

};

// pseudo2D collapses z: every cell loop runs ONLY the k == 0 plane, so the block
// does 1/blockSize of the work.  This is sound because nothing inside a step
// reads a k != 0 cell -- z-gradients are zeroed, there is no z-flux, z-momentum
// is never evolved, there are no z-halo blocks, and restriction/prolongation map
// parent k to child k identically (so each z-plane is independent).  The other
// layers DO go stale, and host-side readers see them (paint samples
// k = blockSize/2, writeSolution and the error norms download whole blocks), so
// broadcastZ() refreshes them at the end of each step().
// k == 0 is exactly (cIdx % blockSizeTot) < blockSize*blockSize.
#define START_CELL_LOOP \
  i32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  i32 bIdx = cIdx / blockSizeTot; \
  while (bIdx < grid.hashTable.nKeys) { \
    if (!grid.pseudo2D || (cIdx % blockSizeTot) < blockSize*blockSize) {
#define END_CELL_LOOP } cIdx += gridDim.x*blockDim.x; \
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