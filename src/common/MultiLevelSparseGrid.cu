#include <thrust/sort.h>
#include <algorithm>
#include <vector>

#include <png++/png.hpp>
#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"
#ifdef USE_MGPU
#include "Comm.cuh"
#endif

MultiLevelSparseGrid::MultiLevelSparseGrid(real *domainSize_, i32 *baseGridSize_, i32 nLvls_, i32 nFields_, bool lean_) {

  domainSize[0] = domainSize_[0];
  domainSize[1] = domainSize_[1];
  domainSize[2] = domainSize_[2];

  baseGridSize[0] = baseGridSize_[0];
  baseGridSize[1] = baseGridSize_[1];
  baseGridSize[2] = baseGridSize_[2];

  nLvls = nLvls_;
  nFields = nFields_;
  lean = lean_;

#ifdef USE_MGPU
  // partition is set up first (before any sizing) so per-PE storage can be sized
  // to THIS PE's subdomain, never the full domain.
  initPartition();
#endif

  // imageDataX holds an x-y slice taken at the mid-z plane (the natural view
  // for the pseudo-2D / quasi-1D Sod problem).
  // Full-domain on every PE: each PE paints only its OWNED leaf cells (the AMR
  // leaves partition the domain, so every finest-pixel is written by exactly
  // one rank), then paintField sum-reduces into one full-domain figure that
  // rank 0 writes.  A Z-curve owned region is not a clean box, so per-PE bbox
  // tiles would overlap and never stitch -- the reduce sidesteps that.
  imageSizeX[0] = (baseGridSize[0])*powi(2,nLvls-1);  // x (width)
  imageSizeX[1] = (baseGridSize[1])*powi(2,nLvls-1);  // y (height)

  imageSizeY[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSizeY[1] = (baseGridSize[2])*powi(2,nLvls-1);

  imageSizeZ[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSizeZ[1] = (baseGridSize[1])*powi(2,nLvls-1);

  imageCounter = 0;

  // grid size checking (the dense base-grid capacity is checked in
  // initializeBaseGrid, where that grid is actually materialized -- a sparse
  // user like the narrowband SDF never fills it and is bounded only by the
  // number of blocks it activates).
  // Non-pow2 blockSize is fine for the leaf-only DG solver (blockSize == p+1,
  // e.g. 3 for p=2): all cell indexing is plain div/mod, no bit tricks.  The
  // FV halo/restriction machinery does assume even blockSize -- keep pow2 for
  // the FV solvers (which always build with the default BLOCK_SIZE=4).
#ifdef DG_ORDER
  assert(blockSize == DG_ORDER + 1);
#else
  assert(isPowerOf2(blockSize));
#endif

  // always needed: block location codes / memory indices / flags (the hash table
  // is a member with its own allocation)
  cudaMallocManaged(&bLocList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&bIdxList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&bFlagsList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&snapValidList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&ibClassList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&dbgCnt, 64*sizeof(i32));   // [1] topo violations, or [3*lvl+{0,1,2}] ghost census
  cudaMallocManaged(&createdCnt, sizeof(i32));
  cudaMemset(bLocList, 0, nBlocksMax*sizeof(u64));
  cudaMemset(bIdxList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(snapValidList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(ibClassList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(dbgCnt, 0, sizeof(i32));
  cudaMemset(createdCnt, 0, sizeof(i32));

  // flow-solver-only per-block arrays + the slice buffer: skipped in lean mode
  // (the narrowband SDF needs none of them -- see the `lean` note in the header).
  prntIdxList = nbrIdxList = cFlagsList = nullptr;
  imageDataX  = nullptr;
  if (!lean) {
    cudaMallocManaged(&prntIdxList, nBlocksMax*sizeof(i32));
    cudaMallocManaged(&nbrIdxList, 27*nBlocksMax*sizeof(i32));
    cudaMallocManaged(&cFlagsList, blockSizeTot*nBlocksMax*sizeof(i32));
    // NOTE: this is a UNIFORM-FINE buffer (baseGrid * 2^(nLvls-1))^2 -- 151 MB
    // at nLvls 7, 604 MB at nLvls 8.  It is the one allocation in the solver
    // that does NOT benefit from adaptivity, so painting must stay optional on
    // deep grids (see --paint in Main.cu).
    cudaMallocManaged(&imageDataX, imageSizeX[0]*imageSizeX[1]*sizeof(real));
    cudaMemset(prntIdxList, 0, nBlocksMax*sizeof(i32));
    cudaMemset(nbrIdxList, 0, 27*nBlocksMax*sizeof(i32));
    cudaMemset(cFlagsList, 0, blockSizeTot*nBlocksMax*sizeof(i32));
    cudaMemset(imageDataX, 0, imageSizeX[0]*imageSizeX[1]*sizeof(real));
  }

  // a solver may carry its own field storage (e.g. the SDF's int16 array) and
  // request zero base fields, in which case fieldData is not allocated.
  fieldData = nullptr;
  if (nFields > 0) {
    // per-PE local storage (each PE holds only its subdomain + halo); the halo
    // moves via comm::neighborExchange, so no symmetric heap is needed.
    size_t bytes = (size_t)nFields*(size_t)blockSizeTot*(size_t)nBlocksMax*sizeof(real);
    cudaMallocManaged(&fieldData, bytes);
    cudaMemset(fieldData, 0, bytes);
  }

  cudaDeviceSynchronize();
}

MultiLevelSparseGrid::~MultiLevelSparseGrid(void) {
  cudaDeviceSynchronize();
  cudaFree(bLocList);
  cudaFree(bIdxList);
  cudaFree(snapValidList);
  cudaFree(ibClassList);
  cudaFree(prntIdxList);
  cudaFree(nbrIdxList);
  cudaFree(cFlagsList);
  cudaFree(fieldData);
#ifdef USE_MGPU
  cudaFree(ownerBase);
#endif
  cudaFree(imageDataX);
}

#ifdef USE_MGPU
// Coarse-grid partition of the base grid across comm::size() PEs.
//   g_partMode 0: regular box split (1-D x-strips), the legacy default.
//   g_partMode 1: Z-curve (Morton) split -- base blocks are ordered along the
//     Morton curve and cut into P contiguous, weight-balanced intervals.
//     Curve intervals are spatially compact (less seam area than strips) and
//     contiguous, so a later re-cut only exchanges blocks with the curve
//     neighbors r-1 / r+1 (the basis for dynamic rebalancing).
i32 g_partMode = 1;

// Space-filling-curve index of a base block.  Pseudo-2D uses a HILBERT curve
// (better locality than Morton -- consecutive indices are always face-adjacent,
// no diagonal jumps -- so a contiguous interval is a compact region with little
// seam area).  True-3D falls back to a 3D Morton (Z) code.
//
// 2D Hilbert distance on an n x n grid (n a power of two), Wikipedia xy2d.
static u64 hilbert2D(i32 n, i32 x, i32 y) {
  u64 d = 0;
  for (i32 s = n/2; s > 0; s /= 2) {
    i32 rx = (x & s) > 0 ? 1 : 0;
    i32 ry = (y & s) > 0 ? 1 : 0;
    d += (u64)s * (u64)s * (u64)((3*rx) ^ ry);
    if (ry == 0) {                         // rotate/flip the quadrant
      if (rx == 1) { x = n-1 - x; y = n-1 - y; }
      i32 t = x; x = y; y = t;
    }
  }
  return d;
}
static u64 morton3D(i32 x, i32 y, i32 z) {
  u64 m = 0;
  for (i32 b = 0; b < 21; b++)
    m |= ((u64)((x>>b)&1))<<(3*b) | ((u64)((y>>b)&1))<<(3*b+1) | ((u64)((z>>b)&1))<<(3*b+2);
  return m;
}
static u64 curveIndex(i32 x, i32 y, i32 z, i32 hn, bool twoD) {
  return twoD ? hilbert2D(hn, x, y) : morton3D(x, y, z);
}

// Fill ownerBase by cutting the Morton-ordered base blocks into P contiguous
// intervals of (approximately) equal total weight.  w = one weight per base
// block (ownerBase indexing); pass nullptr for uniform weights.  Every rank
// computes the same cuts from the same weights, so the map stays replicated.
void MultiLevelSparseGrid::partitionByWeight(const double *w, i32 *dst) {
  if (!dst) dst = ownerBase;
  i32 P = comm::size();
  i32 n = part.nb[0]*part.nb[1]*part.nb[2];
  bool twoD = (part.nb[2] == 1);
  i32 hn = 1; while (hn < part.nb[0] || hn < part.nb[1]) hn <<= 1;   // Hilbert grid side (pow2)
  std::vector<std::pair<u64,i32>> order(n);
  for (i32 k = 0; k < part.nb[2]; k++)
  for (i32 j = 0; j < part.nb[1]; j++)
  for (i32 i = 0; i < part.nb[0]; i++) {
    i32 idx = i + part.nb[0]*(j + part.nb[1]*k);
    order[idx] = {curveIndex(i, j, k, hn, twoD), idx};
  }
  std::sort(order.begin(), order.end());
  double total = 0;
  for (i32 t = 0; t < n; t++) total += w ? w[order[t].second] : 1.0;
  // walk the curve, advancing the rank when its weight share is filled
  double acc = 0, target = total / P;
  i32 r = 0;
  for (i32 t = 0; t < n; t++) {
    // never leave a rank empty: force advance if the remaining blocks are
    // exactly enough for the remaining ranks
    if ((acc >= (r+1)*target && r < P-1) || (n - t) == (P - 1 - r)) r++;
    dst[order[t].second] = r;
    acc += w ? w[order[t].second] : 1.0;
  }
}

// Derive everything the machinery needs from the (replicated) ownership map:
// the owned bounding box (image tiles), the neighbor-PE set (any rank owning a
// base block within the 2-ring of one of ours, with periodic wrap partners
// always included -- they carry zero-count exchanges in non-periodic runs),
// and a base-grid capacity check.  Called after every ownerBase (re)fill.
void MultiLevelSparseGrid::derivePartition(void) {
  i32 P = comm::size();
  // owned bounding box
  for (i32 d = 0; d < 3; d++) { part.b0[d] = part.nb[d]; part.b1[d] = 0; }
  i32 nOwned = 0;
  for (i32 k = 0; k < part.nb[2]; k++)
  for (i32 j = 0; j < part.nb[1]; j++)
  for (i32 i = 0; i < part.nb[0]; i++) {
    if (ownerBase[i + part.nb[0]*(j + part.nb[1]*k)] != part.rank) continue;
    nOwned++;
    i32 v[3] = {i, j, k};
    for (i32 d = 0; d < 3; d++) {
      if (v[d]   < part.b0[d]) part.b0[d] = v[d];
      if (v[d]+1 > part.b1[d]) part.b1[d] = v[d]+1;
    }
  }
  // neighbor set: foreign owners within the 2-ring (wrapped) of owned blocks
  for (i32 r = 0; r < P; r++) nbrOf[r] = -1;
  nNbr = 0;
  i32 dkLim = (part.nb[2] == 1) ? 0 : 2;
  for (i32 k = 0; k < part.nb[2]; k++)
  for (i32 j = 0; j < part.nb[1]; j++)
  for (i32 i = 0; i < part.nb[0]; i++) {
    if (ownerBase[i + part.nb[0]*(j + part.nb[1]*k)] != part.rank) continue;
    for (i32 dk=-dkLim; dk<=dkLim; dk++)
    for (i32 dj=-2; dj<=2; dj++)
    for (i32 di=-2; di<=2; di++) {
      i32 ni = ((i+di) % part.nb[0] + part.nb[0]) % part.nb[0];
      i32 nj = ((j+dj) % part.nb[1] + part.nb[1]) % part.nb[1];
      i32 nk = ((k+dk) % part.nb[2] + part.nb[2]) % part.nb[2];
      i32 o = ownerBase[ni + part.nb[0]*(nj + part.nb[1]*nk)];
      if (o == part.rank || nbrOf[o] >= 0) continue;
      nbrRank[nNbr] = o; nbrOf[o] = nNbr; nNbr++;
    }
  }
  // base grid + ring must fit the block pool (dense-base capacity check)
  assert((size_t)nOwned * (part.nb[2] == 1 ? 25 : 125) < (size_t)nBlocksMax);
}

void MultiLevelSparseGrid::initPartition(void) {
  i32 P = comm::size();
  part.p[0] = P; part.p[1] = 1; part.p[2] = 1;   // process-grid kept for the box mode
  part.rank = comm::rank();
  part.c[0] =  part.rank %  part.p[0];
  part.c[1] = (part.rank /  part.p[0]) % part.p[1];
  part.c[2] =  part.rank / (part.p[0]  * part.p[1]);
  // z divides by blockSizeZ: a collapsed build has a 1-cell-thick block
  for (i32 d = 0; d < 3; d++) part.nb[d] = baseGridSize[d]/(d == 2 ? blockSizeZ : blockSize);

  cudaMallocManaged(&ownerBase, (size_t)part.nb[0]*part.nb[1]*part.nb[2]*sizeof(i32));
  cudaMallocManaged(&nbrRank, (size_t)P*sizeof(i32));      // any rank can be a curve neighbor
  cudaMallocManaged(&nbrOf,   (size_t)P*sizeof(i32));

  if (g_partMode == 1) {
    partitionByWeight(nullptr);            // uniform weights; rebalanced later
  }
  else {
    // legacy regular box split (1-D x-strips): even split per axis with the
    // remainder handed to the low process-columns.
    for (i32 k = 0; k < part.nb[2]; k++)
    for (i32 j = 0; j < part.nb[1]; j++)
    for (i32 i = 0; i < part.nb[0]; i++) {
      i32 idx3[3] = {i, j, k}, col[3];
      for (i32 d = 0; d < 3; d++) {
        i32 q = part.nb[d]/part.p[d], rem = part.nb[d]%part.p[d], lo = 0, c = 0;
        for (c = 0; c < part.p[d]; c++) { i32 w = q + (c<rem?1:0); if (idx3[d] < lo+w) break; lo += w; }
        col[d] = c;
      }
      ownerBase[i + part.nb[0]*(j + part.nb[1]*k)] = col[0] + part.p[0]*(col[1] + part.p[1]*col[2]);
    }
  }
  derivePartition();
  cudaDeviceSynchronize();
}

// rank owning a block: index the coarse ownership map by the block's level-0
// ancestor (ib>>lvl, ...).  Exterior/ghost indices clamp to the edge base block.
__host__ __device__ i32 MultiLevelSparseGrid::ownerPE(i32 lvl, i32 ib, i32 jb, i32 kb) {
  i32 i = ib >> lvl, j = jb >> lvl, k = kb >> lvl;
  if (i < 0) i = 0;  if (i >= part.nb[0]) i = part.nb[0]-1;
  if (j < 0) j = 0;  if (j >= part.nb[1]) j = part.nb[1]-1;
  if (k < 0) k = 0;  if (k >= part.nb[2]) k = part.nb[2]-1;
  return ownerBase[i + part.nb[0]*(j + part.nb[1]*k)];
}

__host__ __device__ bool MultiLevelSparseGrid::isOwnedBlock(i32 lvl, i32 ib, i32 jb, i32 kb) {
  return isInteriorBlock(lvl, ib, jb, kb) && ownerPE(lvl, ib, jb, kb) == part.rank;
}
#endif

void MultiLevelSparseGrid::initializeBaseGrid(void) {
  // the dense base grid must fit in the block pool (this path activates every
  // base block; sparse users that skip it are not subject to this bound)
#ifdef USE_MGPU
  // each PE materializes only its OWNED base box plus a 2-block ghost ring, so
  // the bound is the per-PE share, not the full domain -- nBlocksMax can be sized
  // per GPU and the global grid scales with the number of ranks.
  assert(((size_t)(part.b1[0]-part.b0[0]+4)*(part.b1[1]-part.b0[1]+4)
          *(pseudo2D ? 1 : (part.b1[2]-part.b0[2]+4))) < (size_t)nBlocksMax);
#else
  assert(baseGridSize[0]*baseGridSize[1]*baseGridSize[2]/blockSizeTot < nBlocksMax);
#endif
  // fill the bLocList with base grid blocks
  dim3 cudaBlockSize3(8,8,8);
  dim3 nCudaBlocks3(baseGridSize[0]/blockSize/8+1, 
                    baseGridSize[1]/blockSize/8+1, 
                    baseGridSize[2]/blockSizeZ/8+1);
  initGridKernel<<<nCudaBlocks3, cudaBlockSize3>>>(*this);
  if (!leafMode)   // leaf-only grids have no exterior ghost blocks (weak BCs)
    addBoundaryBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;

  // sort the data by location code
  sortBlocks();
  cudaDeviceSynchronize();
}

// wipe the entire block structure and field data so the grid can be rebuilt
// from scratch (used by the Z-curve weighted re-cut at initialization).
void MultiLevelSparseGrid::resetGrid(void) {
  cudaDeviceSynchronize();
  hashTable.reset();
  thrust::fill(thrust::device, bLocList, bLocList + nBlocksMax, kEmpty);
  thrust::fill(thrust::device, bIdxList, bIdxList + nBlocksMax, bEmpty);
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  if (!lean) {
    cudaMemset(cFlagsList, 0, (size_t)nBlocksMax*blockSizeTot*sizeof(i32));
    thrust::fill(thrust::device, prntIdxList, prntIdxList + nBlocksMax, bEmpty);
  }
  cudaMemset(fieldData, 0, (size_t)nFields*(size_t)nBlocksMax*(size_t)blockSizeTot*sizeof(real));
  nBlocks = 0;
  cudaDeviceSynchronize();
}

// debug: single-GPU stage census matching CompressibleSolver::censusPrint (no comm)
static void censusPrint1(MultiLevelSparseGrid &g, const char *tag) {
  if (!g.dbgChecks) return;
  cudaDeviceSynchronize();
  i32 nOwn = 0;
  for (i32 b = 0; b < g.hashTable.nKeys; b++) {
    u64 loc = g.bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; g.decode(loc, lvl, ib, jb, kb);
    if (!g.isInteriorBlock(lvl, ib, jb, kb)) continue;
    if (g.bFlagsList[b] == DELETE) continue;
    nOwn++;
  }
  printf("[stage] %-14s ownedInterior(nonDelete)=%d\n", tag, nOwn);
}

void MultiLevelSparseGrid::adaptGrid(void) {

  if (nLvls > 1) {
    censusPrint1(*this, "pre-cascade");
    addFineBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    censusPrint1(*this, "post-fine");
    addAdjacentBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    censusPrint1(*this, "post-adjacent");
    for(i32 lvl=nLvls-1; lvl>1; lvl--) {   // lvl>1: level 1 is adaptive, build its ring for level-2 blocks
      setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      addReconstructionBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    }
    addBoundaryBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    censusPrint1(*this, "post-settle");
    cudaDeviceSynchronize();
    nBlocks = hashTable.nKeys;
    deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    updatePrntIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    censusPrint1(*this, "post-prune");
  }
}

void MultiLevelSparseGrid::allocLeafTables(void) {
  if (chldIdxList) return;
  cudaMallocManaged(&chldIdxList, (size_t)8*nBlocksMax*sizeof(i32));
  cudaMallocManaged(&cellMortar, (size_t)4*blockSizeTot*nBlocksMax*sizeof(i32));
  cudaMallocManaged(&rimList,    (size_t)blockSizeTot*nBlocksMax*sizeof(i32));
  cudaMemset(rimList, 0, (size_t)blockSizeTot*nBlocksMax*sizeof(i32));
  cudaMallocManaged(&mortarCnt,  sizeof(i32));
  mortarCap = 2*blockSize*nBlocksMax;
  cudaMallocManaged(&mortarList, (size_t)mortarCap*sizeof(Mortar));
  cudaMemset(chldIdxList, 0, (size_t)8*nBlocksMax*sizeof(i32));
  cudaMemset(cellMortar, 0xff, (size_t)4*blockSizeTot*nBlocksMax*sizeof(i32));   // -1 everywhere
  cudaMemset(mortarCnt,  0, sizeof(i32));
  cudaDeviceSynchronize();
}

void MultiLevelSparseGrid::sortBlocks(void) {

  cudaDeviceSynchronize();
  if (sortCurve || leafFlux) {
    // sort along a level-major space-filling curve (Hilbert/Morton) instead of
    // the row-major location code: face neighbors in all directions and sibling
    // octets land close in memory (locality for gather-heavy solvers).  The loc
    // codes ride along as values; all bindings are rebuilt below either way.
    if (!sortKeyList)
      cudaMalloc(&sortKeyList, (size_t)nBlocksMax*sizeof(u64));
    buildSortKeysKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    thrust::sort_by_key(thrust::device, sortKeyList, sortKeyList+hashTable.nKeys,
                        thrust::make_zip_iterator(thrust::make_tuple(bLocList, bIdxList)));
  } else {
    thrust::sort_by_key(thrust::device, bLocList, bLocList+hashTable.nKeys, bIdxList);
  }
  sortFieldData();
  cudaDeviceSynchronize();
  hashTable.reset();
  hashTable.nKeys = nBlocks;
  updateIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // hash-table rebuild (always)
  if (!lean) {   // parent/neighbor indices + cell flags are flow-solver-only
    updatePrntIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    updateNbrIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (leafFlux) allocLeafTables();
    flagActiveCellsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    flagParentCellsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (leafFlux) {
      // leaf mode: children, the sort groups, and the mortar faces at level jumps
      dbgCnt[8] = 0; dbgCnt[9] = 0;
      cudaDeviceSynchronize();
      cudaMemset(mortarCnt, 0, sizeof(i32));          // reused as the group counter first
      countSortGroupsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, dbgCnt + 8);   // [8] = leaf-bearing, [9] = exterior
      cudaDeviceSynchronize();
      nLeafBlocks = dbgCnt[8]; nExtBlocks = dbgCnt[9];
      updateChldIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      resetLeafFacesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      cudaMemset(mortarCnt, 0, sizeof(i32));
      cudaDeviceSynchronize();
      buildMortarsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      cudaDeviceSynchronize();
      nMortars = *mortarCnt;
      if (nMortars > mortarCap) { printf("[leaf] mortar capacity exceeded (%d > %d)\n", nMortars, mortarCap); nMortars = mortarCap; }
    }
  }
  cudaDeviceSynchronize();
}

__device__ Vec3 MultiLevelSparseGrid::getCellPos(i32 lvl, i32 ib, i32 jb, i32 kb, i32 i, i32 j, i32 k) {
  return Vec3((ib*blockSize + i + .5)*getDx(lvl),  
              (jb*blockSize + j + .5)*getDy(lvl), 
              (kb*blockSizeZ + k + .5)*getDz(lvl));
}

__device__ i32 MultiLevelSparseGrid::getNbrIdx(i32 bIdx, i32 i, i32 j, i32 k) {
  // Collapsed 2-D: a block is one cell thick in z and no z-neighbour blocks
  // exist, so ANY z offset must resolve back to this plane.  Callers still form
  // z-stencil indices unconditionally (the RHS builds its k+-1/k+-2 taps before
  // the pseudo2D guard discards the z-flux), and k = -2 would otherwise index
  // nbrIdxList at kb = -1 -- out of bounds.  Returning the same plane is exactly
  // right: in pseudo2D every z-layer holds identical data.
  // blockSizeZ is constexpr, so this vanishes in the 3-D build -- but the SAME
  // invariant holds whenever pseudo2D is set, even when the block is physically
  // blockSize thick: only the k == 0 plane is ever evolved, the k > 0 layers go
  // stale between broadcastZ() calls, and no z-neighbour block exists.  Without
  // this the wavelet prediction stencil reaches kp +- 1 out of the live plane.
  if (blockSizeZ == 1 || pseudo2D) k = 0;
  i += blockSize;
  j += blockSize;
  k += blockSizeZ;                 // z extent is blockSizeZ, not blockSize
  i32 ib = i / blockSize;
  i32 jb = j / blockSize;
  i32 kb = k / blockSizeZ;
  // A within-block offset (ib=jb=kb=1) is always this block itself; return bIdx
  // directly to skip the neighbor-list read on the common in-block cell lookup.
  i32 nbrIdx = (ib == 1 && jb == 1 && kb == 1)
             ? bIdx
             : nbrIdxList[27*bIdx + ib + 3*jb + 9*kb];
  return blockSizeTot*nbrIdx + (i%blockSize) + (j%blockSize)*blockSize + (k%blockSizeZ)*blockSize*blockSize;
}

__device__ real MultiLevelSparseGrid::getDx(i32 lvl) {
  return real(domainSize[0])/real(baseGridSize[0]*powi(2,lvl));
}

__device__ real MultiLevelSparseGrid::getDy(i32 lvl) {
  return real(domainSize[1])/real(baseGridSize[1]*powi(2,lvl));
}

__device__ real MultiLevelSparseGrid::getDz(i32 lvl) {
  // in pseudo2D the single z-block never refines, so the z-cell size is fixed
  if (pseudo2D) return real(domainSize[2])/real(baseGridSize[2]);
  return real(domainSize[2])/real(baseGridSize[2]*powi(2,lvl));
}

__device__ bool MultiLevelSparseGrid::isInteriorBlock(i32 lvl, i32 i, i32 j, i32 k) {
  i32 gridSize[3] = {i32(baseGridSize[0]/blockSize*powi(2,lvl)),
                     i32(baseGridSize[1]/blockSize*powi(2,lvl)),
                     pseudo2D ? i32(baseGridSize[2]/blockSizeZ)
                              : i32(baseGridSize[2]/blockSizeZ*powi(2,lvl))};
  return i >= 0 && j >= 0 && k >= 0 && i < gridSize[0] && j < gridSize[1] && k < gridSize[2];
}

__device__ bool MultiLevelSparseGrid::isExteriorBlock(i32 lvl, i32 i, i32 j, i32 k) {
  return !isInteriorBlock(lvl, i, j, k);
}

void MultiLevelSparseGrid::broadcastZ(i32 fOnly) {
  if (!pseudo2D) return;
  broadcastZKernel<<<cudaGridSize, cudaBlockSize>>>(*this, fOnly);
}

__host__ __device__ real* MultiLevelSparseGrid::getField(i32 f) {
  // 64-bit offset: f*nBlocksMax*blockSizeTot overflows i32 for
  // nFields*nCellsMax > 2^31 (e.g. 35 fields at the 64M-cell cap)
  return &fieldData[(u64)f*(u64)nBlocksMax*(u64)blockSizeTot];
}

// wrap a block index into the interior range at level lvl (periodic torus).
// pseudo2D keeps the single z-block; identity for already-interior indices.
__device__ void MultiLevelSparseGrid::wrapBlockPeriodic(i32 lvl, i32 &i, i32 &j, i32 &k) {
  i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
  i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
  i = ((i % gx) + gx) % gx;
  j = ((j % gy) + gy) % gy;
  if (!pseudo2D) {
    i32 gz = baseGridSize[2]/blockSizeZ*powi(2,lvl);
    k = ((k % gz) + gz) % gz;
  }
}

// Validated loc->slot lookup: the hash table never removes keys, so getValue can
// resolve a DELETED block's corpse slot (bLoc = kEmpty, fields zeroed).  Callers
// that would read or bind through the returned index must treat a corpse as
// missing (bEmpty) -- otherwise stencil taps silently read wrong/zeroed memory.
__device__ i32 MultiLevelSparseGrid::getBlockIdx(u64 loc) {
  i32 v = hashTable.getValue(loc);
  return (v != bEmpty && bLocList[v] == loc) ? v : bEmpty;
}

__device__ void MultiLevelSparseGrid::activateBlock(i32 lvl, i32 i, i32 j, i32 k) {
  u64 loc = encode(lvl, i, j, k);
  i32 idx = hashTable.insert(loc);
  if (idx == bEmpty) return;
  if (bLocList[idx] != loc || bIdxList[idx] != idx) {
    // CREATE: fresh slot (never used, zeroed fields) or corpse revival (deleted,
    // fields zeroed by deleteData).  All writes are idempotent, so concurrent
    // activators of the same loc are safe.  snapValid=0 marks "no F_OLD snapshot
    // yet" -- reconstituted by prolongation before the inverse reads it.  (The
    // bIdx test also catches the virgin-slot corner case loc==0: the exterior
    // block (-1,-1,-1) at lvl 0 encodes to 0, matching memset-zero slots.)
    bLocList[idx] = loc;
    bIdxList[idx] = idx;
    atomicMax(&bFlagsList[idx], NEW);
    snapValidList[idx] = 0;
    atomicAdd(createdCnt, 1);   // settlement loops: "did any rank CREATE a block this pass?"
  } else {
    // TOUCH: the block is already live -- only raise its flag.  It must keep its
    // snapshot validity and its live data: cascade kernels re-activate existing
    // blocks constantly (a KEEP block's own 1-ring includes itself), and zeroing
    // snapValid here would make the fill overwrite valid F_OLD with a smooth
    // prediction and wipe the block's detail every adaptation cycle.
    atomicMax(&bFlagsList[idx], NEW);
  }
}

// encode ijk indices and resolution level into locational code
__device__ u64 MultiLevelSparseGrid::encode(i32 lvl, i32 i, i32 j, i32 k) {
  i += 1; // add one so that boundary blocks are no longer negative negative
  j += 1;
  k += 1;
  u64 loc = 0;
  loc |= (u64)lvl << 60 | (u64) k << 40 | (u64)j << 20 | (u64)i;
  return loc;
}

// decode locational code into ij idx and resolution level
__device__ void MultiLevelSparseGrid::decode(u64 loc, i32 &lvl, i32 &i, i32 &j, i32 &k) {
  lvl = loc >> 60;
  k = ((loc >> 40) & ((1 << 20)-1)) - 1;
  j = ((loc >> 20) & ((1 << 20)-1)) - 1;
  i = (loc & ((1 << 20)-1)) - 1;
}

// render a single field f (>=0) or the refinement-level map (f=-1) to a png
void MultiLevelSparseGrid::paintField(i32 f, const char *fileName) {
  i32 nPix = imageSizeX[0]*imageSizeX[1];
  // clear first: each PE paints only its owned cells, so unpainted pixels must
  // read 0 for the cross-PE sum (and a shrunk single-PE frame leaves no stale).
  cudaMemset(imageDataX, 0, (size_t)nPix*sizeof(real));
  computeImageData(f);   // virtual: DG overrides it to interpolate from LGL nodes
  cudaDeviceSynchronize();
#ifdef USE_MGPU
  // combine the per-PE owned paints into one full-domain image (exactly one
  // rank wrote each pixel; the rest are 0), then only rank 0 writes the file.
  if (comm::size() > 1) {
    std::vector<double> buf(nPix);
    for (i32 t = 0; t < nPix; t++) buf[t] = (double)imageDataX[t];
    comm::allreduceSum(buf.data(), nPix);
    if (comm::rank() != 0) return;
    for (i32 t = 0; t < nPix; t++) imageDataX[t] = (real)buf[t];
  }
#endif
  png::image<png::gray_pixel_16> image(imageSizeX[0], imageSizeX[1]);

  // normalize image data and fill png image.  VOID pixels (solid side of a cut
  // element, or a dead block) are EXCLUDED from the range and reserved to
  // value 0 -- otherwise -1e30 sets minVal and every real value collapses onto
  // one grey level.  Live data therefore maps to [1, 65535].
  real maxVal = -1e32;
  real minVal = 1e32;
  i64 nVoid = 0;
  for (i32 idx=0; idx<imageSizeX[0]*imageSizeX[1]; idx++) {
    if (imageDataX[idx] <= (real)(0.5*kPaintVoid)) { nVoid++; continue; }
    maxVal = fmax(maxVal, imageDataX[idx]);
    minVal = fmin(minVal, imageDataX[idx]);
  }
  if (nVoid == (i64)imageSizeX[0]*imageSizeX[1]) { minVal = 0; maxVal = 1; }
  if (f == -1) {
    minVal = 0;
    maxVal = nLvls;
  }

  for (i32 j=0; j<imageSizeX[1]; j++) {
    for (i32 i=0; i<imageSizeX[0]; i++) {
      i32 idx = j*imageSizeX[0] + i;
      if (imageDataX[idx] <= (real)(0.5*kPaintVoid)) { image[j][i] = 0; continue; }
      double t = (imageDataX[idx] - minVal) / (maxVal - minVal + 1e-16);
      image[j][i] = (png::gray_pixel_16)(1.0 + t*65534.0);
    }
  }
  image.write(fileName);

  // THE SCALE SIDECAR.  paintField rescales by the frame's own min/max and the
  // PNG records neither, so absolute values were unrecoverable from the file
  // and no two frames shared a scale -- which makes an animation of a decaying
  // or growing field actively misleading.  One append-only CSV fixes both.
  {
    static bool hdr = false;
    FILE *fp = fopen("output/paint_scale.csv", hdr ? "a" : "w");
    if (fp) {
      if (!hdr) { fprintf(fp, "file,field,min,max,nvoid,nx,ny,domx,domy\n"); hdr = true; }
      fprintf(fp, "%s,%d,%.10e,%.10e,%lld,%d,%d,%.8f,%.8f\n", fileName, (i32)f,
              (double)minVal, (double)maxVal, (long long)nVoid,
              imageSizeX[0], imageSizeX[1],
              (double)domainSize[0], (double)domainSize[1]);
      fclose(fp);
    }
  }
}

#ifdef USE_MGPU
// Render the (replicated) domain partition: every pixel coloured by the rank
// owning its base column.  Independent of the distributed grid data -- rank 0
// draws it straight from ownerBase -- so it is cheap and always consistent.
// Called at every point the ownership map changes (Phase B re-cut, Phase C
// migration) to visualise the load partition and its evolution.
void MultiLevelSparseGrid::paintPartition(void) {
  if (comm::rank() != 0) { partImgCounter++; return; }
  i32 P = comm::size();
  i32 sub = blockSize*powi(2, nLvls-1);              // pixels per base column
  png::image<png::gray_pixel_16> image(imageSizeX[0], imageSizeX[1]);
  for (i32 py = 0; py < imageSizeX[1]; py++)
    for (i32 px = 0; px < imageSizeX[0]; px++) {
      i32 ci = px / sub, cj = py / sub;             // pseudo2D: k=0 plane
      i32 owner = ownerBase[ci + part.nb[0]*(cj + part.nb[1]*0)];
      image[py][px] = (png::gray_pixel_16)(((double)owner + 0.5)/(double)P * 65535.0);
    }
  char fn[80];
  sprintf(fn, "output/partition_%05d.png", partImgCounter++);
  image.write(fn);
}
#endif


void MultiLevelSparseGrid::paint(void) {
  cudaDeviceSynchronize();
  char fileName[80];
  // One full-domain figure per field: paintField sum-reduces the per-PE owned
  // paints and rank 0 writes it (single-rank keeps the identical filenames).
  // f = -1 grid, 0 Rho, 1 RhoU, 2 RhoV, 3 RhoW, 4 RhoE (total energy)
  for (i32 f=-1; f<5; f++) {
    if (f >= 0) sprintf(fileName, "output/image%02d_%05d.png", f, imageCounter);
    else        sprintf(fileName, "output/grid_%05d.png", imageCounter);
    paintField(f, fileName);
  }
#ifdef USE_MGPU
  if (dbgChecks) paintRankGrid();   // per-rank owned+ghost view (debug)
#endif
  imageCounter++;
}

#ifdef USE_MGPU
// Per-rank local-grid render (debug): every rank writes its OWN full-domain
// frame showing exactly the blocks it holds -- owned shaded by level (dark ->
// mid grey), partition ghosts in a bright band, absent regions black.  The
// owned/ghost frontier is the partition boundary; the ghost band width shows
// the Chebyshev-2 halo the stencils require.
void MultiLevelSparseGrid::paintRankGrid(void) {
  i32 nPix = imageSizeX[0]*imageSizeX[1];
  cudaMemset(imageDataX, 0, (size_t)nPix*sizeof(real));
  for (i32 L = 0; L < nLvls; L++) {   // coarse -> fine: finer blocks overwrite parents
    paintRankGridKernel<<<cudaGridSize, cudaBlockSize>>>(*this, L);
    cudaDeviceSynchronize();
  }
  png::image<png::gray_pixel_16> image(imageSizeX[0], imageSizeX[1]);
  real maxVal = (real)1;   // paintRankGridKernel writes intensity directly in [0,1]
  for (i32 j = 0; j < imageSizeX[1]; j++)
    for (i32 i = 0; i < imageSizeX[0]; i++)
      image[j][i] = (png::gray_pixel_16)(fmin(fmax(imageDataX[j*imageSizeX[0]+i], (real)0), maxVal) / maxVal * 65535);
  char fileName[80];
  sprintf(fileName, "output/rankgrid_r%d_%05d.png", part.rank, imageCounter);
  image.write(fileName);

  // ghost-layer census: owned/ghost/far-ghost per level
  cudaMemset(dbgCnt, 0, 64*sizeof(i32));
  ghostCensusKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  for (i32 r = 0; r < comm::size(); r++) {
    if (r == part.rank) {
      printf("[ghostcensus] r%d frame %d:", part.rank, imageCounter);
      for (i32 L = 0; L < nLvls; L++)
        printf("  L%d owned=%d ghost=%d far=%d", L, dbgCnt[3*L+0], dbgCnt[3*L+1], dbgCnt[3*L+2]);
      printf("\n");
    }
    comm::barrier();
  }
}
#endif

// default (finite-volume) image build: the GPU kernel that paintField used to
// launch directly.  Virtual so the DG solver can substitute an LGL-interpolating
// version (see DgSolver::computeImageData).
void MultiLevelSparseGrid::computeImageData(i32 f) {
  computeImageDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f);
}

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<cudaGridSize, cudaBlockSize>>>(*this);
}
*/
