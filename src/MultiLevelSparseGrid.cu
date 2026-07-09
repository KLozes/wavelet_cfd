#include <thrust/sort.h>
#include <algorithm>

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
#ifdef USE_MGPU
  // per-PE image TILE: only this PE's owned x-y extent at finest resolution, so
  // no PE ever allocates the full-domain image (each writes its tile to a
  // rank-prefixed png; stitch offline).
  imageSizeX[0] = (part.b1[0]-part.b0[0])*blockSize*powi(2,nLvls-1);
  imageSizeX[1] = (part.b1[1]-part.b0[1])*blockSize*powi(2,nLvls-1);
#else
  imageSizeX[0] = (baseGridSize[0])*powi(2,nLvls-1);  // x (width)
  imageSizeX[1] = (baseGridSize[1])*powi(2,nLvls-1);  // y (height)
#endif

  imageSizeY[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSizeY[1] = (baseGridSize[2])*powi(2,nLvls-1);

  imageSizeZ[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSizeZ[1] = (baseGridSize[1])*powi(2,nLvls-1);

  imageCounter = 0;

  // grid size checking (the dense base-grid capacity is checked in
  // initializeBaseGrid, where that grid is actually materialized -- a sparse
  // user like the narrowband SDF never fills it and is bounded only by the
  // number of blocks it activates).
  assert(isPowerOf2(blockSize));

  // always needed: block location codes / memory indices / flags (the hash table
  // is a member with its own allocation)
  cudaMallocManaged(&bLocList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&bIdxList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&bFlagsList, nBlocksMax*sizeof(i32));
  cudaMemset(bLocList, 0, nBlocksMax*sizeof(u64));
  cudaMemset(bIdxList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));

  // flow-solver-only per-block arrays + the slice buffer: skipped in lean mode
  // (the narrowband SDF needs none of them -- see the `lean` note in the header).
  prntIdxList = nbrIdxList = cFlagsList = nullptr;
  imageDataX  = nullptr;
  if (!lean) {
    cudaMallocManaged(&prntIdxList, nBlocksMax*sizeof(i32));
    cudaMallocManaged(&nbrIdxList, 27*nBlocksMax*sizeof(i32));
    cudaMallocManaged(&cFlagsList, blockSizeTot*nBlocksMax*sizeof(i32));
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

// Morton code of a base block (2D interleave in pseudo-2D, 3D otherwise).
static u64 mortonCode(i32 x, i32 y, i32 z, bool twoD) {
  u64 m = 0;
  for (i32 b = 0; b < 21; b++) {
    if (twoD) {
      m |= ((u64)((x >> b) & 1)) << (2*b) | ((u64)((y >> b) & 1)) << (2*b+1);
    } else {
      m |= ((u64)((x >> b) & 1)) << (3*b)   | ((u64)((y >> b) & 1)) << (3*b+1)
         | ((u64)((z >> b) & 1)) << (3*b+2);
    }
  }
  return m;
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
  std::vector<std::pair<u64,i32>> order(n);
  for (i32 k = 0; k < part.nb[2]; k++)
  for (i32 j = 0; j < part.nb[1]; j++)
  for (i32 i = 0; i < part.nb[0]; i++) {
    i32 idx = i + part.nb[0]*(j + part.nb[1]*k);
    order[idx] = {mortonCode(i, j, k, twoD), idx};
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
  for (i32 d = 0; d < 3; d++) part.nb[d] = baseGridSize[d]/blockSize;

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
                    baseGridSize[2]/blockSize/8+1);
  initGridKernel<<<nCudaBlocks3, cudaBlockSize3>>>(*this);
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

void MultiLevelSparseGrid::adaptGrid(void) {

  if (nLvls > 1) {
    addFineBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    addAdjacentBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    for(i32 lvl=nLvls-1; lvl>2; lvl--) {
      setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      addReconstructionBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    }
    addBoundaryBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    nBlocks = hashTable.nKeys;
    deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    updatePrntIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }
}

void MultiLevelSparseGrid::sortBlocks(void) {

  cudaDeviceSynchronize();
  thrust::sort_by_key(thrust::device, bLocList, bLocList+hashTable.nKeys, bIdxList);
  sortFieldData();
  cudaDeviceSynchronize();
  hashTable.reset();
  hashTable.nKeys = nBlocks;
  updateIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // hash-table rebuild (always)
  if (!lean) {   // parent/neighbor indices + cell flags are flow-solver-only
    updatePrntIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    updateNbrIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    // periodicity is applied in setBoundaryConditions (the exterior ghost blocks
    // are filled from their wrap-around image), not by remapping neighbor slots.
    flagActiveCellsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    flagParentCellsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }
  cudaDeviceSynchronize();
}

__device__ Vec3 MultiLevelSparseGrid::getCellPos(i32 lvl, i32 ib, i32 jb, i32 kb, i32 i, i32 j, i32 k) {
  return Vec3((ib*blockSize + i + .5)*getDx(lvl),  
              (jb*blockSize + j + .5)*getDy(lvl), 
              (kb*blockSize + k + .5)*getDz(lvl));
}

__device__ i32 MultiLevelSparseGrid::getNbrIdx(i32 bIdx, i32 i, i32 j, i32 k) {
  i += blockSize;
  j += blockSize;
  k += blockSize;
  i32 ib = i / blockSize;
  i32 jb = j / blockSize;
  i32 kb = k / blockSize;
  // A within-block offset (ib=jb=kb=1) is always this block itself; return bIdx
  // directly to skip the neighbor-list read on the common in-block cell lookup.
  i32 nbrIdx = (ib == 1 && jb == 1 && kb == 1)
             ? bIdx
             : nbrIdxList[27*bIdx + ib + 3*jb + 9*kb];
  return blockSizeTot*nbrIdx + (i%blockSize) + (j%blockSize)*blockSize + (k%blockSize)*blockSize*blockSize;
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
                     pseudo2D ? i32(baseGridSize[2]/blockSize)
                              : i32(baseGridSize[2]/blockSize*powi(2,lvl))};
  return i >= 0 && j >= 0 && k >= 0 && i < gridSize[0] && j < gridSize[1] && k < gridSize[2];
}

__device__ bool MultiLevelSparseGrid::isExteriorBlock(i32 lvl, i32 i, i32 j, i32 k) {
  return !isInteriorBlock(lvl, i, j, k);
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
    i32 gz = baseGridSize[2]/blockSize*powi(2,lvl);
    k = ((k % gz) + gz) % gz;
  }
}

__device__ void MultiLevelSparseGrid::activateBlock(i32 lvl, i32 i, i32 j, i32 k) {
  u64 loc = encode(lvl, i, j, k);
  i32 idx = hashTable.insert(loc);
  if (idx != bEmpty) { 
    // new key was inserted if not bEmpty
    bLocList[idx] = loc;
    bIdxList[idx] = idx;
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
  png::image<png::gray_pixel_16> image(imageSizeX[0], imageSizeX[1]);
  computeImageDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f);
  cudaDeviceSynchronize();

  // normalize image data and fill png image
  real maxVal = -1e32;
  real minVal = 1e32;
  for (i32 idx=0; idx<imageSizeX[0]*imageSizeX[1]; idx++) {
    maxVal = fmax(maxVal, imageDataX[idx]);
    minVal = fmin(minVal, imageDataX[idx]);
  }
  if (f == -1) {
    minVal = 0;
    maxVal = nLvls;
  }

  for (i32 j=0; j<imageSizeX[1]; j++) {
    for (i32 i=0; i<imageSizeX[0]; i++) {
      i32 idx = j*imageSizeX[0] + i;
      image[j][i] = (imageDataX[idx] - minVal) / (maxVal - minVal + 1e-16) * 65535;
    }
  }
  image.write(fileName);
}

void MultiLevelSparseGrid::paint(void) {
  cudaDeviceSynchronize();
  char fileName[80];
#ifdef USE_MGPU
  // each PE renders its own subdomain tile to a rank-prefixed file so ranks do
  // not collide (single-rank output keeps the original names for A/B diffing).
  // NOTE: computeImageData reads fieldData on the host; under the real MPI
  // backend that must first be staged to host memory (loopback is managed, ok).
  char pre[16] = "";
  if (comm::size() > 1) sprintf(pre, "r%d_", comm::rank());
#else
  const char *pre = "";
#endif
  // f = -1 grid, 0 Rho, 1 RhoU, 2 RhoV, 3 RhoW, 4 RhoE (total energy)
  for (i32 f=-1; f<5; f++) {
    if (f >= 0) sprintf(fileName, "output/%simage%02d_%05d.png", pre, f, imageCounter);
    else        sprintf(fileName, "output/%sgrid_%05d.png", pre, imageCounter);
    paintField(f, fileName);
  }
  imageCounter++;
}

void MultiLevelSparseGrid::computeImageData(i32 f) {

  real *U;
  if (f >= 0) {
    U = getField(f);
  }

  bool gridOn = false;

  // set the pixel values 
  for (uint bIdx=0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    decode(loc, lvl, ib, jb, kb);
    if (isInteriorBlock(lvl, ib, jb, kb) && loc != kEmpty) {
      for (uint j = 0; j < blockSize; j++) {
        for (uint i = 0; i < blockSize; i++) {
          i32 idx = i + blockSize * j + bIdx*blockSizeTot;
          i32 nPixels = powi(2,(nLvls - 1 - lvl));
          for (uint jj=0; jj<nPixels; jj++) {
            for (uint ii=0; ii<nPixels; ii++) {
              i32 iPxl = ib*blockSize*nPixels + i*nPixels + ii;
              i32 jPxl = jb*blockSize*nPixels + j*nPixels + jj;
              if (f >= 0) {
                imageDataX[jPxl*imageSizeX[0] + iPxl] = U[idx];
              }
              else {
                i32 cFlag = cFlagsList[idx];
                imageDataX[jPxl*imageSizeX[0] + iPxl] = lvl+1 - (2-cFlag)/2;
              }
              if (gridOn && ii > 0 && jj > 0) {
                  imageDataX[jPxl*imageSizeX[0] + iPxl] = 0;
              }
            }
          }
        }
      }
    }
  }
}

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<cudaGridSize, cudaBlockSize>>>(*this);
}
*/
