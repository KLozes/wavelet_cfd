
#include <stdio.h>
#include "MultiLevelSparseGridKernels.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid) {
  // initialize the blocks of the base grid level
  i32 i = threadIdx.x + blockIdx.x*blockDim.x - 1;
  i32 j = threadIdx.y + blockIdx.y*blockDim.y - 1;
  i32 k = threadIdx.z + blockIdx.z*blockDim.z - 1;
  if (grid.pseudo2D) k = 0;   // single z-block, no z-halo blocks
  bool inDomain = (i < grid.baseGridSize[0]/blockSize + 1 &&
                   j < grid.baseGridSize[1]/blockSize + 1 &&
                   k < grid.baseGridSize[2]/blockSize + 1);
#ifdef USE_MGPU
  // this PE creates its owned base box plus a ghost ring: 2 blocks thick toward a
  // partition-neighbor PE (the scatter-form flux computes the seam-face flux and
  // needs a full +-2-cell stencil, reaching the 2nd neighbor block), and 1 block
  // (the domain-exterior ring, index -1 / nb) toward a true domain boundary.
  i32 nbx = grid.baseGridSize[0]/blockSize;
  i32 nby = grid.baseGridSize[1]/blockSize;
  i32 nbz = grid.baseGridSize[2]/blockSize;
  i32 lo0 = (grid.part.b0[0] > 0)   ? grid.part.b0[0]-2 : -1;
  i32 hi0 = (grid.part.b1[0] < nbx) ? grid.part.b1[0]+1 : nbx;
  i32 lo1 = (grid.part.b0[1] > 0)   ? grid.part.b0[1]-2 : -1;
  i32 hi1 = (grid.part.b1[1] < nby) ? grid.part.b1[1]+1 : nby;
  i32 lo2 = (grid.part.b0[2] > 0)   ? grid.part.b0[2]-2 : -1;
  i32 hi2 = (grid.part.b1[2] < nbz) ? grid.part.b1[2]+1 : nbz;
  bool inRing = (i >= lo0 && i <= hi0 && j >= lo1 && j <= hi1 &&
                 (grid.pseudo2D || (k >= lo2 && k <= hi2)));
  if (inDomain && inRing) grid.activateBlock(0, i, j, k);
#else
  if (inDomain) grid.activateBlock(0, i, j, k);
#endif
}

__global__ void updateIndicesKernel(MultiLevelSparseGrid &grid) {
  // update the hashtable with new sorted indices
  START_BLOCK_LOOP

    if (grid.bLocList[bIdx] != kEmpty) {
      grid.bIdxList[bIdx] = bIdx;
      grid.hashTable.insertValue(grid.bLocList[bIdx], bIdx);
    }

  END_BLOCK_LOOP
}

__global__ void updatePrntIndicesKernel(MultiLevelSparseGrid &grid) {
  // update the parent indices list
  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl > 0) {
      u64 pLoc = grid.encode(lvl-1, ib/2, jb/2, kb/2);
      i32 prntIdx = grid.hashTable.getValue(pLoc);  
      grid.prntIdxList[bIdx] = prntIdx;
    }

  END_BLOCK_LOOP
}


__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    i32 idx = 0;
    for (i32 dk=-1; dk<2; dk++) {
      for(int dj=-1; dj<2; dj++) {
        for(int di=-1; di<2; di++) {
          u64 nbrLoc = grid.encode(lvl, ib+di, jb+dj, kb+dk);
          grid.nbrIdxList[bIdx*27+idx] = grid.hashTable.getValue(nbrLoc);
          idx++;
        }
      }
    }

  END_BLOCK_LOOP

}

__global__ void flagActiveCellsKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {

      i32 cEmpty = bEmpty * blockSizeTot;
      grid.cFlagsList[cIdx] = ACTIVE;

      if (grid.pseudo2D) {
        // only the x-y reconstruction stencil matters (z is a single block)
        i32 i00 = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize, k);
        i32 i10 = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize, k);
        i32 i01 = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize, k);
        i32 i11 = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize, k);
        if (i00 >= cEmpty || i10 >= cEmpty || i01 >= cEmpty || i11 >= cEmpty) {
          grid.cFlagsList[cIdx] = GHOST;
        }
      }
      else {
        i32 idx000 = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize, k-haloSize);
        i32 idx100 = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize, k-haloSize);
        i32 idx010 = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize, k-haloSize);
        i32 idx110 = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize, k-haloSize);
        i32 idx001 = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize, k+haloSize);
        i32 idx101 = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize, k+haloSize);
        i32 idx011 = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize, k+haloSize);
        i32 idx111 = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize, k+haloSize);
        if (idx000 >= cEmpty || idx100 >= cEmpty || idx010 >= cEmpty || idx110 >= cEmpty ||
            idx001 >= cEmpty || idx101 >= cEmpty || idx011 >= cEmpty || idx111 >= cEmpty) {
          grid.cFlagsList[cIdx] = GHOST;
        }
      }

    }

  END_CELL_LOOP
}

__global__ void flagParentCellsKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && (cFlag == ACTIVE || cFlag == PARENT)) {

      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices (z maps identically in pseudo2D: no z refine)
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);

      // parent cell memory index
      i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);

      grid.cFlagsList[pIdx] = PARENT;

    }

  END_CELL_LOOP
}

__global__ void addFineBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    bool refineBlk = grid.isInteriorBlock(lvl, ib, jb, kb);
#ifdef USE_MGPU
    // multi-GPU: a PE refines (and keeps dense level 1 on) only its OWNED blocks.
    // adaptGrid thus produces owned blocks + the domain-exterior ring; the
    // partition ghost layer is rebuilt separately (rebuildGhosts) from the
    // neighbors' actual blocks, so it can be pruned as features move.
    refineBlk = refineBlk && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    if (refineBlk) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == REFINE) {
        // add finer blocks if not already on finest level
        grid.bFlagsList[bIdx] = KEEP;
        if (lvl < grid.nLvls-1) {
          i32 dkMax = grid.pseudo2D ? 0 : 1;
          for (i32 dk=0; dk<=dkMax; dk++) {
            for (i32 dj=0; dj<=1; dj++) {
              for (i32 di=0; di<=1; di++) {
                i32 kc = grid.pseudo2D ? kb : 2*kb+dk;
                grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj, kc);
              }
            }
          }
        }
      } 
    }

  END_BLOCK_LOOP

}

__global__ void setBlocksKeepKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    if (grid.bFlagsList[bIdx] == NEW ) {
      grid.bFlagsList[bIdx] = KEEP;
    }

  END_BLOCK_LOOP
}

__global__ void setBlocksDeleteKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    grid.bFlagsList[bIdx] = DELETE;

  END_BLOCK_LOOP
}

__global__ void addAdjacentBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    bool gradeBlk = grid.isInteriorBlock(lvl, ib, jb, kb) && grid.bFlagsList[bIdx] == KEEP;
#ifdef USE_MGPU
    gradeBlk = gradeBlk && grid.isOwnedBlock(lvl, ib, jb, kb);   // grade only owned; ghosts via rebuild
#endif
    if (gradeBlk) {
      // add neighboring blocks (x-y only in pseudo2D).  Periodic: wrap the target
      // into the interior so an edge block grades its opposite-edge image (a real
      // interior block) rather than a lone exterior ghost -- keeping the two seam
      // edges refined to matching levels.  Wrap is identity for interior targets.
      i32 dkLim = grid.pseudo2D ? 0 : 1;
      for (i32 dk=-dkLim; dk<=dkLim; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            i32 ni=ib+di, nj=jb+dj, nk=kb+dk;
            if (grid.periodic) grid.wrapBlockPeriodic(lvl, ni, nj, nk);
#ifdef USE_MGPU
            // don't create partition ghosts here (rebuildGhosts does, mirroring
            // the neighbor's real refinement); only conform the owned region.
            if (grid.isInteriorBlock(lvl, ni, nj, nk) && !grid.isOwnedBlock(lvl, ni, nj, nk)) continue;
#endif
            grid.activateBlock(lvl, ni, nj, nk);
          }
        }
      }
    }

  END_BLOCK_LOOP
}

__global__ void addReconstructionBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    // activate parents and neghbors needed for wavelet transform
    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    bool reconBlk = grid.isInteriorBlock(lvl, ib, jb, kb) && lvl > 2 && grid.bFlagsList[bIdx] == KEEP;
#ifdef USE_MGPU
    reconBlk = reconBlk && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    if (reconBlk) {
      // periodic: wrap the parent target so a near-seam block's coarse
      // reconstruction support is built on the opposite edge too (identity for
      // interior targets), keeping the image's parent chain present.
      i32 dkLim = grid.pseudo2D ? 0 : 1;
      for (i32 dk=-dkLim; dk<=dkLim; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            i32 pi=ib/2+di, pj=jb/2+dj, pk=kb/2+dk;
            if (grid.periodic) grid.wrapBlockPeriodic(lvl-1, pi, pj, pk);
#ifdef USE_MGPU
            if (grid.isInteriorBlock(lvl-1, pi, pj, pk) && !grid.isOwnedBlock(lvl-1, pi, pj, pk)) continue;
#endif
            grid.activateBlock(lvl-1, pi, pj, pk);
          }
        }
      }
    }

  END_BLOCK_LOOP
}

__global__ void deleteDataKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP

    if (grid.bFlagsList[bIdx] == DELETE) {
      if (cIdx % blockSizeTot == 0) {
        grid.bLocList[bIdx] = kEmpty;
        grid.bIdxList[bIdx] = bEmpty;
        atomicAdd(&(grid.nBlocks), -1);
      }
      grid.cFlagsList[cIdx] = 0;
      for(i32 f=0; f<grid.nFields; f++) {
        real *F = grid.getField(f);
        F[cIdx] = 0;
      }
    }

  END_CELL_LOOP
}

__global__ void addBoundaryBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    bool onBnd = (ib == 0 || ib == grid.baseGridSize[0]/blockSize*powi(2,lvl)-1 ||
                  jb == 0 || jb == grid.baseGridSize[1]/blockSize*powi(2,lvl)-1);
    if (!grid.pseudo2D) {
      onBnd = onBnd || (kb == 0 || kb == grid.baseGridSize[2]/blockSize*powi(2,lvl)-1);
    }
    if (grid.isInteriorBlock(lvl, ib, jb, kb) && onBnd) {
      // add neighboring exterior blocks (x-y only in pseudo2D)
      i32 dkLim = grid.pseudo2D ? 0 : 1;
      for (i32 dk=-dkLim; dk<=dkLim; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            if (grid.isExteriorBlock(lvl, ib+di, jb+dj, kb+dk)) {
              grid.activateBlock(lvl, ib+di, jb+dj, kb+dk);   // exterior ghost halo
              if (grid.periodic) {
                // the ghost's periodic image (opposite interior edge) must exist
                // at the same level so setBoundaryConditions can fill the ghost
                // from a real block (else the same-level lookup misses -> vacuum)
                i32 ii=ib+di, ij=jb+dj, ik=kb+dk;
                grid.wrapBlockPeriodic(lvl, ii, ij, ik);
                grid.activateBlock(lvl, ii, ij, ik);
              }
            }
          }
        }
      }
    }

  END_BLOCK_LOOP
}

__global__ void computeImageDataKernel(MultiLevelSparseGrid &grid, i32 f) {

  bool gridOn = true;

  real *U;
  if (f >= 0) {
    U = grid.getField(f);
  }

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    // only render the cells covering the mid-z plane (x-y slice).  In pseudo2D
    // z is never refined (all blocks at kb=0, layers identical), so the level-
    // scaled pixel test below would never match a fine cell; just pick a fixed
    // representative z-layer instead.
    bool onMidPlane;
    if (grid.pseudo2D) {
      onMidPlane = (k == blockSize/2);
    }
    else {
      i32 nPixelsZ = powi(2,(grid.nLvls - 1 - lvl));
      i32 zMid = grid.baseGridSize[2]*powi(2, grid.nLvls-1) / 2;
      i32 kLo = (kb*blockSize + k) * nPixelsZ;
      onMidPlane = (zMid >= kLo && zMid < kLo + nPixelsZ);
    }

    if (onMidPlane && grid.isInteriorBlock(lvl, ib, jb, kb) && loc != kEmpty && grid.cFlagsList[cIdx] == ACTIVE) {
      i32 nPixels = powi(2,(grid.nLvls - 1 - lvl));
#ifdef USE_MGPU
      // pixels are indexed into THIS PE's tile (owned extent), so offset by the
      // tile origin; cells outside it (the ghost halo) fall out and are skipped.
      i32 oxPxl = grid.part.b0[0]*blockSize*powi(2, grid.nLvls-1);
      i32 oyPxl = grid.part.b0[1]*blockSize*powi(2, grid.nLvls-1);
#else
      i32 oxPxl = 0, oyPxl = 0;
#endif
      for (uint jj=0; jj<nPixels; jj++) {
        for (uint ii=0; ii<nPixels; ii++) {
          i32 iPxl = ib*blockSize*nPixels + i*nPixels + ii - oxPxl;
          i32 jPxl = jb*blockSize*nPixels + j*nPixels + jj - oyPxl;
          if (iPxl < 0 || iPxl >= grid.imageSizeX[0] || jPxl < 0 || jPxl >= grid.imageSizeX[1]) continue;
          if (f >= 0) {
            grid.imageDataX[jPxl*grid.imageSizeX[0] + iPxl] = U[cIdx];
          }
          else {
            grid.imageDataX[jPxl*grid.imageSizeX[0] + iPxl] = (lvl+1);
          }
          if (f < 0 && gridOn && ii > 0 && jj > 0) {
            grid.imageDataX[jPxl*grid.imageSizeX[0] + iPxl] = 0;
          }
        }
      }
    }

  END_CELL_LOOP
}