
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
  // this PE creates its owned base blocks plus a ghost ring: any candidate
  // (interior, or the -1/nb domain-exterior ring) within Chebyshev distance 2
  // of an owned base block is activated -- 2 blocks toward a partition
  // neighbor (the scatter-form flux needs a full +-2-cell stencil) and the
  // 1-deep domain-exterior ring fall out of the same rule.  Purely ownership-
  // map-driven, so it works for the box split and the Z-curve cut alike.
  bool near = false;
  i32 dkLim = grid.pseudo2D ? 0 : 2;
  for (i32 dk=-dkLim; dk<=dkLim && !near; dk++)
  for (i32 dj=-2; dj<=2 && !near; dj++)
  for (i32 di=-2; di<=2 && !near; di++) {
    i32 ni=i+di, nj=j+dj, nk=k+dk;
    if (ni < 0 || nj < 0 || nk < 0 ||
        ni >= grid.baseGridSize[0]/blockSize || nj >= grid.baseGridSize[1]/blockSize ||
        nk >= grid.baseGridSize[2]/blockSize) continue;
    if (grid.ownerPE(0, ni, nj, nk) == grid.part.rank) near = true;
  }
  if (inDomain && near) grid.activateBlock(0, i, j, k);
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
      // normalize the flag: the sort permuted blocks but not flags, so post-sort
      // flags are stale garbage.  Every surviving block is KEEP by definition;
      // consumers between sorts (directory publication's flag!=DELETE filter,
      // the inverse's DELETE guard) rely on this.
      grid.bFlagsList[bIdx] = KEEP;
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
      // validated lookup: a deleted parent's corpse key must bind bEmpty (the
      // zeroed trash slice), not the corpse slot -- see getBlockIdx
      grid.prntIdxList[bIdx] = grid.getBlockIdx(pLoc);
    }

  END_BLOCK_LOOP
}


__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid) {

  // The trash block's neighbor row must point at the trash block itself: a
  // block with a missing parent gets prntIdx == bEmpty, and the prediction
  // stencil then walks bEmpty's neighbor list.  Self-pointing slots land those
  // reads in the (zeroed) trash slice -- defined zeros, never stale indices
  // into arbitrary memory.  (Single-GPU never has missing parents; the MGPU
  // support closure can gap for one cycle at an advancing rank seam.)
  if (blockIdx.x == 0 && threadIdx.x < 27)
    grid.nbrIdxList[(size_t)bEmpty*27 + threadIdx.x] = bEmpty;

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    i32 idx = 0;
    for (i32 dk=-1; dk<2; dk++) {
      for(int dj=-1; dj<2; dj++) {
        for(int di=-1; di<2; di++) {
          u64 nbrLoc = grid.encode(lvl, ib+di, jb+dj, kb+dk);
          // validated lookup (corpse keys -> bEmpty), see getBlockIdx
          grid.nbrIdxList[bIdx*27+idx] = grid.getBlockIdx(nbrLoc);
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
            // owned targets only; a target in a neighbour's territory is recorded
            // as a NEED and sent to its owner (exchangeStructure), who creates it
            // and exports it back to us as a ghost in the same exchange
            if (!grid.isOwnedBlock(lvl, ni, nj, nk)) {
              if (grid.isInteriorBlock(lvl, ni, nj, nk) && grid.needCnt) {
                i32 o = grid.ownerPE(lvl, ni, nj, nk);
                i32 s = (o >= 0) ? grid.nbrOf[o] : -1;
                if (s >= 0) {
                  i32 q = atomicAdd(&grid.needCnt[s], 1);
                  if (q < grid.needSlot) grid.needLoc[(size_t)s*grid.needSlot + q] = grid.encode(lvl, ni, nj, nk);
                }
              }
              continue;
            }
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
            // owned targets only + NEED record (see addAdjacentBlocksKernel)
            if (!grid.isOwnedBlock(lvl-1, pi, pj, pk)) {
              if (grid.isInteriorBlock(lvl-1, pi, pj, pk) && grid.needCnt) {
                i32 o = grid.ownerPE(lvl-1, pi, pj, pk);
                i32 s = (o >= 0) ? grid.nbrOf[o] : -1;
                if (s >= 0) {
                  i32 q = atomicAdd(&grid.needCnt[s], 1);
                  if (q < grid.needSlot) grid.needLoc[(size_t)s*grid.needSlot + q] = grid.encode(lvl-1, pi, pj, pk);
                }
              }
              continue;
            }
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
    // Only blocks that SURVIVE this cascade (flagged by thresholding/refine/
    // grading) get a ghost halo -- and, under periodic BCs, force their
    // same-level seam image.  Without the flag check a to-be-deleted edge block
    // still forces its image (which is created NEW -> KEEP and survives), and
    // the image forces it back next cycle: seam pairs become immortal and their
    // 3x3 halos spread the band tangentially ~1 block/cycle -- unbounded seam
    // refinement long after the wave has passed.
    if (grid.isInteriorBlock(lvl, ib, jb, kb) && onBnd && grid.bFlagsList[bIdx] != DELETE) {
      // add neighboring exterior blocks (x-y only in pseudo2D)
      i32 dkLim = grid.pseudo2D ? 0 : 1;
      for (i32 dk=-dkLim; dk<=dkLim; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            if (grid.isExteriorBlock(lvl, ib+di, jb+dj, kb+dk)) {
              grid.activateBlock(lvl, ib+di, jb+dj, kb+dk);   // exterior ghost halo
              if (grid.periodic) {
                // the ghost's periodic image (opposite interior edge) is forced
                // at the same level so the ghost fill is an exact same-level
                // copy.  This mirrors seam-touching refinement onto the opposite
                // edge (a 1-block-deep band); removing it (and filling ghosts
                // from a coarser ancestor instead) was tried and is unstable
                // when a shock crosses the seam -- the transient level-
                // mismatched seam NaNs even with monotone-linear ghost fills.
                i32 ii=ib+di, ij=jb+dj, ik=kb+dk;
                grid.wrapBlockPeriodic(lvl, ii, ij, ik);
#ifdef USE_MGPU
                // owned-target rule: the wrapped image is an INTERIOR block that
                // may belong to another rank -- record a NEED for its owner
                // instead of manufacturing an unfillable orphan locally
                if (!grid.isOwnedBlock(lvl, ii, ij, ik)) {
                  i32 o = grid.ownerPE(lvl, ii, ij, ik);
                  i32 s = (o >= 0 && grid.nbrOf) ? grid.nbrOf[o] : -1;
                  if (s >= 0 && grid.needCnt) {
                    i32 q = atomicAdd(&grid.needCnt[s], 1);
                    if (q < grid.needSlot) grid.needLoc[(size_t)s*grid.needSlot + q] = grid.encode(lvl, ii, ij, ik);
                  }
                  continue;
                }
#endif
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

    // paint each ACTIVE leaf cell into its full-domain pixel footprint.  Under
    // MGPU restrict to OWNED cells so exactly one rank writes each pixel (ghost
    // halo cells are skipped) -- paintField then sum-reduces across ranks.
    bool paintCell = onMidPlane && grid.isInteriorBlock(lvl, ib, jb, kb)
                     && loc != kEmpty && grid.cFlagsList[cIdx] == ACTIVE;
#ifdef USE_MGPU
    paintCell = paintCell && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    if (paintCell) {
      i32 nPixels = powi(2,(grid.nLvls - 1 - lvl));
      i32 oxPxl = 0, oyPxl = 0;   // full-domain pixel indexing on every PE
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
//
// D0 integrity checks (runtime-gated by grid.dbgChecks; see the MGPU seam plan).
// Print at most a few violations per launch (dbgCnt-limited) and REPAIR bad
// bindings to bEmpty so a debug run continues past the first hit.
//

// Topology integrity: for every live block verify (1) the hash maps its loc back
// to this slot (catches duplicate keys / split-brain), (2) the parent binding
// points at a block whose loc IS the parent loc, (3) each of the 27 neighbor
// bindings points at a block whose loc IS that neighbor loc.  Stale bindings --
// e.g. a corpse slot of a deleted block, or an index list that predates a
// mid-cycle create/delete -- silently redirect stencil taps to wrong memory.
__global__ void checkTopologyKernel(MultiLevelSparseGrid &grid, i32 phaseTag) {
#ifdef USE_MGPU
  const i32 rk = grid.part.rank;
#else
  const i32 rk = 0;
#endif
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      // (1) hash self-consistency
      i32 self = grid.hashTable.getValue(loc);
      if (self != bIdx) {
        i32 n = atomicAdd(grid.dbgCnt, 1);
        if (n < 8) printf("[topo] r%d ph%d SELF lvl%d (%d,%d,%d) slot=%d getValue=%d\n",
                          rk, phaseTag, lvl, ib, jb, kb, bIdx, self);
      }
      // (2) parent binding
      if (lvl > 0 && grid.prntIdxList) {
        i32 p = grid.prntIdxList[bIdx];
        u64 pLoc = grid.encode(lvl-1, ib/2, jb/2, kb/2);
        if (p != bEmpty && grid.bLocList[p] != pLoc) {
          i32 n = atomicAdd(grid.dbgCnt, 1);
          if (n < 8) printf("[topo] r%d ph%d PRNT lvl%d (%d,%d,%d) slot=%d prnt=%d prntLoc=%llx want=%llx\n",
                            rk, phaseTag, lvl, ib, jb, kb, bIdx, p,
                            (unsigned long long)grid.bLocList[p], (unsigned long long)pLoc);
          grid.prntIdxList[bIdx] = bEmpty;   // repair: read the trash slice, not wrong memory
        }
      }
      // (3) neighbor bindings
      if (grid.nbrIdxList) {
        i32 t = 0;
        for (i32 dk=-1; dk<2; dk++)
        for (i32 dj=-1; dj<2; dj++)
        for (i32 di=-1; di<2; di++) {
          i32 nb = grid.nbrIdxList[(size_t)bIdx*27 + t];
          u64 nLoc = grid.encode(lvl, ib+di, jb+dj, kb+dk);
          if (nb != bEmpty && grid.bLocList[nb] != nLoc) {
            i32 n = atomicAdd(grid.dbgCnt, 1);
            if (n < 8) printf("[topo] r%d ph%d NBR lvl%d (%d,%d,%d)+(%d,%d,%d) slot=%d nbr=%d nbrLoc=%llx want=%llx\n",
                              rk, phaseTag, lvl, ib, jb, kb, di, dj, dk, bIdx, nb,
                              (unsigned long long)grid.bLocList[nb], (unsigned long long)nLoc);
            grid.nbrIdxList[(size_t)bIdx*27 + t] = bEmpty;
          }
          t++;
        }
      }
    }
  END_BLOCK_LOOP
}

// Fill-support check: before prolonging F_OLD into new owned blocks at `level`,
// verify each such block has (a) a valid parent binding, (b) a parent with a
// valid snapshot, and (c) no missing parent-ring tap (the 27-tap wavelet stencil
// touches all 26 parent neighbors; on the union grid the reconstruction closure
// guarantees they exist, so a bEmpty interior tap means the seam protocol failed
// to import or adopt that block).  Prints the missing tap loc and its owner rank.
__global__ void checkFillSupportKernel(MultiLevelSparseGrid &grid, i32 level) {
#ifdef USE_MGPU
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      if (lvl == level && grid.isInteriorBlock(lvl, ib, jb, kb)
          && grid.isOwnedBlock(lvl, ib, jb, kb)
          && grid.snapValidList[bIdx] == 0 && grid.bFlagsList[bIdx] != DELETE) {
        i32 p = grid.prntIdxList[bIdx];
        u64 pLoc = grid.encode(lvl-1, ib/2, jb/2, kb/2);
        if (p == bEmpty || grid.bLocList[p] != pLoc) {
          i32 n = atomicAdd(grid.dbgCnt, 1);
          if (n < 8) printf("[fillchk] r%d lvl%d (%d,%d,%d): parent binding invalid (p=%d)\n",
                            grid.part.rank, lvl, ib, jb, kb, p);
        }
        else if (grid.snapValidList[p] != 1) {
          i32 n = atomicAdd(grid.dbgCnt, 1);
          if (n < 8) printf("[fillchk] r%d lvl%d (%d,%d,%d): parent (%d,%d,%d) snapValid=%d\n",
                            grid.part.rank, lvl, ib, jb, kb, ib/2, jb/2, kb/2, grid.snapValidList[p]);
        }
        else {
          i32 t = 0, dkL = grid.pseudo2D ? 0 : 1;
          for (i32 dk=-1; dk<2; dk++)
          for (i32 dj=-1; dj<2; dj++)
          for (i32 di=-1; di<2; di++) {
            bool inPlane = grid.pseudo2D ? (dk == 0) : true;
            if (inPlane && !(di==0 && dj==0 && dk==0)) {
              i32 pi = ib/2+di, pj = jb/2+dj, pk = kb/2+dk;
              if (grid.isInteriorBlock(lvl-1, pi, pj, pk)
                  && grid.nbrIdxList[(size_t)p*27 + t] == bEmpty) {
                i32 n = atomicAdd(grid.dbgCnt, 1);
                if (n < 8) printf("[fillchk] r%d lvl%d (%d,%d,%d): parent-ring tap lvl%d (%d,%d,%d) MISSING (owner=r%d)\n",
                                  grid.part.rank, lvl, ib, jb, kb, lvl-1, pi, pj, pk,
                                  grid.ownerPE(lvl-1, pi, pj, pk));
              }
            }
            t++;
          }
          (void)dkL;
        }
      }
    }
  END_BLOCK_LOOP
#endif
}

#ifdef USE_MGPU
// Per-rank grid render (debug): paint THIS rank's local blocks at one level --
// owned blocks shaded by refinement level (1..nLvls), ghost (non-owned interior)
// blocks in a bright band (nLvls+2 .. 2*nLvls+1).  Called coarse-to-fine so finer
// blocks overwrite their parents; unpainted pixels stay 0 (not present locally).
// The owned/ghost frontier in the image IS the partition boundary.
__global__ void paintRankGridKernel(MultiLevelSparseGrid &grid, i32 level) {
  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    bool onMidPlane;
    if (grid.pseudo2D) onMidPlane = (k == blockSize/2);
    else {
      i32 nPixelsZ = powi(2,(grid.nLvls - 1 - lvl));
      i32 zMid = grid.baseGridSize[2]*powi(2, grid.nLvls-1) / 2;
      i32 kLo = (kb*blockSize + k) * nPixelsZ;
      onMidPlane = (zMid >= kLo && zMid < kLo + nPixelsZ);
    }

    if (loc != kEmpty && lvl == level && onMidPlane
        && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      bool ghost = !grid.isOwnedBlock(lvl, ib, jb, kb);
      real val = ghost ? (real)(grid.nLvls + 2 + lvl) : (real)(lvl + 1);
      i32 nPixels = powi(2,(grid.nLvls - 1 - lvl));
      for (i32 jj=0; jj<nPixels; jj++) {
        for (i32 ii=0; ii<nPixels; ii++) {
          i32 iPxl = ib*blockSize*nPixels + i*nPixels + ii;
          i32 jPxl = jb*blockSize*nPixels + j*nPixels + jj;
          if (iPxl < 0 || iPxl >= grid.imageSizeX[0] || jPxl < 0 || jPxl >= grid.imageSizeX[1]) continue;
          grid.imageDataX[jPxl*grid.imageSizeX[0] + iPxl] =
            (ii > 0 && jj > 0) ? (real)0 : val;   // gridlines like the grid render
        }
      }
    }

  END_CELL_LOOP
}
#endif
