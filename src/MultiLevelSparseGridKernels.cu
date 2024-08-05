
#include <stdio.h>
#include "MultiLevelSparseGridKernels.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid) {
  // initialize the blocks of the base grid level
  i32 i = threadIdx.x + blockIdx.x*blockDim.x;
	i32 j = threadIdx.y + blockIdx.x*blockDim.y;
  i32 k = threadIdx.z + blockIdx.x*blockDim.z;
  if (i < grid.baseGridSize[0]/blockSize && 
      j < grid.baseGridSize[1]/blockSize && 
      k < grid.baseGridSize[2]/blockSize) {
    grid.activateBlock(0, i, j, k);
  }
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
      u32 prntIdx = grid.hashTable.getValue(pLoc);  
      grid.prntIdxList[bIdx] = prntIdx;
    }

  END_BLOCK_LOOP
}


__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    u32 idx = 0;
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

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {

      u32 idx000 = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize, k-haloSize);
      u32 idx100 = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize, k-haloSize);
      u32 idx010 = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize, k-haloSize);
      u32 idx110 = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize, k-haloSize);
      u32 idx001 = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize, k+haloSize);
      u32 idx101 = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize, k+haloSize);
      u32 idx011 = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize, k+haloSize);
      u32 idx111 = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize, k+haloSize);

      u32 cEmpty = bEmpty * blockSizeTot;
      grid.cFlagsList[cIdx] = ACTIVE;
      if (idx000 >= cEmpty || idx100 >= cEmpty || idx010 >= cEmpty || idx110 >= cEmpty ||
          idx001 >= cEmpty || idx101 >= cEmpty || idx011 >= cEmpty || idx111 >= cEmpty) {
        grid.cFlagsList[cIdx] = GHOST;
      }

    }

  END_CELL_LOOP
}

__global__ void flagParentCellsKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && (cFlag == ACTIVE || cFlag == PARENT)) {

      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = k/2 + kb%2 * blockSize / 2;

      // parent cell memory index
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);

      grid.cFlagsList[pIdx] = PARENT;

    }

  END_CELL_LOOP
}

__global__ void addFineBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == REFINE) {
        // add finer blocks if not already on finest level
        grid.bFlagsList[bIdx] = KEEP;
        if (lvl < grid.nLvls-1) {
          for (i32 dk=0; dk<=1; dk++) {
            for (i32 dj=0; dj<=1; dj++) {
              for (i32 di=0; di<=1; di++) {
                grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj, 2*kb+dk);
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

    if (grid.isInteriorBlock(lvl, ib, jb, kb) && grid.bFlagsList[bIdx] == KEEP) {
      // add neighboring blocks
      for (i32 dk=-1; dk<=1; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            grid.activateBlock(lvl, ib+di, jb+dj, kb+dk);
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

    if (grid.isInteriorBlock(lvl, ib, jb, kb) && lvl > 2 && grid.bFlagsList[bIdx] == KEEP) {
      for (i32 dk=-1; dk<=1; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            grid.activateBlock(lvl-1, ib/2+di, jb/2+dj, kb/2+dk);
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

    if (grid.isInteriorBlock(lvl, ib, jb, kb) && 
       (ib == 0 || ib == grid.baseGridSize[0]/blockSize*powi(2,lvl)-1 ||
        jb == 0 || jb == grid.baseGridSize[1]/blockSize*powi(2,lvl)-1 ||
        kb == 0 || kb == grid.baseGridSize[2]/blockSize*powi(2,lvl)-1)) {
      // add neighboring exterior blocks
      for (i32 dk=-1; dk<=1; dk++) {
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            if (grid.isExteriorBlock(lvl, ib+di, jb+dj, kb+dk)) {
              grid.activateBlock(lvl, ib+di, jb+dj, kb+dk);            
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

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb) && loc != kEmpty && grid.cFlagsList[cIdx] == ACTIVE) {
      u32 nPixels = powi(2,(grid.nLvls - 1 - lvl));
      for (uint jj=0; jj<nPixels; jj++) {
        for (uint ii=0; ii<nPixels; ii++) {
          u32 iPxl = ib*blockSize*nPixels + i*nPixels + ii;
          u32 jPxl = jb*blockSize*nPixels + j*nPixels + jj;
          if (f >= 0) {
            grid.imageData[jPxl*grid.imageSize[0] + iPxl] = U[cIdx];
          }
          else {
            grid.imageData[jPxl*grid.imageSize[0] + iPxl] = (lvl+1);
          }
          if (f < 0 && gridOn && ii > 0 && jj > 0) {
            grid.imageData[jPxl*grid.imageSize[0] + iPxl] = 0;
          }
        }
      }
    }

  END_CELL_LOOP
}