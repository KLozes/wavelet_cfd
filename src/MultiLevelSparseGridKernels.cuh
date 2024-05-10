#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid) {
  // initialize the blocks of the base grid level
  i32 idx = threadIdx.x + blockIdx.x*blockDim.x;
  i32 i = idx % (grid.baseGridSize[0]/blockSize); // plus an exterior block on each side
	i32 j = idx / (grid.baseGridSize[0]/blockSize);
  if (i < grid.baseGridSize[0]/blockSize && j < grid.baseGridSize[1]/blockSize) {
    grid.activateBlock(0, i, j);
  }
}

__global__ void updateIndicesKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    grid.bIdxList[bIdx] = bIdx;
    grid.hashTable.setValue(grid.bLocList[bIdx], bIdx);  

  END_BLOCK_LOOP
}

__global__ void updatePrntIndicesKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (lvl > 0) {
      u64 pLoc = grid.mortonEncode(lvl-1, ib/2, jb/2);
      u32 prntIdx = grid.hashTable.getValue(pLoc);  
      grid.prntIdxList[bIdx] = prntIdx;
    }

  END_BLOCK_LOOP
}


__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid) {

  START_HALO_CELL_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    i32 di = 0;
    if (i < haloSize) {di = -1;}
    if (i >= blockSize + haloSize) {di = 1;}

    i32 dj = 0;
    if (j < haloSize) {dj = -1;}
    if (j >= blockSize + haloSize) {dj = 1;}

    i32 iNbr = ib + di;
    i32 jNbr = jb + dj;

    u64 nbrLoc = grid.mortonEncode(lvl, iNbr, jNbr);
    u32 nbrIdx = grid.hashTable.getValue(nbrLoc);

    i32 il = i - di * blockSize - haloSize;
    i32 jl = j - dj * blockSize - haloSize;

    /*
    if (nbrIdx == bEmpty && lvl > 0) {
      // check lower lvl
      nbrLoc = grid.mortonEncode(lvl-1, iNbr/2, jNbr/2);
      nbrIdx = grid.hashTable.getValue(nbrLoc);
      il = (i+haloSize)/2 % blockSize;
      jl = (j+haloSize)/2 % blockSize;
    }
    */

    u32 lIdx = il + jl * blockSize;
    grid.nbrIdxList[cIdx] = nbrIdx*blockSizeTot + lIdx;

  END_HALO_CELL_LOOP
}


__global__ void addFineBlocks(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == REFINE) {
        // add finer blocks if not already on finest level
        if (lvl < grid.nLvls-1) {
          for (i32 dj=0; dj<=1; dj++) {
            for (i32 di=0; di<=1; di++) {
              grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj);
            }
          }
        }
      } 
    }
  END_BLOCK_LOOP

}

__global__ void addAdjacentBlocks(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == KEEP || grid.bFlagsList[bIdx] == NEW || grid.bFlagsList[bIdx] == REFINE) {
        // add neighboring blocks
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            grid.activateBlock(lvl, ib+di, jb+dj);
          }
        }
      } 
    }

  END_BLOCK_LOOP
}

__global__ void addReconstructionBlocks(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP
    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl > 1 && (grid.bFlagsList[bIdx] == KEEP || grid.bFlagsList[bIdx] == REFINE || grid.bFlagsList[bIdx] == NEW)) {
        // add reconstruction blocks
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            grid.activateBlock(lvl-1, ib/2+di, jb/2+dj);
            atomicMax(&(grid.nBlocks),grid.hashTable.nKeys);
          }
        }
      } 
    }

  END_BLOCK_LOOP
}

__global__ void deleteBlocks(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (lvl > 1 && grid.bFlagsList[bIdx] == DELETE) {
      grid.bLocList[bIdx] = kEmpty;
    }

  END_BLOCK_LOOP
}

__global__ void addBoundaryBlocks(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb) && 
       (ib == 0 || ib == grid.baseGridSize[0]/blockSize*powi(2,lvl)-1 ||
        jb == 0 || jb == grid.baseGridSize[1]/blockSize*powi(2,lvl)-1)) {
      // add neighboring exterior blocks
      for (i32 dj=-1; dj<=1; dj++) {
        for (i32 di=-1; di<=1; di++) {
          if (grid.isExteriorBlock(lvl, ib+di, jb+dj)) {
              grid.activateBlock(lvl, ib+di, jb+dj);            
          }
        }
      }
    }

  END_BLOCK_LOOP
}

#endif
