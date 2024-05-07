#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid) {
  // initialize the blocks of the base grid level
  i32 idx = threadIdx.x + blockIdx.x*blockDim.x;
  i32 i = idx % (grid.baseGridSize[0]/blockSize + 2); // plus an exterior block on each side
	i32 j = idx / (grid.baseGridSize[0]/blockSize + 2);
  if (i < grid.baseGridSize[0]/blockSize + 2 && j < grid.baseGridSize[1]/blockSize + 2) {
    grid.activateBlock(0, i-1, j-1);
  }
}

__global__ void updateIndicesKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    grid.bIdxList[bIdx] = bIdx;
    grid.hashTable.setValue(grid.zLocList[bIdx], bIdx);  

  END_BLOCK_LOOP
}


__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid) {

  START_HALO_CELL_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.zLocList[bIdx];
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

#endif
