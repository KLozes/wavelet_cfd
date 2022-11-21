#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

/*

__global__ void copyBlockListToBlockListOld(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  grid.blockListOld[bIndex] = grid.blockList[bIndex];
  END_BLOCK_LOOP
}

__global__ void copyBlockListOldToBlockList(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  u32 oldIndex = grid.blockList[bIndex].index;
  grid.blockList[bIndex] = grid.blockListOld[oldIndex];
  END_BLOCK_LOOP
}

/*
__global__ void updateBlockConnectivity(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  u32 oldIndex = grid.blockList[bIndex].index;

  // update the index of this block
  grid.blockListOld[oldIndex].index = bIndex;

  // update the child index of this block's parent
  u32 lvl, i, j;
  mortonDecode(grid.blockList[bIndex].loc, lvl,  i, j);
  u32 pIndex = grid.blockList[bIndex].parent;
  grid.blockListOld[pIndex].children(i&1,j&1) = bIndex;

  // update the parent index of each of this blocks children
  for(u32 i=0; i<powi(2,2); i++) {
    u32 cIndex = grid.blockList[bIndex].children(i);
    grid.blockListOld[cIndex].parent = bIndex;
  }

  // update the neighbor index of this blocks neigbors
  for(u32 i=0; i<powi(3,2); i++) {
    u32 nIndex = grid.blockList[bIndex].neighbors(i);
    grid.blockListOld[nIndex].neighbors(8-i) = bIndex;
  }

  END_BLOCK_LOOP
}

__global__ void findNeighbors(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  u64 loc = grid.blockList[bIndex].loc;
  u32 lvl, i, j;
  mortonDecode(loc, lvl, i, j);
  for(u32 n=0; n<9; n++) {
    u32 &nIndex = grid.blockList[bIndex].neighbors(n);
    if (nIndex == bEmpty) {
      u32 di, dj;
      grid.getDijk(n, di, dj);
      nIndex = grid.getBlockIndex(lvl, i+di, j+dj);
    }
  }

  END_BLOCK_LOOP
}

__global__ void reconstructionCheck(MultiLevelSparseGrid &grid) {
  START_DYNAMIC_BLOCK_LOOP

  u32 flags = grid.blockList[bIndex].flags;

  END_DYNAMIC_BLOCK_LOOP
}

__global__ void zeroBlockCounter(MultiLevelSparseGrid &grid) {
  grid.blockCounter = 0;
}
*/

#endif
