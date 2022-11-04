#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "Settings.cuh"
#include "Util.cuh"


__global__ void copyBlockListToBlockListOld(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  grid.blockListOld[bIndex] = grid.blockList[bIndex];
  END_BLOCK_LOOP
}

__global__ void copyBlockListOldToBlockList(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  uint oldIndex = grid.blockList[bIndex].index;
  grid.blockList[bIndex] = grid.blockListOld[oldIndex];
  END_BLOCK_LOOP
}

/*
__global__ void updateBlockConnectivity(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  uint oldIndex = grid.blockList[bIndex].index;

  // update the index of this block
  grid.blockListOld[oldIndex].index = bIndex;

  // update the child index of this block's parent
  uint lvl, i, j;
  mortonDecode(grid.blockList[bIndex].loc, lvl,  i, j);
  uint pIndex = grid.blockList[bIndex].parent;
  grid.blockListOld[pIndex].children(i&1,j&1) = bIndex;

  // update the parent index of each of this blocks children
  for(uint i=0; i<powi(2,2); i++) {
    uint cIndex = grid.blockList[bIndex].children(i);
    grid.blockListOld[cIndex].parent = bIndex;
  }

  // update the neighbor index of this blocks neigbors
  for(uint i=0; i<powi(3,2); i++) {
    uint nIndex = grid.blockList[bIndex].neighbors(i);
    grid.blockListOld[nIndex].neighbors(8-i) = bIndex;
  }

  END_BLOCK_LOOP
}

__global__ void findNeighbors(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  uint64_t loc = grid.blockList[bIndex].loc;
  uint lvl, i, j;
  mortonDecode(loc, lvl, i, j);
  for(uint n=0; n<9; n++) {
    uint &nIndex = grid.blockList[bIndex].neighbors(n);
    if (nIndex == bEmpty) {
      uint di, dj;
      grid.getDijk(n, di, dj);
      nIndex = grid.getBlockIndex(lvl, i+di, j+dj);
    }
  }

  END_BLOCK_LOOP
}
*/

__global__ void reconstructionCheck(MultiLevelSparseGrid &grid) {
  START_DYNAMIC_BLOCK_LOOP

  uint flags = grid.blockList[bIndex].flags;

  END_DYNAMIC_BLOCK_LOOP
}

__global__ void zeroBlockCounter(MultiLevelSparseGrid &grid) {
  grid.blockCounter = 0;
}

#endif
