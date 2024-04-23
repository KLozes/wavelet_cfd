#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid) {
  // initialize the blocks of the base grid level
  u32 i = threadIdx.x + blockIdx.x*blockDim.x;
	u32 j = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < grid.baseGridSize[0]/blockSize && j < grid.baseGridSize[1]/blockSize) {
    grid.activateBlock(0, i, j);
  }
}

/*

__global__ void updateIndices1(MultiLevelSparseGrid &grid) {
  // update the tree connectivity
  START_BLOCK_LOOP
  u64 loc = grid.locList[bIdx];
  u32 lvl, i, j, k;
  mortonDecode(loc, lvl, i, j, k);

  // find the index of my child and update its parent to my new index
  for (u32 i=0; i<powi(2, nDim); i++) {
    grid.prntListOld[grid.childList[bIdx][i]] = bIdx;
  }

  // find the index of my parent and update its child to my new index
  grid.chldListOld[grid.prntList[bIdx]](i&1,j&1,k&1) = bIdx;

  END_BLOCK_LOOP
}

__global__ void updateIndices2(MultiLevelSparseGrid &grid) {
  // finally sort the tree connectivity
  START_BLOCK_LOOP
  bIdxOld = grid.idxList[bIdx];
  grid.idxList[bIdx] = bIdx;
  grid.childList[bIdx] = grid.childListOld[bIdxOld];
  grid.prntList[bIdx] = grid.prntListOld[bIdxOld];
  END_BLOCK_LOOP
}

__global__ void copyBlockListToBlockListOld(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  grid.blockListOld[bIdx] = grid.blockList[bIdx];
  END_BLOCK_LOOP
}

__global__ void copyBlockListOldToBlockList(MultiLevelSparseGrid &grid) {
  START_BLOCK_LOOP
  u32 oldIndex = grid.blockList[bIdx].index;
  grid.blockList[bIdx] = grid.blockListOld[oldIndex];
  END_BLOCK_LOOP
}


__global__ void updateBlockConnectivity(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  u32 oldIndex = grid.blockList[bIdx].index;

  // update the index of this block
  grid.blockListOld[oldIndex].index = bIdx;

  // update the child index of this block's parent
  u32 lvl, i, j;
  mortonDecode(grid.blockList[bIdx].loc, lvl,  i, j);
  u32 pIndex = grid.blockList[bIdx].parent;
  grid.blockListOld[pIndex].children(i&1,j&1) = bIdx;

  // update the parent index of each of this blocks children
  for(u32 i=0; i<powi(2,2); i++) {
    u32 cIndex = grid.blockList[bIdx].children(i);
    grid.blockListOld[cIndex].parent = bIdx;
  }

  // update the neighbor index of this blocks neigbors
  for(u32 i=0; i<powi(3,2); i++) {
    u32 nIndex = grid.blockList[bIdx].neighbors(i);
    grid.blockListOld[nIndex].neighbors(8-i) = bIdx;
  }

  END_BLOCK_LOOP
}

__global__ void findNeighbors(MultiLevelSparseGrid &grid)
{
  START_BLOCK_LOOP

  u64 loc = grid.blockList[bIdx].loc;
  u32 lvl, i, j;
  mortonDecode(loc, lvl, i, j);
  for(u32 n=0; n<9; n++) {
    u32 &nIndex = grid.blockList[bIdx].neighbors(n);
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

  u32 flags = grid.blockList[bIdx].flags;

  END_DYNAMIC_BLOCK_LOOP
}

__global__ void zeroBlockCounter(MultiLevelSparseGrid &grid) {
  grid.blockCounter = 0;
}
*/

#endif
