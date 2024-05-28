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
  
    if (grid.bLocList[bIdx] != kEmpty) {
      grid.bIdxList[bIdx] = bIdx;
      grid.hashTable.insert(grid.bLocList[bIdx]);
      grid.hashTable.setValue(grid.bLocList[bIdx], bIdx);
    }

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

    u32 lIdx = il + jl * blockSize;
    grid.nbrIdxList[cIdx] = nbrIdx*blockSizeTot + lIdx;

  END_HALO_CELL_LOOP
}

__global__ void flagActiveCellsKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      u32 lIdx = grid.getNbrIdx(bIdx, i-haloSize, j);
      u32 rIdx = grid.getNbrIdx(bIdx, i+haloSize, j);
      u32 dIdx = grid.getNbrIdx(bIdx, i, j-haloSize);
      u32 uIdx = grid.getNbrIdx(bIdx, i, j+haloSize);
      u32 ldIdx = grid.getNbrIdx(bIdx, i-haloSize, j-haloSize);
      u32 rdIdx = grid.getNbrIdx(bIdx, i+haloSize, j-haloSize);
      u32 luIdx = grid.getNbrIdx(bIdx, i-haloSize, j+haloSize);
      u32 ruIdx = grid.getNbrIdx(bIdx, i+haloSize, j+haloSize);

      u32 cEmpty = bEmpty * blockSizeTot;
      grid.cFlagsList[cIdx] = ACTIVE;
      if (lIdx >= cEmpty  || rIdx >= cEmpty  || dIdx >= cEmpty  || uIdx >= cEmpty ||
          ldIdx >= cEmpty || rdIdx >= cEmpty || luIdx >= cEmpty || ruIdx >= cEmpty) {
        grid.cFlagsList[cIdx] = GHOST;
      }

    }

  END_CELL_LOOP
}

__global__ void flagParentCellsKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && (cFlag == ACTIVE || cFlag == PARENT)) {

      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent cell memory index
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);

      grid.cFlagsList[pIdx] = PARENT;

    }

  END_CELL_LOOP
}

__global__ void addFineBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == REFINE) {
        // add finer blocks if not already on finest level
        grid.bFlagsList[bIdx] = KEEP;
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

__global__ void addAdjacentBlocksKernel(MultiLevelSparseGrid &grid) {

  START_BLOCK_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl == 0 || grid.bFlagsList[bIdx] == KEEP ) {
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

__global__ void addReconstructionBlocksKernel(MultiLevelSparseGrid &grid) {

  START_DYNAMIC_BLOCK_LOOP
    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      if (lvl > 1 && grid.bFlagsList[bIdx] == KEEP) {
        // add reconstruction blocks
        for (i32 dj=-1; dj<=1; dj++) {
          for (i32 di=-1; di<=1; di++) {
            grid.activateBlock(lvl-1, ib/2+di, jb/2+dj);
            atomicMax(&(grid.nBlocks),grid.hashTable.nKeys);
          }
        }
      } 
    }

  END_DYNAMIC_BLOCK_LOOP
}

__global__ void deleteDataKernel(MultiLevelSparseGrid &grid) {

  START_CELL_LOOP

    i32 lvl, ib, jb;
    u64 loc = grid.bLocList[bIdx];
    grid.mortonDecode(loc, lvl, ib, jb);

    if (lvl > 1 && (grid.bFlagsList[bIdx] == DELETE || grid.bLocList[bIdx] == kEmpty)) {
      grid.bLocList[bIdx] = kEmpty;
      grid.bIdxList[bIdx] = bEmpty;
      grid.cFlagsList[cIdx] = 0;
      for(i32 f=0; f<grid.nFields; f++) {
        dataType *F = grid.getField(f);
        F[cIdx] = 0;
      }
    }

  END_CELL_LOOP
}

__global__ void addBoundaryBlocksKernel(MultiLevelSparseGrid &grid) {

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
