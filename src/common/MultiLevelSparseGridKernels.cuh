#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

__global__ void initGridKernel(MultiLevelSparseGrid &grid);

__global__ void updateIndicesKernel(MultiLevelSparseGrid &grid);

__global__ void buildSortKeysKernel(MultiLevelSparseGrid &grid);   // sortCurve: level-major Hilbert/Morton keys

__global__ void updatePrntIndicesKernel(MultiLevelSparseGrid &grid);

__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid);
__global__ void countSortGroupsKernel(MultiLevelSparseGrid &grid, i32 *cnt);   // --leaf: [0] leaf-bearing, [1] exterior
__global__ void updateChldIndicesKernel(MultiLevelSparseGrid &grid);   // --leaf: child block slots
__global__ void resetLeafFacesKernel(MultiLevelSparseGrid &grid);      // --leaf: cellMortar = -1 on every live block
__global__ void buildMortarsKernel(MultiLevelSparseGrid &grid);        // --leaf: mortar faces at level jumps

__global__ void flagActiveCellsKernel(MultiLevelSparseGrid &grid);

__global__ void flagParentCellsKernel(MultiLevelSparseGrid &grid);

__global__ void addFineBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void setBlocksKeepKernel(MultiLevelSparseGrid &grid);

__global__ void setBlocksDeleteKernel(MultiLevelSparseGrid &grid);

__global__ void addAdjacentBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void addReconstructionBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void activateParentBlocksKernel(MultiLevelSparseGrid &grid, i32 lvl, i32 i, i32 j);

__global__ void deleteDataKernel(MultiLevelSparseGrid &grid);

__global__ void setFlagsToDelete(MultiLevelSparseGrid &grid);

__global__ void addBoundaryBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void computeImageDataKernel(MultiLevelSparseGrid &grid, i32 f);

__global__ void checkTopologyKernel(MultiLevelSparseGrid &grid, i32 phaseTag);

__global__ void checkFillSupportKernel(MultiLevelSparseGrid &grid, i32 level);

#ifdef USE_MGPU
__global__ void paintRankGridKernel(MultiLevelSparseGrid &grid, i32 level);
__global__ void ghostCensusKernel(MultiLevelSparseGrid &grid);
#endif


// pseudo2D: copy the live k == 0 plane into k = 1..blockSize-1 (see the
// START_CELL_LOOP comment in MultiLevelSparseGrid.cuh).
__global__ void broadcastZKernel(MultiLevelSparseGrid &grid, i32 fOnly);

#endif
