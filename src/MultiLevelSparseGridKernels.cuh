#ifndef MULTILEVEL_SPARSE_GRID_KERNELS_H
#define MULTILEVEL_SPARSE_GRID_KERNELS_H

#include "MultiLevelSparseGrid.cuh"

__global__ void initTreeKernel(MultiLevelSparseGrid &grid);
 
__global__ void initGridKernel(MultiLevelSparseGrid &grid);

__global__ void updateIndicesKernel(MultiLevelSparseGrid &grid);

__global__ void updatePrntIndicesKernel(MultiLevelSparseGrid &grid);

__global__ void resetOldTreeKernel(MultiLevelSparseGrid &grid);

__global__ void updateOldTreeKernel(MultiLevelSparseGrid &grid);

__global__ void copyTreeFromOldKernel(MultiLevelSparseGrid &grid);

__global__ void updateNbrIndicesKernel(MultiLevelSparseGrid &grid);

__global__ void flagActiveCellsKernel(MultiLevelSparseGrid &grid);

__global__ void flagParentCellsKernel(MultiLevelSparseGrid &grid);

__global__ void addFineBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void setBlocksKeepKernel(MultiLevelSparseGrid &grid);

__global__ void setBlocksDeleteKernel(MultiLevelSparseGrid &grid);

__global__ void addAdjacentBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void addReconstructionBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void deleteDataKernel(MultiLevelSparseGrid &grid);

__global__ void setFlagsToDelete(MultiLevelSparseGrid &grid);

__global__ void addBoundaryBlocksKernel(MultiLevelSparseGrid &grid);

__global__ void computeImageDataKernel(MultiLevelSparseGrid &grid, i32 f);

#endif
