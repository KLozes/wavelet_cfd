#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"

__device__ static constexpr u32 fields[4] = {RHO, RHOU, RHOV, RHOE};
__device__ static constexpr u32 auxFields[4] = {AUX_RHO, AUX_RHOU, AUX_RHOV, AUX_RHOE};

__global__ void sortfieldArrayKernel(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP
  for (u32 f=0; f<4; f++) {
    fieldArray &dataArray = grid.fieldArrayList[bIdx + fields[f]*nBlocksMax];
    fieldArray &dataArrayAux = grid.fieldArrayList[bIdx + auxFields[f]*nBlocksMax];
    dataArray[idx] = dataArrayAux[idx];
  }
  END_CELL_LOOP
}


/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  __shared__ dataType data;

  START_CELL_LOOP

  u32 flags = solver.blockList[bIdx].flags;

  if (flags & ACTIVE) {
    // calculate wavelet coefficients in active cells
    for (u32 f=0; f<4; f++){
      dataType& q = solver.getFieldValue(fields[f], bIdx, index);
      LOAD_LOW_RES_DATA(aux_fields[f])

    }

  }
  else if (flags & GHOST) {
    // set wavelet coefficients to zero in ghost cells
    for (u32 f=0; f<4; f++){
      solver.getFieldValue(fields[f], bIdx, index) = 0.0;
    }

  }

  END_CELL_LOOP
}


__global__ void inverseWaveletTransform(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  END_CELL_LOOP
}
*/

// restrict conserved fields to lower levels
/*
__global__ void restrictFields(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  u32 flags = solver.blockList[bIdx].flags;

  if (flags & ACTIVE == ACTIVE && lvl > 0) {
    for (u32 f=0; f<4; f++){
      dataType& q = solver.getFieldValue(aux_fields[f], bIdx, i, j);
      dataType& qp = solver.getParentFieldValue(fields[f], bIdx, ib, jb, i, j);
      atomicAdd(&qp, q/4);
    }
  }

  END_CELL_LOOP
}
*/

#endif
