#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"


static const int fields = {RHO, RHOU, RHOV, RHOE};
static const uint aux_fields = {AUX_RHO, AUX_RHOU, AUX_RHOV, AUX_RHOE};

/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  __shared__ dataType data;

  START_CELL_LOOP

  uint flags = solver.blockList[bIndex].flags;

  if (flags & ACTIVE) {
    // calculate wavelet coefficients in active cells
    for (uint f=0; f<4; f++){
      dataType& q = solver.getFieldValue(fields[f], bIndex, index);
      LOAD_LOW_RES_DATA(aux_fields[f])

    }

  }
  else if (flags & GHOST) {
    // set wavelet coefficients to zero in ghost cells
    for (uint f=0; f<4; f++){
      solver.getFieldValue(fields[f], bIndex, index) = 0.0;
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
__global__ void restrictFields(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  uint flags = solver.blockList[bIndex].flags;

  if (flags & ACTIVE == ACTIVE && lvl > 0) {
    for (uint f=0; f<4; f++){
      dataType& q = solver.getFieldValue(aux_fields[f], bIndex, i, j);
      dataType& qp = solver.getParentFieldValue(fields[f], bIndex, ib, jb, i, j);
      atomicAdd(&qp, q/4);
    }
  }

  END_CELL_LOOP
}
