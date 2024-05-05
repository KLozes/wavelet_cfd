#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"


__global__ void sortFieldDataKernel(MultiLevelSparseGrid &grid) {
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  dataType *OldRho  = grid.getField(4);
  dataType *OldRhoU = grid.getField(5);
  dataType *OldRhoV = grid.getField(6);
  dataType *OldRhoE = grid.getField(7);

  START_CELL_LOOP

    u32 bIdxOld = grid.bIdxList[bIdx];
    u32 cIdxOld = bIdxOld * blockSizeTot + idx;
    
    Rho[cIdx] = OldRho[cIdxOld];
    RhoU[cIdx] = OldRhoU[cIdxOld];
    RhoV[cIdx] = OldRhoV[cIdxOld];
    RhoE[cIdx] = OldRhoE[cIdxOld];

  END_CELL_LOOP

}




/*

__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  dataType *Rho   = grid.getField(0);
  dataType *RhoU  = grid.getField(1);
  dataType *RhoV  = grid.getField(2);
  dataType *RhoEi = grid.getField(3);

  START_CELL_LOOP
  
  i32 i,j;
  grid.getLocalIdx(idx, i, j);
  


  END_CELL_LOOP
}
*/

/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  dataType *Rho   = grid.getField(0);
  dataType *RhoU  = grid.getField(1);
  dataType *RhoV  = grid.getField(2);
  dataType *RhoEi = grid.getField(3);

  START_CELL_LOOP

  END_CELL_LOOP
}
*/
/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  __shared__ dataType data;

  dataType *Rho = grid.getFieldData(0, )
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
