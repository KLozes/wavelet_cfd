#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"


enum Fields : u32 {
  RHO = 0,
  RHOU = 1,
  RHOV = 2,
  RHOW = 3,
  RHOE = 4,
  AUX_RHO = 5,
  AUX_RHOU = 6,
  AUX_RHOV = 7,
  AUX_RHOW = 8,
  AUX_RHOE = 9,
  RHS_RHO = 10,
  RHS_RHOU = 11,
  RHS_RHOV = 12,
  RHS_RHOW = 13,
  RHS_RHOE = 14,
  MU = 15
};

class CompressibleSolver : public MultiLevelSparseGrid {
public:
  CompressibleSolver(u32 baseSize_[], u32 nLvlsMax_) :
  MultiLevelSparseGrid(baseSize_, nLvlsMax_, 15){}

};

#endif
