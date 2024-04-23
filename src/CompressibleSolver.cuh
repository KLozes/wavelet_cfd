#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

class CompressibleSolver : public MultiLevelSparseGrid {
public:
  CompressibleSolver(u32 *baseGridSize_, u32 nLvls_) :
    MultiLevelSparseGrid(baseGridSize_, nLvls_, 16) {}

  void sortFieldArray(void);
};

#endif
