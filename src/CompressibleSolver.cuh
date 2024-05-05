#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

typedef struct State {
  dataType rho;
  dataType rhoU;
  dataType rhoV;
  dataType rhoE;
} state;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  static constexpr dataType gamma = 1.4;

  CompressibleSolver(dataType *domainSize_, u32 *baseGridSize_, u32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, 16) {}

  
  void initFieldData(u32 initType);
  State getInitCondition(u32 initType, dataType *pos);

  void sortFieldData(void);

};

#endif
