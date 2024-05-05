#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

typedef struct Flux {
  dataType fRho;
  dataType fRhoU;
  dataType fRhoV;
  dataType fRhoE;
} flux;

static constexpr dataType gam = 1.4;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  CompressibleSolver(dataType *domainSize_, u32 *baseGridSize_, u32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, 16) {}

  void sortFieldData(void);
  void setInitialConditions(i32 icType);
  void setBoundaryConditions(i32 bcType);

  __host__ __device__ Flux HLLEflux(const dataType qL[4], const dataType qR[4], const dataType normal[2]);
  __host__ __device__ Flux Centralflux(const dataType qL[4], const dataType qR[4], const dataType normal[2]);

};

#endif
