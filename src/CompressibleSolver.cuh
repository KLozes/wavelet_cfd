#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

static constexpr real gam = 1.4;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;
  real sos;
  real maxRho;
  real maxMagU;
  real maxP;
  real waveletThresh;

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;
  
  i32 immerserdBcType;
  i32 bcType;
  i32 icType;

  i32 iter;

  CompressibleSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, 13) {
      cfl = .5;
      sos = 20.0;
      waveletThresh = .005;
      iter = 0;
      immerserdBcType = 0;
      bcType = 0;

      tGrid = 0.0;
      tSolver = 0.0;
      tOutput = 0.0;
      tTotal = 0.0;
  }

  void initialize(void);
  real step(real dt);
  void sortFieldData(void);
  void setInitialConditions(void);
  void setBoundaryConditions(void);
  void conservativeToPrimitive(void);
  void primitiveToConservative(void);
  void forwardWaveletTransform(void);
  void inverseWaveletTransform(void);

  void computeDeltaT(void);
  void computeRightHandSide(void);
  void updateFields(i32 stage);

  void restrictFields();
  void interpolateFields();

  __device__ real getBoundaryLevelSet(Vec3 pos);
  __device__ real calcIbMask(real phi);

};

#endif
