#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

static constexpr real gam = 1.4;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;
  real maxRho;
  real maxMagRhoU;
  real maxRhoE;
  real waveletThresh;

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;
  
  i32 immerserdBcType;
  i32 bcType;
  i32 icType;

  u32 iter;

  CompressibleSolver(real *domainSize_, u32 *baseGridSize_, u32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, 13) {
      cfl = .5;
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

  __device__ Vec4 prim2cons(Vec4 prim);
  __device__ Vec4 cons2prim(Vec4 cons);
  __device__ real limU(real &r);
  __device__ real lim(real &r);
  __device__ real tvdRecU(real &ul, real &uc, real &ur);
  __device__ real tvdRec(real &ul, real &uc, real &ur);
  __device__ Vec4 hlleFlux(Vec4 qL, Vec4 qR, Vec2 normal);
  __device__ Vec4 hllcFlux(Vec4 qL, Vec4 qR, Vec2 normal);
  __device__ real getBoundaryLevelSet(Vec2 pos);
  __device__ real calcIbMask(real phi);

};

#endif
