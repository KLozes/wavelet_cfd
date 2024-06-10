#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

static constexpr dataType gam = 1.4;

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  dataType deltaT;
  dataType cfl;
  dataType maxRho;
  dataType maxMagRhoU;
  dataType maxRhoE;
  dataType waveletThresh;

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;
  
  i32 immerserdBcType;
  i32 bcType;
  i32 icType;

  u32 iter;

  CompressibleSolver(dataType *domainSize_, u32 *baseGridSize_, u32 nLvls_, dataType cfl_, dataType waveletThresh_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, 13) {
      cfl = cfl_;
      waveletThresh = waveletThresh_;
      iter = 0;
      immerserdBcType = 0;
      bcType = 0;

      tGrid = 0.0;
      tSolver = 0.0;
      tOutput = 0.0;
      tTotal = 0.0;
  }

  void initialize(void);
  dataType step(dataType dt);
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

  __host__ __device__ Vec4 prim2cons(Vec4 prim);
  __host__ __device__ Vec4 cons2prim(Vec4 cons);
  __host__ __device__ dataType lim(dataType &r);
  __host__ __device__ dataType tvdRec(dataType &ul, dataType &uc, dataType &ur);
  __host__ __device__ Vec4 hlleFlux(Vec4 qL, Vec4 qR, Vec2 normal);
  __host__ __device__ Vec4 hllcFlux(Vec4 qL, Vec4 qR, Vec2 normal);
  __host__ __device__ dataType getBoundaryLevelSet(Vec2 pos);

};

#endif
