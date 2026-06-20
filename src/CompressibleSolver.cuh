#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

static constexpr real gam = 1.4;

//
// 3D compressible Euler solver (HLLC flux + TVD reconstruction + TVD-RK3).
//
// field layout (nFields = 16).  fields 0-4 alternate between conservative and
// primitive storage in place (see conservative/primitive conversion kernels):
//
//   0 : Rho  | Rho
//   1 : RhoU | U
//   2 : RhoV | V
//   3 : RhoW | W
//   4 : RhoE | P
//   5..9   : Old{Rho,RhoU,RhoV,RhoW,RhoE}   (RK3 substep storage)
//   10..14 : Rhs{Rho,RhoU,RhoV,RhoW,RhoE}   (right hand side accumulator)
//   15     : DeltaT / MagRhoU               (scratch, reused)
//
static constexpr i32 nCompressibleFields = 16;

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

  i32 iter;

  CompressibleSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nCompressibleFields) {
      cfl = .5;
      waveletThresh = .005;
      iter = 0;
      immerserdBcType = 0;
      bcType = 0;
      icType = 0;

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

  void writeLineProfile(const char *fileName); // 1D profile dump for validation
  void printDiagnostics(void);                  // AMR-boundary spike / pseudo-2D diagnostics
  void paintPressure(const char *fileName);     // render the pressure field to a png

  __device__ Vec5 prim2cons(Vec5 prim);
  __device__ Vec5 cons2prim(Vec5 cons);
  __device__ real lim(real &r);
  __device__ real tvdRec(real &ul, real &uc, real &ur);
  __device__ Vec5 hllcFlux(Vec5 qL, Vec5 qR, Vec3 normal);

  __device__ real getBoundaryLevelSet(Vec3 pos);
  __device__ real calcIbMask(real phi);

};

#endif
