#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"

__global__ void sortFieldDataKernel(CompressibleSolver &grid);
__global__ void copyFieldKernel(CompressibleSolver &grid, i32 fSrc, i32 fDst);
__global__ void gatherSortedFieldKernel(CompressibleSolver &grid, i32 fSrc, i32 fDst);

__global__ void setInitialConditionsKernel(CompressibleSolver &grid);

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 fOff, i32 prim);


__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid);

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid);

__global__ void computeGlobalScalesKernel(CompressibleSolver &grid);

__global__ void computePressureKernel(CompressibleSolver &grid);

__global__ void envCheckKernel(CompressibleSolver &grid);
__global__ void computeDeltaTKernel(CompressibleSolver &grid);

__global__ void computeRightHandSideKernel(CompressibleSolver &grid);
__global__ void gatherFaceFluxKernel(CompressibleSolver &grid);
__global__ void stateHashKernel(CompressibleSolver &grid);
__global__ void ibForceKernel(CompressibleSolver &grid);

// multiD Osher-type RHS: on-the-fly corner flux tensors + 1D Osher midpoints,
// Simpson face assembly
__global__ void multiDRhsKernel(CompressibleSolver &grid);
// CTU-Hancock half-step predictor (mdFlux==2): predicted primitives -> Old bank
__global__ void hancockPredictKernel(CompressibleSolver &grid);

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage);

#ifdef USE_MGPU
__global__ void markGhostsKernel(CompressibleSolver &grid);
__global__ void countDirKernel(CompressibleSolver &grid);
__global__ void fillDirKernel(CompressibleSolver &grid);
__global__ void consumeDirKernel(CompressibleSolver &grid);
__global__ void packDirKernel(CompressibleSolver &grid, i32 fOff, i32 nf);
__global__ void unpackDirKernel(CompressibleSolver &grid, i32 fOff, i32 nf);
__global__ void countBaseWeightsKernel(CompressibleSolver &grid);
__global__ void migrateInsertKernel(CompressibleSolver &grid, u64 *locs, i32 n, i32 *slots);
#endif

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid);

__global__ void forwardWaveletTransformKernel(CompressibleSolver &grid);

// diagnostic: normalized wavelet-detail indicator -> F_SCRATCH (see kernel)
__global__ void detailToScratchKernel(CompressibleSolver &grid, i32 mode);

__global__ void fillOldSnapshotKernel(CompressibleSolver &grid, i32 level);
__global__ void consumeNeedKernel(CompressibleSolver &grid);
__global__ void dbgVacKernel(CompressibleSolver &grid);
extern __device__ unsigned long long g_vacOwned;
extern __device__ unsigned long long g_ipTaint;
extern __device__ unsigned long long g_ibDetect;
extern __device__ double g_ibFx, g_ibFy;
extern __device__ unsigned long long g_ibFailDip;
extern __device__ unsigned long long g_ibFailSlip;
extern __device__ unsigned long long g_ibFailIp;
extern __device__ unsigned long long g_ibNup;
extern __device__ unsigned long long g_wmGhost;
extern __device__ unsigned long long g_wmCand;
extern __device__ double g_ibMaxDfc;
extern __device__ double g_ibMaxLvl;
extern __device__ unsigned long long g_ibFlux;
extern __device__ unsigned long long g_vacGhost;
__global__ void inverseWaveletTransformKernel(CompressibleSolver &grid);

__global__ void waveletThresholdingKernel(CompressibleSolver &grid);

__global__ void interpolateFieldsKernel(CompressibleSolver &grid, i32 lvlOnly);

__global__ void restrictFieldsKernel(CompressibleSolver &grid, i32 lvlOnly);

__global__ void turbClosureKernel(CompressibleSolver &grid);

__global__ void wallUtauKernel(CompressibleSolver &grid);

__global__ void zeroFieldKernel(CompressibleSolver &grid, i32 f);
__global__ void cutGeometryKernel(CompressibleSolver &grid);
__global__ void ransFieldProbeKernel(CompressibleSolver &grid, i32 which);

__global__ void wallGhostKernel(CompressibleSolver &grid);

__global__ void ibGhostKernel(CompressibleSolver &grid);
__global__ void ibIfaceKernel(CompressibleSolver &grid);
__global__ void shockSensorKernel(CompressibleSolver &grid);
__global__ void rccmReconstructKernel(CompressibleSolver &grid);
__global__ void zeroTrashBlockKernel(CompressibleSolver &grid, i32 f);
__global__ void zeroScalesKernel(CompressibleSolver &grid);
__global__ void zeroFlagsKernel(CompressibleSolver &grid);
__global__ void zeroAccumulatorKernel(CompressibleSolver &grid);
__global__ void ibStampGeometryKernel(CompressibleSolver &grid);
__global__ void stampLocalDtKernel(CompressibleSolver &grid, real dtGlobal, real dtCap);

__global__ void ransShearProbeKernel(CompressibleSolver &grid, real u0, real ky);

__global__ void ransWallProbeKernel(CompressibleSolver &grid, real uTau, real ypMin, i32 comp);

__global__ void ransDecayErrorKernel(CompressibleSolver &grid, real kEx, real tEx, i32 mode);

#endif
