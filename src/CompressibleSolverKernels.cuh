#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"

__global__ void sortFieldDataKernel(CompressibleSolver &grid);

__global__ void setInitialConditionsKernel(CompressibleSolver &grid);

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 fOff);


__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid);

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid);

__global__ void computeGlobalScalesKernel(CompressibleSolver &grid);

__global__ void computePressureKernel(CompressibleSolver &grid);

__global__ void computeDeltaTKernel(CompressibleSolver &grid);

__global__ void computeRightHandSideKernel(CompressibleSolver &grid);

// multiD Osher-type RHS: on-the-fly corner flux tensors + 1D Osher midpoints,
// Simpson face assembly, RT0 slope updates
__global__ void multiDRhsKernel(CompressibleSolver &grid);
// CTU-Hancock half-step predictor (mdFlux==2): predicted primitives -> Old bank
__global__ void hancockPredictKernel(CompressibleSolver &grid);

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage);

#ifdef USE_MGPU
__global__ void haloExchangeKernel(CompressibleSolver &grid, void **peers, i32 fOff, i32 nf);
__global__ void markGhostsKernel(CompressibleSolver &grid);
__global__ void rebuildGhostsKernel(CompressibleSolver &grid, void **peers);
#endif

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid);

__global__ void forwardWaveletTransformKernel(CompressibleSolver &grid);

__global__ void inverseWaveletTransformKernel(CompressibleSolver &grid);

__global__ void waveletThresholdingKernel(CompressibleSolver &grid);

__global__ void interpolateFieldsKernel(CompressibleSolver &grid);

__global__ void restrictFieldsKernel(CompressibleSolver &grid);

#endif
