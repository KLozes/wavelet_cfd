#ifndef STGD_LOW_MACH_SOLVER_KERNELS_H
#define STGD_LOW_MACH_SOLVER_KERNELS_H

#include "StgdLowMachSolver.cuh"

__global__ void sortFieldDataKernel(StgdLowMachSolver &grid);

__global__ void setInitialConditionsKernel(StgdLowMachSolver &grid);

__global__ void setBoundaryConditionsKernel(StgdLowMachSolver &grid);

__global__ void computeMagRhoUKernel(StgdLowMachSolver &grid);

__global__ void computeDeltaTKernel(StgdLowMachSolver &grid);

__global__ void computeRightHandSideKernel(StgdLowMachSolver &grid);

__global__ void updateFieldsKernel(StgdLowMachSolver &grid, i32 stage);

__global__ void updateFieldsRK3Kernel(StgdLowMachSolver &grid, i32 stage);

__global__ void copyToOldFieldsKernel(StgdLowMachSolver &grid);

__global__ void conservativeToPrimitiveKernel(StgdLowMachSolver &grid);

__global__ void primitiveToConservativeKernel(StgdLowMachSolver &grid);

__global__ void forwardWaveletTransformKernel(StgdLowMachSolver &grid);

__global__ void inverseWaveletTransformKernel(StgdLowMachSolver &grid);

__global__ void waveletThresholdingKernel(StgdLowMachSolver &grid);

__global__ void interpolateFieldsKernel(StgdLowMachSolver &grid);

__global__ void restrictFieldsKernel(StgdLowMachSolver &grid);

#endif
