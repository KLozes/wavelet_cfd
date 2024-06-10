#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"

__global__ void sortFieldDataKernel(CompressibleSolver &grid);

__global__ void setInitialConditionsKernel(CompressibleSolver &grid);

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid);

__global__ void computeMagRhoUKernel(CompressibleSolver &grid);

__global__ void computeDeltaTKernel(CompressibleSolver &grid);

__global__ void computeRightHandSideKernel(CompressibleSolver &grid);

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage);

__global__ void updateFieldsRK3Kernel(CompressibleSolver &grid, i32 stage);

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid);

__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid);

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid);

__global__ void forwardWaveletTransformKernel(CompressibleSolver &grid);

__global__ void inverseWaveletTransformKernel(CompressibleSolver &grid);

__global__ void waveletThresholdingKernel(CompressibleSolver &grid);

__global__ void interpolateFieldsKernel(CompressibleSolver &grid);

__global__ void restrictFieldsKernel(CompressibleSolver &grid);

#endif
