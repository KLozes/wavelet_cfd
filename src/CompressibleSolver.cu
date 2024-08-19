#include <iostream>
#include <thrust/extrema.h>


#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"


void CompressibleSolver::initialize(void) {
  initializeBaseGrid();
  setInitialConditions();
  sortBlocks();
  if (bcType == 1) {
    updateNbrIndicesPeriodicKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }
  setBoundaryConditions();
  cudaDeviceSynchronize();
  printf("nblocks %d\n", hashTable.nKeys);
  paint();

  for(i32 lvl=1; lvl<nLvls; lvl++){
    forwardWaveletTransform();
    adaptGrid();
    setInitialConditions();
    setBoundaryConditions();
    sortBlocks();
    cudaDeviceSynchronize();
    printf("nblocks %d\n", hashTable.nKeys);
    paint();
  }
}

real CompressibleSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;

  while (t < tStep) {
    computeDeltaT();
    cudaDeviceSynchronize();
    t += deltaT;
    iter++;
  }

  return t;
}

void CompressibleSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::setInitialConditions(void) {
  setInitialConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(void) {
  setBoundaryConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::forwardWaveletTransform(void) {
  cudaDeviceSynchronize();
  computeMagUKernel<<<cudaGridSize, cudaBlockSize>>>(*this); 
  maxRho = *(thrust::max_element(thrust::device, getField(0), getField(0)+hashTable.nKeys*blockSize));
  maxMagU = *(thrust::max_element(thrust::device, getField(10), getField(10)+hashTable.nKeys*blockSize));
  maxP = *(thrust::max_element(thrust::device, getField(4), getField(4)+hashTable.nKeys*blockSize));

  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this); 
  forwardWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this);

  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  waveletThresholdingKernel<<<cudaGridSize, cudaBlockSize>>>(*this); 
}

void CompressibleSolver::inverseWaveletTransform(void) {
  inverseWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this); 
}


void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
	deltaT = *(thrust::min_element(thrust::device, getField(12), getField(12)+hashTable.nKeys*blockSizeTot));
  deltaT *= cfl;
}

void CompressibleSolver::computeRightHandSide(void) {
  computeRightHandSideKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  updateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage);
}

void CompressibleSolver::restrictFields(void) {
  restrictFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::interpolateFields(void) {
  interpolateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

__device__ real CompressibleSolver::getBoundaryLevelSet(Vec3 pos) {

  if (immerserdBcType == 1) {
    // circle
    real radius = .05;
    real center[2] = {.5, .5};
    return radius - sqrt((pos[0]-center[0])*(pos[0]-center[0]) + (pos[1]-center[1])*(pos[1]-center[1]));
  }
  else {
    return 1e32;
  }

} 

__device__ real CompressibleSolver::calcIbMask(real phi) {
  real dx = min(getDx(nLvls-1), getDy(nLvls-1));
  real eps = .5;
  return (.5 * (1 + tanh(phi / (2 * eps * dx))));
}