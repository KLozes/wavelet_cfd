#include <iostream>
#include <thrust/extrema.h>


#include "StgdLowMachSolver.cuh"
#include "StgdLowMachSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"


void StgdLowMachSolver::initialize(void) {
  initializeBaseGrid();
  setInitialConditions();
  sortBlocks();
  setBoundaryConditions();
  cudaDeviceSynchronize();
  printf("nblocks %d\n", hashTable.nKeys);
  paint();

  for(i32 lvl=0; lvl<nLvls; lvl++){
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

real StgdLowMachSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;

  while (t < tStep) {
    cudaDeviceSynchronize();
    t += deltaT;
    iter++;
  }

  return t;
}

void StgdLowMachSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<1000, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<1000, cudaBlockSize>>>(*this);
}

void StgdLowMachSolver::setInitialConditions(void) {
  setInitialConditionsKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void StgdLowMachSolver::setBoundaryConditions(void) {
  setBoundaryConditionsKernel<<<1000, cudaBlockSize>>>(*this);
}

void StgdLowMachSolver::forwardWaveletTransform(void) {
  cudaDeviceSynchronize();
  computeMagRhoUKernel<<<1000, cudaBlockSize>>>(*this); 
  maxP = *(thrust::max_element(thrust::device, getField(0), getField(0)+hashTable.nKeys*blockSize));
  maxMagU = *(thrust::max_element(thrust::device, getField(8), getField(8)+hashTable.nKeys*blockSize));
  cudaDeviceSynchronize();
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  copyToOldFieldsKernel<<<1000, cudaBlockSize>>>(*this); 
  forwardWaveletTransformKernel<<<1000, cudaBlockSize>>>(*this);
  waveletThresholdingKernel<<<1000, cudaBlockSize>>>(*this); 
}

void StgdLowMachSolver::inverseWaveletTransform(void) {
  inverseWaveletTransformKernel<<<1000, cudaBlockSize>>>(*this); 
}


void StgdLowMachSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
	deltaT = *(thrust::min_element(thrust::device, getField(12), getField(12)+hashTable.nKeys*blockSizeTot));
  deltaT *= cfl;
}

void StgdLowMachSolver::computeRightHandSide(void) {
  computeRightHandSideKernel<<<1000, cudaBlockSize>>>(*this);
}

void StgdLowMachSolver::updateFields(i32 stage) {
  updateFieldsKernel<<<1000, cudaBlockSize>>>(*this, stage);
}

void StgdLowMachSolver::restrictFields(void) {
  restrictFieldsKernel<<<1000, cudaBlockSize>>>(*this);
}

void StgdLowMachSolver::interpolateFields(void) {
  interpolateFieldsKernel<<<1000, cudaBlockSize>>>(*this);
}

__device__ real StgdLowMachSolver::getBoundaryLevelSet(Vec3 pos) {

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

__device__ real StgdLowMachSolver::calcIbMask(real phi) {
  real dx = min(getDx(nLvls-1), getDy(nLvls-1));
  real eps = .5;
  return (.5 * (1 + tanh(phi / (2 * eps * dx))));
}