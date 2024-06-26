#include <string.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  real domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*10, blockSize*10};
  u32 nLvls = 7;
  real cfl = .80;
  real waveletThresh = .0025;
  real tStep = .01;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->cfl = cfl;
  solver->waveletThresh = waveletThresh;
  solver->icType = 0;
  solver->bcType = 0;
  solver->immerserdBcType = 0;
  solver->initialize();

  real t = 0;
  while(t < 100) {

    t += solver->step(tStep);

    solver->paint();
    printf("n: %d, t = %f, tSolver = %d, tGrid = %d, nBlocks = %d\n", solver->imageCounter, t, solver->tSolver , solver->tGrid, solver->hashTable.nKeys);

  }
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
