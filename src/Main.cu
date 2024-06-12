#include <string.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  real domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*10, blockSize*10};
  u32 nLvls = 8;
  real cfl = .80;
  real waveletThresh = .005;
  real tStep = .02;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls, cfl, waveletThresh);
  solver->icType = 0;
  solver->bcType = 0;
  solver->immerserdBcType = 0;
  solver->initialize();

  real t = 0;
  i32 n = 0;
  while(t < 100) {

    t += solver->step(tStep);
    n += 1;

    solver->paint();
    printf("n: %d, t = %f, tSolver = %d, tGrid = %d, nBlocks = %d\n", n, t, solver->tSolver , solver->tGrid, solver->hashTable.nKeys);

  }
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
