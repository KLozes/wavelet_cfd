#include <string.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  real domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*10, blockSize*10};
  u32 nLvls = 7;
  real cfl = .80;
  real waveletThresh = .002;
  real tStep = .02;

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
    real tTotal = solver->tSolver + solver->tGrid;
    real tSolver = real(solver->tSolver) / tTotal;
    real tGrid = real(solver->tGrid) / tTotal;
    real comp = real(solver->hashTable.nKeys) / ((baseGridSize[0])*(baseGridSize[1])*powi(2,nLvls-1)*powi(2,nLvls-1)/16);
    printf("n: %d, t = %f, tSolver = %f, tGrid = %f, compression = %f\n", solver->imageCounter, t, tSolver , tGrid, comp);

  }
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
