#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*10, blockSize*10};
  u32 nLvls = 1;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->initializeBaseGrid();
  solver->setInitialConditions(0);
  solver->paint();

  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
