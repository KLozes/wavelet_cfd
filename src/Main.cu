#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*100, blockSize*100};
  u32 nLvls = 1;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->initGrid();
  solver->initFieldData(0);
  solver->paint();

  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
