#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"
//#include "Kernels.cuh"
//#include "png.hpp"

int main(int argc, char* argv[]) {
  uint baseSize[2] = {blockSize*40, blockSize*40};
  uint nLvlsMax = 1;

  CompressibleSolver *solver = new CompressibleSolver(baseSize, nLvlsMax);
  solver->initGrid();

	cudaDeviceReset();
}
