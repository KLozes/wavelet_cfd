#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"
//#include "Kernels.cuh"
//#include "png.hpp"

int main(int argc, char* argv[]) {
  u32 baseSize[2] = {blockSize*40, blockSize*40};
  u32 nLvlsMax = 1;

  CompressibleSolver *solver = new CompressibleSolver(baseSize, nLvlsMax);
  //solver->initGrid();

	cudaDeviceReset();
}
