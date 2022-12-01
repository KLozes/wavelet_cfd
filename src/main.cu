#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"
//#include "Kernels.cuh"
//#include "png.hpp"

int main(int argc, char* argv[]) {
  u32 baseSize[nDim] = {24, 28};
  u32 nLvls = 1;

  CompressibleSolver *solver = new CompressibleSolver(baseSize, nLvls);

  delete solver;
	cudaDeviceReset();
}
