#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"
//#include "Kernels.cuh"
//#include "png.hpp"

int main(int argc, char* argv[]) {
  u32 baseSize[2] = {blockSize*4, blockSize*6};
  u32 nLvls = 1;

  CompressibleSolver *solver = new CompressibleSolver(baseSize, nLvls);
  cudaDeviceSynchronize();
	cudaDeviceReset();
}
