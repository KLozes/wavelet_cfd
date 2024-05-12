#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*25, blockSize*25};
  u32 nLvls = 3;
  dataType cfl = .1;
  dataType waveletThresh = .002;
  dataType tStep = .001;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls, cfl, waveletThresh);
  solver->initialize(0);

  dataType t = 0;
  i32 n = 0;
  while(t < 100) {

    t += solver->step(tStep);
    n += 1;

    //solver->paint();
    printf("n: %d, t = %f\n", n, t);

  }
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
