#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 2.0};
  u32 baseGridSize[2] = {blockSize*100, blockSize*2*100};
  u32 nLvls = 1;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls, .3);
  solver->initializeBaseGrid();
  solver->setInitialConditions(0);
  solver->setBoundaryConditions(0);
  solver->paint();

  dataType t = 0;
  i32 n = 0;
  while(t < 100) {

    if (n % 1 == 0) {
      solver->computeDeltaT();
    }

    for (i32 stage = 0; stage<3; stage++) {
      solver->computeRightHandSide();
      solver->updateFields(stage);
      solver->setBoundaryConditions(0);
    }
    cudaDeviceSynchronize();
    t += solver->deltaT;
    n++;

    if (n % 50 == 0) {
      printf("n: %d, t = %f\n", n, t);
      solver->paint();
    }

  }

  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
