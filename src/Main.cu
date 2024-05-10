#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*4, blockSize*4};
  u32 nLvls = 7;
  dataType cfl = .3;
  dataType waveletThresh = .01;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls, cfl, waveletThresh);
  solver->initializeBaseGrid();
  solver->setInitialConditions(0);
  solver->setBoundaryConditions(0);
  solver->paint();

  for(i32 lvl=0; lvl<nLvls; lvl++){
    solver->waveletThresholding();
    solver->adaptGrid();
    solver->setInitialConditions(0);
    solver->setBoundaryConditions(0);
    solver->paint();
    printf("nBlocks %d\n" , solver->nBlocks);
  }


  /*
  dataType t = 0;
  i32 n = 0;
  while(t < 100) {

    if (n % 1 == 0) {
      solver->computeDeltaT();
    }

    for (i32 stage = 0; stage<3; stage++) {
      solver->computeRightHandSide();
      solver->primitiveToConservative();
      solver->updateFields(stage);
      solver->conservativeToPrimitive();
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
  */
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
