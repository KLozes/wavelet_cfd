#include <string.h>
#include <iostream>
#include <ctime>

#include "CompressibleSolver.cuh"

int main(int argc, char* argv[]) {
  dataType domainSize[2] = {1.0, 1.0};
  u32 baseGridSize[2] = {blockSize*200, blockSize*200};
  u32 nLvls = 1;
  dataType cfl = .3;
  dataType waveletThresh = .01;
  dataType tStep = .02;

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


  dataType t = 0;
  i32 n = 0;
  while(t < 100) {

    t += solver->step(tStep);
    n += 1;

    solver->paint();
    printf("n: %d, t = %f\n", n, t);

  }
  
  cudaDeviceSynchronize();
  delete solver;
	cudaDeviceReset();
}
