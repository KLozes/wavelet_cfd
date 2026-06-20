#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"

//
// Usage:  ./wave3d [testCase] [nLvls] [nBlocksX]
//
//   testCase 0 : pseudo-2D / quasi-1D Sod shock tube (validated vs exact)
//   testCase 1 : 2D circular Sod explosion with adaptive mesh refinement
//
// Both run the full 3D solver with a single block in z (pseudo-2D mode).
//
int main(int argc, char* argv[]) {

  i32 testCase = (argc > 1) ? atoi(argv[1]) : 0;
  i32 nLvls    = (argc > 2) ? atoi(argv[2]) : (testCase == 1 ? 4 : 1);
  i32 nBlocksX = (argc > 3) ? atoi(argv[3]) : (testCase == 1 ? 16 : 100);
  real wThresh = (argc > 4) ? atof(argv[4]) : (testCase == 1 ? 0.004 : 0.01);

  // base grid in blocks; one block thick in z for pseudo-2D
  i32 nBlocksY = (testCase == 1) ? nBlocksX : (nBlocksX + 9) / 10;

  // cubic cells: dx = dy = dz = 1/(nBlocksX*blockSize)
  real dx = 1.0 / (nBlocksX*blockSize);
  real domainSize[3]   = {1.0, dx*(nBlocksY*blockSize), dx*blockSize};
  i32  baseGridSize[3] = {blockSize*nBlocksX, blockSize*nBlocksY, blockSize*1};

  real cfl  = 0.40;
  real tEnd  = 0.20;
  real tStep = (testCase == 1) ? 0.008 : 0.01;

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->pseudo2D        = (baseGridSize[2] == blockSize) ? 1 : 0;  // collapse z
  solver->cfl             = cfl;
  solver->waveletThresh   = wThresh;
  solver->icType          = (testCase == 1) ? 1 : 0;
  solver->bcType          = (testCase == 1) ? 3 : 3;   // transmissive / outflow
  solver->immerserdBcType = 0;
  solver->initialize();

  real t = 0;
  auto wall0 = std::chrono::steady_clock::now();
  while (t < tEnd) {
    t += solver->step(tStep);
    solver->paint();
    real comp = 100.0 * real(solver->hashTable.nKeys) /
                real(baseGridSize[0]*baseGridSize[1]*baseGridSize[2]/blockSizeTot*powi(powi(2,nLvls-1),2));
    printf("n: %d, t = %f, nblocks = %d, dt = %e, grid = %.1f%% of uniform-fine\n",
           solver->imageCounter, t, solver->hashTable.nKeys, solver->deltaT, comp);
  }
  auto wall1 = std::chrono::steady_clock::now();
  double wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();
  printf("done: t = %f after %d iters in %.1f ms\n", t, solver->iter, wallMs);

  if (testCase == 0) {
    solver->writeLineProfile("output/sod_profile.dat");
  }
  solver->printDiagnostics();
  solver->paintPressure("output/pressure_final.png");

  cudaDeviceSynchronize();
  delete solver;
  cudaDeviceReset();
}
