#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"

//
// Usage:  ./wave3d [testCase] [nLvls] [nBlocksX] [wThresh] [scheme]
//
//   testCase 0 : pseudo-2D / quasi-1D Sod shock tube (validated vs exact)
//   testCase 1 : 2D circular Sod explosion with adaptive mesh refinement
//   testCase 2 : isentropic vortex on [0,10]^2 (RT0/P0 DG validation, pseudo-2D)
//   testCase 3 : true-3D spherical Sod explosion (exercises the z / Gz paths)
//   testCase 4 : Gresho vortex on [0,1]^2, low-Mach (RT0 low-Mach preservation)
//   testCase 5 : Gresho vortex on a STATIC radial AMR grid (fine at centre,
//                coarse outward) — tests flow crossing fixed coarse/fine faces
//
//   scheme (arg 5) : 0 = finite volume (HLLC+TVD), 1 = RT0/P0 DG
//                    (defaults to 1 for testCase >= 2, else 0)
//   Ma     (arg 6) : Gresho Mach number (testCase 4 only; default 0.1)
//
// testCases 0-2,4 run pseudo-2D (single z block); testCase 3 is fully 3D.
//
int main(int argc, char* argv[]) {

  i32 testCase = (argc > 1) ? atoi(argv[1]) : 0;
  bool gresho  = (testCase == 4 || testCase == 5);         // Gresho vortex (uniform / static-AMR)
  i32 nLvls    = (argc > 2) ? atoi(argv[2]) : (testCase == 1 ? 4 : (testCase == 5 ? 3 : 1));
  i32 nBlocksX = (argc > 3) ? atoi(argv[3]) : (testCase == 1 ? 16 : (testCase == 2 ? 40 : (testCase == 3 ? 8 : (testCase == 4 ? 40 : (testCase == 5 ? 10 : 100)))));
  real wThresh = (argc > 4) ? atof(argv[4]) : (testCase == 1 ? 0.004 : 0.01);
  i32 scheme   = (argc > 5) ? atoi(argv[5]) : (testCase >= 2 ? 1 : 0);
  real Ma      = (argc > 6) ? atof(argv[6]) : 0.1;   // Gresho Mach number
  i32 bcArg    = (argc > 7) ? atoi(argv[7]) : -1;    // bcType override (-1 = per-testCase default)
  i32 refluxA  = (argc > 8) ? atoi(argv[8]) : 0;     // 1 = conservative coarse/fine refluxing
  i32 basisG   = (argc > 9) ? atoi(argv[9]) : 2;     // c/f ghost fill: 0=DD, 1=RT0/P0-basis, 2=monotone trilinear (default)

  bool cube   = (testCase == 3);
  bool square = (testCase == 1 || testCase == 2 || gresho);
  i32 nBlocksY = (square || cube) ? nBlocksX : (nBlocksX + 9) / 10;
  i32 nBlocksZ = cube ? nBlocksX : 1;

  // cubic cells; domain length in x is 10 for the isentropic vortex, 1 otherwise
  real domainLenX = (testCase == 2) ? 10.0 : 1.0;
  real dx = domainLenX / (nBlocksX*blockSize);
  real domainSize[3]   = {domainLenX, dx*(nBlocksY*blockSize), dx*(nBlocksZ*blockSize)};
  i32  baseGridSize[3] = {blockSize*nBlocksX, blockSize*nBlocksY, blockSize*nBlocksZ};

  real cfl  = 0.40;
  real tEnd  = (testCase == 2) ? 1.0 : (testCase == 3 ? 0.15 : (gresho ? 1.0 : 0.20));
  real tStep = (testCase == 1) ? 0.008 : (testCase == 2 ? 0.1 : (testCase == 3 ? 0.03 : (gresho ? 0.1 : 0.01)));

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->pseudo2D        = (baseGridSize[2] == blockSize) ? 1 : 0;  // collapse z (pseudo-2D)
  solver->cfl             = cfl;
  solver->waveletThresh   = wThresh;
  solver->scheme          = scheme;
  solver->icType          = (testCase == 1) ? 1 : (testCase == 2 ? 2 : (testCase == 3 ? 3 : (gresho ? 4 : 0)));
  solver->bcType          = (bcArg >= 0) ? bcArg : 3;   // transmissive / outflow (default), or CLI override
  solver->vortexAdvect    = 0.0;                        // stationary vortex
  solver->greshoP0        = 1.0/(gam*Ma*Ma);            // Gresho background pressure -> Mach = Ma
  solver->staticGrid      = (testCase == 5) ? 1 : 0;    // fixed radial refinement about the vortex
  solver->refineRadius    = 0.4;                        // vortex extent
  solver->reflux          = refluxA;                    // conservative coarse/fine flux correction
  solver->basisGhost      = basisG;                     // RT0/P0-basis coarse/fine ghost fill
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
  if (testCase == 2) {
    solver->computeVortexError();
  }
  if (gresho) {
    solver->computeGreshoError();
  }
  solver->printDiagnostics();
  solver->paintPressure("output/pressure_final.png");

  cudaDeviceSynchronize();
  delete solver;
  cudaDeviceReset();
}
