#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#include "CompressibleSolver.cuh"
#include "Comm.cuh"

//
// Usage:  ./wave3d --case N [--flag value ...]        (all flags optional)
//
//   testCase 0 : pseudo-2D / quasi-1D Sod shock tube (validated vs exact)
//   testCase 1 : 2D circular Sod explosion with adaptive mesh refinement
//   testCase 2 : isentropic vortex on [0,10]^2 (RT0/P0 DG validation, pseudo-2D)
//   testCase 3 : true-3D spherical Sod explosion (exercises the z / Gz paths)
//   testCase 4 : Gresho vortex on [0,1]^2, low-Mach (RT0 low-Mach preservation)
//   testCase 5 : Gresho vortex on a STATIC radial AMR grid (fine at centre,
//                coarse outward) — tests flow crossing fixed coarse/fine faces
//   testCase 6 : circular Sod explosion on a static radial AMR grid — measures
//                coarse/fine conservation error (mass/energy drift)
//
//   --case N     test case (default 0); also accepted as the first bare arg
//   --nlvls N    number of refinement levels           (per-case default)
//   --nblocks N  base blocks in x                       (per-case default)
//   --wthresh X  wavelet detail threshold               (per-case default)
//   --scheme N   0 = finite volume (HLLC+TVD), 1 = RT0/P0 DG  (>=2 -> 1)
//   --ma X       Gresho Mach / acConv amplitude / testCase-1 inner pressure
//   --bc N       BC: 0 slip wall, 1 no-slip, 2 periodic, 3 transmissive
//   --recon N    0=TVD, 1=ROUND (default), 2=LD-ROUND, 3=unlimited parabola
//   --tend X     end time override                      (per-case default)
//   --rt0face N  RT0 normal face (scheme 1): 0 = linear modal (default),
//                1 = c=1/6 biased parabola (4th-order face average)
//   --mdflux N   1 = genuinely multidimensional Osher-type corner flux
//                (Gaburro-Ricchiuto-Dumbser, arXiv:2506.00207); 2 = CTU-Hancock:
//                half-step predictor + single-Euler corrector, 2nd order in
//                time, single flux sweep per step.  Stable CFL: FV ~1.2,
//                RT0 ~0.35 (the slope DOFs are near-imaginary modes and the
//                midpoint-class corrector leaks above that; RK3 for margin).
//                pseudo-2D only; states follow --recon/--rt0face
//   --cfl X      CFL number (default 0.40; dt = cfl * min(dx/(|u|+c)))
//
// Back-compat: `./wave3d N` (bare first arg) still selects the test case.
// testCases 0-2,4 run pseudo-2D (single z block); testCase 3 is fully 3D.
//
// per-rank body (run once per PE; loopback runs it on one thread per logical PE)
static void runRank(int argc, char* argv[]) {

  // --flag value parser: findArg returns the token after --flag, or nullptr.
  auto findArg = [&](const char* key) -> const char* {
    for (int a = 1; a < argc - 1; a++)
      if (strcmp(argv[a], key) == 0) return argv[a+1];
    return nullptr;
  };
  auto hasArg = [&](const char* key) { return findArg(key) != nullptr; };
  auto argI   = [&](const char* key, i32 def)    { const char* v = findArg(key); return v ? atoi(v) : def; };
  auto argF   = [&](const char* key, real def)   { const char* v = findArg(key); return v ? (real)atof(v) : def; };

  // guard against the old positional style (silently ignored now): warn if more
  // than one bare arg is passed and no --flag is present.
  bool anyNamed = false;
  for (int a = 1; a < argc; a++) if (strncmp(argv[a], "--", 2) == 0) anyNamed = true;
  if (!anyNamed && argc > 2)
    printf("[warn] positional args are deprecated and ignored; use named flags, "
           "e.g. --case %s --nlvls %s ...  (run with --case only, or -h, for the list)\n",
           argv[1], argv[2]);

  // testCase: --case N, or the first bare (non---) argument for back-compat
  i32 testCase = argI("--case", (argc > 1 && argv[1][0] != '-') ? atoi(argv[1]) : 0);
  bool gresho  = (testCase == 4 || testCase == 5);         // Gresho vortex (uniform / static-AMR)
  bool sodAmr  = (testCase == 6);                          // Sod shock on a static planar AMR grid
  bool acoustic= (testCase == 7);                          // acoustic pulse crossing a static coarse/fine interface
  bool acConv  = (testCase == 8);                          // periodic sine acoustic wave, order-of-accuracy study
  i32 nLvls    = argI("--nlvls", (testCase == 1 ? 4 : (testCase == 5 ? 3 : (sodAmr ? 3 : (acoustic ? 3 : 1)))));
  i32 nBlocksX = argI("--nblocks", (testCase == 1 ? 16 : (testCase == 2 ? 40 : (testCase == 3 ? 8 : (testCase == 4 ? 40 : (testCase == 5 ? 10 : (sodAmr ? 16 : (acoustic ? 64 : (acConv ? 8 : 100)))))))));
  real wThresh = argF("--wthresh", (testCase == 1 ? 0.004 : 0.01));
  i32 scheme   = argI("--scheme", (sodAmr ? 0 : (testCase >= 2 ? 1 : 0)));
  bool haveMa  = hasArg("--ma");
  real Ma      = haveMa ? argF("--ma", 0.1) : 0.1;   // Gresho Mach number / acConv amplitude
  i32 bcArg    = argI("--bc", -1);                   // bcType override (-1 = per-testCase default)
  i32 reconA   = argI("--recon", 1);                 // 0=TVD, 1=ROUND (default), 2=LD-ROUND, 3=unlimited parabola (smooth only)
  real tEndArg = argF("--tend", -1.0);               // tEnd override (-1 = per-testCase default)
  i32 rt0FaceA = argI("--rt0face", 0);               // RT0 normal face (scheme==1): 0=linear modal (default), 1=c=1/6 parabola
  i32 mdFluxA  = argI("--mdflux", 0);                // 1 = multidimensional Osher-type corner flux (first-order states)
  real cflArg  = argF("--cfl", -1.0);                // CFL override (-1 = default 0.40; dt = cfl*min(dx/(|u|+c)))
  real advectA = argF("--advect", 0.0);              // isentropic-vortex (case 2) advection velocity u0=v0 (periodic seam-crossing test)

  bool cube   = (testCase == 3);
  bool square = (testCase == 1 || testCase == 2 || gresho || sodAmr || acoustic || acConv);
  i32 nBlocksY = (square || cube) ? nBlocksX : (nBlocksX + 9) / 10;
  i32 nBlocksZ = cube ? nBlocksX : 1;

  // cubic cells; domain length in x is 10 for the isentropic vortex, 1 otherwise
  real domainLenX = (testCase == 2) ? 10.0 : 1.0;
  real dx = domainLenX / (nBlocksX*blockSize);
  real domainSize[3]   = {domainLenX, dx*(nBlocksY*blockSize), dx*(nBlocksZ*blockSize)};
  i32  baseGridSize[3] = {blockSize*nBlocksX, blockSize*nBlocksY, blockSize*nBlocksZ};

  // acConv: RK3's O(dt^3) global error would cap an order study at 3; scaling
  // cfl ~ dx^(1/3) keeps dt^3 ~ dx^4, below 4th-order spatial error.
  real cfl  = acConv ? 0.40*cbrt(4.0/nBlocksX) : 0.40;
  if (cflArg > 0) cfl = cflArg;                      // CLI override
  // testCase 1: --ma sets the circular-Sod inner pressure (1.0 = classic 10:1
  // ratio; 10 = strong 100:1 blast).  The strong blast's faster shock needs a
  // shorter tEnd to stay inside the domain.
  real sodPin = (testCase == 1) ? (haveMa ? argF("--ma", 1.0) : 1.0) : 0.0;
  real acPeriod = domainLenX / sqrt(gam);   // sound-crossing time (c0=sqrt(gam), p0=rho0=1)
  real tEnd  = (testCase == 1 && sodPin > 2.0) ? 0.06*sqrt(10.0/sodPin) : ((testCase == 2) ? 1.0 : (testCase == 3 ? 0.15 : (gresho ? 1.0 : (sodAmr ? 0.15 : (acoustic ? 0.35 : (acConv ? 2.0*acPeriod : 0.20))))));
  if (tEndArg > 0) tEnd = tEndArg;                   // CLI override (arg 10)
  real tStep = (testCase == 1) ? ((tEndArg > 0) ? tEnd/50.0 : ((sodPin > 2.0) ? 0.06*sqrt(10.0/sodPin)/10.0 : 0.008)) : (testCase == 2 ? 0.1 : (testCase == 3 ? 0.03 : (gresho ? 0.1 : (sodAmr ? 0.02 : (acoustic ? 0.02 : (acConv ? tEnd : 0.01))))));

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->pseudo2D        = (baseGridSize[2] == blockSize) ? 1 : 0;  // collapse z (pseudo-2D)
  solver->cfl             = cfl;
  solver->waveletThresh   = wThresh;
  solver->scheme          = scheme;
  solver->icType          = (testCase == 1 || sodAmr) ? 1 : (testCase == 2 ? 2 : (testCase == 3 ? 3 : (gresho ? 4 : (acoustic ? 5 : (acConv ? 6 : 0)))));
  solver->bcType          = (bcArg >= 0) ? bcArg : ((acConv || testCase == 1) ? 2 : 3);   // periodic for the acoustic wave and circular Sod; else transmissive
  solver->vortexAdvect    = acConv ? Ma : (testCase == 2 ? advectA : sodPin);  // acConv: wave amplitude A; case 2: vortex advection; testCase 1: Sod inner pressure
  solver->greshoP0        = 1.0/(gam*Ma*Ma);            // Gresho background pressure -> Mach = Ma
  solver->staticGrid      = acoustic ? 3 : ((testCase == 5 || sodAmr) ? 1 : 0);   // 1=radial shells, 2=planar band, 3=centre step
  solver->refineRadius    = 0.4;                        // fine-region half-extent (unused for the step)
  solver->recon           = reconA;                     // face reconstruction (TVD / ROUND / LD-ROUND)
  solver->rt0Face         = rt0FaceA;                    // RT0 normal face: 0=linear modal, 1=c=1/6 parabola
  solver->mdFlux          = mdFluxA;                     // multidimensional Osher-type corner flux
  solver->immerserdBcType = 0;
  solver->initialize();

  // Circular-Sod-on-static-radial-AMR conservation test: the cylindrical shock
  // expands from the centre through the radial coarse/fine interface.  Mass and
  // energy are exactly conserved until the shock reaches the (still) boundaries, so
  // their drift is purely the coarse/fine interface flux mismatch.
  double m0 = 0, px0 = 0, e0 = 0;
  if (sodAmr) {
    solver->totalConserved(m0, px0, e0);
    printf("[cons] initial mass=%.12e  energy=%.12e\n", m0, e0);
  }

  real t = 0;
  auto wall0 = std::chrono::steady_clock::now();
  while (t < tEnd) {
    t += solver->step(tStep);
    solver->paint();
    real comp = 100.0 * real(solver->hashTable.nKeys) /
                real(baseGridSize[0]*baseGridSize[1]*baseGridSize[2]/blockSizeTot*powi(powi(2,nLvls-1),2));
    printf("n: %d, t = %f, nblocks = %d, dt = %e, grid = %.1f%% of uniform-fine\n",
           solver->imageCounter, t, solver->hashTable.nKeys, solver->deltaT, comp);
    if (sodAmr) {
      double m, px, e; solver->totalConserved(m, px, e);
      printf("[cons] t=%.3f  dMass/M0=%+.3e  dEnergy/E0=%+.3e\n", t, (m-m0)/m0, (e-e0)/e0);
    }
  }
  auto wall1 = std::chrono::steady_clock::now();
  double wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();
  printf("done: t = %f after %d iters in %.1f ms\n", t, solver->iter, wallMs);
  // adaptation (tGrid: restrict+forward/inverse wavelet+adaptGrid+sortBlocks, every
  // 4th iter) vs solver (tSolver: deltaT + 3 RK stages, every iter)
  double tG = solver->tGrid, tS = solver->tSolver;
  printf("[timing] adaptation = %.0f ms, solver = %.0f ms  ->  adaptation is %.1f%% of step() time\n",
         tG, tS, (tG + tS > 0) ? 100.0*tG/(tG + tS) : 0.0);
  double tFwd = solver->tForwardUs/1000.0, tSrt = solver->tSortUs/1000.0;
  printf("[adapt breakdown] forwardWavelet(6 reductions) = %.0f ms, sortBlocks = %.0f ms, rest = %.0f ms\n",
         tFwd, tSrt, tG - tFwd - tSrt);

  if (testCase == 0) {
    solver->writeLineProfile("output/sod_profile.dat");
  }
  if (testCase == 2) {
    solver->computeVortexError();
  }
  if (gresho) {
    solver->computeGreshoError();
  }
  if (acoustic) {
    solver->computeAcousticReflection("output/acoustic_profile.dat");
  }
  if (acConv) {
    solver->computeAcousticL2Error();
  }
  solver->printDiagnostics();
  solver->paintPressure("output/pressure_final.png");

  cudaDeviceSynchronize();
  delete solver;
}

int main(int argc, char* argv[]) {
  comm::init(&argc, &argv);        // SPMD bring-up: parse --np, set rank/size, pick GPU
  comm::run(argc, argv, runRank);  // run the per-rank body (P threads under loopback)
  comm::finalize();
  cudaDeviceReset();
}
