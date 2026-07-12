#include <string.h>
#include <stdlib.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>

#include "DgSolver.cuh"

//
// Usage:  ./wavedg3d --case N [--flag value ...]
//
//   testCase 0 : quasi-1D Sod shock tube (pseudo-2D), adaptive     [M1/M5]
//   testCase 1 : circular Sod blast (pseudo-2D, walls), adaptive -- the dgsem
//                reference case (rho 11/0.125, p 10/0.1, r=0.25)   [M1/M5]
//   testCase 2 : isentropic vortex on [0,10]^2 (pseudo-2D, periodic),
//                adaptive; --advect for the advecting variant      [M4]
//   testCase 3 : uniform free stream on a STATIC two-level grid
//                (center sphere) -- nonconforming free-stream test [M3]
//   testCase 4 : spherical Sod blast, true 3D, adaptive            [M5]
//   testCase 5 : Gaussian density pulse advecting diagonally
//                (periodic), adaptive                              [M4]
//   testCase 6 : double Mach reflection on [0,4]x[0,1] (pseudo-2D) [M6]
//   testCase 7 : advecting vortex across a STATIC planar coarse/fine
//                interface (periodic)                              [M3]
//   testCase 8 : Gresho vortex on [0,1]^2 (pseudo-2D), low-Mach; --ma
//   testCase 9 : supersonic cylinder in cross flow (immersed boundary,
//                pseudo-2D, 6x4 domain, D=1 cylinder at (1.5,2)); --mach
//
//   --nlvls N      refinement levels             (per-case default)
//   --nblocks N    base elements in x            (per-case default)
//   --cfl X        CFL number                    (default 0.4)
//   --tend X       end time                      (per-case default)
//   --eps X        MRA threshold eps_L override  (default: cthr * h_L^gamma)
//   --cthr X       threshold constant C_thr      (default 1)
//   --gammathr X   threshold exponent gamma      (default 1)
//   --refinefac X  refine at refinefac*eps       (default 2; paper's 2^(p+1)
//                  under-refines shocks under leaf-only evolution)
//   --adaptevery N adaptation cadence in steps   (default 4; must stay < NN/cfl = 10)
//   --refinebuffer N 1 = also refine the neighbor ring of every refined element
//   --shockrefine N  1 = force shocked elements (Ducros sensor) to the finest level
//   --shockthresh X  sensor threshold for shock-driven refinement (default 0.5)
//   --ecflux N     1 = EC flux-differencing volume (default), 0 = collocation
//   --av N         artificial viscosity on/off   (default 1)
//   --avcav X      AV strength C_av              (default 0.25)
//   --avpen X      interface jump-penalty scale (BR2-lite; 0 = off)
//   --sensor N     0 = Ducros, 1 = Persson modal, 2 = max of both (default)
//   --pps0 X       Persson ramp center in log10(S)   (default -4*log10(p))
//   --ppkappa X    Persson ramp half-width           (default 2)
//   --bc N         0 slip wall, 2 periodic, 3 transmissive, 4 DMR
//   --scalemode N  0 = |domain mean| scales (paper), 1 = domain max
//   --indicator N  0 = MRA detail (default), 1 = smoothness-sensor vote
//   --pprefine X   indicator 1: refine above this sensor theta   (default 0.5)
//   --ppcoarsen X  indicator 1: coarsen below this sensor theta  (default 0.01)
//   --static N     static-grid vote override 0..4 (see DgSolver.cuh)
//   --radius X     static fine-region half-extent (default 0.25)
//   --advect X     vortex advection velocity u0=v0
//   --uniform N    force nlvls=1-style uniform run at level N... (unused)
//   --sortcurve N  1 = sort blocks along a Hilbert/Morton curve (memory locality)
//   --ib N         1 = ghost-element immersed boundary (default on for case 9)
//   --ibx/--iby X  cylinder center            (default 1.5, domain mid-height)
//   --ibr X        cylinder radius            (default 0.5)
//   --mach X       freestream Mach, a_inf = 1 (default 3; 0 = rest-state gate)
//   --ibband X     force-finest band, finest-element units    (default 3)
//   --ibcurv N     1 = curvature wall-pressure dp/dn = rho vt^2/R (default 1)
//   --ibord N      reconstruction order 1/2 (3 = unstable)    (default 2)
//   --ibimagefac X image-distance floor, ghost-widths         (default 1.5)
//   --ibtheta X    donor sensor theta forcing order-2 fallback (default 0.5)
//   --ibpen X      ghost-face penalty fraction of lambda      (default 0)
//   --ibgraze X    extra ghost margin, element units (default 0: center rule)
//   --debug N      1 = leaf-cover/integrity checks each adaptation
//   --selftest     run the operator self-test and exit
//
int main(int argc, char* argv[]) {

  auto findArg = [&](const char* key) -> const char* {
    for (int a = 1; a < argc - 1; a++)
      if (strcmp(argv[a], key) == 0) return argv[a+1];
    return nullptr;
  };
  auto hasFlag = [&](const char* key) {
    for (int a = 1; a < argc; a++) if (strcmp(argv[a], key) == 0) return true;
    return false;
  };
  auto argI = [&](const char* key, i32 def)  { const char* v = findArg(key); return v ? atoi(v) : def; };
  auto argF = [&](const char* key, real def) { const char* v = findArg(key); return v ? (real)atof(v) : def; };

  i32 testCase = argI("--case", (argc > 1 && argv[1][0] != '-') ? atoi(argv[1]) : 0);

  bool square = (testCase == 1 || testCase == 2 || testCase == 3 || testCase == 5 || testCase == 7 || testCase == 8);
  bool cube   = (testCase == 4);
  bool dmr    = (testCase == 6);
  bool ibcyl  = (testCase == 9);   // supersonic cylinder: 6 x 4 rectangle

  // base grid in ELEMENTS (= blocks); each element carries 4^3 LGL nodes.
  // Even counts per axis keep the level-0 virtual octets complete.
  i32 nElemX = argI("--nblocks", testCase == 0 ? 32 :
                                 (testCase == 1 ? 16 :
                                 (testCase == 2 || testCase == 7 ? 10 :
                                 (testCase == 3 ? 8 :
                                 (testCase == 4 ? 8 :
                                 (testCase == 5 ? 8 :
                                 (testCase == 8 ? 40 :
                                 (testCase == 9 ? 24 : 16))))))));
  i32 nElemY = cube ? nElemX : (square ? nElemX : (dmr ? nElemX/4 :
               (ibcyl ? (nElemX*2)/3 : 4)));
  i32 nElemZ = cube ? nElemX : 1;

  i32 nLvls = argI("--nlvls", testCase == 0 ? 3 :
                              (testCase == 1 ? 4 :
                              (testCase == 2 ? 3 :
                              (testCase == 3 || testCase == 7 ? 2 :
                              (testCase == 4 ? 3 :
                              (testCase == 5 ? 3 :
                              (testCase == 8 ? 1 :
                              (testCase == 9 ? 4 : 4))))))));

  real domainLenX = (testCase == 2 || testCase == 7) ? 10.0
                  : (dmr ? 4.0 : (ibcyl ? 6.0 : 1.0));
  real hElem = domainLenX / nElemX;
  real domainSize[3]   = {domainLenX, hElem*nElemY, hElem*nElemZ};
  i32  baseGridSize[3] = {blockSize*nElemX, blockSize*nElemY, blockSize*nElemZ};

  real tEnd  = argF("--tend", testCase == 0 ? 0.2 :
                              (testCase == 1 ? 1.0 :
                              (testCase == 2 || testCase == 7 ? 1.0 :
                              (testCase == 3 ? 0.3 :
                              (testCase == 4 ? 0.15 :
                              (testCase == 5 ? 1.0 :
                              (testCase == 8 ? 1.0 :
                              (testCase == 9 ? 10.0 : 0.2))))))));
  real tStep = tEnd/10.0;

  DgSolver *solver = new DgSolver(domainSize, baseGridSize, nLvls);
  solver->pseudo2D    = (baseGridSize[2] == blockSize) ? 1 : 0;
  // blunt-body IB startup at nlvls >= 4 trips the dt margin at cfl 0.4 (the
  // impulsive-start wake transient spikes lambda mid-step); 0.3 is clean
  solver->cfl         = argF("--cfl", ibcyl ? 0.3 : 0.4);
  solver->cThr        = argF("--cthr", 16.0);
  solver->gammaThr    = argF("--gammathr", 1.0);
  solver->epsOverride = argF("--eps", -1.0);
  solver->refineFac   = argF("--refinefac", 1.0);
  solver->adaptEvery  = argI("--adaptevery", 4);
  solver->refineBuffer = argI("--refinebuffer", 1);
  solver->shockRefine  = argI("--shockrefine", 1);
  solver->shockThresh  = argF("--shockthresh", 0.5);
  solver->icDelta      = argF("--icdelta", 0.5);
  solver->ecVolume    = argI("--ecflux", 1);
  solver->avOn        = argI("--av", 1);
  solver->avCav       = argF("--avcav", 0.5);
  solver->avKsensor   = argF("--avk", 0.05);
  solver->avPen       = argF("--avpen", 1.0);
  solver->sensorType  = argI("--sensor", 2);
  solver->ppS0        = argF("--pps0", -4.0*log10((real)dgOrder));
  solver->ppKappa     = argF("--ppkappa", 2.0);
  solver->scaleMode   = argI("--scalemode", 0);
  solver->indicator   = argI("--indicator", 0);
  solver->ppRefine    = argF("--pprefine", 0.5);
  solver->ppCoarsen   = argF("--ppcoarsen", 0.01);
  solver->refineRadius= argF("--radius", 0.25*domainLenX);
  solver->vortexU0    = argF("--advect", 0.0);
  { real ma = argF("--ma", 0.1);
    solver->greshoP0 = 1.0/(dgGam*ma*ma); }
  solver->ibOn        = argI("--ib", ibcyl ? 1 : 0);
  solver->ibX         = argF("--ibx", 1.5);
  solver->ibY         = argF("--iby", 0.5*(domainLenX/nElemX)*nElemY);
  solver->ibR         = argF("--ibr", 0.5);
  solver->machInf     = argF("--mach", 3.0);
  solver->ibBand      = argF("--ibband", 3.0);
  solver->ibCurv      = argI("--ibcurv", 1);
  solver->ibOrder     = argI("--ibord", 2);
  solver->ibImageFac  = argF("--ibimagefac", 1.5);
  solver->ibShockTheta = argF("--ibtheta", 0.5);
  solver->ibPen       = argF("--ibpen", 0.0);
  solver->ibGraze     = argF("--ibgraze", 0.0);
  solver->ibFillEvery = argI("--ibfillstep", 0);
  solver->ibFilt      = argI("--ibfilt", 0);
  solver->ibCut       = argI("--ibcut", 1);
  solver->dbgChecks   = argI("--debug", 0);
  solver->sortCurve   = argI("--sortcurve", 1);

  solver->icType = (testCase == 0) ? 0 :
                   (testCase == 1) ? 1 :
                   (testCase == 2 || testCase == 7) ? 2 :
                   (testCase == 3) ? 3 :
                   (testCase == 4) ? 1 :
                   (testCase == 5) ? 5 :
                   (testCase == 8) ? 6 :
                   (testCase == 9) ? 7 : 4;
  // case 3 defaults to periodic: zero-gradient (transmissive) BCs are ill-posed
  // at a subsonic INFLOW (no incoming characteristic specified) and grow an
  // exponential boundary mode -- unrelated to the nonconforming interface
  // case 8 (Gresho) uses WALLS: zero-gradient at the u~0 far field is the
  // degenerate-characteristic case (no incoming information specified) and at
  // Ma 0.1 the 10x-faster acoustics excite it within t~0.02 -- same boundary
  // mode as the case-3 free-stream lesson.  Gresho is compactly supported, so
  // walls (or periodic) are equally valid and stable.
  solver->bcType = argI("--bc", (testCase == 1 || testCase == 4 || testCase == 8) ? 0 :
                                (testCase == 2 || testCase == 3 || testCase == 5 || testCase == 7) ? 2 :
                                dmr ? 4 : (ibcyl ? 5 : 3));
  solver->staticGrid = argI("--static", (testCase == 3) ? 1 : (testCase == 7 ? 4 : 0));

  if (hasFlag("--selftest")) {
    bool ok = solver->selfTest();
    delete solver;
    cudaDeviceReset();
    return ok ? 0 : 1;
  }

  solver->initialize();

  // per-level element census + finest-band radial width (at t=0 an IC
  // diagnostic; also reprinted at every output time below)
  auto printCensus = [&](void) {
    std::vector<int> perLvl(nLvls, 0);
    double rmin = 1e30, rmax = -1e30;
    int Lf = nLvls - 1;
    double cx = 0.5*domainSize[0], cy = 0.5*domainSize[1];
    for (int b = 0; b < solver->hashTable.nKeys; b++) {
      unsigned long long loc = solver->bLocList[b];
      if (loc == ~0ULL) continue;
      int lvl, ib, jb, kb; solver->decode(loc, lvl, ib, jb, kb);
      if (lvl < 0 || lvl >= nLvls) continue;
      perLvl[lvl]++;
      if (lvl == Lf) {
        double h = domainSize[0]/(nElemX*powi(2, lvl));
        double x = (ib + 0.5)*h - cx, y = (jb + 0.5)*h - cy;
        double r = sqrt(x*x + y*y);
        rmin = fmin(rmin, r); rmax = fmax(rmax, r);
      }
    }
    printf("[census] per-level elements:");
    for (int l = 0; l < nLvls; l++) printf(" L%d=%d", l, perLvl[l]);
    double hL = domainSize[0]/(nElemX*powi(2, Lf));
    printf("  | finest band: r in [%.4f, %.4f] = %.1f finest-elems wide\n",
           rmin, rmax, (rmax - rmin)/hL);
  };
  printCensus();

  // M2 lifecycle test hook: bootstrap at the static target, then flip the
  // static votes so the first stepping adaptation refines/collapses the whole
  // grid through the spawn/prolong/restrict/prune path (conservation gate)
  if (hasFlag("--collapse"))  { solver->staticGrid = 3; solver->adaptLeaves(); }  // merge to base NOW
  if (hasFlag("--refineall")) { solver->staticGrid = 2; solver->adaptLeaves(); }  // refine to finest NOW

  double m0 = 0, px0 = 0, e0 = 0;
  solver->dgTotalConserved(m0, px0, e0);
  printf("[cons] initial mass=%.12e momx=%.12e energy=%.12e\n", m0, px0, e0);

  real t = 0;
  auto wall0 = std::chrono::steady_clock::now();
  while (t < tEnd - (real)0.01*tStep) {   // margin >> float accumulation error
    t += solver->step(tStep);
    solver->paint();
    double m, px, e;
    solver->dgTotalConserved(m, px, e);
    printf("n: %d, t = %f, nblocks = %d, dt = %e, dM/M0 = %+.3e, dE/E0 = %+.3e%s\n",
           solver->imageCounter, (double)t, solver->hashTable.nKeys,
           (double)solver->deltaT, (m-m0)/m0, (e-e0)/e0,
           solver->hashTable.nDropped ? "  [POOL FULL: blocks dropped!]" : "");
    if (solver->hashTable.nDropped) {
      printf("[fatal] block pool exhausted (nBlocksMax=%d, dropped %d): the grid "
             "is inconsistent -- raise -DNCELLS_MAX or lower resolution\n",
             nBlocksMax, solver->hashTable.nDropped);
      fflush(stdout);
      return 3;
    }
    fflush(stdout);
    printCensus();
    if (testCase == 3) {
      real dev = solver->maxDeviationFromUniform();
      printf("[freestream] t=%.4f  max|Q - Q0| = %.6e\n", (double)t, (double)dev);
    }
  }
  auto wall1 = std::chrono::steady_clock::now();
  double wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();
  printf("done: t = %f after %d iters in %.1f ms\n", (double)t, solver->iter, wallMs);
  solver->printPerf();

  if (testCase == 0) solver->writeLineProfile("output/dg_sod_profile.dat");
  if (testCase == 2 || testCase == 7) solver->computeVortexError(t);
  if (testCase == 8) solver->computeGreshoError();
  if (solver->ibOn) {
    if (solver->machInf > 0) solver->computeIbGates();
    else {   // rest-state gate (--mach 0): fluid must stay at rest to roundoff
      real dev = solver->maxDeviationFromUniform();
      printf("[ibrest] max fluid deviation from rest = %.6e\n", (double)dev);
    }
    solver->writeIbSurface("output/ib_surface.dat");
    solver->paintIbClass("output/ib_class.png");
  }
  solver->paintPressure("output/dg_pressure_final.png");

  cudaDeviceSynchronize();
  delete solver;
  cudaDeviceReset();
  return 0;
}
