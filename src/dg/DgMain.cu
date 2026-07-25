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
//   --ibcurv N     1 = curvature wall conditions du_t/ds = -u_t/R (default 1;
//                  0 = flat-wall HO-i/f, low order on curved bodies)
//   --ibtheta X    donor sensor theta forcing the LO fallback  (default 0.5)
//   --ibpen X      ghost-face penalty fraction of lambda      (default 0)
//   --ibgraze X    extra ghost margin, element units (default 0: center rule)
//   The wall method is FRIB HO-i/c (docs/FRIB.pdf); other fills removed.
//   --gauss N      1 = Gauss-Legendre solution points + flux reconstruction
//                  (interior nodes, entropy-projected face traces, correction
//                  functions); 0 = collocated Lobatto DGSEM         (default 0)
//   --fr N         Gauss FR correction function: 0 = g_DG (Radau), 1 = g_HU
//                  (Huynh, wider stability)                          (default 1)
//   --nsfr X       NSFR residual filter sigma in [0,1) (arXiv 2507.09131):
//                  ESFR-c top-mode residual damping -- free on smooth flow,
//                  best shock profiles; 0 disables               (default 0.3)
//   --bulk X       Ducros-gated bulk (dilatation-only) viscosity strength C_b
//                  -- compression-only, shear/contact-transparent; complement
//                  to --dpsbp for multi-D shocks                     (default 0)
//   --dpsbp X      dual-pairing upwind SBP volume dissipation strength tau
//                  (arXiv 2411.06629): intrinsic entropy-dissipative top-mode
//                  volume upwinding.  Fixes the p3 1D Sod that pure DG cannot
//                  run (--av 0 --subfv 0 --dpsbp 0.1) and cleans post-shock
//                  ringing, but does NOT hold the 2D blast alone   (default 0)
//   --dpface X     EXPERIMENTAL: replace HLLC with the paper's Gamma-upwind
//                  interface flux -- UNSTABLE at strong fronts (measured, both
//                  additive and replace forms); keep 0              (default 0)
//   --eslim N      entropy-stable limiter: 1 = everywhere (p=3 M=3 needs
//                  this), 2 = sensor-gated (smooth-exact)      (default 0)
//   --mood N       1 = a-posteriori MOOD limiter (no a priori sensor/AV):
//                  DG -> detect -> local first-order FV redo    (default 0)
//   --rusface N    element-interface flux: 0 HLLC, 1 Rusanov (vacuum-robust),
//                  2 Roe + Harten fix (the NSFR paper pairing)
//   --subfv N      1 = subcell-FV shock capturing (docs/subcellFV.pdf);
//                  alternative to AV -- try --av 0 --subfv 1    (default 0)
//   --submax X     cap on the subcell-FV blend factor          (default 0.5)
//   --subthr X     FV sensor deadband: below it pure high-order + NSFR filter;
//                  above it alpha rescales to full blend at theta=1 (default 0)
//   --subfloor X   amplitude floor for the shock sensor, rel.  (default 0.01)
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
  solver->gauss       = argI("--gauss", 0);   // Gauss-Legendre pts + flux reconstruction
  solver->frType      = argI("--fr", 1);      // 0 = g_DG (Radau), 1 = g_HU (Huynh)
  solver->nsfr        = argF("--nsfr", 0.3);  // NSFR residual top-mode filter sigma (default ON)
  solver->bulkC       = argF("--bulk", 0.0);  // Ducros-gated bulk viscosity C_b
  solver->dpSbp       = argF("--dpsbp", 0.0); // dual-pairing SBP volume upwind tau
  solver->dpFace      = argF("--dpface", 0.0);// DP interface upwind flux: >0 REPLACES
                                              // HLLC with central + Gamma-upwind (paper Eq 17)
                                              // (arXiv 2411.06629; try 0.1 with --av 0 --subfv 0)
  solver->esLim       = argI("--eslim", 0);
  solver->mood        = argI("--mood", 0);
  solver->moodRho     = argF("--moodrho", 1e-6);
  solver->moodP       = argF("--moodp", 1e-6);
  solver->rusFace     = argI("--rusface", 0);
  solver->ibLimit     = argI("--iblimit", 1);
  solver->ibHO        = argI("--ibho", 1);   // 0 = first-order wall reconstruction
  solver->ibSbm       = argI("--ibsbm", 0);   // 1 = shifted boundary wall (no ghosts)
  solver->ibBrink     = argI("--ibbrink", 0);      // 1 = volume-penalization IB
  solver->ibBrinkEps  = argF("--ibbrinkeps", 1e-4);
  solver->ibBrinkDelta= argF("--ibbrinkdelta", 2.0);   // phi transition width in finest cells
  solver->ibBrinkRate = argF("--ibbrinkrate", 1.0);    // Darcy drag rate / CFL-stable rate
  solver->ibSbmCurv   = argF("--ibsbmcurv", 1.0);  // SBM wall curvature coefficient
  solver->ibShift2    = argI("--ibshift2", 0);     // 2nd-order Taylor velocity shift
  solver->ibSbmPen    = argF("--ibsbmpen", 0.2);   // SBM slip Nitsche penalty alpha
  solver->ibSingle    = argI("--ibsingle", 0);    // 1 = single-IP state, 2 = +gradient
  solver->ibRecon     = argI("--ibrecon", 1);      // 0 = H/S (paper) image line; 1 =
                                                   // primitive (p,rho) DEFAULT 2026-07-15:
                                                   // fixes p3-M3, Gauss-FRIB, high-res M=3
  solver->ibPiston    = argI("--ibpiston", 1);     // 0 = no wall-Riemann star (LO instead)
  solver->ibDil       = argF("--ibdil", 0.0);      // image-line length in h (0 = Eq-22 default)
  solver->ibDbg       = getenv("DGDBG") ? 1 : 0;   // TEMP nose-ghost fill trace
  solver->subFv       = argI("--subfv", (solver->mood || solver->gauss) ? 1 : 0);
  // MOOD uses the FV volume; GAUSS FR requires it: the entropy-projected face
  // trace overshoots at shocks and only the troubled-cell constant-extrapolation
  // blend (keyed on slot-6 alpha) contains it -- AV alone blows up (measured).
  solver->subThr      = argF("--subthr", 0.0);  // FV sensor deadband (see DgSolver.cuh)
  solver->subMax      = argF("--submax", solver->gauss ? 1.0 : 0.5);
  // GAUSS: a saturated sensor must drive the face trace to FULLY constant
  // extrapolation -- a 0.5 cap leaves half the overshooting projected trace in
  // f* and detonates at strong shocks (measured: adaptive Sod t=0.28).  The
  // Lobatto trace is a nodal value, where the paper's 0.5 cap is fine.
  solver->subFloor    = argF("--subfloor", 0.01);
  solver->fluxAvgT0   = argF("--fluxavg0", 0.0);   // boundary-flux time-average
  solver->fluxAvgT1   = argF("--fluxavg1", -1.0);  // window [t0,t1] (t1<=t0 off)
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
  solver->ibShockTheta = argF("--ibtheta", 0.5);
  solver->ibPen       = argF("--ibpen", 0.0);
  solver->ibGraze     = argF("--ibgraze", 0.0);
  solver->ibFillEvery = argI("--ibfillstep", 0);
  solver->ibFilt      = argI("--ibfilt", 0);
  solver->ibCut       = argI("--ibcut", 1);
  solver->ibEvolve    = argI("--ibevolve", 0);   // cut elements join the
  // discretization (IB_CUT): fluid-side nodes evolve, solid nodes keep the
  // FRIB fill sampled from NON-CUT fluid donors.  FRIB-path only.
  if (solver->ibEvolve && (solver->ibSbm || solver->ibBrink || !solver->ibCut)) {
    printf("--ibevolve requires the FRIB ghost path (--ibsbm 0 --ibbrink 0 "
           "--ibcut 1); disabling ibevolve\n");
    solver->ibEvolve = 0;
  }
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

  if (solver->gauss && solver->esLim) {
    // the Gauss RHS does not accumulate the entropy-flux bound (SCRATCH slots
    // 3/4), so the ES limiter would read stale garbage -- refuse loudly
    printf("[fatal] --eslim is not supported with --gauss (entropy-flux bound "
           "not accumulated by dgRhsGaussKernel)\n");
    return 1;
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
    if (solver->ibBrink) {   // volume-fraction field alongside the flow frames
      char fn[80]; sprintf(fn, "output/phi_%05d.png", solver->imageCounter-1);
      solver->paintBrinkPhi(fn);
    }
    double m, px, e;
    solver->dgTotalConserved(m, px, e);
    printf("n: %d, t = %f, nblocks = %d, dt = %e, dM/M0 = %+.3e, dE/E0 = %+.3e%s\n",
           solver->imageCounter, (double)t, solver->hashTable.nKeys,
           (double)solver->deltaT, (m-m0)/m0, (e-e0)/e0,
           solver->hashTable.nDropped ? "  [POOL FULL: blocks dropped!]" : "");
    // silent-flatline guard: a NaN wave gets floored to near-vacuum by the
    // sanitizer (fmax(nan, eps) = eps on CUDA), so a detonation can drain the
    // domain WITHOUT collapsing dt and the run "completes" flat (measured:
    // adaptive MOOD, dM/M0 -> -1.0, nan energy).  Closed-box cases conserve
    // mass to ~1e-4, so a large drop is always a masked blow-up -- abort.
    if (!solver->ibOn && solver->bcType != 3 && solver->bcType != 4
        && solver->bcType != 5 && (m < 0.5*m0 || !std::isfinite(e))) {
      printf("[fatal] mass/energy vanished (dM/M0 = %+.3e): a NaN detonation "
             "was flattened by the sanitizer -- treat as blow-up\n", (m-m0)/m0);
      fflush(stdout);
      exit(2);
    }
    { // boundary mass-flux balance: inflow (x-lo), outflow (x-hi), y walls,
      // and the net vs the actual d/dt(mass) -- the gap is IB non-conservation
      double bnd[4]; solver->boundaryMassFlux(bnd);
      static double mPrev = m0; static double tPrev = 0.0;
      double dMdt = (t > tPrev) ? (m - mPrev)/((double)t - tPrev) : 0.0;
      double netOut = bnd[0]+bnd[1]+bnd[2]+bnd[3];
      printf("[flux] t=%.3f  in(x-lo)=%+.4e out(x-hi)=%+.4e  ywall=%+.2e,%+.2e"
             "  netOut=%+.4e  dM/dt=%+.4e  IBsrc=%+.4e\n",
             (double)t, -bnd[0], bnd[1], bnd[2], bnd[3], netOut, dMdt, dMdt + netOut);
      mPrev = m; tPrev = (double)t;
    }
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

  if (solver->fluxAvgTime > 0.0) {   // time-averaged boundary mass flux
    double *A = solver->fluxAvgAcc, T = solver->fluxAvgTime;
    printf("[fluxavg] window [%.2f,%.2f] (dt-weighted, %.4f s):  <in x-lo>=%+.5e  "
           "<out x-hi>=%+.5e  <net out>=%+.5e  imbalance=%.3f%% of inflow\n",
           (double)solver->fluxAvgT0, (double)solver->fluxAvgT1, T,
           -A[0]/T, A[1]/T, (A[0]+A[1]+A[2]+A[3])/T,
           100.0*(A[0]+A[1]+A[2]+A[3])/fmax(-A[0], 1e-30));
  }

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
  solver->paintTroubled("output/dg_troubled_final.png");

  cudaDeviceSynchronize();
  delete solver;
  cudaDeviceReset();
  return 0;
}
