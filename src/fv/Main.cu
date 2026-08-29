#include <string.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <cmath>

#include "CompressibleSolver.cuh"

//
// Usage:  ./wave3d --case N [--flag value ...]        (all flags optional)
//
//   testCase 0 : pseudo-2D / quasi-1D Sod shock tube (validated vs exact)
//   testCase 1 : 2D circular Sod explosion with adaptive mesh refinement
//   testCase 2 : isentropic vortex on [0,10]^2 (stationarity check, pseudo-2D)
//   testCase 3 : true-3D spherical Sod explosion (exercises the z / Gz paths)
//   testCase 4 : Gresho vortex on [0,1]^2, low-Mach preservation
//   testCase 5 : Gresho vortex on a STATIC radial AMR grid (fine at centre,
//                coarse outward) — tests flow crossing fixed coarse/fine faces
//   testCase 6 : circular Sod explosion on a static radial AMR grid — measures
//                coarse/fine conservation error (mass/energy drift)
//   testCase 14: the SAME plate as an IMMERSED body (level set), so the IB wall
//                model can be checked against the validated grid-aligned result.
//                --ibplane sets the plate height inside the domain.
//   testCase 13: flat-plate turbulent boundary layer (TMR).  Re_L = 5e6, M = 0.2,
//                plate from x = 0.5 on a 2.5 x 0.3 domain (the layer is only
//                ~0.017 thick, so a taller box is wasted); writes output/fptbl_cf.dat
//                and reports Cf at x/L = 0.97 against the TMR value 0.0027.
//                Needs --rans 1; --ma sets the Mach number (default 0.2).
//   testCase 12: RANS near-wall equilibrium probe -- Eq. (24) must balance
//                through the solver's own face loop (gates the Appendix-A tau~
//                diffusion plumbing).  --ma sets u_tau, --mu sets nu.
//   testCase 11: RANS frozen-shear probe -- the k~ source against its analytic
//                value, isolating the discrete S/Omega stencil (O(h^2)).
//                Needs --rans 1; --ma sets the shear amplitude.
//   testCase 10: uniform RANS box -- k~/tau~ against the exact 0-D solution of
//                their source terms (and free-stream preservation of the new
//                convective/diffusive fluxes).  Needs --rans 1 --mu.
//   testCase 9 : viscous shear-wave decay (periodic) — exact check on the
//                Navier-Stokes viscous operator: u = U0 sin(ky) decays as
//                exp(-nu k^2 t) with the nonlinear term identically zero.
//                Needs --mu (or --re); --ma sets the shear amplitude.
//
//   --case N     test case (default 0); also accepted as the first bare arg
//   --nlvls N    number of refinement levels           (per-case default)
//   --nblocks N  base blocks in x                       (per-case default)
//   --wthresh X  wavelet detail threshold               (per-case default)
//   --ma X       Gresho Mach / acConv amplitude / testCase-1 inner pressure
//   --bc N       BC: 0 slip wall, 1 no-slip, 2 periodic, 3 transmissive
//   --recon N    0=TVD, 1=ROUND (default), 2=LD-ROUND, 3=unlimited parabola
//   --tend X     end time override                      (per-case default)
//   --mdflux N   1 = genuinely multidimensional Osher-type corner flux
//                (Gaburro-Ricchiuto-Dumbser, arXiv:2506.00207); 2 = CTU-Hancock:
//                half-step predictor + single-Euler corrector, 2nd order in
//                time, single flux sweep per step.  Stable CFL ~1.2.
//                pseudo-2D only; states follow --recon
//   --cfl X      CFL number (default 0.40; dt = cfl * min(dx/(|u|+c)))
//   --mu X       dynamic viscosity (default 0 = inviscid Euler)
//   --re X       set mu = 1/Re for the unit reference state (alternative to --mu)
//   --pr X       Prandtl number (default 0.72)
//   --suth X     Sutherland constant S/Tref (default 0 = constant mu)
//
//   ---- k~-tau~ SST wall-modeled RANS (Tamaki et al., JCP 566 (2026) 115239) ----
//   --rans N     1 = k~-tau~ SST (default 0 = plain Navier-Stokes, unchanged)
//   --kinf X     freestream k~   (paper: 1e-6 u_inf^2)
//   --tauinf X   freestream tau~ (paper: 0.2 L/u_inf)
//   --sustain N  1 = Eq. (32) freestream-sustaining source terms (default 1)
//   --ransv N    1 = "-V" variant, Omega^2 for S^2 in Eq. (19) (default 1)
//   --dcut X     Eq. (38)/(A.5) cutoff distance; <= 0 selects 3 * finest dx
//   --lref X     characteristic length in Gamma3 (default 1)
//   --prt X      turbulent Prandtl number (default 0.9)
//   --wallgeom N wall distance: 0 none, 1 flat plate at y=0, 2 immersed level set
//   --platex0 X  wallGeom 1: leading-edge x
//   --waoff X    wall/grid offset (paper Fig. 5a); <= 0 selects 0.5 * finest dy
//   --wallband X hold the near-wall band at the FINEST level out to this distance
//                (the wall model needs one resolution along the wall); <= 0 -> 8*dCutoff
//   --tref X     Sutherland reference temperature in T = p/rho units (default 1)
//
// Back-compat: `./wave3d N` (bare first arg) still selects the test case.
// testCases 0-2,4 run pseudo-2D (single z block); testCase 3 is fully 3D.
//
// ---- airfoil geometry -------------------------------------------------------
//
// Two sources, both producing a CLOSED counter-clockwise polyline for the type-6
// level set:
//   --airfoil <file>  two columns "x y", one point per line, '#' comments
//                     ignored.  This is the path for RAE 2822 or any tabulated
//                     section.
//   --naca <MPTT>     the analytic 4-digit family, exact from the published
//                     formula, cosine-clustered so the leading edge is resolved.
//                     NACA0012 is the paper's own airfoil case (Sec. 4.3).
//
static i32 loadAirfoilFile(const char *fname, std::vector<real> &xy) {
  FILE *f = fopen(fname, "r");
  if (!f) { printf("[ib] cannot open airfoil file %s\n", fname); return 0; }
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    double a, b;
    if (line[0] == '#') continue;
    if (sscanf(line, "%lf %lf", &a, &b) == 2) { xy.push_back((real)a); xy.push_back((real)b); }
  }
  fclose(f);
  return (i32)(xy.size()/2);
}

static i32 makeNaca4(i32 code, i32 nPerSide, std::vector<real> &xy) {
  const real tt = (real)(code % 100)/(real)100;          // thickness
  const real pp = (real)((code/100) % 10)/(real)10;      // camber position
  const real mm = (real)(code/1000)/(real)100;           // max camber
  std::vector<real> up, lo;
  for (i32 i = 0; i <= nPerSide; i++) {
    const real beta = (real)(M_PI)*(real)i/(real)nPerSide;
    const real x = (real)0.5*((real)1 - cos(beta));      // cosine clustering
    const real t = (real)5*tt*((real)0.2969*sqrt(x) - (real)0.1260*x
                   - (real)0.3516*x*x + (real)0.2843*x*x*x - (real)0.1036*x*x*x*x);
    real yc = 0, dyc = 0;
    if (mm > 0 && pp > 0) {
      if (x < pp) { yc = mm/(pp*pp)*((real)2*pp*x - x*x); dyc = (real)2*mm/(pp*pp)*(pp - x); }
      else { yc = mm/(((real)1-pp)*((real)1-pp))*(((real)1-(real)2*pp) + (real)2*pp*x - x*x);
             dyc = (real)2*mm/(((real)1-pp)*((real)1-pp))*(pp - x); }
    }
    const real th = atan(dyc);
    up.push_back(x - t*sin(th)); up.push_back(yc + t*cos(th));
    lo.push_back(x + t*sin(th)); lo.push_back(yc - t*cos(th));
  }
  // counter-clockwise: lower TE -> LE -> upper TE, LE/TE not duplicated
  for (i32 i = (i32)lo.size()/2 - 1; i >= 1; i--) { xy.push_back(lo[2*i]); xy.push_back(lo[2*i+1]); }
  for (i32 i = 0; i < (i32)up.size()/2; i++)      { xy.push_back(up[2*i]); xy.push_back(up[2*i+1]); }
  return (i32)(xy.size()/2);
}

int main(int argc, char* argv[]) {

  // --flag value parser: findArg returns the token after --flag, or nullptr.
  auto findArg = [&](const char* key) -> const char* {
    for (int a = 1; a < argc - 1; a++)
      if (strcmp(argv[a], key) == 0) return argv[a+1];
    return nullptr;
  };
  auto hasArg = [&](const char* key) { return findArg(key) != nullptr; };
  auto argI   = [&](const char* key, i32 def)    { const char* v = findArg(key); return v ? atoi(v) : def; };
  auto argF   = [&](const char* key, real def)   { const char* v = findArg(key); return v ? (real)atof(v) : def; };
  auto argS   = [&](const char* key, const char* def) { const char* v = findArg(key); return v ? v : def; };

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
  bool shear   = (testCase == 9);
  bool ransBox = (testCase == 10);                         // uniform RANS box (source gate)
  bool ransShr = (testCase == 11);                         // frozen-shear production probe
  bool ransWal = (testCase == 12);                         // near-wall equilibrium probe
  bool fptbl   = (testCase == 13);                         // flat-plate turbulent boundary layer
  bool ibPlate = (testCase == 14);                         // SAME plate, but immersed (IB gate)
  bool afoil   = (testCase == 15);                         // RAE 2822 airfoil (immersed, level set)
  bool svort   = (testCase == 16);                         // supersonic vortex in an annulus (EXACT, curved wall)
  i32 nLvls    = argI("--nlvls", (testCase == 1 ? 4 : (testCase == 5 ? 3 : (sodAmr ? 3 : (acoustic ? 3 : (fptbl ? 3 : (afoil ? 6 : (svort ? 1 : 1))))))));
  i32 nBlocksX = argI("--nblocks", (testCase == 1 ? 16 : (testCase == 2 ? 40 : (testCase == 3 ? 8 : (testCase == 4 ? 40 : (testCase == 5 ? 10 : (sodAmr ? 16 : (acoustic ? 64 : (acConv ? 8 : (shear ? 8 : ((ransBox||ransShr||ransWal) ? 4 : (afoil ? 24 : (svort ? 32 : ((fptbl||ibPlate) ? 64 : 100))))))))))))));
  real wThresh = argF("--wthresh", (testCase == 1 ? 0.004 : 0.01));
  bool haveMa  = hasArg("--ma");
  real Ma      = haveMa ? argF("--ma", 0.1) : 0.1;   // Gresho Mach number / acConv amplitude
  i32 bcArg    = argI("--bc", -1);                   // bcType override (-1 = per-testCase default)
  // Reconstruction default is SPLIT by physics, because the two regimes want
  // opposite things and both were measured:
  //   EULER  -> van Leer (4).  Strictly TVD, which is what shock capturing on a
  //             smeared/immersed wall wants.
  //   RANS   -> ROUND (1).  A TVD limiter clips the SMOOTH near-wall velocity
  //             gradient, and that gradient is exactly what the image point
  //             feeds to the u_tau solve: van Leer on the mean flow cost the
  //             aligned FPTBL gate +14.6% (0.003093 vs 0.002653), CFL-independent.
  //             The paper agrees -- it leaves mass/momentum/energy UNLIMITED and
  //             limits only the turbulence convection (tvdRecVanLeer, always on).
  // NOTE: van Leer needs CFL <~ 0.6; it is unstable at 1.2 even for the sharp IB.
  i32 reconA   = argI("--recon", argI("--rans", 0) ? 1 : 4);
  i32 jfnkVerifyOnce = argI("--jfnkverify", 0);
  i32 debugA   = argI("--debug", 0);                 // 1 = per-cycle integrity/census diagnostics                 // 0=TVD, 1=ROUND (default), 2=LD-ROUND, 3=unlimited parabola (smooth only)
  real tEndArg = argF("--tend", -1.0);               // tEnd override (-1 = per-testCase default)
  i32 mdFluxA  = argI("--mdflux", 0);                // 1 = multidimensional Osher-type corner flux (first-order states)
  real cflArg  = argF("--cfl", -1.0);                // CFL override (-1 = default 0.40; dt = cfl*min(dx/(|u|+c)))
  real advectA = argF("--advect", 0.0);              // isentropic-vortex (case 2) advection velocity u0=v0 (periodic seam-crossing test)
  real muArg   = argF("--mu",  0.0);                  // dynamic viscosity (0 = inviscid)
  real reArg   = argF("--re",  0.0);                 // Reynolds number -> mu = 1/Re
  real prArg   = argF("--pr",  0.72);                // Prandtl number
  real suthArg = argF("--suth", 0.0);                // Sutherland S/Tref (0 = constant mu)
  real trefArg = argF("--tref", 1.0);                // Sutherland reference temperature
  i32 ransA    = argI("--rans", 0);                  // k~-tau~ SST wall-modeled RANS
  real kinfA   = argF("--kinf", 1e-6);   // paper Sec. 4: k~_inf = 1e-6 u_inf^2
  real tauinfA = argF("--tauinf", 0.2);
  i32 sustA    = argI("--sustain", 1);
  i32 ransvA   = argI("--ransv", 1);
  real dcutA   = argF("--dcut", 0.0);
  real lrefA   = argF("--lref", 1.0);
  real prtA    = argF("--prt", 0.9);
  i32 wgeomA   = argI("--wallgeom", -1);   // -1 = unset (per-case default)
  real px0A    = argF("--platex0", 0.0);
  real waoffA  = argF("--waoff", 0.0);               // wall/grid offset; <=0 -> 0.5*finest dy
  real wbandA  = argF("--wallband", 0.0);            // fine-band height above the wall; <=0 -> 8*dCutoff
  // PNG painting allocates and clears a UNIFORM-FINE image, so its cost grows
  // as 4^nLvls while the solver's grows ~linearly with the adaptive block
  // count.  Measured at nLvls 7 (6144^2 = 151 MB per frame): the painter was
  // 83% of CUDA API time and 36% of GPU kernel time; disabling it cut the
  // transonic run 143 s -> 35 s (4.1x) with identical physics.  Default OFF
  // for deep grids; --paint 1 forces it on.
  i32 paintOn  = argI("--paint", (nLvls >= 5 || testCase == 15) ? 0 : 1);
  i32 detailA  = argI("--detail", 0);                // 1 = paint the wavelet-detail indicator at tEnd (white = refine trigger)

  // FPTBL: the layer is only ~0.017 thick at x/L = 1, so a 1.5 x 0.1 box holds
  // it comfortably (about 5 delta of headroom) at a fraction of the cost of the
  // paper's full domain.  --domx / --domy override.
  real domLenXA = argF("--domx", 0.0), domLenYA = argF("--domy", 0.0);
  // Airfoil: a 24-chord square box with the section at the centre.  Far-field
  // vortex/compressibility corrections are not implemented, so the boundary has
  // to be genuinely far -- 12 chords each way is the usual minimum for Cp.
  // Supersonic vortex: r_i = 1, r_o = 1.384, M_i = 2.25 -- the standard
  // parameters for this verification case.  The box just has to CONTAIN the
  // annulus: everything outside r_o is solid, so the domain BCs are shielded
  // by the outer wall and never enter the answer.
  real svRi = argF("--svri", 1.0), svRo = argF("--svro", 1.384);
  real domainLenX = svort ? 2.0*svRo*1.06 : (testCase == 2) ? 10.0
                  : (afoil ? (domLenXA > 0 ? domLenXA : 24.0)
                  : ((fptbl||ibPlate) ? (domLenXA > 0 ? domLenXA : 1.5) : 1.0));

  bool cube   = (testCase == 3);
  bool square = (testCase == 1 || testCase == 2 || gresho || sodAmr || acoustic || acConv || shear || ransBox || ransShr || ransWal || afoil || svort);
  i32 nBlocksY = (square || cube) ? nBlocksX : ((fptbl||ibPlate) ? max(1, (i32)lround((domLenYA > 0 ? domLenYA : 0.1)/domainLenX*nBlocksX)) : (nBlocksX + 9) / 10);
  i32 nBlocksZ = cube ? nBlocksX : 1;

  // cubic cells; domain length in x is 10 for the isentropic vortex, 1 otherwise
  real dx = domainLenX / (nBlocksX*blockSize);
  real domainSize[3]   = {domainLenX, dx*(nBlocksY*blockSize), dx*(nBlocksZ*blockSizeZ)};
  i32  baseGridSize[3] = {blockSize*nBlocksX, blockSize*nBlocksY, blockSizeZ*nBlocksZ};

  // acConv: RK3's O(dt^3) global error would cap an order study at 3; scaling
  // cfl ~ dx^(1/3) keeps dt^3 ~ dx^4, below 4th-order spatial error.
  real cfl  = acConv ? 0.40*cbrt(4.0/nBlocksX) : 0.40;
  if (cflArg > 0) cfl = cflArg;                      // CLI override
  // testCase 1: --ma sets the circular-Sod inner pressure (1.0 = classic 10:1
  // ratio; 10 = strong 100:1 blast).  The strong blast's faster shock needs a
  // shorter tEnd to stay inside the domain.
  real sodPin = (testCase == 1) ? (haveMa ? argF("--ma", 1.0) : 1.0) : 0.0;
  real acPeriod = domainLenX / sqrt(gam);   // sound-crossing time (c0=sqrt(gam), p0=rho0=1)
  real tEnd  = (testCase == 1 && sodPin > 2.0) ? 0.06*sqrt(10.0/sodPin) : ((testCase == 2) ? 1.0 : (testCase == 3 ? 0.15 : (gresho ? 1.0 : (sodAmr ? 0.15 : (acoustic ? 0.35 : (acConv ? 2.0*acPeriod : (shear ? 0.5 : ((ransBox||ransShr||ransWal) ? 1.0 : (afoil ? 40.0 : (svort ? 2.0 : ((fptbl||ibPlate) ? 5.0 : 0.20)))))))))));
  if (tEndArg > 0) tEnd = tEndArg;                   // CLI override (arg 10)
  real tStep = (testCase == 1) ? ((tEndArg > 0) ? tEnd/50.0 : ((sodPin > 2.0) ? 0.06*sqrt(10.0/sodPin)/10.0 : 0.008)) : (testCase == 2 ? 0.1 : (testCase == 3 ? 0.03 : (gresho ? 0.1 : (sodAmr ? 0.02 : (acoustic ? 0.02 : (acConv ? tEnd : (shear ? tEnd : ((ransBox||ransShr||ransWal) ? tEnd : ((fptbl||ibPlate||afoil||svort) ? tEnd/20.0 : 0.01)))))))));

  CompressibleSolver *solver = new CompressibleSolver(domainSize, baseGridSize, nLvls);
  solver->pseudo2D        = (baseGridSize[2] == blockSizeZ) ? 1 : 0;  // collapse z (pseudo-2D)
  solver->cfl             = cfl;
  solver->waveletThresh   = wThresh;
  solver->icType          = (testCase == 1 || sodAmr) ? (argI("--dgblast", 0) ? 7 : 1) : (testCase == 2 ? 2 : (testCase == 3 ? 3 : (gresho ? 4 : (acoustic ? 5 : (acConv ? 6 : (shear ? 8 : (ransBox ? 9 : (ransShr ? 10 : (ransWal ? 11 : (svort ? 13 : ((fptbl||ibPlate||afoil) ? 12 : 0)))))))))));
  // --dgblast 1: DG-matched blast IC (icType 7) for wavedg3d comparison runs
  solver->bcType          = (bcArg >= 0) ? bcArg : (afoil ? 5 : ((fptbl||ibPlate) ? 4 : ((acConv || shear || ransBox || ransShr || testCase == 1) ? 2 : 3)));   // periodic for the acoustic/shear waves and circular Sod; else transmissive
  solver->vortexAdvect    = (acConv || shear || ransBox || ransShr || ransWal) ? Ma : (testCase == 2 ? advectA : sodPin);  // acConv/shear: wave amplitude; case 2: vortex advection; testCase 1: Sod inner pressure
  // shear: fix the background pressure (c ~ 11.8) so --ma sets ONLY the shear
  // amplitude; the flow Mach number is then ~ma/11.8, low enough that the
  // O(Ma^2) viscous-heating contamination stays under the truncation error.
  solver->greshoP0        = shear ? 100.0 : 1.0/(gam*Ma*Ma);   // Gresho background pressure -> Mach = Ma
  solver->staticGrid      = acoustic ? 3 : ((testCase == 5 || sodAmr) ? 1 : 0);   // 1=radial shells, 2=planar band, 3=centre step
  solver->refineRadius    = 0.4;                        // fine-region half-extent (unused for the step)
  solver->recon           = reconA;
  solver->dbgChecks       = debugA;                     // face reconstruction (TVD / ROUND / LD-ROUND)
  solver->mdFlux          = mdFluxA;                     // multidimensional Osher-type corner flux
  solver->mu              = (reArg > 0) ? (real)(1.0/reArg) : muArg;
  solver->Pr              = prArg;
  solver->sutherS         = suthArg;
  solver->sutherTref      = trefArg;
  // testCase 14 puts the wall in as an IMMERSED body.  A full half-space (type 2),
  // not the half-plane of type 4: a half-plane of finite thickness is a
  // forward-facing STEP, and the flow separating over it is not the flat-plate
  // problem the grid-aligned case solves.  With the plane running the whole
  // length, and the wall model is applied only for x >= plateX0 with slip
  // upstream, the geometry AND the boundary treatment match case 13 exactly --
  // same problem, two independent wall implementations.
  solver->immerserdBcType = svort ? 7 : (ibPlate ? argI("--ibtype", 2) : 0);
  // ibtype 8 (planar slab) takes its upper wall from --ibplane2
  if (solver->immerserdBcType == 8) solver->ibRadius2 = argF("--ibplane2", 0.4);
  if (svort) {
    solver->ibCenter[0] = 0.5*domainLenX; solver->ibCenter[1] = 0.5*domainLenX;
    solver->ibCenter[2] = 0.5; solver->ibRadius = svRi; solver->ibRadius2 = svRo;
    solver->svMach = argF("--svmach", 2.25);
  }
  solver->ibDfcMode       = argI("--ibdfc", 0);   // diagnostic: freeze one d_FC term
  solver->dIpFac          = argF("--dipfac", 3.0);  // image-point distance in cells
  solver->ipStandMin      = argF("--ipstand", 0.0);  // min IP standoff above the wall face, cells
  // ARCHITECTURE SPLIT (the two papers do different things):
  //   RANS (WallModeledRans.pdf / UTCart): GHOST-FREE -- "solves the governing
  //   equations only on the fluid cells", wall model imposed as the FC boundary
  //   flux; the word "ghost" does not appear in the paper.  Mixing the ghost
  //   fill with the prescribed flux is what broke the immersed RANS path (NaN
  //   at the degenerate cell-centre plane, d_FC-dependent Cf otherwise).
  //   EULER (FRIB.pdf): ghost states ARE the boundary condition, faces use the
  //   ordinary Riemann solve.  So: ghost-free under --rans, ghosts otherwise.
  // The ghost-cell wall-function architecture NEEDS filled ghosts: the wall
  // stress comes from the near-wall gradient the ordinary flux reads off them.
  // Ghost-free is Tamaki's architecture (prescribed FC flux) and the two cannot
  // be mixed -- that mix is what made an earlier attempt return zero flux.
  solver->ibGhostFree     = argI("--ibgf", (ransA && argI("--ibwm",0)==0) ? 1 : 0);
  solver->ibWmles         = argI("--wmles", 0);   // wall-modeled implicit LES (no RANS)
  solver->wmX1            = argF("--wmx1", 1.0e30); // slip tail past this x (outflow corner)
  solver->detFlux         = argI("--detflux", 1);   // deterministic face-flux gather
  solver->shash           = argI("--shash", 0);     // XOR state-hash per step (2: per phase)
  solver->shashFrom       = argI("--shashfrom", 0);
  solver->shashTo         = argI("--shashto", 1<<30);
  // MEASURED HARMFUL, kept only as a probe: flooring the SA destruction
  // distance caps the one term that holds nu~ down near the surface, and the
  // RAE blows up 5x sooner (34 iters vs 185).  The d = 0 overflow it was
  // written for is fixed properly by the centre-inside-body guard instead.
  solver->saDFloor        = argF("--sadfloor", 0.0);  // SA destruction d-floor (local cells)
  solver->ffVortex        = argI("--ffvortex", 0);   // point-vortex far field (lifting bodies)
  solver->ffEvery         = argI("--ffevery", 100);
  solver->wmGhost         = argI("--wmghost", 0);  // 0 = plain slip ghosts (default); 1 = log-law ghosts
  solver->wmCurv          = argI("--wmcurv", 1);   // curvature pressure on the wall-model face
  solver->wmClip          = argF("--wmclip", 3.0);  // ibwm: near-wall mu_t clamp band (local cells)
  solver->wallPointImplicit = argI("--wallpi", 1);  // point-implicit wall k~/tau~ flux (0 = explicit dt cap)
  solver->paintOn         = paintOn;   // gates the uniform-fine image everywhere
  solver->ibIpQuad        = argI("--ipquad", 0);
  solver->ibThermoRec     = argI("--ibthermo", 0); // close the wall trace on (s, H) not (p, rho)
  solver->ibWls           = argI("--ibwls", 0);    // constrained quadratic WLS wall trace   // 2 image points + biquadratic wall reconstruction
  solver->ransA7Tol       = argF("--a7tol", 1e-6); // (A.7) switch: caps the (A.6) tau ratio
  solver->jfnkOn          = argI("--jfnk", 0);     // implicit k~/tau~ (Newton-Krylov)
  solver->jfnkM           = argI("--jfnkm", 15);   // GMRES restart length
  solver->jfnkCfl         = argF("--jfnkcfl", 50.0); // pseudo-time dtau = jfnkCfl*dt
  solver->ibInfinite      = argI("--ibinf", 0);    // ibtype 5: infinite plane (no tip)
  solver->ibWallMode      = argI("--ibwm", 0);     // 1 = ghost-cell architecture (Processes 2024); 2 = log-law ghost (diagnostic)
  solver->turbModel       = argI("--sa", 0);       // 1 = Spalart-Allmaras
  // SA freestream: the TMR convention is nu~_inf = 3 nu (mu_t/mu ~ 0.21).
  solver->nutInf          = argF("--nutinf", 3.0*(double)solver->mu);
  solver->wmX0            = argF("--wmx0", -1.0);  // wall-model start x; <0 = use plateX0
  solver->wmRamp          = argF("--wmramp", 0.0);  // wall-model blend-in fetch past wmx0 (immersed path)
  solver->brinkPI         = argI("--brinkpi", 2);  // 1 = p grad(phi) only, 2 = full porosity stiffness
  solver->brinkFaceLS     = argI("--brinkface", 3); // 1 pt value, 2 endpoint avg, 3 segmented
  solver->brinkNSeg       = argI("--brinkseg", 4);  // segments for brinkface 3
  solver->brinkDtW        = argI("--brinkdtw", solver->brinkPI >= 2 ? 0 : 1); // phi-ratio dt limit: unneeded once brinkpi 2 absorbs it
  solver->ibNoSlip        = argI("--ibnoslip", 0);   // volume-penalized no-slip (viscous Brinkman)
  solver->ibNoSlipRate    = argF("--noslipRate", 4.0); // penalization rate / (U_ref/h)
  solver->ibSlipModel     = argI("--ibslip", 0);    // 1 = slip-length wall model
  solver->slipA1          = argF("--slipa1", 0.30); // lambda_x = 1 + a1 (delta_f+)^n1
  solver->slipN1          = argF("--slipn1", 0.53);
  solver->slipMatchH      = argF("--wmmatch", 0.0);
  solver->ibTurbShift     = argF("--turbretreat", 2.5);  // wall-modelled band retreat (u AND k~/tau~ masks), cells inside: 2.5 keeps the plate slip tail (-1..-2% Cf) AND damps curved-body interiors (RAE nose blowup at 4) // matching height / h_fine (0 = delta_f)
  solver->ibMassRepair    = argF("--massrepair", 1.0);  // deep-body rho relaxation toward rho_inf (0 = off)
  solver->ibPureSource    = argI("--ibpure", 0);      // 1 = pure-source IB (no porosity flux machinery)
  solver->ibTangOnly      = argI("--ibtang", 0);      // wall model = tangential traction ONLY (pressure-tight normal, Darcy interior)
  solver->wmOrder         = argI("--wmorder", 2);   // ibslip 4: series truncation order
  solver->wmGain          = argF("--wmgain", 1.0);  // ibslip 4: feedback gain
  solver->wmAnchor        = argF("--wmanchor", 1.0); // ibslip 4: anchor start, cells behind wall
  solver->wmNormal        = argI("--wmnormal", 0);   // ibslip 4: feedback the normal component too
  solver->wmPush          = argI("--wmpush", 0);     // ibslip 4: allow accelerating feedback
  solver->ibFieldAllLvls  = argI("--fieldall", 0); // field dump: all leaves, not finest only
  solver->brinkAnalyticGrad = argI("--brinkgrad", 0); // analytic p grad(phi) source
  solver->ibBrink         = argI("--ibbrink", 0);     // pressure-tight volume penalization
  solver->ibBrinkEps      = argF("--brinkeps", 1e-6); // volume fraction inside the body
  solver->ibBrinkDelta    = argF("--brinkdelta", 1.5);// tanh thickness, in cells
  solver->ibBrinkRate     = argF("--brinkrate", 0.125); // interior Darcy damping / (lambda/h)
  solver->ibBrinkShift    = argF("--brinkshift", 4.0);// Darcy mask retreat into the body, cells
  solver->ibBrinkDarcyFac = argF("--brinkdfac", 0.5); // Darcy mask width / delta (paper: 1/2)
  solver->ibTurbFlux      = argI("--ibturb", 1);   // 0 = no k~/tau~ wall flux (diagnostic)
  solver->gridTrace       = argI("--gridtrace", 0);  // dump the build-cascade grids
  solver->adaptEvery      = argI("--adaptevery", 8);  // adaptation cadence (~5 syncs/call)
  solver->dtEvery         = argI("--dtevery", 4);  // dt reduction cadence (hard sync per call)
  solver->dtDipThresh     = argF("--dtdip", 0.0);  // report argmin cell when stable dt < this (0 = off)
  solver->envCheck        = argI("--envcheck", 0); // per-step state-envelope check with neighborhood report
  solver->ibFluxRecon     = argI("--ibrecon", 2);  // 1 = pure ghost Riemann, 2 = FRIB face trace + Riemann
  solver->ibCurv          = argI("--ibcurv", 0);   // FRIB curvature term (measured: overshoots here)
  solver->ibHo            = argI("--ibho", 0);     // FRIB HO-i (k=2): H/S-form wall condition on a 3h image line
  solver->ghostSlip       = argI("--ghostslip", 1);  // slip-mirror ghosts; wall model only on the face
  solver->rkScheme        = argI("--rk", 0);        // 0 = LSRK3, 1 = Jameson 4-stage, 2 = Jameson 5-stage
  solver->nRkStages       = (solver->rkScheme == 1) ? 4 : ((solver->rkScheme == 2) ? 5 : 3);
  solver->precond         = argI("--precond", 0);   // low-Mach preconditioning (STEADY runs only)
  solver->precondK        = argF("--precondk", 5.0);
  solver->lts             = argI("--lts", 0);       // local time stepping (STEADY runs only)
  solver->ltsRatio        = argF("--ltsratio", 100.0);  // cap on dt_local/dt_global
  // Airfoil body (type 6).  --airfoil takes precedence over --naca.
  {
    std::vector<real> afXY;
    // testCase 15 loads the RAE 2822 table shipped in geom/ unless overridden.
    const char *afFile = argS("--airfoil", afoil ? "geom/rae2822.dat" : nullptr);
    const i32   nacaC  = argI("--naca", 0);
    i32 nAf = 0;
    if (afFile)      nAf = loadAirfoilFile(afFile, afXY);
    else if (nacaC)  nAf = makeNaca4(nacaC, argI("--afpts", 120), afXY);
    if (nAf > 2) {
      // Centre the section in the box.  The LE is placed a half cell off the
      // grid lines for the same reason the immersed plane is (see ibPlane): a
      // surface sitting exactly on a face is the degenerate d_FC = 0 geometry.
      const real hFine = dx/powi(2, nLvls-1);
      const real chord = argF("--chord", 1.0);
      // FACE-LINE anchor (user's call, 2026-08-28): chord line and TE exactly
      // ON grid lines.  The old +0.5*hFine "mandatory half-cell offset" was
      // inherited from the PLATE's on-face degeneracy, but an airfoil touches
      // the lattice only at isolated points -- and that offset put the chord
      // line through the CELL-CENTER rows, where the two surfaces' discrete
      // walls part ways.  Measured on the alpha=0 Ma 0.8 gate (nlvls 6):
      // L2(Cp_up - Cp_lo) = 0.166 with the old anchor, 2.6e-4 with this one
      // (shock positions match to 1e-4 c); a quarter-cell control sits at
      // 0.123, so face-line alignment is a genuine symmetry property, not a
      // lucky cancellation.
      const real ax0 = argF("--afx", afoil ? 0.5*domainLenX - 0.5*chord
                                           : (real)0.5);
      const real ay0 = argF("--afy", afoil ? 0.5*domainSize[1] : (real)0.5);
      const real aoa = argF("--aoa", 0.0)*(real)(M_PI/180.0);
      const real ca = cos(aoa), sa = sin(aoa);
      for (i32 i = 0; i < nAf; i++) {            // scale, rotate by -aoa, translate
        const real x = afXY[2*i]*chord, y = afXY[2*i+1]*chord;
        afXY[2*i]   = ax0 + x*ca + y*sa;
        afXY[2*i+1] = ay0 - x*sa + y*ca;
      }
      solver->setAirfoil(afXY.data(), nAf);
      solver->immerserdBcType = 6;
      solver->ibOrigin[0] = ax0; solver->ibOrigin[1] = ay0;
      // far-field vortex sits at the QUARTER CHORD, in the rotated frame
      solver->ffXv = ax0 + (real)(0.25*chord*ca);
      solver->ffYv = ay0 - (real)(0.25*chord*sa);
      solver->ibCosA = ca; solver->ibSinA = sa;
      solver->ibChord = chord;
    }
  }
  // HALF-CELL OFFSET IS MANDATORY.  At a whole multiple of the cell size the
  // immersed surface lands exactly ON a grid face, which is the degenerate
  // geometry wallOffset exists to prevent on the grid-aligned path: d_FC = 0
  // (floored), and the corner test's phi >= 0 tie marks the cell above the
  // surface non-fluid too, so the real face-to-wall distance jumps to a FULL
  // cell -- the large-d_FC regime ibDfcMode was written to study.  Measured at
  // 4.0 dx: max|rhoV| = 2.4 with rho u 73% above freestream in the first fluid
  // row.  At 4.5 dx (d_FC = 0.5h, matching wallOffset): max|rhoV| = 1.6e-2 and
  // C_f tracks the grid-aligned case to 4 digits.
  solver->ibPlane         = argF("--ibplane", 4.5*dx/powi(2, nLvls-1));
  solver->ibAngle         = argF("--ibangle", 0.0);   // type 5: plate inclination, degrees
  // FPTBL (TMR): u_inf = 1, rho_inf = 1, L = 1, M = 0.2 -> p_inf = 1/(gam M^2);
  // Re_L = 5e6 -> mu = 2e-7.  Freestream turbulence per the paper: k~ = 1e-6 u^2,
  // tau~ = 0.2 L/u, with the Eq. (32) sustaining terms on.
  if (afoil) {
    // The SECTION carries the angle of attack (it is rotated by -aoa above), so
    // the freestream stays along +x.  rho_inf = 1, |u_inf| = 1, and the Mach
    // number sets p_inf = 1/(gamma M^2) exactly as in the flat-plate cases, so
    // Cp = (p - p_inf)/(0.5 rho_inf u_inf^2) = 2 (p - p_inf).
    solver->fsU = 1.0;
    solver->fsV = 0.0;
    solver->fsP = 1.0/(gam*Ma*Ma);
  }
  if (fptbl || ibPlate) {
    // The freestream runs ALONG the plate, so an inclined plate needs an
    // inclined freestream -- that is what the paper's "30 degrees between the
    // grid axis and the wall" means.  |u| stays 1 so Re and M are unchanged.
    const double aRad = argF("--ibangle", 0.0)*3.14159265358979323846/180.0;
    solver->fsU = cos(aRad);
    solver->fsV = sin(aRad);
    solver->fsP = 1.0/(gam*Ma*Ma);
  }
  // reference Mach^2 for the preconditioner's beta^2 floor (rho_inf = 1)
  {
    const double u2 = (double)solver->fsU*solver->fsU + (double)solver->fsV*solver->fsV;
    const double c2 = gam*(double)solver->fsP;
    solver->precondMref2 = (c2 > 0) ? u2/c2 : 0.04;
  }
  solver->rans            = ransA;
  solver->kInf            = kinfA;
  // SA mode: rho*nu~ lives in the F_RHOK slot, so the IC and every domain BC
  // (which already set RhoK = kInf) get the right freestream by pointing kInf
  // at nu~_inf.  No kernel needs to know which model is running.
  if (solver->turbModel == 1) { solver->kInf = solver->nutInf; solver->tauInf = 1.0; }
  solver->tauInf          = tauinfA;
  solver->ransSustain     = sustA;
  solver->ransVorticity   = ransvA;
  solver->Lref            = lrefA;
  solver->PrT             = prtA;
  solver->wallGeom        = (wgeomA >= 0) ? wgeomA
                          : ((ibPlate||afoil) ? 2 : ((ransWal || fptbl) ? 1 : 0));   // 2 = level set   // testCases 12/13 are flat plates
  solver->plateX0         = (fptbl && px0A == 0.0) ? 0.25 : px0A;   // FPTBL leading edge
  if (ibPlate && px0A == 0.0) solver->plateX0 = 0.25;               // IB gate: same LE as case 13
  const real plateX0Val   = solver->plateX0;
  // The wall sits this far BELOW the bottom domain face (paper Fig. 5a).  It has
  // to be > 0 or the Appendix-A tau~ flux degenerates at d_FC = 0.
  solver->wallOffset      = (waoffA > 0) ? waoffA : 0.5*dx/powi(2, nLvls-1);
  // Eq. (38) r_d / Eq. (A.5) phi cutoff.  The paper fixes it at the image-point
  // distance, 3 * the minimum cell size; 0 leaves phi == 1 (no wall).
  // Gate on the RESOLVED wallGeom (set just above), not the raw CLI arg: with the
  // arg defaulting to -1 for "unset", testing it here left dCutoff = 0 on exactly
  // the two wall cases, which silently disabled the Eq. (38) r_d augmentation and
  // the Appendix-A phi transform on interior faces while the wall face kept its
  // own 3*dy fallback -- so the (A.4) split no longer telescoped.
  solver->dCutoff         = (dcutA > 0) ? dcutA
                          : ((solver->wallGeom > 0) ? 3.0*dx/powi(2, nLvls-1) : 0.0);
  // Hold the near-wall band at the finest level so d_cutoff and the image-point
  // distance refer to the SAME dy (see wallFineBand).  It has to cover the model's
  // own reach -- the image point at 3*dy plus its interpolation stencil, and the
  // whole r_d region d < d_cutoff -- so a multiple of dCutoff is the natural unit.
  solver->wallFineBand    = (wbandA > 0) ? wbandA : 8.0*solver->dCutoff;

  if ((ransA || solver->ibWmles) && muArg <= 0 && reArg <= 0)
    printf("[warn] --rans/--wmles need a viscous solve; pass --mu or --re\n");
  if ((ransA || solver->ibWmles) && solver->wallGeom && nLvls > 1)
    printf("[rans] adaptive grid: holding the wall band (d < %g, %.1f finest cells) at level %d "
           "so d_cutoff and d_IP share one dy\n", (double)solver->wallFineBand,
           (double)(solver->wallFineBand/(dx/powi(2, nLvls-1))), nLvls-1);
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

  // testCase 11 builds ONE right-hand side and compares it to the analytic
  // source; there is nothing to time-march.
  if (ransShr || ransWal) {
    if (ransShr) solver->computeRansShearProbe();
    else         solver->computeRansWallProbe();
    cudaDeviceSynchronize();
    delete solver;
    cudaDeviceReset();
    return 0;
  }

  real t = 0;
  auto wall0 = std::chrono::steady_clock::now();
  while (t < tEnd) {
    t += solver->step(tStep);
    // The PNG painter allocates and clears a UNIFORM-FINE image
    // (baseGrid * 2^(nLvls-1) squared): 6144^2 = 151 MB at nLvls 7, 604 MB at
    // nLvls 8, cleared AND painted every output.  Profiled at nLvls 7 it was
    // 83% of all CUDA API time and 36% of GPU kernel time -- i.e. the diagnostic
    // dominated the solve.  Off by default for the airfoil cases, which are
    // analysed from the .dat dumps; --paint 1 restores it.
    if (jfnkVerifyOnce) { solver->jfnkVerify(); jfnkVerifyOnce = 0; }
    if (paintOn) solver->paint();
    real comp = 100.0 * real(solver->hashTable.nKeys) /
                real(baseGridSize[0]*baseGridSize[1]*baseGridSize[2]/blockSizeTot*powi(powi(2,nLvls-1),2));
    printf("n: %d, t = %f, nblocks = %d, dt = %e, grid = %.1f%% of uniform-fine\n",
           solver->imageCounter, t, solver->hashTable.nKeys, solver->deltaT, comp);
    if (ibPlate && solver->immerserdBcType == 3)
      solver->writeIbField("output/cyl_field.dat", 0.25);
    if (afoil) { solver->writeIbSurface("output/rae2822_surface.dat");
                 solver->writeIbField("output/rae2822_field.dat", 1.5);
                 solver->writeGridBlocks("output/rae2822_grid.dat");
                 solver->writeIbMask("output/rae2822_mask.dat", 1.0);
                 solver->writeIbGhostLines("output/rae2822_glines.dat", 1.0);
                 solver->writeIbWallFaces("output/rae2822_faces.dat", 1.0);
                 solver->printRansExtremes(); }
    if (fptbl || ibPlate) { solver->writeSolution("output/fptbl_field.dat", "output/fptbl_prof.dat", plateX0Val + 0.97);
                 solver->wallResolutionCheck();
                 solver->printRansExtremes();
                 solver->writeCfProfile("output/fptbl_cf.dat"); }
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
  if (svort) {
    solver->computeSvortexError();
  }
  if (gresho) {
    solver->computeGreshoError();
  }
  if (acoustic) {
    solver->computeAcousticReflection("output/acoustic_profile.dat");
  }
  if (shear) {
    solver->computeShearDecayError(t);
  }
  if (ransBox) {
    solver->computeRansDecayError(t);
  }

  if (acConv) {
    solver->computeAcousticL2Error();
  }
  solver->printDiagnostics();
  if (paintOn) solver->paintPressure("output/pressure_final.png");
  if (detailA) {   // wavelet-detail indicator maps (white = refine trigger)
    solver->paintDetail("output/detail_max.png", 0);
    solver->paintDetail("output/detail_rho.png", 1);
    solver->paintDetail("output/detail_mom.png", 2);
    solver->paintDetail("output/detail_E.png",   3);
  }

  cudaDeviceSynchronize();
  delete solver;
  cudaDeviceReset();
}
