#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

//
// wavedg3d: multi-resolution adaptive DGSEM solver (3D compressible Euler).
//
// Discretization (ported from ../fvStuff/dgsem_lobatto_2d.cu, 3D-ified):
//   nodal collocation DGSEM on Legendre-Gauss-Lobatto nodes, strong form,
//   diagonal GLL mass, Chandrashekar entropy-conservative flux-differencing
//   volume term, HLLC interface flux, SSP-RK3, Zhang-Shu positivity limiter,
//   Ducros-sensor element-local artificial viscosity.
//
// Multi-resolution (wavelet-free MRA of Gerhard/Mueller/Sikstel, CAMC 2022):
//   the grid is a 2:1-graded partition of LEAF elements only -- no coarse
//   parents are stored or evolved.  The refinement indicator restricts each
//   complete sibling octet to its virtual parent by the GLL-discrete L2
//   projection and thresholds the detail (child data minus prolongated
//   parent).  Details >    eps  => children stay;  > 2^(p+1) eps => refine;
//   below eps (whole octet) => merge.  Harten's neighbor rule and a 2:1
//   grading fixpoint make the predicted grid reliable.
//
// Refinement-boundary coupling (NO ghost cells, unlike the FV solver):
//   a fine element's face flux is evaluated at its own face nodes against the
//   coarse neighbor's trace interpolated to those nodes; the coarse element
//   accumulates the identical pointwise fluxes through the fine-subface GLL
//   quadrature and a weighted transpose interpolation back to its face
//   polynomial (a Nitsche-type weak interface integral / Kopriva-style
//   mortar) -- discretely conservative under the diagonal LGL mass.
//   Boundary conditions are imposed weakly per face quadrature point
//   (mirror / copy / periodic-wrap / inflow states), so no exterior blocks
//   exist at all.
//
// One AMR block == one DG element: blockSize (=4) nodes per direction is
// exactly p+1 for p=3, and the grid never stores halos, so the existing
// fieldData layout holds the 4^3 LGL nodal values with zero waste.
//

#ifndef DG_ORDER
#define DG_ORDER 3
#endif
static constexpr i32 dgOrder = DG_ORDER;
static constexpr i32 NNODE   = dgOrder + 1;   // nodes per direction per element
static_assert(NNODE == blockSize, "one AMR block = one DG element requires blockSize == p+1");

static constexpr real dgGam = 1.4;

//
// field layout (nDgFields = 17), all per-node slabs of blockSizeTot (=64):
//   0..4   : Q       conservative nodal state (rho, rhou, rhov, rhow, rhoE)
//   5..9   : Q0      u^n register for SSP-RK3.  Also the block-sort double
//                    buffer and the MRA restriction scratch (virtual-parent
//                    nodal values, stored on the octet's anchor block) --
//                    all uses are temporally disjoint (sort and indicator run
//                    between steps, before Q0 is snapshotted for the stages).
//   10..14 : RHS     L(Q)
//   15     : LAM     per-node dt bound h/(lam*NN) (deltaT reduce); also the
//                    MRA detail-norm accumulator (anchor nodes 0..4) and the
//                    per-element sensor/detail paint slab
//   16     : SCRATCH pressure / diagnostics paint
//
enum DgField {
  D_RHO = 0, D_RHOU = 1, D_RHOV = 2, D_RHOW = 3, D_RHOE = 4,
  D_Q0  = 5,
  D_RHS = 10,
  D_LAM = 15,
  D_SCRATCH = 16
};
static constexpr i32 NEVOLVE_DG = 5;

// immersed-boundary element classes (ibClassList; 0 default = all-fluid)
enum IbClass { IB_FLUID = 0, IB_GHOST = 1, IB_DEAD = 2 };
// managed diagnostic counters (ibCnt[])
enum IbCnt { IB_CNT_NODONOR = 0, IB_CNT_RETRY1 = 1, IB_CNT_FALLBACK = 2, IB_CNT_N = 4 };
static constexpr i32 nDgFields  = 17;

class DgSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;            // dt = cfl * min_e h_e/(lam_e * NNODE); dgsem stable to ~0.6, default 0.4
  real cThr;           // eps_L = cThr * h_L^gammaThr.  Paper uses C~1, but with
                       // the normalized-amplitude detail here that refines almost
                       // everything (h_L is tiny, so eps_L~1e-3); default 16 keeps
                       // the smooth interior coarse and refines only real features.
  real gammaThr;       // 1 with shocks, p for smooth-only runs
  real epsOverride;    // > 0: use this eps_L directly instead of the formula
  real refineFac;      // SINGLE-threshold factor: an octet refines toward the
                       // finest level iff sig > refineFac*eps (default 1).  A
                       // shock's detail does not decay with level, so it keeps
                       // refining to the finest and never straddles a coarse/fine
                       // face; a smooth feature's detail decays and stops it
                       // within a few levels.  (The paper's separate 2^(p+1)
                       // refine band left shocks stalled one level too coarse
                       // under leaf-only evolution.)
  i32  adaptEvery;     // adaptation cadence in steps.  The buffer/neighbor rule
                       // give a one-fine-element margin and a feature crosses one
                       // element in NN/cfl (= 10) steps, so cadence must stay
                       // below that; default 4 (the FV cadence, ~2.5x margin) --
                       // measured: vortex accuracy flat to 8, degrades at 16
  i32  refineBuffer;   // fine-level safety margin around refined elements:
                       // 0 = off; 1 = DIRECTIONAL (extend only toward the half(s)
                       // of the element the detail sits on, by density-gradient
                       // energy -- far fewer neighbors); 2 = full 26-neighbor ring
  i32  shockRefine;    // 1 = force any element whose Ducros shock sensor fires to
                       // REFINE, so shocks are pulled to the finest level and
                       // never straddle a nonconforming coarse/fine interface
                       // (the coarse side cannot represent a shock -> its mortar
                       // trace overshoots).  Paper Remark 2: MRA + shock detector.
  real shockThresh;    // sensor value above which an element is deemed shocked
  real icDelta;        // IC interface smoothing width in units of the FINEST element size
                       // (fixed across levels; ~0.5 keeps a strong jump resolved at p=3)
  i32  ecVolume;       // 1 = Chandrashekar EC flux-differencing volume (default), 0 = collocation
  i32  avOn;           // Ducros-sensor artificial viscosity on/off
  real avCav;          // AV strength C_av (dgsem default 0.25)
  real avKsensor;      // Ducros sensor constant K (dgsem default 0.1)
  real avPen;          // interface jump-penalty scale: sigma = avPen * avCav *
                       // max(theta*lam)/2 (sensor-gated Rusanov).  The only
                       // cross-face AV dissipation; 0 disables
  i32  sensorType;     // AV/penalty sensor: 0 = Ducros compression, 1 = Persson-
                       // Peraire modal smoothness (energy fraction of the top
                       // Legendre modes of PRESSURE; contact-transparent)
  real ppS0;           // Persson ramp center in log10(S) (default -4*log10(p))
  real ppKappa;        // Persson ramp half-width (default 2)
  i32  bcType;         // 0 slip wall, 2 periodic, 3 transmissive, 4 double-Mach-reflection
  i32  icType;         // 0 x-Sod, 1 circular/spherical Sod, 2 isentropic vortex,
                       // 3 uniform free-stream, 4 DMR, 5 Gaussian density pulse
  i32  scaleMode;      // detail normalization c_i: 0 = max(1,|domain mean u_i|) (paper), 1 = domain max |u_i|
  i32  indicator;      // adaptation indicator: 0 = wavelet-free MRA detail (default),
                       // 1 = smoothness-sensor vote (Persson theta hysteresis)
  real ppRefine;       // indicator 1: REFINE when theta_e > ppRefine  (default 0.5)
  real ppCoarsen;      // indicator 1: DELETE when theta_e < ppCoarsen (default 0.01)
  i32  staticGrid;     // 0 dynamic MRA; 1 center-sphere two-level; 2 forced uniform-fine;
                       // 3 forced base collapse; 4 planar x-band two-level (vote overrides)
  real refineRadius;   // staticGrid 1/4: fine-region half-extent about the domain center
  real dmrShockPos;    // DMR: initial shock foot x-position on the bottom wall
  i32  ibOn;           // 1 = ghost-element immersed boundary active (cylinder SDF)
  real ibX, ibY;       // cylinder center
  real ibR;            // cylinder radius
  real machInf;        // freestream Mach (case 9; a_inf = 1 normalization)
  real ibBand;         // force-finest band half-width, in finest-element units
  i32  ibCurv;         // 1 = curvature-corrected wall pressure dp/dn = rho*vt^2/R
  i32  ibOrder;        // reconstruction order: 2 (default) = wall BC + value +
                       // 1st derivative at the image; 1 = linear.  3 (full
                       // cubic, + 2nd derivative) is LINEARLY UNSTABLE: the
                       // donor's 2nd derivative amplifies node-scale noise by
                       // O(1/h^2) into the ghost value -- measured rest-state
                       // e-folding ~0.06, immune to penalty/filter damping
  real ibImageFac;     // image-point distance floor, in ghost-element widths
  real ibShockTheta;   // donor sensor theta above which order drops to 2
  real ibPen;          // ungated fraction of lambda_e ghosts publish as their
                       // face-penalty scale (wall-face jump damping)
  real ibGraze;        // elements closer than ibGraze*h to the wall become
                       // ghosts even when not cut (grazing-sliver guard)
  i32  ibFillEvery;    // 0 = refill ghosts every RK stage; 1 = once per step
  i32  ibFilt;         // 1 = image evaluation reads the donor through a
                       // top-Legendre-mode projection (feedback-loop damping)
  i32  ibCut;          // 1 = cut elements are ghosts (design rule); 0 = FV
                       // center-in-solid rule (A/B; leaks at high Mach)
  real vortexU0;       // isentropic-vortex advection velocity u0=v0
  real greshoP0;       // Gresho background pressure 1/(gam*Ma^2) (sets the Mach number)
  real simT;           // absolute simulation time (time-dependent weak BCs)
  i32  iter;

  real *globalScale;   // [6] managed: GLL-weighted sums of u_i (5) + total volume (1)
                       //     (scaleMode 1: running maxima of |u_i|)
  real *cScale;        // [5] managed: the c_i actually used by the indicator
  i32  *chgCnt;        // [1] managed: grading-fixpoint change counter
  i32  *ibCnt;         // [IB_CNT_N] managed: IB fill diagnostics (no-donor, retries)

  // perf accounting (microseconds, accumulated across the run; printPerf())
  long tVoteUs;      // scales + restrict/detail/vote + neighbor rule + buffer
  long tGradeUs;     // grading fixpoint + merge verdict/apply rounds
  long tSpawnUs;     // spawn + prolong/restrict fills + delete
  long tSortUs;      // sortBlocks (thrust sort + hash + nbr rebuild + gather)
  long tDtUs;        // dgLamKernel + min-reduction
  long tRkUs;        // 3 RHS + RK stages + positivity
  i32  nAdapts, nSortsSkipped, nGradePasses, nMergeRounds;
  void printPerf(void);

  DgSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nDgFields) {
    leafMode    = 1;    // leaf-only grid: no exterior blocks, all interior cells ACTIVE
    sortCurve   = 1;    // Hilbert/Morton memory order: ~3% RHS gather locality win
    deltaT      = 0.0;
    cfl         = 0.4;
    cThr        = 16.0;
    gammaThr    = 1.0;
    epsOverride = -1.0;
    refineFac   = 2.0;
    adaptEvery  = 4;
    refineBuffer = 1;
    shockRefine  = 0;    // the MRA detail sensor pulls shocks to the finest level
    shockThresh  = 0.5;  // on its own; the Ducros refine is an optional belt-and-braces
    icDelta      = 0.5;
    ecVolume    = 1;
    avOn        = 1;
    avCav       = 0.25;
    avKsensor   = 0.1;
    avPen       = 1.0;
    sensorType  = 2;
    ppS0        = -4.0*log10((real)dgOrder);
    ppKappa     = 2.0;
    bcType      = 3;
    icType      = 0;
    scaleMode   = 0;
    indicator   = 0;
    ppRefine    = 0.5;
    ppCoarsen   = 0.01;
    staticGrid  = 0;
    refineRadius = 0.25;
    dmrShockPos  = 1.0/6.0;
    ibOn        = 0;
    ibX = ibY   = 0.0;
    ibR         = 0.5;
    machInf     = 3.0;
    ibBand      = 3.0;
    ibCurv      = 1;
    ibOrder     = 2;
    ibImageFac  = 1.5;
    ibShockTheta = 0.5;
    ibPen       = 0.0;
    ibGraze     = 0.0;
    ibFillEvery = 0;
    ibFilt      = 0;
    ibCut       = 1;
    vortexU0    = 0.0;
    greshoP0    = 1.0/(dgGam*0.01);   // Ma = 0.1
    simT        = 0.0;
    iter        = 0;
    tVoteUs = tGradeUs = tSpawnUs = tSortUs = tDtUs = tRkUs = 0;
    nAdapts = nSortsSkipped = nGradePasses = nMergeRounds = 0;
    cudaMallocManaged(&globalScale, 6*sizeof(real));
    cudaMallocManaged(&cScale, 5*sizeof(real));
    cudaMallocManaged(&chgCnt, sizeof(i32));
    cudaMallocManaged(&ibCnt, IB_CNT_N*sizeof(i32));
    cudaMemset(globalScale, 0, 6*sizeof(real));
    cudaMemset(chgCnt, 0, sizeof(i32));
    cudaMemset(ibCnt, 0, IB_CNT_N*sizeof(i32));
    for (i32 i = 0; i < 5; i++) cScale[i] = 1.0;
  }

  ~DgSolver(void) {
    cudaFree(globalScale);
    cudaFree(cScale);
    cudaFree(chgCnt);
    cudaFree(ibCnt);
  }

  void initialize(void);
  void buildInitialGrid(bool doPaint);   // base grid + IC + refine/re-IC cascade
  real step(real tStep);

  void adaptLeaves(void);       // the leaf-only vote/grade/spawn/fill/prune cascade
  void computeDeltaT(void);
  void setInitialConditions(void);
  void sortFieldData(void) override;     // gather the 5 evolved slabs through the Q0 bank
  void computeImageData(i32 f) override; // paint via LGL->pixel Lagrange interpolation

  // the eps_L threshold on the finest level (Remark 1 heuristic or override)
  real epsFinest(void) const {
    if (epsOverride > 0) return epsOverride;
    real hL = domainSize[0] / (baseGridSize[0] * powi(2, nLvls-1));
    return cThr * pow(hL, gammaThr);
  }

  // diagnostics / validation
  void dgTotalConserved(double &mass, double &momx, double &energy);  // GLL-weighted domain totals
  void writeLineProfile(const char *fileName);   // y/z-midline nodal profile (x, rho, u, p)
  void computeVortexError(real t);               // L2 rho error vs the exact advected vortex
  void computeGreshoError(void);                 // L2 velocity error + KE retention vs exact (GLL-weighted)
  real maxDeviationFromUniform(void);            // max_i |Q_i - Q_i(free stream)| (M3 test)
  void paintPressure(const char *fileName);
  void ibClassify(void);                         // geometry classes (2 kernels), post-sort
  void ibFill(void);                             // Hermite ghost reconstruction
  void computeIbGates(void);                     // standoff + stagnation pressure + Cd
  void writeIbSurface(const char *fileName);     // Cp(theta) around the cylinder
  void paintIbClass(const char *fileName);       // debug: class map
  void paintSensor(const char *fileName);        // per-element Ducros sensor (or MRA detail)
  bool selfTest(void);                           // host operator identities (double precision)

};

#endif
