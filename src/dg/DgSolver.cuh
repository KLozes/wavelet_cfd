#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include "MultiLevelSparseGrid.cuh"
#include "SayeQuad.h"   // SayeNode: the cut-cell rule pools below

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
// IB_CUT exists only under --ibevolve 1: a CUT element (wall crosses it)
// whose fluid-side nodes (phi > 0, split EXACTLY at 0) EVOLVE with the DG
// RHS while its solid-side nodes keep the FRIB fill.  Counts as live for
// RHS/RK/dt/sensor/positivity, as non-fluid for donors/details/metrics.
enum IbClass { IB_FLUID = 0, IB_GHOST = 1, IB_DEAD = 2, IB_CUT = 3 };
// managed diagnostic counters (ibCnt[])
enum IbCnt { IB_CNT_NODONOR = 0, IB_CNT_RETRY1 = 1, IB_CNT_FALLBACK = 2, IB_CNT_N = 4 };
static constexpr i32 nDgFields  = 17;
static constexpr i32 CUT_NBMAX_H = 20;   // total-degree P^3 in 3-D (host mirror)

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
  // Flux reconstruction on GAUSS-LEGENDRE solution points (--gauss).  On the
  // default LOBATTO nodes the correction is vacuous (Huynh's g_HU' vanishes at
  // every interior node and equals -1/w0 at the boundary, so FR-g_HU IS the
  // nodal DGSEM boundary lift -- proven in the selftest).  On GAUSS nodes the
  // solution points are interior: every face trace is an INTERPOLATION (entropy-
  // projected) and the interface correction distributes to ALL nodes via a
  // correction-function derivative g'(xi_i).  The volume stays EC flux-
  // differencing (generalized SBP: Q+Q^T = tR tR^T - tL tL^T holds on Gauss).
  real nsfr;           // NSFR residual filter strength sigma in [0,1) (arXiv
                       // 2507.09131 Srinivasan/Nadarajah): the ESFR correction
                       // K = 2c (D^p)^T M D^p is rank-1 per line, so
                       // (M+K)^-1 M R = R - sigma*phi*(sum w phi R) -- a LINEAR
                       // state-independent top-mode damping of the RESIDUAL,
                       // applied per direction (dimension-split) to the fully
                       // assembled RHS.  sigma = 2cC^2/(1+2cC^2) maps the paper's
                       // c-table (c_DG: 0, c_HU ~ 0.02, c_+x10 ~ 0.3).  Exactly
                       // conservative (sum w phi = 0), free-stream exact; pairs
                       // with the Zhang-Shu positivity limiter as the ONLY other
                       // mechanism (the paper's shock-capturing recipe).  DEFAULT 0.3
                       // (2026-07-14: free on smooth flow, best shock profiles).
  real bulkC;          // BULK (dilatation-only) viscosity strength C_b, gated by
                       // the EXISTING shock sensor (dgAvNuKernel -> SCRATCH slot
                       // 1; --sensor 0 = pure Ducros).  beta = C_b * theta_e *
                       // (h/N) * lam_e * rho -- the same magnitude the AV uses,
                       // applied ONLY to the dilatational flux: the conservative
                       // viscous term d/dx_m [beta divu] (momentum) +
                       // d/dx_m [beta divu u_m] (energy) via the AV's weak-
                       // divergence operator.  Shear layers and contacts are
                       // untouched (no full-field Laplacian).  Complement to
                       // --dpsbp where top-mode damping alone cannot hold a
                       // multi-D shock.  0 = off.
  real dpFace;         // DP-SBP INTERFACE upwind flux strength (the alpha (B_I+B_n)
                       // g surface half of arXiv 2411.06629): > 0 REPLACES the HLLC
                       // face flux with the paper's flux-splitting upwind flux,
                       // central + Gamma-scaled entropy-variable jump (Eq 17).
                       // 1 = paper strength.  Active when dpSbp > 0.  Default 0
                       // (keep HLLC; an ADDITIVE Gamma penalty on top of HLLC
                       // detonates in one step through the 1/w0 lift -- measured).
  real dpSbp;          // dual-pairing upwind SBP volume dissipation strength tau
                       // (arXiv 2411.06629, Stewart/Lee/Duru): adds the intrinsic
                       // volume upwind term (1/2) Gamma (D+ - D-) g to the RHS,
                       // g = thermodynamic-entropy variables, Gamma = per-element
                       // upwind parameters gamma_i = max_x lam/(d2 eta/dU_i2) with
                       // the paper's M^2 gate on the energy channel.  (D+ - D-) is
                       // realized as the rank-1 top-Legendre-mode damping
                       // -tau*phi*(phi^T H g): symmetric negative (A.4), conserva-
                       // tive (sum w_i phi_i = 0, selftest), entropy-dissipative,
                       // degree p-1 exact -- a valid DP-SBP pair on either node
                       // set.  The paper runs Euler shocks with NO artificial
                       // viscosity / limiting / subcell FV: test --av 0 --subfv 0
                       // --dpsbp 0.1.  0 = off (bit-identical scheme).
  i32  gauss;          // 1 = Gauss-Legendre points + flux reconstruction; 0 = collocated Lobatto DGSEM
  i32  frType;         // Gauss FR correction fn: 0 = g_DG (Radau, = nodal DG,
                       // strictly entropy-stable with the SBP lift); 1 = g_HU
                       // (Huynh g2, wider explicit stability) [default].  On
                       // Lobatto both collapse to the DGSEM boundary lift.
  i32  esLim;          // fully-discrete entropy-stable limiter (docs/
                       // EntropyStableDG.pdf): per stage, cap the quadrature
                       // cell entropy by the proper-entropy-flux bound and
                       // enforce by Zhang-Shu scaling toward the mean.
                       // 0 off; 1 limit everywhere (holds p=3 M=3: standoff
                       // -0.32%; costs smooth accuracy); 2 sensor-gated
                       // (smooth-exact, but does NOT hold the p=3 M=3 rear).
                       // See dgEntropyLimitKernel for the measured matrix.
  i32  mood;           // 1 = a-posteriori MOOD limiter (no a priori shock
                       // sensor / AV): each RK stage attempts the pure DG
                       // update, DETECTS failed cells (non-finite, or density/
                       // pressure below a relative floor), and LOCALLY
                       // recomputes only those with the first-order FV volume
                       // (Rusanov subcell); HLLC faces are unchanged so the
                       // recompute never perturbs neighbours (local cascade
                       // DG -> FV).  Replaces the sensor-driven alpha.
  real moodRho;        // MOOD density floor, relative to cScale (freestream)
  real moodP;          // MOOD pressure floor, relative to freestream 1/gam
  i32  rusFace;        // 1 = Rusanov (local Lax-Friedrichs) element-interface
                       // flux instead of HLLC everywhere -- more dissipative
                       // but vacuum-robust (HLLC's intermediate-wave structure
                       // breaks at the near-vacuum wall).  MOOD stays local
                       // (the face TYPE is fixed; only the volume is redone).
  i32  subFv;          // 1 = subcell-FV shock capturing (Hennemann/Gassner,
                       // docs/subcellFV.pdf): blend the DG volume with a
                       // first-order Rusanov FV volume on the LGL subgrid,
                       // a = min(subMax, theta_e) per element.  An ALTERNATIVE
                       // to the artificial viscosity (run --av 0 --subfv 1);
                       // dissipation acts at SUBCELL resolution, so it catches
                       // the single-node startup spikes AV was needed for.
  real subMax;         // cap on the FV blend factor a (paper alpha_max ~ 0.5)
  real subThr;         // FV sensor deadband: theta <= subThr stays pure
                       // high-order (NSFR filter carries mild ringing); above,
                       // alpha = (theta-subThr)/(1-subThr) rescaled so a
                       // saturated sensor still reaches full blend.  0 = the
                       // original alpha = theta gate.
  real subFloor;       // amplitude floor for the shock sensor (relative density
                       // fluctuation): a cell is troubled (refine OR FV-blend)
                       // only if its fluctuation modal amplitude exceeds
                       // subFloor*cScale.  The Persson theta is a scale-free
                       // RATIO -- without this the low-amplitude wake refines
                       // on roundoff wiggle (measured: 4x cell count, +7.7%
                       // mass from positivity flooring the over-refined wake)
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
  // ---- CUT-CELL DG (--cutcell 1) -----------------------------------------
  // A SECOND ELEMENT TYPE, not a modification of the Cartesian path.  Cartesian
  // elements keep collocation, the diagonal GLL mass and tensor sum
  // factorization; a cut element has none of those and carries dense operators
  // built once by cutElemBuild() (src/common/CutElem.h).
  //
  // The cut element's STATE still lives in the ordinary nodal fieldData, so RK,
  // positivity, MOOD, output and state redistribution all work on it unchanged.
  // dgRhsCutKernel converts nodal -> modal, applies the operators, applies the
  // dense M^-1, and writes a nodal RHS back.
  //
  // FACE COUPLING: a cut face's quadrature points are not the tensor face nodes,
  // so the two sides must agree on a rule.  The CUT element owns it and computes
  // BOTH sides of the flux, depositing the neighbour's share by atomicAdd; the
  // Cartesian kernel simply SKIPS any face whose neighbour is cut.  That keeps
  // the exchange conservative by construction and leaves dgRhsKernel almost
  // untouched.
  i32  cutOn = 0;          // 1 = cut-cell path active
  i32  cutHvMean = 0;      // DIAGNOSTIC: let the hyper-viscosity touch mode 0
  real cutHv = 0;          // --cuthv: UNGATED high-order modal hyper-viscosity
                           // on cut elements (see dgRhsCutKernel).  Every other
                           // dissipation here is shock-sensor gated and is
                           // therefore absent exactly where the smooth
                           // instability lives.
  real cutEps = 0;         // global level-set shift applied at sampling time to
                           // step off an exactly-tangent body (see the tangency
                           // guard in DgCutBuild.cu).  0 = geometry is generic.
  i32  cutModal = 0;       // 1 = a cut element's field slots hold MODAL
                           // COEFFICIENTS, not nodal values.  A cut element's
                           // polynomial is supported on {phi>0} only, but the
                           // tensor Lobatto nodes include points buried in the
                           // solid, and evaluating an orthonormal basis there
                           // costs a factor of 449 (measured: max|psi~| 54 over
                           // the fluid rule vs 2.4e+04 at the nodes).  Storing
                           // coefficients means the basis is never evaluated
                           // outside its own support -- the same invariant
                           // Taylor & Chan get by putting their Fekete nodes
                           // inside the cut region.
  i32  cutDbgMask = 15;    // DEBUG term mask: 1 volume, 2 faces, 4 wall, 8 deposit
  i32  cutWallRiem = 0;    // 1 = solid wall as a Riemann problem against the
                           // mirror state (enforces u.n = 0 weakly); 0 = the
                           // legacy pressure-only flux, which does not
  i32  cutFsp = 0;         // DEBUG: transparent wall (exact F.n of the trace)
                           // for the in-solver free-stream gate; free stream is
                           // NOT a solid-wall solution, so gating it against the
                           // reflective wall would test the wrong thing
  i32 *cutDbg = nullptr;   // [2] {starved faces, conforming faces} -- see the
                           // starvation counter in dgRhsCutKernel
  i32  nCutElem = 0;       // number of cut elements
  i32  cutNb = 0;          // modal basis size (total degree N)
  i32  *blkCut = nullptr;  // [nBlocksMax] cut index of a block, or -1
  i32  *cutBlk = nullptr;  // [nCutElem]   owning block of a cut element
  i32  *cutNbOf= nullptr;  // [nCutElem]   modes actually carried (degree may be
                           //              REDUCED on degenerate slivers)
  i32  *cutNbLo= nullptr;  // [nCutElem]   low-degree mode count for the decay sensor
  real *cutM11 = nullptr;  // [nCutElem*CUT_NBMAX^2] low-degree sub-mass inverse
  real  cutEta = (real)0.05;  // modal-decay trouble threshold (--cuteta)
  real *cutCen = nullptr;  // [nCutElem*4] basis centroid (3) + scale (1)
  real *cutLc  = nullptr;  // [nCutElem*CUT_NBMAX^2] Cholesky factor L of the
                           // monomial mass.  The kernel works in the ORTHONORMAL
                           // basis psi~ = L^-1 psi, where the mass is exactly I:
                           // projection is a direct weighted sum, the RHS needs
                           // no solve, and the decay sensor is exact -- the
                           // degree-major Cholesky NESTS, so the first k rows
                           // span the low-degree space
  real *cutQual= nullptr;  // [nCutElem] bndIncons -- per-element geometry quality
  real *cutWallN= nullptr; // [nCutElem*2] mean wall normal (x,y) -- the
                           // characteristic limiter's eigendirection is the
                           // wall TANGENT, per Giuliani SISC 2022
  // rule pools, reference coords, CSR-offset addressed
  SayeNode *cutVolP = nullptr;  i32 *cutVolOff = nullptr;   // [nCutElem+1]
  SayeNode *cutWalP = nullptr;  i32 *cutWalOff = nullptr;   // [nCutElem+1]
  SayeNode *cutFacP = nullptr;  i32 *cutFacOff = nullptr;   // [6*nCutElem+1]
  real *cutFacA = nullptr;      // [6*nCutElem] fluid area of each cut face --
                                // ~1 selects the conforming full-face path

  // ---- ENTROPY-STABLE cut operators (--cutes), Taylor & Chan arXiv:2412.13002 -
  // Built once on the host from the same CutElemOps the baseline path uses, then
  // uploaded.  The surface rule IS the runtime interface rule -- the tensor GLL
  // nodes on any fully-fluid face, the Saye rule on a partial one, the Saye wall
  // rule -- because the flux-differenced surface term only telescopes against the
  // volume term if the two use the same rule, and because a shared face must be
  // integrated identically by both sides or the coupling stops conserving.
  i32   cutEs   = 0;            // --cutes: 1 = entropy-stable cut RHS
  i32   esDbg   = 0;            // ES_CLOSED|ES_NODEPOSIT|ES_NOMETRIC bisection
  i32  *esQOff  = nullptr;      // [nCutElem+1] volume-point CSR offsets
  i32  *esFOff  = nullptr;      // [nCutElem+1] surface-point CSR offsets
  real *esVq    = nullptr;      // [esQOff[n]*CUT_NBMAX] psi~ at volume points
  real *esDVq   = nullptr;      // [3*esQOff[n]*CUT_NBMAX] d psi~/dx_d there
  real *esVf    = nullptr;      // [esFOff[n]*CUT_NBMAX] psi~ at surface points
  real *esWq    = nullptr;      // [esQOff[n]] volume weights (reference measure)
  real *esWf    = nullptr;      // [esFOff[n]] surface weights
  real *esNrm   = nullptr;      // [3*esFOff[n]] outward normal at surface points
  real *esQ     = nullptr;      // [3*sum(nq^2)] Q_d = W dVq_d Pq  (dense, per elem)
  i32  *esQ2Off = nullptr;      // [nCutElem+1] offsets into esQ (units of nq^2)
  real *esEmat  = nullptr;      // [sum(nf*nq)] E = Vf Pq
  i32  *esEOff  = nullptr;      // [nCutElem+1] offsets into esEmat
  i32  *esOwner = nullptr;      // [esFOff[n]] 0..5 = cut face, 6 = wall
  i32  *esNode  = nullptr;      // [esFOff[n]] neighbour tensor-node index on a
                                // full-face GLL point, else -1
  real *esXf    = nullptr;      // [3*esFOff[n]] surface point reference coords
  real *esVtil  = nullptr;      // [nCutElem*CUT_NBMAX*5] entropy-variable modal
                                // coefficients, PUBLISHED so that a cut element
                                // reading a cut neighbour's trace sees the same
                                // entropy-projected state that neighbour uses.
                                // Both sides of a shared face must evaluate the
                                // SAME pair or the single-valued-flux property
                                // -- and with it conservation -- is lost.
  double esGcl  = 0;            // worst Eq-47 residual over the elements

  i32  cutZ2d = 0;     // --cutz2d: zero z-dependent cut modes in a pseudo-2D run
  i32  cutPos = 1;     // --cutpos: Zhang-Shu positivity limiter on cut elements
                       // (modal form -- see dgCutPositivityKernel)
  i32  cutHvGate = 0;  // --cuthvgate 1: gate the cut modal filter on the Persson
                       // modal-decay ramp instead of applying it everywhere
  i32  cutFlux = 0;    // --cutflux 1: cut faces use dgIfaceFlux (HLLC), matching
                       // the Cartesian mesh; 0 = the legacy hardcoded Rusanov
  real srdVolFrac = 0; // --srdvol: SRD small-cell threshold as a fraction of a
                       // background volume (0 = keep StateRedistribution's 0.5)
  i32  subBc = 0;      // --subbc: experimental subsonic characteristic BCs for
                       // bcType 5 (NOT validated -- see the note in dgBcState)
  i32  ibOn;           // 1 = ghost-element immersed boundary active (cylinder SDF)
  real ibX, ibY;       // cylinder center
  real ibR;            // cylinder radius
  real machInf;        // freestream Mach (case 9; a_inf = 1 normalization)
  real ibBand;         // force-finest band half-width, in finest-element units
  i32  ibCurv;         // 1 = curvature wall conditions (du_t/ds = -u_t/R in
                       // the FRIB wall solve; 0 = flat-wall HO-i/f, which the
                       // paper shows is LOW order on curved bodies)
  real ibShockTheta;   // donor sensor theta above which a FRIB node drops to
                       // the LO fallback (H/S are smooth invariants -- they
                       // jump across shocks; raw FRIB died at M=3 iter 5)
  real ibPen;          // ungated fraction of lambda_e ghosts publish as their
                       // face-penalty scale (wall-face jump damping)
  real ibGraze;        // elements closer than ibGraze*h to the wall become
                       // ghosts even when not cut (grazing-sliver guard)
  i32  ibFillEvery;    // 0 = refill ghosts every RK stage; 1 = once per step
  i32  ibFilt;         // 1 = image evaluation reads the donor through a
                       // top-Legendre-mode projection (feedback-loop damping)
  i32  ibCut;          // 1 = cut elements are ghosts (design rule); 0 = FV
                       // center-in-solid rule (A/B; leaks at high Mach)
  i32  ibEvolve;       // 1 = cut elements JOIN the discretization (class
                       // IB_CUT): their fluid-side nodes (phi > 0, exact
                       // split) evolve with the DG RHS; solid-side nodes
                       // keep the FRIB fill, whose image line samples only
                       // NON-CUT fluid donors (the existing march).  The
                       // effective wall moves from the ghost-layer envelope
                       // to the true surface (mirror-era ibevolve measured
                       // standoff -2.1% vs +5.7% at p2).  The old p3
                       // linear instability lived in the H/S fill coupling;
                       // re-test under the primitive line.  Requires
                       // ibCut 1 and the FRIB path (ibSbm 0).  Default 0.
  i32  ibLimit;        // 1 = MUSCL monotonicity limiter on the FRIB image-line
                       // reconstruction (clamp each field to the wall+sample
                       // range): stops the high-order polynomial ring in H/S
                       // that detonates the near-vacuum/high-res wall
  i32  ibSbm;          // 3 = GHOST-FREE FRIB WALL FLUX: the FRIB image-line solve
                       // done per wall-face quadrature point from THE OWNING
                       // ELEMENT's polynomial + levelset (dgIbFluxTrace) --
                       // FRIB physics with SBM locality; no fill/locate/march.
                       // 1 = Shifted Boundary Method wall (no ghost cells): the
                       // active domain is the UNCUT fluid elements; every face
                       // onto a cut/solid (inactive) element is a surrogate WALL
                       // face, imposed as a reflective flux from the interior
                       // trace mirrored about the TRUE radial wall normal.  No
                       // FRIB reconstruction, no piston, no ghost fill is read.
  i32  ibBrink;        // 1 = pressure-tight volume-penalization IB (Reiss 2021,
                       // docs/pressureTIghtBrinkman.pdf): the object is a reduced
                       // volume fraction phi (no ghost cells, no cut cells).  Two
                       // bounded, non-stiff mechanisms are added to standard Euler:
                       //  (a) the flux-form momentum source p*grad(phi) in the
                       //      smeared interface -- a wall reaction pushing fluid
                       //      out of the body, maximal at the stagnation point;
                       //  (b) Darcy drag -chi*(rho u) in the solid interior only
                       //      (phi==eps plateau), freezing the plug so the stream
                       //      cannot advect through.  phi=1 fluid, eps in solid.
  real ibBrinkEps;     // volume fraction inside the object (~1e-4..1e-8)
  real ibBrinkDelta;   // full width of the phi transition, in FINEST cells (~2)
  real ibBrinkRate;    // Darcy drag rate as a fraction of the CFL-stable rate
                       // lam*NNODE/h ("as big as the timestep permits"), ~1
  real ibSbmCurv;      // SBM wall curvature (centripetal) coefficient: the wall
                       // pressure is lowered by ibSbmCurv * rho u_t^2 / R * h
                       // (the FRIB curvature term dp/dn = -rho u_t^2/R), with
                       // u_t taken from FLUID-only data.  0 = flat wall.
  i32  ibShift2;       // 1 = SECOND-order Taylor in the SBM velocity shift:
                       // u_wall ~ u + (d.grad)u + 1/2 (d.grad)^2 u (normal-normal
                       // Hessian from the element polynomial).  A/B knob: the
                       // FRIB --ibord 3 lesson says 2nd derivatives amplify
                       // node-scale noise -- measure, don't assume.  Default 0.
  i32  recov;          // 1 = binary-recovery interface flux on conforming
                       // same-level fluid faces: central flux at the L2
                       // moment-matched quintic trace (Van Leer recovery,
                       // weights [-3,12,23,23,12,-3]/64), Rusanov jump
                       // dissipation scaled by recovK (uniform grids)
  real recovK;         // recovery dissipation fraction (default 0.1)
  real ibSbmPen;       // SBM slip-wall Nitsche penalty coefficient (alpha_slip,
                       // paper default 0.2): weight on c_s*rho*u_n in the wall
                       // momentum flux that drives the SHIFTED normal velocity
                       // to zero at the true wall.
  i32  ibSingle;       // SINGLE-IP FRIB (one donor element, IP at max depth):
                       // 0 = off (multi-IP paper line).  1 = STATE-only: linear
                       // per-field lines wall-BC <-> IP (LO + proper primitive
                       // Neumann/curvature BCs).  2 = STATE+GRADIENT: quadratic
                       // Hermite (wall BC + value + normal slope at the IP) --
                       // the mirror-era measured-optimal order, one clean donor,
                       // no multi-element seams, no march flips, MGPU-local.
  i32  ibRecon;        // FRIB image-line variable set: 0 = (u_n,u_t,w_t,H,S)
                       // (paper; H,S smooth invariants, zero-gradient wall BCs
                       // -- but the back-conversion T*=(g-1)/g(H-q^2/2) CANCELS
                       // large numbers at M=3 and rho=(T*/S)^2.5 is a power-law
                       // amplifier: the documented shock-gate/ring fragilities).
                       // 1 = (u_n,u_t,w_t,p,rho) PRIMITIVE: linear errors,
                       // directly clampable; wall BCs dp/dn = rho u_t^2/R
                       // (centripetal) and drho/dn = dp/dn / a^2 (linearized
                       // isentropy).  Trades smooth-invariant superconvergence
                       // for shock/vacuum robustness.
  i32  ibPiston;       // 1 = exact wall-Riemann (piston) star state for shocked
                       // COMPRESSION ghost nodes (the M=3 bow-shock reflector);
                       // 0 = those nodes drop to the bounded LO fallback like
                       // outflow nodes (A/B: the piston's p* <= 50 p_I states
                       // are the strongest the fill produces -- prime suspect
                       // in the Gauss-points M=3 slam instability)
  real ibDil;          // FRIB image-line length override in units of h
                       // (0 = paper Eq 22 default: 3h at p=2, 5.5h at p=3)
  i32  ibDbg;          // TEMP: 1 = per-path fill trace for the nose ghost block
  i32  ibHO;           // wall reconstruction mode.
                       // 1 = high-order FRIB image-line reconstruction
                       //     (multi-point polynomial + slope BC) [default].
                       // 0 = FIRST ORDER: one interpolated image point + wall
                       //     value, NO SLOPE -- H,S constant = image, u_n linear
                       //     to 0 at the wall, rho/p from (H,S,|u|).
                       // 2 = PAPER wall model (Qi et al. 2024, Eq 19): one
                       //     near-wall image point + algebraic curvature
                       //     correction at the clamped near-wall trace, NO deep
                       //     polynomial and NO piston/LO ladder -- robust by
                       //     construction (bounded u_n, centripetal wall p from
                       //     a floored image p, isentropic rho).
  // The wall method is FRIB HO-i/c (Funada & Imamura, C&F 2023; docs/
  // FRIB.pdf) -- see dgIbFillKernel.  The mirror-world Hermite fill, the
  // --ibevolve per-node hybrid, --ibord/--ibimagefac/--ibrho knobs were
  // REMOVED at user direction 2026-07-13; the project memory keeps their
  // full measured matrix and reimplementation keys.
  real vortexU0;       // isentropic-vortex advection velocity u0=v0
  real greshoP0;       // Gresho background pressure 1/(gam*Ma^2) (sets the Mach number)
  real simT;           // absolute simulation time (time-dependent weak BCs)
  i32  iter;
  real fluxAvgT0, fluxAvgT1;   // boundary-flux time-average window (disabled if
  double fluxAvgAcc[4];        // T1<=T0); dt-weighted accumulators + total time
  double fluxAvgTime;

  real *globalScale;   // [6] managed: GLL-weighted sums of u_i (5) + total volume (1)
                       //     (scaleMode 1: running maxima of |u_i|)
  real *cScale;        // [5] managed: the c_i actually used by the indicator
  i32  *chgCnt;        // [1] managed: grading-fixpoint change counter
  i32  *nanCnt;        // [1] managed: first-NaN probe one-shot latch (--debug 3)
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
    nsfr        = 0.3;
    bulkC       = 0.0;
    dpSbp       = 0.0;
    dpFace      = 1.0;
    gauss       = 0;
    frType      = 1;
    esLim       = 0;
    mood        = 0;
    moodRho     = 1e-6;
    moodP       = 1e-6;
    rusFace     = 0;
    subFv       = 0;
    subMax      = 0.5;
    subThr      = 0.0;
    subFloor    = 0.01;
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
    ibShockTheta = 0.5;
    ibPen       = 0.0;
    ibGraze     = 0.0;
    ibFillEvery = 0;
    ibFilt      = 0;
    ibCut       = 1;
    ibLimit     = 1;
    ibHO        = 1;
    ibSbm       = 0;
    ibBrink     = 0;
    ibBrinkEps  = 1e-4;
    ibBrinkDelta= 2.0;   // full transition width, finest-cell units
    ibBrinkRate = 1.0;
    ibSbmCurv   = 1.0;
    ibSbmPen    = 0.2;
    recov       = 0;
    recovK      = 0.1;
    ibShift2    = 0;
    ibSingle    = 0;
    ibRecon     = 0;
    ibPiston    = 1;
    ibDil       = 0.0;
    ibEvolve    = 0;
    ibDbg       = 0;
    vortexU0    = 0.0;
    greshoP0    = 1.0/(dgGam*0.01);   // Ma = 0.1
    simT        = 0.0;
    iter        = 0;
    fluxAvgT0 = 0.0; fluxAvgT1 = -1.0;   // window disabled by default
    fluxAvgAcc[0]=fluxAvgAcc[1]=fluxAvgAcc[2]=fluxAvgAcc[3]=0.0;
    fluxAvgTime = 0.0;
    tVoteUs = tGradeUs = tSpawnUs = tSortUs = tDtUs = tRkUs = 0;
    nAdapts = nSortsSkipped = nGradePasses = nMergeRounds = 0;
    cudaMallocManaged(&globalScale, 6*sizeof(real));
    cudaMallocManaged(&cScale, 5*sizeof(real));
    cudaMallocManaged(&chgCnt, sizeof(i32));
    cudaMallocManaged(&nanCnt, sizeof(i32));
    cudaMemset(nanCnt, 0, sizeof(i32));
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
    cudaFree(nanCnt);
    cudaFree(ibCnt);
  }

  void initialize(void);
  void buildInitialGrid(bool doPaint);   // base grid + IC + refine/re-IC cascade
  real step(real tStep);

  void adaptLeaves(void);       // the leaf-only vote/grade/spawn/fill/prune cascade
  void buildCutElems(void);     // cut-cell preprocessing: classify, build the dense
                                // operators once, upload.  Requires a STATIC wall band.
  void probeCutRhs(void);       // one RHS apply + per-class |RHS| report (debug)
  void buildSrd(void);          // build the SRD operator (once, after buildCutElems)
  void redistributeFlux(void);  // FLUX redistribution (Chern-Colella): a small
                                // cut member's RHS is clipped to its merge
                                // neighbourhood's volume-weighted rate and the
                                // EXCESS deposited into the partners -- exactly
                                // conservative, removes the 1/vol update spike
                                // at the source.  Runs between DG_RHS and the
                                // RK stage kernel.
  void applyCutLimiter(void);   // characteristic Barth-Jespersen on cut elements
                                // (Giuliani SISC 2022) -- wall-tangent eigenframe,
                                // range condition vs neighbour means, post-SRD
  void buildSrdDevice(void);    // flatten + upload the SRD operator (once)
  void applySrdDevice(void);    // the three-kernel device apply (no host sync)
  void applySrd(void);          // stage-wise state redistribution on the HOST via
                                // managed memory -- microseconds at wall-band size;
                                // port to a kernel only if a profile says so
  // ---- SRD ON THE DEVICE --------------------------------------------------
  // The host apply (applySrd) costs a full device sync plus a serial gather /
  // project / scatter three times per step.  MEASURED on case 9 at only 12 cut
  // elements, M=0.2, t=2: 11.40 s with SRD+FRD, 10.34 s SRD-only, 7.48 s with
  // neither -- i.e. SRD alone was 28% of the run, and the SRD element set grows
  // with the wall band (52 cut elements at h=0.0625, plus two neighbour rings).
  // Everything the operator needs is built ONCE and is small, so it uploads and
  // the apply becomes three kernels with no host round trip.
  // Flat arrays rather than the SrdElem/SrdBasis structs so this header does not
  // have to include StateRedistribution.h.
  i32  srdNE = 0, srdNb = 0, srdDeg = 0;   // elements, modes/neighbourhood, degree
  i32  *srdBlk  = nullptr;   // [nE]      block index of each SRD element
  double *srdX0 = nullptr;   // [3*nE]    element lower corner (physical)
  double *srdH  = nullptr;   // [3*nE]    element size per axis
  i32  *srdQOff = nullptr;   // [nE+1]    slice of the quadrature pool
  SayeNode *srdQ = nullptr;  // [qTot]    quadrature pool (reference coords)
  i32  *srdMOff = nullptr, *srdM = nullptr;   // merge neighbourhoods, CSR
  i32  *srdCOff = nullptr, *srdC = nullptr;   // reverse map (who projects onto i)
  i32  *srdCcnt = nullptr;   // [nE]      |C_k|
  char *srdTriv = nullptr;   // [nE]      neighbourhood is {k}: identity
  double *srdBas  = nullptr; // [4*nE]    neighbourhood centroid (3) + scale
  double *srdChol = nullptr; // [nE*nb*nb] factored neighbourhood mass
  double *srdCoef = nullptr; // [nE*nb*5]  scratch: Pi_k u coefficients
  double *srdU    = nullptr; // [nE*blockSizeTot*5] gathered nodal state
  i32  srdOnDev = 0;         // 1 = device arrays built; CUT_SRDHOST=1 forces host

  struct DgSrd *srd = nullptr;  // opaque SRD state (DgCutBuild.cu); block indices
                                // captured at build time, so the band must stay
                                // STATIC and unsorted (same constraint as the cut
                                // operators)
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
  void paintBrinkPhi(const char *fileName);      // Brinkman volume fraction phi(x)
  void ibClassify(void);                         // geometry classes (2 kernels), post-sort
  void ibFill(void);                             // Hermite ghost reconstruction
  void computeIbGates(void);                     // standoff + stagnation pressure + Cd
  void cutToModal(void);                         // nodal -> modal for cut blocks
  double cutMaxDeviation(const double U0[5]);    // uniform-state check ON the
                                                 // fluid region, not the nodes
  void patchCutImage(i32 f);                     // repaint cut elements from their
                                                 // own polynomial (see DgCutBuild.cu)
  double dgResidualNorm(void);                   // ||dU/dt||_2 / ||U||_2 over the
                                                 // fluid, the steady-state monitor
  void buildCutEs(const void *opsVec);            // ES operators from CutElemOps
  void dgCutConserved(double &mass, double &momx, double &energy);
                                                 // totals over the CUT band only,
                                                 // integrated over the FLUID region
  void writeCutFields(const char *stem);         // cut geometry + the modal solution
                                                 // sampled at Saye volume/wall points
  void writeIbSurface(const char *fileName);     // Cp(theta) around the cylinder
  void paintIbClass(const char *fileName);       // debug: class map
  void paintTroubled(const char *fileName);      // shock indicator / FV blend factor
  void boundaryMassFlux(double bnd[4]);          // outward mass flux per domain boundary
  void paintSensor(const char *fileName);        // per-element Ducros sensor (or MRA detail)
  bool selfTest(void);                           // host operator identities (double precision)

};

#endif
