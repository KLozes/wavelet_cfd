#ifndef COMPRESSIBLE_SOLVER_H
#define COMPRESSIBLE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"

// --leaf: the per-stage kernels stop after the exterior group (blocks are sorted
// leaf-bearing / exterior / fully covered), so covered parents cost nothing.
// Identical to START_CELL_LOOP when leafFlux is off.
#define START_LIVE_CELL_LOOP \
  i32 cIdx = blockIdx.x * blockDim.x + threadIdx.x; \
  i32 bIdx = cIdx / blockSizeTot; \
  const i32 bEndLive = grid.leafFlux ? (grid.nLeafBlocks + grid.nExtBlocks) : grid.hashTable.nKeys; \
  while (bIdx < bEndLive) { \
    if (!grid.pseudo2D || (cIdx % blockSizeTot) < blockSize*blockSize) {
#include "Poly.h"
#include "SayeQuad.h"

static constexpr real gam = 1.4;

//
// 3D compressible flow solver: cell-centred finite volume, HLLC flux with
// TVD/ROUND reconstruction, low-storage (Williamson 2N) RK3 in time, on the
// MultiLevelSparseGrid wavelet AMR core.
//
// field layout (nFields = 17).  fields 0-4 and 5-6 alternate between
// conservative and primitive storage in place (see the conservative/primitive
// conversion kernels):
//
//   0 : Rho    | Rho
//   1 : RhoU   | U
//   2 : RhoV   | V
//   3 : RhoW   | W
//   4 : RhoE   | P
//   5 : RhoK   | K      k-tau SST turbulent kinetic energy     (k~)
//   6 : RhoTau | Tau    k-tau SST inverse specific dissipation (tau~ = 1/omega)
//   7..13  : shared scratch bank (see below)
//   14     : DeltaT / MagRhoU / pressure       (scratch, reused)
//   15     : MuT    eddy viscosity                          (RANS only)
//   16     : TF1    SST blending function F1                (RANS only)
//
// The turbulence pair (5,6) is inert unless the RANS model is switched on (--rans):
// with no source/flux contribution their RHS is identically zero and, being memset
// to zero at allocation, they ride the AMR machinery as a pair of zero fields.
//
// Fields 15-16 hold the two closure quantities that the face loop needs on BOTH
// sides of every face and that are too expensive to recompute there: the eddy
// viscosity and the blending function F1 (which sets sigma_k, sigma_omega, the
// (1-F1) mu of the tau~ diffusion and the (1-F1) cross-diffusion weight).  Both
// need velocity gradients and grad(k~).grad(tau~), so computing them once per
// cell in turbClosureKernel and reading them per face costs one extra pass
// instead of six gradient evaluations per cell.  They are dead weight when
// --rans is off; that is the price of a fixed field count.
//
// Time stepping is low-storage (Williamson 2N) RK3, which needs only q plus
// ONE accumulator bank, so the Old and Rhs banks are a single aliased bank
// (F_OLD == F_RHS == 7) whose uses are temporally disjoint:
//   - during the RK stages: the LSRK accumulator S (the RHS kernels
//     ACCUMULATE L into it; updateFields does q += B dt S, then S *= A_next)
//   - between steps (adaptation): the wavelet-transform reference snapshot
//     and the block-sort double-buffer
//   - during a CTU-Hancock step (mdFlux==2): the half-step predicted
//     primitives (the fused single-stage corrector updates q in place and
//     never touches the bank)
// Any 3-stage 3rd-order explicit RK shares the linear stability polynomial,
// so all measured CFL limits are unchanged vs the previous Shu-Osher SSP-RK3;
// only the formal SSP property is given up.
//
// Only fields 0..6 (NEVOLVE) are sorted, restricted, interpolated and wavelet-
// transformed; 7..14 are transient.  The multiD corner-flux path computes its
// corner tensors on the fly (no flux storage fields).
//
enum CompressibleField {
  F_RHO = 0, F_RHOU = 1, F_RHOV = 2, F_RHOW = 3, F_RHOE = 4,
  F_RHOK = 5, F_RHOTAU = 6,
  F_OLD = 7,       // shared bank 7..13 (snapshot / sort buffer / Hancock)
  F_RHS = 7,       // alias: the LSRK accumulator during the RK stages
  F_SCRATCH = 14,
  F_MUT = 15,      // eddy viscosity            (RANS; written by turbClosureKernel)
  F_TF1 = 16,      // SST blending function F1  (RANS; written by turbClosureKernel)
  // Immersed-body geometry cache, stamped by stampIbGeometry() once at init and
  // once per adaptation (the body is STATIC): the level set and the UTCart
  // corner-test fluid mask are pure functions of cell position, so re-deriving
  // them per stage -- ~15 level-set evaluations per cell, an O(N)-segment loop
  // each for the type-6 polyline -- was pure waste.  CARRIED through every
  // block sort with the flow variables (sortFieldData stages them through
  // F_SCRATCH), so they are never stale for existing blocks; the post-
  // adaptation stamp exists for blocks CREATED since the last stamp.
  F_PHI = 17,      // getBoundaryLevelSet at the cell centre (positive INSIDE)
  F_IBM = 18,      // 1 = fluid (wholly outside the body), 0 = non-fluid
  // Point-implicit wall flux (wallPointImplicit): per-cell diagonal relaxation
  // rates lambda = c |F_wall| A / (rho q) of the stiff k~ / tau~ wall boundary
  // fluxes, stamped by the RHS at wall-adjacent cells each stage and consumed
  // (then zeroed) by updateFields, which divides that cell's k~ / tau~ update
  // by (1 + B dt lambda).  Fixed points of the RHS are untouched, so the
  // converged solution is unchanged; the update is bounded for ANY dt, which
  // is what retires the explicit consumption cap in computeDeltaT.  Zeroed by
  // the allocator memsets; transient within a stage, so never sorted/carried.
  F_LAMK = 19,
  F_LAMT = 20,
  // Local time stepping: this cell's own cfl-scaled stable step.  Needs its own
  // bank because computeDeltaT reports per-cell limits through F_SCRATCH, which
  // the RHS then reuses for the u_tau / C_f stamp within the same step.
  // ---- RCCM cut-cell geometry (Ndiaye et al., IJHFF 114 (2025) 109775) ------
  // Stamped beside F_PHI/F_IBM and carried through the sort for the same reason:
  // a pure function of position for a static body.  2-D (pseudo2D) for now.
  //   F_CUTA  = fluid VOLUME fraction alpha_i = dV_i/dV_uncut (their Eq. 15)
  //   F_CUTAX = open fraction of the cell's LOW-x face
  //   F_CUTAY = open fraction of the cell's LOW-y face
  // High faces are the neighbour's low faces, exactly as the flux scatter is
  // already organised.  The wall segment is NOT stored: the discrete divergence
  // theorem fixes it, A_w n = -sum_f A_f n_f, which is what makes the cut-cell
  // update conservative to machine precision.
  F_CUTA  = 29,
  F_CUTAX = 30,
  F_CUTAY = 32,
  F_DTL = 21,
  // q_n register for the Jameson schemes (rkScheme != 0), banks 22..28.
  // Williamson 2N canNOT express q_k = q_n + alpha_k dt L_{k-1} for m >= 3:
  // carrying the update as a pre-scaled accumulator leaves an uncancelled
  // alpha_1 dt L_0 term from stage 3 on (verified numerically -- the 5-stage
  // diverges even at cfl 0.4 when forced into 2N form).  Jameson's family is
  // 2R-storage in (q_n, q), not 2N in (q, S), so it needs its own bank.
  F_QN = 22,
  // Boundary-condition KIND for each non-fluid cell, cached beside F_PHI/F_IBM
  // because it is (like them) a pure function of position for a static body:
  //   0 = wall      -> the ghost is built to satisfy the wall condition
  //   1 = prescribed-> the ghost carries the exact solution (inflow/outflow)
  // One mechanism for every boundary: ghost cells, then the ordinary Riemann
  // solve on the face (the FRIB architecture the Euler path already documents).
  F_IBBC = 31,
  // Point-implicit relaxation rate for the Brinkman porosity stiffness.  Same
  // lifecycle as F_LAMK/F_LAMT: stamped by the RHS in the penalization band,
  // consumed and zeroed by updateFields.  Under --brinkpi 2 it scales all five
  // mean-flow rows, because the stiff excess (w_f - 1) multiplies the mass and
  // energy fluxes exactly as it does the momentum ones.
  F_LAMM = 33,
  // Brinkman FACE POROSITIES, phibar over this cell's LOW-x and LOW-y faces.
  // Stamped beside F_PHI/F_IBM and carried through the sort for exactly the
  // reason the RCCM apertures are: for a static body they are a pure function
  // of position, so recomputing them every stage was paying an O(N)-segment
  // level-set walk per face per RK stage for an answer that never changes.
  // High faces are the neighbour's low faces, which is how the flux scatter is
  // already organised -- and storing ONE value per face is also what keeps the
  // flux and the p grad(phi) source sharing the identical phibar_f, i.e. what
  // makes the pressure-tight cancellation exact.
  //
  // 2-D (pseudo2D) only; a 3-D run still evaluates its z faces live.  The
  // quadrature stored here is whatever --brinkface selects, so raising
  // --brinkseg is now free at run time: it is paid once per adaptation.
  F_BRINKX = 34,
  F_BRINKY = 35,
  // Cut-cell FLUID CENTROID offset (ox, oy), in units of h, relative to the
  // Cartesian cell centre.  PREPROCESSING: the corner level sets are already
  // evaluated once per cell in ibStampGeometryKernel to get the apertures, so
  // the centroid comes out of the same rccmCutGeom call for free.  Before this,
  // the recon-6 path re-derived it EVERY STAGE via rccmCentroidOff -- 4 level
  // set evaluations per call, and it is called for the cell, its two low
  // neighbours and every cut cell in the 8-point gradient stencil, i.e. up to
  // ~44 evaluations per cell per stage.  For an analytic body that is a cheap
  // closed form; for the type-6 POLYLINE it is an O(nSegments) loop (133 for
  // the RAE 2822) and it made the airfoil cost 17 us/cell/iter against 0.55 for
  // the cylinder -- 32x, and worst exactly where AMR puts its cells.
  F_CUTCX = 36,
  F_CUTCY = 37,
  // Cut-cell WALL NORMAL at the cell, stamped with the rest of the geometry.
  // wallNormal is an EXACT closest-point search over the body, i.e. another
  // O(nSegments) loop for a polyline, and the RHS called it twice per cut cell
  // per stage.  Same preprocessing argument as F_CUTCX/F_CUTCY.
  F_CUTNX = 38,
  F_CUTNY = 39,
  // Tangential centroid of the OPEN part of each LOW face, relative to the face
  // midpoint, in units of the face length -- the same convention rccmFaceCen
  // returns.  Stamped so the reconstruction's evaluation point on a cut face
  // comes from the SAME geometry as alpha and the apertures: under --cutgeom 2
  // those are curved (Saye) while rccmFaceCen is the LINEAR two-corner formula,
  // so leaving it in place would evaluate the reconstruction at the wrong point
  // on an otherwise curved cell.  Also retires the last 3 per-stage level-set
  // evaluations in recon 6 (the f00/f10/f01 corner triple).
  F_CUTTX = 40,
  F_CUTTY = 41,
  // --p1 (modal P1 DG): per conserved variable v in {rho, rhoU, rhoV, rhoE}
  // the two slope DOFs  s_x = F_P1S + v,  s_y = F_P1S + P1_NV + v  (CONSERVED
  // variables, in units of the cell: u_h = q + s_x xi + s_y eta with
  // xi, eta in [-1/2, 1/2]), their LSRK accumulators F_P1SR + (same offsets),
  // and F_P1NEW, a marker that is 1 on every cell that existed before the last
  // adaptation (a 0 after the sort = a block created this cycle, whose slopes
  // are prolonged from the parent polynomial).  Allocated only under --p1.
  F_P1S   = 42,
  F_P1SR  = 50,
  F_P1NEW = 58
};
static constexpr i32 NEVOLVE = 7;                 // evolved DOFs (fields 0..6)
static constexpr i32 nCompressibleFields = 42;
static constexpr i32 nCompressibleFieldsP1 = 59;   // + the --p1 slope banks
static constexpr i32 P1_NV = 4;                      // slope-carrying variables (rho, rhoU, rhoV, rhoE)
__host__ __device__ inline i32 p1Var(i32 v) { return v < 3 ? v : 4; }   // conserved field of slope variable v

// ---- RCCM cut-cell geometry (2-D) ------------------------------------------
// Apertures and volume fraction of a Cartesian cell cut by the level set, from
// the four CORNER values.  phi is POSITIVE INSIDE the body, so the fluid region
// is {phi < 0} and the fluid fraction of an edge is found by linear
// interpolation of phi along it -- the same piecewise-linear interface model
// CutLinQuad uses on the DG side, in its cheapest 2-D form.
//
// The area is the marching-squares polygon area, computed by the shoelace
// formula over the fluid polygon's vertices in order.  Rather than enumerate
// the 16 cases, walk the four edges in cyclic order and emit (a) the corner if
// it is fluid, (b) the crossing point if the edge changes sign.  That single
// loop IS the case table, and it is exact for the bilinear-free (piecewise
// linear along edges) model.
// ---------------------------------------------------------------------------
//  CURVED cut geometry by MOMENT FITTING (--cutgeom 2).
//  rccmCutGeom below reconstructs ONE LINEAR interface per cell from the four
//  corner level-set values.  That is a POLYGONAL body, and Bassi & Rebay's
//  classic result is that a polygonal wall produces spurious entropy that
//  corrupts an Euler solution even on very fine grids; the cut-cell quadrature
//  literature says the same in integral form -- straight-sided cut elements let
//  the DOMAIN APPROXIMATION error dominate once the wall is strongly curved.
//  Measured here: the RAE 2822 nose radius is 1.62 cells at nlvls 6 and the cut
//  cell loses half the lift, while the same scheme beats the sharp IB on a
//  cylinder resolved to 19 cells of radius.
//
//  This builds the Q2 (biquadratic) Lagrange interpolant of phi from a 3x3
//  sample over the cell and integrates {phi < 0} with Saye's implicit-domain
//  quadrature (SayeQuad.h, JCP 448:110720), giving alpha and the fluid centroid
//  to the accuracy of a CURVED interface.  The two low-face apertures come from
//  the exact negative measure of the 1-D quadratic restriction of the same
//  polynomial, so faces and volume see one consistent geometry.
//
//  The wall segment is still NOT taken from sayeSurface: A_w n = -sum_f A_f n_f
//  (the discrete divergence theorem) is what makes the update conservative to
//  machine precision, and the Saye header itself warns that hitting its
//  subdivision cap breaks the closed-surface identity in opposite directions on
//  the two sides.  Keeping the GCL form leaves conservation exact by
//  construction and lets the quadrature improve only alpha and the apertures.
// ---------------------------------------------------------------------------
// negative measure of q(t) = c0 + c1 t + c2 t^2 on [0,1], i.e. |{t : q(t) < 0}|
__host__ __device__ inline real cutNegFrac1D(real c0, real c1, real c2) {
  const real qa = c0, qb = c0 + c1 + c2;                 // endpoint values
  if (fabs(c2) < (real)1e-14) {                          // linear
    if (fabs(c1) < (real)1e-14) return (qa < 0) ? (real)1 : (real)0;
    const real r = -c0/c1;
    if (r <= 0) return (qb < 0) ? (real)1 : (real)0;
    if (r >= 1) return (qa < 0) ? (real)1 : (real)0;
    return (qa < 0) ? r : (real)1 - r;
  }
  const real disc = c1*c1 - (real)4*c2*c0;
  if (disc <= 0) return (qa < 0 && qb < 0) ? (real)1 : ((qa < 0 || qb < 0) ? (real)0.5 : (real)0);
  const real sq = sqrt(disc);
  real r1 = (-c1 - sq)/((real)2*c2), r2 = (-c1 + sq)/((real)2*c2);
  if (r1 > r2) { const real t = r1; r1 = r2; r2 = t; }
  // measure of [0,1] where the parabola is negative: between the roots if c2>0,
  // outside them if c2<0.  Clip to [0,1].
  const real lo1 = fmax(r1, (real)0), hi1 = fmin(r2, (real)1);
  const real inside = fmax(hi1 - lo1, (real)0);
  if (c2 > 0) return inside;
  return fmax((real)1 - inside, (real)0);
}

// alpha, low-face apertures and fluid centroid of the CURVED cut, from a Q2
// interpolant of phi sampled on a 3x3 grid over the cell.  f33[j][i] is phi at
// reference (i/2, j/2).  Returns false if the quadrature overflowed its arena,
// in which case the caller falls back to the linear cut.
// alpha, low-face apertures and fluid centroid of the CURVED cut, from a Q2
// interpolant of phi sampled on a 3x3 grid over the cell.  f33[j][i] is phi at
// reference (i/2, j/2).  The caller supplies the quadrature scratch (`buf`,
// `nbuf` nodes): on the host that is a heap block, which is why this runs there
// -- as a per-thread device array the arena would reserve its frame for every
// thread of the stamp kernel and exhaust local memory at launch.
// Returns false (caller keeps the linear cut) if the quadrature could not
// produce a rule.  `why`: 0 ok, 1 arena overflow, 2 empty rule.
__host__ __device__ inline bool cutGeomMoment(const real f33[3][3],
                                              SayeNode *buf, i32 nbuf,
                                              real &alpha, real &aXlo, real &aYlo,
                                              real *cenX, real *cenY,
                                              real *tanX = nullptr, real *tanY = nullptr,
                                              i32 *why = nullptr)
{
  // monomial coefficients of the 3 Lagrange basis functions on nodes {0,1/2,1}
  const real L[3][3] = {{1,-3,2},{0,4,-4},{0,-1,2}};     // L[node][power]
  PolyND phi; phi.zero(2); phi.deg[0] = 2; phi.deg[1] = 2;
  for (i32 pw = 0; pw < 3; pw++)
    for (i32 qw = 0; qw < 3; qw++) {
      real acc = 0;
      for (i32 jj = 0; jj < 3; jj++)
        for (i32 ii = 0; ii < 3; ii++) acc += f33[jj][ii]*L[ii][pw]*L[jj][qw];
      phi.at(pw, qw, 0) = acc;
    }
  // Volume AND face apertures from the SAME quadrature.  sayeFace drives the
  // identical arrangementRule with axis d deactivated, so the faces inherit the
  // volume rule's subdivision, height-direction choice and sign test.  Computing
  // the apertures by a separate closed form (the negative measure of the 1-D
  // quadratic restriction) is exact for that restriction but is a DIFFERENT
  // integration of the same phi, so the two need not agree and the discrete
  // divergence theorem then holds only for constants (via the GCL definition of
  // A_w n), not for linear fields -- which is what second order on a cut cell
  // rests on.
  //
  // buffer layout, all disjoint:  [0, nOut) volume out | [nOut, nOut+nFace) face
  // out (reused per face) | the rest is arena scratch, reset between calls.
  const i32 nOut  = nbuf/4;
  const i32 nFace = nbuf/8;
  const i32 nScr  = nbuf - nOut - nFace;
  if (nScr < 64) { if (why) *why = 1; return false; }
  SayeArena ar; ar.buf = buf + nOut + nFace; ar.cap = nScr; ar.top = 0;

  SayeSet vol; vol.p = buf; vol.n = 0; vol.cap = nOut; vol.ovf = false;
  // PSEUDO-2D rules: sayeVolume/sayeFace leave the z axis ACTIVE, which for a
  // dim-2 phi silently corrupts both (unit-tested: a fully-fluid face came back
  // as aperture 0.16 instead of 1, and a quarter disc's area was 46% low).
  // sayeSlice2D / sayeEdge1D deactivate z through the same act[] mask and are
  // exact on those cases.
  sayeSlice2D(phi, (real)0, &vol, &ar, SayeCfg::def());
  if (vol.ovf)    { if (why) *why = 1; return false; }
  if (vol.n == 0) { if (why) *why = 2; return false; }

  real aper[2] = {0, 0}, tanc[2] = {0, 0};
  for (i32 d = 0; d < 2; d++) {
    ar.top = 0;                                  // scratch is stack-disciplined
    SayeSet fs; fs.p = buf + nOut; fs.n = 0; fs.cap = nFace; fs.ovf = false;
    sayeEdge1D(phi, d, (real)0, (real)0, &fs, &ar, SayeCfg::def());  // LOW face
    if (fs.ovf) { if (why) *why = 1; return false; }
    // weights sum to the open fraction; their first moment along the face's own
    // tangential axis gives the open part's centroid, in rccmFaceCen's units.
    const i32 t = 1 - d;                         // tangential axis of this face
    real a = 0, m = 0;
    for (i32 q = 0; q < fs.n; q++) { a += fs.p[q].w; m += fs.p[q].w*fs.p[q].x[t]; }
    aper[d] = fmin(fmax(a, (real)0), (real)1);
    tanc[d] = (a > 0) ? (m/a - (real)0.5) : (real)0;
  }
  const real axl = aper[0], ayl = aper[1];

  real w = 0, mx = 0, my = 0;
  for (i32 q = 0; q < vol.n; q++) {
    w  += vol.p[q].w;
    mx += vol.p[q].w*vol.p[q].x[0];
    my += vol.p[q].w*vol.p[q].x[1];
  }
  if (why) *why = 0;
  alpha = fmin(fmax(w, (real)0), (real)1);
  aXlo = axl; aYlo = ayl;
  if (cenX) *cenX = (w > 0) ? mx/w : (real)0.5;
  if (cenY) *cenY = (w > 0) ? my/w : (real)0.5;
  if (tanX) *tanX = tanc[0];
  if (tanY) *tanY = tanc[1];
  return true;
}

// Tangential position of the OPEN part of a face's centroid, relative to the
// face midpoint, in units of the face length.  The face is a segment whose
// fluid part is one contiguous piece (piecewise-linear phi along the edge), so
// the open part runs from one end and its centroid is half the aperture in.
__host__ __device__ inline real rccmFaceCen(real fLo, real fHi) {
  const bool oLo = fLo < (real)0, oHi = fHi < (real)0;
  if (oLo == oHi) return (real)0;                     // fully open or fully shut
  const real a = oLo ? fLo/(fLo - fHi) : fHi/(fHi - fLo);
  return oLo ? (real)0.5*a - (real)0.5 : (real)0.5 - (real)0.5*a;
}

__host__ __device__ inline void rccmCutGeom(const real f[4], real &alpha,
                                   real &aXlo, real &aYlo,
                                   real *cenX = nullptr, real *cenY = nullptr)
{
  // corners in cyclic order: (0,0) (1,0) (1,1) (0,1)
  real px[8], py[8]; i32 n = 0;
  const real cx[4] = {0, 1, 1, 0}, cy[4] = {0, 0, 1, 1};
  for (i32 e = 0; e < 4; e++) {
    const i32 a = e, b = (e + 1) & 3;
    if (f[a] < (real)0) { px[n] = cx[a]; py[n] = cy[a]; n++; }
    if ((f[a] < (real)0) != (f[b] < (real)0)) {
      const real t = f[a]/(f[a] - f[b]);          // phi = 0 crossing
      px[n] = cx[a] + t*(cx[b] - cx[a]);
      py[n] = cy[a] + t*(cy[b] - cy[a]); n++;
    }
  }
  real A2 = 0, Cx = 0, Cy = 0;
  for (i32 v = 0; v < n; v++) {
    const i32 w = (v + 1 == n) ? 0 : v + 1;
    const real cr = px[v]*py[w] - px[w]*py[v];
    A2 += cr;
    Cx += (px[v] + px[w])*cr;
    Cy += (py[v] + py[w])*cr;
  }
  alpha = fmin(fmax((real)0.5*fabs(A2), (real)0), (real)1);
  // Polygon centroid, in cell-local [0,1]^2 coordinates.  This is the point the
  // cut cell's average actually LIVES at -- the paper reconstructs there
  // ("the centroid of the R-Cells is used as the center of the reconstruction"),
  // and for an R-Cell it sits in the FLUID sliver, while the Cartesian centre
  // is inside the solid.  Sampling the Cartesian centre instead hands every
  // R-Cell/NR-Cell face a state taken from the wrong side of the wall.
  if (cenX && cenY) {
    if (fabs(A2) > (real)1e-20) { *cenX = Cx/((real)3*A2); *cenY = Cy/((real)3*A2); }
    else                        { *cenX = (real)0.5;       *cenY = (real)0.5; }
  }
  // low-x face is corner 0 -> corner 3 ; low-y face is corner 0 -> corner 1
  #define RCCM_EDGE(FA, FB) (((FA) < (real)0 && (FB) < (real)0) ? (real)1 : \
                            (((FA) >= (real)0 && (FB) >= (real)0) ? (real)0 : \
                             ((FA) < (real)0 ? (FA)/((FA)-(FB)) : (FB)/((FB)-(FA)))))
  aXlo = RCCM_EDGE(f[0], f[3]);
  aYlo = RCCM_EDGE(f[0], f[1]);
  #undef RCCM_EDGE
}

#include <vector>
struct ClipRec;                // host clip record (CompressibleSolver.cu)
// ---- split-cell geometry records (see CompressibleSolver::cutSplit) ----------
// Offsets are from the CELL CENTRE in units of hR = min(dx,dy) (as F_CUTCX);
// wall vectors are open-fraction differences, the kernel's nwx/nwy units:
//   wnx = -(open fraction on the high-x face - on the low-x face), etc.
struct CutSplitCell {          // one per cell with > 1 fluid piece: its piece 0
  real a0, cx0, cy0;           // area fraction and centroid of piece 0
  real wnx0, wny0, wcx0, wcy0; // its wall vector and wall centroid
  i32  first, n;               // extra pieces: cutPiece[first .. first+n)
  // INTERNAL face: a slit tip inside the cell is extended along its tangent to
  // the boundary; that extension is an open flux face between two pieces
  real iLen;                   // physical length (0: none)
  real icx, icy;               // centroid offset from the cell centre, units of hR
  real inx, iny;               // unit normal from piece iPa into piece iPb
  i32  iPa, iPb;               // the two pieces it separates
};
// DOF HANDLES: h >= 0 is a cell index, h < 0 is piece -h-1.  A piece that is not
// merged is its own DOF (state in cutPieceQ); a merged piece or member cell
// reads its element's state through the owner handle.  CUT_DEAD = no fluid.
#define CUT_DEAD (-1000000000)
__host__ __device__ inline bool cutIsPiece(i32 h) { return h < 0 && h != CUT_DEAD; }
__host__ __device__ inline i32  cutPieceOf(i32 h) { return -h - 1; }
__host__ __device__ inline i32  cutHandle(i32 k)  { return -k - 1; }
struct CutPiece {              // one per EXTRA piece
  i32  cell, owner;            // host cell; DOF handle of its element's owner (itself if unmerged; CUT_DEAD)
  real a, cx, cy;              // area fraction, own centroid (units of hR from the cell centre)
  real ecx, ecy;               // ELEMENT centroid (== cx, cy when unmerged)
  real wnx, wny, wcx, wcy;
};
// ---- --p1 on cut elements (buildP1Cut) ----------------------------------------
// One P1 polynomial per DOF element (owner cell, merged element or piece) over
// its OWN polygon: basis {1, (x-gx)/h, (y-gy)/h} centred at the element
// centroid, so the mean decouples and the slope mass matrix is the 2x2 second-
// moment matrix of the polygon (stored inverted).  The volume term is a fan of
// 3-point triangle rules over the clipper's loop polygons (exact for
// quadratics), the wall is 2 Gauss points per wall edge, and every open face
// interval is a P1Seg fluxed by p1SegKernel with both sides' own traces.
struct P1Elem {
  i32  handle;                  // DOF handle (cell index, or -k-1 for piece k)
  real gx, gy, h, area;         // centroid (physical), basis scale, fluid area
  real m11, m12, m22;           // inverse slope mass matrix
  i32  q0, nq;                  // quadrature points [q0, q0+nq) in p1Qpt
};
struct P1Qpt { real x, y, w, nx, ny; };                       // w != 0: volume point; w == 0: wall point, (nx,ny) = outward normal * length
struct P1Seg { real x0, y0, x1, y1, nx, ny; i32 hA, hB; };   // open face piece; n points from element A into B
struct CutFaceSeg { real len, cen; i32 pC, pN, ownC, ownN; };   // len: fraction of the face, cen: offset from the face midpoint in face lengths
struct CutFace {               // one per cell whose LOW-x or LOW-y face is not a single (0,0) segment
  i32 cell;                    // the cell whose low faces these are
  i32 nX, nY;                  // segment counts; sx[0]/sy[0] is the (0,0) pair (len 0 if absent)
  CutFaceSeg sx[4], sy[4];
};

class CompressibleSolver : public MultiLevelSparseGrid {
public:

  real deltaT;
  real cfl;
  // wavelet-detail normalization: domain maxima of the 3 field scales
  // {|rho|, |momentum|, |rhoE|}, reduced device-side each adaptation.
  // (Local / neighbourhood normalization was tested and is Pareto-dominated by
  // global-with-tighter-threshold on single-feature flows; it over-refines
  // weak-feature regions.)
  real *globalScale;    // [3]  domain max of the 3 scales
  real waveletThresh;

  i32 reconFar;         // --reconfar: under recon 6, cells NOT within two cells of a cut/dead/merged
                        // cell use this 1-D scheme instead of the least-squares gradient (-1 = off)
  i32 *cutNear = nullptr;   // [stride] 1 = within the two-cell band around the body (stamped)
  void buildCutNear(void);
  void leafCensus(void);   // --leaf: block groups, mortars, band cells after a sort
  i32 recon;            // face reconstruction of rho/p/tangential (and FV normal) velocity:
                        // 0 = smooth TVD limiter, 1 = ROUND, 2 = LD-ROUND,
                        // 3 = unlimited 3rd-order parabola (kappa=1/3; smooth tests only),
                        // 4 = van Leer harmonic limiter in NVD form (DEFAULT)
                        // (ROUND/LD-ROUND: Huang, Deng, Matar & Ying, JCP 555 (2026), Eqs. 4.1/4.2)
                        // ROUND: 6-7x lower smooth-wave error than TVD, cleaner low-Mach,
                        // shocks stay spike-free (soft ~1% non-TVD overshoots by design)
  i32 mdFlux;           // 1 = genuinely multidimensional Osher-type corner flux
                        // (Gaburro, Ricchiuto & Dumbser, arXiv:2506.00207, Eq. 23)
                        // with FIRST-ORDER corner states (P0 cell averages).
                        // pseudo-2D only; recon/reflux do not apply.
  // ---- Navier-Stokes viscous terms -------------------------------------
  // mu <= 0 disables the viscous path entirely (pure Euler, bit-for-bit).
  // Units: R = 1, so T = p/rho and cp = gam/(gam-1); the thermal conductivity
  // is kap = mu*gam/((gam-1)*Pr).  Set mu directly (--mu) or via a Reynolds
  // number (--re, which sets mu = 1/Re for the unit reference state).
  real mu;              // dynamic viscosity (constant, or the reference value if sutherS > 0)
  real Pr;              // Prandtl number (default 0.72)
  real sutherS;         // Sutherland constant S/T_ref; <= 0 selects constant mu
  real sutherTref;      // reference temperature for the Sutherland law (T = p/rho units)

  // ---- k~-tau~ SST wall-modeled RANS (src/fv/KtauSst.h) -----------------
  // Tamaki, Friess, Jacob & Imamura, JCP 566 (2026) 115239.  rans = 0 leaves the
  // Navier-Stokes path bit-for-bit unchanged.  Needs --mu (or --re): the model
  // is a correction to a viscous solve, not a substitute for one.
  i32 rans;             // 0 = off, 1 = k~-tau~ SST
  real kInf;            // freestream k~   (paper: 1e-6 u_inf^2)
  real tauInf;          // freestream tau~ (paper: 0.2 L/u_inf)
  i32 ransSustain;      // 1 = freestream-sustaining source terms (Eq. 32)
  i32 ransVorticity;    // 1 = "-V" variant: Omega^2 for S^2 in Eq. (19) (paper default)
  real dCutoff;         // Eq. (38) r_d / Eq. (A.5) phi cutoff; <= 0 selects 3*finest dx
  real Lref;            // characteristic length in Gamma3 (Eq. 31)
  real PrT;             // turbulent Prandtl number (0.9)
  i32 wallGeom;         // wall-distance geometry: 0 = none (freestream), 1 = flat plate below y=0
  real plateX0;         // wallGeom 1: leading-edge x
  // Offset between the wall and the first grid face (the paper's aligned-grid
  // setup, Fig. 5a; default 0.5*dy).  It must be > 0: with the wall exactly ON
  // the face, d_FC = 0 makes phi(d_FC) = 0 and the Appendix-A tau~ flux
  // degenerate (both phi_LR and tau~_LR vanish, and only their ratio is finite).
  // Fig. 7 of the paper shows C_f is insensitive to the offset value.
  real wallOffset;
  real dIpFac;          // image-point distance in cells (paper: 3)
  // Wall-model band: the near-wall region is held at the FINEST level out to this
  // distance.  The model needs ONE resolution along the wall -- d_cutoff is a
  // single length (Eqs. 38, A.5) while the image point is 3 * the LOCAL dy, so on
  // a graded grid the two disagree and the shear-stress balance of Sec. 3.1
  // breaks.  UTCart propagates d_cutoff from the wall by an advection equation;
  // on a Cartesian wall the equivalent is simply to make the local dy BE the
  // finest dy near the wall.  <= 0 selects 8 * dCutoff.  Refinement above the
  // band stays wavelet-driven, which is where the AMR saving actually is.
  real wallFineBand;
  real fsU, fsV, fsP;   // freestream velocity and pressure for the bcType 4 inflow/farfield (rho = 1)

  real vortexAdvect;    // isentropic-vortex IC advection velocity (u0=v0)
  real greshoP0;        // Gresho-vortex background pressure = 1/(gam*Ma^2) (sets Mach)
  i32 staticGrid;       // 1 = fixed refinement (no dynamic wavelet adaptation)
  real refineRadius;    // static-grid: outer radius of the level-1 refinement shell (about the domain centre)

  i32 tGrid;
  i32 tSolver;
  i32 tOutput;
  i32 tTotal;
  long tForwardUs;      // profiling: time in forwardWaveletTransform (the 6 max reductions)
  long tSortUs;         // profiling: time in sortBlocks (hash rebuild + Morton sort)

  // ---- immersed boundary (level-set) ------------------------------------
  // immerserdBcType selects the body:  0 = none, 1 = sphere, 2 = plane y = ibPlane
  // (grid-aligned on purpose -- it lets the IB path be checked against the
  // validated grid-aligned wall model), 3 = cylinder about the z axis.
  // The level set is POSITIVE INSIDE the body throughout.
  real ibCenter[3];
  real ibRadius;
  real ibPlane;
  real ibAngle;         // inclined-plate angle to the x axis, DEGREES (type 5)
  i32 immerserdBcType;
  // Type 6: a closed 2-D polyline body (airfoil).  Points are stored on the
  // device as x,y pairs, counter-clockwise, first point NOT repeated.  The level
  // set is the exact signed distance to the polyline: sign from the winding
  // number (robust on the concave rear lower surface of a supercritical section,
  // where a nearest-segment normal test flips), magnitude from the nearest
  // segment.  Far cells short-circuit on the bounding box, so the O(N) search
  // only runs in the band that isFluidCell and the wall model actually need.
  real *ibPoly;        // device array, 2*ibPolyN reals
  i32   ibPolyN;
  real  ibBox[4];      // xmin, ymin, xmax, ymax
  real  ibChord;       // chord length, for reporting
  // chord frame (set by Main for the airfoil cases): the section was scaled by
  // chord, rotated by -aoa and translated to ibOrigin, so a surface point maps
  // back to x/c via ((X-ox)cos - (Y-oy)sin)/chord.
  real  ibOrigin[2], ibCosA, ibSinA;
  void  setAirfoil(const real *xy, i32 n);   // host: upload + bbox
  i32 bcType;
  // bcType 8 (characteristic far field): width, in units of the boundary sound
  // speed, over which the inflow/outflow branch for ENTROPY and the TANGENTIAL
  // velocity is blended instead of hard-switched at u_n = 0.  0 = the hard
  // switch.  On a box whose top/bottom edges run nearly parallel to the flow,
  // u_n sits at ~1e-4 c for the whole edge, so the branch re-decides every step
  // and the ghost jumps by the (interior - freestream) difference each time.
  real ffBlend;
  i32 icType;

  i32 iter;

  CompressibleSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_, i32 withP1 = 0) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, withP1 ? nCompressibleFieldsP1 : nCompressibleFields) {
      cfl = .5;
      p1 = 0;
      waveletThresh = .005;
      iter = 0;
      immerserdBcType = 0;
      ringlebKmin = 0.6; ringlebKmax = 0.98; ringlebScale = 1.0;
      ringlebQmin = 0.5; ringlebX0 = -1.88; ringlebY0 = -2.4;
      ibPoly = nullptr; ibPolyN = 0; ibChord = 1.0;
      ibBox[0]=ibBox[1]=ibBox[2]=ibBox[3]=0;
      ibCenter[0] = 0.5; ibCenter[1] = 0.5; ibCenter[2] = 0.5;
      ibRadius = 0.05;
      ibPlane  = 0.0;
      ibAngle  = 0.0;
      // 0 = fill immersed ghost cells (the paper's / UTCart's arrangement).
      // Measured: ghost-FREE diverges (|rhoV| 4.7e-1 at t=2 for d_FC=0.5h) with
      // erratic Cf; ghost-FILLED is steady (5.9e-3, below the body-fitted
      // control's 1.1e-2) with Cf = 0.002672, -1.0% vs TMR.  Ghost-free is kept
      // only as a --ibgf 1 experiment.
      ibGhostFree = 0;
      ibWmles = 0;
      wmX1 = 1.0e30;
      detFlux = 1; ffBuf = nullptr; ffN = 0;
      shash = 0; shashFrom = 0; shashTo = 1<<30;
      wmClip = 3.0; saDFloor = 0.0; wmCurv = 1; wmGhost = 0;
      ffVortex = 0; ffEvery = 100; ffGamma = 0; ffXv = 0; ffYv = 0; ffCl = 0; ffPrints = 0;
      ibDfcMode = 0;
      wallPointImplicit = 1;
      lts = 0;
      ltsRatio = 100.0;
      // Measured on the inviscid cylinder (exact max|rhoV| = 1.0): OFF gives
      // 0.99931, ON gives 1.0408.  FRIB uses the curvature term to build a
      // Riemann TRACE in a high-order scheme; here it feeds a prescribed flux
      // directly, with no upwinding to damp it, so it injects momentum.  Kept
      // as --ibcurv 1 for experiment.
      paintOn = 1;
      ibIpQuad = 0;
      ibThermoRec = 0;
      ibWls = 0; ibGMirror = 0; ibGIter = 1; ibGFloor = 0.25; ibGSlip = 0; ibIface = 0;
      ibBrink = 0; ibBrinkEps = 1e-6; ibBrinkDelta = 0.125; brinkNSeg = 4;
      ibRccm = 0; ibRccmAlphaMin = 1e-9; ibRccmPw = 1; ibDirichlet = 0; svQuarter = 0; canalY0 = 0; canalY1 = 1; canalMa = 0.675; canalPout = canalPin = canalRhoIn = 1; canalUin = 0; gradLim = 1; gradLimK = 5.0; kSensor = 0.05;
      ransA7Tol = 1e-6;
      ibWallMode = 0; ibInfinite = 0; turbModel = 0; nutInf = 0;
      wmX0 = -1;
      wmRamp = 0;
      dtDipThresh = 0; dtDipPrints = 0; dtDipCooldown = 0;
      envCheck = 0; envPrints = 0;
      ibTurbFlux = 1;
      gridTrace = 0;
      adaptEvery = 4;
      dtEvery = 4;
      maxIter = 0; residEvery = 0; resTol = 0; resid = 0; resid0 = 0;
      residFar = 0; residMax = 0; residMaxDw = 0; resFar = 4;
      ibFluxRecon = 1;
      ibCurv = 0;
      ibHo = 0;
      ghostSlip = 1;
      rkScheme = 0;
      nRkStages = 3;
      precond = 0;
      precondK = 5.0;
      precondMref2 = 0.04;
      dtScale = 1.0;
      ipStandMin = 0.0;   // paper: d_IP = 3*dx flat, no standoff floor
      bcType = 0;
      icType = 0;
      recon = 1;
      mdFlux = 0;
      mu = 0.0;             // inviscid by default
      Pr = 0.72;
      sutherS = 0.0;
      sutherTref = 1.0;
      rans = 0;
      kInf = 0.0;
      tauInf = 1.0;
      ransSustain = 1;
      ransVorticity = 1;
      dCutoff = 0.0;
      Lref = 1.0;
      PrT = 0.9;
      wallGeom = 0;
      plateX0 = 0.0;
      wallOffset = 0.0;   // <= 0 selects 0.5 * finest dy
      dIpFac = 3.0;
      wallFineBand = 0.0;   // <= 0 selects 8 * dCutoff
      fsU = 1.0;
      fsV = 0.0;
      fsP = 1.0;
      vortexAdvect = 0.0;
      greshoP0 = 0.0;
      staticGrid = 0;
      refineRadius = 0.4;
#ifdef USE_MGPU
      rebalanceEvery = 0;   // dynamic rebalance off by default (experimental)
#endif

      tGrid = 0.0;
      tSolver = 0.0;
      tOutput = 0.0;
      tTotal = 0.0;
      tForwardUs = 0;
      tSortUs = 0;
#ifdef USE_MGPU
      dirSlot = 0;
      dirSendCnt = dirRecvCnt = dirFill = nullptr;
      dirSendLoc = dirRecvLoc = nullptr;
      sendBuf = recvBuf = nullptr;
#endif
  }

  void initialize(void);
  void buildInitialGrid(bool doPaint);   // base grid + IC + refine cascade
  real step(real dt);
  void sortFieldData(void);
  void setInitialConditions(void);
  // fOff selects the state bank (0 = live fields).  prim says whether that bank
  // currently holds PRIMITIVE variables -- the mirror/copy conditions do not care,
  // but the bcType 4 Dirichlet inflow and farfield have to write the right form.
  void setBoundaryConditions(i32 fOff = 0, i32 prim = 0);
  void conservativeToPrimitive(void);
  void primitiveToConservative(void);
  void forwardWaveletTransform(void);
  void inverseWaveletTransform(void);
  void adaptGridConsistent(void);   // refinement cascade w/ cross-rank structure exchange (== adaptGrid on 1 GPU)

  void computeDeltaT(void);
  void computeTurbClosure(void);   // RANS: fill F_MUT/F_TF1 and accumulate the k~/tau~ sources
  void stampIbGeometry(void);
  void checkCutGeometry(void);
  void reportDeadTaps(void);
  void reportIbFaceRows(void);
  void checkWellBalanced(void);      // IB: cache F_PHI/F_IBM (once per adaptation; body is static)
  void applyWallGhosts(void);
  void computeShockSensor(void);
  void reportGhostQuad(void);      // RANS: overwrite wall ghost rows with the wall-model profile
  void computeRightHandSide(void);
  void stateHash(const char *tag, i32 it);
  void updateFarFieldVortex(void);   // measure C_l, refresh ffGamma
  void updateFields(i32 stage);
  void zeroAccumulator(void);   // zero the shared bank before LSRK stage 1
  void topoCheck(i32 phaseTag);          // debug: assert prnt/nbr/hash bindings are loc-consistent (--debug)
  void censusPrint(const char *tag);     // debug: allreduced owned-interior block count with a stage tag
#ifdef USE_MGPU
  void haloExchange(i32 fOff, i32 nf);   // fill partition-boundary ghost blocks from owners
  void exchangeStructure(void);          // publish owned blocks + import neighbors' (structure only, no data)
  void reconstituteOldSnapshot(void);    // rebuild F_OLD for blocks created this cycle (halo + coarse->fine interp)
  void rebuildGhosts(void);              // recreate the 2-ring ghost layer from neighbors' directories
  void rebalanceWeights(void);           // count owned blocks/base column, allreduce, re-cut the Morton curve
  void rebalancePartition(void);         // dynamic: recount, re-cut, migrate (every rebalanceEvery adaptations)
  void migrateBlocks(const i32 *newOwner);   // ship departing base columns to their new owners
  void invalidateCommBuffers(void);      // re-size the directory/halo arrays after a map change
  i32 *ownerScratch = nullptr;           // [nb0*nb1*nb2] candidate map for the re-cut
  i32 rebalanceEvery;                    // rebalance period in adaptation cycles (0 = off)
  double *wBase = nullptr;               // [nb0*nb1*nb2] per-base-column block weights (replicated after allreduce)
  void buildDirectories(void);           // build + exchange the per-neighbor boundary directories
  // Message-passing halo (no peer-memory access).  Per neighbor N, this PE sends
  // a DIRECTORY of the location codes of its owned blocks whose 2-ring reaches
  // into N (dirSend), and receives N's directory (dirRecv).  Ghosts are created
  // from dirRecv; the field data is packed for each neighbor (all of dirSend, in
  // order) and exchanged with comm::neighborExchange -- one contiguous message
  // per neighbor.  Indices are resolved locally by hash lookup of the loc codes,
  // so nothing reaches into a peer's memory.  Rebuilt each adaptation.
  i32   dirSlot;       // per-neighbor directory capacity (blocks), identical on every PE
  i32  *dirSendCnt;    // [nNbr]
  i32  *dirRecvCnt;    // [nNbr]
  u64  *dirSendLoc;    // [nNbr*dirSlot]  my boundary-block loc codes, per neighbor
  u64  *dirRecvLoc;    // [nNbr*dirSlot]  neighbor's boundary-block loc codes
  i32  *dirFill;       // [nNbr]          scratch fill counter
  i32  *needRecvCnt = nullptr;  // [nNbr]           needs received from each neighbor
  u64  *needRecvLoc = nullptr;  // [nNbr*needSlot]  loc codes to adopt (in OUR territory)
  real *sendBuf;       // [nNbr*dirSlot*NEVOLVE*blockSizeTot]  packed field data out
  real *recvBuf;       // [nNbr*dirSlot*NEVOLVE*blockSizeTot]  packed field data in
#endif

  void restrictFields();
  void interpolateFields();

  void writeLineProfile(const char *fileName); // 1D profile dump for validation
  void writeIbSurface(const char *fileName);   // immersed body: surface Cp vs x/c
  void writeIbField(const char *fileName, real halfWidth = 1.5);  // field window around the body
  void writeIbMask(const char *fileName, real halfWidth = 1.0);  // cached phi/mask vs analytic
  void writeIbGhostLines(const char *fileName, real halfWidth = 1.0);  // ghost-fill image lines
  void writeIbWallFaces(const char *fileName, real halfWidth = 1.0);  // wall-model face geometry
  void writeGridBlocks(const char *fileName);  // AMR block structure, for plotting
  void writeCfProfile(const char *fileName);   // skin-friction coefficient along the modeled wall
  void printRansExtremes(void);                // max k~ / tau~ range / mu_t / dt limits
  void scanNonFinite(const char *tag);          // AMR debug: first non-finite evolved cell (--debug 2)
  void scanNonFiniteBase(const char *tag, i32 baseOff);
  real computeResidual(void);                  // RMS of dq/dt over ALL live fluid cells
  void snapshotResidualQ(void);                // q snapshot for the dq/dt residual
  // ---- state redistribution (SRD), Berger & Giuliani JCP 428 (2021) 109820 --
  // The N=0 (finite-volume) case of src/common/StateRedistribution.h: the
  // degree-N L2 projection over a merge neighbourhood collapses to a
  // VOLUME-WEIGHTED AVERAGE, so no cut quadrature and no mass matrix are needed.
  //   Pi_k u = sum_{j in M_k} w_j u_j / sum_{j in M_k} w_j,   w_j = alpha_j V_j/|C_k|
  //   (S u)_i = (1/|C_i|) sum_{k : i in M_k} Pi_k u
  // With that weighting sum_i alpha_i V_i (Su)_i = sum_i alpha_i V_i u_i
  // EXACTLY, so S is conservative -- which is the whole point: it buys the
  // sliver its stable step back without the non-conservative per-cell diagonal
  // that --cutpidamp turns off.
  // Neighbourhood growth is confined to the 5x5 patch around the seed, so a
  // member is always within radius 2 of its seed and the |C_i| gather can scan
  // the same patch -- no atomics, hence deterministic.
  static constexpr i32 SRD_MAXM = 8;   // max cells in a merge neighbourhood
  // --srdreach: how many FACE STEPS from the seed a neighbourhood may grow.
  // 1 = the sliver and its own face neighbours only (3x3 gather); 2 lets a
  // sliver whose face neighbours are themselves small take a neighbour-of-
  // neighbour (5x5 gather).  The gather radius must equal the growth radius so
  // every cell can find every neighbourhood that contains it without atomics.
  // A neighbourhood that cannot reach volFrac within the reach is used anyway
  // and counted in the [srd] line's "never reached the target".
  i32 srdReach;
  i32   srdOn;         // --srd 1
  real  srdVolFrac;    // "small" threshold in background volumes (papers: 1/2)
  i32  *srdM   = nullptr;   // [SRD_MAXM][nBlocksMax*blockSizeTot] neighbourhood lists
  i32  *srdMn  = nullptr;   // members per neighbourhood (0 = dead, 1 = trivial)
  i32  *srdC   = nullptr;   // |C_i|
  real *srdPi  = nullptr;   // [5][...] per-neighbourhood projection of dU (or U*)
  real *srdPi0 = nullptr;   // [5][...] per-neighbourhood projection of U^n (admissibility fallback)
  real *srdTh  = nullptr;   // per-neighbourhood fallback weight theta_k in [0,1]
  // --srdpos 1 (increment mode): ADMISSIBILITY FALLBACK.  The increment form
  // U^n + S(dU) gives up the one thing the state form had going for it at a
  // strong transient: S(U^n) resets a sliver to ~its neighbour's state every
  // step, which is also what keeps it admissible (measured: increment-SRD goes
  // to negative energy at a wall cell within 9 iterations of an impulsive
  // M=1.5 start, at first AND second order, at cfl 0.8 AND 0.3, while the
  // point-implicit survives).  So blend that reset back in ONLY where needed:
  //   U^{n+1} = U^n + S(dU) + theta_k (S(U^n) - U^n),
  // theta_k = 1 if the neighbourhood's SEED would be inadmissible (rho or p
  // below floor) under the pure increment update, else 0.  (S(U^n) - U^n) has
  // zero alpha-weighted total over each neighbourhood by construction of the
  // |C| weights, so this is conservative, and a converged admissible state
  // never triggers it, so the exact steady-state property survives.
  i32  srdPos;
  real srdRhoMin, srdPMin;  // admissibility floors
  size_t srdStride = 0;     // nBlocksMax*blockSizeTot
  real *srdS  = nullptr;    // per-neighbourhood blend s_k (UM-SRD 6)
  real *srdU0 = nullptr;    // [5][...] pre-update state, for the UM-SRD indicator
  // UM-SRD (Karell, arXiv:2605.04863): standard weighted SRD fires every step
  // even when the finite-volume update is zero, so it keeps replacing cells with
  // neighbourhood averages and the base scheme's steady state is NOT preserved.
  // Blend S with the identity through an update-magnitude indicator (their 4-8):
  //   dUmax = max_{i in M} ||U* - U^n||_2 ,  eta = dUmax/(eps + dUmax) ,
  //   s = eta^p/(eta^p + tau^p) ,  R = (1-s) Id + s S.
  // NOTE: their s is PER NEIGHBOURHOOD.  With OVERLAPPING neighbourhoods the
  // conservation identity only survives if s is constant on each connected
  // component of the overlap graph, so a per-cell s would leak.  We use one
  // GLOBAL s (the single-component limit); since the indicator is near-binary by
  // design this costs nothing, and it keeps sum_i alpha_i V_i U_i exact.
  real srdTau;  // tau in (6)
  // eps in (5).  The paper fixes eps = 1e-14 so shut-off happens only at machine
  // precision, which guarantees the steady state is an exact FIXED POINT but does
  // nothing on the APPROACH: with dU ~ dt*R, eps must be ~1/tau times the dU you
  // want to shut off at, or s stays pinned at 1 and you get standard SRD (which
  // measurably floors the residual here).  Exposed so it can be raised.
  real srdEps;
  i32  srdLocal;   // 1 = per-neighbourhood s (paper), 0 = one global s
  // 1 = apply S after EVERY RK stage; 0 = once per STEP (after the last stage,
  // with U^n snapshotted at stage 0 so the indicator sees the whole step's
  // update).  Per-stage lets S fight the stages; per-step lets them settle first.
  i32  srdPerStage;
  // --srdincr 1: redistribute the INCREMENT, U^{n+1} = U^n + S(U* - U^n), instead of
  // the state U^{n+1} = S U*.  S is linear, S(U*) = S(U^n) + S(dU), and it is the
  // S(U^n) term that perturbs a non-constant steady state (their Prop. 3): it
  // averages cells that legitimately differ.  Dropping it gives a scheme that
  //   * vanishes EXACTLY with the update -- the base steady state is a fixed
  //     point with no indicator, eps or tau,
  //   * scales smoothly and proportionally with the update in between,
  //   * is still conservative (S conserves the increment's total, which is the
  //     explicit scheme's boundary flux), and
  //   * still stabilises: the sliver's 1/alpha increment is spread over the
  //     neighbourhood exactly as the state was.
  // This is the paper's own observation about FLUX redistribution ("naturally
  // diminishes with small updates"), carried over to SRD's neighbourhoods.
  i32  srdIncr;
  i32  srdP;    // p   in (6)
  void srdSnapshot(void);   // stash U^n before the stage update

  void stampCutGeomCurved(void);   // --cutgeom 2 host pass
  void stampCutGeomClip(void);     // --cutgeom 3 host pass: clip the cell by the body SEGMENTS
  void buildClipSegments(void);    // segment list for --cutgeom 3 (polyline as is; analytic body polygonised)
  double *clipSeg = nullptr;       // host, 2*clipSegN, closed loop
  i32     clipSegN = 0;
  bool    clipFluidLeftFwd = true; // walking the loop forward keeps the fluid on the left
  double  clipArea = 0;            // area enclosed by the loop (exact reference for the fluid area)
  double  clipBox[4] = {0,0,0,0};  // xmin ymin xmax ymax of the loop
  void buildSrd(void);      // after the geometry stamp; nLvls == 1 only for now
  // ---- cell merging (agglomeration) ------------------------------------
  // Every small cut cell is merged PERMANENTLY with face neighbours until the
  // element holds >= cutMergeFrac of an uncut cell.  One DOF per element: it
  // lives in the OWNER cell and is broadcast to the member cells after each
  // update, so every stencil sees a valid state in every cell.  The RHS of all
  // members scatters to the owner and is divided by the ELEMENT volume: that is
  // the small-cell fix, with nothing to redistribute (SRD) or damp (point-
  // implicit) -- the element is an ordinary control volume.
  void buildCutMerge(void);          // host pass after every geometry stamp
  void buildCutSplit(std::vector<ClipRec> &recs);   // piece records + face segments from the clipper loops
  void cellGeomHost(i32 c, double &px, double &py, double &dx, double &dy);
  void writeCutWindow(const char *fileName, double xc, double yc, i32 nh);   // --cutdump: cut-cell records around a point
  // ---- split cells (--cutsplit, needs --cutgeom 3 + --cutmerge) -------------
  // A cell crossed by more than one wall has several fluid PIECES (the clipper's
  // loops).  The largest keeps the cell DOF (piece 0); every other piece is
  // geometry only, attached to a neighbour element on ITS side of the wall.  A
  // face is then a list of SEGMENTS, one per (piece on this side, piece on the
  // other side) pair, each with its own owners.  Segment (0,0) rides the
  // ordinary flux path; the rest, and the extra pieces' walls, are handled by
  // cutSplitFluxKernel.  This is what lets a body thinner than a cell -- down to
  // zero thickness -- carry two different states across it.
  i32   cutSplit = 0;                // --cutsplit
  i32  *cutSplitId = nullptr;        // [stride] -> CutSplitCell index, -1 if the cell has one piece
  i32  *cutFaceId  = nullptr;        // [stride] -> CutFace index, -1 if both low faces are plain
  CutSplitCell *cutSplitCell = nullptr;
  CutPiece     *cutPiece     = nullptr;
  CutFace      *cutFace      = nullptr;
  i32   nCutSplit = 0, nCutPiece = 0, nCutFace = 0;
  i32   cutSplitCap = 0, cutPieceCap = 0, cutFaceCap = 0;
  // piece-resident DOFs: conservative state and the LSRK accumulator, [NEVOLVE][cutPieceQCap]
  real *cutPieceQ = nullptr, *cutPieceS = nullptr, *cutPieceAlphaE = nullptr;
  i32   cutPieceQCap = 0;
  // --p1 cut elements: tables (rebuilt with every geometry stamp), the per-cell
  // irregular flag (the regular P1 kernel skips every face touching one), and
  // the piece slope DOFs + accumulators [2*P1_NV][cutPieceQCap]
  i32   *p1ElemOfCell = nullptr, *p1ElemOfPiece = nullptr, *p1Irr = nullptr;
  P1Elem *p1Elem = nullptr;  P1Qpt *p1Qpt = nullptr;  P1Seg *p1Seg = nullptr;
  i32   nP1Elem = 0, nP1Qpt = 0, nP1Seg = 0, p1ElemCap = 0, p1QptCap = 0, p1SegCap = 0, p1PieceCap = 0;
  i32   *p1ElemNbrOff = nullptr, *p1ElemNbr = nullptr, p1NbrCap = 0;   // per element: the handles across its face pieces (limiter)
  real  *cutPieceSX = nullptr, *cutPieceSR = nullptr;
  std::vector<ClipRec> *clipRecs = nullptr;   // every clipped cell's loops, kept for buildP1Cut
  void buildP1Cut(void);
  i32  *cutOwner  = nullptr;         // [stride] owner cell of this cell's element (self if unmerged)
  real *cutAlphaE = nullptr;         // [stride] element volume / uncut cell volume
  i32   cutMerge = 0;                // --cutmerge
  i32   cutDbg = 0;                  // --cutdbg: scan the first RHS for exploding cells
  real  cutDbgThr = 1e6;             // --cutdbgthr: |Rhs| threshold of that scan
  real  cutMergeFrac = 0.5;          // --cutmergefrac
  real  cutPieceFrac = 0.25;         // --cutpiecefrac: an extra piece keeps its own DOF above this area fraction
  void applySrd(void);      // after each RK stage update
  real *residQ0 = nullptr;                     // [4][nBlocksMax*blockSizeTot]
  real *residCell = nullptr;                   // per-cell |dq/dt| from the last sample (field dump)
  i32  wallResolutionCheck(bool verbose = true);   // count wall-row blocks NOT at the finest level
  void writeSolution(const char *fieldFile, const char *profFile, real xStation);
  void computeAcousticReflection(const char *fileName); // acoustic wave reflection at coarse/fine interface
  void computeAcousticL2Error(void);            // L2 velocity error for the periodic sine wave (order study)
  void computeShearDecayError(real t);          // L2 error vs the exact viscous shear-wave decay
  void computeRansDecayError(real t);           // k~/tau~ vs the exact 0-D source solution (freestream box)
  void computeRansShearProbe(void);             // production/gradient probe on the frozen shear
  void computeRansWallProbe(void);              // Eq. (24) near-wall balance through the solver's face loop
  void printDiagnostics(void);                  // AMR-boundary spike / pseudo-2D diagnostics
  void computeVortexError(void);
  void computeSvortexError(void);                // L2 error vs the exact stationary isentropic vortex
  void computeCanalMetrics(const char *cpFile); // paper Sect. 4.2: mass-flow-rate error Eq. (39) + floor Cp
  void computeGreshoError(void);                // L2 velocity error + KE retention vs the exact Gresho vortex
  void totalConserved(double &mass, double &momx, double &energy); // domain totals of the conserved variables
  void paintPressure(const char *fileName);     // render the pressure field to a png
  void paintDetail(const char *fileName, i32 mode = 0);  // render the wavelet-detail indicator (white = refine trigger)

  __device__ Vec5 prim2cons(Vec5 prim);
  __device__ Vec5 cons2prim(Vec5 cons);
  __device__ real lim(real &r);
  __device__ real tvdRec(real &ul, real &uc, real &ur, real theta = (real)1, i32 rc = -1);   // rc: limiter override (-1 = recon)
  // van Leer harmonic limiter, unconditionally.  The paper leaves the mass /
  // momentum / energy MUSCL UNLIMITED but limits the turbulence convection, to
  // stop k~ and tau~ going negative at the boundary-layer edge -- so this is a
  // separate entry point rather than a `recon` mode.
  __device__ real tvdRecVanLeer(real ul, real uc, real ur);
  __device__ Vec5 hllcFlux(Vec5 qL, Vec5 qR, Vec3 normal);
  __device__ real viscosity(real T);   // constant mu, or Sutherland when sutherS > 0

  __host__ __device__ real wallDistance(Vec3 pos);   // distance to the nearest viscous wall (RANS)
  __host__ __device__ Vec3 wallNormal(Vec3 pos, real h);   // unit normal, body -> fluid
  __host__ __device__ bool isFluidCell(Vec3 pos, real h);  // entirely outside the body (UTCart: intersecting = non-fluid)
  // Ghost-free immersed boundary: no stencil may read a non-fluid cell, so every
  // difference that would reach into the body is degraded instead.  1 = the
  // paper's formulation (no immersed ghost values at all), 0 = fill them.
  // --ibface 1 (sharp IB): build the MUSCL stencil from the IB FACE VALUE at its
  // TRUE wall distance instead of reading a mirror ghost.  For a slip wall the
  // face state is (rho_c, u_c - (u_c.n)n, p_c), so linearly extrapolating it to
  // where the stencil expects a neighbour, a distance dx away, gives
  //     u_nbr = u_c - s (u_c.n) n,     s = dx / d_wall,
  // with d_wall the distance from the cell centre to the wall ALONG that
  // stencil direction (= -phi/|n.e|).  The classic mirror ghost is the special
  // case s = 2, i.e. it ASSUMES the wall bisects the cell; with a level set the
  // wall sits anywhere in (0, dx) and s = 2 is then simply the wrong slope.
  // rho and p are Neumann at a slip wall, so their stencil value is unchanged.
  i32 ibFaceRec;
  i32 ibGhostFree;
  // --wmles 1: wall-modeled implicit LES.  The RANS wall model's MEAN-flow
  // boundary flux (IP sample -> log-law u_tau -> tau_w + pressure) runs with
  // NO turbulence transport: molecular nu feeds the Newton solve, FwK/FwT
  // stay zero, and the scheme's own dissipation is the SGS model (ILES).
  // Slip upstream of wmX0 exactly as under --rans.
  i32 ibWmles;
  // --detflux 1 (default): deterministic face-flux assembly.  The mean-flow
  // scatter's atomicAdds give each cell's RHS 3+ contributors in warp-schedule
  // order; fp addition is non-associative, so every run rounds differently
  // (measured: rhoV differs at 1e-7 after 8 steps, and 40k steps of chaotic
  // amplification turn that into a 1% Cf band -- or, at the outflow corner,
  // into detonation in one run out of two).  Instead each thread WRITES its
  // west/south(/back) face flux to a face bank (each face has exactly one
  // computing thread already) and a second kernel gathers the 4(6) faces per
  // cell in a fixed expression order: zero atomics, bitwise-reproducible.
  // Resolved OFF when the Brinkman face weights are active (they weight the
  // two sides differently) and on the multiD path.
  // --p1 1: modal P1 discontinuous Galerkin.  Every cell evolves a mean and two
  // slopes per conserved variable (F_P1S), so no reconstruction stencil exists:
  // a face flux is the Riemann flux of the two cells' own polynomial traces at
  // 2 Gauss points, a level jump is the same thing on the mortar sub-faces (the
  // coarse trace taken at the physical point), and the volume term is a 2x2
  // Gauss rule on the cell's own polynomial.  See p1RhsKernel.
  i32  p1;
  i32  detFlux;
  // --shash 1: XOR state hash of banks 0..4 printed per step (XOR is
  // associative, so the atomic accumulation is itself order-independent);
  // --shash 2 adds per-phase hashes inside [--shashfrom, --shashto].  The
  // determinism bisection tool: diff two runs' logs, first differing line
  // localizes the step (then the phase) where roundoff first departs.
  i32  shash, shashFrom, shashTo;
  real saDFloor;  // SA destruction wall-distance floor, in local cells (0 = off)
  // 0 = plain SLIP ghosts (DEFAULT, user's call 2026-08-29 + measured);
  // 1 = the old log-law ghost wall function.  Applying the model at the face
  // AND through the ghosts treats the wall twice and decambers a curved body:
  // RAE 2822 Cl 0.410 -> 0.675 and the shock reappears at 0.561c (exp 0.55)
  // just by switching to slip ghosts, while the Re-5e6 plate gate stays in
  // band (Cf +3.7% vs +3.0%, tol 4%).  Matches the architecture rule that the
  // wall model reconstructs the FACE FLUX state only.
  // ---- point-vortex far field (--ffvortex) --------------------------------
  // Thomas & Salas (AIAA J 24, 1986): a lifting body's far field is NOT the
  // freestream -- it decays only as 1/r, so at 12 chords the induced upwash
  // is still O(1%) of V_inf and the circulation (hence C_l) is under-resolved.
  // Superpose the compressible point vortex of the body's own circulation on
  // the outer boundary and the error drops to O(1/r^2).
  i32  ffVortex;        // 1 = correction on
  i32  ffEvery;         // recompute the circulation every N steps
  real ffGamma;         // Gamma = 0.5 V_inf c C_l  (updated from the surface force)
  real ffXv, ffYv;      // vortex centre (quarter chord)
  real ffCl;            // last measured C_l (diagnostic)
  i32  ffPrints;
  i32  wmGhost;
  i32  wmCurv;    // 1 = curvature pressure correction on the wall-model face
  real wmClip;    // --ibwm modes: clamp near-wall SA mu_t to rho*kappa*u_tau*d
                  // within this many local cells of the wall (0 = off)
  real *ffBuf;    // face banks: [(dir*5+n)*ffN + cellIdx], dir 0=x 1=y 2=z
  u64  ffN;       // bank stride (cells); grown with the hash table
  real wmX1;      // model cutoff before the outflow: faces with x > wmX1 fall
                  // back to the slip trace (the outflow-corner analogue of
                  // wmX0; the corner cell otherwise digs a deficit that feeds
                  // an exponential v-mode and detonates -- measured e-fold
                  // ~0.2t, blowup ~t=1.1 in 1 of 2 identical wmles runs).
  // Diagnostic: freeze ONE d_FC-dependent term of the wall model at the value it
  // would take for d_FC = 0.5h (the stable geometry), leaving the other three
  // live.  Isolates which term drives the large-d_FC instability.
  //   0 = off, 1 = Eq.(36) uFc, 2 = Eq.(39) wallBcKTau, 3 = Eq.(A.5) phiDamp,
  //   4 = image-point standoff
  i32 ibDfcMode;
  // 1 = treat the wall-model k~/tau~ boundary flux point-implicitly (see
  // F_LAMK/F_LAMT above) and drop its explicit dt cap -- the cap is what held
  // the whole solver ~3x below the acoustic CFL, binding at the leading-edge
  // first cell.  0 = the explicit cap (the pre-2026-08-25 behaviour).
  i32 wallPointImplicit;
  // Local time stepping (Jameson): every cell marches at its OWN stable step
  // instead of the global minimum.  The transient becomes non-physical -- it is
  // a pseudo-time march to steady state -- so this is for STEADY runs only, and
  // the reported `t` becomes a pseudo-time counter rather than physical time.
  // Off by default; --lts 1 enables.  ltsRatio caps dt_local/dt_global.
  i32 lts;
  real ltsRatio;
  // Low-Mach (Turkel/Weiss-Smith) local preconditioning, STEADY runs only.
  // The acoustic eigenvalues |u|+c are rescaled to O(|u|), which is the actual
  // stiffness at Ma 0.2 (c = 5u).  The conservative-variable preconditioner is
  // an exact RANK-ONE update -- see precondResidual() -- so it costs ~15 flops
  // per cell and, because P is nonsingular, leaves every steady state (R = 0)
  // untouched.  precondK floors beta^2 at K*M_inf^2 so stagnation points stay
  // well conditioned.
  // Time-integration scheme.  0 = the 3-stage 3rd-order LSRK (time-accurate
  // default), 1 = Jameson 4-stage, 2 = Jameson 5-stage.  The Jameson schemes
  // are the classic steady-state workhorses: only 1st order in time for a
  // nonlinear problem (irrelevant for a pseudo-time march to steady state) but
  // with a much larger stability region PER STAGE.  They need no extra storage
  // here -- Jameson's q_k = q_n + alpha_k dt L_{k-1} is exactly Williamson 2N
  // with B_k = alpha_k and A_k = -alpha_{k-1}/alpha_k, so switching schemes is
  // a pure coefficient change in updateFieldsKernel.
  i32 paintOn;     // 0 = never build the uniform-fine PNG image (see Main --paint)
  // Wall reconstruction along the normal: 0 = one node at the adjacent cell
  // centre, linear (default); 1 = two image points ON the normal, sampled
  // biquadratically, quadratic in s closed by the wall gradient condition.
  i32 ibIpQuad;
  // 1 = close the wall trace on (entropy, total enthalpy) rather than holding
  // (p, rho) Neumann.  s and H are the quantities constant along the normal at
  // an adiabatic slip wall; p and rho are not.  See ibFaceTraceFlux.
  i32 ibThermoRec;
  // 1 = constrained quadratic weighted-least-squares wall trace: fit every
  // fluid cell in a 5x5 window, with u_n Dirichlet at the foot point, u_t free,
  // and (s,H) Neumann along the normal.  2-D only; falls back otherwise.
  // ---- implicit ghost-cell method -----------------------------------------
  // Natural mirror image point (reflect the ghost across the wall) with
  // quadratic interpolation.  The compact stencil then CONTAINS ghost cells, so
  // the ghost values satisfy a coupled linear system G = M G + b; ibGIter Jacobi
  // sweeps before each stage solve it.  The existing scheme instead pushes the
  // image point out to s* = 2h purely to keep the stencil in fluid, which
  // samples 2h from the wall and drops the interpolation order where it matters.
  i32 ibGMirror;   // 1 = natural mirror + ghosts allowed in the stencil
  i32 ibGIter;     // ghost-fill sweeps per stage (1 = explicit, the old behaviour)
  // Floor on the mirror distance, in cells.  |phi| -> 0 for a cell centre that
  // happens to sit near the wall, and the image point then lands ON the wall
  // where the fit degenerates -- a small-cell problem in ghost form.  The floor
  // trades the mirror's accuracy for robustness; too small and the error blows
  // up at high resolution (11% velocity overshoot at N=512 with 0.25h).
  real ibGFloor;
  // Ali et al. Sect. 2.5 slip wall: interpolate at the wall point and drop the
  // normal velocity, instead of mirroring through an image point.  No s* at all.
  i32 ibGSlip;
  // Ali et al. interface-cell architecture: prescribe the first fluid layer
  // each stage instead of imposing wall face states.  0 = off, 1 = paper
  // (bilinear implicit), 2 = implicit triquadratic.
  i32 ibIface;
  // RCCM: 1 = cut-cell discretisation with reconstructed small cells.
  i32 ibRccm;
  // RCCM cell taxonomy (their Sect. 2.3.3), from the cached geometry:
  //   alpha == 0            dead, outside the domain
  //   0 < alpha, phi <  0   NR-Cell: advanced by the cut FVM (Eq. 9)
  //   0 < alpha, phi >= 0   R-Cell : small cell, RECONSTRUCTED (Eq. 10) and
  //                         excluded from the dt reduction (Eq. 11)
  __device__ bool rccmLive(i32 cIdx) {
    // merged: a member sliver is live through its element (alpha_E > 0)
    if (cutMerge && cutAlphaE) return cutAlphaE[cIdx] > ibRccmAlphaMin;
    return getField(F_CUTA)[cIdx] > ibRccmAlphaMin;
  }
  // ---- Brinkman volume penalization, PRESSURE-TIGHT form -------------------
  // Reiss, "A family of energy stable, skew-symmetric finite difference schemes
  // on collocated grids" / the non-stiff pressure-tight penalization
  // (docs/pressureTIghtBrinkman.pdf).  The body is not masked at all: the Euler
  // equations are solved everywhere on a smeared volume fraction
  //     phi(s) = eps + (1-eps) (1 + tanh(s/delta))/2,   s = signed distance,
  // and the wall enters as (a) porosity weights phibar_f/phi_c on every face
  // flux and (b) a p grad(phi) momentum source built from the SAME face
  // porosities.  Sharing phibar_f between the two is what makes a quiescent
  // uniform-pressure state cancel bit-for-bit -- the "pressure-tight" property.
  // Restored 2026-09-02 on the brinkman branch (it had been deleted in the
  // 2026-08-29 IB cleanup); this is the inviscid SLIP wall only -- the no-slip
  // penalization, the wall model and the RANS band sources stay gone.
  i32  ibBrink;        // 1 = volume penalization instead of the sharp/cut IB
  real ibBrinkEps;     // volume fraction deep inside the body (paper: 1e-6..1e-8)
  real ibBrinkDelta;   // tanh band half-width, in FINEST cells
  i32  brinkNSeg;      // sub-segments per face in the stamped porosity quadrature
  __host__ __device__ real brinkPhi(real s, real h);
  __host__ __device__ real brinkPhiFaceAvgSeg(Vec3 p0, Vec3 p1, real h, i32 nseg);
  // ---- point-implicit cut cells (--cutpi) ----------------------------------
  // The small-cell problem and the Brinkman porosity stiffness are the SAME
  // problem: a cell whose update carries a 1/alpha (or 1/phi) amplification of
  // its own flux divergence.  Split it exactly as --brinkpi 2 does,
  //   (1/alpha) sum_f F_f A_f / dV = sum_f F_f A_f / dV
  //                                + (1/alpha - 1) sum_f F_f A_f / dV,
  // and stamp the second, which is local, on F_LAMM; the update then divides by
  // (1 + B dt lambda).  Two consequences that matter:
  //   * fixed points are untouched, so the converged state still satisfies
  //     sum_f F_f A_f = 0 per cell -- the exactly conservative cut-cell answer.
  //     That is the whole point: RCCM instead RECONSTRUCTS its small cells and
  //     is non-conservative because of it (the paper's FRM/FIM/FCM exist to
  //     repair exactly that).  Here every live cell is advanced by its true flux.
  //   * no lag: with B dt lambda >> 1 the update is ~ R alpha h/(B(|u|+a)) while
  //     R itself ~ (|u|+a) dU/(alpha h), so alpha cancels and a sliver relaxes an
  //     O(1) fraction per stage however small it is.
  // Only the DIAGONAL is implicit -- neighbour coupling stays explicit, so a
  // sliver bounded by slivers is outside the argument (the cheap member of the
  // mixed explicit/implicit cut-cell family, cf. May & Berger).
  // --cutpidamp 0 turns the point-implicit DIAGONAL off while still advancing
  // every live cell, i.e. a pure explicit cut-cell scheme.  The diagonal is the
  // only term in the cut-cell update that does not telescope: the face fluxes
  // cancel exactly between neighbours (same aperture, each divided by its own
  // alpha, against a conserved quantity alpha*dV*U), but 1/(1 + B dt lambda) is
  // PER CELL, so while Rhs != 0 the shared-face contributions no longer cancel.
  // It vanishes at the fixed point, so it costs nothing at steady state -- but
  // it is exactly the transient conservation error measured in the closed box.
  // Off, the scheme is exactly conservative and pays the alpha-limited step.
  // Cut-cell WALL PRESSURE (--cutpw, was --rccmpw): extrapolate p to the wall face
  // along the cell's own pressure gradient instead of using the cell average,
  // which captures the centripetal balance dp/dn = rho u_t^2 kappa on a curved
  // wall.  Survives the R-cell removal: a wall-flux term, not reconstruction.
  i32 cutGeom;   // 1 = linear corner cut (default), 2 = curved Q2 moment fitting,
                 // 3 = exact clipping by the body segments (CutClip.h; thin bodies, split cells)
  i32 cutSeg;    // --cutgeom 3 on an ANALYTIC body: number of segments it is polygonised with
  i32 ibRccmPw;
  // --ibdir 1: the immersed segments carry the exact state (verification cases).
  i32 ibDirichlet;
  i32 cutPiDamp;
  real ibRccmAlphaMin;
  i32  gradLim;        // recon 6 limiter: 0 = none, 1 = Barth-Jespersen, 2 = Venkatakrishnan
  real gradLimK;       // Venkatakrishnan threshold: eps^2 = (K h)^3
  // The immersed boundary carries the EXACT solution as a Dirichlet datum
  // (Ndiaye et al. Sect. 4.4: "the analytical solution is imposed as Dirichlet
  // boundary condition on all the boundaries").  Under RCCM the R-Cell fit gets
  // a boundary row for every primitive and the wall segment takes an HLLC flux
  // against the exact state; under the ghost path the ghosts are PRESCRIBED
  // (kind 1) and simply keep the exact initial state.  0 = slip wall.
  // Supersonic vortex on the QUARTER annulus of the paper (Fig. 16): centre at
  // the domain corner, inflow/outflow through the straight x = 0 / y = 0 planes
  // as exact Dirichlet domain boundaries (bcType 6).  The closed full annulus
  // (svQuarter 0) traps its acoustics and its conservation error forever; the
  // open quarter lets both leave, so a genuine steady state exists.
  i32 svQuarter;
  // Transonic canal with a 10% circular-arc bump (Ni 1982; paper Sect. 4.2,
  // immerserdBcType 10 / bcType 7 / icType 15).  The floor (y = canalY0), the
  // bump (disc ibCenter/ibRadius) and the ceiling (y = canalY1) are all
  // immersed; the inlet holds total conditions p0 = rho0 = 1 and the outlet the
  // static pressure canalPout of an isentropic M = canalMa stream.  canalPin /
  // canalRhoIn / canalUin are the inlet static state, used for the initial
  // field and as the Cp reference.
  real canalY0, canalY1, canalMa, canalPout, canalPin, canalRhoIn, canalUin;
  // Ducros-like shock sensor gain for recon 5 (the DG solver's dgAvNuKernel
  // formulation, --avk there): theta = comp^2/(comp^2 + kSensor c^2/h^2) with
  // comp = min(div u, 0) -- compression rate against acoustic rate, so smooth
  // acoustics leave theta ~ 0 and the parabola unlimited.
  real kSensor;
  i32 ibWls;
  // Relative threshold for the Appendix-A (A.7) fallback at a wall face.  The
  // (A.6) branch carries the ratio tau~_1^2/tau~_FC^2, which makes the wall
  // tau~ flux CUBIC in the first cell's tau~; this caps that amplification at
  // 1/ransA7Tol^2.  The paper's condition is "tau~_LR ~ 0", a statement about
  // scale, so a relative switch is faithful -- only its size is a choice.
  real ransA7Tol;
  // x beyond which an immersed face is WALL-MODELLED; slip upstream of it.
  // Negative = follow plateX0 (the old coupled behaviour).  These are two
  // different things and conflating them is what destabilises an immersed
  // plate: for a grid-aligned wall plateX0 is a SLIP RUN-UP on a wall that
  // already spans the domain, but for ibtype 5 plateX0 is the plate's own
  // LEADING EDGE, so the model starts at the sharp tip with no run-up at all.
  // 0 = k~-tau~ SST (default), 1 = Spalart-Allmaras with the Tamaki near-wall
  // modification.  In SA mode rho*nu~ occupies the F_RHOK slot and F_RHOTAU is
  // idle, so the field count, block sort, halo exchange and domain BCs are all
  // untouched.  See SaModel.h for why SA survives an explicit immersed wall
  // coupling where the two-equation model does not.
  // 1 = ghost-cell wall-function architecture (Processes 2024): the wall model
  // enters ONLY through the ghost tangential velocity and the wall face takes
  // the ordinary flux.  Forces filled ghosts.  0 = Tamaki prescribed FC flux.
  i32 ibWallMode;
  i32 ibInfinite;      // ibtype 5: infinite plane, no tip (see getBoundaryLevelSet)
  i32 turbModel;
  real nutInf;         // freestream nu~ (SA convention: a few times nu)
  real wmX0;
  real wmRamp;       // blend-in fetch for the immersed wall model past wmX0 (0 = step)
  // ---- pressure-tight volume penalization (Reiss 2021, arXiv:2103.08144;
  //      docs/pressureTIghtBrinkman.pdf).
  //  The body is NOT masked: every cell stays fluid and the object is a region
  //  of vanishing volume fraction phi = V_fluid/V_total.  Paper Eqs. (4)-(6),
  //      d_t(phi rho) + d_a(phi rho u_a) = 0
  //      d_t(phi rho u_a) + d_b(phi rho u_b u_a) + phi d_a p = phi chi(ut_a - u_a)
  //      d_t(phi rho et) + d_a(phi rho u_a et + phi u_a p) = 0,
  //  i.e. phi scales EVERY flux and divides the whole RHS, with the pressure
  //  written d_a(phi p) - p d_a(phi) so momentum keeps a flux form plus the
  //  source p grad(phi).  Those pieces are inseparable: the source on its own
  //  is an unbalanced body force (that is how the deleted DG port applied it,
  //  and it blows up immediately).  Because phi leaves the speed of sound
  //  untouched, the usual CFL condition still holds.

  // Supersonic vortex (testCase 16): EXACT steady solution of the 2-D Euler
  // equations in a concentric annulus.  With rho_i = 1 and p = rho^gam/gam the
  // sound speed at the inner wall is 1, so |u| = M_i r_i / r and
  //   rho(r) = [1 + (gam-1)/2 M_i^2 (1 - r_i^2/r^2)]^{1/(gam-1)}.
  // Verified against radial momentum: dp/dr = rho M_i^2 r_i^2 / r^3 = rho u^2/r.
  // A FULL annulus needs no inflow/outflow BC at all -- the only boundaries are
  // the two curved slip walls, so every bit of the error is the wall treatment.
  // Ringleb flow (exact, curved streamline walls) -- see CompressibleSolver.cu
  real ringlebKmin, ringlebKmax, ringlebScale, ringlebQmin, ringlebX0, ringlebY0;
  __host__ __device__ void ringlebHodograph(real q, real k, real &x, real &y,
                                            real &rho, real &p) const;
  __host__ __device__ bool ringlebInvert(real x, real y, real &q, real &k) const;
  __host__ __device__ void ringlebExact(real x, real y, real &rho, real &u,
                                        real &v, real &p) const;
  void computeRinglebError(void);
  __host__ __device__ i32 getBoundaryBcKind(Vec3 pos);   // 0 = wall, 1 = prescribed
  // Per-SEGMENT boundary-condition tag, parallel to ibPoly.  All geometry is a
  // segment list (triangles later, in 3-D); the tag says what each piece of the
  // boundary MEANS, so one closed loop can carry walls and inflow/outflow at
  // once.  nullptr = every segment is a wall (the airfoil case).
  i32 *ibPolyBc = nullptr;
  // Does the closed loop bound the SOLID (an airfoil) or the FLUID (a duct such
  // as Ringleb)?  Same segments, opposite sign -- getting this wrong marks the
  // entire flow region solid, which is silent: the run completes and every norm
  // reports zero area.
  i32 ibPolyFluidInside = 0;
  real ibPolyBcTol = 0.0;   // junction width (x chord) biasing a corner to PRESCRIBED
  // Boundary STATE per segment (rho,u,v,p), evaluated FORWARD at setup.  A
  // prescribed ghost needs the state ON the boundary, not at its own centre --
  // and its centre lies outside the map where the inversion fails, which is how
  // the ghosts were silently getting the uniform fallback state.
  real *ibPolyState = nullptr;
  void setPolyline(const real *xy, const i32 *bc, const real *st, i32 n);
  __host__ __device__ void svortexExact(real x, real y, real &rho, real &u,
                                        real &v, real &p);
  // exact state of the running verification case (icType 13 / 14) at (x, y);
  // the bcType 6 exact-Dirichlet domain boundary and the error norms use it
  __host__ __device__ bool exactState(real x, real y, real &rho, real &u,
                                      real &v, real &p);
  i32 ibFieldAllLvls; // writeIbField dumps every leaf, not just the finest level
  real ibRadius2;     // OUTER radius (immerserdBcType 7, annulus)
  real svMach;        // supersonic-vortex Mach number at the inner wall
  i32  ibHo;          // 1 = FRIB high-order (k=2) wall condition in H/S form + curvature-consistent ghosts
  i32 ibTurbFlux;  // 0 = drop the k~/tau~ wall fluxes (diagnostic)
  i32 gridTrace;   // dump the grid at each level of the initial build cascade
  i32 adaptEvery;  // wavelet adaptation cadence in steps (~5 host-device syncs per call)
  i32 dtEvery;     // recompute the global dt every this many steps (the reduction is a hard sync)
  i32 maxIter;     // stop the march after this many steps (0 = no cap)
  i32 residEvery;  // steady-residual cadence in steps (0 = never); the reduction is a hard sync
  real resTol;     // stop the march when R/R0 falls below this (0 = never stop early)
  real resid;      // last computed RMS residual
  real residFar;   // RMS over cells > 4h from the immersed body
  real residMax;   // largest per-cell |L|
  real residMaxDw; // wall distance (local h) of that cell
  real resFar;     // exclusion radius (local h) for the wall-free residual
  real resid0;     // first computed RMS residual (normalizer)
  i32  envCheck;      // --envcheck: per-step envelope check; prints the FIRST out-of-bounds cell + its neighborhood
  i32  envPrints;     // envelope reports emitted (capped)
  void envCheckStep(void);   // run the check and report a hit
  real dtDipThresh;   // --dtdip: report the argmin cell when the stable step drops below this (0 = off)
  i32 dtDipPrints;    // dip reports emitted (capped so a deep dip cannot flood the log)
  i32 dtDipCooldown;  // computeDeltaT calls to skip before the next report
  void reportDtMinCell(const char *tag);   // locate + print the cell that owns the current dt min
  // FRIB-style flux reconstruction for the EULER slip wall: 1 = ghost states +
  // ordinary Riemann flux, 2 = per-face wall trace + Riemann (default).  The
  // RANS path always uses the wall-model boundary flux and ignores this.
  i32 ibFluxRecon;
  // FRIB curvature term in the immersed slip reconstruction.  Measured on the
  // inviscid cylinder (exact max|rhoV| = 1.0): OFF 0.99931, ON 1.0408 -- FRIB
  // uses it for a Riemann TRACE in a high-order scheme, whereas here it feeds a
  // prescribed flux with no upwinding to damp it.  Default OFF; --ibcurv 1.
  i32 ibCurv;
  // ARCHITECTURE (fixed): the wall model reconstructs the FACE FLUX state only;
  // ghost cells are ALWAYS a plain slip wall.  ghostSlip is retained so old
  // command lines still parse -- the wall-model ghost branch has been deleted,
  // so setting it to 0 no longer changes anything.
  i32 ghostSlip;
  i32 rkScheme;
  i32 nRkStages;
  i32 precond;
  real precondK;
  real precondMref2;   // M_inf^2, set on the host
  // last-step rescale: when the host clamps the global step to land exactly on
  // an output time, the local steps are scaled by the same factor
  real dtScale;
  // Minimum image-point standoff ABOVE THE WALL FACE, in cells.  The IP stencil
  // must not reach the wall-adjacent fluid cell: that cell is slaved to the very
  // boundary flux being computed from the IP, so any interpolation weight on it
  // closes a feedback loop.  Measured growth in max|rhoV| per output rises
  // monotonically with that weight w = ipStandMin*h - (d_IP - d_FC):
  //   w = 0.00 -> -0.9% (decaying)      w = 0.45 -> +13.8%
  //   w = 0.25 -> +1.0%                 w = 0.50 -> +15.3%
  //   w = 0.35 -> +9.8%                 w = 0.70 -> +25.3%
  // 2.5 puts the lowest tap on the third fluid cell centre, which is exactly
  // what the validated grid-aligned model does (cUp = 2.0, m0 = 2, wgt = 0).
  real ipStandMin;
  __host__ __device__ real getBoundaryLevelSet(Vec3 pos);
  __device__ real calcIbMask(real phi);

};

#endif
