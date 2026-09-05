#include <cstdio>
#include "CompressibleSolverKernels.cuh"
#include "KtauSst.h"
#include "SaModel.h"

//
// field layout (see CompressibleSolver.cuh):
//   0..4   : Rho, RhoU, RhoV, RhoW, RhoE   (conservative)  /  Rho, U, V, W, P (primitive)
//   5..9   : Old{Rho,RhoU,RhoV,RhoW,RhoE}
//   10..14 : Rhs{Rho,RhoU,RhoV,RhoW,RhoE}
//   15     : DeltaT / MagRhoU
//

__global__ void sortFieldDataKernel(CompressibleSolver &grid) {

  START_CELL_LOOP

    i32 bIdxOld = grid.bIdxList[bIdx];
    i32 cIdxOld = bIdxOld * blockSizeTot + cIdx % blockSizeTot;

    // carry all evolved DOFs through the sort
    for (i32 f = 0; f < NEVOLVE; f++) {
      grid.getField(f)[cIdx] = grid.getField(F_OLD + f)[cIdxOld];
    }
    grid.bFlagsList[bIdxOld] = DELETE;

  END_CELL_LOOP
}

// Carry ONE extra field through the block sort: identity snapshot into a
// staging bank, then gather through the new->old block map -- the same two
// steps the evolved DOFs take, but staged through F_SCRATCH because the F_OLD
// double-buffer holds exactly NEVOLVE fields.  Used for the geometry cache
// (F_PHI / F_IBM), which is block payload like the flow variables: carrying it
// keeps the mask valid through EVERY sort, so the re-stamp after adaptation is
// only needed for blocks that did not exist before (carrying preserves, it
// cannot invent).
__global__ void copyFieldKernel(CompressibleSolver &grid, i32 fSrc, i32 fDst) {
  START_CELL_LOOP
    grid.getField(fDst)[cIdx] = grid.getField(fSrc)[cIdx];
  END_CELL_LOOP
}

__global__ void gatherSortedFieldKernel(CompressibleSolver &grid, i32 fSrc, i32 fDst) {
  START_CELL_LOOP
    const i32 bIdxOld = grid.bIdxList[bIdx];
    const i32 cIdxOld = bIdxOld * blockSizeTot + cIdx % blockSizeTot;
    grid.getField(fDst)[cIdx] = grid.getField(fSrc)[cIdxOld];
  END_CELL_LOOP
}

//
// Isentropic vortex (Barsukow-Ciallella-Ricchiuto-Torlo 2025, Sec 6.2.1), a
// smooth stationary (u0=v0=0) or advecting exact solution of the Euler equations
// on a periodic domain.  Background ρ∞=p∞=T∞=1; strength ε=5; centre (cx,cy).
//
__device__ void vortexPaperExact(real x, real y, real u0, real v0, real cx, real cy,
                                 real &rho, real &u, real &v, real &p) {
  const real eps = 5.0;
  real dx = x - cx, dy = y - cy;
  real r2 = dx*dx + dy*dy;
  real f  = eps/(2.0*PI) * exp(0.5*(1.0 - r2));
  u = u0 - f*dy;
  v = v0 + f*dx;
  real dT = -(gam-1.0)*eps*eps/(8.0*gam*PI*PI) * exp(1.0 - r2);
  real T  = 1.0 + dT;
  if (T < 1e-6) T = 1e-6;
  rho = pow(T, 1.0/(gam-1.0));
  p   = pow(T, gam/(gam-1.0));
}

//
// Gresho vortex (Gresho & Chan 1990): a rotating vortex in exact centrifugal
// balance (∂p/∂r = ρ u_φ²/r), hence a stationary solution of the Euler equations.
// ρ=1; peak azimuthal velocity 1 at r=0.2; background pressure p0 = 1/(γ Ma²)
// sets the Mach number.  Standard low-Mach benchmark: dissipative schemes destroy
// the vortex in O(1/Ma) time, low-Mach-preserving schemes keep it stationary.
//
__device__ void greshoExact(real x, real y, real cx, real cy, real p0,
                            real &rho, real &u, real &v, real &p) {
  real dx = x - cx, dy = y - cy;
  real r  = sqrt(dx*dx + dy*dy);
  real wang, pr;                       // wang = u_phi / r  (angular velocity)
  if (r < 0.2) {
    wang = 5.0;
    pr   = p0 + 12.5*r*r;
  } else if (r < 0.4) {
    wang = 2.0/r - 5.0;
    pr   = p0 + 12.5*r*r + 4.0*log(5.0*r) - 20.0*r + 4.0;
  } else {
    wang = 0.0;
    pr   = p0 - 2.0 + 4.0*log(2.0);
  }
  rho = 1.0;
  u   = -wang*dy;
  v   =  wang*dx;
  p   = pr;
}

//
// Exact right-moving simple wave (Riemann): pick a velocity profile u'(x) and
// hold J- = u - 2c/(gam-1) constant, so c = c0 + (gam-1)/2 u', with density and
// pressure from the isentropic relations.  J- constant => no left-going wave at
// any amplitude, so a uniform grid reflects nothing.
//
__device__ void simpleWaveExact(real x, real A, real x0, real wid,
                                real &rho, real &u, real &p) {
  const real rho0 = 1.0, p0 = 1.0;
  const real c0 = sqrt(gam*p0/rho0);
  real arg = (x - x0)/wid;
  u = A*exp(-arg*arg);                          // u'(x)
  real c = c0 + 0.5*(gam-1.0)*u;                // J- = -2 c0/(gam-1) held constant
  rho = rho0*pow(c/c0, 2.0/(gam-1.0));          // isentropic
  p   = p0 *pow(c/c0, 2.0*gam/(gam-1.0));
}

__global__ void setInitialConditionsKernel(CompressibleSolver &grid) {

  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);

    if (grid.icType == 0) {
      //
      // Sod shock tube along x (uniform in y,z -> pseudo-2D / quasi-1D)
      //
      if (pos[0] < 0.5*grid.domainSize[0]) {
        Rho[cIdx] = 1.0;
        U[cIdx]   = 0.0;
        V[cIdx]   = 0.0;
        W[cIdx]   = 0.0;
        P[cIdx]   = 1.0;
      }
      else {
        Rho[cIdx] = 0.125;
        U[cIdx]   = 0.0;
        V[cIdx]   = 0.0;
        W[cIdx]   = 0.0;
        P[cIdx]   = 0.1;
      }
    }

    if (grid.icType == 7) {
      //
      // DG-MATCHED circular Sod blast (comparison runs vs wavedg3d case 1):
      // center of the domain, radius 0.25 L, dgsem strengths rho 11/0.125,
      // p 10/0.1, tanh-smoothed with the DG run's ABSOLUTE width
      // delta = 0.5 * L/1024 (icDelta 0.5 at 7 DG levels, 1024 elems/side).
      //
      real cx = 0.5*grid.domainSize[0], cy = 0.5*grid.domainSize[1];
      real delta = 0.5*grid.domainSize[0]/1024.0;
      real r = sqrt((pos[0]-cx)*(pos[0]-cx) + (pos[1]-cy)*(pos[1]-cy));
      real phi = 0.5*(1.0 + tanh((0.25*grid.domainSize[0] - r)/delta));
      Rho[cIdx] = 0.125 + (11.0 - 0.125)*phi;
      U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0;
      P[cIdx]   = 0.1 + (10.0 - 0.1)*phi;
    }

    if (grid.icType == 1) {
      //
      // 2D circular Sod explosion (uniform in z -> pseudo-2D).  A circular
      // region of high-pressure gas drives a cylindrical shock outward.  The
      // inner pressure is configurable (vortexAdvect, unused otherwise here):
      // pIn = 1 is the classic 10:1 Sod ratio; pIn = 10 a strong 100:1 blast.
      //
      real pIn = (grid.vortexAdvect > 0.0) ? grid.vortexAdvect : 1.0;
      real cx = grid.domainSize[0]/3;
      real cy = grid.domainSize[1]/3;
      real radius = min(grid.domainSize[0], grid.domainSize[1])/5;
      real dist = sqrt((pos[0]-cx)*(pos[0]-cx) + (pos[1]-cy)*(pos[1]-cy));
      if (dist < radius) {
        Rho[cIdx] = 1.0; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = pIn;
      }
      else {
        Rho[cIdx] = 0.125; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = 0.1;
      }
    }

    if (grid.icType == 3) {
      //
      // 3D spherical Sod explosion (true 3D — exercises the z / w paths
      // and the z-flux path).  A high-pressure ball drives a spherical shock.
      //
      real cx = grid.domainSize[0]/2, cy = grid.domainSize[1]/2, cz = grid.domainSize[2]/2;
      real radius = min(min(grid.domainSize[0], grid.domainSize[1]), grid.domainSize[2])/5;
      real dist = sqrt((pos[0]-cx)*(pos[0]-cx) + (pos[1]-cy)*(pos[1]-cy) + (pos[2]-cz)*(pos[2]-cz));
      if (dist < radius) {
        Rho[cIdx] = 1.0; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = 1.0;
      }
      else {
        Rho[cIdx] = 0.125; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = 0.1;
      }
    }

    if (grid.icType == 2) {
      //
      // Isentropic vortex, z-uniform (validates low-Mach / stationarity
      // property against the 2D reference).  Advection velocity u0=v0 selects the
      // stationary (0) vs moving (nonzero) case; centred on the domain.
      //
      real u0 = grid.vortexAdvect, v0 = grid.vortexAdvect;
      real cx = 0.5*grid.domainSize[0], cy = 0.5*grid.domainSize[1];
      real dx = grid.getDx(lvl), dy = grid.getDy(lvl);

      // cell-centre primitives
      real rho, uc, vc, p;
      vortexPaperExact(pos[0], pos[1], u0, v0, cx, cy, rho, uc, vc, p);
      Rho[cIdx] = rho;  U[cIdx] = uc;  V[cIdx] = vc;  W[cIdx] = 0.0;  P[cIdx] = p;
    }

    if (grid.icType == 6) {
      //
      // Periodic right-moving sine acoustic wave for an order-of-accuracy study.
      // Small amplitude (linear); after an integer number of periods the exact
      // solution returns to the IC, so the L2 velocity error there is the scheme's
      // accumulated truncation error.  Amplitude A is carried in vortexAdvect.
      //
      real Lx = grid.domainSize[0];
      real rho0 = 1.0, p0 = 1.0, c0 = sqrt(gam*p0/rho0);
      real A  = grid.vortexAdvect;
      real k  = 2.0*PI/Lx;
      real up = A*sin(k*pos[0]);
      real dup= A*k*cos(k*pos[0]);
      Rho[cIdx] = rho0 + rho0*up/c0;
      U[cIdx]   = up;  V[cIdx] = 0.0;  W[cIdx] = 0.0;
      P[cIdx]   = p0 + rho0*c0*up;
    }

    if (grid.icType == 8) {
      //
      // Viscous shear wave: u = U0 sin(k y), v = w = 0, uniform rho and p.
      // A parallel shear flow is an exact steady solution of the Euler
      // equations, so under constant-mu Navier-Stokes the profile decays as
      // exp(-nu k^2 t) with the nonlinear term identically zero -- an exact
      // check on the viscous operator alone.  Run at low Mach (large p0) so the
      // O(Ma^2) viscous-heating contamination stays below the truncation error.
      //
      real U0 = grid.vortexAdvect;              // shear amplitude
      real p0 = grid.greshoP0;                  // background pressure (sets Mach)
      real k  = 2.0*PI/grid.domainSize[1];
      Rho[cIdx] = 1.0;
      U[cIdx]   = U0*sin(k*pos[1]);
      V[cIdx]   = 0.0;
      W[cIdx]   = 0.0;
      P[cIdx]   = p0;
    }

    if (grid.icType == 5) {
      //
      // EXACT right-moving simple wave (z-uniform).  The left Riemann invariant
      // J- = u - 2c/(gam-1) is held EXACTLY constant, so the wave is a pure
      // right-runner to all amplitudes -- on a uniform grid it produces NO
      // left-going wave (uniform reflection = 0, up to dispersion).  Any dJ- that
      // appears as it crosses the static coarse/fine interface at the domain
      // centre is a genuine numerical reflection.  Launched in the coarse-left
      // half; amplitude A sets the (small) Mach number A/c0.
      //
      real Lx  = grid.domainSize[0];
      real A   = 1.0e-3;            // pulse amplitude in u'
      real x0  = 0.25*Lx;          // start in the coarse-left half (interface at 0.5)
      real wid = 0.03*Lx;
      real dx  = grid.getDx(lvl);
      real rho, uc, p;
      simpleWaveExact(pos[0], A, x0, wid, rho, uc, p);
      Rho[cIdx] = rho;  U[cIdx] = uc;  V[cIdx] = 0.0;  W[cIdx] = 0.0;  P[cIdx] = p;
    }

    if (grid.icType == 14) {
      // Ringleb: initialise with the EXACT solution.  It is steady, so any later
      // L2 error is purely what the scheme and the curved streamline wall did to
      // an exact equilibrium -- the same design as icType 13.
      real rho, uc, vc, pp;
      grid.ringlebExact(pos[0], pos[1], rho, uc, vc, pp);
      Rho[cIdx] = rho;  U[cIdx] = uc;  V[cIdx] = vc;  W[cIdx] = 0.0;  P[cIdx] = pp;
    }

    if (grid.icType == 15) {
      // Canal with bump: the undisturbed inlet stream everywhere (uniform
      // M = canalMa isentropic state from p0 = rho0 = 1); the bump and the
      // boundary conditions do the rest.
      Rho[cIdx] = grid.canalRhoIn;  U[cIdx] = grid.canalUin;  V[cIdx] = 0.0;  W[cIdx] = 0.0;
      P[cIdx]   = grid.canalPin;
    }

    if (grid.icType == 13) {
      // Supersonic vortex: initialise with the EXACT solution.  It is steady, so
      // the L2 error at any later time is purely what the scheme (and the curved
      // immersed wall) has done to an exact equilibrium.
      real rho, uc, vc, pp;
      grid.svortexExact(pos[0], pos[1], rho, uc, vc, pp);
      Rho[cIdx] = rho;  U[cIdx] = uc;  V[cIdx] = vc;  W[cIdx] = 0.0;  P[cIdx] = pp;
    }

    if (grid.icType == 4) {
      //
      // Gresho vortex on [0,1]^2, z-uniform.  Stationary low-Mach benchmark; the
      // Mach number is set by grid.greshoP0 = 1/(gam*Ma^2).
      //
      real p0 = grid.greshoP0;
      real cx = 0.5*grid.domainSize[0], cy = 0.5*grid.domainSize[1];
      real dx = grid.getDx(lvl), dy = grid.getDy(lvl);

      real rho, uc, vc, p;
      greshoExact(pos[0], pos[1], cx, cy, p0, rho, uc, vc, p);
      Rho[cIdx] = rho;  U[cIdx] = uc;  V[cIdx] = vc;  W[cIdx] = 0.0;  P[cIdx] = p;
    }

    if (grid.icType == 10) {
      // Frozen shear u = u0 sin(2 pi y / Ly), v = w = 0, uniform rho and p.
      // The nonlinear term u.grad(u) = u du/dx vanishes identically for this
      // field, so with mu = 0 the velocity NEVER evolves -- which makes the
      // vorticity known analytically and turns the k~ source into something with
      // a closed-form reference.  That is what gates the solver's own S/Omega
      // gradient stencil and the production term, neither of which the uniform
      // box (Omega == 0) can reach.
      const real ky = (real)(2.0*3.14159265358979323846)/grid.domainSize[1];
      Rho[cIdx] = 1.0;
      U[cIdx] = grid.vortexAdvect*sin(ky*pos[1]);
      V[cIdx] = 0.0;  W[cIdx] = 0.0;
      P[cIdx] = 1.0;
    }

    if (grid.icType == 9) {
      // Uniform stream, periodic.  With zero gradients everywhere the convective
      // and diffusive k~/tau~ fluxes must cancel EXACTLY, leaving a pure 0-D
      // source problem embedded in the 3-D solver -- so this gates the new
      // fluxes (free-stream preservation) and the source integration at once.
      Rho[cIdx] = 1.0;
      U[cIdx] = grid.vortexAdvect;  V[cIdx] = 0.0;  W[cIdx] = 0.0;
      P[cIdx] = 1.0;
    }

    if (grid.icType == 12) {
      // Flat-plate turbulent boundary layer: start from the undisturbed
      // freestream everywhere and let the layer grow.
      Rho[cIdx] = 1.0;
      U[cIdx]   = grid.fsU;
      V[cIdx]   = grid.fsV;  W[cIdx] = 0.0;
      P[cIdx]   = grid.fsP;
    }

    if (grid.icType == 11) {
      // Near-wall equilibrium band (testCase 12): the analytic similarity
      // solution of Eqs. (18) with the wall at y = 0, laid on a band of the
      // domain.  Every term of Eq. (24) is then known, and the solver's own face
      // loop has to reproduce the balance -- which is what gates the Appendix-A
      // plumbing (L/R assignment, scatter signs, face coefficients).  With v = 0
      // and k~ uniform in x the convection is identically zero, so what is left
      // IS the source-plus-diffusion balance.
      const real uTau = grid.vortexAdvect;
      const real nu   = grid.mu;
      // wall distance, not height: the wall sits wallOffset below y = 0
      const real y    = fmax(pos[1] + grid.wallOffset, (real)1e-30);
      // The velocity is the MODEL's own profile, not the raw wall function:
      // the wall function above the image point, and the straight line of
      // Eq. (36) below it.  That linearization is what the r_d augmentation of
      // Eq. (38) is built to be consistent with, so this state -- and only this
      // state -- is the equilibrium the discrete operator should reproduce all
      // the way down to the first cell.
      const real dCut = (grid.dCutoff > 0) ? grid.dCutoff : grid.dIpFac*grid.getDy(lvl);
      real uProf;
      if (y >= dCut) uProf = uTau*ktau::uPlus(y*uTau/nu);
      else {
        const real dudy = uTau*uTau/nu*ktau::dUplusDyplus(dCut*uTau/nu);
        uProf = uTau*ktau::uPlus(dCut*uTau/nu) - dudy*(dCut - y);
      }
      Rho[cIdx] = 1.0;
      U[cIdx]   = uProf;
      V[cIdx]   = 0.0;  W[cIdx] = 0.0;
      P[cIdx]   = 1.0;
      grid.getField(F_RHOK)[cIdx]   = ktau::kNearWall(uTau);
      grid.getField(F_RHOTAU)[cIdx] = ktau::tauNearWall(uTau, y);
    }

    // RANS: the turbulence pair starts at its freestream value everywhere.
    if (grid.rans && grid.icType != 11) {
      grid.getField(F_RHOK)[cIdx]   = grid.kInf;
      grid.getField(F_RHOTAU)[cIdx] = grid.tauInf;
    }

  END_CELL_LOOP
}

// ---- surface pressure force (for the far-field vortex circulation) --------
// Staircase integral: every fluid/solid face carries p_cell * n_out * area.
// The face normals are the grid axes, so the y-faces alone carry the lift and
// their total projected area IS the body's projected area -- first order in h,
// which is far more than the far-field Gamma needs.
__device__ double g_ibFx, g_ibFy;

__global__ void ibForceKernel(CompressibleSolver &grid) {
  real *P = grid.getField(F_RHOE);         // primitive bank here
  const real *Ibm = grid.getField(F_IBM);
  const i32 cEmptyI = bEmpty*blockSizeTot;
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    // FINEST LEVEL ONLY.  A coarse PARENT that overlaps the body carries wall
    // faces too, so summing over every level counts the same physical surface
    // once per level -- measured as a ~2x over-estimate of Gamma, which drove
    // the far-field vortex into a positive-feedback runaway (Cl 0.675 -> 1.323
    // where the 48-chord reference says 0.732).  The wall band holds the whole
    // wetted surface at the finest level, so this restriction is exact here.
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)
        && lvl == grid.nLvls-1 && Ibm[cIdx] > (real)0.5) {
      const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
      Vec3 cpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const real p = P[cIdx];
      #define FF_NBR_SOLID(IDX, PX, PY, HH) \
        (((IDX) < cEmptyI) ? (Ibm[IDX] <= (real)0.5) \
                           : !grid.isFluidCell(Vec3((PX),(PY),(real)0), (HH)))
      // outward normal of the BODY points INTO the fluid cell, so the force on
      // the body from this face is +p * (-n_fluid) -- i.e. toward the solid.
      const i32 l1 = grid.getNbrIdx(bIdx, i-1, j, k);
      const i32 r1 = grid.getNbrIdx(bIdx, i+1, j, k);
      const i32 d1 = grid.getNbrIdx(bIdx, i, j-1, k);
      const i32 u1 = grid.getNbrIdx(bIdx, i, j+1, k);
      if (FF_NBR_SOLID(l1, cpos[0]-dx, cpos[1], dx)) atomicAdd(&g_ibFx, -(double)(p*dy));
      if (FF_NBR_SOLID(r1, cpos[0]+dx, cpos[1], dx)) atomicAdd(&g_ibFx,  (double)(p*dy));
      if (FF_NBR_SOLID(d1, cpos[0], cpos[1]-dy, dy)) atomicAdd(&g_ibFy, -(double)(p*dx));
      if (FF_NBR_SOLID(u1, cpos[0], cpos[1]+dy, dy)) atomicAdd(&g_ibFy,  (double)(p*dx));
      #undef FF_NBR_SOLID
    }
  END_CELL_LOOP
}

// ---- compressible point-vortex far-field state (Thomas & Salas 1986) ------
// Gamma > 0 for positive lift.  The perturbation is written in the polar frame
// of the vortex with the Prandtl-Glauert factor beta and the (1 - M^2 sin^2)
// denominator that makes it the exact linearised subsonic vortex; density and
// pressure follow isentropically from the corrected speed, so total enthalpy
// and entropy are preserved on the boundary.
__device__ inline void ffVortexState(CompressibleSolver &grid, real x, real y,
                                     real &r, real &u, real &v, real &p)
{
  u = grid.fsU; v = grid.fsV; p = grid.fsP; r = (real)1;
  if (!grid.ffVortex || grid.ffGamma == (real)0) return;
  const real Vinf = sqrt(grid.fsU*grid.fsU + grid.fsV*grid.fsV);
  const real cInf = sqrt(gam*grid.fsP);                 // rho_inf = 1
  if (!(cInf > (real)0) || !(Vinf > (real)0)) return;
  const real Minf = Vinf/cInf;
  const real b2   = (real)1 - Minf*Minf;
  if (b2 <= (real)0) return;                            // supersonic: no vortex BC
  const real beta = sqrt(b2);
  const real dx = x - grid.ffXv, dy = y - grid.ffYv;
  const real rr = sqrt(dx*dx + dy*dy);
  if (rr < (real)1e-6) return;
  const real th = atan2(dy, dx);
  const real al = atan2(grid.fsV, grid.fsU);
  const real sd = sin(th - al);
  const real den = (real)1 - Minf*Minf*sd*sd;
  const real f = grid.ffGamma*beta/((real)2*(real)M_PI*rr*fmax(den, (real)1e-3));
  u = grid.fsU + f*sin(th);
  v = grid.fsV - f*cos(th);
  // isentropic + constant total enthalpy
  const real q2 = u*u + v*v;
  const real c2 = cInf*cInf + (real)0.5*(gam-(real)1)*(Vinf*Vinf - q2);
  if (c2 <= (real)0) { u = grid.fsU; v = grid.fsV; return; }
  r = pow(c2/(cInf*cInf), (real)1/(gam-(real)1));
  p = grid.fsP*pow(r, gam);
}

// --leaf: a PARENT partner of a boundary cell gets its children's average (defined with the cut helpers)
__device__ inline void leafMaterialise(CompressibleSolver &grid, i32 c);

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 fOff, i32 prim) {
  // operates on fields fOff+0..4 = (Rho, RhoU, RhoV, RhoW, RhoE).  The same
  // operation (copy density+energy, reflect normal momentum) is valid whether
  // the fields currently hold conservative or primitive variables.  fOff selects
  // the state bank (0 = live fields).
  real *Rho  = grid.getField(fOff + F_RHO);
  real *RhoU = grid.getField(fOff + F_RHOU);
  real *RhoV = grid.getField(fOff + F_RHOV);
  real *RhoW = grid.getField(fOff + F_RHOW);
  real *RhoE = grid.getField(fOff + F_RHOE);
  real *RhoK = grid.getField(fOff + F_RHOK);   // k-tau SST pair (0 unless RANS on)
  real *RhoT = grid.getField(fOff + F_RHOTAU);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isExteriorBlock(lvl, ib, jb, kb)) {
      i32 p1Src = -1;   // --p1: the image cell whose slopes this ghost copies (-1 = zero slopes)
      i32 p1Mirror = -1; bool p1FlipU = false, p1FlipV = false;   // --p1: mirror-wall ghost polynomial
      i32 p1Copy = -1;                                                // --p1: interior partner of a copied ghost
      i32 gridSize[3] = {grid.baseGridSize[0]*powi(2, lvl)/blockSize,
                         grid.baseGridSize[1]*powi(2, lvl)/blockSize,
                         grid.baseGridSize[2]*powi(2, lvl)/blockSizeZ};

      if (grid.bcType == 2) {
        // periodic: this exterior ghost block is the wrap-around image of the
        // opposite-edge interior region.  Fill each ghost cell from the FINEST
        // EXISTING block covering its wrapped position: same level if the image
        // region is refined there (exact copy, as before), else walk up the
        // ancestor chain and sample piecewise-constant -- an ordinary
        // coarse/fine ghost.  The seam is then a regular coarse/fine interface
        // and the two edges need NOT be refined to matching levels (the old
        // same-level-image forcing mirrored any seam-touching refinement onto
        // the opposite edge, refining it far from any physical feature).
        i32 nx = gridSize[0]*blockSize, ny = gridSize[1]*blockSize;
        i32 gcx = ((ib*blockSize + i) % nx + nx) % nx;   // wrapped global cell coords at lvl
        i32 gcy = ((jb*blockSize + j) % ny + ny) % ny;
        i32 gcz = kb*blockSizeZ + k;
        if (!grid.pseudo2D) {
          i32 nz = gridSize[2]*blockSizeZ;
          gcz = ((gcz % nz) + nz) % nz;
        }
        i32 imgBlock = bEmpty, dGap = 0;
        i32 cx = 0, cy = 0, cz = 0;
        for (i32 L = lvl, d = 0; L >= 0; L--, d++) {
          cx = gcx >> d;  cy = gcy >> d;
          cz = grid.pseudo2D ? gcz : (gcz >> d);
          // validated lookup: a deleted image block's corpse key must not stop
          // the ancestor walk at a zeroed slot (see getBlockIdx)
          imgBlock = grid.getBlockIdx(grid.encode(L, cx/blockSize, cy/blockSize, cz/blockSizeZ));
          if (imgBlock != bEmpty) { dGap = d; break; }
        }
        if (imgBlock != bEmpty) {   // L=0 always exists (dense base grid)
          real *F[NEVOLVE] = {Rho, RhoU, RhoV, RhoW, RhoE, RhoK, RhoT};
          i32 ox = cx%blockSize, oy = cy%blockSize, oz = cz%blockSize;
          if (dGap == 0) {          // same level: exact periodic copy (as before)
            i32 bcIdx = imgBlock*blockSizeTot + ox + oy*blockSize + oz*blockSize*blockSize;
            if (grid.leafFlux && grid.cFlagsList[bcIdx] == PARENT) leafMaterialise(grid, bcIdx);
            for (i32 f = 0; f < NEVOLVE; f++) F[f][cIdx] = F[f][bcIdx];
            p1Src = bcIdx;
          }
          else {
            // coarser ancestor: monotone (tri)linear interpolation toward the
            // ghost cell centre (positive convex weights -> no overshoot), the
            // same quality as the interior coarse/fine ghost fill.  The fine
            // centre sits at signed offset f. = ((sub+0.5)/2^d - 0.5) cells from
            // the ancestor cell centre; each axis blends the ancestor cell with
            // its neighbour on that side.  A missing neighbour (refinement
            // boundary at the ancestor level) degrades that axis to PC.
            i32 m = (1 << dGap) - 1;
            real fx = ((real)(gcx & m) + 0.5) / (real)(1 << dGap) - 0.5;
            real fy = ((real)(gcy & m) + 0.5) / (real)(1 << dGap) - 0.5;
            real fz = grid.pseudo2D ? 0.0 : ((real)(gcz & m) + 0.5) / (real)(1 << dGap) - 0.5;
            i32 sx = fx > 0 ? 1 : -1, sy = fy > 0 ? 1 : -1, sz = fz > 0 ? 1 : -1;
            real ax = fabs(fx), ay = fabs(fy), az = fabs(fz);
            i32 cEmpty = bEmpty * blockSizeTot;
            i32 t000 = grid.getNbrIdx(imgBlock, ox,    oy,    oz);
            i32 t100 = grid.getNbrIdx(imgBlock, ox+sx, oy,    oz);
            i32 t010 = grid.getNbrIdx(imgBlock, ox,    oy+sy, oz);
            i32 t110 = grid.getNbrIdx(imgBlock, ox+sx, oy+sy, oz);
            i32 t001 = grid.pseudo2D ? t000 : grid.getNbrIdx(imgBlock, ox,    oy,    oz+sz);
            i32 t101 = grid.pseudo2D ? t100 : grid.getNbrIdx(imgBlock, ox+sx, oy,    oz+sz);
            i32 t011 = grid.pseudo2D ? t010 : grid.getNbrIdx(imgBlock, ox,    oy+sy, oz+sz);
            i32 t111 = grid.pseudo2D ? t110 : grid.getNbrIdx(imgBlock, ox+sx, oy+sy, oz+sz);
            // degrade an axis to PC if any tap on that side is missing, and
            // redirect missing taps to the base cell (never read the trash block)
            if (t100 >= cEmpty || t110 >= cEmpty || t101 >= cEmpty || t111 >= cEmpty) ax = 0.0;
            if (t010 >= cEmpty || t110 >= cEmpty || t011 >= cEmpty || t111 >= cEmpty) ay = 0.0;
            if (t001 >= cEmpty || t101 >= cEmpty || t011 >= cEmpty || t111 >= cEmpty) az = 0.0;
            if (t100 >= cEmpty) t100 = t000;  if (t010 >= cEmpty) t010 = t000;
            if (t110 >= cEmpty) t110 = t000;  if (t001 >= cEmpty) t001 = t000;
            if (t101 >= cEmpty) t101 = t000;  if (t011 >= cEmpty) t011 = t000;
            if (t111 >= cEmpty) t111 = t000;
            real w000 = (1-ax)*(1-ay)*(1-az), w100 = ax*(1-ay)*(1-az);
            real w010 = (1-ax)*ay*(1-az),     w110 = ax*ay*(1-az);
            real w001 = (1-ax)*(1-ay)*az,     w101 = ax*(1-ay)*az;
            real w011 = (1-ax)*ay*az,         w111 = ax*ay*az;
            for (i32 f = 0; f < NEVOLVE; f++) {
              F[f][cIdx] = w000*F[f][t000] + w100*F[f][t100]
                         + w010*F[f][t010] + w110*F[f][t110]
                         + w001*F[f][t001] + w101*F[f][t101]
                         + w011*F[f][t011] + w111*F[f][t111];
            }
          }
        }
      }
      else {
        // find the nearest interior cell (zero-gradient reconstruction)
        i32 ibc = i, jbc = j, kbc = k;
        if (ib < 0)            ibc = blockSize;
        if (ib >= gridSize[0]) ibc = -1;
        if (jb < 0)            jbc = blockSize;
        if (jb >= gridSize[1]) jbc = -1;
        if (kb < 0)            kbc = blockSize;
        if (kb >= gridSize[2]) kbc = -1;
        i32 bcIdx = grid.getNbrIdx(bIdx, ibc, jbc, kbc);
        if (grid.leafFlux && bcIdx < bEmpty*blockSizeTot && grid.cFlagsList[bcIdx] == PARENT)
          leafMaterialise(grid, bcIdx);   // idle parent: give it its children's average first
        if (bcIdx < bEmpty*blockSizeTot) p1Copy = bcIdx;

        bool xWall = (ib < 0 || ib >= gridSize[0]);
        bool yWall = (jb < 0 || jb >= gridSize[1]);
        bool zWall = (kb < 0 || kb >= gridSize[2]);

        if (grid.bcType != 4 || !(ib < 0)) {
          Rho[cIdx]  = Rho[bcIdx];      // Neumann density and energy, except at
          RhoE[cIdx] = RhoE[bcIdx];     // the bcType 4/8 faces, which set them
        }

        if (grid.bcType == 0) {
          // slip wall: reflect the wall-normal momentum, keep tangential
          RhoU[cIdx] = (xWall ? -1.0 : 1.0) * RhoU[bcIdx];
          RhoV[cIdx] = (yWall ? -1.0 : 1.0) * RhoV[bcIdx];
          RhoW[cIdx] = (zWall ? -1.0 : 1.0) * RhoW[bcIdx];
          p1Mirror = bcIdx; p1FlipU = xWall; p1FlipV = yWall;
        }
        else if (grid.bcType == 1) {
          // No-slip wall: MIRROR every velocity component, so the face value
          // (the average of the interior cell and its ghost) is zero for all
          // three -- normal AND tangential.  Zeroing the ghost tangential
          // components instead would put the face velocity at HALF the interior
          // value, which is not no-slip; harmless while inviscid, but it sets
          // the wall shear stress once there is a viscous flux.
          // Energy is copied, so the mirrored velocity leaves rho*e unchanged
          // -> zero wall-normal temperature gradient (adiabatic wall).
          RhoU[cIdx] = -RhoU[bcIdx];
          RhoV[cIdx] = -RhoV[bcIdx];
          RhoW[cIdx] = -RhoW[bcIdx];
          p1Mirror = bcIdx; p1FlipU = true; p1FlipV = true;
        }
        else if (grid.bcType == 4) {
          // ---- flat-plate boundary layer: a different role per face ----------
          //   x-min  subsonic inflow  (freestream state)
          //   x-max  outflow          (zero gradient)
          //   y-max  farfield         (freestream state)
          //   y-min  slip wall        (the wall model overwrites x >= plateX0
          //                            afterwards, in wallGhostKernel)
          const bool inflow   = (ib < 0);
          const bool farfield = (jb >= gridSize[1]);
          const bool outflow  = (ib >= gridSize[0]);
          if (inflow) {
            // subsonic inflow: hold the whole freestream state
            const real r = 1.0, u = grid.fsU, v = grid.fsV, p = grid.fsP;
            Rho[cIdx]  = r;
            RhoU[cIdx] = prim ? u : r*u;
            RhoV[cIdx] = prim ? v : r*v;
            RhoW[cIdx] = 0.0;
            RhoE[cIdx] = prim ? p : (p/(gam - 1.0) + 0.5*r*(u*u + v*v));
            RhoK[cIdx] = prim ? grid.kInf   : r*grid.kInf;
            RhoT[cIdx] = prim ? grid.tauInf : r*grid.tauInf;
          }
          else if (farfield || outflow) {
            // Subsonic pressure boundary: extrapolate the velocity and hold the
            // pressure.  The wall-normal velocity MUST be free here -- a growing
            // boundary layer displaces flow outward, and pinning v = 0 a few
            // delta above the plate blocks that displacement and imposes a
            // spurious favourable pressure gradient, which relaminarizes the
            // layer (C_f decays toward its laminar value instead of settling).
            Rho[cIdx]  = Rho[bcIdx];
            RhoU[cIdx] = RhoU[bcIdx];
            RhoV[cIdx] = RhoV[bcIdx];
            RhoW[cIdx] = RhoW[bcIdx];
            RhoK[cIdx] = RhoK[bcIdx];
            RhoT[cIdx] = RhoT[bcIdx];
            if (prim) RhoE[cIdx] = grid.fsP;
            else {
              const real r = fmax(Rho[cIdx], (real)1e-30);
              const real ke = 0.5*(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx]
                                 + RhoW[cIdx]*RhoW[cIdx])/r;
              RhoE[cIdx] = grid.fsP/(gam - 1.0) + ke;
            }
          }
          else if (yWall) {                 // y-min: slip (mirror the normal)
            RhoU[cIdx] = RhoU[bcIdx];
            RhoV[cIdx] = -RhoV[bcIdx];
            RhoW[cIdx] = RhoW[bcIdx];
            RhoK[cIdx] = RhoK[bcIdx];
            RhoT[cIdx] = RhoT[bcIdx];
            p1Mirror = bcIdx; p1FlipV = true;
          }
          else {                            // x-max outflow, and z in pseudo-2D
            RhoU[cIdx] = RhoU[bcIdx];
            RhoV[cIdx] = RhoV[bcIdx];
            RhoW[cIdx] = RhoW[bcIdx];
            RhoK[cIdx] = RhoK[bcIdx];
            RhoT[cIdx] = RhoT[bcIdx];
          }
        }
        else if (grid.bcType == 5) {
          // ---- airfoil farfield ------------------------------------------
          // Every outer face is either inflow or outflow depending on the sign
          // of u_inf . n_out, so one rule covers all four sides at any angle of
          // attack (unlike bcType 4, whose faces have fixed flat-plate roles).
          //   inflow  : hold the whole freestream state (subsonic Dirichlet)
          //   outflow : extrapolate velocity/density, hold the freestream
          //             pressure -- the same subsonic pressure boundary the
          //             flat plate uses at y-max, which is what lets the body's
          //             displacement and wake leave the domain cleanly.
          const real nxo = xWall ? ((ib < 0) ? (real)-1 : (real)1) : (real)0;
          const real nyo = yWall ? ((jb < 0) ? (real)-1 : (real)1) : (real)0;
          const real udn = grid.fsU*nxo + grid.fsV*nyo;
          Vec3 gpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
          real rF, uF, vF, pF;
          ffVortexState(grid, gpos[0], gpos[1], rF, uF, vF, pF);
          if (udn < (real)0) {                      // inflow
            const real r = rF, u = uF, v = vF, p = pF;
            Rho[cIdx]  = r;
            RhoU[cIdx] = prim ? u : r*u;
            RhoV[cIdx] = prim ? v : r*v;
            RhoW[cIdx] = 0.0;
            RhoE[cIdx] = prim ? p : (p/(gam - 1.0) + 0.5*r*(u*u + v*v));
            RhoK[cIdx] = prim ? grid.kInf   : r*grid.kInf;
            RhoT[cIdx] = prim ? grid.tauInf : r*grid.tauInf;
          } else {                                  // outflow
            Rho[cIdx]  = Rho[bcIdx];
            RhoU[cIdx] = RhoU[bcIdx];
            RhoV[cIdx] = RhoV[bcIdx];
            RhoW[cIdx] = RhoW[bcIdx];
            RhoK[cIdx] = RhoK[bcIdx];
            RhoT[cIdx] = RhoT[bcIdx];
            if (prim) RhoE[cIdx] = pF;            // vortex-corrected back pressure
            else {
              const real r = fmax(Rho[cIdx], (real)1e-30);
              const real ke = 0.5*(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx]
                                 + RhoW[cIdx]*RhoW[cIdx])/r;
              RhoE[cIdx] = pF/(gam - 1.0) + ke;
            }
          }
        }
        else if (grid.bcType == 7) {
          // ---- canal with bump (paper Sect. 4.2, Fig. 9) --------------------
          //   x-min  subsonic inlet: total conditions p0 = rho0 = 1 (so R T0 = 1)
          //          and axial flow; the velocity is taken from the interior and
          //          the static state follows isentropically from it
          //   x-max  subsonic outlet: hold the static pressure canalPout,
          //          extrapolate density and velocity
          //   y      slip (the physical floor/ceiling are immersed; these faces
          //          sit in the dead zone above/below them)
          real r = Rho[bcIdx], uu, vv, ww, p;
          if (prim) { uu = RhoU[bcIdx]; vv = RhoV[bcIdx]; ww = RhoW[bcIdx]; p = RhoE[bcIdx]; }
          else {
            uu = RhoU[bcIdx]/r; vv = RhoV[bcIdx]/r; ww = RhoW[bcIdx]/r;
            p = (gam - 1.0)*(RhoE[bcIdx] - 0.5*r*(uu*uu + vv*vv + ww*ww));
          }
          if (ib < 0) {                            // inlet
            // T/T0 = 1 - (gam-1) u^2/(2 gam) with c_p = gam/(gam-1), R = 1;
            // clamp so a transient over-speed cannot drive T through zero
            const real ui = fmax(uu, (real)0);
            const real tr = fmax((real)1 - (gam - 1.0)*ui*ui/(2.0*gam), (real)0.2);
            p  = pow(tr, gam/(gam - 1.0));
            r  = pow(tr, (real)1/(gam - 1.0));
            uu = ui; vv = 0.0; ww = 0.0;
          }
          else if (ib >= gridSize[0]) {            // outlet
            p = grid.canalPout;
          }
          else {                                   // y / z faces: slip
            vv = yWall ? -vv : vv;
            ww = zWall ? -ww : ww;
          }
          Rho[cIdx]  = r;
          RhoU[cIdx] = prim ? uu : r*uu;
          RhoV[cIdx] = prim ? vv : r*vv;
          RhoW[cIdx] = prim ? ww : r*ww;
          RhoE[cIdx] = prim ? p : (p/(gam - 1.0) + 0.5*r*(uu*uu + vv*vv + ww*ww));
        }
        else if (grid.bcType == 6) {
          // ---- exact-solution Dirichlet (verification cases) --------------
          // Every exterior ghost carries the analytic state of the running
          // case at its own position -- the paper's vortex test (Sect. 4.4)
          // imposes the analytical solution on all boundaries.  Supersonic in
          // AND out here, so a full-state Dirichlet is well posed on both.
          Vec3 gpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
          real r, u, v, p;
          if (grid.exactState(gpos[0], gpos[1], r, u, v, p)) {
            Rho[cIdx]  = r;
            RhoU[cIdx] = prim ? u : r*u;
            RhoV[cIdx] = prim ? v : r*v;
            RhoW[cIdx] = 0.0;
            RhoE[cIdx] = prim ? p : (p/(gam - 1.0) + 0.5*r*(u*u + v*v));
          } else {                                  // no exact state: transmissive
            RhoU[cIdx] = RhoU[bcIdx];
            RhoV[cIdx] = RhoV[bcIdx];
            RhoW[cIdx] = RhoW[bcIdx];
          }
        }
        else if (grid.bcType == 8) {
          // ---- characteristic far field (Riemann invariants) ---------------
          // Zero-gradient extrapolation (bcType 3) is REFLECTIVE for subsonic
          // acoustics: a wave reaching the boundary comes back in.  On an
          // external-aero box that is not a slow leak, it is fatal -- measured
          // on the M=0.3 circle in a 20-diameter box, the far field sat at
          // R ~ 3e-2 for 4000 iterations and then blew up, with the residual
          // maximum 80-160 cells from the body, i.e. AT the boundary.  Both the
          // cut-cell and the RCCM path died the same way, which is how we know
          // it was never the body treatment.
          //
          // The standard fix: decompose into the two acoustic Riemann
          // invariants along the OUTWARD normal and take each from the side its
          // characteristic comes from,
          //   R+ = u_n + 2c/(g-1)   (from the interior if u_n > -c)
          //   R- = u_n - 2c/(g-1)   (from the freestream if u_n < c)
          // then u_n = (R+ + R-)/2 and c = (g-1)(R+ - R-)/4.  Entropy and the
          // tangential velocity ride the convective characteristic, so they come
          // from whichever side the flow is entering from.  Supersonic faces
          // degenerate correctly: all four from upstream.
          const real nx = xWall ? (ib < 0 ? (real)-1 : (real)1) : (real)0;
          const real ny = yWall ? (jb < 0 ? (real)-1 : (real)1) : (real)0;
          const real nz = zWall ? (kb < 0 ? (real)-1 : (real)1) : (real)0;
          // interior state (the ghost's mirror partner)
          const real ri = fmax(Rho[bcIdx], (real)1e-30);
          const real ui = prim ? RhoU[bcIdx] : RhoU[bcIdx]/ri;
          const real vi = prim ? RhoV[bcIdx] : RhoV[bcIdx]/ri;
          const real wi = prim ? RhoW[bcIdx] : RhoW[bcIdx]/ri;
          const real pi_ = prim ? RhoE[bcIdx]
                                : (gam-(real)1)*(RhoE[bcIdx]
                                    - (real)0.5*ri*(ui*ui + vi*vi + wi*wi));
          const real ci = sqrt(fmax(gam*pi_/ri, (real)1e-30));
          const real uni = ui*nx + vi*ny + wi*nz;
          // freestream state
          const real re = (real)1, ue = grid.fsU, ve = grid.fsV, we = (real)0;
          const real pe = grid.fsP;
          const real ce = sqrt(fmax(gam*pe/re, (real)1e-30));
          const real une = ue*nx + ve*ny + we*nz;
          const real tg = (real)2/(gam - (real)1);
          // pick each invariant from its own side of the characteristic
          const real Rp = (uni > -ci) ? (uni + tg*ci) : (une + tg*ce);
          const real Rm = (une <  ce) ? (une - tg*ce) : (uni - tg*ci);
          const real unb = (real)0.5*(Rp + Rm);
          const real cb  = fmax((gam - (real)1)*(real)0.25*(Rp - Rm), (real)1e-30);
          // entropy s = p/rho^gamma and the TANGENTIAL velocity convect, so
          // they come from the side the flow enters from.  wI = weight on the
          // INTERIOR state.  The hard switch (ffBlend 0) is discontinuous in
          // u_n, which matters wherever the boundary is nearly parallel to the
          // flow: there u_n ~ 0, the branch flips back and forth, and the ghost
          // moves by the full interior-freestream difference each flip.  A
          // tanh blend of width ffBlend*c makes the ghost a continuous function
          // of the state; it degenerates to the exact upwind choice as soon as
          // |u_n| > a few ffBlend*c, so inlet and outlet faces are untouched.
          const real wI = (grid.ffBlend > (real)0)
                        ? (real)0.5*((real)1 + tanh(unb/(grid.ffBlend*cb)))
                        : ((unb > (real)0) ? (real)1 : (real)0);
          const real ent = wI*(pi_/pow(ri, gam)) + ((real)1 - wI)*(pe/pow(re, gam));
          const real ut = wI*ui + ((real)1 - wI)*ue;
          const real vt = wI*vi + ((real)1 - wI)*ve;
          const real wt = wI*wi + ((real)1 - wI)*we;
          const real rb = pow(cb*cb/(gam*ent), (real)1/(gam - (real)1));
          const real pb = rb*cb*cb/gam;
          const real unt = ut*nx + vt*ny + wt*nz;          // tangential part kept
          const real ub = ut + (unb - unt)*nx;
          const real vb = vt + (unb - unt)*ny;
          const real wb = wt + (unb - unt)*nz;
          Rho[cIdx]  = rb;
          RhoU[cIdx] = prim ? ub : rb*ub;
          RhoV[cIdx] = prim ? vb : rb*vb;
          RhoW[cIdx] = prim ? wb : rb*wb;
          RhoE[cIdx] = prim ? pb : (pb/(gam - (real)1)
                                    + (real)0.5*rb*(ub*ub + vb*vb + wb*wb));
        }
        else if (grid.bcType == 9) {
          // ---- freestream exterior state, all the upwinding left to HLLC ---
          // The boundary face already runs the SAME approximate Riemann solver
          // as every interior face; the only question a far-field BC answers is
          // what state sits on the outside.  Hand it the freestream and there is
          // no branch anywhere: HLLC is continuous where its own wave speeds
          // cross zero (at S_M = 0 the two star fluxes coincide -- mass and
          // energy flux vanish and the momentum flux is p* n), so nothing can
          // chatter as u_n passes through zero.
          // The cost is that a fixed exterior state over-specifies a SUBSONIC
          // outflow -- four conditions where only the back pressure is well
          // posed -- so the boundary pulls the solution toward the freestream.
          // HLLC weights by wave speed rather than imposing it hard, but on a
          // box this close to the body that is still blockage.
          const real r = (real)1, u = grid.fsU, v = grid.fsV, p = grid.fsP;
          Rho[cIdx]  = r;
          RhoU[cIdx] = prim ? u : r*u;
          RhoV[cIdx] = prim ? v : r*v;
          RhoW[cIdx] = (real)0;
          RhoE[cIdx] = prim ? p : (p/(gam - (real)1) + (real)0.5*r*(u*u + v*v));
        }
        else {
          // bcType == 3 : transmissive / outflow (zero gradient)
          RhoU[cIdx] = RhoU[bcIdx];
          RhoV[cIdx] = RhoV[bcIdx];
          RhoW[cIdx] = RhoW[bcIdx];
        }

        // k-tau SST pair: zero-gradient default (exact for transmissive/farfield;
        // the wall-model BC of Eq. (39) overrides it on wall faces once RANS is
        // active).  A no-op while the fields are zero.  bcType 4 has already set
        // them per face above.
        if (grid.bcType != 4 && grid.bcType != 5) {
          RhoK[cIdx] = RhoK[bcIdx];
          RhoT[cIdx] = RhoT[bcIdx];
        }
      }
      // --p1 imposes every physical boundary condition WEAKLY at the boundary
      // face (p1BoundaryState in p1RhsKernel); the exterior ghost carries a
      // polynomial only as the same-level PERIODIC image, else it is inert
      if (grid.p1 && fOff == 0)
        for (i32 s = 0; s < 2*P1_NV; s++)
          grid.getField(F_P1S + s)[cIdx] = (p1Src >= 0) ? grid.getField(F_P1S + s)[p1Src] : (real)0;
      (void)p1Mirror; (void)p1FlipU; (void)p1FlipV; (void)p1Copy;
    }

  END_CELL_LOOP
}

__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);
  real *RhoK = grid.getField(F_RHOK);
  real *RhoT = grid.getField(F_RHOTAU);

  START_CELL_LOOP   // ALL blocks: idle parents must keep the leaves' form (the wavelet prolongs from them)

    real r = Rho[cIdx];
    Vec5 q = grid.cons2prim(Vec5(r, RhoU[cIdx], RhoV[cIdx], RhoW[cIdx], RhoE[cIdx]));
    Rho[cIdx]  = q[0];
    RhoU[cIdx] = q[1];
    RhoV[cIdx] = q[2];
    RhoW[cIdx] = q[3];
    RhoE[cIdx] = q[4];
    // turbulence pair rides along: rho*k~ -> k~, rho*tau~ -> tau~.  Identically
    // zero (hence a no-op) unless the RANS model is active.
    real rInv  = (r > 0) ? 1.0/r : 0.0;
    RhoK[cIdx] = RhoK[cIdx]*rInv;
    RhoT[cIdx] = RhoT[cIdx]*rInv;

  END_CELL_LOOP
}

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(0);
  real *U   = grid.getField(1);
  real *V   = grid.getField(2);
  real *W   = grid.getField(3);
  real *P   = grid.getField(4);
  real *K   = grid.getField(F_RHOK);
  real *Tau = grid.getField(F_RHOTAU);

  START_CELL_LOOP   // ALL blocks: idle parents must keep the leaves' form (the wavelet prolongs from them)

    real r = Rho[cIdx];
    Vec5 q = grid.prim2cons(Vec5(r, U[cIdx], V[cIdx], W[cIdx], P[cIdx]));
    Rho[cIdx] = q[0];
    U[cIdx]   = q[1];
    V[cIdx]   = q[2];
    W[cIdx]   = q[3];
    P[cIdx]   = q[4];
    K[cIdx]   = K[cIdx]*r;      // k~   -> rho*k~
    Tau[cIdx] = Tau[cIdx]*r;    // tau~ -> rho*tau~

  END_CELL_LOOP
}

// lock-free atomic-max (IEEE ordering via integer CAS; values here are >= 0).
// float and double overloads so the kernel builds at either precision.
__device__ inline float atomicMaxFloat(float *addr, float val) {
  int *ai = (int*)addr;
  int old = *ai;
  while (val > __int_as_float(old)) {
    int assumed = old;
    old = atomicCAS(ai, assumed, __float_as_int(val));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}
__device__ inline double atomicMaxFloat(double *addr, double val) {
  unsigned long long *ai = (unsigned long long*)addr;
  unsigned long long old = *ai;
  while (val > __longlong_as_double(old)) {
    unsigned long long assumed = old;
    old = atomicCAS(ai, assumed, __double_as_longlong(val));
    if (old == assumed) break;
  }
  return __longlong_as_double(old);
}

// Wavelet-thresholding scales: domain maxima of the 3 field scales
// {|rho|, |momentum|, |rhoE|} into globalScale[0..2], pre-zeroed by
// the host.  Warp-level shuffle reduction first, then one atomicMax per warp --
// all device-side, no host round-trip.
// DIAG: count near-vacuum cells (rho < 0.05; physical min is the 0.125 ambient),
// split by owned block vs ghost block, to locate an unphysical density overshoot.
__device__ unsigned long long g_vacOwned;
__device__ unsigned long long g_qUsed;     // implicit-ghost quadratic evaluations
__device__ unsigned long long g_qDecl;     // ... declined (unreachable stencil)
__device__ unsigned long long g_qGhostTap;
__device__ unsigned long long g_rcDeadFace;  // face read with a DEAD neighbour and nonzero aperture
__device__ unsigned long long g_rcDeadGrad;  // gradient stencil that skipped a dead tap
__device__ unsigned long long g_rcLiveFace; // ghost taps actually consumed
// Steady-state residual accumulators (see CompressibleSolver::computeResidual).
__device__ double             g_resSum;   // sum of L(q)^2 over live fluid cells
__device__ unsigned long long g_resCnt;   // cells contributing
__device__ double             g_resSumFar;// same, but only cells > 4h from the body
__device__ unsigned long long g_resCntFar;
__device__ double             g_resMax;   // largest per-cell |L|
__device__ double             g_resMaxPhi;// wall distance (-phi) of that cell, in local h

// --- AMR debug probe: locate the FIRST non-finite evolved value ------------
// g_nfKey packs (cIdx) of the lowest-index offender; g_nfCnt counts all of
// them, split by z-layer so a pseudo2D staleness bug is visible directly.
__device__ int g_nfCnt;      // non-finite cells found (k == 0 plane)
__device__ int g_nfCntZ;     // non-finite cells found (k > 0 stale layers)
__device__ int g_nfCidx;     // lowest offending cIdx (INT_MAX if none)
__device__ int g_nfField;    // which field

// Image-point stencil purity (--debug).  The IP must never interpolate from a
// cell whose OWN reconstruction is wall-degraded: under ibGhostFree the remap
// `if (!isFluidCell(cpos - 2h)) d2R = d1R;` slaves such a cell to the forced
// first fluid cell, so any weight on it makes the wall flux read back the state
// it is setting -- an algebraic loop whose gain is that interpolation weight.
// ipStandMin = 2.5 is supposed to hold the support clear; this counts whether it
// actually does, which a scalar distance-along-normal cannot guarantee for a
// general (angled) surface.
__device__ unsigned long long g_ipTaint;
__device__ unsigned long long g_ibDetect;
// ibWallFlux failure-mode census (debug): why a detected wall face got no flux
__device__ unsigned long long g_ibFailDip;    // dIp <= dFc
__device__ unsigned long long g_ibFailSlip;   // slip-branch sample failed
__device__ unsigned long long g_ibFailIp;     // image-point sample failed
__device__ unsigned long long g_ibNup;
__device__ unsigned long long g_wmGhost;   // ghosts taking the wall-function branch
__device__ unsigned long long g_wmCand;    // ghosts reaching the test        // faces with the body ABOVE (n_d < 0)
__device__ double g_ibMaxDfc;                 // worst dFc/h among the failures
__device__ double g_ibMaxLvl;                 // deepest level among the failures
__device__ unsigned long long g_ibFlux;
__device__ unsigned long long g_vacGhost;
__global__ void dbgVacKernel(CompressibleSolver &grid) {
  START_CELL_LOOP
    GET_CELL_INDICES
    i32 lvl, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.isInteriorBlock(lvl,ib,jb,kb)) {
      real r = grid.getField(F_RHO)[cIdx];
      if (r < 0.05) {
#ifdef USE_MGPU
        bool own = grid.isOwnedBlock(lvl,ib,jb,kb);
        unsigned long long n = own ? atomicAdd(&g_vacOwned, 1ULL) : atomicAdd(&g_vacGhost, 1ULL);
        if (n < 4)   // print the first few offenders: where exactly is the bad data?
          printf("[vac] %s rank=%d lvl=%d blk=(%d,%d) cell=(%d,%d) rho=%.3e flag=%d snap=%d\n",
                 own?"OWNED":"ghost", grid.part.rank, lvl, ib, jb, i, j, r,
                 grid.bFlagsList[bIdx], grid.snapValidList[bIdx]);
#else
        atomicAdd(&g_vacOwned, 1ULL);
#endif
      }
    }
  END_CELL_LOOP
}

__global__ void computeGlobalScalesKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(F_RHO);
  real *RhoU = grid.getField(F_RHOU);
  real *RhoV = grid.getField(F_RHOV);
  real *RhoW = grid.getField(F_RHOW);
  real *RhoE = grid.getField(F_RHOE);

  START_CELL_LOOP

    real r = fabs(Rho[cIdx]);
    real mu = RhoU[cIdx], mv = RhoV[cIdx], mw = RhoW[cIdx];
    real m = sqrt(mu*mu + mv*mv + mw*mw);
    real e = fabs(RhoE[cIdx]);

#ifdef USE_MGPU
    // owned-only so the threshold normalization is rank-count-invariant: a
    // ghost is a stale copy of another PE's cell, and the allreduce-max already
    // folds in every PE's owned max.  Zero (the fmax identity for these
    // magnitudes) drops non-owned lanes without breaking the warp shuffle.
    { i32 lvl,ib,jb,kb; grid.decode(grid.bLocList[bIdx], lvl,ib,jb,kb);
      if (!grid.isOwnedBlock(lvl,ib,jb,kb)) { r = 0; m = 0; e = 0; } }
#endif

    // warp shuffle reduction (grid-stride loop keeps whole warps in-range)
    for (int off = 16; off > 0; off >>= 1) {
      r = fmax(r, __shfl_down_sync(0xffffffff, r, off));
      m = fmax(m, __shfl_down_sync(0xffffffff, m, off));
      e = fmax(e, __shfl_down_sync(0xffffffff, e, off));
    }
    if ((threadIdx.x & 31) == 0) {
      atomicMaxFloat(&grid.globalScale[0], r);
      atomicMaxFloat(&grid.globalScale[1], m);
      atomicMaxFloat(&grid.globalScale[2], e);
    }

  END_CELL_LOOP
}

// compute pressure into the scratch field (15) for visualization (fields conservative)
__global__ void computePressureKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);
  real *P    = grid.getField(F_SCRATCH);

  START_CELL_LOOP

    real r = Rho[cIdx];
    if (r > 0) {
      real u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      P[cIdx] = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
    } else {
      P[cIdx] = 0;
    }

  END_CELL_LOOP
}

// compute the local stable time step in each cell (CFL); fields are conservative
// ---- immersed-boundary sampling -------------------------------------------
//
// The image point sits d_IP = 3 dx from the wall along the NORMAL, so for a
// general normal it can be three cells away in every direction -- outside the
// two-cell halo that getNbrIdx can reach.  It is therefore located by HASH
// lookup of the containing block, exactly as the periodic boundary locates its
// wrap image, and interpolated trilinearly from whichever of the surrounding
// cells are fluid (a solid neighbour carries no solution, so it gets no weight).
//
// Trilinear sample of nf fields at an arbitrary point, fluid cells only.
//
// The stencil is reached with getNbrIdx, NOT a hash lookup.  getNbrIdx walks the
// 27-entry neighbour list, so from any cell it reaches +-blockSize cells; the
// image point sits at most d_IP - d_FC <= 2.9h from the face and the face half a
// cell from the cell centre, so the whole stencil is within +-4 cells and one
// ring of neighbour blocks is exactly enough.  ibRing (below) guarantees that
// ring exists around every block the surface passes through, which is what makes
// this safe -- a hash lookup would silently return "no block" instead, and the
// interpolation would quietly lose points.
//
// (gi,gj,gk) are the CURRENT cell's global indices at this level; (ci,cj,ck) its
// local ones.  Returns false if the stencil leaves the reachable ring or finds
// no fluid, so a failure is visible rather than silent.
__device__ inline bool ibSample(CompressibleSolver &grid, Vec3 p, i32 lvl,
                                i32 bIdx, i32 ci, i32 cj, i32 ck,
                                i32 gi, i32 gj, i32 gk,
                                real **F, i32 nf, real *out, bool tally = false,
                                bool allowGhost = false)
{
  const real dx = grid.getDx(lvl), dy = grid.getDy(lvl), dz = grid.getDz(lvl);
  const real fx = p[0]/dx - (real)0.5, fy = p[1]/dy - (real)0.5;
  const real fz = grid.pseudo2D ? (real)0 : p[2]/dz - (real)0.5;
  const i32 i0 = (i32)floor((double)fx), j0 = (i32)floor((double)fy);
  const i32 k0 = grid.pseudo2D ? gk : (i32)floor((double)fz);
  const real tx = fx - (real)i0, ty = fy - (real)j0;
  const real tz = grid.pseudo2D ? (real)0 : fz - (real)k0;

  // cell offsets from the current cell, and the reachable window of getNbrIdx
  const i32 di = i0 - gi, dj = j0 - gj, dk = grid.pseudo2D ? 0 : k0 - gk;
  const i32 lo = -blockSize, hi = 2*blockSize - 2;      // -2 leaves room for +1
  if (ci+di < lo || ci+di > hi || cj+dj < lo || cj+dj > hi ||
      (!grid.pseudo2D && (ck+dk < lo || ck+dk > hi))) return false;

  for (i32 f = 0; f < nf; f++) out[f] = 0;
  real wSum = 0;
  const i32 nk = grid.pseudo2D ? 1 : 2;
  const i32 cEmpty = bEmpty*blockSizeTot;
  for (i32 a = 0; a < 2; a++)
    for (i32 b = 0; b < 2; b++)
      for (i32 c = 0; c < nk; c++) {
        const real w = (a ? tx : (real)1-tx) * (b ? ty : (real)1-ty)
                     * (grid.pseudo2D ? (real)1 : (c ? tz : (real)1-tz));
        if (w <= 0) continue;
        const i32 m = grid.getNbrIdx(bIdx, ci+di+a, cj+dj+b,
                                     grid.pseudo2D ? ck : ck+dk+c);
        if (m >= cEmpty) continue;
        Vec3 cp((((real)(i0+a)) + (real)0.5)*dx, (((real)(j0+b)) + (real)0.5)*dy,
                (((real)(grid.pseudo2D ? gk : k0+c)) + (real)0.5)*dz);
        // Solid taps carry no solution -- UNLESS this is the implicit ghost
        // method, where first-layer ghosts hold a meaningful (iterated) state
        // and dropping them is exactly the order loss we are trying to remove.
        if (grid.getField(F_IBM)[m] <= (real)0.5) {
          if (!allowGhost) continue;
          const real phiT = grid.getField(F_PHI)[m];
          const real hT   = fmin(dx, dy);
          if (!(phiT <= (real)2.5*hT)) continue;    // beyond the filled band
        }
        if (grid.dbgChecks && tally) {
          // is THIS tap's own reconstruction wall-degraded?  (+-2 taps, matching
          // the d2R/l2R remap condition, not +-1)
          const real hm2 = fmin(dx,dy);
          bool taint = false;
          for (i32 mm = -2; mm <= 2 && !taint; mm++) {
            if (mm == 0) continue;
            if (!grid.isFluidCell(Vec3(cp[0]+(real)mm*dx, cp[1], cp[2]), hm2)) taint = true;
            if (!grid.isFluidCell(Vec3(cp[0], cp[1]+(real)mm*dy, cp[2]), hm2)) taint = true;
          }
          if (taint) atomicAdd(&g_ipTaint, 1ULL);
        }
        for (i32 f = 0; f < nf; f++) out[f] += w*F[f][m];
        wSum += w;
      }
  if (wSum <= (real)1e-12) return false;
  for (i32 f = 0; f < nf; f++) out[f] /= wSum;
  return true;
}

// Biquadratic (3x3, or 3x3x3 off the pseudo-2D path) Lagrange sample at an
// arbitrary point -- third-order in the plane where ibSample above is second.
// Nodes are the three cell centres bracketing the point in each direction, so
// the point is centred on the MIDDLE tap (round, not floor).
//
// Unlike ibSample this does NOT renormalise over a partial stencil.  The outer
// Lagrange weights are NEGATIVE (t(t-1)/2 and t(t+1)/2), so dropping a solid tap
// and rescaling the survivors can return a value far outside the range of the
// data it was built from -- the opposite of what a wall reconstruction wants
// next to the body.  If any tap is solid or out of reach this reports failure
// and the caller falls back to the bilinear sampler, which is monotone.
__device__ inline bool ibSampleQuad(CompressibleSolver &grid, Vec3 p, i32 lvl,
                                    i32 bIdx, i32 ci, i32 cj, i32 ck,
                                    i32 gi, i32 gj, i32 gk,
                                    real **F, i32 nf, real *out,
                                    bool allowGhost = false)
{
  const real dx = grid.getDx(lvl), dy = grid.getDy(lvl), dz = grid.getDz(lvl);
  const real fx = p[0]/dx - (real)0.5, fy = p[1]/dy - (real)0.5;
  const real fz = grid.pseudo2D ? (real)0 : p[2]/dz - (real)0.5;
  const i32 i0 = (i32)floor((double)fx + 0.5);      // NEAREST centre
  const i32 j0 = (i32)floor((double)fy + 0.5);
  const i32 k0 = grid.pseudo2D ? gk : (i32)floor((double)fz + 0.5);
  const real tx = fx - (real)i0, ty = fy - (real)j0;
  const real tz = grid.pseudo2D ? (real)0 : fz - (real)k0;
  real wx[3], wy[3], wz[3];
  #define IBQW(W, T) { W[0] = (real)0.5*(T)*((T)-(real)1);   \
                       W[1] = (real)1 - (T)*(T);             \
                       W[2] = (real)0.5*(T)*((T)+(real)1); }
  IBQW(wx, tx) IBQW(wy, ty)
  if (grid.pseudo2D) { wz[0] = 0; wz[1] = 1; wz[2] = 0; } else IBQW(wz, tz)
  #undef IBQW
  const i32 di = i0 - gi, dj = j0 - gj, dk = grid.pseudo2D ? 0 : k0 - gk;
  const i32 lo = -blockSize + 1, hi = 2*blockSize - 2;   // room for the +-1 ring
  if (ci+di-1 < lo || ci+di+1 > hi || cj+dj-1 < lo || cj+dj+1 > hi ||
      (!grid.pseudo2D && (ck+dk-1 < lo || ck+dk+1 > hi))) {
    if (allowGhost) atomicAdd(&g_qDecl, 1ull);
    return false;
  }
  for (i32 f = 0; f < nf; f++) out[f] = 0;
  const i32 cEmpty = bEmpty*blockSizeTot;
  const i32 c0 = grid.pseudo2D ? 1 : 0, c1 = grid.pseudo2D ? 2 : 3;
  for (i32 a = 0; a < 3; a++)
    for (i32 b = 0; b < 3; b++)
      for (i32 c = c0; c < c1; c++) {
        const i32 m = grid.getNbrIdx(bIdx, ci+di+a-1, cj+dj+b-1,
                                     grid.pseudo2D ? ck : ck+dk+c-1);
        if (m >= cEmpty) { if (allowGhost) atomicAdd(&g_qDecl, 1ull); return false; }
        // Under the implicit ghost method a solid tap is NOT a reason to
        // decline -- accepting ghost taps IS the method.  The ghosts carry an
        // iterated state, so the quadratic always uses its full 3x3 stencil and
        // never silently falls back to a lower-order fluid-only fit.  Declining
        // here would give back exactly the order this is meant to recover.
        if (!allowGhost && grid.getField(F_IBM)[m] <= (real)0.5) return false;
        if (allowGhost && grid.getField(F_IBM)[m] <= (real)0.5) {
          // beyond the filled band the ghost holds stale init data (rho can be
          // 0 there, so its primitives are NaN after the first conversion) --
          // tapping it poisoned the interface cells at the thin outer ring
          const real phiT = grid.getField(F_PHI)[m];
          if (!(phiT <= (real)2.5*fmin(dx, dy))) return false;
          atomicAdd(&g_qGhostTap, 1ull);
        }
        const real w = wx[a]*wy[b]*wz[c];
        for (i32 f = 0; f < nf; f++) out[f] += w*F[f][m];
      }
  if (allowGhost) atomicAdd(&g_qUsed, 1ull);
  return true;
}

// ---- constrained quadratic WLS wall trace (--ibwls 1) ----------------------
//
// Point sampling at one or two image points has two weaknesses that showed up
// under measurement: it reads only 1-2 locations (so a two-point normal
// derivative is noise-dominated, because the near-wall cells feeding it are
// themselves wall-degraded), and it imposes the wall condition by construction
// afterwards rather than as part of the fit.
//
// Here every fluid cell in a 5x5 window is fitted with a quadratic
//      q(xi,eta) = c0 + c1 xi + c2 eta + c3 xi^2 + c4 xi eta + c5 eta^2
// in coordinates scaled by h and CENTRED ON THE FACE, so the reconstructed face
// value is just c0.  The system is overdetermined (typically 15-25 rows for 6
// unknowns), so noise is averaged rather than differenced.
//
// Separate systems for the kinematics and the thermodynamics, because their
// boundary conditions are different -- which is the whole point:
//   u_n : Dirichlet,  u_n = 0 AT THE FOOT POINT        (non-penetration)
//   u_t : unconstrained                                 (slip: no condition)
//   s,H : Neumann,    n.grad(s) = n.grad(H) = 0         (isentropic, isoenergetic)
// Constraints enter as heavily weighted rows rather than a KKT solve: it keeps
// the solve a plain 6x6 SPD Cholesky, and the weight ratio (1e3) sets how nearly
// exact the condition is.
//
// 2-D / pseudo-2D only; a 3-D stencil needs the 10-term basis and two tangents,
// so the caller falls back to the point-sample trace there.
__device__ inline void ibWlsBasis(real xi, real eta, real b[6]) {
  b[0] = 1; b[1] = xi; b[2] = eta; b[3] = xi*xi; b[4] = xi*eta; b[5] = eta*eta;
}
// solve a 6x6 SPD system in place by Cholesky; false if not positive definite
__device__ inline bool ibWlsSolve(real A[6][6], real *rhs, i32 nrhs, real out[][6]) {
  // Jacobi (diagonal) preconditioning before the factorisation.  Normal
  // equations built from a polynomial basis are badly SCALED -- the 1 and the
  // xi^2 columns differ by orders of magnitude, and near the wall the stencil is
  // one-sided, so the raw matrix is ill-conditioned enough to return garbage or
  // NaN from a Cholesky that never trips its own positivity test.  Scaling every
  // row/column by 1/sqrt(diag) puts unit values on the diagonal and costs 6
  // rsqrts.
  real dsc[6], As[6][6];
  for (i32 i = 0; i < 6; i++)
    dsc[i] = (A[i][i] > (real)1e-30) ? (real)1/sqrt(A[i][i]) : (real)0;
  for (i32 i = 0; i < 6; i++) {
    if (dsc[i] == (real)0) return false;             // empty basis direction
    for (i32 j = 0; j < 6; j++) As[i][j] = A[i][j]*dsc[i]*dsc[j];
  }
  // Ridge on the SCALED matrix -- a true relative regularisation.  NOT applied
  // to the constant term: that coefficient IS the reconstructed face value, so
  // ridging it shrinks the answer directly (measured 0.4% low on a gate whose
  // exact answer is a constant).  The higher-order terms are what need taming.
  for (i32 i = 1; i < 6; i++) As[i][i] += (real)1e-4;
  real L[6][6];
  for (i32 i = 0; i < 6; i++)
    for (i32 j = 0; j < 6; j++) L[i][j] = 0;
  for (i32 i = 0; i < 6; i++) {
    for (i32 j = 0; j <= i; j++) {
      real sum = As[i][j];
      for (i32 k = 0; k < j; k++) sum -= L[i][k]*L[j][k];
      if (i == j) {
        if (!(sum > (real)1e-24)) return false;
        L[i][i] = sqrt(sum);
      } else L[i][j] = sum/L[j][j];
    }
  }
  for (i32 r = 0; r < nrhs; r++) {
    real y[6];
    for (i32 i = 0; i < 6; i++) {
      real sum = rhs[r*6 + i]*dsc[i];                // scaled RHS
      for (i32 k = 0; k < i; k++) sum -= L[i][k]*y[k];
      y[i] = sum/L[i][i];
    }
    for (i32 i = 5; i >= 0; i--) {
      real sum = y[i];
      for (i32 k = i+1; k < 6; k++) sum -= L[k][i]*out[r][k];
      out[r][i] = sum/L[i][i];
    }
    for (i32 i = 0; i < 6; i++) {                    // undo the scaling
      out[r][i] *= dsc[i];
      if (!isfinite(out[r][i])) return false;
    }
  }
  return true;
}

__device__ inline bool ibWlsTrace(CompressibleSolver &grid,
    real *Rho, real *U, real *V, real *W, real *P,
    i32 lvl, Vec3 fcPos, i32 d, real h, i32 fluidIdx, bool fluidOnPlus,
    real F[5], i32 bIdx, i32 ci, i32 cj, i32 ck, i32 gi, i32 gj, i32 gk)
{
  if (!grid.pseudo2D) return false;                 // 2-D basis only
  Vec3 n = grid.wallNormal(fcPos, h);
  const real dFc = fmin(fmax(-grid.getBoundaryLevelSet(fcPos), (real)0.05*h),
                        (real)1.5*h);
  const real tvx = -n[1], tvy = n[0];               // unique tangent in 2-D
  const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);

  // ---- assemble the weighted normal equations from the fluid cells ----------
  real M[6][6];
  for (i32 i = 0; i < 6; i++) for (i32 j = 0; j < 6; j++) M[i][j] = 0;
  real rUn[6] = {0,0,0,0,0,0}, rUt[6] = {0,0,0,0,0,0};
  real rS[6]  = {0,0,0,0,0,0}, rH[6]  = {0,0,0,0,0,0};
  const i32 cEmpty = bEmpty*blockSizeTot;
  const real Rw = (real)2.6;                        // support radius, in cells
  i32 nUse = 0;
  real wMax = 0;
  for (i32 a = -2; a <= 2; a++)
    for (i32 b2 = -2; b2 <= 2; b2++) {
      const i32 m = grid.getNbrIdx(bIdx, ci+a, cj+b2, ck);
      if (m >= cEmpty) continue;
      if (grid.getField(F_IBM)[m] <= (real)0.5) continue;      // solid: no data
      // cell centre relative to the FACE, in units of h
      const real cx = (real)(gi + a) + (real)0.5, cy = (real)(gj + b2) + (real)0.5;
      const real xi  = (cx*dx - fcPos[0])/h, eta = (cy*dy - fcPos[1])/h;
      const real rr  = sqrt(xi*xi + eta*eta);
      if (rr > Rw) continue;
      const real u_  = (real)1 - rr/Rw;                        // Wendland C2
      const real w   = u_*u_*u_*u_*((real)4*rr/Rw + (real)1);
      if (w <= 0) continue;
      const real rho = fmax(Rho[m], (real)1e-30), pp = fmax(P[m], (real)1e-30);
      const real vx = U[m], vy = V[m];
      const real un = vx*n[0] + vy*n[1], ut = vx*tvx + vy*tvy;
      const real ss = pp/pow(rho, gam);
      const real HH = gam*pp/((gam-(real)1)*rho) + (real)0.5*(vx*vx + vy*vy);
      real bb[6]; ibWlsBasis(xi, eta, bb);
      for (i32 i = 0; i < 6; i++) {
        for (i32 j = 0; j <= i; j++) M[i][j] += w*bb[i]*bb[j];
        rUn[i] += w*bb[i]*un; rUt[i] += w*bb[i]*ut;
        rS[i]  += w*bb[i]*ss; rH[i]  += w*bb[i]*HH;
      }
      wMax = fmax(wMax, w); nUse++;
    }
  if (nUse < 12) return false;         // 6 unknowns: demand real redundancy
  for (i32 i = 0; i < 6; i++) for (i32 j = i+1; j < 6; j++) M[i][j] = M[j][i];
  // Tikhonov ridge on the quadratic terms only -- keeps a flat stencil solvable
  // without biasing the constant, which IS the answer we read out.

  // ---- constraint rows at the foot point -----------------------------------
  const real xf = -dFc*n[0]/h, yf = -dFc*n[1]/h;
  real bf[6]; ibWlsBasis(xf, yf, bf);
  // n.grad in the scaled coordinates (the 1/h factor divides out of "= 0")
  const real gb[6] = {0, n[0], n[1], (real)2*xf*n[0], yf*n[0] + xf*n[1],
                      (real)2*yf*n[1]};
  // ibWls 2 = constraints OFF (diagnostic), 3 = constraints on, thermo closure off
  const real wc = (grid.ibWls == 2) ? (real)0 : (real)1e3*fmax(wMax, (real)1e-30);

  // u_t is NOT unconstrained at a slip wall.  FRIB Eq. 19 gives
  //     n.grad(u_t) + kappa u_t = 0
  // at the surface, and without it u_t is the one field with no condition at
  // all -- so where half the stencil is solid the face sits at the EDGE of the
  // data and the quadratic extrapolates, overshooting (measured max|u| 2.08 on
  // a cylinder whose exact peak is 2.0).  kappa = div(n) from the level set,
  // capped at 1/(2h) since a feature tighter than two cells is not resolved.
  real kap = 0;
  {
    const real e2 = (real)0.5*h;
    Vec3 nxp = grid.wallNormal(Vec3(fcPos[0]+e2, fcPos[1], fcPos[2]), h);
    Vec3 nxm = grid.wallNormal(Vec3(fcPos[0]-e2, fcPos[1], fcPos[2]), h);
    Vec3 nyp = grid.wallNormal(Vec3(fcPos[0], fcPos[1]+e2, fcPos[2]), h);
    Vec3 nym = grid.wallNormal(Vec3(fcPos[0], fcPos[1]-e2, fcPos[2]), h);
    kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
    const real kMax = (real)0.5/h;
    kap = fmin(fmax(kap, -kMax), kMax);
  }
  // the constraint row lives in SCALED coordinates, where n.grad picks up 1/h
  real gt[6];
  for (i32 i = 0; i < 6; i++) gt[i] = gb[i] + kap*h*bf[i];

  real Mn[6][6], Mg[6][6], Mt[6][6];
  for (i32 i = 0; i < 6; i++)
    for (i32 j = 0; j < 6; j++) {
      Mn[i][j] = M[i][j] + wc*bf[i]*bf[j];           // Dirichlet u_n(foot) = 0
      Mg[i][j] = M[i][j] + ((grid.ibWls == 4) ? (real)0 : wc*gb[i]*gb[j]);
      Mt[i][j] = M[i][j] + wc*gt[i]*gt[j];           // FRIB Eq.19 on u_t
    }
  // constraint right-hand sides are all zero, so the data RHS are unchanged
  real cUt[1][6], cUn[1][6], cSH[2][6];
  real rhsSH[12];
  for (i32 i = 0; i < 6; i++) { rhsSH[i] = rS[i]; rhsSH[6+i] = rH[i]; }
  // NOTE: constraining u_t with FRIB Eq.19 (matrix Mt, built above) was tried and
  // is MUCH worse -- cylinder max|u| 2.73 vs 2.08 unconstrained, L2(Cp) 0.75 vs
  // 0.13.  The condition itself is right (it reproduces potential flow's
  // du_t/dr = -2 u0 sin(theta)/R at r=R analytically); at weight 1e3 it simply
  // over-constrains, forcing the quadratic to bend to meet a derivative the
  // near-wall data cannot support.  It would need a much softer weight, or a
  // proper KKT solve with the weight chosen from the data residual.
  if (!ibWlsSolve(M, rUt, 1, cUt)) return false;
  if (!ibWlsSolve(Mn, rUn, 1, cUn)) return false;
  if (!ibWlsSolve(Mg, rhsSH, 2, cSH)) return false;

  // ---- face values are the constant terms -----------------------------------
  const real utF = cUt[0][0], unF = cUn[0][0];
  const real sF  = cSH[0][0], HF  = cSH[1][0];
  const real vx = utF*tvx + unF*n[0], vy = utF*tvy + unF*n[1];
  const real hF = HF - (real)0.5*(vx*vx + vy*vy);
  if (!(hF > (real)0) || !(sF > (real)0)) return false;
  real rF = pow(hF*(gam-(real)1)/(gam*sF), (real)1/(gam-(real)1));
  real pF = sF*pow(rF, gam);
  if (grid.ibWls == 3) {   // diagnostic: take rho,p straight from the fluid cell
    rF = Rho[fluidIdx]; pF = P[fluidIdx];
  }
  if (!(rF > (real)0) || !(pF > (real)0) || !isfinite(rF) || !isfinite(pF))
    return false;

  Vec5 qW(rF, vx, vy, (real)0, pF);
  Vec5 qF(Rho[fluidIdx], U[fluidIdx], V[fluidIdx], W[fluidIdx], P[fluidIdx]);
  Vec3 e((real)(d==0), (real)(d==1), (real)(d==2));
  Vec5 fl = fluidOnPlus ? grid.hllcFlux(grid.prim2cons(qW), grid.prim2cons(qF), e)
                        : grid.hllcFlux(grid.prim2cons(qF), grid.prim2cons(qW), e);
  for (i32 m = 0; m < 5; m++) F[m] = fl[m];
  return true;
}

// ---- FRIB-style wall face flux (--ibrecon 2) -------------------------------
//
// The paper's construction at a fluid/wall face (FRIB.pdf Sec. 2.3): build the
// WALL-SIDE trace q_FP from the boundary conditions -- image point at s* = 2h
// along the exact normal through the face centre, u_n interpolated linearly to
// zero AT THE WALL (so at the face's own standoff d_FC it is u_n,IP d_FC/s*),
// u_t and rho, p Neumann -- and feed it to the ORDINARY Riemann solver against
// the fluid-side state.  Compared to --ibrecon 1 (pure ghost reconstruction)
// the ghost-side MUSCL trace, which reads TWO cells into the body, is replaced
// by this per-face trace: stencils now reach only ONE ghost deep, and at a
// sub-cell-thin trailing edge the two opposite faces of the SAME cell row each
// get their own trace with their own normal -- the shared-ghost-row conflict
// never arises in the flux.  Upwinding stays with the Riemann solver (the
// earlier PRESCRIBED-flux attempts lacked it and failed).
__device__ inline bool ibFaceTraceFlux(CompressibleSolver &grid,
    real *Rho, real *U, real *V, real *W, real *P,
    i32 lvl, Vec3 fcPos, i32 d, real h, i32 fluidIdx, bool fluidOnPlus,
    real F[5], i32 bIdx, i32 ci, i32 cj, i32 ck, i32 gi, i32 gj, i32 gk)
{
  Vec3 n = grid.wallNormal(fcPos, h);
  const real dFc = fmin(fmax(-grid.getBoundaryLevelSet(fcPos), (real)0.05*h),
                        (real)1.5*h);
  const real sStar = (real)2*h;
  Vec3 foot(fcPos[0] - dFc*n[0], fcPos[1] - dFc*n[1], fcPos[2] - dFc*n[2]);
  Vec3 ip(foot[0] + sStar*n[0], foot[1] + sStar*n[1], foot[2] + sStar*n[2]);
  real *Fs[5] = {Rho, U, V, W, P};
  real q[5];
  real qS[5];              // the nearest image-point sample, whichever path set it
  Vec5 qW(0,0,0,0,0);
  bool haveQ = false;

  // ---- paper slip BC (--ibgslip 1): NO image point ------------------------
  // Ali et al., J Eng Math 146:5 (2024) Sect. 2.5.  Velocity is not built from
  // a mirror: the primitives are interpolated at the WALL point itself and the
  // normal component is simply dropped, leaving the tangential part (their
  // U sin(t)cos(d) - V sin(t)sin(d) written frame-free).  rho and p are Neumann
  // -- the interpolated wall values ARE the face values, their Eq. (26)
  // coefficient (1 - B21) rather than the Dirichlet (2 - B21).
  //
  // The point is structural, not cosmetic: the mirror path below carries
  // u_n * (d_FC/s*), which diverges as s* -> 0.  That division is what made the
  // ghost mirror degenerate -- 11% velocity overshoot at N=512 with a 0.25h
  // floor, needing 0.5h to behave.  Sampling AT the wall has no s* in it.
  // The stencil straddles the boundary and only closes because the ghost values
  // are unknowns of the implicit system: this BC is the reason the method has
  // to be implicit in the first place.
  if (grid.ibGSlip == 1) {
    real qw[5];
    if (ibSampleQuad(grid, foot, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qw, true)
        && qw[0] > (real)0 && qw[4] > (real)0) {
      const real un = qw[1]*n[0] + qw[2]*n[1] + qw[3]*n[2];
      qW = Vec5(qw[0], qw[1] - un*n[0], qw[2] - un*n[1], qw[3] - un*n[2], qw[4]);
      for (i32 m = 0; m < 5; m++) qS[m] = qw[m];
      haveQ = true;
    }
  }

  // ---- two-image-point quadratic trace (--ipquad 1) -----------------------
  // The single-point trace below is zeroth order in rho and p (plain Neumann:
  // the image-point values ARE the face values) and first order in u_n (a line
  // through zero at the wall).  On a curved wall dp/dn is not zero, so the
  // Neumann part is precisely the term that makes neighbouring faces at
  // different d_FC disagree.  With a SECOND node on the same normal:
  //   rho, p, u_t : linear in s through both samples, evaluated at d_FC
  //   u_n         : quadratic, B s + C s^2, since u_n(0) = 0 at the wall
  //                 supplies the third constraint for free
  // Both nodes are sampled biquadratically, so the in-plane interpolation is
  // third order rather than second.
  // modes: 1 = two points + biquadratic, 2 = two points + bilinear (isolates the
  // extra node), 3 = one point + biquadratic (isolates the in-plane order)
  // mode 4 = the only WELL-POSED use of the second node: u_n gets a quadratic
  // (u_n(0)=0 at the wall makes it interpolatory), while rho, p and u_t keep the
  // single-point Neumann value.  Modes 1/2 extrapolate rho,p,u_t back from
  // s = 2h,3h to d_FC, which amplifies noise in samples that are themselves
  // built on wall-degraded stencils -- measured catastrophic with a bilinear
  // sample (max|u| 60 on the cylinder) and merely bad with a biquadratic one.
  const bool qSamp = (grid.ibIpQuad == 1 || grid.ibIpQuad == 3 || grid.ibIpQuad == 4);
  const bool qTwo  = (grid.ibIpQuad == 1 || grid.ibIpQuad == 2 || grid.ibIpQuad == 4);
  const bool qNonly= (grid.ibIpQuad == 4);
  if (!haveQ && qTwo) {
    const real s1 = sStar, s2 = sStar + h;
    real qa[5], qb[5];
    Vec3 ip2(foot[0]+s2*n[0], foot[1]+s2*n[1], foot[2]+s2*n[2]);
    const bool gotA = qSamp
      ? ibSampleQuad(grid, ip,  lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qa)
      : ibSample    (grid, ip,  lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qa);
    const bool gotB = qSamp
      ? ibSampleQuad(grid, ip2, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qb)
      : ibSample    (grid, ip2, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qb);
    if (gotA && gotB) {
      const real ds = s1 - s2;                       // = -h
      const real ea = (dFc - s1)/ds;                 // linear extrapolation weight
      // rho and p: extrapolate, but never further than the two samples'own
      // spread -- a two-point slope built from near-wall cells is noisy, and an
      // unbounded extrapolation toward the wall is what produces impossible C_p.
      real ex[5];
      bool ok = true;
      for (i32 m = 0; m < 5; m++) {
        const real dq = qNonly ? (real)0 : (qa[m] - qb[m]);
        ex[m] = qa[m] + ea*dq;
        if (m == 0 || m == 4) {
          const real lim = (real)2*fabs(dq) + (real)0.02*fabs(qa[m]);
          ex[m] = fmin(fmax(ex[m], qa[m]-lim), qa[m]+lim);
          if (!(ex[m] > (real)0)) ok = false;
        }
      }
      if (ok) {
        // split the extrapolated velocity and re-impose the wall condition on
        // u_n with the quadratic through u_n(0) = 0
        const real na = qa[1]*n[0] + qa[2]*n[1] + qa[3]*n[2];
        const real nb = qb[1]*n[0] + qb[2]*n[1] + qb[3]*n[2];
        const real det = s1*s1*s2 - s2*s2*s1;
        if (fabs(det) > (real)1e-30) {
          const real B = ( na*s2*s2 - nb*s1*s1)/det;
          const real C = (-na*s2     + nb*s1    )/det;
          const real unF = B*dFc + C*dFc*dFc;
          const real ne  = ex[1]*n[0] + ex[2]*n[1] + ex[3]*n[2];
          const real sc2 = unF - ne;                 // replace the normal part
          qW = Vec5(ex[0], ex[1] + sc2*n[0], ex[2] + sc2*n[1], ex[3] + sc2*n[2], ex[4]);
          for (i32 m = 0; m < 5; m++) qS[m] = qa[m];
          haveQ = true;
        }
      }
    }
  }

  if (!haveQ) {
    bool got = (qSamp && !qTwo)
      && ibSampleQuad(grid, ip, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, q);
    if (!got && !ibSample(grid, ip, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, q))
      return false;
    const real un = q[1]*n[0] + q[2]*n[1] + q[3]*n[2];
    const real sc = un*(dFc/sStar - (real)1);        // u - un*n + un*(dFc/s*)*n
    qW = Vec5(q[0], q[1] + sc*n[0], q[2] + sc*n[1], q[3] + sc*n[2], q[4]);
    for (i32 m = 0; m < 5; m++) qS[m] = q[m];
  }

  // ---- mode 2: paper velocity, existing p/rho ------------------------------
  // The paper's velocity rule and its Neumann p,rho are separable, and on a
  // curved wall they grade very differently.  Measured on the annulus at N=256,
  // t=40: the wall-point velocity gives a SMOOTH field (no staircase serration
  // at all) and max|u| = 0.2006 against an exact 0.2, where the mirror is 3.2%
  // low -- but Neumann p,rho throws away dp/dn = rho u_t^2 kappa and costs 5x in
  // L2.  Neither --ibthermo nor --ibcurv can repair it: with the sample taken AT
  // the wall there is no second node on the normal, so the closure is degenerate
  // (measured an exact no-op) and the curvature ramp overshoots 31%.
  // So keep the velocity and let p, rho come from the trace above, which already
  // carries the curvature.
  if (grid.ibGSlip == 2) {
    real qw[5];
    if (ibSampleQuad(grid, foot, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, qw, true)) {
      const real un = qw[1]*n[0] + qw[2]*n[1] + qw[3]*n[2];
      qW[1] = qw[1] - un*n[0];
      qW[2] = qw[2] - un*n[1];
      qW[3] = qw[3] - un*n[2];
    }
  }

  // ---- curvature ramp on u_t (--ibcurv 1) ----------------------------------
  // The missing half of the H/S closure below: with s and H fixed, the face
  // pressure can only differ from the sample's through |u| -- and the trace
  // carries u_t across UNCHANGED, so without this ramp the closure is nearly
  // inert (measured: ibthermo alone moved the vortex error ~nothing).  For the
  // irrotational flow a slip wall supports, du_t/ds = -kappa u_t (zero
  // vorticity at a curved wall), so ramp the tangential part from the sample
  // height s* to the face's own d_FC; the H/S recovery then integrates the
  // centripetal dp/dn = rho u_t^2 kappa automatically and consistently.
  if (grid.ibCurv) {
    const real e2 = (real)0.5*h;
    Vec3 nxp = grid.wallNormal(Vec3(foot[0]+e2, foot[1], foot[2]), h);
    Vec3 nxm = grid.wallNormal(Vec3(foot[0]-e2, foot[1], foot[2]), h);
    Vec3 nyp = grid.wallNormal(Vec3(foot[0], foot[1]+e2, foot[2]), h);
    Vec3 nym = grid.wallNormal(Vec3(foot[0], foot[1]-e2, foot[2]), h);
    real kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
    const real kM = (real)0.5/h;
    kap = fmin(fmax(kap, -kM), kM);
    const real den = (real)1 - kap*sStar;
    if (fabs(den) > (real)0.2) {
      const real ramp = ((real)1 - kap*dFc)/den;
      const real unW = qW[1]*n[0] + qW[2]*n[1] + qW[3]*n[2];
      qW[1] = (qW[1] - unW*n[0])*ramp + unW*n[0];
      qW[2] = (qW[2] - unW*n[1])*ramp + unW*n[1];
      qW[3] = (qW[3] - unW*n[2])*ramp + unW*n[2];
    }
  }
  // ---- entropy / total-enthalpy closure (--ibthermo 1) ---------------------
  // Along the normal at an adiabatic SLIP wall the flow is isentropic and
  // isoenergetic, so the quantities that are CONSTANT there are
  //     s = p/rho^gamma        H = gamma p/((gamma-1) rho) + |u|^2/2
  // and NOT p and rho: those carry the centripetal gradient dp/dn = rho u_t^2
  // kappa that turns the flow around the body.  Holding p and rho Neumann
  // therefore commits exactly that gradient as an error, and estimating it by
  // finite differences between image points is what proved ill-conditioned
  // (near-wall cells are wall-degraded, so the two-point slope is mostly noise).
  //
  // Reconstructing (s, H) instead makes the pressure variation fall out of the
  // VELOCITY rather than out of a difference: fix s and H from the sample, then
  // h = H - |u|^2/2 at the face and (rho, p) follow algebraically -- and stay
  // thermodynamically consistent by construction.  H is exactly constant for a
  // homenthalpic freestream (shocks included), so its "extrapolation" is exact;
  // s is constant along a streamline, so it only errs where a shock crosses the
  // normal.
  if (grid.ibThermoRec) {
    const real rI = fmax(qS[0], (real)1e-30);
    const real pI = fmax(qS[4], (real)1e-30);
    const real sI = pI/pow(rI, gam);
    const real HI = gam*pI/((gam-(real)1)*rI)
                  + (real)0.5*(qS[1]*qS[1] + qS[2]*qS[2] + qS[3]*qS[3]);
    const real vv = qW[1]*qW[1] + qW[2]*qW[2] + qW[3]*qW[3];
    const real hF = HI - (real)0.5*vv;
    if (hF > (real)0 && sI > (real)0) {
      const real rF = pow(hF*(gam-(real)1)/(gam*sI), (real)1/(gam-(real)1));
      const real pF = sI*pow(rF, gam);
      if (rF > (real)0 && pF > (real)0 && isfinite(rF) && isfinite(pF)) {
        qW[0] = rF; qW[4] = pF;
      }
    }
  }
  Vec5 qF(Rho[fluidIdx], U[fluidIdx], V[fluidIdx], W[fluidIdx], P[fluidIdx]);
  Vec3 e((real)(d==0), (real)(d==1), (real)(d==2));
  // low face of the cell: LEFT = the minus-side state.  fluidOnPlus says the
  // fluid cell is on the plus side (the wall is below/left of it).
  Vec5 fl = fluidOnPlus ? grid.hllcFlux(grid.prim2cons(qW), grid.prim2cons(qF), e)
                        : grid.hllcFlux(grid.prim2cons(qF), grid.prim2cons(qW), e);
  for (i32 m = 0; m < 5; m++) F[m] = fl[m];
  return true;
}

// ---- immersed-boundary wall flux ------------------------------------------
//
// The wall condition at the face centre FC between a fluid and a non-fluid cell,
// for a general wall normal.  Same five steps as the grid-aligned model, but the
// stress tensor tau_ij = rho_w u_tau^2 (t_i n_j + t_j n_i) is now projected onto
// the GRID face normal e_d.  The projection carries its own sign: n_d is +1 when
// the body lies below the face and -1 when it lies above, so one expression
// serves both orientations and no case split is needed.
//
__device__ inline bool ibWallFlux(CompressibleSolver &grid,
                                  real *Rho, real *U, real *V, real *W, real *P,
                                  real *K, real *Tau, real *TF1,
                                  i32 lvl, Vec3 fcPos, i32 d, real h,
                                  i32 fluidIdx, real Fw[5], real &FwK, real &FwT,
                                  real &uTauOut,
                                  i32 bIdx, i32 ci, i32 cj, i32 ck,
                                  i32 gi, i32 gj, i32 gk,
                                  bool cons = false)
{
  // Distance from the face centre to the surface.  FLOORED at a fraction of the
  // cell: an immersed surface can pass arbitrarily close to a face, and both
  // tau~_FC (Eq. 39) and phi (Eq. A.5) are proportional to d_FC, so an
  // unbounded-below d_FC makes the Appendix-A flux ill-conditioned -- the exact
  // form stays finite only because phi_LR and C both vanish with d_FC, and the
  // (A.7) fallback is worse still since it carries phi_R/phi_LR.  This floor is
  // the immersed analogue of wallOffset on a grid-aligned wall (a fixed half
  // cell there); the paper's Fig. 7 shows C_f is insensitive to that offset.
  const real dFc = fmax(-grid.getBoundaryLevelSet(fcPos), (real)0.1*h);   // FC -> wall
  // d_IP is measured from the WALL, so the standoff above the FACE is dIp - dFc.
  // Enforce a floor on that standoff (see CompressibleSolver::ipStandMin): with
  // d_IP pinned at 3h the standoff collapses as d_FC grows, dragging the stencil
  // down onto the wall-adjacent cell and destabilising the coupling.
  const real dIp = fmax(grid.dIpFac*h, dFc + grid.ipStandMin*h);
  if (dIp <= dFc) {
    if (grid.dbgChecks) { atomicAdd(&g_ibFailDip, 1ULL);
      atomicMaxFloat(&g_ibMaxDfc, (double)(dFc/h));
      // does the CACHED mask agree with the live level set at this face?
      atomicMaxFloat(&g_ibMaxLvl, (double)lvl); }
    return false; }
  Vec3 n = grid.wallNormal(fcPos, h);
  if (grid.dbgChecks && n[d] < (real)0) atomicAdd(&g_ibNup, 1ULL);

  // Upstream of the leading edge the surface is a SLIP wall, not a modelled one
  // -- the same split the grid-aligned case makes.  Without it the wall runs
  // into the inflow, where the Dirichlet boundary forces freestream at exactly
  // the point the wall model demands no-slip; that corner is singular and seeds
  // a disturbance that grows without bound.  The level set supplies the
  // GEOMETRY; whether a face is modelled or slip is a separate decision.
  // EULER / inviscid: with no turbulence model there is no wall model either --
  // every immersed face is a slip wall.  Same branch the flat plate uses
  // upstream of its leading edge.
  //
  // THE IB FACE IS NOT THE WALL.  It is a grid face sitting d_FC AWAY from the
  // surface, so flow running PARALLEL to an inclined wall legitimately CROSSES
  // it: mass, momentum and energy fluxes through it are real and must be
  // carried.  Zeroing them (pressure only) turns every face into a solid
  // barrier, which is what makes the staircase a blockage -- it was generating
  // a numerical boundary layer, an inviscid wake and per-step suction spikes
  // (gate: uniform flow parallel to a 30-degree slip plane gave max|rhoV| =
  // 5.8e-1 instead of 0).  Slip means removing only the component normal to the
  // TRUE wall; whatever remains is convected through the grid face, exactly as
  // the wall-modelled branch below does with its linearised u_FC.
  // ---- ghost-cell wall-function architecture (--ibwm 1) --------------------
  // Yang, Song & Zhu, Processes 12 (2024) 1182, Sec. 2.3.  The wall model is
  // imposed ONLY through the ghost cells' tangential velocity (their Eqs. 6-8);
  // the wall face then takes the ORDINARY MUSCL + HLLC + viscous flux, whose
  // near-wall gradient reads those ghosts and so carries the wall stress.  No
  // boundary flux is prescribed anywhere -- which is why their solver is stable
  // with EXPLICIT (Adams-Bashforth) time integration.  Declining here hands the
  // face back to the ordinary path; it REQUIRES filled ghosts (--ibgf 0), which
  // Main enforces when this mode is on.
  // Modes 1-2 (ghost wall function): no prescribed flux -- the ordinary path
  // reads the ghosts.  Mode 3: log-law ghosts feed the ORDINARY near-wall
  // stencils (closure + viscous cross-terms) but the wall FACE keeps this
  // prescribed model flux -- the face is the drag sink (= rho u_tau^2 exact);
  // measured: canceling the face sink under mode 1 evaporates the whole layer,
  // and keeping the ordinary ghost-gradient face over-drags ~1.7x.
  if (grid.rans && grid.ibWallMode >= 1 && grid.ibWallMode <= 2 &&
      !(fcPos[0] < ((grid.wmX0 >= (real)0) ? grid.wmX0 : grid.plateX0))) return false;
  if (!(grid.rans || grid.ibWmles)
      || fcPos[0] < ((grid.wmX0 >= (real)0) ? grid.wmX0 : grid.plateX0)
      || fcPos[0] > grid.wmX1) {
    // ---- FRIB image-line reconstruction (docs/FRIB.pdf, Funada & Imamura,
    //      Comput. Fluids 2023, Eqs. 18/19) ------------------------------------
    //
    // The face does NOT sit on the wall: its standoff d_FC varies from ~0.1h to
    // ~1.05h between NEIGHBOURING faces (measured on the RAE section).  Applying
    // the adjacent cell's state at the face regardless of that standoff commits
    // an error dp/dn * d_FC that jumps cell to cell -- a sawtooth C_p along the
    // whole wall, which is exactly what was measured (mean |dCp| 0.099 between
    // adjacent near-wall cells, sign reversing 27 times in 46).
    //
    // FRIB's answer: lay a line along the TRUE normal with the wall at s = 0,
    // solve the WALL state there, then evaluate the reconstruction AT THE FACE'S
    // OWN s = d_FC.  On a curved wall dp/dn is NOT zero -- the flow turning
    // around the surface needs a centripetal gradient dp/ds = rho u_t^2 kappa,
    // which lowers the wall pressure below the outer pressure.  That term is the
    // one a flat-wall extrapolation misses, and it is what makes neighbouring
    // faces at different standoffs agree.
    // ---- FRIB HIGH-ORDER wall condition (--ibho 1): the paper's HO-i/c at
    // k = 2, which is the order the FV interior can support ------------------
    // Their LO (one IP, primitives carried across) is "a first-order method"
    // by their own classification, and the measured vortex order 1.25 of the
    // legacy branch below matches.  HO instead puts Gauss IMAGE POINTS on an
    // image line of length d_IL = 3h along the true normal, SOLVES the
    // wall-adjacent IP from the wall conditions in the H/S form -- u_n = 0,
    // dH/dn = 0, dS/dn = 0 at the wall, plus the curvature relation
    // du_t/ds = -kappa u_t (their Eq. 18d; H/S chosen because curvature then
    // appears in ONE equation) -- and evaluates the face from the constrained
    // polynomial.  The wall-adjacent IP is never sampled, so the near-wall
    // stencil problem that forced the legacy branch's single-cell sample and
    // hard-zeroed u_n never arises: the one sampled IP sits at 2.37h with a
    // fully-fluid bilinear stencil.
    if (grid.ibHo) {
      const real dIL = (real)3*h;
      const real xi1 = (real)-0.5773502691896258, xi2 = (real)0.5773502691896258;
      const real s2  = (real)0.5*((real)1 + xi2)*dIL;      // 2.366h from the wall
      Vec3 ip2(fcPos[0] + (s2 - dFc)*n[0], fcPos[1] + (s2 - dFc)*n[1],
               fcPos[2] + (s2 - dFc)*n[2]);
      real *Fs2[5] = {Rho, U, V, W, P};
      real q2[5];
      if (ibSample(grid, ip2, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs2, 5, q2)) {
        const real r2 = fmax(q2[0], (real)1e-30);
        const real s2c = cons ? (real)1/r2 : (real)1;
        const real u2x = q2[1]*s2c, u2y = q2[2]*s2c, u2z = q2[3]*s2c;
        const real p2 = cons ? (gam-(real)1)*(q2[4] - (real)0.5*r2*(u2x*u2x + u2y*u2y + u2z*u2z))
                             : q2[4];
        const real un2 = u2x*n[0] + u2y*n[1] + u2z*n[2];
        real t2x = u2x - un2*n[0], t2y = u2y - un2*n[1], t2z = u2z - un2*n[2];
        const real ut2 = sqrt(t2x*t2x + t2y*t2y + t2z*t2z);
        const real it2 = (ut2 > (real)1e-30) ? (real)1/ut2 : (real)0;
        t2x *= it2; t2y *= it2; t2z *= it2;
        const real H2 = gam*fmax(p2,(real)1e-30)/((gam-(real)1)*r2)
                      + (real)0.5*(un2*un2 + ut2*ut2);
        const real S2 = fmax(p2,(real)1e-30)/pow(r2, gam);
        // curvature (same FD-of-normal estimate as the legacy branch)
        real kap = 0;
        if (grid.ibCurv) {
          const real e2 = (real)0.5*h;
          Vec3 nxp = grid.wallNormal(Vec3(fcPos[0]+e2, fcPos[1], fcPos[2]), h);
          Vec3 nxm = grid.wallNormal(Vec3(fcPos[0]-e2, fcPos[1], fcPos[2]), h);
          Vec3 nyp = grid.wallNormal(Vec3(fcPos[0], fcPos[1]+e2, fcPos[2]), h);
          Vec3 nym = grid.wallNormal(Vec3(fcPos[0], fcPos[1]-e2, fcPos[2]), h);
          kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
          const real kM = (real)0.5/h;
          kap = fmin(fmax(kap, -kM), kM);
        }
        // linear (k=2) nodal basis on xi with nodes xi1, xi2:
        //   phi1(xi) = (xi2 - xi)/(xi2 - xi1),  phi2(xi) = (xi - xi1)/(xi2 - xi1)
        const real dxi  = xi2 - xi1;
        const real p1w  = (xi2 - (real)-1)/dxi;   // phi1(-1) = 1.366
        const real p2w  = ((real)-1 - xi1)/dxi;   // phi2(-1) = -0.366
        const real d1x  = (real)-1/dxi, d2x = (real)1/dxi;   // phi'
        // wall conditions at xi = -1 solve the first-IP values:
        const real un1 = -p2w/p1w*un2;                     // u_n(-1) = 0
        const real H1  = H2;                               // dH/dxi(-1) = 0 (phi1' = -phi2')
        const real S1  = S2;
        // du_t/ds = -kappa u_t at the wall:  (2/dIL)(phi1' ut1 + phi2' ut2)
        //   = -kappa (phi1(-1) ut1 + phi2(-1) ut2)
        const real aa  = (real)2/dIL*d1x + kap*p1w;
        const real bb  = (real)2/dIL*d2x + kap*p2w;
        const real ut1 = (fabs(aa) > (real)1e-12) ? -bb/aa*ut2 : ut2;
        // evaluate the constrained polynomial at the FACE's own xi
        const real xiF = (real)2*fmin(dFc, dIL)/dIL - (real)1;
        const real f1  = (xi2 - xiF)/dxi, f2 = (xiF - xi1)/dxi;
        const real unF = f1*un1 + f2*un2;
        const real utF = fmax(f1*ut1 + f2*ut2, (real)0);
        const real HF  = f1*H1  + f2*H2;
        const real SF  = fmax(f1*S1 + f2*S2, (real)1e-30);
        // primitives from (H, S, u): a^2 = (gam-1)(H - |u|^2/2); rho from S
        const real a2F = fmax((gam-(real)1)*(HF - (real)0.5*(unF*unF + utF*utF)),
                              (real)1e-6);
        const real rF  = pow(a2F/(gam*SF), (real)1/(gam-(real)1));
        const real pF  = SF*pow(rF, gam);
        const real vx = utF*t2x + unF*n[0];
        const real vy = utF*t2y + unF*n[1];
        const real vz = utF*t2z + unF*n[2];
        const real vdS = (d==0) ? vx : ((d==1) ? vy : vz);
        const real EF = pF/(gam-(real)1) + (real)0.5*rF*(vx*vx + vy*vy + vz*vz);
        Fw[0] = rF*vdS;
        Fw[1] = rF*vdS*vx + ((d==0) ? pF : (real)0);
        Fw[2] = rF*vdS*vy + ((d==1) ? pF : (real)0);
        Fw[3] = rF*vdS*vz + ((d==2) ? pF : (real)0);
        Fw[4] = (EF + pF)*vdS;
        FwK = 0; FwT = 0; uTauOut = 0;
        return true;
      }
      // sample failure: fall through to the legacy single-cell branch
    }
    const real rS   = fmax(Rho[fluidIdx],(real)1e-30);
    const real sS   = cons ? (real)1/rS : (real)1;
    const real ux   = U[fluidIdx]*sS, uy = V[fluidIdx]*sS, uz = W[fluidIdx]*sS;
    const real p1   = cons ? (gam-(real)1)*(P[fluidIdx] - (real)0.5*rS*(ux*ux + uy*uy + uz*uz))
                           : P[fluidIdx];
    // the sample's own wall distance (the line node at s = d1)
    // The FLUID side of this face is the side the wall normal points to, so
    // n[d] picks it without a geometric search.  d1 must also be strictly
    // FARTHER from the wall than the face (the cell centre is half a cell
    // beyond it); enforcing that keeps d_FC/d1 <= 1, so the normal-velocity
    // ramp below can never amplify.
    Vec3 cPos = fcPos;
    cPos[d] += (n[d] > (real)0) ? (real)0.5*h : -(real)0.5*h;
    const real d1 = fmax(-grid.getBoundaryLevelSet(cPos), dFc + (real)0.25*h);

    const real un1 = ux*n[0] + uy*n[1] + uz*n[2];
    real tx1 = ux - un1*n[0], ty1 = uy - un1*n[1], tz1 = uz - un1*n[2];
    const real ut1 = sqrt(tx1*tx1 + ty1*ty1 + tz1*tz1);
    const real inv = (ut1 > (real)1e-30) ? (real)1/ut1 : (real)0;
    tx1 *= inv; ty1 *= inv; tz1 *= inv;            // unit wall tangent

    // wall curvature kappa = div(n) (positive for a CONVEX body), from the level
    // set.  Capped at 1/(2h): a feature tighter than a couple of cells is not
    // resolved, and an uncapped kappa would amplify exactly there.
    real kap = 0;
    if (grid.ibCurv) {
      const real e2 = (real)0.5*h;
      Vec3 nxp = grid.wallNormal(Vec3(fcPos[0]+e2, fcPos[1], fcPos[2]), h);
      Vec3 nxm = grid.wallNormal(Vec3(fcPos[0]-e2, fcPos[1], fcPos[2]), h);
      Vec3 nyp = grid.wallNormal(Vec3(fcPos[0], fcPos[1]+e2, fcPos[2]), h);
      Vec3 nym = grid.wallNormal(Vec3(fcPos[0], fcPos[1]-e2, fcPos[2]), h);
      kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
      const real kMax = (real)0.5/h;
      kap = fmin(fmax(kap, -kMax), kMax);
    }

    // pressure along the line: dp/ds = rho u_t^2 kappa, evaluated at d_FC
    const real dpds = rS*ut1*ut1*kap;
    const real pFc  = fmax(p1 + (dFc - d1)*dpds, (real)1e-6*p1);
    // density: isentropic along the normal
    const real a2   = gam*fmax(p1,(real)1e-30)/rS;
    const real rFc  = fmax(rS + (pFc - p1)/a2, (real)1e-6*rS);
    // tangential: du_t/ds = -kappa u_t at the wall (FRIB Eq. 19), linearised
    const real den  = (real)1 - kap*d1;
    const real utFc = (fabs(den) > (real)0.2) ? ut1*((real)1 - kap*dFc)/den : ut1;
    // Normal component AT THE FACE: zero.  FRIB interpolates u_n from 0 at the
    // wall up to its high-order sample, but here the sample is a single FV cell
    // whose u_n is mostly noise, and any u_n at the face pushes mass toward the
    // solid side -- which is masked, so it acts as a source/sink.  Measured:
    // keeping a scaled u_n gave C_p up to +3.87 (impossible; the max is ~1).
    // Near a slip wall u_n is O(d_FC * dkappa) anyway, so zero is the right
    // leading-order choice at this order.
    const real unFc = 0;

    const real vx = utFc*tx1 + unFc*n[0];
    const real vy = utFc*ty1 + unFc*n[1];
    const real vz = utFc*tz1 + unFc*n[2];
    const real vdS = (d==0) ? vx : ((d==1) ? vy : vz);   // through the GRID face
    const real ES  = pFc/(gam-(real)1) + (real)0.5*rFc*(vx*vx + vy*vy + vz*vz);
    Fw[0] = rFc*vdS;
    Fw[1] = rFc*vdS*vx + ((d==0) ? pFc : (real)0);
    Fw[2] = rFc*vdS*vy + ((d==1) ? pFc : (real)0);
    Fw[3] = rFc*vdS*vz + ((d==2) ? pFc : (real)0);
    Fw[4] = (ES + pFc)*vdS;
    FwK = 0; FwT = 0; uTauOut = 0;
    return true;
  }

  // Diagnostic freeze (see CompressibleSolver::ibDfcMode): substitute the d_FC
  // that the STABLE geometry (d_FC = 0.5h) would supply, one term at a time.
  const real dFcRef = (real)0.5*h;
  const real dFcIp  = (grid.ibDfcMode == 4) ? dFcRef : dFc;
  Vec3 ip(fcPos[0] + (dIp-dFcIp)*n[0],
                fcPos[1] + (dIp-dFcIp)*n[1],
                fcPos[2] + (dIp-dFcIp)*n[2]);

  real *Fs[6] = {Rho, U, V, W, P, Tau};
  real q[6];
  if (!ibSample(grid, ip, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 6, q, true)) {
    if (grid.dbgChecks) atomicAdd(&g_ibFailIp, 1ULL); return false; }
  const real rIp = fmax(q[0],(real)1e-30);
  // computeDeltaTKernel hands us the CONSERVATIVE banks, so q[1..3] are rho*u,
  // q[4] is rho*E and q[5] is rho*tau~.  Convert before anything reads them --
  // the grid-aligned twin (wallModelStateY) carries the same flag for the same
  // reason.  Without this the only dt guard on the stiff immersed tau~ flux is
  // computed from a garbage state (rho*E is ~2.5x p at these conditions).
  const real sIp  = cons ? (real)1/rIp : (real)1;
  const real uIpX = q[1]*sIp, uIpY = q[2]*sIp, uIpZ = q[3]*sIp;
  const real pIp  = cons ? (gam-(real)1)*(q[4] - (real)0.5*rIp*(uIpX*uIpX
                             + uIpY*uIpY + uIpZ*uIpZ))
                         : q[4];
  const real tIp  = q[5]*sIp;

  // wall-parallel velocity at the image point
  const real vn = uIpX*n[0] + uIpY*n[1] + uIpZ*n[2];
  const real ux = uIpX - vn*n[0], uy = uIpY - vn*n[1], uz = uIpZ - vn*n[2];
  const real utMag = sqrt(ux*ux + uy*uy + uz*uz);
  const real tx = (utMag > 0) ? ux/utMag : (real)0;
  const real ty = (utMag > 0) ? uy/utMag : (real)0;
  const real tz = (utMag > 0) ? uz/utMag : (real)0;

  const real nuw  = grid.viscosity(pIp/rIp)/rIp;
  const real uTau = ktau::uTauFromWallFunction(utMag, dIp, nuw);
  uTauOut = uTau;
  const real yp   = dIp*uTau/nuw;
  const real dudy = uTau*uTau/nuw*ktau::dUplusDyplus(yp);          // Eq. (37)
  // Eq. (36) is the LINEARIZED evaluation u(d_FC) = u_IP - du/dy|_IP (d_IP-d_FC)
  // -- exact in the log region, catastrophic at the slip->model TRANSITION: the
  // incoming slip profile has u ~ 1 at d_IP, the inverted u_tau and gradient are
  // huge, the extrapolation lands far below zero and the clamp turns the first
  // modelled faces into full-stagnation walls.  Measured: a steady separation
  // bubble pinned behind wmX0 wherever it is placed (u_min -0.73 at
  // wmx0 + 0.05c, still there at t = 4), whose run-to-run flicker is the whole
  // 2.2x Cf scatter of the sharp plate gate.  Evaluate the wall-function
  // PROFILE at the face's own y+ instead: identical where Eq. 36 is valid,
  // overshoot-free where it is not.
  const real dFcE = (grid.ibDfcMode==1) ? dFcRef : dFc;
  const real ypF  = dFcE*uTau/nuw;
  const real upF  = (ypF < (real)11) ? ypF
                  : log(fmax(ypF,(real)1e-12))/(real)0.41 + (real)5.2;
  const real uFc  = fmin(uTau*upF, utMag);
  real tauW = rIp*uTau*uTau;
  const real nd   = n[d];
  // --wmramp L: blend the wall model in over a fetch L past wmX0.  The step
  // transition injects a stress AND turbulence front (k_FC ~ u_tau^2/sqrt(b*)
  // into a zero-k incoming layer) that trips a STEADY separation bubble pinned
  // behind wmX0 wherever it is placed (measured: u_min -0.73 at +0.05c, alive
  // at t=4; the bubble's flicker is the plate gate's whole 2.2x scatter).
  // Physical transition is not a step either; smoothstep tau_w and the Eq. 39
  // wall values over L.
  real wmBlend = (real)1;
  if (grid.wmRamp > (real)0) {
    const real x0r = (grid.wmX0 >= (real)0) ? grid.wmX0 : grid.plateX0;
    real sR = (fcPos[0] - x0r)/grid.wmRamp;
    sR = fmin(fmax(sR, (real)0), (real)1);
    wmBlend = sR*sR*((real)3 - (real)2*sR);
    tauW *= wmBlend;
  }

  // ---- wall-face pressure: carry it from the IP with the CURVATURE term ----
  // --wmcurv (default on).  This branch used to put the IMAGE-POINT pressure
  // straight onto the wall face.  On a curved body that is an O(d_IP) error:
  // the exact normal-momentum balance is dp/dn = rho u_t^2 kappa, so between
  // the IP and the face the pressure changes by rho u_t^2 kappa (d_FC - d_IP).
  // The validated SLIP path (ibFaceTraceFlux) has always applied it; the
  // wall-MODEL branch did not -- and kappa = 0 on a flat plate, which is
  // exactly why the plate gate passes at +3.0% while the RAE decambers.
  // MEASURED without it (RAE 2822 case 9, nLvls 7): wall model + eddy
  // viscosity gives Cl = 0.410 with no shock (inviscid 0.768, exp 0.803), and
  // wall model WITHOUT eddy viscosity (--wmles) DIVERGES -- i.e. SA was only
  // damping this error into a stable-but-wrong state.
  real pFc = pIp, rFc = rIp;
  if (grid.wmCurv) {
    real kapW = 0;
    {
      const real e2 = (real)0.5*h;
      Vec3 nxp = grid.wallNormal(Vec3(fcPos[0]+e2, fcPos[1], fcPos[2]), h);
      Vec3 nxm = grid.wallNormal(Vec3(fcPos[0]-e2, fcPos[1], fcPos[2]), h);
      Vec3 nyp = grid.wallNormal(Vec3(fcPos[0], fcPos[1]+e2, fcPos[2]), h);
      Vec3 nym = grid.wallNormal(Vec3(fcPos[0], fcPos[1]-e2, fcPos[2]), h);
      kapW = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
      const real kMax = (real)0.5/h;                // sub-2-cell features are noise
      kapW = fmin(fmax(kapW, -kMax), kMax);
    }
    pFc = fmax(pIp + (dFc - dIp)*rIp*utMag*utMag*kapW, (real)1e-6*pIp);
    const real a2w = gam*fmax(pIp, (real)1e-30)/rIp;         // isentropic along n
    rFc = fmax(rIp + (pFc - pIp)/a2w, (real)1e-6*rIp);
  }

  // convective flux of the FC state through the grid face (normal e_d)
  const real vd = uFc*((d==0)?tx:((d==1)?ty:tz));
  const real ke = (real)0.5*rFc*uFc*uFc;
  const real E  = pFc/(gam-(real)1) + ke;
  Fw[0] = rFc*vd;
  Fw[1] = rFc*vd*uFc*tx + ((d==0)?pFc:(real)0);
  Fw[2] = rFc*vd*uFc*ty + ((d==1)?pFc:(real)0);
  Fw[3] = rFc*vd*uFc*tz + ((d==2)?pFc:(real)0);
  Fw[4] = (E + pFc)*vd;
  // minus the viscous part: tau_{d i} = tau_w (t_i n_d + t_d n_i)
  const real td = (d==0)?tx:((d==1)?ty:tz);
  Fw[1] -= tauW*(tx*nd + td*n[0]);
  Fw[2] -= tauW*(ty*nd + td*n[1]);
  Fw[3] -= tauW*(tz*nd + td*n[2]);
  Fw[4] -= tauW*uFc*nd;              // u_j tau_{d j} = tau_w u_FC n_d  (t.n = 0)
  if (grid.ibWallMode == 4) {
    // Mode 4: the ORDINARY inviscid flux (HLLC through the log-law ghosts --
    // mode 1's calm pressure transmission) stays; only the wall-face VISCOUS
    // stress is prescribed, and exactly.  Hand back the tau_w part alone; the
    // caller strips its ordinary viscous contribution and adds this.
    Fw[0] = 0;
    Fw[1] = -tauW*(tx*nd + td*n[0]);
    Fw[2] = -tauW*(ty*nd + td*n[1]);
    Fw[3] = -tauW*(tz*nd + td*n[2]);
    Fw[4] = -tauW*uFc*nd;
  }

  // --wmles: the model ends here.  No turbulence transport exists, so no
  // Eq. (39) / SA wall fluxes -- the mean stress above is the whole model.
  if (!grid.rans) { FwK = 0; FwT = 0; return true; }

  // Eq. (39) turbulence values and the one-sided fluxes into the fluid cell
  real kFc, tFc;
  ktau::wallBcKTau(uTau, (grid.ibDfcMode==2)?dFcRef:dFc, tIp, dIp, kFc, tFc);
  kFc *= wmBlend;   // ramp the turbulence front with the stress (see wmRamp)
  const real s1  = cons ? (real)1/fmax(Rho[fluidIdx],(real)1e-30) : (real)1;
  const real k1 = K[fluidIdx]*s1, t1 = Tau[fluidIdx]*s1, f11 = TF1[fluidIdx];
  // The first fluid cell sits half a cell to the FLUID side of this face, and
  // its wall distance comes from the LEVEL SET -- d_FC + h/2 assumes the wall
  // is parallel to the face.  For a horizontal wall -phi(cell) == d_FC + h/2
  // exactly, so the aligned gate is unchanged; for an inclined wall the true
  // wall-NORMAL separation is (h/2)|n_d|, and both the gradients and the
  // Eq. (A.5) damping distances must be measured along the normal.  The n_d
  // factor then projects the wall-normal diffusive flux onto this grid face;
  // it also carries the orientation, so a body on the HIGH side (n_d < 0)
  // flips the one-sided flux instead of silently keeping the low-side sign.
  Vec3 fPos = fcPos;  fPos[d] += (real)0.5*h;
  if (!grid.isFluidCell(fPos, h)) fPos[d] -= h;
  const real d1 = fmax(-grid.getBoundaryLevelSet(fPos), dFc + (real)0.1*h);
  const real dd = d1 - dFc;      // wall-normal FC -> first-cell separation
  const real muF   = grid.viscosity(pIp/rIp);
  if (grid.turbModel == 1) {
    // ---- SA immersed wall value -------------------------------------------
    // The near-wall SA solution is a SINGLE algebraic value, nu~ = kappa u_tau d
    // (Eq. 16 of the k~-tau~ paper is the two-equation analogue).  It is linear
    // in the wall distance and monotone, so imposing it as a face value cannot
    // set up the production/dissipation balance that destabilises the k~ wall
    // flux -- this is the whole reason for switching models.
    const real nutFc = sa::nutWall(uTau, dFc);
    const real nut1  = fmax(k1, (real)0);         // first cell's nu~ (clamped: it is a diffusivity)
    const real nuW   = muF/fmax(rIp,(real)1e-30);
    FwK = -(nuW + (real)0.5*(nut1 + nutFc))/(real)sa::sigma*(nut1 - nutFc)/dd*nd*rIp;
    FwT = 0;
    if (!grid.ibTurbFlux) FwK = 0;
    return true;
  }
  const real sigKF = f11*ktau::sigK1 + ((real)1-f11)*ktau::sigK2;
  const real sigWF = f11*ktau::sigW1 + ((real)1-f11)*ktau::sigW2;
  FwK = -(muF + sigKF*rIp*kFc*tFc)*(k1 - kFc)/dd*nd;
  const real dCut = (grid.dCutoff > 0) ? grid.dCutoff : grid.dIpFac*h;
  const real dFcPh = (grid.ibDfcMode==3) ? dFcRef : dFc;
  const real phF  = ktau::phiDamp(dFcPh, dCut);
  const real phR  = ktau::phiDamp(dFcPh + dd, dCut);
  const real C    = ((real)1-f11)*muF + sigWF*rIp*kFc*tFc;
  real fdL, fdR;
  ktau::tauDiffFluxes(C, (t1-tFc)/dd, tFc, kFc, phF, tFc, t1, phF, phR,
                      rIp*kFc*tFc, Rho[fluidIdx]*k1*t1, (real)0, (real)0, kFc, k1, fdL, fdR,
                      grid.ransA7Tol);
  FwT = -fdR*nd;
  // diagnostic lever: --ibturb 0 keeps the wall STRESS on the mean flow but
  // drops the k~/tau~ wall fluxes, to separate the two failure paths
  if (!grid.ibTurbFlux) { FwK = 0; FwT = 0; }
  return true;
}

// ---- algebraic wall model: the boundary flux at a grid-aligned y wall -------
//
// Sec. 3 of the paper.  The wall sits grid.wallOffset below the bottom domain
// face, so the face centre (FC) is at wall distance d_FC = wallOffset and the
// image point (IP) at d_IP = 3 dy, both measured from the wall.
//
//   1. interpolate the state at the IP from the two cells straddling it
//   2. Newton-solve u_tau so that (u_IP, d_IP) satisfies the wall function
//   3. wall stress  tau_ij = rho_w u_tau^2 (t_i n_j + t_j n_i), t along the
//      wall-parallel IP velocity  ->  the viscous flux at FC is rho u_tau^2 t_i
//   4. linearize the tangential velocity down to FC (Eqs. 36-37): the profile
//      below the IP is a straight line, which is the ONLY thing a second-order
//      scheme can represent there, and is what the r_d eddy-viscosity
//      augmentation of Eq. (38) is built to be consistent with
//   5. k~ and tau~ at FC from the near-wall similarity solution (Eq. 39)
//
// Returns the TOTAL face flux (convective minus viscous) in the +y sense, so it
// drops straight into the fluxD slot.
//
struct WallState {
  real uTau, tx, tz, dudy, uFc, rw, pw, nuw, kFc, tFc, dFc;
};

// Steps 1-5 of the wall model, shared by the boundary flux and the ghost fill so
// the two can never drift apart.  (i,j,k) is the FIRST INTERIOR cell above the
// wall; its low-y face is the face centre FC.
__device__ inline WallState wallModelStateY(CompressibleSolver &grid,
                                            real *Rho, real *U, real *V, real *W,
                                            real *P, real *Tau,
                                            i32 bIdx, i32 i, i32 j, i32 k, real dy,
                                            bool cons = false)
{
  // `cons` says the bank holds CONSERVATIVE variables.  computeDeltaTKernel runs
  // between RK steps, when it does -- everything below is written in primitives,
  // so it has to convert on read or it would solve the wall function with rho*u
  // in place of u, rho*E in place of p and rho*tau~ in place of tau~.
  WallState w;
  const real dFc = fmax(grid.wallOffset, (real)1e-30);
  w.dFc = dFc;
  const real dIp = grid.dIpFac*dy;

  // --- 1. image point: y_IP = dIp - dFc above the bottom face; the current cell
  // centre sits at dy/2 above it, so the IP is (dIp - dFc)/dy - 0.5 cells up.
  const real cUp = (dIp - dFc)/dy - (real)0.5;
  const i32  m0  = (i32)floor((double)cUp);
  const real wgt = cUp - (real)m0;
  const i32 iA = grid.getNbrIdx(bIdx, i, j+max(m0,0),   k);
  const i32 iB = grid.getNbrIdx(bIdx, i, j+max(m0,0)+1, k);
  const real sA = cons ? (real)1/fmax(Rho[iA],(real)1e-30) : (real)1;
  const real sB = cons ? (real)1/fmax(Rho[iB],(real)1e-30) : (real)1;
  const real uIp = ((real)1-wgt)*U[iA]*sA   + wgt*U[iB]*sB;
  const real vIp = ((real)1-wgt)*V[iA]*sA   + wgt*V[iB]*sB;
  const real wIp = ((real)1-wgt)*W[iA]*sA   + wgt*W[iB]*sB;
  const real rIp = ((real)1-wgt)*Rho[iA]    + wgt*Rho[iB];
  const real tIp = ((real)1-wgt)*Tau[iA]*sA + wgt*Tau[iB]*sB;
  // P holds pressure in primitive form and rho*E in conservative form
  const real eA = cons ? (gam - (real)1)*(P[iA] - (real)0.5*(U[iA]*U[iA] + V[iA]*V[iA]
                          + W[iA]*W[iA])/fmax(Rho[iA],(real)1e-30)) : P[iA];
  const real eB = cons ? (gam - (real)1)*(P[iB] - (real)0.5*(U[iB]*U[iB] + V[iB]*V[iB]
                          + W[iB]*W[iB])/fmax(Rho[iB],(real)1e-30)) : P[iB];
  const real pIp = ((real)1-wgt)*eA + wgt*eB;
  (void)vIp;

  // --- 2. u_tau from the wall function
  const real utMag = sqrt(uIp*uIp + wIp*wIp);
  const real rw    = rIp;                       // Neumann density and pressure
  const real pw    = pIp;
  const real nuw   = grid.viscosity(pw/fmax(rw,(real)1e-30))/fmax(rw,(real)1e-30);
  const real uTau  = ktau::uTauFromWallFunction(utMag, dIp, nuw);
  const real tx = (utMag > 0) ? uIp/utMag : (real)0;
  const real tz = (utMag > 0) ? wIp/utMag : (real)0;

  // --- 3-4. wall stress and the linearized slip velocity at FC
  const real yp   = dIp*uTau/nuw;
  const real dudy = uTau*uTau/nuw*ktau::dUplusDyplus(yp);      // Eq. (37)
  const real uFc  = fmax(utMag - dudy*(dIp - dFc), (real)0);   // Eq. (36)

  ktau::wallBcKTau(uTau, dFc, tIp, dIp, w.kFc, w.tFc);         // Eq. (39)
  w.uTau = uTau;  w.tx = tx;  w.tz = tz;  w.dudy = dudy;
  w.uFc = uFc;    w.rw = rw;  w.pw = pw;  w.nuw = nuw;
  return w;
}

// The total wall face flux (convective minus viscous) in the +y sense, so it
// drops straight into the fluxD slot.
__device__ inline void wallModelFluxY(CompressibleSolver &grid,
                                      real *Rho, real *U, real *V, real *W, real *P,
                                      real *K, real *Tau, real *TF1,
                                      i32 bIdx, i32 i, i32 j, i32 k, real dy,
                                      real Fw[5], real &FwK, real &FwT, real &uTauOut,
                                      bool cons = false)
{
  const WallState w = wallModelStateY(grid, Rho, U, V, W, P, Tau, bIdx, i, j, k, dy, cons);
  const real uTau = w.uTau, tx = w.tx, tz = w.tz;
  const real rw = w.rw, pw = w.pw, uFc = w.uFc, dFc = w.dFc;
  const real kFc = w.kFc, tFc = w.tFc;
  uTauOut = uTau;
  const real tauW = rw*uTau*uTau;

  // total flux = convective - viscous.  No mass crosses the wall, so the
  // convective part is pure pressure; the viscous part is the wall stress.
  Fw[0] = 0;
  Fw[1] = -tauW*tx;
  Fw[2] = pw;
  Fw[3] = -tauW*tz;
  // Adiabatic wall (q_w = 0), but the stress still does work on the SLIPPING
  // face: -u_FC tau_w.  Dropping it is only valid for a no-slip wall, where
  // u_FC = 0 makes it vanish on its own.  Under a wall model u_FC > 0, and
  // zeroing this term converts the entire near-wall kinetic-energy loss into
  // internal energy instead of just the part dissipated inside the cell --
  // which heats the first row without bound and collapses the acoustic dt.
  Fw[4] = -tauW*uFc;

  // --- 5. one-sided turbulence fluxes against the Eq. (39) face values
  const i32  c1  = grid.getNbrIdx(bIdx, i, j, k);
  const real r1  = Rho[c1];
  const real s1  = cons ? (real)1/fmax(r1,(real)1e-30) : (real)1;
  const real k1  = K[c1]*s1, t1 = Tau[c1]*s1, f11 = TF1[c1];
  const real hHalf = (real)0.5*dy;                 // FC -> first cell centre
  const real muF   = grid.viscosity(pw/fmax(rw,(real)1e-30));
  const real sigKF = f11*ktau::sigK1 + ((real)1 - f11)*ktau::sigK2;
  const real sigWF = f11*ktau::sigW1 + ((real)1 - f11)*ktau::sigW2;

  if (grid.turbModel == 1) {
    // SA: same single algebraic wall value on the grid-aligned path
    const real nutFc = sa::nutWall(uTau, dFc);
    const real nuW   = muF/fmax(rw,(real)1e-30);
    FwK = -(nuW + (real)0.5*(k1 + nutFc))/(real)sa::sigma*(k1 - nutFc)/hHalf*rw;
    FwT = 0;
    return;
  }
  // k~: conservative diffusion against the Eq. (39) face value
  FwK = -(muF + sigKF*rw*kFc*tFc)*(k1 - kFc)/hHalf;

  // tau~: the Appendix-A pair with the FC state on the "left".  Only the
  // right-hand (interior) member is used -- the ghost side is discarded.
  const real dCut = (grid.dCutoff > 0) ? grid.dCutoff : (real)3*dy;
  const real phF  = ktau::phiDamp(dFc, dCut);
  const real phR  = ktau::phiDamp(dFc + hHalf, dCut);
  const real C    = ((real)1 - f11)*muF + sigWF*rw*kFc*tFc;
  real fdL, fdR;
  ktau::tauDiffFluxes(C, (t1 - tFc)/hHalf, tFc, kFc, phF,
                      tFc, t1, phF, phR,
                      rw*kFc*tFc, r1*k1*t1, (real)0, (real)0, kFc, k1, fdL, fdR, grid.ransA7Tol);
  FwT = -fdR;
}

// u_tau at every modeled wall face, stamped into F_SCRATCH for the C_f dump.
// Deliberately does NOT go through computeRightHandSide: that would accumulate
// into the LSRK bank, and since the C_f dump runs between steps it would leave
// the accumulator dirty for the next stage 1 (which assumes A_1 = 0).
__global__ void wallUtauKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *Tau = grid.getField(F_RHOTAU);
  real *Sc  = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    Sc[cIdx] = 0;
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb) && grid.immerserdBcType != 0) {
      // Immersed wall: stamp u_tau into the FLUID cell above each modelled face.
      // The grid-aligned branch below scans the domain bottom row, which for an
      // immersed body sits INSIDE it -- reading C_f there measures nothing.
      const real dy = grid.getDy(lvl);
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const i32 dnI = grid.getNbrIdx(bIdx, i, j-1, k);
      if (grid.getField(F_IBM)[cIdx] > (real)0.5 &&
          dnI < bEmpty*blockSizeTot &&
          grid.getField(F_IBM)[dnI] <= (real)0.5 &&
          pos[0] >= grid.plateX0) {
        Vec3 fc(pos[0], pos[1]-(real)0.5*dy, pos[2]);
        if (grid.ibWallMode == 1) {
          // Ghost-wall-function mode prescribes NO wall flux, so ibWallFlux
          // declines and would stamp nothing -- C_f would read 0 everywhere,
          // which is a reporting hole, not a dead wall.  Recover u_tau the same
          // way the model itself does: image point along the normal, then the
          // Eq. (6) Newton solve on its tangential speed.
          Vec3 nrm = grid.wallNormal(fc, dy);
          const real dFc = fmax(-grid.getBoundaryLevelSet(fc), (real)0.05*dy);
          const real sIP = (real)3*dy;
          Vec3 ip(fc[0] + (sIP-dFc)*nrm[0], fc[1] + (sIP-dFc)*nrm[1],
                  fc[2] + (sIP-dFc)*nrm[2]);
          real *Fs[5] = {Rho, U, V, W, P};
          real q5[5];
          if (ibSample(grid, ip, lvl, bIdx, i, j, k,
                       ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k, Fs, 5, q5)) {
            const real rM = fmax(q5[0], (real)1e-30);
            const real vn = q5[1]*nrm[0] + q5[2]*nrm[1] + q5[3]*nrm[2];
            const real ax = q5[1]-vn*nrm[0], ay = q5[2]-vn*nrm[1], az = q5[3]-vn*nrm[2];
            const real utM = sqrt(ax*ax + ay*ay + az*az);
            if (utM > (real)1e-30 && q5[4] > (real)0) {
              const real nuw = grid.viscosity(q5[4]/rM)/rM;
              Sc[cIdx] = ktau::uTauFromWallFunction(utM, sIP, nuw);
            }
          }
        } else {
        real Fi[5], Ki, Ti, Ut = 0;
        if (ibWallFlux(grid, Rho, U, V, W, P, grid.getField(F_RHOK),
                       grid.getField(F_RHOTAU), grid.getField(F_TF1),
                       lvl, fc, 1, dy, cIdx, Fi, Ki, Ti, Ut,
                       bIdx, i, j, k, ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k))
          Sc[cIdx] = Ut;
        }
      }
    }
    else if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      const real dy = grid.getDy(lvl);
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      if (fabs(pos[1] - (real)0.5*dy) < (real)1e-6*dy && pos[0] >= grid.plateX0) {
        const WallState w =
          wallModelStateY(grid, Rho, U, V, W, P, Tau, bIdx, i, j, k, dy);
        Sc[cIdx] = w.uTau;
      }
    }
  END_CELL_LOOP
}

// ---- low-Mach preconditioning ---------------------------------------------
//
// beta^2 = min(max(M^2, K M_inf^2), 1): unpreconditioned (beta = 1) wherever the
// flow is transonic, floored near stagnation so the system stays conditioned.
__device__ inline real precondBeta2(CompressibleSolver &grid, real q2, real c2) {
  const real M2 = q2/fmax(c2, (real)1e-30);
  return fmin(fmax(M2, grid.precondK*grid.precondMref2), (real)1);
}

// Preconditioned acoustic wavespeed (Turkel):
//   u' = 0.5 |u| (1 + b2),  c' = 0.5 sqrt(((b2-1)|u|)^2 + 4 b2 c^2),  lambda = u' + c'
// At Ma 0.2 (c = 5|u|) this is ~1.6 |u| instead of 6 |u|.
__device__ inline real precondLambda(real vel, real c, real b2) {
  const real up = (real)0.5*vel*((real)1 + b2);
  const real d  = (b2 - (real)1)*vel;
  const real cp = (real)0.5*sqrt(d*d + (real)4*b2*c*c);
  return up + cp;
}

// Apply the conservative-variable preconditioner P to a residual in place.
//
// The Turkel/Choi-Merkle preconditioner has the exact rank-one inverse
//   P^-1 = I + d a (x) b,   d = 1/b2 - 1,
//   a = (1, u, v, w, H)^T,  b = (g-1)/c^2 (q^2/2, -u, -v, -w, 1).
// Since b.a = (g-1)/c^2 (H - q^2/2) = 1 IDENTICALLY, Sherman-Morrison collapses
// to  P = I - (d/(1+d)) a (x) b = I - (1 - b2) a (x) b.  So preconditioning a
// residual is one dot product and one axpy -- no matrix, no solve.  b.R is the
// pressure combination of the residual, which is exactly the component whose
// acoustic time scale is being rescaled.  P is nonsingular for b2 > 0, so
// R = 0 <=> P R = 0: the steady state is untouched.
__device__ inline void precondResidual(real R[5], real u, real v, real w,
                                       real H, real c2, real b2) {
  const real q2  = u*u + v*v + w*w;
  const real bR  = (gam - (real)1)/fmax(c2,(real)1e-30)
                 * ((real)0.5*q2*R[0] - u*R[1] - v*R[2] - w*R[3] + R[4]);
  const real s   = ((real)1 - b2)*bR;
  R[0] -= s;
  R[1] -= s*u;
  R[2] -= s*v;
  R[3] -= s*w;
  R[4] -= s*H;
}

// Envelope check (--envcheck): first-writer-wins capture of the FIRST cell
// whose conservative state leaves the physical envelope for the transonic
// airfoil runs.  Runs between steps on CONSERVATIVE banks.  dbgCnt[60] holds
// the winning cIdx+1 (0 = no hit), so the host can print the cell AND its
// geometric neighborhood before amplification wipes the evidence -- every
// post-NaN forensic this week printed 1e27 garbage instead of the seed.
__global__ void envCheckKernel(CompressibleSolver &grid) {
  START_CELL_LOOP
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    // NO continue here (see gatherFaceFluxKernel): it skips END_CELL_LOOP's
    // increment and hangs -- this was the whole "envcheck crawl".
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
    const real r  = grid.getField(F_RHO)[cIdx];
    const real ru = grid.getField(F_RHOU)[cIdx], rv = grid.getField(F_RHOV)[cIdx];
    const real re = grid.getField(F_RHOE)[cIdx];
    const real rk = grid.getField(F_RHOK)[cIdx], rt = grid.getField(F_RHOTAU)[cIdx];
    const real ke = (real)0.5*(ru*ru + rv*rv)/fmax(r, (real)1e-30);
    const real p  = (gam - (real)1)*(re - ke);
    const real pRef = fmax(grid.fsP, (real)1e-30);  // envelope scales with freestream pressure (Ma-dependent nondim)
    bool bad = !(r > (real)0.02 && r < (real)20)
            || fabs(ru) > (real)5 || fabs(rv) > (real)5
            || !(p > (real)0.02*pRef && p < (real)12*pRef)
            || rk < (real)-1e-5 || rk > (real)0.2
            || rt < (real)-1e-4 || rt > (real)10
            || !isfinite(r + ru + rv + re + rk + rt);
    if (bad) atomicCAS((int*)&grid.dbgCnt[60], 0, (int)(cIdx + 1));
    }
  END_CELL_LOOP
}

__global__ void computeDeltaTKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);
  real *DeltaT = grid.getField(F_SCRATCH);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    // Owned-only under decomposition: a ghost's wavespeed is a one-substage-stale
    // copy of another PE's cell, so including it would make each PE's local min
    // (hence the allreduced dt, the step count, and the whole trajectory) depend
    // on the rank count.  Restrict to owned cells so the local min reduces to the
    // true global min -- identical for any number of ranks.
    // cells inside the body are frozen and carry no solution -- excluded, or
    // their (meaningless) state would set the global time step
    bool dtSolid = false;
    if (grid.immerserdBcType != 0)
      dtSolid = grid.getField(F_IBM)[cIdx] <= (real)0.5;   // cached mask
    // Covered parents are not evolved (restriction re-slaves them every
    // stage), so they must not bind the global step either.  Measured before
    // this guard: an nLvls-2 immersed run bound dt at a LEVEL-0 parent wall
    // face at the leading edge, ~10x below what the leaves need.
    if (grid.cFlagsList[cIdx] == PARENT) dtSolid = true;
#ifdef USE_MGPU
    if (grid.isOwnedBlock(lvl, ib, jb, kb) && !dtSolid) {
#else
    if (grid.isInteriorBlock(lvl, ib, jb, kb) && !dtSolid) {
#endif
      Vec5 q = grid.cons2prim(Vec5(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoW[cIdx], RhoE[cIdx]));
      real a   = sqrt(abs(gam*q[4]/(q[0]+1e-32)));
      real vel = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
      real dx  = min(grid.getDx(lvl), min(grid.getDy(lvl), grid.getDz(lvl)));
      real lam = a + vel;
      if (grid.precond) {
        const real b2 = precondBeta2(grid, vel*vel, a*a);
        lam = precondLambda(vel, a, b2);
        // the turbulence pair rides the same step but is convected at |u|, so
        // keep its own convective limit in play once the acoustics are rescaled
        if (grid.rans) lam = fmax(lam, vel);
      }
      DeltaT[cIdx] = dx / (lam + 1e-32);
      // --p1 on a cut element: the slope mode across a THIN element has the mass
      // sqrt(12 lambda_min(M)/A) -- its thickness -- instead of h, and the
      // explicit step must resolve that.  The inverse mass matrix is stored:
      // lambda_min(M) = 1/lambda_max(M^-1).  Pieces of a split cell fold their
      // own thickness into the host cell's entry.
      if (grid.p1 && grid.p1ElemOfCell) {
        auto thick = [&](i32 e) -> real {
          const P1Elem &E = grid.p1Elem[e];
          const real tr = E.m11 + E.m22, det = E.m11*E.m22 - E.m12*E.m12;
          const real disc = sqrt(fmax(tr*tr - (real)4*det, (real)0));
          const real lmax = (real)0.5*(tr + disc);                     // of M^-1 (units 1/h^2... times h^2 below)
          return E.h*sqrt((real)12/fmax(lmax*E.area, (real)1e-300)); }; // = sqrt(12 lambda_min(M_phi) h^2 / A)
        real t = dx;
        const i32 e = grid.p1ElemOfCell[cIdx];
        if (e >= 0) t = fmin(t, thick(e));
        if (grid.cutSplit && grid.cutSplitId && grid.p1ElemOfPiece) {
          const i32 sp = grid.cutSplitId[cIdx];
          if (sp >= 0) for (i32 p = 0; p < grid.cutSplitCell[sp].n; p++) {
            const i32 ep = grid.p1ElemOfPiece[grid.cutSplitCell[sp].first + p];
            if (ep >= 0) t = fmin(t, thick(ep));
          }
        }
        DeltaT[cIdx] = fmin(DeltaT[cIdx], t / (lam + (real)1e-32));
      }
      // Every live cut cell is advanced by its true flux, so none is exempt from
      // the step.  What keeps a sliver stable is --srd (state redistribution) or
      // the --cutpidamp diagonal; the global step is set by the uncut cells.

      // No phi-ratio dt clause: the pressure-tight form (Reiss 2021, Sec. 2.1)
      // leaves u +- c unchanged, so the plain acoustic CFL above is the whole
      // limit, and the point-implicit stamp in the RHS absorbs the band source
      // magnitude without touching the step.

      // Diffusive limit dt <= dx^2 / (2 * ndim * nu_max).  The binding
      // diffusivity is the larger of the kinematic viscosity nu = mu/rho and
      // the thermal diffusivity kap/(rho cp) = nu*gam/Pr (Pr < 1 -> heat wins).
      // Reported through the same DeltaT array, so the host's cfl scales both.
      if (grid.mu > 0) {
        real T     = q[4]/(q[0] + (real)1e-32);
        real nu    = grid.viscosity(T)/(q[0] + (real)1e-32);
        real numax = nu*fmax((real)1.0, gam/grid.Pr);
        real ndim  = grid.pseudo2D ? (real)2.0 : (real)3.0;
        real dtVis = dx*dx / ((real)2.0*ndim*numax + (real)1e-32);
        DeltaT[cIdx] = fmin(DeltaT[cIdx], dtVis);
      }

      // RANS: the eddy viscosity can dwarf the molecular one, so the diffusive
      // limit has to see mu + mu_t; and the k~ destruction beta* k~/tau~ has its
      // own time scale tau~/beta*, which gets short as tau~ -> 0 at a wall.
      if (grid.rans && grid.mu > 0) {
        real muT   = grid.getField(F_MUT)[cIdx];
        real nuT   = muT/(q[0] + (real)1e-32);
        // The mean-flow diffusivity is mu + mu_t; the TURBULENCE equations carry
        // sigma * rho k~ tau~ instead (Eqs. 25 and A.3), which is not bounded by
        // mu_t once the SST limiter of Eq. (28) is active.  sigK2 = 1 is the
        // largest of the four blended sigmas, so one term bounds both.
        const real nuMol = grid.viscosity(q[4]/(q[0]+(real)1e-32))/(q[0]+(real)1e-32);
        const real ktt   = grid.getField(F_RHOK)[cIdx]*grid.getField(F_RHOTAU)[cIdx]
                         /((q[0]+(real)1e-32)*(q[0]+(real)1e-32));
        real numax = fmax(nuT*fmax((real)1.0, gam/grid.PrT)
                            + nuMol*fmax((real)1.0, gam/grid.Pr),
                          nuMol + ktau::sigK2*fmax(ktt,(real)0));
        real ndim  = grid.pseudo2D ? (real)2.0 : (real)3.0;
        DeltaT[cIdx] = fmin(DeltaT[cIdx], dx*dx/((real)2.0*ndim*numax + (real)1e-32));
        real tt = grid.getField(F_RHOTAU)[cIdx]/(q[0] + (real)1e-32);
        if (tt > 0) DeltaT[cIdx] = fmin(DeltaT[cIdx], tt/ktau::betaStar);

        // The wall boundary flux is by far the stiffest thing in the problem and
        // is invisible to every limit above: at the leading edge tau~ has to fall
        // from its freestream value to the Eq. (39) wall value across HALF a
        // cell, and the resulting Appendix-A flux can drive tau~ negative in a
        // single explicit step (which the tau~^2 in the source cannot recover
        // from).  The paper never meets this because it integrates implicitly.
        // So bound dt by the time either wall flux would consume its own cell
        // value.  Only first-row cells above the plate pay for this.
        // Immersed boundary: the same stiff wall tau~ flux, and the same need to
        // bound dt by it.  Guarding the cap below on wallGeom == 1 left the IB
        // path unprotected, so tau~ went negative in a couple of steps and the
        // beta* rho k~/tau~ term then produced NaN.
        if (grid.immerserdBcType != 0 && !grid.wallPointImplicit) {
          const i32 lc2 = cIdx % blockSizeTot;
          const i32 ci = lc2 % blockSize, cj = (lc2/blockSize) % blockSize,
                    ck = lc2/blockSize/blockSize;
          const real hx = grid.getDx(lvl), hy = grid.getDy(lvl);
          Vec3 cp2 = grid.getCellPos(lvl, ib, jb, kb, ci, cj, ck);
          const bool fC2 = grid.getField(F_IBM)[cIdx] > (real)0.5;   // cached mask
          const real rtt = grid.getField(F_RHOTAU)[cIdx];
          const real rkk = grid.getField(F_RHOK)[cIdx];
          const real cap = (real)0.25;
          for (i32 dd = 0; dd < 2; dd++) {          // x and y low faces
            const real hd = dd ? hy : hx;
            const i32 nI2 = grid.getNbrIdx(bIdx, ci - (dd ? 0 : 1),
                                           cj - (dd ? 1 : 0), ck);
            if (nI2 >= bEmpty*blockSizeTot) continue;
            if (fC2 == (grid.getField(F_IBM)[nI2] > (real)0.5)) continue;
            Vec3 fc2(cp2[0] - (dd ? (real)0 : (real)0.5*hd),
                     cp2[1] - (dd ? (real)0.5*hd : (real)0), cp2[2]);
            real Fi[5], Ki, Ti, Ut;
            if (!ibWallFlux(grid, grid.getField(F_RHO), grid.getField(F_RHOU),
                            grid.getField(F_RHOV), grid.getField(F_RHOW),
                            grid.getField(F_RHOE), grid.getField(F_RHOK),
                            grid.getField(F_RHOTAU), grid.getField(F_TF1),
                            lvl, fc2, dd, hd, cIdx, Fi, Ki, Ti, Ut,
                            bIdx, ci, cj, ck, ib*blockSize+ci, jb*blockSize+cj,
                            kb*blockSize+ck, true)) continue;
            if (fabs(Ti) > 0 && rtt > 0)
              DeltaT[cIdx] = fmin(DeltaT[cIdx], cap*rtt*hd/fabs(Ti));
            if (fabs(Ki) > 0 && rkk > 0)
              DeltaT[cIdx] = fmin(DeltaT[cIdx], cap*rkk*hd/fabs(Ki));
          }
        }

        if (grid.wallGeom == 1 && !grid.wallPointImplicit) {
          const real dyL = grid.getDy(lvl);
          const i32 lidx = cIdx % blockSizeTot;
          const i32 i = lidx % blockSize;
          const i32 j = (lidx / blockSize) % blockSize;
          const i32 k = lidx / blockSize / blockSize;
          Vec3 wpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
          if (fabs(wpos[1] - (real)0.5*dyL) < (real)1e-6*dyL && wpos[0] >= grid.plateX0) {
            real Fw[5], FwK, FwT, ut;
            wallModelFluxY(grid, grid.getField(F_RHO), grid.getField(F_RHOU),
                           grid.getField(F_RHOV), grid.getField(F_RHOW),
                           grid.getField(F_RHOE), grid.getField(F_RHOK),
                           grid.getField(F_RHOTAU), grid.getField(F_TF1),
                           bIdx, i, j, k, dyL, Fw, FwK, FwT, ut, true);
            // The step advances the CONSERVATIVE rho*k~ and rho*tau~ by dt*Fw/dy,
            // so the cap has to compare the flux against the CONSERVED cell
            // content, not the primitive value.
            const real rtt = grid.getField(F_RHOTAU)[cIdx];
            const real rkk = grid.getField(F_RHOK)[cIdx];
            const real cap = (real)0.25;      // consume at most a quarter per step
            if (fabs(FwT) > 0 && rtt > 0)
              DeltaT[cIdx] = fmin(DeltaT[cIdx], cap*rtt*dyL/fabs(FwT));
            if (fabs(FwK) > 0 && rkk > 0)
              DeltaT[cIdx] = fmin(DeltaT[cIdx], cap*rkk*dyL/fabs(FwK));
          }
        }
      }
    }
    else {
      DeltaT[cIdx] = 1e32;
    }

  END_CELL_LOOP
}


//
// ---- Navier-Stokes viscous face flux --------------------------------------
//
// Fv at the LOW face of cell (i,j,k) in direction d (the face shared with cell
// (i,j,k)-e_d), matching the convective fluxL/fluxD/fluxB convention.
//
//   tau_dc = mu ( du_d/dx_c + du_c/dx_d ) - (2/3) mu div(u) delta_dc
//   q_d    = -kap dT/dx_d,   kap = mu*gam/((gam-1)*Pr),   T = p/rho  (R = 1)
//   Fv     = [ 0, tau_d0, tau_d1, tau_d2, u_c tau_dc - q_d ]
//
// Face gradients: the FACE-NORMAL derivative is the compact two-point jump
// (u_R - u_L)/h -- this is what keeps the viscous operator free of odd-even
// decoupling.  The two TANGENTIAL derivatives are the average of the two
// neighbouring cells' central differences.  Both are second-order at the face.
// Stencil reach is (+-1, +-1), inside the 2-cell halo the convective path
// already requires.
//
__device__ inline void viscFaceFlux(CompressibleSolver &grid,
                                    real *Rho, real *U, real *V, real *W, real *P,
                                    i32 bIdx, i32 i, i32 j, i32 k, i32 d,
                                    real dx, real dy, real dz, real Fv[5],
                                    real muT = (real)0,
                                    i32 lvl = -1, Vec3 cpos = Vec3(0,0,0)) {
  const i32 di = (d == 0), dj = (d == 1), dk = (d == 2);
  const i32 iL = grid.getNbrIdx(bIdx, i-di, j-dj, k-dk);
  const i32 iR = grid.getNbrIdx(bIdx, i,    j,    k   );

  real h[3]    = {dx, dy, dz};
  real *Vel[3] = {U, V, W};
  (void)lvl; (void)cpos;

  // velocity gradient tensor at the face: g[c][m] = d(u_c)/d(x_m)
  real g[3][3];
  for (i32 c = 0; c < 3; c++) {
    g[c][d] = (Vel[c][iR] - Vel[c][iL]) / h[d];          // normal: compact jump
    for (i32 m = 0; m < 3; m++) {
      if (m == d) continue;
      if (grid.pseudo2D && m == 2) { g[c][m] = 0.0; continue; }   // z is collapsed
      const i32 mi = (m == 0), mj = (m == 1), mk = (m == 2);
      i32 Lp = grid.getNbrIdx(bIdx, i-di+mi, j-dj+mj, k-dk+mk);
      i32 Lm = grid.getNbrIdx(bIdx, i-di-mi, j-dj-mj, k-dk-mk);
      i32 Rp = grid.getNbrIdx(bIdx, i+mi,    j+mj,    k+mk   );
      i32 Rm = grid.getNbrIdx(bIdx, i-mi,    j-mj,    k-mk   );
      real spL = (real)2, spR = (real)2;     // cell-widths spanned by each bracket
      // Ghost-free IB: a tangential tap inside the body carries no solution.
      // Collapse that side onto the face cell so the difference stays one-sided
      // in the fluid instead of reaching across the wall.
      if (lvl >= 0 && grid.immerserdBcType != 0 && grid.ibGhostFree) {
        const real hmin = fmin(h[0], h[1]);
        const real ox = -di*h[0], oy = -dj*h[1], oz = -dk*h[2];
        Vec3 pLp(cpos[0]+ox+mi*h[0], cpos[1]+oy+mj*h[1], cpos[2]+oz+mk*h[2]);
        Vec3 pLm(cpos[0]+ox-mi*h[0], cpos[1]+oy-mj*h[1], cpos[2]+oz-mk*h[2]);
        Vec3 pRp(cpos[0]+mi*h[0],    cpos[1]+mj*h[1],    cpos[2]+mk*h[2]);
        Vec3 pRm(cpos[0]-mi*h[0],    cpos[1]-mj*h[1],    cpos[2]-mk*h[2]);
        if (!grid.isFluidCell(pLp, hmin)) { Lp = iL; spL -= (real)1; }
        if (!grid.isFluidCell(pLm, hmin)) { Lm = iL; spL -= (real)1; }
        if (!grid.isFluidCell(pRp, hmin)) { Rp = iR; spR -= (real)1; }
        if (!grid.isFluidCell(pRm, hmin)) { Rm = iR; spR -= (real)1; }
      }
      // Each bracket is a difference over spL (resp. spR) cell widths -- 2 when
      // both taps survive, 1 when one collapsed onto the face cell, 0 when both
      // did.  Dividing by a fixed 2 (the old 0.25 factor) HALVED every collapsed
      // gradient: an O(1) error that does not vanish as h -> 0, and one that
      // disagreed with turbClosureKernel's own correct one-sided handling
      // (inv = (okP && okM) ? 0.5/h : 1/h) in the very row the wall model uses.
      const real dL = (spL > 0) ? (Vel[c][Lp] - Vel[c][Lm])/(spL*h[m]) : (real)0;
      const real dR = (spR > 0) ? (Vel[c][Rp] - Vel[c][Rm])/(spR*h[m]) : (real)0;
      g[c][m] = (real)0.5*(dL + dR);
    }
  }

  // face state (arithmetic mean) and transport coefficients
  const real rL = Rho[iL], rR = Rho[iR];
  const real TL = P[iL]/(rL + (real)1e-32), TR = P[iR]/(rR + (real)1e-32);
  const real Tf = (real)0.5*(TL + TR);
  // RANS: the Reynolds stress enters as an eddy viscosity added to mu, and the
  // turbulent heat flux as mu_t cp/Pr_t added to the conductivity.  Following the
  // paper, the -(2/3) rho k~ delta_ij TKE term of Eq. (1) is OMITTED.
  // The molecular term keeps its ORIGINAL association so that muT == 0 reproduces
  // the Navier-Stokes path bit-for-bit (adding an exact 0.0 is exact).
  const real muMol = grid.viscosity(Tf);
  const real mu    = muMol + muT;
  const real kap   = muMol*gam/((gam - (real)1.0)*grid.Pr)
                   + muT *gam/((gam - (real)1.0)*grid.PrT);
  const real dTdn = (TR - TL) / h[d];

  const real div = g[0][0] + g[1][1] + g[2][2];
  real tau[3];
  for (i32 c = 0; c < 3; c++)
    tau[c] = mu*(g[d][c] + g[c][d]) - ((c == d) ? (real)(2.0/3.0)*mu*div : (real)0.0);

  const real uf = (real)0.5*(U[iL] + U[iR]);
  const real vf = (real)0.5*(V[iL] + V[iR]);
  const real wf = (real)0.5*(W[iL] + W[iR]);

  Fv[0] = 0.0;
  Fv[1] = tau[0];
  Fv[2] = tau[1];
  Fv[3] = tau[2];
  Fv[4] = uf*tau[0] + vf*tau[1] + wf*tau[2] + kap*dTdn;   // -q_d = +kap dT/dx_d
}

// ---- k~-tau~ SST closure ---------------------------------------------------
//
// One pass over every cell: velocity gradients -> S and Omega, grad(k~).grad(tau~)
// for Gamma3, the wall distance, then the whole algebraic closure (KtauSst.h).
// Writes mu_t and F1 for the face loop to read, and accumulates the cell-local
// source terms of Eqs. (25)-(27) and (32) straight into the LSRK bank.
//
// Runs on EXTERIOR cells too -- their primitives were just set by the boundary
// condition, so the closure there is as valid as anywhere, and the face loop
// needs mu_t/F1 on both sides of every boundary face.  A gradient whose tap is
// missing (the outermost ghost ring) degrades to zero for that direction.
//
// Uniform-box RANS gate: max relative deviation of k~ and tau~ from the exact
// 0-D solution of their source ODEs.  Nonzero here means either the sources are
// wrong OR the new convective/diffusive fluxes failed to cancel on a uniform
// state -- both are failures worth catching, and neither can hide behind the
// other because the exact solution is known.
__global__ void ransDecayErrorKernel(CompressibleSolver &grid, real kEx, real tEx, i32 mode) {
  real *Rho = grid.getField(F_RHO);
  real *K   = grid.getField(F_RHOK);      // conservative here: called after a step
  real *T   = grid.getField(F_RHOTAU);
  real *Sc  = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      const real r = fmax(Rho[cIdx], (real)1e-30);
      const real k = K[cIdx]/r, t = T[cIdx]/r;
      // mode 0: deviation from the exact 0-D solution (source + time error)
      // mode 1/2: k~ itself, for a max (1) and a min (2) over the INTERIOR only,
      //           whose spread measures UNIFORMITY -- whether the convective and
      //           diffusive fluxes cancelled, independently of how accurately
      //           the ODE was integrated.  Non-interior slots take the neutral
      //           element of the reduction, NOT the exact value: filling them
      //           with kEx would silently re-measure the time error instead.
      Sc[cIdx] = (mode == 0)
               ? fmax(fabs(k - kEx)/fmax(fabs(kEx), (real)1e-300),
                      fabs(t - tEx)/fmax(fabs(tEx), (real)1e-300))
               : k;
    }
    else Sc[cIdx] = (mode == 0) ? (real)0
                  : ((mode == 1) ? (real)-1e300 : (real)1e300);
  END_CELL_LOOP
}

// Frozen-shear probe: compare the k~ source the solver actually built against the
// same source evaluated with the ANALYTIC vorticity.  The only difference between
// the two is the discrete gradient stencil, so the residual is that stencil's
// error and must fall like O(h^2).
__global__ void ransShearProbeKernel(CompressibleSolver &grid, real u0, real ky) {
  real *Rho = grid.getField(F_RHO);
  real *P   = grid.getField(F_RHOE);
  real *K   = grid.getField(F_RHOK);        // primitive
  real *Tau = grid.getField(F_RHOTAU);
  real *RhsK = grid.getField(F_RHS + F_RHOK);
  real *Sc   = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const real Om  = fabs(u0*ky*cos(ky*pos[1]));   // exact |du/dy| = S = Omega
      const real r   = fmax(Rho[cIdx], (real)1e-30);
      const real nu  = grid.viscosity(P[cIdx]/r)/r;
      const real d   = grid.wallDistance(pos);
      const ktau::Closure c =
        ktau::closure(r, K[cIdx], Tau[cIdx], nu, d, Om, Om, (real)0,
                      grid.mu, grid.Lref, (real)1, grid.ransVorticity != 0);
      real sk, st;
      ktau::sources(c, r, K[cIdx], Tau[cIdx], grid.kInf, grid.tauInf,
                    grid.ransSustain != 0, sk, st);
      const real scale = ktau::betaStar*r*K[cIdx]/fmax(Tau[cIdx],(real)1e-30);
      Sc[cIdx] = fabs(RhsK[cIdx] - sk)/fmax(scale, (real)1e-300);
    }
    else Sc[cIdx] = 0;
  END_CELL_LOOP
}

// Near-wall equilibrium probe: the residual of Eqs. (25)-(26) on the analytic
// similarity solution, which must vanish.  Reported per band in y+ because the
// stencil cannot resolve u+(y+) near the wall -- that is precisely the region the
// wall model exists to replace -- so only the log region is expected to balance.
__global__ void ransWallProbeKernel(CompressibleSolver &grid, real uTau, real ypMin, i32 comp) {
  real *Rho  = grid.getField(F_RHO);
  real *K    = grid.getField(F_RHOK);
  real *Tau  = grid.getField(F_RHOTAU);
  real *RhsK = grid.getField(F_RHS + F_RHOK);
  real *RhsT = grid.getField(F_RHS + F_RHOTAU);
  real *Sc   = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    Sc[cIdx] = 0;
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const real h  = grid.getDy(lvl);
      const real y  = pos[1];
      const real yp = y*uTau/grid.mu;
      // The TOP rows are still excluded -- that boundary is a generic
      // zero-gradient BC.  The bottom rows are included once the wall model owns
      // that face (wallGeom 1), because reproducing the balance THERE is the
      // whole point.  Below ypMin the stencil cannot resolve u+(y+) anyway,
      // which is exactly the region the wall model exists to replace.
      const real yLoCut = (grid.wallGeom == 1) ? (real)0 : (real)2.5*h;
      const bool inBand = (y > yLoCut)
                       && (y < grid.domainSize[1] - (real)2.5*h)
                       && (yp >= ypMin);
      if (inBand) {
        const real r = Rho[cIdx];
        if (comp == 0) {                 // k~ equation: production = dissipation
          const real scale = ktau::betaStar*r*K[cIdx]/fmax(Tau[cIdx],(real)1e-30);
          Sc[cIdx] = fabs(RhsK[cIdx])/fmax(scale,(real)1e-300);
        } else {                         // tau~ equation: the Eq. (24) balance
          Sc[cIdx] = fabs(RhsT[cIdx])/(ktau::beta1*r);
        }
      }
    }
  END_CELL_LOOP
}

// Scratch probe for the RANS field extremes, so a run that throttles dt or blows
// up can be attributed to a specific quantity instead of guessed at.
//   0: k~   1: tau~   2: -tau~ (max gives the min)   3: mu_t/mu   4: the dt limits
// ---- Jacobian-free Newton-Krylov workspace for the k~/tau~ pair -------------
//
// The immersed wall model's failure mode is a MARGINAL one, not a stiff one:
// with Eq. (19) and Eq. (22) the near-wall balance is P_k/(beta* rho k~/tau~) =
// limSST <= 1 exactly, so the k~ source Jacobian very nearly VANISHES there.
// Nothing damps that direction, any spatial truncation error tips it, and the
// explicit march walks off.  Point-implicit source treatment cannot help (the
// Jacobian it would invert is the one that is ~0), and an exponential
// integrator would reproduce the tipped eigenvalue faithfully rather than
// remove it.  What does work is not integrating the transient at all: solve
// R(q) = 0 for the pair directly.  Newton needs only J*v, and J*v comes from a
// finite difference of the residual -- no Jacobian is ever formed, which matters
// here because R() is one large fused kernel over a 232 MB field bank.
//
// Vector layout: [0,N) = k~, [N,2N) = tau~, N = nKeys*blockSizeTot.  Cells that
// are not active interior are held at zero so they cannot pollute a dot product.
// flat vector helpers (plain kernels, not device lambdas: the build does not
// enable --extended-lambda and this is not worth changing it for)

// FULL-SYSTEM gather: all NEVOLVE conservative DOFs, laid out field-major.
// The paper solves the whole system implicitly with local time stepping, not
// just the turbulence pair, so the Newton state is every evolved variable.
// cells inside the immersed body are NOT evolved -- the BC/ghost machinery sets
// them -- and the shared accumulator bank legitimately holds non-finite junk
// there (the face-flux scatter writes it and nothing cleans it).  Including them
// put NaN into R0, which poisoned every GMRES inner product.  Gather, residual
// and scatter must all use THIS test or the vector space is inconsistent.




// A*v = v/dtau_local - J*v, with dtau taken PER CELL from the local time step
// (F_DTL).  Local time stepping is what the paper pairs with implicit
// integration: each cell advances toward the steady state at its own stable
// rate, which is legitimate precisely because only R(q) = 0 is being sought.
// Zero one field over every live cell (kernel, not a memset of the full cap).
__global__ void zeroFieldKernel(CompressibleSolver &grid, i32 f) {
  real *F = grid.getField(f);
  START_CELL_LOOP
    F[cIdx] = (real)0;
  END_CELL_LOOP
}



__global__ void ransFieldProbeKernel(CompressibleSolver &grid, i32 which) {
  real *Rho = grid.getField(F_RHO);
  real *K   = grid.getField(F_RHOK);
  real *Tau = grid.getField(F_RHOTAU);
  real *MuT = grid.getField(F_MUT);
  real *Sc  = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real v = -1e30;
    // Cells the body contains or intersects carry NO solution -- they hold ghost
    // or initial-condition values -- so they must be excluded from every probe
    // here.  Without this the immersed cases report maxima taken over the INSIDE
    // of the body: that is what made case 14 look like it had |rhoV| ~ 1.2 and a
    // 50x worse wall-normal momentum than the grid-aligned control.  (Same class
    // of error as wallUtauKernel scanning the domain bottom row, which for an
    // immersed body is inside it.)
    const bool probeFluid = (grid.immerserdBcType == 0)
                          || (grid.getField(F_IBM)[cIdx] > (real)0.5);
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb) && probeFluid) {
      const real r = fmax(Rho[cIdx], (real)1e-30);
      const real kk = K[cIdx]/r, tt = Tau[cIdx]/r;    // conservative on entry
      if      (which == 0) v = kk;
      else if (which == 1) v = tt;
      else if (which == 2) v = -tt;
      else if (which == 3) v = MuT[cIdx]/fmax(grid.mu,(real)1e-30);
      else if (which == 4) {
        const real dx = fmin(grid.getDx(lvl), grid.getDy(lvl));
        const real nuT = MuT[cIdx]/r;
        v = -fmin(dx*dx/((real)4*nuT + (real)1e-32), tt/ktau::betaStar);  // max = -min
      }
      else if (which == 9) {
        // count of non-finite evolved values (fmax/fmin hide NaNs, max does not
        // hide this): 1 if this cell is bad, else 0
        real bad = 0;
        for (i32 f = 0; f < NEVOLVE; f++) {
          const real q = grid.getField(f)[cIdx];
          if (!isfinite((double)q)) bad = 1;
        }
        v = bad;
      }
      else if (which == 7 || which == 8) {
        // |rho V| split by level: 7 = finest (leaves), 8 = coarser (parents).
        const bool fine = (lvl == grid.nLvls-1);
        v = ((which == 7) == fine) ? fabs(grid.getField(F_RHOV)[cIdx]) : (real)-1e30;
      }
      else {
        // max sound speed: the acoustic dt limiter's driver.  A hot spot shows
        // up here long before it becomes a visible blow-up.
        real *P = grid.getField(F_RHOE);
        real *RU = grid.getField(F_RHOU), *RV = grid.getField(F_RHOV);
        const real ke = (real)0.5*(RU[cIdx]*RU[cIdx] + RV[cIdx]*RV[cIdx])/r;
        const real pp = (gam - (real)1)*(P[cIdx] - ke);
        v = (which == 5) ? sqrt(fabs(gam*pp/r)) : r;
      }
    }
    Sc[cIdx] = v;
  END_CELL_LOOP
}

__global__ void turbClosureKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *K   = grid.getField(F_RHOK);      // primitive: cons2prim has run
  real *Tau = grid.getField(F_RHOTAU);
  real *MuT = grid.getField(F_MUT);
  real *TF1 = grid.getField(F_TF1);
  real *RhsK = grid.getField(F_RHS + F_RHOK);
  real *RhsT = grid.getField(F_RHS + F_RHOTAU);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    const real hh[3] = {grid.getDx(lvl), grid.getDy(lvl), grid.getDz(lvl)};
    const i32 cEmpty = bEmpty*blockSizeTot;

    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);

    real g[3][3] = {{0,0,0},{0,0,0},{0,0,0}};    // g[c][m] = d(u_c)/d(x_m)
    real gk[3] = {0,0,0}, gt[3] = {0,0,0};
    real *Vel[3] = {U, V, W};
    for (i32 m = 0; m < 3; m++) {
      if (grid.pseudo2D && m == 2) continue;      // z is collapsed
      const i32 mi = (m == 0), mj = (m == 1), mk = (m == 2);
      const i32 ip = grid.getNbrIdx(bIdx, i+mi, j+mj, k+mk);
      const i32 im = grid.getNbrIdx(bIdx, i-mi, j-mj, k-mk);
      if (ip >= cEmpty || im >= cEmpty) continue; // missing tap -> zero gradient
      // Ghost-free IB: a tap inside the body carries no solution, so fall back to
      // the one-sided difference on the fluid side rather than reading it.  Below
      // the image point the model's profile is a straight line, so the one-sided
      // slope IS the modelled slope -- which is exactly what an immersed ghost
      // value would have been constructed to reproduce.
      bool okP = true, okM = true;
      if (grid.immerserdBcType != 0 && grid.ibGhostFree) {
        Vec3 pp(pos[0]+mi*hh[0], pos[1]+mj*hh[1], pos[2]+mk*hh[2]);
        Vec3 pm(pos[0]-mi*hh[0], pos[1]-mj*hh[1], pos[2]-mk*hh[2]);
        const real hmin = fmin(hh[0], hh[1]);
        okP = grid.isFluidCell(pp, hmin);
        okM = grid.isFluidCell(pm, hmin);
      }
      if (!okP && !okM) continue;                 // walled on both sides
      const real inv = (okP && okM) ? (real)0.5/hh[m] : (real)1/hh[m];
      const i32 a = okP ? ip : cIdx;
      const i32 b = okM ? im : cIdx;
      for (i32 c = 0; c < 3; c++) g[c][m] = (Vel[c][a] - Vel[c][b])*inv;
      gk[m] = (K[a]   - K[b]  )*inv;
      gt[m] = (Tau[a] - Tau[b])*inv;
    }

    // S   = sqrt( (du_m/dx_n + du_n/dx_m)^2 / 2 ) = sqrt(2 S_ij S_ij)
    // Om  = sqrt( (du_m/dx_n - du_n/dx_m)^2 / 2 ) = |vorticity| in a shear layer
    real s2 = 0, w2 = 0;
    for (i32 a = 0; a < 3; a++)
      for (i32 b = 0; b < 3; b++) {
        const real sp = g[a][b] + g[b][a], sm = g[a][b] - g[b][a];
        s2 += (real)0.5*sp*sp;
        w2 += (real)0.5*sm*sm;
      }
    const real S    = sqrt(fmax(s2,(real)0));
    const real Om   = sqrt(fmax(w2,(real)0));
    const real gkgt = gk[0]*gt[0] + gk[1]*gt[1] + gk[2]*gt[2];

    const real r   = fmax(Rho[cIdx], (real)1e-30);
    const real nu  = grid.viscosity(P[cIdx]/r)/r;

    // Inside the body there is no solution and no wall distance: the closure
    // would see d = 0 and r_d = dCutoff/d ~ 1e27, which poisons mu_t and then
    // the time step.  Solid cells carry mu_t = 0 and contribute no sources.
    // Under Brinkman, isFluidCell returns TRUE EVERYWHERE by design (the porosity
    // replaces the sharp mask wholesale), so F_IBM cannot identify the body here.
    // Use the cached level set: F_PHI > 0 is inside the solid, where d = 0 makes
    // the closure's near-wall damping phi(d) vanish and its 1/phi terms give
    // ~1e20 -- measured as NaN in k~/tau~ in EVERY cell within one output
    // interval, with mu_t nonzero in all 49408 cells (the sharp path had it in
    // 35072, i.e. fluid only). That count is the tell.
    const bool inBody = (grid.immerserdBcType != 0)
        // >= 0, NOT > 0.  The default ibplane is exactly 4.5 h_fine and cell
        // centres sit at (i+0.5)h, so the wall passes precisely THROUGH a cell
        // centre: that cell's signed distance is EXACTLY zero, its RANS wall
        // distance d = 0, phiDamp(0) = 0, and the closure's 1/phi terms give
        // ~1e20.  A strict > 0 leaves that one cell unguarded and NaNs k~/tau~
        // in every cell.  It only showed up with the DEFAULT ibplane -- any
        // explicit value that is not exactly n+1/2 cells misses the degeneracy.
        // The sharp mask F_IBM is a CORNER test (isFluidCell rejects a cell
        // whose corners are inside), so a body THINNER than a cell slips
        // between the four corners and the cell is flagged FLUID while its
        // CENTRE is inside -- an airfoil trailing edge does exactly this.
        // There the level set gives d = 0 for a "fluid" cell, and SA's
        // destruction (nu~/d)^2 = 2e47 OVERFLOWS fp32 on the FIRST step
        // (measured: RAE 2822 TE at x = 12.44, inf -> NaN in nu~, which then
        // spreads).  The centre test is the one that matches what the closure
        // needs -- a wall distance for THIS cell centre -- so require both.
        && (grid.getField(F_IBM)[cIdx] <= (real)0.5
            || grid.getField(F_PHI)[cIdx] >= (real)0);
    if (inBody) {
      MuT[cIdx] = 0;
      TF1[cIdx] = 0;
    }
    // NOTE: do NOT damp mu_t by phi here.  It was tried 2026-08-28 to smooth the
    // interface and is WRONG twice over.  (1) It double-counts the volume
    // weighting: the viscous flux already picks up alpha_f from the scatter
    // (wLc multiplies the COMBINED flux), so the framework's
    // div(alpha_f (mu + mu_t) grad u) becomes alpha_f (mu + phi mu_t) -- phi^2 on
    // the turbulent part.  (2) It corrupts u_tau in the slip wall model: phi = 1/2
    // AT the wall, so mu_t is halved, u_tau comes out ~30% low, and delta_f+,
    // lambda and l_x all inherit that.
    // The smooth approach to zero comes from k~ -> 0 through the band via the
    // (1-phi) penalization, hence mu_t = rho k~ tau~ -> 0 on its own.
    else {

    // Wall distance from the CACHED level set when the wall IS the level set
    // (wallGeom 2); the analytic branch covers the grid-aligned plate.
    real d         = (grid.wallGeom == 2)
                   ? fmax(-grid.getField(F_PHI)[cIdx], (real)0)
                   : grid.wallDistance(pos);
    // A fluid cell lies wholly outside the body, so its centre is at least half a
    // cell from the wall; flooring d there keeps r_d bounded whatever the level
    // set returns.  (Flooring d ITSELF was tried 2026-08-26 and is WRONG -- it
    // makes the airfoil case worse, polluting the mean flow as well.)
    //
    // UNDER BRINKMAN the closure's d IS floored (2026-08-28), and the sharp-path
    // warning above does not transfer: there d also feeds wall-flux geometry,
    // here it is only the damping-function argument.  A staircase/curved wall
    // puts FLUID cells at arbitrarily small d (the plate never goes below
    // 0.5h, which is why its gates never saw this), and the wall-MODELLED
    // routes hold k~ finite at the wall, so the near-wall source identities
    // (~ (u_tau/kappa d)^2 with k~, tau~ NOT going to zero) inject 1/d^2 into
    // k~/tau~ -- measured as the RAE nose exploding 0 -> 1e27 inside one
    // dtEvery window whenever rans AND ibslip are both on, at every nlvls,
    // while rans+noslip (k~ killed at the wall) and laminar+traction are both
    // stable.  Half a LOCAL cell is the physical resolution limit of "distance
    // to the wall" for a cell centre; below that the identities are being fed
    // sub-grid geometry noise.
    const real rd  = (grid.dCutoff > 0)
                   ? fmax(grid.dCutoff/fmax(d, (real)0.5*fmin(hh[0], hh[1])), (real)1)
                   : (real)1;

    if (grid.turbModel == 1) {
      // ---- Spalart-Allmaras (one equation), near-wall modified -------------
      // rho*nu~ rides in the F_RHOK slot; F_RHOTAU is idle.  Reusing the slot
      // keeps the field count, the block sort, the halo exchange and every
      // domain BC working untouched.  Unlike k~-tau~ there is no second
      // variable and no production/dissipation balance to sit neutrally at
      // unity -- see the header comment in SaModel.h.
      const real nut = K[cIdx]/r;                    // primitive nu~
      const sa::Closure cs = sa::closure(r, nut, nu, d, Om, rd);
      MuT[cIdx] = cs.muT;
      // --wmclip (ghost-wall-function mode): near-wall mu_t control in the
      // spirit of Tamaki's SA modification.  Transported SA overshoots the
      // equilibrium mu_t = rho kappa u_tau d in the first rows (measured
      // 165 mu at row 1 vs the ~30 mu design point -> 6x over-drag, thick
      // mixed layer, u(row1) = 0.15).  Clamp to the local log-law
      // equilibrium within wmClip local cells of the wall; u_tau inverted
      // from the cell's own speed and wall distance.
      if (grid.ibWallMode >= 1 && grid.wmClip > (real)0) {
        const real hLoc = fmin(grid.getDx(lvl), grid.getDy(lvl));
        if (d < grid.wmClip*hLoc && d > (real)0) {
          const real uMag = sqrt(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx]
                               + W[cIdx]*W[cIdx]);
          if (uMag > (real)1e-30) {
            const real uTau = ktau::uTauFromWallFunction(uMag, d, nu);
            MuT[cIdx] = fmin(MuT[cIdx], r*(real)ktau::kappa*uTau*d);
          }
        }
      }
      TF1[cIdx] = 0;                                 // no blending function in SA
      if (grid.isInteriorBlock(lvl, ib, jb, kb))
        atomicAdd(&RhsK[cIdx], sa::source(cs, r, nut, d,
                                          grid.saDFloor*fmin(hh[0], hh[1])));
    }
    else {
    const ktau::Closure c =
      ktau::closure(r, K[cIdx], Tau[cIdx], nu, d, S, Om, gkgt,
                    grid.mu, grid.Lref, rd, grid.ransVorticity != 0);
    MuT[cIdx] = c.muT;
    TF1[cIdx] = c.F1;

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {
      real sk, st;
      ktau::sources(c, r, K[cIdx], Tau[cIdx], grid.kInf, grid.tauInf,
                    grid.ransSustain != 0, sk, st);
      atomicAdd(&RhsK[cIdx], sk);
      atomicAdd(&RhsT[cIdx], st);
    }
    }
    }

  END_CELL_LOOP
}

// ---- k~ / tau~ convective + diffusive face flux ----------------------------
//
// At the LOW face of cell (i,j,k) in direction d, matching the fluxL/fluxD/fluxB
// convention: cIdx is the cell, lIdx its low-side neighbour, mdot the mass flux
// already computed by the HLLC solver (so k~ and tau~ ride the SAME mass flux as
// the mean flow, which is what keeps them consistent with it).
//
// The k~ diffusion is an ordinary conservative flux.  The tau~ diffusion is NOT:
// tauDiffFluxes returns a PAIR, one for each side of the face (Appendix A), and
// the two are scattered separately.
//
__device__ inline void ktauFaceFlux(CompressibleSolver &grid,
                                    real *Rho, real *P, real *K, real *Tau, real *TF1,
                                    real *RhsK, real *RhsT,
                                    i32 bIdx, i32 i, i32 j, i32 k, i32 d,
                                    real h, real area, real mdot,
                                    i32 cIdx, i32 lIdx, Vec3 pos,
                                    // Porosity face weights, exactly as the mean flow uses them:
                                    // the volume-filtered turbulence transport is
                                    // d(alpha_f rho k)/dt + div(alpha_f F_k) = alpha_f S_k,
                                    // so the FLUXES carry phibar_f/phi_c and the volumetric
                                    // SOURCES do not (they are already per unit fluid volume).
                                    real wc = 1, real wn = 1)
{
  const i32 di = (d == 0), dj = (d == 1), dk = (d == 2);
  i32 l2Idx = grid.getNbrIdx(bIdx, i-2*di, j-2*dj, k-2*dk);
  i32 r1Idx = grid.getNbrIdx(bIdx, i+di,   j+dj,   k+dk  );
  // Ghost-free IB: collapse any non-fluid tap onto its inboard neighbour, the
  // same rule the mean-flow MUSCL follows (l2R/d2R in computeRightHandSideKernel).
  // This kernel was the ONE unguarded reader of body cells.  Those cells stay
  // frozen at the INITIAL freestream (k~ = kInf, tau~ = tauInf); kInf sits BELOW
  // the near-wall k~ and tauInf ABOVE it, so a raw tap does not merely add error
  // -- it drives the van Leer limiter outside [0,1] and returns the DOWNWIND
  // state, turning the k~/tau~ convection anti-upwind at the first interior face.
  if (grid.immerserdBcType != 0 && grid.ibGhostFree) {
    const Vec3 pl2(pos[0] - (real)2*di*h, pos[1] - (real)2*dj*h, pos[2] - (real)2*dk*h);
    const Vec3 pr1(pos[0] +        di*h,  pos[1] +        dj*h,  pos[2] +        dk*h);
    if (!grid.isFluidCell(pl2, h)) l2Idx = lIdx;
    if (!grid.isFluidCell(pr1, h)) r1Idx = cIdx;
  }

  // van Leer limited MUSCL states, upwind->downwind on each side
  const real kL = grid.tvdRecVanLeer(K[l2Idx],   K[lIdx], K[cIdx]);
  const real kR = grid.tvdRecVanLeer(K[r1Idx],   K[cIdx], K[lIdx]);
  const real tL = grid.tvdRecVanLeer(Tau[l2Idx], Tau[lIdx], Tau[cIdx]);
  const real tR = grid.tvdRecVanLeer(Tau[r1Idx], Tau[cIdx], Tau[lIdx]);

  // convection: upwind on the sign of the mass flux
  const real Fck = (mdot >= 0) ? mdot*kL : mdot*kR;
  const real Fct = (mdot >= 0) ? mdot*tL : mdot*tR;

  // face state for the diffusion coefficients
  const real rL = Rho[lIdx],  rR = Rho[cIdx];
  const real kcL = K[lIdx],   kcR = K[cIdx];
  const real tcL = Tau[lIdx], tcR = Tau[cIdx];
  const real f1L = TF1[lIdx], f1R = TF1[cIdx];
  const real rF = (real)0.5*(rL + rR), kF = (real)0.5*(kcL + kcR);
  const real tF = (real)0.5*(tcL + tcR), f1F = (real)0.5*(f1L + f1R);
  // molecular viscosity at the face, from the mean face temperature (T = p/rho),
  // exactly as viscFaceFlux does it
  const real muF = grid.viscosity((real)0.5*(P[lIdx]/fmax(rL,(real)1e-30)
                                           + P[cIdx]/fmax(rR,(real)1e-30)));
  const real dkdn = (kcR - kcL)/h;
  const real dtdn = (tcR - tcL)/h;

  if (grid.turbModel == 1) {
    // ---- Spalart-Allmaras transport at this face --------------------------
    // One variable (nu~, in the F_RHOK slot), and an ORDINARY conservative
    // diffusion -- none of the Appendix-A non-conservative machinery the tau~
    // equation needs, because SA has no 1/tau^2 factor to split.
    //   flux = mdot*nu~   -   (1/sigma)(nu + nu~) d(nu~)/dn
    // The cb2 term is a CELL source (it is |grad nu~|^2, not a divergence), so
    // it is added to the owning cell rather than scattered as a face flux.
    const real nuF  = muF/fmax(rF, (real)1e-30);
    // CLAMPED at zero: nu~ is a transported quantity that CAN go negative
    // (standard SA has no positivity mechanism -- that is what the SA-neg
    // variant exists for), and the raw value here is a DIFFUSION COEFFICIENT.
    // Once (nu + nu~) < 0 the nu~ equation runs backwards and blows up in a
    // few steps.  Before the destruction floor below this was hidden: the
    // d = 0 overflow drove nu~ to inf/NaN, which every fmax(nut,0) then read
    // as 0, self-quarantining the bug.
    const real nutF = (real)0.5*(fmax(kcL,(real)0) + fmax(kcR,(real)0));
    const real Fd   = (nuF + nutF)*dkdn/(real)sa::sigma;
    const real F    = Fck - rF*Fd;                      // convective - diffusive
    const real cb2t = (real)(sa::cb2/sa::sigma)*rF*dkdn*dkdn;
    if (grid.detFlux) {
      // deterministic path: this thread owns the face -- store its two signed
      // contributions (A -> own cell, B -> neighbour); gatherFaceFluxKernel
      // sums them in fixed order.  Banks 15+4d..18+4d = {A_K, B_K, A_T, B_T}.
      real *FF = grid.ffBuf; const u64 NN = grid.ffN;
      const u64 b0 = (u64)(15 + 4*d)*NN + (u64)cIdx;
      FF[b0]        =  F*area*wc + (real)0.5*cb2t*wc;
      FF[b0 + NN]   = -F*area*wn + (real)0.5*cb2t*wn;
      FF[b0 + 2*NN] = 0;
      FF[b0 + 3*NN] = 0;
      return;
    }
    atomicAdd(&RhsK[cIdx],  F*area*wc);
    atomicAdd(&RhsK[lIdx], -F*area*wn);
    // cb2/sigma * rho |grad nu~|^2, accumulated one direction at a time
    atomicAdd(&RhsK[cIdx], (real)0.5*cb2t*wc);
    atomicAdd(&RhsK[lIdx], (real)0.5*cb2t*wn);
    return;
  }

  // k~ diffusion: conservative, coefficient mu + sigma_k rho k~ tau~   (Eq. 25)
  const real sigKF = f1F*ktau::sigK1 + ((real)1 - f1F)*ktau::sigK2;
  const real Fdk   = (muF + sigKF*rF*kF*tF)*dkdn;

  // tau~ diffusion + cross-diffusion: the Appendix-A non-conservative pair
  const real sigWF = f1F*ktau::sigW1 + ((real)1 - f1F)*ktau::sigW2;
  const real C     = ((real)1 - f1F)*muF + sigWF*rF*kF*tF;              // Eq. (A.3)
  real dCut = grid.dCutoff;
  real fdL, fdR;
  if (dCut > 0) {
    real dwL, dwR, dwF;
    const Vec3 posL(pos[0] - di*h, pos[1] - dj*h, pos[2] - dk*h);
    const Vec3 posF(pos[0] - (real)0.5*di*h, pos[1] - (real)0.5*dj*h,
                    pos[2] - (real)0.5*dk*h);
    if (grid.wallGeom == 2) {
      // Cached level set: phi at the two cell centres, face = their mean
      // (exact for a plane, second-order for a curved body).  The NEIGHBOUR
      // entry is only addressable when that cell exists at this level; at a
      // coarse/fine interface lIdx is the empty block, whose F_PHI is zero --
      // which reads as wall distance ZERO, drives phiDamp to 0, and makes the
      // Eq. (A.9) term -sigW1 (rho k~ tau~ / phi) divide by ~1e-20.  That is a
      // 1e20 in the residual, i.e. instant NaN, and it is why the airfoil blew
      // up while the uniform-grid plate never did.  Fall back to the analytic
      // level set at the neighbour's POSITION, which is always well defined.
      const real *Phi = grid.getField(F_PHI);
      const real phiL = (lIdx < bEmpty*blockSizeTot) ? Phi[lIdx]
                                                     : grid.getBoundaryLevelSet(posL);
      const real phiR = Phi[cIdx];
      dwL = fmax(-phiL, (real)0);
      dwR = fmax(-phiR, (real)0);
      dwF = fmax(-(real)0.5*(phiL + phiR), (real)0);
    } else {
      dwL = grid.wallDistance(posL);
      dwR = grid.wallDistance(pos);
      dwF = grid.wallDistance(posF);
    }
    const real phL = ktau::phiDamp(dwL, dCut);
    const real phR = ktau::phiDamp(dwR, dCut);
    const real phF = ktau::phiDamp(dwF, dCut);
    const real cdL = (real)2*((real)1 - f1L)*ktau::sigW2*rL*tcL;        // Eq. (A.11)
    const real cdR = (real)2*((real)1 - f1R)*ktau::sigW2*rR*tcR;
    ktau::tauDiffFluxes(C, dtdn, tF, kF, phF, tcL, tcR, phL, phR,
                        rL*kcL*tcL, rR*kcR*tcR, cdL, cdR, kcL, kcR, fdL, fdR);
  } else {
    // no wall in this configuration: phi == 1 everywhere, so the A.9 pair is
    // identically zero and A.6 collapses to the plain non-conservative pair.
    const real cdL = (real)2*((real)1 - f1L)*ktau::sigW2*rL*tcL;
    const real cdR = (real)2*((real)1 - f1R)*ktau::sigW2*rR*tcR;
    ktau::tauDiffFluxes(C, dtdn, tF, kF, (real)1, tcL, tcR, (real)1, (real)1,
                        rL*kcL*tcL, rR*kcR*tcR, cdL, cdR, kcL, kcR, fdL, fdR);
  }

  if (grid.detFlux) {
    // see the SA branch: A/B side-stores handle the non-conservative tau~
    // pair (fdL != fdR) exactly -- each side keeps its own value.
    real *FF = grid.ffBuf; const u64 NN = grid.ffN;
    const u64 b0 = (u64)(15 + 4*d)*NN + (u64)cIdx;
    FF[b0]        =  (Fck - Fdk)*area*wc;
    FF[b0 + NN]   = -(Fck - Fdk)*area*wn;
    FF[b0 + 2*NN] = (Fct*area - fdR*area)*wc;
    FF[b0 + 3*NN] = (-Fct*area + fdL*area)*wn;
    return;
  }
  atomicAdd(&RhsK[cIdx],  (Fck - Fdk)*area*wc);
  atomicAdd(&RhsK[lIdx], -(Fck - Fdk)*area*wn);
  // the tau~ diffusion pair is scattered per side: this face is the HIGH face of
  // the left cell (which therefore takes +fdL) and the LOW face of the right
  // cell (which takes -fdR), reproducing (f_L^{i+1/2} - f_R^{i-1/2})/h.
  atomicAdd(&RhsT[cIdx], (Fct*area - fdR*area)*wc);
  atomicAdd(&RhsT[lIdx], (-Fct*area + fdL*area)*wn);
}

// ---- immersed-boundary ghost fill -----------------------------------------
//
// The IB analogue of wallGhostKernel.  Cells inside the body carry no solution,
// but the fluid cells next to them still read them: the closure's S and Omega
// stencils and the tangential derivatives in viscFaceFlux both reach one cell
// into the body.  Left at their initial freestream state they report u = u_inf
// right where the wall model says the profile is a straight line down to the
// slip velocity -- so the first fluid cell sees roughly zero shear and the whole
// near-wall balance is wrong.
//
// So every solid cell within reach of the fluid is given the wall model's own
// continuation: the tangential velocity on the Eq. (36) line evaluated at its
// (negative) wall distance, the normal component mirrored about the surface so
// the surface velocity has none, density and pressure Neumann, k~ constant and
// tau~ linear through the Eq. (39) face value.
//
// Stamp the static-geometry cache: the level set and the corner-test fluid
// mask per cell.  Runs over EVERY allocated block (exterior and ghost blocks
// included -- their cells are read by neighbour lookups), once per adaptation.
// Stamp each cell's own cfl-scaled step for local time stepping.  Reads the
// per-cell limits computeDeltaTKernel just wrote into F_SCRATCH, before the RHS
// reuses that bank.  Cells excluded from the dt reduction (solid, parents) get
// the global step -- they are not evolved anyway, but a zero here would freeze
// any cell that later becomes active.
__global__ void stampLocalDtKernel(CompressibleSolver &grid, real dtGlobal, real dtCap) {
  real *Dtl = grid.getField(F_DTL);
  real *Sc  = grid.getField(F_SCRATCH);
  START_CELL_LOOP
    const real lim = Sc[cIdx]*grid.cfl;
    Dtl[cIdx] = (lim > dtGlobal) ? fmin(lim, dtCap) : dtGlobal;
  END_CELL_LOOP
}

// Zero the LSRK accumulator over ACTIVE cells only.  The old implementation was
// a cudaMemset over NEVOLVE * blockSizeTot * nBlocksMax -- 56 MB of the MAX
// allocation regardless of how many blocks exist, and on managed memory every
// host-side diagnostic read migrates pages CPU-ward, so the next memset paid
// multi-millisecond page migration on top.  Profiled: cudaMemset was 79% of all
// CUDA API time (15 s in a 680-step run, 4.7 ms per call).  A kernel touches
// only the active slots and never migrates anything.
// tiny zeroing kernels: cudaMemset is banned outside initialization (it also
// serializes the stream; these queue like any other kernel)
__global__ void zeroScalesKernel(CompressibleSolver &grid) {
  if (threadIdx.x < 3 && blockIdx.x == 0) grid.globalScale[threadIdx.x] = 0;
}
__global__ void zeroFlagsKernel(CompressibleSolver &grid) {
  i32 i = blockIdx.x*blockDim.x + threadIdx.x;
  while (i < nBlocksMax) { grid.bFlagsList[i] = 0; i += gridDim.x*blockDim.x; }
}

__global__ void zeroTrashBlockKernel(CompressibleSolver &grid, i32 f) {
  if (threadIdx.x < blockSizeTot)
    grid.getField(f)[(u64)bEmpty*blockSizeTot + threadIdx.x] = 0;
}

__global__ void zeroAccumulatorKernel(CompressibleSolver &grid) {
  START_CELL_LOOP
    for (i32 f = 0; f < NEVOLVE; f++)
      grid.getField(F_RHS + f)[cIdx] = 0;
    if (grid.p1)
      for (i32 s = 0; s < 2*P1_NV; s++)
        grid.getField(F_P1SR + s)[cIdx] = 0;
  END_CELL_LOOP
}

// rccmCutGeom (apertures / volume fraction / centroid from the four corner
// level-set values) lives in CompressibleSolver.cuh so the host error norms can
// evaluate the exact solution at the same fluid centroid the kernels use.

__global__ void ibStampGeometryKernel(CompressibleSolver &grid) {
  real *Phi = grid.getField(F_PHI);
  real *Ibm = grid.getField(F_IBM);
  real *Bc  = grid.getField(F_IBBC);
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty) {
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      Phi[cIdx] = grid.getBoundaryLevelSet(pos);
      Ibm[cIdx] = grid.isFluidCell(pos, fmin(grid.getDx(lvl), grid.getDy(lvl)))
                ? (real)1 : (real)0;
      Bc[cIdx]  = (real)grid.getBoundaryBcKind(pos);
      if (grid.ibBrink) {
        // ---- Brinkman face porosities, stamped once (see F_BRINKX) ---------
        // phibar over this cell's LOW-x and LOW-y faces, by the same rule the
        // RHS used to evaluate live: --brinkface 3 splits the face into
        // --brinkseg pieces and integrates the sigmoid in closed form on each
        // (exact for a plane wall, and its error is geometric, not spectral --
        // it does not grow as the band narrows); --brinkface 2 is the single
        // endpoint-to-endpoint average.  delta is the FIXED physical length
        // taken from the finest level, never the local cell size, so a cell's
        // own dx enters only through the face it is integrating over.
        const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
        const real hb = fmin(grid.getDx(grid.nLvls-1),
                             grid.pseudo2D ? grid.getDx(grid.nLvls-1)
                                           : grid.getDy(grid.nLvls-1));
        const real hx = (real)0.5*dx, hy = (real)0.5*dy;
        const real xl = pos[0]-hx, yd = pos[1]-hy;
        grid.getField(F_BRINKX)[cIdx] =
          grid.brinkPhiFaceAvgSeg(Vec3(xl, pos[1]-hy, pos[2]),
                                  Vec3(xl, pos[1]+hy, pos[2]), hb, grid.brinkNSeg);
        grid.getField(F_BRINKY)[cIdx] =
          grid.brinkPhiFaceAvgSeg(Vec3(pos[0]-hx, yd, pos[2]),
                                  Vec3(pos[0]+hx, yd, pos[2]), hb, grid.brinkNSeg);
      }
      // wall normal: stamped for EVERY immersed body, not just the cut cell -- the
      // sharp IB's --ibface stencil wants it too, and for a polyline it is an
      // O(nSegments) closest-point search, i.e. preprocessing, never per stage.
      if (grid.immerserdBcType != 0) {
        Vec3 nw0 = grid.wallNormal(pos, fmin(grid.getDx(lvl), grid.getDy(lvl)));
        grid.getField(F_CUTNX)[cIdx] = nw0[0];
        grid.getField(F_CUTNY)[cIdx] = nw0[1];
      }
      if (grid.ibRccm) {
        const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
        real f[4];
        f[0] = grid.getBoundaryLevelSet(Vec3(pos[0]-(real)0.5*dx, pos[1]-(real)0.5*dy, pos[2]));
        f[1] = grid.getBoundaryLevelSet(Vec3(pos[0]+(real)0.5*dx, pos[1]-(real)0.5*dy, pos[2]));
        f[2] = grid.getBoundaryLevelSet(Vec3(pos[0]+(real)0.5*dx, pos[1]+(real)0.5*dy, pos[2]));
        f[3] = grid.getBoundaryLevelSet(Vec3(pos[0]-(real)0.5*dx, pos[1]+(real)0.5*dy, pos[2]));
        real al, ax, ay, cx = (real)0.5, cy = (real)0.5;
        rccmCutGeom(f, al, ax, ay, &cx, &cy);
        // --cutgeom 2 overwrites alpha / apertures / centroid for CUT cells in a
        // HOST pass right after this kernel (stampCutGeomCurved).  The curved
        // quadrature needs a ~10 KB node arena, which as a per-thread local array
        // would reserve that frame for EVERY thread of this kernel and exhaust
        // local memory at launch -- and it is preprocessing anyway.
        // same convention as rccmCentroidOff: offset in units of h
        {
          const real hR = fmin(dx, dy);
          grid.getField(F_CUTCX)[cIdx] = (cx - (real)0.5)*dx/hR;
          grid.getField(F_CUTCY)[cIdx] = (cy - (real)0.5)*dy/hR;
        }
        // open-face tangential centroids, linear model (overwritten by the
        // host pass under --cutgeom 2, from the same quadrature as the apertures)
        grid.getField(F_CUTTX)[cIdx] = rccmFaceCen(f[0], f[3]);   // low-x face: corners (0,0),(0,1)
        grid.getField(F_CUTTY)[cIdx] = rccmFaceCen(f[0], f[1]);   // low-y face: corners (0,0),(1,0)
        grid.getField(F_CUTA)[cIdx]  = al;
        grid.getField(F_CUTAX)[cIdx] = ax;
        grid.getField(F_CUTAY)[cIdx] = ay;
      }
      // A PRESCRIBED cell holds the boundary state for the whole run: the body
      // is static and the state is time-independent, so stamp it once here and
      // let ibGhostKernel leave these cells alone.  The value comes from the
      // nearest SEGMENT (forward-evaluated at setup), never from inverting at
      // this cell centre -- which lies outside the map and would fail.
      if (Bc[cIdx] > (real)0.5 && grid.ibPolyState && grid.ibPolyBc) {
        real d2min = (real)1e30; i32 eMin = 0;
        for (i32 e = 0; e < grid.ibPolyN; e++) {
          const i32 f = (e + 1 == grid.ibPolyN) ? 0 : e + 1;
          const real ax = grid.ibPoly[2*e], ay = grid.ibPoly[2*e+1];
          const real bx = grid.ibPoly[2*f], by = grid.ibPoly[2*f+1];
          const real ex = bx-ax, ey = by-ay, L2 = ex*ex + ey*ey;
          real t = (L2 > (real)0) ? ((pos[0]-ax)*ex + (pos[1]-ay)*ey)/L2 : (real)0;
          t = fmin(fmax(t,(real)0),(real)1);
          const real qx = pos[0]-(ax+t*ex), qy = pos[1]-(ay+t*ey);
          const real d2 = qx*qx + qy*qy;
          if (d2 < d2min) { d2min = d2; eMin = e; }
        }
        // ibPolyState holds PRIMITIVES (rho,u,v,p); the solver holds the state
        // CONSERVATIVE between steps, so stamp conservative or every prescribed
        // cell reads back a velocity of u/rho.  That was the whole "|V| = 1.37
        // against a physical max of 0.85" wedge: 0.85/0.62 = 1.37 exactly.
        const real rr = grid.ibPolyState[4*eMin+0];
        const real uu = grid.ibPolyState[4*eMin+1];
        const real vv = grid.ibPolyState[4*eMin+2];
        const real pp = grid.ibPolyState[4*eMin+3];
        grid.getField(F_RHO )[cIdx] = rr;
        grid.getField(F_RHOU)[cIdx] = rr*uu;
        grid.getField(F_RHOV)[cIdx] = rr*vv;
        grid.getField(F_RHOW)[cIdx] = (real)0;
        grid.getField(F_RHOE)[cIdx] = pp/(gam-(real)1) + (real)0.5*rr*(uu*uu + vv*vv);
      }
    }
  END_CELL_LOOP
}

// ---- interface-cell prescription (--ibiface, Ali et al. J Eng Math 146:5) --
//
// The paper's actual architecture: the unknowns of the wall treatment are the
// FIRST FLUID CELLS (interface cells -- fluid cells with a non-fluid face
// neighbour), which are excluded from the solve and PRESCRIBED after each
// sweep.  Solid cells are never touched; there is no wall Riemann problem.
// Here the RK update is left alone and the interface cells are simply
// overwritten each stage before the RHS reads them -- the evolved value never
// reaches any flux, which is the same thing without touching the update kernel.
// (The price the paper does not dwell on: conservation is given up in this one
// cell layer.)
//
// Per interface cell C at wall distance d = -phi (phi < 0 in the fluid):
//   foot P = C - d n,  image I = C + d n  (2d off the wall; C is the midpoint
//   of P and I, which is exactly what makes the paper's Dirichlet relation
//   f = (phi_I + phi_B)/2 a linear interpolation).
//   rho, p (and k~, tau~): Neumann -- the value AT I is the value at C, their
//   Eq. (26): f (1 - B21) = sum of fluid terms.  The stencil at I contains
//   interface cells, C itself included; each pass of this kernel is one Jacobi
//   sweep of that implicit system.
//   velocity: slip -- built at the FOOT, normal component dropped.  Mode 1
//   follows the paper: a plane through the non-interface fluid cells of the
//   bilinear window around I (their points 5, 6, 10), evaluated at P; the
//   interface cell is excluded, so velocity is explicit.  Mode 2 samples at P
//   with the implicit triquadratic, which straddles the wall and taps ghosts.
// mode 1 = paper verbatim (bilinear; the fluid-only renormalising fallback in
//          ibSample IS their Sect. 3.4 three-point corner fix)
// mode 2 = implicit triquadratic (3x3, 3x3x3 off the pseudo-2D path) for
//          everything, falling back to bilinear where the window is out of reach
// ---- Ducros-like shock sensor (recon 5) ------------------------------------
// The DG solver's dgAvNuKernel sensor, cell-centred FV form: compression rate
// against acoustic rate,
//     theta = (div u)^2 / ((div u)^2 + K c^2/h^2)    (K = --ksensor, DG --avk)
// Unlike the DG original this fires on BOTH signs of div u -- see below.
// Central differences of the PRIMITIVE velocity; where the stencil is
// unavailable (domain edge, seam gap) theta = 1, i.e. fall back to van Leer.
// Written into F_RHOK, which is free outside RANS (recon 5 + RANS is refused
// at startup) and rides along in field dumps as a bonus diagnostic.
__global__ void shockSensorKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *Th  = grid.getField(F_RHOK);

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
      const i32 cEmpty = bEmpty*blockSizeTot;
      const i32 xm = grid.getNbrIdx(bIdx, i-1, j, k), xp = grid.getNbrIdx(bIdx, i+1, j, k);
      const i32 ym = grid.getNbrIdx(bIdx, i, j-1, k), yp = grid.getNbrIdx(bIdx, i, j+1, k);
      real th = (real)1;                     // no stencil -> stay limited
      bool ok = xm < cEmpty && xp < cEmpty && ym < cEmpty && yp < cEmpty;
      real divu = 0;
      if (ok) divu = (U[xp]-U[xm])/((real)2*dx) + (V[yp]-V[ym])/((real)2*dy);
      if (ok && !(blockSizeZ == 1 || grid.pseudo2D)) {
        const i32 zm = grid.getNbrIdx(bIdx, i, j, k-1), zp = grid.getNbrIdx(bIdx, i, j, k+1);
        if (zm < cEmpty && zp < cEmpty)
          divu += (W[zp]-W[zm])/((real)2*grid.getDz(lvl));
        else ok = false;
      }
      if (ok) {
        const real c2   = gam*fmax(P[cIdx], (real)1e-30)/fmax(Rho[cIdx], (real)1e-30);
        const real h    = fmin(dx, dy);
        // BOTH signs of div u fire the sensor (user call 2026-09-01): the
        // Brinkman blow-up sat on the EXPANSION side of the forming shock
        // foot, where a compression-only sensor is blind by construction.
        // Resolved smooth waves stay below the K c^2/h^2 acoustic floor
        // either way, so the smooth-flow limit is unchanged.
        const real comp = divu*divu;
        th = comp/(comp + grid.kSensor*c2/(h*h) + (real)1e-30);
      }
      Th[cIdx] = th;
    }
  END_CELL_LOOP
}

// Offset of a cell's FLUID centroid from its Cartesian centre, in units of h.
__device__ inline void rccmCentroidOff(CompressibleSolver &grid, Vec3 cp,
                                       real dx, real dy, real h,
                                       real &ox, real &oy)
{
  real f[4];
  f[0] = grid.getBoundaryLevelSet(Vec3(cp[0]-(real)0.5*dx, cp[1]-(real)0.5*dy, cp[2]));
  f[1] = grid.getBoundaryLevelSet(Vec3(cp[0]+(real)0.5*dx, cp[1]-(real)0.5*dy, cp[2]));
  f[2] = grid.getBoundaryLevelSet(Vec3(cp[0]+(real)0.5*dx, cp[1]+(real)0.5*dy, cp[2]));
  f[3] = grid.getBoundaryLevelSet(Vec3(cp[0]-(real)0.5*dx, cp[1]+(real)0.5*dy, cp[2]));
  real al, ax, ay, cx = (real)0.5, cy = (real)0.5;
  rccmCutGeom(f, al, ax, ay, &cx, &cy);
  ox = (cx - (real)0.5)*dx/h;
  oy = (cy - (real)0.5)*dy/h;
}

// Central-difference gradient of one field over LIVE neighbours.
__device__ inline void rccmGrad(CompressibleSolver &grid, const real *F, i32 bIdx,
                                i32 i, i32 j, i32 k, real dx, real dy,
                                real &gx, real &gy)
{
  const i32 cE = bEmpty*blockSizeTot;
  const i32 xm = grid.getNbrIdx(bIdx, i-1, j, k), xp = grid.getNbrIdx(bIdx, i+1, j, k);
  const i32 ym = grid.getNbrIdx(bIdx, i, j-1, k), yp = grid.getNbrIdx(bIdx, i, j+1, k);
  gx = 0; gy = 0;
  if (xm < cE && xp < cE && grid.rccmLive(xm) && grid.rccmLive(xp))
    gx = (F[xp] - F[xm])/((real)2*dx);
  if (ym < cE && yp < cE && grid.rccmLive(ym) && grid.rccmLive(yp))
    gy = (F[yp] - F[ym])/((real)2*dy);
}

// Move a face state from where the uniform-grid MUSCL implicitly evaluated it
// (the cell's Cartesian centre, pushed half a cell along the face normal) to
// where the cut geometry actually needs it: from the cell's FLUID CENTROID to
// the OPEN FACE's centroid.  Both endpoints move on a cut cell, so the shift is
//     D = (x_faceCentroid - x_cellCentroid) - (half-cell normal step)
// and the state is carried along the cell's own gradient.  Without this the
// scheme is first order at every cut face however good the cell values are --
// the values are simply being used at the wrong points.
__device__ inline void rccmShiftFace(CompressibleSolver &grid, real *Fs[4],
                                     i32 bIdx, i32 i, i32 j, i32 k,
                                     real dx, real dy, real h,
                                     real ocx, real ocy, real tFace,
                                     i32 d, Vec5 &q)
{
  // D in physical units: normal part cancels the assumed half-cell step, the
  // tangential part carries the face centroid offset.
  const real Dx = (d == 0) ? -ocx*h : (tFace*dx - ocx*h);
  const real Dy = (d == 0) ? (tFace*dy - ocy*h) : -ocy*h;
  const i32 slot[4] = {0, 1, 2, 4};                 // rho, u, v, p in q[]
  for (i32 f = 0; f < 4; f++) {
    real gx, gy;
    rccmGrad(grid, Fs[f], bIdx, i, j, k, dx, dy, gx, gy);
    q[slot[f]] += gx*Dx + gy*Dy;
  }
  if (!(q[0] > (real)0) || !(q[4] > (real)0)) {     // never let it invert
    q[0] = fmax(q[0], (real)1e-12); q[4] = fmax(q[4], (real)1e-12);
  }
}

// ---- gradient MUSCL for cut cells (--recon 6) -------------------------------
//
// The 1-D limited slopes of tvdRec assume cell values sit at Cartesian centres
// a uniform dx apart.  On a cut cell neither holds: the average lives at the
// FLUID CENTROID and its neighbours' centroids are at arbitrary offsets.  So
// build the gradient the way an unstructured code does --
//   * least squares over the live neighbours, using centroid-to-centroid
//     separations, which absorbs the non-uniform spacing exactly;
//   * a Barth-Jespersen limiter, which is a CELL-based monotonicity condition
//     (the reconstruction may not exceed the neighbour min/max anywhere it is
//     evaluated) rather than a 1-D stencil test -- this is what an unlimited
//     parabola lacks near a cut cell, where the stencil is not smooth;
//   * evaluation from the cell centroid to the OPEN FACE centroid.
// Gradients are recomputed per thread rather than stored: 8 extra fields would
// be ~0.5 GB here, and the kernel is launch-bound rather than flop-bound.
// ---- DOF handles (see CUT_DEAD / cutIsPiece in the header) -------------------
// A handle h >= 0 is a cell (state in the fields), h < 0 a piece (state in
// cutPieceQ, conservative).  Everything the cut path scatters to or taps from
// goes through these, so a piece is a first-class control volume.
__device__ __forceinline__ real *cutRhsPtr(CompressibleSolver &grid, i32 h, i32 n) {
  return cutIsPiece(h) ? &grid.cutPieceS[(size_t)n*grid.cutPieceQCap + cutPieceOf(h)]
                       : &grid.getField(F_RHS + n)[h];
}
__device__ __forceinline__ real cutAlphaOf(CompressibleSolver &grid, i32 h) {
  if (cutIsPiece(h)) return grid.cutPieceAlphaE[cutPieceOf(h)];
  if (!grid.ibRccm || grid.immerserdBcType == 0) return (real)1;        // no cut geometry stamped
  return (grid.cutMerge && grid.cutAlphaE) ? grid.cutAlphaE[h] : grid.getField(F_CUTA)[h];
}

// ---- --leaf: stencil taps across a level jump, through the mortars -----------
// The neighbour (i+di, j+dj) of cell (bIdx, i, j) is missing (coarser across)
// or PARENT (finer across).  Returns 0..2 far-side cell indices: the coarse
// cell, or the two fine cells of the mortar (one, the nearer, for a diagonal).
__device__ inline i32 leafResolve(CompressibleSolver &grid, i32 bIdx, i32 i, i32 j, i32 k,
                                  i32 di, i32 dj, i32 out[2]) {
  const i32 cE = bEmpty*blockSizeTot;
  i32 n = 0;
  auto viaCell = [&](i32 cell, i32 d, i32 pick) -> bool {
    if (cell < 0 || cell >= cE) return false;
    const i32 mi = grid.cellMortar[(size_t)cell*4 + d];
    if (mi < 0) return false;
    const MultiLevelSparseGrid::Mortar &M = grid.mortarList[mi];
    if (M.coarse == cell) {                       // this side is coarse: the fine cells across
      if (pick < 0) { out[n++] = M.fine[0]; out[n++] = M.fine[1]; }
      else out[n++] = M.fine[pick];
    } else out[n++] = M.coarse;                   // this side is fine: the coarse cell across
    return true;
  };
  const i32 self = grid.getNbrIdx(bIdx, i, j, k);
  if (dj == 0) { viaCell(self, di < 0 ? 0 : 1, -1); return n; }
  if (di == 0) { viaCell(self, dj < 0 ? 2 : 3, -1); return n; }
  // diagonal: through the tangential neighbour's x-face, else the other one's y-face
  if (viaCell(grid.getNbrIdx(bIdx, i, j+dj, k), di < 0 ? 0 : 1, dj < 0 ? 1 : 0)) return n;
  viaCell(grid.getNbrIdx(bIdx, i+di, j, k), dj < 0 ? 2 : 3, di < 0 ? 1 : 0);
  return n;
}

// --leaf: a PARENT partner of a boundary cell gets its children's average so the
// boundary fill reads a live value (parents are idle between adaptations)
__device__ inline void leafMaterialise(CompressibleSolver &grid, i32 c) {
  const i32 b = c/blockSizeTot, cc = c%blockSizeTot;
  const i32 ci = cc%blockSize, cj = (cc/blockSize)%blockSize;
  const i32 o = (ci >= blockSize/2 ? 1 : 0) | (cj >= blockSize/2 ? 2 : 0);
  const i32 cb = grid.chldIdxList[(size_t)b*8 + o];
  if (cb == bEmpty) return;
  const i32 fi = 2*(ci % (blockSize/2)), fj = 2*(cj % (blockSize/2));
  const i32 c00 = cb*blockSizeTot + fi + fj*blockSize;
  for (i32 f = 0; f < NEVOLVE; f++) {
    real *q = grid.getField(f);
    q[c] = (q[c00] + q[c00 + 1] + q[c00 + blockSize] + q[c00 + blockSize + 1])*(real)0.25;
  }
}
// primitive f (0 rho, 1 u, 2 v, 3 p) of a handle; fields hold primitives during the RHS
__device__ __forceinline__ real cutTap(CompressibleSolver &grid, real *Fs[4], i32 h, i32 f) {
  if (!cutIsPiece(h)) return Fs[f][h];
  const i32 k = cutPieceOf(h); const size_t cap = grid.cutPieceQCap;
  const real r = fmax(grid.cutPieceQ[k], (real)1e-30);
  const real u = grid.cutPieceQ[cap + k]/r, v = grid.cutPieceQ[2*cap + k]/r, w = grid.cutPieceQ[3*cap + k]/r;
  if (f == 0) return r;
  if (f == 1) return u;
  if (f == 2) return v;
  return (gam - (real)1)*(grid.cutPieceQ[4*cap + k] - (real)0.5*r*(u*u + v*v + w*w));
}
// absolute ELEMENT centroid of a handle, and its cell size
__device__ inline void cutHandlePos(CompressibleSolver &grid, i32 h, real &x, real &y, real &dxo, real &dyo) {
  const i32 c = cutIsPiece(h) ? grid.cutPiece[cutPieceOf(h)].cell : h;
  const i32 ob = c/blockSizeTot, occ = c%blockSizeTot;
  i32 ol, oib, ojb, okb; grid.decode(grid.bLocList[ob], ol, oib, ojb, okb);
  dxo = grid.getDx(ol); dyo = grid.getDy(ol); const real oh = fmin(dxo, dyo);
  Vec3 op = grid.getCellPos(ol, oib, ojb, okb, occ%blockSize, (occ/blockSize)%blockSize, 0);
  if (cutIsPiece(h)) { const CutPiece &P = grid.cutPiece[cutPieceOf(h)]; x = op[0] + P.ecx*oh; y = op[1] + P.ecy*oh; }
  else { x = op[0] + grid.getField(F_CUTCX)[c]*oh; y = op[1] + grid.getField(F_CUTCY)[c]*oh; }
}
// limited least-squares gradient of a PIECE DOF over the DOFs across its open
// faces (face table) and its tip face partner.  Fewer than 3 taps or a poorly
// conditioned stencil -> no gradient (first order locally, never a blow-up).
__device__ inline void cutPieceGrad(CompressibleSolver &grid, real *Fs[4], i32 k, real g[4][2], real lim[4]) {
  const CutPiece &P = grid.cutPiece[k]; const i32 c = P.cell; const i32 h0 = cutHandle(k);
  const i32 sp = grid.cutSplitId[c]; const CutSplitCell &Sc = grid.cutSplitCell[sp];
  const i32 p = k - Sc.first + 1;
  real x0, y0, dx, dy; cutHandlePos(grid, h0, x0, y0, dx, dy); const real h = fmin(dx, dy);
  real q0[4], qmn[4], qmx[4];
  for (i32 f = 0; f < 4; f++) { q0[f] = cutTap(grid, Fs, h0, f); qmn[f] = q0[f]; qmx[f] = q0[f]; }
  real Sxx = 0, Sxy = 0, Syy = 0, Sxf[4] = {0,0,0,0}, Syf[4] = {0,0,0,0}, nx[10], ny[10]; i32 nn = 0;
  auto addTap = [&](i32 th) {
    if (nn >= 10 || th == CUT_DEAD || th == h0) return;
    real tx, ty, tdx, tdy; cutHandlePos(grid, th, tx, ty, tdx, tdy);
    const real rx = tx - x0, ry = ty - y0, w = (real)1/fmax(rx*rx + ry*ry, (real)1e-30);
    Sxx += w*rx*rx; Sxy += w*rx*ry; Syy += w*ry*ry;
    for (i32 f = 0; f < 4; f++) {
      const real qv = cutTap(grid, Fs, th, f), d = qv - q0[f];
      Sxf[f] += w*rx*d; Syf[f] += w*ry*d;
      qmn[f] = fmin(qmn[f], qv); qmx[f] = fmax(qmx[f], qv);
    }
    nx[nn] = rx; ny[nn] = ry; nn++;
  };
  const i32 b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize;
  const i32 cE = bEmpty*blockSizeTot;
  const i32 fo[4] = { c, grid.getNbrIdx(b, i+1, j, 0), c, grid.getNbrIdx(b, i, j+1, 0) };
  const i32 dr[4] = {0, 0, 1, 1}; const bool own[4] = {true, false, true, false};
  for (i32 q = 0; q < 4; q++) {
    const i32 ow = fo[q]; if (ow < 0 || ow >= cE) continue;
    const i32 fid = grid.cutFaceId[ow]; if (fid < 0) continue;
    const CutFace &F = grid.cutFace[fid];
    const i32 nS = dr[q] ? F.nY : F.nX; const CutFaceSeg *sg = dr[q] ? F.sy : F.sx;
    for (i32 s2 = 0; s2 < nS; s2++) {
      if (sg[s2].len <= 0) continue;
      if ((own[q] ? sg[s2].pC : sg[s2].pN) != p) continue;
      addTap(own[q] ? sg[s2].ownN : sg[s2].ownC);
    }
  }
  if (Sc.iLen > (real)0) {
    auto hp = [&](i32 pp) -> i32 { return pp == 0 ? grid.cutOwner[c] : grid.cutPiece[Sc.first + pp - 1].owner; };
    if (Sc.iPa == p) addTap(hp(Sc.iPb)); else if (Sc.iPb == p) addTap(hp(Sc.iPa));
  }
  const real det = Sxx*Syy - Sxy*Sxy;
  for (i32 f = 0; f < 4; f++) {
    if (nn >= 3 && det > (real)1e-6*Sxx*Syy) {
      g[f][0] = ( Syy*Sxf[f] - Sxy*Syf[f])/det;
      g[f][1] = (-Sxy*Sxf[f] + Sxx*Syf[f])/det;
    } else { g[f][0] = 0; g[f][1] = 0; }
    real phi = (grid.gradLim < 0) ? (real)0 : (real)1;
    if (grid.gradLim == 1) {
      for (i32 t = 0; t < nn; t++) {
        const real d = g[f][0]*nx[t] + g[f][1]*ny[t];
        if (d > (real)1e-30)       phi = fmin(phi, (qmx[f] - q0[f])/d);
        else if (d < (real)-1e-30) phi = fmin(phi, (qmn[f] - q0[f])/d);
      }
    } else if (grid.gradLim == 2) {
      const real kh = grid.gradLimK*h, sc = fmax(fabs(q0[f]), (real)1e-30), e2 = kh*kh*kh*sc*sc;
      for (i32 t = 0; t < nn; t++) {
        const real dm = g[f][0]*nx[t] + g[f][1]*ny[t];
        if (fabs(dm) <= (real)1e-30) continue;
        const real dp = (dm > 0) ? (qmx[f] - q0[f]) : (qmn[f] - q0[f]);
        const real ph = ((dp*dp + e2) + (real)2*dm*dp)/(dp*dp + (real)2*dm*dm + dm*dp + e2);
        phi = fmin(phi, ph);
      }
    }
    lim[f] = fmax(phi, (real)0);
  }
}

__device__ unsigned long long g_ibFaceRows = 0;   // --ibface: wall rows actually added

__device__ inline void rccmGradLimited(CompressibleSolver &grid, real *Fs[4],
                                       i32 bIdx, i32 i, i32 j, i32 k,
                                       Vec3 cpos, real dx, real dy, real h,
                                       real ocx, real ocy,
                                       real g[4][2], real lim[4])
{
  const i32 cE = bEmpty*blockSizeTot;
  i32 cIdx0 = grid.getNbrIdx(bIdx, i, j, k);
  if (grid.cutMerge && cIdx0 < cE) {
    // A MEMBER of a merged element reconstructs with its OWNER's stencil.  The
    // element has one state at one centroid, so the members are not stencil
    // rows for each other -- and a sliver left with only its far-side taps has
    // a near-collinear stencil whose off-line gradient the limiter never sees
    // (measured: 2nd order blew up at iter 20 on the cylinder, 1st order
    // converged 14 orders).  The owner's 8-neighbourhood is well conditioned.
    const i32 o = grid.cutOwner[cIdx0];
    if (cutIsPiece(o)) { cutPieceGrad(grid, Fs, cutPieceOf(o), g, lim); return; }   // element owned by a piece
    if (o != cIdx0) {
      bIdx = o/blockSizeTot;
      const i32 cc = o%blockSizeTot;
      i = cc%blockSize; j = (cc/blockSize)%blockSize;
      cIdx0 = o;
      ocx = grid.getField(F_CUTCX)[o]; ocy = grid.getField(F_CUTCY)[o];
      i32 ol, oib, ojb, okb; grid.decode(grid.bLocList[bIdx], ol, oib, ojb, okb);
      cpos = grid.getCellPos(ol, oib, ojb, okb, i, j, k);
    }
  }
  real Sxx = 0, Sxy = 0, Syy = 0;
  real Sxf[4] = {0,0,0,0}, Syf[4] = {0,0,0,0};
  real qmn[4], qmx[4], q0[4];
  for (i32 f = 0; f < 4; f++) { q0[f] = Fs[f][cIdx0]; qmn[f] = q0[f]; qmx[f] = q0[f]; }
  // neighbour offsets and their centroids
  real nx[16], ny[16], dq[16][4]; i32 nn = 0;
  i32 used[16];
  for (i32 dj = -1; dj <= 1; dj++)
    for (i32 di = -1; di <= 1; di++) {
      if (di == 0 && dj == 0) continue;
      const i32 m0 = grid.getNbrIdx(bIdx, i+di, j+dj, k);
      // --leaf: a missing or covered neighbour is resolved through the mortar to
      // the cell(s) across the jump, tapped at their own element centroids
      i32 cand[2] = {m0, -1}; i32 nCand = 1; bool leafAbs = false;
      if (grid.leafFlux && grid.cellMortar && (m0 >= cE || grid.cFlagsList[m0] == PARENT)) {
        nCand = leafResolve(grid, bIdx, i, j, k, di, dj, cand);
        leafAbs = true;
        if (nCand == 0) continue;
      }
      for (i32 qq = 0; qq < nCand; qq++) {
      const i32 m = cand[qq];
      if (m >= cE || m < 0) continue;
      if (nn >= 16) continue;
      if (leafAbs) {                                 // de-duplicate (two offsets can reach one coarse cell)
        bool dup = (m == cIdx0);
        for (i32 u = 0; u < nn && !dup; u++) dup = (used[u] == m);
        if (dup) continue;
        if (grid.getField(F_RHO)[m] <= (real)0) continue;   // an unfilled exterior corner
      }
      if (grid.ibRccm && !grid.rccmLive(m)) {
        if (grid.dbgChecks) atomicAdd(&g_rcDeadGrad, 1ull);
        continue;
      }
      // a member of the SAME merged element carries this very state at this
      // very centroid: zero separation, zero information -- not a stencil row
      if (grid.cutMerge && grid.cutOwner[m] == grid.cutOwner[cIdx0]) continue;
      // --cutsplit: across a thin wall the neighbour CELL's DOF may be the other
      // side's state.  Through the face table, tap the piece of m that faces OUR
      // piece 0 -- its owner's state at its element centroid -- and skip m if no
      // piece of it does.  Diagonal taps next to a split cell are dropped: no
      // face tells which side they are on.
      i32 tap = m; bool tapAbs = false; real tpx = 0, tpy = 0;
      if (grid.cutSplit && grid.cutFaceId && grid.cutSplitId) {
        const i32 spM = grid.cutSplitId[m], sp0 = grid.cutSplitId[cIdx0];
        if (di != 0 && dj != 0) { if (spM >= 0 || sp0 >= 0) continue; }
        else if (spM >= 0 || sp0 >= 0) {
          const bool weOwn = (di < 0 || dj < 0);            // m on our LOW side: our low face
          const i32 ownerCell = weOwn ? cIdx0 : m;
          const i32 fid = grid.cutFaceId[ownerCell];
          const i32 dir = (di != 0) ? 0 : 1;
          const i32 nS = fid < 0 ? 0 : (dir == 0 ? grid.cutFace[fid].nX : grid.cutFace[fid].nY);
          if (nS > 0) {
            const CutFaceSeg *sg = dir == 0 ? grid.cutFace[fid].sx : grid.cutFace[fid].sy;
            i32 found = -1;
            for (i32 s2 = 0; s2 < nS; s2++) {
              if (sg[s2].len <= 0) continue;
              if ((weOwn ? sg[s2].pC : sg[s2].pN) == 0) { found = s2; break; }
            }
            if (found < 0) continue;
            const i32 theirs = weOwn ? sg[found].pN : sg[found].pC;
            if (theirs != 0) {
              const i32 own = weOwn ? sg[found].ownN : sg[found].ownC;   // a DOF handle
              if (own == CUT_DEAD) continue;
              if (grid.cutMerge && own == grid.cutOwner[cIdx0]) continue;
              tap = own; tapAbs = true;
              real tdx, tdy; cutHandlePos(grid, own, tpx, tpy, tdx, tdy);
            }
          }
        }
      }
      real onx = 0, ony = 0;
      if (grid.ibRccm) {
        onx = grid.getField(F_CUTCX)[m];   // PREPROCESSED (see F_CUTCX); 0 on an uncut,
        ony = grid.getField(F_CUTCY)[m];   // unmerged cell, the ELEMENT centroid if merged
      }
      real rx = (real)di*dx + (onx - ocx)*h;
      real ry = (real)dj*dy + (ony - ocy)*h;
      if (leafAbs && !tapAbs) {                    // across a jump: the far cell's own centroid, absolute
        real tdx, tdy; cutHandlePos(grid, tap, tpx, tpy, tdx, tdy); tapAbs = true;
      }
      if (tapAbs) { rx = tpx - (cpos[0] + ocx*h); ry = tpy - (cpos[1] + ocy*h); }
      used[nn] = m;
      const real w  = (real)1/(rx*rx + ry*ry);
      Sxx += w*rx*rx; Sxy += w*rx*ry; Syy += w*ry*ry;
      for (i32 f = 0; f < 4; f++) {
        const real qv = cutTap(grid, Fs, tap, f);
        const real d = qv - q0[f];
        Sxf[f] += w*rx*d; Syf[f] += w*ry*d;
        qmn[f] = fmin(qmn[f], qv);
        qmx[f] = fmax(qmx[f], qv);
        dq[nn][f] = d;
      }
      nx[nn] = rx; ny[nn] = ry;
      nn++;
      }   // candidates
    }
  // ---- --ibface: the IB FACE as a stencil row ------------------------------
  // The wall is a KNOWN state at a KNOWN place: the foot point x_w = x_c - phi n
  // (phi < 0 in the fluid), carrying the slip-wall face state.  Adding it as a
  // least-squares row IS the reconstruction using the face value itself -- no
  // ghost, and no assumption about where the wall sits inside the cell (the
  // mirror ghost asserts it bisects the gap).  rho and p are Neumann there, so
  // only the normal velocity constrains the fit, which is what the wall imposes.
  // nWall counts toward the nn >= 2 rank test below.
  if (grid.ibFaceRec && !grid.ibRccm && grid.immerserdBcType != 0
      && grid.getField(F_IBM)[cIdx0] > (real)0.5) {
    const real dwv = -grid.getField(F_PHI)[cIdx0];          // > 0 in the fluid
    if (dwv > (real)0 && dwv < (real)2*h) {
      const real nxw = grid.getField(F_CUTNX)[cIdx0];
      const real nyw = grid.getField(F_CUTNY)[cIdx0];
      const real rx = dwv*nxw, ry = dwv*nyw;                // x_w - x_c
      const real w  = (real)1/fmax(rx*rx + ry*ry, (real)1e-30);
      const real un = q0[1]*nxw + q0[2]*nyw;                // Fs = {Rho,U,V,P}
      const real dW[4] = {0, -un*nxw, -un*nyw, 0};          // q_wall - q_c
      Sxx += w*rx*rx; Sxy += w*rx*ry; Syy += w*ry*ry;
      for (i32 f = 0; f < 4; f++) {
        Sxf[f] += w*rx*dW[f]; Syf[f] += w*ry*dW[f];
        const real qw = q0[f] + dW[f];
        qmn[f] = fmin(qmn[f], qw); qmx[f] = fmax(qmx[f], qw);
      }
      nn++;
      if (grid.dbgChecks) atomicAdd(&g_ibFaceRows, 1ull);
    }
  }

  const real det = Sxx*Syy - Sxy*Sxy;
  for (i32 f = 0; f < 4; f++) {
    if (nn >= 2 && fabs(det) > (real)1e-30) {
      g[f][0] = ( Syy*Sxf[f] - Sxy*Syf[f])/det;
      g[f][1] = (-Sxy*Sxf[f] + Sxx*Syf[f])/det;
    } else { g[f][0] = 0; g[f][1] = 0; }
    // Limiter: largest scaling that keeps the reconstruction inside
    // [qmn, qmx] at every neighbour position.
    //   1  Barth-Jespersen: hard min().  Non-differentiable, and at a steady
    //      state it CHATTERS -- a cell whose gradient sits on the clip flips
    //      between phi<1 and phi=1 from step to step, so the residual there
    //      never drops below the size of the flip.  Measured on the annulus:
    //      the residual max sat on the first fluid row at the grid-aligned
    //      wall points, 100x the interior, and did not decay.
    //   2  Venkatakrishnan (AIAA 93-0880): the same bound made smooth, with a
    //      threshold eps^2 = (K h)^3 below which variations are not limited at
    //      all -- so a converged smooth field is left alone and the limiter
    //      cannot hold a limit cycle.  K ~ 5 is the usual choice.
    //   0  none (smooth flows only; blows up on a shock).
    //  -1  FIRST ORDER: the gradient is dropped altogether, which is the
    //      piecewise-constant scheme of the paper (their Sect. 2.2, Roe with
    //      cell averages).  Only for reproducing their convergence tables.
    real phi = (grid.gradLim < 0) ? (real)0 : (real)1;
    if (grid.gradLim == 1) {
      for (i32 t = 0; t < nn; t++) {
        const real d = g[f][0]*nx[t] + g[f][1]*ny[t];
        if (d > (real)1e-30)       phi = fmin(phi, (qmx[f] - q0[f])/d);
        else if (d < (real)-1e-30) phi = fmin(phi, (qmn[f] - q0[f])/d);
      }
    } else if (grid.gradLim == 2) {
      const real kh = grid.gradLimK*h;
      // scale the threshold with the field so rho, p and u are limited alike
      const real sc = fmax(fabs(q0[f]), (real)1e-30);
      const real e2 = kh*kh*kh*sc*sc;
      for (i32 t = 0; t < nn; t++) {
        const real dm = g[f][0]*nx[t] + g[f][1]*ny[t];          // Delta_-
        if (fabs(dm) <= (real)1e-30) continue;
        const real dp = (dm > 0) ? (qmx[f] - q0[f]) : (qmn[f] - q0[f]);   // Delta_+
        const real ph = ((dp*dp + e2) + (real)2*dm*dp)/(dp*dp + (real)2*dm*dm + dm*dp + e2);
        phi = fmin(phi, ph);
      }
    }
    lim[f] = fmin(fmax(phi, (real)0), (real)1);
  }
}
__global__ void ibIfaceKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *K   = grid.getField(F_RHOK);
  real *Tau = grid.getField(F_RHOTAU);

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.getField(F_IBM)[cIdx] > (real)0.5) {
      const i32 cEmpty = bEmpty*blockSizeTot;
      // interface cell?  any non-fluid FACE neighbour.  A neighbour in a
      // PRESCRIBED (inflow/outflow) region vetoes the prescription: those
      // ghosts carry the exact state and the ordinary update is correct.
      bool iface = false, veto = false;
      const i32 nd = (blockSizeZ == 1 || grid.pseudo2D) ? 2 : 3;
      for (i32 d = 0; d < nd; d++)
        for (i32 sgn = -1; sgn <= 1; sgn += 2) {
          const i32 m = grid.getNbrIdx(bIdx, i + (d==0)*sgn, j + (d==1)*sgn,
                                       k + (d==2)*sgn);
          if (m >= cEmpty) continue;
          if (grid.getField(F_IBM)[m] <= (real)0.5) {
            iface = true;
            if (grid.getField(F_IBBC)[m] > (real)0.5) veto = true;
          }
        }
      if (iface && !veto) {
        const real h = fmin(grid.getDx(lvl), grid.getDy(lvl));
        Vec3 p = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
        const real phi = grid.getField(F_PHI)[cIdx];    // < 0 here (fluid side)
        const real d0  = -phi;
        if (d0 > (real)0) {
          Vec3 n = grid.wallNormal(p, h);               // points into the fluid
          Vec3 foot(p[0] + phi*n[0], p[1] + phi*n[1], p[2] + phi*n[2]);
          Vec3 ipt (p[0] + d0 *n[0], p[1] + d0 *n[1], p[2] + d0 *n[2]);
          const i32 gi = ib*blockSize+i, gj = jb*blockSize+j, gk = kb*blockSizeZ+k;
          real *Fp[7] = {Rho, U, V, W, P, Tau, K};
          real q[7];
          bool okI = false;
          if (grid.ibIface >= 2)
            okI = ibSampleQuad(grid, ipt, lvl, bIdx, i, j, k, gi, gj, gk, Fp, 7, q, true);
          if (!okI)
            okI = ibSample(grid, ipt, lvl, bIdx, i, j, k, gi, gj, gk, Fp, 7, q,
                           false, grid.ibIface >= 2);
          if (okI) {
            // mode 1 keeps the paper's Neumann rho/p verbatim.  Mode 2 does not:
            // on a curved wall the normal pressure gradient is not zero, it is
            // the centripetal balance dp/ds = rho u_t^2 kappa (s along n, kappa
            // = div n, positive on a convex body) -- and dropping it is the ONE
            // systematic error every smooth-BC variant measured today shares
            // (paper slip 2.7e-2 vs mirror 5.3e-3 on the annulus, all of the
            // gap in rho/p).  Integrate it from I (s = 2d) down to C (s = d)
            // and move rho along the isentrope.
            real pC = q[4], rC = q[0];
            if (grid.ibIface >= 2) {
              const real e2 = (real)0.5*h;
              Vec3 nxp = grid.wallNormal(Vec3(foot[0]+e2, foot[1], foot[2]), h);
              Vec3 nxm = grid.wallNormal(Vec3(foot[0]-e2, foot[1], foot[2]), h);
              Vec3 nyp = grid.wallNormal(Vec3(foot[0], foot[1]+e2, foot[2]), h);
              Vec3 nym = grid.wallNormal(Vec3(foot[0], foot[1]-e2, foot[2]), h);
              real kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
              const real kM = (real)0.5/h;
              kap = fmin(fmax(kap, -kM), kM);
              const real unI = q[1]*n[0] + q[2]*n[1] + q[3]*n[2];
              const real ut2 = fmax(q[1]*q[1] + q[2]*q[2] + q[3]*q[3] - unI*unI,
                                    (real)0);
              real dp = -q[0]*ut2*kap*d0;              // I -> C is -d0 along n
              dp = fmin(fmax(dp, (real)-0.2*q[4]), (real)0.2*q[4]);
              pC = q[4] + dp;
              const real a2 = gam*q[4]/fmax(q[0], (real)1e-30);
              rC = fmax(q[0] + dp/a2, (real)1e-30);    // along the isentrope
            }
            Rho[cIdx] = rC;
            P[cIdx]   = pC;
            Tau[cIdx] = q[5];
            K[cIdx]   = q[6];
            // ---- velocity at the foot --------------------------------------
            real uf[3]; bool okV = false;
            if (grid.ibIface >= 2) {
              real qf[5];
              okV = ibSampleQuad(grid, foot, lvl, bIdx, i, j, k, gi, gj, gk, Fp, 5, qf, true)
                 || ibSample    (grid, foot, lvl, bIdx, i, j, k, gi, gj, gk, Fp, 5, qf,
                                 false, true);
              if (okV) { uf[0] = qf[1]; uf[1] = qf[2]; uf[2] = qf[3]; }
            }
            else if (grid.ibIface == 1) {
              // paper mode: plane through the NON-interface fluid taps of the
              // bilinear window around I, evaluated at the foot
              // (mode 3 skips this fit and projects the I sample instead --
              // the isolation probe for the mode-1 NaN)
              const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
              const real fx = ipt[0]/dx - (real)0.5, fy = ipt[1]/dy - (real)0.5;
              const i32 i0 = (i32)floor((double)fx), j0 = (i32)floor((double)fy);
              const i32 di = i0 - gi, dj = j0 - gj;
              real A[3] = {0,0,0};                      // sum xx, xy, yy
              real bu[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
              real s1 = 0, sx = 0, sy = 0;
              i32 npt = 0;
              for (i32 a = 0; a <= 1; a++)
                for (i32 b = 0; b <= 1; b++) {
                  const i32 m = grid.getNbrIdx(bIdx, i+di+a, j+dj+b, k);
                  if (m >= cEmpty || grid.getField(F_IBM)[m] <= (real)0.5) continue;
                  bool tapIface = false;                // exclude interface taps
                  for (i32 dd = 0; dd < 2 && !tapIface; dd++)
                    for (i32 sg = -1; sg <= 1; sg += 2) {
                      const i32 mn = grid.getNbrIdx(bIdx, i+di+a + (dd==0)*sg,
                                                    j+dj+b + (dd==1)*sg, k);
                      if (mn < cEmpty && grid.getField(F_IBM)[mn] <= (real)0.5)
                        { tapIface = true; break; }
                    }
                  if (tapIface) continue;
                  const real xr = ((real)(i0+a) + (real)0.5)*dx - foot[0];
                  const real yr = ((real)(j0+b) + (real)0.5)*dy - foot[1];
                  s1 += 1; sx += xr; sy += yr;
                  A[0] += xr*xr; A[1] += xr*yr; A[2] += yr*yr;
                  for (i32 f = 0; f < 3; f++) {
                    const real v = Fp[1+f][m];
                    bu[f][0] += v; bu[f][1] += v*xr; bu[f][2] += v*yr;
                  }
                  npt++;
                }
              if (npt >= 3) {
                // normal equations for c0 + cx x + cy y, centred on the foot
                const real M[3][3] = {{s1, sx, sy}, {sx, A[0], A[1]}, {sy, A[1], A[2]}};
                const real det = M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                               - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                               + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
                // det ~ dx^4: three nearly-collinear taps pass an absolute
                // 1e-20 test and the wall extrapolation then explodes (NaN by
                // t=16 on the annulus).  Guard in the fit's own units.
                if (fabs(det) > (real)1e-2*dx*dx*dx*dx) {
                  for (i32 f = 0; f < 3; f++) {
                    // Cramer, first component only (the value at the foot)
                    uf[f] = (bu[f][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                           - M[0][1]*(bu[f][1]*M[2][2]-M[1][2]*bu[f][2])
                           + M[0][2]*(bu[f][1]*M[2][1]-M[1][1]*bu[f][2]))/det;
                  }
                  okV = true;
                }
              }
            }
            if (!okV) { uf[0] = q[1]; uf[1] = q[2]; uf[2] = q[3]; }
            const real un = uf[0]*n[0] + uf[1]*n[1] + uf[2]*n[2];
            U[cIdx] = uf[0] - un*n[0];                  // tangential part only
            V[cIdx] = uf[1] - un*n[1];
            W[cIdx] = uf[2] - un*n[2];
          }
        }
      }
    }
  END_CELL_LOOP
}

__global__ void ibGhostKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *K   = grid.getField(F_RHOK);
  real *Tau = grid.getField(F_RHOTAU);

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty) {
      const real h = fmin(grid.getDx(lvl), grid.getDy(lvl));
      Vec3 p = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const real phi = grid.getField(F_PHI)[cIdx];   // cached level set
      // EVERY non-fluid cell within reach, not just phi > 0.  A cell the surface
      // merely INTERSECTS has phi < 0 -- it is outside the body yet not wholly
      // outside, so it is masked from the update and carries no solution either.
      // Those cells sit directly beneath the first fluid row: leaving them at
      // their freestream initial state puts u = u_inf immediately under the wall
      // and wrecks every near-wall stencil.
      const bool nonFluid = grid.getField(F_IBM)[cIdx] <= (real)0.5;   // cached mask
      // PRESCRIBED boundary (inflow/outflow): the exact state is known here, so
      // the ghost simply carries it.  No mirror, no extrapolation -- and unlike a
      // wall, where only ONE boundary value is known and the ghost has to be
      // built so the reconstructed FACE satisfies it, a region with a known
      // solution gives the stencil exact data on both sides and keeps full
      // scheme order for free.  2.5h covers the halo-2 reconstruction reach.
      // PRESCRIBED cells were stamped with the boundary state and are constant
      // in time -- leave them.  Only WALL ghosts are rebuilt each stage.
      if (nonFluid && phi <= (real)2.5*h &&
          grid.getField(F_IBBC)[cIdx] <= (real)0.5) {
        Vec3 n = grid.wallNormal(p, h);
        Vec3 surf(p[0] + phi*n[0], p[1] + phi*n[1], p[2] + phi*n[2]);
        // NOT every non-fluid cell is inside the body.  The UTCart rule tags two
        // different populations:
        //   phi > 0  genuinely INSIDE  -> the state is a reflection of the fluid
        //   phi < 0  INTERSECTING      -> the CENTRE IS IN THE FLUID; it is
        //                                 masked only because a corner is inside
        // A blanket mirror is wrong for the second group: it reflects a
        // fluid-side point THROUGH the wall and flips the sign of its normal
        // velocity, assigning it a state from the wrong side.  (And even for the
        // first group the old sample offset max(1.5h, 2|phi|) is not the mirror
        // distance 2 phi whenever the 1.5h floor binds, so the reflection was
        // inconsistent there too.)
        // One rule covers both: near a wall the TANGENTIAL velocity is ~constant
        // along the normal and the NORMAL velocity is linear in wall distance,
        // vanishing at the surface.  So carry the tangential component across
        // unchanged and scale the normal component by the ratio of signed wall
        // distances.  d < 0 inside reproduces the mirror EXACTLY when
        // d_ghost = -d_sample, and d > 0 outside keeps the cell on its own side.
        const real dG = -phi;                       // signed: > 0 on the fluid side
        // Sample at a CONSTANT wall distance s* = 2h through THIS cell's foot
        // point.  The old offset max(1.5h, 2|phi|) floored for every ghost with
        // |phi| < 0.75h -- most of the first layer -- so ADJACENT ghosts drew
        // their Neumann rho/p from DIFFERENT standoffs, and that per-cell
        // inconsistency reached the fluid as a jagged ghost pressure through
        // the Riemann solve (the near-wall p streaks, ~0.3% of p_inf).  With a
        // fixed s* every ghost samples the same smooth offset surface, and the
        // only along-wall variation left is the (smooth) foot-point motion.
        // s* = 2h also keeps the whole bilinear stencil in genuinely fluid
        // cells, so the fluid-only renormalisation inside ibSample -- another
        // per-cell jitter source -- almost never triggers.
        // --ibgmirror: the NATURAL mirror -- reflect this ghost across the wall,
        // so the image sits at its own wall distance rather than a fixed 2h
        // standoff.  Compact and second-order-consistent per cell; the price is
        // that the stencil now contains ghosts (handled by the Jacobi sweeps).
        // dG = -phi is NEGATIVE for a ghost (phi > 0 inside the body), so the mirror
        // distance is |phi|, not dG -- taking dG collapsed this to the 0.25h floor,
        // putting the image point on the wall itself and producing NaN.
        const real sStar = grid.ibGMirror ? fmax(fabs(dG), grid.ibGFloor*h) : (real)2*h;
        Vec3 mir(surf[0] + sStar*n[0], surf[1] + sStar*n[1], surf[2] + sStar*n[2]);
        real dS = -grid.getBoundaryLevelSet(mir);   // measured, not assumed = s*
        if (dS < (real)0.5*h) dS = sStar;           // curvature pathology guard
        real *Fp[7] = {Rho, U, V, W, P, Tau, K};
        real q[7];
        bool okS = grid.ibGMirror
          ? ibSampleQuad(grid, mir, lvl, bIdx, i, j, k,
                         ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k, Fp, 7, q,
                         true /* ghosts allowed: the coupled system */)
          : ibSample(grid, mir, lvl, bIdx, i, j, k,
                     ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k, Fp, 7, q);
        // A declined mirror quad must NOT leave the ghost stale: where the
        // solid is thinner than the window (the outer ring near the axes) the
        // stale cell still holds init data and every later sample of it is
        // poison.  The renormalising bilinear always produces something sane.
        if (!okS && grid.ibGMirror)
          okS = ibSample(grid, mir, lvl, bIdx, i, j, k,
                         ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k, Fp, 7, q,
                         false, true);
        if (okS) {
          Rho[cIdx] = q[0];                     // Neumann density and pressure
          P[cIdx]   = q[4];
          // --ibho: the flat-wall Neumann rho/p above is O(h) wrong on a
          // curved wall (the true dp/dn = -rho u_t^2 kappa is O(1) on the
          // vortex), and these ghosts feed the MUSCL stencils of the first
          // fluid faces -- an O(h) ghost caps the interior reconstruction at
          // first order no matter what the wall flux does.  Carry H and S
          // across instead (both constant along the normal for the smooth
          // slip-wall solution) and ramp u_t with the curvature; rho/p are
          // recovered from (H, S, |u|) after the velocity is built below.
          const real vnM = q[1]*n[0] + q[2]*n[1] + q[3]*n[2];
          {
            // ARCHITECTURE (fixed): ghost cells are ALWAYS a plain slip wall.
            // Tangential component carried across unchanged, normal component
            // linear in the signed wall distance (zero AT the wall), rho/p and
            // k~/tau~ Neumann.  NO wall model here -- no u_tau, no Eq. 36
            // linearisation, no Eq. 39 -- ibWallFlux owns all of that, and it
            // reconstructs the FACE FLUX state only.  Keeping the model out of
            // the ghosts is what removes the second, drifting wall-model solve
            // AND the Eq. 39 singularity (tau~ = 0 stored at d = 0).
            const real sc = (fabs(dS) > (real)1e-30) ? dG/dS : (real)0;
            const real un = vnM*sc;
            U[cIdx] = (q[1] - vnM*n[0]) + un*n[0];
            V[cIdx] = (q[2] - vnM*n[1]) + un*n[1];
            W[cIdx] = (q[3] - vnM*n[2]) + un*n[2];
            if (grid.ibHo) {
              // curvature-consistent ghost (see the comment above): ramp u_t
              // by the linearised du_t/ds = -kappa u_t relation, then recover
              // rho/p from H and S carried across from the sample.
              real kap = 0;
              if (grid.ibCurv) {
                const real e2 = (real)0.5*h;
                Vec3 nxp = grid.wallNormal(Vec3(surf[0]+e2, surf[1], surf[2]), h);
                Vec3 nxm = grid.wallNormal(Vec3(surf[0]-e2, surf[1], surf[2]), h);
                Vec3 nyp = grid.wallNormal(Vec3(surf[0], surf[1]+e2, surf[2]), h);
                Vec3 nym = grid.wallNormal(Vec3(surf[0], surf[1]-e2, surf[2]), h);
                kap = (nxp[0]-nxm[0] + nyp[1]-nym[1])/((real)2*e2);
                const real kM = (real)0.5/h;
                kap = fmin(fmax(kap, -kM), kM);
              }
              const real rSmp = fmax(q[0], (real)1e-30);
              const real pSmp = fmax(q[4], (real)1e-30);
              const real tqx = q[1] - vnM*n[0], tqy = q[2] - vnM*n[1], tqz = q[3] - vnM*n[2];
              const real utS = sqrt(tqx*tqx + tqy*tqy + tqz*tqz);
              const real denS = (real)1 - kap*dS;
              const real ramp = (fabs(denS) > (real)0.2) ? ((real)1 - kap*dG)/denS : (real)1;
              U[cIdx] = tqx*ramp + un*n[0];
              V[cIdx] = tqy*ramp + un*n[1];
              W[cIdx] = tqz*ramp + un*n[2];
              const real Hs = gam*pSmp/((gam-(real)1)*rSmp)
                            + (real)0.5*(utS*utS + vnM*vnM);
              const real Ss = pSmp/pow(rSmp, gam);
              const real u2g = U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx] + W[cIdx]*W[cIdx];
              const real a2g = fmax((gam-(real)1)*(Hs - (real)0.5*u2g), (real)1e-6);
              const real rG  = pow(a2g/(gam*Ss), (real)1/(gam-(real)1));
              Rho[cIdx] = rG;
              P[cIdx]   = Ss*pow(rG, gam);
            }
            // ---- ghost wall function (--ibwm 1) ---------------------------
            // Yang, Song & Zhu, Processes 12 (2024) 1182, Sec. 2.3, Eqs. (6)-(8),
            // itself following Tamaki's near-wall SA modification.  The slip
            // mirror above carries u_t across the wall UNCHANGED, i.e. zero wall
            // shear.  Here the ghost instead carries the log-law profile:
            // Newton-solve u_tau from the mirror-point tangential speed at its
            // wall distance (Eq. 6, our uTauFromWallFunction), evaluate U+ at the
            // GHOST's own y+ (Eq. 8), and take it ANTISYMMETRICALLY through the
            // surface so u_t -> 0 at the wall.  The ORDINARY viscous flux then
            // reads a real near-wall gradient and returns tau_w -- no boundary
            // flux is prescribed anywhere, which is what lets this architecture
            // run with explicit time integration.
            // MODE 1 -- the paper's actual scheme.  Eq. (6) solves u_tau so
            // that u_tau U+(Y+_MP) = u_MP,t EXACTLY; Eq. (8) then assigns that
            // same value to the ghost.  At Newton convergence the "revision" is
            // therefore the tangential velocity carried across UNCHANGED -- the
            // slip mirror already built above.  No further ghost work: the wall
            // face transmits pressure only, and the drag enters through mu_t in
            // the first interior rows (mu_t = rho kappa u_tau d there gives a
            // resolved stress of exactly rho u_tau^2).  u_tau itself appears
            // nowhere in the field equations -- only in the C_f diagnostic.
            // MODE 2 -- an antisymmetric log-law ghost (u_t(-d) = -u_t(d)).
            // NOT the paper: it puts an O(u_inf) velocity jump across the wall
            // face, which the ordinary Riemann flux sees as a strong shear and
            // destabilises (measured: non-finite within t = 3).
            // --wmghost 0: keep the ghosts PLAIN SLIP (the recorded
            // architecture -- the wall model reconstructs the FACE FLUX state
            // only).  Putting the log law into the ghosts too makes the model
            // act twice: once through the prescribed tau_w at the face and
            // again through the near-wall gradient the ghosts impose, which
            // over-thickens the layer on a curved body (RAE: decambered,
            // Cl 0.410 vs the inviscid 0.768).
            if (grid.rans && grid.ibWallMode >= 1 && grid.wmGhost) {
              atomicAdd(&g_wmCand, 1ULL);
              const real tqx = q[1] - vnM*n[0], tqy = q[2] - vnM*n[1],
                         tqz = q[3] - vnM*n[2];
              const real utM = sqrt(tqx*tqx + tqy*tqy + tqz*tqz);
              const real rM  = fmax(q[0], (real)1e-30);
              if (utM > (real)1e-30 && q[4] > (real)0 && dS > (real)0
                  && surf[0] >= grid.plateX0 && surf[0] <= grid.wmX1) {
                const real nuw  = grid.viscosity(q[4]/rM)/rM;
                const real uTau = ktau::uTauFromWallFunction(utM, dS, nuw);
                const real ypG  = fabs(dG)*uTau/fmax(nuw,(real)1e-30);
                const real mag  = uTau*ktau::uPlus(ypG);
                // Mode 1 (the working reading of Eq. 8): the ghost takes the
                // log-law value at its own mirrored distance, POSITIVE sign --
                // below the MP on the log curve, so the first fluid row sees the
                // log GRADIENT (which feeds SA production and hence mu_t and the
                // drag) without the O(u_inf) sign-flip jump of mode 2.
                //   pure slip ghost:  stable but Cf ~40% low and decaying
                //                     (zero resolved shear starves production);
                //   antisymmetric:    non-finite by t = 3 (the jump);
                //   positive log-law: tested below.
                const real f    = ((grid.ibWallMode != 2 || dG >= (real)0)
                                   ? mag : -mag)/utM;
                U[cIdx] = tqx*f + un*n[0];
                V[cIdx] = tqy*f + un*n[1];
                W[cIdx] = tqz*f + un*n[2];
                // SA: the Neumann nu~ mirror is what makes the layer
                // RELAMINARIZE.  It zeroes the nu~ gradient at the wall face,
                // so no diffusion feeds the first cell while the destruction
                // cw1 fw (nu~/d)^2 -- largest exactly there -- drains it, and
                // C_f decays from the wall outward (measured: in-band at t~1,
                // then 0.0027 -> 0.0006 by t=4).  The wall condition is
                // nu~ = 0 AT the surface, i.e. Eq. (1) Dirichlet with
                // nu~_BP = 0: the ghost carries the SIGNED linear extension
                // kappa u_tau d (negative inside), which both restores the
                // diffusive feed and interpolates to zero at the wall.
                if (grid.turbModel == 1)
                  K[cIdx] = ktau::kappa*uTau*dG;
                atomicAdd(&g_wmGhost, 1ULL);
              }
            }
            Tau[cIdx] = q[5];
            if (!(grid.rans && grid.ibWallMode >= 1 && grid.turbModel == 1))
            K[cIdx]   = q[6];
          }
        }
      }
    }
  END_CELL_LOOP
}

// ---- wall-modeled ghost fill ----------------------------------------------
//
// The generic boundary condition mirrors the velocity (no-slip), which is wrong
// under a wall model: below the image point the profile is a straight line with
// slope du/dy|_IP, not one that vanishes at the wall.  Left alone, the first
// interior cell would see |du/dy| ~ 2 u_1/dy instead of the model's slope and
// over-produce k~ badly.
//
// So after the generic BC has run, overwrite the wall ghost rows with the LINEAR
// continuation of the wall-model profile.  Only the normal velocity is still
// mirrored, so the face normal velocity stays zero.  Runs over first-row
// interior cells and writes downward, which keeps the two-cell halo reachable
// (the reverse -- iterating ghosts and reaching up to the image point -- would
// not be).
//
__global__ void wallGhostKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *K   = grid.getField(F_RHOK);
  real *Tau = grid.getField(F_RHOTAU);

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      const real dy = grid.getDy(lvl);
      Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      const real yFace = pos[1] - (real)0.5*dy;
      if (fabs(yFace) < (real)1e-6*dy && pos[0] >= grid.plateX0) {
        const i32 c1 = cIdx;
        const i32 cEmpty = bEmpty*blockSizeTot;
        if (grid.ghostSlip) {
          // Slip mirror about the wall face: ghost row g mirrors interior row
          // g, tangential velocity unchanged and normal reversed, everything
          // else Neumann.  No wall model here -- wallModelFluxY owns the wall,
          // as the face boundary flux.  (No `continue` here: START_CELL_LOOP is
          // a while loop whose stride increment lives in END_CELL_LOOP, so a
          // continue would skip it and hang the kernel.)
          for (i32 g = 1; g <= haloSize; g++) {
            const i32 gIdx = grid.getNbrIdx(bIdx, i, j-g, k);
            const i32 mIdx = grid.getNbrIdx(bIdx, i, j+g-1, k);
            if (gIdx >= cEmpty || mIdx >= cEmpty) continue;   // inner for: safe
            Rho[gIdx] = Rho[mIdx];
            P[gIdx]   = P[mIdx];
            U[gIdx]   = U[mIdx];
            W[gIdx]   = W[mIdx];
            V[gIdx]   = -V[mIdx];
            K[gIdx]   = K[mIdx];
            Tau[gIdx] = Tau[mIdx];
          }
        }
        else {
        const WallState w =
          wallModelStateY(grid, Rho, U, V, W, P, Tau, bIdx, i, j, k, dy);
        for (i32 g = 1; g <= haloSize; g++) {          // both ghost rows
          const i32 gIdx = grid.getNbrIdx(bIdx, i, j-g, k);
          if (gIdx >= cEmpty) continue;
          Rho[gIdx] = Rho[c1];                         // Neumann rho and p
          P[gIdx]   = P[c1];
          // Tangential velocity on the wall-model straight line: u = u_FC at the
          // face and slope du/dy, so the ghost g cells below the first cell sits
          // at u_FC + du/dy (dy/2 - g dy).  (g = 1 therefore averages with the
          // first cell to exactly u_FC.)
          const real ut = w.uFc + w.dudy*((real)0.5*dy - (real)g*dy);
          U[gIdx] = ut*w.tx;
          W[gIdx] = ut*w.tz;
          // normal component mirrored about the face -> zero normal velocity there
          V[gIdx] = (g == 1) ? -V[c1] : -V[grid.getNbrIdx(bIdx, i, j+1, k)];
          // k~ is constant in the wall-normal direction (Eq. 16), i.e. Neumann --
          // the Eq. (39) value k_FC is a DIRICHLET datum for the boundary FLUX,
          // not for the gradient stencil.
          K[gIdx] = K[c1];
          // tau~ is linear in d through (d_FC, tau~_FC) and (d_FC + dy/2, tau~_1):
          // tau~_g = tau~_1 - 2 g (tau~_1 - tau~_FC).
          Tau[gIdx] = fmax(Tau[c1] - (real)(2*g)*(Tau[c1] - w.tFc), (real)0);
        }
        }
      }
    }
  END_CELL_LOOP
}

// Deterministic face-flux gather (--detflux): computeRightHandSideKernel above
// stored every thread's west/south(/back) face flux, area factor included, in
// grid.ffBuf.  Sum each interior cell's own two (three) faces plus its east/
// north(/front) neighbours' stored faces in one fixed expression -- no atomics,
// so the accumulation order (and hence the roundoff) is identical every run.
// A neighbour index at or past the empty block means no same-level neighbour
// exists there; the atomic path never received that face either (level seams
// are reconciled by restriction), so contribute exactly nothing.
__global__ void stateHashKernel(CompressibleSolver &grid) {
  START_LIVE_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {   // no continue: see below
      for (i32 f = 0; f < 5; f++) {
        atomicXor((unsigned int*)&grid.dbgCnt[40+f],
                  __float_as_uint((float)grid.getField(f)[cIdx]));
        atomicXor((unsigned int*)&grid.dbgCnt[45+f],
                  __float_as_uint((float)grid.getField(F_RHS+f)[cIdx]));
      }
    }
  END_CELL_LOOP
}

__global__ void gatherFaceFluxKernel(CompressibleSolver &grid) {
  real *Rhs[5] = {grid.getField(F_RHS + 0), grid.getField(F_RHS + 1),
                  grid.getField(F_RHS + 2), grid.getField(F_RHS + 3),
                  grid.getField(F_RHS + 4)};
  const real *FF = grid.ffBuf;
  const u64 NN = grid.ffN;
  const i32 cE = bEmpty*blockSizeTot;
  START_LIVE_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    // NO continue in START_CELL_LOOP (while-loop; increment lives in
    // END_CELL_LOOP -- a continue skips it and hangs the kernel)
    if (loc != kEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      const i32 r1 = grid.getNbrIdx(bIdx, i+1, j, k);
      const i32 u1 = grid.getNbrIdx(bIdx, i, j+1, k);
      const i32 f1 = grid.pseudo2D ? cE : grid.getNbrIdx(bIdx, i, j, k+1);
      for (i32 n = 0; n < 5; n++) {
        real acc = FF[(u64)n*NN + (u64)cIdx] + FF[(u64)(5+n)*NN + (u64)cIdx];
        if (r1 < cE && (!grid.leafFlux || grid.cFlagsList[r1] != PARENT)) acc -= FF[(u64)n*NN + (u64)r1];
        if (u1 < cE && (!grid.leafFlux || grid.cFlagsList[u1] != PARENT)) acc -= FF[(u64)(5+n)*NN + (u64)u1];
        if (!grid.pseudo2D) {
          acc += FF[(u64)(10+n)*NN + (u64)cIdx];
          if (f1 < cE) acc -= FF[(u64)(10+n)*NN + (u64)f1];
        }
        Rhs[n][cIdx] += acc;
      }
      if (grid.rans) {
        // turbulence banks: A slots are this cell's own west/south(/back)
        // faces; B slots live on the east/north(/front) neighbour.  Signs are
        // already folded into the stored values (the tau~ pair is
        // non-conservative, so each side carries its own value).
        real *RhsK = grid.getField(F_RHS + F_RHOK);
        real *RhsT = grid.getField(F_RHS + F_RHOTAU);
        real aK = FF[(u64)15*NN + (u64)cIdx] + FF[(u64)19*NN + (u64)cIdx];
        real aT = FF[(u64)17*NN + (u64)cIdx] + FF[(u64)21*NN + (u64)cIdx];
        if (r1 < cE) { aK += FF[(u64)16*NN + (u64)r1]; aT += FF[(u64)18*NN + (u64)r1]; }
        if (u1 < cE) { aK += FF[(u64)20*NN + (u64)u1]; aT += FF[(u64)22*NN + (u64)u1]; }
        if (!grid.pseudo2D) {
          aK += FF[(u64)23*NN + (u64)cIdx]; aT += FF[(u64)25*NN + (u64)cIdx];
          if (f1 < cE) { aK += FF[(u64)24*NN + (u64)f1]; aT += FF[(u64)26*NN + (u64)f1]; }
        }
        RhsK[cIdx] += aK;
        RhsT[cIdx] += aT;
      }
    }
  END_CELL_LOOP
}

__global__ void computeRightHandSideKernel(CompressibleSolver &grid) {
  // reads primitive variables (Rho,U,V,W,P) in fields 0..4
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);

  real *RhsRho  = grid.getField(F_RHS + 0);
  real *RhsRhoU = grid.getField(F_RHS + 1);
  real *RhsRhoV = grid.getField(F_RHS + 2);
  real *RhsRhoW = grid.getField(F_RHS + 3);
  real *RhsRhoE = grid.getField(F_RHS + 4);

  START_LIVE_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    real dx = grid.getDx(lvl);
    real dy = grid.getDy(lvl);
    real dz = grid.getDz(lvl);
    real vol = dx*dy*dz;

    // neighbor cell memory indices for the 3 upwind faces (left/down/back)
    i32 l1Idx = grid.getNbrIdx(bIdx, i-1, j, k);
    i32 l2Idx = grid.getNbrIdx(bIdx, i-2, j, k);
    i32 r1Idx = grid.getNbrIdx(bIdx, i+1, j, k);

    i32 d1Idx = grid.getNbrIdx(bIdx, i, j-1, k);
    i32 d2Idx = grid.getNbrIdx(bIdx, i, j-2, k);
    i32 u1Idx = grid.getNbrIdx(bIdx, i, j+1, k);

    i32 b1Idx = grid.getNbrIdx(bIdx, i, j, k-1);
    i32 b2Idx = grid.getNbrIdx(bIdx, i, j, k-2);
    i32 f1Idx = grid.getNbrIdx(bIdx, i, j, k+1);

    Vec3 cpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);

    // Ghost-free IB: the MUSCL stencil reaches TWO cells, so at the first
    // interior face it would read across the wall.  Collapse any non-fluid tap
    // onto its inboard neighbour -- tvdRec with a repeated tap returns the
    // upwind value, so the reconstruction degrades to first order next to the
    // body instead of interpolating through it.  Only the RECONSTRUCTION indices
    // are remapped; the scatter still targets the real neighbours.
    i32 l2R = l2Idx, l1R = l1Idx, r1R = r1Idx;
    i32 d2R = d2Idx, d1R = d1Idx, u1R = u1Idx;
    i32 b2R = b2Idx, b1R = b1Idx, f1R = f1Idx;
    // Under RCCM the collapse is MANDATORY, not an --ibgf option: dead cells
    // (alpha = 0) are never advanced AND never ghost-filled -- the R-Cell
    // reconstruction replaced applyWallGhosts -- so they hold their INITIAL
    // state for the whole run.  A 1-D slope that reads one is differencing
    // against frozen data forever.  Measured with the gradient path (recon 6),
    // which excludes them properly: 1.8e9 dead taps encountered per run.
    if (grid.immerserdBcType != 0 && grid.ibRccm) {
      #define RCCM_DEAD(IDX) ((IDX) < bEmpty*blockSizeTot && !grid.rccmLive(IDX))
      if (RCCM_DEAD(l1Idx)) l1R = cIdx;
      if (RCCM_DEAD(l2Idx)) l2R = l1R;
      if (RCCM_DEAD(r1Idx)) r1R = cIdx;
      if (RCCM_DEAD(d1Idx)) d1R = cIdx;
      if (RCCM_DEAD(d2Idx)) d2R = d1R;
      if (RCCM_DEAD(u1Idx)) u1R = cIdx;
      #undef RCCM_DEAD
    }
    else if (grid.immerserdBcType != 0 && grid.ibGhostFree) {
      const real hm = fmin(dx, dy);
      if (!grid.isFluidCell(Vec3(cpos[0]-dx,   cpos[1], cpos[2]), hm)) l1R = cIdx;
      if (!grid.isFluidCell(Vec3(cpos[0]-2*dx, cpos[1], cpos[2]), hm)) l2R = l1R;
      if (!grid.isFluidCell(Vec3(cpos[0]+dx,   cpos[1], cpos[2]), hm)) r1R = cIdx;
      if (!grid.isFluidCell(Vec3(cpos[0], cpos[1]-dy,   cpos[2]), hm)) d1R = cIdx;
      if (!grid.isFluidCell(Vec3(cpos[0], cpos[1]-2*dy, cpos[2]), hm)) d2R = d1R;
      if (!grid.isFluidCell(Vec3(cpos[0], cpos[1]+dy,   cpos[2]), hm)) u1R = cIdx;
      if (!grid.pseudo2D) {
        if (!grid.isFluidCell(Vec3(cpos[0], cpos[1], cpos[2]-dz),   hm)) b1R = cIdx;
        if (!grid.isFluidCell(Vec3(cpos[0], cpos[1], cpos[2]-2*dz), hm)) b2R = b1R;
        if (!grid.isFluidCell(Vec3(cpos[0], cpos[1], cpos[2]+dz),   hm)) f1R = cIdx;
      }
    }

    Vec5 qL, qR, qD, qU, qB, qF;

    // TVD reconstructed primitive states on each face
    // recon 5: per-face sensor = max of the two adjacent cells' theta.  qL/qR
    // are the two states of THIS cell's minus-x face (l1|c), so both take the
    // same face theta; likewise qD/qU (minus-y) and qB/qF (minus-z).
    real thX = 1, thY = 1, thZ = 1;
    // --reconfar: outside the two-cell band at the body the ordinary 1-D scheme
    // does the reconstruction; inside it (and everywhere when off) it is recon.
    i32 rcCell = (grid.recon == 6 && grid.reconFar >= 0 && grid.cutNear && !grid.cutNear[cIdx])
               ? grid.reconFar : grid.recon;
    // --leaf: a cell whose stencil crosses a level jump reconstructs with the
    // least-squares gradient (its 1-D taps would read the missing neighbour)
    if (grid.leafFlux && grid.cutNear && grid.cutNear[cIdx]) rcCell = 6;
    if (rcCell == 5) {
      const real *Th = grid.getField(F_RHOK);
      thX = fmax(Th[cIdx], Th[l1R]);
      thY = fmax(Th[cIdx], Th[d1R]);
      thZ = (blockSizeZ == 1 || grid.pseudo2D) ? thX : fmax(Th[cIdx], Th[b1R]);
    }
    // ---- --ibface: MUSCL stencil from the IB FACE VALUE (sharp IB) ----------
    // Replaces the mirror GHOST on a wall-adjacent side.  Slip wall: the face
    // state is (rho_c, u_c - (u_c.n) n, p_c); extrapolating it linearly out to
    // where the stencil wants a neighbour (a distance dx away) gives
    //     u_nbr = u_c - s (u_c.n) n,   s = dx / d_wall,
    // d_wall being the distance to the wall ALONG that direction, -phi/|n.e|.
    // The mirror ghost is exactly s = 2, i.e. it ASSUMES the wall bisects the
    // cell; with a level set it does not, and s = 2 is then the wrong slope.
    // rho and p are Neumann at a slip wall, so they keep the cell value.
    real vL1[5] = {Rho[l1R], U[l1R], V[l1R], W[l1R], P[l1R]};
    real vR1[5] = {Rho[r1R], U[r1R], V[r1R], W[r1R], P[r1R]};
    real vD1[5] = {Rho[d1R], U[d1R], V[d1R], W[d1R], P[d1R]};
    real vU1[5] = {Rho[u1R], U[u1R], V[u1R], W[u1R], P[u1R]};
    if (grid.ibFaceRec && grid.immerserdBcType != 0 && !grid.ibRccm) {
      const real hmF   = fmin(dx, dy);
      const real *IbmF = grid.getField(F_IBM);
      const i32   cEf  = bEmpty*blockSizeTot;
      const real  nxw  = grid.getField(F_CUTNX)[cIdx];
      const real  nyw  = grid.getField(F_CUTNY)[cIdx];
      const real  dwn  = fmax(-grid.getField(F_PHI)[cIdx], (real)0.02*hmF);  // normal distance
      const real  unC  = U[cIdx]*nxw + V[cIdx]*nyw;
      // only for a cell that is itself fluid; a solid cell has no face state
      if (IbmF[cIdx] > (real)0.5) {
        #define IBF_SOLID(IDX, PX, PY, HH) \
          (((IDX) < cEf) ? (IbmF[IDX] <= (real)0.5) \
                         : !grid.isFluidCell(Vec3((PX),(PY),cpos[2]), (HH)))
        #define IBF_FILL(VARR, HDIR, EDOT)                                        \
          {                                                                       \
            const real dAl = fmin(HDIR, dwn/fmax(fabs(EDOT), (real)0.3));         \
            const real sF  = fmin(fmax(HDIR/fmax(dAl,(real)1e-30),(real)1),(real)4); \
            VARR[0] = Rho[cIdx];  VARR[4] = P[cIdx];                              \
            VARR[1] = U[cIdx] - sF*unC*nxw;                                       \
            VARR[2] = V[cIdx] - sF*unC*nyw;                                       \
            VARR[3] = W[cIdx];                                                    \
          }
        if (IBF_SOLID(l1Idx, cpos[0]-dx, cpos[1], dx)) IBF_FILL(vL1, dx, nxw)
        if (IBF_SOLID(r1Idx, cpos[0]+dx, cpos[1], dx)) IBF_FILL(vR1, dx, nxw)
        if (IBF_SOLID(d1Idx, cpos[0], cpos[1]-dy, dy)) IBF_FILL(vD1, dy, nyw)
        if (IBF_SOLID(u1Idx, cpos[0], cpos[1]+dy, dy)) IBF_FILL(vU1, dy, nyw)
        #undef IBF_SOLID
        #undef IBF_FILL
      }
    }
    qL[0] = grid.tvdRec(Rho[l2R], vL1[0], Rho[cIdx], thX, rcCell);
    qR[0] = grid.tvdRec(vR1[0], Rho[cIdx],  vL1[0], thX, rcCell);
    qD[0] = grid.tvdRec(Rho[d2R], vD1[0], Rho[cIdx], thY, rcCell);
    qU[0] = grid.tvdRec(vU1[0], Rho[cIdx],  vD1[0], thY, rcCell);
    qB[0] = grid.tvdRec(Rho[b2R], Rho[b1R], Rho[cIdx], thZ, rcCell);
    qF[0] = grid.tvdRec(Rho[f1R], Rho[cIdx],  Rho[b1R], thZ, rcCell);

    qL[1] = grid.tvdRec(U[l2R], vL1[1], U[cIdx], thX, rcCell);
    qR[1] = grid.tvdRec(vR1[1], U[cIdx],  vL1[1], thX, rcCell);
    qD[1] = grid.tvdRec(U[d2R], vD1[1], U[cIdx], thY, rcCell);
    qU[1] = grid.tvdRec(vU1[1], U[cIdx],  vD1[1], thY, rcCell);
    qB[1] = grid.tvdRec(U[b2R], U[b1R], U[cIdx], thZ, rcCell);
    qF[1] = grid.tvdRec(U[f1R], U[cIdx],  U[b1R], thZ, rcCell);

    qL[2] = grid.tvdRec(V[l2R], vL1[2], V[cIdx], thX, rcCell);
    qR[2] = grid.tvdRec(vR1[2], V[cIdx],  vL1[2], thX, rcCell);
    qD[2] = grid.tvdRec(V[d2R], vD1[2], V[cIdx], thY, rcCell);
    qU[2] = grid.tvdRec(vU1[2], V[cIdx],  vD1[2], thY, rcCell);
    qB[2] = grid.tvdRec(V[b2R], V[b1R], V[cIdx], thZ, rcCell);
    qF[2] = grid.tvdRec(V[f1R], V[cIdx],  V[b1R], thZ, rcCell);

    qL[3] = grid.tvdRec(W[l2R], vL1[3], W[cIdx], thX, rcCell);
    qR[3] = grid.tvdRec(vR1[3], W[cIdx],  vL1[3], thX, rcCell);
    qD[3] = grid.tvdRec(W[d2R], vD1[3], W[cIdx], thY, rcCell);
    qU[3] = grid.tvdRec(vU1[3], W[cIdx],  vD1[3], thY, rcCell);
    qB[3] = grid.tvdRec(W[b2R], W[b1R], W[cIdx], thZ, rcCell);
    qF[3] = grid.tvdRec(W[f1R], W[cIdx],  W[b1R], thZ, rcCell);

    qL[4] = grid.tvdRec(P[l2R], vL1[4], P[cIdx], thX, rcCell);
    qR[4] = grid.tvdRec(vR1[4], P[cIdx],  vL1[4], thX, rcCell);
    qD[4] = grid.tvdRec(P[d2R], vD1[4], P[cIdx], thY, rcCell);
    qU[4] = grid.tvdRec(vU1[4], P[cIdx],  vD1[4], thY, rcCell);
    qB[4] = grid.tvdRec(P[b2R], P[b1R], P[cIdx], thZ, rcCell);
    qF[4] = grid.tvdRec(P[f1R], P[cIdx],  P[b1R], thZ, rcCell);

    // ---- gradient MUSCL (--recon 6): centroid -> open-face centroid ---------
    // Replaces the 1-D limited slopes entirely: one limited least-squares
    // gradient per cell, evaluated from that cell's fluid centroid to the open
    // face's centroid.  This is the reconstruction the cut geometry actually
    // calls for -- the earlier fix-up (an unlimited central-difference shift
    // bolted onto a limited 1-D slope) was inconsistent between the value and
    // its correction, and left nothing to stop an overshoot at a cut face.
    if (rcCell == 6) {
      real *Fs[4] = {Rho, U, V, P};
      const real hR = fmin(dx, dy);
      const real hx = (real)0.5*dx, hy = (real)0.5*dy;
      real f00 = -1, f10 = -1, f01 = -1;
      real ocx = 0, ocy = 0, olx = 0, oly = 0, odx = 0, ody = 0;
      real tX = 0, tY = 0;
      if (grid.ibRccm) {
        // PREPROCESSED (F_CUTTX/F_CUTTY): the open-face centroids come from the
        // same geometry as alpha and the apertures -- curved under --cutgeom 2 --
        // instead of being re-derived here from three level-set evaluations.
        tX = grid.getField(F_CUTTX)[cIdx];
        tY = grid.getField(F_CUTTY)[cIdx];
        if (grid.cutSplit && grid.cutFaceId && grid.cutFaceId[cIdx] >= 0) {
          // --cutsplit: this path carries the (piece 0, piece 0) segment only
          const CutFace &Fc = grid.cutFace[grid.cutFaceId[cIdx]];
          if (Fc.nX) tX = Fc.sx[0].cen;
          if (Fc.nY) tY = Fc.sy[0].cen;
        }
        // PREPROCESSED: stamped once per geometry, not re-derived per stage.
        ocx = grid.getField(F_CUTCX)[cIdx];  ocy = grid.getField(F_CUTCY)[cIdx];
        const i32 cEg = bEmpty*blockSizeTot;
        if (l1Idx < cEg) { olx = grid.getField(F_CUTCX)[l1Idx]; oly = grid.getField(F_CUTCY)[l1Idx]; }
        if (d1Idx < cEg) { odx = grid.getField(F_CUTCX)[d1Idx]; ody = grid.getField(F_CUTCY)[d1Idx]; }
      }
      real gC[4][2], lC[4], gL[4][2], lL[4], gD[4][2], lD[4];
      rccmGradLimited(grid, Fs, bIdx, i,   j,   k, cpos, dx, dy, hR, ocx, ocy, gC, lC);
      rccmGradLimited(grid, Fs, bIdx, i-1, j,   k,
                      Vec3(cpos[0]-dx, cpos[1], cpos[2]), dx, dy, hR, olx, oly, gL, lL);
      rccmGradLimited(grid, Fs, bIdx, i,   j-1, k,
                      Vec3(cpos[0], cpos[1]-dy, cpos[2]), dx, dy, hR, odx, ody, gD, lD);
      if (grid.ibRccm && grid.dbgChecks) {
        real *CA = grid.getField(F_CUTA);
        real *AXf = grid.getField(F_CUTAX), *AYf = grid.getField(F_CUTAY);
        const bool deadL = !(CA[l1Idx] > grid.ibRccmAlphaMin);
        const bool deadD = !(CA[d1Idx] > grid.ibRccmAlphaMin);
        if (deadL && AXf[cIdx] > (real)1e-12) atomicAdd(&g_rcDeadFace, 1ull);
        if (deadD && AYf[cIdx] > (real)1e-12) atomicAdd(&g_rcDeadFace, 1ull);
        if (grid.rccmLive(cIdx)) atomicAdd(&g_rcLiveFace, 2ull);
      }
      const i32 slot[4] = {0, 1, 2, 4};
      // face centroid positions relative to each cell's own centroid
      const real xfL = -hx - ocx*hR,          yfL = tX*dy - ocy*hR;   // c  -> low-x face
      const real xfLn =  hx - olx*hR,         yfLn = tX*dy - oly*hR;  // l1 -> same face
      const real xfD = tY*dx - ocx*hR,        yfD = -hy - ocy*hR;     // c  -> low-y face
      const real xfDn = tY*dx - odx*hR,       yfDn =  hy - ody*hR;    // d1 -> same face
      for (i32 f = 0; f < 4; f++) {
        qR[slot[f]] = Rho[cIdx]*0 + Fs[f][cIdx]   + lC[f]*(gC[f][0]*xfL  + gC[f][1]*yfL);
        qL[slot[f]] =               Fs[f][l1Idx]  + lL[f]*(gL[f][0]*xfLn + gL[f][1]*yfLn);
        qU[slot[f]] =               Fs[f][cIdx]   + lC[f]*(gC[f][0]*xfD  + gC[f][1]*yfD);
        qD[slot[f]] =               Fs[f][d1Idx]  + lD[f]*(gD[f][0]*xfDn + gD[f][1]*yfDn);
      }
      // positivity is guaranteed by the limiter for the neighbour hull, but a
      // hull that already contains a tiny value can still land near zero
      qR[0] = fmax(qR[0], (real)1e-12); qR[4] = fmax(qR[4], (real)1e-12);
      qL[0] = fmax(qL[0], (real)1e-12); qL[4] = fmax(qL[4], (real)1e-12);
      qU[0] = fmax(qU[0], (real)1e-12); qU[4] = fmax(qU[4], (real)1e-12);
      qD[0] = fmax(qD[0], (real)1e-12); qD[4] = fmax(qD[4], (real)1e-12);
    }

    Vec5 fluxL = grid.hllcFlux(grid.prim2cons(qL), grid.prim2cons(qR), Vec3(1,0,0));
    Vec5 fluxD = grid.hllcFlux(grid.prim2cons(qD), grid.prim2cons(qU), Vec3(0,1,0));

    real ax = dy*dz/vol;   // = 1/dx
    real ay = dx*dz/vol;   // = 1/dy

    // Viscous face fluxes.  The conservative form is dU/dt + dFc/dx = dFv/dx,
    // so the viscous flux enters the SAME face-scatter with the opposite sign;
    // folding it into the convective flux keeps one accumulation per face and
    // leaves the two sides of every face exactly equal and opposite.
    real *MuT = grid.getField(F_MUT);

    // Grid-aligned wall model: is this cell's LOW-y face the wall face?  The
    // wall lies wallOffset below the bottom domain face, so the face itself is
    // at y = 0 and the cell centre half a cell above it.
    const real yFace = cpos[1] - (real)0.5*dy;
    // EVERY level's wall row takes the wall flux, not just the finest.  Skipping
    // the coarse parents (they are restricted from their children anyway, so it
    // looked free) leaves their bottom face to the ordinary interior path -- and
    // there phi(d_face -> 0) -> 0, so the Eq. (A.9) term -sigW1 (rho k~ tau~/phi)
    // divides by ~0 and the parent's wall-normal momentum explodes.  The wall
    // face is exactly where the Appendix-A form is singular, which is why the
    // wall model has to own it at every level.  On a coarse parent d_IP is the
    // local 3*dy and disagrees with the global d_cutoff, but that only has to be
    // BOUNDED there: restrictFieldsKernel overwrites the parent from its
    // children every stage.  wallFineBand keeps the LEAVES at one resolution,
    // which is what the model's accuracy actually depends on.
    const bool wallFaceY = grid.rans && grid.wallGeom == 1
                        && grid.isInteriorBlock(lvl, ib, jb, kb)
                        && fabs(yFace) < (real)1e-6*dy
                        && cpos[0] >= grid.plateX0;
    real Fwall[5] = {0,0,0,0,0}, FwallK = 0, FwallT = 0, uTauFace = 0;
    if (wallFaceY) {
      wallModelFluxY(grid, Rho, U, V, W, P,
                     grid.getField(F_RHOK), grid.getField(F_RHOTAU),
                     grid.getField(F_TF1), bIdx, i, j, k, dy,
                     Fwall, FwallK, FwallT, uTauFace);
      grid.getField(F_SCRATCH)[cIdx] = uTauFace;   // for the C_f / probe output
    }

    if (grid.mu > 0) {
      real Fv[5];
      const real mtL = grid.rans ? (real)0.5*(MuT[l1Idx] + MuT[cIdx]) : (real)0;
      const real mtD = grid.rans ? (real)0.5*(MuT[d1Idx] + MuT[cIdx]) : (real)0;
      viscFaceFlux(grid, Rho, U, V, W, P, bIdx, i, j, k, 0, dx, dy, dz, Fv, mtL, lvl, cpos);
      for (i32 n = 1; n < 5; n++) fluxL[n] -= Fv[n];
      if (!wallFaceY) {
        viscFaceFlux(grid, Rho, U, V, W, P, bIdx, i, j, k, 1, dx, dy, dz, Fv, mtD, lvl, cpos);
        for (i32 n = 1; n < 5; n++) fluxD[n] -= Fv[n];
      }
    }

    // the wall-model flux REPLACES the interior face flux, it does not add to it
    if (wallFaceY) for (i32 n = 0; n < 5; n++) fluxD[n] = Fwall[n];

    // ---- immersed boundary: same replacement on any fluid/solid face -------
    // A face between a fluid and a non-fluid cell IS the wall boundary; the
    // interior flux there is meaningless because the solid side carries no
    // solution.  ibWallFlux returns the flux in the +e_d sense with the wall
    // normal's own orientation folded in, so the ordinary scatter below is
    // correct whichever side the body is on.
    bool ibX = false, ibY = false, ibZ = false;
    i32  ibFluidX = cIdx, ibFluidY = cIdx, ibFluidZ = cIdx;
    real ibKx = 0, ibTx = 0, ibKy = 0, ibTy = 0, ibKz = 0, ibTz = 0;
    // Faces on which the ORDINARY interior k~/tau~ flux may run.  Inside the
    // body the wall distance is identically zero, so phi = phiDamp(0) = 0 and
    // the Eq. (A.9) term -sigW1 (rho k~ tau~ / phi) divides by the 1e-20 floor:
    // a 1e20 in the residual.  Only the wall FACES are legitimate there, and
    // ibWallFlux already owns those, so any face touching a non-fluid cell must
    // contribute no interior turbulence flux at all.
    bool ktX = true, ktY = true, ktZ = true;
    // ibFluxRecon (Euler): NO flux replacement -- the ghost state carries the
    // wall condition and the ordinary MUSCL+HLLC path computes every face,
    // fluid/wall included.  The RANS wall model keeps its boundary flux.
    // NOT under Brinkman.  The sharp path REPLACES the face flux on fluid/solid
    // faces with ibWallFlux; the porosity formulation has no such faces and
    // already carries the wall through phi-weighting plus the volumetric
    // penalization.  Running both at once is a direct conflict -- it put NaN in
    // k~/tau~ on the first Brinkman+RANS attempt while the mean flow stayed
    // finite, because ibWallFlux was writing wall fluxes into a smeared interface.
    if (grid.immerserdBcType != 0
        && (grid.rans || grid.ibFluxRecon != 1)) {
      real Fi[5], Ki, Ti, Ut;
      // SLIP faces under RANS must use the SAME trace+Riemann path the
      // inviscid gates validate -- routing them to ibWallFlux's legacy
      // prescribed-flux slip branch (the old gate) let the slip REGION of the
      // wall-modelled plate blow itself apart before the model ever engaged:
      // measured u = -2.4 at the first cell at x = wmX0 - 0.01 while the
      // grid-aligned reference's slip region is u = +1.00 exactly.  Every
      // downstream symptom (the pinned separation bubble, the 9x k excess,
      // mu_t/mu ~ 500, the 2.2x run-to-run scatter) was fed by this wreckage.
      const bool trace2 = (grid.ibFluxRecon == 2);
      const real x0wSlip = (grid.wmX0 >= (real)0) ? grid.wmX0 : grid.plateX0;
      // Neighbour fluid flag.  The cached mask is only addressable when the
      // neighbour CELL exists at this level; where the same-level block is
      // absent (a coarse/fine interface) getNbrIdx returns the empty block,
      // whose F_IBM is zero -- i.e. "solid" -- which manufactures a wall face
      // wherever the refined region ends.  That is invisible on a uniform grid
      // (case 14: every neighbour exists, 1312/1312 fluxes applied) but on the
      // nLvls-6 airfoil it produced 880 spurious wall faces with the surface up
      // to 71 CELLS away.  Fall back to the geometric test at the neighbour's
      // position, which is always well defined.
      const real *Ibm = grid.getField(F_IBM);
      const i32 cEmptyI = bEmpty*blockSizeTot;
      const bool fC = Ibm[cIdx] > (real)0.5;
      // The WLS/face traces impose a SLIP WALL (u_n = 0 plus FRIB Eq. 19 on
      // u_t).  That is only valid where the boundary IS a wall -- applying it at
      // a prescribed inflow/outflow face walls off the flow, which is exactly
      // what made --ibwls degrade the Ringleb duct (order 1.48 -> 0.58).
      real *BcF = grid.getField(F_IBBC);
      #define IB_FACE_IS_WALL(NBRIDX) \
        (!grid.ibPolyBc ? true \
          : (((NBRIDX) < cEmptyI) ? (BcF[NBRIDX] <= (real)0.5) \
                                  : (BcF[cIdx]  <= (real)0.5)))
      #define IB_NBR_FLUID(IDX, PX, PY, PZ, HH) \
        (((IDX) < cEmptyI) ? (Ibm[IDX] > (real)0.5) \
                           : grid.isFluidCell(Vec3((PX),(PY),(PZ)), (HH)))
      if (fC != IB_NBR_FLUID(l1Idx, cpos[0]-dx, cpos[1], cpos[2], dx)) {
        Vec3 fc(cpos[0]-(real)0.5*dx, cpos[1], cpos[2]);
        if (trace2 && (!(grid.rans || grid.ibWmles) || fc[0] < x0wSlip
                       || fc[0] > grid.wmX1)) {
          // constrained quadratic WLS first; it declines (and falls through to
          // the point-sample trace) in 3-D or when the stencil is too cut up
          bool got = grid.ibWls && IB_FACE_IS_WALL(fC ? l1Idx : cIdx)
                  && ibWlsTrace(grid, Rho, U, V, W, P, lvl, fc, 0, dx,
                                fC ? cIdx : l1Idx, fC, Fi, bIdx, i, j, k,
                                ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k);
          if (got || ibFaceTraceFlux(grid, Rho, U, V, W, P, lvl, fc, 0, dx,
                              fC ? cIdx : l1Idx, fC, Fi, bIdx, i, j, k,
                              ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k))
            for (i32 nq = 0; nq < 5; nq++) fluxL[nq] = Fi[nq];
        }
        else if (ibWallFlux(grid, Rho, U, V, W, P, grid.getField(F_RHOK),
                       grid.getField(F_RHOTAU), grid.getField(F_TF1),
                       lvl, fc, 0, dx, fC ? cIdx : l1Idx, Fi, Ki, Ti, Ut,
                       bIdx, i, j, k, ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k)) {
          if (grid.ibWallMode == 4 && grid.mu > 0) {
            real Fv[5];
            const real mtL2 = (real)0.5*(MuT[l1Idx] + MuT[cIdx]);
            viscFaceFlux(grid, Rho, U, V, W, P, bIdx, i, j, k, 0, dx, dy, dz,
                         Fv, mtL2, lvl, cpos);
            for (i32 n = 1; n < 5; n++) fluxL[n] += Fv[n];
            for (i32 n = 1; n < 5; n++) fluxL[n] += Fi[n];
          }
          else for (i32 n = 0; n < 5; n++) fluxL[n] = Fi[n];
          ibX = true; ibFluidX = fC ? cIdx : l1Idx; ibKx = Ki; ibTx = Ti;
        }

      }
      if (fC != IB_NBR_FLUID(d1Idx, cpos[0], cpos[1]-dy, cpos[2], dy)) {
        atomicAdd(&g_ibDetect, 1ULL);
        Vec3 fc(cpos[0], cpos[1]-(real)0.5*dy, cpos[2]);
        if (trace2 && (!(grid.rans || grid.ibWmles) || fc[0] < x0wSlip
                       || fc[0] > grid.wmX1)) {
          bool got = grid.ibWls && IB_FACE_IS_WALL(fC ? d1Idx : cIdx)
                  && ibWlsTrace(grid, Rho, U, V, W, P, lvl, fc, 1, dy,
                                fC ? cIdx : d1Idx, fC, Fi, bIdx, i, j, k,
                                ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k);
          if (got || ibFaceTraceFlux(grid, Rho, U, V, W, P, lvl, fc, 1, dy,
                              fC ? cIdx : d1Idx, fC, Fi, bIdx, i, j, k,
                              ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k)) {
            atomicAdd(&g_ibFlux, 1ULL);
            for (i32 nq = 0; nq < 5; nq++) fluxD[nq] = Fi[nq];
          }
        }
        else if (ibWallFlux(grid, Rho, U, V, W, P, grid.getField(F_RHOK),
                       grid.getField(F_RHOTAU), grid.getField(F_TF1),
                       lvl, fc, 1, dy, fC ? cIdx : d1Idx, Fi, Ki, Ti, Ut,
                       bIdx, i, j, k, ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k)) {
          atomicAdd(&g_ibFlux, 1ULL);
          if (grid.ibWallMode == 4 && grid.mu > 0) {
            // strip the ordinary viscous part (same inputs -> bitwise), then
            // add the exact tau_w-only flux ibWallFlux handed back
            real Fv[5];
            const real mtD2 = (real)0.5*(MuT[d1Idx] + MuT[cIdx]);
            viscFaceFlux(grid, Rho, U, V, W, P, bIdx, i, j, k, 1, dx, dy, dz,
                         Fv, mtD2, lvl, cpos);
            for (i32 n = 1; n < 5; n++) fluxD[n] += Fv[n];
            for (i32 n = 1; n < 5; n++) fluxD[n] += Fi[n];
          }
          else for (i32 n = 0; n < 5; n++) fluxD[n] = Fi[n];
          ibY = true; ibFluidY = fC ? cIdx : d1Idx; ibKy = Ki; ibTy = Ti;
          grid.getField(F_SCRATCH)[ibFluidY] = Ut;      // for the C_f dump
        }

      }
      const bool fL = IB_NBR_FLUID(l1Idx, cpos[0]-dx, cpos[1], cpos[2], dx);
      const bool fD = IB_NBR_FLUID(d1Idx, cpos[0], cpos[1]-dy, cpos[2], dy);
      ktX = fC && fL;
      ktY = fC && fD;
      // Ghost-cell wall architecture: the turbulence transport must ALSO cross
      // the wall face, reading the ghost's nu~ (the signed linear extension
      // that interpolates to zero at the surface).  Without this the first
      // cell's nu~ has no diffusive feed against its destruction term
      // cw1 fw (nu~/d)^2 -- the largest in the domain exactly there -- and the
      // layer relaminarizes over a few flow times (measured: C_f lands in the
      // band at t ~ 1 and then decays 0.0027 -> 0.0006, IDENTICALLY for two
      // different nu~ ghost values, which is what betrayed the missing flux).
      if (grid.ibWallMode >= 1 && grid.turbModel == 1) {
        const i32 cE = bEmpty*blockSizeTot;
        if (fC && !fL && l1Idx < cE) ktX = true;
        if (fC && !fD && d1Idx < cE) ktY = true;
      }
      if (!grid.pseudo2D) {
        const bool fB = IB_NBR_FLUID(b1Idx, cpos[0], cpos[1], cpos[2]-dz, dz);
        if (fC != fB) ibZ = true;   // handled in the z block
        ktZ = fC && fB;
      }
      #undef IB_NBR_FLUID
    }

    // ---- Brinkman porosity face weights (pressure-tight penalization) -------
    // Every flux entering this cell is scaled by w_f = phibar_f/phi_c, and the
    // p grad(phi) source below is built from the SAME phibar_f.  With that one
    // sharing, a quiescent uniform-pressure state cancels bit-for-bit: the
    // pressure part of the flux divergence, sum_f p phibar_f n_f A_f / (phi_c V),
    // is exactly what the source subtracts.  Measured on a quiescent body:
    // residual 0.000000e+00 shared, 8.76e-03 with an analytic grad(phi) that
    // does not share the face quadrature -- which is why only the shared form
    // survives here.
    real wLc = 1, wLn = 1, wDc = 1, wDn = 1, wBc = 1, wBn = 1, phiC = 1;
    if (grid.ibBrink && grid.immerserdBcType != 0) {
      real *PhiG = grid.getField(F_PHI);
      const i32 cEmptyI = bEmpty*blockSizeTot;
      // The cached level set is only meaningful for real cells; anything past
      // cEmptyI is evaluated.  phi is positive INSIDE, so s = -phi.
      #define BRINK_S(IDX, PX, PY, PZ) \
        (-(((IDX) < cEmptyI) ? PhiG[IDX] \
                             : grid.getBoundaryLevelSet(Vec3((PX),(PY),(PZ)))))
      // delta is a FIXED PHYSICAL length taken from the finest level, never the
      // local cell size.  phi is a property of the body, so it has to be one
      // single field: keying it to the local h redefines the wall across every
      // coarse-fine interface, the two sides of a hanging face disagree about
      // phi, and the solution diverges there (nLvls 1 fine, nLvls >= 2 NaN).
      const real hb = fmin(grid.getDx(grid.nLvls-1),
                           grid.pseudo2D ? grid.getDx(grid.nLvls-1)
                                         : grid.getDy(grid.nLvls-1));
      const real sC = BRINK_S(cIdx,  cpos[0], cpos[1], cpos[2]);
      phiC = grid.brinkPhi(sC, hb);
      const real sL = BRINK_S(l1Idx, cpos[0]-dx, cpos[1], cpos[2]);
      const real sD = BRINK_S(d1Idx, cpos[0], cpos[1]-dy, cpos[2]);
      // ---- face porosity: CACHED (F_BRINKX/F_BRINKY) ------------------------
      // The body is static, so phibar over each low face was stamped once by
      // ibStampGeometry and carried through every block sort.  Two consequences,
      // both load-bearing: the segmented quadrature costs nothing per stage
      // (measured 24.3 -> 5.0 ms/iteration at nlvls 4), and both cells sharing a
      // face read the ONE stored number, which is what keeps the flux and the
      // p grad(phi) source exactly consistent.
      const real pbL = grid.getField(F_BRINKX)[cIdx];
      const real pbD = grid.getField(F_BRINKY)[cIdx];
      wLc = pbL/phiC;   wLn = pbL/grid.brinkPhi(sL, hb);
      wDc = pbD/phiC;   wDn = pbD/grid.brinkPhi(sD, hb);
      // 2-D ONLY.  The z faces have no stamped bank, and evaluating them live
      // here is not an option: brinkPhiFaceAvgSeg loops over the polyline SDF,
      // and inlining that into this kernel -- already at REG:255 with stack
      // spills -- cost 3x the wall time even with the branch dead under
      // pseudo2D (measured 22.6 s -> 64.9 s on the RAE at nlvls 4).  A 3-D run
      // needs a third stamped field, F_BRINKZ, not a live quadrature.
      #undef BRINK_S
    }

    // ---- RCCM cut-cell weighting (their Eqs. 4, 9) --------------------------
    // The FVM update on a cut cell is  dU/dt = -(1/dV_i) sum_f F_f A_f, with
    // dV_i = alpha_i dx dy and A_f the OPEN part of each face.  The existing
    // scatter already divides each face by the receiving cell (that structure
    // outlived the Brinkman weights it was built for), so the aperture goes in
    // as A_f/alpha_recv and nothing else in the flux path changes.
    i32 tC = cIdx, tL = l1Idx, tD = d1Idx;        // scatter targets (owners under --cutmerge)
    if (grid.ibRccm) {
      real *CA = grid.cutMerge ? grid.cutAlphaE : grid.getField(F_CUTA);   // element volume when merged
      real *AXf = grid.getField(F_CUTAX), *AYf = grid.getField(F_CUTAY);
      const real aC = fmax(CA[cIdx], grid.ibRccmAlphaMin);
      const real aL = fmax(CA[l1Idx], grid.ibRccmAlphaMin);
      const real aD = fmax(CA[d1Idx], grid.ibRccmAlphaMin);
      real apX = AXf[cIdx], apY = AYf[cIdx];   // this cell's LOW faces
      if (grid.cutSplit && grid.cutFaceId && grid.cutFaceId[cIdx] >= 0) {
        // --cutsplit: only the (piece 0, piece 0) segment rides this path; the
        // other segments and the extra pieces' walls are cutSplitFluxKernel's
        const CutFace &Fc = grid.cutFace[grid.cutFaceId[cIdx]];
        if (Fc.nX) apX = Fc.sx[0].len;
        if (Fc.nY) apY = Fc.sy[0].len;
      }
      wLc = apX/aC;  wLn = apX/aL;
      wDc = apY/aC;  wDn = apY/aD;
      if (grid.cutMerge) {
        // Merged element: every member's face flux lands on the OWNER, divided
        // by the element volume.  A face between two members of the same
        // element is internal -- its two contributions would only cancel to
        // roundoff under atomics, so it is skipped outright.
        tC = grid.cutOwner[cIdx]; tL = grid.cutOwner[l1Idx]; tD = grid.cutOwner[d1Idx];
        if (tL == tC) { wLc = 0; wLn = 0; }
        if (tD == tC) { wDc = 0; wDn = 0; }
      }
    }

    // --leaf: a face across a level jump belongs to the mortar (coarser across:
    // the neighbour is missing; finer across: the neighbour is a PARENT), and a
    // covered PARENT cell inside a mixed block owns nothing
    bool skipL = false, skipD = false;
    if (grid.leafFlux) {
      const i32 cEl = bEmpty*blockSizeTot;
      const bool covered = grid.cFlagsList[cIdx] == PARENT;
      skipL = covered || l1Idx >= cEl || grid.cFlagsList[l1Idx] == PARENT;
      skipD = covered || d1Idx >= cEl || grid.cFlagsList[d1Idx] == PARENT;
      if (skipL) { wLc = 0; wLn = 0; }
      if (skipD) { wDc = 0; wDn = 0; }
    }
    real *Rhs[5] = {RhsRho, RhsRhoU, RhsRhoV, RhsRhoW, RhsRhoE};
    if (grid.detFlux) {
      // deterministic path: store this thread's two faces; gatherFaceFluxKernel
      // sums each cell's faces in fixed order.  Face weights are 1 here (the
      // flag is resolved off under Brinkman), so only the area factor rides in.
      real *FF = grid.ffBuf; const u64 NN = grid.ffN;
      for (i32 n = 0; n < 5; n++) {
        FF[(u64)n*NN + (u64)cIdx]       = skipL ? (real)0 : fluxL[n]*ax;
        FF[(u64)(5+n)*NN + (u64)cIdx]   = skipD ? (real)0 : fluxD[n]*ay;
      }
    } else
    for (i32 n = 0; n < 5; n++) {
      atomicAdd(cutRhsPtr(grid, tC, n),   fluxL[n]*ax*wLc + fluxD[n]*ay*wDc);
      atomicAdd(cutRhsPtr(grid, tL, n), - fluxL[n]*ax*wLn);
      atomicAdd(cutRhsPtr(grid, tD, n), - fluxD[n]*ay*wDn);
    }

    // ---- point-implicit small cells (--cutpi) --------------------------------
    // Stamp the 1/alpha excess of THIS cell's own divergence.  The Jacobian
    // spectral radius of  (1/alpha - 1) sum_f F_f A_f / dV  is
    // (|u| + a) (1/alpha - 1) sum_f A_f / dV, and every open face contributes --
    // the wall segment is NOT special.  A corner sliver of legs `a` has volume
    // a^2/2, two Cartesian faces of length a and a wall of a*sqrt(2): the
    // flux-to-volume ratios are 2F/a and 2*sqrt(2)F/a, the same order, so
    // stamping only the wall flux would leave the CFL limit exactly where it was.
    // (Directly evidenced next door: --brinkpi 1 stamps one term and buys
    // nothing, --brinkpi 2 stamps the full sum and retires the restriction.)
    if (grid.cutPiDamp && grid.ibRccm && grid.immerserdBcType != 0 && grid.rccmLive(cIdx)) {
      real *CA = grid.getField(F_CUTA);
      real *AXf = grid.getField(F_CUTAX), *AYf = grid.getField(F_CUTAY);
      const real aC = fmax(CA[cIdx], grid.ibRccmAlphaMin);
      if (aC < (real)1 - (real)1e-12) {                 // uncut cells: lambda = 0
        // this cell's four open faces: its own low faces, the neighbours' lows
        const i32 cE2 = bEmpty*blockSizeTot;
        const real apXlo = AXf[cIdx], apYlo = AYf[cIdx];
        const real apXhi = (r1Idx < cE2) ? AXf[r1Idx] : (real)0;
        const real apYhi = (u1Idx < cE2) ? AYf[u1Idx] : (real)0;
        // wall segment from the discrete divergence theorem, as the wall flux does
        const real awx = (apXhi - apXlo)*dy, awy = (apYhi - apYlo)*dx;
        const real aWall = sqrt(awx*awx + awy*awy);
        const real sumA = (apXlo + apXhi)*dy + (apYlo + apYhi)*dx + aWall;
        const real a2   = gam*P[cIdx]/fmax(Rho[cIdx],(real)1e-30);
        const real lamC = sqrt(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx] + W[cIdx]*W[cIdx])
                        + sqrt(fmax(a2,(real)0));
        // (1/alpha - 1) sum_f A_f / dV_uncut, i.e. the EXCESS over the uncut cell
        const real exc = ((real)1/aC - (real)1)*sumA/vol;
        atomicAdd(&grid.getField(F_LAMM)[cIdx], lamC*exc);
      }
    }

    if (grid.ibBrink && grid.immerserdBcType != 0) {
      // ---- p grad(phi) in flux-scatter form, POINT-IMPLICIT ----------------
      // The scatter mirrors the flux scatter exactly (same face weights), so a
      // quiescent uniform-pressure state cancels bit-for-bit; only the momentum
      // UPDATE is relaxed rather than applied outright.
      //
      // Why this term and not the others: in the smeared band |grad phi| ~
      // 1/delta, so p grad(phi) can change a cell's momentum by many times its
      // own magnitude in one explicit step.  It is the stiffest term in the
      // penalization.  Dividing the update by (1 + B dt lambda) caps the change
      // at the momentum actually present, and fixed points are untouched: where
      // the source vanishes so does lambda.
      atomicAdd(&RhsRhoU[cIdx], -P[cIdx]*ax*wLc);
      atomicAdd(&RhsRhoU[l1Idx],  P[l1Idx]*ax*wLn);
      atomicAdd(&RhsRhoV[cIdx], -P[cIdx]*ay*wDc);
      atomicAdd(&RhsRhoV[d1Idx],  P[d1Idx]*ay*wDn);
      {
        // Point-implicit stamp of the porosity stiffness.  Split the weighted
        // divergence into the plain one plus a pointwise excess:
        //   (1/phi_c) sum_f phibar_f F_f n_f / h
        //     = sum_f F_f n_f / h + sum_f (w_f - 1) F_f n_f / h,  w_f = phibar_f/phi_c.
        // Only the second is stiff -- it carries the exp(h/delta) amplification
        // that sets the step -- and it is LOCAL, so a diagonal treatment absorbs
        // it.  Its Jacobian spectral radius is (|u| + a) sum_f |w_f - 1|/h.
        //
        // This is what makes a narrow band affordable, and it is not optional:
        // at delta = h/8 on the RAE this reaches t = 2 in 239 iterations, while
        // stamping only the pressure term on the momentum rows (the old
        // --brinkpi 1) had not reached t = 40 in 31000.  The delta >= 1.5h floor
        // in the original method was an artifact of not having this.
        //
        // The cell's own HIGH faces are not carried by the low-face scatter, so
        // read them from the neighbours' stamped LOW faces.  Guard the index:
        // past cEmptyI the bank was never stamped (it reads as the allocator's
        // zero), and a phibar of 0 would inflate this cell's stamp.
        const i32 cE2 = bEmpty*blockSizeTot;
        const real wR = (r1Idx < cE2) ? grid.getField(F_BRINKX)[r1Idx]/fmax(phiC,(real)1e-30) : (real)1;
        const real wU = (u1Idx < cE2) ? grid.getField(F_BRINKY)[u1Idx]/fmax(phiC,(real)1e-30) : (real)1;
        const real a2   = gam*P[cIdx]/fmax(Rho[cIdx],(real)1e-30);
        const real lamC = sqrt(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx] + W[cIdx]*W[cIdx])
                        + sqrt(fmax(a2,(real)0));
        const real sw = (fabs(wLc-(real)1) + fabs(wR-(real)1))*ax
                      + (fabs(wDc-(real)1) + fabs(wU-(real)1))*ay;
        atomicAdd(&grid.getField(F_LAMM)[cIdx], lamC*sw);
      }
    }

    // (the cut cell's WALL flux lives in cutCellKernel now: no cut branch here)

    // k~ / tau~ transport: same mass flux as the mean flow, van Leer limited
    // states, plus the two diffusion forms.  fluxL[0] / fluxD[0] ARE the HLLC
    // mass fluxes through those faces.
    if (grid.rans) {
      real *K    = grid.getField(F_RHOK);
      real *Tau  = grid.getField(F_RHOTAU);
      real *TF1  = grid.getField(F_TF1);
      real *RhsK = grid.getField(F_RHS + F_RHOK);
      real *RhsT = grid.getField(F_RHS + F_RHOTAU);
      real *LamK = grid.getField(F_LAMK), *LamT = grid.getField(F_LAMT);
      if (grid.detFlux) {
        // Banks are STORAGE, not accumulators: a face whose turbulence flux is
        // suppressed (ibX/ibY replacement, the ktX/ktY gates) must read as
        // ZERO in the gather, so blank this cell's slots before the
        // conditional writers below overwrite them.
        real *FF = grid.ffBuf; const u64 NN = grid.ffN;
        const i32 ndd = grid.pseudo2D ? 2 : 3;
        for (i32 dd2 = 0; dd2 < ndd; dd2++) {
          const u64 b0 = (u64)(15 + 4*dd2)*NN + (u64)cIdx;
          FF[b0] = 0; FF[b0 + NN] = 0; FF[b0 + 2*NN] = 0; FF[b0 + 3*NN] = 0;
        }
      }
      if (ibX) {
        if (grid.detFlux) {
          // wall flux is one-sided into the fluid cell: A slot if that is this
          // thread's own cell, B slot if it is the west neighbour.  (LamK/LamT
          // stay atomic below: <=2 contributors to a zeroed slot is a
          // commutative sum, deterministic as is.)
          real *FF = grid.ffBuf; const u64 NN = grid.ffN;
          const u64 b0 = (u64)15*NN + (u64)cIdx;
          if (ibFluidX == cIdx) { FF[b0]      = ibKx*ax; FF[b0 + 2*NN] = ibTx*ax; }
          else                  { FF[b0 + NN] = ibKx*ax; FF[b0 + 3*NN] = ibTx*ax; }
        } else { atomicAdd(&RhsK[ibFluidX], ibKx*ax); atomicAdd(&RhsT[ibFluidX], ibTx*ax); }
        if (grid.wallPointImplicit) {
          atomicAdd(&LamK[ibFluidX], (real)3*fabs(ibKx)*ax
                    / fmax(Rho[ibFluidX]*K[ibFluidX],   (real)1e-30));
          atomicAdd(&LamT[ibFluidX], (real)3*fabs(ibTx)*ax
                    / fmax(Rho[ibFluidX]*Tau[ibFluidX], (real)1e-30));
        }
      }
      else if (ktX) ktauFaceFlux(grid, Rho, P, K, Tau, TF1, RhsK, RhsT,
                        bIdx, i, j, k, 0, dx, ax, fluxL[0], cIdx, l1Idx, cpos, wLc, wLn);
      if (ibY) {
        if (grid.detFlux) {
          real *FF = grid.ffBuf; const u64 NN = grid.ffN;
          const u64 b0 = (u64)19*NN + (u64)cIdx;
          if (ibFluidY == cIdx) { FF[b0]      = ibKy*ay; FF[b0 + 2*NN] = ibTy*ay; }
          else                  { FF[b0 + NN] = ibKy*ay; FF[b0 + 3*NN] = ibTy*ay; }
        } else { atomicAdd(&RhsK[ibFluidY], ibKy*ay); atomicAdd(&RhsT[ibFluidY], ibTy*ay); }
        if (grid.wallPointImplicit) {
          atomicAdd(&LamK[ibFluidY], (real)3*fabs(ibKy)*ay
                    / fmax(Rho[ibFluidY]*K[ibFluidY],   (real)1e-30));
          atomicAdd(&LamT[ibFluidY], (real)3*fabs(ibTy)*ay
                    / fmax(Rho[ibFluidY]*Tau[ibFluidY], (real)1e-30));
        }
      }
      else if (wallFaceY) {
        // Eq. (39) face values already gave the one-sided fluxes; there is no
        // convective part because no mass crosses the wall.
        if (grid.detFlux) {
          real *FF = grid.ffBuf; const u64 NN = grid.ffN;
          const u64 b0 = (u64)19*NN + (u64)cIdx;
          FF[b0] = FwallK*ay; FF[b0 + 2*NN] = FwallT*ay;
        } else {
        atomicAdd(&RhsK[cIdx], FwallK*ay);
        atomicAdd(&RhsT[cIdx], FwallT*ay);
        }
        if (grid.wallPointImplicit) {
          atomicAdd(&LamK[cIdx], (real)3*fabs(FwallK)*ay
                    / fmax(Rho[cIdx]*K[cIdx],   (real)1e-30));
          atomicAdd(&LamT[cIdx], (real)3*fabs(FwallT)*ay
                    / fmax(Rho[cIdx]*Tau[cIdx], (real)1e-30));
        }
      } else if (ktY) {
        ktauFaceFlux(grid, Rho, P, K, Tau, TF1, RhsK, RhsT,
                     bIdx, i, j, k, 1, dy, ay, fluxD[0], cIdx, d1Idx, cpos, wDc, wDn);
      }
    }

    // z-flux only in true 3D; pseudo2D never updates z-momentum (W stays 0)
    if (!grid.pseudo2D) {
      Vec5 fluxB = grid.hllcFlux(grid.prim2cons(qB), grid.prim2cons(qF), Vec3(0,0,1));
      if (grid.mu > 0) {
        real Fv[5];
        const real mtB = grid.rans ? (real)0.5*(MuT[b1Idx] + MuT[cIdx]) : (real)0;
        viscFaceFlux(grid, Rho, U, V, W, P, bIdx, i, j, k, 2, dx, dy, dz, Fv, mtB, lvl, cpos);
        for (i32 n = 1; n < 5; n++) fluxB[n] -= Fv[n];
      }
      real az = dx*dy/vol;   // = 1/dz
      if (grid.detFlux) {
        real *FF = grid.ffBuf; const u64 NN = grid.ffN;
        for (i32 n = 0; n < 5; n++) FF[(u64)(10+n)*NN + (u64)cIdx] = fluxB[n]*az;
      } else
      for (i32 n = 0; n < 5; n++) {
        atomicAdd(&Rhs[n][cIdx],    fluxB[n]*az*wBc);
        atomicAdd(&Rhs[n][b1Idx], - fluxB[n]*az*wBn);
      }
      if (grid.rans && ktZ) {
        ktauFaceFlux(grid, Rho, P, grid.getField(F_RHOK), grid.getField(F_RHOTAU),
                     grid.getField(F_TF1), grid.getField(F_RHS + F_RHOK),
                     grid.getField(F_RHS + F_RHOTAU),
                     bIdx, i, j, k, 2, dz, az, fluxB[0], cIdx, b1Idx, cpos);
      }
    }

  END_CELL_LOOP
}

//
// ---- Genuinely multidimensional Osher-type corner flux --------------------
// (Gaburro, Ricchiuto & Dumbser, arXiv:2506.00207, Sec. 3-4, adapted from
//  their Voronoi d+1-cell corners to the 4-cell corners of a Cartesian grid.)
//
// Each grid vertex p carries a full numerical flux TENSOR (Eq. 15/23)
//   F^_p,i = (1/4) sum_c f_i(Q_c)  -  (h/4) |A_i(Qbar_p)| (h grad_p Q)_i / h
// built from the 4 corner cell states; the edge flux is then the trapezoidal
// average of its two endpoint corner fluxes (Eq. 9), which makes the cell
// update conservative by construction (Eq. 10).  The corner gradient is the
// Green-Gauss formula (Eq. 20), exact for linear data on the 2x2 corner patch.
// One-point quadrature at the corner-average state replaces the path integral
// of |A_i| (first-order context).  FIRST-ORDER corner states: P0 cell averages.
// Pseudo-2D only.
//

// |A_n| dQ for the Euler system: standard wave decomposition at the mean
// primitive state (rb,ub,vb,wb,pb), unit normal (nx,ny,0).
__device__ void mdAbsJacDq(real rb, real ub, real vb, real wb, real pb,
                           real nx, real ny, const real dQ[5], real out[5]) {
  real cb = sqrt(gam*pb/rb);
  real q2 = ub*ub + vb*vb + wb*wb;
  real Hb = 0.5*q2 + cb*cb/(gam - 1.0);          // total enthalpy

  real d0 = dQ[0];
  real du = (dQ[1] - ub*d0)/rb;                   // primitive velocity deltas
  real dv = (dQ[2] - vb*d0)/rb;
  real dw = (dQ[3] - wb*d0)/rb;
  real dp = (gam - 1.0)*(dQ[4] - 0.5*q2*d0 - rb*(ub*du + vb*dv + wb*dw));

  real un  = ub*nx + vb*ny;
  real dun = du*nx + dv*ny;
  real am  = (dp - rb*cb*dun)/(2.0*cb*cb);        // acoustic (-) wave strength
  real ap  = (dp + rb*cb*dun)/(2.0*cb*cb);        // acoustic (+)
  real ae  = d0 - dp/(cb*cb);                     // entropy wave
  real dtx = du - dun*nx, dty = dv - dun*ny;      // tangential velocity delta

  real l1 = fabs(un - cb), l2 = fabs(un), l3 = fabs(un + cb);

  out[0] = l1*am              + l2*ae                              + l3*ap;
  out[1] = l1*am*(ub - cb*nx) + l2*(ae*ub + rb*dtx)                + l3*ap*(ub + cb*nx);
  out[2] = l1*am*(vb - cb*ny) + l2*(ae*vb + rb*dty)                + l3*ap*(vb + cb*ny);
  out[3] = l1*am*wb           + l2*(ae*wb + rb*dw)                 + l3*ap*wb;
  out[4] = l1*am*(Hb - cb*un) + l2*(ae*0.5*q2 + rb*(ub*dtx + vb*dty + wb*dw))
                              + l3*ap*(Hb + cb*un);
}

// physical flux in direction d (0=x,1=y) from a primitive corner state
__device__ inline void mdPhysFlux(real r, real u, real v, real w, real p,
                                  i32 d, real f[5]) {
  real E  = p/(gam - 1.0) + 0.5*r*(u*u + v*v + w*w);
  real vn = (d == 0) ? u : v;
  f[0] = r*vn;
  f[1] = r*vn*u + ((d == 0) ? p : 0.0);
  f[2] = r*vn*v + ((d == 1) ? p : 0.0);
  f[3] = r*vn*w;
  f[4] = vn*(E + p);
}

// 1D Osher-Solomon flux (the 1D special case of the multiD corner solver,
// paper Eq. 25; Dumbser-Toro form):
//   f^ = 1/2 (f(QL) + f(QR)) - 1/2 [ int_0^1 |A_d(psi(xi))| dxi ] (QR - QL)
// with the straight-line path psi(xi) = QL + xi (QR - QL) integrated by 3-point
// Gauss-Legendre (a 1-point rule degrades to Roe and can admit expansion
// shocks at sonic points).  Primitive L/R inputs; direction d (0=x, 1=y).
__device__ void osher1dFlux(real rL, real uL, real vL, real wL, real pL,
                            real rR, real uR, real vR, real wR, real pR,
                            i32 d, real out[5]) {
  real QL[5], QR[5], dQ[5], fL[5], fR[5];
  QL[0] = rL; QL[1] = rL*uL; QL[2] = rL*vL; QL[3] = rL*wL;
  QL[4] = pL/(gam - 1.0) + 0.5*rL*(uL*uL + vL*vL + wL*wL);
  QR[0] = rR; QR[1] = rR*uR; QR[2] = rR*vR; QR[3] = rR*wR;
  QR[4] = pR/(gam - 1.0) + 0.5*rR*(uR*uR + vR*vR + wR*wR);
  for (i32 n = 0; n < 5; n++) dQ[n] = QR[n] - QL[n];
  mdPhysFlux(rL, uL, vL, wL, pL, d, fL);
  mdPhysFlux(rR, uR, vR, wR, pR, d, fR);

  const real xg[3] = {0.11270166537925831, 0.5, 0.88729833462074169};
  const real wg[3] = {5.0/18.0, 8.0/18.0, 5.0/18.0};
  real nx = (d == 0) ? 1.0 : 0.0, ny = 1.0 - nx;
  real acc[5] = {0, 0, 0, 0, 0};
  for (i32 g = 0; g < 3; g++) {
    real Qg[5];
    for (i32 n = 0; n < 5; n++) Qg[n] = QL[n] + xg[g]*dQ[n];
    real rg = Qg[0];
    real ug = Qg[1]/rg, vg = Qg[2]/rg, wgv = Qg[3]/rg;
    real pg = (gam - 1.0)*(Qg[4] - 0.5*rg*(ug*ug + vg*vg + wgv*wgv));
    if (pg < (real)1e-12) pg = (real)1e-12;
    real tmp[5];
    mdAbsJacDq(rg, ug, vg, wgv, pg, nx, ny, dQ, tmp);
    for (i32 n = 0; n < 5; n++) acc[n] += wg[g]*tmp[n];
  }
  for (i32 n = 0; n < 5; n++) out[n] = 0.5*(fL[n] + fR[n]) - 0.5*acc[n];
}

// unlimited 3rd-order upwind parabola: right-face value of cell b with left
// neighbour a and right neighbour c (kappa = 1/3 MUSCL; mirror args for left)
__device__ inline real rec3(real a, real b, real c) {
  return (-a + 5.0*b + 2.0*c)/6.0;
}

// Raw ingredients (rho, u, v, p) of cell (ii,jj)'s face value
// in direction d at `side`, reconstructed along d by rec3 when recon==3.
__device__ void mdFaceState1D(CompressibleSolver &grid,
                              real *Rho, real *U, real *V, real *P,
                              i32 bIdx, i32 ii, i32 jj, i32 kk,
                              i32 d, real side, real h, real out[5]) {
  i32 di = (d == 0), dj = (d == 1);
  i32 id  = grid.getNbrIdx(bIdx, ii, jj, kk);
  real r = Rho[id], u = U[id], v = V[id], p = P[id];

  if (grid.recon == 3) {
    i32 idm = grid.getNbrIdx(bIdx, ii - di, jj - dj, kk);
    i32 idp = grid.getNbrIdx(bIdx, ii + di, jj + dj, kk);
    i32 iu  = (side > 0) ? idm : idp;    // rec3 mirrored by face side
    i32 idn = (side > 0) ? idp : idm;
    real rr = rec3(Rho[iu], r, Rho[idn]);
    real pr = rec3(P[iu],   p, P[idn]);
    real tr = (d == 0) ? rec3(V[iu], v, V[idn]) : rec3(U[iu], u, U[idn]);
    real nr = (d == 0) ? rec3(U[iu], u, U[idn]) : rec3(V[iu], v, V[idn]);
    if (rr > (real)1e-10 && pr > (real)1e-10) {   // positivity fallback
      r = rr; p = pr;
      if (d == 0) { v = tr; u = nr; }
      else        { u = tr; v = nr; }
    }
  }
  out[0] = r; out[1] = u; out[2] = v; out[3] = 0.0; out[4] = p;
}

// Primitive state (r,u,v,w,p) contributed by cell (ii,jj) to direction d's face
// plane at `side`, with optional TANGENTIAL corner offset tside (+-1 = evaluate
// at the edge endpoint / cell corner, 0 = face midpoint).  With recon==3 the
// corner value is the nested tensor reconstruction: rec3 along d in three
// tangential rows, then rec3 across the rows toward tside -- Simpson's edge
// quadrature needs the flux at the edge ENDPOINTS, and row-centre values there
// cap the whole scheme at 2nd order.  tside==0 or recon!=3 reduces to the 1D
// form.  Positivity falls back to the 1D (or raw) value.
__device__ void mdFaceState(CompressibleSolver &grid,
                            real *Rho, real *U, real *V, real *P,
                            i32 bIdx, i32 ii, i32 jj, i32 kk,
                            i32 d, real side, real tside, real h, real out[5]) {
  mdFaceState1D(grid, Rho, U, V, P, bIdx, ii, jj, kk, d, side, h, out);
  if (tside == 0.0 || grid.recon != 3) return;

  // face values of the two tangential neighbour rows
  i32 ti = (d == 0) ? 0 : 1, tj = 1 - ti;      // tangential unit offset
  real qm[5], qp[5];
  mdFaceState1D(grid, Rho, U, V, P, bIdx, ii - ti, jj - tj, kk, d, side, h, qm);
  mdFaceState1D(grid, Rho, U, V, P, bIdx, ii + ti, jj + tj, kk, d, side, h, qp);

  real c[5];
  for (i32 n = 0; n < 5; n++)
    c[n] = (tside > 0) ? rec3(qm[n], out[n], qp[n]) : rec3(qp[n], out[n], qm[n]);
  if (c[0] > (real)1e-10 && c[4] > (real)1e-10)
    for (i32 n = 0; n < 5; n++) out[n] = c[n];
}

// Corner flux tensor for the vertex at (ic-1/2, jc-1/2) of cell (ic, jc)
// (block-local coordinates relative to the calling thread's cell; the corner's
// owner is the (+,+) cell of its quad).  Computed on the fly -- no storage --
// so each corner is evaluated identically by up to 4 sharing cells (bitwise
// deterministic: same inputs, same expression) which keeps assembly conservative.
__device__ void mdCornerFlux(CompressibleSolver &grid,
                             real *Rho, real *U, real *V, real *W, real *P,
                              i32 bIdx, i32 ic, i32 jc, i32 k,
                             real dx, real dy, real Fx[5], real Fy[5]) {
    // the 4 cells sharing the vertex: (ic-1..ic) x (jc-1..jc)
    // sign of the corner relative to each cell's centre (sx, sy)
    const real csx[4] = { 1.0, -1.0,  1.0, -1.0};
    const real csy[4] = { 1.0,  1.0, -1.0, -1.0};

    // Per-direction corner states: the direction-i flux tensor sees the
    // momentum in its own normal component (u for F^x, v for F^y).
    real Qx[4][5], Qy[4][5];   // conservative corner states per flux direction
    real fx[4][5], fy[4][5];
    for (i32 m = 0; m < 4; m++) {
      i32 xi = ic - 1 + (m & 1), yj = jc - 1 + (m >> 1);   // quad cell coords
      real px[5], py[5];
      mdFaceState(grid, Rho, U, V, P, bIdx, xi, yj, k, 0, csx[m], csy[m], dx, px);
      mdFaceState(grid, Rho, U, V, P, bIdx, xi, yj, k, 1, csy[m], csx[m], dy, py);
      Qx[m][0] = px[0];  Qx[m][1] = px[0]*px[1];  Qx[m][2] = px[0]*px[2];  Qx[m][3] = 0.0;
      Qx[m][4] = px[4]/(gam - 1.0) + 0.5*px[0]*(px[1]*px[1] + px[2]*px[2]);
      Qy[m][0] = py[0];  Qy[m][1] = py[0]*py[1];  Qy[m][2] = py[0]*py[2];  Qy[m][3] = 0.0;
      Qy[m][4] = py[4]/(gam - 1.0) + 0.5*py[0]*(py[1]*py[1] + py[2]*py[2]);
      mdPhysFlux(px[0], px[1], px[2], px[3], px[4], 0, fx[m]);
      mdPhysFlux(py[0], py[1], py[2], py[3], py[4], 1, fy[m]);
    }

    // Green-Gauss corner gradient (times h), per direction; exact for linears
    real hgx[5], hgy[5], Qbx[5], Qby[5];
    for (i32 n = 0; n < 5; n++) {
      hgx[n] = 0.5*((Qx[1][n] - Qx[0][n]) + (Qx[3][n] - Qx[2][n]));
      hgy[n] = 0.5*((Qy[2][n] - Qy[0][n]) + (Qy[3][n] - Qy[1][n]));
      Qbx[n] = 0.25*(Qx[0][n] + Qx[1][n] + Qx[2][n] + Qx[3][n]);
      Qby[n] = 0.25*(Qy[0][n] + Qy[1][n] + Qy[2][n] + Qy[3][n]);
    }

    // mean primitives for the one-point |A_i| quadrature (per direction)
    real rbx = Qbx[0];
    real ubx = Qbx[1]/rbx, vbx = Qbx[2]/rbx, wbx = Qbx[3]/rbx;
    real pbx = (gam - 1.0)*(Qbx[4] - 0.5*rbx*(ubx*ubx + vbx*vbx + wbx*wbx));
    if (pbx < (real)1e-12) pbx = (real)1e-12;
    real rby = Qby[0];
    real uby = Qby[1]/rby, vby = Qby[2]/rby, wby = Qby[3]/rby;
    real pby = (gam - 1.0)*(Qby[4] - 0.5*rby*(uby*uby + vby*vby + wby*wby));
    if (pby < (real)1e-12) pby = (real)1e-12;

    real ax[5], ay[5];
    mdAbsJacDq(rbx, ubx, vbx, wbx, pbx, 1.0, 0.0, hgx, ax);
    mdAbsJacDq(rby, uby, vby, wby, pby, 0.0, 1.0, hgy, ay);

    // dissipation coefficient 1/2 calibrated so the 1D (y-uniform) limit of the
    // assembled face flux is exactly the Roe/Osher flux 1/2(fL+fR) - 1/2|A|dQ
    // (h grad Q reduces to the 1D jump there); 1/4 (the naive 1/|C_p| carry-over
    // from the paper's d+1-cell corners) is half-dissipative and unstable.
    for (i32 n = 0; n < 5; n++) {
      real cx = 0.0, cy = 0.0;
      for (i32 m = 0; m < 4; m++) { cx += fx[m][n]; cy += fy[m][n]; }
      Fx[n] = 0.25*cx - 0.5*ax[n];
      Fy[n] = 0.25*cy - 0.5*ay[n];
    }
}

// CTU-Hancock half-step predictor (mdFlux == 2): every cell is advanced
//   Q* = Q^n - (dt/2) [ (f(Q_r1)-f(Q_l1))/(2dx) + (g(Q_u1)-g(Q_d1))/(2dy) ]
// (central cell-flux gradients in BOTH directions) and the predicted
// PRIMITIVES are stored in the Old bank.  The whole multiD flux assembly
// (corners, midpoints, reconstruction) then runs on the predicted field, so
// every flux is time-centred at t + dt/2: Colella's corner-transport stability
// (CFL ~ 1) plus Hancock's 2nd-order-in-time accuracy, in one Euler corrector.
// Falls back to the raw state on positivity loss.  The Old bank is free during
// the RHS (updateFieldsKernel rewrites it afterwards).
__global__ void hancockPredictKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *P   = grid.getField(F_RHOE);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
    real cfx = 0.25*grid.deltaT/dx;    // (dt/2) * central-diff/(2 dx)
    real cfy = 0.25*grid.deltaT/dy;

    i32 l1 = grid.getNbrIdx(bIdx, i-1, j, k);
    i32 r1 = grid.getNbrIdx(bIdx, i+1, j, k);
    i32 d1 = grid.getNbrIdx(bIdx, i, j-1, k);
    i32 u1 = grid.getNbrIdx(bIdx, i, j+1, k);

    real r0 = Rho[cIdx], u0 = U[cIdx], v0 = V[cIdx], p0 = P[cIdx];
    real ro = r0, uo = u0, vo = v0, po = p0;   // fallback

    real fm[5], fp[5], gm[5], gp[5];
    mdPhysFlux(Rho[l1], U[l1], V[l1], 0.0, P[l1], 0, fm);
    mdPhysFlux(Rho[r1], U[r1], V[r1], 0.0, P[r1], 0, fp);
    mdPhysFlux(Rho[d1], U[d1], V[d1], 0.0, P[d1], 1, gm);
    mdPhysFlux(Rho[u1], U[u1], V[u1], 0.0, P[u1], 1, gp);

    real Q[5];
    Q[0] = r0;  Q[1] = r0*u0;  Q[2] = r0*v0;  Q[3] = 0.0;
    Q[4] = p0/(gam - 1.0) + 0.5*r0*(u0*u0 + v0*v0);
    for (i32 n = 0; n < 5; n++) Q[n] -= cfx*(fp[n] - fm[n]) + cfy*(gp[n] - gm[n]);

    real r = Q[0];
    if (r > (real)1e-10) {
      real u = Q[1]/r, v = Q[2]/r;
      real p = (gam - 1.0)*(Q[4] - 0.5*r*(u*u + v*v));
      if (p > (real)1e-10) { ro = r; uo = u; vo = v; po = p; }
    }

    grid.getField(F_OLD + F_RHO )[cIdx] = ro;
    grid.getField(F_OLD + F_RHOU)[cIdx] = uo;
    grid.getField(F_OLD + F_RHOV)[cIdx] = vo;
    grid.getField(F_OLD + F_RHOW)[cIdx] = 0.0;
    grid.getField(F_OLD + F_RHOE)[cIdx] = po;

    grid.getField(F_OLD + F_RHOK  )[cIdx] = grid.getField(F_RHOK  )[cIdx];
    grid.getField(F_OLD + F_RHOTAU)[cIdx] = grid.getField(F_RHOTAU)[cIdx];

  END_CELL_LOOP
}

// MultiD corner-flux RHS: computes the cell's 4 vertex flux tensors on the fly,
// the 4 face-midpoint 1D Osher fluxes, and assembles the faces by Simpson's
// rule (Eq. 9 generalized).  Gather form -- no atomics; Rhs was zeroed by
// updateFieldsKernel.
__global__ void multiDRhsKernel(CompressibleSolver &grid) {
  // mdFlux==2 (CTU-Hancock): every state -- corners, midpoints, reconstruction
  // stencils -- comes from the HALF-STEP-PREDICTED primitives in the Old bank,
  // so all fluxes are time-centred at t + dt/2 (2nd-order time with the single
  // Euler corrector).  Otherwise the live primitive fields.
  i32 fb = (grid.mdFlux == 2) ? F_OLD : 0;
  real *Rho = grid.getField(fb + F_RHO);
  real *U   = grid.getField(fb + F_RHOU);
  real *V   = grid.getField(fb + F_RHOV);
  real *W   = grid.getField(fb + F_RHOW);
  real *P   = grid.getField(fb + F_RHOE);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    real dx = grid.getDx(lvl), dy = grid.getDy(lvl);

    // flux tensors at this cell's 4 vertices, computed in place
    real xLL[5], yLL[5], xLR[5], yLR[5], xUL[5], yUL[5], xUR[5], yUR[5];
    mdCornerFlux(grid, Rho, U, V, W, P, bIdx, i,   j,   k, dx, dy, xLL, yLL);
    mdCornerFlux(grid, Rho, U, V, W, P, bIdx, i+1, j,   k, dx, dy, xLR, yLR);
    mdCornerFlux(grid, Rho, U, V, W, P, bIdx, i,   j+1, k, dx, dy, xUL, yUL);
    mdCornerFlux(grid, Rho, U, V, W, P, bIdx, i+1, j+1, k, dx, dy, xUR, yUR);

    // Face-midpoint 1D Osher-Solomon fluxes (the 1D member of the same Osher
    // family as the corner tensors).  Each face is evaluated identically by its
    // two sharing cells -> conservative.

    // Face-midpoint donor states via mdFaceState (raw P0, or recon==3
    // unlimited parabolic), evaluated
    // on the fb bank (predicted field under CTU-Hancock).
    real qA[5], qB[5];
    real FxLm[5], FxRm[5], FyBm[5], FyTm[5];
    {
      mdFaceState(grid, Rho, U, V, P, bIdx, i-1, j, k, 0,  1.0, 0.0, dx, qA);
      mdFaceState(grid, Rho, U, V, P, bIdx, i,   j, k, 0, -1.0, 0.0, dx, qB);
      osher1dFlux(qA[0],qA[1],qA[2],qA[3],qA[4], qB[0],qB[1],qB[2],qB[3],qB[4], 0, FxLm);
      mdFaceState(grid, Rho, U, V, P, bIdx, i,   j, k, 0,  1.0, 0.0, dx, qA);
      mdFaceState(grid, Rho, U, V, P, bIdx, i+1, j, k, 0, -1.0, 0.0, dx, qB);
      osher1dFlux(qA[0],qA[1],qA[2],qA[3],qA[4], qB[0],qB[1],qB[2],qB[3],qB[4], 0, FxRm);
      mdFaceState(grid, Rho, U, V, P, bIdx, i, j-1, k, 1,  1.0, 0.0, dy, qA);
      mdFaceState(grid, Rho, U, V, P, bIdx, i, j,   k, 1, -1.0, 0.0, dy, qB);
      osher1dFlux(qA[0],qA[1],qA[2],qA[3],qA[4], qB[0],qB[1],qB[2],qB[3],qB[4], 1, FyBm);
      mdFaceState(grid, Rho, U, V, P, bIdx, i, j,   k, 1,  1.0, 0.0, dy, qA);
      mdFaceState(grid, Rho, U, V, P, bIdx, i, j+1, k, 1, -1.0, 0.0, dy, qB);
      osher1dFlux(qA[0],qA[1],qA[2],qA[3],qA[4], qB[0],qB[1],qB[2],qB[3],qB[4], 1, FyTm);
    }

    // Simpson-rule face assembly (Balsara-style): 1/6 corner + 2/3 midpoint +
    // 1/6 corner.  The row-local midpoint solver supplies the upwind coupling
    // and dissipation that the pure corner trapezoid lacks -- in particular it
    // damps the diagonal-checkerboard mode the corner tensors are exactly
    // blind to.  1D limit stays exactly the Roe/Osher-scaled flux.
    real L[NEVOLVE] = {0};
    for (i32 n = 0; n < 5; n++) {
      const real w6 = 1.0/6.0, w23 = 2.0/3.0;
      real FxL = w6*(xLL[n] + xUL[n]) + w23*FxLm[n];   // left  face (i-1/2, j)
      real FxR = w6*(xLR[n] + xUR[n]) + w23*FxRm[n];   // right face (i+1/2, j)
      real FyB = w6*(yLL[n] + yLR[n]) + w23*FyBm[n];   // bottom face (i, j-1/2)
      real FyT = w6*(yUL[n] + yUR[n]) + w23*FyTm[n];   // top    face (i, j+1/2)
      L[n] = (FxL - FxR)/dx + (FyB - FyT)/dy;
    }

    if (grid.mdFlux == 2) {
      // CTU-Hancock fused single-stage corrector: all neighbour data came from
      // the PREDICTED bank (fb = F_OLD), so the live fields can be updated in
      // place -- convert this cell's live primitives to conservative, apply
      // q^{n+1} = q^n + dt L, and store CONSERVATIVE (step() skips the usual
      // primitiveToConservative/updateFields for mdFlux==2).  The accumulator
      // bank is never touched: it holds the predicted states.
      real dt = grid.deltaT;
      real rl = grid.getField(F_RHO)[cIdx],  ul = grid.getField(F_RHOU)[cIdx];
      real vl = grid.getField(F_RHOV)[cIdx], pl = grid.getField(F_RHOE)[cIdx];
      grid.getField(F_RHO )[cIdx] = rl + dt*L[0];
      grid.getField(F_RHOU)[cIdx] = rl*ul + dt*L[1];
      grid.getField(F_RHOV)[cIdx] = rl*vl + dt*L[2];
      grid.getField(F_RHOW)[cIdx] = 0.0;
      grid.getField(F_RHOE)[cIdx] = pl/(gam - 1.0) + 0.5*rl*(ul*ul + vl*vl) + dt*L[4];
    } else {
      // LSRK: accumulate L onto the pre-scaled accumulator bank
      for (i32 n = 0; n < NEVOLVE; n++)
        grid.getField(F_RHS + n)[cIdx] += L[n];
    }

  END_CELL_LOOP
}

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage) {
  //
  // Low-storage (Williamson 2N) RK3 update of the NEVOLVE evolved DOFs.
  // The RHS kernels ACCUMULATE L onto the pre-scaled bank, so on entry the
  // accumulator holds  S_i = A_i S_{i-1} + L(q_{i-1});  here
  //   q += B_i dt S_i,   then   S *= A_{i+1}
  // (A_1 = 0 is realized by zeroing S after the last stage).  Any 3-stage
  // 3rd-order explicit RK shares the linear stability polynomial of the
  // previous Shu-Osher SSP-RK3, so all measured CFL limits are unchanged;
  // only the formal SSP property is given up.
  //
  // rkScheme 0: Williamson 2N LSRK3 (q += B dt S, S carried between stages).
  // rkScheme 1/2: Jameson, q_k = q_n + alpha_k dt L_{k-1}.  Bw then holds
  // alpha directly and EVERY Anext is 0, which zeroes the accumulator after
  // each stage so the next stage's RHS accumulates a pure L_k -- no separate
  // zeroing kernel, and the q_n snapshot is taken below at stage 0.
  real Bw[5]    = {(real)(1.0/3.0), (real)(15.0/16.0), (real)(8.0/15.0), 0, 0};
  real Anext[5] = {(real)(-5.0/9.0), (real)(-153.0/128.0), 0, 0, 0};
  if (grid.rkScheme == 1) {          // Jameson 4-stage, alpha = (1/4,1/3,1/2,1)
    Bw[0] = (real)0.25; Bw[1] = (real)(1.0/3.0); Bw[2] = (real)0.5; Bw[3] = (real)1;
    Anext[0] = Anext[1] = Anext[2] = Anext[3] = 0;
  } else if (grid.rkScheme == 2) {   // Jameson 5-stage, alpha = (1/4,1/6,3/8,1/2,1)
    Bw[0] = (real)0.25; Bw[1] = (real)(1.0/6.0); Bw[2] = (real)0.375;
    Bw[3] = (real)0.5;  Bw[4] = (real)1;
    Anext[0] = Anext[1] = Anext[2] = Anext[3] = Anext[4] = 0;
  }
  const real dtG = grid.deltaT;

  START_LIVE_CELL_LOOP

    // Local time stepping: this cell's own step, rescaled by the same factor the
    // host applied when clamping the global step onto an output time.
    const real dt = grid.lts ? grid.getField(F_DTL)[cIdx]*grid.dtScale : dtG;

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    // Cells the body intersects or contains carry NO solution (UTCart's rule).
    // Freezing them at their initial freestream state keeps every stencil that
    // reaches into the body finite, and the fluid/solid faces are boundary
    // fluxes anyway, so nothing physical is read from them.
    bool ibSolid = false;
    if (grid.immerserdBcType != 0)
      ibSolid = grid.getField(F_IBM)[cIdx] <= (real)0.5;   // cached mask
    // RCCM: R-Cells get their new value from the reconstruction (Eq. 10), never
    // from the FVM update -- advancing them is exactly the small-cell blow-up
    // the method exists to avoid.
    if (grid.ibRccm && grid.immerserdBcType != 0)
      ibSolid = !grid.rccmLive(cIdx);      // advance every live cell
    // merged element: only the OWNER holds the DOF; members are refreshed from
    // it by cutBroadcastKernel right after this kernel
    if (grid.cutMerge && grid.cutOwner[cIdx] != cIdx) ibSolid = true;

    // Covered parents are NOT evolved -- restrictFields overwrites them from
    // their children after every stage, so their only legitimate source is
    // restriction and the RK update is dead work.  At an immersed wall it was
    // UNSTABLE dead work: the parent-level wall-face flux (parent-geometry
    // d_FC / image point against a child-averaged state) is exactly where the
    // bounded-but-garbage parent transients came from.  The accumulator
    // pre-scale below still runs for every cell, which keeps the shared
    // bank's between-step roles intact (it ends the step at zero).
    if (grid.isInteriorBlock(lvl, ib, jb, kb) && !ibSolid
        && grid.cFlagsList[cIdx] != PARENT) {

      // Point-implicit wall flux: at a wall-adjacent cell the RHS stamped the
      // diagonal relaxation rate of the stiff k~/tau~ wall flux; dividing the
      // update by (1 + B dt lambda) bounds it for ANY dt while leaving every
      // fixed point of the RHS -- hence the converged solution -- unchanged.
      real lamK = 0, lamT = 0;
      if (grid.rans && grid.wallPointImplicit) {
        lamK = grid.getField(F_LAMK)[cIdx];
        lamT = grid.getField(F_LAMT)[cIdx];
      }
      // Brinkman porosity stiffness, stamped by the RHS in the band.  Read and
      // CLEARED here (not in the sweep below) because a stale stamp must never
      // survive a stage: the rate is a per-stage quantity.
      real lamM = 0;
      if (grid.cutPiDamp) {
        lamM = grid.getField(F_LAMM)[cIdx];
        grid.getField(F_LAMM)[cIdx] = 0;
      }
      // Low-Mach preconditioning of the mean-flow residual (rank-one; steady
      // states are fixed points of P R exactly as they are of R).
      real Rm[5];
      if (grid.precond) {
        for (i32 f = 0; f < 5; f++) Rm[f] = grid.getField(F_RHS + f)[cIdx];
        const real r  = fmax(grid.getField(F_RHO)[cIdx], (real)1e-30);
        const real u  = grid.getField(F_RHOU)[cIdx]/r;
        const real v  = grid.getField(F_RHOV)[cIdx]/r;
        const real w  = grid.getField(F_RHOW)[cIdx]/r;
        const real E  = grid.getField(F_RHOE)[cIdx];
        const real q2 = u*u + v*v + w*w;
        const real p  = (gam - (real)1)*(E - (real)0.5*r*q2);
        const real c2 = gam*fmax(p,(real)1e-30)/r;
        const real Hh = (E + p)/r;
        precondResidual(Rm, u, v, w, Hh, c2, precondBeta2(grid, q2, c2));
      }
      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q = grid.getField(f);
        real *S = grid.getField(F_RHS + f);
        real fac = 1;
        if (f == F_RHOK   && lamK > 0) fac = (real)1/((real)1 + Bw[stage]*dt*lamK);
        if (f == F_RHOTAU && lamT > 0) fac = (real)1/((real)1 + Bw[stage]*dt*lamT);
        // The stiff excess (w_f - 1) multiplies the mass and energy fluxes
        // just as it does the momentum ones, so the same diagonal applies to
        // all five mean-flow rows.
        if (lamM > 0 && f < 5)
          fac = (real)1/((real)1 + Bw[stage]*dt*lamM);
        const real Sv = (grid.precond && f < 5) ? Rm[f] : S[cIdx];
        if (grid.rkScheme != 0) {
          // Jameson: every stage restarts from q_n.  At stage 0 the field still
          // holds q_n (the RHS only read it), so snapshot it here.
          real *QN = grid.getField(F_QN + f);
          if (stage == 0) QN[cIdx] = Q[cIdx];
          Q[cIdx] = QN[cIdx] + fac * Bw[stage] * dt * Sv;
        } else {
          Q[cIdx] += fac * Bw[stage] * dt * Sv;
        }
      }

      // --p1: the slope DOFs take the same LSRK stage (rkScheme 0 only)
      if (grid.p1)
        for (i32 s = 0; s < 2*P1_NV; s++)
          grid.getField(F_P1S + s)[cIdx] += Bw[stage]*dt*grid.getField(F_P1SR + s)[cIdx];

      // pseudo2D: z-momentum is never evolved
      if (grid.pseudo2D) {
        grid.getField(F_RHOW)[cIdx] = 0;
      }
    }

    // pre-scale the accumulator for the next stage (0 after the last stage:
    // that is the next step's A_1 = 0, and it leaves the bank clean for its
    // between-step roles -- wavelet snapshot / sort buffer / Hancock predictor)
    for (i32 f = 0; f < NEVOLVE; f++)
      grid.getField(F_RHS + f)[cIdx] *= Anext[stage];
    if (grid.p1)
      for (i32 s = 0; s < 2*P1_NV; s++)
        grid.getField(F_P1SR + s)[cIdx] *= Anext[stage];
    // consume the wall-flux relaxation rates: they are per-stage quantities,
    // and this also clears stamps on cells that skipped the update (parents)
    if (grid.rans && grid.wallPointImplicit) {
      real *LK = grid.getField(F_LAMK), *LT = grid.getField(F_LAMT);
      if (LK[cIdx] != 0) LK[cIdx] = 0;
      if (LT[cIdx] != 0) LT[cIdx] = 0;
    }
    if (grid.cutPiDamp) {
      real *LM = grid.getField(F_LAMM);
      if (LM[cIdx] != 0) LM[cIdx] = 0;      // parents / skipped cells
    }

  END_CELL_LOOP
}

#ifdef USE_MGPU
// Rebuild the partition ghost layer after an owned-only adaptGrid: for each owned
// block, activate the 2-block ring of same-level neighbor-owned blocks that the
// owning PE actually has (queried from its hash).  Iterating owned blocks at all
// levels yields both the fine seam ghosts and, through each owned block's own
// ring, the coarser parent-level ghosts the wavelet DD stencil needs.  Because it
// mirrors the neighbors' CURRENT refinement and is rebuilt from scratch each
// adaptation, the ghost layer prunes automatically as features move.
// flag current ghosts (interior, not owned) for deletion; keep owned + exterior.
// rebuildGhostsKernel then un-deletes (atomicMax NEW) the ones still needed.
// (keepLocalSupportKernel was removed: under the owned-target + NEED/adopt
// protocol every needed non-owned block is directory-backed by its owner --
// locally-manufactured orphans no longer exist, and un-deleting a ghost no
// directory names would keep a permanently-unfillable zero block alive.)

__global__ void markGhostsKernel(CompressibleSolver &grid) {
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      bool ghost = grid.isInteriorBlock(lvl, ib, jb, kb) && !grid.isOwnedBlock(lvl, ib, jb, kb);
      grid.bFlagsList[bIdx] = ghost ? DELETE : KEEP;
    }
  END_BLOCK_LOOP
}

// --- directory-based halo (message-passing; no peer-memory access) ---------
// Distinct neighbor slots whose territory this owned block's 2-ring reaches.
__device__ i32 blockNbrSlots(CompressibleSolver &grid, i32 lvl, i32 ib, i32 jb, i32 kb, i32 *slot) {
  // 1-block ring, corners included (Chebyshev-1).  Sufficient for blockSize>=4:
  // the widest read is +-2 CELLS (flux) / the parent 27-tap (parent is same base
  // column -> owned, only its 1-ring is a ghost) -- both stay within +-1 block.
  i32 n = 0, dkLim = grid.pseudo2D ? 0 : 1;
  for (i32 dk=-dkLim; dk<=dkLim; dk++)
  for (i32 dj=-1; dj<=1; dj++)
  for (i32 di=-1; di<=1; di++) {
    i32 ni=ib+di, nj=jb+dj, nk=kb+dk;
    // A 2-ring position past the domain edge is, under periodic BCs, the
    // wrap-around image on the far side -- which may be owned by another PE.
    // Wrapping it here makes an edge block's directory reach that PE, so the
    // periodic image blocks the far side's setBoundaryConditions needs become
    // ghosts on it (else the wrap-image hash lookup misses -> NaN across ranks).
    if (!grid.isInteriorBlock(lvl,ni,nj,nk)) {
      if (!grid.periodic) continue;
      grid.wrapBlockPeriodic(lvl, ni, nj, nk);
    }
    i32 o = grid.ownerPE(lvl,ni,nj,nk);
    if (o == grid.part.rank) continue;
    i32 s = grid.nbrOf[o];
    if (s < 0) continue;
    bool seen = false;
    for (i32 t=0; t<n; t++) if (slot[t]==s) { seen=true; break; }
    if (!seen && n < 27) slot[n++] = s;
  }
  return n;
}

// Count / fill the per-neighbor directory of this PE's owned boundary blocks
// (loc codes of owned blocks whose 2-ring reaches into a neighbor's territory).
__global__ void countDirKernel(CompressibleSolver &grid) {
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl,ib,jb,kb; grid.decode(loc,lvl,ib,jb,kb);
      // Publish owned SURVIVING blocks to every neighbour whose 2-ring reaches
      // them (they become ghosts there).  The flag filter is essential: a block
      // thresholding marked DELETE (to be coarsened) must NOT be exported -- the
      // neighbour would import it as a KEEP ghost, grade from it, and NEED its
      // ring back to us, resurrecting the region every cycle (a one-way ratchet
      // to a uniformly-dense grid).  Legitimate cross-seam support is instead
      // revived explicitly: a NEED/adopt touch raises the flag above DELETE.
      if (grid.isOwnedBlock(lvl,ib,jb,kb) && grid.bFlagsList[bIdx] != DELETE) {
        i32 slot[27]; i32 n = blockNbrSlots(grid,lvl,ib,jb,kb,slot);
        for (i32 t=0; t<n; t++) atomicAdd(&grid.dirSendCnt[slot[t]], 1);
      }
    }
  END_BLOCK_LOOP
}

__global__ void fillDirKernel(CompressibleSolver &grid) {
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl,ib,jb,kb; grid.decode(loc,lvl,ib,jb,kb);
      if (grid.isOwnedBlock(lvl,ib,jb,kb) && grid.bFlagsList[bIdx] != DELETE) {   // survivors only (see countDirKernel)
        i32 slot[27]; i32 n = blockNbrSlots(grid,lvl,ib,jb,kb,slot);
        for (i32 t=0; t<n; t++) {
          i32 s = slot[t], i = atomicAdd(&grid.dirFill[s], 1);
          if (i < grid.dirSlot) grid.dirSendLoc[(size_t)s*grid.dirSlot + i] = loc;
        }
      }
    }
  END_BLOCK_LOOP
}

// Create ghosts from received neighbor directories: for each received boundary
// block, if this PE has an owned block within its 2-ring, activate it as a ghost.
// Adopt the neighbours' NEED lists: each received loc is a support block the
// sender's closure required in OUR territory (owned-target rule forbade the
// sender creating it).  We create it as an owned block; the same exchange's
// directory pass then exports it back to the sender as a ghost, and its data is
// prolonged from our (valid) parent level by reconstituteOldSnapshot.
__global__ void consumeNeedKernel(CompressibleSolver &grid) {
#ifdef USE_MGPU
  i32 slot = grid.needSlot, nN = grid.nNbr;
  for (i64 tid = (i64)blockIdx.x*blockDim.x+threadIdx.x; tid < (i64)nN*slot; tid += (i64)gridDim.x*blockDim.x) {
    i32 s = tid / slot, i = tid % slot;
    if (i >= min(grid.needRecvCnt[s], slot)) continue;
    u64 loc = grid.needRecvLoc[(size_t)s*slot + i];
    i32 lvl,ni,nj,nk; grid.decode(loc,lvl,ni,nj,nk);
    if (grid.isInteriorBlock(lvl,ni,nj,nk) && grid.isOwnedBlock(lvl,ni,nj,nk))
      grid.activateBlock(lvl, ni, nj, nk);
  }
#endif
}

__global__ void consumeDirKernel(CompressibleSolver &grid) {
  i32 slot = grid.dirSlot, nN = grid.nNbr, dkLim = grid.pseudo2D ? 0 : 1;
  for (i64 tid = (i64)blockIdx.x*blockDim.x+threadIdx.x; tid < (i64)nN*slot; tid += (i64)gridDim.x*blockDim.x) {
    i32 s = tid / slot, i = tid % slot;
    if (i >= grid.dirRecvCnt[s]) continue;
    u64 loc = grid.dirRecvLoc[(size_t)s*slot + i];
    i32 lvl,ni,nj,nk; grid.decode(loc,lvl,ni,nj,nk);
    bool want = false;
    for (i32 dk=-dkLim; dk<=dkLim && !want; dk++)
    for (i32 dj=-1; dj<=1 && !want; dj++)
    for (i32 di=-1; di<=1 && !want; di++) {
      i32 ai=ni+di, aj=nj+dj, ak=nk+dk;
      // periodic: a 2-ring position past the edge wraps to the far side (see
      // blockNbrSlots) -- so this received block is my periodic image if an
      // owned block sits within its wrapped 2-ring.
      if (!grid.isInteriorBlock(lvl,ai,aj,ak)) {
        if (!grid.periodic) continue;
        grid.wrapBlockPeriodic(lvl, ai, aj, ak);
      }
      if (grid.ownerPE(lvl,ai,aj,ak)==grid.part.rank
          && grid.getBlockIdx(grid.encode(lvl,ai,aj,ak)) != bEmpty)   // validated: corpses don't count
        want = true;
    }
    // store the ghost under the sender's TRUE interior code so the far side's
    // periodic wrap-image lookup (setBoundaryConditions) resolves to it locally.
    if (want) grid.activateBlock(lvl, ni, nj, nk);
  }
}

// Pack this PE's directory blocks for each neighbor into the send buffer,
// contiguous per neighbor as [block][field][cell]; index resolved by loc lookup.
__global__ void packDirKernel(CompressibleSolver &grid, i32 fOff, i32 nf) {
  i32 slot = grid.dirSlot, nN = grid.nNbr, bst = blockSizeTot, fs = nf*bst;
  for (i64 tid = (i64)blockIdx.x*blockDim.x+threadIdx.x; tid < (i64)nN*slot*bst; tid += (i64)gridDim.x*blockDim.x) {
    i32 cell = tid % bst, i = (tid / bst) % slot, s = tid / bst / slot;
    if (i >= grid.dirSendCnt[s]) continue;
    // validated lookup: after the D5 tail reorder the directories are always
    // fresh, so a corpse here is a protocol bug -- ship zeros, never stale memory
    i32 blk = grid.getBlockIdx(grid.dirSendLoc[(size_t)s*slot + i]);
    real *dst = grid.sendBuf + ((size_t)s*slot + i)*fs;
    if (blk == bEmpty) { for (i32 f=0; f<nf; f++) dst[f*bst+cell] = 0.0; continue; }
    for (i32 f=0; f<nf; f++) dst[f*bst+cell] = grid.getField(fOff+f)[(size_t)blk*bst+cell];
  }
}

// Unpack received neighbor data into this PE's ghost blocks: each received
// directory entry this PE holds as a (non-owned) ghost gets copied from recvBuf.
__global__ void unpackDirKernel(CompressibleSolver &grid, i32 fOff, i32 nf) {
  i32 slot = grid.dirSlot, nN = grid.nNbr, bst = blockSizeTot, fs = nf*bst;
  for (i64 tid = (i64)blockIdx.x*blockDim.x+threadIdx.x; tid < (i64)nN*slot*bst; tid += (i64)gridDim.x*blockDim.x) {
    i32 cell = tid % bst, i = (tid / bst) % slot, s = tid / bst / slot;
    if (i >= grid.dirRecvCnt[s]) continue;
    u64 loc = grid.dirRecvLoc[(size_t)s*slot + i];
    i32 ghost = grid.getBlockIdx(loc);   // validated: never unpack into a corpse slot
    if (ghost == bEmpty) continue;
    i32 lvl,ib,jb,kb; grid.decode(loc,lvl,ib,jb,kb);
    if (grid.isOwnedBlock(lvl,ib,jb,kb)) continue;   // only my ghosts
    real *src = grid.recvBuf + ((size_t)s*slot + i)*fs;
    for (i32 f=0; f<nf; f++) grid.getField(fOff+f)[(size_t)ghost*bst+cell] = src[f*bst+cell];
  }
}
#endif

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP

    for (i32 f = 0; f < NEVOLVE; f++)
      grid.getField(F_OLD + f)[cIdx] = grid.getField(f)[cIdx];
    // this block now carries a valid F_OLD snapshot; blocks created afterwards
    // (refinement/support/imported ghosts) are marked 0 by activateBlock and get
    // their F_OLD reconstituted (halo + hierarchical interpolation) before the inverse
    if (cIdx % blockSizeTot == 0) grid.snapValidList[bIdx] = 1;

  END_CELL_LOOP
}

//
// 3D second-order interpolating-wavelet prediction of a child cell value from
// its parent block (trilinear Deslauriers-Dubuc stencil).
//
__device__ real waveletPredict(MultiLevelSparseGrid &grid, real *Q, i32 prntIdx,
                               i32 ip, i32 jp, i32 kp, real xs, real ys, real zs) {
  i32 p   = grid.getNbrIdx(prntIdx, ip,   jp,   kp);
  i32 l   = grid.getNbrIdx(prntIdx, ip-1, jp,   kp);
  i32 r   = grid.getNbrIdx(prntIdx, ip+1, jp,   kp);
  i32 d   = grid.getNbrIdx(prntIdx, ip,   jp-1, kp);
  i32 u   = grid.getNbrIdx(prntIdx, ip,   jp+1, kp);
  i32 b   = grid.getNbrIdx(prntIdx, ip,   jp,   kp-1);
  i32 f   = grid.getNbrIdx(prntIdx, ip,   jp,   kp+1);

  i32 lu  = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp);
  i32 ru  = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp);
  i32 ld  = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp);
  i32 rd  = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp);

  i32 lb  = grid.getNbrIdx(prntIdx, ip-1, jp,   kp-1);
  i32 rb  = grid.getNbrIdx(prntIdx, ip+1, jp,   kp-1);
  i32 lf  = grid.getNbrIdx(prntIdx, ip-1, jp,   kp+1);
  i32 rf  = grid.getNbrIdx(prntIdx, ip+1, jp,   kp+1);

  i32 db  = grid.getNbrIdx(prntIdx, ip,   jp-1, kp-1);
  i32 ub  = grid.getNbrIdx(prntIdx, ip,   jp+1, kp-1);
  i32 df  = grid.getNbrIdx(prntIdx, ip,   jp-1, kp+1);
  i32 uf  = grid.getNbrIdx(prntIdx, ip,   jp+1, kp+1);

  i32 ruf = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp+1);
  i32 luf = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp+1);
  i32 rdf = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp+1);
  i32 ldf = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp+1);
  i32 rub = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp-1);
  i32 lub = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp-1);
  i32 rdb = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp-1);
  i32 ldb = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp-1);

  return Q[p]
       + xs/8.0*(Q[r]-Q[l]) + ys/8.0*(Q[u]-Q[d]) + zs/8.0*(Q[f]-Q[b])
       + xs*ys/64.0*(Q[ru]-Q[lu]-Q[rd]+Q[ld])
       + xs*zs/64.0*(Q[rf]-Q[lf]-Q[rb]+Q[lb])
       + ys*zs/64.0*(Q[uf]-Q[ub]-Q[df]+Q[db])
       + xs*ys*zs/512.0*(Q[ruf]-Q[luf]-Q[rdf]+Q[ldf]-Q[rub]+Q[lub]+Q[rdb]-Q[ldb]);
}

//
// Predict evolved field f at a child cell from its parent block by the
// interpolating-wavelet (Deslauriers-Dubuc) stencil.  Every evolved field is a
// P0 cell average, so one prolongation serves them all.
//
__device__ real
predictEvolvedField(CompressibleSolver &grid, i32 baseOff, i32 f, i32 prntIdx,
                    i32 ip, i32 jp, i32 kp, real xs, real ys, real zs, i32 lvl) {
  real *Q = grid.getField(baseOff + f);
  return waveletPredict(grid, Q, prntIdx, ip, jp, kp, xs, ys, zs);
}

__global__ void forwardWaveletTransformKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      real xs = 2*(i % 2) - 1;
      real ys = 2*(j % 2) - 1;
      real zs = grid.pseudo2D ? 0.0 : (2*(k % 2) - 1);

      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = Q[cIdx] - predictEvolvedField(grid, F_OLD, f, prntIdx, ip, jp, kp, xs, ys, zs, lvl);
      }
    }
    else if (lvl > 0 && cFlag == GHOST) {
      // zero the detail of coarse/fine GHOST cells so they carry no stale
      // coefficient into the transform; the inverse restores them by prediction.
      // ONLY at lvl>0: a level-0 cell has no wavelet detail and no inverse to
      // restore it, so zeroing an (authoritative, halo/BC-filled) level-0 GHOST
      // cell would destroy its data permanently -- which is what produced the
      // transient boundary vacuum right after a domain-edge column migration.
      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = 0.0;
      }
    }

  END_CELL_LOOP
}

//
// Diagnostic: per-cell normalized wavelet-detail indicator into F_SCRATCH,
// exactly as waveletThresholdingKernel would see it (max over the primary
// fields of |Q - predict| / globalScale), but predicting from the LIVE fields
// (baseOff 0) so nothing is mutated.  Saturated at the refine threshold
// (2*waveletThresh): a white pixel in the painted image = would refine.
// mode: 0 = max over primary fields, 1 = rho only, 2 = momentum, 3 = rhoE.
//
__global__ void detailToScratchKernel(CompressibleSolver &grid, i32 mode) {
  real *S = grid.getField(F_SCRATCH);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    S[cIdx] = 0.0;
    i32 cFlag = grid.cFlagsList[cIdx];
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      real xs = 2*(i % 2) - 1;
      real ys = 2*(j % 2) - 1;
      real zs = grid.pseudo2D ? 0.0 : (2*(k % 2) - 1);

      real ind = 0.0;
      for (i32 f = 0; f < NEVOLVE; f++) {
        if (f >= F_RHOK) continue;                     // match thresholding: primary fields only
        if (grid.pseudo2D && f == F_RHOW) continue;
        if (mode == 1 && f != F_RHO)  continue;
        if (mode == 2 && !(f >= F_RHOU && f <= F_RHOW)) continue;
        if (mode == 3 && f != F_RHOE) continue;
        real *Q = grid.getField(f);
        real pred = predictEvolvedField(grid, 0, f, prntIdx, ip, jp, kp, xs, ys, zs, lvl);
        i32 sc = (f == F_RHO) ? 0 : (f <= F_RHOW ? 1 : 2);
        real mag = fmax(grid.globalScale[sc], (real)1e-32);
        ind = fmax(ind, fabs((Q[cIdx] - pred)/mag));
      }
      real trig = (real)2.0 * grid.waveletThresh;
      S[cIdx] = fmin(ind / trig, (real)1.0);   // 1.0 == refine trigger
    }

  END_CELL_LOOP
}

// Reconstitute the F_OLD wavelet snapshot for blocks created this cycle (new
// refinement / support / imported ghosts) that never went through copyToOld.
// Run per level, coarse->fine: a new block's snapshot is the smooth wavelet
// prediction from its parent (one level coarser, already valid), and it carries
// no detail (Q=0).  Only OWNED new blocks are filled here; ghosts get the
// owner's reconstituted snapshot from the F_OLD halo between levels.  Without
// this a new block's slot holds stale garbage that the inverse would read
// through the reconstruction stencil (single-GPU is safe because a MISSING block
// resolves to the zeroed trash slice; an imported ghost is a live slot).
__global__ void fillOldSnapshotKernel(CompressibleSolver &grid, i32 level) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl == level && lvl > 0 && grid.snapValidList[bIdx] == 0
        && grid.isInteriorBlock(lvl, ib, jb, kb) && grid.bFlagsList[bIdx] != DELETE
#ifdef USE_MGPU
        && grid.isOwnedBlock(lvl, ib, jb, kb)
#endif
       ) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      if (prntIdx == bEmpty) continue;   // no parent -> stay zero (single-GPU zero-block behaviour)
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      real xs = 2*(i % 2) - 1;
      real ys = 2*(j % 2) - 1;
      real zs = grid.pseudo2D ? 0.0 : (2*(k % 2) - 1);
      for (i32 f = 0; f < NEVOLVE; f++) {
        // wavelet prolongation from the parent (whose F_OLD -- and neighbour
        // taps, via the freshly-rebuilt index lists -- is up to date).  The live
        // field stays 0 (a new block carries no detail); the inverse then adds
        // this same prediction, so Q lands on the prolonged value exactly as it
        // would on a single GPU.
        grid.getField(F_OLD + f)[cIdx] = predictEvolvedField(grid, F_OLD, f, prntIdx, ip, jp, kp, xs, ys, zs, lvl);
        grid.getField(f)[cIdx] = 0.0;
      }
      if (cIdx % blockSizeTot == 0) grid.snapValidList[bIdx] = 1;   // valid parent for the next level
    }

  END_CELL_LOOP
}

__global__ void inverseWaveletTransformKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && grid.bFlagsList[bIdx] != DELETE) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      real xs = 2*(i % 2) - 1;
      real ys = 2*(j % 2) - 1;
      real zs = grid.pseudo2D ? 0.0 : (2*(k % 2) - 1);

      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = Q[cIdx] + predictEvolvedField(grid, F_OLD, f, prntIdx, ip, jp, kp, xs, ys, zs, lvl);
      }
    }

  END_CELL_LOOP
}

// Virtual level-(-1) detail on the base grid.  Level 0 is the coarsest grid: it
// has no parent to predict from, hence no wavelet detail and no inverse -- so on
// its own it would have to be refined everywhere (a permanently dense level 1).
// Instead synthesize a coarse prediction of a level-0 cell as the simple average
// of its surrounding 5x5x5 box of level-0 cells (5x5 in pseudo-2D).  A symmetric
// box average is exact for a linear field, so smooth flow yields ~0 detail while
// curvature / a discontinuity yields a large one.  The returned signed deviation
// (value - average) is used ONLY to decide whether to refine level 0 -> level 1;
// it is never written back into the authoritative level-0 live data.  The +-2
// reach lands one block away (haloSize 2 < blockSize 4) so every tap resolves
// through getNbrIdx; a missing tap (the bEmpty trash slice, e.g. a domain corner)
// is skipped so the mean is taken over present cells only.
__device__ real virtualDetailLevel0(CompressibleSolver &grid, i32 f, i32 bIdx,
                                     i32 i, i32 j, i32 k) {
  real *Q = grid.getField(f);
  real self = Q[grid.getNbrIdx(bIdx, i, j, k)];
  i32 dkLim = grid.pseudo2D ? 0 : 2;
  real sum = 0.0;
  i32 cnt = 0;
  for (i32 dk = -dkLim; dk <= dkLim; dk++) {
    for (i32 dj = -2; dj <= 2; dj++) {
      for (i32 di = -2; di <= 2; di++) {
        i32 nIdx = grid.getNbrIdx(bIdx, i+di, j+dj, k+dk);
        if (nIdx / blockSizeTot == bEmpty) continue;   // missing tap -> skip
        sum += Q[nIdx];
        cnt++;
      }
    }
  }
  return self - (cnt > 0 ? sum / (real)cnt : self);
}

__global__ void waveletThresholdingKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    // Dynamic (wavelet) grids force only level 0 dense; level 1 is now adaptive,
    // driven by the virtual-(-1) detail below.  Static grids keep the old dense
    // level 0 AND 1 (their level 1 is built full at init, see the refineBase
    // branch) so those tests stay byte-identical.
    bool keepCoarse = grid.staticGrid ? (lvl < 2) : (lvl < 1);
#ifdef USE_MGPU
    keepCoarse = keepCoarse && grid.isOwnedBlock(lvl, ib, jb, kb);   // owned coarse only
    // no blanket ghost-KEEP: the ghost layer is rebuilt each cycle by the seam
    // sync (create+delete propagation, both directions), and the F_OLD the
    // inverse reads is rebuilt by reconstituteOldSnapshot
#endif
    if (keepCoarse) {
      grid.bFlagsList[bIdx] = KEEP;
    }

    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
    real dx = min(grid.getDx(lvl), min(grid.getDy(lvl), grid.getDz(lvl)));
    real ls = grid.getBoundaryLevelSet(pos);

    bool refineHere = (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb));
#ifdef USE_MGPU
    // only owned blocks drive detail-based refinement (and KEEP their parent).
    // Ghosts get no flag -> adaptGrid deletes them; rebuildGhosts recreates the
    // exact 2-ring from the neighbors' real blocks (so it prunes as features move).
    refineHere = refineHere && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    // ---- wall-model band -----------------------------------------------
    // Hold everything within wallFineBand of the wall at the FINEST level, so
    // the local dy near the wall IS the finest dy and d_cutoff (one global
    // length) agrees with the image-point distance (3 * local dy) by
    // construction.  Sits OUTSIDE the detail-driven branch below because that
    // one starts at level 1, and the band has to lift level 0 too.
    // wallFineBand exists so the RANS wall model's d_cutoff and d_IP refer to
    // ONE dy -- a MODEL-consistency need.  An inviscid run has neither, so the
    // 8*dCutoff = 24-fine-cell forced band was pure waste there (it was the
    // big yellow blob dominating the transonic grid).
    bool wallBand = (grid.wallGeom != 0) && (grid.rans || grid.ibWmles)
                 && grid.isInteriorBlock(lvl, ib, jb, kb);
#ifdef USE_MGPU
    wallBand = wallBand && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    // ---- immersed boundary ------------------------------------------------
    // The block-level IB ring criterion was REMOVED (user's call, 2026-08-26).
    // It forced every block the surface cut to the finest level and activated a
    // 3x3 block ring around it.  Both jobs are already covered:
    //   - the wavelet criterion below carries its own |phi| < dx term, which
    //     refines every CELL within one cell of the surface (a tighter, per-cell
    //     footprint than the per-block test it replaced);
    //   - the 2:1 grading and reconstruction-support closure
    //     (addAdjacentBlocksKernel / addReconstructionBlocksKernel) already
    //     create the neighbour blocks the halo stencils need.
    // Watch the ibWallFlux fail census (--debug) if this is ever revisited: a
    // missing same-level neighbour shows up there as ipSample failures.

    // Two-sided surface distance for immersed walls: wallDistance() clips to 0
    // INSIDE the body, so every body-interior cell would pass the band test and
    // the whole body gets held at the finest level.  Invisible when the body is
    // a few cells thin (the aligned-plate gates), catastrophic when it isn't:
    // the 30-deg inclined plate's body is 46% of the domain, measured 22k of
    // 31k blocks pinned at finest with the run 9x the aligned cost.  |ls| keeps
    // the band a BAND on both sides; the fluid side is unchanged.
    const real dWb = (grid.immerserdBcType != 0) ? fabs(ls) : grid.wallDistance(pos);
    if (wallBand && dWb < grid.wallFineBand) {
      grid.bFlagsList[bIdx] = KEEP;
      if (lvl > 0) grid.bFlagsList[grid.prntIdxList[bIdx]] = KEEP;
      if (lvl < grid.nLvls-1) {
        i32 bSize = blockSize/2;
        i32 kc = grid.pseudo2D ? kb : (2*kb + k/bSize);
        grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, kc);
      }
    }

    if (refineHere) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      grid.bFlagsList[prntIdx] = KEEP;

      if (grid.staticGrid == 3) {
        // single planar interface at the domain centre: fine for x > cx, coarse
        // for x < cx.  One clean coarse/fine face at x=cx for the acoustic-
        // reflection test (a wave launched in the coarse half crosses it).
        real cx = 0.5*grid.domainSize[0];
        if (pos[0] > cx) {
          grid.bFlagsList[bIdx] = KEEP;
          if (lvl < grid.nLvls-1) {
            i32 bSize = blockSize/2;
            i32 kc = grid.pseudo2D ? kb : (2*kb + k/bSize);
            grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, kc);
          }
        }
      }
      else if (grid.staticGrid) {
        // fixed refinement in nested shells about the domain centre, finest at the
        // centre: level L occupies d < refineRadius*(nLvls-L)/(nLvls-1), so each
        // coarse/fine interface sits at a shrinking distance.  staticGrid==1 uses
        // the radial distance (a vortex core); ==2 uses |x-centre| (a planar band,
        // for a shock crossing an x-normal interface).  Independent of the solution.
        real cx = 0.5*grid.domainSize[0], cy = 0.5*grid.domainSize[1];
        real d  = (grid.staticGrid == 2) ? fabs(pos[0]-cx)
                  : sqrt((pos[0]-cx)*(pos[0]-cx) + (pos[1]-cy)*(pos[1]-cy));
        real invN   = 1.0 / (real)(grid.nLvls - 1);
        real Rkeep  = grid.refineRadius * (real)(grid.nLvls - lvl)     * invN;
        real Rchild = grid.refineRadius * (real)(grid.nLvls - 1 - lvl) * invN;
        if (d < Rkeep) grid.bFlagsList[bIdx] = KEEP;
        if (lvl < grid.nLvls-1 && d < Rchild) {
          i32 bSize = blockSize/2;
          i32 kc = grid.pseudo2D ? kb : (2*kb + k/bSize);
          grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, kc);
        }
      }
      else
      for (i32 f = 0; f < NEVOLVE; f++) {
        // Refine only on the PRIMARY conserved fields (rho, momentum, rhoE).
        //  - The k~/tau~ pair is a modelled quantity, not a primary field, and
        //    is excluded so it cannot drive the grid on its own.
        //  - z-momentum is identically 0 in pseudo-2D, so it never fires anyway;
        //    skip it explicitly so a stray roundoff detail can never trigger.
        if (f >= F_RHOK) continue;                             // skip the k~/tau~ pair
        if (grid.pseudo2D && f == F_RHOW) continue;            // z-mom is 0 in 2D
        real *Q = grid.getField(f);
        // normalize the detail by the domain max of this field's scale
        // (computed pre-transform, device-side): rho / |momentum| / rhoE
        i32 sc = (f == F_RHO) ? 0 : (f <= F_RHOW ? 1 : 2);
        real mag = fmax(grid.globalScale[sc], (real)1e-32);

        // --leaf: a former rim cell is a leaf but its detail must not drive the
        // grid (the overlap scheme zeroed it); its parent is kept by its block
        const real det = (grid.leafFlux && grid.rimList && grid.rimList[cIdx]) ? (real)0 : abs(Q[cIdx]/mag);
        if (det > grid.waveletThresh || abs(ls) < dx) {
          if (lvl < grid.nLvls-1 && (det > grid.waveletThresh*2 || abs(ls) < dx)) {
            i32 bSize = blockSize/2;
            i32 cx = 2*ib+i/bSize, cy = 2*jb+j/bSize;
            i32 cz = grid.pseudo2D ? kb : (2*kb + k/bSize);
            grid.activateBlock(lvl+1, cx, cy, cz);
            if (grid.periodic) {
              // refine the across-seam partner child in the SAME cycle: an edge
              // child (leftmost/rightmost column at lvl+1) and its opposite-edge
              // partner are periodic neighbors, so activate the partner too --
              // else the opposite edge reaches the finest level a cycle late (a
              // transient coarse/fine seam).  Identity for non-edge children.
              i32 gcx = grid.baseGridSize[0]/blockSize*powi(2,lvl+1);
              i32 gcy = grid.baseGridSize[1]/blockSize*powi(2,lvl+1);
              i32 px = (cx==0) ? gcx-1 : (cx==gcx-1 ? 0 : cx);
              i32 py = (cy==0) ? gcy-1 : (cy==gcy-1 ? 0 : cy);
              i32 pz = cz;
              if (!grid.pseudo2D) {
                i32 gcz = grid.baseGridSize[2]/blockSizeZ*powi(2,lvl+1);
                pz = (cz==0) ? gcz-1 : (cz==gcz-1 ? 0 : cz);
              }
              if (px!=cx || py!=cy || pz!=cz) grid.activateBlock(lvl+1, px, py, pz);
            }
          }
          grid.bFlagsList[bIdx] = KEEP;
        }
      }
    }

    // Level 0 -> level 1 refinement.  Level 0 has no parent, so it uses the
    // virtual level-(-1) detail (5x5x5 base-grid average) instead of the wavelet
    // detail: refine only where the base grid carries sub-cell structure.  This
    // replaces addFineBlocksKernel's old unconditional "level 0 refines
    // everywhere" so level 1 is adaptive.  staticGrid preserves the dense level 1
    // by spawning every child (matches the old addFineBlocksKernel behavior).
    bool refineBase = (lvl == 0 && grid.isInteriorBlock(lvl, ib, jb, kb));
#ifdef USE_MGPU
    refineBase = refineBase && grid.isOwnedBlock(lvl, ib, jb, kb);
#endif
    if (refineBase && grid.nLvls > 1) {
      i32 bSize = blockSize/2;
      i32 cx = 2*ib+i/bSize, cy = 2*jb+j/bSize;
      i32 cz = grid.pseudo2D ? kb : (2*kb + k/bSize);
      bool doRefine = false;
      if (grid.staticGrid) {
        doRefine = true;                                   // dense level 1 (static)
      }
      else {
        for (i32 f = 0; f < NEVOLVE; f++) {
          if (f >= F_RHOK) continue;                       // primary fields only
          if (grid.pseudo2D && f == F_RHOW) continue;      // z-mom is 0 in 2D
          i32 sc = (f == F_RHO) ? 0 : (f <= F_RHOW ? 1 : 2);
          real mag = fmax(grid.globalScale[sc], (real)1e-32);
          real vdet = virtualDetailLevel0(grid, f, bIdx, i, j, k);
          if (abs(vdet/mag) > grid.waveletThresh*2 || abs(ls) < dx) { doRefine = true; break; }
        }
      }
      if (doRefine) {
        grid.activateBlock(1, cx, cy, cz);
        if (grid.periodic && !grid.staticGrid) {
          // refine the across-seam partner child in the same cycle (see the
          // lvl>0 branch above); static grids spawn every child anyway.
          i32 gcx = grid.baseGridSize[0]/blockSize*powi(2,1);
          i32 gcy = grid.baseGridSize[1]/blockSize*powi(2,1);
          i32 px = (cx==0) ? gcx-1 : (cx==gcx-1 ? 0 : cx);
          i32 py = (cy==0) ? gcy-1 : (cy==gcy-1 ? 0 : cy);
          i32 pz = cz;
          if (!grid.pseudo2D) {
            i32 gcz = grid.baseGridSize[2]/blockSizeZ*powi(2,1);
            pz = (cz==0) ? gcz-1 : (cz==gcz-1 ? 0 : cz);
          }
          if (px!=cx || py!=cy || pz!=cz) grid.activateBlock(1, px, py, pz);
        }
      }
    }

  END_CELL_LOOP
}

// Monotone (tri)linear interpolation of a coarse field to a fine ghost cell at
// octant (xs,ys,zs) of parent cell (ip,jp,kp).  The ghost centre sits 1/4 of the
// way from the covering coarse cell toward its neighbour in each axis, so the
// weights are a positive convex combination (no overshoot), unlike DD.
__device__ real
trilinearGhost(MultiLevelSparseGrid &grid, real *Q, i32 prntIdx,
               i32 ip, i32 jp, i32 kp, real xs, real ys, real zs) {
  i32 sx = xs > 0 ? 1 : -1, sy = ys > 0 ? 1 : -1;
  i32 C   = grid.getNbrIdx(prntIdx, ip,    jp,    kp);
  i32 Cx  = grid.getNbrIdx(prntIdx, ip+sx, jp,    kp);
  i32 Cy  = grid.getNbrIdx(prntIdx, ip,    jp+sy, kp);
  i32 Cxy = grid.getNbrIdx(prntIdx, ip+sx, jp+sy, kp);
  if (grid.pseudo2D)
    return (9.0*Q[C] + 3.0*Q[Cx] + 3.0*Q[Cy] + 1.0*Q[Cxy]) / 16.0;
  i32 sz  = zs > 0 ? 1 : -1;
  i32 Cz  = grid.getNbrIdx(prntIdx, ip,    jp,    kp+sz);
  i32 Cxz = grid.getNbrIdx(prntIdx, ip+sx, jp,    kp+sz);
  i32 Cyz = grid.getNbrIdx(prntIdx, ip,    jp+sy, kp+sz);
  i32 Cxyz= grid.getNbrIdx(prntIdx, ip+sx, jp+sy, kp+sz);
  return (27.0*Q[C] + 9.0*(Q[Cx]+Q[Cy]+Q[Cz]) + 3.0*(Q[Cxy]+Q[Cxz]+Q[Cyz]) + 1.0*Q[Cxyz]) / 64.0;
}

__global__ void interpolateFieldsKernel(CompressibleSolver &grid, i32 lvlOnly) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if ((lvlOnly < 0 || lvl == lvlOnly)
        && lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag == GHOST) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      real xs = 2*(i % 2) - 1;
      real ys = 2*(j % 2) - 1;
      real zs = grid.pseudo2D ? 0.0 : (2*(k % 2) - 1);

      // monotone (tri)linear ghost fill: smooth (no piecewise-constant jump) and
      // overshoot-free (no DD ringing), which is what keeps the coarse/fine
      // interface low-Mach consistent (a positive-weighted average of the coarse
      // cells cannot overshoot, so no spurious pressure is injected).
      for (i32 f = 0; f < NEVOLVE; f++)
        grid.getField(f)[cIdx] = trilinearGhost(grid, grid.getField(f), prntIdx, ip, jp, kp, xs, ys, zs);
      // --p1: the ghost polynomial is the parent's restricted to this child
      // (mean q + s_x xi0 + s_y eta0 with xi0 = xs/4, slopes halved) -- exact
      // for a linear field and conservative over the four children
      if (grid.p1) {
        const i32 p = grid.getNbrIdx(prntIdx, ip, jp, kp);
        for (i32 v = 0; v < P1_NV; v++) {
          real *SX = grid.getField(F_P1S + v), *SY = grid.getField(F_P1S + P1_NV + v);
          grid.getField(p1Var(v))[cIdx] = grid.getField(p1Var(v))[p] + (real)0.25*(xs*SX[p] + ys*SY[p]);
          SX[cIdx] = (real)0.5*SX[p];
          SY[cIdx] = (real)0.5*SY[p];
        }
      }
    }

  END_CELL_LOOP
}

__global__ void restrictFieldsKernel(CompressibleSolver &grid, i32 lvlOnly) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    bool restrictCell = grid.pseudo2D ? (i%2==0 && j%2==0)
                                      : (i%2==0 && j%2==0 && k%2==0);
    if ((lvlOnly < 0 || lvl == lvlOnly)
        && lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST && restrictCell) {

      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize / 2);
      i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);

      if (grid.pseudo2D) {
        // average the 4 x-y children at this z-layer (z is not refined)
        i32 c00 = cIdx;
        i32 c10 = cIdx + 1;
        i32 c01 = cIdx + blockSize;
        i32 c11 = cIdx + blockSize + 1;
        for (i32 f = 0; f < NEVOLVE; f++) {
          real *q = grid.getField(f);
          q[pIdx] = (q[c00] + q[c10] + q[c01] + q[c11]) / 4.0;
        }
        // --p1: the parent slope is the L2 projection of the children's
        // polynomials: s = 3/4 (sum of the two means on the + side - the two on
        // the - side) + 1/8 (sum of the four children's slopes); exact for a
        // linear field (means a + b xi0 with xi0 = +-1/4, children slopes b/2).
        if (grid.p1)
          for (i32 v = 0; v < P1_NV; v++) {
            real *q  = grid.getField(p1Var(v));
            real *SX = grid.getField(F_P1S + v), *SY = grid.getField(F_P1S + P1_NV + v);
            SX[pIdx] = (real)0.75*((q[c10] + q[c11]) - (q[c00] + q[c01])) + (real)0.125*(SX[c00] + SX[c10] + SX[c01] + SX[c11]);
            SY[pIdx] = (real)0.75*((q[c01] + q[c11]) - (q[c00] + q[c10])) + (real)0.125*(SY[c00] + SY[c10] + SY[c01] + SY[c11]);
          }
      }
      else {
        // average the 8 children
        i32 c000 = cIdx;
        i32 c100 = cIdx + 1;
        i32 c010 = cIdx + blockSize;
        i32 c110 = cIdx + blockSize + 1;
        i32 c001 = cIdx + blockSize*blockSize;
        i32 c101 = cIdx + blockSize*blockSize + 1;
        i32 c011 = cIdx + blockSize*blockSize + blockSize;
        i32 c111 = cIdx + blockSize*blockSize + blockSize + 1;
        for (i32 f = 0; f < NEVOLVE; f++) {
          real *q = grid.getField(f);
          q[pIdx] = (q[c000] + q[c100] + q[c010] + q[c110] +
                     q[c001] + q[c101] + q[c011] + q[c111]) / 8.0;
        }
      }
    }

  END_CELL_LOOP
}




#ifdef USE_MGPU
// one weight per level-0 base column: the number of OWNED blocks (all levels)
// whose footprint lies in that column -- the load metric for the Z-curve cut
__global__ void countBaseWeightsKernel(CompressibleSolver &grid) {
  START_BLOCK_LOOP
    u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      if (grid.isOwnedBlock(lvl, ib, jb, kb)) {
        i32 i = ib >> lvl, j = jb >> lvl, k = kb >> lvl;
        atomicAdd(&grid.wBase[i + grid.part.nb[0]*(j + grid.part.nb[1]*k)], 1.0);
      }
    }
  END_BLOCK_LOOP
}
#endif

#ifdef USE_MGPU
// insert migrated blocks: one thread per received block.  A block already
// present locally (e.g. as a ghost mirror of its old owner) resolves to its
// existing slot -- the caller then overwrites its fields with the shipped
// (authoritative) data.  slots[t] = bEmpty if the pool is full (dropped).
__global__ void migrateInsertKernel(CompressibleSolver &grid, u64 *locs, i32 n, i32 *slots) {
  i32 t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= n) return;
  i32 idx = grid.hashTable.insert(locs[t]);
  if (idx != bEmpty) {
    grid.bLocList[idx] = locs[t];
    grid.bIdxList[idx] = idx;
    atomicMax(&grid.bFlagsList[idx], KEEP);
    grid.snapValidList[idx] = 0;   // F_OLD regenerated by copyToOld before it is read
  }
  slots[t] = idx;
}
#endif

// AMR debug probe: scan every live cell (both z-planes) for non-finite evolved
// data.  Deliberately NOT cell-looped: pseudo2D's START_CELL_LOOP skips k > 0,
// and the stale z-layers are exactly what we need to see.
__global__ void scanNonFiniteKernel(CompressibleSolver &grid, i32 baseOff) {
  i32 cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const i32 nCell = grid.hashTable.nKeys * blockSizeTot;
  while (cIdx < nCell) {
    i32 bIdx = cIdx / blockSizeTot;
    u64 loc  = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      for (i32 f = 0; f < NEVOLVE; f++) {
        real v = grid.getField(baseOff + f)[cIdx];
        if (!isfinite((double)v)) {
          if ((cIdx % blockSizeTot) < blockSize*blockSize) atomicAdd(&g_nfCnt, 1);
          else                                             atomicAdd(&g_nfCntZ, 1);
          if (atomicMin(&g_nfCidx, cIdx) > cIdx) g_nfField = f;
          break;
        }
      }
    }
    cIdx += gridDim.x*blockDim.x;
  }
}

// ---------------------------------------------------------------------------
// Steady-state residual.  The RHS kernels ACCUMULATE L onto the pre-scaled
// accumulator as S_i = A_i S_{i-1} + L(q_{i-1}), and A_1 = 0, so immediately
// after computeRightHandSide() on stage 0 the bank holds exactly L(q^n) -- the
// residual of the semi-discrete system, with no dt or RK coefficient in it.
//
// Masked to LIVE INTERIOR FLUID cells: the accumulator legitimately holds junk
// (including non-finite values) in exterior/ghost slots and in cells buried in
// the immersed body, because the face-flux scatter writes there and nothing
// ever cleans it.  An unmasked norm is NaN, not large.
//
__global__ void residualNormKernel(CompressibleSolver &grid, real *q0, real dtGlobal, real *rCell) {
  real *Q[4] = {grid.getField(F_RHO),  grid.getField(F_RHOU),
                grid.getField(F_RHOV), grid.getField(F_RHOE)};
  real *Dtl  = grid.getField(F_DTL);
  real *Ibm  = grid.getField(F_IBM);
  const i32 stride = nBlocksMax*blockSizeTot;

  START_CELL_LOOP

    double s2 = 0.0, s2f = 0.0;
    unsigned long long cnt = 0, cntf = 0;
    if (rCell) rCell[cIdx] = (real)0;   // non-live cells map to 0 in the dump

    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);
    // EVERY fluid cell counts.  Only true non-DOFs are excluded: exterior
    // blocks, coarse/fine GHOST cells, and cells inside the body.
    const bool live = grid.isInteriorBlock(lvl, ib, jb, kb)
                   && grid.cFlagsList[cIdx] != GHOST
                   && (!grid.leafFlux || grid.cFlagsList[cIdx] == ACTIVE)
                   && (!grid.immerserdBcType || Ibm[cIdx] != (real)0);
    if (live) {
      // dq/dt is the residual of the WHOLE update, however it is composed --
      // RK accumulator, wall ghosts, prescribed wall flux, any source.  Unlike
      // ||L(q)|| it cannot be defeated by a term applied outside the bank.
      const double dt = (double)((grid.lts && Dtl[cIdx] > (real)0) ? Dtl[cIdx] : dtGlobal);
      double a = 0.0;
      for (i32 f = 0; f < 4; f++) {
        const double d = ((double)Q[f][cIdx] - (double)q0[(size_t)f*stride + cIdx])/dt;
        a += d*d;
      }
      if (isfinite(a)) {
        s2 = a; cnt = 1;
        if (rCell) rCell[cIdx] = (real)sqrt(a);
        const double h  = (double)grid.getDx(lvl);
        const double dw = grid.immerserdBcType
                        ? -(double)grid.getField(F_PHI)[cIdx]/h : 1e30;
        if (dw > (double)grid.resFar) { s2f = a; cntf = 1; }
        double old = g_resMax;
        if (a > old) { atomicMax((unsigned long long*)&g_resMax,
                                 (unsigned long long)__double_as_longlong(a));
                       g_resMaxPhi = dw; }
      }
    }

    for (int off = 16; off > 0; off >>= 1) {
      s2   += __shfl_down_sync(0xffffffff, s2,   off);
      cnt  += __shfl_down_sync(0xffffffff, cnt,  off);
      s2f  += __shfl_down_sync(0xffffffff, s2f,  off);
      cntf += __shfl_down_sync(0xffffffff, cntf, off);
    }
    if ((threadIdx.x & 31) == 0) {
      atomicAdd(&g_resSum, s2);      atomicAdd(&g_resCnt, cnt);
      atomicAdd(&g_resSumFar, s2f);  atomicAdd(&g_resCntFar, cntf);
    }

  END_CELL_LOOP
}


// ---------------------------------------------------------------------------
//  STATE REDISTRIBUTION, N = 0 (finite volume).
//  Berger & Giuliani JCP 428 (2021) 109820.  The degree-N machinery lives in
//  src/common/StateRedistribution.h, but for CELL AVERAGES the weighted L2
//  projection over a merge neighbourhood IS a volume-weighted mean, so neither
//  the cut quadrature nor the mass matrix is needed:
//      Pi_k u  = sum_{j in M_k} w_j u_j / sum_{j in M_k} w_j,  w_j = alpha_j/|C_j|
//      (S u)_i = (1/|C_i|) sum_{k : i in M_k} Pi_k u
//  which gives sum_i alpha_i V_i (Su)_i = sum_i alpha_i V_i u_i exactly, i.e. S
//  is CONSERVATIVE -- the point of using it instead of the per-cell diagonal.
//  Growth is confined to the 5x5 patch about the seed, so a member is always
//  within radius 2 of its seed and the |C_i| / averaging gathers can scan the
//  same patch rather than scattering with atomics: deterministic by construction.
//  V is taken uniform, so this is restricted to nLvls == 1 (checked on the host).
// ---------------------------------------------------------------------------
__device__ unsigned long long g_srdShort = 0;   // neighbourhoods that never reached volFrac

__device__ __forceinline__ bool srdLive(CompressibleSolver &grid, i32 m, i32 cE) {
  // ACTIVE only.  On a multi-level grid a lookup from a fine block into a coarser
  // region comes back as that block's own GHOST cell (interpolated from the coarse
  // parent, not an independent DOF), and PARENT cells are covered by finer ones.
  // Excluding both confines every merge neighbourhood to a SINGLE LEVEL, which is
  // exactly the condition that makes the alpha-only weights right: the cell volume
  // is then a common factor of Pi_k's numerator and denominator and cancels, while
  // the conservation identity (which needs alpha_j V_j / |C_j|) still telescopes
  // because each neighbourhood is internally uniform in V.
  return m < cE && grid.cFlagsList[m] == ACTIVE
      && grid.getField(F_CUTA)[m] > grid.ibRccmAlphaMin;
}

__device__ __forceinline__ bool srdInM(CompressibleSolver &grid, i32 kk, i32 target) {
  const i32 nk = grid.srdMn[kk];
  for (i32 q = 0; q < nk; q++)
    if (grid.srdM[(size_t)q*grid.srdStride + kk] == target) return true;
  return false;
}

__global__ void srdBuildKernel(CompressibleSolver &grid) {
  real *CA = grid.getField(F_CUTA);
  const i32 cE = bEmpty*blockSizeTot;
  const i32 R  = grid.srdReach, W = 2*R + 1, NP = W*W, C0 = R*W + R;   // patch geometry
  START_CELL_LOOP
    GET_CELL_INDICES
    grid.srdMn[cIdx] = 0;
    grid.srdC [cIdx] = 1;
    if (srdLive(grid, cIdx, cE)) {
      grid.srdM[cIdx] = cIdx;                    // slot 0 is always the seed
      const real a0 = CA[cIdx];
      if (a0 >= grid.srdVolFrac) {
        grid.srdMn[cIdx] = 1;                    // healthy: M_k = {k}
      } else {
        i32  pid[25]; real pa[25]; bool inM[25];   // sized for reach <= 2
        for (i32 dj = -R; dj <= R; dj++)
          for (i32 di = -R; di <= R; di++) {
            const i32 t = (dj+R)*W + (di+R);
            const i32 m = grid.getNbrIdx(bIdx, i+di, j+dj, k);
            pid[t] = m; inM[t] = false;
            pa[t]  = srdLive(grid, m, cE) ? CA[m] : (real)-1;
          }
        inM[C0] = true;                          // the seed itself
        real vol = a0; i32 n = 1;
        while (vol < grid.srdVolFrac && n < CompressibleSolver::SRD_MAXM) {
          i32 bt = -1; real ba = 0;              // greedy by volume, face-adjacent only
          for (i32 t = 0; t < NP; t++) {
            if (!inM[t]) continue;
            const i32 tdi = t%W, tdj = t/W;
            const i32 adj[4] = { (tdi>0)? t-1 : -1, (tdi<W-1)? t+1 : -1,
                                 (tdj>0)? t-W : -1, (tdj<W-1)? t+W : -1 };
            for (i32 q = 0; q < 4; q++) {
              const i32 u = adj[q];
              if (u >= 0 && !inM[u] && pa[u] > ba) { ba = pa[u]; bt = u; }
            }
          }
          if (bt < 0) break;                     // nothing left to absorb
          inM[bt] = true; vol += ba;
          grid.srdM[(size_t)n*grid.srdStride + cIdx] = pid[bt];
          n++;
        }
        grid.srdMn[cIdx] = n;
        if (vol < grid.srdVolFrac) atomicAdd(&g_srdShort, 1ull);
      }
    }
  END_CELL_LOOP
}

__global__ void srdCountKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  const i32 R  = grid.srdReach;
  START_CELL_LOOP
    GET_CELL_INDICES
    if (srdLive(grid, cIdx, cE)) {
      i32 c = 0;
      for (i32 dj = -R; dj <= R; dj++)
        for (i32 di = -R; di <= R; di++) {
          const i32 kk = grid.getNbrIdx(bIdx, i+di, j+dj, k);
          if (srdLive(grid, kk, cE) && srdInM(grid, kk, cIdx)) c++;
        }
      grid.srdC[cIdx] = (c > 0) ? c : 1;
    }
  END_CELL_LOOP
}

__global__ void srdProjectKernel(CompressibleSolver &grid) {
  real *CA = grid.getField(F_CUTA);
  const i32 cE = bEmpty*blockSizeTot;
  START_CELL_LOOP
    if (srdLive(grid, cIdx, cE)) {
      const i32 n = grid.srdMn[cIdx];
      double num[5] = {0,0,0,0,0}, den = 0;
      for (i32 q = 0; q < n; q++) {
        const i32 m = grid.srdM[(size_t)q*grid.srdStride + cIdx];
        const double w = (double)CA[m] / (double)max(1, grid.srdC[m]);
        den += w;
        for (i32 f = 0; f < 5; f++) {
          double v = (double)grid.getField(f)[m];
          if (grid.srdIncr) v -= (double)grid.srdU0[(size_t)f*grid.srdStride + m];   // project dU
          num[f] += w*v;
        }
      }
      for (i32 f = 0; f < 5; f++)
        grid.srdPi[(size_t)f*grid.srdStride + cIdx] =
          (den > 0) ? (real)(num[f]/den) : grid.getField(f)[cIdx];
      if (grid.srdIncr && grid.srdPos) {
        // projection of U^n over the same neighbourhood, same weights
        double num0[5] = {0,0,0,0,0};
        for (i32 q = 0; q < n; q++) {
          const i32 m = grid.srdM[(size_t)q*grid.srdStride + cIdx];
          const double w = (double)CA[m] / (double)max(1, grid.srdC[m]);
          for (i32 f = 0; f < 5; f++) num0[f] += w*(double)grid.srdU0[(size_t)f*grid.srdStride + m];
        }
        for (i32 f = 0; f < 5; f++)
          grid.srdPi0[(size_t)f*grid.srdStride + cIdx] =
            (den > 0) ? (real)(num0[f]/den) : grid.srdU0[(size_t)f*grid.srdStride + cIdx];
        // theta_k: would the SEED be inadmissible under U^n + Pi_k(dU)?  The seed
        // sits in exactly one non-trivial neighbourhood (its own), so for it
        // (S dU) = Pi_k(dU) exactly and this test is the true increment result.
        real th = 0;
        if (n > 1 && den > 0) {
          const double r  = (double)grid.srdU0[cIdx] + num[0]/den;
          const double mx = (double)grid.srdU0[(size_t)1*grid.srdStride + cIdx] + num[1]/den;
          const double my = (double)grid.srdU0[(size_t)2*grid.srdStride + cIdx] + num[2]/den;
          const double mz = (double)grid.srdU0[(size_t)3*grid.srdStride + cIdx] + num[3]/den;
          const double E  = (double)grid.srdU0[(size_t)4*grid.srdStride + cIdx] + num[4]/den;
          const double pr = (gam - 1.0)*(E - 0.5*(mx*mx + my*my + mz*mz)/fmax(r, 1e-300));
          if (!(r > (double)grid.srdRhoMin) || !(pr > (double)grid.srdPMin)) th = 1;
        }
        grid.srdTh[cIdx] = th;
      }
    }
  END_CELL_LOOP
}

// The state of a DOF handle, reconstructed from its element centroid with its
// limited gradient, evaluated at (xq, yq).
__device__ inline void cutOwnerState(CompressibleSolver &grid, real *Fs[4], i32 own,
                                     real xq, real yq, Vec5 &q) {
  real g[4][2], lim[4], x0, y0, dxo, dyo;
  cutHandlePos(grid, own, x0, y0, dxo, dyo);
  if (cutIsPiece(own)) cutPieceGrad(grid, Fs, cutPieceOf(own), g, lim);
  else {
    const i32 ob = own/blockSizeTot, occ = own%blockSizeTot;
    const i32 oi = occ%blockSize, oj = (occ/blockSize)%blockSize;
    i32 ol, oib, ojb, okb; grid.decode(grid.bLocList[ob], ol, oib, ojb, okb);
    Vec3 op = grid.getCellPos(ol, oib, ojb, okb, oi, oj, 0);
    rccmGradLimited(grid, Fs, ob, oi, oj, 0, op, dxo, dyo, fmin(dxo, dyo),
                    grid.getField(F_CUTCX)[own], grid.getField(F_CUTCY)[own], g, lim);
  }
  const real ddx = xq - x0, ddy = yq - y0;
  const i32 slot[4] = {0, 1, 2, 4};
  for (i32 f = 0; f < 4; f++) q[slot[f]] = cutTap(grid, Fs, own, f) + lim[f]*(g[f][0]*ddx + g[f][1]*ddy);
  q[3] = cutIsPiece(own) ? grid.cutPieceQ[3*(size_t)grid.cutPieceQCap + cutPieceOf(own)]/fmax(q[0], (real)1e-30)
                         : grid.getField(F_RHOW)[own];
  q[0] = fmax(q[0], (real)1e-12); q[4] = fmax(q[4], (real)1e-12);
}

// ---- the CUT-CELL kernel -----------------------------------------------------
// Everything a cut cell needs beyond the axis-aligned face fluxes of the main
// kernel, so that kernel stays branch-free: the wall flux of every cut cell
// (piece 0 of a split cell), and under --cutsplit the face segments beyond the
// (piece 0, piece 0) pair, the extra pieces' walls and the slit-tip internal
// faces.  All scatters go through DOF handles into the same accumulator.
__global__ void cutCellKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(F_RHO), *U = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV), *W = grid.getField(F_RHOW), *P = grid.getField(F_RHOE);
  real *Fs[4] = {Rho, U, V, P};
  START_LIVE_CELL_LOOP
    GET_CELL_INDICES
    if (grid.cFlagsList[cIdx] == ACTIVE) {
      u64 loc = grid.bLocList[bIdx]; i32 lvl, ib, jb, kb; grid.decode(loc, lvl, ib, jb, kb);
      const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
      const real ax = (real)1/dx, ay = (real)1/dy, hR = fmin(dx, dy);
      Vec3 cpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
    // ---- RCCM wall face ----------------------------------------------------
      // The cut cell's remaining face is the wall segment.  Its area-normal is
      // NOT stored: the discrete divergence theorem fixes it exactly,
      //     A_w n_w = -sum_f A_f n_f  =  -( (aXhi-aXlo) dy , (aYhi-aYlo) dx ),
      // which is what makes the update conservative to machine precision (a
      // stored normal from a separate geometric fit would not close).  For an
      // inviscid slip wall the flux is pressure only.
      if (grid.ibRccm && grid.rccmLive(cIdx)) {
        real *CA = grid.getField(F_CUTA);
        real *AXf = grid.getField(F_CUTAX), *AYf = grid.getField(F_CUTAY);
        const real aC = fmax(grid.cutMerge ? grid.cutAlphaE[cIdx] : CA[cIdx], grid.ibRccmAlphaMin);
        const i32  wT = grid.cutMerge ? grid.cutOwner[cIdx] : cIdx;   // wall flux lands on the owner
        const i32 spId = (grid.cutSplit && grid.cutSplitId) ? grid.cutSplitId[cIdx] : -1;
        // A ZERO-THICKNESS body leaves alpha = 1 exactly: the cell is all fluid and
        // still has a wall through it.  A split cell always takes the wall path.
        if (CA[cIdx] < (real)1 - (real)1e-12 || spId >= 0) {
          const i32 cE = bEmpty*blockSizeTot;
          const i32 xp = grid.getNbrIdx(bIdx, i+1, j, k);
          const i32 yp = grid.getNbrIdx(bIdx, i, j+1, k);
          const real axHi = (xp < cE) ? AXf[xp] : AXf[cIdx];
          const real ayHi = (yp < cE) ? AYf[yp] : AYf[cIdx];
          real nwx = -(axHi - AXf[cIdx]);      // already area/dy
          real nwy = -(ayHi - AYf[cIdx]);
          if (spId >= 0) {                     // split cell: piece 0's own wall, not the union's
            nwx = grid.cutSplitCell[spId].wnx0; nwy = grid.cutSplitCell[spId].wny0;
          }
          // Wall pressure.  The cell AVERAGE is only first order at the wall, and
          // on a curved wall the normal pressure gradient is not small -- it is
          // the centripetal balance dp/dn = rho u_t^2 kappa that turns the flow.
          // Extrapolate to the wall face along the cell's own pressure gradient
          // (central differences over live neighbours), which captures that term
          // without needing the curvature explicitly and is the ordinary
          // cut-cell treatment.  --rccmpw 0 restores the cell-average pressure.
          real pw = P[cIdx];
          if (grid.ibRccmPw) {
            const i32 xm = grid.getNbrIdx(bIdx, i-1, j, k);
            const i32 ym = grid.getNbrIdx(bIdx, i, j-1, k);
            real gpx = 0, gpy = 0;
            // The central difference needs BOTH neighbours live.  At a wall cell the
            // neighbour on the WALL SIDE is dead by definition, and that side is the
            // NORMAL direction -- so the component this zeroes is exactly the one
            // dpw = gp.(phi n) is dominated by, i.e. the centripetal term
            // dp/dn = rho u_t^2 kappa that this extrapolation exists to capture.
            // p_w then collapses to p_c (first order) at precisely the cells that
            // set the lift.  --cutpw 2 falls back to a ONE-SIDED difference there.
            const bool xpL = (xp < cE) && grid.rccmLive(xp);
            const bool xmL = (xm < cE) && grid.rccmLive(xm);
            const bool ypL = (yp < cE) && grid.rccmLive(yp);
            const bool ymL = (ym < cE) && grid.rccmLive(ym);
            if (xpL && xmL)                  gpx = (P[xp] - P[xm])/((real)2*dx);
            else if (grid.ibRccmPw >= 2 && xpL) gpx = (P[xp] - P[cIdx])/dx;
            else if (grid.ibRccmPw >= 2 && xmL) gpx = (P[cIdx] - P[xm])/dx;
            if (ypL && ymL)                  gpy = (P[yp] - P[ym])/((real)2*dy);
            else if (grid.ibRccmPw >= 2 && ypL) gpy = (P[yp] - P[cIdx])/dy;
            else if (grid.ibRccmPw >= 2 && ymL) gpy = (P[cIdx] - P[ym])/dy;
            Vec3 nb(grid.getField(F_CUTNX)[cIdx], grid.getField(F_CUTNY)[cIdx], (real)0);  // PREPROCESSED
            const real ph = grid.getField(F_PHI)[cIdx];        // < 0 in the fluid
            const real dpw = gpx*ph*nb[0] + gpy*ph*nb[1];      // x_w - x_c = phi n
            // never let the extrapolation invert the pressure
            pw = fmax(P[cIdx] + dpw, (real)0.2*P[cIdx]);
          }
          if (spId >= 0) {
            // split cell: the wall pressure from the ELEMENT's limited gradient at
            // piece 0's wall centroid.  F_PHI / F_CUTNX point at the NEAREST wall,
            // which for a thin body may be the other side.
            real *Fs4[4] = {Rho, U, V, P}; real gW[4][2], lW[4];
            const real hR2 = fmin(dx, dy);
            const real ocx2 = grid.getField(F_CUTCX)[cIdx], ocy2 = grid.getField(F_CUTCY)[cIdx];
            rccmGradLimited(grid, Fs4, bIdx, i, j, k, cpos, dx, dy, hR2, ocx2, ocy2, gW, lW);
            const CutSplitCell &Sc = grid.cutSplitCell[spId];
            const real ddx = (Sc.wcx0 - ocx2)*hR2, ddy = (Sc.wcy0 - ocy2)*hR2;
            pw = fmax(P[cIdx] + lW[3]*(gW[3][0]*ddx + gW[3][1]*ddy), (real)0.2*P[cIdx]);
          }
          real rD, uD, vD, pD;
          if (grid.ibDirichlet) {
            // ---- Dirichlet boundary (--ibdir 1) --------------------------------
            // The exact state is known on the segment, so the wall is an ordinary
            // Riemann face against that state -- ghost = Dirichlet datum, which is
            // what a Dirichlet condition means in a Roe/HLLC FV code and what the
            // paper's Sect. 4.4 imposes on the arcs.  Evaluate the datum at the
            // segment MIDPOINT: the two edge crossings of the corner level sets
            // (a saddle-cut cell, 0 or 4 crossings, falls back to the foot of the
            // cell centre).
            const real hx = (real)0.5*dx, hy = (real)0.5*dy;
            const real fc[4] = {
              grid.getBoundaryLevelSet(Vec3(cpos[0]-hx, cpos[1]-hy, cpos[2])),
              grid.getBoundaryLevelSet(Vec3(cpos[0]+hx, cpos[1]-hy, cpos[2])),
              grid.getBoundaryLevelSet(Vec3(cpos[0]+hx, cpos[1]+hy, cpos[2])),
              grid.getBoundaryLevelSet(Vec3(cpos[0]-hx, cpos[1]+hy, cpos[2]))};
            const real ccx[4] = {-hx, hx, hx, -hx}, ccy[4] = {-hy, -hy, hy, hy};
            real mx = 0, my = 0; i32 nc = 0;
            for (i32 e = 0; e < 4; e++) {
              const i32 a = e, b = (e + 1) & 3;
              if ((fc[a] < (real)0) != (fc[b] < (real)0)) {
                const real t = fc[a]/(fc[a] - fc[b]);
                mx += ccx[a] + t*(ccx[b] - ccx[a]);
                my += ccy[a] + t*(ccy[b] - ccy[a]); nc++;
              }
            }
            Vec3 pw3;
            if (nc == 2) pw3 = Vec3(cpos[0] + (real)0.5*mx, cpos[1] + (real)0.5*my, cpos[2]);
            else {
              Vec3 nb(grid.getField(F_CUTNX)[cIdx], grid.getField(F_CUTNY)[cIdx], (real)0);  // PREPROCESSED
              const real ph = grid.getField(F_PHI)[cIdx];
              pw3 = Vec3(cpos[0] + ph*nb[0], cpos[1] + ph*nb[1], cpos[2]);
            }
            if (grid.exactState(pw3[0], pw3[1], rD, uD, vD, pD)) {
              // unit outward (fluid -> solid) normal and |A_w|/dV of the segment
              const real anx = nwx*ax, any = nwy*ay;          // A_w n_w / dV_uncut
              const real am  = sqrt(anx*anx + any*any);
              if (am > (real)1e-30) {
                const Vec3 nu(anx/am, any/am, 0);
                const real rC = Rho[cIdx], uC = U[cIdx], vC = V[cIdx], wC = W[cIdx];
                const real pC = P[cIdx];
                Vec5 qLw(rC, rC*uC, rC*vC, rC*wC,
                         pC/(gam-(real)1) + (real)0.5*rC*(uC*uC + vC*vC + wC*wC));
                Vec5 qRw(rD, rD*uD, rD*vD, (real)0,
                         pD/(gam-(real)1) + (real)0.5*rD*(uD*uD + vD*vD));
                Vec5 Fw = grid.hllcFlux(qLw, qRw, nu);
                atomicAdd(cutRhsPtr(grid, wT, 0), -Fw[0]*am/aC);
                atomicAdd(cutRhsPtr(grid, wT, 1), -Fw[1]*am/aC);
                atomicAdd(cutRhsPtr(grid, wT, 2), -Fw[2]*am/aC);
                atomicAdd(cutRhsPtr(grid, wT, 4), -Fw[4]*am/aC);
                pw = (real)-1;                                 // handled
              }
            }
          }
          if (pw >= (real)0) {
            atomicAdd(cutRhsPtr(grid, wT, 1), -pw*nwx*ax/aC);
            atomicAdd(cutRhsPtr(grid, wT, 2), -pw*nwy*ay/aC);
          }
        }
      }


      const i32 fid = grid.cutFaceId ? grid.cutFaceId[cIdx] : -1;
      const i32 sid = grid.cutSplitId ? grid.cutSplitId[cIdx] : -1;
      if (fid >= 0) {
        const CutFace &F = grid.cutFace[fid];
        for (i32 dir = 0; dir < 2; dir++) {
          const i32 nS = (dir == 0) ? F.nX : F.nY;
          const CutFaceSeg *sg = (dir == 0) ? F.sx : F.sy;
          for (i32 s2 = 1; s2 < nS; s2++) {
            const CutFaceSeg &S = sg[s2];
            if (S.len <= 0 || S.ownC == CUT_DEAD || S.ownN == CUT_DEAD || S.ownC == S.ownN) continue;
            const real xq = (dir == 0) ? cpos[0] - (real)0.5*dx : cpos[0] + S.cen*dx;
            const real yq = (dir == 0) ? cpos[1] + S.cen*dy : cpos[1] - (real)0.5*dy;
            Vec5 qC, qN;
            cutOwnerState(grid, Fs, S.ownC, xq, yq, qC);
            cutOwnerState(grid, Fs, S.ownN, xq, yq, qN);
            const Vec3 nrm = (dir == 0) ? Vec3(1,0,0) : Vec3(0,1,0);
            Vec5 Fl = grid.hllcFlux(grid.prim2cons(qN), grid.prim2cons(qC), nrm);
            const real fac = S.len*((dir == 0) ? ax : ay);
            const real wC = fac/fmax(cutAlphaOf(grid, S.ownC), grid.ibRccmAlphaMin);
            const real wN = fac/fmax(cutAlphaOf(grid, S.ownN), grid.ibRccmAlphaMin);
            for (i32 n = 0; n < 5; n++) {
              atomicAdd(cutRhsPtr(grid, S.ownC, n),  Fl[n]*wC);
              atomicAdd(cutRhsPtr(grid, S.ownN, n), -Fl[n]*wN);
            }
          }
        }
      }
      if (sid >= 0) {
        const CutSplitCell &Sc = grid.cutSplitCell[sid];
        auto ownerOfPiece = [&](i32 p) -> i32 {
          return p == 0 ? grid.cutOwner[cIdx] : grid.cutPiece[Sc.first + p - 1].owner; };
        if (Sc.iLen > (real)0 && Sc.iPa >= 0) {
          // INTERNAL face: the slit-tip extension between pieces iPa and iPb
          const i32 oA = ownerOfPiece(Sc.iPa), oB = ownerOfPiece(Sc.iPb);
          if (oA != CUT_DEAD && oB != CUT_DEAD && oA != oB) {
            const real xq = cpos[0] + Sc.icx*hR, yq = cpos[1] + Sc.icy*hR;
            Vec5 qA, qB;
            cutOwnerState(grid, Fs, oA, xq, yq, qA);
            cutOwnerState(grid, Fs, oB, xq, yq, qB);
            const Vec3 nrm(Sc.inx, Sc.iny, (real)0);              // from A into B
            Vec5 Fl = grid.hllcFlux(grid.prim2cons(qA), grid.prim2cons(qB), nrm);
            const real fac = Sc.iLen/(dx*dy);                       // area / cell volume (dz cancels)
            const real wA = fac/fmax(cutAlphaOf(grid, oA), grid.ibRccmAlphaMin);
            const real wB = fac/fmax(cutAlphaOf(grid, oB), grid.ibRccmAlphaMin);
            for (i32 n = 0; n < 5; n++) {
              atomicAdd(cutRhsPtr(grid, oB, n),  Fl[n]*wB);
              atomicAdd(cutRhsPtr(grid, oA, n), -Fl[n]*wA);
            }
          }
        }
        for (i32 p = 0; p < Sc.n; p++) {
          const CutPiece &Pc = grid.cutPiece[Sc.first + p];
          if (Pc.owner == CUT_DEAD) continue;
          const real xw = cpos[0] + Pc.wcx*hR, yw = cpos[1] + Pc.wcy*hR;
          Vec5 qw; cutOwnerState(grid, Fs, Pc.owner, xw, yw, qw);
          const real pw = fmax(qw[4], (real)0.2*cutTap(grid, Fs, Pc.owner, 3));
          const real aE = fmax(cutAlphaOf(grid, Pc.owner), grid.ibRccmAlphaMin);
          atomicAdd(cutRhsPtr(grid, Pc.owner, 1), -pw*Pc.wnx*ax/aE);
          atomicAdd(cutRhsPtr(grid, Pc.owner, 2), -pw*Pc.wny*ay/aE);
        }
      }
    }
  END_CELL_LOOP
}

// piece-resident DOFs: the same low-storage RK stage as updateFieldsKernel
// (LSRK3 only; the accumulator pre-scale keeps the bank clean between steps)
__global__ void cutPieceUpdateKernel(CompressibleSolver &grid, i32 stage) {
  const real Bw[3]    = {(real)(1.0/3.0), (real)(15.0/16.0), (real)(8.0/15.0)};
  const real Anext[3] = {(real)(-5.0/9.0), (real)(-153.0/128.0), 0};
  const i32 k = blockIdx.x*blockDim.x + threadIdx.x;
  if (k >= grid.nCutPiece) return;
  const CutPiece &P = grid.cutPiece[k]; const size_t cap = grid.cutPieceQCap;
  if (P.owner == cutHandle(k)) {                       // its own DOF
    const real dt = grid.lts ? grid.getField(F_DTL)[P.cell]*grid.dtScale : grid.deltaT;
    for (i32 n = 0; n < 5; n++) grid.cutPieceQ[n*cap + k] += Bw[stage]*dt*grid.cutPieceS[n*cap + k];
    if (grid.pseudo2D) grid.cutPieceQ[3*cap + k] = 0;
    if (grid.p1 && grid.cutPieceSX)
      for (i32 s = 0; s < 2*P1_NV; s++) grid.cutPieceSX[s*cap + k] += Bw[stage]*dt*grid.cutPieceSR[s*cap + k];
  }
  for (i32 n = 0; n < NEVOLVE; n++) grid.cutPieceS[n*cap + k] *= Anext[stage];
  if (grid.p1 && grid.cutPieceSR)
    for (i32 s = 0; s < 2*P1_NV; s++) grid.cutPieceSR[s*cap + k] *= Anext[stage];
}

// --leaf: the level-jump faces.  One thread per mortar: the coarse cell and
// each fine cell are reconstructed (owner state + limited gradient) at the two
// sub-face centroids, one HLLC per sub-face, and each side receives the flux
// with ITS OWN area/volume factor (fine 1/h_f, coarse 1/(2 h_c) per sub-face in
// pseudo-2D) -- the interface is conservative to roundoff.
__global__ void mortarFluxKernel(CompressibleSolver &grid) {
  const i32 m = blockIdx.x*blockDim.x + threadIdx.x;
  if (m >= grid.nMortars) return;
  const MultiLevelSparseGrid::Mortar &M = grid.mortarList[m];
  real *Fs[4] = { grid.getField(F_RHO), grid.getField(F_RHOU), grid.getField(F_RHOV), grid.getField(F_RHOE) };
  const bool merged = grid.cutMerge && grid.cutOwner;
  const i32 ownC = merged ? grid.cutOwner[M.coarse] : M.coarse;
  i32 lc, cib, cjb, ckb; grid.decode(grid.bLocList[M.coarse/blockSizeTot], lc, cib, cjb, ckb);
  i32 lf, fib, fjb, fkb; grid.decode(grid.bLocList[M.fine[0]/blockSizeTot], lf, fib, fjb, fkb);
  const real hc = grid.getDx(lc), hf = grid.getDx(lf);
  const Vec3 nrm = (M.dir == 0) ? Vec3(1,0,0) : Vec3(0,1,0);
  const real amin = grid.ibRccmAlphaMin;
  const real wC = ((real)1/((real)2*hc))/fmax(cutAlphaOf(grid, ownC), amin);
  if (grid.ibRccm && !grid.rccmLive(M.coarse)) return;      // a dead (solid) coarse cell: no fluid face
  for (i32 q = 0; q < 2; q++) {
    const i32 fc = M.fine[q];
    if (grid.cFlagsList[fc] != ACTIVE) continue;
    if (grid.ibRccm && !grid.rccmLive(fc)) continue;         // dead fine cell
    const i32 ownF = merged ? grid.cutOwner[fc] : fc;
    if (ownF == ownC) continue;
    Vec5 qC, qF;
    cutOwnerState(grid, Fs, ownC, M.cen[q][0], M.cen[q][1], qC);
    cutOwnerState(grid, Fs, ownF, M.cen[q][0], M.cen[q][1], qF);
    Vec5 Fl = (M.side == 0) ? grid.hllcFlux(grid.prim2cons(qC), grid.prim2cons(qF), nrm)
                            : grid.hllcFlux(grid.prim2cons(qF), grid.prim2cons(qC), nrm);
    const real wF = ((real)1/hf)/fmax(cutAlphaOf(grid, ownF), amin);
    for (i32 n = 0; n < 5; n++) {
      if (M.side == 0) { atomicAdd(cutRhsPtr(grid, ownF, n),  Fl[n]*wF); atomicAdd(cutRhsPtr(grid, ownC, n), -Fl[n]*wC); }
      else             { atomicAdd(cutRhsPtr(grid, ownC, n),  Fl[n]*wC); atomicAdd(cutRhsPtr(grid, ownF, n), -Fl[n]*wF); }
    }
  }
}

// --cutmerge: members take the owner's evolved state (all evolved DOFs), so the
// next stage's stencils read one value per element wherever they tap it.
__global__ void cutBroadcastKernel(CompressibleSolver &grid) {
  START_CELL_LOOP
    const i32 o = grid.cutOwner[cIdx];
    if (o != cIdx) {
      if (cutIsPiece(o)) {
        const i32 k = cutPieceOf(o); const size_t cap = grid.cutPieceQCap;
        for (i32 f = 0; f < NEVOLVE; f++) grid.getField(f)[cIdx] = grid.cutPieceQ[f*cap + k];
        if (grid.p1 && grid.cutPieceSX) for (i32 s = 0; s < 2*P1_NV; s++) grid.getField(F_P1S + s)[cIdx] = grid.cutPieceSX[s*cap + k];
      } else {
        for (i32 f = 0; f < NEVOLVE; f++) grid.getField(f)[cIdx] = grid.getField(f)[o];
        if (grid.p1) for (i32 s = 0; s < 2*P1_NV; s++) grid.getField(F_P1S + s)[cIdx] = grid.getField(F_P1S + s)[o];
      }
    }
  END_CELL_LOOP
}

__global__ void srdSnapKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  START_CELL_LOOP
    if (srdLive(grid, cIdx, cE))
      for (i32 f = 0; f < 5; f++)
        grid.srdU0[(size_t)f*grid.srdStride + cIdx] = grid.getField(f)[cIdx];
  END_CELL_LOOP
}

// dUmax = max ||U* - U^n||_2 over cells that belong to a NON-TRIVIAL
// neighbourhood -- those are the only cells S can move.
__device__ double g_srdDU = 0;
// PER-NEIGHBOURHOOD indicator, faithful to UM-SRD (4)-(6):
//   dUmax_k = max_{i in M_k} ||U*_i - U^n_i||_2,  eta_k = dUmax_k/(eps + dUmax_k),
//   s_k = eta_k^p/(eta_k^p + tau^p).
// A neighbourhood whose cells have all gone quiet shuts itself off even while
// the rest of the wall is still moving -- which a single global s cannot do.
__global__ void srdIndicatorKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  START_CELL_LOOP
    if (srdLive(grid, cIdx, cE)) {
      const i32 n = grid.srdMn[cIdx];
      double dmax = 0;
      for (i32 q = 0; q < n; q++) {
        const i32 m = grid.srdM[(size_t)q*grid.srdStride + cIdx];
        double s2 = 0;
        for (i32 f = 0; f < 5; f++) {
          const double d = (double)grid.getField(f)[m]
                         - (double)grid.srdU0[(size_t)f*grid.srdStride + m];
          s2 += d*d;
        }
        const double d = sqrt(s2);
        if (d > dmax) dmax = d;
      }
      const double eta = dmax/((double)grid.srdEps + dmax);
      double ep = 1, tp = 1;
      for (i32 q = 0; q < grid.srdP; q++) { ep *= eta; tp *= (double)grid.srdTau; }
      grid.srdS[cIdx] = (real)((ep + tp > 0) ? ep/(ep + tp) : 0.0);
    }
  END_CELL_LOOP
}

__global__ void srdAverageKernel(CompressibleSolver &grid, real sBlend) {
  const i32 cE = bEmpty*blockSizeTot;
  const i32 R  = grid.srdReach;
  START_CELL_LOOP
    GET_CELL_INDICES
    if (srdLive(grid, cIdx, cE)) {
      double acc[5] = {0,0,0,0,0}, acc0[5] = {0,0,0,0,0}; double thMax = 0;
      const bool pos = grid.srdIncr && grid.srdPos;
      for (i32 dj = -R; dj <= R; dj++)
        for (i32 di = -R; di <= R; di++) {
          const i32 kk = grid.getNbrIdx(bIdx, i+di, j+dj, k);
          if (srdLive(grid, kk, cE) && srdInM(grid, kk, cIdx)) {
            for (i32 f = 0; f < 5; f++)
              acc[f] += (double)grid.srdPi[(size_t)f*grid.srdStride + kk];
            if (pos) {
              for (i32 f = 0; f < 5; f++)
                acc0[f] += (double)grid.srdPi0[(size_t)f*grid.srdStride + kk];
              thMax = fmax(thMax, (double)grid.srdTh[kk]);
            }
          }
        }
      const double ci = (double)max(1, grid.srdC[cIdx]);
      // UM-SRD blend: R = (1-s) Id + s S.  s = 0 leaves U* untouched, which is
      // what makes the base scheme's steady state an exact fixed point.
      // s for THIS cell is the max over the neighbourhoods it belongs to (a cell
      // that any active neighbourhood still needs must stay stabilised).
      double sb = (double)sBlend;
      if (sBlend < (real)0) {
        sb = 0;
        for (i32 dj = -R; dj <= R; dj++)
          for (i32 di = -R; di <= R; di++) {
            const i32 kk = grid.getNbrIdx(bIdx, i+di, j+dj, k);
            if (srdLive(grid, kk, cE) && srdInM(grid, kk, cIdx))
              sb = fmax(sb, (double)grid.srdS[kk]);
          }
      }
      if (grid.srdIncr) {
        // U^{n+1} = U^n + S(dU) [+ theta (S(U^n) - U^n) where a sliver would go bad]
        for (i32 f = 0; f < 5; f++) {
          const double u0 = (double)grid.srdU0[(size_t)f*grid.srdStride + cIdx];
          double v = u0 + acc[f]/ci;
          if (pos && thMax > 0) v += thMax*(acc0[f]/ci - u0);
          grid.getField(f)[cIdx] = (real)v;
        }
      } else
      for (i32 f = 0; f < 5; f++) {
        const double uStar = (double)grid.getField(f)[cIdx];
        grid.getField(f)[cIdx] = (real)((1.0 - sb)*uStar + sb*(acc[f]/ci));
      }
    }
  END_CELL_LOOP
}

// Snapshot q for the dq/dt residual.  Taken AFTER adaptation (sortBlocks
// renumbers blocks) and before the RK stages, so indices match at compare time.
__global__ void residualSnapshotKernel(CompressibleSolver &grid, real *q0) {
  real *Q[4] = {grid.getField(F_RHO),  grid.getField(F_RHOU),
                grid.getField(F_RHOV), grid.getField(F_RHOE)};
  const i32 stride = nBlocksMax*blockSizeTot;
  START_CELL_LOOP
    for (i32 f = 0; f < 4; f++) q0[(size_t)f*stride + cIdx] = Q[f][cIdx];
  END_CELL_LOOP
}


// =============================================================================
// --p1: modal P1 discontinuous Galerkin (pseudo-2D Euler)
//
// Every cell carries, per conserved variable, the mean q (the FV storage) and
// two slope DOFs (F_P1S):
//     u_h(x,y) = q + s_x xi + s_y eta,   xi = (x-x_c)/dx,  eta = (y-y_c)/dy
// on [-1/2,1/2]^2.  The basis {1, xi, eta} is orthogonal on the box, so the mass
// matrix is diag(1, 1/12, 1/12) dx dy and the semi-discrete equations are
//     dq/dt  =  1/(dx dy)  [            - sum_faces  L sum_g w_g  F_g.n           ]
//     ds/dt  = 12/(dx dy)  [ int F.grad(phi) dV  - sum_faces  L sum_g w_g (F_g.n) phi_g ]
// with 2 Gauss points per face piece and a 2x2 Gauss rule for the volume term.
// A face PIECE of length L on the side of a cell with outward normal n, whose
// Riemann fluxes F_g (along +e_dir) sit at points with weights w_g (sum 1) and
// basis values (xi_g, eta_g) in the cell's OWN coordinates, contributes
//     mean:  fac sum w_g F_g,   s_x: 12 fac sum w_g F_g xi_g,   s_y: same with eta_g,
//     fac = -(n.e_dir) L / (dx dy).
// That one formula (p1Scatter) serves the regular faces (L = dy or dx, both
// cells), the mortar sub-faces of a level jump (L = h_fine on BOTH sides; the
// coarse cell's xi, eta are taken at the physical point) and, later, the
// clipped face segments and wall segments of a cut cell.  No reconstruction
// stencil exists anywhere, so a jump needs no band and no least squares.
//
// Free-stream check of the slope equation: for a constant state the two
// x-faces give -6 F/dx each (xi = -1/2 on the low face with fac = +1/dx, +1/2
// on the high face with fac = -1/dx) and the volume term gives +12 F/dx, so
// ds/dt = 0 exactly.
// =============================================================================

struct P1Poly { real q[5]; real sx[5]; real sy[5]; };

// load a cell polynomial.  The mean bank holds PRIMITIVES during the RHS
// (conservativeToPrimitive precedes it), so the mean is converted back; the
// slopes are stored in CONSERVED variables.  rhoW carries no slope (pseudo-2D).
__device__ inline void p1Load(CompressibleSolver &grid, i32 c, P1Poly &P) {
  Vec5 qc = grid.prim2cons(Vec5(grid.getField(F_RHO)[c], grid.getField(F_RHOU)[c],
                                grid.getField(F_RHOV)[c], grid.getField(F_RHOW)[c], grid.getField(F_RHOE)[c]));
  for (i32 f = 0; f < 5; f++) { P.q[f] = qc[f]; P.sx[f] = 0; P.sy[f] = 0; }
  for (i32 v = 0; v < P1_NV; v++) {
    const i32 f = p1Var(v);
    P.sx[f] = grid.getField(F_P1S + v)[c];
    P.sy[f] = grid.getField(F_P1S + P1_NV + v)[c];
  }
}

// polynomial trace at local (xi, eta); falls back to the mean when the point
// state is not physical (rho or p <= 0), which an unlimited slope can produce
__device__ inline Vec5 p1Trace(const P1Poly &P, real xi, real eta) {
  Vec5 q;
  for (i32 f = 0; f < 5; f++) q[f] = P.q[f] + P.sx[f]*xi + P.sy[f]*eta;
  const real ke = (real)0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/fmax(q[0], (real)1e-300);
  if (!(q[0] > (real)0) || !(q[4] - ke > (real)0))
    for (i32 f = 0; f < 5; f++) q[f] = P.q[f];
  return q;
}

// physical Euler flux along +x (dir 0) or +y (dir 1)
__device__ inline Vec5 p1EulerFlux(Vec5 q, i32 dir) {
  const real r = fmax(q[0], (real)1e-300), u = q[1]/r, v = q[2]/r, w = q[3]/r;
  const real p = (gam - (real)1)*(q[4] - (real)0.5*r*(u*u + v*v + w*w));
  const real un = (dir == 0) ? u : v;
  return Vec5(r*un, q[1]*un + (dir == 0 ? p : (real)0), q[2]*un + (dir == 1 ? p : (real)0),
              q[3]*un, (q[4] + p)*un);
}

// one face piece into a cell's mean + slope accumulators (see the header note)
__device__ inline void p1Scatter(CompressibleSolver &grid, i32 c, real fac, const Vec5 *Fg,
                                 const real *wg, const real *xig, const real *etag, i32 ng) {
  for (i32 f = 0; f < 5; f++) {
    real m = 0, mx = 0, my = 0;
    for (i32 g = 0; g < ng; g++) {
      const real F = wg[g]*Fg[g].data[f];
      m += F;  mx += F*xig[g];  my += F*etag[g];
    }
    atomicAdd(&grid.getField(F_RHS + f)[c], fac*m);
    if (f == F_RHOW) continue;
    const i32 v = (f < 3) ? f : 3;
    atomicAdd(&grid.getField(F_P1SR + v)[c],         (real)12*fac*mx);
    atomicAdd(&grid.getField(F_P1SR + P1_NV + v)[c], (real)12*fac*my);
  }
}

static constexpr real P1_G1 = (real)0.28867513459481287;   // Gauss point 1/(2 sqrt 3) on [-1/2, 1/2]

// ---- --p1 domain boundaries WITHOUT ghost cells --------------------------------
// A DG scheme imposes its boundary conditions weakly: the flux through a face on
// the domain boundary is the Riemann flux of the interior TRACE against a
// boundary state built from that trace and the boundary type.  side: 0 x-min,
// 1 x-max, 2 y-min, 3 y-max (outward normal of the domain).  Returns false for
// a type that keeps the ghost path (periodic images; the inlet/outlet types 6, 7).
__device__ inline bool p1BoundaryState(CompressibleSolver &grid, i32 side, Vec5 qIn, real x, real y, Vec5 &qB) {
  const real nx = (side == 0) ? (real)-1 : (side == 1) ? (real)1 : (real)0;
  const real ny = (side == 2) ? (real)-1 : (side == 3) ? (real)1 : (real)0;
  const real ri = fmax(qIn[0], (real)1e-30), ui = qIn[1]/ri, vi = qIn[2]/ri, wi = qIn[3]/ri;
  const real pi_ = (gam - (real)1)*(qIn[4] - (real)0.5*ri*(ui*ui + vi*vi + wi*wi));
  auto cons = [&](real r, real u, real v, real w, real p) {
    qB = Vec5(r, r*u, r*v, r*w, p/(gam - (real)1) + (real)0.5*r*(u*u + v*v + w*w)); };
  auto mirror = [&](bool allComponents) {
    const real un = ui*nx + vi*ny;
    if (allComponents) cons(ri, -ui, -vi, -wi, pi_);
    else cons(ri, ui - (real)2*un*nx, vi - (real)2*un*ny, wi, pi_); };
  switch (grid.bcType) {
    case 0: mirror(false); return true;                                   // slip walls
    case 1: mirror(true);  return true;                                   // no-slip walls
    case 3: qB = qIn;      return true;                                   // transmissive
    case 4:                                                               // flat plate box
      if (side == 0) { cons((real)1, grid.fsU, grid.fsV, (real)0, grid.fsP); return true; }   // inflow
      if (side == 2) { mirror(false); return true; }                                       // y-min slip wall
      cons(ri, ui, vi, wi, grid.fsP); return true;                                         // far field / outflow: back pressure
    case 5: {                                                             // point-vortex far field
      real rF, uF, vF, pF; ffVortexState(grid, x, y, rF, uF, vF, pF);
      if (grid.fsU*nx + grid.fsV*ny < (real)0) cons(rF, uF, vF, (real)0, pF);   // inflow
      else cons(ri, ui, vi, wi, pF);                                              // outflow: vortex-corrected back pressure
      return true; }
    case 8: {                                                             // Riemann-invariant far field
      const real ci = sqrt(fmax(gam*pi_/ri, (real)1e-30));
      const real uni = ui*nx + vi*ny;
      const real re = (real)1, ue = grid.fsU, ve = grid.fsV, pe = grid.fsP;
      const real ce = sqrt(fmax(gam*pe/re, (real)1e-30));
      const real une = ue*nx + ve*ny;
      const real tg = (real)2/(gam - (real)1);
      const real Rp = (uni > -ci) ? (uni + tg*ci) : (une + tg*ce);
      const real Rm = (une <  ce) ? (une - tg*ce) : (uni - tg*ci);
      const real unb = (real)0.5*(Rp + Rm);
      const real cb  = fmax((gam - (real)1)*(real)0.25*(Rp - Rm), (real)1e-30);
      const real wI = (grid.ffBlend > (real)0) ? (real)0.5*((real)1 + tanh(unb/(grid.ffBlend*cb))) : ((unb > (real)0) ? (real)1 : (real)0);
      const real ent = wI*(pi_/pow(ri, gam)) + ((real)1 - wI)*(pe/pow(re, gam));
      const real ut = wI*ui + ((real)1 - wI)*ue, vt = wI*vi + ((real)1 - wI)*ve;
      const real rb = pow(cb*cb/(gam*ent), (real)1/(gam - (real)1)), pb = rb*cb*cb/gam;
      const real unt = ut*nx + vt*ny;
      cons(rb, ut + (unb - unt)*nx, vt + (unb - unt)*ny, (real)0, pb); return true; }
    case 9: cons((real)1, grid.fsU, grid.fsV, (real)0, grid.fsP); return true;   // freestream
    default: return false;
  }
}
// which domain boundary a cell of an EXTERIOR block sits beyond (-1: interior block)
__device__ inline bool p1ExtBlock(CompressibleSolver &grid, i32 c) {
  i32 lvl, ib, jb, kb; grid.decode(grid.bLocList[c/blockSizeTot], lvl, ib, jb, kb);
  return grid.isExteriorBlock(lvl, ib, jb, kb);
}

__global__ void p1RhsKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  const real w2[2] = {(real)0.5, (real)0.5};
  START_LIVE_CELL_LOOP
    GET_CELL_INDICES
    const u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
      // --leaf: an idle parent owns nothing, and a face against a parent is a
      // level jump that p1MortarKernel fluxes on the fine sub-faces
      const bool covered = grid.leafFlux && grid.cFlagsList[cIdx] == PARENT;
      // cut geometry: a cell whose faces live in the P1 cut tables (clipped,
      // dead, merged) is fluxed by p1SegKernel / p1ElemKernel instead
      const bool irrC = grid.p1Irr && grid.p1Irr[cIdx];
      P1Poly Pc;
      p1Load(grid, cIdx, Pc);

      const bool extC = p1ExtBlock(grid, cIdx);
      Vec3 cpos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
      // low-x face: this cell owns it, the left neighbour receives the same flux.
      // A face between an interior cell and an EXTERIOR block is a domain
      // boundary: the flux is the Riemann flux of the interior trace against the
      // weak boundary state, scattered to the interior side only (no ghost).
      const i32 l1 = grid.getNbrIdx(bIdx, i-1, j, k);
      if (l1 < cE && !covered && !irrC && !(grid.p1Irr && grid.p1Irr[l1]) && !(grid.leafFlux && grid.cFlagsList[l1] == PARENT)) {
        const bool extL = p1ExtBlock(grid, l1);
        const real eta[2] = {-P1_G1, P1_G1};
        const real xiC[2] = {(real)-0.5, (real)-0.5}, xiL[2] = {(real)0.5, (real)0.5};
        Vec5 Fg[2]; bool weak = false;
        if (extC != extL) {                        // domain boundary face (x-min if l1 is exterior, else x-max)
          const i32 side = extL ? 0 : 1, in = extL ? cIdx : l1;
          P1Poly Pin; if (in == cIdx) Pin = Pc; else p1Load(grid, in, Pin);
          const real xf = cpos[0] - (real)0.5*dx;
          weak = true;
          for (i32 g = 0; g < 2 && weak; g++) {
            Vec5 qIn = p1Trace(Pin, extL ? (real)-0.5 : (real)0.5, eta[g]), qB;
            if (!p1BoundaryState(grid, side, qIn, xf, cpos[1] + eta[g]*dy, qB)) { weak = false; break; }
            Fg[g] = extL ? grid.hllcFlux(qB, qIn, Vec3(1,0,0)) : grid.hllcFlux(qIn, qB, Vec3(1,0,0));
          }
          if (weak) { if (extL) p1Scatter(grid, cIdx, (real)1/dx, Fg, w2, xiC, eta, 2); else p1Scatter(grid, l1, (real)-1/dx, Fg, w2, xiL, eta, 2); }
        }
        if (!weak && !(extC && extL)) {            // interior face (or a boundary type that keeps its ghost)
          P1Poly Pl;
          p1Load(grid, l1, Pl);
          for (i32 g = 0; g < 2; g++)
            Fg[g] = grid.hllcFlux(p1Trace(Pl, (real)0.5, eta[g]), p1Trace(Pc, (real)-0.5, eta[g]), Vec3(1,0,0));
          p1Scatter(grid, cIdx, (real)1/dx,  Fg, w2, xiC, eta, 2);
          p1Scatter(grid, l1,   (real)-1/dx, Fg, w2, xiL, eta, 2);
        }
      }
      // low-y face
      const i32 d1 = grid.getNbrIdx(bIdx, i, j-1, k);
      if (d1 < cE && !covered && !irrC && !(grid.p1Irr && grid.p1Irr[d1]) && !(grid.leafFlux && grid.cFlagsList[d1] == PARENT)) {
        const bool extD = p1ExtBlock(grid, d1);
        const real xi[2] = {-P1_G1, P1_G1};
        const real etC[2] = {(real)-0.5, (real)-0.5}, etD[2] = {(real)0.5, (real)0.5};
        Vec5 Fg[2]; bool weak = false;
        if (extC != extD) {                        // domain boundary face (y-min if d1 is exterior, else y-max)
          const i32 side = extD ? 2 : 3, in = extD ? cIdx : d1;
          P1Poly Pin; if (in == cIdx) Pin = Pc; else p1Load(grid, in, Pin);
          const real yf = cpos[1] - (real)0.5*dy;
          weak = true;
          for (i32 g = 0; g < 2 && weak; g++) {
            Vec5 qIn = p1Trace(Pin, xi[g], extD ? (real)-0.5 : (real)0.5), qB;
            if (!p1BoundaryState(grid, side, qIn, cpos[0] + xi[g]*dx, yf, qB)) { weak = false; break; }
            Fg[g] = extD ? grid.hllcFlux(qB, qIn, Vec3(0,1,0)) : grid.hllcFlux(qIn, qB, Vec3(0,1,0));
          }
          if (weak) { if (extD) p1Scatter(grid, cIdx, (real)1/dy, Fg, w2, xi, etC, 2); else p1Scatter(grid, d1, (real)-1/dy, Fg, w2, xi, etD, 2); }
        }
        if (!weak && !(extC && extD)) {
          P1Poly Pd;
          p1Load(grid, d1, Pd);
          for (i32 g = 0; g < 2; g++)
            Fg[g] = grid.hllcFlux(p1Trace(Pd, xi[g], (real)0.5), p1Trace(Pc, xi[g], (real)-0.5), Vec3(0,1,0));
          p1Scatter(grid, cIdx, (real)1/dy,  Fg, w2, xi, etC, 2);
          p1Scatter(grid, d1,   (real)-1/dy, Fg, w2, xi, etD, 2);
        }
      }
      // volume term 12/(dx dy) int F.grad(phi) dV on the cell's own polynomial:
      // grad(xi) = (1/dx, 0) -> 12 <F_x>/dx, grad(eta) -> 12 <G_y>/dy, 2x2 Gauss
      if (!covered && !irrC) {
        real Fx[5] = {0,0,0,0,0}, Gy[5] = {0,0,0,0,0};
        for (i32 gx = 0; gx < 2; gx++)
          for (i32 gy = 0; gy < 2; gy++) {
            Vec5 q = p1Trace(Pc, gx ? P1_G1 : -P1_G1, gy ? P1_G1 : -P1_G1);
            Vec5 F = p1EulerFlux(q, 0), G = p1EulerFlux(q, 1);
            for (i32 f = 0; f < 5; f++) { Fx[f] += (real)0.25*F[f]; Gy[f] += (real)0.25*G[f]; }
          }
        for (i32 v = 0; v < P1_NV; v++) {
          const i32 f = p1Var(v);
          atomicAdd(&grid.getField(F_P1SR + v)[cIdx],         (real)12*Fx[f]/dx);
          atomicAdd(&grid.getField(F_P1SR + P1_NV + v)[cIdx], (real)12*Gy[f]/dy);
        }
      }
    }
  END_CELL_LOOP
}

// --leaf level jumps: one thread per mortar, 2 sub-faces x 2 Gauss points.  Each
// side evaluates its OWN polynomial at the physical point, so the coupling is
// stencil-free and conservative (the same flux enters both sides).
__global__ void p1MortarKernel(CompressibleSolver &grid) {
  const i32 m = blockIdx.x*blockDim.x + threadIdx.x;
  if (m >= grid.nMortars) return;
  const MultiLevelSparseGrid::Mortar &M = grid.mortarList[m];
  const i32 cE = bEmpty*blockSizeTot;
  const real w2[2] = {(real)0.5, (real)0.5};
  const i32 c = M.coarse;
  if (c < 0 || c >= cE || grid.cFlagsList[c] != ACTIVE) return;
  if (grid.p1Irr && grid.p1Irr[c]) return;          // cut geometry never touches a jump (refined to the finest level)
  i32 lc, cib, cjb, ckb;
  grid.decode(grid.bLocList[c/blockSizeTot], lc, cib, cjb, ckb);
  const i32 cc = c % blockSizeTot;
  Vec3 xc = grid.getCellPos(lc, cib, cjb, ckb, cc % blockSize, (cc/blockSize) % blockSize, cc/(blockSize*blockSize));
  const real hc = grid.getDx(lc);
  P1Poly Pc;
  p1Load(grid, c, Pc);
  const Vec3 nrm = (M.dir == 0) ? Vec3(1,0,0) : Vec3(0,1,0);
  for (i32 q = 0; q < 2; q++) {
    const i32 fc = M.fine[q];
    if (fc < 0 || fc >= cE || grid.cFlagsList[fc] != ACTIVE) continue;
    i32 lf, fib, fjb, fkb;
    grid.decode(grid.bLocList[fc/blockSizeTot], lf, fib, fjb, fkb);
    const i32 fcc = fc % blockSizeTot;
    Vec3 xf = grid.getCellPos(lf, fib, fjb, fkb, fcc % blockSize, (fcc/blockSize) % blockSize, fcc/(blockSize*blockSize));
    const real hf = grid.getDx(lf);
    P1Poly Pf;
    p1Load(grid, fc, Pf);
    real xiF[2], etF[2], xiC[2], etC[2];
    Vec5 Fg[2];
    for (i32 g = 0; g < 2; g++) {
      const real s  = (g ? P1_G1 : -P1_G1)*hf;          // tangential offset along the sub-face
      const real px = M.cen[q][0] + (M.dir == 0 ? (real)0 : s);
      const real py = M.cen[q][1] + (M.dir == 0 ? s : (real)0);
      xiF[g] = (px - xf[0])/hf;  etF[g] = (py - xf[1])/hf;
      xiC[g] = (px - xc[0])/hc;  etC[g] = (py - xc[1])/hc;
      Vec5 qF = p1Trace(Pf, xiF[g], etF[g]), qC = p1Trace(Pc, xiC[g], etC[g]);
      Fg[g] = (M.side == 0) ? grid.hllcFlux(qC, qF, nrm) : grid.hllcFlux(qF, qC, nrm);
    }
    // side 0: the coarse cell is on the LOW side, so the sub-face is the fine
    // cell's low face (outward -e: fac = +L/(hf hf) = 1/hf) and the coarse
    // cell's high face (fac = -hf/(hc hc)); side 1 mirrors both
    const real sgn = (M.side == 0) ? (real)1 : (real)-1;
    p1Scatter(grid, fc, sgn/hf,          Fg, w2, xiF, etF, 2);
    p1Scatter(grid, c, -sgn*hf/(hc*hc),  Fg, w2, xiC, etC, 2);
  }
}

// slope limiter after every stage (gradLim 1 = Barth-Jespersen, 2 =
// Venkatakrishnan with eps^2 = (K h)^3), per conserved variable, against the
// face-neighbour MEANS; the polynomial is checked at the four face midpoints.
// A jump tap goes through the mortar: the coarse cell, or both fine cells.
__global__ void p1LimitKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  START_LIVE_CELL_LOOP
    GET_CELL_INDICES
    const u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty && grid.cFlagsList[cIdx] == ACTIVE && !(grid.p1Irr && grid.p1Irr[cIdx])) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      if (grid.isInteriorBlock(lvl, ib, jb, kb)) {
        const real h = fmin(grid.getDx(lvl), grid.getDy(lvl));
        i32 taps[8]; i32 nt = 0;
        const i32 di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
        for (i32 d = 0; d < 4; d++) {
          const i32 n = grid.getNbrIdx(bIdx, i+di[d], j+dj[d], k);
          if (n < cE && !(grid.leafFlux && grid.cFlagsList[n] == PARENT)) { taps[nt++] = n; continue; }
          if (grid.leafFlux && grid.cellMortar) {
            const i32 mm = grid.cellMortar[(size_t)cIdx*4 + d];
            if (mm >= 0) {
              const MultiLevelSparseGrid::Mortar &M = grid.mortarList[mm];
              if (M.coarse == cIdx) {
                for (i32 q = 0; q < 2; q++)
                  if (M.fine[q] >= 0 && M.fine[q] < cE && grid.cFlagsList[M.fine[q]] == ACTIVE) taps[nt++] = M.fine[q];
              } else if (M.coarse >= 0 && M.coarse < cE) taps[nt++] = M.coarse;
            }
          }
        }
        const real eps2 = (grid.gradLim == 2) ? pow(grid.gradLimK*h, (real)3) : (real)0;
        for (i32 v = 0; v < P1_NV; v++) {
          real *Q  = grid.getField(p1Var(v));
          real *SX = grid.getField(F_P1S + v), *SY = grid.getField(F_P1S + P1_NV + v);
          const real qc = Q[cIdx];
          real qmin = qc, qmax = qc;
          for (i32 t = 0; t < nt; t++) { qmin = fmin(qmin, Q[taps[t]]); qmax = fmax(qmax, Q[taps[t]]); }
          const real sx = SX[cIdx], sy = SY[cIdx];
          const real dmax = qmax - qc, dmin = qmin - qc;
          const real dpts[4] = {-(real)0.5*sx, (real)0.5*sx, -(real)0.5*sy, (real)0.5*sy};
          real a = 1;
          for (i32 p = 0; p < 4; p++) {
            const real d = dpts[p];
            if (fabs(d) < (real)1e-300) continue;
            const real D = (d > 0) ? dmax : dmin;
            real psi;
            if (grid.gradLim == 2) {
              const real D2 = D*D, d2 = d*d;
              psi = (D2 + eps2 + (real)2*d*D) / (D2 + (real)2*d2 + d*D + eps2 + (real)1e-300);
            } else {
              psi = fmin((real)1, D/d);
            }
            a = fmin(a, fmax(psi, (real)0));
          }
          SX[cIdx] = a*sx;
          SY[cIdx] = a*sy;
          // (zeroing the LSRK slope accumulator where the limiter acts was tried
          // against the plate's leading-edge limit cycle: no effect, removed)
        }
      }
    }
  END_CELL_LOOP
}

// initial slopes: central differences of the conserved means (a 2nd-order
// projection of the initial field); a missing tap gives a zero slope
__global__ void p1InitSlopesKernel(CompressibleSolver &grid) {
  const i32 cE = bEmpty*blockSizeTot;
  START_CELL_LOOP
    GET_CELL_INDICES
    const i32 l1 = grid.getNbrIdx(bIdx, i-1, j, k), r1 = grid.getNbrIdx(bIdx, i+1, j, k);
    const i32 d1 = grid.getNbrIdx(bIdx, i, j-1, k), u1 = grid.getNbrIdx(bIdx, i, j+1, k);
    for (i32 v = 0; v < P1_NV; v++) {
      real *Q = grid.getField(p1Var(v));
      grid.getField(F_P1S + v)[cIdx]         = (l1 < cE && r1 < cE) ? (real)0.5*(Q[r1] - Q[l1]) : (real)0;
      grid.getField(F_P1S + P1_NV + v)[cIdx] = (d1 < cE && u1 < cE) ? (real)0.5*(Q[u1] - Q[d1]) : (real)0;
    }
    grid.getField(F_P1NEW)[cIdx] = 1;
  END_CELL_LOOP
}

// mark every existing cell before an adaptation; a 0 after the sort is a block
// created this cycle
__global__ void p1MarkKernel(CompressibleSolver &grid) {
  START_CELL_LOOP
    grid.getField(F_P1NEW)[cIdx] = 1;
  END_CELL_LOOP
}

// slopes of the blocks created this cycle: the parent polynomial restricted to
// the child (slopes halved; the mean was predicted by the inverse wavelet).
// Called level by level, coarse to fine, so a new parent is filled first.
__global__ void p1ProlongNewKernel(CompressibleSolver &grid, i32 lvlOnly) {
  START_CELL_LOOP
    GET_CELL_INDICES
    const u64 loc = grid.bLocList[bIdx];
    if (loc != kEmpty) {
      i32 lvl, ib, jb, kb;
      grid.decode(loc, lvl, ib, jb, kb);
      real *Mk = grid.getField(F_P1NEW);
      if (lvl == lvlOnly && Mk[cIdx] == (real)0) {
        const i32 prntIdx = grid.prntIdxList[bIdx];
        if (prntIdx != bEmpty && grid.isInteriorBlock(lvl, ib, jb, kb)) {
          const i32 ip = i/2 + ib%2 * blockSize/2, jp = j/2 + jb%2 * blockSize/2;
          const i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize/2);
          const i32 p = grid.getNbrIdx(prntIdx, ip, jp, kp);
          if (p < bEmpty*blockSizeTot)
            for (i32 s = 0; s < 2*P1_NV; s++)
              grid.getField(F_P1S + s)[cIdx] = (real)0.5*grid.getField(F_P1S + s)[p];
        }
        Mk[cIdx] = 1;
      }
    }
  END_CELL_LOOP
}

// =============================================================================
// --p1 on cut elements (tables from CompressibleSolver::buildP1Cut)
// =============================================================================
struct P1Geo { real gx, gy, h, area, m11, m12, m22; };

__device__ inline i32 p1ElemIdx(CompressibleSolver &grid, i32 h) {
  if (!grid.p1ElemOfCell) return -1;
  return cutIsPiece(h) ? (grid.p1ElemOfPiece ? grid.p1ElemOfPiece[cutPieceOf(h)] : -1) : grid.p1ElemOfCell[h];
}
// basis geometry of a handle: its P1Elem, or the regular cell (centre, h,
// diagonal mass matrix -- identical to the regular kernel on square cells)
__device__ inline void p1GeoOf(CompressibleSolver &grid, i32 h, P1Geo &G) {
  const i32 e = p1ElemIdx(grid, h);
  if (e >= 0) {
    const P1Elem &E = grid.p1Elem[e];
    G.gx = E.gx; G.gy = E.gy; G.h = E.h; G.area = E.area; G.m11 = E.m11; G.m12 = E.m12; G.m22 = E.m22;
    return;
  }
  const i32 c = cutIsPiece(h) ? grid.cutPiece[cutPieceOf(h)].cell : h;
  const i32 b = c/blockSizeTot, cc = c%blockSizeTot;
  i32 lvl, ib, jb, kb; grid.decode(grid.bLocList[b], lvl, ib, jb, kb);
  Vec3 p = grid.getCellPos(lvl, ib, jb, kb, cc%blockSize, (cc/blockSize)%blockSize, 0);
  const real dx = grid.getDx(lvl), dy = grid.getDy(lvl);
  G.gx = p[0]; G.gy = p[1]; G.h = fmin(dx, dy); G.area = dx*dy; G.m11 = G.m22 = (real)12/(dx*dy); G.m12 = 0;
}
// polynomial of a handle (piece DOFs are conserved already)
__device__ inline void p1LoadH(CompressibleSolver &grid, i32 h, P1Poly &P) {
  if (!cutIsPiece(h)) { p1Load(grid, h, P); return; }
  const i32 k = cutPieceOf(h); const size_t cap = grid.cutPieceQCap;
  for (i32 f = 0; f < 5; f++) { P.q[f] = grid.cutPieceQ[(size_t)f*cap + k]; P.sx[f] = 0; P.sy[f] = 0; }
  for (i32 v = 0; v < P1_NV; v++) {
    const i32 f = p1Var(v);
    P.sx[f] = grid.cutPieceSX[(size_t)v*cap + k];
    P.sy[f] = grid.cutPieceSX[(size_t)(P1_NV + v)*cap + k];
  }
}
__device__ __forceinline__ real *p1SlopePtr(CompressibleSolver &grid, i32 h, i32 s) {
  return cutIsPiece(h) ? &grid.cutPieceSR[(size_t)s*grid.cutPieceQCap + cutPieceOf(h)]
                       : &grid.getField(F_P1SR + s)[h];
}
__device__ __forceinline__ real p1Pressure(Vec5 q) {
  const real r = fmax(q[0], (real)1e-300);
  return (gam - (real)1)*(q[4] - (real)0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/r);
}
// one face piece into an element through its handle.  sgnLen = +L when the
// flux along n ENTERS the element, -L when it leaves; phi at the physical
// points relative to the element's own centroid; the inverse mass matrix
// mixes the two slope rows (linear, so each contributor applies it itself).
__device__ inline void p1ScatterH(CompressibleSolver &grid, i32 h, const P1Geo &G, real sgnLen,
                                  const Vec5 *Fg, const real *wg, const real *xg, const real *yg, i32 ng) {
  for (i32 f = 0; f < 5; f++) {
    real m = 0, mx = 0, my = 0;
    for (i32 g = 0; g < ng; g++) {
      const real F = wg[g]*Fg[g].data[f];
      m += F;  mx += F*(xg[g] - G.gx)/G.h;  my += F*(yg[g] - G.gy)/G.h;
    }
    atomicAdd(cutRhsPtr(grid, h, f), sgnLen*m/G.area);
    if (f == F_RHOW) continue;
    const i32 v = (f < 3) ? f : 3;
    const real r1 = sgnLen*mx, r2 = sgnLen*my;
    atomicAdd(p1SlopePtr(grid, h, v),         G.m11*r1 + G.m12*r2);
    atomicAdd(p1SlopePtr(grid, h, P1_NV + v), G.m12*r1 + G.m22*r2);
  }
}

// open face pieces between two elements: 2 Gauss points, each side's own trace
__global__ void p1SegKernel(CompressibleSolver &grid) {
  const i32 s = blockIdx.x*blockDim.x + threadIdx.x;
  if (s >= grid.nP1Seg) return;
  const P1Seg &S = grid.p1Seg[s];
  P1Geo GA, GB; p1GeoOf(grid, S.hA, GA); p1GeoOf(grid, S.hB, GB);
  P1Poly PA, PB; p1LoadH(grid, S.hA, PA); p1LoadH(grid, S.hB, PB);
  const real ex = S.x1 - S.x0, ey = S.y1 - S.y0, len = sqrt(ex*ex + ey*ey);
  const real w2[2] = {(real)0.5, (real)0.5};
  real xg[2], yg[2]; Vec5 Fg[2];
  const Vec3 nrm(S.nx, S.ny, (real)0);
  for (i32 g = 0; g < 2; g++) {
    const real sg = (real)0.5 + (g ? P1_G1 : -P1_G1);
    xg[g] = S.x0 + sg*ex; yg[g] = S.y0 + sg*ey;
    Vec5 qA = p1Trace(PA, (xg[g] - GA.gx)/GA.h, (yg[g] - GA.gy)/GA.h);
    Vec5 qB = p1Trace(PB, (xg[g] - GB.gx)/GB.h, (yg[g] - GB.gy)/GB.h);
    Fg[g] = grid.hllcFlux(qA, qB, nrm);
  }
  p1ScatterH(grid, S.hA, GA, -len, Fg, w2, xg, yg, 2);
  p1ScatterH(grid, S.hB, GB,  len, Fg, w2, xg, yg, 2);
}

// every cut element: the volume rule on its own polynomial and the wall
// pressure integral over its wall edges (slip wall: F.n = p n)
__global__ void p1ElemKernel(CompressibleSolver &grid) {
  const i32 e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= grid.nP1Elem) return;
  const P1Elem &E = grid.p1Elem[e];
  const i32 h = E.handle;
  P1Geo G; G.gx = E.gx; G.gy = E.gy; G.h = E.h; G.area = E.area; G.m11 = E.m11; G.m12 = E.m12; G.m22 = E.m22;
  P1Poly P; p1LoadH(grid, h, P);
  real r1[5] = {0,0,0,0,0}, r2[5] = {0,0,0,0,0}, m[5] = {0,0,0,0,0};
  for (i32 q = E.q0; q < E.q0 + E.nq; q++) {
    const P1Qpt &Q = grid.p1Qpt[q];
    const real xi = (Q.x - G.gx)/G.h, eta = (Q.y - G.gy)/G.h;
    Vec5 u = p1Trace(P, xi, eta);
    if (Q.w != (real)0) {                       // volume point: int F.grad(phi), grad(phi1) = (1/h, 0)
      Vec5 F = p1EulerFlux(u, 0), Gf = p1EulerFlux(u, 1);
      for (i32 f = 0; f < 5; f++) { r1[f] += Q.w*F[f]/G.h; r2[f] += Q.w*Gf[f]/G.h; }
    } else {                                    // wall point: -(F.n) phi with F.n = p (nx, ny) * length
      const real p = p1Pressure(u);
      m[1]  -= p*Q.nx;      m[2]  -= p*Q.ny;
      r1[1] -= p*Q.nx*xi;   r2[1] -= p*Q.nx*eta;
      r1[2] -= p*Q.ny*xi;   r2[2] -= p*Q.ny*eta;
    }
  }
  for (i32 f = 0; f < 5; f++) if (m[f] != (real)0) atomicAdd(cutRhsPtr(grid, h, f), m[f]/G.area);
  for (i32 v = 0; v < P1_NV; v++) {
    const i32 f = p1Var(v);
    atomicAdd(p1SlopePtr(grid, h, v),         G.m11*r1[f] + G.m12*r2[f]);
    atomicAdd(p1SlopePtr(grid, h, P1_NV + v), G.m12*r1[f] + G.m22*r2[f]);
  }
}

// slope limiter on the cut elements: Barth-Jespersen / Venkatakrishnan (the
// same gradLim switch) per conserved variable against the MEANS of the
// elements across its face pieces, the polynomial checked at the element's
// own polygon vertices (the extrema of a linear function).  Cell means are
// conserved here (after the update); piece means always are.
__device__ i32 g_p1LimPieces = 1;   // debug: 0 = the cut limiter leaves piece DOFs alone
__global__ void p1LimitCutKernel(CompressibleSolver &grid) {
  const i32 e = blockIdx.x*blockDim.x + threadIdx.x;
  if (e >= grid.nP1Elem) return;
  const P1Elem &E = grid.p1Elem[e];
  const i32 h = E.handle;
  if (cutIsPiece(h) && !g_p1LimPieces) return;
  const real eps2 = (grid.gradLim == 2) ? pow(grid.gradLimK*E.h, (real)3) : (real)0;
  auto meanOf = [&](i32 hh, i32 f) -> real {
    return cutIsPiece(hh) ? grid.cutPieceQ[(size_t)f*grid.cutPieceQCap + cutPieceOf(hh)] : grid.getField(f)[hh]; };
  for (i32 v = 0; v < P1_NV; v++) {
    const i32 f = p1Var(v);
    const real qc = meanOf(h, f);
    real qmin = qc, qmax = qc;
    for (i32 n = grid.p1ElemNbrOff[e]; n < grid.p1ElemNbrOff[e+1]; n++) {
      const i32 hn = grid.p1ElemNbr[n];
      if (hn == CUT_DEAD) continue;
      const real qn = meanOf(hn, f);
      qmin = fmin(qmin, qn); qmax = fmax(qmax, qn);
    }
    real *SX = cutIsPiece(h) ? &grid.cutPieceSX[(size_t)v*grid.cutPieceQCap + cutPieceOf(h)] : &grid.getField(F_P1S + v)[h];
    real *SY = cutIsPiece(h) ? &grid.cutPieceSX[(size_t)(P1_NV + v)*grid.cutPieceQCap + cutPieceOf(h)] : &grid.getField(F_P1S + P1_NV + v)[h];
    const real sx = *SX, sy = *SY;
    const real dmax = qmax - qc, dmin = qmin - qc;
    real a = 1;
    for (i32 q = E.q0; q < E.q0 + E.nq; q++) {
      const P1Qpt &Q = grid.p1Qpt[q];
      if (Q.w != (real)0 || Q.nx != (real)0 || Q.ny != (real)0) continue;   // vertices only
      const real d = sx*(Q.x - E.gx)/E.h + sy*(Q.y - E.gy)/E.h;
      if (fabs(d) < (real)1e-300) continue;
      const real D = (d > 0) ? dmax : dmin;
      real psi;
      if (grid.gradLim == 2) { const real D2 = D*D, d2 = d*d; psi = (D2 + eps2 + (real)2*d*D)/(D2 + (real)2*d2 + d*D + eps2 + (real)1e-300); }
      else psi = fmin((real)1, D/d);
      a = fmin(a, fmax(psi, (real)0));
    }
    if (grid.cutDbg && cutIsPiece(h) && grid.iter < 2 && v == 0)
      printf("[p1lim] iter %d piece %d elem %d: rho mean %.6f nbrs [%.6f, %.6f] (%d nbrs) slopes (%.3e, %.3e) -> a = %.3f  h %.4f area %.3e nq %d\n",
             grid.iter, cutPieceOf(h), e, (double)qc, (double)qmin, (double)qmax, grid.p1ElemNbrOff[e+1]-grid.p1ElemNbrOff[e], (double)sx, (double)sy, (double)a, (double)E.h, (double)E.area, E.nq);
    *SX = a*sx; *SY = a*sy;
  }
}
