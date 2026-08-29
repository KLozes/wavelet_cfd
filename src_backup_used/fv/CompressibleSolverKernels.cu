#include <cstdio>
#include "CompressibleSolverKernels.cuh"
#include "KtauSst.h"

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
            for (i32 f = 0; f < NEVOLVE; f++) F[f][cIdx] = F[f][bcIdx];
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

        bool xWall = (ib < 0 || ib >= gridSize[0]);
        bool yWall = (jb < 0 || jb >= gridSize[1]);
        bool zWall = (kb < 0 || kb >= gridSize[2]);

        if (grid.bcType != 4 || !(ib < 0)) {
          Rho[cIdx]  = Rho[bcIdx];      // Neumann density and energy, except at
          RhoE[cIdx] = RhoE[bcIdx];     // the bcType 4 Dirichlet faces
        }

        if (grid.bcType == 0) {
          // slip wall: reflect the wall-normal momentum, keep tangential
          RhoU[cIdx] = (xWall ? -1.0 : 1.0) * RhoU[bcIdx];
          RhoV[cIdx] = (yWall ? -1.0 : 1.0) * RhoV[bcIdx];
          RhoW[cIdx] = (zWall ? -1.0 : 1.0) * RhoW[bcIdx];
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
          if (udn < (real)0) {                      // inflow
            const real r = 1.0, u = grid.fsU, v = grid.fsV, p = grid.fsP;
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
            if (prim) RhoE[cIdx] = grid.fsP;
            else {
              const real r = fmax(Rho[cIdx], (real)1e-30);
              const real ke = 0.5*(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx]
                                 + RhoW[cIdx]*RhoW[cIdx])/r;
              RhoE[cIdx] = grid.fsP/(gam - 1.0) + ke;
            }
          }
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

  START_CELL_LOOP

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

  START_CELL_LOOP

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
__device__ unsigned long long g_ibNup;        // faces with the body ABOVE (n_d < 0)
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
                                real **F, i32 nf, real *out, bool tally = false)
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
        if (grid.getField(F_IBM)[m] <= (real)0.5) continue;   // solid: no solution (cached mask)
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
  if (!ibSample(grid, ip, lvl, bIdx, ci, cj, ck, gi, gj, gk, Fs, 5, q)) return false;
  const real un = q[1]*n[0] + q[2]*n[1] + q[3]*n[2];
  const real sc = un*(dFc/sStar - (real)1);        // u - un*n + un*(dFc/s*)*n
  Vec5 qW(q[0], q[1] + sc*n[0], q[2] + sc*n[1], q[3] + sc*n[2], q[4]);
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
  if (!grid.rans || fcPos[0] < grid.plateX0) {
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
  const real uFc  = fmax(utMag - dudy*(dIp - ((grid.ibDfcMode==1)?dFcRef:dFc)), (real)0);  // Eq. (36)
  const real tauW = rIp*uTau*uTau;
  const real nd   = n[d];

  // convective flux of the FC state through the grid face (normal e_d)
  const real vd = uFc*((d==0)?tx:((d==1)?ty:tz));
  const real ke = (real)0.5*rIp*uFc*uFc;
  const real E  = pIp/(gam-(real)1) + ke;
  Fw[0] = rIp*vd;
  Fw[1] = rIp*vd*uFc*tx + ((d==0)?pIp:(real)0);
  Fw[2] = rIp*vd*uFc*ty + ((d==1)?pIp:(real)0);
  Fw[3] = rIp*vd*uFc*tz + ((d==2)?pIp:(real)0);
  Fw[4] = (E + pIp)*vd;
  // minus the viscous part: tau_{d i} = tau_w (t_i n_d + t_d n_i)
  const real td = (d==0)?tx:((d==1)?ty:tz);
  Fw[1] -= tauW*(tx*nd + td*n[0]);
  Fw[2] -= tauW*(ty*nd + td*n[1]);
  Fw[3] -= tauW*(tz*nd + td*n[2]);
  Fw[4] -= tauW*uFc*nd;              // u_j tau_{d j} = tau_w u_FC n_d  (t.n = 0)

  // Eq. (39) turbulence values and the one-sided fluxes into the fluid cell
  real kFc, tFc;
  ktau::wallBcKTau(uTau, (grid.ibDfcMode==2)?dFcRef:dFc, tIp, dIp, kFc, tFc);
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
                      rIp*kFc*tFc, Rho[fluidIdx]*k1*t1, (real)0, (real)0, kFc, k1, fdL, fdR);
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
                      rw*kFc*tFc, r1*k1*t1, (real)0, (real)0, kFc, k1, fdL, fdR);
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
        real Fi[5], Ki, Ti, Ut = 0;
        if (ibWallFlux(grid, Rho, U, V, W, P, grid.getField(F_RHOK),
                       grid.getField(F_RHOTAU), grid.getField(F_TF1),
                       lvl, fc, 1, dy, cIdx, Fi, Ki, Ti, Ut,
                       bIdx, i, j, k, ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k))
          Sc[cIdx] = Ut;
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
    if (grid.immerserdBcType != 0 && grid.getField(F_IBM)[cIdx] <= (real)0.5) {
      MuT[cIdx] = 0;
      TF1[cIdx] = 0;
    }
    else {

    // Wall distance from the CACHED level set when the wall IS the level set
    // (wallGeom 2); the analytic branch covers the grid-aligned plate.
    const real d   = (grid.wallGeom == 2)
                   ? fmax(-grid.getField(F_PHI)[cIdx], (real)0)
                   : grid.wallDistance(pos);
    // A fluid cell lies wholly outside the body, so its centre is at least half a
    // cell from the wall; flooring d there keeps r_d bounded whatever the level
    // set returns.  (Flooring d ITSELF was tried 2026-08-26 and is WRONG -- it
    // makes the airfoil case worse, polluting the mean flow as well.)
    const real rd  = (grid.dCutoff > 0)
                   ? fmax(grid.dCutoff/fmax(d, (real)0.5*fmin(hh[0], hh[1])), (real)1)
                   : (real)1;

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
                                    i32 cIdx, i32 lIdx, Vec3 pos)
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

  atomicAdd(&RhsK[cIdx],  (Fck - Fdk)*area);
  atomicAdd(&RhsK[lIdx], -(Fck - Fdk)*area);
  // the tau~ diffusion pair is scattered per side: this face is the HIGH face of
  // the left cell (which therefore takes +fdL) and the LOW face of the right
  // cell (which takes -fdR), reproducing (f_L^{i+1/2} - f_R^{i-1/2})/h.
  atomicAdd(&RhsT[cIdx],  Fct*area - fdR*area);
  atomicAdd(&RhsT[lIdx], -Fct*area + fdL*area);
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
  END_CELL_LOOP
}

__global__ void ibStampGeometryKernel(CompressibleSolver &grid) {
  real *Phi = grid.getField(F_PHI);
  real *Ibm = grid.getField(F_IBM);
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
      if (nonFluid && phi <= (real)2.5*h) {
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
        const real sStar = (real)2*h;
        Vec3 mir(surf[0] + sStar*n[0], surf[1] + sStar*n[1], surf[2] + sStar*n[2]);
        real dS = -grid.getBoundaryLevelSet(mir);   // measured, not assumed = s*
        if (dS < (real)0.5*h) dS = sStar;           // curvature pathology guard
        real *Fp[7] = {Rho, U, V, W, P, Tau, K};
        real q[7];
        if (ibSample(grid, mir, lvl, bIdx, i, j, k,
                     ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k, Fp, 7, q)) {
          Rho[cIdx] = q[0];                     // Neumann density and pressure
          P[cIdx]   = q[4];
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
            Tau[cIdx] = q[5];
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

  START_CELL_LOOP
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
    if (grid.immerserdBcType != 0 && grid.ibGhostFree) {
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
    qL[0] = grid.tvdRec(Rho[l2R], Rho[l1R], Rho[cIdx]);
    qR[0] = grid.tvdRec(Rho[r1R], Rho[cIdx],  Rho[l1R]);
    qD[0] = grid.tvdRec(Rho[d2R], Rho[d1R], Rho[cIdx]);
    qU[0] = grid.tvdRec(Rho[u1R], Rho[cIdx],  Rho[d1R]);
    qB[0] = grid.tvdRec(Rho[b2R], Rho[b1R], Rho[cIdx]);
    qF[0] = grid.tvdRec(Rho[f1R], Rho[cIdx],  Rho[b1R]);

    qL[1] = grid.tvdRec(U[l2R], U[l1R], U[cIdx]);
    qR[1] = grid.tvdRec(U[r1R], U[cIdx],  U[l1R]);
    qD[1] = grid.tvdRec(U[d2R], U[d1R], U[cIdx]);
    qU[1] = grid.tvdRec(U[u1R], U[cIdx],  U[d1R]);
    qB[1] = grid.tvdRec(U[b2R], U[b1R], U[cIdx]);
    qF[1] = grid.tvdRec(U[f1R], U[cIdx],  U[b1R]);

    qL[2] = grid.tvdRec(V[l2R], V[l1R], V[cIdx]);
    qR[2] = grid.tvdRec(V[r1R], V[cIdx],  V[l1R]);
    qD[2] = grid.tvdRec(V[d2R], V[d1R], V[cIdx]);
    qU[2] = grid.tvdRec(V[u1R], V[cIdx],  V[d1R]);
    qB[2] = grid.tvdRec(V[b2R], V[b1R], V[cIdx]);
    qF[2] = grid.tvdRec(V[f1R], V[cIdx],  V[b1R]);

    qL[3] = grid.tvdRec(W[l2R], W[l1R], W[cIdx]);
    qR[3] = grid.tvdRec(W[r1R], W[cIdx],  W[l1R]);
    qD[3] = grid.tvdRec(W[d2R], W[d1R], W[cIdx]);
    qU[3] = grid.tvdRec(W[u1R], W[cIdx],  W[d1R]);
    qB[3] = grid.tvdRec(W[b2R], W[b1R], W[cIdx]);
    qF[3] = grid.tvdRec(W[f1R], W[cIdx],  W[b1R]);

    qL[4] = grid.tvdRec(P[l2R], P[l1R], P[cIdx]);
    qR[4] = grid.tvdRec(P[r1R], P[cIdx],  P[l1R]);
    qD[4] = grid.tvdRec(P[d2R], P[d1R], P[cIdx]);
    qU[4] = grid.tvdRec(P[u1R], P[cIdx],  P[d1R]);
    qB[4] = grid.tvdRec(P[b2R], P[b1R], P[cIdx]);
    qF[4] = grid.tvdRec(P[f1R], P[cIdx],  P[b1R]);

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
    if (grid.immerserdBcType != 0 && (grid.rans || grid.ibFluxRecon != 1)) {
      real Fi[5], Ki, Ti, Ut;
      const bool trace2 = (!grid.rans && grid.ibFluxRecon == 2);
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
      #define IB_NBR_FLUID(IDX, PX, PY, PZ, HH) \
        (((IDX) < cEmptyI) ? (Ibm[IDX] > (real)0.5) \
                           : grid.isFluidCell(Vec3((PX),(PY),(PZ)), (HH)))
      if (fC != IB_NBR_FLUID(l1Idx, cpos[0]-dx, cpos[1], cpos[2], dx)) {
        Vec3 fc(cpos[0]-(real)0.5*dx, cpos[1], cpos[2]);
        if (trace2) {
          if (ibFaceTraceFlux(grid, Rho, U, V, W, P, lvl, fc, 0, dx,
                              fC ? cIdx : l1Idx, fC, Fi, bIdx, i, j, k,
                              ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k))
            for (i32 nq = 0; nq < 5; nq++) fluxL[nq] = Fi[nq];
        }
        else if (ibWallFlux(grid, Rho, U, V, W, P, grid.getField(F_RHOK),
                       grid.getField(F_RHOTAU), grid.getField(F_TF1),
                       lvl, fc, 0, dx, fC ? cIdx : l1Idx, Fi, Ki, Ti, Ut,
                       bIdx, i, j, k, ib*blockSize+i, jb*blockSize+j, kb*blockSizeZ+k)) {
          for (i32 n = 0; n < 5; n++) fluxL[n] = Fi[n];
          ibX = true; ibFluidX = fC ? cIdx : l1Idx; ibKx = Ki; ibTx = Ti;
        }
      }
      if (fC != IB_NBR_FLUID(d1Idx, cpos[0], cpos[1]-dy, cpos[2], dy)) {
        atomicAdd(&g_ibDetect, 1ULL);
        Vec3 fc(cpos[0], cpos[1]-(real)0.5*dy, cpos[2]);
        if (trace2) {
          if (ibFaceTraceFlux(grid, Rho, U, V, W, P, lvl, fc, 1, dy,
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
          for (i32 n = 0; n < 5; n++) fluxD[n] = Fi[n];
          ibY = true; ibFluidY = fC ? cIdx : d1Idx; ibKy = Ki; ibTy = Ti;
          grid.getField(F_SCRATCH)[ibFluidY] = Ut;      // for the C_f dump
        }
      }
      const bool fL = IB_NBR_FLUID(l1Idx, cpos[0]-dx, cpos[1], cpos[2], dx);
      const bool fD = IB_NBR_FLUID(d1Idx, cpos[0], cpos[1]-dy, cpos[2], dy);
      ktX = fC && fL;
      ktY = fC && fD;
      if (!grid.pseudo2D) {
        const bool fB = IB_NBR_FLUID(b1Idx, cpos[0], cpos[1], cpos[2]-dz, dz);
        if (fC != fB) ibZ = true;   // handled in the z block
        ktZ = fC && fB;
      }
      #undef IB_NBR_FLUID
    }

    // ---- pressure-tight volume penalization (Reiss 2021) -------------------
    // Paper Eqs. (4)-(6); see CompressibleSolver::brinkPhi and the block comment
    // on ibBrink.  Every face flux is scaled by the face volume fraction and the
    // RHS is divided by the cell's own volume fraction.
    //
    // Two implementation choices matter.  (a) We divide EACH FACE by the
    // receiving cell's phi rather than dividing the summed RHS, so every factor
    // is the O(1) ratio phi_face/phi_cell -- with phi ~ 1e-8 the alternative
    // multiplies then divides by 1e-8 and loses the result to cancellation in
    // fp32.  (b) The momentum source uses the SAME phi_face as the flux, which
    // makes it exactly -p_cell * (that same weight): a quiescent uniform-pressure
    // state then cancels to zero in every cell bit-for-bit, however steep phi is.
    // That exact cancellation is the "pressure-tight" property -- lose it and the
    // smeared wall leaks a spurious force proportional to grad(phi).
    real wLc = 1, wLn = 1, wDc = 1, wDn = 1, wBc = 1, wBn = 1, phiC = 1;
    real sCb = 0, hbF = 1;   // cell signed distance and the finest h, for the Darcy mask
    if (grid.ibBrink && grid.immerserdBcType != 0) {
      real *PhiG = grid.getField(F_PHI);
      const i32 cEmptyI = bEmpty*blockSizeTot;
      // Same validity guard the wall-face code uses: the cached level set is
      // only meaningful for real cells; anything past cEmptyI is evaluated.
      // Level set is positive INSIDE, so the signed distance is its negation.
      #define BRINK_S(IDX, PX, PY, PZ) \
        (-(((IDX) < cEmptyI) ? PhiG[IDX] \
                             : grid.getBoundaryLevelSet(Vec3((PX),(PY),(PZ)))))
      // delta is a FIXED PHYSICAL length taken from the finest level, never the
      // local cell size.  phi is a property of the immersed body, so it has to
      // be one single field: keying it to the local h redefines the wall across
      // every coarse-fine interface, the two sides of a hanging face disagree
      // about phi, and the solution diverges there (nLvls 1 is fine, nLvls >= 2
      // goes NaN).  The band is refined to the finest level anyway.
      const real hb = fmin(grid.getDx(grid.nLvls-1),
                           grid.pseudo2D ? grid.getDx(grid.nLvls-1)
                                         : grid.getDy(grid.nLvls-1));
      const real sC = BRINK_S(cIdx,  cpos[0], cpos[1], cpos[2]);
      sCb = sC; hbF = hb;
      const real sL = BRINK_S(l1Idx, cpos[0]-dx, cpos[1], cpos[2]);
      const real sD = BRINK_S(d1Idx, cpos[0], cpos[1]-dy, cpos[2]);
      phiC = grid.brinkPhi(sC, hb);
      // Face value from the average signed distance: second order, and it costs
      // no extra level-set evaluations on top of the two cached cell values.
      const real pfL = grid.brinkPhi((real)0.5*(sC + sL), hb);
      const real pfD = grid.brinkPhi((real)0.5*(sC + sD), hb);
      wLc = pfL/phiC;  wLn = pfL/grid.brinkPhi(sL, hb);
      wDc = pfD/phiC;  wDn = pfD/grid.brinkPhi(sD, hb);
      if (!grid.pseudo2D) {
        const real sB = BRINK_S(b1Idx, cpos[0], cpos[1], cpos[2]-dz);
        const real pfB = grid.brinkPhi((real)0.5*(sC + sB), hb);
        wBc = pfB/phiC;  wBn = pfB/grid.brinkPhi(sB, hb);
      }
      #undef BRINK_S
    }

    real *Rhs[5] = {RhsRho, RhsRhoU, RhsRhoV, RhsRhoW, RhsRhoE};
    for (i32 n = 0; n < 5; n++) {
      atomicAdd(&Rhs[n][cIdx],    fluxL[n]*ax*wLc + fluxD[n]*ay*wDc);
      atomicAdd(&Rhs[n][l1Idx], - fluxL[n]*ax*wLn);
      atomicAdd(&Rhs[n][d1Idx], - fluxD[n]*ay*wDn);
    }
    if (grid.ibBrink && grid.immerserdBcType != 0) {
      // -p grad(phi) in flux-scatter form, sharing the face weights above.
      atomicAdd(&RhsRhoU[cIdx],  -P[cIdx] *ax*wLc);
      atomicAdd(&RhsRhoU[l1Idx],  P[l1Idx]*ax*wLn);
      atomicAdd(&RhsRhoV[cIdx],  -P[cIdx] *ay*wDc);
      atomicAdd(&RhsRhoV[d1Idx],  P[d1Idx]*ay*wDn);
      if (grid.ibBrinkRate > (real)0) {
        // Darcy friction through the RETREATED mask (see ibBrinkRate).  Damps
        // the body interior, where the collapsing phi would otherwise amplify
        // any inbound disturbance like 1/sqrt(phi); ~0 at the wall, so the wall
        // itself stays slip.  Explicit stability needs chi*dt < O(1), and
        // chi*dt = ibBrinkRate*CFL by construction.
        const real rC = fmax(Rho[cIdx], (real)1e-30);
        const real cS = sqrt(fabs(gam*P[cIdx]/rC));
        const real mask = grid.brinkDarcyMask(sCb, hbF);
        const real chi = grid.ibBrinkRate*mask
                       * (fabs(U[cIdx]) + fabs(V[cIdx]) + fabs(W[cIdx]) + cS)/hbF;
        atomicAdd(&RhsRhoU[cIdx], -chi*rC*U[cIdx]);
        atomicAdd(&RhsRhoV[cIdx], -chi*rC*V[cIdx]);
        if (!grid.pseudo2D) atomicAdd(&RhsRhoW[cIdx], -chi*rC*W[cIdx]);
        atomicAdd(&RhsRhoE[cIdx], -chi*rC*(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx]
                                         + W[cIdx]*W[cIdx]));
      }
    }

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
      if (ibX) { atomicAdd(&RhsK[ibFluidX], ibKx*ax); atomicAdd(&RhsT[ibFluidX], ibTx*ax);
        if (grid.wallPointImplicit) {
          atomicAdd(&LamK[ibFluidX], (real)3*fabs(ibKx)*ax
                    / fmax(Rho[ibFluidX]*K[ibFluidX],   (real)1e-30));
          atomicAdd(&LamT[ibFluidX], (real)3*fabs(ibTx)*ax
                    / fmax(Rho[ibFluidX]*Tau[ibFluidX], (real)1e-30));
        }
      }
      else if (ktX) ktauFaceFlux(grid, Rho, P, K, Tau, TF1, RhsK, RhsT,
                        bIdx, i, j, k, 0, dx, ax, fluxL[0], cIdx, l1Idx, cpos);
      if (ibY) { atomicAdd(&RhsK[ibFluidY], ibKy*ay); atomicAdd(&RhsT[ibFluidY], ibTy*ay);
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
        atomicAdd(&RhsK[cIdx], FwallK*ay);
        atomicAdd(&RhsT[cIdx], FwallT*ay);
        if (grid.wallPointImplicit) {
          atomicAdd(&LamK[cIdx], (real)3*fabs(FwallK)*ay
                    / fmax(Rho[cIdx]*K[cIdx],   (real)1e-30));
          atomicAdd(&LamT[cIdx], (real)3*fabs(FwallT)*ay
                    / fmax(Rho[cIdx]*Tau[cIdx], (real)1e-30));
        }
      } else if (ktY) {
        ktauFaceFlux(grid, Rho, P, K, Tau, TF1, RhsK, RhsT,
                     bIdx, i, j, k, 1, dy, ay, fluxD[0], cIdx, d1Idx, cpos);
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
      for (i32 n = 0; n < 5; n++) {
        atomicAdd(&Rhs[n][cIdx],    fluxB[n]*az*wBc);
        atomicAdd(&Rhs[n][b1Idx], - fluxB[n]*az*wBn);
      }
      if (grid.ibBrink && grid.immerserdBcType != 0) {
        atomicAdd(&RhsRhoW[cIdx],  -P[cIdx] *az*wBc);
        atomicAdd(&RhsRhoW[b1Idx],  P[b1Idx]*az*wBn);
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

  START_CELL_LOOP

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
    // consume the wall-flux relaxation rates: they are per-stage quantities,
    // and this also clears stamps on cells that skipped the update (parents)
    if (grid.rans && grid.wallPointImplicit) {
      real *LK = grid.getField(F_LAMK), *LT = grid.getField(F_LAMT);
      if (LK[cIdx] != 0) LK[cIdx] = 0;
      if (LT[cIdx] != 0) LT[cIdx] = 0;
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
    bool wallBand = (grid.wallGeom != 0) && grid.rans
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

    // ---- volume-penalization band -----------------------------------------
    // Hold the whole phi transition at the finest level; see inBrinkBand().
    // Same shape as the wall-model band below, and for the same reason: it must
    // sit outside the detail-driven branch, which only starts at level 1, so it
    // can lift level 0 too.
    if (grid.ibBrink && grid.immerserdBcType != 0
        && grid.isInteriorBlock(lvl, ib, jb, kb)
#ifdef USE_MGPU
        && grid.isOwnedBlock(lvl, ib, jb, kb)
#endif
        && grid.inBrinkBand(ls)) {
      grid.bFlagsList[bIdx] = KEEP;
      if (lvl > 0) grid.bFlagsList[grid.prntIdxList[bIdx]] = KEEP;
      if (lvl < grid.nLvls-1) {
        i32 bSize = blockSize/2;
        i32 kc = grid.pseudo2D ? kb : (2*kb + k/bSize);
        grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, kc);
      }
    }

    if (wallBand && grid.wallDistance(pos) < grid.wallFineBand) {
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

        if (abs(Q[cIdx]/mag) > grid.waveletThresh || abs(ls) < dx) {
          if (lvl < grid.nLvls-1 && (abs(Q[cIdx]/mag) > grid.waveletThresh*2 || abs(ls) < dx)) {
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

__global__ void interpolateFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag == GHOST) {
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
    }

  END_CELL_LOOP
}

__global__ void restrictFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    bool restrictCell = grid.pseudo2D ? (i%2==0 && j%2==0)
                                      : (i%2==0 && j%2==0 && k%2==0);
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST && restrictCell) {

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
