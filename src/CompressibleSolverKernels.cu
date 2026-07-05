#include "CompressibleSolverKernels.cuh"

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

    // carry all evolved DOFs (P0 ρ,E + momentum avg + RT0 slopes) through the sort
    for (i32 f = 0; f < NEVOLVE; f++) {
      grid.getField(f)[cIdx] = grid.getField(F_OLD + f)[cIdxOld];
    }
    grid.bFlagsList[bIdxOld] = DELETE;

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
  real *Gx  = grid.getField(F_GX);
  real *Gy  = grid.getField(F_GY);
  real *Gz  = grid.getField(F_GZ);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);

    // RT0 slope DOFs default to zero (all zero-velocity ICs); the vortex IC below
    // overwrites them with the projected momentum gradients.
    Gx[cIdx] = 0.0; Gy[cIdx] = 0.0; Gz[cIdx] = 0.0;

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

    if (grid.icType == 1) {
      //
      // 2D circular Sod explosion (uniform in z -> pseudo-2D).  A circular
      // region of high-pressure gas drives a cylindrical shock outward.  The
      // inner pressure is configurable (vortexAdvect, unused otherwise here):
      // pIn = 1 is the classic 10:1 Sod ratio; pIn = 10 a strong 100:1 blast.
      //
      real pIn = (grid.vortexAdvect > 0.0) ? grid.vortexAdvect : 1.0;
      real cx = grid.domainSize[0]/2;
      real cy = grid.domainSize[1]/2;
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
      // 3D spherical Sod explosion (true 3D — exercises all three RT0 slope DOFs
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
      // zero-velocity IC → RT0 slope DOFs stay 0 (already set above)
    }

    if (grid.icType == 2) {
      //
      // Isentropic vortex, z-uniform (validates the RT0 low-Mach / stationarity
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

      // RT0 slope DOFs from the face-midpoint momenta:  Gx = (ρu|_xR − ρu|_xL)/dx
      real rR, uR, vR, pR;  vortexPaperExact(pos[0]+0.5*dx, pos[1], u0, v0, cx, cy, rR, uR, vR, pR);
      real rL, uL, vL, pL;  vortexPaperExact(pos[0]-0.5*dx, pos[1], u0, v0, cx, cy, rL, uL, vL, pL);
      real rT, uT, vT, pT;  vortexPaperExact(pos[0], pos[1]+0.5*dy, u0, v0, cx, cy, rT, uT, vT, pT);
      real rB, uB, vB, pB;  vortexPaperExact(pos[0], pos[1]-0.5*dy, u0, v0, cx, cy, rB, uB, vB, pB);
      Gx[cIdx] = (rR*uR - rL*uL) / dx;
      Gy[cIdx] = (rT*vT - rB*vB) / dy;
      Gz[cIdx] = 0.0;   // z-uniform
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
      Gx[cIdx]  = rho0*dup*(1.0 + 2.0*up/c0);   // d(rho u)/dx, rho u = rho0 up(1+up/c0)
      Gy[cIdx]  = 0.0;  Gz[cIdx] = 0.0;
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
      // RT0 slope Gx = d(rho u)/dx from the face-midpoint momenta (as the vortex IC)
      real rR, uR, pR;  simpleWaveExact(pos[0]+0.5*dx, A, x0, wid, rR, uR, pR);
      real rL, uL, pL;  simpleWaveExact(pos[0]-0.5*dx, A, x0, wid, rL, uL, pL);
      Gx[cIdx] = (rR*uR - rL*uL)/dx;  Gy[cIdx] = 0.0;  Gz[cIdx] = 0.0;
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

      // RT0 slope DOFs from face-midpoint momenta (ρ≡1 here)
      real rR, uR, vR, pR;  greshoExact(pos[0]+0.5*dx, pos[1], cx, cy, p0, rR, uR, vR, pR);
      real rL, uL, vL, pL;  greshoExact(pos[0]-0.5*dx, pos[1], cx, cy, p0, rL, uL, vL, pL);
      real rT, uT, vT, pT;  greshoExact(pos[0], pos[1]+0.5*dy, cx, cy, p0, rT, uT, vT, pT);
      real rB, uB, vB, pB;  greshoExact(pos[0], pos[1]-0.5*dy, cx, cy, p0, rB, uB, vB, pB);
      Gx[cIdx] = (rR*uR - rL*uL) / dx;
      Gy[cIdx] = (rT*vT - rB*vB) / dy;
      Gz[cIdx] = 0.0;
    }

  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid) {
  // operates on fields 0..4 = (Rho, RhoU, RhoV, RhoW, RhoE).  The same
  // operation (copy density+energy, reflect normal momentum) is valid whether
  // the fields currently hold conservative or primitive variables.
  real *Rho  = grid.getField(F_RHO);
  real *RhoU = grid.getField(F_RHOU);
  real *RhoV = grid.getField(F_RHOV);
  real *RhoW = grid.getField(F_RHOW);
  real *RhoE = grid.getField(F_RHOE);
  real *Gx   = grid.getField(F_GX);   // RT0 slope DOFs (0 in FV mode)
  real *Gy   = grid.getField(F_GY);
  real *Gz   = grid.getField(F_GZ);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isExteriorBlock(lvl, ib, jb, kb)) {
      i32 gridSize[3] = {grid.baseGridSize[0]*powi(2, lvl)/blockSize,
                         grid.baseGridSize[1]*powi(2, lvl)/blockSize,
                         grid.baseGridSize[2]*powi(2, lvl)/blockSize};

      if (grid.bcType == 2) {
        // periodic: the ghost block's center slot 13 holds the opposite-edge image
        // block (set by updateNbrIndicesPeriodicKernel); copy its matching cell.
        i32 imgBlock = grid.nbrIdxList[27*bIdx + 13];
        i32 bcIdx = imgBlock*blockSizeTot + i + j*blockSize + k*blockSize*blockSize;
        Rho[cIdx]  = Rho[bcIdx];
        RhoU[cIdx] = RhoU[bcIdx];
        RhoV[cIdx] = RhoV[bcIdx];
        RhoW[cIdx] = RhoW[bcIdx];
        RhoE[cIdx] = RhoE[bcIdx];
        Gx[cIdx]   = Gx[bcIdx];   // RT0 slopes wrap unchanged under periodicity
        Gy[cIdx]   = Gy[bcIdx];
        Gz[cIdx]   = Gz[bcIdx];
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

        Rho[cIdx]  = Rho[bcIdx];
        RhoE[cIdx] = RhoE[bcIdx];

        if (grid.bcType == 0) {
          // slip wall: reflect the wall-normal momentum, keep tangential
          RhoU[cIdx] = (xWall ? -1.0 : 1.0) * RhoU[bcIdx];
          RhoV[cIdx] = (yWall ? -1.0 : 1.0) * RhoV[bcIdx];
          RhoW[cIdx] = (zWall ? -1.0 : 1.0) * RhoW[bcIdx];
        }
        else if (grid.bcType == 1) {
          // no-slip wall: reflect normal, zero tangential
          RhoU[cIdx] = xWall ? -RhoU[bcIdx] : 0.0;
          RhoV[cIdx] = yWall ? -RhoV[bcIdx] : 0.0;
          RhoW[cIdx] = zWall ? -RhoW[bcIdx] : 0.0;
          if (!xWall && !yWall && !zWall) {
            RhoU[cIdx] = RhoU[bcIdx]; RhoV[cIdx] = RhoV[bcIdx]; RhoW[cIdx] = RhoW[bcIdx];
          }
        }
        else {
          // bcType == 3 : transmissive / outflow (zero gradient)
          RhoU[cIdx] = RhoU[bcIdx];
          RhoV[cIdx] = RhoV[bcIdx];
          RhoW[cIdx] = RhoW[bcIdx];
        }

        // RT0 slope DOFs: zero-gradient copy (exact for transmissive; a wall-
        // normal slope reflection is a follow-up — not exercised by the periodic
        // RT0 validation).
        Gx[cIdx] = Gx[bcIdx];
        Gy[cIdx] = Gy[bcIdx];
        Gz[cIdx] = Gz[bcIdx];
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

  START_CELL_LOOP

    Vec5 q = grid.cons2prim(Vec5(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoW[cIdx], RhoE[cIdx]));
    Rho[cIdx]  = q[0];
    RhoU[cIdx] = q[1];
    RhoV[cIdx] = q[2];
    RhoW[cIdx] = q[3];
    RhoE[cIdx] = q[4];

  END_CELL_LOOP
}

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(0);
  real *U   = grid.getField(1);
  real *V   = grid.getField(2);
  real *W   = grid.getField(3);
  real *P   = grid.getField(4);

  START_CELL_LOOP

    Vec5 q = grid.prim2cons(Vec5(Rho[cIdx], U[cIdx], V[cIdx], W[cIdx], P[cIdx]));
    Rho[cIdx] = q[0];
    U[cIdx]   = q[1];
    V[cIdx]   = q[2];
    W[cIdx]   = q[3];
    P[cIdx]   = q[4];

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

// Wavelet-thresholding scales: domain maxima of the 4 field scales
// {|rho|, |momentum|, |rhoE|, max|grad|} into globalScale[0..3], pre-zeroed by
// the host.  Warp-level shuffle reduction first, then one atomicMax per warp --
// all device-side, no host round-trip.
__global__ void computeGlobalScalesKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(F_RHO);
  real *RhoU = grid.getField(F_RHOU);
  real *RhoV = grid.getField(F_RHOV);
  real *RhoW = grid.getField(F_RHOW);
  real *RhoE = grid.getField(F_RHOE);
  real *Gx   = grid.getField(F_GX);
  real *Gy   = grid.getField(F_GY);
  real *Gz   = grid.getField(F_GZ);

  START_CELL_LOOP

    real r = fabs(Rho[cIdx]);
    real mu = RhoU[cIdx], mv = RhoV[cIdx], mw = RhoW[cIdx];
    real m = sqrt(mu*mu + mv*mv + mw*mw);
    real e = fabs(RhoE[cIdx]);
    real g = fmax(fabs(Gx[cIdx]), fmax(fabs(Gy[cIdx]), fabs(Gz[cIdx])));

    // warp shuffle reduction (grid-stride loop keeps whole warps in-range)
    for (int off = 16; off > 0; off >>= 1) {
      r = fmax(r, __shfl_down_sync(0xffffffff, r, off));
      m = fmax(m, __shfl_down_sync(0xffffffff, m, off));
      e = fmax(e, __shfl_down_sync(0xffffffff, e, off));
      g = fmax(g, __shfl_down_sync(0xffffffff, g, off));
    }
    if ((threadIdx.x & 31) == 0) {
      atomicMaxFloat(&grid.globalScale[0], r);
      atomicMaxFloat(&grid.globalScale[1], m);
      atomicMaxFloat(&grid.globalScale[2], e);
      atomicMaxFloat(&grid.globalScale[3], g);
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

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {
      Vec5 q = grid.cons2prim(Vec5(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoW[cIdx], RhoE[cIdx]));
      real a   = sqrt(abs(gam*q[4]/(q[0]+1e-32)));
      real vel = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
      real dx  = min(grid.getDx(lvl), min(grid.getDy(lvl), grid.getDz(lvl)));
      DeltaT[cIdx] = dx / (a + vel + 1e-32);
    }
    else {
      DeltaT[cIdx] = 1e32;
    }

  END_CELL_LOOP
}

// One-sided biased-parabola face state for the RT0 normal velocity (value+slope
// from `self`, value from the neighbour); selected by grid.rt0Face == 1.  The
// weights are set so the AVERAGE of the two face states is the c=1/6 face value
//   u_f = (uL+uR)/2 + (dx/6)(gL - gR),
// whose flux difference is the 4th-order-accurate divergence operator.  Each
// state reduces exactly to the linear RT0 modal value uSelf+sSelf for smooth
// data (qL==qR: no jump -> low-Mach preserving); across a discontinuity they
// differ, and HLLC upwinds the jump (shock capture + slope relaxation).  `sSelf`
// is the signed half-cell RT0 increment toward the face (+mxs/rho on a right
// face, -mxs/rho on a left face) -- the same term the linear modal state uses.
__device__ inline real parabolicFace(real uSelf, real uNbr, real sSelf) {
  return (5.0/6.0)*uSelf + (1.0/6.0)*uNbr + (2.0/3.0)*sSelf;
}

__global__ void computeRightHandSideKernel(CompressibleSolver &grid) {
  // reads primitive variables (Rho,U,V,W,P) in fields 0..4
  real *Rho = grid.getField(F_RHO);
  real *U   = grid.getField(F_RHOU);
  real *V   = grid.getField(F_RHOV);
  real *W   = grid.getField(F_RHOW);
  real *P   = grid.getField(F_RHOE);
  real *Gx  = grid.getField(F_GX);   // RT0 slope DOFs (∂(ρu)/∂x, …); 0 in FV mode
  real *Gy  = grid.getField(F_GY);
  real *Gz  = grid.getField(F_GZ);

  real *RhsRho  = grid.getField(F_RHS + 0);
  real *RhsRhoU = grid.getField(F_RHS + 1);
  real *RhsRhoV = grid.getField(F_RHS + 2);
  real *RhsRhoW = grid.getField(F_RHS + 3);
  real *RhsRhoE = grid.getField(F_RHS + 4);
  real *RhsGx   = grid.getField(F_RHS + 5);
  real *RhsGy   = grid.getField(F_RHS + 6);
  real *RhsGz   = grid.getField(F_RHS + 7);

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

    Vec5 qL, qR, qD, qU, qB, qF;

    // TVD reconstructed primitive states on each face
    qL[0] = grid.tvdRec(Rho[l2Idx], Rho[l1Idx], Rho[cIdx]);
    qR[0] = grid.tvdRec(Rho[r1Idx], Rho[cIdx],  Rho[l1Idx]);
    qD[0] = grid.tvdRec(Rho[d2Idx], Rho[d1Idx], Rho[cIdx]);
    qU[0] = grid.tvdRec(Rho[u1Idx], Rho[cIdx],  Rho[d1Idx]);
    qB[0] = grid.tvdRec(Rho[b2Idx], Rho[b1Idx], Rho[cIdx]);
    qF[0] = grid.tvdRec(Rho[f1Idx], Rho[cIdx],  Rho[b1Idx]);

    qL[1] = grid.tvdRec(U[l2Idx], U[l1Idx], U[cIdx]);
    qR[1] = grid.tvdRec(U[r1Idx], U[cIdx],  U[l1Idx]);
    qD[1] = grid.tvdRec(U[d2Idx], U[d1Idx], U[cIdx]);
    qU[1] = grid.tvdRec(U[u1Idx], U[cIdx],  U[d1Idx]);
    qB[1] = grid.tvdRec(U[b2Idx], U[b1Idx], U[cIdx]);
    qF[1] = grid.tvdRec(U[f1Idx], U[cIdx],  U[b1Idx]);

    qL[2] = grid.tvdRec(V[l2Idx], V[l1Idx], V[cIdx]);
    qR[2] = grid.tvdRec(V[r1Idx], V[cIdx],  V[l1Idx]);
    qD[2] = grid.tvdRec(V[d2Idx], V[d1Idx], V[cIdx]);
    qU[2] = grid.tvdRec(V[u1Idx], V[cIdx],  V[d1Idx]);
    qB[2] = grid.tvdRec(V[b2Idx], V[b1Idx], V[cIdx]);
    qF[2] = grid.tvdRec(V[f1Idx], V[cIdx],  V[b1Idx]);

    qL[3] = grid.tvdRec(W[l2Idx], W[l1Idx], W[cIdx]);
    qR[3] = grid.tvdRec(W[r1Idx], W[cIdx],  W[l1Idx]);
    qD[3] = grid.tvdRec(W[d2Idx], W[d1Idx], W[cIdx]);
    qU[3] = grid.tvdRec(W[u1Idx], W[cIdx],  W[d1Idx]);
    qB[3] = grid.tvdRec(W[b2Idx], W[b1Idx], W[cIdx]);
    qF[3] = grid.tvdRec(W[f1Idx], W[cIdx],  W[b1Idx]);

    qL[4] = grid.tvdRec(P[l2Idx], P[l1Idx], P[cIdx]);
    qR[4] = grid.tvdRec(P[r1Idx], P[cIdx],  P[l1Idx]);
    qD[4] = grid.tvdRec(P[d2Idx], P[d1Idx], P[cIdx]);
    qU[4] = grid.tvdRec(P[u1Idx], P[cIdx],  P[d1Idx]);
    qB[4] = grid.tvdRec(P[b2Idx], P[b1Idx], P[cIdx]);
    qF[4] = grid.tvdRec(P[f1Idx], P[cIdx],  P[b1Idx]);

    // RT0/P0 DG: replace the limited face-normal velocity by the RT0 face state.
    // ρ, pressure and the tangential velocities keep the reconstruction above.
    // Modal slope mxs = (dx/2)·Gx, so a cell's right-face value is u + mxs/ρ and
    // its left-face value is u − mxs/ρ.  rt0Face selects how the two sides meet:
    //   0 = linear RT0 modal (each cell extrapolates its own slope), or
    //   1 = c=1/6 biased parabola (both states average to the 4th-order c=1/6
    //       face value; see parabolicFace).  Both are unlimited; HLLC upwinds.
    if (grid.scheme == 1) {
      real sxL = 0.5*dx*Gx[l1Idx]/Rho[l1Idx], sxR = 0.5*dx*Gx[cIdx]/Rho[cIdx];
      real syL = 0.5*dy*Gy[d1Idx]/Rho[d1Idx], syR = 0.5*dy*Gy[cIdx]/Rho[cIdx];
      real szL = 0.5*dz*Gz[b1Idx]/Rho[b1Idx], szR = 0.5*dz*Gz[cIdx]/Rho[cIdx];
      if (grid.rt0Face == 1) {
        qL[1] = parabolicFace(U[l1Idx], U[cIdx],  sxL);  qR[1] = parabolicFace(U[cIdx], U[l1Idx], -sxR);
        qD[2] = parabolicFace(V[d1Idx], V[cIdx],  syL);  qU[2] = parabolicFace(V[cIdx], V[d1Idx], -syR);
        qB[3] = parabolicFace(W[b1Idx], W[cIdx],  szL);  qF[3] = parabolicFace(W[cIdx], W[b1Idx], -szR);
      } else {
        qL[1] = U[l1Idx] + sxL;   qR[1] = U[cIdx] - sxR;   // x-face normal (u)
        qD[2] = V[d1Idx] + syL;   qU[2] = V[cIdx] - syR;   // y-face normal (v)
        qB[3] = W[b1Idx] + szL;   qF[3] = W[cIdx] - szR;   // z-face normal (w)
      }
    }

    Vec5 fluxL = grid.hllcFlux(grid.prim2cons(qL), grid.prim2cons(qR), Vec3(1,0,0));
    Vec5 fluxD = grid.hllcFlux(grid.prim2cons(qD), grid.prim2cons(qU), Vec3(0,1,0));

    real ax = dy*dz/vol;   // = 1/dx
    real ay = dx*dz/vol;   // = 1/dy

    real *Rhs[5] = {RhsRho, RhsRhoU, RhsRhoV, RhsRhoW, RhsRhoE};
    for (i32 n = 0; n < 5; n++) {
      atomicAdd(&Rhs[n][cIdx],    fluxL[n]*ax + fluxD[n]*ay);
      atomicAdd(&Rhs[n][l1Idx], - fluxL[n]*ax);
      atomicAdd(&Rhs[n][d1Idx], - fluxD[n]*ay);
    }

    // RT0 slope-DOF update: only the direction-normal faces feed a slope, and
    // both neighbours of a face get the same-sign contribution.  Stored as the
    // physical gradient, so d(Gx)/dt = (2/dx)·d(mxs)/dt.
    //   face  : d(mxs)/dt −= 3·F_xmom/dx   →   RhsGx −= 6·fluxL[1]/dx²
    //   volume: d(mxs)/dt += (6/dx)·[(mxa²+mxs²/3)/ρ + p] → RhsGx += 12/dx²·[…]
    if (grid.scheme == 1) {
      real rc = Rho[cIdx], pc = P[cIdx];
      real gxF = -6.0*fluxL[1]/(dx*dx);
      atomicAdd(&RhsGx[cIdx],  gxF);
      atomicAdd(&RhsGx[l1Idx], gxF);
      real mxa = rc*U[cIdx], mxs = 0.5*dx*Gx[cIdx];
      atomicAdd(&RhsGx[cIdx], 12.0/(dx*dx)*((mxa*mxa + mxs*mxs/3.0)/rc + pc));

      real gyF = -6.0*fluxD[2]/(dy*dy);
      atomicAdd(&RhsGy[cIdx],  gyF);
      atomicAdd(&RhsGy[d1Idx], gyF);
      real mya = rc*V[cIdx], mys = 0.5*dy*Gy[cIdx];
      atomicAdd(&RhsGy[cIdx], 12.0/(dy*dy)*((mya*mya + mys*mys/3.0)/rc + pc));
    }

    // z-flux only in true 3D; pseudo2D never updates z-momentum (W stays 0)
    if (!grid.pseudo2D) {
      Vec5 fluxB = grid.hllcFlux(grid.prim2cons(qB), grid.prim2cons(qF), Vec3(0,0,1));
      real az = dx*dy/vol;   // = 1/dz
      for (i32 n = 0; n < 5; n++) {
        atomicAdd(&Rhs[n][cIdx],    fluxB[n]*az);
        atomicAdd(&Rhs[n][b1Idx], - fluxB[n]*az);
      }
      if (grid.scheme == 1) {
        real rc = Rho[cIdx], pc = P[cIdx];
        real gzF = -6.0*fluxB[3]/(dz*dz);
        atomicAdd(&RhsGz[cIdx],  gzF);
        atomicAdd(&RhsGz[b1Idx], gzF);
        real mza = rc*W[cIdx], mzs = 0.5*dz*Gz[cIdx];
        atomicAdd(&RhsGz[cIdx], 12.0/(dz*dz)*((mza*mza + mzs*mzs/3.0)/rc + pc));
      }
    }

  END_CELL_LOOP
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-dimension flux-array RHS (used when grid.reflux == 1).  One reused
// lower-face conserved-flux vector per cell (fields F_FLUX..F_FLUX+4) serves all
// three dimensions in turn: computeFluxDim fills it for dimension `dim`,
// refluxDim reconciles coarse/fine faces in it, applyFluxDim scatters it to the
// RHS.  RT0 slope terms for `dim` are done inline in computeFluxDim.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void computeFluxDimKernel(CompressibleSolver &grid, i32 dim) {
  real *prim[5] = {grid.getField(F_RHO), grid.getField(F_RHOU), grid.getField(F_RHOV),
                   grid.getField(F_RHOW), grid.getField(F_RHOE)};
  real *G    = grid.getField(F_GX + dim);          // this dim's RT0 slope (0 in FV mode)
  real *RhsG = grid.getField(F_RHS + F_GX + dim);
  real *Fx[5] = {grid.getField(F_FLUX+0), grid.getField(F_FLUX+1), grid.getField(F_FLUX+2),
                 grid.getField(F_FLUX+3), grid.getField(F_FLUX+4)};
  i32 di = (dim==0), dj = (dim==1), dk = (dim==2);
  i32 nv = dim + 1;                                 // normal velocity component (u/v/w)

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx]; i32 lvl, ib, jb, kb; grid.decode(loc, lvl, ib, jb, kb);
    real h = (dim==0) ? grid.getDx(lvl) : (dim==1 ? grid.getDy(lvl) : grid.getDz(lvl));

    i32 lm1 = grid.getNbrIdx(bIdx, i-di,   j-dj,   k-dk);
    i32 lm2 = grid.getNbrIdx(bIdx, i-2*di, j-2*dj, k-2*dk);
    i32 lp1 = grid.getNbrIdx(bIdx, i+di,   j+dj,   k+dk);

    // lower-face left/right states (TVD reconstruction), normal component
    // replaced by the RT0 modal face value in DG mode
    Vec5 qL, qR;
    for (i32 m = 0; m < 5; m++) {
      qL[m] = grid.tvdRec(prim[m][lm2], prim[m][lm1], prim[m][cIdx]);
      qR[m] = grid.tvdRec(prim[m][lp1], prim[m][cIdx], prim[m][lm1]);
    }
    if (grid.scheme == 1) {   // RT0 normal states: linear modal (0) or c=1/6 (1)
      real sL = 0.5*h*G[lm1]/prim[0][lm1], sR = 0.5*h*G[cIdx]/prim[0][cIdx];
      if (grid.rt0Face == 1) {
        qL[nv] = parabolicFace(prim[nv][lm1], prim[nv][cIdx],  sL);
        qR[nv] = parabolicFace(prim[nv][cIdx], prim[nv][lm1], -sR);
      } else {
        qL[nv] = prim[nv][lm1]  + sL;
        qR[nv] = prim[nv][cIdx] - sR;
      }
    }
    Vec5 flux = grid.hllcFlux(grid.prim2cons(qL), grid.prim2cons(qR), Vec3(di, dj, dk));
    for (i32 n = 0; n < 5; n++) Fx[n][cIdx] = flux[n];

    if (grid.scheme == 1) {   // RT0 slope DOF for this dimension (face + volume)
      real slopeFace = -6.0*flux[nv]/(h*h);
      atomicAdd(&RhsG[cIdx], slopeFace);
      atomicAdd(&RhsG[lm1],  slopeFace);
      real rc = prim[0][cIdx], mavg = rc*prim[nv][cIdx], mslope = 0.5*h*G[cIdx];
      atomicAdd(&RhsG[cIdx], 12.0/(h*h)*((mavg*mavg + mslope*mslope/3.0)/rc + prim[4][cIdx]));
    }
  END_CELL_LOOP
}

// Conservative coarse/fine flux correction, done fine→coarse like restriction.
// Every `dim`-face separating a coarse cell from finer cells has its coarse flux
// replaced by the area-average of the finer fluxes tiling it, so both sides update
// with the same flux (Berger-Colella refluxing).  The flux of a face is stored in
// the cell on its +`dim` side (that cell's lower face), so a face is a coarse/fine
// interface exactly when a cell and its lower-`dim` neighbour differ in ACTIVE vs
// PARENT.  This handles both sides of a refined region symmetrically.
//   phase 0 (refluxDim):      zero the stored coarse flux at every such face
//   phase 1 (refluxAccumDim): each fine face on the interface adds its flux/nFine
__global__ void refluxDimKernel(CompressibleSolver &grid, i32 dim) {
  real *Fx[5] = {grid.getField(F_FLUX+0), grid.getField(F_FLUX+1), grid.getField(F_FLUX+2),
                 grid.getField(F_FLUX+3), grid.getField(F_FLUX+4)};
  i32 di = (dim==0), dj = (dim==1), dk = (dim==2);

  START_CELL_LOOP
    GET_CELL_INDICES
    i32 cf  = grid.cFlagsList[cIdx];
    i32 lm1 = grid.getNbrIdx(bIdx, i-di, j-dj, k-dk);
    i32 lf  = grid.cFlagsList[lm1];
    // this cell's lower-`dim` face is a coarse/fine interface (coarse side)
    if ((cf == PARENT && lf == ACTIVE) || (cf == ACTIVE && lf == PARENT))
      for (i32 n = 0; n < 5; n++) Fx[n][cIdx] = 0.0;
  END_CELL_LOOP
}

__global__ void refluxAccumDimKernel(CompressibleSolver &grid, i32 dim) {
  real *Fx[5] = {grid.getField(F_FLUX+0), grid.getField(F_FLUX+1), grid.getField(F_FLUX+2),
                 grid.getField(F_FLUX+3), grid.getField(F_FLUX+4)};
  i32 di = (dim==0), dj = (dim==1), dk = (dim==2);
  real w = 1.0 / (grid.pseudo2D ? 2.0 : 4.0);

  // Each fine cell at a `dim`-edge of the refined region contributes its interface
  // flux to the coarse cell owning that face.  (nested ifs, no `continue`: the
  // grid-stride increment lives in END_CELL_LOOP.)
  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx]; i32 lvl, ib, jb, kb; grid.decode(loc, lvl, ib, jb, kb);
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && grid.cFlagsList[cIdx] == ACTIVE) {
      i32 prntIdx = grid.prntIdxList[bIdx];
      i32 ip = i/2 + ib%2 * blockSize/2;
      i32 jp = j/2 + jb%2 * blockSize/2;
      i32 kp = grid.pseudo2D ? k : (k/2 + kb%2 * blockSize/2);

      // lower-`dim` interface: this fine cell's lower face → parent cell P (P's
      // own lower face is the interface, stored in Fx[P])
      i32 lm1 = grid.getNbrIdx(bIdx, i-di, j-dj, k-dk);
      if (grid.cFlagsList[lm1] == GHOST) {
        i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);
        for (i32 n = 0; n < 5; n++) atomicAdd(&Fx[n][pIdx], w * Fx[n][cIdx]);
      }
      // upper-`dim` interface: this fine cell's upper face (flux stored in the
      // GHOST cell beyond it) → coarse cell C = parent's +`dim` neighbour (C's own
      // lower face is the interface, stored in Fx[C])
      i32 gp1 = grid.getNbrIdx(bIdx, i+di, j+dj, k+dk);
      if (grid.cFlagsList[gp1] == GHOST) {
        i32 cIdxB = grid.getNbrIdx(prntIdx, ip+di, jp+dj, kp+dk);
        for (i32 n = 0; n < 5; n++) {
          real fv = Fx[n][gp1];
          if (isfinite(fv)) atomicAdd(&Fx[n][cIdxB], w * fv);
        }
      }
    }
  END_CELL_LOOP
}

__global__ void applyFluxDimKernel(CompressibleSolver &grid, i32 dim) {
  real *Rhs[5] = {grid.getField(F_RHS+0), grid.getField(F_RHS+1), grid.getField(F_RHS+2),
                  grid.getField(F_RHS+3), grid.getField(F_RHS+4)};
  real *Fx[5] = {grid.getField(F_FLUX+0), grid.getField(F_FLUX+1), grid.getField(F_FLUX+2),
                 grid.getField(F_FLUX+3), grid.getField(F_FLUX+4)};
  i32 di = (dim==0), dj = (dim==1), dk = (dim==2);

  START_CELL_LOOP
    GET_CELL_INDICES
    u64 loc = grid.bLocList[bIdx]; i32 lvl, ib, jb, kb; grid.decode(loc, lvl, ib, jb, kb);
    real h = (dim==0) ? grid.getDx(lvl) : (dim==1 ? grid.getDy(lvl) : grid.getDz(lvl));
    real a = 1.0/h;
    i32 lm1 = grid.getNbrIdx(bIdx, i-di, j-dj, k-dk);
    for (i32 n = 0; n < 5; n++) {
      atomicAdd(&Rhs[n][cIdx],  Fx[n][cIdx]*a);
      atomicAdd(&Rhs[n][lm1],  -Fx[n][cIdx]*a);
    }
  END_CELL_LOOP
}

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage) {
  //
  // TVD Runge-Kutta 3 update of the NEVOLVE evolved DOFs
  //   (P0 ρ,E + momentum cell-averages + RT0 slope DOFs Gx,Gy,Gz).
  //
  real dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {

      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q   = grid.getField(f);
        real *Old = grid.getField(F_OLD + f);
        real *Rhs = grid.getField(F_RHS + f);

        if (stage == 0) {
          Old[cIdx] = Q[cIdx];
          Q[cIdx]   = Q[cIdx] + dt * Rhs[cIdx];
        }
        else if (stage == 1) {
          Q[cIdx]   = 3.0/4.0*Old[cIdx] + 1.0/4.0*Q[cIdx] + 1.0/4.0 * dt * Rhs[cIdx];
        }
        else {
          Q[cIdx]   = 1.0/3.0*Old[cIdx] + 2.0/3.0*Q[cIdx] + 2.0/3.0 * dt * Rhs[cIdx];
        }
      }

      // pseudo2D: z-momentum and its RT0 slope are never evolved
      if (grid.pseudo2D) {
        grid.getField(F_RHOW)[cIdx]        = 0;
        grid.getField(F_OLD + F_RHOW)[cIdx] = 0;
        grid.getField(F_GZ)[cIdx]          = 0;
        grid.getField(F_OLD + F_GZ)[cIdx]   = 0;
      }
    }

    // reset the rhs accumulators for the next substep
    for (i32 f = 0; f < NEVOLVE; f++)
      grid.getField(F_RHS + f)[cIdx] = 0;

  END_CELL_LOOP
}

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP

    for (i32 f = 0; f < NEVOLVE; f++)
      grid.getField(F_OLD + f)[cIdx] = grid.getField(f)[cIdx];

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
// RT0 flux-preserving prolongation of the momentum DOFs.  The Deslauriers-Dubuc
// prediction above is second-order accurate but does NOT reproduce a cell's RT0
// face-normal-momentum DOF on refinement, so it leaks flux at coarse/fine faces.
//
// For a momentum component with normal direction d (x-mom -> d=x, ...) the child
// cell-average is corrected so that the four fine faces tiling a parent face
// average exactly to the parent's face DOF, and the interior fine faces stay
// continuous.  The correction is: replace the DD normal-direction slope of the
// average by the exact RT0 own-slope  s_d*(dx_child/2)*grad_p , and (for the
// gradient DOF) drop its normal-direction variation entirely (RT0 momentum is
// linear along its own axis, so its axis-derivative is constant there).  All the
// transverse (y,z) and cross terms of DD are kept — they are zero-mean over the
// four sub-faces, so they add tangential detail without breaking conservation.
//
// Normal-direction stencil offsets (which DD term to remove):
//   d=0(x): s=xs, (nm,np)=(l,r)   d=1(y): s=ys,(d,u)   d=2(z): s=zs,(b,f)
//
__device__ real
waveletPredictNormalSlope(MultiLevelSparseGrid &grid, real *Q, i32 prntIdx,
                          i32 ip, i32 jp, i32 kp, real xs, real ys, real zs, i32 dir) {
  real s; i32 nm, np;
  if (dir == 0)      { s = xs; nm = grid.getNbrIdx(prntIdx, ip-1, jp, kp); np = grid.getNbrIdx(prntIdx, ip+1, jp, kp); }
  else if (dir == 1) { s = ys; nm = grid.getNbrIdx(prntIdx, ip, jp-1, kp); np = grid.getNbrIdx(prntIdx, ip, jp+1, kp); }
  else               { s = zs; nm = grid.getNbrIdx(prntIdx, ip, jp, kp-1); np = grid.getNbrIdx(prntIdx, ip, jp, kp+1); }
  return s/8.0*(Q[np] - Q[nm]);
}

// prolongate the momentum cell-average (avg) paired with its gradient DOF (grad):
//   DD prediction  -  DD normal-slope  +  RT0 own-slope [ = s*(dx_child/2)*grad_p ]
__device__ real
waveletPredictMomentum(MultiLevelSparseGrid &grid, real *avg, real *grad, i32 prntIdx,
                       i32 ip, i32 jp, i32 kp, real xs, real ys, real zs,
                       i32 dir, real dxChild) {
  real base = waveletPredict(grid, avg, prntIdx, ip, jp, kp, xs, ys, zs);
  base -= waveletPredictNormalSlope(grid, avg, prntIdx, ip, jp, kp, xs, ys, zs, dir);
  real s  = (dir == 0) ? xs : (dir == 1 ? ys : zs);
  i32  p  = grid.getNbrIdx(prntIdx, ip, jp, kp);
  base += s * (0.5*dxChild) * grad[p];
  return base;
}

// prolongate a momentum-gradient DOF: DD prediction minus its normal-direction
// variation (kept constant along its own axis, per the RT0 linear-in-normal model)
__device__ real
waveletPredictGrad(MultiLevelSparseGrid &grid, real *grad, i32 prntIdx,
                   i32 ip, i32 jp, i32 kp, real xs, real ys, real zs, i32 dir) {
  real base = waveletPredict(grid, grad, prntIdx, ip, jp, kp, xs, ys, zs);
  base -= waveletPredictNormalSlope(grid, grad, prntIdx, ip, jp, kp, xs, ys, zs, dir);
  return base;
}

//
// Predict evolved field f at a child cell from its parent block, dispatching to
// the RT0 flux-preserving prolongation for the momentum DOFs and to plain DD for
// the P0 scalars (ρ, E).  baseOff selects the field bank (0 = current, F_OLD =
// the wavelet-transform reference snapshot).
//
__device__ real
predictEvolvedField(CompressibleSolver &grid, i32 baseOff, i32 f, i32 prntIdx,
                    i32 ip, i32 jp, i32 kp, real xs, real ys, real zs, i32 lvl) {
  // finite-volume mode carries no RT0 slope DOFs -> plain DD for every field
  if (grid.scheme != 1) {
    real *Q = grid.getField(baseOff + f);
    return waveletPredict(grid, Q, prntIdx, ip, jp, kp, xs, ys, zs);
  }
  if (f >= F_RHOU && f <= F_RHOW) {
    i32 d = f - F_RHOU;
    real *avg  = grid.getField(baseOff + F_RHOU + d);
    real *grad = grid.getField(baseOff + F_GX   + d);
    real dxd = (d == 0) ? grid.getDx(lvl) : (d == 1 ? grid.getDy(lvl) : grid.getDz(lvl));
    return waveletPredictMomentum(grid, avg, grad, prntIdx, ip, jp, kp, xs, ys, zs, d, dxd);
  }
  else if (f >= F_GX && f <= F_GZ) {
    i32 d = f - F_GX;
    real *grad = grid.getField(baseOff + F_GX + d);
    return waveletPredictGrad(grid, grad, prntIdx, ip, jp, kp, xs, ys, zs, d);
  }
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
    else if (cFlag == GHOST) {
      for (i32 f = 0; f < NEVOLVE; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = 0.0;
      }
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

__global__ void waveletThresholdingKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl < 2) {
      grid.bFlagsList[bIdx] = KEEP;
    }

    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
    real dx = min(grid.getDx(lvl), min(grid.getDy(lvl), grid.getDz(lvl)));
    real ls = grid.getBoundaryLevelSet(pos);

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb)) {
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
        //  - The RT0 slope DOFs (Gx,Gy,Gz) are auxiliary sub-cell moments, not
        //    primary fields: refining on them adds spurious grid (their detail
        //    is noisy where G/rho is large, i.e. low-density regions) and is the
        //    main source of the RT0-mode over-refinement.
        //  - z-momentum is identically 0 in pseudo-2D, so it never fires anyway;
        //    skip it explicitly so a stray roundoff detail can never trigger.
        if (f >= F_GX) continue;                               // skip Gx,Gy,Gz
        if (grid.pseudo2D && f == F_RHOW) continue;            // z-mom is 0 in 2D
        real *Q = grid.getField(f);
        // normalize the detail by the domain max of this field's scale
        // (computed pre-transform, device-side): rho / |momentum| / rhoE
        i32 sc = (f == F_RHO) ? 0 : (f <= F_RHOW ? 1 : 2);
        real mag = fmax(grid.globalScale[sc], (real)1e-32);

        if (abs(Q[cIdx]/mag) > grid.waveletThresh || abs(ls) < dx) {
          if (lvl < grid.nLvls-1 && (abs(Q[cIdx]/mag) > grid.waveletThresh*2 || abs(ls) < dx)) {
            i32 bSize = blockSize/2;
            i32 kc = grid.pseudo2D ? kb : (2*kb + k/bSize);
            grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, kc);
          }
          grid.bFlagsList[bIdx] = KEEP;
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
