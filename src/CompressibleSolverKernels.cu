#include "CompressibleSolverKernels.cuh"

//
// field layout (see CompressibleSolver.cuh):
//   0..4   : Rho, RhoU, RhoV, RhoW, RhoE   (conservative)  /  Rho, U, V, W, P (primitive)
//   5..9   : Old{Rho,RhoU,RhoV,RhoW,RhoE}
//   10..14 : Rhs{Rho,RhoU,RhoV,RhoW,RhoE}
//   15     : DeltaT / MagRhoU
//

__global__ void sortFieldDataKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);

  real *OldRho  = grid.getField(5);
  real *OldRhoU = grid.getField(6);
  real *OldRhoV = grid.getField(7);
  real *OldRhoW = grid.getField(8);
  real *OldRhoE = grid.getField(9);

  START_CELL_LOOP

    i32 bIdxOld = grid.bIdxList[bIdx];
    i32 cIdxOld = bIdxOld * blockSizeTot + cIdx % blockSizeTot;

    Rho[cIdx]  = OldRho[cIdxOld];
    RhoU[cIdx] = OldRhoU[cIdxOld];
    RhoV[cIdx] = OldRhoV[cIdxOld];
    RhoW[cIdx] = OldRhoW[cIdxOld];
    RhoE[cIdx] = OldRhoE[cIdxOld];
    grid.bFlagsList[bIdxOld] = DELETE;

  END_CELL_LOOP
}

__global__ void setInitialConditionsKernel(CompressibleSolver &grid) {

  real *Rho = grid.getField(0);
  real *U   = grid.getField(1);
  real *V   = grid.getField(2);
  real *W   = grid.getField(3);
  real *P   = grid.getField(4);

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

    if (grid.icType == 1) {
      //
      // 2D circular Sod explosion (uniform in z -> pseudo-2D).  A circular
      // region of high-pressure gas drives a cylindrical shock outward.
      //
      real cx = grid.domainSize[0]/2;
      real cy = grid.domainSize[1]/2;
      real radius = min(grid.domainSize[0], grid.domainSize[1])/5;
      real dist = sqrt((pos[0]-cx)*(pos[0]-cx) + (pos[1]-cy)*(pos[1]-cy));
      if (dist < radius) {
        Rho[cIdx] = 1.0; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = 1.0;
      }
      else {
        Rho[cIdx] = 0.125; U[cIdx] = 0.0; V[cIdx] = 0.0; W[cIdx] = 0.0; P[cIdx] = 0.1;
      }
    }

  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid) {
  // operates on fields 0..4 = (Rho, RhoU, RhoV, RhoW, RhoE).  The same
  // operation (copy density+energy, reflect normal momentum) is valid whether
  // the fields currently hold conservative or primitive variables.
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);

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
        // periodic: self-neighbor has been remapped to the periodic interior cell
        i32 bcIdx = grid.getNbrIdx(bIdx, i, j, k);
        Rho[cIdx]  = Rho[bcIdx];
        RhoU[cIdx] = RhoU[bcIdx];
        RhoV[cIdx] = RhoV[bcIdx];
        RhoW[cIdx] = RhoW[bcIdx];
        RhoE[cIdx] = RhoE[bcIdx];
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

__global__ void computeMagUKernel(CompressibleSolver &grid) {
  // magnitude of momentum (fields are conservative when this is called)
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *MagRhoU = grid.getField(15);

  START_CELL_LOOP

    MagRhoU[cIdx] = sqrt(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx] + RhoW[cIdx]*RhoW[cIdx]);

  END_CELL_LOOP
}

// compute pressure into the scratch field (15) for visualization (fields conservative)
__global__ void computePressureKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);
  real *P    = grid.getField(15);

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
  real *DeltaT = grid.getField(15);

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

__global__ void computeRightHandSideKernel(CompressibleSolver &grid) {
  // reads primitive variables (Rho,U,V,W,P) in fields 0..4
  real *Rho = grid.getField(0);
  real *U   = grid.getField(1);
  real *V   = grid.getField(2);
  real *W   = grid.getField(3);
  real *P   = grid.getField(4);

  real *RhsRho  = grid.getField(10);
  real *RhsRhoU = grid.getField(11);
  real *RhsRhoV = grid.getField(12);
  real *RhsRhoW = grid.getField(13);
  real *RhsRhoE = grid.getField(14);

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

    // z-flux only in true 3D; pseudo2D never updates z-momentum (W stays 0)
    if (!grid.pseudo2D) {
      Vec5 fluxB = grid.hllcFlux(grid.prim2cons(qB), grid.prim2cons(qF), Vec3(0,0,1));
      real az = dx*dy/vol;   // = 1/dz
      for (i32 n = 0; n < 5; n++) {
        atomicAdd(&Rhs[n][cIdx],    fluxB[n]*az);
        atomicAdd(&Rhs[n][b1Idx], - fluxB[n]*az);
      }
    }

  END_CELL_LOOP
}

__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage) {
  //
  // TVD Runge-Kutta 3 update of the conservative fields
  //
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);

  real *OldRho  = grid.getField(5);
  real *OldRhoU = grid.getField(6);
  real *OldRhoV = grid.getField(7);
  real *OldRhoW = grid.getField(8);
  real *OldRhoE = grid.getField(9);

  real *RhsRho  = grid.getField(10);
  real *RhsRhoU = grid.getField(11);
  real *RhsRhoV = grid.getField(12);
  real *RhsRhoW = grid.getField(13);
  real *RhsRhoE = grid.getField(14);

  real dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {

      if (stage == 0) {
        OldRho[cIdx]  = Rho[cIdx];
        OldRhoU[cIdx] = RhoU[cIdx];
        OldRhoV[cIdx] = RhoV[cIdx];
        OldRhoW[cIdx] = RhoW[cIdx];
        OldRhoE[cIdx] = RhoE[cIdx];

        Rho[cIdx]  = Rho[cIdx]  + dt * RhsRho[cIdx];
        RhoU[cIdx] = RhoU[cIdx] + dt * RhsRhoU[cIdx];
        RhoV[cIdx] = RhoV[cIdx] + dt * RhsRhoV[cIdx];
        RhoW[cIdx] = RhoW[cIdx] + dt * RhsRhoW[cIdx];
        RhoE[cIdx] = RhoE[cIdx] + dt * RhsRhoE[cIdx];
      }

      if (stage == 1) {
        Rho[cIdx]  = 3.0/4.0*OldRho[cIdx]  + 1.0/4.0*Rho[cIdx]  + 1.0/4.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 3.0/4.0*OldRhoU[cIdx] + 1.0/4.0*RhoU[cIdx] + 1.0/4.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 3.0/4.0*OldRhoV[cIdx] + 1.0/4.0*RhoV[cIdx] + 1.0/4.0 * dt * RhsRhoV[cIdx];
        RhoW[cIdx] = 3.0/4.0*OldRhoW[cIdx] + 1.0/4.0*RhoW[cIdx] + 1.0/4.0 * dt * RhsRhoW[cIdx];
        RhoE[cIdx] = 3.0/4.0*OldRhoE[cIdx] + 1.0/4.0*RhoE[cIdx] + 1.0/4.0 * dt * RhsRhoE[cIdx];
      }

      if (stage == 2) {
        Rho[cIdx]  = 1.0/3.0*OldRho[cIdx]  + 2.0/3.0*Rho[cIdx]  + 2.0/3.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 1.0/3.0*OldRhoU[cIdx] + 2.0/3.0*RhoU[cIdx] + 2.0/3.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 1.0/3.0*OldRhoV[cIdx] + 2.0/3.0*RhoV[cIdx] + 2.0/3.0 * dt * RhsRhoV[cIdx];
        RhoW[cIdx] = 1.0/3.0*OldRhoW[cIdx] + 2.0/3.0*RhoW[cIdx] + 2.0/3.0 * dt * RhsRhoW[cIdx];
        RhoE[cIdx] = 1.0/3.0*OldRhoE[cIdx] + 2.0/3.0*RhoE[cIdx] + 2.0/3.0 * dt * RhsRhoE[cIdx];
      }

      // pseudo2D: z-momentum is never evolved
      if (grid.pseudo2D) {
        RhoW[cIdx]    = 0;
        OldRhoW[cIdx] = 0;
      }
    }

    // reset the rhs accumulator for the next substep
    RhsRho[cIdx]  = 0;
    RhsRhoU[cIdx] = 0;
    RhsRhoV[cIdx] = 0;
    RhsRhoW[cIdx] = 0;
    RhsRhoE[cIdx] = 0;

  END_CELL_LOOP
}

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoW = grid.getField(3);
  real *RhoE = grid.getField(4);

  real *OldRho  = grid.getField(5);
  real *OldRhoU = grid.getField(6);
  real *OldRhoV = grid.getField(7);
  real *OldRhoW = grid.getField(8);
  real *OldRhoE = grid.getField(9);

  START_CELL_LOOP

    OldRho[cIdx]  = Rho[cIdx];
    OldRhoU[cIdx] = RhoU[cIdx];
    OldRhoV[cIdx] = RhoV[cIdx];
    OldRhoW[cIdx] = RhoW[cIdx];
    OldRhoE[cIdx] = RhoE[cIdx];

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

      for (i32 f = 0; f < 5; f++) {
        real *Q    = grid.getField(f);
        real *OldQ = grid.getField(f+5);
        Q[cIdx] = Q[cIdx] - waveletPredict(grid, OldQ, prntIdx, ip, jp, kp, xs, ys, zs);
      }
    }
    else if (cFlag == GHOST) {
      for (i32 f = 0; f < 5; f++) {
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

      for (i32 f = 0; f < 5; f++) {
        real *Q    = grid.getField(f);
        real *OldQ = grid.getField(f+5);
        Q[cIdx] = Q[cIdx] + waveletPredict(grid, OldQ, prntIdx, ip, jp, kp, xs, ys, zs);
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

      for (i32 f = 0; f < 5; f++) {
        real *Q = grid.getField(f);
        real mag = 1e-32;
        if (f == 0)            mag = grid.maxRho;
        if (f > 0 && f < 4)    mag = grid.maxMagRhoU;
        if (f == 4)            mag = grid.maxRhoE;

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

      for (i32 f = 0; f < 5; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = waveletPredict(grid, Q, prntIdx, ip, jp, kp, xs, ys, zs);
      }
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
        for (i32 f = 0; f < 5; f++) {
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
        for (i32 f = 0; f < 5; f++) {
          real *q = grid.getField(f);
          q[pIdx] = (q[c000] + q[c100] + q[c010] + q[c110] +
                     q[c001] + q[c101] + q[c011] + q[c111]) / 8.0;
        }
      }
    }

  END_CELL_LOOP
}
