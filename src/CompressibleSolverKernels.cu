#include "CompressibleSolverKernels.cuh"

__global__ void sortFieldDataKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  real *OldRho  = grid.getField(4);
  real *OldRhoU = grid.getField(5);
  real *OldRhoV = grid.getField(6);
  real *OldRhoE = grid.getField(7);

  START_CELL_LOOP

    u32 bIdxOld = grid.bIdxList[bIdx];
    u32 cIdxOld = bIdxOld * blockSizeTot + idx;
    
    Rho[cIdx] = OldRho[cIdxOld];
    RhoU[cIdx] = OldRhoU[cIdxOld];
    RhoV[cIdx] = OldRhoV[cIdxOld];
    RhoE[cIdx] = OldRhoE[cIdxOld];
    grid.bFlagsList[bIdxOld] = DELETE;

  END_CELL_LOOP

}

__global__ void setInitialConditionsKernel(CompressibleSolver &grid) {

  real *Rho  = grid.getField(0);
  real *U    = grid.getField(1);
  real *V    = grid.getField(2);
  real *P    = grid.getField(3);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);
    Vec2 pos = grid.getCellPos(lvl, ib, jb, i, j);

    if (grid.icType == 0) {
      //
      // sod shock explosion
      //
      real centerX = grid.domainSize[0]/2;
      real centerY = grid.domainSize[1]/2;
      real radius = min(grid.domainSize[0], grid.domainSize[1])/5;

      real dist = sqrt((pos[0]-centerX)*(pos[0]-centerX) + (pos[1]-centerY)*(pos[1]-centerY));
    
      // inside
      if (dist < radius) {
        Rho[cIdx]  = 10.0;
        U[cIdx]    = 0.0;
        V[cIdx]    = 0.0;
        P[cIdx]    = 10.0;
      }
      else {
        Rho[cIdx]  = 0.125;
        U[cIdx]    = 0.0;
        V[cIdx]    = 0.0;
        P[cIdx]    = 0.1;
      }
    }

    if (grid.icType == 1) {
      //
      // gaussian explosion
      //
    
      Rho[cIdx] = 10.0*exp(-3000 * ((pos[0] - .4)*(pos[0] - .4) + (pos[1] - .4)*(pos[1] - .4))) + .125;
      P[cIdx] = 10.0*exp(-3000 * ((pos[0] - .4)*(pos[0] - .4) + (pos[1] - .4)*(pos[1] - .4))) + .1;
      U[cIdx]    = 0.0;
      V[cIdx]    = 0.0;
    }

    if (grid.icType == 2) {
      //
      // wind tunnel
      //
      Rho[cIdx] = 1.0;
      P[cIdx]   = 1.0;
      U[cIdx]   = 3.0;
      V[cIdx]   = 0.0;
    }



  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid) {

  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    if (grid.isExteriorBlock(lvl, ib, jb)) {
      u32 gridSize[2] = {grid.baseGridSize[0]*powi(2, lvl)/blockSize, 
                         grid.baseGridSize[1]*powi(2, lvl)/blockSize};
      if (grid.bcType == 0) {
        //
        // slip wall
        //

        // figure out internal cell for neuman boundary conditions
        i32 ibc = i;
        i32 jbc = j;
        if (ib < 0) {
          ibc = blockSize;
        }
        if (jb < 0) {
          jbc = blockSize;
        }
        if (ib >= i32(gridSize[0])) {
          ibc = -1; 
        }
        if (jb >= i32(gridSize[1])) {
          jbc = -1;
        }
        u32 bcIdx = grid.getNbrIdx(bIdx, ibc, jbc); 

        // apply boundary conditions
        Rho[cIdx] = Rho[bcIdx];
        RhoE[cIdx] = RhoE[bcIdx];
        
        if (ib < 0 || ib >= gridSize[0]) {
          RhoU[cIdx] = -RhoU[bcIdx];
          RhoV[cIdx] =  RhoV[bcIdx];
        }
        if (jb < 0 || jb >= gridSize[1]) {
          RhoU[cIdx] =  RhoU[bcIdx];
          RhoV[cIdx] = -RhoV[bcIdx];
        }
        if ((ib < 0 || ib >= gridSize[0]) && (jb < 0 || jb >= gridSize[1])) {
          RhoU[cIdx] = -RhoU[bcIdx];
          RhoV[cIdx] = -RhoV[bcIdx];
        }
      }

      if (grid.bcType == 1) {
        //
        // wind tunnel
        //

        // figure out internal cell for neuman boundary conditions
        i32 ibc = i;
        i32 jbc = j;
        if (ib < 0) {
          ibc = blockSize;
        }
        if (jb < 0) {
          jbc = blockSize;
        }
        if (ib >= i32(gridSize[0])) {
          ibc = -1; 
        }
        if (jb >= i32(gridSize[1])) {
          jbc = -1;
        }
        u32 bcIdx = grid.getNbrIdx(bIdx, ibc, jbc); 

        // apply boundary conditions
        Rho[cIdx] = Rho[bcIdx];
        RhoE[cIdx] = RhoE[bcIdx];
        
        // left wall
        if (ib < 0) {
          RhoU[cIdx] = 3.0;
          RhoV[cIdx] = 0.0;
          Rho[cIdx] = 1.0;
          RhoE[cIdx] = 1.0/(gam-1) + .5*(3.0*3.0);
        }

        // right wall
        if (ib >= gridSize[0]) {
          RhoU[cIdx] = RhoU[bcIdx];
          RhoV[cIdx] = RhoV[bcIdx];
        }

        // top and bottom wall
        if (jb >= gridSize[1] || jb < 0) {
          RhoU[cIdx] = RhoU[bcIdx];
          RhoV[cIdx] =  -RhoV[bcIdx];
        }

      }
    }

  END_CELL_LOOP
}


// compute max wavespeed in each cell, will be used for CFL condition
__global__ void computeDeltaTKernel(CompressibleSolver &grid) {
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);
  real *DeltaT = grid.getField(12);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      real a, dx, vel;
      Vec4 q = grid.cons2prim(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]));
      a = sqrt(abs(gam*q[3]/(q[0]+1e-32)));
      vel = sqrt(q[1]*q[1] + q[2]*q[2]);
      dx = min(grid.getDx(lvl), grid.getDy(lvl));
      DeltaT[cIdx] = dx / (a + vel + 1e-32);
    }
    else {
      DeltaT[cIdx] = 1e32;
    }

  END_CELL_LOOP

}


__global__ void computeRightHandSideKernel(CompressibleSolver &grid) {
  real *Rho = grid.getField(0);
  real *U   = grid.getField(1);
  real *V   = grid.getField(2);
  real *P   = grid.getField(3);

  real *RhsRho  = grid.getField(8);
  real *RhsRhoU = grid.getField(9);
  real *RhsRhoV = grid.getField(10);
  real *RhsRhoE = grid.getField(11);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    real dx = grid.getDx(lvl);
    real dy = grid.getDy(lvl);
    real vol = dx*dy;

    u32 l1Idx = grid.getNbrIdx(bIdx, i-1, j);
    u32 l2Idx = grid.getNbrIdx(bIdx, i-2, j);
    u32 r1Idx = grid.getNbrIdx(bIdx, i+1, j);

    u32 d1Idx = grid.getNbrIdx(bIdx, i, j-1);
    u32 d2Idx = grid.getNbrIdx(bIdx, i, j-2);
    u32 u1Idx = grid.getNbrIdx(bIdx, i, j+1);

    //u32 cFlag = grid.cFlagsList[cIdx];
    //u32 lFlag = grid.cFlagsList[l1Idx];
    //u32 dFlag = grid.cFlagsList[d1Idx];

    //if (grid.isInteriorBlock(lvl, ib, jb)) {
      
      Vec4 fluxL;
      Vec4 fluxD;
      Vec4 qL;
      Vec4 qR;
      Vec4 qD;
      Vec4 qU;

      // left flux
      qL[0] = grid.tvdRec(Rho[l2Idx], Rho[l1Idx], Rho[cIdx]);
      qR[0] = grid.tvdRec(Rho[r1Idx], Rho[cIdx],  Rho[l1Idx]);
      qD[0] = grid.tvdRec(Rho[d2Idx], Rho[d1Idx], Rho[cIdx]);
      qU[0] = grid.tvdRec(Rho[u1Idx], Rho[cIdx],  Rho[d1Idx]);

      qL[1] = grid.tvdRec(U[l2Idx], U[l1Idx], U[cIdx]);
      qR[1] = grid.tvdRec(U[r1Idx], U[cIdx],  U[l1Idx]);
      qD[1] = grid.tvdRec(U[d2Idx], U[d1Idx], U[cIdx]);
      qU[1] = grid.tvdRec(U[u1Idx], U[cIdx],  U[d1Idx]);

      qL[2] = grid.tvdRec(V[l2Idx], V[l1Idx], V[cIdx]);
      qR[2] = grid.tvdRec(V[r1Idx], V[cIdx],  V[l1Idx]);
      qD[2] = grid.tvdRec(V[d2Idx], V[d1Idx], V[cIdx]);
      qU[2] = grid.tvdRec(V[u1Idx], V[cIdx],  V[d1Idx]);

      qL[3] = grid.tvdRec(P[l2Idx], P[l1Idx], P[cIdx]);
      qR[3] = grid.tvdRec(P[r1Idx], P[cIdx],  P[l1Idx]);
      qD[3] = grid.tvdRec(P[d2Idx], P[d1Idx], P[cIdx]);
      qU[3] = grid.tvdRec(P[u1Idx], P[cIdx],  P[d1Idx]);

      fluxL = grid.hllcFlux(grid.prim2cons(qL), grid.prim2cons(qR), Vec2(1,0)); 
      fluxD = grid.hllcFlux(grid.prim2cons(qD), grid.prim2cons(qU), Vec2(0,1)); 
      
      //fluxL = grid.hllcFlux(grid.prim2cons(Vec4(Rho[l1Idx], U[l1Idx], V[l1Idx], P[l1Idx])),
      //                     grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx])),
      //                     Vec2(1,0)); 
      //fluxD = grid.hllcFlux(grid.prim2cons(Vec4(Rho[d1Idx], U[d1Idx], V[d1Idx], P[d1Idx])),
      //                     grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx])),
      //                     Vec2(0,1)); 

      atomicAdd(&RhsRho[cIdx],     fluxL[0] * dy / vol + fluxD[0] * dx / vol);
      atomicAdd(&RhsRho[l1Idx],  - fluxL[0] * dy / vol);
      atomicAdd(&RhsRho[d1Idx],  - fluxD[0] * dx / vol);

      atomicAdd(&RhsRhoU[cIdx],    fluxL[1] * dy / vol + fluxD[1] * dx / vol);
      atomicAdd(&RhsRhoU[l1Idx], - fluxL[1] * dy / vol);
      atomicAdd(&RhsRhoU[d1Idx], - fluxD[1] * dx / vol);
    
      atomicAdd(&RhsRhoV[cIdx],    fluxL[2] * dy / vol + fluxD[2] * dx / vol);
      atomicAdd(&RhsRhoV[l1Idx], - fluxL[2] * dy / vol);
      atomicAdd(&RhsRhoV[d1Idx], - fluxD[2] * dx / vol);

      atomicAdd(&RhsRhoE[cIdx],    fluxL[3] * dy / vol + fluxD[3] * dx / vol);
      atomicAdd(&RhsRhoE[l1Idx], - fluxL[3] * dy / vol);
      atomicAdd(&RhsRhoE[d1Idx], - fluxD[3] * dx / vol);
 

    // immersed boundary
    //Vec2 pos = grid.getCellPos(lvl, ib, jb, i, j);
    //real phi = grid.getBoundaryLevelSet(pos);
    //real X = grid.calcIbMask(phi);
//
    //real dt = grid.deltaT;
    //atomicAdd(&RhsRhoU[cIdx], 1.5*X/dt*Rho[cIdx]*(0.0 - U[cIdx]));
    //atomicAdd(&RhsRhoV[cIdx], 1.5*X/dt*Rho[cIdx]*(0.0 - V[cIdx]));

    //}

  END_CELL_LOOP

}


__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage) {
  //
  // update fields with low storage runge kutta
  //
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  real *RhsRho  = grid.getField(8);
  real *RhsRhoU = grid.getField(9);
  real *RhsRhoV = grid.getField(10);
  real *RhsRhoE = grid.getField(11);

  constexpr real alpha[3] = {5.0/9.0, 153.0/128.0, 0.0};
  constexpr real beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

  real dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      Rho[cIdx] += beta[stage] * dt * RhsRho[cIdx];
      RhoU[cIdx] += beta[stage] * dt * RhsRhoU[cIdx];
      RhoV[cIdx] += beta[stage] * dt * RhsRhoV[cIdx];
      RhoE[cIdx] += beta[stage] * dt * RhsRhoE[cIdx];

      RhsRho[cIdx]  *= - alpha[stage];
      RhsRhoU[cIdx] *= - alpha[stage];
      RhsRhoV[cIdx] *= - alpha[stage];
      RhsRhoE[cIdx] *= - alpha[stage];
    }

  END_CELL_LOOP

}

__global__ void updateFieldsRK3Kernel(CompressibleSolver &grid, i32 stage) {
  //
  // update fields with tvd runge kutta 3
  //
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  real *OldRho  = grid.getField(4);
  real *OldRhoU = grid.getField(5);
  real *OldRhoV = grid.getField(6);
  real *OldRhoE = grid.getField(7);

  real *RhsRho  = grid.getField(8);
  real *RhsRhoU = grid.getField(9);
  real *RhsRhoV = grid.getField(10);
  real *RhsRhoE = grid.getField(11);

  real dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      if (stage == 0) {
        OldRho[cIdx] = Rho[cIdx];
        OldRhoU[cIdx] = RhoU[cIdx];
        OldRhoV[cIdx] = RhoV[cIdx];
        OldRhoE[cIdx] = RhoE[cIdx];

        Rho[cIdx]  = Rho[cIdx]  + dt * RhsRho[cIdx];
        RhoU[cIdx] = RhoU[cIdx] + dt * RhsRhoU[cIdx];
        RhoV[cIdx] = RhoV[cIdx] + dt * RhsRhoV[cIdx];
        RhoE[cIdx] = RhoE[cIdx] + dt * RhsRhoE[cIdx];
      }

      if (stage == 1) {
        Rho[cIdx]  = 3.0/4.0*OldRho[cIdx]  + 1.0/4.0*Rho[cIdx]  + 1.0/4.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 3.0/4.0*OldRhoU[cIdx] + 1.0/4.0*RhoU[cIdx] + 1.0/4.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 3.0/4.0*OldRhoV[cIdx] + 1.0/4.0*RhoV[cIdx] + 1.0/4.0 * dt * RhsRhoV[cIdx];
        RhoE[cIdx] = 3.0/4.0*OldRhoE[cIdx] + 1.0/4.0*RhoE[cIdx] + 1.0/4.0 * dt * RhsRhoE[cIdx];
      }

      if (stage == 2) {
        Rho[cIdx]  = 1.0/3.0*OldRho[cIdx]  + 2.0/3.0*Rho[cIdx]  + 2.0/3.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 1.0/3.0*OldRhoU[cIdx] + 2.0/3.0*RhoU[cIdx] + 2.0/3.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 1.0/3.0*OldRhoV[cIdx] + 2.0/3.0*RhoV[cIdx] + 2.0/3.0 * dt * RhsRhoV[cIdx];
        RhoE[cIdx] = 1.0/3.0*OldRhoE[cIdx] + 2.0/3.0*RhoE[cIdx] + 2.0/3.0 * dt * RhsRhoE[cIdx];
      }

    }

    RhsRho[cIdx]  = 0;
    RhsRhoU[cIdx] = 0;
    RhsRhoV[cIdx] = 0;
    RhsRhoE[cIdx] = 0;

  END_CELL_LOOP

}

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid) {
  //
  // update fields with low storage runge kutta
  //
  real *Rho  = grid.getField(0);
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  real *OldRho  = grid.getField(4);
  real *OldRhoU = grid.getField(5);
  real *OldRhoV = grid.getField(6);
  real *OldRhoE = grid.getField(7);

  START_CELL_LOOP

    OldRho[cIdx] = Rho[cIdx];
    OldRhoU[cIdx] = RhoU[cIdx];
    OldRhoV[cIdx] = RhoV[cIdx];
    OldRhoE[cIdx] = RhoE[cIdx];

  END_CELL_LOOP
}

__global__ void computeMagRhoUKernel(CompressibleSolver &grid) {
  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);

  real *MagRhoU = grid.getField(12);

  START_CELL_LOOP

    MagRhoU[cIdx] = sqrt(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx]);

  END_CELL_LOOP
}

__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid) {

  real *Rho  = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *P = grid.getField(3);

  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  START_CELL_LOOP

      Vec4 qPrim = grid.cons2prim(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]));
      Rho[cIdx] = qPrim[0];
      U[cIdx]   = qPrim[1];
      V[cIdx]   = qPrim[2];
      P[cIdx]   = qPrim[3];

  END_CELL_LOOP

}

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid) {

  real *Rho  = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *P = grid.getField(3);

  real *RhoU = grid.getField(1);
  real *RhoV = grid.getField(2);
  real *RhoE = grid.getField(3);

  START_CELL_LOOP

    Vec4 qCons = grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx]));
    Rho[cIdx]  = qCons[0];
    RhoU[cIdx] = qCons[1];
    RhoV[cIdx] = qCons[2];
    RhoE[cIdx] = qCons[3];

  END_CELL_LOOP

}


__global__ void forwardWaveletTransformKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    u32 cFlag = grid.cFlagsList[cIdx];
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag != GHOST) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);
      u32 lIdx = grid.getNbrIdx(prntIdx, ip-1, jp);
      u32 rIdx = grid.getNbrIdx(prntIdx, ip+1, jp);
      u32 dIdx = grid.getNbrIdx(prntIdx, ip, jp-1);
      u32 uIdx = grid.getNbrIdx(prntIdx, ip, jp+1);
      u32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1);
      u32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1);
      u32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1);
      u32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1);

      real xs = 2 * (i % 2) - 1 ; // sign for interp weights
      real ys = 2 * (j % 2) - 1;

      // calculate detail coefficients for each field and set block to refine if large
      for(i32 f=0; f<4; f++) {
        real *Q  = grid.getField(f);
        real *OldQ  = grid.getField(f+4);
        Q[cIdx] = Q[cIdx] - (OldQ[pIdx] 
                + xs * 1/8 * (OldQ[rIdx] - OldQ[lIdx]) 
                + ys * 1/8 * (OldQ[uIdx] - OldQ[dIdx])
                + xs * ys * 1/64 * (OldQ[ruIdx] - OldQ[luIdx] - OldQ[rdIdx] + OldQ[ldIdx])); 
      }
    } 
    else if (cFlag == GHOST) {
      for(i32 f=0; f<4; f++) {
        real *Q = grid.getField(f);
        Q[cIdx] = 0.0; 
      }
    }

  END_CELL_LOOP
}

__global__ void inverseWaveletTransformKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && grid.bFlagsList[bIdx] != DELETE) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);
      u32 lIdx = grid.getNbrIdx(prntIdx, ip-1, jp);
      u32 rIdx = grid.getNbrIdx(prntIdx, ip+1, jp);
      u32 dIdx = grid.getNbrIdx(prntIdx, ip, jp-1);
      u32 uIdx = grid.getNbrIdx(prntIdx, ip, jp+1);
      u32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1);
      u32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1);
      u32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1);
      u32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1);

      real xs = 2 * (i % 2) - 1 ; // sign for interp weights
      real ys = 2 * (j % 2) - 1;

      // calculate detail coefficients for each field and set block to refine if large
      for(i32 f=0; f<4; f++) {
        real *Q  = grid.getField(f);
        real *OldQ  = grid.getField(f+4);
        Q[cIdx] = Q[cIdx] + (OldQ[pIdx] 
                + xs * 1/8 * (OldQ[rIdx] - OldQ[lIdx]) 
                + ys * 1/8 * (OldQ[uIdx] - OldQ[dIdx])
                + xs * ys * 1/64 * (OldQ[ruIdx] - OldQ[luIdx] - OldQ[rdIdx] + OldQ[ldIdx])); 
      }
    }

  END_CELL_LOOP
}


__global__ void waveletThresholdingKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);
    
    if (lvl < 2) {
      grid.bFlagsList[bIdx] = KEEP;
    }

    Vec2 pos = grid.getCellPos(lvl, ib, jb, i, j);
    real dx = min(grid.getDx(lvl), grid.getDy(lvl)) ;
    real ls = grid.getBoundaryLevelSet(Vec2(pos[0], pos[1]));
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb)) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];
      grid.bFlagsList[prntIdx] = KEEP;

      // calculate detail coefficients for each field and set block to refine if large
      for(i32 f=0; f<4; f++) {
        real *Q  = grid.getField(f);

        real mag = 1e-32;
        if (f == 0) {mag = grid.maxRho;}
        if (f == 1 || f == 2) {mag = grid.maxMagRhoU;}
        if (f == 3) {mag = grid.maxRhoE;}

        // refine block if large wavelet detail
        if (abs(Q[cIdx]/mag) > grid.waveletThresh || abs(ls) < dx) {
          if (lvl < grid.nLvls-1 && (abs(Q[cIdx]/mag) > grid.waveletThresh*2 || abs(ls) < dx)) {
            i32 bSize = blockSize/2;
            grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize);
          }
          grid.bFlagsList[bIdx] = KEEP;
        }
      }
    }

  END_CELL_LOOP
}

__global__ void interpolateFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    u32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag == GHOST) {
      
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);
      u32 lIdx = grid.getNbrIdx(prntIdx, ip-1, jp);
      u32 rIdx = grid.getNbrIdx(prntIdx, ip+1, jp);
      u32 dIdx = grid.getNbrIdx(prntIdx, ip, jp-1);
      u32 uIdx = grid.getNbrIdx(prntIdx, ip, jp+1);
      u32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1);
      u32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1);
      u32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1);
      u32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1);

      real xs = 2 * (i % 2) - 1 ; // sign for interp weights
      real ys = 2 * (j % 2) - 1;

      // interpolate each field from lower resolustion
      for(i32 f=0; f<4; f++) {
        real *Q  = grid.getField(f);
        Q[cIdx] = Q[pIdx]
                + xs * 1/8 * (Q[rIdx] - Q[lIdx]) 
                + ys * 1/8 * (Q[uIdx] - Q[dIdx])
                + xs * ys * 1/64 * (Q[ruIdx] - Q[luIdx] - Q[rdIdx] + Q[ldIdx]); 
      }
    }

  END_CELL_LOOP
}

__global__ void restrictFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.decode(loc, lvl, ib, jb);

    u32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag != GHOST && i%2==0 && j%2==0) {
      // sister cell indices
      i32 rIdx = cIdx + 1;
      i32 uIdx = cIdx + blockSize;
      i32 ruIdx = cIdx + blockSize + 1;

      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);

      for (u32 f=0; f<4; f++){
        real *q = grid.getField(f);
        q[pIdx] = (q[cIdx] + q[rIdx] + q[uIdx] + q[ruIdx])/4.0;
      }
    }

  END_CELL_LOOP
}
