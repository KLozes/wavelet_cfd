#ifndef COMPRESSIBLE_SOLVER_KERNELS_H
#define COMPRESSIBLE_SOLVER_KERNELS_H

#include "CompressibleSolver.cuh"


__global__ void sortFieldDataKernel(CompressibleSolver &grid) {
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  dataType *OldRho  = grid.getField(4);
  dataType *OldRhoU = grid.getField(5);
  dataType *OldRhoV = grid.getField(6);
  dataType *OldRhoE = grid.getField(7);

  START_CELL_LOOP

    u32 bIdxOld = grid.bIdxList[bIdx];
    u32 cIdxOld = bIdxOld * blockSizeTot + idx;
    
    Rho[cIdx] = OldRho[cIdxOld];
    RhoU[cIdx] = OldRhoU[cIdxOld];
    RhoV[cIdx] = OldRhoV[cIdxOld];
    RhoE[cIdx] = OldRhoE[cIdxOld];

  END_CELL_LOOP

}

__global__ void setInitialConditionsKernel(CompressibleSolver &grid, i32 icType) {

  dataType *Rho  = grid.getField(0);
  dataType *U    = grid.getField(1);
  dataType *V    = grid.getField(2);
  dataType *P    = grid.getField(3);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);
    dataType pos[2];
    grid.getCellPos(lvl, ib, jb, i, j, pos);

    if (icType == 0) {
      //
      // sod shock explosion
      //
      dataType centerX = grid.domainSize[0]/2;
      dataType centerY = grid.domainSize[1]/2;
      dataType radius = grid.domainSize[0]/5;

      dataType dist = sqrt((pos[0]-centerX)*(pos[0]-centerX) + (pos[1]-centerY)*(pos[1]-centerY));
    
      // inside
      if (dist < radius) {
        Rho[cIdx]  = 1.0;
        U[cIdx]    = 0.0;
        V[cIdx]    = 0.0;
        P[cIdx]    = 1.0;
      }
      else {
        Rho[cIdx]  = 0.125;
        U[cIdx]    = 0.0;
        V[cIdx]    = 0.0;
        P[cIdx]    = 0.1;
      }
    }

    if (icType == 1) {
      //
      // gaussian explosion
      //
    
      Rho[cIdx] = 10.0*exp(-3000 * ((pos[0] - .4)*(pos[0] - .4) + (pos[1] - .4)*(pos[1] - .4))) + .125;
      P[cIdx] = 10.0*exp(-3000 * ((pos[0] - .4)*(pos[0] - .4) + (pos[1] - .4)*(pos[1] - .4))) + .1;
      U[cIdx]    = 0.0;
      V[cIdx]    = 0.0;
    }


  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 bcType) {

  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isExteriorBlock(lvl, ib, jb)) {
      u32 gridSize[2] = {grid.baseGridSize[0]*powi(2, lvl)/blockSize, 
                         grid.baseGridSize[1]*powi(2, lvl)/blockSize};
      if (bcType == 0) {
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
    }

  END_CELL_LOOP
}


// compute max wavespeed in each cell, will be used for CFL condition
__global__ void computeDeltaTKernel(CompressibleSolver &grid) {
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);
  dataType *DeltaT = grid.getField(12);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {
      dataType a, dx, vel;
      Vec4 q = grid.cons2prim(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]));
      a = sqrt(abs(gam*q[3]/q[0]));
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
  dataType *Rho = grid.getField(0);
  dataType *U   = grid.getField(1);
  dataType *V   = grid.getField(2);
  dataType *P   = grid.getField(3);

  dataType *RhsRho  = grid.getField(8);
  dataType *RhsRhoU = grid.getField(9);
  dataType *RhsRhoV = grid.getField(10);
  dataType *RhsRhoE = grid.getField(11);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    dataType dx = grid.getDx(lvl);
    dataType dy = grid.getDy(lvl);
    dataType vol = dx*dy;

    u32 l1Idx = grid.getNbrIdx(bIdx, i-1, j);
    u32 l2Idx = grid.getNbrIdx(bIdx, i-2, j);
    u32 r1Idx = grid.getNbrIdx(bIdx, i+1, j);

    u32 d1Idx = grid.getNbrIdx(bIdx, i, j-1);
    u32 d2Idx = grid.getNbrIdx(bIdx, i, j-2);
    u32 u1Idx = grid.getNbrIdx(bIdx, i, j+1);

    u32 cFlag = grid.cFlagsList[cIdx];
    u32 lFlag = grid.cFlagsList[l1Idx];
    u32 dFlag = grid.cFlagsList[d1Idx];

    //if (grid.isInteriorBlock(lvl, ib, jb)) {
      
      Vec4 fluxL;
      Vec4 fluxD;
      Vec4 qL;
      Vec4 qR;
      Vec4 qD;
      Vec4 qU;

      // left flux
      qL[0] = grid.tvdRec(Rho[l2Idx], Rho[l1Idx], Rho[cIdx]);
      qR[0] = grid.tvdRec(Rho[r1Idx], Rho[cIdx], Rho[l1Idx]);
      qD[0] = grid.tvdRec(Rho[d2Idx], Rho[d1Idx], Rho[cIdx]);
      qU[0] = grid.tvdRec(Rho[u1Idx], Rho[cIdx], Rho[d1Idx]);

      qL[1] = grid.tvdRec(U[l2Idx], U[l1Idx], U[cIdx]);
      qR[1] = grid.tvdRec(U[r1Idx], U[cIdx], U[l1Idx]);
      qD[1] = grid.tvdRec(U[d2Idx], U[d1Idx], U[cIdx]);
      qU[1] = grid.tvdRec(U[u1Idx], U[cIdx], U[d1Idx]);

      qL[2] = grid.tvdRec(V[l2Idx], V[l1Idx], V[cIdx]);
      qR[2] = grid.tvdRec(V[r1Idx], V[cIdx], V[l1Idx]);
      qD[2] = grid.tvdRec(V[d2Idx], V[d1Idx], V[cIdx]);
      qU[2] = grid.tvdRec(V[u1Idx], V[cIdx], V[d1Idx]);

      qL[3] = grid.tvdRec(P[l2Idx], P[l1Idx], P[cIdx]);
      qR[3] = grid.tvdRec(P[r1Idx], P[cIdx], P[l1Idx]);
      qD[3] = grid.tvdRec(P[d2Idx], P[d1Idx], P[cIdx]);
      qU[3] = grid.tvdRec(P[u1Idx], P[cIdx], P[d1Idx]);

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
 

    //}

  END_CELL_LOOP

}


__global__ void updateFieldsKernel(CompressibleSolver &grid, i32 stage) {
  //
  // update fields with low storage runge kutta
  //
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  dataType *RhsRho  = grid.getField(8);
  dataType *RhsRhoU = grid.getField(9);
  dataType *RhsRhoV = grid.getField(10);
  dataType *RhsRhoE = grid.getField(11);

  constexpr dataType alpha[3] = {5.0/9.0, 153.0/128.0, 0.0};
  constexpr dataType beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

  dataType dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

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
  // update fields with low storage runge kutta
  //
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  dataType *OldRho  = grid.getField(4);
  dataType *OldRhoU = grid.getField(5);
  dataType *OldRhoV = grid.getField(6);
  dataType *OldRhoE = grid.getField(7);

  dataType *RhsRho  = grid.getField(8);
  dataType *RhsRhoU = grid.getField(9);
  dataType *RhsRhoV = grid.getField(10);
  dataType *RhsRhoE = grid.getField(11);

  dataType dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

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
        Rho[cIdx] = 3.0/4.0*OldRho[cIdx]   + 1.0/4.0*Rho[cIdx]   + 1.0/4.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 3.0/4.0*OldRhoU[cIdx] + 1.0/4.0*RhoU[cIdx] + 1.0/4.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 3.0/4.0*OldRhoV[cIdx] + 1.0/4.0*RhoV[cIdx] + 1.0/4.0 * dt * RhsRhoV[cIdx];
        RhoE[cIdx] = 3.0/4.0*OldRhoE[cIdx] + 1.0/4.0*RhoE[cIdx] + 1.0/4.0 * dt * RhsRhoE[cIdx];
      }

      if (stage == 2) {
        Rho[cIdx] = 1.0/3.0*OldRho[cIdx]   + 2.0/3.0*Rho[cIdx]  + 2.0/3.0 * dt * RhsRho[cIdx];
        RhoU[cIdx] = 1.0/3.0*OldRhoU[cIdx] + 2.0/3.0*RhoU[cIdx] + 2.0/3.0 * dt * RhsRhoU[cIdx];
        RhoV[cIdx] = 1.0/3.0*OldRhoV[cIdx] + 2.0/3.0*RhoV[cIdx] + 2.0/3.0 * dt * RhsRhoV[cIdx];
        RhoE[cIdx] = 1.0/3.0*OldRhoE[cIdx] + 2.0/3.0*RhoE[cIdx] + 2.0/3.0 * dt * RhsRhoE[cIdx];
      }

      RhsRho[cIdx]  = 0;
      RhsRhoU[cIdx] = 0;
      RhsRhoV[cIdx] = 0;
      RhsRhoE[cIdx] = 0;
    }

  END_CELL_LOOP

}

__global__ void copyToOldFieldsKernel(CompressibleSolver &grid) {
  //
  // update fields with low storage runge kutta
  //
  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  dataType *OldRho  = grid.getField(4);
  dataType *OldRhoU = grid.getField(5);
  dataType *OldRhoV = grid.getField(6);
  dataType *OldRhoE = grid.getField(7);

  START_CELL_LOOP

    OldRho[cIdx] = Rho[cIdx];
    OldRhoU[cIdx] = RhoU[cIdx];
    OldRhoV[cIdx] = RhoV[cIdx];
    OldRhoE[cIdx] = RhoE[cIdx];

  END_CELL_LOOP
}

__global__ void computeMagRhoUKernel(CompressibleSolver &grid) {
  //
  // update fields with low storage runge kutta
  //
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);

  dataType *MagRhoU = grid.getField(12);

  START_CELL_LOOP

    MagRhoU[cIdx] = sqrt(RhoU[cIdx]*RhoU[cIdx] + RhoV[cIdx]*RhoV[cIdx]);

  END_CELL_LOOP
}


__global__ void conservativeToPrimitiveKernel(CompressibleSolver &grid) {

  dataType *Rho  = grid.getField(0);
  dataType *U = grid.getField(1);
  dataType *V = grid.getField(2);
  dataType *P = grid.getField(3);

  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  START_CELL_LOOP

      Vec4 qPrim = grid.cons2prim(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]));
      Rho[cIdx] = qPrim[0];
      U[cIdx]   = qPrim[1];
      V[cIdx]   = qPrim[2];
      P[cIdx]   = qPrim[3];

  END_CELL_LOOP

}

__global__ void primitiveToConservativeKernel(CompressibleSolver &grid) {

  dataType *Rho  = grid.getField(0);
  dataType *U = grid.getField(1);
  dataType *V = grid.getField(2);
  dataType *P = grid.getField(3);

  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

  START_CELL_LOOP

      Vec4 qCons = grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx]));
      Rho[cIdx]  = qCons[0];
      RhoU[cIdx] = qCons[1];
      RhoV[cIdx] = qCons[2];
      RhoE[cIdx] = qCons[3];

  END_CELL_LOOP

}


__global__ void waveletThresholdingKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (lvl < 2) {
      grid.bFlagsList[bIdx] = KEEP;
    }

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb)) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];
      //grid.bFlagsList[prntIdx] = KEEP;
      atomicMax(&(grid.bFlagsList[prntIdx]),KEEP);

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

      dataType xs = 2 * (i % 2) - 1 ; // sign for interp weights
      dataType ys = 2 * (j % 2) - 1;

      // calculate detail coefficients for each field and set block to refine if large
      for(i32 f=0; f<4; f++) {
        dataType *Q  = grid.getField(f);
        dataType detail = Q[cIdx] - (Q[pIdx] 
                + xs * 1/8 * (Q[rIdx] - Q[lIdx]) 
                + ys * 1/8 * (Q[uIdx] - Q[dIdx])
                + xs * ys * 1/64 * (Q[ruIdx] - Q[luIdx] - Q[rdIdx] + Q[ldIdx])); 

        dataType mag = 1e-32;
        if (f == 0) {mag = grid.maxRho;}
        if (f == 1 || f == 2) {mag = grid.maxMagRhoU;}
        if (f == 3) {mag = grid.maxRhoE;}

        // refine block if large wavelet detail
        if (abs(detail/mag) > grid.waveletThresh) {
          grid.bFlagsList[bIdx] = REFINE;
          break;
        }
      }
    }

  END_CELL_LOOP
}

__global__ void interpolateFieldsKernel(CompressibleSolver &grid) {

  START_CELL_LOOP
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    u32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag == GHOST) {

      grid.bFlagsList[bIdx] = KEEP;
      
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

      dataType xs = 2 * (i % 2) - 1 ; // sign for interp weights
      dataType ys = 2 * (j % 2) - 1;

      // interpolate each field from lower resolustion
      for(i32 f=0; f<4; f++) {
        dataType *Q  = grid.getField(f);
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
    grid.mortonDecode(loc, lvl, ib, jb);

    u32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag == ACTIVE) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);

      if (lvl > 0) {
        for (u32 f=0; f<4; f++){
          dataType* q = grid.getField(f);
          q[pIdx] = 0.0;
        }
      }
    }

    __syncthreads();

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb) && cFlag == ACTIVE) {
      // parent block memory index
      u32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;

      // parent cell memory indices
      u32 pIdx = grid.getNbrIdx(prntIdx, ip, jp);

      if (grid.cFlagsList[cIdx] == ACTIVE && lvl > 0) {
        for (u32 f=0; f<4; f++){
          dataType *q = grid.getField(f);
          atomicAdd(&q[pIdx], q[cIdx]/4);
        }
      }
    }

  END_CELL_LOOP
}

#endif
