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

    u64 loc = grid.zLocList[bIdx];
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

  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 bcType) {

  dataType *Rho = grid.getField(0);
  dataType *U   = grid.getField(1);
  dataType *V   = grid.getField(2);
  dataType *P   = grid.getField(3);

  START_CELL_LOOP

    u64 loc = grid.zLocList[bIdx];
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
        P[cIdx] = P[bcIdx];
        
        if (ib < 0 || ib >= gridSize[0]) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] =  V[bcIdx];
        }
        if (jb < 0 || jb >= gridSize[1]) {
          U[cIdx] =  U[bcIdx];
          V[cIdx] = -V[bcIdx];
        }
        if ((ib < 0 || ib >= gridSize[0]) && (jb < 0 || jb >= gridSize[1])) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] = -V[bcIdx];
        }
      }
    }

  END_CELL_LOOP
}


// compute max wavespeed in each cell, will be used for CFL condition
__global__ void computeDeltaTKernel(CompressibleSolver &grid) {
  dataType *Rho = grid.getField(0);
  dataType *U   = grid.getField(1);
  dataType *V   = grid.getField(2);
  dataType *P   = grid.getField(3);
  dataType *DeltaT = grid.getField(12);

  START_CELL_LOOP

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    dataType rho, u, v, p, a, dx, vel;
    rho = Rho[cIdx];
    u = U[cIdx];
    v = V[cIdx];
    p = P[cIdx];
    a = sqrt(abs(gam*p/rho));
    vel = sqrt(u*u + v*v);
    dx = min(grid.getDx(lvl), grid.getDy(lvl));
    DeltaT[cIdx] = dx / (a + vel + 1e-32);

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

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    //if (grid.isInteriorBlock(lvl, ib, jb)) {

      dataType dx = grid.getDx(lvl);
      dataType dy = grid.getDy(lvl);
      dataType vol = dx*dy;

      u32 l1Idx = grid.getNbrIdx(bIdx, i-1, j);
      u32 l2Idx = grid.getNbrIdx(bIdx, i-2, j);
      u32 r1Idx = grid.getNbrIdx(bIdx, i+1, j);

      u32 d1Idx = grid.getNbrIdx(bIdx, i, j-1);
      u32 d2Idx = grid.getNbrIdx(bIdx, i, j-2);
      u32 u1Idx = grid.getNbrIdx(bIdx, i, j+1);
      
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
  dataType *U = grid.getField(1);
  dataType *V = grid.getField(2);
  dataType *P = grid.getField(3);

  dataType *RhsRho  = grid.getField(4);
  dataType *RhsRhoU = grid.getField(5);
  dataType *RhsRhoV = grid.getField(6);
  dataType *RhsRhoE = grid.getField(7);

  constexpr dataType alpha[3] = {5.0/9.0, 153.0/128.0, 0.0};
  constexpr dataType beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

  dataType dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      Vec4 qCons = grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx]));
      qCons[0] += beta[stage] * dt * RhsRho[cIdx];
      qCons[1] += beta[stage] * dt * RhsRhoU[cIdx];
      qCons[2] += beta[stage] * dt * RhsRhoV[cIdx];
      qCons[3] += beta[stage] * dt * RhsRhoE[cIdx];

      Vec4 qPrim = grid.cons2prim(qCons);
      Rho[cIdx] = qPrim[0];
      U[cIdx]   = qPrim[1];
      V[cIdx]   = qPrim[2];
      P[cIdx]   = qPrim[3];

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
  dataType *U = grid.getField(1);
  dataType *V = grid.getField(2);
  dataType *P = grid.getField(3);

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

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      Vec4 qCons = grid.prim2cons(Vec4(Rho[cIdx], U[cIdx], V[cIdx], P[cIdx]));
      if (stage == 0) {
        OldRho[cIdx] = qCons[0];
        OldRhoU[cIdx] = qCons[1];
        OldRhoV[cIdx] = qCons[2];
        OldRhoE[cIdx] = qCons[3];

        qCons[0] = qCons[0] + dt * RhsRho[cIdx];
        qCons[1] = qCons[1] + dt * RhsRhoU[cIdx];
        qCons[2] = qCons[2] + dt * RhsRhoV[cIdx];
        qCons[3] = qCons[3] + dt * RhsRhoE[cIdx];
      }

      if (stage == 1) {
        qCons[0] = 3.0/4.0*OldRho[cIdx]  + 1.0/4.0*qCons[0] + 1.0/4.0*dt* RhsRho[cIdx];
        qCons[1] = 3.0/4.0*OldRhoU[cIdx] + 1.0/4.0*qCons[1] + 1.0/4.0*dt* RhsRhoU[cIdx];
        qCons[2] = 3.0/4.0*OldRhoV[cIdx] + 1.0/4.0*qCons[2] + 1.0/4.0*dt* RhsRhoV[cIdx];
        qCons[3] = 3.0/4.0*OldRhoE[cIdx] + 1.0/4.0*qCons[3] + 1.0/4.0*dt* RhsRhoE[cIdx];
      }

      if (stage == 2) {
        qCons[0] = 1.0/3.0*OldRho[cIdx]  + 2.0/3.0*qCons[0] + 2.0/3.0*dt* RhsRho[cIdx];
        qCons[1] = 1.0/3.0*OldRhoU[cIdx] + 2.0/3.0*qCons[1] + 2.0/3.0*dt* RhsRhoU[cIdx];
        qCons[2] = 1.0/3.0*OldRhoV[cIdx] + 2.0/3.0*qCons[2] + 2.0/3.0*dt* RhsRhoV[cIdx];
        qCons[3] = 1.0/3.0*OldRhoE[cIdx] + 2.0/3.0*qCons[3] + 2.0/3.0*dt* RhsRhoE[cIdx];
      }

      Vec4 qPrim = grid.cons2prim(qCons);
      Rho[cIdx] = qPrim[0];
      U[cIdx]   = qPrim[1];
      V[cIdx]   = qPrim[2];
      P[cIdx]   = qPrim[3];

      RhsRho[cIdx]  = 0;
      RhsRhoU[cIdx] = 0;
      RhsRhoV[cIdx] = 0;
      RhsRhoE[cIdx] = 0;
    }

  END_CELL_LOOP

}

/*

__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  dataType *Rho   = grid.getField(0);
  dataType *RhoU  = grid.getField(1);
  dataType *RhoV  = grid.getField(2);
  dataType *RhoEi = grid.getField(3);

  START_CELL_LOOP
  
  i32 i,j;
  grid.getLocalIdx(idx, i, j);
  


  END_CELL_LOOP
}
*/

/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  dataType *Rho   = grid.getField(0);
  dataType *RhoU  = grid.getField(1);
  dataType *RhoV  = grid.getField(2);
  dataType *RhoEi = grid.getField(3);

  START_CELL_LOOP

  END_CELL_LOOP
}
*/
/*
__global__ void forwardWaveletTransform(MultiLevelSparseGrid &grid)
{
  __shared__ dataType data;

  dataType *Rho = grid.getFieldData(0, )
  START_CELL_LOOP

  u32 flags = solver.blockList[bIdx].flags;

  if (flags & ACTIVE) {
    // calculate wavelet coefficients in active cells
    for (u32 f=0; f<4; f++){
      dataType& q = solver.getFieldValue(fields[f], bIdx, index);
      LOAD_LOW_RES_DATA(aux_fields[f])

    }

  }
  else if (flags & GHOST) {
    // set wavelet coefficients to zero in ghost cells
    for (u32 f=0; f<4; f++){
      solver.getFieldValue(fields[f], bIdx, index) = 0.0;
    }

  }

  END_CELL_LOOP
}


__global__ void inverseWaveletTransform(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  END_CELL_LOOP
}
*/

// restrict conserved fields to lower levels
/*
__global__ void restrictFields(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  u32 flags = solver.blockList[bIdx].flags;

  if (flags & ACTIVE == ACTIVE && lvl > 0) {
    for (u32 f=0; f<4; f++){
      dataType& q = solver.getFieldValue(aux_fields[f], bIdx, i, j);
      dataType& qp = solver.getParentFieldValue(fields[f], bIdx, ib, jb, i, j);
      atomicAdd(&qp, q/4);
    }
  }

  END_CELL_LOOP
}
*/

#endif
