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
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

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
        Rho[cIdx]  = 1.0;
        RhoU[cIdx] = 0.0;
        RhoV[cIdx] = 0.0;
        RhoE[cIdx] = 1.0/(gam-1);
      }
      else {
        Rho[cIdx]  = 0.125;
        RhoU[cIdx] = 0.0;
        RhoV[cIdx] = 0.0;
        RhoE[cIdx] = 0.1/(gam-1);
      }
    }

  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(CompressibleSolver &grid, i32 bcType) {

  dataType *Rho  = grid.getField(0);
  dataType *RhoU = grid.getField(1);
  dataType *RhoV = grid.getField(2);
  dataType *RhoE = grid.getField(3);

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
          ibc = 4;
        }
        if (jb < 0) {
          jbc = 4;
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
  dataType *DeltaT = grid.getField(8);

  START_CELL_LOOP

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    dataType rho, u, v, p, a, dx, vel;
    rho = Rho[cIdx];
    u = RhoU[cIdx]/rho;
    v = RhoV[cIdx]/rho;
    p = (gam-1.0)*(RhoE[cIdx] - rho*(u*u+v*v)/2.0);
    a = sqrt(abs(gam*p/rho));
    vel = sqrt(u*u + v*v);
    dx = fminf(grid.getDx(lvl), grid.getDy(lvl));
    DeltaT[cIdx] = dx / (a + vel + 1e-32);

  END_CELL_LOOP

}


__global__ void computeRightHandSideKernel(CompressibleSolver &grid) {
  dataType *Rho  = grid.getField(0);
  dataType *U = grid.getField(1);
  dataType *V = grid.getField(2);
  dataType *P = grid.getField(3);

  dataType *RhsRho  = grid.getField(4);
  dataType *RhsRhoU = grid.getField(5);
  dataType *RhsRhoV = grid.getField(6);
  dataType *RhsRhoE = grid.getField(7);


  START_CELL_LOOP

    u64 loc = grid.zLocList[bIdx];
    i32 lvl, ib, jb;
    grid.mortonDecode(loc, lvl, ib, jb);

    if (grid.isInteriorBlock(lvl, ib, jb)) {

      dataType dx = grid.getDx(lvl);
      dataType dy = grid.getDy(lvl);
      dataType vol = dx*dy;

      u32 l1Idx = grid.getNbrIdx(bIdx, i-1, j);
      u32 l2Idx = grid.getNbrIdx(bIdx, i-2, j);
      u32 r1Idx = grid.getNbrIdx(bIdx, i+1, j);
      u32 r2Idx = grid.getNbrIdx(bIdx, i+2, j);

      u32 d1Idx = grid.getNbrIdx(bIdx, i, j-1);
      u32 d2Idx = grid.getNbrIdx(bIdx, i, j-1);
      u32 u1Idx = grid.getNbrIdx(bIdx, i, j+1);
      u32 u2Idx = grid.getNbrIdx(bIdx, i, j+2);
      
      Vec4 flux;
      // left flux
      Vec4 qL;
      Vec4 qR;
      qL[0] = grid.tvdRec(Rho[l2Idx], Rho[l1Idx], Rho[cIdx]);
      qR[0] = grid.tvdRec(Rho[r1Idx], Rho[cIdx], Rho[l1Idx]);
      qL[1] = grid.tvdRec(U[l2Idx], U[l1Idx], U[cIdx]);
      qR[1] = grid.tvdRec(U[r1Idx], U[cIdx], U[l1Idx]);
      qL[2] = grid.tvdRec(V[l2Idx], V[l1Idx], V[cIdx]);
      qR[2] = grid.tvdRec(V[r1Idx], V[cIdx], V[l1Idx]);
      qL[3] = grid.tvdRec(P[l2Idx], P[l1Idx], P[cIdx]);
      qR[3] = grid.tvdRec(P[r1Idx], P[cIdx], P[l1Idx]);
      flux = grid.hlleFlux(grid.prim2cons(qL), grid.prim2cons(qR), Vec2(1,0)); 
      RhsRho[cIdx]  += flux[0] * dy / vol;
      RhsRhoU[cIdx] += flux[1] * dy / vol;
      RhsRhoV[cIdx] += flux[2] * dy / vol;
      RhsRhoE[cIdx] += flux[3] * dy / vol;

      // right flux
      flux = grid.hlleFlux(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]),
                           Vec4(Rho[rIdx], RhoU[rIdx], RhoV[rIdx], RhoE[rIdx]),
                           Vec2(1,0)); 
      RhsRho[cIdx]  -= flux[0] * dy / vol;
      RhsRhoU[cIdx] -= flux[1] * dy / vol;
      RhsRhoV[cIdx] -= flux[2] * dy / vol;
      RhsRhoE[cIdx] -= flux[3] * dy / vol;

      // down flux
      flux = grid.hlleFlux(Vec4(Rho[dIdx], RhoU[dIdx], RhoV[dIdx], RhoE[dIdx]),
                           Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]),
                           Vec2(0,1)); 
      RhsRho[cIdx]  += flux[0] * dx / vol;
      RhsRhoU[cIdx] += flux[1] * dx / vol;
      RhsRhoV[cIdx] += flux[2] * dx / vol;
      RhsRhoE[cIdx] += flux[3] * dx / vol;

      // up flux
      flux = grid.hlleFlux(Vec4(Rho[cIdx], RhoU[cIdx], RhoV[cIdx], RhoE[cIdx]),
                           Vec4(Rho[uIdx], RhoU[uIdx], RhoV[uIdx], RhoE[uIdx]),
                           Vec2(0,1)); 
      RhsRho[cIdx]  -= flux[0] * dx / vol;
      RhsRhoU[cIdx] -= flux[1] * dx / vol;
      RhsRhoV[cIdx] -= flux[2] * dx / vol;
      RhsRhoE[cIdx] -= flux[3] * dx / vol;            

    }

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

  dataType *RhsRho  = grid.getField(4);
  dataType *RhsRhoU = grid.getField(5);
  dataType *RhsRhoV = grid.getField(6);
  dataType *RhsRhoE = grid.getField(7);

  constexpr dataType alpha[3] = {5.0/9.0, 153.0/128.0, 0.0};
  constexpr dataType beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

  dataType dt = .5*grid.deltaT;

  START_CELL_LOOP

    Rho[cIdx]  +=  beta[stage] * dt * RhsRho[cIdx];
    RhoU[cIdx] +=  beta[stage] * dt * RhsRhoU[cIdx];
    RhoV[cIdx] +=  beta[stage] * dt * RhsRhoV[cIdx];
    RhoE[cIdx] +=  beta[stage] * dt * RhsRhoE[cIdx];

    RhsRho[cIdx]  *= - alpha[stage];
    RhsRhoU[cIdx] *= - alpha[stage];
    RhsRhoV[cIdx] *= - alpha[stage];
    RhsRhoE[cIdx] *= - alpha[stage];

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
