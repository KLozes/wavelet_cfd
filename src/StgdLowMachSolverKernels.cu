#include "StgdLowMachSolverKernels.cuh"

__global__ void sortFieldDataKernel(StgdLowMachSolver &grid) {
  real *P = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  real *OldP = grid.getField(4);
  real *OldU = grid.getField(5);
  real *OldV = grid.getField(6);
  real *OldW = grid.getField(7);

  START_CELL_LOOP

    i32 bIdxOld = grid.bIdxList[bIdx];
    i32 cIdxOld = bIdxOld * blockSizeTot + cIdx%blockSizeTot;
    
    P[cIdx] = OldP[cIdxOld];
    U[cIdx] = OldU[cIdxOld];
    V[cIdx] = OldV[cIdxOld];
    W[cIdx] = OldW[cIdxOld];
    grid.bFlagsList[bIdxOld] = DELETE;

  END_CELL_LOOP

}

__global__ void setInitialConditionsKernel(StgdLowMachSolver &grid) {

  real *P  = grid.getField(0);
  real *U  = grid.getField(1);
  real *V  = grid.getField(2);
  real *W  = grid.getField(3);

  START_CELL_LOOP
   GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);

    if (grid.icType == 0) { // quiescent
      P[cIdx] = 0.0;
      U[cIdx] = 0.0;
      V[cIdx] = 0.0;
      W[cIdx] = 0.0;
    }

    if (grid.icType == 1) {
    }

    if (grid.icType == 2) {
      //
      // wind tunnel
      //
      P[cIdx] = 1.0;
      U[cIdx] = 3.0;
      V[cIdx] = 0.0;
      U[cIdx] = 0.0;
    }



  END_CELL_LOOP
}

__global__ void setBoundaryConditionsKernel(StgdLowMachSolver &grid) {

  real *P  = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isExteriorBlock(lvl, ib, jb, kb)) {
      // grid size at this resolution level
      i32 gridSize[3] = {grid.baseGridSize[0]*powi(2, lvl)/blockSize, 
                         grid.baseGridSize[1]*powi(2, lvl)/blockSize, 
                         grid.baseGridSize[2]*powi(2, lvl)/blockSize};
      if (grid.bcType == 0) {
        //
        // slip wall
        //

        // figure out internal cell for neuman boundary conditions
        i32 ibc = i;
        i32 jbc = j;
        i32 kbc = k;
        if (ib < 0) {
          ibc = blockSize;
        }
        if (jb < 0) {
          jbc = blockSize;
        }
        if (kb < 0) {
          kbc = blockSize;
        }
        if (ib >= i32(gridSize[0])) {
          ibc = -1; 
        }
        if (jb >= i32(gridSize[1])) {
          jbc = -1;
        }
        if (jb >= i32(gridSize[2])) {
          jbc = -1;
        }
        i32 bcIdx = grid.getNbrIdx(bIdx, ibc, jbc, kbc); 

        // apply boundary conditions
        P[cIdx] = P[bcIdx];
        
        if (ib < 0 || ib >= gridSize[0]) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] =  V[bcIdx];
          W[cIdx] =  W[bcIdx];
        }
        if (jb < 0 || jb >= gridSize[1]) {
          U[cIdx] =  U[bcIdx];
          V[cIdx] = -V[bcIdx];
          W[cIdx] =  W[bcIdx];
        }
        if (kb < 0 || kb >= gridSize[2]) {
          U[cIdx] =  U[bcIdx];
          V[cIdx] =  V[bcIdx];
          W[cIdx] = -W[bcIdx];
        }

        if ((ib < 0 || ib >= gridSize[0]) && (jb < 0 || jb >= gridSize[1])) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] = -V[bcIdx];
          W[cIdx] = W[bcIdx];
        }

        if ((ib < 0 || ib >= gridSize[0]) && (kb < 0 || kb >= gridSize[2])) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] = V[bcIdx];
          W[cIdx] = -W[bcIdx];
        }

        if ((jb < 0 || jb >= gridSize[1]) && (kb < 0 || kb >= gridSize[2])) {
          U[cIdx] = U[bcIdx];
          V[cIdx] = -V[bcIdx];
          W[cIdx] = -W[bcIdx];
        }
      }

      if (grid.bcType == 1) {
        //
        // no-slip wall
        //

        // figure out internal cell for neuman boundary conditions
        i32 ibc = i;
        i32 jbc = j;
        i32 kbc = k;
        if (ib < 0) {
          ibc = blockSize;
        }
        if (jb < 0) {
          jbc = blockSize;
        }
        if (kb < 0) {
          kbc = blockSize;
        }
        if (ib >= i32(gridSize[0])) {
          ibc = -1; 
        }
        if (jb >= i32(gridSize[1])) {
          jbc = -1;
        }
        if (jb >= i32(gridSize[2])) {
          jbc = -1;
        }
        i32 bcIdx = grid.getNbrIdx(bIdx, ibc, jbc, kbc); 

        // apply boundary conditions
        P[cIdx] = P[bcIdx];
        
        if (ib < 0 || ib >= gridSize[0]) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] =  0.0;
          W[cIdx] =  0.0;
        }
        if (jb < 0 || jb >= gridSize[1]) {
          U[cIdx] =  0.0;
          V[cIdx] = -V[bcIdx];
          W[cIdx] =  0.0;
        }
        if (kb < 0 || kb >= gridSize[2]) {
          U[cIdx] =  0.0;
          V[cIdx] =  0.0;
          W[cIdx] = -W[bcIdx];
        }

        if ((ib < 0 || ib >= gridSize[0]) && (jb < 0 || jb >= gridSize[1])) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] = -V[bcIdx];
          W[cIdx] = W[bcIdx];
        }

        if ((ib < 0 || ib >= gridSize[0]) && (kb < 0 || kb >= gridSize[2])) {
          U[cIdx] = -U[bcIdx];
          V[cIdx] = V[bcIdx];
          W[cIdx] = -W[bcIdx];
        }

        if ((jb < 0 || jb >= gridSize[1]) && (kb < 0 || kb >= gridSize[2])) {
          U[cIdx] = U[bcIdx];
          V[cIdx] = -V[bcIdx];
          W[cIdx] = -W[bcIdx];
        }
      }
    }

  END_CELL_LOOP
}


// compute max wavespeed in each cell, will be used for CFL condition
__global__ void computeDeltaTKernel(StgdLowMachSolver &grid) {
  real *P  = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);
  real *DeltaT = grid.getField(12);

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {
      real vel = sqrt(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx]);
      real dx = min(grid.getDx(lvl), min(grid.getDy(lvl), grid.getDz(lvl)));
      DeltaT[cIdx] = dx / (grid.sos + vel);
    }
    else {
      DeltaT[cIdx] = 1e32;
    }

  END_CELL_LOOP

}


__global__ void computeRightHandSideKernel(StgdLowMachSolver &grid) {
  real *P = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  real *RhsP = grid.getField(4);
  real *RhsU = grid.getField(5);
  real *RhsV = grid.getField(6);
  real *RhsW = grid.getField(7);

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    real dx = grid.getDx(lvl);
    real dy = grid.getDy(lvl);
    real dz = grid.getDz(lvl);
    real vol = dx*dy*dx;

    i32 l1Idx = grid.getNbrIdx(bIdx, i-1, j, k);
    i32 l2Idx = grid.getNbrIdx(bIdx, i-2, j, k);
    i32 r1Idx = grid.getNbrIdx(bIdx, i+1, j, k);

    i32 d1Idx = grid.getNbrIdx(bIdx, i, j-1, k);
    i32 d2Idx = grid.getNbrIdx(bIdx, i, j-2, k);
    i32 u1Idx = grid.getNbrIdx(bIdx, i, j+1, k);

  
  END_CELL_LOOP

}


__global__ void updateFieldsKernel(StgdLowMachSolver &grid, i32 stage) {
  //
  // update fields with low storage runge kutta
  //
  real *P  = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  real *RhsP  = grid.getField(4);
  real *RhsU = grid.getField(5);
  real *RhsV = grid.getField(6);
  real *RhsW = grid.getField(7);

  constexpr real alpha[3] = {5.0/9.0, 153.0/128.0, 0.0};
  constexpr real beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

  real dt = grid.deltaT;

  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (grid.isInteriorBlock(lvl, ib, jb, kb)) {

      P[cIdx] += beta[stage] * dt * RhsP[cIdx];
      U[cIdx] += beta[stage] * dt * RhsU[cIdx];
      V[cIdx] += beta[stage] * dt * RhsV[cIdx];
      W[cIdx] += beta[stage] * dt * RhsW[cIdx];

      RhsP[cIdx] *= - alpha[stage];
      RhsU[cIdx] *= - alpha[stage];
      RhsV[cIdx] *= - alpha[stage];
      RhsW[cIdx] *= - alpha[stage];
    }

  END_CELL_LOOP

}

__global__ void copyToOldFieldsKernel(StgdLowMachSolver &grid) {
  //
  // copy fields to auxilary memory
  //
  real *P = grid.getField(0);
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  real *OldP = grid.getField(4);
  real *OldU = grid.getField(5);
  real *OldV = grid.getField(6);
  real *OldW = grid.getField(7);

  START_CELL_LOOP

    OldP[cIdx] = P[cIdx];
    OldU[cIdx] = U[cIdx];
    OldV[cIdx] = V[cIdx];
    OldW[cIdx] = W[cIdx];

  END_CELL_LOOP
}

__global__ void computeMagRhoUKernel(StgdLowMachSolver &grid) {
  real *U = grid.getField(1);
  real *V = grid.getField(2);
  real *W = grid.getField(3);

  real *MagU = grid.getField(8);

  START_CELL_LOOP

    MagU[cIdx] = sqrt(U[cIdx]*U[cIdx] + V[cIdx]*V[cIdx] + W[cIdx]*W[cIdx]);

  END_CELL_LOOP
}

__global__ void forwardWaveletTransformKernel(StgdLowMachSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES
  
    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST) {
      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = j/2 + kb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      i32 pIdx  = grid.getNbrIdx(prntIdx,   ip,   jp, kp);
      i32 lIdx  = grid.getNbrIdx(prntIdx, ip-1,   jp, kp);
      i32 rIdx  = grid.getNbrIdx(prntIdx, ip+1,   jp, kp);
      i32 dIdx  = grid.getNbrIdx(prntIdx,   ip, jp-1, kp);
      i32 uIdx  = grid.getNbrIdx(prntIdx,   ip, jp+1, kp);
      i32 bIdx  = grid.getNbrIdx(prntIdx,   ip, jp, kp-1);
      i32 fIdx  = grid.getNbrIdx(prntIdx,   ip, jp, kp+1);

      i32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp);
      i32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp);
      i32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp);
      i32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp);

      real xs = 2 * (i % 2) - 1 ; // sign for interp weights
      real ys = 2 * (j % 2) - 1;
      real zs = 2 * (k % 2) - 1;

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

__global__ void inverseWaveletTransformKernel(StgdLowMachSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && grid.bFlagsList[bIdx] != DELETE) {
      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = k/2 + kb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);
      i32 lIdx = grid.getNbrIdx(prntIdx, ip-1, jp, kp);
      i32 rIdx = grid.getNbrIdx(prntIdx, ip+1, jp, kp);
      i32 dIdx = grid.getNbrIdx(prntIdx, ip, jp-1, kp);
      i32 uIdx = grid.getNbrIdx(prntIdx, ip, jp+1, kp);
      i32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp);
      i32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp);
      i32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp);
      i32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp);

      real xs = 2 * (i % 2) - 1 ; // sign for interp weights
      real ys = 2 * (j % 2) - 1;
      real zs = 2 * (k % 2) - 1;

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


__global__ void waveletThresholdingKernel(StgdLowMachSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);
    
    if (lvl < 2) {
      grid.bFlagsList[bIdx] = KEEP;
    }

    Vec3 pos = grid.getCellPos(lvl, ib, jb, kb, i, j, k);
    real dx = min(grid.getDx(lvl), grid.getDy(lvl)) ;
    real ls = grid.getBoundaryLevelSet(pos);
    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];
      grid.bFlagsList[prntIdx] = KEEP;

      // calculate detail coefficients for each field and set block to refine if large
      for(i32 f=0; f<4; f++) {
        real *Q  = grid.getField(f);

        real mag = 1e-32;
        if (f == 0) {mag = grid.maxP;}
        if (f > 0  && f < 4) {mag = grid.maxU;}

        // refine block if large wavelet detail
        if (abs(Q[cIdx]/mag) > grid.waveletThresh || abs(ls) < dx) {
          if (lvl < grid.nLvls-1 && (abs(Q[cIdx]/mag) > grid.waveletThresh*2 || abs(ls) < dx)) {
            i32 bSize = blockSize/2;
            grid.activateBlock(lvl+1, 2*ib+i/bSize, 2*jb+j/bSize, 2*kb+k/bSize);
          }
          grid.bFlagsList[bIdx] = KEEP;
        }
      }
    }

  END_CELL_LOOP
}

__global__ void interpolateFieldsKernel(StgdLowMachSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag == GHOST) {
      
      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = k/2 + kb%2 * blockSize / 2;

      // parent and neigboring cell memory indices
      i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);
      i32 lIdx = grid.getNbrIdx(prntIdx, ip-1, jp, kp);
      i32 rIdx = grid.getNbrIdx(prntIdx, ip+1, jp, kp);
      i32 dIdx = grid.getNbrIdx(prntIdx, ip, jp-1, kp);
      i32 uIdx = grid.getNbrIdx(prntIdx, ip, jp+1, kp);
      i32 ldIdx = grid.getNbrIdx(prntIdx, ip-1, jp-1, kp);
      i32 rdIdx = grid.getNbrIdx(prntIdx, ip+1, jp-1, kp);
      i32 luIdx = grid.getNbrIdx(prntIdx, ip-1, jp+1, kp);
      i32 ruIdx = grid.getNbrIdx(prntIdx, ip+1, jp+1, kp);

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

__global__ void restrictFieldsKernel(StgdLowMachSolver &grid) {

  START_CELL_LOOP
    GET_CELL_INDICES


    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    i32 cFlag = grid.cFlagsList[cIdx];

    if (lvl > 0 && grid.isInteriorBlock(lvl, ib, jb, kb) && cFlag != GHOST && i%2==0 && j%2==0) {
      // sister cell indices
      i32 rIdx = cIdx + 1;
      i32 uIdx = cIdx + blockSize;
      i32 ruIdx = cIdx + blockSize + 1;

      // parent block memory index
      i32 prntIdx = grid.prntIdxList[bIdx];

      // parent cell local indices
      i32 ip = i/2 + ib%2 * blockSize / 2;
      i32 jp = j/2 + jb%2 * blockSize / 2;
      i32 kp = k/2 + kp%2 * blockSize / 2;

      // parent cell memory indices
      i32 pIdx = grid.getNbrIdx(prntIdx, ip, jp, kp);

      for (i32 f=0; f<4; f++){
        real *q = grid.getField(f);
        q[pIdx] = (q[cIdx] + q[rIdx] + q[uIdx] + q[ruIdx])/4.0;
      }
    }

  END_CELL_LOOP
}
