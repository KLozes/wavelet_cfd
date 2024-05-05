#include <iostream>

#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"


void CompressibleSolver::sortFieldData(void) {
  sortFieldDataKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::initFieldData(u32 initType) {

  dataType *Rho  = getField(0);
  dataType *RhoU = getField(1);
  dataType *RhoV = getField(2);
  dataType *RhoE = getField(3);

  for (uint bIdx=0; bIdx < nBlocks; bIdx++) {
    u64 loc = zLocList[bIdx];
    i32 lvl, ib, jb;
    mortonDecode(loc, lvl, ib, jb);
    for (uint j = 0; j < blockSize; j++) {
      for (uint i = 0; i < blockSize; i++) {
        dataType pos[2];
        getCellPos(lvl, ib, jb, i, j, pos);
        u32 idx = i + j*blockSize + bIdx * blockSizeTot;
        State q = getInitCondition(initType, pos);
        Rho[idx] = q.rho;
        RhoU[idx] = q.rhoU;
        RhoV[idx] = q.rhoV;
        RhoE[idx] = q.rhoE;
      }
    }
  }
}

State CompressibleSolver::getInitCondition(u32 initType, dataType *pos) {
  if (initType == 0) {
    // sod shock explosion
    dataType centerX = domainSize[0]/2;
    dataType centerY = domainSize[1]/2;
    dataType radius = domainSize[0]/5;

    dataType dist = sqrt((pos[0]-centerX)*(pos[0]-centerX) + (pos[1]-centerY)*(pos[1]-centerY));
  
    // inside
    State q;
    if (dist < radius) {
      q.rho = 1.0;
      q.rhoU = 0.0;
      q.rhoV = 0.0;
      q.rhoE = 1.0/(gamma-1);
    }
    else {
      q.rho = 0.125;
      q.rhoU = 0.0;
      q.rhoV = 0.0;
      q.rhoE = 0.1/(gamma-1);
    }
    return q;
  }
  else {
    printf("ERROR: Ivalid initType: %d", initType);
    exit(0);
  }
}
