#include <iostream>
#include <thrust/extrema.h>

#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"

void CompressibleSolver::initialize(i32 icType) {
  initializeBaseGrid();
  setInitialConditions(icType);
  primitiveToConservative();
  setBoundaryConditions(0);
  paint();

  for(i32 lvl=0; lvl<nLvls+3; lvl++){
    waveletThresholding();
    adaptGrid();
    setInitialConditions(icType);
    primitiveToConservative();
    setBoundaryConditions(0);
    sortBlocks();
    printf("nblocks %d\n", nBlocks);
    paint();
  }
}

dataType CompressibleSolver::step(dataType tStep) {

  dataType t = 0;

  while (t < tStep) {

    u32 nBlocksPrev = nBlocks;
    if (iter % 1 == 0) {
      waveletThresholding();
      adaptGrid();
      setBoundaryConditions(0);
      interpolateFields();
      paint();
      sortBlocks();
      computeDeltaT();
    }

    for (i32 stage = 0; stage<3; stage++) {
      conservativeToPrimitive();
      computeRightHandSide();
      primitiveToConservative();
      updateFields(stage);
      setBoundaryConditions(0);

      if (nLvls > 2) {
        interpolateFields();
      }
      
    }

    cudaDeviceSynchronize();
    t += deltaT;
    iter++;
  }

  return t;
}

void CompressibleSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::setInitialConditions(i32 icType) {
  setInitialConditionsKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this, icType);
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(i32 bcType) {
  setBoundaryConditionsKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this, bcType);
}

void CompressibleSolver::conservativeToPrimitive(void) {
  conservativeToPrimitiveKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::primitiveToConservative(void) {
  primitiveToConservativeKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::waveletThresholding(void) {
  computeMagRhoUKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this); 
  maxRho = *(thrust::min_element(thrust::device, getField(0), getField(0)+nBlocks*blockSize));
  maxMagRhoU = *(thrust::min_element(thrust::device, getField(12), getField(12)+nBlocks*blockSize));
  maxRhoE = *(thrust::min_element(thrust::device, getField(3), getField(3)+nBlocks*blockSize));
  cudaMemset(bFlagsList, 0, nBlocks*sizeof(u32));
  waveletThresholdingKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this); 
}

void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
	deltaT = *(thrust::min_element(thrust::device, getField(12), getField(12)+nBlocks*blockSize));
  deltaT *= cfl;
}

void CompressibleSolver::computeRightHandSide(void) {
  computeRightHandSideKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  updateFieldsRK3Kernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this, stage);
}

void CompressibleSolver::interpolateFields(void) {
  restrictFieldsKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
  interpolateFieldsKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

__host__ __device__ dataType CompressibleSolver::lim(dataType &r) {
  // new TVD
  //return ((r > 0.0 && r < 1.0) ? (2.0*r + r*r*r) / (1.0 + 2.0*r*r) : r);

  // low dissipation tvd scheme
  dataType gam0 = 1100.0;
  dataType gam1 = 800.0;
  dataType lam1 = .15;
  dataType r4 = (r - .5)*(r - .5)*(r - .5)*(r - .5); // (r-.5) is a correction to the paper, which has (r-1)
  dataType w0 = 1.0 / ((1.0 + gam0*r4)*(1.0 + gam0*r4)); 
  dataType w1 = 1.0 / ((1.0 + gam1*r4)*(1.0 + gam1*r4));

  dataType u = r;
  dataType temp0 = 1.0/3.0 + 5.0/6.0*r;
  dataType temp1 = 2.0*r;
  dataType temp2 = lam1*r - lam1 + 1.0;
  if (0 < r && r <= 0.5){
    u = min(temp0 * w0 + temp1 * (1.0 - w0), temp1);
  }
  if ( 0.5 < r && r <= 1.0) {
    u = min(temp0 * w1 + temp2 * (1.0 - w1), temp2);
  }
  return u;
}

__host__ __device__ dataType CompressibleSolver::tvdRec(dataType &ul, dataType &uc, dataType &ur) {
  dataType r = (uc - ul) / (copysign(1.0, ur - ul)*fmaxf(abs(ur - ul), 1e-32));
  return ul + lim(r) * (ur - ul);
}

__host__ __device__ Vec4 CompressibleSolver::prim2cons(Vec4 prim) {
  Vec4 cons;
  cons[0] = prim[0];
  cons[1] = prim[1]*prim[0];
  cons[2] = prim[2]*prim[0];
  cons[3] = prim[3]/(gam-1.0) + .5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2]);
  return cons;
}

__host__ __device__ Vec4 CompressibleSolver::cons2prim(Vec4 cons) {
  Vec4 prim;
  prim[0] = cons[0];
  prim[1] = cons[1]/cons[0];
  prim[2] = cons[2]/cons[0];
  prim[3] = (gam-1.0)*(cons[3] - .5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2]));
  return prim;
}

__host__ __device__ Vec4 CompressibleSolver::hlleFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute HLLE flux
  //
  dataType nx = normal[0];
  dataType ny = normal[1];

  // Left state
  dataType rL = qL[0];
  dataType sqrL = sqrt(rL);
  dataType uL = qL[1]/qL[0];
  dataType vL = qL[2]/qL[0];
  dataType vnL = uL*nx+vL*ny;
  dataType eL = qL[3];
  dataType pL = (gam-1.0)*(eL - .5*rL*(uL*uL + vL*vL));
  dataType hL = (eL + pL)/rL;
  dataType aL = sqrt(abs(gam*pL/rL));

  // Right state
  dataType rR = qR[0];
  dataType sqrR = sqrt(rR);
  dataType uR = qR[1]/qR[0];
  dataType vR = qR[2]/qR[0];
  dataType vnR = uR*nx+vR*ny;
  dataType eR = qR[3];
  dataType pR = (gam-1.0)*(eR - .5*rR*(uR*uR + vR*vR));
  dataType hR = (eR + pR)/rR;
  dataType aR = sqrt(abs(gam*pR/rR));

  // Roe Averages
  dataType rSum = sqrL + sqrR;
  dataType u = (uL*sqrL + uR*sqrR) / rSum;
  dataType v = (vL*sqrL + vR*sqrR) / rSum;
  dataType a2 = (aL*aL*sqrL + aR*aR*sqrR) / rSum + .5*sqrL*sqrR/(rSum*rSum)*(vnR - vnL)*(vnR - vnL);
  dataType a = sqrt(a2);
  dataType vn = u*nx+v*ny;

  // Wave speed estimates
  dataType SL = fminf(fminf(vnL-aL, vn-a), 0);
  dataType SR = fmaxf(fmaxf(vnR+aR, vn+a), 0);

  // Left and Right fluxes
  dataType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*hL};
  dataType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*hR};

  // Compute the HLLE flux.
  Vec4 Flux((SR*FL[0] - SL*FR[0] + SL*SR*(qR[0] - qL[0]) )/(SR-SL), 
            (SR*FL[1] - SL*FR[1] + SL*SR*(qR[1] - qL[1]) )/(SR-SL),
            (SR*FL[2] - SL*FR[2] + SL*SR*(qR[2] - qL[2]) )/(SR-SL),
            (SR*FL[3] - SL*FR[3] + SL*SR*(qR[3] - qL[3]) )/(SR-SL));
  return Flux;
}


__host__ __device__ Vec4 CompressibleSolver::hllcFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute HLLE flux
  //
  dataType nx = normal[0];
  dataType ny = normal[1];

  // Left state
  dataType rL = qL[0];
  dataType sqrL = sqrt(rL);
  dataType uL = qL[1]/qL[0];
  dataType vL = qL[2]/qL[0];
  dataType vnL = uL*nx+vL*ny;
  dataType eL = qL[3];
  dataType pL = (gam-1.0)*(eL - .5*rL*(uL*uL + vL*vL));
  dataType hL = (eL + pL)/rL;
  dataType aL = sqrt(abs(gam*pL/rL));

  // Right state
  dataType rR = qR[0];
  dataType sqrR = sqrt(rR);
  dataType uR = qR[1]/qR[0];
  dataType vR = qR[2]/qR[0];
  dataType vnR = uR*nx+vR*ny;
  dataType eR = qR[3];
  dataType pR = (gam-1.0)*(eR - .5*rR*(uR*uR + vR*vR));
  dataType hR = (eR + pR)/rR;
  dataType aR = sqrt(abs(gam*pR/rR));

  // Roe Averages
  dataType rSum = sqrL + sqrR;
  dataType u = (uL*sqrL + uR*sqrR) / rSum;
  dataType v = (vL*sqrL + vR*sqrR) / rSum;
  dataType a2 = (aL*aL*sqrL + aR*aR*sqrR) / rSum + .5*sqrL*sqrR/(rSum*rSum)*(vnR - vnL)*(vnR - vnL);
  dataType a = sqrt(a2);
  dataType vn = u*nx+v*ny;

  // Wave speed estimates
  dataType SL = fminf(vnL-aL, vn-a);
  dataType SR = fmaxf(vnR+aR, vn+a);
  dataType SM = (pL-pR + rR*vnR*(SR-vnR) - rL*vnL*(SL-vnL))/(rR*(SR-vnR) - rL*(SL-vnL));

  // Left and Right fluxes
  dataType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*hL};
  dataType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*hR};

  // Q star
  Vec4 qLS((SL-vnL)/(SL-SM) * rL, 
           (SL-vnL)/(SL-SM) * rL*(nx*SM + ny*uL),
           (SL-vnL)/(SL-SM) * rL*(ny*SM + nx*vL),
           (SL-vnL)/(SL-SM) * (eL + (SM-vnL)*(rL*SM + pL/(SL - vnL))));

  Vec4 qRS((SR-vnR)/(SR-SM) * rR, 
           (SR-vnR)/(SR-SM) * rR*(nx*SM + ny*uR),
           (SR-vnR)/(SR-SM) * rR*(ny*SM + nx*vR),
           (SR-vnR)/(SR-SM) * (eR + (SM-vnR)*(rR*SM + pR/(SR - vnR))));

  /*
  // hllm state
  aR = rR*(SR-vnR);
  aL = rL*(SL-vnL);
  dataType vtR = ny*uR + nx*vR;
  dataType vtL = ny*uL + nx*vL;
  Vec4 qLS(aL/(SL-SM) * 1, 
           aL/(SL-SM) * (nx*SM + ny*(aR*vtR - aL*vtL)/(aR-aL)),
           aL/(SL-SM) * (ny*SM + nx*(aR*vtR - aL*vtL)/(aR-aL)),
           aL/(SL-SM) * (eL/rL + (SM-vnL)*(SM + pL/aL) + .5*((aR*vtR*vtR - aL*vtL*vtL)/(aR-aL) - vtL*vtL)));

  Vec4 qRS(aR/(SR-SM) * 1, 
           aR/(SR-SM) * (nx*SM + ny*(aR*vtR - aL*vtL)/(aR-aL)),
           aR/(SR-SM) * (ny*SM + nx*(aR*vtR - aL*vtL)/(aR-aL)),
           aR/(SR-SM) * (eR/rR + (SM-vnR)*(SM + pR/aR) + .5*((aR*vtR*vtR - aL*vtL*vtL)/(aR-aL) - vtR*vtR)));
  */

  // Compute the HLLC flux.
  Vec4 Flux(.5*(FL[0]+FR[0]) - .5*(abs(SL)*(qLS[0]-qL[0]) + abs(SM)*(qRS[0]-qLS[0]) + abs(SR)*(qR[0]-qRS[0])), 
            .5*(FL[1]+FR[1]) - .5*(abs(SL)*(qLS[1]-qL[1]) + abs(SM)*(qRS[1]-qLS[1]) + abs(SR)*(qR[1]-qRS[1])),
            .5*(FL[2]+FR[2]) - .5*(abs(SL)*(qLS[2]-qL[2]) + abs(SM)*(qRS[2]-qLS[2]) + abs(SR)*(qR[2]-qRS[2])),
            .5*(FL[3]+FR[3]) - .5*(abs(SL)*(qLS[3]-qL[3]) + abs(SM)*(qRS[3]-qLS[3]) + abs(SR)*(qR[3]-qRS[3])));
  return Flux;
}
