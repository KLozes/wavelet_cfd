#include <iostream>
#include <thrust/extrema.h>


#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"


void CompressibleSolver::initialize(void) {
  initializeBaseGrid();
  setInitialConditions();
  primitiveToConservative();
  sortBlocks();
  setBoundaryConditions();
  cudaDeviceSynchronize();
  printf("nblocks %d\n", nBlocks);
  paint();

  for(i32 lvl=0; lvl<1; lvl++){
    forwardWaveletTransform();
    adaptGrid();
    setInitialConditions();
    primitiveToConservative();
    setBoundaryConditions();
    sortBlocks();
    cudaDeviceSynchronize();
    printf("nblocks %d\n", nBlocks);
    paint();
  }
}

real CompressibleSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;

  while (t < tStep) {
    clock.tick();
    if (iter % 4 == 0 && nLvls > 1) {
      restrictFields();
      forwardWaveletTransform();
      adaptGrid();
      inverseWaveletTransform();
      sortBlocks();
      setBoundaryConditions();
    }
    cudaDeviceSynchronize();
    clock.tock();
    tGrid += clock.duration().count();

    clock.tick();
    computeDeltaT();
    for (i32 stage = 0; stage<3; stage++) {
      conservativeToPrimitive();
      computeRightHandSide();
      primitiveToConservative();
      updateFields(stage);
      setBoundaryConditions();

      if (nLvls > 1) {
        restrictFields();
        interpolateFields();
        setBoundaryConditions();
      }
    }
    cudaDeviceSynchronize();
    clock.tock();
    tSolver += clock.duration().count();

    tTotal += tSolver + tGrid;

    cudaDeviceSynchronize();
    t += deltaT;
    iter++;
  }

  return t;
}

void CompressibleSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<1000, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::setInitialConditions(void) {
  setInitialConditionsKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(void) {
  setBoundaryConditionsKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::conservativeToPrimitive(void) {
  conservativeToPrimitiveKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::primitiveToConservative(void) {
  primitiveToConservativeKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::forwardWaveletTransform(void) {
  cudaDeviceSynchronize();
  computeMagRhoUKernel<<<1000, cudaBlockSize>>>(*this); 
  maxRho = *(thrust::max_element(thrust::device, getField(0), getField(0)+nBlocks*blockSize));
  maxMagRhoU = *(thrust::max_element(thrust::device, getField(12), getField(12)+nBlocks*blockSize));
  maxRhoE = *(thrust::max_element(thrust::device, getField(3), getField(3)+nBlocks*blockSize));
  cudaDeviceSynchronize();
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(u32));
  copyToOldFieldsKernel<<<1000, cudaBlockSize>>>(*this); 
  forwardWaveletTransformKernel<<<1000, cudaBlockSize>>>(*this);
  waveletThresholdingKernel<<<1000, cudaBlockSize>>>(*this); 
}

void CompressibleSolver::inverseWaveletTransform(void) {
  inverseWaveletTransformKernel<<<1000, cudaBlockSize>>>(*this); 
}


void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
	deltaT = *(thrust::min_element(thrust::device, getField(12), getField(12)+nBlocks*blockSizeTot));
  deltaT *= cfl;
}

void CompressibleSolver::computeRightHandSide(void) {
  computeRightHandSideKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  updateFieldsRK3Kernel<<<1000, cudaBlockSize>>>(*this, stage);
}

void CompressibleSolver::restrictFields(void) {
  restrictFieldsKernel<<<1000, cudaBlockSize>>>(*this);
}

void CompressibleSolver::interpolateFields(void) {
  interpolateFieldsKernel<<<1000, cudaBlockSize>>>(*this);
}

__device__ real CompressibleSolver::lim(real &r) {
  // new TVD
  //return ((r > 0.0 && r < 1.0) ? (2.0*r + r*r*r) / (1.0 + 2.0*r*r) : r);

  // low dissipation tvd scheme
  real gam0 = 1100.0;
  real gam1 = 800.0;
  real lam1 = .15;
  real r4 = (r - .5)*(r - .5)*(r - .5)*(r - .5); // (r-.5) is a correction to the paper, which has (r-1)
  real w0 = 1.0 / ((1.0 + gam0*r4)*(1.0 + gam0*r4)); 
  real w1 = 1.0 / ((1.0 + gam1*r4)*(1.0 + gam1*r4));

  real u = r;
  real temp0 = 1.0/3.0 + 5.0/6.0*r;
  real temp1 = 2.0*r;
  real temp2 = lam1*r - lam1 + 1.0;
  if (0 < r && r <= 0.5){
    u = min(temp0 * w0 + temp1 * (1.0 - w0), temp1);
  }
  if ( 0.5 < r && r <= 1.0) {
    u = min(temp0 * w1 + temp2 * (1.0 - w1), temp2);
  }
  return u;
}

__device__ real CompressibleSolver::tvdRec(real &ul, real &uc, real &ur) {
  real r = (uc - ul) / (copysign(1.0, ur - ul)*fmaxf(abs(ur - ul), 1e-32));
  return ul + lim(r) * (ur - ul);
}

__device__ Vec4 CompressibleSolver::prim2cons(Vec4 prim) {
  Vec4 cons;
  cons[0] = prim[0];
  cons[1] = prim[1]*prim[0];
  cons[2] = prim[2]*prim[0];
  cons[3] = prim[3]/(gam-1.0) + .5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2]);
  return cons;
}

__device__ Vec4 CompressibleSolver::cons2prim(Vec4 cons) {
  Vec4 prim;
  prim[0] = cons[0];
  prim[1] = cons[1]/cons[0];
  prim[2] = cons[2]/cons[0];
  prim[3] = (gam-1.0)*(cons[3] - .5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2]));
  return prim;
}

__device__ real CompressibleSolver::getBoundaryLevelSet(Vec2 pos) {

  if (immerserdBcType == 1) {
    // circle
    real radius = .1;
    real center[2] = {1.0, .5};
    return radius - sqrt((pos[0]-center[0])*(pos[0]-center[0]) + (pos[1]-center[1])*(pos[1]-center[1]));
  }
  else {
    return 1e32;
  }

} 

__device__ Vec4 CompressibleSolver::hlleFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute HLLE flux
  //
  real nx = normal[0];
  real ny = normal[1];

  // Left state
  real rL = qL[0];
  real sqrL = sqrt(rL);
  real uL = qL[1]/qL[0];
  real vL = qL[2]/qL[0];
  real vnL = uL*nx+vL*ny;
  real eL = qL[3];
  real pL = (gam-1.0)*(eL - .5*rL*(uL*uL + vL*vL));
  real hL = (eL + pL)/rL;
  real aL = sqrt(abs(gam*pL/rL));

  // Right state
  real rR = qR[0];
  real sqrR = sqrt(rR);
  real uR = qR[1]/qR[0];
  real vR = qR[2]/qR[0];
  real vnR = uR*nx+vR*ny;
  real eR = qR[3];
  real pR = (gam-1.0)*(eR - .5*rR*(uR*uR + vR*vR));
  real hR = (eR + pR)/rR;
  real aR = sqrt(abs(gam*pR/rR));

  // Roe Averages
  real rSum = sqrL + sqrR;
  real u = (uL*sqrL + uR*sqrR) / rSum;
  real v = (vL*sqrL + vR*sqrR) / rSum;
  real a2 = (aL*aL*sqrL + aR*aR*sqrR) / rSum + .5*sqrL*sqrR/(rSum*rSum)*(vnR - vnL)*(vnR - vnL);
  real a = sqrt(a2);
  real vn = u*nx+v*ny;

  // Wave speed estimates
  real SL = fminf(fminf(vnL-aL, vn-a), 0);
  real SR = fmaxf(fmaxf(vnR+aR, vn+a), 0);

  // Left and Right fluxes
  real FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*hL};
  real FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*hR};

  // Compute the HLLE flux.
  Vec4 Flux((SR*FL[0] - SL*FR[0] + SL*SR*(qR[0] - qL[0]) )/(SR-SL), 
            (SR*FL[1] - SL*FR[1] + SL*SR*(qR[1] - qL[1]) )/(SR-SL),
            (SR*FL[2] - SL*FR[2] + SL*SR*(qR[2] - qL[2]) )/(SR-SL),
            (SR*FL[3] - SL*FR[3] + SL*SR*(qR[3] - qL[3]) )/(SR-SL));
  return Flux;
}


__device__ Vec4 CompressibleSolver::hllcFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute HLLC flux
  //
  real nx = normal[0];
  real ny = normal[1];

  // Left state
  real rL = qL[0];
  real sqrL = sqrt(rL);
  real uL = qL[1]/qL[0];
  real vL = qL[2]/qL[0];
  real vnL = uL*nx+vL*ny;
  real eL = qL[3];
  real pL = (gam-1.0)*(eL - .5*rL*(uL*uL + vL*vL));
  real hL = (eL + pL)/rL;
  real aL = sqrt(abs(gam*pL/rL));

  // Right state
  real rR = qR[0];
  real sqrR = sqrt(rR);
  real uR = qR[1]/qR[0];
  real vR = qR[2]/qR[0];
  real vnR = uR*nx+vR*ny;
  real eR = qR[3];
  real pR = (gam-1.0)*(eR - .5*rR*(uR*uR + vR*vR));
  real hR = (eR + pR)/rR;
  real aR = sqrt(abs(gam*pR/rR));

  // Roe Averages
  real rSum = sqrL + sqrR;
  real u = (uL*sqrL + uR*sqrR) / rSum;
  real v = (vL*sqrL + vR*sqrR) / rSum;
  real a2 = (aL*aL*sqrL + aR*aR*sqrR) / rSum + .5*sqrL*sqrR/(rSum*rSum)*(vnR - vnL)*(vnR - vnL);
  real a = sqrt(a2);
  real vn = u*nx+v*ny;

  // Wave speed estimates
  real SL = fminf(vnL-aL, vn-a);
  real SR = fmaxf(vnR+aR, vn+a);
  real SM = (pL-pR + rR*vnR*(SR-vnR) - rL*vnL*(SL-vnL))/(rR*(SR-vnR) - rL*(SL-vnL));

  // Left and Right fluxes
  real FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*hL};
  real FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*hR};

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
  real vtR = ny*uR + nx*vR;
  real vtL = ny*uL + nx*vL;
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
