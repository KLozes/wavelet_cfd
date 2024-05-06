#include <iostream>
#include <thrust/extrema.h>

#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"


void CompressibleSolver::sortFieldData(void) {
  sortFieldDataKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::setInitialConditions(i32 icType) {
  setInitialConditionsKernel<<<nBlocks*blockSize/cudaBlockSize+1, cudaBlockSize>>>(*this, icType);
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(i32 bcType) {
  setBoundaryConditionsKernel<<<nBlocks*blockSize/cudaBlockSize+1, cudaBlockSize>>>(*this, bcType);
}

void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<nBlocks*blockSize/cudaBlockSize+1, cudaBlockSize>>>(*this);
	deltaT = *(thrust::min_element(thrust::device, getField(8), getField(8)+nBlocks*blockSize));
}

void CompressibleSolver::computeRightHandSide(void) {
  computeRightHandSideKernel<<<nBlocks*blockSize/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  updateFieldsKernel<<<nBlocks*blockSize/cudaBlockSize+1, cudaBlockSize>>>(*this, stage);
}


__host__ __device__ dataType CompressibleSolver::lim(dataType r) {
  return ((r > 0.0 && r < 1.0) ? (2.0*r + r*r*r) / (1.0 + 2.0*r*r) : r);
}

__host__ __device__ dataType CompressibleSolver::tvdRec(dataType ul, dataType uc, dataType ur) {
  dataType r = (uc - ul) / (copysign(1.0, ur - ul)*fmaxf(abs(ur - ul), 1e-32));
  return ul + lim(r) * (ur - ul);
}

__host__ __device__ Vec4 CompressibleSolver::centralFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute Central KEEP flux
  //
  dataType nx, ny, rL, uL, vL, vnL, pL, aL, HL, rR, uR, vR, vnR, pR, aR, HR, RT, u, v, H, a, vn, SLm, SRp;
  nx = normal[0];
  ny = normal[1];

  // Left state
  rL = qL[0];
  uL = qL[1]/rL;
  vL = qL[2]/rL;
  vnL = uL*nx+vL*ny;
  pL = (gam-1.0)*( qL[3] - rL*(uL*uL+vL*vL)/2.0 );
  aL = sqrt(abs(gam*pL/rL));
  HL = ( qL[3] + pL ) / rL;

  // Right state
  rR = qR[0];
  uR = qR[1]/rR;
  vR = qR[2]/rR;
  vnR = uR*nx+vR*ny;
  pR = (gam-1)*( qR[3] - rR*(uR*uR+vR*vR)/2.0 );
  aR = sqrt(abs(gam*pR/rR));
  HR = ( qR[3] + pR ) / rR;

  // irst compute the Roe Averages
  RT = sqrt(rR/rL); // r = RT*rL;
  u = (uL+RT*uR)/(1.0+RT);
  v = (vL+RT*vR)/(1.0+RT);
  H = ( HL+RT* HR)/(1.0+RT);
  a = sqrt( abs((gam-1.0)*(H-(u*u+v*v)/2.0)) );
  vn = u*nx+v*ny;

  // Wave speed estimates
  SLm = fminf(fminf(vnL-aL, vn-a), 0);
  SRp = fmaxf(fmaxf(vnR+aR, vn+a), 0);

  // Left and Right fluxes
  dataType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL};
  dataType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR};

  // Compute the HLLE flux.
  Vec4 Flux((SRp*FL[0] - SLm*FR[0] + SLm*SRp*(qR[0]-qL[0]) )/(SRp-SLm), 
            (SRp*FL[1] - SLm*FR[1] + SLm*SRp*(qR[1]-qL[1]) )/(SRp-SLm),
            (SRp*FL[2] - SLm*FR[2] + SLm*SRp*(qR[2]-qL[2]) )/(SRp-SLm),
            (SRp*FL[3] - SLm*FR[3] + SLm*SRp*(qR[3]-qL[3]) )/(SRp-SLm));
  return Flux;
}

__host__ __device__ Vec4 CompressibleSolver::hlleFlux(Vec4 qL, Vec4 qR, Vec2 normal) {
  //
  // Compute HLLE flux
  //
  dataType nx, ny, rL, uL, vL, vnL, pL, aL, HL, rR, uR, vR, vnR, pR, aR, HR, RT, u, v, H, a, vn, SLm, SRp;
  nx = normal[0];
  ny = normal[1];

  // Left state
  rL = qL[0];
  uL = qL[1]/rL;
  vL = qL[2]/rL;
  vnL = uL*nx+vL*ny;
  pL = (gam-1.0)*( qL[3] - rL*(uL*uL+vL*vL)/2.0 );
  aL = sqrt(abs(gam*pL/rL));
  HL = ( qL[3] + pL ) / rL;

  // Right state
  rR = qR[0];
  uR = qR[1]/rR;
  vR = qR[2]/rR;
  vnR = uR*nx+vR*ny;
  pR = (gam-1)*( qR[3] - rR*(uR*uR+vR*vR)/2.0 );
  aR = sqrt(abs(gam*pR/rR));
  HR = ( qR[3] + pR ) / rR;

  // irst compute the Roe Averages
  RT = sqrt(rR/rL); // r = RT*rL;
  u = (uL+RT*uR)/(1.0+RT);
  v = (vL+RT*vR)/(1.0+RT);
  H = ( HL+RT* HR)/(1.0+RT);
  a = sqrt( abs((gam-1.0)*(H-(u*u+v*v)/2.0)) );
  vn = u*nx+v*ny;

  // Wave speed estimates
  SLm = fminf(fminf(vnL-aL, vn-a), 0);
  SRp = fmaxf(fmaxf(vnR+aR, vn+a), 0);

  // Left and Right fluxes
  dataType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL};
  dataType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR};

  // Compute the HLLE flux.
  Vec4 Flux((SRp*FL[0] - SLm*FR[0] + SLm*SRp*(qR[0]-qL[0]) )/(SRp-SLm), 
            (SRp*FL[1] - SLm*FR[1] + SLm*SRp*(qR[1]-qL[1]) )/(SRp-SLm),
            (SRp*FL[2] - SLm*FR[2] + SLm*SRp*(qR[2]-qL[2]) )/(SRp-SLm),
            (SRp*FL[3] - SLm*FR[3] + SLm*SRp*(qR[3]-qL[3]) )/(SRp-SLm));
  return Flux;
}
