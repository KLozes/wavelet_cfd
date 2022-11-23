#ifndef CNS_KERNELS_H
#define CNS_KERNELS_H

#include <string>
#include "Global.cuh"
//#include "Flux.cuh"

enum
{
  RHO = 0,
  RHOU = 1,
  RHOV = 2,
  E = 3,
  OLD_RHO = 4,
  OLD_RHOU = 5,
  OLD_RHOV = 5,
  OLD_E = 6
};

__global__ void sortFieldData2D(MultiLevelSparseGrid &grid)
{
  START_CELL_LOOP

  BlockType &block grid.blockList[bIdx]
  u32 old_bIndex = block.index;

  Array &rhou = grid.getFieldArray(bIdx, RHOU);
  Array &rhov = grid.getFieldArray(bIdx, RHOV);

  Array &old_rhou = grid.getFieldArray(bIdx, RHOU);
  Array &old_rhov = grid.getFieldArray(bIdx, RHOV);

  rho(bIdx) = old_rho(old_bIndex);
  rhou(bIdx) = old_rhou(old_bIndex);
  rhov(bIdx) = old_rhov(old_bIndex);
  e(bIdx) = old_e(old_bIndex);

  END_CELL_LOOP
}

/*
// compute max wavespeed in each cell, will be used for CFL condition
__global__ void calcWaveSpeed(Array<dType> &Q, Array<dType> &WS)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	dType p, r, a, u, v, vel;
	dType gamma = 1.4;
	if(i < Q.len_j && j < Q.len_k){
		r = Q(0,i,j);
		u = Q(1,i,j)/r;
		v = Q(2,i,j)/r;
		p = (gamma-1.0)*( Q(3,i,j) - r*(u*u+v*v)/2.0 );
		a = sqrt(abs(gamma*p/r));
		vel = sqrt(u*u + v*v);
		WS(i,j) = a + vel;
	}

}

// copy data from 1 array to the other
__global__ void GPUcopy(Array<dType> &Q, Array<dType> &Qold )
{

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if(i < Q.len_j && j < Q.len_k){
		Qold(0,i,j) = Q(0,i,j);
		Qold(1,i,j) = Q(1,i,j);
		Qold(2,i,j) = Q(2,i,j);
		Qold(3,i,j) = Q(3,i,j);
	}

}


__global__ void SetIC(Array<dType> &Q, dType W, dType H, dType hX, dType hY)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	dType gamma = 1.4;

	dType rIn = 10;
	dType uIn = 0.0;
	dType vIn = 0.0;
	dType pIn = 10;

	dType rOut = 0.125;
	dType uOut = 0.0;
	dType vOut = 0.0;
	dType pOut = 0.1;

	//dType rb = 1;
	//dType ub = 0.0;
	//dType vb = 0.0;
	//dType pb = 0.1;

	dType x = i*hX + .5*hX;
	dType y = j*hY + .5*hY;

	dType dist = sqrtf((x-W*.5)*(x-W*.5) + (y-H*.5)*(y-H*.5));
	dType dist1 = sqrtf((x-W*.15)*(x-W*.15) + (y-H*.15)*(y-H*.15));
	dType dist2 = sqrtf((x-W*.15)*(x-W*.15) + (y-H*.85)*(y-H*.85));
	dType dist3 = sqrtf((x-W*.85)*(x-W*.85) + (y-H*.15)*(y-H*.15));
	dType dist5 = sqrtf((x-W*.15)*(x-W*.15) + (y-H*.5)*(y-H*.5));
	dType dist4 = sqrtf((x-W*.85)*(x-W*.85) + (y-H*.85)*(y-H*.85));
	dType dist6 = sqrtf((x-W*.5)*(x-W*.5) + (y-H*.85)*(y-H*.85));
	dType dist7 = sqrtf((x-W*.85)*(x-W*.85) + (y-H*.5)*(y-H*.5));
	dType dist8 = sqrtf((x-W*.5)*(x-W*.5) + (y-H*.15)*(y-H*.15));

	if(i < Q.len_j && j < Q.len_k){
		if(dist < .2){
			Q(0,i,j) = rIn;
			Q(1,i,j) = rIn*uIn;
			Q(2,i,j) = rIn*vIn;
			Q(3,i,j) = rIn*(pIn/((gamma-1)*rIn) + 0.5*(uIn*uIn + vIn*vIn));
		}


		//else if( dist5 <= .1 || dist6 <= .1 || dist7 <= .1 || dist8 <= .1){
		//	Q(0,i,j) = rb;
		//	Q(1,i,j) = rb*ub;
		//	Q(2,i,j) = rb*vb;
		//	Q(3,i,j) = rb*(pb/((gamma-1)*rb) + 0.5*(ub*ub + vb*vb));
		//}

		else{
			Q(0,i,j) = rOut;
			Q(1,i,j) = rOut*uOut;
			Q(2,i,j) = rOut*vOut;
			Q(3,i,j) = rOut*(pOut/((gamma-1)*rOut) + 0.5*(uOut*uOut + vOut*vOut));

		}

	}

}


__global__ void CalcFluxX(Array<dType> &Q, Array<dType> &Fx)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x+1;
	int j = threadIdx.y + blockIdx.y*blockDim.y+1;
	int f;
	dType qL[4], qR[4], rL, rR;
	dType w = .8;

	if(i > 1 && i < Fx.len_j-2 && j > 0 && j < Fx.len_k-1 ){
		for(f=0; f<4; f++){
			rL = (Q(f,i,j) - Q(f,i-1,j))/(Q(f,i-1,j) - Q(f,i-2,j));
			rR = (Q(f,i,j) - Q(f,i-1,j))/(Q(f,i+1,j) - Q(f,i,j));
			qL[f] = Q(f,i-1,j) + 0.5f*McLim(rL) * (Q(f,i-1,j) - Q(f,i-2,j));
			qR[f] = Q(f,i,j)   - 0.5f*McLim(rR) * (Q(f,i+1,j) - Q(f,i,j));
		}

		dType normal[2] = {1.0f,0.0f};
		HLLCflux(qL, qR, Fx.getPtr(0,i,j), normal);
	}

}

__global__ void CalcFluxY(Array<dType> &Q, Array<dType> &Fy)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x+1;
	int j = threadIdx.y + blockIdx.y*blockDim.y+1;
	int f;
	dType qL[4], qR[4], rL, rR;

	if(i > 0 && i < Fy.len_j-1 && j > 1 && j < Fy.len_k-2 ){
		for(f=0; f<4; f++){
			rL = (Q(f,i,j) - Q(f,i,j-1))/(Q(f,i,j-1) - Q(f,i,j-2));
			rR = (Q(f,i,j) - Q(f,i,j-1))/(Q(f,i,j+1) - Q(f,i,j));
			qL[f] = Q(f,i,j-1) + 0.5f*McLim(rL) * (Q(f,i,j-1) - Q(f,i,j-2));
			qR[f] = Q(f,i,j)   - 0.5f*McLim(rR) * (Q(f,i,j+1) - Q(f,i,j));
		}

		dType normal[2] = {0.0f,1.0f};
		HLLCflux(qL, qR, Fy.getPtr(0,i,j), normal);
	}

}


__global__ void Update1(Array<dType> &Q, Array<dType> &Fx, Array<dType> &Fy, dType hX, dType hY, dType dt)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x+1;
	int j = threadIdx.y + blockIdx.y*blockDim.y+1;
	int f;

	if(i > 0 && j > 0 && i < Q.len_j-1 && j < Q.len_k-1){  //interior cells
		for(f=0; f<4; f++){
			Q(f,i,j) = Q(f,i,j) - dt*((Fx(f,i+1,j) - Fx(f,i,j))/hX + (Fy(f,i,j+1) - Fy(f,i,j))/hY);
		}

	}

}

__global__ void Update2(Array<dType> &Q, Array<dType> &Qold, Array<dType> &Fx, Array<dType> &Fy, dType hX, dType hY, dType dt)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x+1;
	int j = threadIdx.y + blockIdx.y*blockDim.y+1;
	int f;

	if(i > 0 && j > 0 && i < Q.len_j-1 && j < Q.len_k-1){  //interior cells
		for(f=0; f<4; f++){
			Q(f,i,j) = 0.5f*(Q(f,i,j) + Qold(f,i,j) - dt*((Fx(f,i+1,j) - Fx(f,i,j))/hX + (Fy(f,i,j+1) - Fy(f,i,j))/hY));
		}

	}

}


__global__ void ApplyBCs(Array<dType> &Q, Array<dType> &Fx, Array<dType> &Fy, int nX, int nY)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int side = blockIdx.y;
	int i;
	int j;

	dType QL[4], QR[4];

	if(side == 0)
	{
		i = 0;
		if(idx > 0 && idx < nY+1){
			Q(0,i,idx) = Q(0,i+1,idx);
			Q(1,i,idx) = -Q(1,i+1,idx);
			Q(2,i,idx) = Q(2,i+1,idx);
			Q(3,i,idx) = Q(3,i+1,idx);

			QL[0] = Q(0,i,idx);
			QL[1] = Q(1,i,idx);
			QL[2] = Q(2,i,idx);
			QL[3] = Q(3,i,idx);

			QR[0] = Q(0,i+1,idx);
			QR[1] = Q(1,i+1,idx);
			QR[2] = Q(2,i+1,idx);
			QR[3] = Q(3,i+1,idx);

			dType normal[2] = {1.0,0.0};
			HLLEflux(QL, QR, Fx.getPtr(0,i+1,idx), normal);

		}
	}
	if(side == 1)
	{
		i =  nX+1;
		if(idx > 0 && idx < nY+1){
			Q(0,i,idx) = Q(0,i-1,idx);
			Q(1,i,idx) = -Q(1,i-1,idx);
			Q(2,i,idx) = Q(2,i-1,idx);
			Q(3,i,idx) = Q(3,i-1,idx);

			QL[0] = Q(0,i-1,idx);
			QL[1] = Q(1,i-1,idx);
			QL[2] = Q(2,i-1,idx);
			QL[3] = Q(3,i-1,idx);

			QR[0] = Q(0,i,idx);
			QR[1] = Q(1,i,idx);
			QR[2] = Q(2,i,idx);
			QR[3] = Q(3,i,idx);

			dType normal[2] = {1.0,0.0};
			HLLEflux(QL, QR, Fx.getPtr(0,i,idx), normal);

		}
	}
	if(side == 2)
	{
		j = 0;
		if(idx > 0 && idx < nX+1){
			Q(0,idx,j) = Q(0,idx,j+1);
			Q(1,idx,j) = Q(1,idx,j+1);
			Q(2,idx,j) = -Q(2,idx,j+1);
			Q(3,idx,j) = Q(3,idx,j+1);


			QL[0] = Q(0,idx,j);
			QL[1] = Q(1,idx,j);
			QL[2] = Q(2,idx,j);
			QL[3] = Q(3,idx,j);

			QR[0] = Q(0,idx,j+1);
			QR[1] = Q(1,idx,j+1);
			QR[2] = Q(2,idx,j+1);
			QR[3] = Q(3,idx,j+1);

			dType normal[2] = {0.0,1.0};
			HLLEflux(QL, QR, Fy.getPtr(0,idx,j+1), normal);
		}
	}
	if(side == 3)
	{
		j = nY+1;
		if(idx > 0 && idx < nX+1){
			Q(0,idx,j) = Q(0,idx,j-1);
			Q(1,idx,j) = Q(1,idx,j-1);
			Q(2,idx,j) = -Q(2,idx,j-1);
			Q(3,idx,j) = Q(3,idx,j-1);

			QL[0] = Q(0,idx,j-1);
			QL[1] = Q(1,idx,j-1);
			QL[2] = Q(2,idx,j-1);
			QL[3] = Q(3,idx,j-1);

			QR[0] = Q(0,idx,j);
			QR[1] = Q(1,idx,j);
			QR[2] = Q(2,idx,j);
			QR[3] = Q(3,idx,j);

			dType normal[2] = {0.0,1.0};

			HLLEflux(QL, QR, Fy.getPtr(0,idx,j), normal);

		}
	}

}

__global__ void copyToImData(Array<dType> &Q, Array<dType> &ImData, int f)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
  if(i > 0 && j > 0 && i < ImData.len_i && j < ImData.len_j){  //interior cells
    ImData(i,j) = Q(f,i+1,j+1);
  }
}

__global__ void normalizeImData(Array<dType> &ImData, dType maxImData, dType minImData)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

  dType delta = maxImData - minImData;
  if(i > 0 && j > 0 && i < ImData.len_i && j < ImData.len_j){  //interior cells
    ImData(i,j) = ((ImData(i,j)-minImData)/delta)*65535;     // normalize 0to1 then multiply by max 16 bit u32
  }

}
*/

#endif
