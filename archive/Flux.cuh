#ifndef FLUX_H
#define FLUX_H

__device__ dType SuperBeeLim(dType r)
{
	return fmaxf(0.0f, fmaxf(fminf(2.0f*r,1.0f), fminf(r,2.0f)));
}

__device__ dType McLim(dType r)
{
	return fmaxf(0.0f, fminf(2.0f*r, fminf(.5*(r+1), 2.0f)));
}

__device__ dType MinModLim(dType r)
{
	return fmaxf(0.0f, fminf(1,r));
}

__device__ void HLLEflux(const dType qL[4], const dType qR[4], dType* Fn, const dType normal[2])
{
    //Compute HLLE flux
		dType gamma = 1.4;

		dType nx, ny, rL, uL, vL, vnL, pL, aL, HL, rR, uR, vR, vnR, pR, aR, HR, RT, u, v, H, a, vn, SLm, SRp;
    // normal vectors
    nx = normal[0];
    ny = normal[1];

    // Left state
    rL = qL[0];
    uL = qL[1]/rL;
    vL = qL[2]/rL;
    vnL = uL*nx+vL*ny;
    pL = (gamma-1.0)*( qL[3] - rL*(uL*uL+vL*vL)/2.0 );
    aL = sqrt(abs(gamma*pL/rL));
    HL = ( qL[3] + pL ) / rL;

    // Right state
    rR = qR[0];
    uR = qR[1]/rR;
    vR = qR[2]/rR;
    vnR = uR*nx+vR*ny;
    pR = (gamma-1)*( qR[3] - rR*(uR*uR+vR*vR)/2.0 );
    aR = sqrt(abs(gamma*pR/rR));
    HR = ( qR[3] + pR ) / rR;

    // irst compute the Roe Averages
    RT = sqrt(rR/rL); // r = RT*rL;
    u = (uL+RT*uR)/(1.0+RT);
    v = (vL+RT*vR)/(1.0+RT);
    H = ( HL+RT* HR)/(1.0+RT);
    a = sqrt( abs((gamma-1.0)*(H-(u*u+v*v)/2.0)) );
    vn = u*nx+v*ny;

    // Wave speed estimates
    SLm = fminf(fminf(vnL-aL, vn-a), 0);
    SRp = fmaxf(fmaxf(vnR+aR, vn+a), 0);

    // Left and Right fluxes
    dType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL};
    dType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR};

    // Compute the HLL flux.
    Fn[0] = ( SRp*FL[0] - SLm*FR[0] + SLm*SRp*(qR[0]-qL[0]) )/(SRp-SLm);
		Fn[1] = ( SRp*FL[1] - SLm*FR[1] + SLm*SRp*(qR[1]-qL[1]) )/(SRp-SLm);
		Fn[2] = ( SRp*FL[2] - SLm*FR[2] + SLm*SRp*(qR[2]-qL[2]) )/(SRp-SLm);
		Fn[3] = ( SRp*FL[3] - SLm*FR[3] + SLm*SRp*(qR[3]-qL[3]) )/(SRp-SLm);
}


__device__ void HLLCflux(const dType qL[4], const dType qR[4], dType* Fn, const dType normal[2])
{
	//Compute HLLC flux
	dType gamma = 1.4;

	dType nx, ny, rL, uL, vL, vnL, pL, aL, HL, rR, uR, vR, vnR, pR, aR, HR, PPV, pmin, pmax, Qmax, Quser, pM, PQ, uM, PTL, PTR, GEL, GER, zL, zR, SL, SR, SM;
	// normal vectors
	nx = normal[0];
	ny = normal[1];

	// Left state
	rL = qL[0];
	uL = qL[1]/rL;
	vL = qL[2]/rL;
	vnL = uL*nx+vL*ny;
	pL = (gamma-1.0)*( qL[3] - rL*(uL*uL+vL*vL)/2.0 );
	aL = sqrtf(gamma*pL/rL);
	HL = ( qL[3] + pL ) / rL;

	// Right state
	rR = qR[0];
	uR = qR[1]/rR;
	vR = qR[2]/rR;
	vnR = uR*nx+vR*ny;
	pR = (gamma-1.0)*( qR[3] - rR*(uR*uR+vR*vR)/2.0 );
	aR = sqrtf(gamma*pR/rR);
	HR = ( qR[3] + pR ) / rR;

  // Left and Right fluxes
  dType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL};
  dType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR};

  // Compute guess pressure from PVRS Riemann solver
  PPV  = fmaxf(0 , 0.5*(pL+pR) + 0.5*(vnL-vnR) * (0.25*(rL+rR)*(aL+aR)));
  pmin = fminf(pL,pR);
  pmax = fmaxf(pL,pR);
  Qmax = pmax/pmin;
  Quser= 2.0; // <--- parameter manually set (I don't like this!)

  if((Qmax <= Quser) && (pmin <= PPV) && (PPV <= pmax)){
   	// Select PRVS Riemann solver
    pM = PPV;
	}
  else{
  	if(PPV < pmin){
    	// Select Two-Rarefaction Riemann solver
    	PQ  = powf(pL/pR,(gamma - 1.0))/(2.0*gamma);
      uM  = (PQ*vnL/aL + vnR/aR + 2.0/(gamma-1.0)*(PQ-1.0))/(PQ/aL+1.0/aR);
      PTL = 1.0 + (gamma-1.0)/2.0*(vnL - uM)/aL;
      PTR = 1.0 + (gamma-1.0)/2.0*(uM - vnR)/aR;
      pM  = 0.5*(pL*powf(PTL,(2.0*gamma/(gamma-1.0))) + pR*powf(PTR,(2.0*gamma/(gamma-1.0))));
		}
    else{
    	// Use Two-Shock Riemann solver with PVRS as estimate
      GEL = sqrtf((2.0/(gamma+1.0)/rL)/((gamma-1.0)/(gamma+1.0)*pL + PPV));
      GER = sqrtf((2.0/(gamma+1.0)/rR)/((gamma-1.0)/(gamma+1.0)*pR + PPV));
      pM  = (GEL*pL + GER*pR - (vnR - vnL))/(GEL + GER);
		}
	}

  // Estimate wave speeds: SL, SR and SM (Toro, 1994)
  zL=1.0;
  zR=1.0;
  if(pM>pL){zL=sqrtf(1.0+(gamma+1.0)/(2.0*gamma)*(pM/pL-1.0));}
  if(pM>pR){zR=sqrtf(1.0+(gamma+1.0)/(2.0*gamma)*(pM/pR-1.0));}

	SL = vnL - aL*zL;
  SR = vnR + aR*zR;
  SM = (pL-pR + rR*vnR*(SR-vnR) - rL*vnL*(SL-vnL))/(rR*(SR-vnR) - rL*(SL-vnL));

  dType qsL[4], qsR[4];
  // Compute the HLL flux.
  if(0 <= SL){  // Right-going supersonic flow
    Fn[0] = FL[0];
    Fn[1] = FL[1];
    Fn[2] = FL[2];
    Fn[3] = FL[3];
	}
  else if((SL <= 0) && (0 <= SM)){	// Subsonic flow to the right
    qsL[0] = rL*(SL-vnL)/(SL-SM);
    qsL[1] = rL*(SL-vnL)/(SL-SM)*(SM*nx+uL*fabsf(ny));
    qsL[2] = rL*(SL-vnL)/(SL-SM)*(SM*ny+vL*fabsf(nx));
    qsL[3] = rL*(SL-vnL)/(SL-SM)*(qL[3]/rL + (SM-vnL)*(SM+pL/(rL*(SL-vnL))));
    Fn[0] = FL[0] + SL*(qsL[0] - qL[0]);
    Fn[1] = FL[1] + SL*(qsL[1] - qL[1]);
    Fn[2] = FL[2] + SL*(qsL[2] - qL[2]);
    Fn[3] = FL[3] + SL*(qsL[3] - qL[3]);
	}
  else if ((SM <= 0) && (0 <= SR)){	// Subsonic flow to the Left
    qsR[0] = rR*(SR-vnR)/(SR-SM);
    qsR[1] = rR*(SR-vnR)/(SR-SM)*(SM*nx+uR*fabsf(ny));
    qsR[2] = rR*(SR-vnR)/(SR-SM)*(SM*ny+vR*fabsf(nx));
    qsR[3] = rR*(SR-vnR)/(SR-SM)*(qR[3]/rR + (SM-vnR)*(SM+pR/(rR*(SR-vnR))));
    Fn[0] = FR[0] + SR*(qsR[0] - qR[0]);
    Fn[1] = FR[1] + SR*(qsR[1] - qR[1]);
    Fn[2] = FR[2] + SR*(qsR[2] - qR[2]);
    Fn[3] = FR[3] + SR*(qsR[3] - qR[3]);
	}
  else if(0 >= SR){ // Left-going supersonic flow
  	Fn[0] = FR[0];
    Fn[1] = FR[1];
    Fn[2] = FR[2];
    Fn[3] = FR[3];
	}
}


__device__ void ROEflux(const dType qL[4], const dType qR[4], dType* Fn, const dType normal[2])
{
	// Compute Roe flux
	dType dws[4];
	dType nx, ny, tx, ty, rL, uL, vL, vnL, vtL, pL, HL, rR, uR, vR, vnR, vtR, pR, HR, RT, r, u, v, H, a, vn, vt, dr, dp, dvn, dvt;

	dType gamma = 1.4f;
	// normal vectors
	nx = normal[0];
	ny = normal[1];

	// Tangent vectors
	tx = -ny;
	ty = nx;

	// Left state
	rL = qL[0];
	uL = qL[1]/rL;
	vL = qL[2]/rL;
	vnL = uL*nx+vL*ny;
	vtL = uL*tx+vL*ty;
	pL = (gamma-1.0f)*(qL[3] - rL*(uL*uL+vL*vL)/2.0f );
	//aL = sqrt(gamma*pL/rL);
	HL = ( qL[3] + pL ) / rL;

	// Right state
	rR = qR[0];
	uR = qR[1]/rR;
	vR = qR[2]/rR;
	vnR = uR*nx+vR*ny;
	vtR = uR*tx+vR*ty;
	pR = (gamma-1.0)*( qR[3] - rR*(uR*uR+vR*vR)/2.0f );
	//aR = sqrt(gamma*pR/rR);
	HR = (qR[3] + pR) / rR;

	// First compute the Roe Averages
	RT = sqrtf(rR/rL);
	r = RT*rL;
	u = (uL+RT*uR)/(1.0f+RT);
	v = (vL+RT*vR)/(1.0f+RT);
	H = ( HL+RT* HR)/(1.0f+RT);
	a = sqrtf( (gamma-1.0f)*(H-(u*u+v*v)/2.0f) );
	vn = u*nx+v*ny;
	vt = u*tx+v*ty;

	// Wave Strengths
	dr = rR - rL;
  dp = pR - pL;
  dvn= vnR - vnL;
  dvt= vtR - vtL;
	dType dV[4] = {(dp-r*a*dvn )/(2.0f*a*a), r*dvt/a, dr-dp/(a*a), (dp+r*a*dvn)/(2.0f*a*a)};

	// Wave Speed
	dType ws[4] = {fabsf(vn-a), fabsf(vn), fabsf(vn), fabsf(vn+a)};

	// Harten's Entropy Fix JCP(1983), 49, pp357-393:
 	// only for the nonlinear fields.
	dws[0]=0.2f; if(ws[0]<dws[0]){ws[0] = (ws[0]*ws[0]/dws[0]+dws[0] )/2.0f;}
	dws[3]=0.2f; if(ws[3]<dws[3]){ws[3] = (ws[3]*ws[3]/dws[3]+dws[3] )/2.0f;}

	// Right Eigenvectors
	dType Rv[4][4] = {{1.0f   ,  0.0f   ,    1.0f      ,  1.0f},
				{u-a*nx, a*tx ,    u      ,u+a*nx},
				{u-a*ny, a*ty ,    u      ,u+a*ny},
				{H-vn*a, vt*a , (u*u+v*v)/2.0f, H+vn*a}};

	// Left and Right fluxes
	dType FL[4] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*HL};
	dType FR[4] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*HR};

	// Dissipation Term
	Fn[0] = (FL[0] + FR[0] - (Rv[0][0]*(ws[0]*dV[0]) + Rv[0][1]*(ws[1]*dV[1]) \
														+ Rv[0][2]*(ws[2]*dV[2]) + Rv[0][3]*(ws[3]*dV[3])))/2.0f;

	Fn[1] = (FL[1] + FR[1] - (Rv[1][0]*(ws[0]*dV[0]) + Rv[1][1]*(ws[1]*dV[1]) \
														+ Rv[1][2]*(ws[2]*dV[2]) + Rv[1][3]*(ws[3]*dV[3])))/2.0f;

	Fn[2] = (FL[2] + FR[2] - (Rv[2][0]*(ws[0]*dV[0]) + Rv[2][1]*(ws[1]*dV[1]) \
														+ Rv[2][2]*(ws[2]*dV[2]) + Rv[2][3]*(ws[3]*dV[3])))/2.0f;

	Fn[3] = (FL[3] + FR[3] - (Rv[3][0]*(ws[0]*dV[0]) + Rv[3][1]*(ws[1]*dV[1]) \
														+ Rv[3][2]*(ws[2]*dV[2]) + Rv[3][3]*(ws[3]*dV[3])))/2.0f;

}


#endif
