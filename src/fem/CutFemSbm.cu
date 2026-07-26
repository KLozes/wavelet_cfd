//
// Shifted-boundary (GSBM) path of the CutFEM solver -- CutFemSolver::runSbm().
//
// Production realization of the method verified standalone in SbmMms.cu /
// SbmSolve.h, sitting ALONGSIDE the Saye cut-cell path (runQp): same octree +
// oracle (buildMesh), same Qp node lattice and gather/scatter, but
//
//   * the domain is the SURROGATE Omega~ = cells whose CENTRE is inside
//     (phi < 0): every element is FULL -- no cut quadrature, no Saye, no
//     interface polynomial;
//   * boundary conditions are imposed on the surrogate boundary Gamma~ (mesh
//     faces between a surrogate and a non-surrogate cell) by the SHIFTED
//     Nitsche of GSBM Eq. (35) (Colomes, Modderman, Scovazzi, CMAME 452 (2026)
//     118748): plain-v consistency, -adjoint with the Taylor shift S_d u, and
//     the beta1/h penalty on S_d u, with d, nu from the oracle;
//   * the ghost penalty (l = 1..p) acts on interior faces of the boundary BAND
//     (elements touching Gamma~).  It is ESSENTIAL at p >= 2: the shift
//     activates derivatives up to order p and without their jump control the
//     operator is indefinite (verified in M1).
//
// Parameters (verified in SbmMms):  beta1 = 20 (p+1)^2 (2mu+lam) / h,
// beta2 = 0.5 (2mu+lam).  The operator is NON-symmetric (plain-v consistency
// vs shifted adjoint) -- but only MILDLY so once the ghost penalty is in
// (measured asymmetry ~5e-4), and the short-recurrence solvers converge on it:
// default is Jacobi-BiCGStab (O(1) memory; verified same L2 as GMRES to 5
// digits).  SBM_SOLVER=cg|gmres selects the alternatives -- see the solver
// block.  (The historical CG pAp<0 / BiCGStab divergence were artifacts of the
// UNSTABILIZED operator, pre-ghost-penalty.)
//
// Boundary geometry (d, nu) is computed ONCE per face quadrature point and
// cached: with a BVH-backed oracle each evaluation is expensive, and
// recomputing it inside the Krylov loop made blade solves non-terminating.
//
// Cartesian (coordMode 0) only for now; the cylindrical sector needs the
// curved-metric face terms and is the next step.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CutFemSolver.cuh"
#include "QpBasis.h"
#include "PolyFit.h"
#include "QpElem.h"
#include "SbmShift.h"

static inline u64 sbKey(i32 I, i32 J, i32 K) {
  return (u64)I | ((u64)J << 21) | ((u64)K << 42);
}
static inline void sbBlockDec(u64 loc, i32 &i, i32 &j, i32 &k) {
  k = (i32)((loc >> 40) & ((1u<<20)-1)) - 1;
  j = (i32)((loc >> 20) & ((1u<<20)-1)) - 1;
  i = (i32)( loc        & ((1u<<20)-1)) - 1;
}
static long sbNowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- curvature (parallel-surface Jacobian) from a fitted level set ----------
// Ported verbatim from the host M1 gate (SbmSolve.h polyDeriv/shapeInvariants);
// used by the GAP-SBM Neumann path so the geometric transfers keep the optimal
// O(h^{p+1}) rate at p >= 2.  J(s) = 1 + trS*s + detK*s^2.
static double sbPolyDeriv(const PolyND &P, const double xr[3], int mx, int my, int mz) {
  double s = 0;
  for (int k = 0; k < PNC; k++) for (int j = 0; j < PNC; j++) for (int i = 0; i < PNC; i++) {
    if (i < mx || j < my || k < mz) continue;
    double cf = (double)P.at(i, j, k); if (cf == 0) continue;
    for (int q = 0; q < mx; q++) cf *= (i - q);
    for (int q = 0; q < my; q++) cf *= (j - q);
    for (int q = 0; q < mz; q++) cf *= (k - q);
    for (int q = 0; q < i - mx; q++) cf *= xr[0];
    for (int q = 0; q < j - my; q++) cf *= xr[1];
    for (int q = 0; q < k - mz; q++) cf *= xr[2];
    s += cf;
  }
  return s;
}
static void sbShapeInvariants(const PolyND &P, const double xr[3], double h,
                              double &trS, double &detK) {
  double g[3] = {sbPolyDeriv(P,xr,1,0,0)/h, sbPolyDeriv(P,xr,0,1,0)/h, sbPolyDeriv(P,xr,0,0,1)/h};
  double gm = sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]); if (gm < 1e-30) { trS = 0; detK = 0; return; }
  double nu[3] = {g[0]/gm, g[1]/gm, g[2]/gm};
  double H[3][3];
  H[0][0]=sbPolyDeriv(P,xr,2,0,0)/(h*h); H[1][1]=sbPolyDeriv(P,xr,0,2,0)/(h*h); H[2][2]=sbPolyDeriv(P,xr,0,0,2)/(h*h);
  H[0][1]=H[1][0]=sbPolyDeriv(P,xr,1,1,0)/(h*h);
  H[0][2]=H[2][0]=sbPolyDeriv(P,xr,1,0,1)/(h*h);
  H[1][2]=H[2][1]=sbPolyDeriv(P,xr,0,1,1)/(h*h);
  double Pr[3][3]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Pr[i][j]=(i==j?1.0:0.0)-nu[i]*nu[j];
  double T[3][3], S[3][3];
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) { double s2=0; for (int m=0;m<3;m++) s2+=Pr[i][m]*H[m][j]/gm; T[i][j]=s2; }
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) { double s2=0; for (int m=0;m<3;m++) s2+=T[i][m]*Pr[m][j]; S[i][j]=s2; }
  trS = S[0][0]+S[1][1]+S[2][2];
  double tr2 = 0; for (int i=0;i<3;i++) for (int j=0;j<3;j++) tr2 += S[i][j]*S[j][i];
  detK = 0.5*(trS*trS - tr2);
}
// value + reference gradient (d/d xr) of a fitted Q_p SDF at ref point xr.  Used
// by the Newton to the surrogate's zero-set, which replaces the BVH oracle in the
// boundary geometry (isoparametric: geometry from the cached solution-node SDF).
static void sbPolyVG(const PolyND &P, const double xr[3], double &val, double g[3]) {
  val  = sbPolyDeriv(P,xr,0,0,0);
  g[0] = sbPolyDeriv(P,xr,1,0,0);
  g[1] = sbPolyDeriv(P,xr,0,1,0);
  g[2] = sbPolyDeriv(P,xr,0,0,1);
}
// qpElemCore with PRECOMPUTED reference basis gradients ggb[q][3*ndof] (constant
// across cells/applies on a Cartesian mesh).  Same math as qpElemCore but never
// re-derives the gradients -- the density band-matrix assembly calls this 81x per
// cell, so hoisting allGradRef out of the loop removes ~81x redundant work.
static inline void qpElemCoreG(const QpBasis &B, real mu, real lam, real h,
                               const real *ggb, const real *w, i32 npts,
                               const real *u, real *y) {
  i32 ndof=B.n*B.n*B.n;
  for (i32 a=0;a<3*ndof;a++) y[a]=0;
  for (i32 q=0;q<npts;q++){ const real *gb=ggb+(size_t)q*3*ndof;
    real gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    for (i32 a=0;a<ndof;a++) for(i32 i=0;i<3;i++){ real ui=u[3*a+i];
      gU[i][0]+=ui*gb[3*a]; gU[i][1]+=ui*gb[3*a+1]; gU[i][2]+=ui*gb[3*a+2]; }
    real eps[3][3]; for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) eps[i][j]=(real)0.5*(gU[i][j]+gU[j][i]);
    real tr=eps[0][0]+eps[1][1]+eps[2][2], sig[3][3];
    for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) sig[i][j]=2*mu*eps[i][j]+(i==j?lam*tr:(real)0);
    real wq=w[q];
    for (i32 a=0;a<ndof;a++) for(i32 i=0;i<3;i++)
      y[3*a+i]+=wq*(sig[i][0]*gb[3*a]+sig[i][1]*gb[3*a+1]+sig[i][2]*gb[3*a+2]); }
  for (i32 a=0;a<3*ndof;a++) y[a]*=h;
}

// =====================================================================
//  GPU matrix-free SBM operator (SBM_GPU=1).  The continuous Q_p nodal
//  solution lives in flat DEVICE arrays -- dof-space x/y (3*nReal) and
//  node-space xn/yn (3*nNode) -- exactly the p=1 CutFEM pattern
//  (CutFemSolverKernels).  applyA = prolong -> {bulk,face,ghost} -> restrict,
//  continuous-node accumulation via atomicAdd.  Stage 1: Cartesian, the
//  Dirichlet shifted-Nitsche operator (Neumann/gap/cyl kernels are stage 2).
// =====================================================================
struct SbmDev {
  QpBasis B;
  i32 nE, nBF, nGFQ, nNode, ndof, ndof3, mG, NQF, gqn;
  real h, mu, lam, gammaD, cph, sph;
  const i32 *eNode, *nMap; const char *nRot;
  const i32 *bfE, *bfD, *bfS, *gfM, *gfP, *gfD;
  const real *shTab, *gbTab, *vbTab, *gqw; const char *neuPt;
  const real *Kg[3], *Kbulk;
  const real *rhoE;          // density solver: ersatz density per element (tanh mask)
  const char *dofDir;        // density solver: 1 if this dof is a strong-Dirichlet dof
};
#define SBM_QN3 (QN_MAX*QN_MAX*QN_MAX)
#define SBM_STRIDE (blockIdx.x*blockDim.x+threadIdx.x)
// xn = P x  (dof -> node, pitch rotation on tied slaves)
__global__ void sbmProlongK(SbmDev S, const real *x, real *xn) {
  for (i32 nd=SBM_STRIDE; nd<S.nNode; nd+=gridDim.x*blockDim.x) {
    i32 m=S.nMap[nd]; real x0=x[3*m],x1=x[3*m+1],x2=x[3*m+2];
    if (S.nRot[nd]) { xn[3*nd]=S.cph*x0-S.sph*x1; xn[3*nd+1]=S.sph*x0+S.cph*x1; xn[3*nd+2]=x2; }
    else            { xn[3*nd]=x0; xn[3*nd+1]=x1; xn[3*nd+2]=x2; }
  }
}
// y += P^T yn  (node -> dof, inverse rotation; y pre-zeroed)
__global__ void sbmRestrictK(SbmDev S, const real *yn, real *y) {
  for (i32 nd=SBM_STRIDE; nd<S.nNode; nd+=gridDim.x*blockDim.x) {
    i32 m=S.nMap[nd]; real c0=yn[3*nd],c1=yn[3*nd+1],c2=yn[3*nd+2],a0,a1,a2;
    if (S.nRot[nd]) { a0=S.cph*c0+S.sph*c1; a1=-S.sph*c0+S.cph*c1; a2=c2; }
    else            { a0=c0; a1=c1; a2=c2; }
    atomicAdd(&y[3*m],a0); atomicAdd(&y[3*m+1],a1); atomicAdd(&y[3*m+2],a2);
  }
}
// bulk elasticity: yn += K_bulk xn.  BLOCK per element, THREAD per element dof:
// each thread owns one output dof r and computes row r of the (constant, affine)
// element matrix times the shared-memory-gathered element solution.
__global__ void sbmBulkK(SbmDev S, const real *xn, real *yn) {
  i32 ndof=S.ndof, m3=S.ndof3; extern __shared__ real uls[];   // [m3]
  for (i32 e=blockIdx.x; e<S.nE; e+=gridDim.x) {
    const i32 *nod=S.eNode+(size_t)e*ndof;
    for (i32 c=threadIdx.x;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)];
    __syncthreads();
    for (i32 r=threadIdx.x;r<m3;r+=blockDim.x){
      const real *Kr=S.Kbulk+(size_t)r*m3; real acc=0;
      for (i32 c=0;c<m3;c++) acc+=Kr[c]*uls[c];
      atomicAdd(&yn[3*nod[r/3]+(r%3)], acc);
    }
    __syncthreads();
  }
}
// DENSITY SOLVER: ersatz density-weighted bulk elasticity, yn += rho_e * K_bulk xn.
// Same thread-per-dof dense matvec, scaled by the per-element tanh(phi) density.
// SPD -> CG.  (Elemental density; a nodal/quad-point mask is a later refinement.)
__global__ void densBulkK(SbmDev S, const real *xn, real *yn) {
  i32 ndof=S.ndof, m3=S.ndof3; extern __shared__ real uls[];
  for (i32 e=blockIdx.x; e<S.nE; e+=gridDim.x) {
    real re=S.rhoE[e]; const i32 *nod=S.eNode+(size_t)e*ndof;
    for (i32 c=threadIdx.x;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)];
    __syncthreads();
    for (i32 r=threadIdx.x;r<m3;r+=blockDim.x){
      const real *Kr=S.Kbulk+(size_t)r*m3; real acc=0;
      for (i32 c=0;c<m3;c++) acc+=Kr[c]*uls[c];
      atomicAdd(&yn[3*nod[r/3]+(r%3)], re*acc);
    }
    __syncthreads();
  }
}
// zero the strong-Dirichlet dofs of a dof-space vector (keeps them fixed in CG)
__global__ void densZeroDirK(const char *dofDir, real *x, i32 nDof){
  for (i32 i=SBM_STRIDE;i<nDof;i+=gridDim.x*blockDim.x) if(dofDir[i]) x[i]=0;
}
// unified per-element dense matvec: yn += scale_e * (K_e . u_e).  K_e is at
// Kall+matOff[e] -- the shared constant K_bulk (matOff 0, scale rho_e) for
// nearly-constant-density cells, or a precomputed HIGH-RES element matrix
// (matOff>0, scale 1) for band cells.  The quadrature is done ONCE at assembly to
// build K_e; the CG matvec never re-quadratures.  BLOCK per element, THREAD per dof.
__global__ void densMatvecK(SbmDev S, const i32 *matOff, const real *scale, const real *Kall,
                            const real *xn, real *yn){
  i32 ndof=S.ndof, m3=S.ndof3; extern __shared__ real uls[];
  for (i32 e=blockIdx.x; e<S.nE; e+=gridDim.x){
    const i32 *nod=S.eNode+(size_t)e*ndof; const real *K=Kall+(size_t)matOff[e]; real sc=scale[e];
    for (i32 c=threadIdx.x;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)];
    __syncthreads();
    for (i32 r=threadIdx.x;r<m3;r+=blockDim.x){ const real *Kr=K+(size_t)r*m3; real acc=0;
      for (i32 c=0;c<m3;c++) acc+=Kr[c]*uls[c];
      atomicAdd(&yn[3*nod[r/3]+(r%3)], sc*acc); }
    __syncthreads();
  }
}
// QUAD-POINT density operator: yn += int rho(x) sigma:eps.  rho varies WITHIN the
// element (sub-element profile from nodal phi), so quadrature is weighted by the
// precomputed wrhoQ[e*NQP+q] = w_q * rho_q.  gbQ[q] = reference basis gradients at
// quad point q (constant, affine).  BLOCK per element, THREAD per dof, two-phase.
__global__ void densBulkQK(SbmDev S, const real *gbQ, const real *wrhoQ, const real *xn, real *yn) {
  i32 ndof=S.ndof, m3=S.ndof3, n=S.B.n, NQP=n*n*n, t=threadIdx.x;
  extern __shared__ real sm[];
  real *uls=sm, *sq=uls+m3;                          // uls[m3], sq[NQP*6] weighted stress
  for (i32 e=blockIdx.x; e<S.nE; e+=gridDim.x){
    const i32 *nod=S.eNode+(size_t)e*ndof;
    for (i32 c=t;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)];
    __syncthreads();
    for (i32 q=t;q<NQP;q+=blockDim.x){
      const real *gb=gbQ+(size_t)q*3*ndof;
      real gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++){ real u0=uls[3*a],u1=uls[3*a+1],u2=uls[3*a+2], g0=gb[3*a],g1=gb[3*a+1],g2=gb[3*a+2];
        gU[0][0]+=u0*g0;gU[0][1]+=u0*g1;gU[0][2]+=u0*g2; gU[1][0]+=u1*g0;gU[1][1]+=u1*g1;gU[1][2]+=u1*g2;
        gU[2][0]+=u2*g0;gU[2][1]+=u2*g1;gU[2][2]+=u2*g2; }
      real tr=gU[0][0]+gU[1][1]+gU[2][2], w=wrhoQ[(size_t)e*NQP+q];
      sq[q*6+0]=w*(2*S.mu*gU[0][0]+S.lam*tr); sq[q*6+1]=w*(2*S.mu*gU[1][1]+S.lam*tr); sq[q*6+2]=w*(2*S.mu*gU[2][2]+S.lam*tr);
      sq[q*6+3]=w*S.mu*(gU[0][1]+gU[1][0]); sq[q*6+4]=w*S.mu*(gU[0][2]+gU[2][0]); sq[q*6+5]=w*S.mu*(gU[1][2]+gU[2][1]);
    }
    __syncthreads();
    if (t<m3){ i32 a=t/3,l=t%3; real y=0;
      for (i32 q=0;q<NQP;q++){ const real *gb=gbQ+(size_t)q*3*ndof; real g0=gb[3*a],g1=gb[3*a+1],g2=gb[3*a+2],s0,s1,s2;
        if(l==0){s0=sq[q*6+0];s1=sq[q*6+3];s2=sq[q*6+4];} else if(l==1){s0=sq[q*6+3];s1=sq[q*6+1];s2=sq[q*6+5];} else {s0=sq[q*6+4];s1=sq[q*6+5];s2=sq[q*6+2];}
        y+=s0*g0+s1*g1+s2*g2; }
      atomicAdd(&yn[3*nod[a]+l], S.h*y);
    }
    __syncthreads();
  }
}
// Dirichlet shifted-Nitsche on Gamma~ (Cartesian).  BLOCK per face, THREAD per dof.
// Phase 1 (thread per quad point): stress trace tu, shifted trace Shu, Shu.n into
// shared.  Phase 2 (thread per output dof a,l): integrate over quad points.
// Neumann quad points contribute 0 (their shared data is zeroed).  Stage 1.
__global__ void sbmFaceK(SbmDev S, const real *xn, real *yn) {
  i32 ndof=S.ndof, m3=S.ndof3, gqn=S.gqn, NQF=S.NQF, t=threadIdx.x;
  extern __shared__ real sm[];
  real *uls=sm, *qtu=uls+m3, *qShu=qtu+NQF*3, *qShun=qShu+NQF*3;   // [m3],[NQF*3],[NQF*3],[NQF]
  for (i32 f=blockIdx.x; f<S.nBF; f+=gridDim.x) {
    i32 e=S.bfE[f], d=S.bfD[f], s=S.bfS[f];
    const i32 *nod=S.eNode+(size_t)e*ndof;
    real nsign=s?(real)1:(real)-1, nn[3]={0,0,0}; nn[d]=nsign;
    for (i32 c=t;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)];
    __syncthreads();
    if (t<NQF) {                                    // ---- phase 1: per quad point ----
      size_t qp=(size_t)f*NQF+t;
      if (S.neuPt[qp]) { qtu[3*t]=qtu[3*t+1]=qtu[3*t+2]=0; qShu[3*t]=qShu[3*t+1]=qShu[3*t+2]=0; qShun[t]=0; }
      else {
        const real *sh=S.shTab+qp*ndof, *gx=S.gbTab+(qp*3+0)*ndof, *gy=S.gbTab+(qp*3+1)*ndof, *gz=S.gbTab+(qp*3+2)*ndof;
        real gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
        for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ real ua=uls[3*a+i2];
          gradU[i2][0]+=ua*gx[a]; gradU[i2][1]+=ua*gy[a]; gradU[i2][2]+=ua*gz[a]; Shu[i2]+=ua*sh[a]; }
        real tr=gradU[0][0]+gradU[1][1]+gradU[2][2], sig[3][3];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=S.mu*(gradU[i2][j2]+gradU[j2][i2])+(i2==j2?S.lam*tr:0);
        for(i32 i2=0;i2<3;i2++) qtu[3*t+i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
        qShu[3*t]=Shu[0]; qShu[3*t+1]=Shu[1]; qShu[3*t+2]=Shu[2];
        qShun[t]=Shu[0]*nn[0]+Shu[1]*nn[1]+Shu[2]*nn[2];
      }
    }
    __syncthreads();
    if (t<m3) {                                     // ---- phase 2: per output dof (a,l) ----
      i32 a=t/3, l=t%3; real y=0;
      for (i32 ql=0;ql<NQF;ql++){
        real Shul=qShu[3*ql+l], Shun=qShun[ql]; if (Shul==0&&qtu[3*ql+l]==0&&Shun==0) continue;
        size_t qp=(size_t)f*NQF+ql; i32 q1=ql/gqn,q2=ql%gqn; real hw=S.gqw[q1]*S.gqw[q2]*S.h;
        real gA[3]={S.gbTab[(qp*3+0)*ndof+a],S.gbTab[(qp*3+1)*ndof+a],S.gbTab[(qp*3+2)*ndof+a]};
        real gan=gA[d]*nsign, ugb=qShu[3*ql]*gA[0]+qShu[3*ql+1]*gA[1]+qShu[3*ql+2]*gA[2];
        real c1=-qtu[3*ql+l]*S.vbTab[qp*ndof+a];
        real c2=-(S.mu*(Shul*gan+ugb*nn[l])+S.lam*gA[l]*Shun);
        real c3=S.gammaD*Shul*S.shTab[qp*ndof+a];
        y+=hw*(c1+c2+c3);
      }
      atomicAdd(&yn[3*nod[a]+l], y);
    }
    __syncthreads();
  }
}
// ghost penalty: yn += Kghost[d] [uM;uP].  BLOCK per ghost face, THREAD per row of
// the dense mG x mG operator (mG = 2*ndof3; rows 0..ndof3-1 -> M, rest -> P).
__global__ void sbmGhostK(SbmDev S, const real *xn, real *yn) {
  i32 ndof=S.ndof, ndof3=S.ndof3, mG=S.mG; extern __shared__ real uMP[];  // [mG]
  for (i32 f=blockIdx.x; f<S.nGFQ; f+=gridDim.x) {
    i32 eM=S.gfM[f], eP=S.gfP[f], dd=S.gfD[f];
    const i32 *nodM=S.eNode+(size_t)eM*ndof, *nodP=S.eNode+(size_t)eP*ndof;
    const real *K=S.Kg[dd];
    for (i32 c=threadIdx.x;c<mG;c+=blockDim.x){
      if (c<ndof3){ i32 a=c/3; uMP[c]=xn[3*nodM[a]+(c%3)]; }
      else { i32 cc=c-ndof3,a=cc/3; uMP[c]=xn[3*nodP[a]+(cc%3)]; }
    }
    __syncthreads();
    for (i32 r=threadIdx.x;r<mG;r+=blockDim.x){
      const real *Kr=K+(size_t)r*mG; real acc=0;
      for (i32 c=0;c<mG;c++) acc+=Kr[c]*uMP[c];
      if (r<ndof3){ i32 a=r/3; atomicAdd(&yn[3*nodM[a]+(r%3)], acc); }
      else { i32 cc=r-ndof3,a=cc/3; atomicAdd(&yn[3*nodP[a]+(cc%3)], acc); }
    }
    __syncthreads();
  }
}
// vector kernels (self-contained; dot reduces into a managed double)
__global__ void sbmSetK(real *x, real v, i32 n){ for(i32 i=SBM_STRIDE;i<n;i+=gridDim.x*blockDim.x) x[i]=v; }
__global__ void sbmDotK(const real *a, const real *b, i32 n, double *out){
  double s=0; for(i32 i=SBM_STRIDE;i<n;i+=gridDim.x*blockDim.x) s+=(double)a[i]*b[i];
  __shared__ double sh[256]; sh[threadIdx.x]=s; __syncthreads();
  for(i32 o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o) sh[threadIdx.x]+=sh[threadIdx.x+o]; __syncthreads(); }
  if(threadIdx.x==0) atomicAdd(out, sh[0]);
}
__global__ void sbmAxpyK(real *y, const real *x, real a, i32 n){ for(i32 i=SBM_STRIDE;i<n;i+=gridDim.x*blockDim.x) y[i]+=a*x[i]; }
__global__ void sbmJacobiK(real *z, const real *r, const real *d, i32 n){ for(i32 i=SBM_STRIDE;i<n;i+=gridDim.x*blockDim.x) z[i]=r[i]/d[i]; }
// combined: p = r + be*(p - om*v)
__global__ void sbmBicgPK(real *p, const real *r, const real *v, real be, real om, i32 n){
  for(i32 i=SBM_STRIDE;i<n;i+=gridDim.x*blockDim.x) p[i]=r[i]+be*(p[i]-om*v[i]);
}

void CutFemSolver::runSbm(void) {
  const i32 p = femOrder;
  QpBasis Bp; Bp.init(p);
  const i32 n = p+1, ndof = n*n*n, ndof3 = 3*ndof, mG = 2*ndof3;
  const real h = cellSize();
  const double mu = prob.mu, lam = prob.lam;
  const double gammaD_ = 20.0*(p+1)*(p+1)*(2*mu+lam);  // GSBM Eq.(35): chi=20
  const bool per = (periodic != 0);
  const double cph = cos((double)pitchAngle), sph = sin((double)pitchAngle);
  const double kapG = getenv("SBM_KAPPA") ? atof(getenv("SBM_KAPPA")) : 0.5;
  const double gammaG_ = kapG*(2*mu+lam);              // GSBM Eq.(17): kappa=0.5 default

  const bool cyl = (ls.coordMode == 1);
  if (ls.coordMode != 0 && !cyl) {
    printf("ERROR: SBM path supports coordMode 0/1 only\n"); return; }
  if (p > QP_MAX) { printf("ERROR: femOrder %d exceeds QP_MAX=%d\n", p, QP_MAX); return; }

  printf("higher : GSBM shifted boundary (Eq.35), p=%d  %s%s  beta1=%.4g/h beta2=%.4g\n",
         p, cyl?"CYLINDRICAL isoparametric":"Cartesian", per?" + cyclic pitch tie":"",
         gammaD_, gammaG_);

  long t0 = sbNowUs();
  initialize();
  buildMesh();

  // -------------------------------------------------------------------------
  //  surrogate elements: cells whose CENTRE is inside (phi < 0).  All full.
  // -------------------------------------------------------------------------
  struct SElem { i32 ci, cj, ck; };
  std::vector<SElem> elems;
  const i32 nB = hashTable.nKeys;
  #pragma omp parallel
  {
    std::vector<SElem> le;
    #pragma omp for schedule(dynamic,4) nowait
    for (i32 b = 0; b < nB; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty) continue;
      i32 ib, jb, kb; sbBlockDec(loc, ib, jb, kb);
      for (i32 cz = 0; cz < blockSize; cz++)
      for (i32 cy = 0; cy < blockSize; cy++)
      for (i32 cx = 0; cx < blockSize; cx++) {
        i32 ci = ib*blockSize + cx, cj = jb*blockSize + cy, ck = kb*blockSize + cz;
        if (ls.phi((ci+(real)0.5)*h, (cj+(real)0.5)*h, (ck+(real)0.5)*h) < 0)
          le.push_back({ci,cj,ck});
      }
    }
    #pragma omp critical
    { for (auto &E:le) elems.push_back(E); }
  }
  i32 nE = (i32)elems.size();
  if (nE == 0) { printf("ERROR: no surrogate elements\n"); return; }

  // ---- surrogate connectivity audit + optional pruning ---------------------
  // Thin features (the blade TE at coarse h) can leave cells attached to the
  // body of the surrogate by <=1 face: floppy chains whose near-null modes are
  // held only by boundary penalties -- a Krylov solver's nightmare.  The
  // surrogate is OURS to define, so SBM_PRUNE=1 drops cells with <2 surrogate
  // face-neighbours (iterated to a fixed point; Gamma~ just moves slightly,
  // which the shift absorbs by construction).  The audit prints either way.
  {
    bool prune = getenv("SBM_PRUNE") && atoi(getenv("SBM_PRUNE"))!=0;
    for (i32 pass=0;;pass++){
      std::unordered_map<u64,i32> cid; cid.reserve((size_t)nE*2);
      for (i32 e=0;e<nE;e++) cid[sbKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;
      std::vector<i32> nnb(nE,0);
      for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
        for (i32 d=0;d<3;d++) for (i32 s2=0;s2<2;s2++){
          i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]+= s2?1:-1;
          if (per && d==1){ if (nb[1]==nThetaCells) nb[1]=0; else if (nb[1]<0) nb[1]=nThetaCells-1; }
          if (cid.find(sbKey(nb[0],nb[1],nb[2]))!=cid.end()) nnb[e]++; } }
      i32 c0=0,c1=0;
      for (i32 e=0;e<nE;e++){ if (nnb[e]==0) c0++; else if (nnb[e]==1) c1++; }
      if (pass==0 || c0+c1==0)
        printf("connect: %d elems with 0 nbrs, %d with 1 nbr (of %d)%s\n",
               c0, c1, nE, prune?"":"  [SBM_PRUNE=1 to drop]");
      if (!prune || c0+c1==0) break;
      std::vector<SElem> keep; keep.reserve(nE);
      for (i32 e=0;e<nE;e++) if (nnb[e]>=2) keep.push_back(elems[e]);
      printf("connect: pass %d pruned %d weakly-connected cells\n", pass, nE-(i32)keep.size());
      elems.swap(keep); nE=(i32)elems.size();
      if (nE==0){ printf("ERROR: pruning removed everything\n"); return; }
    }
  }

  // -------------------------------------------------------------------------
  //  node numbering on the p-finer lattice (no cyclic tie: Cartesian only)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> nodeId; nodeId.reserve((size_t)nE*ndof);
  std::vector<i32> eNodeQ((size_t)nE*ndof);
  std::vector<i32> nI, nJ, nK;
  i32 nNodeQ=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 i=a%n, j=(a/n)%n, k=a/(n*n);
    i32 I=p*elems[e].ci+i, J=p*elems[e].cj+j, K=p*elems[e].ck+k;
    u64 key=sbKey(I,J,K);
    auto it=nodeId.find(key); i32 id;
    if (it==nodeId.end()){ id=nNodeQ++; nodeId[key]=id; nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    else id=it->second;
    eNodeQ[(size_t)e*ndof+a]=id;
  }
  // cyclic dof map: theta-seam slave columns (J = p*nThetaCells) tie to the
  // master at J = 0 through the pitch rotation (identity when non-periodic).
  const i32 Jmax = p*nThetaCells;
  std::vector<i32> realIdx(nNodeQ,-1);
  std::vector<char> rotFlag(nNodeQ,0);
  i32 nDofNode=0, nTie=0, nOrphan=0;
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (per && nJ[nd]==Jmax) continue;
    realIdx[nd]=nDofNode++;
  }
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (realIdx[nd]>=0) continue;
    auto it=nodeId.find(sbKey(nI[nd],0,nK[nd]));
    if (it==nodeId.end()){ realIdx[nd]=nDofNode++; nOrphan++; }
    else { realIdx[nd]=realIdx[it->second]; rotFlag[nd]=1; nTie++; }
  }
  const i32 nDofQ=3*nDofNode;
  if (per) printf("cyclic : %d SBM nodes tied across the pitch (%d unmatched kept free), %d -> %d dofs\n",
                  nTie, nOrphan, 3*nNodeQ, nDofQ);

  // gather/scatter through the tie (rotation-aware; identity when non-periodic)
  auto gather3=[&](const std::vector<real>&x,i32 nd,double u[3]){
    i32 b=3*realIdx[nd];
    if (rotFlag[nd]){ u[0]=cph*x[b]-sph*x[b+1]; u[1]=sph*x[b]+cph*x[b+1]; u[2]=x[b+2]; }
    else { u[0]=x[b]; u[1]=x[b+1]; u[2]=x[b+2]; }
  };
  auto scatter3=[&](std::vector<real>&y,i32 nd,const double c[3]){
    i32 b=3*realIdx[nd]; double a0,a1,a2v;
    if (rotFlag[nd]){ a0=cph*c[0]+sph*c[1]; a1=-sph*c[0]+cph*c[1]; a2v=c[2]; }
    else { a0=c[0]; a1=c[1]; a2v=c[2]; }
    #pragma omp atomic
    y[b]+=(real)a0;
    #pragma omp atomic
    y[b+1]+=(real)a1;
    #pragma omp atomic
    y[b+2]+=(real)a2v;
  };

  std::vector<real> nodeXQ((size_t)3*nNodeQ);
  for (i32 nd=0;nd<nNodeQ;nd++){
    real X0,X1,X2; ls.toPhys((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p,X0,X1,X2);
    nodeXQ[3*nd]=X0; nodeXQ[3*nd+1]=X1; nodeXQ[3*nd+2]=X2;
  }

  // -------------------------------------------------------------------------
  //  ISOPARAMETRIC GEOMETRY.  Sample the SDF at the p-order solution nodes -- the
  //  LAST oracle use -- and build a per-element Q_p interpolant of phi.  From here
  //  the boundary geometry (d, nu, curvature, the shift's zero-set) is derived
  //  from this Q_p SDF, NOT the oracle: the geometry is represented at the SAME
  //  order as the displacement, so it is C-infinity within a cell (clean Hessian
  //  -> JAC>=2 works on faceted/BVH input) and d,nu,curvature are mutually
  //  consistent.  Zero BVH calls in the Krylov-adjacent geometry.  SBM_ORACLEGEO=1
  //  restores the old per-face-point oracle Newton (A/B).
  // -------------------------------------------------------------------------
  const bool oracleGeo = getenv("SBM_ORACLEGEO");
  std::vector<real> phiN(nNodeQ);
  #pragma omp parallel for schedule(dynamic,256)
  for (i32 nd=0;nd<nNodeQ;nd++)
    phiN[nd]=ls.phi((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p);
  std::vector<PolyND> Pelem(nE);
  #pragma omp parallel for schedule(dynamic,64)
  for (i32 e=0;e<nE;e++){ real v[QN_MAX*QN_MAX*QN_MAX];
    const i32*nod=&eNodeQ[(size_t)e*ndof];
    for (i32 a=0;a<ndof;a++) v[a]=phiN[nod[a]];
    Pelem[e]=fitPoly3(p,v); }

  // Cut cells: cache phi at the solution nodes of the NON-surrogate cells around
  // the surrogate too -- the cut band, where the true wall actually lives.  With
  // their Q_p SDF in hand the boundary Newton can MARCH across cells to a wall
  // that is more than one cell from the surrogate face (still no oracle in the
  // march).  Pmap holds the SDF of every cell the march may enter.  SBM_BAND sets
  // the band radius (cells); it caps the reachable |d|.
  std::unordered_map<u64,PolyND> Pmap;
  for (i32 e=0;e<nE;e++) Pmap.emplace(sbKey(elems[e].ci,elems[e].cj,elems[e].ck), Pelem[e]);
  if (!oracleGeo){
    const i32 RB = getenv("SBM_BAND") ? atoi(getenv("SBM_BAND")) : 2;
    std::unordered_set<u64> seen;
    for (i32 e=0;e<nE;e++) seen.insert(sbKey(elems[e].ci,elems[e].cj,elems[e].ck));
    std::vector<u64> cut;
    for (i32 e=0;e<nE;e++){ i32 c[3]={elems[e].ci,elems[e].cj,elems[e].ck};
      for (i32 dz=-RB;dz<=RB;dz++) for (i32 dy=-RB;dy<=RB;dy++) for (i32 dx=-RB;dx<=RB;dx++){
        i32 nb[3]={c[0]+dx,c[1]+dy,c[2]+dz};
        if (nb[0]<0||nb[1]<0||nb[2]<0) continue;
        u64 k=sbKey(nb[0],nb[1],nb[2]);
        if (seen.insert(k).second) cut.push_back(k); } }
    std::vector<PolyND> cutP(cut.size());
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t ci=0;ci<cut.size();ci++){ u64 k=cut[ci];
      i32 c0=(i32)(k&0x1FFFFF), c1=(i32)((k>>21)&0x1FFFFF), c2=(i32)((k>>42)&0x1FFFFF);
      real v[QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ i32 i=a%n,j=(a/n)%n,kk=a/(n*n);
        v[a]=ls.phi((c0+(real)i/p)*h,(c1+(real)j/p)*h,(c2+(real)kk/p)*h); }
      cutP[ci]=fitPoly3(p,v); }
    for (size_t ci=0;ci<cut.size();ci++) Pmap.emplace(cut[ci], cutP[ci]);
    printf("isogeo : %d surrogate + %zu cut-band cells cached at solution nodes (band %d)\n",
           nE, cut.size(), RB);
  }

  // -------------------------------------------------------------------------
  //  surrogate boundary faces + ghost faces (boundary band)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> cellId; cellId.reserve((size_t)nE*2);
  for (i32 e=0;e<nE;e++) cellId[sbKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;

  struct BF{ i32 e, d, s; };            // owning elem, axis, side (0:-,1:+)
  std::vector<BF> bf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++) for (i32 s=0;s<2;s++){
      i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]+= s?1:-1;
      if (per && d==1){ if (nb[1]==nThetaCells) nb[1]=0; else if (nb[1]<0) nb[1]=nThetaCells-1; }
      if (cellId.find(sbKey(nb[0],nb[1],nb[2]))==cellId.end()) bf.push_back({e,d,s});
    } }
  i32 nBF=(i32)bf.size();

  std::vector<char> nearB((size_t)nE,0);
  for (i32 f=0;f<nBF;f++) nearB[bf[f].e]=1;
  struct GF{ i32 eM,eP,d; };
  std::vector<GF> gf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
      if (per && d==1 && nb[1]==nThetaCells) nb[1]=0;
      auto it=cellId.find(sbKey(nb[0],nb[1],nb[2])); if (it==cellId.end()) continue;
      i32 ep=it->second; if (nearB[e]||nearB[ep]) gf.push_back({e,ep,d}); } }
  i32 nGFQ=(i32)gf.size();

  // cylindrical metric at a reference point of element E: Jinv (=Jref^{-1}), detJ
  auto inv3s=[&](const double J[3][3], double Ji[3][3])->double{
    double c00=J[1][1]*J[2][2]-J[1][2]*J[2][1];
    double c01=J[1][2]*J[2][0]-J[1][0]*J[2][2];
    double c02=J[1][0]*J[2][1]-J[1][1]*J[2][0];
    double det=J[0][0]*c00+J[0][1]*c01+J[0][2]*c02;
    double id=(fabs(det)>1e-300)?1.0/det:0.0;
    Ji[0][0]=c00*id; Ji[1][0]=c01*id; Ji[2][0]=c02*id;
    Ji[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id;
    Ji[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id;
    Ji[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id;
    Ji[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id;
    Ji[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
    Ji[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;
    return det;
  };
  auto metric=[&](const SElem&E,const real xr[3],double Jinv[3][3],double&detJ){
    double r=(E.ci+xr[0])*h+ls.org[0], s2=(E.cj+xr[1])*h+ls.org[1], z=(E.ck+xr[2])*h+ls.org[2];
    double th=s2/ls.rRef+(double)ls.thc((real)z), thp=(double)ls.thcSlope((real)z);
    double ct=cos(th), st=sin(th);
    double J[3][3]={{ h*ct, h*(-r*st/ls.rRef), h*(-r*st*thp) },
                    { h*st, h*( r*ct/ls.rRef), h*( r*ct*thp) },
                    { 0,    0,                 h            }};
    detJ=inv3s(J,Jinv);
  };

  // Newton MARCH to the isoparametric wall: step toward {phi=0} from a surrogate
  // face point, crossing cells via Pmap (each cell's own Q_p SDF).  Reaches a wall
  // more than one cell away with NO oracle.  Returns false (caller clamps) if the
  // march leaves the cached band or fails to converge.  d,nu are computational.
  auto marchZero=[&](double sx,double sy,double sz,double d[3],double nu[3])->bool{
    double x[3]={sx,sy,sz};
    for (i32 it=0;it<12;it++){
      i32 ci=(i32)floor(x[0]/h), cj=(i32)floor(x[1]/h), ck=(i32)floor(x[2]/h);
      if (ci<0||cj<0||ck<0) return false;
      auto pit=Pmap.find(sbKey(ci,cj,ck)); if (pit==Pmap.end()) return false;
      double xr[3]={x[0]/h-ci, x[1]/h-cj, x[2]/h-ck};
      double val,g[3]; sbPolyVG(pit->second,xr,val,g);
      double gm2=g[0]*g[0]+g[1]*g[1]+g[2]*g[2]; if (gm2<1e-30) return false;
      if (fabs(val) < 1e-6*(double)h){                   // reached the wall
        double gm=sqrt(gm2);
        nu[0]=g[0]/gm; nu[1]=g[1]/gm; nu[2]=g[2]/gm;      // ref-grad dir = computational normal
        d[0]=x[0]-sx; d[1]=x[1]-sy; d[2]=x[2]-sz; return true; }
      double s=val*h/gm2;                                // computational Newton step
      x[0]-=s*g[0]; x[1]-=s*g[1]; x[2]-=s*g[2];
    }
    return false;
  };

  // -------------------------------------------------------------------------
  //  boundary geometry cache: d (computational frame) and nu at every face
  //  quadrature point, from a Newton MARCH on the cached Q_p SDF (Pmap) -- NO
  //  oracle (SBM_ORACLEGEO=1 restores the legacy BVH Newton for A/B).
  // -------------------------------------------------------------------------
  GaussRule gq = gaussLegendre(n);
  const i32 NQF = gq.n*gq.n;
  std::vector<double> geoD((size_t)nBF*NQF*6, 0.0);
  long nClamp=0;
  const double dCapF = getenv("SBM_DCAP") ? atof(getenv("SBM_DCAP")) : 3.0;
  const double dCap = dCapF*(double)h;   // safety: a surrogate face's true wall
                                         // should be within ~1 cell; beyond this
                                         // the Newton went wrong -> d=0 (plain
                                         // Nitsche at the face, exact-data MMS)
  #pragma omp parallel for schedule(dynamic,16) reduction(+:nClamp)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    const SElem&E=elems[F.e];
    const PolyND &Pe=Pelem[F.e];
    i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[F.d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      double dv[3], nu3[3];
      if (oracleGeo){                                  // --- legacy: BVH oracle Newton (A/B) ---
        double c0=(E.ci+xr[0])*h, c1=(E.cj+xr[1])*h, c2=(E.ck+xr[2])*h;
        double cx=c0, cy=c1, cz=c2; real g3[3]; double gm2=1;
        for (i32 it=0; it<3; it++){ real fv=ls.phiGrad((real)cx,(real)cy,(real)cz,g3);
          gm2=(double)g3[0]*g3[0]+(double)g3[1]*g3[1]+(double)g3[2]*g3[2];
          if (gm2<1e-30) break;
          cx-=fv*g3[0]/gm2; cy-=fv*g3[1]/gm2; cz-=fv*g3[2]/gm2; }
        dv[0]=cx-c0; dv[1]=cy-c1; dv[2]=cz-c2;
        ls.phiGrad((real)cx,(real)cy,(real)cz,g3);
        double gm=sqrt((double)g3[0]*g3[0]+(double)g3[1]*g3[1]+(double)g3[2]*g3[2]); if (gm<1e-30) gm=1;
        nu3[0]=g3[0]/gm; nu3[1]=g3[1]/gm; nu3[2]=g3[2]/gm;
      } else {                                         // --- isoparametric: march to the wall via Pmap ---
        double c0=(E.ci+xr[0])*h, c1=(E.cj+xr[1])*h, c2=(E.ck+xr[2])*h;
        if (!marchZero(c0,c1,c2,dv,nu3)){               // wall not reachable -> clamp d, keep owning-cell nu
          double xrd[3]={xr[0],xr[1],xr[2]}, val, g[3]; sbPolyVG(Pe,xrd,val,g);
          double gm=sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]); if (gm<1e-30) gm=1;
          nu3[0]=g[0]/gm; nu3[1]=g[1]/gm; nu3[2]=g[2]/gm;
          dv[0]=dv[1]=dv[2]=0; nClamp++;
        }
      }
      double dm=sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]);
      if (!(dm<=dCap)){ dv[0]=dv[1]=dv[2]=0; nClamp++; }
      double *g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      g6[0]=dv[0]; g6[1]=dv[1]; g6[2]=dv[2];
      g6[3]=nu3[0]; g6[4]=nu3[1]; g6[5]=nu3[2];
    } }
  if (nClamp) printf("warning: %ld boundary quadrature points clamped (|d| > %g)\n",
                     nClamp, dCap);
  { double dmax=0,dsum=0; long cnt=0; i32 nbig=0;
    for (i32 f=0;f<nBF;f++) for (i32 q=0;q<NQF;q++){
      const double*g6=&geoD[((size_t)f*NQF+q)*6];
      double dm=sqrt(g6[0]*g6[0]+g6[1]*g6[1]+g6[2]*g6[2])/h;
      dmax=fmax(dmax,dm); dsum+=dm; cnt++; if (dm>1.0) nbig++; }
    printf("shift  : |d|/h max %.3f mean %.3f, >1h at %d/%ld face pts\n",
           dmax, dsum/(cnt?cnt:1), nbig, cnt); }

  // -------------------------------------------------------------------------
  //  GAP-SBM Neumann (traction) faces: per-face-QP mask + cached curvature.
  //  The gap-augmented natural BC verified in the host M1 gate (SbmSolve.h,
  //  NEU/JAC/NOGAP), ported here to run ALONGSIDE the Dirichlet shifted-Nitsche.
  //  Cartesian only for now (the gap sliver needs the affine metric).
  //    SBM_NEU=1  manufactured test: Neumann where the true normal nu_z>0 (the
  //               upper "hemisphere"); Dirichlet elsewhere (fixes rigid modes).
  //    SBM_NEU=0  physical BC: Neumann where !prob.isDirichlet (e.g. CASE_LOAD).
  //    SBM_JAC    curvature-Jacobian order (>=2 fitted; needed for optimal p>=2;
  //               =2 quadratic fit, =3 cubic).  0/1 -> linearized (no Jacobian).
  //    SBM_NOGAP  ablation: drop the gap sliver term.
  // -------------------------------------------------------------------------
  const int sbmNeuMode = getenv("SBM_NEU")   ? atoi(getenv("SBM_NEU"))   : 0;
  const int sbmJac     = getenv("SBM_JAC")   ? atoi(getenv("SBM_JAC"))   : 2;
  const int sbmGap     = getenv("SBM_NOGAP") ? 0 : 1;
  GaussRule gt = gaussLegendre(n);            // 1-D rule along d for the gap sliver
  std::vector<char>   neuPt((size_t)nBF*NQF, 0);
  std::vector<double> curvT;                  // (trS,detK) per Neumann QP
  long nNeuPt = 0;
  {
    #pragma omp parallel for schedule(dynamic,16) reduction(+:nNeuPt)
    for (i32 f=0;f<nBF;f++){ const BF&F=bf[f]; const SElem&E=elems[F.e];
      i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
      for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2;
        const double*g6=&geoD[qp*6];
        real xr[3]; xr[F.d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
        real XT[3]; ls.toPhys((E.ci+xr[0])*h+(real)g6[0],(E.cj+xr[1])*h+(real)g6[1],
                              (E.ck+xr[2])*h+(real)g6[2],XT[0],XT[1],XT[2]);
        double nuz = g6[5];   // physical z-normal (== computational z-normal for Cartesian)
        if (cyl){             // cyl: transform the computational normal by the metric
          real xrT[3]={(real)(xr[0]+g6[0]/h),(real)(xr[1]+g6[1]/h),(real)(xr[2]+g6[2]/h)};
          double Jinv[3][3],detJ; metric(E,xrT,Jinv,detJ);
          double np[3]; for (i32 i2=0;i2<3;i2++) np[i2]=Jinv[0][i2]*g6[3]+Jinv[1][i2]*g6[4]+Jinv[2][i2]*g6[5];
          double nm=sqrt(np[0]*np[0]+np[1]*np[1]+np[2]*np[2]); nuz = nm>0? np[2]/nm : 0;
        }
        bool neu = sbmNeuMode ? (nuz>0.0) : (!prob.isDirichlet(XT[0],XT[1],XT[2]));
        neuPt[qp] = neu?1:0; if (neu) nNeuPt++;
      } }
  }
  if (nNeuPt && sbmJac>=1 && !cyl) {   // curvature Jacobian from the cached Q_p SDF (Cartesian)
    // Isoparametric: the SAME per-element interpolant that gives d,nu -- so the
    // curvature is self-consistent with the shift (this is what makes JAC work on
    // faceted/BVH input; the old path fitted a SEPARATE oracle sampling).
    curvT.assign((size_t)nBF*NQF*2, 0.0);
    #pragma omp parallel for schedule(dynamic,16)
    for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
      i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
      const PolyND &Pe=Pelem[F.e];
      for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2; if (!neuPt[qp]) continue;
        double xr[3]; xr[F.d]=F.s?1.0:0.0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
        double trS,detK; sbShapeInvariants(Pe,xr,h,trS,detK);
        curvT[qp*2]=trS; curvT[qp*2+1]=detK;
      } }
  }
  if (nNeuPt) printf("sbm-neu: %ld Neumann face pts  (mode %d, jac %d, gap %d)\n",
                     nNeuPt, sbmNeuMode, sbmJac, sbmGap);

  // -------------------------------------------------------------------------
  //  shift matrices + ghost D^l rows
  // -------------------------------------------------------------------------
  real Vm[QN_MAX][QN_MAX]; sbmDerivMatrix(Bp, Vm);

  // Per-face-quadrature-point basis tables, hoisted out of the Krylov loop:
  // sh (the Taylor shift S_d phi_a -- the expensive one), gb, vb.  Everything
  // the face apply needs per point is then a table read; profiling showed the
  // per-iteration sbmShiftAll evaluation dominated the GMRES cost.
  std::vector<real> shTab, gbTab, vbTab;
  double Dl0[QP_MAX+1][QN_MAX], Dl1[QP_MAX+1][QN_MAX];
  { double Dp[QN_MAX][QN_MAX];
    for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Bp.D[i][a];
    for (i32 l=1;l<=p;l++){ for (i32 a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if (l<p){ double Nw[QN_MAX][QN_MAX];
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++){ double s=0; for(i32 m=0;m<n;m++) s+=Dp[i][m]*Bp.D[m][a]; Nw[i][a]=s; }
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Nw[i][a]; } } }

  shTab.assign((size_t)nBF*NQF*ndof,0); gbTab.assign((size_t)nBF*NQF*3*ndof,0);
  vbTab.assign((size_t)nBF*NQF*ndof,0);
  std::vector<double> wNTab;                 // cyl: weighted physical normal nraw*|detJ|*w
  if (cyl) wNTab.assign((size_t)nBF*NQF*3,0.0);
  #pragma omp parallel for schedule(dynamic,16)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
    real gb[3*QN_MAX*QN_MAX*QN_MAX], sh[QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[F.d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      const double*g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      real dref[3]={(real)(g6[0]/h),(real)(g6[1]/h),(real)(g6[2]/h)};
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); sbmShiftAll(Bp,Vm,xr,dref,sh);
      size_t qp=(size_t)f*NQF + q1*gq.n + q2;
      for (i32 a=0;a<ndof;a++){ shTab[qp*ndof+a]=sh[a]; vbTab[qp*ndof+a]=vb[a]; }
      if (!cyl){
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++) gbTab[(qp*3+d2)*ndof+a]=gb[3*a+d2];
      } else {
        // PHYSICAL gradients + Nanson weighted normal (reference face normal = +-e_d)
        double Jinv[3][3],detJ; metric(elems[F.e],xr,Jinv,detJ);
        double nsign=F.s?1.0:-1.0, wf=gq.w[q1]*gq.w[q2];
        for (i32 i2=0;i2<3;i2++)
          wNTab[qp*3+i2]=nsign*Jinv[F.d][i2]*fabs(detJ)*wf;
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
          gbTab[(qp*3+d2)*ndof+a]=(real)(Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2]);
      } } }

  // ghost matrices (one per axis, reference-invariant; same as runQp but kappa=0.5)
  auto ghostLocal=[&](i32 d,const double*uMP,double*yMP){
    for (i32 i=0;i<mG;i++) yMP[i]=0.0;
    i32 t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(p+1);
    for (i32 q1=0;q1<g1.n;q1++) for (i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2];
      real L1[QN_MAX],L2[QN_MAX]; Bp.basis1(g1.x[q1],L1); Bp.basis1(g1.x[q2],L2);
      double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ i32 idx[3]={a%n,(a/n)%n,a/(n*n)}; i32 idn=idx[d];
        double Lt=L1[idx[t1]]*L2[idx[t2]]; for (i32 l=1;l<=p;l++){ cP[l][a]=Dl0[l][idn]*Lt; cM[l][a]=Dl1[l][idn]*Lt; } }
      for (i32 l=1;l<=p;l++){ double cf=gammaG_*h*w; double jU[3]={0,0,0};
        for (i32 a=0;a<ndof;a++) for(i32 comp=0;comp<3;comp++) jU[comp]+=uMP[ndof3+3*a+comp]*cP[l][a]-uMP[3*a+comp]*cM[l][a];
        for (i32 a=0;a<ndof;a++) for(i32 comp=0;comp<3;comp++){ yMP[ndof3+3*a+comp]+=cf*cP[l][a]*jU[comp];
          yMP[3*a+comp]+=cf*(-cM[l][a])*jU[comp]; } } } };
  std::vector<double> Kghost[3];
  for (i32 d=0;d<3;d++){ Kghost[d].assign((size_t)mG*mG,0.0);
    double ue[2*3*QN_MAX*QN_MAX*QN_MAX], ye[2*3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 cq=0;cq<mG;cq++){ for(i32 a=0;a<mG;a++) ue[a]=(a==cq)?1.0:0.0;
      ghostLocal(d,ue,ye); for (i32 r=0;r<mG;r++) Kghost[d][(size_t)r*mG+cq]=ye[r]; } }

  // -------------------------------------------------------------------------
  //  SBM face local apply (GSBM Eq. 35, reference frame; verified in SbmSolve.h)
  // -------------------------------------------------------------------------
  auto sbmFace=[&](i32 f,const double*uloc,double*yloc){
    const BF&F=bf[f];
    i32 d=F.d;
    double nsign=F.s?1.0:-1.0, nn[3]={0,0,0}; nn[d]=nsign;
    for (i32 a=0;a<ndof3;a++) yloc[a]=0.0;
    if (cyl){
      for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2;
        if (neuPt[qp]){                    // ---- cyl GAP-SBM Neumann: gap bulk stiffness ----
          if (!sbmGap) continue;
          const double*g6=&geoD[qp*6];
          double dref[3]={g6[0]/h,g6[1]/h,g6[2]/h};
          double dnR=dref[d]*nsign, wf=gq.w[q1]*gq.w[q2];
          i32 td1=(d+1)%3, td2=(d+2)%3;
          real xrq[3]; xrq[d]=F.s?(real)1:(real)0; xrq[td1]=(real)gq.x[q1]; xrq[td2]=(real)gq.x[q2];
          real gbr[3*QN_MAX*QN_MAX*QN_MAX]; double gX[QN_MAX*QN_MAX*QN_MAX][3];
          for (i32 qt=0;qt<gt.n;qt++){ double tau=gt.x[qt];
            real xrt[3]={(real)(xrq[0]+tau*dref[0]),(real)(xrq[1]+tau*dref[1]),(real)(xrq[2]+tau*dref[2])};
            double Jinv[3][3],detJ; metric(elems[F.e],xrt,Jinv,detJ);
            Bp.allGradRef(xrt,gbr);
            for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
              gX[a][d2]=Jinv[0][d2]*gbr[3*a]+Jinv[1][d2]*gbr[3*a+1]+Jinv[2][d2]*gbr[3*a+2];
            double W=fabs(detJ)*dnR*wf*gt.w[qt];
            double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
            for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
              gU[i2][0]+=ua*gX[a][0]; gU[i2][1]+=ua*gX[a][1]; gU[i2][2]+=ua*gX[a][2]; }
            double e2[3][3],tr2; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) e2[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
            tr2=e2[0][0]+e2[1][1]+e2[2][2];
            double s2[3][3]; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) s2[i2][j2]=2*mu*e2[i2][j2]+(i2==j2?lam*tr2:0);
            for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++)
              yloc[3*a+l]+=W*(s2[l][0]*gX[a][0]+s2[l][1]*gX[a][1]+s2[l][2]*gX[a][2]);
          }
          continue;
        }
        const real*sh=&shTab[qp*ndof]; const real*vb=&vbTab[qp*ndof];
        const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
        const double*wN=&wNTab[qp*3];
        double dS=sqrt(wN[0]*wN[0]+wN[1]*wN[1]+wN[2]*wN[2]);
        double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
        for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
          gradU[i2][0]+=ua*gx[a]; gradU[i2][1]+=ua*gy[a]; gradU[i2][2]+=ua*gz[a];
          Shu[i2]+=ua*sh[a]; }
        double eps[3][3],tr=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
        tr=eps[0][0]+eps[1][1]+eps[2][2];
        double sig[3][3];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
        double tw[3]; for(i32 i2=0;i2<3;i2++) tw[i2]=sig[i2][0]*wN[0]+sig[i2][1]*wN[1]+sig[i2][2]*wN[2];
        double ShuN=Shu[0]*wN[0]+Shu[1]*wN[1]+Shu[2]*wN[2];
        double penC=gammaD_/h;
        real gA[3];
        for (i32 a=0;a<ndof;a++){
          gA[0]=gx[a]; gA[1]=gy[a]; gA[2]=gz[a];
          double gan=gA[0]*wN[0]+gA[1]*wN[1]+gA[2]*wN[2];
          double ugb=Shu[0]*gA[0]+Shu[1]*gA[1]+Shu[2]*gA[2];
          for (i32 l=0;l<3;l++){
            double c1=-tw[l]*vb[a];                                   // -(v, sigma(u).n) dS
            double c2=-(mu*(Shu[l]*gan+ugb*wN[l])+lam*gA[l]*ShuN);    // -(sigma(v).n, S_d u) dS
            double c3=penC*Shu[l]*sh[a]*dS;                           // beta1/h penalty
            yloc[3*a+l]+=c1+c2+c3; } }
      }
      return;
    }
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      double hw=gq.w[q1]*gq.w[q2]*h;
      size_t qp=(size_t)f*NQF + q1*gq.n + q2;
      if (neuPt[qp]){                        // ---- GAP-SBM Neumann: gap bulk stiffness ----
        // Traction is pure data (RHS only) -> the operator contribution is just
        // sigma(u~):eps(v~) over the gap sliver with the extended field.  No
        // shifted-Nitsche on a Neumann face.
        if (!sbmGap) continue;               // NOGAP ablation: no operator term
        const double*g6=&geoD[qp*6];
        double dd[3]={g6[0],g6[1],g6[2]}, nu[3]={g6[3],g6[4],g6[5]};
        double dref[3]={dd[0]/h,dd[1]/h,dd[2]/h};
        double dn=dd[0]*nn[0]+dd[1]*nn[1]+dd[2]*nn[2];
        double sTot=dd[0]*nu[0]+dd[1]*nu[1]+dd[2]*nu[2];
        double trS=curvT.empty()?0:curvT[qp*2], detK=curvT.empty()?0:curvT[qp*2+1];
        i32 td1=(d+1)%3, td2=(d+2)%3;
        real xrq[3]; xrq[d]=F.s?(real)1:(real)0; xrq[td1]=(real)gq.x[q1]; xrq[td2]=(real)gq.x[q2];
        real gT[3*QN_MAX*QN_MAX*QN_MAX];
        for (i32 qt=0;qt<gt.n;qt++){ double tau=gt.x[qt];
          real xrt[3]={(real)(xrq[0]+tau*dref[0]),(real)(xrq[1]+tau*dref[1]),(real)(xrq[2]+tau*dref[2])};
          Bp.allGradRef(xrt,gT);
          double sq=tau*sTot, jg=1.0+trS*sq+detK*sq*sq;
          double wgb=dn*gq.w[q1]*gq.w[q2]*gt.w[qt]*jg;
          double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
          for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
            gU[i2][0]+=ua*gT[3*a]; gU[i2][1]+=ua*gT[3*a+1]; gU[i2][2]+=ua*gT[3*a+2]; }
          double e2[3][3],tr2; for (i32 i2=0;i2<3;i2++) for(i32 j2=0;j2<3;j2++) e2[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
          tr2=e2[0][0]+e2[1][1]+e2[2][2];
          double s2[3][3]; for (i32 i2=0;i2<3;i2++) for(i32 j2=0;j2<3;j2++) s2[i2][j2]=2*mu*e2[i2][j2]+(i2==j2?lam*tr2:0);
          for (i32 a=0;a<ndof;a++) for (i32 l=0;l<3;l++)
            yloc[3*a+l]+=wgb*(s2[l][0]*gT[3*a]+s2[l][1]*gT[3*a+1]+s2[l][2]*gT[3*a+2]);
        }
        continue;
      }
      const real*sh=&shTab[qp*ndof]; const real*vb=&vbTab[qp*ndof];
      const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
      const real*gd=&gbTab[(qp*3+d)*ndof];
      double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
        gradU[i2][0]+=ua*gx[a]; gradU[i2][1]+=ua*gy[a]; gradU[i2][2]+=ua*gz[a];
        Shu[i2]+=ua*sh[a]; }
      double eps[3][3],tr=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2];
      double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
      double Shun=Shu[0]*nn[0]+Shu[1]*nn[1]+Shu[2]*nn[2];
      real gA[3];
      for (i32 a=0;a<ndof;a++){ double gan=gd[a]*nsign;
        gA[0]=gx[a]; gA[1]=gy[a]; gA[2]=gz[a];
        double ugb=Shu[0]*gA[0]+Shu[1]*gA[1]+Shu[2]*gA[2];
        for (i32 l=0;l<3;l++){
          double c1=-tu[l]*vb[a];                                     // -(v, sigma(u).n~)
          double c2=-(mu*(Shu[l]*gan+ugb*nn[l])+lam*gA[l]*Shun);      // -(sigma(v).n~, S_d u)
          double c3=gammaD_*Shu[l]*sh[a];                             // (beta1/h S_d v, S_d u)
          yloc[3*a+l]+=hw*(c1+c2+c3); } }
    } };

  // -------------------------------------------------------------------------
  //  matrix-free operator apply
  // -------------------------------------------------------------------------
  auto applyA=[&](const std::vector<real>&x,std::vector<real>&y){
    std::fill(y.begin(),y.end(),(real)0);
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nod[a],u3);
        ul[3*a]=(real)u3[0]; ul[3*a+1]=(real)u3[1]; ul[3*a+2]=(real)u3[2]; }
      if (cyl){
        double ylc[3*QN_MAX*QN_MAX*QN_MAX]; for (i32 a2=0;a2<ndof3;a2++) ylc[a2]=0.0;
        real gb2[3*QN_MAX*QN_MAX*QN_MAX]; double gX2[QN_MAX*QN_MAX*QN_MAX][3];
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
          real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
          double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          double wdet=fabs(detJ)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k];
          Bp.allGradRef(xr,gb2);
          for (i32 a2=0;a2<ndof;a2++) for (i32 d2=0;d2<3;d2++)
            gX2[a2][d2]=Jinv[0][d2]*gb2[3*a2]+Jinv[1][d2]*gb2[3*a2+1]+Jinv[2][d2]*gb2[3*a2+2];
          double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
          for (i32 a2=0;a2<ndof;a2++) for(i32 i2=0;i2<3;i2++){ double ua=ul[3*a2+i2];
            gU[i2][0]+=ua*gX2[a2][0]; gU[i2][1]+=ua*gX2[a2][1]; gU[i2][2]+=ua*gX2[a2][2]; }
          double ep2[3][3],tr2=0;
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ep2[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
          tr2=ep2[0][0]+ep2[1][1]+ep2[2][2];
          double sg[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sg[i2][j2]=2*mu*ep2[i2][j2]+(i2==j2?lam*tr2:0);
          for (i32 a2=0;a2<ndof;a2++) for (i32 l=0;l<3;l++)
            ylc[3*a2+l]+=wdet*(sg[l][0]*gX2[a2][0]+sg[l][1]*gX2[a2][1]+sg[l][2]*gX2[a2][2]);
        }
        for (i32 a2=0;a2<ndof;a2++) scatter3(y,nod[a2],&ylc[3*a2]);
        continue;
      }
      qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
      for (i32 a=0;a<ndof;a++){ double c3[3]={(double)yl[3*a],(double)yl[3*a+1],(double)yl[3*a+2]};
        scatter3(y,nod[a],c3); } }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nBF;f++){ const i32*nod=&eNodeQ[(size_t)bf[f].e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX], yloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++) gather3(x,nod[a],&uloc[3*a]);
      sbmFace(f,uloc,yloc);
      for (i32 a=0;a<ndof;a++) scatter3(y,nod[a],&yloc[3*a]); }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      double uMP[2*3*QN_MAX*QN_MAX*QN_MAX], yMP[2*3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ gather3(x,nodM[a],&uMP[3*a]); gather3(x,nodP[a],&uMP[ndof3+3*a]); }
      const double*K=Kghost[F.d].data();
      for (i32 r=0;r<mG;r++){ double s=0; const double*Kr=&K[(size_t)r*mG]; for(i32 c=0;c<mG;c++) s+=Kr[c]*uMP[c]; yMP[r]=s; }
      for (i32 a=0;a<ndof;a++){ scatter3(y,nodM[a],&yMP[3*a]); scatter3(y,nodP[a],&yMP[ndof3+3*a]); } }
  };

  // -------------------------------------------------------------------------
  //  RHS (body force + shifted-Nitsche Dirichlet data) and Jacobi diagonal
  // -------------------------------------------------------------------------
  std::vector<real> bvec(nDofQ,(real)0);
  std::vector<double> diagN((size_t)3*nNodeQ,0.0);   // per-NODE diag (folded to dofs below)
  std::vector<double> diagX((size_t)3*nNodeQ,0.0);   // per-node xy,xz,yz cross terms
  #pragma omp parallel for schedule(dynamic,64)
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    double gXl[QN_MAX*QN_MAX*QN_MAX][3];
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
      double wv=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h*h*h, wb=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h;
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
      real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
      real fb[3]; prob.bodyForce(X[0],X[1],X[2],fb);
      if (cyl){
        double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
        double wdet=fabs(detJ)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k];
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
          gXl[a][d2]=Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2];
        for (i32 a=0;a<ndof;a++){
          double gsq=gXl[a][0]*gXl[a][0]+gXl[a][1]*gXl[a][1]+gXl[a][2]*gXl[a][2];
          { double c3[3]={wdet*fb[0]*vb[a],wdet*fb[1]*vb[a],wdet*fb[2]*vb[a]}; scatter3(bvec,nod[a],c3); }
          for (i32 l=0;l<3;l++){
            double dvl=wdet*(mu*(gsq+gXl[a][l]*gXl[a][l])+lam*gXl[a][l]*gXl[a][l]);
            #pragma omp atomic
            diagN[3*nod[a]+l]+=dvl; }
          double c01=wdet*(mu+lam)*gXl[a][0]*gXl[a][1];
          double c02=wdet*(mu+lam)*gXl[a][0]*gXl[a][2];
          double c12=wdet*(mu+lam)*gXl[a][1]*gXl[a][2];
          #pragma omp atomic
          diagX[3*nod[a]  ]+=c01;
          #pragma omp atomic
          diagX[3*nod[a]+1]+=c02;
          #pragma omp atomic
          diagX[3*nod[a]+2]+=c12; }
        continue;
      }
      for (i32 a=0;a<ndof;a++){
        double gsq=(double)gb[3*a]*gb[3*a]+(double)gb[3*a+1]*gb[3*a+1]+(double)gb[3*a+2]*gb[3*a+2];
        { double c3[3]={wv*fb[0]*vb[a],wv*fb[1]*vb[a],wv*fb[2]*vb[a]}; scatter3(bvec,nod[a],c3); }
        for (i32 l=0;l<3;l++){
          double dvl=wb*(mu*(gsq+(double)gb[3*a+l]*gb[3*a+l])+lam*(double)gb[3*a+l]*gb[3*a+l]);
          #pragma omp atomic
          diagN[3*nod[a]+l]+=dvl; }
        // 3x3 nodal-block cross terms (bulk only; penalty/ghost diag are
        // component-isotropic): K_aa[l][m] = wb*(mu+lam)*g_l*g_m for l != m
        double c01=wb*(mu+lam)*(double)gb[3*a]*gb[3*a+1];
        double c02=wb*(mu+lam)*(double)gb[3*a]*gb[3*a+2];
        double c12=wb*(mu+lam)*(double)gb[3*a+1]*gb[3*a+2];
        #pragma omp atomic
        diagX[3*nod[a]  ]+=c01;
        #pragma omp atomic
        diagX[3*nod[a]+1]+=c02;
        #pragma omp atomic
        diagX[3*nod[a]+2]+=c12; } } }
  i32 nNeuSkip=0;
  #pragma omp parallel for schedule(dynamic,16) reduction(+:nNeuSkip)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    const SElem&E=elems[F.e]; const i32*nod=&eNodeQ[(size_t)F.e*ndof];
    i32 d=F.d, t1=(d+1)%3, t2=(d+2)%3;
    double nsign=F.s?1.0:-1.0, nn[3]={0,0,0}; nn[d]=nsign;
    real gb[3*QN_MAX*QN_MAX*QN_MAX], sh[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      double hw=gq.w[q1]*gq.w[q2]*h;
      const double*g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      real dref[3]={(real)(g6[0]/h),(real)(g6[1]/h),(real)(g6[2]/h)};
      Bp.allGradRef(xr,gb); sbmShiftAll(Bp,Vm,xr,dref,sh);
      real XT[3]; ls.toPhys((E.ci+xr[0])*h+(real)g6[0],(E.cj+xr[1])*h+(real)g6[1],
                            (E.ck+xr[2])*h+(real)g6[2],XT[0],XT[1],XT[2]);
      size_t qp=(size_t)f*NQF + q1*gq.n + q2;
      if (neuPt[qp]){                        // ---- GAP-SBM Neumann (traction) RHS ----
        if (cyl){                            // ---- cylindrical: curved-metric gap + Nanson transfer ----
          double dref[3]={g6[0]/h,g6[1]/h,g6[2]/h};
          double dnR=dref[d]*nsign, wf=gq.w[q1]*gq.w[q2];
          double nu_c[3]={g6[3],g6[4],g6[5]};
          i32 td1=(d+1)%3, td2=(d+2)%3;
          real xrq[3]; xrq[d]=F.s?(real)1:(real)0; xrq[td1]=(real)gq.x[q1]; xrq[td2]=(real)gq.x[q2];
          real gbr[3*QN_MAX*QN_MAX*QN_MAX], vbr[QN_MAX*QN_MAX*QN_MAX];
          for (i32 qt=0;qt<gt.n && sbmGap;qt++){ double tau=gt.x[qt];
            real xrt[3]={(real)(xrq[0]+tau*dref[0]),(real)(xrq[1]+tau*dref[1]),(real)(xrq[2]+tau*dref[2])};
            double Jinv[3][3],detJ; metric(E,xrt,Jinv,detJ);
            Bp.allGradRef(xrt,gbr); Bp.allVal(xrt,vbr);
            double gX[QN_MAX*QN_MAX*QN_MAX][3];
            for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
              gX[a][d2]=Jinv[0][d2]*gbr[3*a]+Jinv[1][d2]*gbr[3*a+1]+Jinv[2][d2]*gbr[3*a+2];
            double W=fabs(detJ)*dnR*wf*gt.w[qt];
            real Xt[3]; ls.toPhys((E.ci+xrt[0])*h,(E.cj+xrt[1])*h,(E.ck+xrt[2])*h,Xt[0],Xt[1],Xt[2]);
            real fb[3]; prob.bodyForce(Xt[0],Xt[1],Xt[2],fb);
            for (i32 a=0;a<ndof;a++){
              double gsq=gX[a][0]*gX[a][0]+gX[a][1]*gX[a][1]+gX[a][2]*gX[a][2];
              { double c3[3]={W*fb[0]*vbr[a],W*fb[1]*vbr[a],W*fb[2]*vbr[a]}; scatter3(bvec,nod[a],c3); }
              for (i32 l=0;l<3;l++){ double dv=W*(mu*(gsq+gX[a][l]*gX[a][l])+lam*gX[a][l]*gX[a][l]);
                #pragma omp atomic
                diagN[3*nod[a]+l]+=dv; } }
          }
          const double*wN=&wNTab[qp*3];
          real xrT[3]={(real)(xrq[0]+dref[0]),(real)(xrq[1]+dref[1]),(real)(xrq[2]+dref[2])};
          double JinvT[3][3],detJT; metric(E,xrT,JinvT,detJT);
          double np[3]; for (i32 i2=0;i2<3;i2++) np[i2]=JinvT[0][i2]*nu_c[0]+JinvT[1][i2]*nu_c[1]+JinvT[2][i2]*nu_c[2];
          double nm=sqrt(np[0]*np[0]+np[1]*np[1]+np[2]*np[2]); if(nm<1e-30) nm=1;
          for (i32 i2=0;i2<3;i2++) np[i2]/=nm;
          real vtr[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xrT,vtr);
          real nuv[3]={(real)np[0],(real)np[1],(real)np[2]}, tv[3];
          prob.neumannData(XT[0],XT[1],XT[2],nuv,tv);
          double wS=wN[0]*np[0]+wN[1]*np[1]+wN[2]*np[2];   // = |wN| (n~.nu_phys)
          for (i32 a=0;a<ndof;a++){ double c3[3]={wS*tv[0]*vtr[a],wS*tv[1]*vtr[a],wS*tv[2]*vtr[a]}; scatter3(bvec,nod[a],c3); }
          continue;
        }
        double dd[3]={g6[0],g6[1],g6[2]}, nu[3]={g6[3],g6[4],g6[5]};
        double drf[3]={dd[0]/h,dd[1]/h,dd[2]/h};
        double dn=dd[0]*nn[0]+dd[1]*nn[1]+dd[2]*nn[2];      // signed gap thickness
        double sTot=dd[0]*nu[0]+dd[1]*nu[1]+dd[2]*nu[2];    // signed offset along nu
        double wf=gq.w[q1]*gq.w[q2];
        double trS=curvT.empty()?0:curvT[qp*2], detK=curvT.empty()?0:curvT[qp*2+1];
        // gap sliver: body-force load (full gap dV) + gap-stiffness Jacobi diagonal
        real gbt[3*QN_MAX*QN_MAX*QN_MAX], vbt[QN_MAX*QN_MAX*QN_MAX];
        for (i32 qt=0;qt<gt.n && sbmGap;qt++){ double tau=gt.x[qt];
          real xrt[3]={(real)(xr[0]+tau*drf[0]),(real)(xr[1]+tau*drf[1]),(real)(xr[2]+tau*drf[2])};
          Bp.allGradRef(xrt,gbt); Bp.allVal(xrt,vbt);
          double sq=tau*sTot, jg=1.0+trS*sq+detK*sq*sq;
          double wgb=dn*wf*gt.w[qt]*jg, wgl=h*h*wgb;
          real Xt[3]; ls.toPhys((E.ci+xr[0])*h+(real)(tau*dd[0]),(E.cj+xr[1])*h+(real)(tau*dd[1]),
                                (E.ck+xr[2])*h+(real)(tau*dd[2]),Xt[0],Xt[1],Xt[2]);
          real fb[3]; prob.bodyForce(Xt[0],Xt[1],Xt[2],fb);
          for (i32 a=0;a<ndof;a++){
            double gsq=(double)gbt[3*a]*gbt[3*a]+(double)gbt[3*a+1]*gbt[3*a+1]+(double)gbt[3*a+2]*gbt[3*a+2];
            { double c3[3]={wgl*fb[0]*vbt[a],wgl*fb[1]*vbt[a],wgl*fb[2]*vbt[a]}; scatter3(bvec,nod[a],c3); }
            for (i32 l=0;l<3;l++){ double dv=wgb*(mu*(gsq+(double)gbt[3*a+l]*gbt[3*a+l])+lam*(double)gbt[3*a+l]*gbt[3*a+l]);
              #pragma omp atomic
              diagN[3*nod[a]+l]+=dv; } } }
        // surface traction transfer at x_true = x~ + d:  int_Gamma~ (n~.nu) t(x_true).v
        real vtr[QN_MAX*QN_MAX*QN_MAX];
        real xrT[3]={(real)(xr[0]+drf[0]),(real)(xr[1]+drf[1]),(real)(xr[2]+drf[2])};
        Bp.allVal(xrT,vtr);
        real nuv[3]={(real)nu[0],(real)nu[1],(real)nu[2]}, tv[3];
        prob.neumannData(XT[0],XT[1],XT[2],nuv,tv);        // t(x_true) = sigma(u).nu
        double nvn=nn[0]*nu[0]+nn[1]*nu[1]+nn[2]*nu[2];
        double jS=1.0+trS*sTot+detK*sTot*sTot;
        double wS=h*h*wf*nvn*jS;
        for (i32 a=0;a<ndof;a++){ double c3[3]={wS*tv[0]*vtr[a],wS*tv[1]*vtr[a],wS*tv[2]*vtr[a]}; scatter3(bvec,nod[a],c3); }
        continue;
      }
      real g[3]; prob.dirichletData(XT[0],XT[1],XT[2],g);
      if (cyl){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2;
        const real*shc=&shTab[qp*ndof];
        const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
        const double*wN=&wNTab[qp*3];
        double dS=sqrt(wN[0]*wN[0]+wN[1]*wN[1]+wN[2]*wN[2]);
        double gnw=g[0]*wN[0]+g[1]*wN[1]+g[2]*wN[2];
        double penC=gammaD_/h;
        for (i32 a=0;a<ndof;a++){
          double gA[3]={gx[a],gy[a],gz[a]};
          double gan=gA[0]*wN[0]+gA[1]*wN[1]+gA[2]*wN[2];
          double ggb=g[0]*gA[0]+g[1]*gA[1]+g[2]*gA[2];
          { double c3[3];
            for (i32 l=0;l<3;l++)
              c3[l]=-(mu*(g[l]*gan+ggb*wN[l])+lam*gA[l]*gnw)+penC*g[l]*shc[a]*dS;
            scatter3(bvec,nod[a],c3); }
          for (i32 l=0;l<3;l++){
            double dvl=penC*shc[a]*shc[a]*dS;
            #pragma omp atomic
            diagN[3*nod[a]+l]+=dvl; } }
        continue;
      }
      double gn=g[0]*nn[0]+g[1]*nn[1]+g[2]*nn[2];
      for (i32 a=0;a<ndof;a++){ double gan=gb[3*a+d]*nsign;
        double ggb=g[0]*gb[3*a]+g[1]*gb[3*a+1]+g[2]*gb[3*a+2];
        { double c3[3];
          for (i32 l=0;l<3;l++){
            double rhs=-(mu*(g[l]*gan+ggb*nn[l])+lam*gb[3*a+l]*gn)+gammaD_*g[l]*sh[a];
            c3[l]=hw*rhs; }
          scatter3(bvec,nod[a],c3); }
        for (i32 l=0;l<3;l++){
          double dvl=hw*gammaD_*sh[a]*sh[a];
          #pragma omp atomic
          diagN[3*nod[a]+l]+=dvl; } } } }
  if (nNeuSkip) printf("warning: %d Neumann face points skipped (cyl gap not supported yet)\n", nNeuSkip);
  // ghost diagonal
  for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
    const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
    const double*K=Kghost[F.d].data();
    for (i32 a=0;a<ndof;a++) for (i32 l=0;l<3;l++){
      diagN[3*nodM[a]+l]+=K[(size_t)(3*a+l)*mG+(3*a+l)];
      diagN[3*nodP[a]+l]+=K[(size_t)(ndof3+3*a+l)*mG+(ndof3+3*a+l)]; } }
  std::vector<double> diagv(nDofQ,0.0);
  std::vector<double> diagXD((size_t)3*nDofNode,0.0);
  for (i32 nd=0;nd<nNodeQ;nd++){ i32 b=3*realIdx[nd];
    for (i32 l=0;l<3;l++) diagv[b+l]+=diagN[3*nd+l];
    for (i32 l=0;l<3;l++) diagXD[b+l]+=diagX[3*nd+l]; }
  for (i32 i=0;i<nDofQ;i++) if (diagv[i]<=0) diagv[i]=1.0;

  // -------------------------------------------------------------------------
  //  preconditioner: SBM_PC = jacobi (default) | bjac (3x3 nodal blocks) |
  //  poly[k] (k damped-Jacobi sweeps, default k=2).  bjac captures the
  //  (mu+lam) g_l g_m component coupling of elasticity that scalar Jacobi
  //  ignores; poly is a fixed linear polynomial in A (matrix-free friendly).
  // -------------------------------------------------------------------------
  const char* pcName = getenv("SBM_PC") ? getenv("SBM_PC") : "jacobi";
  i32 polyK = 2;
  if (pcName[0]=='p' && pcName[4]) polyK = atoi(pcName+4);
  std::vector<double> Binv;
  if (pcName[0]=='b'){
    Binv.assign((size_t)9*nDofNode,0.0);
    #pragma omp parallel for
    for (i32 nd=0;nd<nDofNode;nd++){
      double B[3][3]={{diagv[3*nd],diagXD[3*nd],diagXD[3*nd+1]},
                      {diagXD[3*nd],diagv[3*nd+1],diagXD[3*nd+2]},
                      {diagXD[3*nd+1],diagXD[3*nd+2],diagv[3*nd+2]}};
      double c00=B[1][1]*B[2][2]-B[1][2]*B[2][1];
      double c01=B[1][2]*B[2][0]-B[1][0]*B[2][2];
      double c02=B[1][0]*B[2][1]-B[1][1]*B[2][0];
      double det=B[0][0]*c00+B[0][1]*c01+B[0][2]*c02;
      double *Bi=&Binv[(size_t)9*nd];
      if (fabs(det)<1e-300){ Bi[0]=1.0/diagv[3*nd]; Bi[4]=1.0/diagv[3*nd+1]; Bi[8]=1.0/diagv[3*nd+2]; continue; }
      double id=1.0/det;
      Bi[0]=c00*id; Bi[3]=c01*id; Bi[6]=c02*id;
      Bi[1]=(B[0][2]*B[2][1]-B[0][1]*B[2][2])*id;
      Bi[4]=(B[0][0]*B[2][2]-B[0][2]*B[2][0])*id;
      Bi[7]=(B[0][1]*B[2][0]-B[0][0]*B[2][1])*id;
      Bi[2]=(B[0][1]*B[1][2]-B[0][2]*B[1][1])*id;
      Bi[5]=(B[0][2]*B[1][0]-B[0][0]*B[1][2])*id;
      Bi[8]=(B[0][0]*B[1][1]-B[0][1]*B[1][0])*id;
    }
    printf("precond: 3x3 nodal block-Jacobi\n");
  } else if (pcName[0]=='p') printf("precond: polynomial (%d damped-Jacobi sweeps)\n", polyK);
  else printf("precond: Jacobi\n");
  std::vector<real> ptmp1, ptmp2;
  if (pcName[0]=='p'){ ptmp1.assign(nDofQ,(real)0); ptmp2.assign(nDofQ,(real)0); }
  auto precond=[&](const std::vector<real>&r,std::vector<real>&z){
    if (pcName[0]=='b'){
      #pragma omp parallel for
      for (i32 nd=0;nd<nDofNode;nd++){ const double*Bi=&Binv[(size_t)9*nd];
        double r0=r[3*nd],r1=r[3*nd+1],r2=r[3*nd+2];
        z[3*nd  ]=(real)(Bi[0]*r0+Bi[1]*r1+Bi[2]*r2);
        z[3*nd+1]=(real)(Bi[3]*r0+Bi[4]*r1+Bi[5]*r2);
        z[3*nd+2]=(real)(Bi[6]*r0+Bi[7]*r1+Bi[8]*r2); }
    } else if (pcName[0]=='p'){
      const double om=2.0/3.0;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=(real)(om*r[i]/diagv[i]);
      for (i32 k2=0;k2<polyK;k2++){
        applyA(z,ptmp1);
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) z[i]+=(real)(om*(r[i]-ptmp1[i])/diagv[i]);
      }
    } else {
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=(real)(r[i]/diagv[i]);
    }
  };

  // ---- consistency probe (SBM_CONSISTENCY): ||b - A*I(u_exact)|| / ||b|| ------
  // Interpolates the manufactured solution at the nodes and measures the operator
  // residual.  This tests operator/RHS CONSISTENCY (the correctness of the gap-
  // Neumann assembly) WITHOUT needing the Krylov solve to converge -- so it
  // validates the port even on ill-conditioned geometry (the blade).  A correct
  // discretization gives a small residual that -> 0 with h; a wrong sign/scaling
  // in the Neumann terms shows up as an O(1) residual.
  if (getenv("SBM_CONSISTENCY")){
    std::vector<real> uh(nDofQ,(real)0), Ax(nDofQ,(real)0);
    for (i32 nd=0; nd<nNodeQ; nd++){
      real ue[3]; prob.exactU(nodeXQ[3*nd],nodeXQ[3*nd+1],nodeXQ[3*nd+2],ue);
      double u0=ue[0],u1=ue[1];
      if (rotFlag[nd]){ double a0=cph*u0+sph*u1, a1=-sph*u0+cph*u1; u0=a0; u1=a1; }  // store in master frame
      i32 b=3*realIdx[nd]; uh[b]=(real)u0; uh[b+1]=(real)u1; uh[b+2]=ue[2];
    }
    applyA(uh,Ax);
    double rn=0,bn=0; for (i32 i=0;i<nDofQ;i++){ double rr=(double)bvec[i]-(double)Ax[i]; rn+=rr*rr; bn+=(double)bvec[i]*bvec[i]; }
    printf("consist: ||b - A u_exact|| / ||b|| = %.4e   (||b|| = %.3e)\n", sqrt(rn/(bn+1e-300)), sqrt(bn));
  }

  // -------------------------------------------------------------------------
  //  solver.  Default: restarted Jacobi-GMRES(m) -- the SBM operator is
  //  NON-symmetric, and on the UNSTABILIZED operator CG broke down (pAp<0) and
  //  BiCGStab diverged (measured, M1).  BUT the production operator carries the
  //  ghost penalty and its measured asymmetry is ~5e-4, so the short-recurrence
  //  methods may work as inexact solvers here: SBM_SOLVER=bicgstab|cg|gmres
  //  selects, to settle that experimentally.  BiCGStab/CG are O(1) memory
  //  (~8/5 vectors) vs GMRES's (m+1) -- the memory driver of this path.
  // -------------------------------------------------------------------------
  std::vector<real> uv(nDofQ,(real)0);
  i32 it=0;
  // Default: BiCGStab -- verified on the stabilized operator (533 its, same L2 as
  // GMRES to 5 digits), principled for a non-symmetric system, O(1) memory
  // (8 vectors vs GMRES's m+1; the memory driver of this path at scale).
  // "cg" is faster still (inexact: rides the ~5e-4 asymmetry) but theoretically
  // unsound if the shift grows; "gmres" is the fallback if BiCGStab misbehaves.
  const char* sbmSolver = getenv("SBM_SOLVER") ? getenv("SBM_SOLVER") : "bicgstab";
  if (getenv("SBM_GPU")) {                    // ---- GPU Jacobi-BiCGStab (stage 1: Cartesian Dirichlet) ----
    const i32 nR=nDofQ, nN=3*nNodeQ, BS=256, GS=1024;
    auto cpI=[&](const i32*s,size_t m){ i32*d; cudaMallocManaged(&d,m*sizeof(i32)); memcpy(d,s,m*sizeof(i32)); return d; };
    auto cpR=[&](const real*s,size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); memcpy(d,s,m*sizeof(real)); return d; };
    auto cpD=[&](const double*s,size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); for(size_t i=0;i<m;i++) d[i]=(real)s[i]; return d; };
    auto alR=[&](size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); cudaMemset(d,0,m*sizeof(real)); return d; };
    i32 *d_eNode=cpI(eNodeQ.data(),(size_t)nE*ndof), *d_nMap=cpI(realIdx.data(),nNodeQ);
    char *d_nRot; cudaMallocManaged(&d_nRot,nNodeQ); for(i32 i=0;i<nNodeQ;i++) d_nRot[i]=rotFlag[i];
    i32 *d_bfE,*d_bfD,*d_bfS; cudaMallocManaged(&d_bfE,(size_t)nBF*4); cudaMallocManaged(&d_bfD,(size_t)nBF*4); cudaMallocManaged(&d_bfS,(size_t)nBF*4);
    for(i32 f=0;f<nBF;f++){ d_bfE[f]=bf[f].e; d_bfD[f]=bf[f].d; d_bfS[f]=bf[f].s; }
    i32 *d_gfM,*d_gfP,*d_gfD; cudaMallocManaged(&d_gfM,(size_t)nGFQ*4+4); cudaMallocManaged(&d_gfP,(size_t)nGFQ*4+4); cudaMallocManaged(&d_gfD,(size_t)nGFQ*4+4);
    for(i32 f=0;f<nGFQ;f++){ d_gfM[f]=gf[f].eM; d_gfP[f]=gf[f].eP; d_gfD[f]=gf[f].d; }
    real *d_shTab=cpR(shTab.data(),shTab.size()), *d_gbTab=cpR(gbTab.data(),gbTab.size()), *d_vbTab=cpR(vbTab.data(),vbTab.size());
    char *d_neuPt; cudaMallocManaged(&d_neuPt,(size_t)nBF*NQF+1); for(size_t i=0;i<(size_t)nBF*NQF;i++) d_neuPt[i]=neuPt[i];
    real *d_Kg0=cpD(Kghost[0].data(),(size_t)mG*mG), *d_Kg1=cpD(Kghost[1].data(),(size_t)mG*mG), *d_Kg2=cpD(Kghost[2].data(),(size_t)mG*mG);
    real *d_gqw; cudaMallocManaged(&d_gqw,gq.n*sizeof(real)); for(i32 q=0;q<gq.n;q++) d_gqw[q]=(real)gq.w[q];
    real *d_bvec=cpR(bvec.data(),nR), *d_diag=cpD(diagv.data(),nR), *d_uv=alR(nR);
    real *d_r=alR(nR),*d_rh=alR(nR),*d_v=alR(nR),*d_pd=alR(nR),*d_ph=alR(nR),*d_ss=alR(nR),*d_sh2=alR(nR),*d_tt=alR(nR),*d_Ax=alR(nR);
    real *d_xn=alR(nN),*d_yn=alR(nN); double *d_acc; cudaMalloc(&d_acc,sizeof(double));
    real *d_Kbulk; cudaMallocManaged(&d_Kbulk,(size_t)ndof3*ndof3*sizeof(real));
    { real ulc[3*QN_MAX*QN_MAX*QN_MAX], ylc[3*QN_MAX*QN_MAX*QN_MAX];   // constant affine bulk element matrix
      for (i32 c=0;c<ndof3;c++){ for(i32 i=0;i<ndof3;i++) ulc[i]=(i==c)?(real)1:(real)0;
        qpElemUncut(Bp,(real)mu,(real)lam,h,ulc,ylc);
        for (i32 r=0;r<ndof3;r++) d_Kbulk[(size_t)r*ndof3+c]=ylc[r]; } }
    SbmDev S; S.B=Bp; S.nE=nE; S.nBF=nBF; S.nGFQ=nGFQ; S.nNode=nNodeQ; S.ndof=ndof; S.ndof3=ndof3; S.mG=mG; S.NQF=NQF; S.gqn=gq.n;
    S.h=h; S.mu=(real)mu; S.lam=(real)lam; S.gammaD=(real)gammaD_; S.cph=(real)cph; S.sph=(real)sph;
    S.eNode=d_eNode; S.nMap=d_nMap; S.nRot=d_nRot; S.bfE=d_bfE; S.bfD=d_bfD; S.bfS=d_bfS; S.gfM=d_gfM; S.gfP=d_gfP; S.gfD=d_gfD;
    S.shTab=d_shTab; S.gbTab=d_gbTab; S.vbTab=d_vbTab; S.gqw=d_gqw; S.neuPt=d_neuPt; S.Kg[0]=d_Kg0; S.Kg[1]=d_Kg1; S.Kg[2]=d_Kg2; S.Kbulk=d_Kbulk;
    const i32 GBe=(nE<65535?nE:65535), GBg=(nGFQ<65535?nGFQ:65535), GBf=(nBF<65535?nBF:65535);
    const size_t shF=(size_t)(ndof3+7*NQF)*sizeof(real);
    auto gpuApply=[&](const real*x,real*y){
      sbmProlongK<<<GS,BS>>>(S,x,d_xn); sbmSetK<<<GS,BS>>>(d_yn,(real)0,nN);
      sbmBulkK<<<GBe,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn);
      if(nBF) sbmFaceK<<<GBf,128,shF>>>(S,d_xn,d_yn);
      if(nGFQ) sbmGhostK<<<GBg,256,(size_t)mG*sizeof(real)>>>(S,d_xn,d_yn);
      sbmSetK<<<GS,BS>>>(y,(real)0,nR); sbmRestrictK<<<GS,BS>>>(S,d_yn,y); cudaDeviceSynchronize(); };
    auto gpuDot=[&](const real*a,const real*b)->double{ cudaMemset(d_acc,0,sizeof(double)); sbmDotK<<<GS,BS>>>(a,b,nR,d_acc); double hv; cudaMemcpy(&hv,d_acc,sizeof(double),cudaMemcpyDeviceToHost); return hv; };
    gpuApply(d_uv,d_Ax);
    cudaMemcpy(d_r,d_bvec,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
    sbmAxpyK<<<GS,BS>>>(d_r,d_Ax,(real)-1,nR); cudaMemcpy(d_rh,d_r,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
    double bn=sqrt(gpuDot(d_bvec,d_bvec)); if(bn==0)bn=1; double rho=1,al=1,om=1;
    for(; it<cgMaxIt; it++){
      double rho2=gpuDot(d_rh,d_r); if(fabs(rho2)<1e-290){ printf("warning: GPU BiCGStab breakdown\n"); break; }
      double be=(rho2/rho)*(al/om); rho=rho2;
      sbmBicgPK<<<GS,BS>>>(d_pd,d_r,d_v,(real)be,(real)om,nR); sbmJacobiK<<<GS,BS>>>(d_ph,d_pd,d_diag,nR); cudaDeviceSynchronize();
      gpuApply(d_ph,d_v); double al2=rho/gpuDot(d_rh,d_v); al=al2;
      cudaMemcpy(d_ss,d_r,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); sbmAxpyK<<<GS,BS>>>(d_ss,d_v,(real)-al,nR); cudaDeviceSynchronize();
      double sn=sqrt(gpuDot(d_ss,d_ss));
      if(sn<=cgTol*bn){ sbmAxpyK<<<GS,BS>>>(d_uv,d_ph,(real)al,nR); cudaDeviceSynchronize(); it++; cgRes=sn/bn; break; }
      sbmJacobiK<<<GS,BS>>>(d_sh2,d_ss,d_diag,nR); cudaDeviceSynchronize(); gpuApply(d_sh2,d_tt);
      double ts=gpuDot(d_tt,d_ss), ttn=gpuDot(d_tt,d_tt); om=ts/ttn;
      sbmAxpyK<<<GS,BS>>>(d_uv,d_ph,(real)al,nR); sbmAxpyK<<<GS,BS>>>(d_uv,d_sh2,(real)om,nR);
      cudaMemcpy(d_r,d_ss,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); sbmAxpyK<<<GS,BS>>>(d_r,d_tt,(real)-om,nR); cudaDeviceSynchronize();
      double rn=sqrt(gpuDot(d_r,d_r)); cgRes=rn/bn;
      if(getenv("SBM_DBG")&&it%50==0) printf("    [gpu-bicgstab it=%d rres=%.3e]\n",it,rn/bn);
      if(rn<=cgTol*bn){ it++; break; }
    }
    cudaDeviceSynchronize(); memcpy(uv.data(),d_uv,(size_t)nR*sizeof(real));
    for(void*pp:{(void*)d_eNode,(void*)d_nMap,(void*)d_nRot,(void*)d_bfE,(void*)d_bfD,(void*)d_bfS,(void*)d_gfM,(void*)d_gfP,(void*)d_gfD,
                 (void*)d_shTab,(void*)d_gbTab,(void*)d_vbTab,(void*)d_neuPt,(void*)d_Kg0,(void*)d_Kg1,(void*)d_Kg2,(void*)d_gqw,
                 (void*)d_bvec,(void*)d_diag,(void*)d_uv,(void*)d_r,(void*)d_rh,(void*)d_v,(void*)d_pd,(void*)d_ph,(void*)d_ss,
                 (void*)d_sh2,(void*)d_tt,(void*)d_Ax,(void*)d_xn,(void*)d_yn,(void*)d_acc,(void*)d_Kbulk}) cudaFree(pp);
    sbmSolver="gpu"; printf("solver : GPU Jacobi-BiCGStab (kernels over the sparse-grid nodes)\n");
  } else if (sbmSolver[0]=='b') {                    // ---- Jacobi-BiCGStab ----
    std::vector<real> r(nDofQ),rh(nDofQ),v(nDofQ,(real)0),pd(nDofQ,(real)0),
                      ph(nDofQ),ss(nDofQ),sh2(nDofQ),tt(nDofQ),Ax(nDofQ);
    applyA(uv,Ax);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++){ r[i]=bvec[i]-Ax[i]; rh[i]=r[i]; }
    double bn=0; for (i32 i=0;i<nDofQ;i++) bn+=(double)bvec[i]*bvec[i]; bn=sqrt(bn); if(bn==0)bn=1;
    double rho=1,al=1,om=1; bool fail=false;
    for (; it<cgMaxIt; it++){
      double rho2=0; for (i32 i=0;i<nDofQ;i++) rho2+=(double)rh[i]*r[i];
      if (fabs(rho2)<1e-290){ printf("warning: BiCGStab breakdown rho=%.2e at it %d\n",rho2,it); fail=true; break; }
      double be=(rho2/rho)*(al/om); rho=rho2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=(real)(r[i]+be*(pd[i]-om*v[i]));
      precond(pd,ph);
      applyA(ph,v);
      double rhv=0; for (i32 i=0;i<nDofQ;i++) rhv+=(double)rh[i]*v[i];
      al=rho/rhv;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) ss[i]=(real)(r[i]-al*v[i]);
      double sn=0; for (i32 i=0;i<nDofQ;i++) sn+=(double)ss[i]*ss[i]; sn=sqrt(sn);
      if (sn<=cgTol*bn){
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) uv[i]+=(real)(al*ph[i]);
        it++; cgRes=sn/bn; break; }
      precond(ss,sh2);
      applyA(sh2,tt);
      double ts=0,ttn=0; for (i32 i=0;i<nDofQ;i++){ ts+=(double)tt[i]*ss[i]; ttn+=(double)tt[i]*tt[i]; }
      om=ts/ttn;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=(real)(al*ph[i]+om*sh2[i]); r[i]=(real)(ss[i]-om*tt[i]); }
      double rn=0; for (i32 i=0;i<nDofQ;i++) rn+=(double)r[i]*r[i]; rn=sqrt(rn);
      cgRes=rn/bn;
      if (getenv("SBM_DBG") && it%50==0)
        printf("    [bicgstab it=%d rres=%.3e rho=%.3e al=%.3e om=%.3e]\n",it,rn/bn,rho,al,om);
      if (rn<=cgTol*bn){ it++; break; }
    }
    if (fail) printf("warning: BiCGStab did not converge cleanly\n");
    printf("solver : Jacobi-BiCGStab (O(1) memory)\n");
  } else if (sbmSolver[0]=='l') {        // ---- BiCGStab(ell), Sleijpen-Fokkema ----
    // Left-Jacobi-preconditioned: operates on Ahat = M^-1 A, bhat = M^-1 b.
    // The ell-dimensional MR polynomial replaces plain BiCGStab's degree-1
    // stabilizer -- the standard cure for the "BiCGStab stalls where GMRES
    // works" signature (complex spectrum / non-normality).  Storage:
    // 2(ell+1)+2 vectors, still O(1) in restart-free memory.
    i32 ell = sbmSolver[1] ? atoi(sbmSolver+1) : 2; if (ell<1) ell=2; if (ell>8) ell=8;
    printf("solver : Jacobi-BiCGStab(%d)\n", ell);
    auto applyHat=[&](const std::vector<real>&xx,std::vector<real>&yy){
      applyA(xx,yy);
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) yy[i]=(real)(yy[i]/diagv[i]); };
    auto dotv=[&](const std::vector<real>&a2,const std::vector<real>&b2){
      double s2=0;
      #pragma omp parallel for reduction(+:s2)
      for (i32 i=0;i<nDofQ;i++) s2+=(double)a2[i]*b2[i]; return s2; };
    std::vector<std::vector<real>> rv(ell+1,std::vector<real>(nDofQ,(real)0)),
                                   uv2(ell+1,std::vector<real>(nDofQ,(real)0));
    std::vector<real> rt(nDofQ), bhat(nDofQ);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) bhat[i]=(real)(bvec[i]/diagv[i]);
    { std::vector<real> Ax0(nDofQ); applyHat(uv,Ax0);
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ rv[0][i]=(real)(bhat[i]-Ax0[i]); rt[i]=rv[0][i]; } }
    double bn=sqrt(dotv(bhat,bhat)); if (bn==0) bn=1;
    double rho=1, alpha=0, omega=1;
    std::vector<double> tau((size_t)(ell+1)*(ell+1),0.0), sg(ell+1,0.0),
                        gp(ell+1,0.0), ga(ell+1,0.0), gpp(ell+1,0.0);
    i32 nRestart=0;
    for (; it<cgMaxIt; ){
      double rn=sqrt(dotv(rv[0],rv[0]));
      cgRes=rn/bn;
      if (getenv("SBM_DBG") && it%50<2*ell)
        printf("    [bl%d it=%d rres=%.3e rho=%.3e om=%.3e]\n",ell,it,rn/bn,rho,omega);
      if (rn<=cgTol*bn) break;
      rho=-omega*rho;
      bool broke=false;
      for (i32 j=0;j<ell;j++){
        double rho1=dotv(rv[j],rt);
        if (fabs(rho)<1e-290||fabs(rho1)<1e-290){ broke=true; break; }
        double beta=alpha*rho1/rho; rho=rho1;
        for (i32 i2=0;i2<=j;i2++){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) uv2[i2][q]=(real)(rv[i2][q]-beta*uv2[i2][q]); }
        applyHat(uv2[j],uv2[j+1]); it++;
        double gm=dotv(uv2[j+1],rt);
        if (fabs(gm)<1e-290){ broke=true; break; }
        alpha=rho/gm;
        for (i32 i2=0;i2<=j;i2++){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) rv[i2][q]=(real)(rv[i2][q]-alpha*uv2[i2+1][q]); }
        applyHat(rv[j],rv[j+1]); it++;
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++) uv[q]+=(real)(alpha*uv2[0][q]);
      }
      if (broke){ // shadow restart
        nRestart++;
        if (nRestart>50){ printf("warning: BiCGStab(%d) exceeded restarts\n",ell); break; }
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){ rt[q]=rv[0][q]; uv2[0][q]=(real)0; }
        rho=1; alpha=0; omega=1; continue; }
      // MR part: minimize ||r0 - sum gamma_j r_j|| (modified Gram-Schmidt)
      for (i32 j=1;j<=ell;j++){
        for (i32 i2=1;i2<j;i2++){
          double t=dotv(rv[j],rv[i2])/sg[i2];
          tau[(size_t)i2*(ell+1)+j]=t;
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) rv[j][q]-=(real)(t*rv[i2][q]); }
        sg[j]=dotv(rv[j],rv[j]);
        if (sg[j]<1e-290){ broke=true; break; }
        gp[j]=dotv(rv[0],rv[j])/sg[j]; }
      if (broke){ nRestart++;
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){ rt[q]=rv[0][q]; uv2[0][q]=(real)0; }
        rho=1; alpha=0; omega=1; continue; }
      ga[ell]=gp[ell]; omega=ga[ell];
      for (i32 j=ell-1;j>=1;j--){ double s2=gp[j];
        for (i32 i2=j+1;i2<=ell;i2++) s2-=tau[(size_t)j*(ell+1)+i2]*ga[i2];
        ga[j]=s2; }
      for (i32 j=1;j<ell;j++){ double s2=ga[j+1];
        for (i32 i2=j+1;i2<ell;i2++) s2+=tau[(size_t)j*(ell+1)+i2]*ga[i2+1];
        gpp[j]=s2; }
      #pragma omp parallel for
      for (i32 q=0;q<nDofQ;q++){
        uv[q]+=(real)(ga[1]*rv[0][q]);
        rv[0][q]-=(real)(gp[ell]*rv[ell][q]);
        uv2[0][q]-=(real)(ga[ell]*uv2[ell][q]); }
      for (i32 j=1;j<ell;j++){
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){
          uv2[0][q]-=(real)(ga[j]*uv2[j][q]);
          uv[q]+=(real)(gpp[j]*rv[j][q]);
          rv[0][q]-=(real)(gp[j]*rv[j][q]); } }
    }
    if (nRestart) printf("solver : %d shadow restarts\n",nRestart);
  } else if (sbmSolver[0]=='c') {             // ---- Jacobi-CG (inexact: operator is ~1e-4 asymmetric) ----
    std::vector<real> r(nDofQ),z(nDofQ),pd(nDofQ),Ap(nDofQ);
    applyA(uv,Ap);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) r[i]=bvec[i]-Ap[i];
    precond(r,z);
    pd=z;
    double rz=0; for (i32 i=0;i<nDofQ;i++) rz+=(double)r[i]*z[i];
    double bn=0; for (i32 i=0;i<nDofQ;i++) bn+=(double)bvec[i]*bvec[i]; bn=sqrt(bn); if(bn==0)bn=1;
    for (; it<cgMaxIt; it++){
      applyA(pd,Ap);
      double pAp=0; for (i32 i=0;i<nDofQ;i++) pAp+=(double)pd[i]*Ap[i];
      if (!(pAp>0)){ printf("warning: CG indefinite pAp=%.3e at it %d\n",pAp,it); break; }
      double al=rz/pAp;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=(real)(al*pd[i]); r[i]-=(real)(al*Ap[i]); }
      double rn=0; for (i32 i=0;i<nDofQ;i++) rn+=(double)r[i]*r[i]; rn=sqrt(rn);
      cgRes=rn/bn;
      if (rn<=cgTol*bn){ it++; break; }
      precond(r,z);
      double rz2=0; for (i32 i=0;i<nDofQ;i++) rz2+=(double)r[i]*z[i];
      double be=rz2/rz; rz=rz2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=(real)(z[i]+be*pd[i]);
    }
    printf("solver : Jacobi-CG (inexact on the ~1e-4-asymmetric operator)\n");
  } else {
    const i32 m=200;
    const double tol=(double)cgTol;
    std::vector<std::vector<real>> V(m+1,std::vector<real>(nDofQ));
    std::vector<double> Hm((size_t)(m+1)*m,0.0),cs(m,0.0),sn(m,0.0),ss(m+1,0.0),yv(m,0.0);
    std::vector<real> w(nDofQ),Ax(nDofQ),rr(nDofQ);
    bool conv=false; double beta0=0;
    for (i32 outer=0;outer<2000&&!conv&&it<cgMaxIt;outer++){
      applyA(uv,Ax);
      { std::vector<real> t2(nDofQ);
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) t2[i]=(real)(bvec[i]-Ax[i]);
        precond(t2,rr); }
      double beta=0;
      #pragma omp parallel for reduction(+:beta)
      for (i32 i=0;i<nDofQ;i++) beta+=(double)rr[i]*rr[i];
      beta=sqrt(beta);
      if (outer==0) beta0=beta>0?beta:1;
      cgRes=beta/beta0;
      if (getenv("SBM_DBG")) printf("    [gmres outer=%d it=%d pres=%.3e]\n",outer,it,beta/beta0);
      if (beta<=tol*beta0){ conv=true; break; }
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) V[0][i]=(real)(rr[i]/beta);
      std::fill(ss.begin(),ss.end(),0.0); ss[0]=beta; i32 jj=0;
      for (i32 j=0;j<m&&it<cgMaxIt;j++){ jj=j; it++;
        applyA(V[j],Ax);
        precond(Ax,w);
        for (i32 i2=0;i2<=j;i2++){ double hij=0;
          #pragma omp parallel for reduction(+:hij)
          for (i32 q=0;q<nDofQ;q++) hij+=(double)w[q]*V[i2][q];
          Hm[(size_t)i2*m+j]=hij;
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) w[q]-=(real)(hij*V[i2][q]); }
        double hj1=0;
        #pragma omp parallel for reduction(+:hj1)
        for (i32 q=0;q<nDofQ;q++) hj1+=(double)w[q]*w[q];
        hj1=sqrt(hj1); Hm[(size_t)(j+1)*m+j]=hj1;
        if (hj1>1e-300){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) V[j+1][q]=(real)(w[q]/hj1); }
        for (i32 i2=0;i2<j;i2++){ double t=cs[i2]*Hm[(size_t)i2*m+j]+sn[i2]*Hm[(size_t)(i2+1)*m+j];
          Hm[(size_t)(i2+1)*m+j]=-sn[i2]*Hm[(size_t)i2*m+j]+cs[i2]*Hm[(size_t)(i2+1)*m+j]; Hm[(size_t)i2*m+j]=t; }
        double d0=Hm[(size_t)j*m+j],d1=Hm[(size_t)(j+1)*m+j],r2=sqrt(d0*d0+d1*d1); if(r2<1e-300)r2=1e-300;
        cs[j]=d0/r2; sn[j]=d1/r2; Hm[(size_t)j*m+j]=cs[j]*d0+sn[j]*d1; Hm[(size_t)(j+1)*m+j]=0;
        double t=cs[j]*ss[j]; ss[j+1]=-sn[j]*ss[j]; ss[j]=t;
        if (fabs(ss[j+1])<=tol*beta0) break; }
      i32 sz=jj+1;
      for (i32 i2=sz-1;i2>=0;i2--){ double s2=ss[i2];
        for (i32 k2=i2+1;k2<sz;k2++) s2-=Hm[(size_t)i2*m+k2]*yv[k2];
        yv[i2]=s2/Hm[(size_t)i2*m+i2]; }
      for (i32 i2=0;i2<sz;i2++){
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++) uv[q]+=(real)(yv[i2]*V[i2][q]); }
    }
  }
  cgIters=it;

  // -------------------------------------------------------------------------
  //  errors over the surrogate (tensor GLL; every element is full)
  // -------------------------------------------------------------------------
  double l2e=0,l2n=0,ene=0,enn=0,vol=0;
  #pragma omp parallel for schedule(dynamic,64) reduction(+:l2e,l2n,ene,enn,vol)
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
      double dw=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h*h*h;
      double JinvE[3][3],detJE=0;
      if (cyl){ metric(elems[e],xr,JinvE,detJE); dw=fabs(detJE)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; }
      Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
      double uh[3]={0,0,0}, gh[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      double ulocE[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++) gather3(uv,nod[a],&ulocE[3*a]);
      for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++){ double ua=ulocE[3*a+l];
        uh[l]+=ua*vb[a];
        if (cyl){
          double gXa[3];
          for (i32 d2=0;d2<3;d2++) gXa[d2]=JinvE[0][d2]*gb[3*a]+JinvE[1][d2]*gb[3*a+1]+JinvE[2][d2]*gb[3*a+2];
          gh[l][0]+=ua*gXa[0]; gh[l][1]+=ua*gXa[1]; gh[l][2]+=ua*gXa[2];
        } else {
          gh[l][0]+=ua*gb[3*a]/h; gh[l][1]+=ua*gb[3*a+1]/h; gh[l][2]+=ua*gb[3*a+2]/h; } }
      real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
      real ue[3]; prob.exactU(X[0],X[1],X[2],ue);
      real ge[3][3]; prob.exactGradU(X[0],X[1],X[2],ge);
      vol+=dw;
      for (i32 l=0;l<3;l++){ double dd=uh[l]-ue[l]; l2e+=dd*dd*dw; l2n+=(double)ue[l]*ue[l]*dw; }
      double ee[3][3],se[3][3],tre=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ee[i2][j2]=0.5*((gh[i2][j2]-ge[i2][j2])+(gh[j2][i2]-ge[j2][i2]));
      tre=ee[0][0]+ee[1][1]+ee[2][2];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) se[i2][j2]=2*mu*ee[i2][j2]+(i2==j2?lam*tre:0);
      double en=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) en+=se[i2][j2]*ee[i2][j2]; ene+=en*dw;
      double eeE[3][3],seE[3][3],trE=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eeE[i2][j2]=0.5*(ge[i2][j2]+ge[j2][i2]);
      trE=eeE[0][0]+eeE[1][1]+eeE[2][2];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) seE[i2][j2]=2*mu*eeE[i2][j2]+(i2==j2?lam*trE:0);
      double enE=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) enE+=seE[i2][j2]*eeE[i2][j2]; enn+=enE*dw; } }
  errL2=sqrt(l2e); normL2=sqrt(l2n); errEnergy=sqrt(ene); normEnergy=sqrt(enn);
  volOmega=vol; areaGamma=0;

  // -------------------------------------------------------------------------
  //  report + VTU
  // -------------------------------------------------------------------------
  double ms=(sbNowUs()-t0)/1000.0;
  printf("active : %d surrogate elements (all full), %d boundary faces, %d ghost faces\n", nE, nBF, nGFQ);
  printf("dofs   : %d nodes -> %d unknowns   h = %.6g   p = %d\n", nNodeQ, nDofQ, (double)h, p);
  printf("geom   : |Omega~| = %.8g  (surrogate: centre-in cells; differs from Omega by O(h))\n", volOmega);
  printf("solve  : %s %d iters, rel res %.2e   (%.0f ms)\n", sbmSolver, cgIters, cgRes, ms);
  if ((prob.caseId==CASE_MMS || prob.caseId==CASE_MMS_CYL) && normL2>0)
    printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)   [over Omega~]\n",
           errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);

  if (wantVtu && !outTag.empty()){ mkdir("output",0755);
    std::string fn="output/"+outTag+"_femsbm.vtu"; std::ofstream os(fn.c_str(),std::ios::binary);
    i64 nSub=(i64)nE*p*p*p; static const i32 HEX[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    os<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
      <<"  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nNodeQ<<"\" NumberOfCells=\""<<nSub<<"\">\n";
    os<<"      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (i32 nd=0;nd<nNodeQ;nd++) os<<(float)nodeXQ[3*nd]<<" "<<(float)nodeXQ[3*nd+1]<<" "<<(float)nodeXQ[3*nd+2]<<"\n";
    os<<"        </DataArray>\n      </Points>\n";
    os<<"      <PointData Vectors=\"u\">\n        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (i32 nd=0;nd<nNodeQ;nd++){ double u3[3]; gather3(uv,nd,u3);
      os<<(float)u3[0]<<" "<<(float)u3[1]<<" "<<(float)u3[2]<<"\n"; }
    os<<"        </DataArray>\n      </PointData>\n";
    os<<"      <Cells>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      for (i32 sk=0;sk<p;sk++)for(i32 sj=0;sj<p;sj++)for(i32 si=0;si<p;si++){
        for (i32 v=0;v<8;v++){ i32 ii=si+HEX[v][0],jj=sj+HEX[v][1],kk=sk+HEX[v][2]; os<<nod[ii+n*(jj+n*kk)]<<" "; } os<<"\n"; } }
    os<<"        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (i64 s=0;s<nSub;s++) os<<8*(s+1)<<"\n";
    os<<"        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (i64 s=0;s<nSub;s++) os<<"12\n";
    os<<"        </DataArray>\n      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
    printf("wrote %s\n", fn.c_str());
  }
}

// =====================================================================
//  DENSITY-MASK ELASTICITY SOLVER (--method density).  Ersatz-material:
//  the geometry is a smooth tanh(phi) density rho in [rmin,1] on the FULL grid
//  (no surrogate/cut/shift), the operator is the SPD rho-weighted bulk
//  elasticity, solved with GPU CG.  Stage: Q2, rho=1 box-MMS patch test
//  (validates the operator+CG+Dirichlet); the tanh mask + interface |grad rho|
//  traction is the next milestone.
// =====================================================================
void CutFemSolver::runDensity(void) {
  const i32 p = femOrder<2?2:femOrder;
  QpBasis Bp; Bp.init(p);
  const i32 n=p+1, ndof=n*n*n, ndof3=3*ndof;
  const real h = cellSize();
  const double mu=prob.mu, lam=prob.lam;
  if (ls.coordMode!=0){ printf("ERROR: density path is Cartesian-only for now\n"); return; }
  const double beta=getenv("DENS_BETA")?atof(getenv("DENS_BETA")):2.0;   // interface ~1 element (mesh-resolvable sweet spot)
  const double rmin=getenv("DENS_RMIN")?atof(getenv("DENS_RMIN")):1e-3;
  const bool rho1=getenv("DENS_RHO1")!=nullptr;   // rho=1 patch/box test
  const bool mmsI=getenv("DENS_MMS")!=nullptr;    // add the -int(sigma.grad rho).v MMS consistency term
  printf("density: ersatz tanh(phi) mask elasticity, Q%d Cartesian (SPD, GPU-CG)  beta=%.2f rmin=%.0e%s\n",
         p, beta, rmin, rho1?"  [rho=1 box test]":"");
  initialize(); buildMesh();

  // ---- active cells: everything in the active blocks (the geometry's box) -----
  struct DE{ i32 ci,cj,ck; };
  std::vector<DE> elems;
  const i32 nB=hashTable.nKeys;
  for (i32 b=0;b<nB;b++){ u64 loc=bLocList[b]; if(loc==kEmpty) continue;
    i32 ib,jb,kb; sbBlockDec(loc,ib,jb,kb);
    for (i32 cz=0;cz<blockSize;cz++)for(i32 cy=0;cy<blockSize;cy++)for(i32 cx=0;cx<blockSize;cx++)
      elems.push_back({ib*blockSize+cx,jb*blockSize+cy,kb*blockSize+cz}); }
  i32 nE=(i32)elems.size();
  if(!nE){ printf("ERROR: no active cells\n"); return; }

  // ---- Q2 node numbering (continuous; no cyclic tie -> dof == node) -----------
  std::unordered_map<u64,i32> nodeId; nodeId.reserve((size_t)nE*ndof);
  std::vector<i32> eNode((size_t)nE*ndof); std::vector<i32> nI,nJ,nK;
  i32 nNode=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 i=a%n,j=(a/n)%n,k=a/(n*n);
    i32 I=p*elems[e].ci+i,J=p*elems[e].cj+j,K=p*elems[e].ck+k;
    u64 key=sbKey(I,J,K); auto it=nodeId.find(key); i32 id;
    if(it==nodeId.end()){ id=nNode++; nodeId[key]=id; nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    else id=it->second;
    eNode[(size_t)e*ndof+a]=id;
  }
  const i32 nDof=3*nNode;

  // ---- Dirichlet nodes: on the boundary of the active set (a face with no active
  //      neighbour cell); pinned to u_exact -> far-field + rigid-mode fix ---------
  std::unordered_map<u64,i32> cellId; cellId.reserve((size_t)nE*2);
  for (i32 e=0;e<nE;e++) cellId[sbKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;
  std::vector<char> nodeDir(nNode,0);
  for (i32 e=0;e<nE;e++){ i32 c[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++) for (i32 s=0;s<2;s++){ i32 nb[3]={c[0],c[1],c[2]}; nb[d]+=s?1:-1;
      if (cellId.count(sbKey(nb[0],nb[1],nb[2]))) continue;         // interior face
      i32 t1=(d+1)%3,t2=(d+2)%3;                                    // mark the p+1 x p+1 face nodes
      for (i32 b2=0;b2<n;b2++) for (i32 a2=0;a2<n;a2++){ i32 lo[3]; lo[d]=s?n-1:0; lo[t1]=a2; lo[t2]=b2;
        i32 la=lo[0]+n*(lo[1]+n*lo[2]); nodeDir[eNode[(size_t)e*ndof+la]]=1; } } }

  // node physical coords + density per element
  std::vector<real> nodeX((size_t)3*nNode);
  for (i32 nd=0;nd<nNode;nd++){ real X0,X1,X2; ls.toPhys((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p,X0,X1,X2);
    nodeX[3*nd]=X0; nodeX[3*nd+1]=X1; nodeX[3*nd+2]=X2; }
  // phi at the SOLUTION nodes (same points as u) -> a Q2 phi field -> sub-element rho.
  std::vector<real> phiN(nNode);
  for (i32 nd=0;nd<nNode;nd++) phiN[nd]=ls.phi((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p);
  auto mask=[&](double ph)->double{ return rho1?1.0:rmin+(1.0-rmin)*0.5*(1.0-tanh(beta*ph/(double)h)); };

  // constant affine bulk element matrix (GLL) -- shared by all nearly-constant cells
  std::vector<real> Kbulk((size_t)ndof3*ndof3);
  { real ulc[3*QN_MAX*QN_MAX*QN_MAX], ylc[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 c=0;c<ndof3;c++){ for(i32 i=0;i<ndof3;i++) ulc[i]=(i==c)?(real)1:(real)0;
      qpElemUncut(Bp,(real)mu,(real)lam,h,ulc,ylc);
      for (i32 r=0;r<ndof3;r++) Kbulk[(size_t)r*ndof3+c]=ylc[r]; } }

  // high-res Gauss rule for the band cells (points, weights, basis values -- reused)
  const i32 nG = getenv("DENS_QG")?atoi(getenv("DENS_QG")):std::max(2*p+2,(i32)ceil(3.0*beta));  // finer -> resolve the sharp band (setup-only cost)
  GaussRule gr = gaussLegendre(nG); const i32 nq3=nG*nG*nG;
  std::vector<real> gpts((size_t)nq3*3), gvb((size_t)nq3*ndof), gwt(nq3), ggb((size_t)nq3*3*ndof);
  { i32 q=0; for(i32 k=0;k<nG;k++)for(i32 j=0;j<nG;j++)for(i32 i=0;i<nG;i++,q++){
      gpts[3*q]=gr.x[i]; gpts[3*q+1]=gr.x[j]; gpts[3*q+2]=gr.x[k]; gwt[q]=gr.w[i]*gr.w[j]*gr.w[k];
      real xr[3]={gr.x[i],gr.x[j],gr.x[k]}; real vb[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xr,vb);
      real gb[3*QN_MAX*QN_MAX*QN_MAX]; Bp.allGradRef(xr,gb);   // reference gradients ONCE (constant)
      for(i32 a=0;a<ndof;a++) gvb[(size_t)q*ndof+a]=vb[a];
      for(i32 a=0;a<3*ndof;a++) ggb[(size_t)q*3*ndof+a]=gb[a]; } }

  // ---- SPLIT: classify by nodal rho range.  Near-constant -> rho_e*Kbulk (matOff
  //      0, scale rho_e).  Band (sharp transition) -> a HIGH-RES element matrix
  //      built ONCE in Kall.  CG matvec is then a dense matvec, never re-quadratures.
  const double rtol = getenv("DENS_RTOL")?atof(getenv("DENS_RTOL")):1e-4;   // const-cell rho tolerance = the error floor (looser -> less memory, more error)
  std::vector<i32> matOff(nE,0), bandE; std::vector<real> scale(nE,(real)1), rhoE(nE);
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNode[(size_t)e*ndof];
    double rlo=2,rhi=-1,ravg=0; i32 q=0;
    for(i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++,q++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]}; real vb[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xr,vb);
      double phq=0; for(i32 a=0;a<ndof;a++) phq+=(double)phiN[nod[a]]*vb[a]; double rq=mask(phq);
      rlo=fmin(rlo,rq); rhi=fmax(rhi,rq); ravg+=rq*Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; }
    rhoE[e]=(real)ravg;
    if (!rho1 && (rhi-rlo)>rtol) bandE.push_back(e);
    else { matOff[e]=0; scale[e]=(real)ravg; } }
  const i32 nBand=(i32)bandE.size();
  std::vector<real> Kall(Kbulk); Kall.resize((size_t)(1+nBand)*ndof3*ndof3);
  for (i32 b=0;b<nBand;b++){ matOff[bandE[b]]=(i32)((size_t)(1+b)*ndof3*ndof3); scale[bandE[b]]=(real)1; }
  #pragma omp parallel for schedule(dynamic,8)
  for (i32 b=0;b<nBand;b++){ i32 e=bandE[b]; const i32*nod=&eNode[(size_t)e*ndof];
    std::vector<real> wr(nq3);
    for(i32 q=0;q<nq3;q++){ double phq=0; const real*vbq=&gvb[(size_t)q*ndof];
      for(i32 a=0;a<ndof;a++) phq+=(double)phiN[nod[a]]*vbq[a]; wr[q]=(real)(gwt[q]*mask(phq)); }
    real *Ke=&Kall[(size_t)matOff[e]]; real ulc[3*QN_MAX*QN_MAX*QN_MAX], ylc[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 c=0;c<ndof3;c++){ for(i32 i=0;i<ndof3;i++) ulc[i]=(i==c)?(real)1:(real)0;
      qpElemCoreG(Bp,(real)mu,(real)lam,h,ggb.data(),wr.data(),nq3,ulc,ylc);   // precomputed grads (no re-derivation)
      for (i32 r=0;r<ndof3;r++) Ke[(size_t)r*ndof3+c]=ylc[r]; } }

  // ---- RHS (high-res in band cells) + diagonal (from the precomputed matrices) --
  std::vector<real> bvec(nDof,(real)0); std::vector<double> diag(nDof,0.0);
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNode[(size_t)e*ndof];
    if (matOff[e]==0){                                   // constant cell: rho_e * GLL body load
      real re=scale[e]; i32 q=0;
      for(i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++,q++){
        real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]}; real vb[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xr,vb);
        real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
        real fb[3]; prob.bodyForce(X[0],X[1],X[2],fb); double w3=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*(double)h*h*h*re;
        for(i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++) bvec[3*nod[a]+l]+=(real)(w3*fb[l]*vb[a]); }
      for(i32 r=0;r<ndof3;r++) diag[3*nod[r/3]+(r%3)]+=(double)re*Kbulk[(size_t)r*ndof3+r];
    } else {                                             // band cell: high-res body load (+ MMS term)
      const real *Ke=&Kall[(size_t)matOff[e]];
      for(i32 q=0;q<nq3;q++){ const real*vbq=&gvb[(size_t)q*ndof];
        double phq=0; for(i32 a=0;a<ndof;a++) phq+=(double)phiN[nod[a]]*vbq[a]; double rq=mask(phq);
        real xr[3]={gpts[3*q],gpts[3*q+1],gpts[3*q+2]};
        real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
        real fb[3]; prob.bodyForce(X[0],X[1],X[2],fb); double w3=gwt[q]*(double)h*h*h*rq;
        for(i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++) bvec[3*nod[a]+l]+=(real)(w3*fb[l]*vbq[a]);
        if (mmsI){ real gbq[3*QN_MAX*QN_MAX*QN_MAX]; Bp.allGradRef(xr,gbq);
          double gphi[3]={0,0,0}; for(i32 a=0;a<ndof;a++){ double pa=phiN[nod[a]]; gphi[0]+=pa*gbq[3*a];gphi[1]+=pa*gbq[3*a+1];gphi[2]+=pa*gbq[3*a+2]; }
          double th=tanh(beta*phq/(double)h), rp=-(1.0-rmin)*0.5*beta/(double)h*(1.0-th*th);
          double grho[3]={rp*gphi[0]/h,rp*gphi[1]/h,rp*gphi[2]/h};
          real gu[3][3]; prob.exactGradU(X[0],X[1],X[2],gu); double trg=gu[0][0]+gu[1][1]+gu[2][2],sig[3][3];
          for(i32 a2=0;a2<3;a2++)for(i32 b2=0;b2<3;b2++) sig[a2][b2]=mu*(gu[a2][b2]+gu[b2][a2])+(a2==b2?lam*trg:0.0);
          double sgr[3]; for(i32 l=0;l<3;l++) sgr[l]=sig[l][0]*grho[0]+sig[l][1]*grho[1]+sig[l][2]*grho[2];
          double w3n=gwt[q]*(double)h*h*h;
          for(i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++) bvec[3*nod[a]+l]-=(real)(w3n*sgr[l]*vbq[a]); } }
      for(i32 r=0;r<ndof3;r++) diag[3*nod[r/3]+(r%3)]+=(double)Ke[(size_t)r*ndof3+r];
    } }
  // apply strong Dirichlet: u_D at Dirichlet dofs; b zeroed there; diag=1
  std::vector<char> dofDir(nDof,0);
  for (i32 nd=0;nd<nNode;nd++) if(nodeDir[nd]) for(i32 l=0;l<3;l++){ dofDir[3*nd+l]=1; }
  std::vector<real> uD(nDof,(real)0);
  for (i32 nd=0;nd<nNode;nd++) if(nodeDir[nd]){ real ue[3]; prob.exactU(nodeX[3*nd],nodeX[3*nd+1],nodeX[3*nd+2],ue);
    for(i32 l=0;l<3;l++){ uD[3*nd+l]=ue[l]; } }
  for (i32 i=0;i<nDof;i++) if(dofDir[i]) diag[i]=1.0;

  printf("density: %d cells (%d band), %d nodes -> %d dofs, %d Dirichlet;  band quad %d^3, matrices %.0f MB\n",
         nE, nBand, nNode, nDof, (i32)std::count(nodeDir.begin(),nodeDir.end(),(char)1),
         nG, (double)Kall.size()*sizeof(real)/1e6);

  // ================= GPU CG (SPD rho-weighted operator) =====================
  const i32 BS=256, GS=1024, GBe=(nE<65535?nE:65535);
  auto cpI=[&](const i32*s,size_t m){ i32*d; cudaMallocManaged(&d,m*sizeof(i32)); memcpy(d,s,m*sizeof(i32)); return d; };
  auto cpR=[&](const real*s,size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); memcpy(d,s,m*sizeof(real)); return d; };
  auto cpD=[&](const double*s,size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); for(size_t i=0;i<m;i++) d[i]=(real)s[i]; return d; };
  auto alR=[&](size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); cudaMemset(d,0,m*sizeof(real)); return d; };
  i32 *d_eNode=cpI(eNode.data(),(size_t)nE*ndof);
  char *d_dofDir; cudaMallocManaged(&d_dofDir,nDof); for(i32 i=0;i<nDof;i++) d_dofDir[i]=dofDir[i];
  real *d_Kall=cpR(Kall.data(),Kall.size()); i32 *d_matOff=cpI(matOff.data(),nE); real *d_scale=cpR(scale.data(),nE);
  real *d_b=cpR(bvec.data(),nDof), *d_diag=cpD(diag.data(),nDof), *d_uD=cpR(uD.data(),nDof);
  real *d_u=alR(nDof),*d_r=alR(nDof),*d_z=alR(nDof),*d_pv=alR(nDof),*d_Ap=alR(nDof);
  double *d_acc; cudaMalloc(&d_acc,sizeof(double));
  SbmDev S; S.B=Bp; S.nE=nE; S.ndof=ndof; S.ndof3=ndof3; S.h=h; S.mu=(real)mu; S.lam=(real)lam;
  S.eNode=d_eNode; S.dofDir=d_dofDir;
  auto zeroDir=[&](real*x){ densZeroDirK<<<GS,BS>>>(d_dofDir,x,nDof); };
  auto apply=[&](const real*x,real*y){ sbmSetK<<<GS,BS>>>(y,(real)0,nDof);
    densMatvecK<<<GBe,128,(size_t)ndof3*sizeof(real)>>>(S,d_matOff,d_scale,d_Kall,x,y); zeroDir(y); cudaDeviceSynchronize(); };
  auto dot=[&](const real*a,const real*b)->double{ cudaMemset(d_acc,0,sizeof(double));
    sbmDotK<<<GS,BS>>>(a,b,nDof,d_acc); double hv; cudaMemcpy(&hv,d_acc,sizeof(double),cudaMemcpyDeviceToHost); return hv; };
  // u = uD (Dirichlet) ; r = b - A u ; zeroDir(r)
  cudaMemcpy(d_u,d_uD,(size_t)nDof*sizeof(real),cudaMemcpyDeviceToDevice);
  apply(d_u,d_Ap);
  cudaMemcpy(d_r,d_b,(size_t)nDof*sizeof(real),cudaMemcpyDeviceToDevice);
  sbmAxpyK<<<GS,BS>>>(d_r,d_Ap,(real)-1,nDof); zeroDir(d_r); cudaDeviceSynchronize();
  sbmJacobiK<<<GS,BS>>>(d_z,d_r,d_diag,nDof); zeroDir(d_z);
  cudaMemcpy(d_pv,d_z,(size_t)nDof*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
  double bn=sqrt(dot(d_b,d_b)); if(bn==0)bn=1; double rz=dot(d_r,d_z); i32 it=0; double rn=0;
  for(; it<cgMaxIt; it++){
    apply(d_pv,d_Ap); double pAp=dot(d_pv,d_Ap); if(!(pAp>0)){ printf("density: CG pAp=%.2e break\n",pAp); break; }
    double al=rz/pAp;
    sbmAxpyK<<<GS,BS>>>(d_u,d_pv,(real)al,nDof); sbmAxpyK<<<GS,BS>>>(d_r,d_Ap,(real)-al,nDof); cudaDeviceSynchronize();
    rn=sqrt(dot(d_r,d_r)); if(rn<=cgTol*bn){ it++; break; }
    sbmJacobiK<<<GS,BS>>>(d_z,d_r,d_diag,nDof); zeroDir(d_z); cudaDeviceSynchronize();
    double rz2=dot(d_r,d_z), be=rz2/rz; rz=rz2;
    sbmSetK<<<GS,BS>>>(d_Ap,(real)0,0); // no-op keep
    sbmAxpyK<<<GS,BS>>>(d_pv,d_pv,(real)(be-1),nDof);    // pv = be*pv (via +=(be-1)pv)
    sbmAxpyK<<<GS,BS>>>(d_pv,d_z,(real)1,nDof); cudaDeviceSynchronize(); // pv += z  -> pv = z + be*pv
    if(getenv("DENS_DBG")&&it%50==0) printf("    [dens-cg it=%d rres=%.3e]\n",it,rn/bn);
  }
  std::vector<real> uh(nDof); cudaDeviceSynchronize(); memcpy(uh.data(),d_u,(size_t)nDof*sizeof(real));
  printf("solve  : density-CG %d iters, rel res %.2e\n", it, rn/bn);

  // ---- L2 error over cells with rho>0.5 (the solid) --------------------------
  double l2e=0,l2n=0;
  for (i32 e=0;e<nE;e++){ if(rhoE[e]<=0.5) continue; const i32*nod=&eNode[(size_t)e*ndof];
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]}; real vb[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xr,vb);
      double w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h*h*h;
      double u3[3]={0,0,0}; for(i32 a=0;a<ndof;a++)for(i32 l=0;l<3;l++) u3[l]+=uh[3*nod[a]+l]*vb[a];
      real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
      real ue[3]; prob.exactU(X[0],X[1],X[2],ue);
      for(i32 l=0;l<3;l++){ double dd=u3[l]-ue[l]; l2e+=dd*dd*w; l2n+=(double)ue[l]*ue[l]*w; } } }
  printf("error  : L2 %.6e (rel %.4e)   [density, over rho>0.5, %d cells h=%.5f]\n",
         sqrt(l2e), sqrt(l2e/(l2n+1e-300)), nE, (double)h);
  for(void*pp:{(void*)d_eNode,(void*)d_dofDir,(void*)d_Kall,(void*)d_matOff,(void*)d_scale,(void*)d_b,(void*)d_diag,
               (void*)d_uD,(void*)d_u,(void*)d_r,(void*)d_z,(void*)d_pv,(void*)d_Ap,(void*)d_acc}) cudaFree(pp);
}
