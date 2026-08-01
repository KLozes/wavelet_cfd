//
// Higher-order path of the CutFEM solver -- CutFemSolver::runIga():
// IMMERSED ISOGEOMETRIC analysis on uniform C^{p-1} B-splines.
//
// Production realization of the method verified standalone in QpMms.cu, reusing
// the sparse octree + oracle (buildMesh) for the active block set and the
// verified modules IgaBasis / PolyFit / SayeQuad / IgaElem.  The p=1 GPU path
// (run()) is untouched; run() dispatches here when femOrder>=2.
//
// Two geometry modes:
//   * CARTESIAN (coordMode 0): the element is the cube [0,h]^3, Jacobian h*I;
//     interior elements share one reference matrix, cut elements carry their own.
//   * CYLINDRICAL (coordMode 1): the element is a curved (r, r*theta', z) brick.
//     The physical Jacobian is evaluated ANALYTICALLY from the cylindrical map at
//     every quadrature point (isoparametric-exact), so EVERY element carries its
//     own matrix.  The Nitsche surface uses the Nanson-transformed physical
//     normal and area, and the one-pitch sector is closed by the cyclic node tie
//     u(theta+pitch)=R(pitch)u(theta) -- a rotation applied at gather/scatter.
//
// The level set is sampled by the oracle at the (p+1)^3 GLL solution nodes; the
// Qp normal-derivative-jump ghost penalty (computational frame, l=1..p) on cut-
// element faces restores conditioning and the O(h^{p+1}) rate on slivers.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include <cusolverSp.h>
#include <cusparse.h>

#include "CutFemSolver.cuh"
#include "IgaBasis.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "IgaElem.h"

static inline u64 qpKey(i32 I, i32 J, i32 K) {
  return (u64)I | ((u64)J << 21) | ((u64)K << 42);
}
static inline u64 qpCellKey(i32 i, i32 j, i32 k) { return qpKey(i,j,k); }
static inline void qpBlockDec(u64 loc, i32 &i, i32 &j, i32 &k) {
  k = (i32)((loc >> 40) & ((1u<<20)-1)) - 1;
  j = (i32)((loc >> 20) & ((1u<<20)-1)) - 1;
  i = (i32)( loc        & ((1u<<20)-1)) - 1;
}
static long qpNowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
// invert a 3x3, return det
__host__ __device__ static inline double inv3(const double J[3][3], double Ji[3][3]) {
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
}


// =====================================================================
//  GPU solve for the Cartesian cut-cell operator (CUT_GPU=1).  Reuses runIga's
//  host setup verbatim -- the continuous nodal dofs live in flat device arrays,
//  and the operator is: interior cells share one reference matrix K_ref (dense
//  matvec), cut cells RE-QUADRATURE from their stored Saye rule (volume stiffness
//  via qpElemCoreSaye + Nitsche surface), ghost penalty via per-axis K_ghost.
//  Symmetric -> Jacobi-PCG.  All data device-resident: nodal u/phi, the Saye
//  integration rules (CSR pools), K_ref, K_ghost.  Stage: p2/p3 Cartesian.
// dense SPD inverse (Cholesky A=L L^T, then A^-1 by column solves); in-place, A->A^-1.
// returns false if not positive-definite (caller regularizes with a diagonal shift).
// cyclic Jacobi eigensolver for a small dense SYMMETRIC matrix (Rayleigh-Ritz projection).
// A is overwritten; w gets the eigenvalues ASCENDING, V the eigenvectors in COLUMNS.
static void jacobiEig(double *A, double *V, double *w, int n) {
  for (int i=0;i<n;i++) for (int j=0;j<n;j++) V[i*n+j]=(i==j)?1.0:0.0;
  for (int sweep=0; sweep<80; sweep++) {
    double off=0; for(int i=0;i<n;i++)for(int j=i+1;j<n;j++) off+=A[i*n+j]*A[i*n+j];
    if (off < 1e-26) break;
    for (int p2=0;p2<n;p2++) for (int q=p2+1;q<n;q++) {
      double apq=A[p2*n+q]; if (fabs(apq)<1e-30) continue;
      double th=(A[q*n+q]-A[p2*n+p2])/(2*apq);
      double t=(th>=0?1.0:-1.0)/(fabs(th)+sqrt(th*th+1.0));
      double c=1.0/sqrt(t*t+1.0), s=t*c;
      for (int k=0;k<n;k++){ double akp=A[k*n+p2], akq=A[k*n+q];
        A[k*n+p2]=c*akp-s*akq; A[k*n+q]=s*akp+c*akq; }
      for (int k=0;k<n;k++){ double apk=A[p2*n+k], aqk=A[q*n+k];
        A[p2*n+k]=c*apk-s*aqk; A[q*n+k]=s*apk+c*aqk; }
      for (int k=0;k<n;k++){ double vkp=V[k*n+p2], vkq=V[k*n+q];
        V[k*n+p2]=c*vkp-s*vkq; V[k*n+q]=s*vkp+c*vkq; }
    } }
  for (int i=0;i<n;i++) w[i]=A[i*n+i];
  for (int i=0;i<n;i++){ int m=i; for(int j=i+1;j<n;j++) if(w[j]<w[m]) m=j;
    if (m!=i){ double t=w[i]; w[i]=w[m]; w[m]=t;
      for(int k=0;k<n;k++){ double v=V[k*n+i]; V[k*n+i]=V[k*n+m]; V[k*n+m]=v; } } }
}
static bool invertSPD(double *A, int n) {
  std::vector<double> L((size_t)n*n, 0.0);
  for (int j=0;j<n;j++){ double d=A[(size_t)j*n+j];
    for (int k=0;k<j;k++) d-=L[(size_t)j*n+k]*L[(size_t)j*n+k];
    if (d<=0) return false; d=sqrt(d); L[(size_t)j*n+j]=d;
    for (int i=j+1;i<n;i++){ double s=A[(size_t)i*n+j]; for(int k=0;k<j;k++) s-=L[(size_t)i*n+k]*L[(size_t)j*n+k]; L[(size_t)i*n+j]=s/d; } }
  std::vector<double> y(n);
  for (int c=0;c<n;c++){                          // solve A x = e_c  -> column c of A^-1
    for (int i=0;i<n;i++){ double s=(i==c)?1.0:0.0; for(int k=0;k<i;k++) s-=L[(size_t)i*n+k]*y[k]; y[i]=s/L[(size_t)i*n+i]; }
    for (int i=n-1;i>=0;i--){ double s=y[i]; for(int k=i+1;k<n;k++) s-=L[(size_t)k*n+i]*A[(size_t)k*n+c]; A[(size_t)i*n+c]=s/L[(size_t)i*n+i]; } }
  return true;
}
// =====================================================================
struct CutDev {
  IgaBasis B; i32 nE,nCut,nGFQ,nNode,ndof,ndof3,mG;
  real h, mu, lam, gammaD;
  const i32 *eNode, *nMap; const char *nRot; real cph, sph;
  const i32 *intList, *cutElem;                    // interior cells; cut-cell -> element
  const SayeNode *volP, *surfP; const i32 *volOff, *surfOff; const char *surfDir;
  const i32 *gfM, *gfP, *gfD; const real *Kref, *Kg[3];
  i32 cyl; LevelSet ls; const i32 *eCijk, *eCut;   // cylindrical: analytic metric per element (ci,cj,ck), per-element cut index
  const real *volJ, *surfJ;                        // precomputed metric [Jinv(9),detJ(1)] at Saye vol/surf points (shared across p-levels)
  i32 sp[3];                                       // h-MG: fine cells agglomerated per coarse cell (1,1,1 on the fine grid).
                                                   // Cell extent is (sp[d]*h); eCijk holds the FINE index of the low corner.
};
#define CUT_QN3 (QN_MAX*QN_MAX*QN_MAX)
#define CUT_STRIDE (blockIdx.x*blockDim.x+threadIdx.x)
__global__ void cutProlongK(CutDev S,const real*x,real*xn){ for(i32 nd=CUT_STRIDE;nd<S.nNode;nd+=gridDim.x*blockDim.x){
  i32 m=S.nMap[nd]; real x0=x[3*m],x1=x[3*m+1],x2=x[3*m+2];
  if(S.nRot[nd]){xn[3*nd]=S.cph*x0-S.sph*x1;xn[3*nd+1]=S.sph*x0+S.cph*x1;xn[3*nd+2]=x2;} else{xn[3*nd]=x0;xn[3*nd+1]=x1;xn[3*nd+2]=x2;} } }
__global__ void cutRestrictK(CutDev S,const real*yn,real*y){ for(i32 nd=CUT_STRIDE;nd<S.nNode;nd+=gridDim.x*blockDim.x){
  i32 m=S.nMap[nd]; real c0=yn[3*nd],c1=yn[3*nd+1],c2=yn[3*nd+2],a0,a1,a2;
  if(S.nRot[nd]){a0=S.cph*c0+S.sph*c1;a1=-S.sph*c0+S.cph*c1;a2=c2;} else{a0=c0;a1=c1;a2=c2;}
  atomicAdd(&y[3*m],a0);atomicAdd(&y[3*m+1],a1);atomicAdd(&y[3*m+2],a2); } }
__global__ void cutSetK(real*x,real v,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)x[i]=v;}
__global__ void cutAxpyK(real*y,const real*x,real a,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[i]+=a*x[i];}
__global__ void cutJacK(real*z,const real*r,const real*d,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)z[i]=r[i]/d[i];}
// real<->double bridges: the assembled CSR / IC(0) factor are always fp64 even in the fp32 build
__global__ void cutD2K(double*y,const real*x,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[i]=(double)x[i];}
__global__ void cutR2K(real*y,const double*x,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[i]=(real)x[i];}
// multicolor permutation bridges (perm[newIdx]=oldIdx)
__global__ void cutD2PK(double*y,const real*x,const i32*p,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[i]=(double)x[p[i]];}
__global__ void cutR2PK(real*y,const double*x,const i32*p,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[p[i]]=(real)x[i];}
__global__ void cutDMulK(double*y,const double*d,i32 n){for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)y[i]*=d[i];}
// dense linear combination of a flat block of k vectors: out = sum_j coef[j]*S[j]  (LOBPCG basis rotate)
__global__ void cutCombK(real*out,const real*Sflat,const real*coef,i32 k,i32 n){
  for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x){ real s=0;
    for(i32 j=0;j<k;j++) s+=coef[j]*Sflat[(size_t)j*n+i]; out[i]=s; } }
// CSR mat-vec (used for the modal mass matrix; one row per thread is plenty at these sizes)
__global__ void cutCsrMvK(const i32*rp,const i32*ci,const real*va,const real*x,real*y,i32 n){
  for(i32 r=CUT_STRIDE;r<n;r+=gridDim.x*blockDim.x){ real s=0;
    for(i32 k=rp[r];k<rp[r+1];k++) s+=va[k]*x[ci[k]]; y[r]=s; } }
// nodal 3x3 block-Jacobi: z_node = Binv_node * r_node  (Binv = per-dof-node inverse block, 9 reals row-major)
__global__ void cutBJacK(real*z,const real*r,const real*Binv,i32 nDofNode){ for(i32 nd=CUT_STRIDE;nd<nDofNode;nd+=gridDim.x*blockDim.x){
  const real*B=Binv+(size_t)9*nd; real r0=r[3*nd],r1=r[3*nd+1],r2=r[3*nd+2];
  z[3*nd]=B[0]*r0+B[1]*r1+B[2]*r2; z[3*nd+1]=B[3]*r0+B[4]*r1+B[5]*r2; z[3*nd+2]=B[6]*r0+B[7]*r1+B[8]*r2; } }
__global__ void cutDotK(const real*a,const real*b,i32 n,double*out){ double s=0; for(i32 i=CUT_STRIDE;i<n;i+=gridDim.x*blockDim.x)s+=(double)a[i]*b[i];
  __shared__ double sh[256]; sh[threadIdx.x]=s; __syncthreads();
  for(i32 o=blockDim.x/2;o>0;o>>=1){ if(threadIdx.x<o)sh[threadIdx.x]+=sh[threadIdx.x+o]; __syncthreads(); }
  if(threadIdx.x==0) atomicAdd(out,sh[0]); }
// interior cells: yn += K_ref xn  (shared constant matrix, thread per dof)
__global__ void cutInteriorK(CutDev S,const real*xn,real*yn){ i32 ndof=S.ndof,m3=S.ndof3; extern __shared__ real uls[];
  for(i32 ii=blockIdx.x;ii<S.nE-S.nCut;ii+=gridDim.x){ i32 e=S.intList[ii]; const i32*nod=S.eNode+(size_t)e*ndof;
    for(i32 c=threadIdx.x;c<m3;c+=blockDim.x) uls[c]=xn[3*nod[c/3]+(c%3)]; __syncthreads();
    for(i32 r=threadIdx.x;r<m3;r+=blockDim.x){ const real*Kr=S.Kref+(size_t)r*m3; real acc=0; for(i32 c=0;c<m3;c++)acc+=Kr[c]*uls[c];
      atomicAdd(&yn[3*nod[r/3]+(r%3)],acc); } __syncthreads(); } }
// cut cells: re-quadrature the Saye rule (volume stiffness + Nitsche surface).
// BLOCK per cut cell, THREAD per quadrature point: ul/yl live in shared memory
// (no per-thread element-size spill), points accumulate into yl via shared atomics.
__global__ void cutCellK(CutDev S,const real*xn,real*yn){ IgaBasis B=S.B; i32 ndof=S.ndof,ndof3=S.ndof3;
  extern __shared__ real sh[]; real*ul=sh,*yl=sh+ndof3;                 // ul[ndof3] + yl[ndof3]
  for(i32 c=blockIdx.x;c<S.nCut;c+=gridDim.x){ i32 e=S.cutElem[c]; const i32*nod=S.eNode+(size_t)e*ndof;
    for(i32 a=threadIdx.x;a<ndof;a+=blockDim.x){ i32 nd=nod[a]; ul[3*a]=xn[3*nd]; ul[3*a+1]=xn[3*nd+1]; ul[3*a+2]=xn[3*nd+2]; }
    for(i32 i=threadIdx.x;i<ndof3;i+=blockDim.x) yl[i]=(real)0;
    __syncthreads();
    // volume stiffness (per-point form of qpElemCoreSaye; single h folded into wq)
    i32 v0=S.volOff[c], nv=S.volOff[c+1]-v0;
    for(i32 q=threadIdx.x;q<nv;q+=blockDim.x){ const SayeNode*vn=&S.volP[v0+q];
      real gb[3*CUT_QN3], xr[3]={vn->x[0],vn->x[1],vn->x[2]}; B.allGradRef(xr,gb);
      real gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++){ real ua=ul[3*a+i2];
        gU[i2][0]+=ua*gb[3*a]; gU[i2][1]+=ua*gb[3*a+1]; gU[i2][2]+=ua*gb[3*a+2]; }
      real eps[3][3],tr; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=(real)0.5*(gU[i2][j2]+gU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; real sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*S.mu*eps[i2][j2]+(i2==j2?S.lam*tr:0);
      real wq=vn->w*S.h;
      for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++) atomicAdd(&yl[3*a+i2],wq*(sig[i2][0]*gb[3*a]+sig[i2][1]*gb[3*a+1]+sig[i2][2]*gb[3*a+2])); }
    // Nitsche surface (Dirichlet points only)
    i32 s0=S.surfOff[c], ns=S.surfOff[c+1]-s0; const SayeNode*sn=S.surfP+s0;
    for(i32 q=threadIdx.x;q<ns;q+=blockDim.x){ if(!S.surfDir[s0+q]) continue;
      real gb[3*CUT_QN3], vb[CUT_QN3];
      real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]}, nn[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
      B.allGradRef(xr,gb); B.allVal(xr,vb); real hw=sn[q].w*S.h;
      real uval[3]={0,0,0}, gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++){ real ua=ul[3*a+i2]; uval[i2]+=ua*vb[a];
        gU[i2][0]+=ua*gb[3*a]; gU[i2][1]+=ua*gb[3*a+1]; gU[i2][2]+=ua*gb[3*a+2]; }
      real eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=(real)0.5*(gU[i2][j2]+gU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; real sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*S.mu*eps[i2][j2]+(i2==j2?S.lam*tr:0);
      real tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
      real un=uval[0]*nn[0]+uval[1]*nn[1]+uval[2]*nn[2];
      for(i32 a=0;a<ndof;a++){ real gan=gb[3*a]*nn[0]+gb[3*a+1]*nn[1]+gb[3*a+2]*nn[2];
        real ugb=uval[0]*gb[3*a]+uval[1]*gb[3*a+1]+uval[2]*gb[3*a+2];
        for(i32 l=0;l<3;l++){ real t1=-tu[l]*vb[a], t2=-(S.mu*(uval[l]*gan+ugb*nn[l])+S.lam*gb[3*a+l]*un), t3=S.gammaD*uval[l]*vb[a];
          atomicAdd(&yl[3*a+l],hw*(t1+t2+t3)); } } }
    __syncthreads();
    for(i32 a=threadIdx.x;a<ndof;a+=blockDim.x){ i32 nd=nod[a]; atomicAdd(&yn[3*nd],yl[3*a]); atomicAdd(&yn[3*nd+1],yl[3*a+1]); atomicAdd(&yn[3*nd+2],yl[3*a+2]); }
    __syncthreads();
  } }
__device__ inline void cutLoadJ(const real*J,double Jinv[3][3],double& detJ){ for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++)Jinv[i][j]=J[3*i+j]; detJ=J[9]; }
// analytic cylindrical metric at a reference point of element e: Jinv (=Jref^-1), detJ
__device__ inline void cutMetric(const CutDev& S,i32 e,const real xr[3],double Jinv[3][3],double& detJ){
  const i32* cj=S.eCijk+(size_t)3*e; double h=S.h;
  // cell extent per direction (h-MG coarse cells are anisotropic: sp = (2,1,2) semi-coarsened)
  double hx=S.sp[0]*h, hy=S.sp[1]*h, hz=S.sp[2]*h;
  if(!S.cyl){   // Cartesian: constant diagonal metric (exact for a hx x hy x hz brick)
    for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) Jinv[i][j]=0;
    Jinv[0][0]=1.0/hx; Jinv[1][1]=1.0/hy; Jinv[2][2]=1.0/hz; detJ=hx*hy*hz; return; }
  double r=cj[0]*h+hx*xr[0]+S.ls.org[0], s=cj[1]*h+hy*xr[1]+S.ls.org[1], z=cj[2]*h+hz*xr[2]+S.ls.org[2];
  double th=s/S.ls.rRef+(double)S.ls.thc((real)z), thp=(double)S.ls.thcSlope((real)z), ct=cos(th), st=sin(th);
  double J[3][3]={{ hx*ct, hy*(-r*st/S.ls.rRef), hz*(-r*st*thp) },{ hx*st, hy*( r*ct/S.ls.rRef), hz*( r*ct*thp) },{ 0,0,hz }};
  detJ=inv3(J,Jinv); }
// CYLINDRICAL operator: block per element (ALL elements re-quadrature the analytic metric;
// interior = tensor-GLL, cut = Saye), + Nanson Nitsche on cut Dirichlet faces.
__global__ void cutCylK(CutDev S,const real*xn,real*yn){ IgaBasis B=S.B; i32 ndof=S.ndof,ndof3=S.ndof3,n=B.n;
  extern __shared__ real sh[]; real*ul=sh,*yl=sh+ndof3;
  for(i32 e=blockIdx.x;e<S.nE;e+=gridDim.x){ const i32*nod=S.eNode+(size_t)e*ndof; i32 c=S.eCut[e]; bool cut=(c>=0);
    for(i32 a=threadIdx.x;a<ndof;a+=blockDim.x){ i32 nd=nod[a]; ul[3*a]=xn[3*nd]; ul[3*a+1]=xn[3*nd+1]; ul[3*a+2]=xn[3*nd+2]; }
    for(i32 i=threadIdx.x;i<ndof3;i+=blockDim.x) yl[i]=(real)0;
    __syncthreads();
    i32 nv = cut ? (S.volOff[c+1]-S.volOff[c]) : ndof;
    for(i32 q=threadIdx.x;q<nv;q+=blockDim.x){ real xr[3],wq;
      if(cut){ const SayeNode*vp=&S.volP[S.volOff[c]+q]; xr[0]=vp->x[0];xr[1]=vp->x[1];xr[2]=vp->x[2]; wq=vp->w; }
      else{ i32 i=q%n,j=(q/n)%n,k=q/(n*n); xr[0]=B.t[i];xr[1]=B.t[j];xr[2]=B.t[k]; wq=B.wq[i]*B.wq[j]*B.wq[k]; }
      double Jinv[3][3],detJ; if(cut&&S.volJ) cutLoadJ(S.volJ+(size_t)10*(S.volOff[c]+q),Jinv,detJ); else cutMetric(S,e,xr,Jinv,detJ);
      real gb[3*CUT_QN3], gxl[3*CUT_QN3]; B.allGradRef(xr,gb);
      for(i32 a=0;a<ndof;a++){ gxl[3*a]=(real)(Jinv[0][0]*gb[3*a]+Jinv[1][0]*gb[3*a+1]+Jinv[2][0]*gb[3*a+2]);
        gxl[3*a+1]=(real)(Jinv[0][1]*gb[3*a]+Jinv[1][1]*gb[3*a+1]+Jinv[2][1]*gb[3*a+2]); gxl[3*a+2]=(real)(Jinv[0][2]*gb[3*a]+Jinv[1][2]*gb[3*a+1]+Jinv[2][2]*gb[3*a+2]); }
      double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++){ real ua=ul[3*a+i2]; gU[i2][0]+=ua*gxl[3*a]; gU[i2][1]+=ua*gxl[3*a+1]; gU[i2][2]+=ua*gxl[3*a+2]; }
      double eps[3][3],tr; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3]; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*S.mu*eps[i2][j2]+(i2==j2?S.lam*tr:0);
      double wdet=fabs(detJ)*wq;
      for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++) atomicAdd(&yl[3*a+i2],(real)(wdet*(sig[i2][0]*gxl[3*a]+sig[i2][1]*gxl[3*a+1]+sig[i2][2]*gxl[3*a+2])));
    }
    if(cut){ i32 s0=S.surfOff[c], ns=S.surfOff[c+1]-s0; const SayeNode*sn=S.surfP+s0;
      i32 spm=S.sp[0]<S.sp[1]?S.sp[0]:S.sp[1]; spm=spm<S.sp[2]?spm:S.sp[2];   // Nitsche penalty ~ 1/h_elem: smallest extent (coercivity-safe)
      double penC=(double)S.gammaD/(spm*S.h);
      for(i32 q=threadIdx.x;q<ns;q+=blockDim.x){ if(!S.surfDir[s0+q])continue; real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
        double Jinv[3][3],detJ; if(S.surfJ) cutLoadJ(S.surfJ+(size_t)10*(s0+q),Jinv,detJ); else cutMetric(S,e,xr,Jinv,detJ);
        double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]},nraw[3]; for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
        double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if(nmag<=0)continue;
        double nP[3]={nraw[0]/nmag,nraw[1]/nmag,nraw[2]/nmag}, dS=fabs(detJ)*nmag*sn[q].w;
        real gb[3*CUT_QN3],vb[CUT_QN3],gxl[3*CUT_QN3]; B.allGradRef(xr,gb); B.allVal(xr,vb);
        for(i32 a=0;a<ndof;a++){ gxl[3*a]=(real)(Jinv[0][0]*gb[3*a]+Jinv[1][0]*gb[3*a+1]+Jinv[2][0]*gb[3*a+2]);
          gxl[3*a+1]=(real)(Jinv[0][1]*gb[3*a]+Jinv[1][1]*gb[3*a+1]+Jinv[2][1]*gb[3*a+2]); gxl[3*a+2]=(real)(Jinv[0][2]*gb[3*a]+Jinv[1][2]*gb[3*a+1]+Jinv[2][2]*gb[3*a+2]); }
        double uval[3]={0,0,0}, gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        for(i32 a=0;a<ndof;a++)for(i32 i2=0;i2<3;i2++){ real ua=ul[3*a+i2]; uval[i2]+=ua*vb[a]; gU[i2][0]+=ua*gxl[3*a]; gU[i2][1]+=ua*gxl[3*a+1]; gU[i2][2]+=ua*gxl[3*a+2]; }
        double eps[3][3],tr; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
        tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3]; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*S.mu*eps[i2][j2]+(i2==j2?S.lam*tr:0);
        double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nP[0]+sig[i2][1]*nP[1]+sig[i2][2]*nP[2];
        double un=uval[0]*nP[0]+uval[1]*nP[1]+uval[2]*nP[2];
        for(i32 a=0;a<ndof;a++){ double gan=gxl[3*a]*nP[0]+gxl[3*a+1]*nP[1]+gxl[3*a+2]*nP[2], ugb=uval[0]*gxl[3*a]+uval[1]*gxl[3*a+1]+uval[2]*gxl[3*a+2];
          for(i32 l=0;l<3;l++){ double t1=-tu[l]*vb[a], t2=-(S.mu*(uval[l]*gan+ugb*nP[l])+S.lam*gxl[3*a+l]*un), t3=penC*uval[l]*vb[a]; atomicAdd(&yl[3*a+l],(real)(dS*(t1+t2+t3))); } } } }
    __syncthreads();
    for(i32 a=threadIdx.x;a<ndof;a+=blockDim.x){ i32 nd=nod[a]; atomicAdd(&yn[3*nd],yl[3*a]); atomicAdd(&yn[3*nd+1],yl[3*a+1]); atomicAdd(&yn[3*nd+2],yl[3*a+2]); }
    __syncthreads();
  } }
// ghost penalty (thread per row of the dense mG x mG per-axis matrix)
__global__ void cutGhostK(CutDev S,const real*xn,real*yn){ i32 ndof=S.ndof,ndof3=S.ndof3,mG=S.mG; extern __shared__ real uMP[];
  for(i32 f=blockIdx.x;f<S.nGFQ;f+=gridDim.x){ i32 eM=S.gfM[f],eP=S.gfP[f],dd=S.gfD[f];
    const i32*nodM=S.eNode+(size_t)eM*ndof,*nodP=S.eNode+(size_t)eP*ndof; const real*K=S.Kg[dd];
    for(i32 c=threadIdx.x;c<mG;c+=blockDim.x){ if(c<ndof3){i32 a=c/3;uMP[c]=xn[3*nodM[a]+(c%3)];} else{i32 cc=c-ndof3,a=cc/3;uMP[c]=xn[3*nodP[a]+(cc%3)];} }
    __syncthreads();
    for(i32 r=threadIdx.x;r<mG;r+=blockDim.x){ const real*Kr=K+(size_t)r*mG; real acc=0; for(i32 c=0;c<mG;c++)acc+=Kr[c]*uMP[c];
      if(r<ndof3){i32 a=r/3;atomicAdd(&yn[3*nodM[a]+(r%3)],acc);} else{i32 cc=r-ndof3,a=cc/3;atomicAdd(&yn[3*nodP[a]+(cc%3)],acc);} }
    __syncthreads(); } }
// additive-Schwarz apply: zn += Kinv_e * rn over an element list (dense matvec, thread per row).
// perBlock=1 -> per-element inverse invPool[ii]; perBlock=0 -> one shared block invPool[0].
__global__ void cutSchwarzK(CutDev S,const real*rn,real*zn,const i32*elemList,i32 nList,const real*invPool,i32 perBlock){
  i32 ndof=S.ndof,m3=S.ndof3; extern __shared__ real re[];
  for(i32 ii=blockIdx.x;ii<nList;ii+=gridDim.x){ i32 e=elemList[ii]; const i32*nod=S.eNode+(size_t)e*ndof;
    const real*Inv=invPool+(size_t)(perBlock?ii:0)*m3*m3;
    for(i32 c=threadIdx.x;c<m3;c+=blockDim.x) re[c]=rn[3*nod[c/3]+(c%3)]; __syncthreads();
    for(i32 r=threadIdx.x;r<m3;r+=blockDim.x){ const real*Ir=Inv+(size_t)r*m3; real acc=0; for(i32 c=0;c<m3;c++)acc+=Ir[c]*re[c];
      atomicAdd(&zn[3*nod[r/3]+(r%3)],acc); } __syncthreads(); } }
// ---- p-multigrid helpers (all in NODE space) ----
// damped-Jacobi smoothing step: x += omega * (r - Ax) / dN   (node-space, dN = node diagonal)
__global__ void cutSmoothK(real*x,const real*r,const real*Ax,const real*dN,real omega,i32 n3){
  for(i32 i=CUT_STRIDE;i<n3;i+=gridDim.x*blockDim.x) x[i]+=omega*(r[i]-Ax[i])/dN[i]; }
// Chebyshev direction update: dir = c1*dir + c2 * (res / dN)   (res = r - Ax already formed)
__global__ void cutChebDirK(real*dir,const real*res,const real*dN,real c1,real c2,i32 n3){
  for(i32 i=CUT_STRIDE;i<n3;i+=gridDim.x*blockDim.x) dir[i]=c1*dir[i]+c2*res[i]/dN[i]; }
// LINE-IMPLICIT Chebyshev update: dir = c1*dir + c2 * (M_line^-1 res).  One block per span-wise
// (radial) line; Minv[mOff[L]] is the dense inverse of that line's node-space block (3m x 3m).
// Solving implicitly ALONG the span resolves the 1-D cantilever bending stiffness exactly, which is
// the soft mode p-coarsening at fixed h cannot reach.  Off-line (theta/z) coupling stays explicit.
__global__ void cutLineChebK(real*dir,const real*res,const real*Minv,const size_t*mOff,
                             const i32*lOff,const i32*lNode,real c1,real c2,i32 nLine){
  extern __shared__ real rl[];
  for(i32 L=blockIdx.x;L<nLine;L+=gridDim.x){
    i32 o=lOff[L], m=lOff[L+1]-o, m3=3*m; const real* Mi=Minv+mOff[L];
    for(i32 i=threadIdx.x;i<m3;i+=blockDim.x) rl[i]=res[3*lNode[o+i/3]+(i%3)];
    __syncthreads();
    for(i32 i=threadIdx.x;i<m3;i+=blockDim.x){ real s=0;
      for(i32 j=0;j<m3;j++) s+=Mi[(size_t)i*m3+j]*rl[j];
      i32 g=3*lNode[o+i/3]+(i%3); dir[g]=c1*dir[g]+c2*s; }
    __syncthreads(); } }
// p-prolongation (coarse-node -> fine-node), ADD: fn[3f+i] += sum_k W[w*f+k]*ec[3*col[w*f+k]+i]  (width w = coarse ndof)
__global__ void cutPProlongWK(const i32*col,const real*W,i32 w,const real*ec,real*fn,i32 nFine){
  for(i32 f=CUT_STRIDE;f<nFine;f+=gridDim.x*blockDim.x){ real a0=0,a1=0,a2=0;
    for(i32 k=0;k<w;k++){ real ww=W[(size_t)w*f+k]; i32 c=col[(size_t)w*f+k]; a0+=ww*ec[3*c];a1+=ww*ec[3*c+1];a2+=ww*ec[3*c+2]; }
    fn[3*f]+=a0; fn[3*f+1]+=a1; fn[3*f+2]+=a2; } }
// p-restriction (fine-node -> coarse-node) = P^T: ec[3*col+i] += W*fn[3f+i]
__global__ void cutPRestrictWK(const i32*col,const real*W,i32 w,const real*fn,real*ec,i32 nFine){
  for(i32 f=CUT_STRIDE;f<nFine;f+=gridDim.x*blockDim.x){ real f0=fn[3*f],f1=fn[3*f+1],f2=fn[3*f+2];
    for(i32 k=0;k<w;k++){ real ww=W[(size_t)w*f+k]; i32 c=col[(size_t)w*f+k]; atomicAdd(&ec[3*c],ww*f0);atomicAdd(&ec[3*c+1],ww*f1);atomicAdd(&ec[3*c+2],ww*f2); } } }

void CutFemSolver::runIga(void) {
  const i32 p = femOrder;
  IgaBasis Bp; Bp.init(p);
  const i32 n = p+1, ndof = n*n*n, ndof3 = 3*ndof, mG = 2*ndof3;
  const real h = cellSize();
  const double mu = prob.mu, lam = prob.lam;
  // Penalty coefficients.  kappa scales with BOTH of these, so the safety factors are a direct
  // conditioning cost -- CUT_GAMD / CUT_GAMG expose them for tuning (defaults = the verified values).
  double gamDfac=100.0; { const char*e=getenv("CUT_GAMD"); if(e) gamDfac=atof(e); }
  double gamGfac=0.1;   { const char*e=getenv("CUT_GAMG"); if(e) gamGfac=atof(e); }
  if (getenv("CUT_NOGHOST")) gamGfac=0.0;
  const double gammaD_ = gamDfac*(2*mu+lam)*p*p;   // Nitsche penalty (verified scaling)
  const double gammaG_ = gamGfac*(2*mu+lam);       // ghost-penalty coeff
  const bool cyl = (ls.coordMode == 1);
  const bool per = (periodic != 0);
  const double cph = cos((double)pitchAngle), sph = sin((double)pitchAngle);

  if (ls.coordMode != 0 && !cyl) {
    printf("ERROR: IGA path supports coordMode 0 (Cartesian) or 1 (cylindrical) only\n"); return; }
  if (p > QP_MAX) { printf("ERROR: femOrder %d exceeds QP_MAX=%d\n", p, QP_MAX); return; }

  printf("higher : immersed IGA CutFEM, C^%d B-spline (Saye cut quadrature), p=%d  %s%s  gammaD=%.4g gammaG=%.4g\n",
         p-1, p, cyl?"CYLINDRICAL isoparametric":"Cartesian", per?" + cyclic pitch tie":"",
         gammaD_, gammaG_);

  long t0 = qpNowUs(); long tSolveEnd = 0;
  initialize();
  buildMesh();
  real tgll[PNC]; gllNodes(p, tgll);

  // -------------------------------------------------------------------------
  //  active elements: sample phi at the (p+1)^3 GLL nodes via the oracle
  // -------------------------------------------------------------------------
  struct QElem { i32 ci, cj, ck; bool cut; };
  std::vector<QElem> elems;
  std::vector<real>  ephi;
  const i32 nB = hashTable.nKeys;
  #pragma omp parallel
  {
    std::vector<QElem> le; std::vector<real> lp;
    #pragma omp for schedule(dynamic,4) nowait
    for (i32 b = 0; b < nB; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty) continue;
      i32 ib, jb, kb; qpBlockDec(loc, ib, jb, kb);
      for (i32 cz = 0; cz < blockSize; cz++)
      for (i32 cy = 0; cy < blockSize; cy++)
      for (i32 cx = 0; cx < blockSize; cx++) {
        i32 ci = ib*blockSize + cx, cj = jb*blockSize + cy, ck = kb*blockSize + cz;
        real v[PNC*PNC*PNC];
        bool anyNeg=false, anyPos=false;
        for (i32 k=0;k<n;k++) for(i32 j=0;j<n;j++) for(i32 i=0;i<n;i++){
          real f = ls.phi((ci+tgll[i])*h, (cj+tgll[j])*h, (ck+tgll[k])*h);
          v[i+n*(j+n*k)]=f; if(f<0)anyNeg=true; else anyPos=true;
        }
        if (!anyNeg) continue;
        QElem E; E.ci=ci; E.cj=cj; E.ck=ck; E.cut=anyPos;
        le.push_back(E);
        for (i32 t=0;t<ndof;t++) lp.push_back(v[t]);
      }
    }
    #pragma omp critical
    { for (auto &E:le) elems.push_back(E); for (real f:lp) ephi.push_back(f); }
  }
  i32 nE = (i32)elems.size();
  if (nE == 0) { printf("ERROR: no active elements\n"); return; }

  // -------------------------------------------------------------------------
  //  node numbering (p-finer lattice) + cyclic dof map (master + rotation)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> nodeId;           // fallback only (see latOK below)
  std::vector<i32> eNodeQ((size_t)nE*ndof);
  std::vector<i32> nI, nJ, nK;                 // per-node lattice coords
  // Lattice stride: C^0 Q_p shares only the p-th node with the next cell, so the
  // node lattice is p-times finer than the cell grid.  The p+1 B-splines nonzero
  // on span ci are exactly ci..ci+p, so the control-point lattice has STRIDE 1 --
  // nCells+p per axis instead of p*nCells+1, which is the whole dof-count win.
  const i32 lstr = 1;
  auto gcoord=[&](const QElem&E,i32 a,i32&I,i32&J,i32&K){ i32 i=a%n,j=(a/n)%n,k=a/(n*n);
    I=lstr*E.ci+i; J=lstr*E.cj+j; K=lstr*E.ck+k; };
  // ---- GRID-NATIVE node numbering ----------------------------------------
  // The node lattice is a regular integer grid, so identity is DIRECT-INDEXABLE:
  // a dense (Imax-Imin+1)^3 table of node ids replaces the u64-key hash, turning
  // every lookup into address arithmetic.  Ids are still handed out in first-
  // encounter (e,a) order, so eNodeQ / nI / nJ / nK come out BIT-IDENTICAL to the
  // hashed path -- this is a pure setup-cost change, not a numbering change.
  // Falls back to the hash if the bounding box is pathologically larger than the
  // narrowband (a very thin body in a very large domain at high p).
  i32 lo[3]={INT32_MAX,INT32_MAX,INT32_MAX}, hi[3]={INT32_MIN,INT32_MIN,INT32_MIN};
  for (i32 e=0;e<nE;e++){ const i32 c[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 v=lstr*c[d]; if(v<lo[d])lo[d]=v; if(v+p>hi[d])hi[d]=v+p; } }
  const size_t lnx=(size_t)(hi[0]-lo[0]+1), lny=(size_t)(hi[1]-lo[1]+1), lnz=(size_t)(hi[2]-lo[2]+1);
  const size_t latN=lnx*lny*lnz;
  const bool latOK = latN <= (size_t)256e6 && !getenv("CUT_NODENSE");  // 1 GB ceiling on the i32 table
  std::vector<i32> latId;
  if (latOK) latId.assign(latN,-1); else nodeId.reserve((size_t)nE*ndof);
  auto latAt=[&](i32 I,i32 J,i32 K)->size_t{
    return (size_t)(I-lo[0]) + lnx*((size_t)(J-lo[1]) + lny*(size_t)(K-lo[2])); };
  auto latFind=[&](i32 I,i32 J,i32 K)->i32{    // -1 if absent (both backends)
    if (latOK){ if(I<lo[0]||I>hi[0]||J<lo[1]||J>hi[1]||K<lo[2]||K>hi[2]) return -1;
                return latId[latAt(I,J,K)]; }
    auto it=nodeId.find(qpKey(I,J,K)); return (it==nodeId.end())?-1:it->second; };
  const bool tmr = getenv("CUT_TIMING")!=nullptr;
  long tNum0 = qpNowUs();
  i32 nNodeQ=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 I,J,K; gcoord(elems[e],a,I,J,K);
    i32 id=latFind(I,J,K);
    if (id<0){ id=nNodeQ++;
      if (latOK) latId[latAt(I,J,K)]=id; else nodeId[qpKey(I,J,K)]=id;
      nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    eNodeQ[(size_t)e*ndof+a]=id;
  }
  if (tmr) printf("timing : node numbering %.1f ms (%s, %d nodes from %d elem-dofs, table %.1f MB)\n",
                  (qpNowUs()-tNum0)*1e-3, latOK?"dense":"hash", nNodeQ, nE*ndof,
                  latOK?latN*4.0/1048576.0:0.0);
  const i32 Jmax = lstr*nThetaCells;           // theta slave column (periodic)
  std::vector<i32> realIdx(nNodeQ,-1);
  std::vector<char> rotFlag(nNodeQ,0);
  i32 nDofNode=0, nTie=0, nOrphan=0;
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (per && nJ[nd]==Jmax) continue;         // slave, resolved below
    realIdx[nd]=nDofNode++;
  }
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (realIdx[nd]>=0) continue;
    i32 mst=latFind(nI[nd],0,nK[nd]);              // master at J=0
    if (mst<0){ realIdx[nd]=nDofNode++; nOrphan++; }   // no partner: own dof
    else { realIdx[nd]=realIdx[mst]; rotFlag[nd]=1; nTie++; }
  }
  const i32 nDofQ=3*nDofNode;
  if (per) printf("cyclic : %d control points tied across the pitch (%d unmatched kept free), %d -> %d dofs\n",
                  nTie, nOrphan, 3*nNodeQ, nDofQ);

  // gather/scatter with the pitch rotation (identity when non-periodic).
  // Vectors are stored in `real` (fp32 in the wavefem build, fp64 in wavefem_dp):
  // with the matrix bank gone (matrix-free below), storage is the memory driver,
  // so single precision halves it.  Element APPLY still computes in double.
  auto gather3=[&](const std::vector<real>&x,i32 nd,double u[3]){
    i32 b=3*realIdx[nd];
    if (rotFlag[nd]){ u[0]=cph*x[b]-sph*x[b+1]; u[1]=sph*x[b]+cph*x[b+1]; u[2]=x[b+2]; }
    else { u[0]=x[b]; u[1]=x[b+1]; u[2]=x[b+2]; }
  };
  auto scatter3=[&](std::vector<real>&y,i32 nd,const double c[3]){
    i32 b=3*realIdx[nd]; double a0,a1,a2;
    if (rotFlag[nd]){ a0=cph*c[0]+sph*c[1]; a1=-sph*c[0]+cph*c[1]; a2=c[2]; }
    else { a0=c[0]; a1=c[1]; a2=c[2]; }
    #pragma omp atomic
    y[b]+=a0;
    #pragma omp atomic
    y[b+1]+=a1;
    #pragma omp atomic
    y[b+2]+=a2;
  };

  // node physical positions; per-dof "owner" position (for rigid-mode build/VTU)
  std::vector<real> nodeXQ((size_t)3*nNodeQ);
  std::vector<double> dofPos((size_t)3*nDofNode, 0.0);
  for (i32 nd=0;nd<nNodeQ;nd++){
    real X0,X1,X2; ls.toPhys((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p,X0,X1,X2);
    nodeXQ[3*nd]=X0; nodeXQ[3*nd+1]=X1; nodeXQ[3*nd+2]=X2;
    if (!rotFlag[nd]){ i32 d=realIdx[nd]; dofPos[3*d]=X0; dofPos[3*d+1]=X1; dofPos[3*d+2]=X2; }
  }

  // -------------------------------------------------------------------------
  //  per-element level-set polynomial + Saye cut quadrature
  // -------------------------------------------------------------------------
  std::vector<PolyND> ePoly(nE);
  std::vector<i32> cutIdx(nE,-1); i32 nCutQ=0;
  for (i32 e=0;e<nE;e++){ ePoly[e]=fitPoly3(p,&ephi[(size_t)e*ndof]); if (elems[e].cut) cutIdx[e]=nCutQ++; }
  std::vector<SayeNode> volPool, surfPool;
  std::vector<i32> volOff(nCutQ+1,0), surfOff(nCutQ+1,0);
  { std::vector<SayeNode> arena(1<<18), out(1<<16);
    for (i32 e=0;e<nE;e++) if (elems[e].cut){ i32 c=cutIdx[e];
      SayeArena ar; ar.buf=arena.data(); ar.cap=1<<18; ar.top=0;
      SayeSet ov; ov.p=out.data(); ov.n=0; ov.cap=1<<16; ov.ovf=false;
      sayeVolume(ePoly[e],&ov,&ar,SayeCfg::def());
      for (i32 q=0;q<ov.n;q++) volPool.push_back(ov.p[q]); volOff[c+1]=(i32)volPool.size();
      SayeArena ar2; ar2.buf=arena.data(); ar2.cap=1<<18; ar2.top=0;
      SayeSet sv; sv.p=out.data(); sv.n=0; sv.cap=1<<16; sv.ovf=false;
      sayeSurface(ePoly[e],&sv,&ar2,SayeCfg::def());
      for (i32 q=0;q<sv.n;q++) surfPool.push_back(sv.p[q]); surfOff[c+1]=(i32)surfPool.size();
    } }

  // ---- NNLS quadrature compression (ON by default; CUT_NOPRUNE=1 to disable): shrink each
  //      cut cell's Saye volume rule to a minimal positive rule with the same Q_{2p} moments.
  //      Exact for polynomial integrands (stiffness always; polynomial body load) ----
  if (!getenv("CUT_NOPRUNE") && nCutQ>0) {
    long t0=qpNowUs(); size_t n0=volPool.size();
    const char*ue=getenv("CUT_UNIFORM"); int gN=ue?atoi(ue):0;   // >0 : paper's uniform-grid candidates
    std::vector<i32> cutE(nCutQ); for (i32 e=0;e<nE;e++) if (elems[e].cut) cutE[cutIdx[e]]=e;
    std::vector<std::vector<SayeNode>> pc(nCutQ); std::vector<double> resid(nCutQ,0.0);
    #pragma omp parallel for schedule(dynamic,1)
    for (i32 c=0;c<nCutQ;c++){ i32 v0=volOff[c],nv=volOff[c+1]-v0;
      if(gN>0) resid[c]=compressVolUniform(&volPool[v0],nv,ePoly[cutE[c]],p,gN,pc[c]);
      else compressVol(&volPool[v0],nv,p,pc[c]); }
    std::vector<SayeNode> vp2; vp2.reserve(n0); std::vector<i32> vo2(nCutQ+1,0);
    for (i32 c=0;c<nCutQ;c++){ for (auto&s:pc[c]) vp2.push_back(s); vo2[c+1]=(i32)vp2.size(); }
    printf("prune  : NNLS %s volume %zu -> %zu pts (%.1fx, %.0f/cell -> %.0f/cell) in %.2fs\n",
           gN>0?"uniform-grid":"Saye", n0, vp2.size(), (double)n0/std::max<size_t>(vp2.size(),1),
           (double)n0/nCutQ, (double)vp2.size()/nCutQ, (qpNowUs()-t0)*1e-6);
    if(gN>0){ double mr=0,ar=0; for(double v:resid){ if(v>mr)mr=v; ar+=v; }
      printf("       : uniform grid %d^3, moment residual: max %.2e, mean %.2e (0 => reproduces Saye moments exactly)\n",gN,mr,ar/nCutQ); }
    volPool.swap(vp2); volOff.swap(vo2);
  }

  // ghost faces (interior faces of cut elements; wrap the theta seam if periodic)
  // GRID-NATIVE: face pairing is a +1 step on a regular cell grid, so the partner
  // is found by direct indexing into a dense cell->element table rather than by
  // hashing a u64 cell key.  Same (e,d) visit order => identical face list.
  // The theta-seam wrap is why the table spans cells, not blocks: nb[1] jumps to 0.
  i32 clo[3]={INT32_MAX,INT32_MAX,INT32_MAX}, chi[3]={INT32_MIN,INT32_MIN,INT32_MIN};
  for (i32 e=0;e<nE;e++){ const i32 c[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ if(c[d]<clo[d])clo[d]=c[d]; if(c[d]>chi[d])chi[d]=c[d]; } }
  if (per) { clo[1]=std::min(clo[1],0); chi[1]=std::max(chi[1],nThetaCells); }
  const size_t cnx=(size_t)(chi[0]-clo[0]+1), cny=(size_t)(chi[1]-clo[1]+1), cnz=(size_t)(chi[2]-clo[2]+1);
  const size_t cellN=cnx*cny*cnz;
  const bool cellOK = cellN <= (size_t)256e6 && !getenv("CUT_NODENSE");
  std::unordered_map<u64,i32> cellId;            // fallback only
  std::vector<i32> cellTab;
  if (cellOK) cellTab.assign(cellN,-1); else cellId.reserve((size_t)nE*2);
  auto cellPut=[&](i32 i,i32 j,i32 k,i32 e){
    if (cellOK) cellTab[(size_t)(i-clo[0])+cnx*((size_t)(j-clo[1])+cny*(size_t)(k-clo[2]))]=e;
    else cellId[qpCellKey(i,j,k)]=e; };
  auto cellGet=[&](i32 i,i32 j,i32 k)->i32{
    if (cellOK){ if(i<clo[0]||i>chi[0]||j<clo[1]||j>chi[1]||k<clo[2]||k>chi[2]) return -1;
                 return cellTab[(size_t)(i-clo[0])+cnx*((size_t)(j-clo[1])+cny*(size_t)(k-clo[2]))]; }
    auto it=cellId.find(qpCellKey(i,j,k)); return (it==cellId.end())?-1:it->second; };
  long tGf0 = qpNowUs();
  for (i32 e=0;e<nE;e++) cellPut(elems[e].ci,elems[e].cj,elems[e].ck,e);
  struct GF{ i32 eM,eP,d; };
  std::vector<GF> gf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
      if (per && d==1 && nb[1]==nThetaCells) nb[1]=0;      // theta seam wraps
      i32 ep=cellGet(nb[0],nb[1],nb[2]); if (ep<0) continue;
      if (elems[e].cut||elems[ep].cut) gf.push_back({e,ep,d}); } }
  if (tmr) printf("timing : ghost pairing %.1f ms (%s, %zu faces from %d cells, table %.1f MB)\n",
                  (qpNowUs()-tGf0)*1e-3, cellOK?"dense":"hash", gf.size(), nE,
                  cellOK?cellN*4.0/1048576.0:0.0);
  i32 nGFQ=(i32)gf.size();

  // ghost-penalty face traces: Dl0[l][a] / Dl1[l][a] = l-th reference derivative
  // of 1-D basis function a at xi=0 / xi=1.
  double Dl0[QP_MAX+1][QN_MAX], Dl1[QP_MAX+1][QN_MAX];
  for (i32 l=1;l<=p;l++) for (i32 a=0;a<n;a++){
    real v=Bp.dlFace(l,a); Dl0[l][a]=(double)v; Dl1[l][a]=(double)v; }

  auto physOf=[&](const QElem&E,const real xr[3],real X[3]){
    ls.toPhys((E.ci+xr[0])*h,(E.cj+xr[1])*h,(E.ck+xr[2])*h,X[0],X[1],X[2]); };

  // cylindrical metric at a reference point: Jinv (=Jref^{-1}), detJ (analytic)
  auto metric=[&](const QElem&E,const real xr[3],double Jinv[3][3],double&detJ){
    double r=(E.ci+xr[0])*h+ls.org[0], s=(E.cj+xr[1])*h+ls.org[1], z=(E.ck+xr[2])*h+ls.org[2];
    double th=s/ls.rRef+(double)ls.thc((real)z), thp=(double)ls.thcSlope((real)z);
    double ct=cos(th), st=sin(th);
    double J[3][3]={{ h*ct, h*(-r*st/ls.rRef), h*(-r*st*thp) },
                    { h*st, h*( r*ct/ls.rRef), h*( r*ct*thp) },
                    { 0,    0,                 h            }};
    detJ=inv3(J,Jinv);
  };

  // ---- optional geometry dump (CUT_DUMP=1): active cells, cut flag, Saye points; then exit ----
  if (getenv("CUT_DUMP")) {
    mkdir("output",0755);
    std::string base = "output/" + (outTag.empty()?std::string("dump"):outTag);
    { std::ofstream os((base+"_cells.csv").c_str());
      os<<"cut";for(i32 c=0;c<8;c++)os<<",x"<<c<<",y"<<c<<",z"<<c; os<<"\n";
      for (i32 e=0;e<nE;e++){ os<<(elems[e].cut?1:0);
        for (i32 c=0;c<8;c++){ real xr[3]={(real)(c&1),(real)((c>>1)&1),(real)((c>>2)&1)},X[3]; physOf(elems[e],xr,X); os<<","<<X[0]<<","<<X[1]<<","<<X[2]; } os<<"\n"; } }
    { std::ofstream os((base+"_sayevol.csv").c_str()); os<<"x,y,z,w\n";
      for (i32 e=0;e<nE;e++) if(elems[e].cut){ i32 c=cutIdx[e]; for(i32 q=volOff[c];q<volOff[c+1];q++){ real xr[3]={volPool[q].x[0],volPool[q].x[1],volPool[q].x[2]},X[3]; physOf(elems[e],xr,X); os<<X[0]<<","<<X[1]<<","<<X[2]<<","<<volPool[q].w<<"\n"; } } }
    { std::ofstream os((base+"_sayesurf.csv").c_str()); os<<"x,y,z\n";
      for (i32 e=0;e<nE;e++) if(elems[e].cut){ i32 c=cutIdx[e]; for(i32 q=surfOff[c];q<surfOff[c+1];q++){ real xr[3]={surfPool[q].x[0],surfPool[q].x[1],surfPool[q].x[2]},X[3]; physOf(elems[e],xr,X); os<<X[0]<<","<<X[1]<<","<<X[2]<<"\n"; } } }
    printf("dump   : %d cells (%d cut), %zu Saye vol pts, %zu surf pts -> %s_{cells,sayevol,sayesurf}.csv\n",
           nE,nCutQ,volPool.size(),surfPool.size(),base.c_str());
    return;
  }

  // -------------------------------------------------------------------------
  //  local element operators
  // -------------------------------------------------------------------------
  // CARTESIAN cut-element local apply (bulk Saye + Nitsche), reference frame
  auto applyCutCart=[&](i32 e,const double*uloc,double*yloc){
    i32 c=cutIdx[e]; i32 nv=volOff[c+1]-volOff[c]; const SayeNode*vn=&volPool[volOff[c]];
    real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 a=0;a<ndof3;a++) ul[a]=(real)uloc[a];
    qpElemCoreSaye(Bp,(real)mu,(real)lam,h,vn,nv,ul,yl);
    for (i32 a=0;a<ndof3;a++) yloc[a]=yl[a];
    i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
      real X[3]; physOf(elems[e],xr,X); if (!prob.isDirichlet(X[0],X[1],X[2])) continue;
      double nn[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); double hw=sn[q].w*h;
      double uval[3]={0,0,0}, gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ uval[i2]+=uloc[3*a+i2]*vb[a];
        gradU[i2][0]+=uloc[3*a+i2]*gb[3*a+0]; gradU[i2][1]+=uloc[3*a+i2]*gb[3*a+1]; gradU[i2][2]+=uloc[3*a+i2]*gb[3*a+2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
      double un=uval[0]*nn[0]+uval[1]*nn[1]+uval[2]*nn[2];
      for (i32 a=0;a<ndof;a++){ double gan=gb[3*a+0]*nn[0]+gb[3*a+1]*nn[1]+gb[3*a+2]*nn[2];
        double ugb=uval[0]*gb[3*a+0]+uval[1]*gb[3*a+1]+uval[2]*gb[3*a+2];
        for (i32 l=0;l<3;l++){ double t1=-tu[l]*vb[a];
          double t2=-(mu*(uval[l]*gan+ugb*nn[l])+lam*gb[3*a+l]*un); double t3=gammaD_*uval[l]*vb[a];
          yloc[3*a+l]+=hw*(t1+t2+t3); } } }
  };

  // CYLINDRICAL element local apply (curved bulk + Nanson Nitsche).  nitsche=false
  // gives bulk only (for the rigid-body self-check).
  auto applyElemCyl=[&](i32 e,const double*uloc,double*yloc,bool nitsche){
    for (i32 a=0;a<ndof3;a++) yloc[a]=0;
    // volume quadrature: tensor GLL (uncut) or Saye (cut)
    std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
    if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
      for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
        tens[qi].x[0]=Bp.qx[i];tens[qi].x[1]=Bp.qx[j];tens[qi].x[2]=Bp.qx[k];
        tens[qi].w=Bp.qw[i]*Bp.qw[j]*Bp.qw[k]; qi++; } vn=tens.data(); nv=ndof;
    } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
    real gb[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
      double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
      Bp.allGradRef(xr,gb);
      double gX[QN_MAX*QN_MAX*QN_MAX][3];
      for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
      double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ gradU[i2][0]+=uloc[3*a+i2]*gX[a][0];
        gradU[i2][1]+=uloc[3*a+i2]*gX[a][1]; gradU[i2][2]+=uloc[3*a+i2]*gX[a][2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double wdet=fabs(detJ)*vn[q].w;
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++)
        yloc[3*a+i2]+=wdet*(sig[i2][0]*gX[a][0]+sig[i2][1]*gX[a][1]+sig[i2][2]*gX[a][2]); }
    if (!nitsche || !elems[e].cut) return;
    // Nitsche on the physical (Nanson) surface at Dirichlet points
    i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
    real vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
      real X[3]; physOf(elems[e],xr,X); if (!prob.isDirichlet(X[0],X[1],X[2])) continue;
      double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
      double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
      double nraw[3]; for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
      double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if (nmag<=0) continue;
      double nP[3]={nraw[0]/nmag,nraw[1]/nmag,nraw[2]/nmag};
      double dS=fabs(detJ)*nmag*sn[q].w;
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
      double gX[QN_MAX*QN_MAX*QN_MAX][3];
      for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
      double uval[3]={0,0,0}, gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ uval[i2]+=uloc[3*a+i2]*vb[a];
        gradU[i2][0]+=uloc[3*a+i2]*gX[a][0]; gradU[i2][1]+=uloc[3*a+i2]*gX[a][1]; gradU[i2][2]+=uloc[3*a+i2]*gX[a][2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nP[0]+sig[i2][1]*nP[1]+sig[i2][2]*nP[2];
      double un=uval[0]*nP[0]+uval[1]*nP[1]+uval[2]*nP[2];
      // penalty carries the 1/h Nitsche scaling (Cartesian folds it into hw=sn.w*h;
      // here dS is the FULL physical area, so divide the penalty coefficient by h)
      double penC=gammaD_/h;
      for (i32 a=0;a<ndof;a++){ double gan=gX[a][0]*nP[0]+gX[a][1]*nP[1]+gX[a][2]*nP[2];
        double ugb=uval[0]*gX[a][0]+uval[1]*gX[a][1]+uval[2]*gX[a][2];
        for (i32 l=0;l<3;l++){ double t1=-tu[l]*vb[a];
          double t2=-(mu*(uval[l]*gan+ugb*nP[l])+lam*gX[a][l]*un); double t3=penC*uval[l]*vb[a];
          yloc[3*a+l]+=dS*(t1+t2+t3); } } }
  };

  // ---- consistency self-check (cylindrical curved bulk operator) ----
  // A CONSTANT displacement is in Q_p exactly, so eps=0 and K_bulk*u_const must
  // be machine-zero regardless of the curved metric -- this validates that the
  // analytic Jacobian is applied consistently (partition-of-unity of grad).  (A
  // rigid ROTATION is only O(h^{p+1}) here because the geometry is exact, not
  // isoparametric-interpolated, so it is not a machine-zero test.)  Also checks
  // the reference-cell volume integral of |detJ| against the analytic curved
  // cell volume on a sample element, which validates the determinant.
  if (cyl) {
    double worstC=0;
    for (i32 e=0;e<nE;e++){
      double uc[3*QN_MAX*QN_MAX*QN_MAX], yc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ uc[3*a]=1.0; uc[3*a+1]=-0.7; uc[3*a+2]=0.3; }  // const
      applyElemCyl(e,uc,yc,false);
      double yn2=0; for (i32 a=0;a<ndof3;a++) yn2+=yc[a]*yc[a];
      worstC=std::max(worstC,sqrt(yn2)); }
    // |detJ| volume of an uncut sample cell vs the analytic curved-brick volume
    // int_cell dV = int_{r0}^{r1} int_{s0}^{s1} int_{z0}^{z1} r dr ds dz
    //             = (r1^2-r0^2)/2 * (s1-s0) * (z1-z0)   (theta = s/rRef+thc(z))
    double volErr=-1;
    for (i32 e=0;e<nE && volErr<0;e++) if (!elems[e].cut){
      double num=0; GaussRule g=gaussLegendre(5);
      for (i32 i=0;i<g.n;i++)for(i32 j=0;j<g.n;j++)for(i32 k=0;k<g.n;k++){
        real xr[3]={(real)g.x[i],(real)g.x[j],(real)g.x[k]}; double Jinv[3][3],detJ;
        metric(elems[e],xr,Jinv,detJ); num+=fabs(detJ)*g.w[i]*g.w[j]*g.w[k]; }
      double r0=elems[e].ci*h+ls.org[0], r1=r0+h;
      double ana=0.5*(r1*r1-r0*r0)*h*h/ls.rRef;   // (s1-s0)=h, (z1-z0)=h, ds=h -> /rRef in theta but dV=r dr ds dz uses s directly
      // dV_phys = r dr dtheta dz = r dr (ds/rRef) dz ; here s is the arc coord so dtheta=ds/rRef
      volErr=fabs(num-ana)/std::max(1e-30,ana);
    }
    printf("check  : curved-bulk const-field residual %.3e %s;  |detJ| cell-volume rel err %.3e\n",
           worstC, worstC<1e-9?"[OK]":"[WARN]", volErr);
  }

  // -------------------------------------------------------------------------
  //  MATRIX-FREE operator.  The cylindrical per-element matrix bank
  //  (~n_elem * (3(p+1)^3)^2 reals) was the memory driver; instead of storing it,
  //  recompute each element's action per CG apply by calling the SAME verified
  //  local applies used to build it (so the operator is algebraically identical,
  //  just recomputed).  Only the tiny reference-invariant per-axis ghost matrices
  //  are kept.  Combined with the fp32 `real` vectors, this removes the tens-of-GB
  //  bank and leaves the Saye pools + O(dof) vectors as the footprint.
  // -------------------------------------------------------------------------
  auto applyElem=[&](i32 e,const double*uloc,double*yloc){
    if (cyl) { applyElemCyl(e,uloc,yloc,true); return; }
    if (elems[e].cut) { applyCutCart(e,uloc,yloc); return; }
    real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 a=0;a<ndof3;a++) ul[a]=(real)uloc[a];
    qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
    for (i32 a=0;a<ndof3;a++) yloc[a]=yl[a];
  };

  // ghost matrices (computational frame, one per axis; shared, tiny)
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
  //  operator apply (dense local matvecs + rotation-aware gather/scatter)
  // -------------------------------------------------------------------------
  auto applyA=[&](const std::vector<real>&x,std::vector<real>&y){
    std::fill(y.begin(),y.end(),(real)0);
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX], yloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nod[a],u3); uloc[3*a]=u3[0]; uloc[3*a+1]=u3[1]; uloc[3*a+2]=u3[2]; }
      applyElem(e,uloc,yloc);                    // matrix-free: recompute the action
      for (i32 a=0;a<ndof;a++){ double c3[3]={yloc[3*a],yloc[3*a+1],yloc[3*a+2]}; scatter3(y,nod[a],c3); } }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      double uMP[2*3*QN_MAX*QN_MAX*QN_MAX], yMP[2*3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nodM[a],u3); uMP[3*a]=u3[0];uMP[3*a+1]=u3[1];uMP[3*a+2]=u3[2];
        gather3(x,nodP[a],u3); uMP[ndof3+3*a]=u3[0];uMP[ndof3+3*a+1]=u3[1];uMP[ndof3+3*a+2]=u3[2]; }
      const double*K=Kghost[F.d].data();
      for (i32 r=0;r<mG;r++){ double s=0; const double*Kr=&K[(size_t)r*mG]; for(i32 c=0;c<mG;c++) s+=Kr[c]*uMP[c]; yMP[r]=s; }
      for (i32 a=0;a<ndof;a++){ double cm[3]={yMP[3*a],yMP[3*a+1],yMP[3*a+2]}; scatter3(y,nodM[a],cm);
        double cp[3]={yMP[ndof3+3*a],yMP[ndof3+3*a+1],yMP[ndof3+3*a+2]}; scatter3(y,nodP[a],cp); } }
  };

  // -------------------------------------------------------------------------
  //  RHS (body force + Nitsche data + Neumann traction) and Jacobi diagonal
  // -------------------------------------------------------------------------
  std::vector<real> bvec(nDofQ,(real)0);                 // fp32 storage (wavefem build)
  std::vector<double> diagNode((size_t)3*nNodeQ,0.0);   // per-node, folded to dof below (temp, freed)
  std::vector<double> blkNode((size_t)9*nNodeQ,0.0);    // per-node SPD 3x3 block (for block-Jacobi PC)
  {
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double bloc[3*QN_MAX*QN_MAX*QN_MAX]; for (i32 a=0;a<ndof3;a++) bloc[a]=0;
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      // volume quadrature source
      std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
      if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.qx[i];tens[qi].x[1]=Bp.qx[j];tens[qi].x[2]=Bp.qx[k];
          tens[qi].w=Bp.qw[i]*Bp.qw[j]*Bp.qw[k]; qi++; } vn=tens.data(); nv=ndof;
      } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
      for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); real X[3]; physOf(elems[e],xr,X);
        real f[3]; prob.bodyForce(X[0],X[1],X[2],f);
        double gX[QN_MAX*QN_MAX*QN_MAX][3]; double wdet, wdiag;
        if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          wdet=fabs(detJ)*vn[q].w; wdiag=wdet;
        } else { for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]/h;
          wdet=vn[q].w*h*h*h; wdiag=vn[q].w*h; }
        for (i32 a=0;a<ndof;a++){ for (i32 l=0;l<3;l++) bloc[3*a+l]+=wdet*f[l]*vb[a];
          double gsq=gX[a][0]*gX[a][0]+gX[a][1]*gX[a][1]+gX[a][2]*gX[a][2];
          double sc = cyl ? 1.0 : (h*h);   // diag uses physical grads; Cart gX=gb/h so rescale to match wdiag*h powers
          for (i32 l=0;l<3;l++) diagNode[3*nod[a]+l]+=wdiag*(mu*(gsq+gX[a][l]*gX[a][l])+lam*gX[a][l]*gX[a][l])*sc;
          // full 3x3 volume block:  mu(|g|^2 dij + gi gj) + lam gi gj  (SPD)
          double*Bn=&blkNode[9*nod[a]]; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++)
            Bn[3*i2+j2]+=wdiag*(mu*((i2==j2?gsq:0)+gX[a][i2]*gX[a][j2])+lam*gX[a][i2]*gX[a][j2])*sc; } }
      // surface source
      if (elems[e].cut){ i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
        for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]}; real X[3]; physOf(elems[e],xr,X);
          Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
          double nP[3], dS; double gX[QN_MAX*QN_MAX*QN_MAX][3];
          if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
            double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]}, nraw[3];
            for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
            double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if (nmag<=0) continue;
            nP[0]=nraw[0]/nmag; nP[1]=nraw[1]/nmag; nP[2]=nraw[2]/nmag; dS=fabs(detJ)*nmag*sn[q].w;
            for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          } else { nP[0]=sn[q].n[0]; nP[1]=sn[q].n[1]; nP[2]=sn[q].n[2]; dS=sn[q].w*h;   // Cart: reference frame, one h
            for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]; }
          // penalty coefficient: Cartesian folds 1/h into dS=sn.w*h; cylindrical
          // uses the full physical dS, so the penalty carries the explicit 1/h
          double penC = cyl ? (gammaD_/h) : gammaD_;
          bool dir=prob.isDirichlet(X[0],X[1],X[2]);
          if (dir){ real g[3]; prob.dirichletData(X[0],X[1],X[2],g);
            double gn=g[0]*nP[0]+g[1]*nP[1]+g[2]*nP[2];
            for (i32 a=0;a<ndof;a++){ double gan=gX[a][0]*nP[0]+gX[a][1]*nP[1]+gX[a][2]*nP[2];
              double ggb=g[0]*gX[a][0]+g[1]*gX[a][1]+g[2]*gX[a][2];
              for (i32 l=0;l<3;l++){ double rhs=-(mu*(g[l]*gan+ggb*nP[l])+lam*gX[a][l]*gn)+penC*g[l]*vb[a];
                bloc[3*a+l]+=dS*rhs; double pd=dS*penC*vb[a]*vb[a];
                diagNode[3*nod[a]+l]+=pd; blkNode[9*nod[a]+3*l+l]+=pd; } }
          } else { real nr[3]={(real)nP[0],(real)nP[1],(real)nP[2]}, g[3]; prob.neumannData(X[0],X[1],X[2],nr,g);
            for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++) bloc[3*a+l]+=dS*g[l]*vb[a]; } } }
      // scatter the element load through the cyclic tie
      for (i32 a=0;a<ndof;a++){ double c3[3]={bloc[3*a],bloc[3*a+1],bloc[3*a+2]}; scatter3(bvec,nod[a],c3); }
    }
    // ghost-penalty diagonal (computational frame)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      i32 d=F.d,t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(p+1);
      for (i32 q1=0;q1<g1.n;q1++) for (i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2];
        real L1[QN_MAX],L2[QN_MAX]; Bp.basis1(g1.x[q1],L1); Bp.basis1(g1.x[q2],L2);
        for (i32 a=0;a<ndof;a++){ i32 idx[3]={a%n,(a/n)%n,a/(n*n)}; i32 idn=idx[d]; double Lt=L1[idx[t1]]*L2[idx[t2]];
          for (i32 l=1;l<=p;l++){ double cP=Dl0[l][idn]*Lt, cM=Dl1[l][idn]*Lt, cf=gammaG_*h*w;
            for (i32 comp=0;comp<3;comp++){ diagNode[3*nodP[a]+comp]+=cf*cP*cP; diagNode[3*nodM[a]+comp]+=cf*cM*cM;
              blkNode[9*nodP[a]+3*comp+comp]+=cf*cP*cP; blkNode[9*nodM[a]+3*comp+comp]+=cf*cM*cM; } } } } }
    // fold node diagonal into dof space (rotation ignored: preconditioner only)
    std::vector<real> diagv(nDofQ,(real)0);
    for (i32 nd=0;nd<nNodeQ;nd++){ i32 b=3*realIdx[nd];
      for (i32 l=0;l<3;l++) diagv[b+l]+=(real)diagNode[3*nd+l]; }
    for (i32 i=0;i<nDofQ;i++) if (diagv[i]<=0) diagv[i]=(real)1;

    // ---- Jacobi-PCG (fp32 vectors, fp64 scalars/accumulators = mixed precision) ----
    std::vector<real> uv(nDofQ,(real)0);
    if (getenv("CUT_GPU")) {                  // ===== GPU cut-cell CG (Cartesian or cylindrical) =====
      const i32 nR=nDofQ, nN=3*nNodeQ, BS=256, GS=1024;
      std::vector<i32> cutElem(nCutQ), intList; intList.reserve(nE-nCutQ);
      for (i32 e=0;e<nE;e++){ if (elems[e].cut) cutElem[cutIdx[e]]=e; else intList.push_back(e); }
      std::vector<char> surfDir(surfPool.size()?surfPool.size():1,0);
      for (i32 e=0;e<nE;e++) if (elems[e].cut){ i32 c=cutIdx[e]; for (i32 q=surfOff[c];q<surfOff[c+1];q++){
        real xr[3]={surfPool[q].x[0],surfPool[q].x[1],surfPool[q].x[2]}, X[3]; physOf(elems[e],xr,X);
        surfDir[q]=prob.isDirichlet(X[0],X[1],X[2])?1:0; } }
      std::vector<real> Kref((size_t)ndof3*ndof3);
      { real ulc[3*QN_MAX*QN_MAX*QN_MAX], ylc[3*QN_MAX*QN_MAX*QN_MAX];
        for (i32 cc=0;cc<ndof3;cc++){ for(i32 i=0;i<ndof3;i++) ulc[i]=(i==cc)?(real)1:(real)0;
          qpElemUncut(Bp,(real)mu,(real)lam,h,ulc,ylc); for(i32 r=0;r<ndof3;r++) Kref[(size_t)r*ndof3+cc]=ylc[r]; } }
      auto cpI=[&](const i32*s,size_t m){ i32*d; cudaMallocManaged(&d,(m?m:1)*sizeof(i32)); if(m)memcpy(d,s,m*sizeof(i32)); return d; };
      auto cpR=[&](const real*s,size_t m){ real*d; cudaMallocManaged(&d,(m?m:1)*sizeof(real)); if(m)memcpy(d,s,m*sizeof(real)); return d; };
      auto cpDr=[&](const double*s,size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); for(size_t i=0;i<m;i++)d[i]=(real)s[i]; return d; };
      auto alR=[&](size_t m){ real*d; cudaMallocManaged(&d,m*sizeof(real)); cudaMemset(d,0,m*sizeof(real)); return d; };
      i32 *d_eNode=cpI(eNodeQ.data(),(size_t)nE*ndof), *d_nMap=cpI(realIdx.data(),nNodeQ);
      char *d_nRot; cudaMallocManaged(&d_nRot,nNodeQ); for(i32 i=0;i<nNodeQ;i++)d_nRot[i]=rotFlag[i];
      i32 *d_intList=cpI(intList.data(),intList.size()), *d_cutElem=cpI(cutElem.data(),nCutQ);
      std::vector<i32> eCijkH((size_t)3*nE); for(i32 e=0;e<nE;e++){ eCijkH[3*e]=elems[e].ci; eCijkH[3*e+1]=elems[e].cj; eCijkH[3*e+2]=elems[e].ck; }
      i32 *d_eCijk=cpI(eCijkH.data(),(size_t)3*nE), *d_eCut=cpI(cutIdx.data(),nE);   // per-element (ci,cj,ck) and cut index (cyl metric)
      real *d_volJ=nullptr,*d_surfJ=nullptr;   // precompute cyl metric [Jinv,detJ] at Saye points once (shared across p-levels)
      if (cyl) { std::vector<real> volJH((size_t)10*(volPool.size()?volPool.size():1)), surfJH((size_t)10*(surfPool.size()?surfPool.size():1));
        for(i32 e=0;e<nE;e++) if(elems[e].cut){ i32 c=cutIdx[e];
          for(i32 q=volOff[c];q<volOff[c+1];q++){ real xr[3]={volPool[q].x[0],volPool[q].x[1],volPool[q].x[2]}; double Ji[3][3],dJ; metric(elems[e],xr,Ji,dJ); real*J=&volJH[(size_t)10*q]; for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++)J[3*i+j]=(real)Ji[i][j]; J[9]=(real)dJ; }
          for(i32 q=surfOff[c];q<surfOff[c+1];q++){ real xr[3]={surfPool[q].x[0],surfPool[q].x[1],surfPool[q].x[2]}; double Ji[3][3],dJ; metric(elems[e],xr,Ji,dJ); real*J=&surfJH[(size_t)10*q]; for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++)J[3*i+j]=(real)Ji[i][j]; J[9]=(real)dJ; } }
        d_volJ=cpR(volJH.data(),volJH.size()); d_surfJ=cpR(surfJH.data(),surfJH.size()); }
      SayeNode *d_volP,*d_surfP; cudaMallocManaged(&d_volP,(volPool.size()?volPool.size():1)*sizeof(SayeNode)); if(volPool.size())memcpy(d_volP,volPool.data(),volPool.size()*sizeof(SayeNode));
      cudaMallocManaged(&d_surfP,(surfPool.size()?surfPool.size():1)*sizeof(SayeNode)); if(surfPool.size())memcpy(d_surfP,surfPool.data(),surfPool.size()*sizeof(SayeNode));
      i32 *d_volOff=cpI(volOff.data(),nCutQ+1), *d_surfOff=cpI(surfOff.data(),nCutQ+1);
      char *d_surfDir; cudaMallocManaged(&d_surfDir,surfDir.size()); memcpy(d_surfDir,surfDir.data(),surfDir.size());
      i32 *d_gfM,*d_gfP,*d_gfD; cudaMallocManaged(&d_gfM,(nGFQ?nGFQ:1)*sizeof(i32)); cudaMallocManaged(&d_gfP,(nGFQ?nGFQ:1)*sizeof(i32)); cudaMallocManaged(&d_gfD,(nGFQ?nGFQ:1)*sizeof(i32));
      for(i32 f=0;f<nGFQ;f++){ d_gfM[f]=gf[f].eM; d_gfP[f]=gf[f].eP; d_gfD[f]=gf[f].d; }
      real *d_Kref=cpR(Kref.data(),(size_t)ndof3*ndof3), *d_Kg0=cpDr(Kghost[0].data(),(size_t)mG*mG), *d_Kg1=cpDr(Kghost[1].data(),(size_t)mG*mG), *d_Kg2=cpDr(Kghost[2].data(),(size_t)mG*mG);
      real *d_b=cpR(bvec.data(),nR), *d_diag=cpR(diagv.data(),nR);
      real *d_u=alR(nR),*d_r=alR(nR),*d_z=alR(nR),*d_pd=alR(nR),*d_Ap=alR(nR),*d_xn=alR(nN),*d_yn=alR(nN);
      double *d_acc; cudaMalloc(&d_acc,sizeof(double));
      // ---- preconditioner selection: CUT_PC = jac (default) | bjac | schwarz | pmg | hmg | ic ----
      //  'ic' = incomplete Cholesky IC(0) on the ASSEMBLED CSR.  Unlike pmg/hmg it needs no coarse
      //  space, so the aspect-ratio shear locking that caps the h-hierarchy at 2 levels cannot bite it.
      const char*pcEnv=getenv("CUT_PC"); std::string pcName=pcEnv?pcEnv:"jac";
      //  'l1jac' = Jacobi on the l1 row-sum d_i = sum_j |a_ij| instead of a_ii.  Identical apply cost;
      //  differs from plain jac ONLY where |row|/|diag| VARIES (cut rows), since CG is invariant to a
      //  constant rescaling of M -- so it is a direct probe of whether diagonal quality matters here.
      //  'line' = span-wise line-block Jacobi standalone (not as an MG smoother): the one cheap PC that
      //  is GLOBAL along the bending direction, so it can change the rate, not just the constant.
      i32 pcMode=(pcName=="bjac")?1:(pcName=="schwarz")?2:(pcName=="pmg")?3:(pcName=="hmg")?4:(pcName=="ic")?5
                :(pcName=="l1jac")?6:(pcName=="line")?7:(pcName=="linex")?8:(pcName=="ssor")?9:0;
      if (cyl && (pcMode==1||pcMode==2)){ printf("note   : PC '%s' not yet cylindrical-ready; using jac\n",pcName.c_str()); pcMode=0; pcName="jac"; }
      const i32 nDofNode=nDofQ/3; real *d_Binv=nullptr;
      if (pcMode==1) {   // nodal 3x3 block-Jacobi: fold blkNode by realIdx, invert per dof-node (SPD)
        std::vector<double> Bd((size_t)9*nDofNode,0.0);
        for (i32 nd=0;nd<nNodeQ;nd++){ i32 b=realIdx[nd]; const double*Bn=&blkNode[9*nd]; for(i32 k=0;k<9;k++) Bd[9*b+k]+=Bn[k]; }
        std::vector<real> Bi((size_t)9*nDofNode);
        for (i32 nd=0;nd<nDofNode;nd++){ double*M=&Bd[9*nd]; real*O=&Bi[9*nd];
          double a=M[0],b2=M[1],c2=M[2],d=M[4],e=M[5],f=M[8];
          double C00=d*f-e*e,C01=c2*e-b2*f,C02=b2*e-c2*d,det=a*C00+b2*C01+c2*C02;
          if (fabs(det)>1e-300){ double id=1.0/det,C11=a*f-c2*c2,C12=b2*c2-a*e,C22=a*d-b2*b2;
            O[0]=(real)(C00*id);O[1]=(real)(C01*id);O[2]=(real)(C02*id); O[3]=O[1];O[4]=(real)(C11*id);O[5]=(real)(C12*id); O[6]=O[2];O[7]=O[5];O[8]=(real)(C22*id);
          } else { for(i32 k=0;k<9;k++)O[k]=0; O[0]=O[4]=O[8]=(real)((M[0]>0)?1.0/M[0]:1.0); } }
        d_Binv=cpR(Bi.data(),(size_t)9*nDofNode);
      }
      real *d_intInv=nullptr,*d_cutInv=nullptr;
      if (pcMode==2) {   // element additive-Schwarz: dense per-element SPD inverses (vol + Nitsche penalty + shift)
        const i32 m3=ndof3; const char*she=getenv("CUT_SHIFT"); const double shift=she?atof(she):1e-3;
        std::vector<double> Ki((size_t)m3*m3); for(size_t i=0;i<(size_t)m3*m3;i++) Ki[i]=Kref[i];
        { double tr=0; for(i32 i=0;i<m3;i++) tr+=Ki[(size_t)i*m3+i]; double sh=shift*tr/m3;
          for(i32 i=0;i<m3;i++) Ki[(size_t)i*m3+i]+=sh; if(!invertSPD(Ki.data(),m3)) printf("WARNING: interior Schwarz block not SPD\n"); }
        std::vector<real> Kir((size_t)m3*m3); for(size_t i=0;i<(size_t)m3*m3;i++) Kir[i]=(real)Ki[i]; d_intInv=cpR(Kir.data(),(size_t)m3*m3);
        std::vector<real> cutInvH((size_t)nCutQ*m3*m3);
        #pragma omp parallel for schedule(dynamic,2)
        for(i32 c=0;c<nCutQ;c++){ i32 e=cutElem[c]; std::vector<double> Ke((size_t)m3*m3,0.0);
          i32 v0=volOff[c], nv=volOff[c+1]-v0, s0=surfOff[c], ns=surfOff[c+1]-s0;
          real ulc[3*QN_MAX*QN_MAX*QN_MAX], ylc[3*QN_MAX*QN_MAX*QN_MAX];
          for(i32 k=0;k<m3;k++){ for(i32 i=0;i<m3;i++) ulc[i]=(i==k)?(real)1:(real)0;
            qpElemCoreSaye(Bp,(real)mu,(real)lam,h,&volPool[v0],nv,ulc,ylc);          // volume stiffness column
            for(i32 q=0;q<ns;q++){ if(!surfDir[s0+q]) continue;                        // + Nitsche penalty (SPD part)
              real xr[3]={surfPool[s0+q].x[0],surfPool[s0+q].x[1],surfPool[s0+q].x[2]}, vb[QN_MAX*QN_MAX*QN_MAX];
              Bp.allVal(xr,vb); real hw=surfPool[s0+q].w*h; real uval[3]={0,0,0};
              for(i32 b=0;b<ndof;b++)for(i32 l=0;l<3;l++) uval[l]+=ulc[3*b+l]*vb[b];
              for(i32 a=0;a<ndof;a++)for(i32 l=0;l<3;l++) ylc[3*a+l]+=hw*(real)gammaD_*uval[l]*vb[a]; }
            for(i32 r=0;r<m3;r++) Ke[(size_t)r*m3+k]=ylc[r]; }
          double tr=0; for(i32 i=0;i<m3;i++) tr+=Ke[(size_t)i*m3+i]; double sh=shift*tr/m3;
          for(i32 i=0;i<m3;i++) Ke[(size_t)i*m3+i]+=sh;
          if(!invertSPD(Ke.data(),m3)){ std::vector<double> di(m3); for(i32 i=0;i<m3;i++) di[i]=Ke[(size_t)i*m3+i];
            for(size_t i=0;i<(size_t)m3*m3;i++) Ke[i]=0; for(i32 i=0;i<m3;i++) Ke[(size_t)i*m3+i]=(di[i]>0?1.0/di[i]:1.0); }
          for(i32 i=0;i<m3*m3;i++) cutInvH[(size_t)c*m3*m3+i]=(real)Ke[i]; }
        d_cutInv=cpR(cutInvH.data(),(size_t)nCutQ*m3*m3);
        printf("schwarz: %d cut blocks + 1 shared interior (%dx%d dense inverses, %.0f MB)\n",
               nCutQ,m3,m3,(double)((size_t)(nCutQ+1)*m3*m3*sizeof(real))/1e6);
      }
      // ---- p-multigrid p-hierarchy (p -> p-1 -> ... -> 1); NODE-space V-cycle; non-periodic only ----
      #define MAXLEV 10
      i32 nLev=1; real *d_diagN=nullptr,*d_cz=nullptr,*d_cp=nullptr,*d_cAp=nullptr,*d_cr=nullptr;
      CutDev Slev[MAXLEV]{}; IgaBasis Blev[MAXLEV];
      i32 nNodeLev[MAXLEV]={0}, nNLev[MAXLEV]={0}, ndofLev[MAXLEV]={0}, nd3Lev[MAXLEV]={0}, mGLev[MAXLEV]={0}, wLev[MAXLEV]={0}, nDofNodeLev[MAXLEV]={0};
      real *d_diagNLev[MAXLEV]={0}, *d_xLev[MAXLEV]={0}, *d_rLev[MAXLEV]={0}, *d_tmpLev[MAXLEV]={0}, *d_resLev[MAXLEV]={0}, *d_dirLev[MAXLEV]={0}, *d_pwLev[MAXLEV]={0};
      real *d_mult3Lev[MAXLEV]={0}, *d_projLev[MAXLEV]={0}, *d_ptmpLev[MAXLEV]={0};   // per-level pitch-tie projection (dof buffer, mult, node scratch)
      i32 *d_pcolLev[MAXLEV]={0};
      double aLev[MAXLEV]={0}, bLev[MAXLEV]={0};
      i32 nELev[MAXLEV]={0}, nCutLev[MAXLEV]={0}, nGFLev[MAXLEV]={0}, hcylLev[MAXLEV]={0};   // h-MG: per-level grid sizes; hcyl=1 -> use the general metric kernel
      std::vector<std::vector<i32>> eNodeLev(MAXLEV); std::vector<void*> pmgFree;
      for(i32 L=0;L<MAXLEV;L++){ nELev[L]=nE; nCutLev[L]=nCutQ; nGFLev[L]=nGFQ; hcylLev[L]=cyl?1:0; }   // p-levels share the fine grid
      // line-implicit smoother state (built further below; lambdas capture by reference so the
      // declarations only need to precede chebSmoothL, not the build)
      i32 lineSm=0; { const char*e=getenv("CUT_SMOOTH"); if(e&&std::string(e)=="line") lineSm=1; }
      i32 nLine=0, lineMax3=0; real* d_Minv=nullptr; size_t* d_mOff=nullptr; i32 *d_lOff=nullptr,*d_lNode=nullptr;
      auto lineApply=[&](real*dir,const real*res,real c1,real c2){
        cutLineChebK<<<(nLine<65535?nLine:65535),128,(size_t)lineMax3*sizeof(real)>>>(dir,res,d_Minv,d_mOff,d_lOff,d_lNode,c1,c2,nLine); };
      i32 cDeg=3,cMaxit=100; double cTol=5e-2; { const char*e; if((e=getenv("CUT_CHEBDEG")))cDeg=atoi(e); if((e=getenv("CUT_CMAXIT")))cMaxit=atoi(e); if((e=getenv("CUT_CTOL")))cTol=atof(e); }
      if (pcMode==3) {
        nLev=p;   // p3 -> {p3,p2,p1}; p2 -> {p2,p1}
        auto buildLevel=[&](i32 pc,i32 L){ IgaBasis Bc; Bc.init(pc); Blev[L]=Bc; i32 nc=pc+1,ndc=nc*nc*nc,nd3c=3*ndc,mGc=2*nd3c;
          ndofLev[L]=ndc; nd3Lev[L]=nd3c; mGLev[L]=mGc;
          std::unordered_map<u64,i32> nid; nid.reserve((size_t)nE*ndc); std::vector<i32>& en=eNodeLev[L]; en.assign((size_t)nE*ndc,0); i32 nn=0;
          for(i32 e=0;e<nE;e++)for(i32 a=0;a<ndc;a++){ i32 i=a%nc,j=(a/nc)%nc,k=a/(nc*nc); i32 I=pc*elems[e].ci+i,J=pc*elems[e].cj+j,K=pc*elems[e].ck+k; u64 key=qpKey(I,J,K);
            auto it=nid.find(key); i32 id; if(it==nid.end()){id=nn++;nid[key]=id;}else id=it->second; en[(size_t)e*ndc+a]=id; }
          nNodeLev[L]=nn; nNLev[L]=3*nn;
          // periodic pitch tie at this level: nodes at J=pc*nThetaCells are slaves of masters at J=0
          std::vector<i32> nIc(nn),nJc(nn),nKc(nn);
          for(auto&kv:nid){ u64 key=kv.first; i32 id=kv.second; nIc[id]=(i32)(key&0x1FFFFF); nJc[id]=(i32)((key>>21)&0x1FFFFF); nKc[id]=(i32)((key>>42)&0x1FFFFF); }
          i32 JmaxC=pc*nThetaCells; std::vector<i32> ridC(nn,-1); std::vector<char> rotC(nn,0); i32 ndn=0;
          for(i32 nd=0;nd<nn;nd++){ if(per&&nJc[nd]==JmaxC) continue; ridC[nd]=ndn++; }
          for(i32 nd=0;nd<nn;nd++){ if(ridC[nd]>=0)continue; auto it=nid.find(qpKey(nIc[nd],0,nKc[nd])); if(it==nid.end()){ridC[nd]=ndn++;} else {ridC[nd]=ridC[it->second];rotC[nd]=1;} }
          nDofNodeLev[L]=ndn;
          std::vector<real> mult3((size_t)3*ndn,(real)0); for(i32 nd=0;nd<nn;nd++){ real*m=&mult3[3*ridC[nd]]; m[0]+=1;m[1]+=1;m[2]+=1; }
          i32* d_ridC=cpI(ridC.data(),nn); char* d_rotC; cudaMallocManaged(&d_rotC,nn); for(i32 i=0;i<nn;i++)d_rotC[i]=rotC[i];
          d_mult3Lev[L]=cpR(mult3.data(),(size_t)3*ndn); d_projLev[L]=alR((size_t)3*ndn); d_ptmpLev[L]=alR((size_t)3*nn);
          pmgFree.push_back(d_ridC); pmgFree.push_back(d_rotC); pmgFree.push_back(d_mult3Lev[L]); pmgFree.push_back(d_projLev[L]); pmgFree.push_back(d_ptmpLev[L]);
          std::vector<real> Kr((size_t)nd3c*nd3c);
          { std::vector<real> u(nd3c),y(nd3c); for(i32 cc=0;cc<nd3c;cc++){ for(i32 i=0;i<nd3c;i++)u[i]=(i==cc)?(real)1:(real)0; qpElemUncut(Bc,(real)mu,(real)lam,h,u.data(),y.data()); for(i32 r=0;r<nd3c;r++)Kr[(size_t)r*nd3c+cc]=y[r]; } }
          double Dl0c[QP_MAX+1][QN_MAX], Dl1c[QP_MAX+1][QN_MAX];
          for(i32 l=1;l<=pc;l++)for(i32 a=0;a<nc;a++){ real v=Bc.dlFace(l,a); Dl0c[l][a]=(double)v; Dl1c[l][a]=(double)v; }
          auto ghostc=[&](i32 d,const double*uMP,double*yMP){ for(i32 i=0;i<mGc;i++)yMP[i]=0; i32 t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(pc+1);
            for(i32 q1=0;q1<g1.n;q1++)for(i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2]; real L1[QN_MAX],L2[QN_MAX]; Bc.basis1(g1.x[q1],L1); Bc.basis1(g1.x[q2],L2);
              double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
              for(i32 a=0;a<ndc;a++){ i32 ix[3]={a%nc,(a/nc)%nc,a/(nc*nc)}; double Lt=L1[ix[t1]]*L2[ix[t2]]; for(i32 l=1;l<=pc;l++){cP[l][a]=Dl0c[l][ix[d]]*Lt;cM[l][a]=Dl1c[l][ix[d]]*Lt;} }
              for(i32 l=1;l<=pc;l++){ double cf=gammaG_*h*w,jU[3]={0,0,0}; for(i32 a=0;a<ndc;a++)for(i32 cc=0;cc<3;cc++)jU[cc]+=uMP[nd3c+3*a+cc]*cP[l][a]-uMP[3*a+cc]*cM[l][a];
                for(i32 a=0;a<ndc;a++)for(i32 cc=0;cc<3;cc++){yMP[nd3c+3*a+cc]+=cf*cP[l][a]*jU[cc];yMP[3*a+cc]+=cf*(-cM[l][a])*jU[cc];} } } };
          std::vector<double> Kg[3];
          for(i32 d=0;d<3;d++){ Kg[d].assign((size_t)mGc*mGc,0.0); std::vector<double>ue(mGc),ye(mGc); for(i32 cq=0;cq<mGc;cq++){for(i32 a=0;a<mGc;a++)ue[a]=(a==cq)?1.0:0.0; ghostc(d,ue.data(),ye.data()); for(i32 r=0;r<mGc;r++)Kg[d][(size_t)r*mGc+cq]=ye[r];} }
          std::vector<double> dN((size_t)3*nn,0.0);
          if (cyl) {   // cylindrical: metric re-quadrature diagonal (tensor-GLL interior + Saye cut, + Nanson penalty)
            real gb[3*QN_MAX*QN_MAX*QN_MAX]; double penC=gammaD_/h;
            for(i32 e=0;e<nE;e++){ const i32*nod=&en[(size_t)e*ndc]; bool cut=elems[e].cut; i32 c=cut?cutIdx[e]:-1;
              i32 nvv=cut?(volOff[c+1]-volOff[c]):ndc;
              for(i32 q=0;q<nvv;q++){ real xr[3],wq;
                if(cut){ const SayeNode&vp=volPool[volOff[c]+q]; xr[0]=vp.x[0];xr[1]=vp.x[1];xr[2]=vp.x[2]; wq=vp.w; }
                else{ i32 i=q%nc,j=(q/nc)%nc,k=q/(nc*nc); xr[0]=Bc.t[i];xr[1]=Bc.t[j];xr[2]=Bc.t[k]; wq=Bc.wq[i]*Bc.wq[j]*Bc.wq[k]; }
                double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ); Bc.allGradRef(xr,gb); double wdet=fabs(detJ)*wq;
                for(i32 a=0;a<ndc;a++){ double gX[3]; for(i32 d2=0;d2<3;d2++) gX[d2]=Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2];
                  double gsq=gX[0]*gX[0]+gX[1]*gX[1]+gX[2]*gX[2]; for(i32 l=0;l<3;l++) dN[3*nod[a]+l]+=wdet*(mu*(gsq+gX[l]*gX[l])+lam*gX[l]*gX[l]); } }
              if(cut){ i32 s0=surfOff[c],ns=surfOff[c+1]-s0;
                for(i32 q=0;q<ns;q++){ if(!surfDir[s0+q])continue; real xr[3]={surfPool[s0+q].x[0],surfPool[s0+q].x[1],surfPool[s0+q].x[2]};
                  double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ); double nref[3]={surfPool[s0+q].n[0],surfPool[s0+q].n[1],surfPool[s0+q].n[2]},nraw[3];
                  for(i32 i2=0;i2<3;i2++)nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2]; double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if(nmag<=0)continue;
                  double dS=fabs(detJ)*nmag*surfPool[s0+q].w; std::vector<real>vb(ndc); Bc.allVal(xr,vb.data()); for(i32 a=0;a<ndc;a++)for(i32 l=0;l<3;l++)dN[3*nod[a]+l]+=dS*penC*vb[a]*vb[a]; } } }
          } else
          for(i32 e=0;e<nE;e++){ const i32*nod=&en[(size_t)e*ndc];
            if(!elems[e].cut){ for(i32 a=0;a<ndc;a++)for(i32 l=0;l<3;l++)dN[3*nod[a]+l]+=Kr[(size_t)(3*a+l)*nd3c+(3*a+l)]; }
            else { i32 c=cutIdx[e],v0=volOff[c],nv=volOff[c+1]-v0,s0=surfOff[c],ns=surfOff[c+1]-s0; std::vector<real>u(nd3c),y(nd3c);
              for(i32 cc=0;cc<nd3c;cc++){ for(i32 i=0;i<nd3c;i++)u[i]=(i==cc)?(real)1:(real)0; qpElemCoreSaye(Bc,(real)mu,(real)lam,h,&volPool[v0],nv,u.data(),y.data());
                for(i32 q=0;q<ns;q++){ if(!surfDir[s0+q])continue; real xr[3]={surfPool[s0+q].x[0],surfPool[s0+q].x[1],surfPool[s0+q].x[2]}; std::vector<real>vb(ndc); Bc.allVal(xr,vb.data()); real hw=surfPool[s0+q].w*h;
                  real uval[3]={0,0,0}; for(i32 b=0;b<ndc;b++)for(i32 l=0;l<3;l++)uval[l]+=u[3*b+l]*vb[b]; for(i32 a=0;a<ndc;a++)for(i32 l=0;l<3;l++)y[3*a+l]+=hw*(real)gammaD_*uval[l]*vb[a]; }
                i32 a=cc/3,l=cc%3; dN[3*nod[a]+l]+=y[cc]; } } }
          for(i32 f=0;f<nGFQ;f++){ const GF&F=gf[f]; const i32*nodM=&en[(size_t)F.eM*ndc],*nodP=&en[(size_t)F.eP*ndc]; const double*K=Kg[F.d].data();
            for(i32 a=0;a<ndc;a++)for(i32 l=0;l<3;l++){dN[3*nodM[a]+l]+=K[(size_t)(3*a+l)*mGc+(3*a+l)];dN[3*nodP[a]+l]+=K[(size_t)(nd3c+3*a+l)*mGc+(nd3c+3*a+l)];} }
          for(size_t i=0;i<(size_t)3*nn;i++)if(dN[i]<=0)dN[i]=1.0;
          i32* d_en=cpI(en.data(),(size_t)nE*ndc); real* d_Kr=cpR(Kr.data(),(size_t)nd3c*nd3c); real* d_KgL[3];
          for(i32 d=0;d<3;d++){ std::vector<real>t((size_t)mGc*mGc); for(size_t i=0;i<(size_t)mGc*mGc;i++)t[i]=(real)Kg[d][i]; d_KgL[d]=cpR(t.data(),(size_t)mGc*mGc); }
          { std::vector<real>t((size_t)3*nn); for(size_t i=0;i<(size_t)3*nn;i++)t[i]=(real)dN[i]; d_diagNLev[L]=cpR(t.data(),(size_t)3*nn); }
          CutDev Sc{}; Sc.B=Bc; Sc.nE=nE; Sc.nCut=nCutQ; Sc.nGFQ=nGFQ; Sc.nNode=nn; Sc.ndof=ndc; Sc.ndof3=nd3c; Sc.mG=mGc;
          Sc.h=h; Sc.mu=(real)mu; Sc.lam=(real)lam; Sc.gammaD=(real)gammaD_; Sc.cph=(real)cph; Sc.sph=(real)sph;
          Sc.eNode=d_en; Sc.nMap=d_ridC; Sc.nRot=d_rotC; Sc.intList=d_intList; Sc.cutElem=d_cutElem;
          Sc.volP=d_volP; Sc.surfP=d_surfP; Sc.volOff=d_volOff; Sc.surfOff=d_surfOff; Sc.surfDir=d_surfDir;
          Sc.gfM=d_gfM; Sc.gfP=d_gfP; Sc.gfD=d_gfD; Sc.Kref=d_Kr; Sc.Kg[0]=d_KgL[0]; Sc.Kg[1]=d_KgL[1]; Sc.Kg[2]=d_KgL[2];
          Sc.cyl=cyl?1:0; Sc.ls=ls; Sc.eCijk=d_eCijk; Sc.eCut=d_eCut; Sc.volJ=d_volJ; Sc.surfJ=d_surfJ;
          Sc.sp[0]=Sc.sp[1]=Sc.sp[2]=1;   // p-levels share the fine grid
          Slev[L]=Sc;
          d_xLev[L]=alR(3*nn); d_rLev[L]=alR(3*nn); d_tmpLev[L]=alR(3*nn); d_resLev[L]=alR(3*nn); d_dirLev[L]=alR(3*nn);
          for(void*pp:{(void*)d_en,(void*)d_Kr,(void*)d_KgL[0],(void*)d_KgL[1],(void*)d_KgL[2],(void*)d_diagNLev[L],(void*)d_xLev[L],(void*)d_rLev[L],(void*)d_tmpLev[L],(void*)d_resLev[L],(void*)d_dirLev[L]}) pmgFree.push_back(pp); };
        // level 0 = existing fine operator (reuse d_eNode/d_Kref/d_Kg*), fine node diagonal
        { std::vector<real> t((size_t)3*nNodeQ); for(size_t i=0;i<(size_t)3*nNodeQ;i++)t[i]=(real)diagNode[i]; d_diagN=cpR(t.data(),(size_t)3*nNodeQ); }
        Blev[0]=Bp; nNodeLev[0]=nNodeQ; nNLev[0]=nN; ndofLev[0]=ndof; nd3Lev[0]=ndof3; mGLev[0]=mG; d_diagNLev[0]=d_diagN; eNodeLev[0]=eNodeQ;
        { nDofNodeLev[0]=nDofNode; std::vector<real> mult3((size_t)3*nDofNode,(real)0); for(i32 nd=0;nd<nNodeQ;nd++){ real*m=&mult3[3*realIdx[nd]]; m[0]+=1;m[1]+=1;m[2]+=1; }
          d_mult3Lev[0]=cpR(mult3.data(),(size_t)3*nDofNode); d_projLev[0]=alR((size_t)3*nDofNode); d_ptmpLev[0]=alR((size_t)3*nNodeQ);
          for(void*pp:{(void*)d_mult3Lev[0],(void*)d_projLev[0],(void*)d_ptmpLev[0]}) pmgFree.push_back(pp); }
        { CutDev S0{}; S0.B=Bp; S0.nE=nE; S0.nCut=nCutQ; S0.nGFQ=nGFQ; S0.nNode=nNodeQ; S0.ndof=ndof; S0.ndof3=ndof3; S0.mG=mG;
          S0.h=h; S0.mu=(real)mu; S0.lam=(real)lam; S0.gammaD=(real)gammaD_; S0.cph=(real)cph; S0.sph=(real)sph;
          S0.eNode=d_eNode; S0.nMap=d_nMap; S0.nRot=d_nRot; S0.intList=d_intList; S0.cutElem=d_cutElem;
          S0.volP=d_volP; S0.surfP=d_surfP; S0.volOff=d_volOff; S0.surfOff=d_surfOff; S0.surfDir=d_surfDir;
          S0.gfM=d_gfM; S0.gfP=d_gfP; S0.gfD=d_gfD; S0.Kref=d_Kref; S0.Kg[0]=d_Kg0; S0.Kg[1]=d_Kg1; S0.Kg[2]=d_Kg2;
          S0.cyl=cyl?1:0; S0.ls=ls; S0.eCijk=d_eCijk; S0.eCut=d_eCut; S0.volJ=d_volJ; S0.surfJ=d_surfJ;
          S0.sp[0]=S0.sp[1]=S0.sp[2]=1; Slev[0]=S0; }
        d_tmpLev[0]=alR(nN); d_resLev[0]=alR(nN); d_dirLev[0]=alR(nN);
        for(void*pp:{(void*)d_diagN,(void*)d_tmpLev[0],(void*)d_resLev[0],(void*)d_dirLev[0]}) pmgFree.push_back(pp);
        for(i32 L=1; L<nLev; L++) buildLevel(p-L,L);
        for(i32 L=0; L<nLev-1; L++){ i32 pf=p-L, ncf=pf+1, ndf=ncf*ncf*ncf, ndc=ndofLev[L+1]; IgaBasis Bc=Blev[L+1]; i32 ncc=Bc.n;
          real tf[PNC]; gllNodes(pf,tf); wLev[L]=ndc;
          std::vector<i32> pcol((size_t)ndc*nNodeLev[L],0); std::vector<real> pw((size_t)ndc*nNodeLev[L],(real)0);
          for(i32 e=0;e<nE;e++){ const i32*nodF=&eNodeLev[L][(size_t)e*ndf]; const i32*nodC=&eNodeLev[L+1][(size_t)e*ndc];
            for(i32 a=0;a<ndf;a++){ i32 fi=a%ncf,fj=(a/ncf)%ncf,fk=a/(ncf*ncf); real rx=tf[fi],ry=tf[fj],rz=tf[fk]; i32 fn=nodF[a];
              real Lx[QN_MAX],Ly[QN_MAX],Lz[QN_MAX]; Bc.basis1(rx,Lx); Bc.basis1(ry,Ly); Bc.basis1(rz,Lz);
              for(i32 b=0;b<ndc;b++){ i32 bx=b%ncc,by=(b/ncc)%ncc,bz=b/(ncc*ncc); pcol[(size_t)ndc*fn+b]=nodC[b]; pw[(size_t)ndc*fn+b]=(real)(Lx[bx]*Ly[by]*Lz[bz]); } } }
          d_pcolLev[L]=cpI(pcol.data(),(size_t)ndc*nNodeLev[L]); d_pwLev[L]=cpR(pw.data(),(size_t)ndc*nNodeLev[L]);
          pmgFree.push_back(d_pcolLev[L]); pmgFree.push_back(d_pwLev[L]); }
        i32 ncoarse=nNLev[nLev-1]; d_cz=alR(ncoarse); d_cp=alR(ncoarse); d_cAp=alR(ncoarse); d_cr=alR(ncoarse);
        for(void*pp:{(void*)d_cz,(void*)d_cp,(void*)d_cAp,(void*)d_cr}) pmgFree.push_back(pp);
        printf("pmg    : %d-level p-hierarchy [", nLev); for(i32 L=0;L<nLev;L++) printf("p%d(%dn)%s", p-L, nNodeLev[L], L<nLev-1?" -> ":""); printf("]\n");
      }
      // ================ h-multigrid: SEMI-COARSENED geometric hierarchy (CUT_PC=hmg) ================
      //  A coarse cell agglomerates sp=(2,1,2) fine cells by default: coarsen r and z, keep theta FINE.
      //  Semi-coarsening is mandatory here -- the blade is only ~1-2 cells thick across the 4-cell pitch,
      //  so isotropic coarsening would drive it sub-cell and the coarse geometry would vanish.
      //  Coarse quadrature = UNION of the sub-cells' existing (already NNLS-pruned) Saye rules remapped
      //  into the coarse reference cell -- no re-cutting, no new level-set/Saye work.  Coarse cells are
      //  anisotropic (2h x h x 2h), which cutMetric/cutCylK handle via CutDev::sp.
      std::vector<std::vector<i32>> hI(MAXLEV),hJ(MAXLEV),hK(MAXLEV);   // per-level cell lattice coords
      i32 hSp[MAXLEV][3];
      if (pcMode==4) {
        i32 s0c[3]={2,1,2}; { const char*e=getenv("CUT_HSEMI"); if(e) sscanf(e,"%d,%d,%d",&s0c[0],&s0c[1],&s0c[2]); }
        i32 nLmax=MAXLEV; { const char*e=getenv("CUT_HLEV"); if(e) nLmax=atoi(e); }
        const i32 nsub0=s0c[0]*s0c[1]*s0c[2];
        if(nsub0<1){ printf("ERROR: bad CUT_HSEMI\n"); return; }
        // ---- level 0 = the existing fine operator (reuse fine device arrays) ----
        { std::vector<real> t((size_t)3*nNodeQ); for(size_t i=0;i<(size_t)3*nNodeQ;i++)t[i]=(real)diagNode[i]; d_diagN=cpR(t.data(),(size_t)3*nNodeQ); }
        Blev[0]=Bp; nNodeLev[0]=nNodeQ; nNLev[0]=nN; ndofLev[0]=ndof; nd3Lev[0]=ndof3; mGLev[0]=mG; d_diagNLev[0]=d_diagN; eNodeLev[0]=eNodeQ;
        hSp[0][0]=hSp[0][1]=hSp[0][2]=1;
        hI[0].resize(nE); hJ[0].resize(nE); hK[0].resize(nE);
        for(i32 e=0;e<nE;e++){ hI[0][e]=elems[e].ci; hJ[0][e]=elems[e].cj; hK[0][e]=elems[e].ck; }
        { nDofNodeLev[0]=nDofNode; std::vector<real> mult3((size_t)3*nDofNode,(real)0); for(i32 nd=0;nd<nNodeQ;nd++){ real*m=&mult3[3*realIdx[nd]]; m[0]+=1;m[1]+=1;m[2]+=1; }
          d_mult3Lev[0]=cpR(mult3.data(),(size_t)3*nDofNode); d_projLev[0]=alR((size_t)3*nDofNode); d_ptmpLev[0]=alR((size_t)3*nNodeQ);
          for(void*pp:{(void*)d_mult3Lev[0],(void*)d_projLev[0],(void*)d_ptmpLev[0]}) pmgFree.push_back(pp); }
        { CutDev S0{}; S0.B=Bp; S0.nE=nE; S0.nCut=nCutQ; S0.nGFQ=nGFQ; S0.nNode=nNodeQ; S0.ndof=ndof; S0.ndof3=ndof3; S0.mG=mG;
          S0.h=h; S0.mu=(real)mu; S0.lam=(real)lam; S0.gammaD=(real)gammaD_; S0.cph=(real)cph; S0.sph=(real)sph;
          S0.eNode=d_eNode; S0.nMap=d_nMap; S0.nRot=d_nRot; S0.intList=d_intList; S0.cutElem=d_cutElem;
          S0.volP=d_volP; S0.surfP=d_surfP; S0.volOff=d_volOff; S0.surfOff=d_surfOff; S0.surfDir=d_surfDir;
          S0.gfM=d_gfM; S0.gfP=d_gfP; S0.gfD=d_gfD; S0.Kref=d_Kref; S0.Kg[0]=d_Kg0; S0.Kg[1]=d_Kg1; S0.Kg[2]=d_Kg2;
          S0.cyl=cyl?1:0; S0.ls=ls; S0.eCijk=d_eCijk; S0.eCut=d_eCut; S0.volJ=d_volJ; S0.surfJ=d_surfJ;
          S0.sp[0]=S0.sp[1]=S0.sp[2]=1; Slev[0]=S0; }
        d_tmpLev[0]=alR(nN); d_resLev[0]=alR(nN); d_dirLev[0]=alR(nN);
        for(void*pp:{(void*)d_diagN,(void*)d_tmpLev[0],(void*)d_resLev[0],(void*)d_dirLev[0]}) pmgFree.push_back(pp);

        // host metric with per-direction cell extent (mirrors cutMetric; Cartesian -> diagonal)
        auto metricSp=[&](i32 ci,i32 cj,i32 ck,const i32 sp[3],const real xr[3],double Jinv[3][3],double&detJ){
          double hx=sp[0]*h, hy=sp[1]*h, hz=sp[2]*h;
          if(!cyl){ for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++)Jinv[i][j]=0;
            Jinv[0][0]=1.0/hx; Jinv[1][1]=1.0/hy; Jinv[2][2]=1.0/hz; detJ=hx*hy*hz; return; }
          double r=ci*h+hx*xr[0]+ls.org[0], s=cj*h+hy*xr[1]+ls.org[1], z=ck*h+hz*xr[2]+ls.org[2];
          double th=s/ls.rRef+(double)ls.thc((real)z), thp=(double)ls.thcSlope((real)z), ct=cos(th), st=sin(th);
          double J[3][3]={{ hx*ct, hy*(-r*st/ls.rRef), hz*(-r*st*thp) },{ hx*st, hy*( r*ct/ls.rRef), hz*( r*ct*thp) },{ 0,0,hz }};
          detJ=inv3(J,Jinv); };

        // ---- build coarse level L from level L-1 by agglomerating s0c fine-of-that-level cells ----
        //  Returns false when the level would be degenerate (no further coarsening possible).
        auto buildH=[&](i32 L,const std::vector<SayeNode>&vPf,const std::vector<i32>&vOf,
                        const std::vector<SayeNode>&sPf,const std::vector<i32>&sOf,const std::vector<char>&sDf,
                        const std::vector<i32>&cutIdxF,const std::vector<char>&cutF,
                        std::vector<SayeNode>&vPc,std::vector<i32>&vOc,std::vector<SayeNode>&sPc,
                        std::vector<i32>&sOc,std::vector<char>&sDc,std::vector<i32>&cutIdxC,std::vector<char>&cutC)->bool{
          const i32 F=L-1; i32 nEf=(i32)hI[F].size();
          for(i32 d=0;d<3;d++) hSp[L][d]=hSp[F][d]*s0c[d];
          // agglomerate: coarse cell index = (level-(L-1) cell index) / s0c
          std::unordered_map<u64,i32> cmap; std::vector<std::vector<i32>> sub;
          hI[L].clear(); hJ[L].clear(); hK[L].clear();
          std::vector<i32> owner(nEf);
          for(i32 e=0;e<nEf;e++){ i32 C0=hI[F][e]/s0c[0], C1=hJ[F][e]/s0c[1], C2=hK[F][e]/s0c[2];
            u64 key=qpCellKey(C0,C1,C2); auto it=cmap.find(key); i32 ce;
            if(it==cmap.end()){ ce=(i32)hI[L].size(); cmap[key]=ce; hI[L].push_back(C0);hJ[L].push_back(C1);hK[L].push_back(C2); sub.push_back({}); }
            else ce=it->second;
            sub[ce].push_back(e); owner[e]=ce; }
          i32 nEc=(i32)hI[L].size();
          if(nEc<2 || nEc==nEf) return false;         // no useful coarsening left
          // a coarse cell is UNCUT only when every sub-cell is present AND uncut
          cutC.assign(nEc,1);
          for(i32 ce=0;ce<nEc;ce++){ bool allIn=((i32)sub[ce].size()==nsub0), anyCut=false;
            for(i32 fe:sub[ce]) if(cutF[fe]) anyCut=true;
            cutC[ce]=(!allIn||anyCut)?1:0; }
          // ---- agglomerated quadrature: remap each sub-cell rule into the coarse reference cell ----
          //  vol : x_c = (off + x_f)/s ,  w_c = w_f / (sx*sy*sz)
          //  surf: n~ = diag(s)*n_f , n_c = n~/|n~| , w_c = w_f*|n~|/(sx*sy*sz)   (exact Nanson under the
          //        diagonal remap A=diag(1/s): J_f = J_c A, so |detJ_c||J_c^-T n_c| w_c = |detJ_f||J_f^-T n_f| w_f)
          const double sc=1.0/(double)nsub0;
          vPc.clear(); sPc.clear(); sDc.clear(); vOc.assign(1,0); sOc.assign(1,0); cutIdxC.assign(nEc,-1);
          i32 nCutC=0; std::vector<i32> cutList;
          for(i32 ce=0;ce<nEc;ce++) if(cutC[ce]){ cutIdxC[ce]=nCutC++; cutList.push_back(ce); }
          std::vector<std::vector<SayeNode>> cellV(nCutC);
          for(i32 ci2=0;ci2<nCutC;ci2++){
            i32 ce=cutList[ci2]; std::vector<SayeNode>& vAcc=cellV[ci2];
            for(i32 fe:sub[ce]){
              i32 o[3]={ hI[F][fe]-hI[L][ce]*s0c[0], hJ[F][fe]-hJ[L][ce]*s0c[1], hK[F][fe]-hK[L][ce]*s0c[2] };
              if(cutF[fe]){
                i32 c=cutIdxF[fe];
                for(i32 q=vOf[c];q<vOf[c+1];q++){ SayeNode nd=vPf[q];
                  for(i32 d=0;d<3;d++) nd.x[d]=(real)((o[d]+(double)nd.x[d])/s0c[d]);
                  nd.w=(real)(nd.w*sc); vAcc.push_back(nd); }
                for(i32 q=sOf[c];q<sOf[c+1];q++){ SayeNode nd=sPf[q];
                  double nt[3]={(double)nd.n[0]*s0c[0],(double)nd.n[1]*s0c[1],(double)nd.n[2]*s0c[2]};
                  double nm=sqrt(nt[0]*nt[0]+nt[1]*nt[1]+nt[2]*nt[2]); if(nm<=0) continue;
                  for(i32 d=0;d<3;d++){ nd.x[d]=(real)((o[d]+(double)nd.x[d])/s0c[d]); nd.n[d]=(real)(nt[d]/nm); }
                  nd.w=(real)(nd.w*nm*sc); sPc.push_back(nd); sDc.push_back(sDf[q]); }
              } else {   // fully-interior sub-cell: its tensor-GLL rule, remapped
                for(i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
                  SayeNode nd{}; nd.x[0]=(real)((o[0]+(double)Bp.t[i])/s0c[0]); nd.x[1]=(real)((o[1]+(double)Bp.t[j])/s0c[1]); nd.x[2]=(real)((o[2]+(double)Bp.t[k])/s0c[2]);
                  nd.w=(real)(Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*sc); vAcc.push_back(nd); }
              } }
            sOc.push_back((i32)sPc.size()); }
          // NNLS-compress each agglomerated rule back to the ~(2p+1)^3 moment floor.  Agglomeration
          // PRESERVES the total point count, so without this every coarse level costs as much as the
          // fine one and the V-cycle buys nothing.  Exact for the (polynomial) Cartesian stiffness;
          // on the curved cyl metric it is an approximation, which is fine inside a preconditioner.
          size_t rawV=0; for(auto&v:cellV) rawV+=v.size();
          if(!getenv("CUT_NOPRUNE")&&nCutC>0){
            #pragma omp parallel for schedule(dynamic,1)
            for(i32 c=0;c<nCutC;c++){ std::vector<SayeNode> o; compressVol(cellV[c].data(),(i32)cellV[c].size(),p,o);
              if(o.size()<cellV[c].size()) cellV[c].swap(o); } }
          size_t cmpV=0; for(auto&v:cellV) cmpV+=v.size();
          for(i32 c=0;c<nCutC;c++){ for(const SayeNode&nd:cellV[c]) vPc.push_back(nd); vOc.push_back((i32)vPc.size()); }
          nELev[L]=nEc; nCutLev[L]=nCutC; hcylLev[L]=1;   // coarse cells are anisotropic -> always the metric kernel
          ndofLev[L]=ndof; nd3Lev[L]=ndof3; mGLev[L]=mG; Blev[L]=Bp;
          // ---- node numbering on the coarse grid (same order p) + periodic pitch tie ----
          std::unordered_map<u64,i32> nid; std::vector<i32>& en=eNodeLev[L]; en.assign((size_t)nEc*ndof,0); i32 nn=0;
          std::vector<i32> nIc,nJc,nKc;
          for(i32 e=0;e<nEc;e++)for(i32 a=0;a<ndof;a++){ i32 i=a%n,j=(a/n)%n,k=a/(n*n);
            i32 I=p*hI[L][e]+i, J=p*hJ[L][e]+j, K=p*hK[L][e]+k; u64 key=qpKey(I,J,K);
            auto it=nid.find(key); i32 id; if(it==nid.end()){id=nn++;nid[key]=id;nIc.push_back(I);nJc.push_back(J);nKc.push_back(K);}else id=it->second;
            en[(size_t)e*ndof+a]=id; }
          nNodeLev[L]=nn; nNLev[L]=3*nn;
          i32 nThC=nThetaCells; for(i32 q=0;q<L;q++) nThC=(nThC+s0c[1]-1)/s0c[1];
          i32 JmaxC=p*nThC; std::vector<i32> ridC(nn,-1); std::vector<char> rotC(nn,0); i32 ndn=0;
          for(i32 nd=0;nd<nn;nd++){ if(per&&nJc[nd]==JmaxC) continue; ridC[nd]=ndn++; }
          for(i32 nd=0;nd<nn;nd++){ if(ridC[nd]>=0)continue; auto it=nid.find(qpKey(nIc[nd],0,nKc[nd]));
            if(it==nid.end()){ridC[nd]=ndn++;} else {ridC[nd]=ridC[it->second];rotC[nd]=1;} }
          nDofNodeLev[L]=ndn;
          { std::vector<real> mult3((size_t)3*ndn,(real)0); for(i32 nd=0;nd<nn;nd++){ real*m=&mult3[3*ridC[nd]]; m[0]+=1;m[1]+=1;m[2]+=1; }
            d_mult3Lev[L]=cpR(mult3.data(),(size_t)3*ndn); }
          d_projLev[L]=alR((size_t)3*ndn); d_ptmpLev[L]=alR((size_t)3*nn);
          i32* d_ridC=cpI(ridC.data(),nn); char* d_rotC; cudaMallocManaged(&d_rotC,nn); for(i32 i=0;i<nn;i++)d_rotC[i]=rotC[i];
          for(void*pp:{(void*)d_ridC,(void*)d_rotC,(void*)d_mult3Lev[L],(void*)d_projLev[L],(void*)d_ptmpLev[L]}) pmgFree.push_back(pp);
          // ---- ghost faces on the coarse grid; anisotropic ghost matrices cf = gammaG*(h_t1*h_t2/h_d)*w ----
          std::unordered_map<u64,i32> cellIdC; for(i32 e=0;e<nEc;e++) cellIdC[qpCellKey(hI[L][e],hJ[L][e],hK[L][e])]=e;
          std::vector<i32> gfM,gfP,gfD;
          for(i32 e=0;e<nEc;e++){ i32 cc[3]={hI[L][e],hJ[L][e],hK[L][e]};
            for(i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
              if(per&&d==1&&nb[1]==nThC) nb[1]=0;
              auto it=cellIdC.find(qpCellKey(nb[0],nb[1],nb[2])); if(it==cellIdC.end())continue;
              i32 ep=it->second; if(cutC[e]||cutC[ep]){ gfM.push_back(e); gfP.push_back(ep); gfD.push_back(d); } } }
          i32 nGFc=(i32)gfM.size(); nGFLev[L]=nGFc;
          std::vector<double> Kg[3];
          { double Dl0c[QP_MAX+1][QN_MAX], Dl1c[QP_MAX+1][QN_MAX];
            for(i32 l=1;l<=p;l++)for(i32 a=0;a<n;a++){ real v=Bp.dlFace(l,a); Dl0c[l][a]=(double)v; Dl1c[l][a]=(double)v; }
            for(i32 d=0;d<3;d++){ Kg[d].assign((size_t)mG*mG,0.0); i32 t1=(d+1)%3,t2=(d+2)%3;
              double hd=hSp[L][d]*h, ht1=hSp[L][t1]*h, ht2=hSp[L][t2]*h, gscale=ht1*ht2/hd;
              std::vector<double> ue(mG),ye(mG); GaussRule g1=gaussLegendre(p+1);
              for(i32 cq=0;cq<mG;cq++){ for(i32 a=0;a<mG;a++)ue[a]=(a==cq)?1.0:0.0; for(i32 i=0;i<mG;i++)ye[i]=0;
                for(i32 q1=0;q1<g1.n;q1++)for(i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2];
                  real L1[QN_MAX],L2[QN_MAX]; Bp.basis1((real)g1.x[q1],L1); Bp.basis1((real)g1.x[q2],L2);
                  double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
                  for(i32 a=0;a<ndof;a++){ i32 ix[3]={a%n,(a/n)%n,a/(n*n)}; double Lt=L1[ix[t1]]*L2[ix[t2]];
                    for(i32 l=1;l<=p;l++){cP[l][a]=Dl0c[l][ix[d]]*Lt;cM[l][a]=Dl1c[l][ix[d]]*Lt;} }
                  for(i32 l=1;l<=p;l++){ double cf=gammaG_*gscale*w, jU[3]={0,0,0};
                    for(i32 a=0;a<ndof;a++)for(i32 cc2=0;cc2<3;cc2++) jU[cc2]+=ue[ndof3+3*a+cc2]*cP[l][a]-ue[3*a+cc2]*cM[l][a];
                    for(i32 a=0;a<ndof;a++)for(i32 cc2=0;cc2<3;cc2++){ye[ndof3+3*a+cc2]+=cf*cP[l][a]*jU[cc2];ye[3*a+cc2]+=cf*(-cM[l][a])*jU[cc2];} } }
                for(i32 r=0;r<mG;r++)Kg[d][(size_t)r*mG+cq]=ye[r]; } } }
          real* d_KgL[3]; for(i32 d=0;d<3;d++){ std::vector<real>t((size_t)mG*mG); for(size_t i=0;i<(size_t)mG*mG;i++)t[i]=(real)Kg[d][i]; d_KgL[d]=cpR(t.data(),(size_t)mG*mG); }
          // ---- coarse device arrays + node diagonal (metric re-quadrature, matches cutCylK exactly) ----
          std::vector<i32> eCijkC((size_t)3*nEc), eCutC(nEc);
          for(i32 e=0;e<nEc;e++){ eCijkC[3*e]=hI[L][e]*hSp[L][0]; eCijkC[3*e+1]=hJ[L][e]*hSp[L][1]; eCijkC[3*e+2]=hK[L][e]*hSp[L][2]; eCutC[e]=cutIdxC[e]; }
          i32 spm=hSp[L][0]<hSp[L][1]?hSp[L][0]:hSp[L][1]; spm=spm<hSp[L][2]?spm:hSp[L][2];
          double penC=gammaD_/(spm*h);
          std::vector<double> dN((size_t)3*nn,0.0); double volC=0, srfC=0;
          { real gb[3*QN_MAX*QN_MAX*QN_MAX];
            for(i32 e=0;e<nEc;e++){ const i32*nod=&en[(size_t)e*ndof]; bool cut=cutC[e]!=0; i32 c=cutIdxC[e];
              i32 nvv=cut?(vOc[c+1]-vOc[c]):ndof;
              for(i32 q=0;q<nvv;q++){ real xr[3],wq;
                if(cut){ const SayeNode&vp=vPc[vOc[c]+q]; xr[0]=vp.x[0];xr[1]=vp.x[1];xr[2]=vp.x[2]; wq=vp.w; }
                else{ i32 i=q%n,j=(q/n)%n,k=q/(n*n); xr[0]=Bp.t[i];xr[1]=Bp.t[j];xr[2]=Bp.t[k]; wq=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; }
                double Jinv[3][3],detJ; metricSp(eCijkC[3*e],eCijkC[3*e+1],eCijkC[3*e+2],hSp[L],xr,Jinv,detJ);
                Bp.allGradRef(xr,gb); double wdet=fabs(detJ)*wq; volC+=wdet;
                for(i32 a=0;a<ndof;a++){ double gX[3]; for(i32 d2=0;d2<3;d2++) gX[d2]=Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2];
                  double gsq=gX[0]*gX[0]+gX[1]*gX[1]+gX[2]*gX[2];
                  for(i32 l=0;l<3;l++) dN[3*nod[a]+l]+=wdet*(mu*(gsq+gX[l]*gX[l])+lam*gX[l]*gX[l]); } }
              if(cut){ i32 s0=sOc[c],ns=sOc[c+1]-s0;
                for(i32 q=0;q<ns;q++){ if(!sDc[s0+q])continue; real xr[3]={sPc[s0+q].x[0],sPc[s0+q].x[1],sPc[s0+q].x[2]};
                  double Jinv[3][3],detJ; metricSp(eCijkC[3*e],eCijkC[3*e+1],eCijkC[3*e+2],hSp[L],xr,Jinv,detJ);
                  double nref[3]={sPc[s0+q].n[0],sPc[s0+q].n[1],sPc[s0+q].n[2]},nraw[3];
                  for(i32 i2=0;i2<3;i2++)nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
                  double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if(nmag<=0)continue;
                  double dS=fabs(detJ)*nmag*sPc[s0+q].w; srfC+=dS;
                  std::vector<real>vb(ndof); Bp.allVal(xr,vb.data());
                  for(i32 a=0;a<ndof;a++)for(i32 l=0;l<3;l++) dN[3*nod[a]+l]+=dS*penC*vb[a]*vb[a]; } } }
            for(i32 f=0;f<nGFc;f++){ const i32*nodM=&en[(size_t)gfM[f]*ndof],*nodP=&en[(size_t)gfP[f]*ndof]; const double*K=Kg[gfD[f]].data();
              for(i32 a=0;a<ndof;a++)for(i32 l=0;l<3;l++){dN[3*nodM[a]+l]+=K[(size_t)(3*a+l)*mG+(3*a+l)];dN[3*nodP[a]+l]+=K[(size_t)(ndof3+3*a+l)*mG+(ndof3+3*a+l)];} }
            for(size_t i=0;i<(size_t)3*nn;i++) if(dN[i]<=0) dN[i]=1.0; }
          i32* d_enC=cpI(en.data(),(size_t)nEc*ndof);
          i32* d_eCijkC=cpI(eCijkC.data(),(size_t)3*nEc); i32* d_eCutC=cpI(eCutC.data(),nEc);
          SayeNode *d_vP,*d_sP;
          cudaMallocManaged(&d_vP,(vPc.size()?vPc.size():1)*sizeof(SayeNode)); if(vPc.size())memcpy(d_vP,vPc.data(),vPc.size()*sizeof(SayeNode));
          cudaMallocManaged(&d_sP,(sPc.size()?sPc.size():1)*sizeof(SayeNode)); if(sPc.size())memcpy(d_sP,sPc.data(),sPc.size()*sizeof(SayeNode));
          i32* d_vO=cpI(vOc.data(),vOc.size()); i32* d_sO=cpI(sOc.data(),sOc.size());
          char* d_sD; cudaMallocManaged(&d_sD,std::max((size_t)1,sDc.size())); for(size_t i=0;i<sDc.size();i++)d_sD[i]=sDc[i];
          i32* d_gfM=cpI(gfM.size()?gfM.data():&nEc,std::max((size_t)1,gfM.size()));
          i32* d_gfP=cpI(gfP.size()?gfP.data():&nEc,std::max((size_t)1,gfP.size()));
          i32* d_gfD=cpI(gfD.size()?gfD.data():&nEc,std::max((size_t)1,gfD.size()));
          { std::vector<real>t((size_t)3*nn); for(size_t i=0;i<(size_t)3*nn;i++)t[i]=(real)dN[i]; d_diagNLev[L]=cpR(t.data(),(size_t)3*nn); }
          CutDev Sc{}; Sc.B=Bp; Sc.nE=nEc; Sc.nCut=nCutC; Sc.nGFQ=nGFc; Sc.nNode=nn; Sc.ndof=ndof; Sc.ndof3=ndof3; Sc.mG=mG;
          Sc.h=h; Sc.mu=(real)mu; Sc.lam=(real)lam; Sc.gammaD=(real)gammaD_; Sc.cph=(real)cph; Sc.sph=(real)sph;
          Sc.eNode=d_enC; Sc.nMap=d_ridC; Sc.nRot=d_rotC; Sc.intList=nullptr; Sc.cutElem=nullptr;
          Sc.volP=d_vP; Sc.surfP=d_sP; Sc.volOff=d_vO; Sc.surfOff=d_sO; Sc.surfDir=d_sD;
          Sc.gfM=d_gfM; Sc.gfP=d_gfP; Sc.gfD=d_gfD; Sc.Kref=nullptr; Sc.Kg[0]=d_KgL[0]; Sc.Kg[1]=d_KgL[1]; Sc.Kg[2]=d_KgL[2];
          Sc.cyl=cyl?1:0; Sc.ls=ls; Sc.eCijk=d_eCijkC; Sc.eCut=d_eCutC; Sc.volJ=nullptr; Sc.surfJ=nullptr;
          for(i32 d=0;d<3;d++) Sc.sp[d]=hSp[L][d];
          Slev[L]=Sc;
          d_xLev[L]=alR(3*nn); d_rLev[L]=alR(3*nn); d_tmpLev[L]=alR(3*nn); d_resLev[L]=alR(3*nn); d_dirLev[L]=alR(3*nn);
          for(void*pp:{(void*)d_enC,(void*)d_eCijkC,(void*)d_eCutC,(void*)d_vP,(void*)d_sP,(void*)d_vO,(void*)d_sO,(void*)d_sD,
                       (void*)d_gfM,(void*)d_gfP,(void*)d_gfD,(void*)d_KgL[0],(void*)d_KgL[1],(void*)d_KgL[2],
                       (void*)d_diagNLev[L],(void*)d_xLev[L],(void*)d_rLev[L],(void*)d_tmpLev[L],(void*)d_resLev[L],(void*)d_dirLev[L]}) pmgFree.push_back(pp);
          // ---- h-transfer (level L-1 -> L): coarse Q_p basis sampled at the fine nodes' coarse-ref position ----
          wLev[F]=ndof;
          std::vector<i32> pcol((size_t)ndof*nNodeLev[F],0); std::vector<real> pw((size_t)ndof*nNodeLev[F],(real)0);
          for(i32 e=0;e<nEf;e++){ const i32*nodF=&eNodeLev[F][(size_t)e*ndof]; i32 ce=owner[e]; const i32*nodC=&en[(size_t)ce*ndof];
            i32 o[3]={ hI[F][e]-hI[L][ce]*s0c[0], hJ[F][e]-hJ[L][ce]*s0c[1], hK[F][e]-hK[L][ce]*s0c[2] };
            for(i32 a=0;a<ndof;a++){ i32 fi=a%n,fj=(a/n)%n,fk=a/(n*n); i32 fn=nodF[a];
              real rx=(real)((o[0]+(double)Bp.t[fi])/s0c[0]), ry=(real)((o[1]+(double)Bp.t[fj])/s0c[1]), rz=(real)((o[2]+(double)Bp.t[fk])/s0c[2]);
              real Lx[QN_MAX],Ly[QN_MAX],Lz[QN_MAX]; Bp.basis1(rx,Lx); Bp.basis1(ry,Ly); Bp.basis1(rz,Lz);
              for(i32 b=0;b<ndof;b++){ i32 bx=b%n,by=(b/n)%n,bz=b/(n*n);
                pcol[(size_t)ndof*fn+b]=nodC[b]; pw[(size_t)ndof*fn+b]=(real)(Lx[bx]*Ly[by]*Lz[bz]); } } }
          d_pcolLev[F]=cpI(pcol.data(),(size_t)ndof*nNodeLev[F]); d_pwLev[F]=cpR(pw.data(),(size_t)ndof*nNodeLev[F]);
          pmgFree.push_back(d_pcolLev[F]); pmgFree.push_back(d_pwLev[F]);
          printf("hmg    : L%d cells %d->%d (%d cut) %d nodes %d ghost sp=(%d,%d,%d)  volpts %zu->%zu (%.1fx)  |Om|=%.7f |Gam|=%.6f\n",
                 L,nEf,nEc,nCutC,nn,nGFc,hSp[L][0],hSp[L][1],hSp[L][2],rawV,cmpV,cmpV?(double)rawV/cmpV:1.0,volC,srfC);
          return true; };

        // ---- build the hierarchy ----
        std::vector<SayeNode> vPa(volPool), sPa(surfPool); std::vector<i32> vOa(volOff), sOa(surfOff);
        std::vector<char> sDa(surfDir.begin(),surfDir.end());
        std::vector<i32> cutIdxA(cutIdx); std::vector<char> cutA(nE); for(i32 e=0;e<nE;e++) cutA[e]=elems[e].cut?1:0;
        nLev=1;
        for(i32 L=1;L<nLmax;L++){
          std::vector<SayeNode> vPn,sPn; std::vector<i32> vOn,sOn,cutIdxN; std::vector<char> sDn,cutN;
          if(!buildH(L,vPa,vOa,sPa,sOa,sDa,cutIdxA,cutA,vPn,vOn,sPn,sOn,sDn,cutIdxN,cutN)) break;
          vPa.swap(vPn); vOa.swap(vOn); sPa.swap(sPn); sOa.swap(sOn); sDa.swap(sDn); cutIdxA.swap(cutIdxN); cutA.swap(cutN);
          nLev=L+1; }
        if(nLev<2){ printf("note   : hmg found no coarsenable level; using jac\n"); pcMode=0; pcName="jac"; }
        else { i32 ncoarse=nNLev[nLev-1]; d_cz=alR(ncoarse); d_cp=alR(ncoarse); d_cAp=alR(ncoarse); d_cr=alR(ncoarse);
          for(void*pp:{(void*)d_cz,(void*)d_cp,(void*)d_cAp,(void*)d_cr}) pmgFree.push_back(pp);
          printf("hmg    : %d-level SEMI-COARSENED h-hierarchy [", nLev);
          for(i32 L=0;L<nLev;L++) printf("%dn%s", nNodeLev[L], L<nLev-1?" -> ":""); printf("]\n"); }
      }
      CutDev S; S.B=Bp; S.nE=nE; S.nCut=nCutQ; S.nGFQ=nGFQ; S.nNode=nNodeQ; S.ndof=ndof; S.ndof3=ndof3; S.mG=mG;
      S.h=h; S.mu=(real)mu; S.lam=(real)lam; S.gammaD=(real)gammaD_; S.cph=(real)cph; S.sph=(real)sph;
      S.eNode=d_eNode; S.nMap=d_nMap; S.nRot=d_nRot; S.intList=d_intList; S.cutElem=d_cutElem;
      S.volP=d_volP; S.surfP=d_surfP; S.volOff=d_volOff; S.surfOff=d_surfOff; S.surfDir=d_surfDir;
      S.gfM=d_gfM; S.gfP=d_gfP; S.gfD=d_gfD; S.Kref=d_Kref; S.Kg[0]=d_Kg0; S.Kg[1]=d_Kg1; S.Kg[2]=d_Kg2;
      S.cyl=cyl?1:0; S.ls=ls; S.eCijk=d_eCijk; S.eCut=d_eCut; S.volJ=d_volJ; S.surfJ=d_surfJ;
      S.sp[0]=S.sp[1]=S.sp[2]=1;
      const i32 GBi=((nE-nCutQ)<65535?(nE-nCutQ):65535), GBg=(nGFQ<65535?nGFQ:65535), GBc=(nCutQ<65535?nCutQ:65535), GBall=(nE<65535?nE:65535);
      auto apply=[&](const real*x,real*y){ cutProlongK<<<GS,BS>>>(S,x,d_xn); cutSetK<<<GS,BS>>>(d_yn,(real)0,nN);
        if(cyl){ cutCylK<<<GBall,128,(size_t)2*ndof3*sizeof(real)>>>(S,d_xn,d_yn); }
        else { if(nE-nCutQ) cutInteriorK<<<GBi,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn);
          if(nCutQ) cutCellK<<<GBc,128,(size_t)2*ndof3*sizeof(real)>>>(S,d_xn,d_yn); }
        if(nGFQ) cutGhostK<<<GBg,256,(size_t)mG*sizeof(real)>>>(S,d_xn,d_yn);
        cutSetK<<<GS,BS>>>(y,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,y); cudaDeviceSynchronize(); };
      auto dot=[&](const real*a,const real*b)->double{ cudaMemset(d_acc,0,sizeof(double)); cutDotK<<<GS,BS>>>(a,b,nR,d_acc); double hv; cudaMemcpy(&hv,d_acc,sizeof(double),cudaMemcpyDeviceToHost); return hv; };
      // node-space operator apply at level L (Slev[L]); dot at arbitrary length
      auto applyN_L=[&](i32 L,const real*xn,real*yn){ i32 nd3=nd3Lev[L],mg=mGLev[L],nnl=nNLev[L]; cutSetK<<<GS,BS>>>(yn,(real)0,nnl);
        i32 nEl=nELev[L],nCl=nCutLev[L],nGl=nGFLev[L];
        i32 gball=nEl<65535?nEl:65535, gbi=(nEl-nCl)<65535?(nEl-nCl):65535, gbc=nCl<65535?nCl:65535, gbg=nGl<65535?nGl:65535;
        if(hcylLev[L]){ cutCylK<<<gball,128,(size_t)2*nd3*sizeof(real)>>>(Slev[L],xn,yn); }
        else { if(nEl-nCl)cutInteriorK<<<gbi,128,(size_t)nd3*sizeof(real)>>>(Slev[L],xn,yn);
          if(nCl)cutCellK<<<gbc,128,(size_t)2*nd3*sizeof(real)>>>(Slev[L],xn,yn); }
        if(nGl)cutGhostK<<<gbg,256,(size_t)mg*sizeof(real)>>>(Slev[L],xn,yn); cudaDeviceSynchronize(); };
      // pitch-tie subspace projection Pi = prolong * (1/mult) * restrict (identity when non-periodic)
      auto proj_L=[&](i32 L,const real*v,real*out){ i32 ndn=3*nDofNodeLev[L];
        cutSetK<<<GS,BS>>>(d_projLev[L],(real)0,ndn); cutRestrictK<<<GS,BS>>>(Slev[L],v,d_projLev[L]);
        cutJacK<<<GS,BS>>>(d_projLev[L],d_projLev[L],d_mult3Lev[L],ndn); cutProlongK<<<GS,BS>>>(Slev[L],d_projLev[L],out); cudaDeviceSynchronize(); };
      // tied operator A_tied = Pi A_n Pi (symmetric; kills untied theta-face modes) -- used by smoother/coarse when periodic
      auto applyTied_L=[&](i32 L,const real*xn,real*yn){ if(per){ proj_L(L,xn,d_ptmpLev[L]); applyN_L(L,d_ptmpLev[L],yn); proj_L(L,yn,yn); } else applyN_L(L,xn,yn); };
      auto ndotL=[&](const real*a,const real*b,i32 nnl)->double{ cudaMemset(d_acc,0,sizeof(double)); cutDotK<<<GS,BS>>>(a,b,nnl,d_acc); double hv; cudaMemcpy(&hv,d_acc,sizeof(double),cudaMemcpyDeviceToHost); return hv; };
      auto residualL=[&](i32 L,const real*x,const real*rhs,real*out){ i32 nnl=nNLev[L]; applyTied_L(L,x,d_tmpLev[L]);
        cutSetK<<<GS,BS>>>(out,(real)0,nnl); cutAxpyK<<<GS,BS>>>(out,rhs,(real)1,nnl); cutAxpyK<<<GS,BS>>>(out,d_tmpLev[L],(real)-1,nnl); cudaDeviceSynchronize(); };
      // Chebyshev direction step: line-implicit M^-1 on the fine level when enabled, else diagonal
      auto chebDir=[&](i32 L,real c1,real c2,i32 nnl){
        if(lineSm&&L==0) lineApply(d_dirLev[L],d_resLev[L],c1,c2);
        else cutChebDirK<<<GS,BS>>>(d_dirLev[L],d_resLev[L],d_diagNLev[L],c1,c2,nnl); };
      auto precApply=[&](i32 L,real*w,const real*v2,i32 nnl){   // M^-1 v (for the power iteration)
        if(lineSm&&L==0) lineApply(w,v2,(real)0,(real)1); else cutJacK<<<GS,BS>>>(w,v2,d_diagNLev[L],nnl); };
      auto chebSmoothL=[&](i32 L,real*x,const real*rhs,i32 deg){ i32 nnl=nNLev[L]; double a=aLev[L],b=bLev[L],theta=(b+a)/2,delta=(b-a)/2,s1=theta/delta,rho=1.0/s1;
        residualL(L,x,rhs,d_resLev[L]); chebDir(L,(real)0,(real)(1.0/theta),nnl); cutAxpyK<<<GS,BS>>>(x,d_dirLev[L],(real)1,nnl); cudaDeviceSynchronize();
        for(i32 k=1;k<deg;k++){ residualL(L,x,rhs,d_resLev[L]); double rn2=1.0/(2*s1-rho);
          chebDir(L,(real)(rho*rn2),(real)(2*rn2/delta),nnl); cutAxpyK<<<GS,BS>>>(x,d_dirLev[L],(real)1,nnl); cudaDeviceSynchronize(); rho=rn2; }
        if(per) proj_L(L,x,x); };   // strip untied component from the smoothed correction
      auto coarseSolve=[&](i32 L,const real*rhs,real*x,i32 maxit,double tol){ i32 nnl=nNLev[L]; cutSetK<<<GS,BS>>>(x,(real)0,nnl);
        cudaMemcpy(d_cr,rhs,(size_t)nnl*sizeof(real),cudaMemcpyDeviceToDevice); if(per) proj_L(L,d_cr,d_cr);
        cutJacK<<<GS,BS>>>(d_cz,d_cr,d_diagNLev[L],nnl); cudaMemcpy(d_cp,d_cz,(size_t)nnl*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
        double bn=sqrt(ndotL(d_cr,d_cr,nnl)); if(bn==0)bn=1; double rz=ndotL(d_cr,d_cz,nnl);
        for(i32 it=0;it<maxit;it++){ applyTied_L(L,d_cp,d_cAp); double pAp=ndotL(d_cp,d_cAp,nnl); if(!(pAp>0))break;
          double al=rz/pAp; cutAxpyK<<<GS,BS>>>(x,d_cp,(real)al,nnl); cutAxpyK<<<GS,BS>>>(d_cr,d_cAp,(real)-al,nnl); cudaDeviceSynchronize();
          if(sqrt(ndotL(d_cr,d_cr,nnl))<=tol*bn)break;
          cutJacK<<<GS,BS>>>(d_cz,d_cr,d_diagNLev[L],nnl); cudaDeviceSynchronize(); double rz2=ndotL(d_cr,d_cz,nnl),be=rz2/rz; rz=rz2;
          cutAxpyK<<<GS,BS>>>(d_cp,d_cp,(real)(be-1),nnl); cutAxpyK<<<GS,BS>>>(d_cp,d_cz,(real)1,nnl); cudaDeviceSynchronize(); }
        if(per) proj_L(L,x,x); };
      auto restrictP=[&](i32 L){ cutSetK<<<GS,BS>>>(d_rLev[L+1],(real)0,nNLev[L+1]); cutPRestrictWK<<<GS,BS>>>(d_pcolLev[L],d_pwLev[L],wLev[L],d_resLev[L],d_rLev[L+1],nNodeLev[L]); cudaDeviceSynchronize(); if(per) proj_L(L+1,d_rLev[L+1],d_rLev[L+1]); };
      auto prolongAdd=[&](i32 L,real*xfine){ cutPProlongWK<<<GS,BS>>>(d_pcolLev[L],d_pwLev[L],wLev[L],d_xLev[L+1],xfine,nNodeLev[L]); cudaDeviceSynchronize(); if(per) proj_L(L,xfine,xfine); };
      auto vcycle=[&](const real*rn,real*zn){ const i32 deg=cDeg;   // multi-level V-cycle (descend, coarse solve, ascend); Pi-tied when periodic
        cutSetK<<<GS,BS>>>(zn,(real)0,nNLev[0]); chebSmoothL(0,zn,rn,deg); residualL(0,zn,rn,d_resLev[0]); restrictP(0);
        for(i32 L=1; L<nLev-1; L++){ cutSetK<<<GS,BS>>>(d_xLev[L],(real)0,nNLev[L]); chebSmoothL(L,d_xLev[L],d_rLev[L],deg); residualL(L,d_xLev[L],d_rLev[L],d_resLev[L]); restrictP(L); }
        coarseSolve(nLev-1,d_rLev[nLev-1],d_xLev[nLev-1],cMaxit,cTol);
        for(i32 L=nLev-2; L>=1; L--){ prolongAdd(L,d_xLev[L]); chebSmoothL(L,d_xLev[L],d_rLev[L],deg); }
        prolongAdd(0,zn); chebSmoothL(0,zn,rn,deg); };
      // ---- IC(0) state (filled by the CSR block below; precond captures by reference) ----
      bool icReady=false; double *d_icV=nullptr, *d_icX=nullptr, *d_icY=nullptr;
      i32 *d_icRp=nullptr,*d_icCi=nullptr; void *d_icW1=nullptr,*d_icW2=nullptr;
      real *d_l1=nullptr;   // l1 row-sum diagonal (CUT_PC=l1jac)
      i32 *d_icPerm=nullptr;   // multicolor permutation: newIdx -> oldIdx (CUT_ICMC)
      double *d_ssorD=nullptr; // SSOR middle factor D/omega (non-null => CUT_PC=ssor)
      cusparseHandle_t icH=nullptr; cusparseSpMatDescr_t icL=nullptr;
      cusparseDnVecDescr_t icVb=nullptr,icVy=nullptr,icVx=nullptr;
      cusparseSpSVDescr_t icSvL=nullptr,icSvU=nullptr;
      auto precond=[&](real*z,const real*r){
        if((pcMode==5||pcMode==9)&&icReady){   // z = L^-T [D/w] L^-1 r  (two sparse triangular solves;
                                               // the middle diagonal is present for SSOR only)
          const double one=1.0;   // icVb,icVx wrap d_icX; icVy wraps d_icY
          if(d_icPerm) cutD2PK<<<GS,BS>>>(d_icX,r,d_icPerm,nR); else cutD2K<<<GS,BS>>>(d_icX,r,nR);
          cusparseSpSV_solve(icH,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,icL,icVb,icVy,CUDA_R_64F,
                             CUSPARSE_SPSV_ALG_DEFAULT,icSvL);
          if(d_ssorD) cutDMulK<<<GS,BS>>>(d_icY,d_ssorD,nR);
          cusparseSpSV_solve(icH,CUSPARSE_OPERATION_TRANSPOSE,&one,icL,icVy,icVx,CUDA_R_64F,
                             CUSPARSE_SPSV_ALG_DEFAULT,icSvU);
          if(d_icPerm) cutR2PK<<<GS,BS>>>(z,d_icX,d_icPerm,nR); else cutR2K<<<GS,BS>>>(z,d_icX,nR);
          return; }
        if(pcMode==6&&d_l1){ cutJacK<<<GS,BS>>>(z,r,d_l1,nR); return; }
        if(pcMode==8){ lineApply(z,r,(real)0,(real)1); return; }   // EXACT line blocks, already in dof space
        if(pcMode==7){   // standalone span-wise line-block Jacobi: z = R M_line^-1 P r
          cutProlongK<<<GS,BS>>>(S,r,d_xn); lineApply(d_yn,d_xn,(real)0,(real)1);
          cutSetK<<<GS,BS>>>(z,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,z); return; }
        if(pcMode==1){ cutBJacK<<<GS,BS>>>(z,r,d_Binv,nDofNode); return; }
        if(pcMode==2){ cutProlongK<<<GS,BS>>>(S,r,d_xn); cutSetK<<<GS,BS>>>(d_yn,(real)0,nN);
          if(nE-nCutQ) cutSchwarzK<<<GBi,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn,d_intList,nE-nCutQ,d_intInv,0);
          if(nCutQ) cutSchwarzK<<<GBc,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn,d_cutElem,nCutQ,d_cutInv,1);
          cutSetK<<<GS,BS>>>(z,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,z); return; }
        if(pcMode==3||pcMode==4){ cutProlongK<<<GS,BS>>>(S,r,d_xn); vcycle(d_xn,d_yn); cutSetK<<<GS,BS>>>(z,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,z); return; }
        cutJacK<<<GS,BS>>>(z,r,d_diag,nR); };
      // ============ LINE-IMPLICIT smoother on the FINE level (CUT_SMOOTH=line) ============
      //  Anisotropy needs EITHER line-smoother+full-coarsening OR point-smoother+semi-coarsening.
      //  A span-wise (radial, I-varying at fixed J,K) implicit solve resolves the 1-D cantilever
      //  bending stiffness exactly -- the soft mode p-coarsening at fixed h cannot reach -- and it
      //  builds NO coarse geometry, so the aspect-ratio locking that killed deep hmg cannot occur.
      if ((lineSm && (pcMode==3||pcMode==4)) || pcMode==7) {
        long tl=qpNowUs();
        //  CUT_LINEDIR picks which grid direction the lines run along: 0 = I (radial/span, default),
        //  1 = J (theta = the blade's THIN direction), 2 = K (z = chord).  The blade is TWISTED, so the
        //  beam axis is not exactly a grid line and the best direction is an empirical question.
        i32 lineDir=0; { const char*e=getenv("CUT_LINEDIR"); if(e) lineDir=atoi(e); }
        const i32 *nA,*nB,*nC;   // nC = the direction the line RUNS along; nA,nB = the key
        if(lineDir==1){ nA=nI.data(); nB=nK.data(); nC=nJ.data(); }
        else if(lineDir==2){ nA=nI.data(); nB=nJ.data(); nC=nK.data(); }
        else { nA=nJ.data(); nB=nK.data(); nC=nI.data(); }
        const i32 ghostD=lineDir;   // only ghost faces normal to the line direction couple ALONG a line
        std::unordered_map<u64,std::vector<i32>> lm;
        for(i32 nd=0;nd<nNodeQ;nd++) lm[((u64)nA[nd]<<21)|(u64)nB[nd]].push_back(nd);
        std::vector<std::vector<i32>> lines; lines.reserve(lm.size());
        for(auto&kv:lm){ std::vector<i32> v=kv.second; std::sort(v.begin(),v.end(),[&](i32 a,i32 b){return nC[a]<nC[b];}); lines.push_back(std::move(v)); }
        nLine=(i32)lines.size();
        std::vector<i32> nLid(nNodeQ,-1), nPos(nNodeQ,-1);
        for(i32 L2=0;L2<nLine;L2++) for(size_t t=0;t<lines[L2].size();t++){ nLid[lines[L2][t]]=L2; nPos[lines[L2][t]]=(i32)t; }
        // dense element matrices (bulk + Nitsche), probed column by column
        std::vector<real> Kel((size_t)nE*ndof3*ndof3);
        #pragma omp parallel for schedule(dynamic,4)
        for(i32 e=0;e<nE;e++){
          std::vector<double> u(ndof3),y(ndof3); real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
          for(i32 cc=0;cc<ndof3;cc++){
            if(cyl){ for(i32 i=0;i<ndof3;i++)u[i]=(i==cc)?1.0:0.0; applyElemCyl(e,u.data(),y.data(),true); }
            else { for(i32 i=0;i<ndof3;i++)ul[i]=(i==cc)?(real)1:(real)0;
              if(!elems[e].cut) qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
              else { i32 c=cutIdx[e],v0=volOff[c],nv=volOff[c+1]-v0,s0=surfOff[c],ns=surfOff[c+1]-s0;
                qpElemCoreSaye(Bp,(real)mu,(real)lam,h,&volPool[v0],nv,ul,yl);
                for(i32 q=0;q<ns;q++){ if(!surfDir[s0+q])continue; real xr[3]={surfPool[s0+q].x[0],surfPool[s0+q].x[1],surfPool[s0+q].x[2]};
                  real vb[QN_MAX*QN_MAX*QN_MAX]; Bp.allVal(xr,vb); real hw=surfPool[s0+q].w*h;
                  real uval[3]={0,0,0}; for(i32 b=0;b<ndof;b++)for(i32 l=0;l<3;l++)uval[l]+=ul[3*b+l]*vb[b];
                  for(i32 a=0;a<ndof;a++)for(i32 l=0;l<3;l++) yl[3*a+l]+=hw*(real)gammaD_*uval[l]*vb[a]; } }
              for(i32 i=0;i<ndof3;i++)y[i]=yl[i]; }
            for(i32 r=0;r<ndof3;r++) Kel[(size_t)e*ndof3*ndof3+(size_t)r*ndof3+cc]=(real)y[r]; } }
        // assemble each line's node-space block from the element matrices (+ r-direction ghost faces,
        // which are the only ghost couplings that lie ALONG a line; theta/z ghost stays explicit)
        std::vector<std::vector<double>> blk(nLine);
        for(i32 L2=0;L2<nLine;L2++){ i32 m3=3*(i32)lines[L2].size(); blk[L2].assign((size_t)m3*m3,0.0); lineMax3=std::max(lineMax3,m3); }
        for(i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof]; const real*K=&Kel[(size_t)e*ndof3*ndof3];
          for(i32 a=0;a<ndof;a++){ i32 na=nod[a], La=nLid[na]; i32 m3=3*(i32)lines[La].size();
            for(i32 b=0;b<ndof;b++){ i32 nb2=nod[b]; if(nLid[nb2]!=La) continue;
              i32 ia=3*nPos[na], ib=3*nPos[nb2];
              for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++)
                blk[La][(size_t)(ia+i)*m3+(ib+j)] += (double)K[(size_t)(3*a+i)*ndof3+(3*b+j)]; } } }
        for(i32 f=0;f<nGFQ;f++){ if(gf[f].d!=ghostD) continue;
          const i32*nodM=&eNodeQ[(size_t)gf[f].eM*ndof], *nodP=&eNodeQ[(size_t)gf[f].eP*ndof]; const double*K=Kghost[ghostD].data();
          auto dec=[&](i32 r,i32&nd2,i32&cp){ if(r<ndof3){ nd2=nodM[r/3]; cp=r%3; } else { i32 rr=r-ndof3; nd2=nodP[rr/3]; cp=rr%3; } };
          for(i32 r=0;r<mG;r++){ i32 nr,cr; dec(r,nr,cr); i32 La=nLid[nr]; i32 m3=3*(i32)lines[La].size();
            for(i32 c2=0;c2<mG;c2++){ i32 nc,cc2; dec(c2,nc,cc2); if(nLid[nc]!=La) continue;
              blk[La][(size_t)(3*nPos[nr]+cr)*m3+(3*nPos[nc]+cc2)] += K[(size_t)r*mG+c2]; } } }
        // Put the TRUE full node diagonal (which includes the theta/z ghost coupling that is OFF-line)
        // onto each block's diagonal.  Without this, lines that only graze the domain lose their ghost
        // stabilization, their blocks go near-singular, and M^-1 blows up (lambda_max ~ 5e3).
        for(i32 L2=0;L2<nLine;L2++){ i32 m3=3*(i32)lines[L2].size();
          for(i32 t=0;t<(i32)lines[L2].size();t++){ i32 nd=lines[L2][t];
            for(i32 c3=0;c3<3;c3++) blk[L2][(size_t)(3*t+c3)*m3+(3*t+c3)]=diagNode[3*nd+c3]; } }
        // invert (dense Cholesky per line; tiny -- 3m x 3m with m ~ p*nr+1)
        std::vector<size_t> mOffH(nLine+1,0);
        for(i32 L2=0;L2<nLine;L2++){ i32 m3=3*(i32)lines[L2].size(); mOffH[L2+1]=mOffH[L2]+(size_t)m3*m3; }
        std::vector<real> MinvH(mOffH[nLine]); i32 nbad=0;
        #pragma omp parallel for schedule(dynamic,4) reduction(+:nbad)
        for(i32 L2=0;L2<nLine;L2++){ i32 m3=3*(i32)lines[L2].size(); std::vector<double> A(blk[L2]);
          for(i32 i=0;i<m3;i++) if(A[(size_t)i*m3+i]<=0) A[(size_t)i*m3+i]=1.0;
          if(!invertSPD(A.data(),m3)){ nbad++; std::fill(A.begin(),A.end(),0.0);
            for(i32 i=0;i<m3;i++) A[(size_t)i*m3+i]=1.0/std::max(1e-30,blk[L2][(size_t)i*m3+i]); }   // fall back to diagonal
          for(size_t i=0;i<(size_t)m3*m3;i++) MinvH[mOffH[L2]+i]=(real)A[i]; }
        std::vector<i32> lOffH(nLine+1,0), lNodeH; lNodeH.reserve(nNodeQ);
        for(i32 L2=0;L2<nLine;L2++){ for(i32 nd:lines[L2]) lNodeH.push_back(nd); lOffH[L2+1]=(i32)lNodeH.size(); }
        d_Minv=cpR(MinvH.data(),MinvH.size());
        cudaMallocManaged(&d_mOff,(size_t)(nLine+1)*sizeof(size_t)); memcpy(d_mOff,mOffH.data(),(size_t)(nLine+1)*sizeof(size_t));
        d_lOff=cpI(lOffH.data(),nLine+1); d_lNode=cpI(lNodeH.data(),lNodeH.size());
        for(void*pp:{(void*)d_Minv,(void*)d_mOff,(void*)d_lOff,(void*)d_lNode}) pmgFree.push_back(pp);
        { double avg=(double)nNodeQ/std::max(1,nLine); i32 nLong=0;
          for(i32 L2=0;L2<nLine;L2++) if((i32)lines[L2].size()>=lineMax3/3/2) nLong++;
          printf("line   : dir %c  %d lines (max %d nodes, avg %.1f, %d >=half-max), %.1f MB inv, %d non-SPD%s  in %.2fs\n",
                 "IJK"[lineDir&3],nLine,lineMax3/3,avg,nLong,
                 (double)(MinvH.size()*sizeof(real))/1e6,nbad,nbad?" (diag fallback)":"",(qpNowUs()-tl)*1e-6); }
      }
      if (pcMode==3||pcMode==4) {   // per-level Chebyshev interval via power iteration on M^-1 A (smoothed levels 0..nLev-2)
        for(i32 L=0; L<nLev-1; L++){ i32 nnl=nNLev[L]; real*v=d_resLev[L],*Av=d_tmpLev[L],*w=d_dirLev[L];
          cutSetK<<<GS,BS>>>(v,(real)1,nnl);
          for(i32 it=0;it<20;it++){ applyTied_L(L,v,Av); precApply(L,w,Av,nnl); cudaDeviceSynchronize();
            double nrm=sqrt(ndotL(w,w,nnl)); if(nrm<=0)break; cutSetK<<<GS,BS>>>(v,(real)0,nnl); cutAxpyK<<<GS,BS>>>(v,w,(real)(1.0/nrm),nnl); cudaDeviceSynchronize(); }
          applyTied_L(L,v,Av); precApply(L,w,Av,nnl); cudaDeviceSynchronize();
          double lam=ndotL(v,w,nnl)/ndotL(v,v,nnl); double clo=30.0; { const char*e=getenv("CUT_CHEBLO"); if(e) clo=atof(e); }
          bLev[L]=1.1*lam; aLev[L]=bLev[L]/clo; }
        printf("pmg    : cheb intervals"); for(i32 L=0;L<nLev-1;L++) printf(" L%d[%.2f,%.2f]",L,aLev[L],bLev[L]); printf("\n");
      }
      // ================= assembled CSR + SPARSE DIRECT solve (CUT_DIRECT=1) =================
      //  The matrix-free design was chosen for p3 scalability, but the blade is only ~7.7k unknowns --
      //  small enough to factor exactly.  Assembling also unlocks AMG.  Dof-space assembly folds the
      //  periodic pitch tie directly: K_dof(da,db) += R_a^T K_node(a,b) R_b  (R = pitch rotation on
      //  slave nodes, identity elsewhere), matching gather3/scatter3.
      bool directOK=false;
      const bool wantDirect=getenv("CUT_DIRECT")!=nullptr;
      i32 nModal=0; { const char*e=getenv("CUT_MODAL"); if(e) nModal=atoi(e); }
      i32 *d_mRp=nullptr,*d_mCi=nullptr; real *d_mVa=nullptr;   // mass CSR (modal only)
      if (nModal>0 && pcMode!=5) { printf("note   : CUT_MODAL needs the assembled CSR; forcing CUT_PC=ic\n");
                                   pcMode=5; pcName="ic"; }
      if (wantDirect || pcMode==5 || pcMode==6 || pcMode==8 || pcMode==9) {
        long ta=qpNowUs();
        double Rm[3][3]={{cph,-sph,0},{sph,cph,0},{0,0,1}};
        std::vector<std::unordered_map<i32,double>> row((size_t)nDofQ);
        auto addBlk=[&](i32 na,i32 nb,const double Kab[3][3]){
          i32 da=realIdx[na], db=realIdx[nb];
          double T[3][3];   // T = R_a^T K R_b
          for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++){ double s=0;
            for(i32 k2=0;k2<3;k2++){ double kb=0; for(i32 l2=0;l2<3;l2++) kb+=Kab[k2][l2]*(rotFlag[nb]?Rm[l2][j]:(l2==j?1.0:0.0));
              s+=(rotFlag[na]?Rm[k2][i]:(k2==i?1.0:0.0))*kb; } T[i][j]=s; }
          for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) if(T[i][j]!=0.0) row[3*da+i][3*db+j]+=T[i][j]; };
        // element (bulk + Nitsche) blocks, probed column by column
        #pragma omp parallel for schedule(dynamic,4) ordered
        for(i32 e=0;e<nE;e++){
          std::vector<double> Ke((size_t)ndof3*ndof3); std::vector<double> u(ndof3),y(ndof3);
          real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
          for(i32 cc=0;cc<ndof3;cc++){
            if(cyl){ for(i32 i=0;i<ndof3;i++)u[i]=(i==cc)?1.0:0.0; applyElemCyl(e,u.data(),y.data(),true); }
            else { for(i32 i=0;i<ndof3;i++)ul[i]=(i==cc)?(real)1:(real)0;
              if(!elems[e].cut) qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
              else { i32 c=cutIdx[e],v0=volOff[c],nv=volOff[c+1]-v0,s0=surfOff[c],ns=surfOff[c+1]-s0;
                qpElemCoreSaye(Bp,(real)mu,(real)lam,h,&volPool[v0],nv,ul,yl);
                // Nitsche: must mirror cutCellK EXACTLY -- consistency (t1) + adjoint (t2) + penalty (t3).
                // An earlier version copied the *diagonal* builder, which keeps only t3; that assembles a
                // different operator than the matrix-free apply and silently poisons every CSR consumer.
                for(i32 q=0;q<ns;q++){ if(!surfDir[s0+q])continue;
                  real xr[3]={surfPool[s0+q].x[0],surfPool[s0+q].x[1],surfPool[s0+q].x[2]};
                  real nn[3]={surfPool[s0+q].n[0],surfPool[s0+q].n[1],surfPool[s0+q].n[2]};
                  real vb[CUT_QN3], gb[3*CUT_QN3]; Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
                  real hw=surfPool[s0+q].w*h;
                  real uval[3]={0,0,0}, gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
                  for(i32 b=0;b<ndof;b++)for(i32 i2=0;i2<3;i2++){ real ua=ul[3*b+i2]; uval[i2]+=ua*vb[b];
                    gU[i2][0]+=ua*gb[3*b]; gU[i2][1]+=ua*gb[3*b+1]; gU[i2][2]+=ua*gb[3*b+2]; }
                  real eps[3][3]; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=(real)0.5*(gU[i2][j2]+gU[j2][i2]);
                  real tr=eps[0][0]+eps[1][1]+eps[2][2], sig[3][3];
                  for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*(real)mu*eps[i2][j2]+(i2==j2?(real)lam*tr:(real)0);
                  real tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
                  real un=uval[0]*nn[0]+uval[1]*nn[1]+uval[2]*nn[2];
                  for(i32 a=0;a<ndof;a++){ real gan=gb[3*a]*nn[0]+gb[3*a+1]*nn[1]+gb[3*a+2]*nn[2];
                    real ugb=uval[0]*gb[3*a]+uval[1]*gb[3*a+1]+uval[2]*gb[3*a+2];
                    for(i32 l=0;l<3;l++){ real t1=-tu[l]*vb[a],
                      t2=-((real)mu*(uval[l]*gan+ugb*nn[l])+(real)lam*gb[3*a+l]*un), t3=(real)gammaD_*uval[l]*vb[a];
                      yl[3*a+l]+=hw*(t1+t2+t3); } } } }
              for(i32 i=0;i<ndof3;i++)y[i]=yl[i]; }
            for(i32 r=0;r<ndof3;r++) Ke[(size_t)r*ndof3+cc]=y[r]; }
          #pragma omp ordered
          { const i32*nod=&eNodeQ[(size_t)e*ndof];
            for(i32 a=0;a<ndof;a++)for(i32 b=0;b<ndof;b++){ double Kab[3][3];
              for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) Kab[i][j]=Ke[(size_t)(3*a+i)*ndof3+(3*b+j)];
              addBlk(nod[a],nod[b],Kab); } } }
        // ghost-penalty face blocks (mG x mG over the [minus | plus] element node pair)
        for(i32 f=0;f<nGFQ;f++){ const i32*nodM=&eNodeQ[(size_t)gf[f].eM*ndof],*nodP=&eNodeQ[(size_t)gf[f].eP*ndof];
          const double*K=Kghost[gf[f].d].data();
          auto nodeOf=[&](i32 r)->i32{ return (r<ndof3)? nodM[r/3] : nodP[(r-ndof3)/3]; };
          for(i32 a=0;a<2*ndof;a++)for(i32 b=0;b<2*ndof;b++){ double Kab[3][3];
            for(i32 i=0;i<3;i++)for(i32 j=0;j<3;j++) Kab[i][j]=K[(size_t)(3*a+i)*mG+(3*b+j)];
            addBlk(nodeOf(3*a),nodeOf(3*b),Kab); } }
        // -> CSR
        std::vector<i32> rp(nDofQ+1,0), ci2; std::vector<double> va;
        for(i32 r=0;r<nDofQ;r++) rp[r+1]=rp[r]+(i32)row[r].size();
        ci2.resize(rp[nDofQ]); va.resize(rp[nDofQ]);
        for(i32 r=0;r<nDofQ;r++){ std::vector<std::pair<i32,double>> e2(row[r].begin(),row[r].end());
          std::sort(e2.begin(),e2.end()); for(size_t t=0;t<e2.size();t++){ ci2[rp[r]+t]=e2[t].first; va[rp[r]+t]=e2[t].second; } }
        i32 nnz=rp[nDofQ];
        { std::vector<std::unordered_map<i32,double>>().swap(row); }   // free the maps before the factor allocates
        printf("csr    : assembled %d x %d, %d nnz (%.1f/row, %.1f MB) in %.2fs\n",
               nDofQ,nDofQ,nnz,(double)nnz/nDofQ,(double)(nnz*(sizeof(double)+sizeof(i32)))/1e6,(qpNowUs()-ta)*1e-6);
        // ---------------- MASS matrix for modal analysis (CUT_MODAL=N) ----------------
        //  M_ab = rho * integral_Omega phi_a phi_b, SAME quadrature as the stiffness (tensor GLL on
        //  uncut cells -> lumped there; Saye on cut cells -> consistent).  Formed directly (rank-1 per
        //  point, no column probing) and pushed through the SAME addBlk, so the periodic pitch tie is
        //  folded identically to K -- required, or K and M live in different spaces.
        //  NOTE the NNLS prune preserves moments to degree 2p and phi_a*phi_b IS degree 2p, so the
        //  pruned rule integrates the mass matrix EXACTLY too.
        if (nModal>0) {
          long tm=qpNowUs();
          std::vector<std::unordered_map<i32,double>> mrow((size_t)nDofQ);
          std::vector<std::unordered_map<i32,double>>* saveRow=nullptr; (void)saveRow;
          auto addMass=[&](i32 na,i32 nb,double m){   // mass block is m*I3; R^T (m I) R = m I, so the
            i32 da=realIdx[na], db=realIdx[nb];       // pitch rotation is a no-op -- but keep it explicit
            for(i32 i=0;i<3;i++) mrow[3*da+i][3*db+i]+=m; };
          std::vector<double> Ml((size_t)ndof*ndof);
          for (i32 e=0;e<nE;e++){
            std::fill(Ml.begin(),Ml.end(),0.0);
            std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
            if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
              for (i32 k2=0;k2<n;k2++)for(i32 j2=0;j2<n;j2++)for(i32 i2=0;i2<n;i2++){
                tens[qi].x[0]=Bp.qx[i2];tens[qi].x[1]=Bp.qx[j2];tens[qi].x[2]=Bp.qx[k2];
                tens[qi].w=Bp.qw[i2]*Bp.qw[j2]*Bp.qw[k2]; qi++; } vn=tens.data(); nv=ndof;
            } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
            real vb[QN_MAX*QN_MAX*QN_MAX];
            for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
              double detJ=h*h*h, Jinv[3][3];
              if (cyl) metric(elems[e],xr,Jinv,detJ);
              Bp.allVal(xr,vb);
              double w=(double)prob.rho*fabs(detJ)*(double)vn[q].w;
              for (i32 a=0;a<ndof;a++){ double wa=w*(double)vb[a];
                for (i32 b=0;b<ndof;b++) Ml[(size_t)a*ndof+b]+=wa*(double)vb[b]; } }
            const i32*nod=&eNodeQ[(size_t)e*ndof];
            for (i32 a=0;a<ndof;a++) for(i32 b=0;b<ndof;b++)
              if (Ml[(size_t)a*ndof+b]!=0.0) addMass(nod[a],nod[b],Ml[(size_t)a*ndof+b]); }
          std::vector<i32> mrp(nDofQ+1,0), mci; std::vector<real> mva;
          for(i32 r2=0;r2<nDofQ;r2++) mrp[r2+1]=mrp[r2]+(i32)mrow[r2].size();
          mci.resize(mrp[nDofQ]); mva.resize(mrp[nDofQ]);
          double mtot=0;
          for(i32 r2=0;r2<nDofQ;r2++){ std::vector<std::pair<i32,double>> e2(mrow[r2].begin(),mrow[r2].end());
            std::sort(e2.begin(),e2.end());
            for(size_t t=0;t<e2.size();t++){ mci[mrp[r2]+t]=e2[t].first; mva[mrp[r2]+t]=(real)e2[t].second;
              if(r2%3==0) mtot+=e2[t].second; } }
          d_mRp=cpI(mrp.data(),nDofQ+1); d_mCi=cpI(mci.data(),mci.size()); d_mVa=cpR(mva.data(),mva.size());
          // total mass = sum of ALL M entries / 3 ; must equal rho*|Omega| -- the sanity check that
          // the cut quadrature and the tie folding did not lose or double-count material.
          printf("modal  : mass CSR %d nnz, total mass %.8g (should be rho*|Omega_h|) in %.2fs\n",
                 mrp[nDofQ],mtot,(qpNowUs()-tm)*1e-6);
        }
        // ------------- EXACT line-block Jacobi from the CSR (CUT_PC=linex) -------------
        //  The probed builder (CUT_PC=line) drops ghost couplings whose face normal is not the line
        //  direction, then patches the damage by overwriting the block diagonal -- so its M is not a
        //  submatrix of A at all.  Here the block IS A restricted to the line: every ghost direction
        //  included, tie already folded (the CSR is dof-space), no diagonal hack, no prolong/restrict.
        if (pcMode==8) {
          i32 lineDir=0; { const char*e=getenv("CUT_LINEDIR"); if(e) lineDir=atoi(e); }
          std::vector<i32> dI(nDofNode,-1),dJ(nDofNode,-1),dK(nDofNode,-1);
          for(i32 nd=0;nd<nNodeQ;nd++){ i32 da=realIdx[nd]; if(da>=0&&dI[da]<0){ dI[da]=nI[nd]; dJ[da]=nJ[nd]; dK[da]=nK[nd]; } }
          const i32 *nA,*nB,*nC;
          if(lineDir==1){ nA=dI.data(); nB=dK.data(); nC=dJ.data(); }
          else if(lineDir==2){ nA=dI.data(); nB=dJ.data(); nC=dK.data(); }
          else { nA=dJ.data(); nB=dK.data(); nC=dI.data(); }
          std::unordered_map<u64,std::vector<i32>> lm;
          for(i32 da=0;da<nDofNode;da++) lm[((u64)(u32)nA[da]<<21)|(u64)(u32)nB[da]].push_back(da);
          std::vector<std::vector<i32>> lines; lines.reserve(lm.size());
          for(auto&kv:lm){ std::vector<i32> v=kv.second; std::sort(v.begin(),v.end(),[&](i32 a,i32 b){return nC[a]<nC[b];}); lines.push_back(std::move(v)); }
          nLine=(i32)lines.size();
          std::vector<i32> pos(nDofQ,-1), lid(nDofNode,-1);
          for(i32 L2=0;L2<nLine;L2++){ for(size_t t=0;t<lines[L2].size();t++){ lid[lines[L2][t]]=L2;
            for(i32 c3=0;c3<3;c3++) pos[3*lines[L2][t]+c3]=3*(i32)t+c3; } }
          std::vector<size_t> mOffH(nLine+1,0);
          for(i32 L2=0;L2<nLine;L2++){ i32 m3=3*(i32)lines[L2].size(); lineMax3=std::max(lineMax3,m3);
            mOffH[L2+1]=mOffH[L2]+(size_t)m3*m3; }
          std::vector<real> MinvH(mOffH[nLine]); i32 nbad=0;
          #pragma omp parallel for schedule(dynamic,4) reduction(+:nbad)
          for(i32 L2=0;L2<nLine;L2++){ i32 m=(i32)lines[L2].size(), m3=3*m;
            std::vector<double> A((size_t)m3*m3,0.0);
            for(i32 t=0;t<m;t++) for(i32 c3=0;c3<3;c3++){ i32 gr=3*lines[L2][t]+c3, lr=3*t+c3;
              for(i32 k2=rp[gr];k2<rp[gr+1];k2++){ i32 gc=ci2[k2];
                if(lid[gc/3]!=L2) continue;   // keep ONLY couplings inside this line
                A[(size_t)lr*m3+pos[gc]]+=va[k2]; } }
            if(!invertSPD(A.data(),m3)){ nbad++; std::fill(A.begin(),A.end(),0.0);
              for(i32 t=0;t<m;t++) for(i32 c3=0;c3<3;c3++){ i32 gr=3*lines[L2][t]+c3, lr=3*t+c3; double dg=1;
                for(i32 k2=rp[gr];k2<rp[gr+1];k2++) if(ci2[k2]==gr) dg=va[k2];
                A[(size_t)lr*m3+lr]=1.0/std::max(1e-30,dg); } }
            for(size_t i=0;i<(size_t)m3*m3;i++) MinvH[mOffH[L2]+i]=(real)A[i]; }
          std::vector<i32> lOffH(nLine+1,0), lNodeH; lNodeH.reserve(nDofNode);
          for(i32 L2=0;L2<nLine;L2++){ for(i32 da:lines[L2]) lNodeH.push_back(da); lOffH[L2+1]=(i32)lNodeH.size(); }
          d_Minv=cpR(MinvH.data(),MinvH.size());
          cudaMallocManaged(&d_mOff,(size_t)(nLine+1)*sizeof(size_t)); memcpy(d_mOff,mOffH.data(),(size_t)(nLine+1)*sizeof(size_t));
          d_lOff=cpI(lOffH.data(),nLine+1); d_lNode=cpI(lNodeH.data(),lNodeH.size());
          for(void*pp:{(void*)d_Minv,(void*)d_mOff,(void*)d_lOff,(void*)d_lNode}) pmgFree.push_back(pp);
          printf("linex  : dir %c  %d exact line blocks (max %d dofnodes, avg %.1f), %.1f MB inv, %d non-SPD\n",
                 "IJK"[lineDir&3],nLine,lineMax3/3,(double)nDofNode/std::max(1,nLine),
                 (double)(MinvH.size()*sizeof(real))/1e6,nbad);
        }
        // ---------------- l1 row-sum diagonal (CUT_PC=l1jac) ----------------
        if (pcMode==6) {
          std::vector<real> l1H(nDofQ); double mn=1e300,mx=0;
          for(i32 r2=0;r2<nDofQ;r2++){ double s=0,dg=1;
            for(i32 k2=rp[r2];k2<rp[r2+1];k2++){ s+=fabs(va[k2]); if(ci2[k2]==r2) dg=va[k2]; }
            l1H[r2]=(real)s; double q=s/std::max(1e-300,dg); mn=std::min(mn,q); mx=std::max(mx,q); }
          d_l1=cpR(l1H.data(),l1H.size());
          // ratio SPREAD is the whole story: CG is invariant to a constant rescale of M, so l1jac can
          // only differ from jac to the extent |row|/|diag| varies across rows.
          printf("l1jac  : |row|/|diag| in [%.3f, %.3f] (spread %.1fx)\n",mn,mx,mx/std::max(1e-300,mn));
        }
        // ---------------- IC(0) incomplete Cholesky factor (CUT_PC=ic) ----------------
        //  A ~ L L^T with the sparsity of A (no fill).  Breakdown (a non-positive pivot) is the usual
        //  failure mode on ill-conditioned elasticity; CUT_ICSHIFT adds a Manteuffel diagonal shift
        //  A + alpha*diag(A) to restore positivity -- the factor stays a valid SPD preconditioner
        //  for the UNSHIFTED A, it just gets less accurate as alpha grows.
        //  SSOR (pcMode 9) shares this ENTIRE path -- same CSR, same optional multicolor reorder, same
        //  two SpSV -- but skips csric02 and feeds A itself (diagonal scaled by 1/omega) to the solves,
        //  with D/omega applied between them.  SPD for any 0<omega<2, so unlike IC it CANNOT break down.
        if (pcMode==5 || pcMode==9) {
          double icShift=0.0; { const char*e=getenv("CUT_ICSHIFT"); if(e) icShift=atof(e); }
          double omega=1.0; { const char*e=getenv("CUT_SSOROMEGA"); if(e) omega=atof(e); }
          long tf=qpNowUs();
          // ---- optional MULTICOLOR reordering (CUT_ICMC=1) ----
          //  SpSV is level-scheduled, and in the natural (lexicographic) order the chain is ~3*(nx+ny+nz)
          //  deep -- launch-latency bound, not flop bound.  Reordering by color makes rows of one color
          //  mutually uncoupled, so the level count collapses to the NUMBER OF COLORS.  Costs iterations
          //  (IC(0) quality depends on the ordering) and buys apply time; nColor is the whole story.
          std::vector<i32> Rp,Ci,perm; std::vector<double> Va;
          const i32 *rpU=rp.data(),*ciU=ci2.data(); const double*vaU=va.data();
          if (getenv("CUT_ICMC")) {
            std::vector<i32> color(nDofQ,-1),stamp(nDofQ+2,-1); i32 nColor=0;
            for(i32 r2=0;r2<nDofQ;r2++){
              for(i32 k2=rp[r2];k2<rp[r2+1];k2++){ i32 c=ci2[k2]; if(c!=r2&&color[c]>=0) stamp[color[c]]=r2; }
              i32 c=0; while(c<nDofQ&&stamp[c]==r2) c++;
              color[r2]=c; if(c>=nColor) nColor=c+1; }
            std::vector<i32> ofs(nColor+1,0);
            for(i32 r2=0;r2<nDofQ;r2++) ofs[color[r2]+1]++;
            for(i32 c=0;c<nColor;c++) ofs[c+1]+=ofs[c];
            perm.resize(nDofQ); { std::vector<i32> w(ofs);
              for(i32 r2=0;r2<nDofQ;r2++) perm[w[color[r2]]++]=r2; }
            std::vector<i32> iperm(nDofQ); for(i32 i=0;i<nDofQ;i++) iperm[perm[i]]=i;
            Rp.assign(nDofQ+1,0); Ci.resize(nnz); Va.resize(nnz);
            for(i32 i=0;i<nDofQ;i++){ i32 o=perm[i]; Rp[i+1]=Rp[i]+(rp[o+1]-rp[o]); }
            #pragma omp parallel for schedule(static)
            for(i32 i=0;i<nDofQ;i++){ i32 o=perm[i];
              std::vector<std::pair<i32,double>> e2; e2.reserve(rp[o+1]-rp[o]);
              for(i32 k2=rp[o];k2<rp[o+1];k2++) e2.push_back({iperm[ci2[k2]],va[k2]});
              std::sort(e2.begin(),e2.end());
              for(size_t t=0;t<e2.size();t++){ Ci[Rp[i]+t]=e2[t].first; Va[Rp[i]+t]=e2[t].second; } }
            rpU=Rp.data(); ciU=Ci.data(); vaU=Va.data();
            d_icPerm=cpI(perm.data(),nDofQ);
            printf("ic     : multicolor reorder -> %d colors (= SpSV level count)\n",nColor);
          }
          cudaMalloc(&d_icRp,(size_t)(nDofQ+1)*sizeof(i32)); cudaMalloc(&d_icCi,(size_t)nnz*sizeof(i32));
          cudaMalloc(&d_icV,(size_t)nnz*sizeof(double));
          cudaMalloc(&d_icX,(size_t)nDofQ*sizeof(double)); cudaMalloc(&d_icY,(size_t)nDofQ*sizeof(double));
          cudaMemcpy(d_icRp,rpU,(size_t)(nDofQ+1)*sizeof(i32),cudaMemcpyHostToDevice);
          cudaMemcpy(d_icCi,ciU,(size_t)nnz*sizeof(i32),cudaMemcpyHostToDevice);
          std::vector<double> vs(vaU,vaU+nnz);
          cusparseStatus_t zs=CUSPARSE_STATUS_SUCCESS; int zpiv=-1;
          cusparseCreate(&icH);
          if (pcMode==9) {   // ---- SSOR: feed A with diag/omega; no factorization, no breakdown ----
            std::vector<double> dH(nDofQ,1.0);
            for(i32 r2=0;r2<nDofQ;r2++) for(i32 k2=rpU[r2];k2<rpU[r2+1];k2++) if(ciU[k2]==r2){ vs[k2]/=omega; dH[r2]=vs[k2]; }
            cudaMemcpy(d_icV,vs.data(),(size_t)nnz*sizeof(double),cudaMemcpyHostToDevice);
            cudaMalloc(&d_ssorD,(size_t)nDofQ*sizeof(double));
            cudaMemcpy(d_ssorD,dH.data(),(size_t)nDofQ*sizeof(double),cudaMemcpyHostToDevice);
          } else {
          if(icShift>0){ for(i32 r2=0;r2<nDofQ;r2++) for(i32 k2=rpU[r2];k2<rpU[r2+1];k2++) if(ciU[k2]==r2) vs[k2]*=(1.0+icShift); }
          cudaMemcpy(d_icV,vs.data(),(size_t)nnz*sizeof(double),cudaMemcpyHostToDevice);
          cusparseMatDescr_t dM=nullptr; csric02Info_t info=nullptr;
          cusparseCreateMatDescr(&dM); cusparseSetMatType(dM,CUSPARSE_MATRIX_TYPE_GENERAL);
          cusparseSetMatIndexBase(dM,CUSPARSE_INDEX_BASE_ZERO);
          cusparseCreateCsric02Info(&info);
          int bs=0; cusparseDcsric02_bufferSize(icH,nDofQ,nnz,dM,d_icV,d_icRp,d_icCi,info,&bs);
          void*buf=nullptr; cudaMalloc(&buf,(size_t)bs);
          cusparseDcsric02_analysis(icH,nDofQ,nnz,dM,d_icV,d_icRp,d_icCi,info,CUSPARSE_SOLVE_POLICY_USE_LEVEL,buf);
          cusparseDcsric02(icH,nDofQ,nnz,dM,d_icV,d_icRp,d_icCi,info,CUSPARSE_SOLVE_POLICY_USE_LEVEL,buf);
          zs=cusparseXcsric02_zeroPivot(icH,info,&zpiv);
          cusparseDestroyCsric02Info(info); cusparseDestroyMatDescr(dM); cudaFree(buf);
          }
          if(zs!=CUSPARSE_STATUS_ZERO_PIVOT){
            // wrap the lower triangle of the in-place factor; SpSV with FILL_MODE_LOWER ignores the
            // untouched strictly-upper entries, and OPERATION_TRANSPOSE gives the L^T back-solve.
            cusparseCreateCsr(&icL,nDofQ,nDofQ,nnz,d_icRp,d_icCi,d_icV,
                              CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
            cusparseFillMode_t fm=CUSPARSE_FILL_MODE_LOWER; cusparseDiagType_t dt=CUSPARSE_DIAG_TYPE_NON_UNIT;
            cusparseSpMatSetAttribute(icL,CUSPARSE_SPMAT_FILL_MODE,&fm,sizeof(fm));
            cusparseSpMatSetAttribute(icL,CUSPARSE_SPMAT_DIAG_TYPE,&dt,sizeof(dt));
            cusparseCreateDnVec(&icVb,nDofQ,d_icX,CUDA_R_64F);
            cusparseCreateDnVec(&icVy,nDofQ,d_icY,CUDA_R_64F);
            cusparseCreateDnVec(&icVx,nDofQ,d_icX,CUDA_R_64F);
            cusparseSpSV_createDescr(&icSvL); cusparseSpSV_createDescr(&icSvU);
            const double one=1.0; size_t b1=0,b2=0;
            cusparseSpSV_bufferSize(icH,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,icL,icVb,icVy,CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,icSvL,&b1);
            cudaMalloc(&d_icW1,b1);
            cusparseSpSV_analysis(icH,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,icL,icVb,icVy,CUDA_R_64F,
                                  CUSPARSE_SPSV_ALG_DEFAULT,icSvL,d_icW1);
            cusparseSpSV_bufferSize(icH,CUSPARSE_OPERATION_TRANSPOSE,&one,icL,icVy,icVx,CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,icSvU,&b2);
            cudaMalloc(&d_icW2,b2);
            cusparseSpSV_analysis(icH,CUSPARSE_OPERATION_TRANSPOSE,&one,icL,icVy,icVx,CUDA_R_64F,
                                  CUSPARSE_SPSV_ALG_DEFAULT,icSvU,d_icW2);
            icReady=(cudaGetLastError()==cudaSuccess);
            if(pcMode==9) printf("ssor   : omega %.3g + SpSV analysis in %.2fs%s\n",
                                 omega,(qpNowUs()-tf)*1e-6, icReady?"":"  [FAILED -> jac]");
            else printf("ic     : IC(0) factored (shift %.3g) + SpSV analysis in %.2fs%s\n",
                        icShift,(qpNowUs()-tf)*1e-6, icReady?"":"  [FAILED -> jac]");
          } else {
            printf("ic     : IC(0) BREAKDOWN at pivot %d (shift %.3g)\n",zpiv,icShift);
          }
          // HARD FAIL by default.  The silent jac fallback cost two whole p3 modal runs: LOBPCG
          // depends on PC quality, so it burned its full sweep cap and returned plausible-looking
          // but WRONG frequencies.  Opt back in with CUT_ICFALLBACK=1 if a degraded run is wanted.
          if(!icReady){
            if(!getenv("CUT_ICFALLBACK")){
              printf("ERROR  : IC(0) unavailable and CUT_ICFALLBACK is not set -- ABORTING.\n"
                     "         Raise CUT_ICSHIFT (needs ~0.05 at p2, ~0.15 at p3; higher p needs more).\n");
              exit(1); }
            printf("ic     : falling back to jac (CUT_ICFALLBACK) -- RESULTS MAY NOT CONVERGE\n");
            pcMode=0; pcName="jac";
          } else if(pcMode==5) pcName="ic";
        }
        // cuSOLVER sparse Cholesky (host API; reorder=3 => symamd fill-reduction)
        if (wantDirect) {
        long ts=qpNowUs();
        std::vector<double> bH(nDofQ), xH(nDofQ,0.0);
        for(i32 i=0;i<nDofQ;i++) bH[i]=(double)bvec[i];
        cusolverSpHandle_t hs=nullptr; cusparseMatDescr_t descr=nullptr;
        cusolverSpCreate(&hs); cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        int sing=-1; double tolD=1e-12;
        cusolverStatus_t st=cusolverSpDcsrlsvcholHost(hs,nDofQ,nnz,descr,va.data(),rp.data(),ci2.data(),
                                                      bH.data(),tolD,3,xH.data(),&sing);
        cusolverSpDestroy(hs); cusparseDestroyMatDescr(descr);
        if(st!=CUSOLVER_STATUS_SUCCESS || sing>=0){
          printf("direct : FAILED (status %d, singularity %d) -- falling back to iterative\n",(int)st,sing);
        } else {
          for(i32 i=0;i<nDofQ;i++) uv[i]=(real)xH[i];
          double bn2=0,rn2=0; for(i32 i=0;i<nDofQ;i++) bn2+=bH[i]*bH[i];
          { std::vector<double> Ax(nDofQ,0.0);
            for(i32 r=0;r<nDofQ;r++){ double s=0; for(i32 k2=rp[r];k2<rp[r+1];k2++) s+=va[k2]*xH[ci2[k2]]; Ax[r]=s; }
            for(i32 i=0;i<nDofQ;i++){ double d=bH[i]-Ax[i]; rn2+=d*d; } }
          cgIters=0; cgRes=sqrt(rn2)/std::max(1e-300,sqrt(bn2));
          printf("direct : cuSOLVER sparse Cholesky solved in %.2fs   true rel res %.3e\n",(qpNowUs()-ts)*1e-6,cgRes);
          for(i32 i=0;i<nR;i++) d_u[i]=(real)xH[i];   // CG loop is skipped below; normal reporting path follows
          directOK=true;
        }
        }
      }
      real *d_zold=alR(nR);   // flexible-CG (Polak-Ribiere) tolerates the nonlinear V-cycle PC
      t0=qpNowUs();           // time the CG loop only (excludes one-time setup / NNLS prune)
      cudaMemcpy(d_r,d_b,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);   // u=0 -> r=b
      precond(d_z,d_r); cudaMemcpy(d_pd,d_z,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
      double bn=sqrt(dot(d_b,d_b)); if(bn==0)bn=1; double rz=dot(d_r,d_z); i32 it=0; double rn=0;
      for(; !directOK && it<cgMaxIt; it++){ apply(d_pd,d_Ap); double pAp=dot(d_pd,d_Ap); if(!(pAp>0)){ printf("WARNING: IGA GPU-CG breakdown pAp=%.3e\n",pAp); break; }
        double al=rz/pAp; cutAxpyK<<<GS,BS>>>(d_u,d_pd,(real)al,nR); cutAxpyK<<<GS,BS>>>(d_r,d_Ap,(real)-al,nR); cudaDeviceSynchronize();
        rn=sqrt(dot(d_r,d_r)); if(rn<=cgTol*bn){ it++; break; }
        cudaMemcpy(d_zold,d_z,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
        precond(d_z,d_r); cudaDeviceSynchronize();
        double rzn=dot(d_r,d_z), rzo=dot(d_r,d_zold), be=(rzn-rzo)/rz; rz=rzn;
        cutAxpyK<<<GS,BS>>>(d_pd,d_pd,(real)(be-1),nR); cutAxpyK<<<GS,BS>>>(d_pd,d_z,(real)1,nR); cudaDeviceSynchronize(); }
      cgIters=it; cgRes=rn/bn; cudaDeviceSynchronize(); memcpy(uv.data(),d_u,(size_t)nR*sizeof(real));
      tSolveEnd=qpNowUs();   // freeze the STATIC solve time before modal runs on the same t0
      // ================= MODAL ANALYSIS (CUT_MODAL=N) =================
      //  K phi = lambda M phi,  lambda = omega^2.  Block INVERSE iteration + Rayleigh-Ritz.
      //  Inverse iteration is the right choice for cut FEM: a dof whose basis has no support in
      //  Omega carries ZERO mass, i.e. lambda = infinity, which maps to 0 under K^-1 M and is
      //  damped out automatically instead of polluting the spectrum.  K is SPD (Nitsche clamp),
      //  so there are no rigid-body modes to deflate.
      if (nModal>0 && d_mVa) {
        long tmo=qpNowUs(); const i32 nev=nModal;
        std::vector<real*> Xv(nev),Yv(nev);
        for(i32 j=0;j<nev;j++){ Xv[j]=alR(nR); Yv[j]=alR(nR); }
        std::vector<real> h0(nR);
        for(i32 j=0;j<nev;j++){   // deterministic, linearly independent start
          for(i32 i=0;i<nR;i++) h0[i]=(real)sin(0.7*(j+1)*(i%97+1)+0.31*j);
          cudaMemcpy(Xv[j],h0.data(),(size_t)nR*sizeof(real),cudaMemcpyHostToDevice); }
        auto massMv=[&](const real*x,real*y){ cutCsrMvK<<<GS,BS>>>(d_mRp,d_mCi,d_mVa,x,y,nR); cudaDeviceSynchronize(); };
        //  tolK TIGHTENS with the sweep: solving to 1e-9 while the iterate is still noise is pure waste
        //  (that is what cost 337931 CG its at res48).  Inverse iteration only needs the DIRECTION.
        double tolK=1e-3;
        auto solveK=[&](const real*b,real*x){    // PCG with the SAME preconditioner as the static solve
          cutSetK<<<GS,BS>>>(x,(real)0,nR);
          cudaMemcpy(d_r,b,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
          precond(d_z,d_r); cudaMemcpy(d_pd,d_z,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
          double bn2=sqrt(dot(b,b)); if(bn2==0) return 0; double rz2=dot(d_r,d_z); i32 k2=0;
          for(;k2<cgMaxIt;k2++){ apply(d_pd,d_Ap); double pAp=dot(d_pd,d_Ap); if(!(pAp>0)) break;
            double al=rz2/pAp; cutAxpyK<<<GS,BS>>>(x,d_pd,(real)al,nR); cutAxpyK<<<GS,BS>>>(d_r,d_Ap,(real)-al,nR);
            cudaDeviceSynchronize(); if(sqrt(dot(d_r,d_r))<=tolK*bn2){ k2++; break; }
            precond(d_z,d_r); cudaDeviceSynchronize();
            double rzn=dot(d_r,d_z), be=rzn/rz2; rz2=rzn;
            cutAxpyK<<<GS,BS>>>(d_pd,d_pd,(real)(be-1),nR); cutAxpyK<<<GS,BS>>>(d_pd,d_z,(real)1,nR);
            cudaDeviceSynchronize(); }
          return (int)k2; };
        std::vector<double> lam(nev,0.0);
        const bool useInv = getenv("CUT_MODALINV")!=nullptr;
        // ---------------- LOBPCG (default) ----------------
        //  Search space S = [X | W | P] with W = T*residual (T = the SAME multicolor-IC preconditioner,
        //  which already approximates K^-1) and P the previous conjugate direction.  Costs ~3m matvecs
        //  per sweep instead of m FULL CG SOLVES -- the inverse-iteration path spent 337931 CG its at
        //  res48 because every one of its 150 solves ran to 1e-9 even while the vectors were garbage.
        //  Basis is K-orthonormalized by MGS, so the small Rayleigh-Ritz problem has B = I and the
        //  nested-span property makes the [0,m) / [m,k) split for the P update exact.
        //  !! We solve the RECIPROCAL pencil  M x = mu K x  for the LARGEST mu (lambda = 1/mu).
        //  Reason: M is SINGULAR here -- a dof whose basis has no support in Omega has ZERO mass, so
        //  the (K,M) pencil has an infinite-eigenvalue cloud.  Rayleigh-Ritz on (K,M) locks straight
        //  onto that cloud (measured: it returned lambda ~ 3e3 and the basis collapsed to k=3).  On
        //  (M,K) those same modes give mu = 0, i.e. the MINIMUM, so seeking maxima pushes them away.
        //  K is SPD (Nitsche clamp) so the K inner product is a genuine inner product.
        if (!useInv) {
          const i32 m=nev, mx=3*m;
          real *Sf=nullptr,*MSf=nullptr,*Xn=nullptr,*Pn=nullptr,*coef=nullptr,*tmp=nullptr;
          cudaMalloc(&Sf,(size_t)mx*nR*sizeof(real)); cudaMalloc(&MSf,(size_t)mx*nR*sizeof(real));
          cudaMalloc(&Xn,(size_t)m*nR*sizeof(real));  cudaMalloc(&Pn,(size_t)m*nR*sizeof(real));
          cudaMallocManaged(&coef,(size_t)mx*sizeof(real)); cudaMalloc(&tmp,(size_t)nR*sizeof(real));
          auto V=[&](real*b,i32 j)->real*{ return b+(size_t)j*nR; };
          for(i32 j=0;j<m;j++) cudaMemcpy(V(Sf,j),Xv[j],(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
          i32 nP=0; std::vector<double> lamL(m,0.0), lamPrev(m,0.0);
          std::vector<double> Ad((size_t)mx*mx), Vd((size_t)mx*mx), wd(mx);
          i32 nOutL=300; { const char*e=getenv("CUT_MODALIT"); if(e) nOutL=atoi(e); }
          i32 nApply=0;
          for (i32 sweep=0; sweep<nOutL; sweep++) {
            i32 k=m+(sweep>0?m:0)+nP;   // [X | W | P]
            // --- M-orthonormalize the basis (modified Gram-Schmidt in the M inner product) ---
            i32 kk=0;
            for (i32 j=0;j<k;j++) {
              if (j!=kk) cudaMemcpy(V(Sf,kk),V(Sf,j),(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
              apply(V(Sf,kk),tmp); nApply++; double n0=sqrt(std::max(1e-300,dot(V(Sf,kk),tmp)));
              // TWO passes: the K inner product has kappa(K) ~ 1e6 here, and single-pass MGS loses
              // orthogonality badly at that conditioning (measured: basis collapsed k=12 -> 3).
              for (i32 pass=0;pass<2;pass++)
                for (i32 i2=0;i2<kk;i2++){ double c=dot(V(Sf,kk),V(MSf,i2));
                  cutAxpyK<<<GS,BS>>>(V(Sf,kk),V(Sf,i2),(real)(-c),nR); cudaDeviceSynchronize(); }
              apply(V(Sf,kk),tmp); nApply++; double nm=sqrt(std::max(0.0,dot(V(Sf,kk),tmp)));
              if (nm < 1e-11*std::max(n0,1e-300)) continue;          // linearly dependent -> drop
              cutAxpyK<<<GS,BS>>>(V(Sf,kk),V(Sf,kk),(real)(1.0/nm-1.0),nR); cudaDeviceSynchronize();
              apply(V(Sf,kk),V(MSf,kk)); nApply++; kk++; }   // MSf now caches K*s_i
            k=kk; if(k<m){ printf("modal  : LOBPCG basis collapsed (k=%d<%d)\n",k,m); break; }
            // --- Rayleigh-Ritz on the RECIPROCAL pencil: A = S^T M S, B = I (K-orthonormal S) ---
            for (i32 j=0;j<k;j++){ massMv(V(Sf,j),tmp);
              for(i32 i2=0;i2<k;i2++) Ad[(size_t)i2*k+j]=dot(V(Sf,i2),tmp); }
            for(i32 a=0;a<k;a++)for(i32 b=a+1;b<k;b++){ double s=0.5*(Ad[(size_t)a*k+b]+Ad[(size_t)b*k+a]);
              Ad[(size_t)a*k+b]=s; Ad[(size_t)b*k+a]=s; }
            jacobiEig(Ad.data(),Vd.data(),wd.data(),k);
            // --- rotate.  jacobiEig sorts ASCENDING and we want the LARGEST mu, so column (k-1-j). ---
            for (i32 j=0;j<m;j++){
              for(i32 i2=0;i2<k;i2++) coef[i2]=(real)Vd[(size_t)i2*k+(k-1-j)];
              cutCombK<<<GS,BS>>>(V(Xn,j),Sf,coef,k,nR); cudaDeviceSynchronize(); }
            if (k>m){ for (i32 j=0;j<m;j++){
                for(i32 i2=0;i2<k-m;i2++) coef[i2]=(real)Vd[(size_t)(i2+m)*k+(k-1-j)];
                cutCombK<<<GS,BS>>>(V(Pn,j),Sf+(size_t)m*nR,coef,k-m,nR); cudaDeviceSynchronize(); } nP=m; }
            cudaDeviceSynchronize();
            double drift=0;
            for(i32 j=0;j<m;j++){ lamPrev[j]=lamL[j];
              double mu=wd[k-1-j]; lamL[j]=(mu>1e-300? 1.0/mu : 0.0);   // lambda = 1/mu
              drift=std::max(drift,fabs(lamL[j]-lamPrev[j])/std::max(1e-300,fabs(lamL[j]))); }
            // --- rebuild [X | W | P]: W_j = T (M x_j - mu_j K x_j), T ~ K^-1 ---
            double rmax=0;
            for (i32 j=0;j<m;j++){
              double mu=(lamL[j]>0? 1.0/lamL[j] : 0.0);
              massMv(V(Xn,j),tmp); apply(V(Xn,j),V(MSf,0));   // MSf[0] reused as scratch
              cutAxpyK<<<GS,BS>>>(tmp,V(MSf,0),(real)(-mu),nR); cudaDeviceSynchronize();
              rmax=std::max(rmax,sqrt(dot(tmp,tmp)));
              precond(V(Sf,m+j),tmp); }
            cudaDeviceSynchronize();
            for(i32 j=0;j<m;j++) cudaMemcpy(V(Sf,j),V(Xn,j),(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
            if (nP) for(i32 j=0;j<m;j++) cudaMemcpy(V(Sf,2*m+j),V(Pn,j),(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
            if (sweep%10==0 || drift<1e-9){
              printf("modal  : lobpcg %3d  drift %.2e  |res| %.2e  f =",sweep,drift,rmax);
              for(i32 j=0;j<m;j++) printf(" %.6g",(lamL[j]>0?sqrt(lamL[j])/(2*M_PI):0.0));
              printf("\n"); fflush(stdout); }
            if (sweep>2 && drift<1e-9) break;
          }
          for(i32 j=0;j<nev;j++) cudaMemcpy(Xv[j],Sf+(size_t)j*nR,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
          for(i32 j=0;j<nev;j++) lam[j]=lamL[j];
          printf("modal  : LOBPCG done, %d K-applies total (vs 337931 CG its for inverse iteration at res48)\n",nApply);
          for(void*pp:{(void*)Sf,(void*)MSf,(void*)Xn,(void*)Pn,(void*)coef,(void*)tmp}) cudaFree(pp);
        }
        std::vector<double> lamOld(nev,0.0);
        if (useInv) {
        std::vector<double> Amat((size_t)nev*nev), Vmat((size_t)nev*nev), wv(nev);
        i32 nOut=25; { const char*e=getenv("CUT_MODALIT"); if(e) nOut=atoi(e); }
        i32 totCg=0;
        for (i32 sweep=0; sweep<nOut; sweep++) {
          tolK=std::max(1e-10,1e-3*pow(0.25,(double)sweep));
          for(i32 j=0;j<nev;j++){ massMv(Xv[j],Yv[j]); totCg+=solveK(Yv[j],Xv[j]); }
          // M-orthonormalize the block (modified Gram-Schmidt in the M inner product)
          for(i32 j=0;j<nev;j++){
            for(i32 k2=0;k2<j;k2++){ massMv(Xv[k2],d_Ap); double c=dot(Xv[j],d_Ap);
              cutAxpyK<<<GS,BS>>>(Xv[j],Xv[k2],(real)(-c),nR); cudaDeviceSynchronize(); }
            massMv(Xv[j],d_Ap); double nm=sqrt(std::max(1e-300,dot(Xv[j],d_Ap)));
            cutAxpyK<<<GS,BS>>>(Xv[j],Xv[j],(real)(1.0/nm-1.0),nR); cudaDeviceSynchronize(); }
          // Rayleigh-Ritz: with M-orthonormal X, B = I and A = X^T K X
          for(i32 j=0;j<nev;j++){ apply(Xv[j],d_Ap);
            for(i32 k2=0;k2<nev;k2++) Amat[(size_t)k2*nev+j]=dot(Xv[k2],d_Ap); }
          for(i32 a=0;a<nev;a++)for(i32 b=a+1;b<nev;b++){ double s=0.5*(Amat[(size_t)a*nev+b]+Amat[(size_t)b*nev+a]);
            Amat[(size_t)a*nev+b]=s; Amat[(size_t)b*nev+a]=s; }
          jacobiEig(Amat.data(),Vmat.data(),wv.data(),nev);
          // rotate the block by the Ritz vectors
          { std::vector<std::vector<real>> hx(nev,std::vector<real>(nR));
            std::vector<std::vector<real>> hs(nev);
            for(i32 j=0;j<nev;j++){ hs[j].resize(nR); cudaMemcpy(hs[j].data(),Xv[j],(size_t)nR*sizeof(real),cudaMemcpyDeviceToHost); }
            for(i32 j=0;j<nev;j++){ std::fill(hx[j].begin(),hx[j].end(),(real)0);
              for(i32 k2=0;k2<nev;k2++){ real c=(real)Vmat[(size_t)k2*nev+j];
                for(i32 i=0;i<nR;i++) hx[j][i]+=c*hs[k2][i]; } }
            for(i32 j=0;j<nev;j++) cudaMemcpy(Xv[j],hx[j].data(),(size_t)nR*sizeof(real),cudaMemcpyHostToDevice); }
          double drift=0; for(i32 j=0;j<nev;j++){ lamOld[j]=lam[j]; lam[j]=wv[j];
            drift=std::max(drift,fabs(lam[j]-lamOld[j])/std::max(1e-300,fabs(lam[j]))); }
          printf("modal  : sweep %2d  drift %.2e  f =",sweep,drift);
          for(i32 j=0;j<nev;j++) printf(" %.5g",(lam[j]>0?sqrt(lam[j])/(2*M_PI):0.0));
          printf("\n"); fflush(stdout);
          if (sweep>0 && drift<1e-7) break;
        }
        printf("modal  : %d modes, %d total CG its, %.1fs\n",nev,totCg,(qpNowUs()-tmo)*1e-6);
        }
        printf("modal  : elapsed %.1fs\n",(qpNowUs()-tmo)*1e-6);
        printf("modal  :  mode      lambda        omega [rad/s]      f [Hz]\n");
        for(i32 j=0;j<nev;j++){ double om=(lam[j]>0?sqrt(lam[j]):0.0);
          printf("modal  :   %2d   %13.6e   %13.6e   %13.6e\n",j+1,lam[j],om,om/(2*M_PI)); }
        { std::ofstream os(("output/"+outTag+"_modes.csv").c_str());
          os<<"mode,lambda,omega,f\n";
          for(i32 j=0;j<nev;j++){ double om=(lam[j]>0?sqrt(lam[j]):0.0);
            os<<j+1<<","<<lam[j]<<","<<om<<","<<om/(2*M_PI)<<"\n"; }
          for(i32 j=0;j<nev;j++){ std::vector<real> hx(nR);
            cudaMemcpy(hx.data(),Xv[j],(size_t)nR*sizeof(real),cudaMemcpyDeviceToHost);
            std::ofstream om2(("output/"+outTag+"_mode"+std::to_string(j+1)+".csv").c_str());
            om2<<"x,y,z,ux,uy,uz\n";
            for(i32 nd=0;nd<nNodeQ;nd++){ double u3[3]; gather3(hx,nd,u3);
              om2<<nodeXQ[3*nd]<<","<<nodeXQ[3*nd+1]<<","<<nodeXQ[3*nd+2]<<","
                 <<u3[0]<<","<<u3[1]<<","<<u3[2]<<"\n"; } } }
        for(i32 j=0;j<nev;j++){ cudaFree(Xv[j]); cudaFree(Yv[j]); }
        for(void*pp:{(void*)d_mRp,(void*)d_mCi,(void*)d_mVa}) if(pp) cudaFree(pp);
      }
      if(icSvL) cusparseSpSV_destroyDescr(icSvL); if(icSvU) cusparseSpSV_destroyDescr(icSvU);
      if(icVb) cusparseDestroyDnVec(icVb); if(icVy) cusparseDestroyDnVec(icVy); if(icVx) cusparseDestroyDnVec(icVx);
      if(icL) cusparseDestroySpMat(icL); if(icH) cusparseDestroy(icH);
      for(void*pp:{(void*)d_icV,(void*)d_icX,(void*)d_icY,(void*)d_icRp,(void*)d_icCi,d_icW1,d_icW2,(void*)d_l1,(void*)d_icPerm,(void*)d_ssorD}) if(pp) cudaFree(pp);
      cudaFree(d_Binv); cudaFree(d_intInv); cudaFree(d_cutInv); cudaFree(d_zold);
      for(void*pp:pmgFree) cudaFree(pp);
      for(void*pp:{(void*)d_eNode,(void*)d_nMap,(void*)d_nRot,(void*)d_intList,(void*)d_cutElem,(void*)d_eCijk,(void*)d_eCut,(void*)d_volJ,(void*)d_surfJ,(void*)d_volP,(void*)d_surfP,(void*)d_volOff,(void*)d_surfOff,(void*)d_surfDir,(void*)d_gfM,(void*)d_gfP,(void*)d_gfD,(void*)d_Kref,(void*)d_Kg0,(void*)d_Kg1,(void*)d_Kg2,(void*)d_b,(void*)d_diag,(void*)d_u,(void*)d_r,(void*)d_z,(void*)d_pd,(void*)d_Ap,(void*)d_xn,(void*)d_yn,(void*)d_acc}) cudaFree(pp);
      printf("solver : GPU %s-PCG cut-cell (interior K_ref + Saye re-quadrature + ghost)\n", pcName.c_str());
    } else {
    std::vector<real> r=bvec, z(nDofQ), pd(nDofQ), Ap(nDofQ);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) z[i]=r[i]/diagv[i];
    pd=z;
    double rz=0; for (i32 i=0;i<nDofQ;i++) rz+=r[i]*z[i];
    double bnorm=0; for (i32 i=0;i<nDofQ;i++) bnorm+=bvec[i]*bvec[i]; bnorm=sqrt(bnorm);
    i32 it=0;
    for (; it<cgMaxIt && bnorm>0; it++){
      applyA(pd,Ap);
      double pAp=0;
      #pragma omp parallel for reduction(+:pAp)
      for (i32 i=0;i<nDofQ;i++) pAp+=pd[i]*Ap[i];
      if (!(pAp>0)){ printf("WARNING: IGA CG breakdown pAp=%.3e\n",pAp); break; }
      double al=rz/pAp;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=al*pd[i]; r[i]-=al*Ap[i]; }
      double rn=0;
      #pragma omp parallel for reduction(+:rn)
      for (i32 i=0;i<nDofQ;i++) rn+=r[i]*r[i];
      rn=sqrt(rn);
      if (rn<=cgTol*bnorm){ it++; cgRes=rn/bnorm; break; }
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=r[i]/diagv[i];
      double rz2=0;
      #pragma omp parallel for reduction(+:rz2)
      for (i32 i=0;i<nDofQ;i++) rz2+=r[i]*z[i];
      double be=rz2/rz; rz=rz2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=z[i]+be*pd[i];
      cgRes=rn/bnorm;
    }
    cgIters=it;
    }   // end host-CG else

    // ---- errors + geometry ----
    double l2e=0,l2n=0,ene=0,enn=0,vol=0,area=0;
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(uv,nod[a],u3); uloc[3*a]=u3[0];uloc[3*a+1]=u3[1];uloc[3*a+2]=u3[2]; }
      std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
      if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.qx[i];tens[qi].x[1]=Bp.qx[j];tens[qi].x[2]=Bp.qx[k];
          tens[qi].w=Bp.qw[i]*Bp.qw[j]*Bp.qw[k]; qi++; } vn=tens.data(); nv=ndof;
      } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
        double gX[QN_MAX*QN_MAX*QN_MAX][3], dw;
        if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          dw=fabs(detJ)*vn[q].w; }
        else { for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]/h; dw=vn[q].w*h*h*h; }
        double uh[3]={0,0,0}, gh[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++){ uh[l]+=uloc[3*a+l]*vb[a];
          gh[l][0]+=uloc[3*a+l]*gX[a][0]; gh[l][1]+=uloc[3*a+l]*gX[a][1]; gh[l][2]+=uloc[3*a+l]*gX[a][2]; }
        real X[3]; physOf(elems[e],xr,X); real ue[3]; prob.exactU(X[0],X[1],X[2],ue);
        real ge[3][3]; prob.exactGradU(X[0],X[1],X[2],ge);
        vol+=dw; for (i32 l=0;l<3;l++){ double d=uh[l]-ue[l]; l2e+=d*d*dw; l2n+=ue[l]*ue[l]*dw; }
        double ee[3][3],se[3][3],tre=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ee[i2][j2]=0.5*((gh[i2][j2]-ge[i2][j2])+(gh[j2][i2]-ge[j2][i2]));
        tre=ee[0][0]+ee[1][1]+ee[2][2];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) se[i2][j2]=2*mu*ee[i2][j2]+(i2==j2?lam*tre:0);
        double en=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) en+=se[i2][j2]*ee[i2][j2]; ene+=en*dw;
        double eeE[3][3],seE[3][3],trE=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eeE[i2][j2]=0.5*(ge[i2][j2]+ge[j2][i2]);
        trE=eeE[0][0]+eeE[1][1]+eeE[2][2];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) seE[i2][j2]=2*mu*eeE[i2][j2]+(i2==j2?lam*trE:0);
        double enE=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) enE+=seE[i2][j2]*eeE[i2][j2]; enn+=enE*dw; }
      if (elems[e].cut){ i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
        for (i32 q=0;q<ns;q++){ if (!cyl){ area+=sn[q].w*h*h; }
          else { real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]}; double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
            double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]}, nraw[3];
            for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
            area+=fabs(detJ)*sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2])*sn[q].w; } } } }
    errL2=sqrt(l2e); normL2=sqrt(l2n); errEnergy=sqrt(ene); normEnergy=sqrt(enn);
    volOmega=vol; areaGamma=area;

    // ---- report ----
    double ms=((tSolveEnd?tSolveEnd:qpNowUs())-t0)/1000.0; i32 nCutTrueQ=0; for (i32 e=0;e<nE;e++) if (elems[e].cut) nCutTrueQ++;
    printf("active : %d elements (%d cut), %d ghost faces\n", nE, nCutTrueQ, nGFQ);
    printf("dofs   : %d nodes -> %d unknowns   h = %.6g   p = %d\n", nNodeQ, nDofQ, (double)h, p);
    printf("geom   : |Omega_h| = %.8g", volOmega);
    if (volExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", volExact, volOmega-volExact, 100.0*fabs(volOmega-volExact)/volExact);
    printf("\n         |Gamma_h| = %.8g", areaGamma);
    if (areaExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", areaExact, areaGamma-areaExact, 100.0*fabs(areaGamma-areaExact)/areaExact);
    printf("\nsolve  : CG %d iters, rel res %.2e   (%.0f ms)\n", cgIters, cgRes, ms);
    if ((prob.caseId==CASE_MMS || prob.caseId==CASE_MMS_CYL) && normL2>0)
      printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)\n", errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);

    // ---- nodal STRESS recovery -------------------------------------------------
    //  sigma = 2 mu eps(u) + lam tr(eps) I, evaluated at the GLL nodes of every element
    //  (same metric path as applyElemCyl) and averaged over the elements sharing a node.
    //  phi is written alongside because stress at a node OUTSIDE Omega is meaningless --
    //  there u is only the polynomial extension the cut basis happens to carry, so any
    //  plot or max-stress query MUST mask on phi < 0.
    std::vector<float> vmN(nNodeQ,0.f), phiN(nNodeQ,0.f), sigN((size_t)6*nNodeQ,0.f);
    std::vector<float> cntN(nNodeQ,0.f);
    if (wantVtu && !outTag.empty()) {
      real gb[3*QN_MAX*QN_MAX*QN_MAX];
      std::vector<double> uloc((size_t)3*ndof);
      for (i32 e=0;e<nE;e++){
        const i32*nod=&eNodeQ[(size_t)e*ndof];
        for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(uv,nod[a],u3);
          uloc[3*a]=u3[0]; uloc[3*a+1]=u3[1]; uloc[3*a+2]=u3[2]; }
        for (i32 kk=0;kk<n;kk++)for(i32 jj=0;jj<n;jj++)for(i32 ii=0;ii<n;ii++){
          i32 a0=ii+n*(jj+n*kk); i32 nd=nod[a0];
          real xr[3]={Bp.t[ii],Bp.t[jj],Bp.t[kk]};
          double Jinv[3][3],detJ;
          if (cyl) metric(elems[e],xr,Jinv,detJ);
          else { for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) Jinv[i2][j2]=0;
                 Jinv[0][0]=Jinv[1][1]=Jinv[2][2]=1.0/h; detJ=h*h*h; }
          Bp.allGradRef(xr,gb);
          double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
          for (i32 a=0;a<ndof;a++){ double gX[3];
            for(i32 d=0;d<3;d++) gX[d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
            for(i32 i2=0;i2<3;i2++){ gU[i2][0]+=uloc[3*a+i2]*gX[0];
              gU[i2][1]+=uloc[3*a+i2]*gX[1]; gU[i2][2]+=uloc[3*a+i2]*gX[2]; } }
          double eps[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
          double trc=eps[0][0]+eps[1][1]+eps[2][2], sg[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sg[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*trc:0);
          double vm=sqrt(0.5*((sg[0][0]-sg[1][1])*(sg[0][0]-sg[1][1])
                             +(sg[1][1]-sg[2][2])*(sg[1][1]-sg[2][2])
                             +(sg[2][2]-sg[0][0])*(sg[2][2]-sg[0][0]))
                         +3.0*(sg[0][1]*sg[0][1]+sg[1][2]*sg[1][2]+sg[2][0]*sg[2][0]));
          vmN[nd]+=(float)vm; cntN[nd]+=1.f;
          const i32 I6[6][2]={{0,0},{1,1},{2,2},{0,1},{1,2},{2,0}};
          for(i32 c6=0;c6<6;c6++) sigN[6*(size_t)nd+c6]+=(float)sg[I6[c6][0]][I6[c6][1]];
          phiN[nd]=(float)ls.phi((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h);
        } }
      double vmMax=0; i32 ndMax=-1;
      for (i32 nd=0;nd<nNodeQ;nd++){ if(cntN[nd]>0){ vmN[nd]/=cntN[nd];
          for(i32 c6=0;c6<6;c6++) sigN[6*(size_t)nd+c6]/=cntN[nd]; }
        if (phiN[nd]<0 && vmN[nd]>vmMax){ vmMax=vmN[nd]; ndMax=nd; } }
      if (ndMax>=0) printf("stress : peak von Mises %.6e at (%.4f, %.4f, %.4f)  [solid nodes only, phi<0]\n",
                           vmMax,nodeXQ[3*ndMax],nodeXQ[3*ndMax+1],nodeXQ[3*ndMax+2]);
    }
    // ---- constant-z SECTION sample (CUT_SLICEZ=<z>, CUT_SLICEN=<pts/elem/axis>) ----
    // Samples u and von Mises on a plane by EVALUATING THE BASIS at reference
    // points, not by reading nodal values.  That is the only basis-fair way to
    // compare fem against iga: B-splines are NOT interpolatory, so a control-point
    // coefficient is not the displacement there and the nodal arrays above (which
    // average GLL-point values onto nodes) are meaningless for iga.
    // Physical z equals grid z in both coord systems (the metric's third row is
    // {0,0,h}), so the plane inverts exactly: xr2 = (z0-org2)/h - ck.
    if (getenv("CUT_SLICEZ")) {
      mkdir("output",0755);
      // "<axis>,<coord>" in the COMPUTATIONAL frame (axis 0/1/2, coord in grid
      // units).  For the cylindrical blade the span is RADIAL, so a constant-z
      // world cut is oblique and exaggerates thickness -- axis 0 (constant r) is
      // the true blade-to-blade section.  A bare number means axis 2 (legacy).
      const char* sarg = getenv("CUT_SLICEZ");
      i32 sax = 2; double z0 = 0;
      { const char* cm = strchr(sarg,',');
        if (cm) { sax = atoi(sarg); z0 = atof(cm+1); } else z0 = atof(sarg); }
      const i32 t1=(sax+1)%3, t2=(sax+2)%3;
      const i32 ms = getenv("CUT_SLICEN") ? atoi(getenv("CUT_SLICEN")) : 12;
      std::string fn = "output/" + (outTag.empty()?std::string("dump"):outTag) + "_slice.csv";
      std::ofstream os(fn.c_str());
      os<<"s1,s2,x,y,z,phi,ux,uy,uz,umag,vm\n";
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      std::vector<double> uloc((size_t)3*ndof);
      size_t nOut=0;
      for (i32 e=0;e<nE;e++){
        const i32 ec[3]={elems[e].ci,elems[e].cj,elems[e].ck};
        double zr = (z0 - (double)ls.org[sax])/h - (double)ec[sax];
        if (zr < 0.0 || zr > 1.0) continue;                 // plane misses this element
        const i32*nod=&eNodeQ[(size_t)e*ndof];
        for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(uv,nod[a],u3);
          uloc[3*a]=u3[0]; uloc[3*a+1]=u3[1]; uloc[3*a+2]=u3[2]; }
        for (i32 jj=0;jj<ms;jj++) for (i32 ii=0;ii<ms;ii++){
          real xr[3]; xr[sax]=(real)zr; xr[t1]=(real)((ii+0.5)/ms); xr[t2]=(real)((jj+0.5)/ms);
          double s1=(ec[t1]+xr[t1])*h+(double)ls.org[t1], s2=(ec[t2]+xr[t2])*h+(double)ls.org[t2];
          real X[3]; physOf(elems[e],xr,X);
          double phi = (double)ls.phi((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h);
          double Jinv[3][3],detJ;
          if (cyl) metric(elems[e],xr,Jinv,detJ);
          else { for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) Jinv[i2][j2]=0;
                 Jinv[0][0]=Jinv[1][1]=Jinv[2][2]=1.0/h; detJ=h*h*h; }
          Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
          double uu[3]={0,0,0}, gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
          for (i32 a=0;a<ndof;a++){ double gX[3];
            for(i32 d=0;d<3;d++) gX[d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
            for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2]; uu[i2]+=ua*vb[a];
              gU[i2][0]+=ua*gX[0]; gU[i2][1]+=ua*gX[1]; gU[i2][2]+=ua*gX[2]; } }
          double eps[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
          double trc=eps[0][0]+eps[1][1]+eps[2][2], sg[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sg[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*trc:0);
          double vm=sqrt(0.5*((sg[0][0]-sg[1][1])*(sg[0][0]-sg[1][1])
                             +(sg[1][1]-sg[2][2])*(sg[1][1]-sg[2][2])
                             +(sg[2][2]-sg[0][0])*(sg[2][2]-sg[0][0]))
                         +3.0*(sg[0][1]*sg[0][1]+sg[1][2]*sg[1][2]+sg[2][0]*sg[2][0]));
          double um=sqrt(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2]);
          os<<s1<<","<<s2<<","<<X[0]<<","<<X[1]<<","<<X[2]<<","<<phi<<","
            <<uu[0]<<","<<uu[1]<<","<<uu[2]<<","<<um<<","<<vm<<"\n";
          nOut++;
        } }
      printf("slice  : axis %d at %.6f, %zu samples (%d^2/elem) -> %s\n", sax, z0, nOut, ms, fn.c_str());
    }
    // ---- VTU (p^3 sub-hexes) ----
    if (wantVtu && !outTag.empty()){ mkdir("output",0755);
      std::string fn="output/"+outTag+"_femqp.vtu"; std::ofstream os(fn.c_str(),std::ios::binary);
      i64 nSub=(i64)nE*p*p*p; static const i32 HEX[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
      os<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        <<"  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nNodeQ<<"\" NumberOfCells=\""<<nSub<<"\">\n";
      os<<"      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++) os<<(float)nodeXQ[3*nd]<<" "<<(float)nodeXQ[3*nd+1]<<" "<<(float)nodeXQ[3*nd+2]<<"\n";
      os<<"        </DataArray>\n      </Points>\n";
      os<<"      <PointData Vectors=\"u\">\n        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++){ double u3[3]; gather3(uv,nd,u3); os<<(float)u3[0]<<" "<<(float)u3[1]<<" "<<(float)u3[2]<<"\n"; }
      os<<"        </DataArray>\n";
      os<<"        <DataArray type=\"Float32\" Name=\"vonMises\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++) os<<vmN[nd]<<"\n";
      os<<"        </DataArray>\n";
      os<<"        <DataArray type=\"Float32\" Name=\"phi\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++) os<<phiN[nd]<<"\n";
      os<<"        </DataArray>\n";
      os<<"        <DataArray type=\"Float32\" Name=\"sigma\" NumberOfComponents=\"6\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++){ for(i32 c6=0;c6<6;c6++) os<<sigN[6*(size_t)nd+c6]<<" "; os<<"\n"; }
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
}
