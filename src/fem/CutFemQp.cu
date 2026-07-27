//
// Higher-order (Qp) path of the CutFEM solver -- CutFemSolver::runQp().
//
// Production realization of the method verified standalone in QpMms.cu, reusing
// the sparse octree + oracle (buildMesh) for the active block set and the
// verified modules QpBasis / PolyFit / SayeQuad / QpElem.  The p=1 GPU path
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

#include "CutFemSolver.cuh"
#include "QpBasis.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "QpElem.h"

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

// ---- quadrature compression (Potter, "Fast Construction of Efficient Cut Cell
//      Quadratures"): prune a dense Saye rule to a minimal POSITIVE rule that
//      reproduces the same polynomial moments, via Lawson-Hanson NNLS. ---------
static inline void legShift(double x,int K,double*P){ double t=2*x-1; P[0]=1; if(K>=1)P[1]=t;
  for(int k=1;k<K;k++) P[k+1]=((2*k+1)*t*P[k]-k*P[k-1])/(k+1); }
static bool solveSPD(const double*Gin,const double*r,int k,double*z){   // G z = r, Cholesky
  std::vector<double> L((size_t)k*k,0.0);
  for(int j=0;j<k;j++){ double d=Gin[(size_t)j*k+j]; for(int q=0;q<j;q++) d-=L[(size_t)j*k+q]*L[(size_t)j*k+q];
    if(d<=1e-300)return false; d=sqrt(d); L[(size_t)j*k+j]=d;
    for(int i=j+1;i<k;i++){ double s=Gin[(size_t)i*k+j]; for(int q=0;q<j;q++) s-=L[(size_t)i*k+q]*L[(size_t)j*k+q]; L[(size_t)i*k+j]=s/d; } }
  std::vector<double> y(k);
  for(int i=0;i<k;i++){ double s=r[i]; for(int q=0;q<i;q++) s-=L[(size_t)i*k+q]*y[q]; y[i]=s/L[(size_t)i*k+i]; }
  for(int i=k-1;i>=0;i--){ double s=y[i]; for(int q=i+1;q<k;q++) s-=L[(size_t)q*k+i]*z[q]; z[i]=s/L[(size_t)i*k+i]; }
  return true; }
// Lawson-Hanson NNLS: min ||sum_q w_q a_q - b||_2 s.t. w>=0.  A is NODE-major: a_q = &A[q*m]
// (contiguous per candidate -> cache-friendly).  Incremental Cholesky of the passive-set Gram
// matrix (rebuilt only on the rare backtracking removal).  w has <= m nonzeros.
static void nnls(const std::vector<double>&A,const std::vector<double>&b,int m,int n,std::vector<double>&w){
  w.assign(n,0.0); std::vector<char> P(n,0); std::vector<double> r(b),z(n),zk(m),y(m);
  std::vector<int> idx; idx.reserve(m);
  std::vector<double> G((size_t)m*m,0.0), L((size_t)m*m,0.0), rhs(m,0.0); int k=0;
  auto dot=[&](const double*u,const double*v){ double s=0; for(int i=0;i<m;i++)s+=u[i]*v[i]; return s; };
  auto reform=[&](){ k=(int)idx.size();
    for(int a=0;a<k;a++){ const double*Aa=&A[(size_t)idx[a]*m]; rhs[a]=dot(Aa,b.data());
      for(int c=0;c<=a;c++) G[(size_t)a*m+c]=G[(size_t)c*m+a]=dot(Aa,&A[(size_t)idx[c]*m]); }
    for(int a=0;a<k;a++) G[(size_t)a*m+a]+=1e-12;
    for(int j=0;j<k;j++){ double d=G[(size_t)j*m+j]; for(int q=0;q<j;q++) d-=L[(size_t)j*m+q]*L[(size_t)j*m+q]; d=d>1e-300?sqrt(d):1e-150; L[(size_t)j*m+j]=d;
      for(int i=j+1;i<k;i++){ double s=G[(size_t)i*m+j]; for(int q=0;q<j;q++) s-=L[(size_t)i*m+q]*L[(size_t)j*m+q]; L[(size_t)i*m+j]=s/d; } } };
  auto addcol=[&](int jn){ const double*Aj=&A[(size_t)jn*m];        // append candidate jn to passive set
    for(int a=0;a<k;a++){ double g=dot(&A[(size_t)idx[a]*m],Aj); G[(size_t)a*m+k]=G[(size_t)k*m+a]=g; }
    G[(size_t)k*m+k]=dot(Aj,Aj)+1e-12;
    for(int a=0;a<k;a++){ double s=G[(size_t)k*m+a]; for(int q=0;q<a;q++) s-=L[(size_t)k*m+q]*L[(size_t)a*m+q]; L[(size_t)k*m+a]=s/L[(size_t)a*m+a]; }
    { double s=G[(size_t)k*m+k]; for(int q=0;q<k;q++) s-=L[(size_t)k*m+q]*L[(size_t)k*m+q]; L[(size_t)k*m+k]=s>1e-300?sqrt(s):1e-150; }
    rhs[k]=dot(Aj,b.data()); idx.push_back(jn); k++; };
  auto solveLS=[&](){ for(int i=0;i<k;i++){ double s=rhs[i]; for(int q=0;q<i;q++) s-=L[(size_t)i*m+q]*y[q]; y[i]=s/L[(size_t)i*m+i]; }
    for(int i=k-1;i>=0;i--){ double s=y[i]; for(int q=i+1;q<k;q++) s-=L[(size_t)q*m+i]*zk[q]; zk[i]=s/L[(size_t)i*m+i]; } };
  for(int outer=0;outer<m;outer++){
    int jm=-1; double gm=1e-9;
    for(int j=0;j<n;j++) if(!P[j]){ double g=dot(&A[(size_t)j*m],r.data()); if(g>gm){gm=g;jm=j;} }
    if(jm<0) break; P[jm]=1; addcol(jm);
    for(int inner=0;inner<3*n;inner++){
      solveLS(); std::fill(z.begin(),z.end(),0.0); double zmin=1e300;
      for(int a=0;a<k;a++){ z[idx[a]]=zk[a]; if(zk[a]<zmin)zmin=zk[a]; }
      if(zmin>1e-13){ for(int a=0;a<k;a++) w[idx[a]]=zk[a]; break; }
      double alpha=1e300; for(int a=0;a<k;a++){ int j=idx[a]; if(z[j]<=1e-13){ double t=w[j]/(w[j]-z[j]); if(t<alpha)alpha=t; } }
      for(int a=0;a<k;a++){ int j=idx[a]; w[j]+=alpha*(z[j]-w[j]); }
      std::vector<int> keep; keep.reserve(k); bool rem=false;
      for(int a=0;a<k;a++){ int j=idx[a]; if(w[j]<=1e-13){ P[j]=0; w[j]=0; rem=true; } else keep.push_back(j); }
      if(rem){ idx.swap(keep); reform(); }
    }
    for(int i=0;i<m;i++){ double s=b[i]; for(int a=0;a<k;a++) s-=w[idx[a]]*A[(size_t)idx[a]*m+i]; r[i]=s; }
  }
}
// compress a Saye VOLUME rule (points in the reference cube [0,1]^3) to a positive rule
// matching all tensor Q_{2p} moments; reuses a subset of the input node positions.
static void compressVol(const SayeNode*in,int nIn,int p,std::vector<SayeNode>&out){
  int K=2*p,n1=K+1,m=n1*n1*n1,n=nIn; out.clear();
  if(n<=m){ for(int q=0;q<n;q++) out.push_back(in[q]); return; }   // already minimal
  std::vector<double> A((size_t)n*m),b(m,0.0),Px(n1),Py(n1),Pz(n1);   // node-major
  for(int q=0;q<n;q++){ legShift(in[q].x[0],K,Px.data()); legShift(in[q].x[1],K,Py.data()); legShift(in[q].x[2],K,Pz.data());
    double*Aq=&A[(size_t)q*m];
    for(int i=0;i<n1;i++)for(int j=0;j<n1;j++)for(int k=0;k<n1;k++){ int rr=(i*n1+j)*n1+k; double v=Px[i]*Py[j]*Pz[k]; Aq[rr]=v; b[rr]+=(double)in[q].w*v; } }
  std::vector<double> w; nnls(A,b,m,n,w);
  for(int q=0;q<n;q++) if(w[q]>1e-13){ SayeNode s=in[q]; s.w=(real)w[q]; out.push_back(s); }
  if(out.empty()){ for(int q=0;q<nIn;q++) out.push_back(in[q]); }   // NNLS failed -> keep original
}
// Paper's discretization: candidates on a UNIFORM grid inside Omega ({phi<0} via the level-set fit),
// moments still taken from the Saye rule.  Returns ||Aw-b||_inf (moment mismatch: ~0 iff the grid can
// reproduce the exact moments, which the Saye-node candidate set does BY CONSTRUCTION).
static double compressVolUniform(const SayeNode*sayeIn,int nSaye,const PolyND&phi,int p,int gN,std::vector<SayeNode>&out){
  int K=2*p,n1=K+1,m=n1*n1*n1; out.clear();
  std::vector<double> b(m,0.0),Px(n1),Py(n1),Pz(n1);
  for(int q=0;q<nSaye;q++){ legShift(sayeIn[q].x[0],K,Px.data()); legShift(sayeIn[q].x[1],K,Py.data()); legShift(sayeIn[q].x[2],K,Pz.data());
    for(int i=0;i<n1;i++)for(int j=0;j<n1;j++)for(int k=0;k<n1;k++) b[(i*n1+j)*n1+k]+=(double)sayeIn[q].w*Px[i]*Py[j]*Pz[k]; }
  std::vector<SayeNode> cand;
  for(int a=0;a<gN;a++)for(int bb=0;bb<gN;bb++)for(int c=0;c<gN;c++){ real x[3]={(real)((a+0.5)/gN),(real)((bb+0.5)/gN),(real)((c+0.5)/gN)};
    if(phi.eval(x)<0){ SayeNode s{}; s.x[0]=x[0];s.x[1]=x[1];s.x[2]=x[2]; cand.push_back(s); } }
  int n=(int)cand.size();
  if(n<m){ for(int q=0;q<nSaye;q++) out.push_back(sayeIn[q]); return 0; }
  std::vector<double> A((size_t)n*m);
  for(int q=0;q<n;q++){ legShift(cand[q].x[0],K,Px.data()); legShift(cand[q].x[1],K,Py.data()); legShift(cand[q].x[2],K,Pz.data());
    double*Aq=&A[(size_t)q*m]; for(int i=0;i<n1;i++)for(int j=0;j<n1;j++)for(int k=0;k<n1;k++) Aq[(i*n1+j)*n1+k]=Px[i]*Py[j]*Pz[k]; }
  std::vector<double> w; nnls(A,b,m,n,w);
  double res=0; for(int r=0;r<m;r++){ double s=-b[r]; for(int q=0;q<n;q++) s+=A[(size_t)q*m+r]*w[q]; if(fabs(s)>res)res=fabs(s); }
  for(int q=0;q<n;q++) if(w[q]>1e-13){ SayeNode s=cand[q]; s.w=(real)w[q]; out.push_back(s); }
  if(out.empty()){ for(int q=0;q<nSaye;q++) out.push_back(sayeIn[q]); }
  return res;
}

// =====================================================================
//  GPU solve for the Cartesian Qp cut-cell operator (CUT_GPU=1).  Reuses runQp's
//  host setup verbatim -- the continuous nodal dofs live in flat device arrays,
//  and the operator is: interior cells share one reference matrix K_ref (dense
//  matvec), cut cells RE-QUADRATURE from their stored Saye rule (volume stiffness
//  via qpElemCoreSaye + Nitsche surface), ghost penalty via per-axis K_ghost.
//  Symmetric -> Jacobi-PCG.  All data device-resident: nodal u/phi, the Saye
//  integration rules (CSR pools), K_ref, K_ghost.  Stage: p2/p3 Cartesian.
// dense SPD inverse (Cholesky A=L L^T, then A^-1 by column solves); in-place, A->A^-1.
// returns false if not positive-definite (caller regularizes with a diagonal shift).
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
  QpBasis B; i32 nE,nCut,nGFQ,nNode,ndof,ndof3,mG;
  real h, mu, lam, gammaD;
  const i32 *eNode, *nMap; const char *nRot; real cph, sph;
  const i32 *intList, *cutElem;                    // interior cells; cut-cell -> element
  const SayeNode *volP, *surfP; const i32 *volOff, *surfOff; const char *surfDir;
  const i32 *gfM, *gfP, *gfD; const real *Kref, *Kg[3];
  i32 cyl; LevelSet ls; const i32 *eCijk, *eCut;   // cylindrical: analytic metric per element (ci,cj,ck), per-element cut index
  const real *volJ, *surfJ;                        // precomputed metric [Jinv(9),detJ(1)] at Saye vol/surf points (shared across p-levels)
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
__global__ void cutCellK(CutDev S,const real*xn,real*yn){ QpBasis B=S.B; i32 ndof=S.ndof,ndof3=S.ndof3;
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
  double r=(cj[0]+xr[0])*h+S.ls.org[0], s=(cj[1]+xr[1])*h+S.ls.org[1], z=(cj[2]+xr[2])*h+S.ls.org[2];
  double th=s/S.ls.rRef+(double)S.ls.thc((real)z), thp=(double)S.ls.thcSlope((real)z), ct=cos(th), st=sin(th);
  double J[3][3]={{ h*ct, h*(-r*st/S.ls.rRef), h*(-r*st*thp) },{ h*st, h*( r*ct/S.ls.rRef), h*( r*ct*thp) },{ 0,0,h }};
  detJ=inv3(J,Jinv); }
// CYLINDRICAL operator: block per element (ALL elements re-quadrature the analytic metric;
// interior = tensor-GLL, cut = Saye), + Nanson Nitsche on cut Dirichlet faces.
__global__ void cutCylK(CutDev S,const real*xn,real*yn){ QpBasis B=S.B; i32 ndof=S.ndof,ndof3=S.ndof3,n=B.n;
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
    if(cut){ i32 s0=S.surfOff[c], ns=S.surfOff[c+1]-s0; const SayeNode*sn=S.surfP+s0; double penC=(double)S.gammaD/S.h;
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
// p-prolongation (coarse-node -> fine-node), ADD: fn[3f+i] += sum_k W[w*f+k]*ec[3*col[w*f+k]+i]  (width w = coarse ndof)
__global__ void cutPProlongWK(const i32*col,const real*W,i32 w,const real*ec,real*fn,i32 nFine){
  for(i32 f=CUT_STRIDE;f<nFine;f+=gridDim.x*blockDim.x){ real a0=0,a1=0,a2=0;
    for(i32 k=0;k<w;k++){ real ww=W[(size_t)w*f+k]; i32 c=col[(size_t)w*f+k]; a0+=ww*ec[3*c];a1+=ww*ec[3*c+1];a2+=ww*ec[3*c+2]; }
    fn[3*f]+=a0; fn[3*f+1]+=a1; fn[3*f+2]+=a2; } }
// p-restriction (fine-node -> coarse-node) = P^T: ec[3*col+i] += W*fn[3f+i]
__global__ void cutPRestrictWK(const i32*col,const real*W,i32 w,const real*fn,real*ec,i32 nFine){
  for(i32 f=CUT_STRIDE;f<nFine;f+=gridDim.x*blockDim.x){ real f0=fn[3*f],f1=fn[3*f+1],f2=fn[3*f+2];
    for(i32 k=0;k<w;k++){ real ww=W[(size_t)w*f+k]; i32 c=col[(size_t)w*f+k]; atomicAdd(&ec[3*c],ww*f0);atomicAdd(&ec[3*c+1],ww*f1);atomicAdd(&ec[3*c+2],ww*f2); } } }

void CutFemSolver::runQp(void) {
  const i32 p = femOrder;
  QpBasis Bp; Bp.init(p);
  const i32 n = p+1, ndof = n*n*n, ndof3 = 3*ndof, mG = 2*ndof3;
  const real h = cellSize();
  const double mu = prob.mu, lam = prob.lam;
  const double gammaD_ = 100.0*(2*mu+lam)*p*p;   // Nitsche penalty (verified scaling)
  const double gammaG_ = (getenv("CUT_NOGHOST")?0.0:0.1)*(2*mu+lam);   // ghost-penalty coeff (CUT_NOGHOST=1 disables it, to expose small-cut conditioning)
  const bool cyl = (ls.coordMode == 1);
  const bool per = (periodic != 0);
  const double cph = cos((double)pitchAngle), sph = sin((double)pitchAngle);

  if (ls.coordMode != 0 && !cyl) {
    printf("ERROR: Qp path supports coordMode 0 (Cartesian) or 1 (cylindrical) only\n"); return; }
  if (p > QP_MAX) { printf("ERROR: femOrder %d exceeds QP_MAX=%d\n", p, QP_MAX); return; }

  printf("higher : Qp CutFEM (Saye cut quadrature), p=%d  %s%s  gammaD=%.4g gammaG=%.4g\n",
         p, cyl?"CYLINDRICAL isoparametric":"Cartesian", per?" + cyclic pitch tie":"",
         gammaD_, gammaG_);

  long t0 = qpNowUs();
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
  std::unordered_map<u64,i32> nodeId;
  nodeId.reserve((size_t)nE*ndof);
  std::vector<i32> eNodeQ((size_t)nE*ndof);
  std::vector<i32> nI, nJ, nK;                 // per-node lattice coords
  auto gcoord=[&](const QElem&E,i32 a,i32&I,i32&J,i32&K){ i32 i=a%n,j=(a/n)%n,k=a/(n*n);
    I=p*E.ci+i; J=p*E.cj+j; K=p*E.ck+k; };
  i32 nNodeQ=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 I,J,K; gcoord(elems[e],a,I,J,K); u64 key=qpKey(I,J,K);
    auto it=nodeId.find(key); i32 id;
    if (it==nodeId.end()){ id=nNodeQ++; nodeId[key]=id; nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    else id=it->second;
    eNodeQ[(size_t)e*ndof+a]=id;
  }
  const i32 Jmax = p*nThetaCells;              // theta slave column (periodic)
  std::vector<i32> realIdx(nNodeQ,-1);
  std::vector<char> rotFlag(nNodeQ,0);
  i32 nDofNode=0, nTie=0, nOrphan=0;
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (per && nJ[nd]==Jmax) continue;         // slave, resolved below
    realIdx[nd]=nDofNode++;
  }
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (realIdx[nd]>=0) continue;
    auto it=nodeId.find(qpKey(nI[nd],0,nK[nd]));   // master at J=0
    if (it==nodeId.end()){ realIdx[nd]=nDofNode++; nOrphan++; }   // no partner: own dof
    else { realIdx[nd]=realIdx[it->second]; rotFlag[nd]=1; nTie++; }
  }
  const i32 nDofQ=3*nDofNode;
  if (per) printf("cyclic : %d Qp nodes tied across the pitch (%d unmatched kept free), %d -> %d dofs\n",
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
  std::unordered_map<u64,i32> cellId; cellId.reserve((size_t)nE*2);
  for (i32 e=0;e<nE;e++) cellId[qpCellKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;
  struct GF{ i32 eM,eP,d; };
  std::vector<GF> gf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
      if (per && d==1 && nb[1]==nThetaCells) nb[1]=0;      // theta seam wraps
      auto it=cellId.find(qpCellKey(nb[0],nb[1],nb[2])); if (it==cellId.end()) continue;
      i32 ep=it->second; if (elems[e].cut||elems[ep].cut) gf.push_back({e,ep,d}); } }
  i32 nGFQ=(i32)gf.size();

  double Dl0[QP_MAX+1][QN_MAX], Dl1[QP_MAX+1][QN_MAX];
  { double Dp[QN_MAX][QN_MAX];
    for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Bp.D[i][a];
    for (i32 l=1;l<=p;l++){ for (i32 a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if (l<p){ double Nw[QN_MAX][QN_MAX];
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++){ double s=0; for(i32 m=0;m<n;m++) s+=Dp[i][m]*Bp.D[m][a]; Nw[i][a]=s; }
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Nw[i][a]; } } }

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
        tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
        tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
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
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
          tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
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
      // ---- preconditioner selection: CUT_PC = jac (default) | bjac | schwarz | pmg ----
      const char*pcEnv=getenv("CUT_PC"); std::string pcName=pcEnv?pcEnv:"jac";
      i32 pcMode=(pcName=="bjac")?1:(pcName=="schwarz")?2:(pcName=="pmg")?3:0;
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
      i32 nLev=1; real *d_diagN=nullptr,*d_cz=nullptr,*d_cp=nullptr,*d_cAp=nullptr,*d_cr=nullptr;
      CutDev Slev[QP_MAX+1]{}; QpBasis Blev[QP_MAX+1];
      i32 nNodeLev[QP_MAX+1]={0}, nNLev[QP_MAX+1]={0}, ndofLev[QP_MAX+1]={0}, nd3Lev[QP_MAX+1]={0}, mGLev[QP_MAX+1]={0}, wLev[QP_MAX+1]={0}, nDofNodeLev[QP_MAX+1]={0};
      real *d_diagNLev[QP_MAX+1]={0}, *d_xLev[QP_MAX+1]={0}, *d_rLev[QP_MAX+1]={0}, *d_tmpLev[QP_MAX+1]={0}, *d_resLev[QP_MAX+1]={0}, *d_dirLev[QP_MAX+1]={0}, *d_pwLev[QP_MAX+1]={0};
      real *d_mult3Lev[QP_MAX+1]={0}, *d_projLev[QP_MAX+1]={0}, *d_ptmpLev[QP_MAX+1]={0};   // per-level pitch-tie projection (dof buffer, mult, node scratch)
      i32 *d_pcolLev[QP_MAX+1]={0};
      double aLev[QP_MAX+1]={0}, bLev[QP_MAX+1]={0};
      std::vector<std::vector<i32>> eNodeLev(QP_MAX+1); std::vector<void*> pmgFree;
      i32 cDeg=3,cMaxit=100; double cTol=5e-2; { const char*e; if((e=getenv("CUT_CHEBDEG")))cDeg=atoi(e); if((e=getenv("CUT_CMAXIT")))cMaxit=atoi(e); if((e=getenv("CUT_CTOL")))cTol=atof(e); }
      if (pcMode==3) {
        nLev=p;   // p3 -> {p3,p2,p1}; p2 -> {p2,p1}
        auto buildLevel=[&](i32 pc,i32 L){ QpBasis Bc; Bc.init(pc); Blev[L]=Bc; i32 nc=pc+1,ndc=nc*nc*nc,nd3c=3*ndc,mGc=2*nd3c;
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
          { double Dp[QN_MAX][QN_MAX]; for(i32 i=0;i<nc;i++)for(i32 a=0;a<nc;a++)Dp[i][a]=Bc.D[i][a];
            for(i32 l=1;l<=pc;l++){ for(i32 a=0;a<nc;a++){Dl0c[l][a]=Dp[0][a];Dl1c[l][a]=Dp[nc-1][a];}
              if(l<pc){ double Nw[QN_MAX][QN_MAX]; for(i32 i=0;i<nc;i++)for(i32 a=0;a<nc;a++){double s=0;for(i32 m=0;m<nc;m++)s+=Dp[i][m]*Bc.D[m][a];Nw[i][a]=s;} for(i32 i=0;i<nc;i++)for(i32 a=0;a<nc;a++)Dp[i][a]=Nw[i][a]; } } }
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
          Sc.cyl=cyl?1:0; Sc.ls=ls; Sc.eCijk=d_eCijk; Sc.eCut=d_eCut; Sc.volJ=d_volJ; Sc.surfJ=d_surfJ; Slev[L]=Sc;
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
          S0.cyl=cyl?1:0; S0.ls=ls; S0.eCijk=d_eCijk; S0.eCut=d_eCut; S0.volJ=d_volJ; S0.surfJ=d_surfJ; Slev[0]=S0; }
        d_tmpLev[0]=alR(nN); d_resLev[0]=alR(nN); d_dirLev[0]=alR(nN);
        for(void*pp:{(void*)d_diagN,(void*)d_tmpLev[0],(void*)d_resLev[0],(void*)d_dirLev[0]}) pmgFree.push_back(pp);
        for(i32 L=1; L<nLev; L++) buildLevel(p-L,L);
        for(i32 L=0; L<nLev-1; L++){ i32 pf=p-L, ncf=pf+1, ndf=ncf*ncf*ncf, ndc=ndofLev[L+1]; QpBasis Bc=Blev[L+1]; i32 ncc=Bc.n;
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
      CutDev S; S.B=Bp; S.nE=nE; S.nCut=nCutQ; S.nGFQ=nGFQ; S.nNode=nNodeQ; S.ndof=ndof; S.ndof3=ndof3; S.mG=mG;
      S.h=h; S.mu=(real)mu; S.lam=(real)lam; S.gammaD=(real)gammaD_; S.cph=(real)cph; S.sph=(real)sph;
      S.eNode=d_eNode; S.nMap=d_nMap; S.nRot=d_nRot; S.intList=d_intList; S.cutElem=d_cutElem;
      S.volP=d_volP; S.surfP=d_surfP; S.volOff=d_volOff; S.surfOff=d_surfOff; S.surfDir=d_surfDir;
      S.gfM=d_gfM; S.gfP=d_gfP; S.gfD=d_gfD; S.Kref=d_Kref; S.Kg[0]=d_Kg0; S.Kg[1]=d_Kg1; S.Kg[2]=d_Kg2;
      S.cyl=cyl?1:0; S.ls=ls; S.eCijk=d_eCijk; S.eCut=d_eCut; S.volJ=d_volJ; S.surfJ=d_surfJ;
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
        if(cyl){ cutCylK<<<GBall,128,(size_t)2*nd3*sizeof(real)>>>(Slev[L],xn,yn); }
        else { if(nE-nCutQ)cutInteriorK<<<GBi,128,(size_t)nd3*sizeof(real)>>>(Slev[L],xn,yn);
          if(nCutQ)cutCellK<<<GBc,128,(size_t)2*nd3*sizeof(real)>>>(Slev[L],xn,yn); }
        if(nGFQ)cutGhostK<<<GBg,256,(size_t)mg*sizeof(real)>>>(Slev[L],xn,yn); cudaDeviceSynchronize(); };
      // pitch-tie subspace projection Pi = prolong * (1/mult) * restrict (identity when non-periodic)
      auto proj_L=[&](i32 L,const real*v,real*out){ i32 ndn=3*nDofNodeLev[L];
        cutSetK<<<GS,BS>>>(d_projLev[L],(real)0,ndn); cutRestrictK<<<GS,BS>>>(Slev[L],v,d_projLev[L]);
        cutJacK<<<GS,BS>>>(d_projLev[L],d_projLev[L],d_mult3Lev[L],ndn); cutProlongK<<<GS,BS>>>(Slev[L],d_projLev[L],out); cudaDeviceSynchronize(); };
      // tied operator A_tied = Pi A_n Pi (symmetric; kills untied theta-face modes) -- used by smoother/coarse when periodic
      auto applyTied_L=[&](i32 L,const real*xn,real*yn){ if(per){ proj_L(L,xn,d_ptmpLev[L]); applyN_L(L,d_ptmpLev[L],yn); proj_L(L,yn,yn); } else applyN_L(L,xn,yn); };
      auto ndotL=[&](const real*a,const real*b,i32 nnl)->double{ cudaMemset(d_acc,0,sizeof(double)); cutDotK<<<GS,BS>>>(a,b,nnl,d_acc); double hv; cudaMemcpy(&hv,d_acc,sizeof(double),cudaMemcpyDeviceToHost); return hv; };
      auto residualL=[&](i32 L,const real*x,const real*rhs,real*out){ i32 nnl=nNLev[L]; applyTied_L(L,x,d_tmpLev[L]);
        cutSetK<<<GS,BS>>>(out,(real)0,nnl); cutAxpyK<<<GS,BS>>>(out,rhs,(real)1,nnl); cutAxpyK<<<GS,BS>>>(out,d_tmpLev[L],(real)-1,nnl); cudaDeviceSynchronize(); };
      auto chebSmoothL=[&](i32 L,real*x,const real*rhs,i32 deg){ i32 nnl=nNLev[L]; double a=aLev[L],b=bLev[L],theta=(b+a)/2,delta=(b-a)/2,s1=theta/delta,rho=1.0/s1;
        residualL(L,x,rhs,d_resLev[L]); cutChebDirK<<<GS,BS>>>(d_dirLev[L],d_resLev[L],d_diagNLev[L],(real)0,(real)(1.0/theta),nnl); cutAxpyK<<<GS,BS>>>(x,d_dirLev[L],(real)1,nnl); cudaDeviceSynchronize();
        for(i32 k=1;k<deg;k++){ residualL(L,x,rhs,d_resLev[L]); double rn2=1.0/(2*s1-rho);
          cutChebDirK<<<GS,BS>>>(d_dirLev[L],d_resLev[L],d_diagNLev[L],(real)(rho*rn2),(real)(2*rn2/delta),nnl); cutAxpyK<<<GS,BS>>>(x,d_dirLev[L],(real)1,nnl); cudaDeviceSynchronize(); rho=rn2; }
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
      auto precond=[&](real*z,const real*r){
        if(pcMode==1){ cutBJacK<<<GS,BS>>>(z,r,d_Binv,nDofNode); return; }
        if(pcMode==2){ cutProlongK<<<GS,BS>>>(S,r,d_xn); cutSetK<<<GS,BS>>>(d_yn,(real)0,nN);
          if(nE-nCutQ) cutSchwarzK<<<GBi,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn,d_intList,nE-nCutQ,d_intInv,0);
          if(nCutQ) cutSchwarzK<<<GBc,128,(size_t)ndof3*sizeof(real)>>>(S,d_xn,d_yn,d_cutElem,nCutQ,d_cutInv,1);
          cutSetK<<<GS,BS>>>(z,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,z); return; }
        if(pcMode==3){ cutProlongK<<<GS,BS>>>(S,r,d_xn); vcycle(d_xn,d_yn); cutSetK<<<GS,BS>>>(z,(real)0,nR); cutRestrictK<<<GS,BS>>>(S,d_yn,z); return; }
        cutJacK<<<GS,BS>>>(z,r,d_diag,nR); };
      if (pcMode==3) {   // per-level Chebyshev interval via power iteration on D^-1 A (smoothed levels 0..nLev-2)
        for(i32 L=0; L<nLev-1; L++){ i32 nnl=nNLev[L]; real*v=d_resLev[L],*Av=d_tmpLev[L],*w=d_dirLev[L];
          cutSetK<<<GS,BS>>>(v,(real)1,nnl);
          for(i32 it=0;it<20;it++){ applyTied_L(L,v,Av); cutJacK<<<GS,BS>>>(w,Av,d_diagNLev[L],nnl); cudaDeviceSynchronize();
            double nrm=sqrt(ndotL(w,w,nnl)); if(nrm<=0)break; cutSetK<<<GS,BS>>>(v,(real)0,nnl); cutAxpyK<<<GS,BS>>>(v,w,(real)(1.0/nrm),nnl); cudaDeviceSynchronize(); }
          applyTied_L(L,v,Av); cutJacK<<<GS,BS>>>(w,Av,d_diagNLev[L],nnl); cudaDeviceSynchronize();
          double lam=ndotL(v,w,nnl)/ndotL(v,v,nnl); bLev[L]=1.1*lam; aLev[L]=bLev[L]/30.0; }
        printf("pmg    : cheb intervals"); for(i32 L=0;L<nLev-1;L++) printf(" L%d[%.2f,%.2f]",L,aLev[L],bLev[L]); printf("\n");
      }
      real *d_zold=alR(nR);   // flexible-CG (Polak-Ribiere) tolerates the nonlinear V-cycle PC
      t0=qpNowUs();           // time the CG loop only (excludes one-time setup / NNLS prune)
      cudaMemcpy(d_r,d_b,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);   // u=0 -> r=b
      precond(d_z,d_r); cudaMemcpy(d_pd,d_z,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice); cudaDeviceSynchronize();
      double bn=sqrt(dot(d_b,d_b)); if(bn==0)bn=1; double rz=dot(d_r,d_z); i32 it=0; double rn=0;
      for(; it<cgMaxIt; it++){ apply(d_pd,d_Ap); double pAp=dot(d_pd,d_Ap); if(!(pAp>0)){ printf("WARNING: Qp GPU-CG breakdown pAp=%.3e\n",pAp); break; }
        double al=rz/pAp; cutAxpyK<<<GS,BS>>>(d_u,d_pd,(real)al,nR); cutAxpyK<<<GS,BS>>>(d_r,d_Ap,(real)-al,nR); cudaDeviceSynchronize();
        rn=sqrt(dot(d_r,d_r)); if(rn<=cgTol*bn){ it++; break; }
        cudaMemcpy(d_zold,d_z,(size_t)nR*sizeof(real),cudaMemcpyDeviceToDevice);
        precond(d_z,d_r); cudaDeviceSynchronize();
        double rzn=dot(d_r,d_z), rzo=dot(d_r,d_zold), be=(rzn-rzo)/rz; rz=rzn;
        cutAxpyK<<<GS,BS>>>(d_pd,d_pd,(real)(be-1),nR); cutAxpyK<<<GS,BS>>>(d_pd,d_z,(real)1,nR); cudaDeviceSynchronize(); }
      cgIters=it; cgRes=rn/bn; cudaDeviceSynchronize(); memcpy(uv.data(),d_u,(size_t)nR*sizeof(real));
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
      if (!(pAp>0)){ printf("WARNING: Qp CG breakdown pAp=%.3e\n",pAp); break; }
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
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
          tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
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
    double ms=(qpNowUs()-t0)/1000.0; i32 nCutTrueQ=0; for (i32 e=0;e<nE;e++) if (elems[e].cut) nCutTrueQ++;
    printf("active : %d Qp elements (%d cut), %d ghost faces\n", nE, nCutTrueQ, nGFQ);
    printf("dofs   : %d nodes -> %d unknowns   h = %.6g   p = %d\n", nNodeQ, nDofQ, (double)h, p);
    printf("geom   : |Omega_h| = %.8g", volOmega);
    if (volExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", volExact, volOmega-volExact, 100.0*fabs(volOmega-volExact)/volExact);
    printf("\n         |Gamma_h| = %.8g", areaGamma);
    if (areaExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", areaExact, areaGamma-areaExact, 100.0*fabs(areaGamma-areaExact)/areaExact);
    printf("\nsolve  : CG %d iters, rel res %.2e   (%.0f ms)\n", cgIters, cgRes, ms);
    if ((prob.caseId==CASE_MMS || prob.caseId==CASE_MMS_CYL) && normL2>0)
      printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)\n", errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);

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
