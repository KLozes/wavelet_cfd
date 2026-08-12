#ifndef COMMON_CUTQUADCOMPRESS_H
#define COMMON_CUTQUADCOMPRESS_H

// ---------------------------------------------------------------------------
//  NNLS compression of a cut-cell quadrature rule.
//
//  A Saye rule is an *inefficient composite* rule -- it over-resolves badly
//  (measured: 216 pts/cut-cell on a sphere, 1478 on the cylindrical blade at
//  res16, 2485 at p3).  Potter, "Fast Construction of Efficient Cut Cell
//  Quadratures", prunes it to a minimal POSITIVE rule reproducing the SAME
//  polynomial moments, via Lawson-Hanson NNLS.  The result has at most
//  m = (2p+1)^3 points and is EXACT for polynomial integrands, so the stiffness
//  operator is unchanged while the matvec gets 13-22x cheaper.
//
//  Solver-agnostic: it takes SayeNode rules in and gives SayeNode rules out, so
//  it serves any method that integrates over a cut cell -- continuous FEM, IGA,
//  or a cut-cell DG.  Compresses VOLUME rules only; a Saye SURFACE rule is
//  already at or below its own moment count and has nothing to prune.
//
//  NOTE the cost: NNLS scales roughly as m^3 = (2p+1)^9, which at p3-p4 makes
//  this the dominant setup cost (95 s on the blade at p3 res32).  reform()'s
//  O(k^2 m) Gram rebuild is the suspect.
// ---------------------------------------------------------------------------

#include <cmath>
#include <vector>

#include "Poly.h"
#include "SayeQuad.h"

// ---- quadrature compression (Potter, "Fast Construction of Efficient Cut Cell
//      Quadratures"): prune a dense Saye rule to a minimal POSITIVE rule that
//      reproduces the same polynomial moments, via Lawson-Hanson NNLS. ---------
static inline void legShift(double x,int K,double*P){ double t=2*x-1; P[0]=1; if(K>=1)P[1]=t;
  for(int k=1;k<K;k++) P[k+1]=((2*k+1)*t*P[k]-k*P[k-1])/(k+1); }
// matrix (rebuilt only on the rare backtracking removal).  w has <= m nonzeros.
// gtol: gradient threshold for admitting a column (ABSOLUTE).  The default
// 1e-9 is what the FEM compression path has always used and is kept so that
// path is bit-identical; the cut-element fit passes a value scaled to its own
// data.  nOuter caps the passive set; default m reproduces the old behaviour.
// instrumentation (CUT_NNLSSTAT=1): where does the prune time actually go?
struct NnlsStat { long long reformCalls=0, reformFlops=0, scanFlops=0, outer=0; };
extern NnlsStat g_nnlsStat;
#pragma omp threadprivate(g_nnlsStat)
NnlsStat g_nnlsStat;
static void nnls(const std::vector<double>&A,const std::vector<double>&b,int m,int n,std::vector<double>&w,
                 double gtol=1e-9, int nOuter=-1){
  if (nOuter<0 || nOuter>m) nOuter=m;   // G/L are m x m: the passive set
                                       // cannot exceed m columns
  w.assign(n,0.0); std::vector<char> P(n,0); std::vector<double> r(b),z(n),zk(m),y(m);
  std::vector<int> idx; idx.reserve(m);
  std::vector<int> keep, keepPos; keep.reserve(m); keepPos.reserve(m);   // hoisted out of the inner loop
  std::vector<double> G((size_t)m*m,0.0), L((size_t)m*m,0.0), rhs(m,0.0); int k=0;
  auto dot=[&](const double*u,const double*v){ double s=0; for(int i=0;i<m;i++)s+=u[i]*v[i]; return s; };
  // Cholesky of the leading k x k of G (G assumed already correct).
  auto refactor=[&](){
    for(int j=0;j<k;j++){ double d=G[(size_t)j*m+j]; for(int q=0;q<j;q++) d-=L[(size_t)j*m+q]*L[(size_t)j*m+q]; d=d>1e-300?sqrt(d):1e-150; L[(size_t)j*m+j]=d;
      for(int i=j+1;i<k;i++){ double s=G[(size_t)i*m+j]; for(int q=0;q<j;q++) s-=L[(size_t)i*m+q]*L[(size_t)j*m+q]; L[(size_t)i*m+j]=s/d; } } };
  // Drop columns from the passive set.  G's ENTRIES are unchanged by a removal --
  // only their POSITIONS shift -- so compact in place (k^2/2 copies) instead of
  // recomputing them as k^2 dot products of length m.  That rebuild was measured
  // at 76% of the p3 prune's flops (reform/scan 3.1x at p3, 0.5x at p2), and it
  // is the (2p+1)^9 scaling: k^2*m grows as m^3 while the useful work does not.
  // kp = kept POSITIONS in the old passive set, ascending; the copy is safe in
  // place because kp[a] >= a, so every source (kp[a],kp[c]) is at or after its
  // destination (a,c) in both index directions.
  auto shrink=[&](const std::vector<int>&kp){
    int kn=(int)kp.size();
    g_nnlsStat.reformCalls++;
    g_nnlsStat.reformFlops += (long long)kn*kn/2 + (long long)kn*kn*kn/3;
    for(int a=0;a<kn;a++){ int oa=kp[a]; rhs[a]=rhs[oa];
      for(int c=0;c<=a;c++){ int oc=kp[c]; double v=G[(size_t)oa*m+oc];
        G[(size_t)a*m+c]=v; G[(size_t)c*m+a]=v; } }
    k=kn; refactor(); };
  auto addcol=[&](int jn){ const double*Aj=&A[(size_t)jn*m];        // append candidate jn to passive set
    for(int a=0;a<k;a++){ double g=dot(&A[(size_t)idx[a]*m],Aj); G[(size_t)a*m+k]=G[(size_t)k*m+a]=g; }
    G[(size_t)k*m+k]=dot(Aj,Aj)+1e-12;
    for(int a=0;a<k;a++){ double s=G[(size_t)k*m+a]; for(int q=0;q<a;q++) s-=L[(size_t)k*m+q]*L[(size_t)a*m+q]; L[(size_t)k*m+a]=s/L[(size_t)a*m+a]; }
    { double s=G[(size_t)k*m+k]; for(int q=0;q<k;q++) s-=L[(size_t)k*m+q]*L[(size_t)k*m+q]; L[(size_t)k*m+k]=s>1e-300?sqrt(s):1e-150; }
    rhs[k]=dot(Aj,b.data()); idx.push_back(jn); k++; };
  auto solveLS=[&](){ for(int i=0;i<k;i++){ double s=rhs[i]; for(int q=0;q<i;q++) s-=L[(size_t)i*m+q]*y[q]; y[i]=s/L[(size_t)i*m+i]; }
    for(int i=k-1;i>=0;i--){ double s=y[i]; for(int q=i+1;q<k;q++) s-=L[(size_t)q*m+i]*zk[q]; zk[i]=s/L[(size_t)i*m+i]; } };
  for(int outer=0;outer<nOuter;outer++){
    int jm=-1; double gm=gtol;
    g_nnlsStat.outer++; g_nnlsStat.scanFlops += (long long)n*m;
    for(int j=0;j<n;j++) if(!P[j]){ double g=dot(&A[(size_t)j*m],r.data()); if(g>gm){gm=g;jm=j;} }
    if(jm<0) break; P[jm]=1; addcol(jm);
    for(int inner=0;inner<3*n;inner++){
      solveLS(); std::fill(z.begin(),z.end(),0.0); double zmin=1e300;
      for(int a=0;a<k;a++){ z[idx[a]]=zk[a]; if(zk[a]<zmin)zmin=zk[a]; }
      if(zmin>1e-13){ for(int a=0;a<k;a++) w[idx[a]]=zk[a]; break; }
      double alpha=1e300; for(int a=0;a<k;a++){ int j=idx[a]; if(z[j]<=1e-13){ double t=w[j]/(w[j]-z[j]); if(t<alpha)alpha=t; } }
      for(int a=0;a<k;a++){ int j=idx[a]; w[j]+=alpha*(z[j]-w[j]); }
      keep.clear(); keepPos.clear(); bool rem=false;
      for(int a=0;a<k;a++){ int j=idx[a]; if(w[j]<=1e-13){ P[j]=0; w[j]=0; rem=true; }
                            else { keep.push_back(j); keepPos.push_back(a); } }
      if(rem){ idx.swap(keep); shrink(keepPos); }
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

#endif
