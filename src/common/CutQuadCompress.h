#ifndef COMMON_CUTQUADCOMPRESS_H
#define COMMON_CUTQUADCOMPRESS_H

// ---------------------------------------------------------------------------
//  NNLS compression of a cut-cell quadrature rule.
//
//  A Saye rule is an *inefficient composite* rule -- it over-resolves badly
//  (measured: 216 pts/cut-cell on a sphere, 1478 on the cylindrical blade at
//  res16, 2485 at p3).  Potter, "Fast Construction of Efficient Cut Cell
//  Quadratures", prunes it to a minimal POSITIVE rule reproducing the SAME
//  polynomial moments, via Lawson-Hanson NNLS.  The result has at most m points,
//  m = dim of the target moment space, so the matvec gets 13-60x cheaper.
//
//  The TARGET SPACE is Potter's TOTAL degree P^d_{2p} (m = C(2p+3,3) = 35/84/165
//  for p = 2/3/4), NOT the tensor space (m = (2p+1)^3 = 125/343/729).  See the
//  long note at the moment-target switch in compressVol: total degree is 3.6-4.4x
//  smaller, preserves convergence order, and is invisible on real cut geometry --
//  at the price of ~30% higher L2 on SMOOTH geometry at fine h, where it
//  underintegrates the tensor-product stiffness integrand.
//
//  Solver-agnostic: it takes SayeNode rules in and gives SayeNode rules out, so
//  it serves any method that integrates over a cut cell -- continuous FEM, IGA,
//  or a cut-cell DG.  Compresses VOLUME rules only; a Saye SURFACE rule is
//  already at or below its own moment count and has nothing to prune.
//
//  COST: NNLS scales roughly as m^3, so shrinking m is the single biggest lever
//  on setup time -- the total-degree target cut the blade res64 prune 7.46 -> 0.62 s
//  (12x).  Two earlier fixes are also in here: reform() now COMPACTS the Gram
//  matrix in place instead of rebuilding it (it only loses a row/col on a removal),
//  and the candidate matrix is stored FACTORED (3*n1 Legendre factors per node,
//  expanded on the fly) because the gradient scan was DDR-bandwidth-bound.
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
inline NnlsStat g_nnlsStat;
// PF holds the FACTORED candidate matrix: 3*n1 Legendre factors per node
// (Px,Py,Pz), from which column q of A is the tensor product
// A_q[(i0*n1+i1)*n1+i2] = (Px[i0]*Py[i1])*Pz[i2].
//
// Storing the factors instead of the expanded n1^3 column is a 16x smaller
// working set at p3 (3*7 vs 343 doubles/node: 414 KB vs 6.8 MB per cell), which
// is the whole ballgame: CUT_NNLSSTAT showed the gradient scan re-streams the
// candidate matrix EVERY outer iteration -- 329 GB at p3, i.e. ~18 GB/s = DDR4
// saturation, and essentially 100% of the prune time.  Expanding on the fly
// costs 2 extra multiplies per element (free -- we were never compute-bound) and
// keeps the working set L2-resident.  Products and summation order are UNCHANGED,
// so results stay bit-identical.
// mi holds the moment index TRIPLES (i0,i1,i2), 3 ints per moment, in the order
// that defines the moment vector.  Passing the full tensor set reproduces the
// previous behaviour exactly; passing the total-degree subset (i0+i1+i2 <= 2p)
// is Potter's P^d_N target, which is 3.6-4.4x smaller in 3D.
// Core NNLS, parameterized on HOW a candidate column is accessed, so the two
// callers can store A differently: the FEM path keeps it FACTORED (Legendre
// factors, expanded on the fly -- bandwidth), while the cut-element DG path
// passes a DENSE A built from a general basis it evaluates itself.
//   Acc must provide:  void expand(int q,double*dst)   and
//                      double dotv (int q,const double*v)
template<class Acc>
static void nnlsCore(const Acc&acc,const std::vector<double>&b,int m,int n,
                     std::vector<double>&w, double gtol, int nOuter){
  w.assign(n,0.0); std::vector<char> P(n,0); std::vector<double> r(b),z(n),zk(m),y(m);
  std::vector<int> idx; idx.reserve(m);
  std::vector<int> keep, keepPos; keep.reserve(m); keepPos.reserve(m);   // hoisted out of the inner loop
  std::vector<double> G((size_t)m*m,0.0), L((size_t)m*m,0.0), rhs(m,0.0); int k=0;
  std::vector<double> col(m);          // scratch: one expanded column
  auto expand=[&](int q,double*dst){ acc.expand(q,dst); };
  auto dotAv =[&](int q,const double*v){ return acc.dotv(q,v); };
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
  auto addcol=[&](int jn){ expand(jn,col.data()); const double*Aj=col.data();   // append candidate jn
    for(int a=0;a<k;a++){ double g=dotAv(idx[a],Aj); G[(size_t)a*m+k]=G[(size_t)k*m+a]=g; }
    G[(size_t)k*m+k]=dot(Aj,Aj)+1e-12;
    for(int a=0;a<k;a++){ double s=G[(size_t)k*m+a]; for(int q=0;q<a;q++) s-=L[(size_t)k*m+q]*L[(size_t)a*m+q]; L[(size_t)k*m+a]=s/L[(size_t)a*m+a]; }
    { double s=G[(size_t)k*m+k]; for(int q=0;q<k;q++) s-=L[(size_t)k*m+q]*L[(size_t)k*m+q]; L[(size_t)k*m+k]=s>1e-300?sqrt(s):1e-150; }
    rhs[k]=dot(Aj,b.data()); idx.push_back(jn); k++; };
  auto solveLS=[&](){ for(int i=0;i<k;i++){ double s=rhs[i]; for(int q=0;q<i;q++) s-=L[(size_t)i*m+q]*y[q]; y[i]=s/L[(size_t)i*m+i]; }
    for(int i=k-1;i>=0;i--){ double s=y[i]; for(int q=i+1;q<k;q++) s-=L[(size_t)q*m+i]*zk[q]; zk[i]=s/L[(size_t)i*m+i]; } };
  for(int outer=0;outer<nOuter;outer++){
    int jm=-1; double gm=gtol;
    g_nnlsStat.outer++; g_nnlsStat.scanFlops += (long long)n*m;
    for(int j=0;j<n;j++) if(!P[j]){ double g=dotAv(j,r.data()); if(g>gm){gm=g;jm=j;} }
    if(jm<0) break;
    // THE PASSIVE SET is what must stay within the m x m scratch -- not the
    // iteration count.  Each outer pass adds ONE column and the inner
    // active-set loop can remove SEVERAL, so capping outer at m stops the
    // solve early with columns still to find: measured on a case-9 cut cell,
    // 21 columns admitted for a 35-dimensional moment space, terminating on
    // the cap with 2.0e-05 (wedge) / 3.2e-04 (quarter) of moment residual --
    // landing on moments that are EXACTLY ZERO in the true geometry (every
    // odd power of z, which a z-invariant cylinder cannot have).  The raw
    // Saye rule gives those as 1e-15, so an exact non-negative solution
    // provably exists and NNLS was simply stopped before finding it.
    if(k>=m) break;
    P[jm]=1; addcol(jm);
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
    // r = b - sum_a w_a A_{idx[a]}.  Column-outer so each column is expanded once;
    // per-i the subtractions still happen in ascending a, so rounding is unchanged.
    for(int i=0;i<m;i++) r[i]=b[i];
    for(int a=0;a<k;a++){ expand(idx[a],col.data()); double wa=w[idx[a]];
      for(int i=0;i<m;i++) r[i]-=wa*col[i]; }
  }
}

// --- accessor: DENSE A, row-major n x m (the cut-element DG path builds this
//     from its own basis, so there is no tensor factorization to exploit) ---
struct NnlsDenseAcc {
  const double*A; int m;
  void   expand(int q,double*dst)          const { const double*Aq=A+(size_t)q*m; for(int i=0;i<m;i++) dst[i]=Aq[i]; }
  double dotv  (int q,const double*v)      const { const double*Aq=A+(size_t)q*m; double s=0; for(int i=0;i<m;i++) s+=Aq[i]*v[i]; return s; }
};
// --- accessor: FACTORED A (FEM/IGA path).  Column q is the tensor product of
//     3*n1 Legendre factors, expanded on the fly over the moment list mi. ---
struct NnlsFactAcc {
  const double*PF; const int*mi; int m,n1;
  void expand(int q,double*dst) const {
    const double*F=PF+(size_t)q*3*n1,*Px=F,*Py=F+n1,*Pz=F+2*n1;
    for(int t=0;t<m;t++){ const int*I=mi+3*t; dst[t]=Px[I[0]]*Py[I[1]]*Pz[I[2]]; } }
  double dotv(int q,const double*v) const {
    const double*F=PF+(size_t)q*3*n1,*Px=F,*Py=F+n1,*Pz=F+2*n1; double s=0;
    for(int t=0;t<m;t++){ const int*I=mi+3*t; s+=Px[I[0]]*Py[I[1]]*Pz[I[2]]*v[t]; }
    return s; }
};
// Original DENSE entry point -- signature unchanged, so existing callers
// (src/common/CutElem.h) are untouched.
static void nnls(const std::vector<double>&A,const std::vector<double>&b,int m,int n,
                 std::vector<double>&w, double gtol=1e-9, int nOuter=-1){
  // nOuter<0 keeps the historical default (= m) so the FEM/IGA path is
  // bit-identical; a caller that needs the solve to CONVERGE passes more.
  // The passive set is bounded inside nnlsCore (k >= m breaks), so a larger
  // iteration budget cannot overflow the scratch.
  if (nOuter<0) nOuter=m;
  NnlsDenseAcc acc{A.data(),m};
  nnlsCore(acc,b,m,n,w,gtol,nOuter);
}
// FACTORED entry point (FEM/IGA compressVol).
static void nnlsFactored(const std::vector<double>&PF,const std::vector<double>&b,
                         const std::vector<int>&mi,int m,int n,int n1,
                         std::vector<double>&w, double gtol=1e-9, int nOuter=-1){
  if (nOuter<0 || nOuter>m) nOuter=m;
  NnlsFactAcc acc{PF.data(),mi.data(),m,n1};
  nnlsCore(acc,b,m,n,w,gtol,nOuter);
}

// compress a Saye VOLUME rule (points in the reference cube [0,1]^3) to a positive rule
// matching all tensor Q_{2p} moments; reuses a subset of the input node positions.
static void compressVol(const SayeNode*in,int nIn,int p,std::vector<SayeNode>&out){
  int K=2*p,n1=K+1,n=nIn; out.clear();
  // MOMENT TARGET -- DEFAULT IS POTTER'S TOTAL-DEGREE SPACE P^d_{2p} (i0+i1+i2 <= 2p).
  //  The alternative (CUT_PRUNETOTAL=0) is the full TENSOR set (partial degree
  //  <= 2p), which is what a tensor-product FEM stiffness integrand formally needs
  //  -- d(phi_a)/dx * d(phi_b)/dx has partial degrees (2p-2, 2p, 2p), so tensor is
  //  EXACT and total-degree UNDERINTEGRATES.  Potter Sec 2.2 explicitly sanctions
  //  systematic underintegration in FEM, and measurement says the exactness is not
  //  worth its price here:
  //    m: 125->35 (p2), 343->84 (p3), 729->165 (p4)  = 3.6-4.4x fewer moments, and
  //    the compressed rule floors at ~m points/cell, so the rule shrinks likewise.
  //    sphere p2 res16: 119 -> 28 pts/cell, prune 0.37 -> 0.04 s, solve 751 -> 480 ms
  //    sphere p3 res16: 183 -> 61 pts/cell, prune 3.56 -> 0.23 s, wall 12.4 -> 6.3 s
  //    blade  res64 p2: 110 -> 27 pts/cell, prune 7.46 -> 0.62 s, wall 18.1 -> 8.9 s
  //  CONVERGENCE ORDER IS PRESERVED (sphere p2 L2 orders 3.42/3.40 vs tensor's
  //  3.36/3.86, both above the design 3) and CG iteration counts are unchanged
  //  (629 vs 634), so conditioning is untouched.
  //  THE COST, stated honestly: on SMOOTH geometry at fine h the underintegration
  //  error becomes visible -- ~30% higher L2 (sphere p2 res32 +31%, p3 res16 +27%),
  //  because tensor's error keeps dropping at 3.86 while total's drops at 3.40.
  //  On the BLADE it is invisible (peak stress 293.0 -> 292.5 MPa = 0.16%, tip
  //  deflection 0.8788 -> 0.8772 mm = 0.18%, same peak location) because geometry
  //  error dominates there by orders of magnitude.
  //  => Set CUT_PRUNETOTAL=0 for convergence studies on smooth geometry where the
  //     last 30% of L2 matters; leave the default everywhere else.
  static const int useTotal = getenv("CUT_PRUNETOTAL") ? atoi(getenv("CUT_PRUNETOTAL")) : 1;
  std::vector<int> mi; mi.reserve((size_t)3*n1*n1*n1);
  for(int i=0;i<n1;i++)for(int j=0;j<n1;j++)for(int k=0;k<n1;k++){
    if(useTotal && (i+j+k)>K) continue;
    mi.push_back(i); mi.push_back(j); mi.push_back(k); }
  const int m=(int)(mi.size()/3);
  if(n<=m){ for(int q=0;q<n;q++) out.push_back(in[q]); return; }   // already minimal
  // FACTORED candidate matrix: 3*n1 Legendre factors per node, expanded on the fly
  // inside nnls (16x smaller working set at p3, L2-resident -- see the note there).
  std::vector<double> PF((size_t)n*3*n1),b(m,0.0),Px(n1),Py(n1),Pz(n1);
  for(int q=0;q<n;q++){ legShift(in[q].x[0],K,Px.data()); legShift(in[q].x[1],K,Py.data()); legShift(in[q].x[2],K,Pz.data());
    double*F=&PF[(size_t)q*3*n1];
    for(int i=0;i<n1;i++){ F[i]=Px[i]; F[n1+i]=Py[i]; F[2*n1+i]=Pz[i]; }
    for(int t=0;t<m;t++){ const int*I=&mi[3*t]; b[t]+=(double)in[q].w*Px[I[0]]*Py[I[1]]*Pz[I[2]]; } }
  std::vector<double> w; nnlsFactored(PF,b,mi,m,n,n1,w);
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
  std::vector<double> PF((size_t)n*3*n1);
  for(int q=0;q<n;q++){ legShift(cand[q].x[0],K,Px.data()); legShift(cand[q].x[1],K,Py.data()); legShift(cand[q].x[2],K,Pz.data());
    double*F=&PF[(size_t)q*3*n1];
    for(int i=0;i<n1;i++){ F[i]=Px[i]; F[n1+i]=Py[i]; F[2*n1+i]=Pz[i]; } }
  std::vector<int> mi; mi.reserve((size_t)3*m);
  for(int i=0;i<n1;i++)for(int j=0;j<n1;j++)for(int k=0;k<n1;k++){ mi.push_back(i); mi.push_back(j); mi.push_back(k); }
  std::vector<double> w; nnlsFactored(PF,b,mi,m,n,n1,w);
  double res=0;
  { std::vector<double> acc(m,0.0);
    for(int q=0;q<n;q++){ if(w[q]==0.0) continue; const double*F=&PF[(size_t)q*3*n1];
      for(int i=0;i<n1;i++)for(int j=0;j<n1;j++){ double pxy=F[i]*F[n1+j]; int base=(i*n1+j)*n1;
        for(int kk=0;kk<n1;kk++) acc[base+kk]+=pxy*F[2*n1+kk]*w[q]; } }
    for(int r2=0;r2<m;r2++){ double s=acc[r2]-b[r2]; if(fabs(s)>res)res=fabs(s); } }
  for(int q=0;q<n;q++) if(w[q]>1e-13){ SayeNode s=cand[q]; s.w=(real)w[q]; out.push_back(s); }
  if(out.empty()){ for(int q=0;q<nSaye;q++) out.push_back(sayeIn[q]); }
  return res;
}

#endif
