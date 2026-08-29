//
// State-redistribution gate.  Builds a real 3-D cut mesh (sphere carved out of
// a Cartesian box, Saye rules on the cut cells) and checks the three properties
// the papers rely on:
//
//   1. CONSERVATION       INT_Omega Su = INT_Omega u   for arbitrary u.
//   2. POLYNOMIAL EXACTNESS   Su == u when u is a global degree-N polynomial,
//                         so SRD costs no formal order.
//   3. CONTRACTIVITY      ||Su||_L2 <= ||u||_L2   (Taylor et al. Thm 2.1).
//                         This is the property the energy-stability proof rests
//                         on: an energy-stable scheme survives a contractive
//                         filter, so if this fails the whole argument fails.
//
// Measured by power iteration on the M-inner-product Rayleigh quotient
// (Su,Su)_M / (u,u)_M, whose square root is the operator norm.  The number
// reported must be <= 1.
//
// Also reports the SMALL-CELL RELIEF: the worst volume ratio before and after
// merging, which is what sets the explicit stable time step.
//
// build:  make dgsrd_test
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Util.cuh"
#include "Poly.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "StateRedistribution.h"
#include "LagrangeBasis.h"

static constexpr i32 ARENA = 1 << 20, OUTCAP = 1 << 16;

int main(int argc, char **argv) {
  const i32 p = DG_ORDER, n = p+1, ndof = n*n*n;
  const i32 NB = getenv("SRD_NB") ? atoi(getenv("SRD_NB")) : 8;   // cells/axis
  const double R  = getenv("SRD_R")  ? atof(getenv("SRD_R"))  : 0.699;
  const double L = 2.0, h = L/NB;                                  // box [-1,1]^3
  LagrangeBasis B; B.init(p);

  // ---- build the cut mesh: fluid is OUTSIDE the sphere --------------------
  std::vector<SrdElem> elem;
  std::vector<SayeNode> qpool;
  std::vector<i32> gid((size_t)NB*NB*NB, -1);
  std::vector<SayeNode> ab(ARENA), ob(OUTCAP);
  SayeArena ar; ar.buf=ab.data(); ar.cap=ARENA; ar.top=0;

  for (i32 kz=0;kz<NB;kz++) for (i32 ky=0;ky<NB;ky++) for (i32 kx=0;kx<NB;kx++) {
    double x0[3] = { -1.0+kx*h, -1.0+ky*h, -1.0+kz*h };
    std::vector<real> v(ndof);
    bool anyF=false, anyS=false;
    for (i32 c=0;c<ndof;c++) {
      i32 i=c%n, j=(c/n)%n, k=c/(n*n);
      double X=x0[0]+h*B.t[i], Y=x0[1]+h*B.t[j], Z=x0[2]+h*B.t[k];
      double f = R*R-(X*X+Y*Y+Z*Z);      // <0 OUTSIDE sphere = fluid
      v[c]=(real)f; if (f<0) anyF=true; else anyS=true;
    }
    if (!anyF) continue;                                   // fully solid
    SrdElem E{}; for (i32 f=0;f<6;f++) E.nbr[f]=-1;
    E.x0[0]=x0[0]; E.x0[1]=x0[1]; E.x0[2]=x0[2]; E.h[0]=E.h[1]=E.h[2]=h;
    E.qOff=(i32)qpool.size();
    if (!anyS) {                                           // uncut: tensor GLL
      for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
        SayeNode s{}; s.x[0]=B.t[i]; s.x[1]=B.t[j]; s.x[2]=B.t[k];
        s.w=B.wq[i]*B.wq[j]*B.wq[k]; qpool.push_back(s); }
    } else {
      PolyND phi = fitPoly3(p, v.data());
      SayeSet out; out.p=ob.data(); out.n=0; out.cap=OUTCAP; out.ovf=false;
      SayeCfg cfg=SayeCfg::def(); cfg.ng=10;
      sayeVolume(phi,&out,&ar,cfg);
      std::vector<SayeNode> cmp;
      compressVol(out.p,out.n,p,cmp);                      // positive NNLS rule
      for (const SayeNode &s : cmp) qpool.push_back(s);
    }
    E.qN=(i32)qpool.size()-E.qOff;
    E.vol=0; for (i32 q=E.qOff;q<E.qOff+E.qN;q++) E.vol += (double)qpool[q].w*E.hv();
    gid[(size_t)(kz*NB+ky)*NB+kx]=(i32)elem.size();
    elem.push_back(E);
  }
  // face neighbours
  for (i32 kz=0;kz<NB;kz++) for (i32 ky=0;ky<NB;ky++) for (i32 kx=0;kx<NB;kx++) {
    i32 g=gid[(size_t)(kz*NB+ky)*NB+kx]; if (g<0) continue;
    const i32 off[6][3]={{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    for (i32 f=0;f<6;f++){
      i32 a=kx+off[f][0], b=ky+off[f][1], c=kz+off[f][2];
      if (a<0||a>=NB||b<0||b>=NB||c<0||c>=NB) continue;
      elem[g].nbr[f]=gid[(size_t)(c*NB+b)*NB+a];
    }
  }

  const i32 nE=(i32)elem.size();
  const double vFull=h*h*h;
  double vmin=1e300; for (const SrdElem &E : elem) if (E.vol<vmin) vmin=E.vol;

  // ---- build SRD ---------------------------------------------------------
  SrdOperator S; S.volFrac = getenv("SRD_FRAC") ? atof(getenv("SRD_FRAC")) : 0.5;
  S.buildNeighborhoods(elem);
  S.buildReverse();
  S.factor(elem, qpool.data(), p);

  i32 nSmall=0, maxM=0; double vmerged=1e300;
  for (i32 k=0;k<nE;k++) {
    if (!S.trivial[k]) nSmall++;
    if ((i32)S.M[k].size()>maxM) maxM=(i32)S.M[k].size();
    double v=0; for (i32 j : S.M[k]) v+=elem[j].vol;
    if (v<vmerged) vmerged=v;
  }
  printf("mesh   : %d^3 background, sphere R=%.3f -> %d fluid elements, %d cut-small\n",
         NB, R, nE, nSmall);
  printf("volume : smallest element %.4e = 1/%.0f of a full cell\n", vmin, vFull/vmin);
  printf("       : smallest MERGE NEIGHBOURHOOD %.4e = 1/%.1f of a full cell\n",
         vmerged, vFull/vmerged);
  printf("       : merged/unmerged volume ratio %.1fx -- the neighbourhood is back at\n"
         "         background scale, which is the property SRD is claimed to restore\n",
         vmerged/vmin);
  printf("       : largest neighbourhood %d elements, basis %d monomials (degree %d)\n",
         maxM, S.nb, p);

  // ---- helpers -----------------------------------------------------------
  const size_t NDOF=(size_t)nE*ndof;
  auto integral=[&](const double *u){                      // INT_Omega u
    double s=0; std::vector<real> vb(ndof);
    for (i32 e=0;e<nE;e++){ double hv=elem[e].hv();
      for (i32 q=elem[e].qOff;q<elem[e].qOff+elem[e].qN;q++){
        real xr[3]={qpool[q].x[0],qpool[q].x[1],qpool[q].x[2]};
        B.allVal(xr,vb.data()); double uq=0;
        for (i32 a=0;a<ndof;a++) uq+=u[(size_t)e*ndof+a]*(double)vb[a];
        s += (double)qpool[q].w*hv*uq; } }
    return s; };
  auto l2sq=[&](const double *u){                          // ||u||^2_L2(Omega)
    double s=0; std::vector<real> vb(ndof);
    for (i32 e=0;e<nE;e++){ double hv=elem[e].hv();
      for (i32 q=elem[e].qOff;q<elem[e].qOff+elem[e].qN;q++){
        real xr[3]={qpool[q].x[0],qpool[q].x[1],qpool[q].x[2]};
        B.allVal(xr,vb.data()); double uq=0;
        for (i32 a=0;a<ndof;a++) uq+=u[(size_t)e*ndof+a]*(double)vb[a];
        s += (double)qpool[q].w*hv*uq*uq; } }
    return s; };
  auto nodeX=[&](i32 e,i32 a,double X[3]){
    i32 i=a%n,j=(a/n)%n,k=a/(n*n);
    X[0]=elem[e].x0[0]+elem[e].h[0]*(double)B.t[i];
    X[1]=elem[e].x0[1]+elem[e].h[1]*(double)B.t[j];
    X[2]=elem[e].x0[2]+elem[e].h[2]*(double)B.t[k]; };

  std::vector<double> u(NDOF), su(NDOF);

  // ---- 1. conservation ---------------------------------------------------
  unsigned seed=12345;
  auto rnd=[&](){ seed=seed*1664525u+1013904223u; return (double)(seed>>8)/16777216.0-0.5; };
  for (size_t t=0;t<NDOF;t++) u[t]=rnd();
  srdApply(S, elem, qpool.data(), B, u.data(), su.data(), 1);
  double I0=integral(u.data()), I1=integral(su.data());
  printf("\n1 conservation      INT u %.12e -> %.12e   rel %.3e  %s\n",
         I0, I1, fabs(I1-I0)/fabs(I0), fabs(I1-I0)/fabs(I0)<1e-10?"ok":"FAIL");

  // ---- 2. polynomial exactness ------------------------------------------
  double worstP=0;
  for (i32 deg=0; deg<=p; deg++) {
    for (size_t t=0;t<NDOF;t++) {
      i32 e=(i32)(t/ndof), a=(i32)(t%ndof); double X[3]; nodeX(e,a,X);
      u[t]=pow(X[0]+0.3,deg)+pow(X[1]-0.2,deg)+pow(X[2]+0.1,deg);
    }
    srdApply(S, elem, qpool.data(), B, u.data(), su.data(), 1);
    double d=0,r=0; for (size_t t=0;t<NDOF;t++){ d+=(su[t]-u[t])*(su[t]-u[t]); r+=u[t]*u[t]; }
    double rel=sqrt(d/r); if (rel>worstP) worstP=rel;
    printf("2 poly degree %d     ||Su-u||/||u|| = %.3e  %s\n", deg, rel, rel<1e-9?"ok":"FAIL");
  }

  // ---- 3. SELF-ADJOINTNESS ----------------------------------------------
  // (Su,v) == (u,Sv) in L2(Omega).  This is not decoration: it follows from
  // Pi_j being self-adjoint in the 1/|C_k|-weighted neighbourhood inner product
  // (their Eq. 24/26), and it is precisely that weighting the implementation
  // could get wrong.  It also makes the power iteration below measure the true
  // operator norm rather than merely the spectral radius.
  {
    std::vector<double> v(NDOF), sv(NDOF);
    for (size_t t=0;t<NDOF;t++){ u[t]=rnd(); v[t]=rnd(); }
    srdApply(S, elem, qpool.data(), B, u.data(), su.data(), 1);
    srdApply(S, elem, qpool.data(), B, v.data(), sv.data(), 1);
    auto ip=[&](const double *a,const double *b){
      double s=0; std::vector<real> vb(ndof);
      for (i32 e=0;e<nE;e++){ double hv=elem[e].hv();
        for (i32 q=elem[e].qOff;q<elem[e].qOff+elem[e].qN;q++){
          real xr[3]={qpool[q].x[0],qpool[q].x[1],qpool[q].x[2]};
          B.allVal(xr,vb.data()); double aq=0,bq=0;
          for (i32 c=0;c<ndof;c++){ aq+=a[(size_t)e*ndof+c]*(double)vb[c];
                                    bq+=b[(size_t)e*ndof+c]*(double)vb[c]; }
          s += (double)qpool[q].w*hv*aq*bq; } }
      return s; };
    double lhs=ip(su.data(),v.data()), rhs2=ip(u.data(),sv.data());
    double sc=sqrt(l2sq(u.data())*l2sq(v.data()));
    double rel=fabs(lhs-rhs2)/sc;
    printf("3 self-adjoint      (Su,v)-(u,Sv) = %.3e (rel)  %s\n", rel,
           rel<1e-10?"ok  -> power iteration gives the true norm":"FAIL");
    if (rel>=1e-10) printf("   !! the 1/|C_j| neighbourhood weighting is wrong\n");
  }

  // ---- 4. contractivity --------------------------------------------------
  // power iteration on the M-Rayleigh quotient: ||S||^2 = max (Su,Su)_M/(u,u)_M
  for (size_t t=0;t<NDOF;t++) u[t]=rnd();
  double lam=0;
  std::vector<double> w(NDOF);
  const i32 nPow = getenv("SRD_POW") ? atoi(getenv("SRD_POW")) : 60;
  for (i32 it=0; it<nPow; it++) {
    double nu=sqrt(l2sq(u.data())); if (nu<=0) break;
    for (size_t t=0;t<NDOF;t++) u[t]/=nu;
    srdApply(S, elem, qpool.data(), B, u.data(), su.data(), 1);
    lam = l2sq(su.data());                       // = ||Su||^2 with ||u||=1
    // S is self-adjoint (checked above), so plain power iteration converges to
    // the largest |eigenvalue| = the operator norm.  Expect EXACTLY 1: degree-N
    // polynomials are fixed points (test 2), so 1 is attained; the theorem says
    // nothing exceeds it.  The meaningful check is that it is not > 1.
    for (size_t t=0;t<NDOF;t++) u[t]=su[t];
    if (l2sq(u.data())<=0) break;
  }
  double nrm=sqrt(lam);
  printf("4 contractivity     ||S||_L2 = %.10f   %s\n", nrm,
         nrm<=1.0+1e-9?"ok  (energy stability preserved)":"FAIL");

  bool pass = fabs(I1-I0)/fabs(I0)<1e-10 && worstP<1e-9 && nrm<=1.0+1e-9;
  printf("\n%s\n", pass?"SRD PASS":"SRD FAIL");
  return pass?0:1;
}
