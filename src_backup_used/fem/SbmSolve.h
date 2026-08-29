#ifndef FEM_SBMSOLVE_H
#define FEM_SBMSOLVE_H
//
// M1 gate: standalone host SBM MMS convergence test on a structured SURROGATE
// mesh -- the decisive check that the shifted-boundary path reaches O(h^{p+1})
// with NO cut quadrature, before wiring into the octree solver.  Mirrors
// QpMms.cu (same MMS/sphere/material, so the results A/B directly), but:
//   * the domain is the union of FULL cells whose CENTRE is inside (the surrogate
//     Omega~); no cell is ever cut.  Gamma~ then sits ~h/2 from Gamma (small |d|).
//   * BCs live on the SURROGATE boundary Gamma~ (mesh faces between a surrogate
//     and a non-surrogate cell), imposed by SHIFTED Nitsche: the trace is
//     Taylor-shifted (SbmShift.h) to x_true = x~ + d, with d,nu from the level set.
//
// Weak form = GSBM Eq. (35) (Colomes, Modderman, Scovazzi, "The generalized
// shifted boundary method", CMAME 452 (2026) 118748):
//     (eps(v), sigma(u))_Omega~
//   - (v, sigma(u).n~)_Gamma~                          <- consistency, PLAIN v
//   - (sigma(v).n~, S_d u - u_D)_Gamma~                 <- adjoint, SHIFTED u
//   + (beta1/h S_d v, S_d u - u_D)_Gamma~               <- penalty, SHIFTED both
//   + sum_{l=1..r} beta2 (2mu+lam) h^{2l-1} ([[D_n^l v]],[[D_n^l u]])_ghost
// with beta1 = chi (r+1)^2 (2mu+lam), chi=20;  beta2 = kappa = 0.5.
// The GHOST PENALTY is ESSENTIAL at p>=2: the shift S_d activates derivatives up
// to order p, and without control of their jumps the operator is INDEFINITE
// (CG pAp<0, BiCGStab diverges, restarted GMRES stagnates).  Verified:
// p=1 O(h^2), p=2 O(h^3).
//
// NEU=1 (opt-in): GAP-SBM traction (Neumann) BC on the upper hemisphere,
// Dirichlet elsewhere.  With the gap included the traction is a PURE NATURAL BC:
//     int_Omega~ sigma:eps(v) + int_gap sigma:eps(v)
//       = int_Omega~ f.v + int_gap f.v + int_Gamma~ (n~.nu) t(x_true).v(x_true)
// Verified O(h^2) at p=1.  NOGAP=1 ablates the gap; JAC=1 uses the exact
// parallel-surface (curvature) Jacobians instead of the linearized transfers.
// SCALING (the bug that made the first attempt diverge ~1/h): the traction
// carries NO physical gradient, so its weight is h^2*w_face -- NOT the
// hw = h*w_face used by the Nitsche terms, which carry ONE gradient (1/h).
//   weight cheat-sheet: volume load h^3*w | stiffness (2 grads) h*w |
//   Nitsche flux/adjoint (1 grad) h*w | Nitsche penalty (beta/h) h*w |
//   SURFACE TRACTION (0 grads) h^2*w | ghost penalty h*w
// Measured Neumann rates (p=2, N=8/16/24): the LINEARIZED geometric transfers
// cap the rate -- no gap 2.47 then 1.13; gap only 3.38 then 1.75; but
// gap + EXACT Jacobians (JAC=1) holds 3.48 then 3.33 = O(h^{p+1}), optimal, and
// 2.75x better error.  BOTH ingredients are needed at p>=2: the gap fixes the
// constant, the curvature Jacobians fix the RATE.  p=1 is O(h^2) either way.
// For general geometry the factor is the parallel-surface Jacobian
// (1+k1*s)(1+k2*s); JAC=2/3 obtain it from the FITTED level set (fitSdfCell:
// sample phi at the (q+1)^3 GLL nodes, fitPoly3 -> phi, grad, HESSIAN in closed
// form), so only POINT SAMPLES of phi are needed -- no oracle Hessian.
// The fitted QUADRATIC (JAC=2, degree p) matches the exact analytic curvature to
// 0.1%: 3.47 then 3.33, identical to JAC=1.  A cubic fit is no better.
// Default (NEU unset) is the pure-Dirichlet path, which IS optimal at all p.
// p=3 is NOT verified: Jacobi-GMRES cannot solve it (stagnates ~3e-5, L2 still
// tolerance-dependent) -- a PRECONDITIONER limit, not a formulation one; the
// Saye path has the same known p=3 behaviour and wants block-Jacobi.
//
//   build: nvcc -O2 -DUSE_DOUBLE -Xcompiler -fopenmp -I src/common -I src/fem src/fem/SbmMms.cu -o sbm_mms
//   run:   ./sbm_mms <p> <N1> <N2> ...  env: CHI KAP TOL GM DBG NEU NOGAP JAC
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "IgaElem.h"      // qpElemUncut (bulk), LagrangeBasis
#include "SbmShift.h"    // sbmShiftAll (Taylor shift), sbmDerivMatrix

static double MU=0.8, LAM=1.2, KK=M_PI;
static double SPH_R=0.75, SPH_C[3]={0.0123,-0.0071,0.0055};

// Pluggable geometry: default = the analytic sphere (the verified MMS case);
// set g_sdfFn to run the SAME solver on any other level set (e.g. the blade).
typedef double (*SbmSdfFn)(double,double,double);
static SbmSdfFn g_sdfFn = nullptr;
static double sdfSphere(double x,double y,double z){
  double dx=x-SPH_C[0],dy=y-SPH_C[1],dz=z-SPH_C[2]; return sqrt(dx*dx+dy*dy+dz*dz)-SPH_R; }
static double sdf(double x,double y,double z){
  return g_sdfFn ? g_sdfFn(x,y,z) : sdfSphere(x,y,z); }
static void uex(double x,double y,double z,double u[3]){
  u[0]=sin(KK*y)*sin(KK*z); u[1]=sin(KK*z)*sin(KK*x); u[2]=sin(KK*x)*sin(KK*y); }
static void fbody(double x,double y,double z,double f[3]){
  double u[3]; uex(x,y,z,u); for(int i=0;i<3;i++) f[i]=2*MU*KK*KK*u[i]; }
// exact stress sigma(u_exact) = mu(grad u + grad u^T)  (div u = 0 so the lam term drops)
static void exactStress(double x,double y,double z,double s[3][3]){
  double k=KK, sx=sin(k*x),sy=sin(k*y),sz=sin(k*z), cx=cos(k*x),cy=cos(k*y),cz=cos(k*z);
  s[0][0]=s[1][1]=s[2][2]=0.0;
  s[0][1]=s[1][0]=MU*k*sz*(cy+cx);
  s[0][2]=s[2][0]=MU*k*sy*(cz+cx);
  s[1][2]=s[2][1]=MU*k*sx*(cz+cy); }
// ---- fitted ("isoparametric") level set: sample the SDF at the (q+1)^3 GLL
// nodes of a cell and fit a tensor polynomial of degree q (same trick the Saye
// path uses).  phi, grad phi AND the HESSIAN then come from that polynomial in
// closed form -- no extra oracle capability beyond point samples of phi.
static PolyND fitSdfCell(double x0,double y0,double z0,double h,int q){
  real tq[PNC]; gllNodes(q,tq);
  real v[PNC*PNC*PNC]; int nq=q+1;
  for(int k=0;k<nq;k++)for(int j=0;j<nq;j++)for(int i=0;i<nq;i++)
    v[i+nq*(j+nq*k)] = (real)sdf(x0+tq[i]*h, y0+tq[j]*h, z0+tq[k]*h);
  return fitPoly3(q,v);
}
// d^(mx+my+mz) P / dx^mx dy^my dz^mz  at reference point xr
static double polyDeriv(const PolyND&P,const double xr[3],int mx,int my,int mz){
  double s=0;
  for(int k=0;k<PNC;k++)for(int j=0;j<PNC;j++)for(int i=0;i<PNC;i++){
    if(i<mx||j<my||k<mz) continue;
    double cf=(double)P.at(i,j,k); if(cf==0) continue;
    for(int q=0;q<mx;q++) cf*=(i-q);
    for(int q=0;q<my;q++) cf*=(j-q);
    for(int q=0;q<mz;q++) cf*=(k-q);
    for(int q=0;q<i-mx;q++) cf*=xr[0];
    for(int q=0;q<j-my;q++) cf*=xr[1];
    for(int q=0;q<k-mz;q++) cf*=xr[2];
    s+=cf; }
  return s;
}
// Parallel-surface (curvature) Jacobian J(s) = (1+k1 s)(1+k2 s) from the fitted
// level set, using ONLY the invariants of the shape operator S = P(H/|g|)P with
// P = I - nu nu^T:  J(s) = 1 + tr(S) s + (1/2)[(tr S)^2 - tr(S^2)] s^2.
// (No eigen-decomposition needed.)  Sphere check: S=(I-xx^T)/r -> tr=2/r,
// k1k2=1/r^2 -> J=(1+s/r)^2 = ((r+s)/r)^2, matching the exact radial factor.
static void shapeInvariants(const PolyND&P,const double xr[3],double h,
                            double&trS,double&detK){
  double g[3]={polyDeriv(P,xr,1,0,0)/h, polyDeriv(P,xr,0,1,0)/h, polyDeriv(P,xr,0,0,1)/h};
  double gm=sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]); if(gm<1e-30){ trS=0; detK=0; return; }
  double nu[3]={g[0]/gm,g[1]/gm,g[2]/gm};
  double H[3][3];
  H[0][0]=polyDeriv(P,xr,2,0,0)/(h*h); H[1][1]=polyDeriv(P,xr,0,2,0)/(h*h); H[2][2]=polyDeriv(P,xr,0,0,2)/(h*h);
  H[0][1]=H[1][0]=polyDeriv(P,xr,1,1,0)/(h*h);
  H[0][2]=H[2][0]=polyDeriv(P,xr,1,0,1)/(h*h);
  H[1][2]=H[2][1]=polyDeriv(P,xr,0,1,1)/(h*h);
  double Pr[3][3]; for(int i=0;i<3;i++)for(int j=0;j<3;j++) Pr[i][j]=(i==j?1.0:0.0)-nu[i]*nu[j];
  double T[3][3],S[3][3];
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){ double s2=0; for(int m=0;m<3;m++) s2+=Pr[i][m]*H[m][j]/gm; T[i][j]=s2; }
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){ double s2=0; for(int m=0;m<3;m++) s2+=T[i][m]*Pr[m][j]; S[i][j]=s2; }
  trS=S[0][0]+S[1][1]+S[2][2];
  double tr2=0; for(int i=0;i<3;i++)for(int j=0;j<3;j++) tr2+=S[i][j]*S[j][i];
  detK=0.5*(trS*trS-tr2);
}
// distance vector d (to the sphere surface) and true normal nu, at an interior x
static double g_fdEps = 0;      // finite-difference step for a general level set
// If >0, any surrogate face whose true boundary is farther than this is treated
// as an ARTIFICIAL CROP boundary (the sub-box cutting through the solid) rather
// than an immersed one: d is zeroed, so the shift becomes the identity and the
// face gets plain Nitsche with the exact solution at the face itself.  SBM needs
// |d| = O(h); crop faces would otherwise give |d| of tens of cells.
static double g_dMax = 0;
static long   g_nClamp = 0, g_nFacePt = 0;
static void distNormal(double x,double y,double z,double d[3],double nu[3]){
  if(!g_sdfFn){   // analytic sphere
    double dx=x-SPH_C[0],dy=y-SPH_C[1],dz=z-SPH_C[2], r=sqrt(dx*dx+dy*dy+dz*dz);
    nu[0]=dx/r; nu[1]=dy/r; nu[2]=dz/r;
    double s=SPH_R-r; d[0]=s*nu[0]; d[1]=s*nu[1]; d[2]=s*nu[2]; return; }
  // general level set: central-difference gradient + Newton steps to {phi=0}
  double e=g_fdEps>0?g_fdEps:1e-5, cx=x,cy=y,cz=z;
  for(int it=0; it<3; it++){
    double f=sdf(cx,cy,cz);
    double gx=(sdf(cx+e,cy,cz)-sdf(cx-e,cy,cz))/(2*e);
    double gy=(sdf(cx,cy+e,cz)-sdf(cx,cy-e,cz))/(2*e);
    double gz=(sdf(cx,cy,cz+e)-sdf(cx,cy,cz-e))/(2*e);
    double g2=gx*gx+gy*gy+gz*gz; if(g2<1e-30) break;
    cx-=f*gx/g2; cy-=f*gy/g2; cz-=f*gz/g2;
  }
  d[0]=cx-x; d[1]=cy-y; d[2]=cz-z;
  g_nFacePt++;
  if(g_dMax>0){ double dm=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    if(!(dm<=g_dMax)){ d[0]=d[1]=d[2]=0; g_nClamp++; } }
  double gx=(sdf(cx+e,cy,cz)-sdf(cx-e,cy,cz))/(2*e);
  double gy=(sdf(cx,cy+e,cz)-sdf(cx,cy-e,cz))/(2*e);
  double gz=(sdf(cx,cy,cz+e)-sdf(cx,cy,cz-e))/(2*e);
  double gm=sqrt(gx*gx+gy*gy+gz*gz); if(gm<1e-30) gm=1;
  nu[0]=gx/gm; nu[1]=gy/gm; nu[2]=gz/gm; }

struct SbmOut { double l2rel, l2abs; long nd3; int iters; int nBF; int nE; double h; };

// One SBM solve on a CUBIC box [lo3, lo3+L]^3 divided into N^3 cells.
// Geometry comes from sdf() (see g_sdfFn).  Returns the MMS L2 error.
inline SbmOut sbmSolveOne(int p, int N, const double lo3[3], double L){
  LagrangeBasis B; B.init(p); int n=B.n, ndof=n*n*n, ndof3=3*ndof;
  real Vm[QN_MAX][QN_MAX]; sbmDerivMatrix(B, Vm);
  // GSBM (Colomes-Modderman-Scovazzi, CMAME 452 (2026) 118748), Eq. (35):
  //   beta_1 = chi (r+1)^2 (2mu+lam), chi = 20   (shifted-Nitsche penalty, /h)
  //   beta_2 = kappa = 0.5                        (ghost penalty, Eq. (17))
  double chi = getenv("CHI") ? atof(getenv("CHI")) : 20.0;
  double kap = getenv("KAP") ? atof(getenv("KAP")) : 0.5;
  double gammaD = chi*(p+1)*(p+1)*(2*MU+LAM);
  double gammaG = kap*(2*MU+LAM);
  // GPL: cap the ghost-penalty order (default = p, i.e. GSBM Eq.(35) l=1..r).
  // The l-th term scales like gammaG*h*(D^l)^2 and the reference D^l entries grow
  // ~p^{2l}, so the top-l term dominates the spectrum at high p -- prime suspect
  // for the p=3 conditioning wall.
  int gpl = getenv("GPL") ? atoi(getenv("GPL")) : 0;

  // l-th normal-derivative of each 1-D basis at the two faces (xi=0, xi=1):
  //   Dl0[l][a]=(D^l)[0][a], Dl1[l][a]=(D^l)[n-1][a], l=1..p  (ghost penalty)
  static double Dl0[QN_MAX+1][QN_MAX], Dl1[QN_MAX+1][QN_MAX];
  { double Dp[QN_MAX][QN_MAX];
    for(int i=0;i<n;i++) for(int a=0;a<n;a++) Dp[i][a]=B.D[i][a];        // D^1
    for(int l=1;l<=p;l++){
      for(int a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if(l<p){ double Nw[QN_MAX][QN_MAX];
        for(int i=0;i<n;i++) for(int a=0;a<n;a++){
          double s=0; for(int m2=0;m2<n;m2++) s+=Dp[i][m2]*B.D[m2][a]; Nw[i][a]=s; }
        for(int i=0;i<n;i++) for(int a=0;a<n;a++) Dp[i][a]=Nw[i][a]; } } }

  if(gpl<=0||gpl>p) gpl=p;

  real t[PNC]; gllNodes(p,t);
    double h=L/N;
    if(g_sdfFn) g_dMax = 1.5*h;      // crop-face cutoff (general geometry only)
    g_nClamp = 0; g_nFacePt = 0;
    int nAx=p*N+1;
    std::vector<int> nodeDof((size_t)nAx*nAx*nAx,-1);
    auto gnode=[&](int cx,int cy,int cz,int i,int j,int k)->long{
      return (long)(p*cx+i)+(long)nAx*((p*cy+j)+(long)nAx*(p*cz+k)); };

    // ---- pass 1: SURROGATE cells (cell CENTER inside) treated as full (SBM);
    //      Gamma~ then sits ~h/2 from Gamma -> small |d|, clean O(h^{p+1}) ----
    std::vector<int> cellKind((size_t)N*N*N,0);   // 0 = not surrogate, 2 = surrogate
    std::vector<int> actEl;
    for(int cz=0;cz<N;cz++)for(int cy=0;cy<N;cy++)for(int cx=0;cx<N;cx++){
      double x0=lo3[0]+cx*h,y0=lo3[1]+cy*h,z0=lo3[2]+cz*h;
      double fc=sdf(x0+0.5*h,y0+0.5*h,z0+0.5*h);   // cell CENTER inside -> surrogate
      int id=cx+N*(cy+N*cz);
      if(fc<0){ cellKind[id]=2; actEl.push_back(id);
        for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++) nodeDof[gnode(cx,cy,cz,i,j,k)]=0; }
    }
    int nDofNode=0; for(size_t q=0;q<nodeDof.size();q++) if(nodeDof[q]==0) nodeDof[q]=nDofNode++;
    int nE=actEl.size(); long nd3=(long)3*nDofNode;

    std::vector<int> eNodes((size_t)nE*ndof), eCx(nE),eCy(nE),eCz(nE);
    for(int e=0;e<nE;e++){ int id=actEl[e],cx=id%N,cy=(id/N)%N,cz=id/(N*N);
      eCx[e]=cx;eCy[e]=cy;eCz[e]=cz;
      for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++)
        eNodes[(size_t)e*ndof+(i+n*(j+n*k))]=nodeDof[gnode(cx,cy,cz,i,j,k)]; }

    // ---- surrogate boundary faces: interior cell face with non-interior nbr ----
    struct BFace{ int e; int d; int s; char neu; };  // elem, axis d, side s, Neumann?
    std::vector<BFace> bf;
    // NEU=1: traction (Neumann) BC on the upper hemisphere (true normal nu_z>0),
    // Dirichlet elsewhere (a Dirichlet part is required to fix the rigid modes).
    int neuOn = getenv("NEU") ? atoi(getenv("NEU")) : 0;
    int useGap = getenv("NOGAP") ? 0 : 1;   // ablation: NOGAP=1 drops the gap term
    // JAC=1: exact parallel-surface (curvature) Jacobians for the sphere.  The
    // closest-point map x~ -> x_true is RADIAL, so r(tau)=r+tau*(R-r) is linear;
    // a surface patch scales by (r_tau/r)^2 and the gap cross-section likewise.
    // Without these, both geometric transfers are only LINEARIZED (an O(h)
    // relative error) which is a candidate cap on the p>=2 Neumann rate.
    int useJac = getenv("JAC") ? atoi(getenv("JAC")) : 0;
    for(int e=0;e<nE;e++){ int c[3]={eCx[e],eCy[e],eCz[e]};
      for(int d=0;d<3;d++) for(int s=0;s<2;s++){ int nb[3]={c[0],c[1],c[2]}; nb[d]+= s?1:-1;
        int kind = (nb[d]<0||nb[d]>=N)?0:cellKind[nb[0]+N*(nb[1]+N*nb[2])];
        if(kind==2) continue;                      // neighbor interior -> not a surrogate face
        char isn=0;
        if(neuOn){ double xc=lo3[0]+(c[0]+0.5)*h, yc=lo3[1]+(c[1]+0.5)*h, zc=lo3[2]+(c[2]+0.5)*h;
          xc+= (d==0)?(s?0.5*h:-0.5*h):0; yc+= (d==1)?(s?0.5*h:-0.5*h):0; zc+= (d==2)?(s?0.5*h:-0.5*h):0;
          double dd0[3],nu0[3]; distNormal(xc,yc,zc,dd0,nu0); isn = (nu0[2]>0.0)?1:0; }
        bf.push_back({e,d,s,isn}); } }
    int nBF=bf.size();
    int nNeu=0; for(int i=0;i<nBF;i++) nNeu+=bf[i].neu;


    // ---- ghost-penalty faces (GSBM Eq. (35), l=1..r): interior faces of the
    //      surrogate where at least one side touches Gamma~ (the boundary band).
    //      Controls the high normal derivatives that the Taylor shift S_d
    //      activates -- essential at p>=2 (without it the operator is indefinite).
    std::vector<char> nearB((size_t)nE,0);
    for(int fi=0;fi<nBF;fi++) nearB[bf[fi].e]=1;
    std::vector<int> elemIdx((size_t)N*N*N,-1);
    for(int e=0;e<nE;e++) elemIdx[actEl[e]]=e;
    struct GFace{ int eM,eP,d; };
    std::vector<GFace> gfaces;
    for(int e=0;e<nE;e++){ int c[3]={eCx[e],eCy[e],eCz[e]};
      for(int d=0;d<3;d++){ if(c[d]+1>=N) continue;
        int nb[3]={c[0],c[1],c[2]}; nb[d]++;
        int ep=elemIdx[nb[0]+N*(nb[1]+N*nb[2])];
        if(ep<0) continue;
        if(nearB[e]||nearB[ep]) gfaces.push_back({e,ep,d}); } }
    int nGF=gfaces.size();
    // SBMDBG: surrogate health check.  The shift S_d uses dref = d/h, so a large
    // |d|/h (a surrogate face far from the true boundary, or a bad Newton step on
    // a non-ideal SDF) blows up the Taylor shift and wrecks conditioning.
    if(getenv("SBMDBG")){
      double dmax=0,dsum=0; int nbad=0; long cnt=0;
      for(int fi=0;fi<nBF;fi++){ BFace F=bf[fi]; int e=F.e,dd0=F.d,s2=F.s;
        double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h;
        int t1=(dd0+1)%3,t2=(dd0+2)%3;
        GaussRule gd=gaussLegendre(B.n);
        for(int q1=0;q1<gd.n;q1++)for(int q2=0;q2<gd.n;q2++){
          real xr[3]; xr[dd0]=s2?1.0:0.0; xr[t1]=gd.x[q1]; xr[t2]=gd.x[q2];
          double xg=x0+xr[0]*h,yg=y0+xr[1]*h,zg=z0+xr[2]*h;
          double dv[3],nu[3]; distNormal(xg,yg,zg,dv,nu);
          double dm=sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2])/h;
          dmax=fmax(dmax,dm); dsum+=dm; cnt++; if(dm>2.0) nbad++; } }
      int iso=0;
      for(int e=0;e<nE;e++){ int c[3]={eCx[e],eCy[e],eCz[e]}, nb2=0;
        for(int dd0=0;dd0<3;dd0++)for(int s2=0;s2<2;s2++){ int q[3]={c[0],c[1],c[2]}; q[dd0]+= s2?1:-1;
          if(q[dd0]<0||q[dd0]>=N) continue;
          if(cellKind[q[0]+N*(q[1]+N*q[2])]==2) nb2++; }
        if(nb2==0) iso++; }
      printf("  [SBMDBG N=%d] nElem=%d nBF=%d nGhostF=%d isolated=%d | |d|/h max %.3f mean %.3f, >2h at %d/%ld pts, clamped %ld/%ld\n",
             N,nE,nBF,nGF,iso,dmax,dsum/(cnt?cnt:1),nbad,cnt,g_nClamp,g_nFacePt);
      if(getenv("SBMDBG")[0]=='2'){ SbmOut o; o.l2rel=0; o.l2abs=0; o.nd3=nd3; o.iters=0;
        o.nBF=nBF; o.nE=nE; o.h=h; return o; }
    }

    // ghost-face contribution: j_h = sum_l gammaG*h*int_F [D_n^l u][D_n^l v] dS_ref
    // (the 1/h^{2l} from physical derivs and h^2 from dS collapse to a single h).
    GaussRule g1=gaussLegendre(p+1);
    auto ghostFace=[&](const GFace& gf, const std::vector<double>* X,
                       std::vector<double>* Y, std::vector<double>* Dg){
      int d=gf.d, t1=(d+1)%3, t2=(d+2)%3;
      const int* nodM=&eNodes[(size_t)gf.eM*ndof];
      const int* nodP=&eNodes[(size_t)gf.eP*ndof];
      double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
      for(int q1=0;q1<g1.n;q1++) for(int q2=0;q2<g1.n;q2++){
        double w=g1.w[q1]*g1.w[q2];
        real L1[QN_MAX],L2[QN_MAX]; B.basis1(g1.x[q1],L1); B.basis1(g1.x[q2],L2);
        for(int a=0;a<ndof;a++){ int idx[3]={a%n,(a/n)%n,a/(n*n)}; int idn=idx[d];
          double Lt=L1[idx[t1]]*L2[idx[t2]];
          for(int l=1;l<=p;l++){ cP[l][a]=Dl0[l][idn]*Lt; cM[l][a]=Dl1[l][idn]*Lt; } }
        for(int l=1;l<=gpl;l++){ double cf=gammaG*h*w;
          if(Dg){ for(int a=0;a<ndof;a++) for(int cc=0;cc<3;cc++){
              #pragma omp atomic
              (*Dg)[3*nodP[a]+cc]+=cf*cP[l][a]*cP[l][a];
              #pragma omp atomic
              (*Dg)[3*nodM[a]+cc]+=cf*cM[l][a]*cM[l][a]; } }
          else { double jU[3]={0,0,0};
            for(int a=0;a<ndof;a++) for(int cc=0;cc<3;cc++)
              jU[cc]+=(*X)[3*nodP[a]+cc]*cP[l][a]-(*X)[3*nodM[a]+cc]*cM[l][a];
            for(int a=0;a<ndof;a++) for(int cc=0;cc<3;cc++){
              #pragma omp atomic
              (*Y)[3*nodP[a]+cc]+=cf*cP[l][a]*jU[cc];
              #pragma omp atomic
              (*Y)[3*nodM[a]+cc]-=cf*cM[l][a]*jU[cc]; } } } } };

    // ---- PRECOMPUTED boundary geometry -------------------------------------
    // d and nu at every surrogate-boundary quadrature point, evaluated ONCE.
    // Critical for a BVH-backed level set: distNormal costs ~18 SDF evaluations
    // (FD gradient + Newton) and applyA runs once per Krylov iteration, so
    // recomputing it inside the solver made a blade solve non-terminating.
    // Negligible cost for an analytic SDF, so it is always on.
    GaussRule ggp = gaussLegendre(B.n);
    const int NQF = ggp.n*ggp.n;
    std::vector<double> geoD((size_t)nBF*NQF*6, 0.0);
    #pragma omp parallel for schedule(dynamic,32)
    for(int fi=0;fi<nBF;fi++){ BFace F=bf[fi]; int e=F.e,d=F.d,s2=F.s;
      double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h;
      int t1=(d+1)%3,t2=(d+2)%3;
      for(int q1=0;q1<ggp.n;q1++)for(int q2=0;q2<ggp.n;q2++){
        real xr[3]; xr[d]=s2?1.0:0.0; xr[t1]=ggp.x[q1]; xr[t2]=ggp.x[q2];
        double xg=x0+xr[0]*h,yg=y0+xr[1]*h,zg=z0+xr[2]*h;
        double dv[3],nv[3]; distNormal(xg,yg,zg,dv,nv);
        double *g6=&geoD[((size_t)fi*NQF + q1*ggp.n + q2)*6];
        g6[0]=dv[0]; g6[1]=dv[1]; g6[2]=dv[2]; g6[3]=nv[0]; g6[4]=nv[1]; g6[5]=nv[2]; } }

    // ---- Jacobi diag + RHS (body force + shifted-Nitsche Dirichlet data) ----
    std::vector<double> diagv(nd3,0.0), b(nd3,0.0);
    GaussRule gg=gaussLegendre(B.n);       // face (surrogate boundary) quadrature
    GaussRule gt=gaussLegendre(B.n);       // 1-D rule along d for the gap sliver
    // bulk load + diag (tensor GLL over full interior cells)
    #pragma omp parallel for schedule(dynamic,64)
    for(int e=0;e<nE;e++){ const int*nod=&eNodes[(size_t)e*ndof];
      double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h;
      real gb[3*QN_MAX*QN_MAX*QN_MAX],vb[QN_MAX*QN_MAX*QN_MAX];
      for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++){
        real xr[3]={B.t[i],B.t[j],B.t[k]}; double wv=B.wq[i]*B.wq[j]*B.wq[k]*h*h*h, wb=B.wq[i]*B.wq[j]*B.wq[k]*h;
        B.allGradRef(xr,gb); B.allVal(xr,vb);
        double f[3]; fbody(x0+B.t[i]*h,y0+B.t[j]*h,z0+B.t[k]*h,f);
        for(int a=0;a<ndof;a++){ for(int l=0;l<3;l++){ double vv=wv*f[l]*vb[a];
            #pragma omp atomic
            b[3*nod[a]+l]+=vv; }
          double gsq=(gb[3*a]*gb[3*a]+gb[3*a+1]*gb[3*a+1]+gb[3*a+2]*gb[3*a+2]);
          for(int l=0;l<3;l++){ double dv=wb*(MU*(gsq+gb[3*a+l]*gb[3*a+l])+LAM*gb[3*a+l]*gb[3*a+l]);
            #pragma omp atomic
            diagv[3*nod[a]+l]+=dv; } } } }
    // Surrogate Nitsche SBM (Main-Scovazzi), NIPG variant: flux sigma(u).n~ at the
    // surrogate face x~ (IN-CELL gradient gb -> standard inverse inequality, well
    // conditioned), plain-v consistency, +adjoint sign (coercive for ANY penalty),
    // shifted penalty (S_h).  Data g = u(x_true).  No cut quad, no ghost penalty,
    // no gap.  Non-symmetric but coercive -> BiCGStab.
    #pragma omp parallel for schedule(dynamic,64)
    for(int fi=0;fi<nBF;fi++){ BFace F=bf[fi]; int e=F.e,d=F.d,s=F.s; const int*nod=&eNodes[(size_t)e*ndof];
      double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h; double nsign=s?1.0:-1.0;
      int t1=(d+1)%3,t2=(d+2)%3; double nn[3]={0,0,0}; nn[d]=nsign;
      real gb[3*QN_MAX*QN_MAX*QN_MAX],sh[QN_MAX*QN_MAX*QN_MAX],vb[QN_MAX*QN_MAX*QN_MAX];
      for(int q1=0;q1<gg.n;q1++)for(int q2=0;q2<gg.n;q2++){
        real xr[3]; xr[d]=s?1.0:0.0; xr[t1]=gg.x[q1]; xr[t2]=gg.x[q2];
        double hw=gg.w[q1]*gg.w[q2]*h;
        double xg=x0+xr[0]*h, yg=y0+xr[1]*h, zg=z0+xr[2]*h;   // x~ on Gamma~
        const double *g6=&geoD[((size_t)fi*NQF + q1*gg.n + q2)*6];
        double dd[3]={g6[0],g6[1],g6[2]}, nu[3]={g6[3],g6[4],g6[5]};
        real dref[3]={(real)(dd[0]/h),(real)(dd[1]/h),(real)(dd[2]/h)};
        if(F.neu){
          // GAP-SBM NEUMANN.  With the GAP included the bulk domain is Omega (not
          // Omega~), so the IBP boundary IS the true Gamma and the traction is a
          // pure NATURAL BC -- no flux/tangential patch needed:
          //   int_Omega~ sigma:eps(v) + int_gap sigma:eps(v)
          //     = int_Omega~ f.v + int_gap f.v + int_Gamma t.v
          // Gap sliver: x(tau)=x~+tau*d, signed volume element (d.n~) dtau dGamma~
          // (negative where the surrogate over-covers).  Surface transfer:
          //   int_Gamma t.v dGamma = int_Gamma~ (n~.nu) t(x_true).v(x_true) dGamma~
          // (orthogonal projection between the two surfaces scales area by n~.nu).
          // SCALING: dGamma~ = h^2*wf and the traction carries NO gradient, so the
          // weight is h^2*wf -- NOT the hw=h*wf used by the Nitsche terms (those
          // carry one physical gradient = 1/h).
          real gbt[3*QN_MAX*QN_MAX*QN_MAX],vbt[QN_MAX*QN_MAX*QN_MAX],vtr[QN_MAX*QN_MAX*QN_MAX];
          double wf=gg.w[q1]*gg.w[q2];
          double dn=dd[0]*nn[0]+dd[1]*nn[1]+dd[2]*nn[2];      // signed gap thickness
          double sTot=dd[0]*nu[0]+dd[1]*nu[1]+dd[2]*nu[2];    // signed offset along nu
          double trS=0,detK=0;
          if(useJac>=2){ PolyND Pf=fitSdfCell(x0,y0,z0,h,(useJac==2)?p:PDEG);
            double xrd[3]={(double)xr[0],(double)xr[1],(double)xr[2]};
            shapeInvariants(Pf,xrd,h,trS,detK); }
          for(int qt=0;qt<gt.n && useGap;qt++){ double tau=gt.x[qt];
            double xt=xg+tau*dd[0],yt=yg+tau*dd[1],zt=zg+tau*dd[2];
            real xrt[3]={(real)((xt-x0)/h),(real)((yt-y0)/h),(real)((zt-z0)/h)};
            B.allGradRef(xrt,gbt); B.allVal(xrt,vbt);
            double jg=1.0;
            if(useJac==1){ double rr=sqrt((xg-SPH_C[0])*(xg-SPH_C[0])+(yg-SPH_C[1])*(yg-SPH_C[1])+(zg-SPH_C[2])*(zg-SPH_C[2]));
              double rt=rr+tau*(SPH_R-rr); jg=(rt/rr)*(rt/rr); }
            else if(useJac>=2){ double sq=tau*sTot; jg=1.0+trS*sq+detK*sq*sq; }
            double wgb=dn*wf*gt.w[qt]*jg;     // stiffness weight (the two 1/h cancel h^2)
            double wgl=h*h*wgb;               // load weight (full gap dV)
            double f[3]; fbody(xt,yt,zt,f);
            for(int a=0;a<ndof;a++){
              double gsq=gbt[3*a]*gbt[3*a]+gbt[3*a+1]*gbt[3*a+1]+gbt[3*a+2]*gbt[3*a+2];
              for(int l=0;l<3;l++){ double vv=wgl*f[l]*vbt[a];
                #pragma omp atomic
                b[3*nod[a]+l]+=vv;
                double dv=wgb*(MU*(gsq+gbt[3*a+l]*gbt[3*a+l])+LAM*gbt[3*a+l]*gbt[3*a+l]);
                #pragma omp atomic
                diagv[3*nod[a]+l]+=dv; } } }
          double xT=xg+dd[0],yT=yg+dd[1],zT=zg+dd[2];
          real xrT[3]={(real)((xT-x0)/h),(real)((yT-y0)/h),(real)((zT-z0)/h)};
          B.allVal(xrT,vtr);
          double se[3][3]; exactStress(xT,yT,zT,se);
          double tv[3]; for(int i2=0;i2<3;i2++) tv[i2]=se[i2][0]*nu[0]+se[i2][1]*nu[1]+se[i2][2]*nu[2];
          double nvn=nn[0]*nu[0]+nn[1]*nu[1]+nn[2]*nu[2];
          double jS=1.0;
          if(useJac==1){ double rr=sqrt((xg-SPH_C[0])*(xg-SPH_C[0])+(yg-SPH_C[1])*(yg-SPH_C[1])+(zg-SPH_C[2])*(zg-SPH_C[2]));
            jS=(SPH_R/rr)*(SPH_R/rr); }
          else if(useJac>=2){ jS=1.0+trS*sTot+detK*sTot*sTot; }
          double wS=h*h*wf*nvn*jS;
          for(int a=0;a<ndof;a++) for(int l=0;l<3;l++){
            double vv=wS*tv[l]*vtr[a];
            #pragma omp atomic
            b[3*nod[a]+l]+=vv; }
          continue;
        }
        B.allGradRef(xr,gb); sbmShiftAll(B,Vm,xr,dref,sh);
        double g[3]; uex(xg+dd[0],yg+dd[1],zg+dd[2],g);        // data at x_true
        double gn=g[0]*nn[0]+g[1]*nn[1]+g[2]*nn[2];
        for(int a=0;a<ndof;a++){ double gan=gb[3*a+d]*nsign;
          double ggb=g[0]*gb[3*a]+g[1]*gb[3*a+1]+g[2]*gb[3*a+2];
          for(int l=0;l<3;l++){
            // L(v) = -(sigma(v).n, u_D) + (beta1/h S_d v, u_D)   [GSBM Eq. (35)]
            double radj=-(MU*(g[l]*gan+ggb*nn[l])+LAM*gb[3*a+l]*gn);
            double rpen=gammaD*g[l]*sh[a];
            #pragma omp atomic
            b[3*nod[a]+l]+=hw*(radj+rpen);
            double dv=hw*gammaD*sh[a]*sh[a];
            #pragma omp atomic
            diagv[3*nod[a]+l]+=dv; } } } }
    // ghost-penalty diagonal
    #pragma omp parallel for schedule(dynamic,64)
    for(int gf=0;gf<nGF;gf++) ghostFace(gfaces[gf],nullptr,nullptr,&diagv);
    for(long i=0;i<nd3;i++) if(diagv[i]<=0) diagv[i]=1.0;

    // ---- matrix-free operator: bulk (qpElemUncut) + shifted Nitsche ----
    auto applyA=[&](const std::vector<double>&x,std::vector<double>&y){
      std::fill(y.begin(),y.end(),0.0);
      #pragma omp parallel for schedule(dynamic,64)
      for(int e=0;e<nE;e++){ const int*nod=&eNodes[(size_t)e*ndof];
        real ul[3*QN_MAX*QN_MAX*QN_MAX],yl[3*QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<ndof;a++)for(int l=0;l<3;l++) ul[3*a+l]=(real)x[3*nod[a]+l];
        qpElemUncut(B,(real)MU,(real)LAM,h,ul,yl);
        for(int a=0;a<ndof;a++)for(int l=0;l<3;l++){
          #pragma omp atomic
          y[3*nod[a]+l]+=yl[3*a+l]; } }
      #pragma omp parallel for schedule(dynamic,64)
      for(int fi=0;fi<nBF;fi++){ BFace F=bf[fi]; int e=F.e,d=F.d,s=F.s; const int*nod=&eNodes[(size_t)e*ndof];
        double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h; double nsign=s?1.0:-1.0;
        int t1=(d+1)%3,t2=(d+2)%3; double nn[3]={0,0,0}; nn[d]=nsign;
        real gb[3*QN_MAX*QN_MAX*QN_MAX],sh[QN_MAX*QN_MAX*QN_MAX],vb[QN_MAX*QN_MAX*QN_MAX];
        real gT[3*QN_MAX*QN_MAX*QN_MAX];
        for(int q1=0;q1<gg.n;q1++)for(int q2=0;q2<gg.n;q2++){
          real xr[3]; xr[d]=s?1.0:0.0; xr[t1]=gg.x[q1]; xr[t2]=gg.x[q2];
          double hw=gg.w[q1]*gg.w[q2]*h;
          double xg=x0+xr[0]*h,yg=y0+xr[1]*h,zg=z0+xr[2]*h;
          const double *g6=&geoD[((size_t)fi*NQF + q1*gg.n + q2)*6];
          double dd[3]={g6[0],g6[1],g6[2]}, nu[3]={g6[3],g6[4],g6[5]};
          real dref[3]={(real)(dd[0]/h),(real)(dd[1]/h),(real)(dd[2]/h)};
          if(F.neu){
            // GAP bulk operator: sigma(u~):eps(v~) over the sliver, extended field.
            // (Traction is pure data -> RHS only; no Nitsche on a Neumann face.)
            double wf=gg.w[q1]*gg.w[q2];
            double dn=dd[0]*nn[0]+dd[1]*nn[1]+dd[2]*nn[2];
            double sTot=dd[0]*nu[0]+dd[1]*nu[1]+dd[2]*nu[2];
            double trS=0,detK=0;
            if(useJac>=2){ PolyND Pf=fitSdfCell(x0,y0,z0,h,(useJac==2)?p:PDEG);
              double xrd[3]={(double)xr[0],(double)xr[1],(double)xr[2]};
              shapeInvariants(Pf,xrd,h,trS,detK); }
            for(int qt=0;qt<gt.n && useGap;qt++){ double tau=gt.x[qt];
              double xt=xg+tau*dd[0],yt=yg+tau*dd[1],zt=zg+tau*dd[2];
              real xrt[3]={(real)((xt-x0)/h),(real)((yt-y0)/h),(real)((zt-z0)/h)};
              B.allGradRef(xrt,gT);
              double jg=1.0;
              if(useJac==1){ double rr=sqrt((xg-SPH_C[0])*(xg-SPH_C[0])+(yg-SPH_C[1])*(yg-SPH_C[1])+(zg-SPH_C[2])*(zg-SPH_C[2]));
                double rt=rr+tau*(SPH_R-rr); jg=(rt/rr)*(rt/rr); }
              else if(useJac>=2){ double sq=tau*sTot; jg=1.0+trS*sq+detK*sq*sq; }
              double wgb=dn*wf*gt.w[qt]*jg;
              double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
              for(int a=0;a<ndof;a++)for(int i2=0;i2<3;i2++){ double ua=x[3*nod[a]+i2];
                gU[i2][0]+=ua*gT[3*a]; gU[i2][1]+=ua*gT[3*a+1]; gU[i2][2]+=ua*gT[3*a+2]; }
              double e2[3][3],tr2; for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) e2[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
              tr2=e2[0][0]+e2[1][1]+e2[2][2];
              double s2[3][3]; for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) s2[i2][j2]=2*MU*e2[i2][j2]+(i2==j2?LAM*tr2:0);
              for(int a=0;a<ndof;a++) for(int l=0;l<3;l++){
                double vv=s2[l][0]*gT[3*a]+s2[l][1]*gT[3*a+1]+s2[l][2]*gT[3*a+2];
                #pragma omp atomic
                y[3*nod[a]+l]+=wgb*vv; } }
            continue;
          }
          B.allGradRef(xr,gb); B.allVal(xr,vb); sbmShiftAll(B,Vm,xr,dref,sh);
          double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
          for(int a=0;a<ndof;a++)for(int i2=0;i2<3;i2++){ double ua=x[3*nod[a]+i2];
            gradU[i2][0]+=ua*gb[3*a]; gradU[i2][1]+=ua*gb[3*a+1]; gradU[i2][2]+=ua*gb[3*a+2];
            Shu[i2]+=ua*sh[a]; }
          double eps[3][3],tr=0; for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
          tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
          for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) sig[i2][j2]=2*MU*eps[i2][j2]+(i2==j2?LAM*tr:0);
          double tu[3]; for(int i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
          double Shun=Shu[0]*nn[0]+Shu[1]*nn[1]+Shu[2]*nn[2];
          for(int a=0;a<ndof;a++){ double gan=gb[3*a+d]*nsign;
            double ugb=Shu[0]*gb[3*a]+Shu[1]*gb[3*a+1]+Shu[2]*gb[3*a+2];
            for(int l=0;l<3;l++){
              double c1=-tu[l]*vb[a];                                    // -(v, sigma(u).n)
              double c2=-(MU*(Shu[l]*gan+ugb*nn[l])+LAM*gb[3*a+l]*Shun);  // -(sigma(v).n, S_d u)
              double c3=gammaD*Shu[l]*sh[a];                             // (beta1/h S_d v, S_d u)
              #pragma omp atomic
              y[3*nod[a]+l]+=hw*(c1+c2+c3); } } } }
      // ghost penalty (GSBM Eq. (35), l=1..r)
      #pragma omp parallel for schedule(dynamic,64)
      for(int gf=0;gf<nGF;gf++) ghostFace(gfaces[gf],&x,&y,nullptr);
    };

    if(getenv("DBG")){                    // symmetry + definiteness probe on A
      auto dotp=[&](const std::vector<double>&a2,const std::vector<double>&b2){
        double s=0; for(long i=0;i<nd3;i++) s+=a2[i]*b2[i]; return s; };
      std::vector<double> vv(nd3),ww(nd3),Av(nd3),Aw(nd3);
      unsigned sd=12345u;
      for(long i=0;i<nd3;i++){ sd=sd*1664525u+1013904223u; vv[i]=(double)(sd>>9)/8388608.0-1.0;
        sd=sd*1664525u+1013904223u; ww[i]=(double)(sd>>9)/8388608.0-1.0; }
      applyA(vv,Av); applyA(ww,Aw);
      double vAw=dotp(vv,Aw), wAv=dotp(ww,Av), vAv=dotp(vv,Av), wAw=dotp(ww,Aw);
      printf("  [DBG p=%d N=%d] sym |vAw-wAv|/|vAw|=%.2e  vAv=%.3e wAw=%.3e (defpos? %s)\n",
             p,N,fabs(vAw-wAv)/(fabs(vAw)+1e-300),vAv,wAw,(vAv>0&&wAw>0)?"yes":"NO");
    }

    // ---- PC=block : element-block additive Schwarz -- *** DOES NOT WORK ***
    // Kept because the block EXTRACTION is verified correct and encodes a real
    // finding (see CST below), but the assembled preconditioner fails: CG halts
    // after ~15 iterations with a TRUE residual ||b-Au||/||b|| ~ 0.99 (the
    // recursive residual collapses while the real one never moves), and GMRES is
    // no better.  A and M^-1 both test positive on random vectors, so the cause is
    // NOT indefiniteness and has not been isolated.  DEFAULT is Jacobi (PC unset),
    // which is verified.  Do not use PC=block without re-diagnosing this.
    // ---- (original notes) -----------------------------------------------------
    // Jacobi is far too weak at p=3: the ghost penalty's top-order term scales as
    // gammaG*h*(D^l)^2 with reference D^l entries growing ~p^{2l}, so it dominates
    // the spectrum -- yet it CANNOT be dropped (capping l wrecks accuracy).  So we
    // precondition with the exact DIAGONAL BLOCKS A_ee of the assembled operator.
    // Those blocks are extracted by COLORED PROBING: elements are 8-coloured by
    // (cx,cy,cz) parity so same-coloured elements share no dof; then one applyA per
    // (colour, local index) yields a whole colour's blocks at once.  This reuses the
    // verified applyA -- no duplicated element math -- at 8*mB applies total.
    int mB=3*ndof;
    std::vector<double> Ablk; std::vector<int> Apiv; std::vector<double> mult;
    bool useBlock = getenv("PC") && getenv("PC")[0]=='b';
    if(useBlock){
      Ablk.assign((size_t)nE*mB*mB,0.0); Apiv.assign((size_t)nE*mB,0);
      mult.assign(nd3,0.0);
      for(int e=0;e<nE;e++){ const int*nod=&eNodes[(size_t)e*ndof];
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++) mult[3*nod[a]+l]+=1.0; }
      for(long i=0;i<nd3;i++) if(mult[i]<=0) mult[i]=1.0;
      std::vector<double> xp(nd3),yp(nd3);
      // COLOURING MUST BE STRIDE 3, NOT 2.  Two elements 2 apart share no dof, but
      // the operator still couples them THROUGH the intervening element's bulk
      // stencil (e at cx and e'' at cx+2 both touch element cx+1's dofs), so a
      // stride-2 probe pollutes the extracted rows.  A[i][j]!=0 only for dofs in a
      // common element or in ghost-adjacent elements, so distance >=3 is clean.
      int CST=getenv("CST")?atoi(getenv("CST")):3;
      for(int col=0;col<CST*CST*CST;col++){
        std::vector<int> ce;
        for(int e=0;e<nE;e++) if((eCx[e]%CST)+CST*(eCy[e]%CST)+CST*CST*(eCz[e]%CST)==col) ce.push_back(e);
        if(ce.empty()) continue;
        for(int a=0;a<mB;a++){
          std::fill(xp.begin(),xp.end(),0.0);
          for(int e : ce){ const int*nod=&eNodes[(size_t)e*ndof]; xp[3*nod[a/3]+(a%3)]=1.0; }
          applyA(xp,yp);
          for(int e : ce){ const int*nod=&eNodes[(size_t)e*ndof];
            double*Ae=&Ablk[(size_t)e*mB*mB];
            for(int bq=0;bq<mB;bq++) Ae[(size_t)bq*mB+a]=yp[3*nod[bq/3]+(bq%3)]; } }
      }
      // LU with partial pivoting, per element
      #pragma omp parallel for schedule(dynamic,4)
      for(int e=0;e<nE;e++){ double*Ae=&Ablk[(size_t)e*mB*mB]; int*pv=&Apiv[(size_t)e*mB];
        for(int k=0;k<mB;k++){
          int piv=k; double mx=fabs(Ae[(size_t)k*mB+k]);
          for(int i=k+1;i<mB;i++){ double v=fabs(Ae[(size_t)i*mB+k]); if(v>mx){mx=v;piv=i;} }
          pv[k]=piv;
          if(piv!=k) for(int j=0;j<mB;j++) std::swap(Ae[(size_t)k*mB+j],Ae[(size_t)piv*mB+j]);
          double d=Ae[(size_t)k*mB+k]; if(fabs(d)<1e-300) d=1e-300;
          for(int i=k+1;i<mB;i++){ double f=Ae[(size_t)i*mB+k]/d; Ae[(size_t)i*mB+k]=f;
            if(f!=0) for(int j=k+1;j<mB;j++) Ae[(size_t)i*mB+j]-=f*Ae[(size_t)k*mB+j]; } } }
      printf("  [PC=block] %d element blocks of %dx%d (%.0f MB)\n",
             nE,mB,mB,(double)nE*mB*mB*8.0/1e6);
      if(getenv("LUCHK")){   // LU round-trip: solve A_ee x = A_ee u, expect x==u
        // rebuild a raw block (Ablk is LU-overwritten) for e0
        int e0=nE/2; const int*nod0=&eNodes[(size_t)e0*ndof];
        std::vector<double> Ar((size_t)mB*mB), xp2(nd3),yp2(nd3);
        for(int a=0;a<mB;a++){ std::fill(xp2.begin(),xp2.end(),0.0);
          xp2[3*nod0[a/3]+(a%3)]=1.0; applyA(xp2,yp2);
          for(int bq=0;bq<mB;bq++) Ar[(size_t)bq*mB+a]=yp2[3*nod0[bq/3]+(bq%3)]; }
        std::vector<double> uu(mB),rr(mB); unsigned sd=7u;
        for(int a=0;a<mB;a++){ sd=sd*1664525u+1013904223u; uu[a]=(double)(sd>>9)/8388608.0-1.0; }
        for(int i=0;i<mB;i++){ double t=0; for(int j=0;j<mB;j++) t+=Ar[(size_t)i*mB+j]*uu[j]; rr[i]=t; }
        const double*Ae=&Ablk[(size_t)e0*mB*mB]; const int*pv=&Apiv[(size_t)e0*mB];
        std::vector<double> rl(rr);
        for(int k=0;k<mB;k++){ if(pv[k]!=k) std::swap(rl[k],rl[pv[k]]);
          for(int i=k+1;i<mB;i++) rl[i]-=Ae[(size_t)i*mB+k]*rl[k]; }
        for(int i=mB-1;i>=0;i--){ double s2=rl[i];
          for(int j=i+1;j<mB;j++) s2-=Ae[(size_t)i*mB+j]*rl[j];
          double d=Ae[(size_t)i*mB+i]; rl[i]=s2/((fabs(d)<1e-300)?1e-300:d); }
        double em=0,um=0; for(int a=0;a<mB;a++){ em=fmax(em,fabs(rl[a]-uu[a])); um=fmax(um,fabs(uu[a])); }
        // also report the block's extreme diagonal magnitudes after LU (cond proxy)
        double dmin=1e300,dmax=0;
        for(int i=0;i<mB;i++){ double d=fabs(Ae[(size_t)i*mB+i]); dmin=fmin(dmin,d); dmax=fmax(dmax,d); }
        printf("  [LUCHK] max|x-u|=%.3e (|u|max %.3e, rel %.2e)  U diag min %.3e max %.3e ratio %.2e\n",
               em,um,em/um,dmin,dmax,dmax/(dmin+1e-300));
      }
      if(getenv("DBG")){   // verify a probed block against the true operator action
        std::vector<double> xv(nd3,0.0),yv2(nd3);
        int e0=nE/2; const int*nod0=&eNodes[(size_t)e0*ndof];
        unsigned sd=999u; double ul0[3*QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<mB;a++){ sd=sd*1664525u+1013904223u;
          ul0[a]=(double)(sd>>9)/8388608.0-1.0; xv[3*nod0[a/3]+(a%3)]=ul0[a]; }
        applyA(xv,yv2);
        // NOTE: Ablk has already been LU-overwritten, so rebuild this one block raw
        std::vector<double> Araw((size_t)mB*mB);
        std::vector<double> xp2(nd3),yp2(nd3);
        for(int a=0;a<mB;a++){ std::fill(xp2.begin(),xp2.end(),0.0);
          xp2[3*nod0[a/3]+(a%3)]=1.0; applyA(xp2,yp2);
          for(int bq=0;bq<mB;bq++) Araw[(size_t)bq*mB+a]=yp2[3*nod0[bq/3]+(bq%3)]; }
        double emax=0,ymax=0;
        for(int bq=0;bq<mB;bq++){ double s2=0;
          for(int a=0;a<mB;a++) s2+=Araw[(size_t)bq*mB+a]*ul0[a];
          double tgt=yv2[3*nod0[bq/3]+(bq%3)];
          emax=fmax(emax,fabs(s2-tgt)); ymax=fmax(ymax,fabs(tgt)); }
        printf("  [DBG probe] max|A_ee*u - (A u)|_e = %.3e  (|Au|max %.3e)  rel %.2e\n",
               emax,ymax,emax/(ymax+1e-300));
      }
    }
    std::vector<double> rsc;   // scratch for the symmetric weighting
    auto precond=[&](const std::vector<double>&r,std::vector<double>&z){
      if(!useBlock){ for(long i=0;i<nd3;i++) z[i]=r[i]/diagv[i]; return; }
      // SYMMETRIC weighted additive Schwarz:  M^-1 = D^-1/2 (sum_e R^T A_e^-1 R) D^-1/2
      // with D = dof multiplicity.  Scaling on ONE side only makes M^-1
      // non-symmetric, which breaks CG (it simply fails to converge).
      if((long)rsc.size()!=nd3) rsc.assign(nd3,0.0);
      for(long i=0;i<nd3;i++) rsc[i]=r[i]/sqrt(mult[i]);
      std::fill(z.begin(),z.end(),0.0);
      #pragma omp parallel for schedule(dynamic,4)
      for(int e=0;e<nE;e++){ const int*nod=&eNodes[(size_t)e*ndof];
        const double*Ae=&Ablk[(size_t)e*mB*mB]; const int*pv=&Apiv[(size_t)e*mB];
        double rl[3*QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<mB;a++) rl[a]=rsc[3*nod[a/3]+(a%3)];
        for(int k=0;k<mB;k++){ if(pv[k]!=k) std::swap(rl[k],rl[pv[k]]);
          for(int i=k+1;i<mB;i++) rl[i]-=Ae[(size_t)i*mB+k]*rl[k]; }
        for(int i=mB-1;i>=0;i--){ double s2=rl[i];
          for(int j=i+1;j<mB;j++) s2-=Ae[(size_t)i*mB+j]*rl[j];
          double d=Ae[(size_t)i*mB+i]; rl[i]=s2/((fabs(d)<1e-300)?1e-300:d); }
        for(int a=0;a<mB;a++){
          #pragma omp atomic
          z[3*nod[a/3]+(a%3)]+=rl[a]; } }
      for(long i=0;i<nd3;i++) z[i]/=sqrt(mult[i]);
    };

    if(getenv("PCCHK")){   // definiteness of A and of M^-1 over random vectors
      std::vector<double> rv(nd3),zv(nd3),av(nd3);
      unsigned sd=4242u; int negA=0,negM=0; int NT=12;
      double minRA=1e300,minRM=1e300;
      for(int t=0;t<NT;t++){
        for(long i=0;i<nd3;i++){ sd=sd*1664525u+1013904223u; rv[i]=(double)(sd>>9)/8388608.0-1.0; }
        applyA(rv,av); double rAr=0; for(long i=0;i<nd3;i++) rAr+=rv[i]*av[i];
        precond(rv,zv);  double rMr=0; for(long i=0;i<nd3;i++) rMr+=rv[i]*zv[i];
        if(rAr<=0) negA++; if(rMr<=0) negM++;
        minRA=fmin(minRA,rAr); minRM=fmin(minRM,rMr);
      }
      printf("  [PCCHK] over %d random vectors: r.Ar<=0 in %d (min %.3e) | r.M^-1r<=0 in %d (min %.3e)\n",
             NT,negA,minRA,negM,minRM);
    }
    auto dot=[&](const std::vector<double>&a2,const std::vector<double>&b2){
      double s=0;
      #pragma omp parallel for reduction(+:s)
      for(long i=0;i<nd3;i++) s+=a2[i]*b2[i]; return s; };
    // SOLVER=cg|gmres.  History: WITHOUT the ghost penalty the operator is
    // symmetric-INDEFINITE at p>=2 (the shift cross-term breaks Nitsche
    // coercivity) so CG hit pAp<0 and BiCGStab diverged, leaving restarted GMRES
    // -- which then STAGNATES on the ill-conditioned p=3 system.  But the ghost
    // penalty (added later, GSBM Eq. (35)) RESTORES definiteness, so CG becomes
    // viable again and does not suffer restart stagnation.  CG is the default.
    double tol=getenv("TOL")?atof(getenv("TOL")):1e-9;
    const char* slv=getenv("SOLVER")?getenv("SOLVER"):"gmres";  // gmres = the verified default
    std::vector<double> uOut(nd3,0.0); int itOut=0;
    if(slv[0]=='c'){                       // ---- Jacobi-PCG ----
      std::vector<double> u(nd3,0.0),r(nd3),z(nd3),pd(nd3),Ap(nd3);
      applyA(u,Ap); for(long i=0;i<nd3;i++) r[i]=b[i]-Ap[i];
      precond(r,z);
      pd=z; double rz=dot(r,z), bn=sqrt(dot(b,b)); if(bn==0) bn=1;
      int it2=0, maxit=getenv("MAXIT")?atoi(getenv("MAXIT")):400000;
      for(;it2<maxit;it2++){ applyA(pd,Ap);
        double pAp=dot(pd,Ap);
        if(!(pAp>0)){ printf("  PCG indefinite pAp=%.2e at it=%d\n",pAp,it2); break; }
        double al=rz/pAp;
        for(long i=0;i<nd3;i++){ u[i]+=al*pd[i]; r[i]-=al*Ap[i]; }
        double rn=sqrt(dot(r,r));
        if(getenv("DBG")&&(it2<12||it2%500==0)) printf("    [cg it=%d rres=%.3e rz=%.3e pAp=%.3e bn=%.3e]\n",it2,rn/bn,rz,pAp,bn);
        if(rn<=tol*bn){ it2++; break; }
        precond(r,z);
        double rz2=dot(r,z), be=rz2/rz; rz=rz2;
        for(long i=0;i<nd3;i++) pd[i]=z[i]+be*pd[i]; }
      if(getenv("DBG")){ std::vector<double> tr(nd3); applyA(u,tr);
        double s2=0,b2=0; for(long i=0;i<nd3;i++){ double t=b[i]-tr[i]; s2+=t*t; b2+=b[i]*b[i]; }
        printf("  [cg done] it=%d  TRUE ||b-Au||/||b|| = %.3e\n",it2,sqrt(s2/(b2+1e-300))); }
      uOut=u; itOut=it2;
    } else {
    const int m=getenv("GM")?atoi(getenv("GM")):200;
    std::vector<std::vector<double>> V(m+1,std::vector<double>(nd3));
    std::vector<double> Hm((m+1)*m,0.0),cs(m,0.0),sn(m,0.0),ss(m+1,0.0),yv(m,0.0),w(nd3),Ax(nd3),r(nd3);
    std::vector<double> u(nd3,0.0);
    double bn=sqrt(dot(b,b)); if(bn==0) bn=1; int it=0; bool conv=false; double beta0=0;
    for(int outer=0;outer<2000&&!conv;outer++){
      applyA(u,Ax); { std::vector<double> t0(nd3); for(long i=0;i<nd3;i++) t0[i]=b[i]-Ax[i]; precond(t0,r); }
      double beta=sqrt(dot(r,r)); if(outer==0) beta0=beta>0?beta:1;
      if(getenv("DBG")) printf("    [outer=%d it=%d pres=%.3e]\n",outer,it,beta/beta0);
      if(beta<=tol*beta0){ conv=true; break; }
      for(long i=0;i<nd3;i++) V[0][i]=r[i]/beta;
      std::fill(ss.begin(),ss.end(),0.0); ss[0]=beta; int jj=0;
      for(int j=0;j<m;j++){ jj=j; it++;
        applyA(V[j],Ax); precond(Ax,w);
        for(int i=0;i<=j;i++){ double hij=dot(w,V[i]); Hm[i*m+j]=hij;
          for(long q=0;q<nd3;q++) w[q]-=hij*V[i][q]; }
        double hj1=sqrt(dot(w,w)); Hm[(j+1)*m+j]=hj1;
        if(hj1>1e-300) for(long q=0;q<nd3;q++) V[j+1][q]=w[q]/hj1;
        for(int i=0;i<j;i++){ double t=cs[i]*Hm[i*m+j]+sn[i]*Hm[(i+1)*m+j];
          Hm[(i+1)*m+j]=-sn[i]*Hm[i*m+j]+cs[i]*Hm[(i+1)*m+j]; Hm[i*m+j]=t; }
        double d0=Hm[j*m+j],d1=Hm[(j+1)*m+j],rr2=sqrt(d0*d0+d1*d1); if(rr2<1e-300) rr2=1e-300;
        cs[j]=d0/rr2; sn[j]=d1/rr2; Hm[j*m+j]=cs[j]*d0+sn[j]*d1; Hm[(j+1)*m+j]=0;
        double t=cs[j]*ss[j]; ss[j+1]=-sn[j]*ss[j]; ss[j]=t;
        if(fabs(ss[j+1])<=tol*beta0){ break; } }
      int sz=jj+1;
      for(int i=sz-1;i>=0;i--){ double s2=ss[i];
        for(int k=i+1;k<sz;k++) s2-=Hm[i*m+k]*yv[k];
        yv[i]=s2/Hm[i*m+i]; }
      for(int i=0;i<sz;i++) for(long q=0;q<nd3;q++) u[q]+=yv[i]*V[i][q]; }
    uOut=u; itOut=it;
    }

    // ---- L2 error over the surrogate (interior cells) ----
    double l2e=0,l2n=0;
    #pragma omp parallel for schedule(dynamic,64) reduction(+:l2e,l2n)
    for(int e=0;e<nE;e++){ const int*nod=&eNodes[(size_t)e*ndof];
      double x0=lo3[0]+eCx[e]*h,y0=lo3[1]+eCy[e]*h,z0=lo3[2]+eCz[e]*h; real vb[QN_MAX*QN_MAX*QN_MAX];
      for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++){
        real xr[3]={B.t[i],B.t[j],B.t[k]}; B.allVal(xr,vb); double w=B.wq[i]*B.wq[j]*B.wq[k]*h*h*h;
        double uh[3]={0,0,0}; for(int a=0;a<ndof;a++)for(int l=0;l<3;l++) uh[l]+=uOut[3*nod[a]+l]*vb[a];
        double ue[3]; uex(x0+B.t[i]*h,y0+B.t[j]*h,z0+B.t[k]*h,ue);
        for(int l=0;l<3;l++){ double dd=uh[l]-ue[l]; l2e+=dd*dd*w; l2n+=ue[l]*ue[l]*w; } } }
    l2e=sqrt(l2e); l2n=sqrt(l2n);
    SbmOut out; out.l2rel=(l2n>0)?l2e/l2n:l2e; out.l2abs=l2e; out.nd3=nd3; out.iters=itOut;
    out.nBF=nBF; out.nE=nE; out.h=h; return out;
}
#endif
