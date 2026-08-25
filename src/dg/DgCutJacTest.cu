//
// Tier-0 diagnostic: the cut element's own Jacobian, and the rule mismatch.
//
// Two questions, both answered without touching the solver:
//
//  D1/D2  Does the RUNTIME face rule equal the rule the GCL correction was
//         FITTED against?  cutElemBuild corrects the volume weights so that
//              SUM_q w_q d(psi~_m)/dxi_d  ==  SUM_faces sg INT psi~_m
//                                            + INT_wall psi~_m n_d
//         with the Saye face rules on the right.  dgRhsCutKernel does NOT use
//         those rules on a face that is fully fluid with a non-cut neighbour:
//         it substitutes a tensor GLL mortar (DgSolverKernels.cu:6242-6285).
//         If the two disagree the corrected identity is void AT RUNTIME even
//         though every host gate passes -- which is exactly the observed
//         signature (host free stream 1e-10, solver seed 1e-8..1e-4 and a
//         quadrature-INVARIANT growth rate).
//
//  J      The element-local Jacobian A = dR~/dc~ about the free stream, with
//         the neighbour traces FROZEN.  If max Re lambda(A) reproduces the
//         measured +13/time the instability is element-local and a modal
//         damping can reach it; if A is stable the instability lives in the
//         COUPLING and no element-local operator can fix it.
//         A is written to a CSV for the eigen-analysis (scripts/cut_jac.py).
//
// Everything is in the solver's REFERENCE measure with h = (0.25,0.25,0.25):
// volume, face and wall terms each carry exactly one 1/h_d, as the kernel does.
//
// build:  make dgcutjac_test
// run  :  ./dgcutjac_test [outdir]
//

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Util.cuh"
#include "Poly.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "CutElem.h"
#include "LagrangeBasis.h"

static constexpr i32 ARENA = 1<<22, SCRATCH = 1<<19;
static constexpr double GAM = 1.4;

static void p2c(const double W[5], double U[5]) {
  U[0]=W[0]; U[1]=W[0]*W[1]; U[2]=W[0]*W[2]; U[3]=W[0]*W[3];
  U[4]=W[4]/(GAM-1.0)+0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
}
static void c2p(const double U[5], double W[5]) {
  W[0]=fmax(U[0],1e-14); W[1]=U[1]/W[0]; W[2]=U[2]/W[0]; W[3]=U[3]/W[0];
  W[4]=(GAM-1.0)*(U[4]-0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/W[0]);
}
static void fluxAxis(const double W[5], i32 dir, double F[5]) {
  double un=W[1+dir];
  double E=W[4]/(GAM-1.0)+0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
  F[0]=W[0]*un; F[1]=W[0]*un*W[1]; F[2]=W[0]*un*W[2]; F[3]=W[0]*un*W[3];
  F[1+dir]+=W[4]; F[4]=(E+W[4])*un;
}
static void rusanovAxis(const double WL[5], const double WR[5], i32 dir, double F[5]) {
  double UL[5],UR[5],FL[5],FR[5]; p2c(WL,UL); p2c(WR,UR);
  fluxAxis(WL,dir,FL); fluxAxis(WR,dir,FR);
  double cL=sqrt(fmax(GAM*WL[4]/WL[0],1e-30)), cR=sqrt(fmax(GAM*WR[4]/WR[0],1e-30));
  double lam=fmax(fabs(WL[1+dir])+cL, fabs(WR[1+dir])+cR);
  for (i32 q=0;q<5;q++) F[q]=0.5*(FL[q]+FR[q])-0.5*lam*(UR[q]-UL[q]);
}

// forward solve L psi~ = psi -- the device's dgCutSolveL, host double
static void solveL(const std::vector<double> &L, i32 nb, double *v) {
  for (i32 i=0;i<nb;i++){ double t=v[i];
    for (i32 j=0;j<i;j++) t -= L[(size_t)i*nb+j]*v[j];
    v[i] = t/L[(size_t)i*nb+i]; }
}

int main(int argc, char **argv) {
  const char *outdir = (argc>1) ? argv[1] : ".";
  const i32 p=DG_ORDER, n=p+1;
  const double h[3] = {0.25, 0.25, 0.25};     // pseudo2D: domainSize[2]=hElem
  const double cx=1.5, cy=2.0, R=0.5;
  LagrangeBasis GL; GL.init(p);
  std::vector<SayeNode> ab(ARENA), sc(SCRATCH);
  SayeArena ar; ar.buf=ab.data(); ar.cap=ARENA; ar.top=0;
  SayeCfg cfg=SayeCfg::def();
  if (const char *e=getenv("CUT_NG")) cfg.ng=atoi(e);

  // the free stream of the growth case: M=3, a=1, rho=1
  double Wf[5] = {1.0, 3.0, 0.0, 0.0, 1.0/GAM};
  double U0[5]; p2c(Wf,U0);

  struct Cell { i32 ib, jb; const char *name; };
  Cell cells[] = { {6,6,"wedge (6,6)"}, {7,6,"quarter (7,6)"} };
  const i32 degs[] = {2,3};

  printf("cut-element Jacobian / rule-mismatch probe   h=%.3f  R=%.3f  ng=%d\n",
         h[0], R, cfg.ng);
  printf("free stream rho=%.3f u=%.3f p=%.5f  (a=1, M=3, lam=%.3f, lam/h=%.2f)\n\n",
         Wf[0], Wf[1], Wf[4], fabs(Wf[1])+1.0, (fabs(Wf[1])+1.0)/h[0]);

  for (const Cell &c : cells) for (i32 N : degs) {
    std::vector<real> v((size_t)n*n*n);
    for (i32 k=0;k<n;k++) for (i32 j=0;j<n;j++) for (i32 i=0;i<n;i++) {
      const double X=(c.ib+GL.t[i])*h[0], Y=(c.jb+GL.t[j])*h[1];
      v[i+n*(j+n*k)] = (real)(-(sqrt((X-cx)*(X-cx)+(Y-cy)*(Y-cy))-R));
    }
    PolyND phi=fitPoly3(p,v.data());
    // CUT_QUALTOL: raising it past every achievable bndIncons DISABLES the
    // tangency perturbation retry, which is required for any A/B comparison of
    // face rules -- otherwise the two arms build DIFFERENT geometries.
    const double qtol = getenv("CUT_QUALTOL") ? atof(getenv("CUT_QUALTOL")) : 1e-6;
    CutElemOps E;
    if (!cutElemBuild(phi,N,E,ar,cfg,sc,qtol)) { printf("%-14s N=%d BUILD FAILED\n",c.name,N); continue; }
    if (E.snap) { printf("%-14s N=%d SNAPPED\n",c.name,N); continue; }

    // ---- OPTIONAL REPAIR (CUT_GLLFACE=1): fit the build against the rule the
    // SOLVER actually uses.  dgRhsCutKernel replaces E.face[f] with the tensor
    // GLL rule on any fully-fluid face with a non-cut neighbour, so the GCL
    // correction is fitted against a rule that is then not used.  faceOverride
    // already exists for the cut<->cut shared-rule case; feeding it the GLL
    // tensor rule on full faces makes build and run agree exactly.
    const bool gllFit = getenv("CUT_GLLFACE") && atoi(getenv("CUT_GLLFACE"));
    std::vector<SayeNode> ovS[6];
    if (gllFit) {
      const std::vector<SayeNode> *ovP[6]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
      bool any=false;
      for (i32 f=0; f<6; f++) {
        double a=0; for (const SayeNode &s : E.face[f]) a+=(double)s.w;
        if (fabs(a-1.0) > 1e-6) continue;                 // not a full face
        const i32 d=f/2, side=f%2, t1=(d==0)?1:0, t2=(d==2)?1:2;
        ovS[f].clear();
        for (i32 fb=0; fb<n; fb++) for (i32 fa=0; fa<n; fa++) {
          SayeNode s{};
          s.x[d]=(real)(side?1.0:0.0);
          s.x[t1]=GL.t[fa]; s.x[t2]=GL.t[fb];
          s.w=(real)((double)GL.qw[fa]*(double)GL.qw[fb]);
          ovS[f].push_back(s);
        }
        ovP[f]=&ovS[f]; any=true;
      }
      if (any) {
        CutElemOps E2;
        if (cutElemBuild(phi,N,E2,ar,cfg,sc,qtol,ovP) && !E2.snap) E = std::move(E2);
        else printf("%-14s N=%d  GLL-face rebuild FAILED, keeping Saye build\n",c.name,N);
      }
    }
    const i32 nb=E.B.nb;
    std::vector<double> L(E.Mchol.begin(), E.Mchol.end());     // nb x nb lower

    auto psiO=[&](const double X[3], double *psi){ E.B.eval(X,psi); solveL(L,nb,psi); };
    auto gradO=[&](const double X[3], double *dpsi){
      E.B.grad(X,dpsi);
      std::vector<double> col(nb);
      for (i32 d=0;d<3;d++){ for(i32 m=0;m<nb;m++) col[m]=dpsi[3*m+d];
        solveL(L,nb,col.data()); for(i32 m=0;m<nb;m++) dpsi[3*m+d]=col[m]; } };

    // ---- D1: per-face fluid fraction, and the Saye-vs-GLL moment mismatch --
    // both rules integrate over the SAME reference face measure (total 1 on a
    // full face), so the moments of psi~_m must agree if the substitution is
    // to be legitimate.
    double facA[6], mism[6]; i32 nFull=0;
    std::vector<double> psi(nb), mS(nb), mG(nb);
    for (i32 f=0; f<6; f++) {
      const i32 d=f/2, side=f%2;
      facA[f]=0; for (const SayeNode &s : E.face[f]) facA[f]+=(double)s.w;
      std::fill(mS.begin(),mS.end(),0.0);
      for (const SayeNode &s : E.face[f]) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        psiO(X,psi.data());
        for (i32 m=0;m<nb;m++) mS[m]+=(double)s.w*psi[m];
      }
      std::fill(mG.begin(),mG.end(),0.0);
      const i32 t1=(d==0)?1:0, t2=(d==2)?1:2;
      for (i32 fb=0; fb<n; fb++) for (i32 fa=0; fa<n; fa++) {
        double X[3]; X[d]=side?1.0:0.0;
        X[t1]=(double)GL.t[fa]; X[t2]=(double)GL.t[fb];
        psiO(X,psi.data());
        const double w2d=(double)GL.qw[fa]*(double)GL.qw[fb];
        for (i32 m=0;m<nb;m++) mG[m]+=w2d*psi[m];
      }
      mism[f]=0; for (i32 m=0;m<nb;m++) mism[f]=fmax(mism[f],fabs(mS[m]-mG[m]));
      if (fabs(facA[f]-1.0)<1e-6) nFull++;
    }

    // ---- D2: the reference GCL residual under each face rule ---------------
    // G[m][d] = SUM_vol w dpsi~_m/dxi_d
    //         - ( SUM_f sg SUM_face w psi~_m  +  SUM_wall w n_d psi~_m )
    auto gclResid=[&](bool useGll){
      std::vector<double> G((size_t)nb*3,0.0), dpsi((size_t)nb*3);
      for (const SayeNode &s : E.vol) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        gradO(X,dpsi.data());
        for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++) G[(size_t)m*3+d]+=(double)s.w*dpsi[3*m+d];
      }
      for (i32 f=0;f<6;f++) {
        const i32 d=f/2, side=f%2; const double sg=side?1.0:-1.0;
        const bool gll = useGll && fabs(facA[f]-1.0)<1e-6;
        if (gll) {
          const i32 t1=(d==0)?1:0, t2=(d==2)?1:2;
          for (i32 fb=0; fb<n; fb++) for (i32 fa=0; fa<n; fa++) {
            double X[3]; X[d]=side?1.0:0.0;
            X[t1]=(double)GL.t[fa]; X[t2]=(double)GL.t[fb];
            psiO(X,psi.data());
            const double w2d=(double)GL.qw[fa]*(double)GL.qw[fb];
            for (i32 m=0;m<nb;m++) G[(size_t)m*3+d]-=sg*w2d*psi[m];
          }
        } else {
          for (const SayeNode &s : E.face[f]) {
            double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
            psiO(X,psi.data());
            for (i32 m=0;m<nb;m++) G[(size_t)m*3+d]-=sg*(double)s.w*psi[m];
          }
        }
      }
      for (const SayeNode &s : E.wall) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        psiO(X,psi.data());
        for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++)
          G[(size_t)m*3+d]-=(double)s.w*(double)s.n[d]*psi[m];
      }
      double w=0; for (size_t t=0;t<G.size();t++) w=fmax(w,fabs(G[t]));
      return w;
    };
    const double gSaye=gclResid(false), gGll=gclResid(true);

    // ---- the RHS the device computes, as a function of the modal state -----
    // wallMode: 0 = transparent (--cutfsp), 1 = mirror Riemann (--cutwallriem)
    // faceGll : true = substitute the tensor GLL mortar on full fluid faces
    bool faceSelf=false;
    auto rhs=[&](const double *cM, double *Rm, bool faceGll, i32 wallMode){
      std::fill(Rm, Rm+(size_t)nb*5, 0.0);
      std::vector<double> ps(nb), dps((size_t)nb*3);
      // volume
      for (const SayeNode &s : E.vol) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        E.B.eval(X,ps.data()); solveL(L,nb,ps.data());
        gradO(X,dps.data());
        double U[5],W[5];
        for (i32 q=0;q<5;q++){ double t=0; for(i32 m=0;m<nb;m++) t+=cM[m*5+q]*ps[m]; U[q]=t; }
        c2p(U,W);
        for (i32 d=0;d<3;d++){ double F[5]; fluxAxis(W,d,F); const double jac=1.0/h[d];
          for (i32 m=0;m<nb;m++) for (i32 q=0;q<5;q++) Rm[m*5+q]+=(double)s.w*F[q]*dps[3*m+d]*jac; }
      }
      // wall
      for (const SayeNode &s : E.wall) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        E.B.eval(X,ps.data()); solveL(L,nb,ps.data());
        double U[5],W[5];
        for (i32 q=0;q<5;q++){ double t=0; for(i32 m=0;m<nb;m++) t+=cM[m*5+q]*ps[m]; U[q]=t; }
        c2p(U,W);
        double nt[3]={(double)s.n[0]/h[0],(double)s.n[1]/h[1],(double)s.n[2]/h[2]};
        double nm=sqrt(nt[0]*nt[0]+nt[1]*nt[1]+nt[2]*nt[2]);
        if (nm<=0) continue;
        double np[3]={nt[0]/nm,nt[1]/nm,nt[2]/nm}, dS=(double)s.w*nm, Fw[5];
        if (wallMode==0) { double Fx[5],Fy[5],Fz[5];
          fluxAxis(W,0,Fx); fluxAxis(W,1,Fy); fluxAxis(W,2,Fz);
          for (i32 q=0;q<5;q++) Fw[q]=Fx[q]*np[0]+Fy[q]*np[1]+Fz[q]*np[2];
        } else { const double un=W[1]*np[0]+W[2]*np[1]+W[3]*np[2];
          const double lam=fabs(un)+sqrt(fmax(GAM*W[4]/W[0],1e-30));
          const double pw=W[4]+W[0]*un*un+lam*W[0]*un;
          Fw[0]=0; Fw[1]=pw*np[0]; Fw[2]=pw*np[1]; Fw[3]=pw*np[2]; Fw[4]=0; }
        for (i32 m=0;m<nb;m++) for (i32 q=0;q<5;q++) Rm[m*5+q]-=dS*Fw[q]*ps[m];
      }
      // faces -- neighbour trace FROZEN at the free stream (isolated element)
      double Wo[5]; { double Uo[5]; for(i32 q=0;q<5;q++) Uo[q]=U0[q]; c2p(Uo,Wo); }
      for (i32 f=0;f<6;f++) {
        const i32 d=f/2, side=f%2; const double sg=side?1.0:-1.0;
        const bool gll = faceGll && fabs(facA[f]-1.0)<1e-6;
        auto one=[&](const double X[3], double wq){
          E.B.eval(X,ps.data()); solveL(L,nb,ps.data());
          double U[5],W[5];
          for (i32 q=0;q<5;q++){ double t=0; for(i32 m=0;m<nb;m++) t+=cM[m*5+q]*ps[m]; U[q]=t; }
          c2p(U,W);
          double fs[5];
          if (faceSelf)   fluxAxis(W,d,fs);
          else if (side)  rusanovAxis(W,Wo,d,fs);
          else            rusanovAxis(Wo,W,d,fs);
          for (i32 m=0;m<nb;m++) for (i32 q=0;q<5;q++) Rm[m*5+q]-=sg*wq*fs[q]*ps[m]/h[d];
        };
        if (gll) {
          const i32 t1=(d==0)?1:0, t2=(d==2)?1:2;
          for (i32 fb=0; fb<n; fb++) for (i32 fa=0; fa<n; fa++) {
            double X[3]; X[d]=side?1.0:0.0;
            X[t1]=(double)GL.t[fa]; X[t2]=(double)GL.t[fb];
            one(X, (double)GL.qw[fa]*(double)GL.qw[fb]);
          }
        } else for (const SayeNode &s : E.face[f]) {
          double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
          one(X, (double)s.w);
        }
      }
    };

    // uniform state in the orthonormal frame: u = U0 => c~ = L^T c with
    // c = (U0,0,..) in the monomial frame, so c~_m = L[0][m]... but L is lower
    // triangular, so only c~_0 = L00*U0 survives.
    std::vector<double> c0((size_t)nb*5,0.0);
    for (i32 q=0;q<5;q++) c0[0*5+q]=L[0]*U0[q];
    std::vector<double> R0((size_t)nb*5);

    double fsp=0, fspG=0;
    rhs(c0.data(),R0.data(),false,0);
    for (size_t t=0;t<R0.size();t++) fsp=fmax(fsp,fabs(R0[t]));
    rhs(c0.data(),R0.data(),true,0);
    for (size_t t=0;t<R0.size();t++) fspG=fmax(fspG,fabs(R0[t]));

    printf("%-14s N=%d  nb=%d  vol=%.4e (frac %.4f)  nFull=%d\n",
           c.name, N, nb, E.volume, E.volume/1.0, nFull);
    printf("   face fluid fraction : "); for (i32 f=0;f<6;f++) printf("%7.4f ",facA[f]); printf("\n");
    printf("   1 - facA (full only): ");
    for (i32 f=0;f<6;f++) { if (fabs(facA[f]-1.0)<1e-6) printf("%7.1e ",1.0-facA[f]); else printf("%7s ","-"); }
    printf("\n   npts Saye face      : "); for (i32 f=0;f<6;f++) printf("%7zu ",E.face[f].size()); printf("\n");
    printf("   Saye-vs-GLL moment  : "); for (i32 f=0;f<6;f++) printf("%7.1e ",mism[f]); printf("\n");
    // the CONSTANT-mode closure: CLOSED INT n_d dS must be 0 for ANY closed
    // surface.  This is the one row of the GCL no volume-weight correction can
    // touch, and it is a statement about the boundary rules ALONE -- so it is
    // the clean way to ask whether a set of rules closes.
    auto closure=[&](bool useGll){
      double cl[3]={0,0,0};
      for (i32 f=0;f<6;f++){ const i32 d=f/2, side=f%2; const double sg=side?1.0:-1.0;
        if (useGll && fabs(facA[f]-1.0)<1e-6) { cl[d]+=sg*1.0; continue; }   // exact
        double a=0; for (const SayeNode &s : E.face[f]) a+=(double)s.w; cl[d]+=sg*a; }
      for (const SayeNode &s : E.wall)
        for (i32 d=0;d<3;d++) cl[d]+=(double)s.w*(double)s.n[d];
      double w=0; for(i32 d=0;d<3;d++) w=fmax(w,fabs(cl[d])); return w; };
    printf("   GCL residual        : Saye rules %.3e   RUNTIME (GLL on full) %.3e\n", gSaye, gGll);
    printf("   bndIncons           : %.3e   (uncorrectable floor of the GCL)\n", E.bndIncons);
    printf("   CLOSED INT n_d dS   : Saye rules %.3e   GLL on full faces     %.3e\n",
           closure(false), closure(true));
    printf("   free-stream |R~|max : Saye rules %.3e   RUNTIME (GLL on full) %.3e\n", fsp, fspG);

    // ---- CUT_EVOLVE: the matching run to DgEsCutTest's ES_EVOLVE ----------
    // Same element, same free stream, same deterministic seed, same dt, same
    // SSP-RK3, and the same CLOSED configuration (neighbour trace = own trace,
    // so no exterior dissipation enters and the two operators are compared on
    // equal terms).  The entropy-stable operator is entropy CONSERVATIVE, so
    // its response to the constant truncation forcing is POLYNOMIAL; if this
    // one is exponential, that difference is the whole argument.
    if (getenv("CUT_EVOLVE")) {
      faceSelf = true;
      std::vector<double> cs(c0);
      for (i32 m = 1; m < nb; m++)
        for (i32 q = 0; q < 5; q++)
          cs[(size_t)m*5+q] = 1e-6*fabs(c0[q])*((m%2) ? 1.0 : -1.0);
      const double dt = getenv("ES_DT") ? atof(getenv("ES_DT")) : 2e-4;
      const i32 nStep = getenv("ES_NSTEP") ? atoi(getenv("ES_NSTEP")) : 5000;
      auto dev = [&](const std::vector<double> &x){ double a=0;
        for (size_t t2=0;t2<x.size();t2++) a += (x[t2]-c0[t2])*(x[t2]-c0[t2]);
        return sqrt(a); };
      std::vector<double> k1(nb*5), k2(nb*5), k3(nb*5), tmp(nb*5);
      const double d0 = dev(cs);
      printf("   [CUT_EVOLVE] baseline RHS  dt=%.1e nStep=%d  ||dc||_0 = %.3e\n", dt, nStep, d0);
      bool blew = false;
      for (i32 it = 0; it < nStep && !blew; it++) {
        rhs(cs.data(), k1.data(), true, 0);
        for (i32 t2=0;t2<nb*5;t2++) tmp[t2] = cs[t2] + dt*k1[t2];
        rhs(tmp.data(), k2.data(), true, 0);
        for (i32 t2=0;t2<nb*5;t2++) tmp[t2] = cs[t2] + 0.25*dt*(k1[t2]+k2[t2]);
        rhs(tmp.data(), k3.data(), true, 0);
        for (i32 t2=0;t2<nb*5;t2++) cs[t2] += dt*(k1[t2]+k2[t2]+4.0*k3[t2])/6.0;
        for (i32 t2=0;t2<nb*5;t2++) if (!std::isfinite(cs[t2])) blew = true;
        if (((it+1) % (nStep/5)) == 0 || blew) {
          const double dd = dev(cs), T = (it+1)*dt;
          printf("      t=%7.4f   ||dc|| = %.4e   ratio %.3e   rate %+8.2f/time%s\n",
                 T, dd, dd/d0, (dd>0&&d0>0)?log(dd/d0)/T:0.0, blew?"   NON-FINITE":"");
        }
      }
      faceSelf = false;
    }

    // ---- the Jacobian, central differences, RUNTIME rules ------------------
    const i32 nu=nb*5;
    struct Mode { const char *tag; bool self; i32 wall; };
    const Mode modes[] = { {"fsp",false,0}, {"riem",false,1}, {"closed",true,0} };
    for (const Mode &md : modes) {
    faceSelf = md.self;
    std::vector<double> A((size_t)nu*nu), cp(c0), Rp(nu), Rm2(nu);
    for (i32 j=0;j<nu;j++) {
      const double sca=fmax(fabs(c0[j]),1.0), eps=1e-6*sca;
      cp=c0; cp[j]+=eps; rhs(cp.data(),Rp.data(),true,md.wall);
      cp=c0; cp[j]-=eps; rhs(cp.data(),Rm2.data(),true,md.wall);
      for (i32 i=0;i<nu;i++) A[(size_t)i*nu+j]=(Rp[i]-Rm2[i])/(2.0*eps);
    }
    char fn[512];
    snprintf(fn,sizeof fn,"%s/cutjac_%c%d_N%d_%s.csv", outdir, c.name[0], c.ib*10+c.jb, N, md.tag);
    FILE *fp=fopen(fn,"w");
    if (fp) {
      // header: mode degree map, so the eigenvector spectrum can be binned
      fprintf(fp,"# nb=%d nu=%d h=%.6f\n#deg:",nb,nu,h[0]);
      for (i32 m=0;m<nb;m++){ i32 e[3]; E.B.expo(m,e); fprintf(fp,"%d ",e[0]+e[1]+e[2]); }
      fprintf(fp,"\n#e2:");
      for (i32 m=0;m<nb;m++){ i32 e[3]; E.B.expo(m,e); fprintf(fp,"%d ",e[2]); }
      fprintf(fp,"\n");
      for (i32 i=0;i<nu;i++){ for(i32 j=0;j<nu;j++) fprintf(fp,"%.16e%c",A[(size_t)i*nu+j], j==nu-1?'\n':',');}
      fclose(fp);
      printf("   Jacobian[%-6s] -> %s  (%dx%d)\n", md.tag, fn, nu, nu);
    }
    }
    faceSelf=false;
    printf("\n");
  }
  return 0;
}
