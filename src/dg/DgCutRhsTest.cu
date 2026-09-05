//
// S2 gate (scheme): the cut-element DG RHS for the Euler equations.
//
// Assembles the full nonlinear RHS on a cut element -- volume flux term over the
// fitted rule, numerical flux over each cut face, wall flux over the Saye
// surface rule -- and checks the two states a correct scheme must hold EXACTLY:
//
//   A. FREE STREAM with a transparent wall (wall flux = the exact F.n).
//      Uniform flow in every direction.  This isolates GEOMETRIC consistency:
//      it is zero iff SUM_q w_q d(psi_m)/dx_d == CLOSED INT psi_m n_d, which is
//      what S1's correction was built to deliver.
//
//   B. STAGNANT state with a REFLECTIVE wall (pressure-only flux).
//      rho, p uniform and u = 0 is a genuine steady solution of the Euler
//      equations with a solid wall, so the RHS must vanish.  This is the
//      physical test -- note free stream is NOT a solid-wall solution, which is
//      why A needs the transparent wall.
//
// Both use a uniform state, so a neighbour's trace equals the element's own and
// the face coupling is exercised without needing a full mesh.  Conservation
// under a VARYING state needs real neighbour exchange and is checked once this
// is wired into the solver.
//
// build:  make dgcutrhs_test
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
#include "CutElem.h"
#include "LagrangeBasis.h"

static constexpr i32 ARENA = 1<<21, SCRATCH = 1<<18;
static constexpr double GAM = 1.4;

// primitive W = [rho, u, v, w, p] -> conservative U
static void p2c(const double W[5], double U[5]) {
  U[0]=W[0]; U[1]=W[0]*W[1]; U[2]=W[0]*W[2]; U[3]=W[0]*W[3];
  U[4]=W[4]/(GAM-1.0)+0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
}
static void c2p(const double U[5], double W[5]) {
  W[0]=U[0]; W[1]=U[1]/U[0]; W[2]=U[2]/U[0]; W[3]=U[3]/U[0];
  W[4]=(GAM-1.0)*(U[4]-0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/U[0]);
}
// Euler flux along axis dir, from primitives
static void fluxAxis(const double W[5], i32 dir, double F[5]) {
  double un=W[1+dir];
  double E=W[4]/(GAM-1.0)+0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
  F[0]=W[0]*un; F[1]=W[0]*un*W[1]; F[2]=W[0]*un*W[2]; F[3]=W[0]*un*W[3];
  F[1+dir]+=W[4]; F[4]=(E+W[4])*un;
}
// flux projected on an arbitrary normal
static void fluxNormal(const double W[5], const double n[3], double F[5]) {
  double Fd[5];
  for (i32 q=0;q<5;q++) F[q]=0;
  for (i32 d=0;d<3;d++){ fluxAxis(W,d,Fd); for(i32 q=0;q<5;q++) F[q]+=Fd[q]*n[d]; }
}
// Rusanov along an axis (conservative form)
static void rusanovAxis(const double WL[5], const double WR[5], i32 dir, double F[5]) {
  double UL[5],UR[5],FL[5],FR[5]; p2c(WL,UL); p2c(WR,UR);
  fluxAxis(WL,dir,FL); fluxAxis(WR,dir,FR);
  double cL=sqrt(GAM*WL[4]/WL[0]), cR=sqrt(GAM*WR[4]/WR[0]);
  double lam=fmax(fabs(WL[1+dir])+cL, fabs(WR[1+dir])+cR);
  for (i32 q=0;q<5;q++) F[q]=0.5*(FL[q]+FR[q])-0.5*lam*(UR[q]-UL[q]);
}

int main(void) {
  const i32 p=DG_ORDER, n=p+1, nd=n*n*n, N=p;
  LagrangeBasis GL; GL.init(p);
  std::vector<SayeNode> ab(ARENA), sc(SCRATCH);
  SayeArena ar; ar.buf=ab.data(); ar.cap=ARENA; ar.top=0;
  SayeCfg cfg=SayeCfg::def(); cfg.ng=10;

  struct Case { double cx,cy,cz,R; const char *name; };
  Case cases[] = {
    {0.5,0.5,0.5, 0.75, "centred, big cut"},
    {0.0,0.0,0.0, 1.20, "corner cut"},
    {1.3,0.5,0.5, 1.00, "oblique, off-axis"},
    {0.0,0.0,0.0, 1.70, "sliver (tiny solid)"},
    {0.5,0.5,0.5, 0.52, "near-face tangency (geom-limited)"},
  };

  // uniform states
  double Wfree[5] = {1.225, 0.8, -0.35, 0.22, 1.0e5};   // free stream
  double Wstag[5] = {1.225, 0.0,  0.0,  0.0,  1.0e5};   // stagnant

  printf("cut-element Euler RHS   N=%d, %d modes\n", N, CutBasis::count(N));
  printf("      flux imbalance (gated)          |  reported only\n");
  printf("%-34s %11s %11s %10s %9s %9s\n", "case",
         "A free-str", "B stagnant", "bndIncons", "kappa(M)", "drift");

  bool allok=true;
  for (const Case &c : cases) {
    std::vector<real> v(nd);
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      double X=GL.t[i]-c.cx, Y=GL.t[j]-c.cy, Z=GL.t[k]-c.cz;
      v[i+n*(j+n*k)]=(real)(c.R*c.R-(X*X+Y*Y+Z*Z));
    }
    PolyND phi=fitPoly3(p,v.data());
    CutElemOps E;
    if (!cutElemBuild(phi,N,E,ar,cfg,sc)) { printf("%-34s BUILD FAILED\n",c.name); allok=false; continue; }
    // snapped == not a cut element; no operators exist to test (see CutElem.h)
    if (E.snap) { printf("%-34s SNAPPED to %s (sub-resolution)\n",
                         c.name, E.snap==1?"FLUID":"SOLID"); continue; }
    const i32 nb=E.B.nb;

    // modal coefficients of a UNIFORM conservative state: only the constant
    // mode is nonzero, and psi_0 == 1, so c[0] = U and the rest vanish.
    // total boundary measure -- the natural scale for a flux imbalance
    double Abnd=E.wallArea;
    for (i32 f=0;f<6;f++) for (const SayeNode &s : E.face[f]) Abnd += (double)s.w;

    auto rhsUniform=[&](const double W[5], bool reflectiveWall,
                        double *imbalOut, double *driftOut){
      double U[5]; p2c(W,U);
      std::vector<double> R((size_t)nb*5, 0.0), psi(nb), dpsi((size_t)nb*3);

      // ---- volume:  + INT F_d(u) d(psi_m)/dx_d dV ------------------------
      double Fd[3][5];
      for (i32 d=0;d<3;d++) fluxAxis(W,d,Fd[d]);
      for (const SayeNode &s : E.vol) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        E.B.grad(X,dpsi.data());
        for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++)
          for (i32 q=0;q<5;q++) R[(size_t)m*5+q] += (double)s.w*Fd[d][q]*dpsi[3*m+d];
      }
      // ---- cut faces:  - CLOSED INT (F*.n) psi_m dS -----------------------
      // uniform state => the neighbour trace equals ours, so Rusanov reduces to
      // the physical flux; the dissipation term vanishes identically.
      for (i32 d=0;d<3;d++) for (i32 side=0;side<2;side++) {
        double sg = side?1.0:-1.0, Fs[5];
        rusanovAxis(W,W,d,Fs);
        for (const SayeNode &s : E.face[2*d+side]) {
          double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
          E.B.eval(X,psi.data());
          for (i32 m=0;m<nb;m++) for (i32 q=0;q<5;q++)
            R[(size_t)m*5+q] -= (double)s.w*sg*Fs[q]*psi[m];
        } }
      // ---- wall:  - INT (F_wall.n) psi_m dS ------------------------------
      for (const SayeNode &s : E.wall) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        double nr[3]={(double)s.n[0],(double)s.n[1],(double)s.n[2]};
        E.B.eval(X,psi.data());
        double Fw[5];
        if (reflectiveWall) {          // solid wall: pressure only, no mass flux
          Fw[0]=0; Fw[1]=W[4]*nr[0]; Fw[2]=W[4]*nr[1]; Fw[3]=W[4]*nr[2]; Fw[4]=0;
        } else {                       // transparent: the exact flux
          fluxNormal(W,nr,Fw);
        }
        for (i32 m=0;m<nb;m++) for (i32 q=0;q<5;q++)
          R[(size_t)m*5+q] -= (double)s.w*Fw[q]*psi[m];
      }
      double scale=0;
      for (i32 d=0;d<3;d++) for (i32 q=0;q<5;q++) scale=fmax(scale,fabs(Fd[d][q]));
      scale=fmax(scale,W[4]);

      // (1) FLUX IMBALANCE -- the scheme property.  R before M^-1 is the
      // residual of  INT F.grad(psi) - CLOSED INT (F*.n) psi, so scaling it by
      // |F| times the boundary measure asks exactly "does the discrete flux
      // balance close?".  This is free-stream preservation, and it is what S1's
      // correction was built to deliver.
      double imb=0;
      for (size_t t=0;t<(size_t)nb*5;t++) imb=fmax(imb,fabs(R[t]));
      *imbalOut = imb/(scale*fmax(Abnd,1e-300));

      // (2) SOLUTION DRIFT -- the same residual after M^-1, i.e. what actually
      // moves the state.  On an isolated awkward cut cell M is badly conditioned
      // and amplifies (1) by orders of magnitude.  That is a CONDITIONING
      // problem, not a discretisation one, and it is precisely what state
      // redistribution removes by merging such cells; reported, not gated.
      std::vector<double> col(nb);
      double worst=0;
      for (i32 q=0;q<5;q++){
        for (i32 m=0;m<nb;m++) col[m]=R[(size_t)m*5+q];
        E.massSolve(col.data());
        for (i32 m=0;m<nb;m++) worst=fmax(worst,fabs(col[m]));
      }
      *driftOut = worst/scale;
    };

    // conditioning of the cut mass matrix, from the Cholesky diagonal.
    // R before M^-1 is |F| * (GCL residual); anything bigger in the answer is
    // M^-1 amplification, and on an isolated awkward cut cell that is large.
    double dmin=1e300, dmax=0;
    for (i32 i=0;i<nb;i++){ double d=E.Mchol[(size_t)i*nb+i];
      dmin=fmin(dmin,d); dmax=fmax(dmax,d); }
    double kappa=(dmax/dmin)*(dmax/dmin);

    double rA=0, rB=0, dA=0, dB=0;
    rhsUniform(Wfree, false, &rA, &dA);   // A: free stream, transparent wall
    rhsUniform(Wstag, true,  &rB, &dB);   // B: stagnant, reflective wall
    // a geometry-limited cell can only reach its own boundary inconsistency
    double tol = fmax(1e-11, 20.0*E.bndIncons);
    bool ok = rA < tol && rB < tol;
    if (!ok) allok=false;
    printf("%-34s %11.3e %11.3e %10.2e %9.1e %9.1e %s\n", c.name, rA, rB,
           E.bndIncons, kappa, fmax(dA,dB), ok?"ok":"FAIL");
  }
  printf("\n%s\n", allok
    ? "S2(scheme) PASS -- the discrete flux balance closes for both states, so\n"
      "                   the cut RHS is free-stream preserving.  The `drift`\n"
      "                   column is the same residual amplified by kappa(M) on an\n"
      "                   ISOLATED awkward cell -- a conditioning problem that\n"
      "                   state redistribution removes by merging those cells."
    : "S2(scheme) FAIL");
  return allok?0:1;
}
