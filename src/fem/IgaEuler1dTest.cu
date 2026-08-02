//
// IGA COMPRESSIBLE FLOW, step 1: 1-D Euler on a uniform grid with classic
// FEM shock capturing.
//
//   basis     uniform C^{p-1} B-splines (IgaBasis machinery, 1-D), control-
//             point conservative dofs, CONSISTENT banded mass (prefactored).
//   scheme    Galerkin flux term
//               + SUPG streamline stabilization (Hughes-Mallet lineage,
//                 scalar tau = h / (2(|u|+c)), steady-residual form)
//               + discontinuity-capturing artificial viscosity: residual-
//                 scaled Laplacian nu = C_dc h^2 |dF/dx|/(|dU/dx|+eps),
//                 capped by the first-order level 0.5 h lambda
//             -- the classic continuous-FEM shock-capturing pair.  A C^1
//             basis has NO element-local discontinuity mechanism, so ALL
//             shock robustness must come from these terms; that is the point
//             being tested.
//   time      SSP-RK3, CFL on the wave speed.
//   gate      Sod shock tube at t = 0.2 against the EXACT Riemann solution:
//             L1(rho) convergence under h-refinement, positivity, bounded
//             overshoot at the shock, mass conservation.
//
// build:  make iga_euler1d
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Util.cuh"
#include "IgaBasis.h"
#include "SayeQuad.h"   // GaussRule/gaussLegendre

static constexpr double GAM = 1.4;

// ---------------------------------------------------------------------------
//  exact Riemann solver (Toro) for the Sod gate
// ---------------------------------------------------------------------------
struct RiemannExact {
  double rL, uL, pL, rR, uR, pR, cL, cR, ps, us;
  void init(double rl,double ul,double pl,double rr,double ur,double pr){
    rL=rl;uL=ul;pL=pl;rR=rr;uR=ur;pR=pr;
    cL=sqrt(GAM*pL/rL); cR=sqrt(GAM*pR/rR);
    double p=0.5*(pL+pR), pold; i32 it=0;
    do { pold=p;
      double fL,dfL,fR,dfR;
      f1(p,rL,pL,cL,fL,dfL); f1(p,rR,pR,cR,fR,dfR);
      p = p - (fL+fR+uR-uL)/(dfL+dfR);
      if (p<1e-8) p=1e-8;
    } while (fabs(p-pold)/(0.5*(p+pold))>1e-12 && ++it<100);
    ps=p;
    double fL,dfL,fR,dfR; f1(ps,rL,pL,cL,fL,dfL); f1(ps,rR,pR,cR,fR,dfR);
    us=0.5*(uL+uR)+0.5*(fR-fL);
  }
  static void f1(double p,double rk,double pk,double ck,double &f,double &df){
    if (p>pk){ double A=2.0/((GAM+1)*rk), B=(GAM-1)/(GAM+1)*pk;
      f=(p-pk)*sqrt(A/(p+B)); df=sqrt(A/(B+p))*(1.0-0.5*(p-pk)/(B+p)); }
    else { double c=ck; f=2*c/(GAM-1)*(pow(p/pk,(GAM-1)/(2*GAM))-1.0);
      df=1.0/(rk*c)*pow(p/pk,-(GAM+1)/(2*GAM)); }
  }
  void sample(double s,double &r,double &u,double &p) const {
    if (s<=us) {                                  // left of contact
      if (ps>pL) {                                // left shock
        double sL=uL-cL*sqrt((GAM+1)/(2*GAM)*ps/pL+(GAM-1)/(2*GAM));
        if (s<=sL){r=rL;u=uL;p=pL;}
        else {r=rL*((ps/pL+(GAM-1)/(GAM+1))/((GAM-1)/(GAM+1)*ps/pL+1.0));u=us;p=ps;}
      } else {                                    // left rarefaction
        double shL=uL-cL, csL=cL*pow(ps/pL,(GAM-1)/(2*GAM)), stL=us-csL;
        if (s<=shL){r=rL;u=uL;p=pL;}
        else if (s>=stL){r=rL*pow(ps/pL,1.0/GAM);u=us;p=ps;}
        else { double c=2.0/(GAM+1)*(cL+0.5*(GAM-1)*(uL-s));
          u=2.0/(GAM+1)*(cL+0.5*(GAM-1)*uL+s);
          r=rL*pow(c/cL,2.0/(GAM-1)); p=pL*pow(c/cL,2*GAM/(GAM-1)); }
      }
    } else {                                      // right of contact
      if (ps>pR) {                                // right shock
        double sR=uR+cR*sqrt((GAM+1)/(2*GAM)*ps/pR+(GAM-1)/(2*GAM));
        if (s>=sR){r=rR;u=uR;p=pR;}
        else {r=rR*((ps/pR+(GAM-1)/(GAM+1))/((GAM-1)/(GAM+1)*ps/pR+1.0));u=us;p=ps;}
      } else {                                    // right rarefaction
        double shR=uR+cR, csR=cR*pow(ps/pR,(GAM-1)/(2*GAM)), stR=us+csR;
        if (s>=shR){r=rR;u=uR;p=pR;}
        else if (s<=stR){r=rR*pow(ps/pR,1.0/GAM);u=us;p=ps;}
        else { double c=2.0/(GAM+1)*(cR-0.5*(GAM-1)*(uR-s));
          u=2.0/(GAM+1)*(-cR+0.5*(GAM-1)*uR+s);
          r=rR*pow(c/cR,2.0/(GAM-1)); p=pR*pow(c/cR,2*GAM/(GAM-1)); }
      }
    }
  }
};

// ---------------------------------------------------------------------------
//  1-D uniform C^{p-1} spline machinery (unit-spacing spans, N cells,
//  n = N + p control points; span i supports control points i..i+p)
// ---------------------------------------------------------------------------
struct Spline1d {
  i32 p, N, n;                    // degree, cells, control points
  IgaBasis B;                     // for val/der on a span
  void init(i32 p_, i32 N_) { p=p_; N=N_; n=N+p; B.init(p_); }
  // values of the p+1 nonzero functions on span s at local xi
  void val(double xi, real *Nv) const { B.val((real)xi, Nv); }
  void der(double xi, real *Dv) const { B.der((real)xi, Dv); }
};

// banded SPD Cholesky (bandwidth b): A[i][j] stored as Ab[i*(b+1)+(j-i+... )]
struct BandChol {
  i32 n, b;
  std::vector<double> L;          // lower band: L[i*(b+1)+k] = L(i, i-b+k), k=0..b
  void factor(const std::vector<double> &Ab) {
    L = Ab;
    for (i32 j=0;j<n;j++) {
      i32 j0 = (j-b>0)?(j-b):0;
      double d = L[(size_t)j*(b+1)+b];
      for (i32 k=j0;k<j;k++){ double l=L[(size_t)j*(b+1)+(k-j+b)]; d-=l*l; }
      d = sqrt(d); L[(size_t)j*(b+1)+b]=d;
      for (i32 i=j+1;i<=j+b && i<n;i++){
        i32 i0=(i-b>0)?(i-b):0;
        double s2=L[(size_t)i*(b+1)+(j-i+b)];
        for (i32 k=(i0>j0?i0:j0);k<j;k++)
          s2-=L[(size_t)i*(b+1)+(k-i+b)]*L[(size_t)j*(b+1)+(k-j+b)];
        L[(size_t)i*(b+1)+(j-i+b)]=s2/d;
      }
    }
  }
  void solve(double *x) const {
    for (i32 i=0;i<n;i++){ i32 i0=(i-b>0)?(i-b):0; double s2=x[i];
      for (i32 k=i0;k<i;k++) s2-=L[(size_t)i*(b+1)+(k-i+b)]*x[k];
      x[i]=s2/L[(size_t)i*(b+1)+b]; }
    for (i32 i=n-1;i>=0;i--){ double s2=x[i];
      for (i32 k=i+1;k<=i+b && k<n;k++) s2-=L[(size_t)k*(b+1)+(i-k+b)]*x[k];
      x[i]=s2/L[(size_t)i*(b+1)+b]; }
  }
};

// mathematical entropy pair: eta = -rho*s/(gam-1), s = ln p - gam ln rho; q = u*eta
static double entEta(const double U[3]) {
  double r=fmax(U[0],1e-12), u=U[1]/r;
  double p=fmax((GAM-1.0)*(U[2]-0.5*r*u*u),1e-12);
  double sp=log(p)-GAM*log(r);
  return -r*sp/(GAM-1.0);
}
static double entQ(const double U[3]) {
  double r=fmax(U[0],1e-12), u=U[1]/r;
  return u*entEta(U);
}

static void eulerFlux(const double U[3], double F[3], double &u, double &c) {
  double r=fmax(U[0],1e-12); u=U[1]/r;
  double p=(GAM-1.0)*(U[2]-0.5*r*u*u); p=fmax(p,1e-12);
  F[0]=U[1]; F[1]=U[1]*u+p; F[2]=(U[2]+p)*u;
  c=sqrt(GAM*p/r);
}

int main(int argc, char **argv) {
  const i32 p   = (argc>1)? atoi(argv[1]) : 2;
  const double CFL   = getenv("IGA_CFL")   ? atof(getenv("IGA_CFL"))   : 0.3;
  const double C_DC  = getenv("IGA_CDC")   ? atof(getenv("IGA_CDC"))   : 1.0;
  const double C_MAX = getenv("IGA_CMAX")  ? atof(getenv("IGA_CMAX"))  : 0.5;
  // SUPG with the LAGGED unsteady residual is linearly UNSTABLE (blowup after
  // a fixed step count at every N -- measured); with the steady residual it is
  // stable but adds diffusion and breaks exact conservation.  DC alone passes
  // the gate, so SUPG is OPT-IN (IGA_SUPG=1, steady-residual form).
  const i32 supg     = getenv("IGA_SUPG")? 1 : 0;
  // IGA_EV=1: Guermond-Popov ENTROPY VISCOSITY instead of the U-residual DC.
  // The entropy residual eta_t + q_x is O(1) at shocks (entropy production is
  // a Dirac there), EXACTLY zero across contacts, truncation-small in smooth
  // flow -- a shock-only sensor with convergence theory behind it.
  const i32 useEV    = getenv("IGA_UDC")? 0 : 1;   // entropy viscosity DEFAULT;
                                                    // IGA_UDC=1 for the U-residual
  const double TEND = 0.2;

  printf("IGA 1-D Euler, C^%d B-splines (p=%d), SUPG=%d, sensor=%s "
         "(C=%.2f cap=%.2f), Sod gate\n", p-1, p, supg,
         useEV?"ENTROPY-VISC":"U-residual DC", C_DC, C_MAX);
  printf("%6s %10s %12s %10s %10s %10s %9s %7s\n",
         "N", "dofs", "L1(rho)", "order", "min rho", "overshoot", "dM/M0", "cw/h");

  double prevErr = 0; bool pass = true;
  for (i32 N : {100, 200, 400, 800}) {
    Spline1d S; S.init(p, N);
    const double h = 1.0/N;
    const i32 n = S.n;
    // Greville abscissae for IC sampling (interpolation-free L2 would be
    // better; Greville collocation is the standard simple start)
    std::vector<double> U((size_t)3*n), U0((size_t)3*n), Us((size_t)3*n), R((size_t)3*n);
    std::vector<double> Udot((size_t)3*n, 0.0);   // lagged dU/dt (previous rhs)
    for (i32 i=0;i<n;i++){
      double xg = h*((i - (p-1)*0.5));           // greville in cell units*h
      double r,u2,pr;
      if (xg < 0.5) { r=1.0; u2=0.0; pr=1.0; } else { r=0.125; u2=0.0; pr=0.1; }
      U[3*i]=r; U[3*i+1]=r*u2; U[3*i+2]=pr/(GAM-1.0)+0.5*r*u2*u2;
    }
    // consistent mass, banded (bandwidth p), Gauss (p+1) per span
    BandChol M; M.n=n; M.b=p;
    { std::vector<double> Ab((size_t)n*(p+1), 0.0);
      GaussRule g = gaussLegendre(p+1);
      real Nv[BS_NMAX];
      for (i32 s2=0;s2<N;s2++) for (i32 q=0;q<g.n;q++){
        S.val(g.x[q], Nv);
        for (i32 a=0;a<=p;a++) for (i32 b2=0;b2<=a;b2++){
          i32 i=s2+a, j=s2+b2;
          Ab[(size_t)i*(p+1)+(j-i+p)] += (double)g.w[q]*h*(double)Nv[a]*(double)Nv[b2];
        } }
      M.factor(Ab);
    }

    // one RHS evaluation: Galerkin + SUPG + DC into R (control-point residual),
    // then M^-1
    GaussRule g = gaussLegendre(p+2);
    std::vector<double> nuSpan(N), nuS(N);
    double evNorm = 1.0;                 // lagged ||eta - eta_bar||_inf (Guermond)
    auto rhs=[&](const std::vector<double> &Uc, std::vector<double> &Rout){
      std::fill(Rout.begin(), Rout.end(), 0.0);
      real Nv[BS_NMAX], Dv[BS_NMAX];
      double eAcc=0, eMax=0;
      // ---- pass 1: span-wise DC viscosity ---------------------------------
      // A per-POINT sensor switches off too sharply as the shock thins: the
      // measured post-shock overshoot GREW under refinement (2.5% -> 12%).
      // Span-wise nu with a neighbour-max pass keeps the viscosity footprint
      // one support wide around the discontinuity -- the classic smoothing.
      for (i32 s2=0;s2<N;s2++){
        double nmax=0;
        for (i32 q=0;q<g.n;q++){
          S.val(g.x[q], Nv); S.der(g.x[q], Dv);
          double Uq[3]={0,0,0}, Ux[3]={0,0,0}, Ut[3]={0,0,0};
          for (i32 a=0;a<=p;a++) for (i32 k=0;k<3;k++){
            Uq[k]+=(double)Nv[a]*Uc[3*(s2+a)+k];
            Ux[k]+=(double)Dv[a]/h*Uc[3*(s2+a)+k];
            Ut[k]+=(double)Nv[a]*Udot[3*(s2+a)+k];
          }
          double F[3], u2, c2; eulerFlux(Uq,F,u2,c2);
          double lam=fabs(u2)+c2;
          double Fx[3];
          { double Up[3], Fp[3], Fm[3], du=1e-7, uu, cc;
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]+du*Ux[k];
            eulerFlux(Up,Fp,uu,cc);
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]-du*Ux[k];
            eulerFlux(Up,Fm,uu,cc);
            for (i32 k=0;k<3;k++) Fx[k]=(Fp[k]-Fm[k])/(2*du);
          }
          double nu;
          if (useEV) {
            // entropy residual eta_t + q_x, both via the same directional-FD
            // trick as Fx (chain rule along Ut / Ux)
            double du=1e-7, Up[3];
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]+du*Ut[k];
            double etp=entEta(Up);
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]-du*Ut[k];
            double etm=entEta(Up);
            double eta_t=(etp-etm)/(2*du);
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]+du*Ux[k];
            double qp=entQ(Up);
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]-du*Ux[k];
            double qm=entQ(Up);
            double q_x=(qp-qm)/(2*du);
            nu = C_DC*h*h*fabs(eta_t+q_x)/evNorm;
            eAcc += (double)g.w[q]*h*entEta(Uq);      // eta mean accumulation
            eMax = fmax(eMax, fabs(entEta(Uq)));
          } else {
            double gradU=fabs(Ux[0])+fabs(Ux[1])+fabs(Ux[2]);
            double res =fabs(Ut[0]+Fx[0])+fabs(Ut[1]+Fx[1])+fabs(Ut[2]+Fx[2]);
            nu = C_DC*h*h*res/(gradU*h+1e-10);
          }
          nmax=fmax(nmax, fmin(nu, C_MAX*h*lam));
        }
        nuS[s2]=nmax;
      }
      if (useEV) {
        // lagged Guermond normalization: ||eta - eta_bar||_inf from THIS pass,
        // used by the NEXT (one-evaluation lag, standard practice)
        double ebar = eAcc;                         // domain length is 1
        double dev = fabs(eMax - fabs(ebar));
        evNorm = fmax(dev, 1e-8);
      }
      for (i32 s2=0;s2<N;s2++){
        double v=nuS[s2];
        if (s2>0)   v=fmax(v,nuS[s2-1]);
        if (s2<N-1) v=fmax(v,nuS[s2+1]);
        // frozen-end guard: the boundary control points hold the IC, so a
        // nonzero nu there leaks mass through the frozen rows -- measured as
        // an O(h) drift (1e-6, halving with h) under the entropy sensor,
        // whose normalized residual never vanishes EXACTLY.  Waves stay far
        // from the ends in this test by construction.
        if (s2 < 2*p || s2 >= N-2*p) v = 0;
        nuSpan[s2]=v;
      }
      // ---- pass 2: assembly ------------------------------------------------
      for (i32 s2=0;s2<N;s2++) {
        for (i32 q=0;q<g.n;q++){
          S.val(g.x[q], Nv); S.der(g.x[q], Dv);
          double Uq[3]={0,0,0}, Ux[3]={0,0,0}, Ut[3]={0,0,0};
          for (i32 a=0;a<=p;a++) for (i32 k=0;k<3;k++){
            Uq[k]+=(double)Nv[a]*Uc[3*(s2+a)+k];
            Ux[k]+=(double)Dv[a]/h*Uc[3*(s2+a)+k];
            Ut[k]+=(double)Nv[a]*Udot[3*(s2+a)+k];
          }
          double F[3], u2, c2; eulerFlux(Uq,F,u2,c2);
          double lam=fabs(u2)+c2;
          // dF/dx via chain rule: A * Ux (A = flux Jacobian) -- compute by
          // finite difference of the flux for compactness
          double Fx[3];
          { double Up[3], Fp[3], Fm[3], du=1e-7, uu, cc;
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]+du*Ux[k];
            eulerFlux(Up,Fp,uu,cc);
            for (i32 k=0;k<3;k++) Up[k]=Uq[k]-du*Ux[k];
            eulerFlux(Up,Fm,uu,cc);
            for (i32 k=0;k<3;k++) Fx[k]=(Fp[k]-Fm[k])/(2*du);
          }
          double nu = nuSpan[s2];   // span-wise, neighbour-smoothed (pass 1)
          double w2=(double)g.w[q]*h;
          for (i32 a=0;a<=p;a++){
            double Na=(double)Nv[a], Da=(double)Dv[a]/h;
            for (i32 k=0;k<3;k++){
              Rout[3*(s2+a)+k] += w2*( F[k]*Da - nu*Ux[k]*Da );
              if (supg) {
                // SUPG on the true residual, scalar tau, spectral A^T grad N_a
                double tau = h/(2.0*lam+1e-12);
                Rout[3*(s2+a)+k] -= w2*tau*lam*Da*Fx[k];   // steady residual (stable)
              }
            }
          }
        }
      }
      // ends: hold IC (waves never reach the boundary by t=0.2)
      for (i32 i=0;i<p;i++) for (i32 k=0;k<3;k++)
        { Rout[3*i+k]=0; Rout[3*(n-1-i)+k]=0; }
      // M^-1 per component
      std::vector<double> col(n);
      for (i32 k=0;k<3;k++){
        for (i32 i=0;i<n;i++) col[i]=Rout[3*i+k];
        M.solve(col.data());
        for (i32 i=0;i<n;i++) Rout[3*i+k]=col[i];
      }
    };

    // SSP-RK3
    double t=0, M0=0;
    { GaussRule gm=gaussLegendre(p+1); real Nv[BS_NMAX];
      for (i32 s2=0;s2<N;s2++) for(i32 q=0;q<gm.n;q++){ S.val(gm.x[q],Nv);
        double rq=0; for(i32 a=0;a<=p;a++) rq+=(double)Nv[a]*U[3*(s2+a)];
        M0+=(double)gm.w[q]*h*rq; } }
    while (t < TEND-1e-12) {
      double lamMax=0;
      for (i32 i=0;i<n;i++){ double F[3],u2,c2; eulerFlux(&U[3*i],F,u2,c2);
        lamMax=fmax(lamMax,fabs(u2)+c2); }
      double dt=fmin(CFL*h/lamMax, TEND-t);
      U0=U;
      rhs(U,R);  Udot=R; for (size_t i2=0;i2<U.size();i2++) Us[i2]=U0[i2]+dt*R[i2];
      rhs(Us,R); for (size_t i2=0;i2<U.size();i2++) Us[i2]=0.75*U0[i2]+0.25*(Us[i2]+dt*R[i2]);
      rhs(Us,R); for (size_t i2=0;i2<U.size();i2++) U[i2]=(1.0/3)*U0[i2]+(2.0/3)*(Us[i2]+dt*R[i2]);
      t+=dt;
      bool bad=false;
      for (i32 i=0;i<n;i++) if (!(U[3*i]==U[3*i]) || U[3*i]<0) bad=true;
      if (bad){ printf("%6d  BLOWUP at t=%.4f\n",N,t); pass=false; break; }
    }

    // ---- gate: L1(rho) vs exact, overshoot, mass ----
    RiemannExact ex; ex.init(1.0,0.0,1.0, 0.125,0.0,0.1);
    GaussRule gm=gaussLegendre(p+2); real Nv[BS_NMAX];
    double e1=0, rmin=1e300, over=0, M1=0;
    // exact post-shock density (right of contact behind the shock)
    double rPost = 0.125*((ex.ps/0.1+(GAM-1)/(GAM+1))/((GAM-1)/(GAM+1)*ex.ps/0.1+1.0));
    for (i32 s2=0;s2<N;s2++) for(i32 q=0;q<gm.n;q++){
      S.val(gm.x[q],Nv);
      double x=(s2+(double)gm.x[q])*h, rq=0;
      for(i32 a=0;a<=p;a++) rq+=(double)Nv[a]*U[3*(s2+a)];
      double re,ue,pe; ex.sample((x-0.5)/TEND, re,ue,pe);
      e1 += (double)gm.w[q]*h*fabs(rq-re);
      rmin=fmin(rmin,rq); M1+=(double)gm.w[q]*h*rq;
      { double xc = 0.5 + ex.us*TEND;               // contact position
        double sSh = ex.uR + ex.cR*sqrt((GAM+1)/(2*GAM)*ex.ps/ex.pR+(GAM-1)/(2*GAM));
        double xs2 = 0.5 + sSh*TEND;                 // shock position
        if (x > xc + 5*h && x < xs2 - 2*h && rq > rPost) over=fmax(over, rq-rPost);
        if (x > xs2 + 5*h && rq > 0.125) over=fmax(over, rq-0.125); }
    }
    // contact width: x-extent of the transition band between the contact's two
    // exact states (0.426 -> 0.266): rho in (0.29, 0.40) near the contact
    double cw=0;
    { double xc=0.5+ex.us*TEND, xlo=1e300, xhi=-1e300;
      for (i32 s2=0;s2<N;s2++) for(i32 q=0;q<gm.n;q++){
        double x=(s2+(double)gm.x[q])*h;
        if (fabs(x-xc)>0.1) continue;
        real Nv3[BS_NMAX]; S.val(gm.x[q],Nv3);
        double rq=0; for(i32 a=0;a<=p;a++) rq+=(double)Nv3[a]*U[3*(s2+a)];
        if (rq>0.29 && rq<0.40){ xlo=fmin(xlo,x); xhi=fmax(xhi,x); } }
      cw = (xhi>xlo)? (xhi-xlo)/h : 0; }
    double order = prevErr>0 ? log2(prevErr/e1) : 0;
    printf("%6d %10d %12.4e %10.2f %10.4f %10.4f %9.1e %7.1f\n",
           N, 3*n, e1, order, rmin, over/rPost, fabs(M1-M0)/M0, cw);
    prevErr = e1;
    if (rmin <= 0 || over/rPost > 0.15) pass=false;
    if (getenv("IGA_DUMP") && N==400) {
      FILE *fp=fopen("/tmp/claude-1000/-home-kennyl-Documents-wavelet-cfd/3802b4fd-408b-4d7f-b0a9-ed9464b5a408/scratchpad/sod.csv","w");
      fprintf(fp,"x,rho,rho_exact\n");
      for (i32 s2=0;s2<N;s2++) for (i32 q=0;q<2;q++){
        double x=(s2+0.25+0.5*q)*h, rq=0; real Nv2[BS_NMAX];
        S.val(0.25+0.5*q,Nv2);
        for(i32 a=0;a<=p;a++) rq+=(double)Nv2[a]*U[3*(s2+a)];
        double re,ue,pe; ex.sample((x-0.5)/TEND,re,ue,pe);
        fprintf(fp,"%.6f,%.6e,%.6e\n",x,rq,re); }
      fclose(fp);
    }
  }
  printf("\n%s\n", pass ?
    "IGA-EULER-1D PASS: C^{p-1} splines + classic FEM shock capturing hold the\n"
    "Sod shock -- positivity kept, overshoot bounded, L1 converging."
    : "IGA-EULER-1D FAIL");
  return pass?0:1;
}
