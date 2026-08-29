//
// Gate test for the k~ - tau~ SST closure (KtauSst.h).
//
//   Y. Tamaki et al., J. Comput. Phys. 566 (2026) 115239  (docs/WallModeledRans.pdf)
//
// Every check here is ANALYTIC -- none of it needs a flow solve, which matters
// because this solver's cubic cells can never reach the y+ ~ 1 of the paper's
// wall-RESOLVED validation (Sec. 4.1).  What can be checked exactly is the
// internal consistency the model is built on:
//
//   1  uPlus() is the exact integral of Eq. (34)             (vs quadrature)
//   2  ... and therefore has exactly the log-law slope 1/kappa
//   3  the u_tau Newton inversion of the wall function round-trips
//   4  Theta = (u_tau/(kappa d))^2 EXACTLY, for any f_v1     (Eqs. 18-23)
//   5  mu_t and du/dy are CONSTANT below the image point     (Eq. 38's r_d)
//   6  k~ production = dissipation = rho u_tau^3/(kappa d)   (Sec. 2.3)
//   7  the three tau~ terms of Eq. (24) sum to zero, with the tau~ diffusion
//      evaluated through the Appendix-A non-conservative face fluxes
//   8  a manufactured solution exercises A.6 + A.11 with F1 < 1  (gate 7 runs at
//      F1 = 1 with k~ constant, so it reaches neither the mu(1-F1) term nor the
//      cross-diffusion pair)
//   9  the Eq. (A.7) fallback branch is consistent and finite
//  10  all of the above compiles and runs in device code
//
//   build: make ktau_test        run: ./ktau_test
//
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "KtauSst.h"

using namespace ktau;

// The solver builds in FLOAT; this test builds in both (ktau_test / ktau_test_sp).
// Roundoff-limited tolerances scale with the precision, discretization-limited
// ones do not.  Two gates -- the finite-difference cross-check and the y+ = 1e12
// log-law asymptote -- are pure double-precision probes and are skipped in the
// float build, where they measure nothing but cancellation.
static const bool gDp = (sizeof(real) == 8);
static const real gS  = gDp ? (real)1 : (real)5.4e8;    // eps_float / eps_double
static bool gPass = true;
static void check(const char *what, real err, real tol, const char *note = "") {
  const bool ok = (err <= tol) && !std::isnan((double)err);
  if (!ok) gPass = false;
  printf("   %-46s %.3e  (tol %.0e)  %s%s\n", what, (double)err, (double)tol,
         ok ? "ok" : "FAIL", note);
}

// ---- the near-wall similarity state, Eqs. (18) + (21) ----------------------
// Everything the model assumes holds throughout the viscous sublayer AND the
// log layer.  rd = max(dCut/d, 1) switches on the wall-modeled augmentation.
struct NearWall {
  real k, tau, chi, F1, F2, fv1, S, muT, rho, nu, d, uTau, rd;
};
static NearWall nearWallState(real uTau, real d, real nu, real rho, real dCut)
{
  NearWall w;
  w.rho = rho;  w.nu = nu;  w.d = d;  w.uTau = uTau;
  w.rd  = (dCut > 0) ? fmax(dCut/d, (real)1) : (real)1;
  w.k   = kNearWall(uTau);
  w.tau = tauNearWall(uTau, d);
  w.chi = chiOf(w.k, w.tau, nu);                       // = kappa y+
  // k~ is constant in the wall-normal direction, so grad k~ . grad tau~ = 0.
  blendFuncs(w.k, w.tau, nu, d, (real)0, nu, (real)1, w.F1, w.F2);
  w.fv1 = fv1Of(w.chi, w.F1, w.rd);
  // thin-layer momentum balance (mu + mu_t) S = rho u_tau^2   (Eqs. 20-21)
  w.S   = uTau*uTau/(nu*((real)1 + w.chi*w.fv1));
  w.muT = rho*w.k*w.tau*w.fv1;
  return w;
}
static Closure nearWallClosure(const NearWall &w) {
  return closure(w.rho, w.k, w.tau, w.nu, w.d, w.S, w.S, (real)0,
                 w.nu, (real)1, w.rd, true);
}

// ---- high-accuracy quadrature of Eq. (34), the wall-function DEFINITION ----
// Substituting y+ = t^2 removes the y+^2.5 kink at the wall, leaving a smooth
// integrand; composite 5-point Gauss-Legendre then lands at machine precision.
static real quadUplus(real yp)
{
  const real T = sqrt(fmax(yp,(real)0));
  const i32  M = 4000;
  const real gx[5] = {(real)-0.9061798459386640, (real)-0.5384693101056831, (real)0,
                      (real) 0.5384693101056831, (real) 0.9061798459386640};
  const real gw[5] = {(real) 0.2369268850561891, (real) 0.4786286704993665,
                      (real) 0.5688888888888889,
                      (real) 0.4786286704993665, (real) 0.2369268850561891};
  const real dt = T/(real)M;
  real acc = 0;
  for (i32 m = 0; m < M; m++) {
    const real t0 = (real)m*dt, tc = t0 + (real)0.5*dt;
    for (i32 q = 0; q < 5; q++) {
      const real t = tc + (real)0.5*dt*gx[q];
      acc += gw[q]*(real)0.5*dt*(real)2*t*dUplusDyplus(t*t);   // dy+ = 2t dt
    }
  }
  return acc;
}


// ---- manufactured solution for the Appendix-A operator ---------------------
// Smooth k~(x), tau~(x) and F1(x) strictly inside (0,1), so the mu(1-F1) half of
// Eq. (A.3) and the whole cross-diffusion pair (A.11) are live -- neither of
// which gate 7 can reach (it runs at F1 = 1 with k~ constant).  Always evaluated
// in double, so the reference is exact in both builds.
static const double mmsMu = 0.3, mmsRho = 1.0;
static double mmsK  (double x){ return 1.0 + 0.3*sin(2*M_PI*x); }
static double mmsKp (double x){ return 0.3*2*M_PI*cos(2*M_PI*x); }
static double mmsT  (double x){ return 0.5*(1.0 + 0.4*cos(2*M_PI*x + 0.7)); }
static double mmsTp (double x){ return -0.5*0.4*2*M_PI*sin(2*M_PI*x + 0.7); }
static double mmsF1 (double x){ return 0.5 + 0.25*sin(2*M_PI*x + 1.3); }
static double mmsC  (double x){
  const double f = mmsF1(x);
  const double sw = f*(double)sigW1 + (1.0-f)*(double)sigW2;
  return (1.0-f)*mmsMu + sw*mmsRho*mmsK(x)*mmsT(x);          // Eq. (A.3)
}
static double mmsG  (double x){ return mmsC(x)*mmsTp(x)/(mmsT(x)*mmsT(x)); }
// tau~^2 d/dx[ C tau~^-2 dtau~/dx ]  +  (1-F1) 2 sigW2 rho tau~ dk~/dx dtau~/dx
static double mmsTarget(double x){
  const double h = 1e-5;
  const double Gp = (-mmsG(x+2*h) + 8*mmsG(x+h) - 8*mmsG(x-h) + mmsG(x-2*h))/(12*h);
  const double f = mmsF1(x);
  return mmsT(x)*mmsT(x)*Gp
       + (1.0-f)*2.0*(double)sigW2*mmsRho*mmsT(x)*mmsKp(x)*mmsTp(x);
}

// ---- 8: the same identity, evaluated on the GPU ---------------------------
__global__ void deviceGateKernel(real uTau, real nu, real rho, real *out) {
  const i32 i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= 64) return;
  const real yp  = (real)0.05*pow((real)10, (real)4*i/(real)63);   // y+ in [0.05, 500]
  const real d   = yp*nu/uTau;
  const real k   = kNearWall(uTau);
  const real tau = tauNearWall(uTau, d);
  const real chi = chiOf(k, tau, nu);
  real F1, F2;
  blendFuncs(k, tau, nu, d, (real)0, nu, (real)1, F1, F2);
  const real fv1 = fv1Of(chi, F1, (real)1);
  const real S   = uTau*uTau/(nu*((real)1 + chi*fv1));
  const Closure c = closure(rho, k, tau, nu, d, S, S, (real)0, nu, (real)1, (real)1, true);
  const real want = (uTau/(kappa*d))*(uTau/(kappa*d));
  out[i] = fabs(c.Theta - want)/want;
}

int main(int argc, char **argv)
{
  printf("k~-tau~ SST closure gates  (Tamaki et al., JCP 566 (2026) 115239)\n");
  printf("constants:  gamma1 = %.6f  gamma2 = %.6f   (TMR: 0.5532, 0.4403)\n\n",
         (double)gam1, (double)gam2);

  // ---------------------------------------------------------------- 1 + 2 --
  printf("1  uPlus() is the exact integral of Eq.(34)\n");
  {
    // primary gate: against a machine-precision quadrature of Eq. (34), which
    // is what actually DEFINES the wall function (Eq. 35 is its closed form).
    real worstQ = 0;
    for (real yp = (real)1e-4; yp < (real)3e4; yp *= (real)2.3) {
      const real q = quadUplus(yp);
      worstQ = fmax(worstQ, fabs(uPlus(yp) - q));
    }
    check("max |uPlus - quad(Eq.34)| over y+ in [1e-4, 3e4]", worstQ, (real)1e-12*gS);
    check("|uPlus(0)|  (the integration constant)", fabs(uPlus((real)0)), (real)1e-14*gS);
    // secondary: differentiating the closed form must return Eq. (34) itself.
    // 4th-order stencil -- a 2nd-order one is roundoff-limited near y+ ~ 0.01,
    // where uPlus is a ~1e-2 difference of ~10-sized terms.
    if (gDp) {
      real worstD = 0;
      for (real yp = (real)0.01; yp < (real)3e4; yp *= (real)1.7) {
        const real h  = fmax(yp*(real)1e-3, (real)1e-12);
        const real fd = (-uPlus(yp+2*h) + (real)8*uPlus(yp+h)
                         - (real)8*uPlus(yp-h) + uPlus(yp-2*h))/((real)12*h);
        worstD = fmax(worstD, fabs(fd - dUplusDyplus(yp))/dUplusDyplus(yp));
      }
      check("max |du+/dy+ (FD) - 1/(1+chi f_v1)| / exact", worstD, (real)1e-9);
    }
  }
  printf("2  log-law asymptote  u+ -> ln(y+)/kappa + B%s\n", gDp ? "" : "   (double-only probe, skipped)");
  if (gDp) {
    // The approach to the log law is O(1/sqrt(y+)) (the atan and log remainders),
    // so B is only the asymptote in the limit -- report it converging.
    const real y0 = (real)1e10;
    const real Bslope = fabs((uPlus(10*y0) - uPlus(y0)) - log((real)10)/kappa)/(log((real)10)/kappa);
    check("|slope - 1/kappa| / (1/kappa)  at y+ = 1e10", Bslope, (real)1e-8);
    for (real yy = (real)1e5; yy <= (real)1e12; yy *= (real)1e4)
      printf("   B = u+ - ln(y+)/kappa at y+ = %-8.0e     %.9f   (limit 5.219917073)\n",
             (double)yy, (double)(uPlus(yy) - log(yy)/kappa));
    // closed form: B = (6.6501959 - 4.0953338) pi/2 + 1.2067490, the atan limits
    // plus the integration constant.
    check("|B(y+=1e12) - 5.219917072941399|",
          fabs(uPlus((real)1e12) - log((real)1e12)/kappa - (real)5.219917072941399), (real)1e-8);
    // the log residues must sum to 1/kappa -- that IS the log-law slope, exactly
    const real slopeSum = (real)(-0.18730207210813268) + (real)4.0077031531887295
                        + (real)(-2.7627533816733845)*(real)0.5;
    check("|sum of log residues - 1/kappa|", fabs(slopeSum - (real)1/kappa), (real)1e-14*gS);
  }

  // -------------------------------------------------------------------- 3 --
  printf("3  u_tau Newton inversion of the wall function\n");
  {
    real worst = 0;
    const real nu = (real)1e-5;
    const real ds[3] = {(real)1e-5, (real)1e-3, (real)1e-1};
    const real us[5] = {(real)1e-3, (real)1e-2, (real)5e-2, (real)2e-1, (real)1};
    for (i32 a = 0; a < 3; a++)
      for (i32 b = 0; b < 5; b++) {
        const real ut = us[b], d = ds[a];
        const real u  = ut*uPlus(d*ut/nu);              // forward
        const real r  = uTauFromWallFunction(u, d, nu); // inverse
        worst = fmax(worst, fabs(r-ut)/ut);
      }
    check("max |u_tau(recovered) - u_tau| / u_tau", worst, (real)1e-10*gS);
  }

  // -------------------------------------------------------------------- 4 --
  // The identity the whole construction rests on (Sec. 2.3): with the near-wall
  // solution and the thin-layer balance, Theta of Eq. (19) is (u_tau/(kappa d))^2
  // for ANY f_v1 -- including the r_d-augmented one of Eq. (38).
  printf("4  Theta = (u_tau/(kappa d))^2 exactly, for any f_v1\n");
  {
    real worst = 0, worstF1 = 0;
    const real nu = (real)1e-5, rho = (real)1, uTau = (real)0.05;
    for (i32 v = 0; v < 2; v++) {                       // rd = 1 and rd > 1
      const real dCut = v ? (real)0.02 : (real)0;
      for (real yp = (real)0.05; yp < (real)1e4; yp *= (real)1.6) {
        const real d = yp*nu/uTau;
        NearWall w = nearWallState(uTau, d, nu, rho, dCut);
        Closure  c = nearWallClosure(w);
        const real want = (uTau/(kappa*d))*(uTau/(kappa*d));
        worst   = fmax(worst, fabs(c.Theta - want)/want);
        worstF1 = fmax(worstF1, fabs(c.F1 - (real)1));
      }
    }
    check("max |Theta - (u_tau/kappa d)^2| / exact", worst, (real)1e-12*gS);
    check("max |F1 - 1| in the near-wall region", worstF1, (real)1e-14*gS,
          "  (Gamma1 = kappa/betaStar^0.75 = 2.495 there)");
  }

  // -------------------------------------------------------------------- 5 --
  // Sec. 3.1: below the image point the velocity is LINEARIZED, so du/dy is
  // constant, so mu_t must be too.  This is what the r_d factor OUTSIDE the max
  // in Eq. (38) buys -- and the check that catches reading it as an r_d inside
  // the second branch (which gives mu_t ~ d).
  printf("5  mu_t and du/dy are constant below the image point (Eq. 38)\n");
  {
    const real nu = (real)1e-5, rho = (real)1, uTau = (real)0.05;
    const real dCut = (real)0.02;                       // = 3 dx in the solver
    real muMin = 1e300, muMax = -1e300, sMin = 1e300, sMax = -1e300;
    for (real f = (real)0.02; f <= (real)1.0001; f += (real)0.02) {
      NearWall w = nearWallState(uTau, f*dCut, nu, rho, dCut);
      Closure  c = nearWallClosure(w);
      muMin = fmin(muMin, c.muT);  muMax = fmax(muMax, c.muT);
      sMin  = fmin(sMin,  w.S);    sMax  = fmax(sMax,  w.S);
    }
    // reference: mu_t = rho kappa u_tau dCut * bracket(chi at the cutoff)
    const real chiCut = kappa*dCut*uTau/nu;
    const real want   = rho*kappa*uTau*dCut*pow15(chiCut)/(pow15(chiCut) + psiFv);
    check("mu_t spread (max-min)/mean over 0 < d < dCut", (muMax-muMin)/(real)0.5/(muMax+muMin), (real)1e-12*gS);
    check("|mu_t - rho kappa u_tau dCut fv1_cut| / exact", fabs(muMax-want)/want, (real)1e-12*gS);
    check("du/dy spread (max-min)/mean below the IP", (sMax-sMin)/(real)0.5/(sMax+sMin), (real)1e-12*gS);
  }

  // -------------------------------------------------------------------- 6 --
  printf("6  k~ equation balances: P = beta* rho k~/tau~ = rho u_tau^3/(kappa d)\n");
  {
    real worstBal = 0, worstVal = 0;
    const real nu = (real)1e-5, rho = (real)1, uTau = (real)0.05;
    for (real yp = (real)0.05; yp < (real)1e4; yp *= (real)1.6) {
      const real d = yp*nu/uTau;
      NearWall w = nearWallState(uTau, d, nu, rho, (real)0);
      Closure  c = nearWallClosure(w);
      real sk, st;
      sources(c, rho, w.k, w.tau, (real)0, (real)1, false, sk, st);
      const real scale = rho*uTau*uTau*uTau/(kappa*d);
      worstBal = fmax(worstBal, fabs(sk)/scale);                     // P - D = 0
      worstVal = fmax(worstVal, fabs(betaStar*rho*w.k/w.tau - scale)/scale);
    }
    check("max |P - dissipation| / (rho u_tau^3/kappa d)", worstBal, (real)1e-12*gS);
    check("max |dissipation - rho u_tau^3/(kappa d)| / exact", worstVal, (real)1e-13*gS);
  }

  // -------------------------------------------------------------------- 7 --
  // Eq. (24): the three tau~ terms sum to zero, with the diffusion evaluated
  // through the Appendix-A non-conservative face fluxes on a 1-D near-wall
  // column.  The diffusion alone must converge to -kappa^2 sigW1 sqrt(beta*) rho.
  printf("7  tau~ equation Eq.(24) balances with the Appendix-A diffusion\n");
  {
    const real nu = (real)1e-5, rho = (real)1, uTau = (real)0.05;
    const real D = (real)0.2, dCut = (real)0.05;     // y+ up to 1000; cutoff at y+ = 250
    const real wantDiff = -kappa*kappa*sigW1*sqrtBetaStar*rho;
    real errPrev = 0;
    printf("   %-8s %-14s %-14s %-14s %s\n", "N", "diffusion", "want", "Linf(balance)", "order");
    for (i32 pass = 0; pass < 3; pass++) {
      const i32 N = 400 << pass;
      const real h = D/(real)N;
      real *fL = (real*)malloc((N+1)*sizeof(real));
      real *fR = (real*)malloc((N+1)*sizeof(real));
      for (i32 f = 1; f < N; f++) {                  // interior faces only
        const real dL = ((real)(f-1) + (real)0.5)*h, dR = ((real)f + (real)0.5)*h;
        const real dF = (real)f*h;
        const real tL = tauNearWall(uTau, dL), tR = tauNearWall(uTau, dR);
        const real kc = kNearWall(uTau);
        const real tLR = (real)0.5*(tL + tR);         // exact: tau~ is linear in d
        const real pL = phiDamp(dL,dCut), pR = phiDamp(dR,dCut), pF = phiDamp(dF,dCut);
        // F1 = 1 throughout (gate 4), so mu(1-F1) drops and sigma_omega = sigW1
        const real C    = sigW1*rho*kc*tLR;
        const real dTdn = (tR - tL)/h;
        tauDiffFluxes(C, dTdn, tLR, kc, pF, tL, tR, pL, pR,
                      rho*kc*tL, rho*kc*tR, (real)0, (real)0, kc, kc, fL[f], fR[f]);
      }
      // Split the sweep at d/h = 100.  The residual is a 1/h-amplified difference
      // of near-equal fluxes, so its roundoff floor grows like eps (d/h)^2 -- the
      // generic single-precision limit of ANY second-derivative operator, not
      // something particular to the Appendix-A form.  In double that floor is
      // 5e-10 at the finest grid and the discretization error O(h^2) dominates;
      // in float it is ~15% at d/h = 1600 and the table below diverges.  The
      // near-wall band (where this balance is what holds the model up, and where
      // F1 = 1) is the part that has to be right.
      real linfDiff = 0, linfBal = 0, linfNear = 0;
      for (i32 i = 1; i < N-1; i++) {
        const real d = ((real)i + (real)0.5)*h;
        const real diff = (fL[i+1] - fR[i])/h;
        NearWall w = nearWallState(uTau, d, nu, rho, (real)0);
        Closure  c = nearWallClosure(w);
        const real src = -c.gam*rho*w.tau*w.tau*c.Theta + c.beta*rho;
        linfDiff = fmax(linfDiff, fabs(diff - wantDiff));
        linfBal  = fmax(linfBal,  fabs(src + diff));
        if (d <= (real)100*h) linfNear = fmax(linfNear, fabs(src + diff));
      }
      const real order = (pass && errPrev > 0) ? log(errPrev/linfNear)/log((real)2) : (real)0;
      char ordBuf[32];
      if (pass && errPrev > 0) sprintf(ordBuf, "%.2f", (double)order);
      else                     sprintf(ordBuf, "-");
      printf("   %-8d %-14.6e %-14.6e %-11.3e %-11.3e %s\n", N,
             (double)((fL[N/2+1]-fR[N/2])/h), (double)wantDiff,
             (double)linfNear, (double)linfBal, ordBuf);
      errPrev = linfNear;
      if (pass == 2) {
        // In double the residual is discretization-limited (O(h^2), the table).
        // In float it is roundoff-limited: the residual is a 1/h-amplified
        // difference of near-equal fluxes, so it floors at ~eps/h * |f| and
        // does NOT converge -- the table above shows that directly.
        // gated on the near-wall band; the target itself is 2.52e-2, so the
        // float tolerance below is a 0.4% relative error there.
        check("Linf | Eq.(24) balance |, d/h < 100", linfNear, gDp ? (real)2e-5 : (real)1e-4);
        check("Linf | diffusion + kappa^2 sigW1 sqrt(b*) rho |, all d",
              linfDiff, gDp ? (real)2e-5 : (real)2e-2);
        if (gDp) check("observed convergence order >= 1.9", (real)2.0 - order, (real)0.1);
      }
      free(fL); free(fR);
    }
  }


  // -------------------------------------------------------------------- 8 --
  // Gate 7 pins A.6 + A.9, but at F1 = 1 with k~ constant it never touches the
  // mu(1-F1) half of Eq. (A.3) nor the cross-diffusion pair (A.11) -- whose
  // prefactor is an INPUT to tauDiffFluxes, so nothing but a comment asserts it.
  // This runs beyond the cutoff (phi == 1) on purpose: Eq. (A.9) is derived
  // ASSUMING F1 = 1, the paper's own caveat, so F1 < 1 and phi < 1 must not be
  // combined.  With phi == 1 the A.9 pair vanishes identically and the target is
  // exact, leaving A.6 and A.11 as the only things under test.
  printf("8  manufactured solution: A.6 + A.11 with F1 < 1, in the phi == 1 region\n");
  {
    const real dCut = (real)1, dOff = (real)2;          // d in [2,3] -> phi == 1
    real errPrev = 0, ordFinest = 0, relCoarse = 0, relFinest = 0;
    printf("   %-8s %-14s %-14s %s\n", "N", "Linf", "Linf/|target|", "order");
    for (i32 pass = 0; pass < 4; pass++) {
      const i32 N = 200 << pass;
      const real h = (real)1/(real)N;
      real *fL = (real*)malloc((N+2)*sizeof(real));
      real *fR = (real*)malloc((N+2)*sizeof(real));
      for (i32 f = 1; f < N; f++) {
        const real xL = ((real)(f-1) + (real)0.5)*h, xR = ((real)f + (real)0.5)*h;
        const real tL = (real)mmsT(xL),  tR = (real)mmsT(xR);
        const real kL = (real)mmsK(xL),  kR = (real)mmsK(xR);
        const real f1L= (real)mmsF1(xL), f1R= (real)mmsF1(xR);
        const real tLR = (real)0.5*(tL+tR), kLR = (real)0.5*(kL+kR);
        const real f1LR = (real)0.5*(f1L+f1R);
        const real swLR = f1LR*sigW1 + ((real)1-f1LR)*sigW2;
        const real C    = ((real)1-f1LR)*(real)mmsMu + swLR*(real)mmsRho*kLR*tLR;
        const real dTdn = (tR - tL)/h;
        const real pL = phiDamp(dOff+xL,dCut), pR = phiDamp(dOff+xR,dCut);
        const real pF = phiDamp(dOff+(real)f*h,dCut);
        const real cdL = (real)2*((real)1-f1L)*sigW2*(real)mmsRho*tL;   // Eq. (A.11)
        const real cdR = (real)2*((real)1-f1R)*sigW2*(real)mmsRho*tR;
        tauDiffFluxes(C, dTdn, tLR, kLR, pF, tL, tR, pL, pR,
                      (real)mmsRho*kL*tL, (real)mmsRho*kR*tR, cdL, cdR, kL, kR,
                      fL[f], fR[f]);
      }
      real linf = 0, scale = 0;
      for (i32 i = 1; i < N-1; i++) {
        const real x    = ((real)i + (real)0.5)*h;
        const real got  = (fL[i+1] - fR[i])/h;
        const real want = (real)mmsTarget((double)x);
        linf  = fmax(linf,  fabs(got - want));
        scale = fmax(scale, fabs(want));
      }
      const real order = (pass && errPrev > 0) ? log(errPrev/linf)/log((real)2) : (real)0;
      char ob[32];
      if (pass && errPrev > 0) sprintf(ob, "%.2f", (double)order); else sprintf(ob, "-");
      printf("   %-8d %-14.4e %-14.4e %s\n", N, (double)linf, (double)(linf/scale), ob);
      errPrev = linf;  relFinest = linf/scale;  ordFinest = order;
      if (pass == 0) relCoarse = linf/scale;
      free(fL); free(fR);
    }
    // Double is discretization-limited: gate the FINEST level and demand 2nd
    // order.  Float hits the same eps (L/h)^2 roundoff wall as gate 7 and cannot
    // converge, so gate the COARSEST level -- the finer ones are in the table to
    // SHOW the wall, not to pass.  (L/h here is cells across the profile, so the
    // practical float floor is eps * (cells across the boundary layer)^2: 5e-5
    // at 30 cells, 5e-3 at 300.)
    check(gDp ? "relative Linf of A.6+A.11 at the finest level"
              : "relative Linf of A.6+A.11 at N = 200 (float floor)",
          gDp ? relFinest : relCoarse, gDp ? (real)2e-5 : (real)3e-3);
    if (gDp) check("observed convergence order >= 1.9", (real)2.0 - ordFinest, (real)0.1);
  }

  // -------------------------------------------------------------------- 9 --
  // Eq. (A.7), the tau~_LR -> 0 fallback.  Gate 7 never enters this branch (its
  // faces all sit at d = f*h > 0), so it is checked directly: the substitution
  // must be CONSISTENT with the exact branch where both are valid, and it must
  // stay finite where the exact branch cannot.
  printf("9  Eq.(A.7) fallback branch at a face with vanishing wall distance\n");
  {
    const real C = (real)0.7, dTdn = (real)2, pL = (real)0.3, pR = (real)0.31;
    const real pLR = (real)0.5*(pL + pR);
    // (a) consistency: where tau~_LR really equals tau~_L phi_LR/phi_L, the exact
    //     branch must already give the fallback's (phi_L/phi_LR) C dtau~/dx.
    const real tL = (real)1e-3;
    const real tLR = tL*pLR/pL, tR = (real)2*tLR - tL;
    real aL, aR;
    tauDiffFluxes(C, dTdn, tLR, (real)0, pLR, tL, tR, pL, pR,
                  (real)0, (real)0, (real)0, (real)0, (real)0, (real)0, aL, aR);
    check("|exact branch - (phi_L/phi_LR) C dtau/dx| / exact",
          fabs(aL - (pL/pLR)*C*dTdn)/fabs((pL/pLR)*C*dTdn), (real)1e-6);
    // (b) finiteness: below the guard, where tau~_LR^2 would underflow
    real bL, bR;
    tauDiffFluxes(C, dTdn, (real)1e-25, (real)0, pLR, (real)1e-25, (real)1e-25, pL, pR,
                  (real)0, (real)0, (real)0, (real)0, (real)0, (real)0, bL, bR);
    const bool fin = std::isfinite((double)bL) && std::isfinite((double)bR);
    if (!fin) gPass = false;
    printf("   %-46s fL=%.6e fR=%.6e  %s\n", "fallback stays finite at tau~_LR = 1e-25",
           (double)bL, (double)bR, fin ? "ok" : "FAIL");
    check("|fallback fL - (phi_L/phi_LR) C dtau/dx|",
          fabs(bL - (pL/pLR)*C*dTdn), (real)1e-6);
    check("|fallback fR - (phi_R/phi_LR) C dtau/dx|",
          fabs(bR - (pR/pLR)*C*dTdn), (real)1e-6);
  }

  // ------------------------------------------------------------------- 10 --
  printf("10 device-code gate (the closure must run inside the solver kernels)\n");
  {
    real *dOut = nullptr, hOut[64];
    if (cudaMalloc(&dOut, 64*sizeof(real)) == cudaSuccess) {
      deviceGateKernel<<<1,64>>>((real)0.05, (real)1e-5, (real)1, dOut);
      cudaMemcpy(hOut, dOut, 64*sizeof(real), cudaMemcpyDeviceToHost);
      cudaFree(dOut);
      real worst = 0;
      for (i32 i = 0; i < 64; i++) worst = fmax(worst, hOut[i]);
      check("max device |Theta - (u_tau/kappa d)^2| / exact", worst, (real)1e-12*gS);
    } else {
      printf("   (no CUDA device available -- device gate skipped)\n");
    }
  }

  printf("\n%s\n", gPass ? "KTAU PASS" : "KTAU FAIL");
  return gPass ? 0 : 1;
}
