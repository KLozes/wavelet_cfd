#ifndef KTAU_SST_H
#define KTAU_SST_H

//
// k~ - tau~ SST turbulence model and its algebraic wall model.
//
//   Y. Tamaki, C. Friess, J. Jacob, T. Imamura, "Robust implementation strategy
//   for k-omega SST turbulence model with immersed boundary methods",
//   J. Comput. Phys. 566 (2026) 115239.   (docs/WallModeledRans.pdf; every
//   equation number below refers to that paper.)
//
// The model replaces omega by tau = 1/omega, which VANISHES at the wall instead
// of diverging, and adds a viscous damping function f_v1 that extends the
// LOG-LAYER solutions of k and tau all the way to the wall.  The near-wall
// profiles are then constant (k~) and linear (tau~) -- both representable on a
// Cartesian grid that does not resolve the viscous sublayer, which is exactly
// what makes the wall-modeled coupling work.
//
// Everything here is algebraic and stateless: given the local state and the wall
// distance it returns the closure (F1, F2, f_v1, Theta, mu_t, blended
// coefficients) and the point sources.  The transport-equation discretization
// lives in the solver -- with one exception: the tau~ diffusion, whose
// NON-conservative face fluxes (Appendix A) are here too, because getting them
// right is the whole difficulty and ktau_test gates them directly.
//
// The design identity everything hangs on: with the near-wall solution (Eq. 18)
// and the thin-layer momentum balance (Eq. 21), Theta of Eq. (19) equals
// (u_tau/(kappa d))^2 EXACTLY, for ANY f_v1.  That makes the production and the
// dissipation of the k~ equation both equal rho u_tau^3/(kappa d), and the three
// tau~ terms of Eq. (24) sum to zero.  ktau_test checks all of it.
//

#include "Settings.cuh"
#include "Util.cuh"
#include <cmath>

namespace ktau {

// ---- model constants (Eqs. 9, 10; identical to the k-omega SST model) ------
constexpr real betaStar     = 0.09;
constexpr real sqrtBetaStar = 0.3;            // sqrt(0.09), exact
constexpr real kappa        = 0.41;
constexpr real a1           = 0.31;

constexpr real sigK1 = 0.85,  sigK2 = 1.0;
constexpr real sigW1 = 0.5,   sigW2 = 0.856;
constexpr real beta1 = 0.075, beta2 = 0.0828;

// gamma_i = beta_i/betaStar - sigW_i kappa^2 / sqrt(betaStar)        (Eq. 10)
constexpr real gam1 = beta1/betaStar - sigW1*kappa*kappa/sqrtBetaStar;   // 0.553167
constexpr real gam2 = beta2/betaStar - sigW2*kappa*kappa/sqrtBetaStar;   // 0.440355

// wall damping exponent/constant (Sec. 2.5).  alpha = 1.5 is chosen so that
// Eq. (34) integrates in CLOSED FORM -- that closed form is uPlus() below --
// and psi = 26.5 is fitted so the FPTBL C_f matches the original k-omega SST.
constexpr real alphaFv = 1.5;
constexpr real psiFv   = 26.5;

constexpr real kPi = (real)3.14159265358979323846;

// ---- small helpers --------------------------------------------------------
__host__ __device__ inline real pow15(real x) { return x*sqrt(x); }   // x^1.5, no pow()

// chi = k~ tau~ / nu  (Eq. 19).  Equivalently mu_t/(rho nu f_v1) (Eq. 17), and
// in the near-wall solution chi = kappa y+.
__host__ __device__ inline real chiOf(real kt, real tt, real nu) {
  return fmax(kt,(real)0)*fmax(tt,(real)0)/fmax(nu,(real)1e-30);
}

// Eq. (33) with F1 = 1 and r_d = 1: the damping the WALL FUNCTION is integrated
// with (chi = kappa y+).  uPlus()/dUplusDyplus() must stay consistent with this.
__host__ __device__ inline real fv1Plain(real chi) {
  const real c = pow15(fmax(chi,(real)0));
  return c/(c + psiFv);
}

// Eqs. (33) / (38): the full damping function.  r_d = max(dCutoff/d, 1) is the
// eddy-viscosity augmentation of the wall-modeled path; r_d = 1 recovers Eq. (33).
//
// NOTE the r_d OUTSIDE the max.  It is what makes mu_t = rho k~ tau~ f_v1
// CONSTANT below the image point: k~ tau~ = kappa u_tau d there, so f_v1 has to
// go like 1/d, and only the outside factor does that ((chi r_d) is constant
// below the cutoff, so the bracket alone is constant too).  Reading r_d as a
// factor on the second branch instead gives mu_t ~ d and silently breaks the
// shear-stress balance under the IP -- which is the whole point of Sec. 3.1.
__host__ __device__ inline real fv1Of(real chi, real F1, real rd) {
  const real c = pow15(fmax(chi*rd,(real)0));
  return fmax(c/(c + psiFv), (real)1 - F1) * rd;
}

// phi, the Appendix-A wall damping of the tau~ diffusion (Eq. A.5).  Linear in d
// at the wall (which is what tames f~ ~ 1/d) and flat beyond the cutoff.
__host__ __device__ inline real phiDamp(real d, real dCut) {
  return sin(kPi*fmin(fmax(d,(real)0), dCut)/((real)2*fmax(dCut,(real)1e-30)));
}

// ---- blending functions (Eqs. 30, 31) --------------------------------------
//
// Gamma1/Gamma2 are the k-omega SST arg1/arg2 branches rewritten with tau = 1/omega;
// Gamma3 is the cross-diffusion branch, whose k-omega form 4 rho sigW2 k/(CD d^2)
// becomes -2 k~ tau~ / (d^2 grad k~ . grad tau~).  The min() clamps the
// denominator to a strictly NEGATIVE value, so Gamma3 stays positive and finite
// even where grad k~ . grad tau~ >= 0.
//
__host__ __device__ inline void blendFuncs(real kt, real tt, real nu, real d,
                                           real gradKdotGradT, real nuInf, real Lref,
                                           real &F1, real &F2)
{
  // Clamp here as well as in closure(): this is a public entry point, and a
  // transiently NEGATIVE k~ (routine before the solver clips a k-equation
  // update) would flip the sign of Gamma3's numerator, make arg1 negative, and
  // then be laundered back to +1 by the EVEN power in tanh(arg1^4) -- handing a
  // freestream cell F1 = 1, i.e. the k-omega coefficient set, no cross-diffusion
  // and no sustaining terms, at exactly the point the model wants k-epsilon.
  kt = fmax(kt,(real)0);
  tt = fmax(tt,(real)0);
  const real dd = fmax(d,(real)1e-30);
  const real G1 = sqrt(fmax(kt,(real)0))*tt/(betaStar*dd);
  const real G2 = (real)500*nu*tt/(dd*dd);
  const real floor3 = -(real)1e-12*nuInf/fmax(Lref*Lref,(real)1e-30);
  const real G3 = -(real)2*kt*tt/(dd*dd) / fmin(gradKdotGradT, floor3);
  // tanh saturates to 1 well before arg = 10, so capping the arguments there is
  // numerically a no-op -- but it keeps arg1^4 from overflowing the float build,
  // where G2 = 500 nu tau~/d^2 can get large in the first cell off a fine wall.
  const real arg1 = fmin(fmin(fmax(G1,G2), G3), (real)10);
  const real arg2 = fmin(fmax((real)2*G1, G2), (real)10);
  const real a1sq = arg1*arg1;
  F1 = tanh(a1sq*a1sq);          // tanh(arg1^4)
  F2 = tanh(arg2*arg2);          // tanh(arg2^2)
}

// ---- the closure at a point ------------------------------------------------
struct Closure {
  real F1, F2;                   // Eq. (30)
  real chi, fv1, fv2;            // Eqs. (19), (33)/(38)
  real Theta;                    // Eq. (19): damped strain (or vorticity) squared
  real limSST;                   // the min[1, a1/...] of Eq. (28)
  real muT;                      // Eq. (28)
  real sigK, sigW, beta, gam;    // F1-blended coefficients (Eq. 8)
};

// S and Om are the strain and vorticity magnitudes; useVorticity selects the
// "-V" variant (S^2 -> Omega^2 in Eq. 19), which is what the paper validates
// with.  rd = max(dCutoff/d, 1) for the wall-modeled path, 1 otherwise.
__host__ __device__ inline Closure closure(real rho, real kt, real tt, real nu,
                                           real d, real S, real Om,
                                           real gradKdotGradT,
                                           real nuInf, real Lref, real rd,
                                           bool useVorticity = true)
{
  Closure c;
  kt = fmax(kt,(real)0);
  tt = fmax(tt,(real)1e-30);

  blendFuncs(kt, tt, nu, d, gradKdotGradT, nuInf, Lref, c.F1, c.F2);

  c.chi = chiOf(kt, tt, nu);
  c.fv1 = fv1Of(c.chi, c.F1, rd);
  const real q = c.chi/((real)1 + c.chi*c.fv1);        // chi/(1 + chi f_v1)
  c.fv2 = (real)1 - q*q;                                // Eq. (19)

  const real G   = useVorticity ? Om : S;               // "-V" variant
  const real sbt = sqrtBetaStar/tt;                     // = u_tau/(kappa d) near the wall
  c.Theta = fmax(G*G + sbt*sbt*c.fv2*c.F1, (real)0.09*G*G);   // (0.3 G)^2 floor

  // eddy viscosity, Eqs. (28)-(29).  Om~ carries (1 - q) -- NOT f_v2, which
  // carries (1 - q^2).
  const real OmTil = fmax(Om + sqrtBetaStar*c.F1/tt*((real)1 - q), (real)0.3*Om);
  c.limSST = fmin((real)1, a1/fmax(OmTil*tt*c.F2,(real)1e-12));
  c.muT    = rho*kt*tt*c.fv1*c.limSST;

  const real g = (real)1 - c.F1;
  c.sigK = c.F1*sigK1 + g*sigK2;
  c.sigW = c.F1*sigW1 + g*sigW2;
  c.beta = c.F1*beta1 + g*beta2;
  c.gam  = c.F1*gam1  + g*gam2;
  return c;
}

// ---- point sources ---------------------------------------------------------
//
// The non-diffusive right-hand sides of Eqs. (25)-(26): production and
// dissipation for k~, and the -gamma rho tau~^2 Theta + beta rho pair for tau~,
// plus the optional freestream-sustaining terms of Eq. (32).
//
// P_k~ is Eq. (27) written as rho k~ tau~ limSST Theta rather than mu_t Theta/f_v1
// -- algebraically identical, but finite where f_v1 -> 0.
//
__host__ __device__ inline void sources(const Closure &c, real rho, real kt, real tt,
                                        real kInf, real tInf, bool sustain,
                                        real &sk, real &st)
{
  kt = fmax(kt,(real)0);
  tt = fmax(tt,(real)1e-30);
  const real Pk = fmin(rho*kt*tt*c.limSST*c.Theta, (real)20*betaStar*rho*kt/tt);
  sk = Pk - betaStar*rho*kt/tt;
  st = -c.gam*rho*tt*tt*c.Theta + c.beta*rho;
  if (sustain) {                                                    // Eq. (32)
    sk += betaStar*rho*(kInf/tInf)*((real)1 - c.F1);
    st -= c.beta*rho*(kInf/fmax(kt,kInf))*(tt/tInf)*((real)1 - c.F1);
  }
}

// ---- near-wall similarity solution (Eqs. 16, 18) ---------------------------
__host__ __device__ inline real kNearWall(real uTau)          { return uTau*uTau/sqrtBetaStar; }
__host__ __device__ inline real tauNearWall(real uTau, real d) {
  return kappa*sqrtBetaStar*d/fmax(uTau,(real)1e-30);
}

// ---- velocity wall function ------------------------------------------------
//
// Eq. (35): the closed-form integral of Eq. (34) with chi = kappa y+, alpha=1.5,
// psi = 26.5.  Asymptotically u+ = ln(y+)/kappa + 5.2199171 -- exactly the
// log-law slope, because the three log residues sum to 1/kappa (to 2e-15), and
// the intercept is (6.6501959 - 4.0953338) pi/2 + 1.2067490 in closed form.
//
// The coefficients are the paper's to every digit it prints, but carried to
// full precision: they are the partial-fraction expansion of
//   du+/dsigma = 2 sigma (kappa^1.5 sigma^3 + psi) / (kappa^2.5 sigma^5 + kappa^1.5 sigma^3 + psi),
// sigma = sqrt(y+), over the roots of that quintic (two conjugate pairs -> the
// two log/atan pairs, one real root -> the lone log).  Eq. (34) is the model;
// Eq. (35) is only its printed 6-digit rounding, which is off by up to 2.2e-5
// in u+ (mostly in the additive constant: 1.20677 vs 1.2067490).  Using the
// exact residues instead costs nothing and lets ktau_test gate uPlus against a
// quadrature of Eq. (34) at 1e-12 rather than at the rounding level.
//
__host__ __device__ inline real uPlus(real yp) {
  const real y = fmax(yp,(real)0);
  const real s = sqrt(y);
  return (real)(-6.6501958897564286)*atan((real)(-0.53910202540295604)*s + (real)1.2396760212086757)
       - (real)  4.0953337526867921 *atan((real)  0.33111991742187391*s + (real)0.2888463815305457)
       - (real)  0.18730207210813268*log (y - (real)4.5990404887908563*s + (real)8.7285826072687183)
       + (real)  4.0077031531887295 *log (y + (real)1.7446632856128177*s + (real)9.8816880908378995)
       - (real)  2.7627533816733845 *log (s + (real)2.8543772031780388)
       + (real)  1.2067490125650551;
}

// Eq. (34): du+/dy+ = 1/(1 + chi f_v1) with chi = kappa y+.
__host__ __device__ inline real dUplusDyplus(real yp) {
  const real chi = kappa*fmax(yp,(real)0);
  return (real)1/((real)1 + chi*fv1Plain(chi));
}

// Solve  u = u_tau uPlus(d u_tau / nu)  for u_tau (Sec. 3.1, "Newton iterations
// so that u_IP and d_IP satisfy the wall function").
//
// g(u_tau) = u_tau uPlus(y+) - u is monotone increasing (both uPlus and y+ grow
// with u_tau) and convex, so Newton converges; the bisection safeguard just
// keeps a stray step inside the bracket.  g' = uPlus + y+ uPlus'.
__host__ __device__ inline real uTauFromWallFunction(real u, real d, real nu,
                                                     i32 maxIt = 40,
                                                     real tol = (real)1e-12)
{
  u = fabs(u);
  const real dd = fmax(d,(real)1e-30);
  if (u <= (real)0) return (real)0;

  // viscous-sublayer guess: u+ = y+  =>  u_tau = sqrt(nu u / d).  Exact for
  // y+ << 1 and an under-estimate elsewhere, so it brackets from below.
  real lo = (real)0;
  real ut = sqrt(nu*u/dd);
  real hi = fmax(ut,(real)1e-30);
  for (i32 k = 0; k < 200; k++) {              // expand until g(hi) >= 0
    if (hi*uPlus(dd*hi/nu) - u >= (real)0) break;
    hi *= (real)2;
  }
  ut = (real)0.5*(lo + hi);

  for (i32 it = 0; it < maxIt; it++) {
    const real yp = dd*ut/nu;
    const real up = uPlus(yp);
    const real g  = ut*up - u;
    if (g > (real)0) hi = ut; else lo = ut;
    if (fabs(g) <= tol*u) break;
    const real dg = up + yp*dUplusDyplus(yp);   // dg/du_tau
    real next = (dg > (real)0) ? ut - g/dg : (real)0.5*(lo + hi);
    if (!(next > lo && next < hi)) next = (real)0.5*(lo + hi);   // safeguard
    if (fabs(next - ut) <= tol*ut) { ut = next; break; }
    ut = next;
  }
  return ut;
}

// ---- turbulence wall boundary conditions (Eq. 39) --------------------------
// k~ is Neumann at the wall but its near-wall value is the constant u_tau^2/sqrt(b*);
// tau~ is linear in d, capped by the image-point value to stay finite as u_tau -> 0.
__host__ __device__ inline void wallBcKTau(real uTau, real dFc, real tauIp, real dIp,
                                           real &kFc, real &tauFc)
{
  kFc   = uTau*uTau/sqrtBetaStar;
  tauFc = fmin(kappa*sqrtBetaStar*dFc/fmax(uTau,(real)1e-30), tauIp*dFc/fmax(dIp,(real)1e-30));
}

// ---- tau~ diffusion: Appendix A non-conservative face fluxes ---------------
//
// The tau~ diffusion  tau~^2 d/dx[ C (1/tau~^2) dtau~/dx ]  is NOT in
// conservative form: the tau~^2 prefactor must not be differentiated.  The
// remedy (Eq. A.2) is a face flux PAIR that differs between the two cells
// sharing the face -- cell L takes tau~_L^2 f~_LR, cell R takes tau~_R^2 f~_LR --
// so the cell residual (f_L^{i+1/2} - f_R^{i-1/2})/h reproduces
// tau~_i^2 (f~_{i+1/2} - f~_{i-1/2})/h exactly.  Splitting off a source term
// instead (Eq. A.1) puts a spurious tau~ peak at the boundary-layer edge.
//
// f~_LR ~ 1/d near the wall, so the damping phi (Eq. A.5) is folded in: phi f~
// is bounded at the wall, and the leftover -(1/phi) C dtau~/dx dphi/dx comes
// back as a second flux pair (Eq. A.9, evaluated at F1 = 1 since phi is active
// only near the wall).  A third pair carries the cross-diffusion (Eq. A.11).
// All three share the face-centre quantities.
//
// PRECISION: the cell residual is a 1/h-amplified difference of near-equal
// fluxes, so its roundoff floor grows like eps (d/h)^2 -- the generic limit of
// any second-derivative operator, not something particular to this form.  In
// double that floor is ~5e-10 at d/h = 1600 and the O(h^2) truncation dominates;
// in FLOAT (which is what wave3d builds) it is 0.18% of the near-wall balance at
// d/h = 100 and 15% at d/h = 1600.  So the balance is solid through the boundary
// layer and degrades in the far field -- where F1 -> 0 and this term stops being
// a leading-order balance anyway.  ktau_test/ktau_test_sp report both bands.
//
// C      = mu(1-F1) + sigW rho k~ tau~   at the face                  (Eq. A.3)
// dTdn   = (tau~_R - tau~_L)/h, the compact jump (this is the "gradient at LR
//          that correctly reflects the difference between tau~_L and tau~_R",
//          i.e. what keeps the operator free of even-odd decoupling)
// rkTau. = rho k~ tau~ at L / R;  cd. = 2(1-F1) sigW2 rho tau~ at L / R
//
// MIXED PRECISION (measured 2026-08-25): the pair is computed internally in
// DOUBLE regardless of the build's `real`.  This does NOT move the eps (d/h)^2
// float floor -- gates 7/8 are unchanged by it (gate 8 still diverges at order
// -2.2 in float), because that floor is a STORAGE limit: the second difference
// must dig a signal of size tau~'' h^2 out of stored values carrying eps tau~
// of rounding, and no internal precision -- nor any smooth re-map of the
// stored variable (tau~/d, log tau~: reconstruction multiplies the noise
// straight back) -- can recover bits fp32 storage discarded.  The only true
// cures are fp64 storage for tau~ or subtracting an ANALYTIC reference
// profile.  Mitigation in practice: on the AMR grid d/h is bounded per level
// (~30-100 in the wall band), so the divergent d/h -> 1600 regime never
// arises there.  What the double internals DO buy is conditioning of the
// (A.6)/(A.7) ratio branch at an immersed wall face, where tau~_LR ~ d_FC can
// be arbitrarily small and the fp32 intermediates (1/tau~_LR^2, phi ratios)
// lose their delicate cancellation.  Bit-identical in the double build; ~30
// fp64 flops per face in the float build.
__host__ __device__ inline void tauDiffFluxes(
    real C_, real dTdn_, real tauLR_, real kLR_, real phiLR_,
    real tauL_, real tauR_, real phiL_, real phiR_,
    real rkTauL_, real rkTauR_, real cdL_, real cdR_, real kL_, real kR_,
    real &fL, real &fR, real a7Tol_ = (real)1e-6)
{
  const double C = (double)C_, dTdn = (double)dTdn_, tauLR = (double)tauLR_;
  const double kLR = (double)kLR_, tauL = (double)tauL_, tauR = (double)tauR_;
  const double rkTauL = (double)rkTauL_, rkTauR = (double)rkTauR_;
  const double cdL = (double)cdL_, cdR = (double)cdR_;
  const double kL = (double)kL_, kR = (double)kR_;
  const double tiny = 1e-20;
  const double pL  = fmax((double)phiL_,  tiny);
  const double pR  = fmax((double)phiR_,  tiny);
  const double pLR = fmax((double)phiLR_, tiny);

  // (A.6): first term of Eq. (A.4).  Keep tau~_L^2/tau~_LR^2 as a RATIO -- that
  // ratio is precisely what makes the pair non-conservative; cancelling it
  // would collapse both sides to the ordinary conservative flux.
  //
  // The (A.7) fallback triggers on tau~_LR being small RELATIVE to the cells it
  // sits between, not on an absolute floor.  The paper's condition is
  // "tau~_LR ~ 0", which is a statement about scale: on an immersed boundary the
  // face-to-surface distance d_FC -- and with it tau~_FC -- can be arbitrarily
  // small, so 1/tau~_LR^2 blows up long before any absolute epsilon is reached.
  // (A grid-aligned wall never exposes this: there d_FC is a fixed half cell.)
  // The switch is RELATIVE, and its threshold decides how large the (A.6)
  // ratio tau~_L^2/tau~_LR^2 is allowed to get.  At 1e-6 it effectively never
  // fires: once the first cell's tau~ has grown, 1e-6*tRef is far below tau~_FC,
  // so (A.6) keeps running with an amplification of (tau~_1/tau~_FC)^2 -- and
  // that wall flux is CUBIC in tau~_1, i.e. a positive feedback.  a7Tol caps the
  // amplification at 1/a7Tol^2.
  const double tRef = fmax(fabs(tauL), fabs(tauR));
  const double a7 = fmax((double)a7Tol_, 0.0);
  double d1L, d1R;
  if (tauLR > tiny && tauLR > a7*tRef) {
    const double ftil = C*dTdn/(tauLR*tauLR);                     // (A.3)
    d1L = (pLR/pL)*tauL*tauL*ftil;
    d1R = (pLR/pR)*tauR*tauR*ftil;
  } else {
    // (A.7): at a face with vanishing wall distance use phi_LR/tau~_LR ~
    // phi_L/tau~_L, under which the ratio collapses to phi_L/phi_LR.
    d1L = (pL/pLR)*C*dTdn;
    d1R = (pR/pLR)*C*dTdn;
  }

  // (A.9): second term of Eq. (A.4).
  const double d2L = -(double)sigW1*(rkTauL/pL)*dTdn*(pLR - pL);
  const double d2R = -(double)sigW1*(rkTauR/pR)*dTdn*(pLR - pR);

  // (A.11): cross-diffusion in the same non-conservative split.
  const double cL = cdL*dTdn*(kLR - kL);
  const double cR = cdR*dTdn*(kLR - kR);

  fL = (real)(d1L + d2L + cL);
  fR = (real)(d1R + d2R + cR);
}

}  // namespace ktau

#endif
