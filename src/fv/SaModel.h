#ifndef SA_MODEL_H
#define SA_MODEL_H

//
// Spalart-Allmaras turbulence model with the near-wall modification for
// immersed boundaries.
//
//   P. Spalart, S. Allmaras, "A one-equation turbulence model for aerodynamic
//   flows", AIAA Paper 92-0439 (1992)                       -- the base model
//   S.R. Allmaras, F.T. Johnson, "Modifications and clarifications for the
//   implementation of the Spalart-Allmaras turbulence model", ICCFD7 (2012)
//                                                           -- the "SA-noft2" form
//   Y. Tamaki, M. Harada, T. Imamura, "Near-wall modification of
//   Spalart-Allmaras turbulence model for immersed boundary method",
//   AIAA J. 55 (9) (2017) 3027-3039                         -- the IB modification
//
// WHY THIS MODEL AND NOT k~-tau~.  The k~-tau~ SST wall model is built on an
// exact near-wall identity: Theta = (u_tau/(kappa d))^2, which makes the k~
// production and dissipation EQUAL there, so P/D = limSST <= 1 and the k~
// source Jacobian is ~0.  That balance is neutrally stable by construction: any
// spatial truncation error tips it, nothing damps it, and an explicit march
// walks off (measured: k~ -> 800x its physical value on the paper's own
// inclined immersed gate, then NaN).  SA is a ONE-equation model whose near-wall
// solution nu~ = kappa u_tau d is algebraic and monotone -- there is no
// production/dissipation pair to sit in balance, so that failure mode cannot
// arise.  This is why the SA-lineage immersed wall models in the literature run
// stably with fully EXPLICIT time integration.
//
// Everything here is algebraic and stateless, matching KtauSst.h: given the
// local state and the wall distance it returns mu_t and the point source.
//

#include "Settings.cuh"
#include "Util.cuh"
#include <cmath>

namespace sa {

// ---- model constants (SA-noft2) -------------------------------------------
constexpr real cb1   = 0.1355;
constexpr real cb2   = 0.622;
constexpr real sigma = 2.0/3.0;
constexpr real kappa = 0.41;
constexpr real cw2   = 0.3;
constexpr real cw3   = 2.0;
constexpr real cv1   = 7.1;
// cw1 = cb1/kappa^2 + (1 + cb2)/sigma
constexpr real cw1   = cb1/(kappa*kappa) + (1.0 + cb2)/sigma;
constexpr real cv1c  = cv1*cv1*cv1;
constexpr real cw3c  = cw3*cw3*cw3*cw3*cw3*cw3;

// ---- damping, with the Tamaki near-wall (immersed) modification ------------
//
// Base model: f_v1 = chi^3/(chi^3 + cv1^3),  chi = nu~/nu.
//
// Immersed modification: below the image point the wall model has LINEARISED
// the velocity, so du/dy is constant there.  To keep the shear stress
// (mu + mu_t) du/dy balanced under that linearisation, the eddy viscosity must
// also be constant below the IP -- which is what r_d = max(d_cutoff/d, 1)
// buys: it freezes the argument at its d_cutoff value as d shrinks, and the
// trailing factor r_d cancels the 1/d that mu_t would otherwise inherit.
// r_d = 1 outside the cutoff, so the model is untouched away from the wall.
__host__ __device__ inline real fv1Of(real chi, real rd) {
  const real c = chi*rd;
  const real c3 = c*c*c;
  return (c3/(c3 + cv1c))*rd;
}

// ---- modified vorticity S~ -------------------------------------------------
// S~ = Omega + nu~ f_v2/(kappa^2 d^2), with the Allmaras positivity limiter
// that keeps S~ from going negative (S~ >= 0.3 Omega) rather than clipping it.
__host__ __device__ inline real sTilde(real Om, real nut, real d, real chi, real fv1) {
  const real fv2 = (real)1 - chi/((real)1 + chi*fv1);
  const real kd2 = fmax(kappa*kappa*d*d, (real)1e-30);
  const real S   = Om + nut*fv2/kd2;
  return fmax(S, (real)0.3*Om);
}

// ---- destruction function f_w ---------------------------------------------
__host__ __device__ inline real fwOf(real nut, real St, real d) {
  const real kd2 = fmax(kappa*kappa*d*d, (real)1e-30);
  real r = nut/fmax(St*kd2, (real)1e-30);
  r = fmin(r, (real)10);                       // standard cap
  const real r6 = r*r*r*r*r*r;
  const real g  = r + cw2*(r6 - r);
  const real g6 = g*g*g*g*g*g;
  return g*pow(((real)1 + cw3c)/fmax(g6 + cw3c,(real)1e-30), (real)(1.0/6.0));
}

struct Closure {
  real chi, fv1, muT, St, fw;
};

__host__ __device__ inline Closure closure(real rho, real nut, real nu, real d,
                                           real Om, real rd) {
  Closure c;
  nut    = fmax(nut, (real)0);
  c.chi  = nut/fmax(nu, (real)1e-30);
  c.fv1  = fv1Of(c.chi, rd);
  c.muT  = rho*nut*c.fv1;
  c.St   = sTilde(Om, nut, d, c.chi, c.fv1);
  c.fw   = fwOf(nut, c.St, d);
  return c;
}

// ---- point source: production - destruction --------------------------------
// The conservative diffusion and the cb2 gradient term are face quantities and
// live in the solver; this is the algebraic part.
//   P = cb1 S~ nu~,   D = cw1 f_w (nu~/d)^2
// dMin: the sub-grid floor for the DESTRUCTION denominator (half a local cell).
// A cell flagged FLUID can still have its CENTRE inside the body -- isFluidCell
// tests the four CORNERS, so a body thinner than a cell (an airfoil trailing
// edge) slips between them -- and there the level set gives d = 0 exactly.
// (nu~/1e-30)^2 is then 2e47, which OVERFLOWS fp32 (3.4e38) and puts inf, then
// NaN, into nu~ on the FIRST step (measured: RAE 2822 TE, x = 12.44).  The
// grid-aligned plate never saw it because the half-cell ibPlane offset keeps
// every fluid centre at d >= 0.5h -- the same floor applied here.  Only the
// destruction is floored: closure() is already safe at d = 0 (the S~ positivity
// limiter and the r <= 10 cap both hold), and flooring the d that feeds
// f_v1/r_d/mu_t was tried 2026-08-26 and pollutes the mean flow.
__host__ __device__ inline real source(const Closure &c, real rho, real nut,
                                       real d, real dMin = (real)0) {
  nut = fmax(nut, (real)0);
  const real P = cb1*c.St*nut;
  const real dD = fmax(fmax(d, dMin), (real)1e-30);
  const real D = cw1*c.fw*(nut/dD)*(nut/dD);
  return rho*(P - D);
}

// ---- near-wall solution: the immersed wall boundary value -------------------
// In the log layer the SA solution is nu~ = kappa u_tau d, EXACTLY linear in the
// wall distance and monotone.  That is the whole reason this model survives an
// explicit immersed wall coupling where the two-equation one does not: there is
// a single algebraic value to impose, not a production/dissipation balance.
__host__ __device__ inline real nutWall(real uTau, real d) {
  return kappa*fmax(uTau,(real)0)*fmax(d,(real)0);
}

// ---- gate: the log-layer equilibrium ---------------------------------------
// With nu~ = kappa u_tau d, Omega = u_tau/(kappa d) and f_v1 -> 1, the SA source
// must vanish: production cb1 S~ nu~ balances destruction cw1 f_w (nu~/d)^2.
// Returns the residual normalised by the production, which must be ~0.
__host__ __device__ inline real logLayerResidual(real uTau, real d, real nu) {
  const real nut = nutWall(uTau, d);
  const real Om  = uTau/(kappa*fmax(d,(real)1e-30));
  Closure c = closure((real)1, nut, nu, d, Om, (real)1);
  const real P = cb1*c.St*nut;
  return (P > 0) ? source(c, (real)1, nut, d)/(P) : (real)0;
}

}  // namespace sa
#endif
