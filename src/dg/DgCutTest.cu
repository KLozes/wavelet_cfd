//
// Cut-cell DG feasibility gate: the DISCRETE DIVERGENCE THEOREM.
//
// A DG element's semi-discrete RHS for basis function i is
//
//     M_ij dQ_j/dt = INT_{K n W} F(Q) . grad(phi_i) dV
//                  - SUM_faces INT_{f n W} (F*.n) phi_i dS
//                  - INT_{wall}          (F*.n) phi_i dS
//
// For a CONSTANT state F is constant, so the RHS collapses to
//
//     F_d . [ INT_{K n W} d(phi_i)/dx_d dV  -  CLOSED INT phi_i n_d dS ]
//
// which is exactly the divergence theorem applied to phi_i.  If the volume,
// face and wall quadratures do not satisfy it DISCRETELY, a cut cell emits a
// spurious source under a uniform flow -- free-stream preservation fails, and
// that is the classic way cut-cell DG dies.  Nothing downstream is worth
// building until this residual is at round-off.
//
// It is a sharp test of the QUADRATURE ALONE: no equation, no time stepping,
// no small-cell fix.  The three rules are produced independently by the Saye
// recursion (volume on {phi<0}, sayeFace on each cell face, sayeSurface on
// {phi=0}) and are only consistent if their geometry agrees.
//
// build:  make dgcut_test
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Util.cuh"
#include "Poly.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "LagrangeBasis.h"

static constexpr i32 ARENA = 1 << 20;
static constexpr i32 OUTCAP = 1 << 18;

// element in world space: lower corner x0, size h.  Level set is a SPHERE;
// FLUID is {phi < 0} to match Saye's convention (the DG solver's cylinder SDF
// is the other sign, which is a negation at the sampling step, nothing more).
struct Case { real x0[3], h, cx, cy, cz, R; const char *name; };

// QUADRATIC form R^2 - r^2, not the SDF R - r: it is exactly degree 2, so the
// degree-3 fit reproduces it EXACTLY and the fitted geometry IS the sphere.
// That removes level-set fit error from the test, isolating the quadrature.
static double sphPhi(const Case &c, double x, double y, double z) {
  double dx=x-c.cx, dy=y-c.cy, dz=z-c.cz;
  if (getenv("USE_SDF")) return c.R - sqrt(dx*dx+dy*dy+dz*dz);
  return c.R*c.R - (dx*dx+dy*dy+dz*dz);        // <0 OUTSIDE the sphere = fluid
}

// exactly-linear level set: the cut region is a POLYHEDRON and the wall a PLANE,
// so every integrand in the GCL identity is a genuine polynomial.  If GCL fails
// here it is the rules, not the geometry.
static double planePhi(double x, double y, double z) {
  return (real)(0.37 - (0.6*x + 0.5*y + 0.3*z));
}

int main(void) {
  const i32 p = DG_ORDER, n = p+1, nd = n*n*n;
  LagrangeBasis B; B.init(p);                  // DGSEM nodal basis (GLL)

  Case cases[] = {
    // cell straddling the sphere surface at various offsets -> a range of cut
    // fractions, including deliberate SLIVERS (the hard case)
    {{0,0,0}, 1.0, 0, 0, 0, -1.0, "PLANE (exact polyhedron)"},
    {{0,0,0}, 1.0, 0.5, 0.5, 0.5, 0.75, "centred, big cut"},
    {{0,0,0}, 1.0, 0.5, 0.5, 0.5, 0.52, "near-face tangency"},
    {{0,0,0}, 1.0, 0.0, 0.0, 0.0, 1.20, "corner cut"},
    {{0,0,0}, 1.0, 0.0, 0.0, 0.0, 0.10, "SLIVER (tiny fluid)"},
    {{0,0,0}, 1.0, 0.0, 0.0, 0.0, 1.70, "SLIVER (tiny solid)"},
    {{0,0,0}, 1.0, 1.3, 0.5, 0.5, 1.00, "oblique, off-axis"},
  };

  printf("cut-cell DG quadrature gate   p=%d  (%d nodes/elem)\n", p, nd);
  printf("%-24s %10s %10s %12s %12s %12s\n",
         "case", "|K n W|", "|wall|", "GCL(sum_i)", "GCL(max_i)", "verdict");

  double worst = 0;
  for (const Case &c : cases) {
    // ---- degree-p fit of phi at the element's own GLL nodes ----------------
    // The DG element IS the fit stencil: blockSize == p+1, so the (p+1)^3 LGL
    // nodes the solver already stores are exactly what fitPoly3 wants.
    std::vector<real> v(nd);
    for (i32 k=0;k<n;k++) for (i32 j=0;j<n;j++) for (i32 i=0;i<n;i++)
      { double X=c.x0[0]+c.h*B.t[i], Y=c.x0[1]+c.h*B.t[j], Z=c.x0[2]+c.h*B.t[k];
        v[i+n*(j+n*k)] = (real)(c.R < 0 ? planePhi(X,Y,Z) : sphPhi(c,X,Y,Z)); }
    PolyND phi = fitPoly3(p, v.data());

    std::vector<SayeNode> arenaBuf(ARENA), volBuf(OUTCAP), srfBuf(OUTCAP), facBuf(OUTCAP);
    SayeArena ar; ar.buf = arenaBuf.data(); ar.cap = ARENA; ar.top = 0;
    auto mkset = [&](std::vector<SayeNode> &b){ SayeSet s; s.p=b.data(); s.n=0; s.cap=OUTCAP; s.ovf=false; return s; };

    SayeCfg cfg = SayeCfg::def();
    if (getenv("SAYE_NG"))    cfg.ng       = atoi(getenv("SAYE_NG"));
    if (getenv("SAYE_DEPTH")) cfg.maxDepth = atoi(getenv("SAYE_DEPTH"));
    SayeSet vol = mkset(volBuf); sayeVolume (phi, &vol, &ar, cfg);
    SayeSet srf = mkset(srfBuf); sayeSurface(phi, &srf, &ar, cfg);

    // ---- GCL residual per basis function and direction --------------------
    //   r_i,d = INT dphi_i/dx_d dV  -  [ SUM_faces INT phi_i n_d dS
    //                                  + INT_wall  phi_i n_d dS ]
    // Everything is on the REFERENCE cell, so the h factors cancel identically
    // (volume h^3 * gradient 1/h vs surface h^2) and this is a pure geometry
    // identity -- no metric bookkeeping can hide an inconsistency.
    std::vector<double> lhs(3*nd, 0.0), rhs(3*nd, 0.0);
    real gb[QN_MAX*QN_MAX*QN_MAX*3], vb[QN_MAX*QN_MAX*QN_MAX];

    for (i32 q=0;q<vol.n;q++) {
      B.allGradRef(vol.p[q].x, gb);
      for (i32 i=0;i<nd;i++) for (i32 d=0;d<3;d++)
        lhs[3*i+d] += (double)vol.p[q].w * gb[3*i+d];
    }
    // wall: Saye's node normal is grad(phi)/|grad(phi)|, i.e. it points from
    // {phi<0} into {phi>0} -- OUTWARD from the fluid region, which is the sign
    // the divergence theorem wants.
    double area = 0;
    for (i32 q=0;q<srf.n;q++) {
      B.allVal(srf.p[q].x, vb);
      area += (double)srf.p[q].w;
      for (i32 i=0;i<nd;i++) for (i32 d=0;d<3;d++)
        rhs[3*i+d] += (double)srf.p[q].w * vb[i] * (double)srf.p[q].n[d];
    }
    // the six cell faces: outward normal is -e_d at side 0, +e_d at side 1
    for (i32 d=0;d<3;d++) for (i32 side=0;side<2;side++) {
      SayeSet fc = mkset(facBuf); sayeFace(phi, d, side, &fc, &ar, cfg);
      double sgn = side ? 1.0 : -1.0;
      for (i32 q=0;q<fc.n;q++) {
        B.allVal(fc.p[q].x, vb);
        for (i32 i=0;i<nd;i++) rhs[3*i+d] += (double)fc.p[q].w * vb[i] * sgn;
      }
    }

    double volume=0; for (i32 q=0;q<vol.n;q++) volume += (double)vol.p[q].w;
    // sum_i phi_i == 1 (partition of unity), so summing the residual over i is
    // the CLOSED-SURFACE normal integral -- it must vanish on its own.
    double sumr=0, maxr=0;
    for (i32 d=0;d<3;d++) {
      double s=0;
      for (i32 i=0;i<nd;i++) { double r=lhs[3*i+d]-rhs[3*i+d]; s+=r; if (fabs(r)>maxr) maxr=fabs(r); }
      if (fabs(s)>sumr) sumr=fabs(s);
    }
    if (maxr > worst) worst = maxr;
    printf("%-24s %10.6f %10.6f %12.3e %12.3e %12s\n",
           c.name, volume, area, sumr, maxr, maxr < 1e-9 ? "ok" : "FAIL");
  }
  printf("\nworst GCL residual over all cases and basis functions: %.3e   %s\n",
         worst, worst < 1e-9 ? "PASS -- free-stream preservation is achievable"
                             : "FAIL -- cut-cell DG would emit spurious sources");
  return worst < 1e-9 ? 0 : 1;
}
