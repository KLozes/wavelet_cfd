#ifndef FEM_BLADE_SDF_H
#define FEM_BLADE_SDF_H

//
// The bladed-sector solid, as SDF algebra.
//
// Instead of trimming and offsetting the blade surface (which ties the fillet
// to span stations and cannot follow the true, oblique blade/platform
// intersection curve), the geometry is composed from signed distance fields:
//
//     blade     = phi_blade(x)   (THIS blade only -- no neighbour copies)
//     platform  = annular slab, r in [r_hub - t, r_hub], z in [z0, z1]
//     solid     = roundUnion(blade, platform, R_fillet)      <- the root fillet
//     solid     = max(solid, (r_hub - t) - r)                <- flat underside
//     solid     = max(solid, r - (r_cas - gap))              <- the tip gap
//     sector    = max(solid, (|theta - thc(z)| - pi/N) * r)  <- one pitch, exact
//
// roundUnion is the exact rolling-ball blend for two locally-perpendicular
// surfaces, so the fillet is a true constant-radius blend that automatically
// follows wherever the blade meets the platform -- no span stations involved.
//
// The sector is centred on thc(z), the blade's own mean tangential position at
// each AXIAL station.  A blade sweeps more than a pitch of arc from LE to TE
// (~20 degrees against a 15.7 degree pitch here), so cut planes at a FIXED
// theta would put the blade hard against one boundary at the LE and the other
// at the TE.  Following thc(z) keeps the passage centred on the blade the whole
// way through -- the standard single-passage periodic boundary -- and the two
// faces stay exactly one pitch apart, so they remain exact cyclic partners.
//
// PERIODICITY WITHOUT NEIGHBOUR COPIES.  Earlier versions took a min over the
// blade and its +-1 rotated copies, so a slice of a neighbouring blade could
// enter the sector.  That is geometrically defensible but it made those slices
// FREE-FLOATING bodies in a single passage: a neighbour slice entering above
// the platform touches nothing else in the domain and, on the load case where
// only the platform underside is clamped, adds rigid-body null modes that make
// CG diverge.  So the SDF now uses THIS blade alone.  The two theta faces still
// carry matching material because near them the only solid is the PLATFORM,
// which is a full annulus (theta-independent) and therefore exactly periodic;
// the blade is centred and narrow (half-width ~0.05 rad << pi/N ~ 0.14 rad) and
// never reaches a face.  So Omega is periodic at the faces by the platform, and
// the cyclic node tie connects the sector through it -- one simply-connected
// solid, blade on platform on clamp.
//
// Frame: machine axis +Z, X = r cos(theta), Y = r sin(theta).
//

#include "Util.cuh"
#include "Vec3f.cuh"
#include "Bvh.h"
#include "BvhQuery.h"

struct BladeSdf {
  // ---- blade solid (BVH over the full-span loft, centred on theta = 0) ----
  const BvhNode *bvhNodes = nullptr;
  const i32     *bvhOrder = nullptr;
  const TriFeat *bvhTris  = nullptr;
  real           orient   = 1;

  // ---- flow-path walls, tabulated on a uniform z grid ---------------------
  const real *hubTab = nullptr;   // [nWall] hub radius
  const real *casTab = nullptr;   // [nWall] casing radius
  i32   nWall = 0;
  real  wallZ0 = 0, wallDz = 1;

  // ---- platform -----------------------------------------------------------
  real platThick = 0;             // radial thickness below the hub line
  real platZ0 = 0, platZ1 = 0;    // axial extent
  i32  platOn = 0;

  // ---- blends / cuts ------------------------------------------------------
  real filletR = 0;               // root fillet radius
  real tipGap  = 0;               // tip clearance below the casing
  real halfPitch = 0;             // pi/nSectors; <= 0 disables the sector cut
  real pitch = 0;                 // 2*pi/nSectors (neighbour rotation)
  const real *thcTab = nullptr;   // [nWall] sector centreline theta_c(z)
  i32  nCopies = 1;               // number of blade copies: 1 = this one only,
                                  // otherwise the (nCopies-1)/2 neighbours each
                                  // side.  The field is only EXACTLY periodic
                                  // once every blade that can be nearest to a
                                  // point in the sector is included; this root
                                  // sweeps 1.25 pitches, so +-1 is not enough
                                  // and leaves the two theta faces slightly
                                  // mismatched.

  // ---- coordinate system --------------------------------------------------
  // COORD_CART: the computational coordinates ARE the physical ones.
  // COORD_CYL : (q0,q1,q2) = (r, rRef*theta', z) with theta = theta'+thc(z), so
  //   the swept one-pitch sector is a BOX.  q1 is an arc length at the
  //   reference radius rather than an angle, which keeps all three
  //   computational coordinates in length units so cubic cells stay cubic.
  i32  coordMode = 0;
  real rRef = 1;
  // Evaluate the level set PERIODICALLY: theta' is wrapped into one pitch
  // before the blade is queried, so phi is periodic to machine precision by
  // construction rather than by relying on enough neighbour copies.  Without
  // it the one-cell active-mesh halo comes out asymmetric -- elements are
  // activated from interior corners, which are not periodic images -- and the
  // node columns on the two theta faces do not match, leaving nodes that
  // cannot be tied.
  i32  wrapTheta = 0;

  // grid -> world offset.  The solver's grid spans 0..domainSize, but the
  // sector SDF is built about the MACHINE AXIS, so the geometry cannot simply
  // be shifted into the grid frame the way an STL body can -- the origin is
  // added here instead.
  real org[3] = {0, 0, 0};

  // computational (grid-frame) -> physical Cartesian
  __host__ __device__ void toPhys(real q0, real q1, real q2,
                                  real &X, real &Y, real &Z) const {
    q0 += org[0]; q1 += org[1]; q2 += org[2];
    if (coordMode == 0) { X = q0; Y = q1; Z = q2; return; }
    real th = q1/rRef + thcRaw(q2);
    X = q0*cos(th); Y = q0*sin(th); Z = q2;
  }

  // ---- wall lookup --------------------------------------------------------
  __host__ __device__ real wallAt(const real *tab, real z) const {
    if (nWall <= 1) return 0;
    real u = (z - wallZ0)/wallDz;
    i32 i = (i32)floor(u);
    if (i < 0) { i = 0; u = 0; }
    if (i > nWall-2) { i = nWall-2; u = (real)(nWall-1); }
    real w = u - i;
    return tab[i] + w*(tab[i+1] - tab[i]);
  }
  // slope of a tabulated wall, d(radius)/dz -- the hub and casing surfaces are
  // r = r_wall(z), so their gradients carry this term
  __host__ __device__ real wallSlope(const real *tab, real z) const {
    if (nWall <= 1) return 0;
    real u = (z - wallZ0)/wallDz;
    i32 i = (i32)floor(u);
    if (i < 0) i = 0;
    if (i > nWall-2) i = nWall-2;
    return (tab[i+1] - tab[i])/wallDz;
  }
  __host__ __device__ real hubR(real z) const { return wallAt(hubTab, z); }
  __host__ __device__ real casR(real z) const { return wallAt(casTab, z); }
  __host__ __device__ real thcRaw(real z) const {
    return thcTab ? wallAt(thcTab, z) : (real)0;
  }
  __host__ __device__ real thc(real z) const { return thcRaw(z); }

  // ---- one blade copy, rotated by -k*pitch about the axis -----------------
  //
  // One blade copy, rotated by -k*pitch about the axis, WITH its gradient.
  //
  // The oracle already computes the gradient on its way to the sign (it is the
  // eikonal direction to the nearest feature), so carrying it out costs nothing.
  // The query point is rotated by -k*pitch, so the gradient in the unrotated
  // frame is R(+k*pitch) applied to what comes back.
  //
  __host__ __device__ real bladePhi(real x, real y, real z, i32 k, real g[3]) const {
    real ca = 1, sa = 0;
    if (k) {
      real a = -k*pitch;
      ca = cos(a); sa = sin(a);
      real xr = ca*x - sa*y, yr = sa*x + ca*y;
      x = xr; y = yr;
    }
    float3 gf;
    // pseudonormal sign, not ray parity: see signedDistancePseudo
    real d = (real)signedDistancePseudo(bvhNodes, bvhOrder, bvhTris,
                                        make_float3((float)x, (float)y, (float)z), gf);
    if (k) {                                  // rotate the gradient back: R(+k*pitch)
      g[0] =  ca*(real)gf.x + sa*(real)gf.y;
      g[1] = -sa*(real)gf.x + ca*(real)gf.y;
    } else {
      g[0] = (real)gf.x; g[1] = (real)gf.y;
    }
    g[2] = (real)gf.z;
    return d;
  }
  __host__ __device__ real bladePhi(real x, real y, real z, i32 k) const {
    real g[3];
    return bladePhi(x, y, z, k, g);
  }

  // ---- exact box SDF from the signed slab distances -----------------------
  __host__ __device__ static real boxPhi(real a, real b) {
    real ax = fmax(a, (real)0), bx = fmax(b, (real)0);
    return sqrt(ax*ax + bx*bx) + fmin(fmax(a, b), (real)0);
  }
  // ... and its gradient, given the gradients of the two slab distances
  __host__ __device__ static real boxPhiG(real a, const real ga[3],
                                          real b, const real gb[3], real g[3]) {
    real ax = fmax(a, (real)0), bx = fmax(b, (real)0);
    real out = sqrt(ax*ax + bx*bx);
    if (out > (real)1e-30) {                    // outside: distance to the corner
      for (i32 d = 0; d < 3; d++) g[d] = (ax*ga[d] + bx*gb[d])/out;
      return out;
    }
    const real *gm = (a > b) ? ga : gb;         // inside: the nearer wall governs
    for (i32 d = 0; d < 3; d++) g[d] = gm[d];
    return fmin(fmax(a, b), (real)0);
  }

  // ---- rolling-ball round union (the fillet) ------------------------------
  //   u = max(R-a,0), v = max(R-b,0);  max(R, min(a,b)) - sqrt(u^2+v^2)
  // Exact circular blend where the two surfaces are locally perpendicular, and
  // degenerates to min(a,b) once either surface is farther than R away.
  __host__ __device__ static real roundUnion(real a, real b, real R) {
    if (R <= 0) return fmin(a, b);
    real u = fmax(R - a, (real)0), v = fmax(R - b, (real)0);
    return fmax(R, fmin(a, b)) - sqrt(u*u + v*v);
  }
  __host__ __device__ static real roundUnionG(real a, const real ga[3],
                                              real b, const real gb[3],
                                              real R, real g[3]) {
    if (R <= 0) {
      const real *gm = (a < b) ? ga : gb;
      for (i32 d = 0; d < 3; d++) g[d] = gm[d];
      return fmin(a, b);
    }
    real u = fmax(R - a, (real)0), v = fmax(R - b, (real)0);
    real w = sqrt(u*u + v*v);
    real mn = fmin(a, b);
    const real *gmn = (a < b) ? ga : gb;
    // d/dx [ max(R,min) - sqrt(u^2+v^2) ] ; du/dx = -ga where u > 0
    for (i32 d = 0; d < 3; d++) {
      real t = (mn > R) ? gmn[d] : (real)0;
      if (w > (real)1e-30) t += (u*ga[d] + v*gb[d])/w;
      g[d] = t;
    }
    return fmax(R, mn) - w;
  }

  // ---- gradients of the analytic surfaces (physical frame) ----------------
  // r = sqrt(x^2+y^2): grad_x r = (x/r, y/r, 0)
  __host__ __device__ void gradR(real x, real y, real rr, real g[3]) const {
    if (rr > (real)1e-30) { g[0] = x/rr; g[1] = y/rr; g[2] = 0; }
    else { g[0] = 1; g[1] = 0; g[2] = 0; }
  }

  //
  // Composite level set WITH its true gradient, both in the COMPUTATIONAL
  // frame (the frame the marching-tets cut machinery works in).  The gradient
  // is threaded through every CSG operation -- min/max/roundUnion each just
  // SELECT or blend the operands' gradients -- so the stored normal is the
  // normal of the actual solid, not of the blade BVH alone.
  //
  // Physical -> computational: in Cartesian the two frames coincide.  In
  // cylindrical mode q = (r, rRef*theta', z) with x = r cos(th), y = r sin(th),
  // th = q1/rRef + thc(z).  For a scalar f the chain rule gives
  //   df/dq0 =  cos th f_x + sin th f_y                       (= f_r)
  //   df/dq1 = (-sin th f_x + cos th f_y) * (r/rRef)          (= f_theta * r/rRef)
  //   df/dq2 =  f_z + thc'(z) * (-y f_x + x f_y)
  // The theta faces are grid faces (wrapTheta), so phi is periodic and this map
  // is what a consumer in computational coordinates needs.
  //
  __host__ __device__ real phiGrad(real q0, real q1, real q2, real g[3]) const {
    real x, y, z;
    toPhys(q0, q1, q2, x, y, z);
    real rr   = (coordMode == 0) ? sqrt(x*x + y*y) : (q0 + org[0]);
    real thAbs = (coordMode == 0) ? atan2(y, x) : ((q1 + org[1])/rRef + thc(z));

    // blades (min over copies), carrying the winning gradient
    real gb[3];
    real pb = bladePhi(x, y, z, 0, gb);
    for (i32 k = 1; k <= (nCopies-1)/2; k++) {
      real gk[3];
      real p1 = bladePhi(x, y, z, -k, gk);
      if (p1 < pb) { pb = p1; for (i32 d=0; d<3; d++) gb[d]=gk[d]; }
      real p2 = bladePhi(x, y, z,  k, gk);
      if (p2 < pb) { pb = p2; for (i32 d=0; d<3; d++) gb[d]=gk[d]; }
    }

    real s = pb, gs[3] = {gb[0], gb[1], gb[2]};
    real gRad[3]; gradR(x, y, rr, gRad);

    if (platOn) {
      real rh = hubR(z), slope = wallSlope(hubTab, z);
      // dr = max(rr - rh, (rh-t) - rr).  grad(rr) = gRad; grad(rh) = slope * e_z
      real a = rr - rh, b = (rh - platThick) - rr;
      real ga[3] = {gRad[0], gRad[1], gRad[2] - slope};
      real gbb[3] = {-gRad[0], -gRad[1], slope - gRad[2]};
      real dr, gdr[3];
      if (a > b) { dr = a; for (i32 d=0;d<3;d++) gdr[d]=ga[d]; }
      else       { dr = b; for (i32 d=0;d<3;d++) gdr[d]=gbb[d]; }
      // dz = max(z0 - z, z - z1)
      real dz, gdz[3] = {0,0,0};
      if (platZ0 - z > z - platZ1) { dz = platZ0 - z; gdz[2] = -1; }
      else                          { dz = z - platZ1; gdz[2] =  1; }
      real gbox[3]; real box = boxPhiG(dr, gdr, dz, gdz, gbox);
      s = roundUnionG(pb, gb, box, gbox, filletR, gs);
      // flat underside trim: max(s, (rh - t) - rr)
      real u = (rh - platThick) - rr;
      real gu[3] = {-gRad[0], -gRad[1], slope - gRad[2]};
      if (u > s) { s = u; for (i32 d=0;d<3;d++) gs[d]=gu[d]; }
    }

    if (tipGap > 0) {
      real slope = wallSlope(casTab, z);
      real u = rr - (casR(z) - tipGap);
      real gu[3] = {gRad[0], gRad[1], gRad[2] - slope};
      if (u > s) { s = u; for (i32 d=0;d<3;d++) gs[d]=gu[d]; }
    }

    if (halfPitch > 0) {                       // Cartesian-only sector cut
      real th = thAbs - thc(z);
      while (th >  (real)M_PI) th -= (real)(2*M_PI);
      while (th < -(real)M_PI) th += (real)(2*M_PI);
      real u = (fabs(th) - halfPitch)*rr;
      // grad(th) = (-y, x)/r^2 (physical); the r factor and sign(th) fold in
      real sgn = (th >= 0) ? (real)1 : (real)-1;
      real gth[3] = {-y/(rr*rr), x/(rr*rr), 0};
      real gu[3];
      for (i32 d=0; d<3; d++) gu[d] = sgn*gth[d]*rr + (fabs(th)-halfPitch)*gRad[d];
      if (u > s) { s = u; for (i32 d=0;d<3;d++) gs[d]=gu[d]; }
    }

    // physical gradient gs -> computational frame
    if (coordMode == 0) {
      g[0] = gs[0]; g[1] = gs[1]; g[2] = gs[2];
    } else {
      real c = cos(thAbs), sn = sin(thAbs);
      real fr = c*gs[0] + sn*gs[1];
      real ft = -sn*gs[0] + c*gs[1];
      real slopeThc = thcSlope(z);
      g[0] = fr;
      g[1] = ft*(rr/rRef);
      g[2] = gs[2] + slopeThc*(-y*gs[0] + x*gs[1]);
    }
    return s;
  }

  // slope of the sector centreline, d(thc)/dz (finite difference on the table)
  __host__ __device__ real thcSlope(real z) const {
    if (!thcTab || nWall <= 1) return 0;
    real u = (z - wallZ0)/wallDz;
    i32 i = (i32)floor(u);
    if (i < 0) i = 0;
    if (i > nWall-2) i = nWall-2;
    return (thcTab[i+1] - thcTab[i])/wallDz;
  }

  // ---- the composite solid ------------------------------------------------
  __host__ __device__ real phi(real q0, real q1, real q2) const {
    real x, y, z;
    toPhys(q0, q1, q2, x, y, z);
    // in cylindrical mode r and theta' are the computational coordinates
    // themselves, so take them directly instead of rebuilding them from x,y
    real rr   = (coordMode == 0) ? sqrt(x*x + y*y) : (q0 + org[0]);
    real thp0 = (coordMode == 0) ? (real)0 : (q1 + org[1])/rRef;

    // periodic evaluation: fold theta' back into one pitch and re-form the
    // Cartesian query point for the blade BVH
    if (wrapTheta && pitch > 0 && coordMode != 0) {
      real thw = thp0 - pitch*rint(thp0/pitch);
      real ta  = thw + thc(z);
      x = rr*cos(ta); y = rr*sin(ta);
      thp0 = thw;
    }

    // blades: this one plus, optionally, the two neighbours that reach into
    // the sector
    real pb = bladePhi(x, y, z, 0);
    for (i32 k = 1; k <= (nCopies-1)/2; k++) {
      pb = fmin(pb, bladePhi(x, y, z, -k));
      pb = fmin(pb, bladePhi(x, y, z,  k));
    }

    real s = pb;
    if (platOn) {
      real rh = hubR(z);
      real dr = fmax(rr - rh, (rh - platThick) - rr);
      real dz = fmax(platZ0 - z, z - platZ1);
      // full annulus in theta: the sector is cut ONCE at the end so the blade
      // and the platform are cut on exactly the same planes
      s = roundUnion(pb, boxPhi(dr, dz), filletR);
      // The loft runs well below the hub so the SDF can do the cutting, which
      // leaves blade sticking out under the platform.  Trim it flat AFTER the
      // union, not before: clipping the blade first would make its underside
      // coincide with the platform's, and roundUnion of two coincident surfaces
      // blisters outward by 0.41*R instead of leaving a flat face.
      s = fmax(s, (rh - platThick) - rr);
    }

    // tip clearance: intersect with { r <= r_cas - gap }
    if (tipGap > 0) s = fmax(s, rr - (casR(z) - tipGap));

    // one exact pitch, centred on the blade at this axial station
    if (halfPitch > 0) {
      real th = thp0;
      if (coordMode == 0) {
        th = atan2(y, x) - thc(z);
        while (th >  (real)M_PI) th -= (real)(2*M_PI);
        while (th < -(real)M_PI) th += (real)(2*M_PI);
      }
      s = fmax(s, (fabs(th) - halfPitch)*rr);
    }
    return s;
  }
};

#endif
