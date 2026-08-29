#ifndef COMMON_CUTELEM_H
#define COMMON_CUTELEM_H

// ---------------------------------------------------------------------------
//  CUT-ELEMENT OPERATORS for a discontinuous Galerkin discretisation.
//
//  Builds, per cut element, everything a DG RHS needs over an implicitly
//  defined region: a positive volume rule, the six cut-face rules, the wall
//  rule, and the dense mass matrix that replaces DGSEM's diagonal one.
//
//  THE CENTRAL TRICK -- BOUNDARY-DERIVED MOMENTS.
//
//  A DG scheme is FREE-STREAM PRESERVING only if the discrete divergence
//  theorem holds on every element:
//
//      SUM_q w_q d(psi_m)/dx_d (x_q)  ==  CLOSED INT psi_m n_d dS          (*)
//
//  If it does not, a cut cell emits a spurious source under uniform flow.
//  Measured on raw Saye rules this fails at ~1e-3 for a wall grazing a cell
//  face, and neither raising the quadrature order nor the skew-symmetric form
//  fixes it -- the skew form buys ENERGY STABILITY but still leaves
//  1/2 * (residual of *) on a constant state.
//
//  The fix, following Taylor, Wilcox & Chan (arXiv:2404.06630) Appendix 6.3:
//  do not ask the volume rule to be accurate on its own.  Compute its TARGET
//  MOMENTS from the BOUNDARY rules via the divergence theorem, then fit weights
//  to those targets.  (*) then holds to the accuracy of the closed-surface
//  identity CLOSED INT n dS = 0, which is a far more robust quantity -- it
//  measures 1e-14 on the same cells where the raw rules are 1e-3.
//
//  We differ from the paper in the fitting step, and better: they moment-fit
//  onto approximate Fekete points via a pivoted QR, which "can return negative
//  quadrature weights" (their 6.3.3, conditioning up to 6.15).  We fit with
//  Lawson-Hanson NNLS onto the Saye points, so the weights are NON-NEGATIVE by
//  construction and the conditioning is 1 identically.
//
//  Basis is TOTAL-DEGREE P^N, not tensor Q^N: the operators are dense either
//  way, and P^N drops the basis 64 -> 20 at N=3 and the moment count 343 -> 84,
//  which is what keeps both this setup and the eventual entropy-stable flux
//  differencing affordable.  Same split the paper makes (their Eqs. 8-9).
//
//  Everything is on the REFERENCE cell [0,1]^3; the caller folds in h.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <vector>

#include "Util.cuh"
#include "Poly.h"
#include "SayeQuad.h"
#include "CutQuadCompress.h"

// small dense SPD solve (Cholesky, in place on A; b overwritten with the answer)
inline bool srdSolveSPDLocal(std::vector<double> &A, std::vector<double> &b, i32 n) {
  for (i32 j=0;j<n;j++) {
    double d=A[(size_t)j*n+j];
    for (i32 q=0;q<j;q++) d-=A[(size_t)j*n+q]*A[(size_t)j*n+q];
    if (d<=1e-300) return false;
    d=sqrt(d); A[(size_t)j*n+j]=d;
    for (i32 i=j+1;i<n;i++){ double s=A[(size_t)i*n+j];
      for (i32 q=0;q<j;q++) s-=A[(size_t)i*n+q]*A[(size_t)j*n+q]; A[(size_t)i*n+j]=s/d; }
  }
  for (i32 i=0;i<n;i++){ double s=b[i]; for(i32 q=0;q<i;q++) s-=A[(size_t)i*n+q]*b[q]; b[i]=s/A[(size_t)i*n+i]; }
  for (i32 i=n-1;i>=0;i--){ double s=b[i]; for(i32 q=i+1;q<n;q++) s-=A[(size_t)q*n+i]*b[q]; b[i]=s/A[(size_t)i*n+i]; }
  return true;
}

// Total-degree-N monomials, centred and scaled for conditioning.  Provides the
// value, the gradient, and the ANALYTIC primitive along each axis -- the last
// is what turns the boundary moment integrals into exact evaluations rather
// than yet another quadrature.
struct CutBasis {
  i32 N, nb;
  double c[3], s;
  __host__ __device__ static i32 count(i32 N) { return (N+1)*(N+2)*(N+3)/6; }

  __host__ __device__ void init(i32 N_, const double cc[3], double ss) {
    N=N_; nb=count(N); c[0]=cc[0]; c[1]=cc[1]; c[2]=cc[2]; s=(ss>0)?ss:1.0;
  }
  // exponent triple of basis function m, ordered by total degree
  __host__ __device__ void expo(i32 m, i32 e[3]) const {
    i32 t=0;
    for (i32 d=0; d<=N; d++)
      for (i32 i=d;i>=0;i--) for (i32 j=d-i;j>=0;j--) {
        if (t==m){ e[0]=i; e[1]=j; e[2]=d-i-j; return; }
        t++;
      }
    e[0]=e[1]=e[2]=0;
  }
  __host__ __device__ void ref(const double X[3], double u[3]) const {
    for (i32 d=0;d<3;d++) u[d]=(X[d]-c[d])/s;
  }
  __host__ __device__ static double ipow(double x, i32 k){ double t=1; for(i32 a=0;a<k;a++) t*=x; return t; }

  __host__ __device__ void eval(const double X[3], double *psi) const {
    double u[3]; ref(X,u); i32 e[3];
    for (i32 m=0;m<nb;m++){ expo(m,e); psi[m]=ipow(u[0],e[0])*ipow(u[1],e[1])*ipow(u[2],e[2]); }
  }
  // dpsi[3*m+d]
  __host__ __device__ void grad(const double X[3], double *dpsi) const {
    double u[3]; ref(X,u); i32 e[3];
    for (i32 m=0;m<nb;m++){ expo(m,e);
      for (i32 d=0;d<3;d++){
        if (e[d]==0){ dpsi[3*m+d]=0; continue; }
        double t=(double)e[d]/s;
        for (i32 a=0;a<3;a++) t *= ipow(u[a], (a==d)? e[a]-1 : e[a]);
        dpsi[3*m+d]=t;
      } }
  }
  // Primitive of psi_m along axis d: P with dP/dX_d == psi_m.  Exact, since a
  // monomial integrates to a monomial.  The additive constant is irrelevant --
  // a constant integrates to zero against a closed surface.
  __host__ __device__ double prim(i32 m, i32 d, const double X[3]) const {
    double u[3]; ref(X,u); i32 e[3]; expo(m,e);
    double t = s/(double)(e[d]+1);
    for (i32 a=0;a<3;a++) t *= ipow(u[a], (a==d)? e[a]+1 : e[a]);
    return t;
  }
};

struct CutElemOps {
  CutBasis B;
  std::vector<SayeNode> vol;        // FITTED positive volume rule
  std::vector<SayeNode> face[6];    // cut-face rules (normal = +/- e_d)
  std::vector<SayeNode> wall;       // wall rule, .n carries the outward normal
  std::vector<double>   Mchol;      // Cholesky factor of M = V^T W V
  std::vector<double>   M11inv;     // inverse of the degree<=(N-1) sub-mass --
                                    // the monomial ordering is DEGREE-MAJOR, so
                                    // this is the leading block of M.  Feeds the
                                    // modal-decay trouble sensor: with r = M c in
                                    // hand, eta = 1 - (z^T b)/(c^T r), z = M11^-1 b,
                                    // b = r[0..k) -- no extra mass matvec needed.
  i32                   nbLo=0;     // its size = count(deg-1); 0 => no sensor
  double volume=0, wallArea=0;
  double momResid=0;                // GCL residual BEFORE the correction
  i32    nNegW=0;                   // weights driven negative by the correction
  i32    snap=0;                    // 0 = genuine cut element; 1 = SNAPPED TO FLUID
                                    // (sub-resolution solid pocket dropped); 2 = SNAPPED
                                    // TO SOLID.  A half-seen feature makes the rules
                                    // mutually inconsistent, which is WORSE than not
                                    // seeing it: snapping keeps the GCL exact and costs
                                    // only the pocket volume, which is below resolution
                                    // anyway.
  double bndIncons=0;               // SELF-INCONSISTENCY of the boundary rules:
                                    // max |CLOSED INT psi_m n_d - CLOSED INT psi_m' n_d'|
                                    // over pairs whose derivative functions COINCIDE,
                                    // TOGETHER WITH |CLOSED INT n_d dS| (the constant
                                    // mode, which has no pair -- and is the only row that
                                    // survives at nb == 1).
                                    // The true identity forces these equal, so this is a
                                    // pure boundary-quadrature defect -- and it is the part
                                    // of the GCL residual NO volume-weight correction can
                                    // remove, because those rows of G are identical (or,
                                    // for the constant mode, identically zero).
  bool   ok=false;

  // solve M z = b in place using the stored factor
  void massSolve(double *b) const {
    const i32 n=B.nb;
    for (i32 i=0;i<n;i++){ double t=b[i]; for(i32 q=0;q<i;q++) t-=Mchol[(size_t)i*n+q]*b[q]; b[i]=t/Mchol[(size_t)i*n+i]; }
    for (i32 i=n-1;i>=0;i--){ double t=b[i]; for(i32 q=i+1;q<n;q++) t-=Mchol[(size_t)q*n+i]*b[q]; b[i]=t/Mchol[(size_t)i*n+i]; }
  }
};

// CUT_TRACEFACE=1: print each face rule as it is built.  This exists because a
// TANGENCY cell's accepted geometry is the output of a SEARCH: if the raw build
// leaves bndIncons > qualTol, cutElemBuild below retries over a ladder of
// level-set perturbations and keeps the best.  Anything that changes the raw
// build -- including a shared-face override -- can therefore change which
// perturbation wins, and with it the cell's volume.  MEASURED on the case-9
// wedge (6,6): a NO-OP override (handing face 2 back its own Saye rule) moved
// the accepted volume 8.688436e-02 -> 8.684247e-02, because the unperturbed
// build drops face 0's 4.0e-05 tangency sliver and the perturbed one keeps it.
// That is the search being sensitive, not the override being wrong -- with the
// retry disabled (qualTol = inf) the two builds agree exactly.
static const bool cutTraceFace = (getenv("CUT_TRACEFACE") != nullptr);

// ---------------------------------------------------------------------------
//  Build the operators for one cut element from its level-set fit.
//  `phi < 0` is the ACTIVE region (negate at the caller if your fluid is phi>0).
// ---------------------------------------------------------------------------
// z exponent of mode m in the degree-major total-degree ordering CutBasis uses
inline i32 cutModeKz(i32 m, i32 deg) {
  i32 idx = 0;
  for (i32 d = 0; d <= deg; d++)
    for (i32 i = d; i >= 0; i--)
      for (i32 j = d-i; j >= 0; j--) { if (idx == m) return d-i-j; idx++; }
  return 0;
}

inline bool cutElemBuildRaw(const PolyND &phi, i32 N, CutElemOps &E,
                         SayeArena &ar, const SayeCfg &cfg,
                         std::vector<SayeNode> &scratch,
                         const std::vector<SayeNode> * const *faceOverride = nullptr,
                         const double *ffXi = nullptr, const double *ffW = nullptr,
                         i32 ffN = 0, i32 p2dNz = 0) {
  E.ok=false;
  auto mkset=[&](std::vector<SayeNode> &b){
    SayeSet s; s.p=b.data(); s.n=0; s.cap=(i32)b.size(); s.ovf=false; return s; };

  // ---- boundary rules first: they define the geometry the volume rule must
  //      be consistent with, not the other way round ------------------------
  // ---- PSEUDO-2D: build every rule as (slice) x (Gauss in z) --------------
  //  Symmetric in z BY CONSTRUCTION, because the Gauss rule is, and every
  //  point carries the same in-plane position.  See the note on sayeSlice2D.
  const GaussRule gz = gaussLegendre(p2dNz > 0 ? p2dNz : 1);
  const real z0 = (real)0.5;
  auto extrude = [&](const SayeSet &src, std::vector<SayeNode> &dst) {
    dst.clear(); dst.reserve((size_t)src.n*gz.n);
    for (i32 i = 0; i < src.n; i++)
      for (i32 k = 0; k < gz.n; k++) {
        SayeNode s = src.p[i]; s.x[2] = gz.x[k]; s.w = src.p[i].w*gz.w[k];
        dst.push_back(s);
      }
  };
  if (p2dNz > 0) {
    { SayeSet w=mkset(scratch); sayeCurve2D(phi,z0,&w,&ar,cfg); extrude(w, E.wall); }
    for (i32 d=0; d<2; d++) for (i32 side=0; side<2; side++) {
      const i32 fi = 2*d+side;
      if (faceOverride && faceOverride[fi]) { E.face[fi] = *faceOverride[fi];
        if (cutTraceFace) printf("   [face] fi=%d OVERRIDE n=%zu\n", fi, E.face[fi].size());
        continue; }
      SayeSet f=mkset(scratch); sayeEdge1D(phi,d,(real)(side?1:0),z0,&f,&ar,cfg);
      extrude(f, E.face[fi]);
      if (cutTraceFace) printf("   [face] fi=%d edge1D n=%d -> %zu\n", fi, f.n, E.face[fi].size());
    }
    // the z faces ARE the slice, placed at z = 0 and z = 1
    { SayeSet f=mkset(scratch); sayeSlice2D(phi,z0,&f,&ar,cfg);
      for (i32 side=0; side<2; side++) {
        const i32 fi = 4+side;
        if (faceOverride && faceOverride[fi]) { E.face[fi] = *faceOverride[fi]; continue; }
        E.face[fi].clear();
        for (i32 i=0;i<f.n;i++){ SayeNode s=f.p[i]; s.x[2]=(real)side; E.face[fi].push_back(s); }
      } }
  } else {
  { SayeSet w=mkset(scratch); sayeSurface(phi,&w,&ar,cfg);
    E.wall.assign(w.p, w.p+w.n); }
  for (i32 d=0; d<3; d++) for (i32 side=0; side<2; side++) {
    const i32 fi = 2*d+side;
    // SHARED-FACE RULE.  If an already-built cut neighbour owns this face, take
    // ITS rule (flipped into our frame by the caller) instead of building our
    // own.  The two fits restrict IDENTICALLY to the shared face in exact
    // arithmetic, but in floating point each side evaluates its restriction
    // from different 3-D coefficients and at a tangency the ulp differences
    // flip Saye recursion branches.  MEASURED on case 9 before this line
    // existed: 6 of 8 shared cut<->cut faces disagreed on their OWN AREA by
    // 4.6e-05 (1.7e-04 relative), so the two elements integrated the same
    // physical face over different quadrature -- each self-consistent with its
    // own GCL (free stream still passed at 1e-12) but NOT CONSERVATIVE across
    // the face: what one side says leaves is not what the other says enters.
    // The mismatches come in mirrored pairs of opposite sign, which is why a
    // symmetric gate cancels it and a wake does not.
    // e6c25c2 introduced the override, threaded it through two call levels, and
    // never read it here; this is the line that was missing.
    if (faceOverride && faceOverride[fi]) { E.face[fi] = *faceOverride[fi];
      if (cutTraceFace) printf("   [face] fi=%d OVERRIDE n=%zu\n", fi, E.face[fi].size());
      continue; }
    SayeSet f=mkset(scratch); sayeFace(phi,d,side,&f,&ar,cfg);
    E.face[fi].assign(f.p, f.p+f.n);
    if (cutTraceFace) printf("   [face] fi=%d saye n=%d ovf=%d\n", fi, f.n, (i32)f.ovf);
  }
  }
  // ---- FULLY FLUID FACES TAKE THE SOLVER'S OWN TENSOR RULE ----------------
  //  A face whose fluid area is the whole face is not cut, so there is nothing
  //  for Saye to resolve on it -- and asking Saye anyway is both wasteful and
  //  WRONG IN THE ONE WAY THAT MATTERS.  Wasteful: the recursion subdivides for
  //  a near-tangency that does not touch this face, and it is measured, not
  //  hypothetical -- CUT_TRACEFACE on case 9 shows a single fully-fluid face
  //  carrying 6300 points, and the ordinary ones carrying 100 (a 10x10 Gauss
  //  tensor rule with extra steps).  Wrong: the GCL correction below fits the
  //  volume weights against THESE face integrals, while the DG kernel, on
  //  exactly these faces, substitutes the tensor rule its Cartesian neighbour
  //  lifts over and skips the Saye rule entirely.  The identity is then fitted
  //  against one rule and evaluated against another -- the D1/D2 hypothesis
  //  dgcutjac_test was written to test, which measured the two agreeing to only
  //  5.7e-10 where they should be identical.
  //
  //  So take the caller's rule (ffXi/ffW, a 1-D rule on [-1,1] with weights
  //  summing to 2 -- the DG solution points) and tensor it onto the face.  This
  //  loses NO accuracy: face integrals here carry psi_m of the SOLUTION degree
  //  N only, and a (N+1)-point GLL rule is exact through degree 2N-1 >= N.
  //  Both sides of a shared face get the same point set, because the node set
  //  is symmetric about the face centre and the mirror map only reverses the
  //  tangential indices.  ffN == 0 keeps the Saye rule (the FEM callers, which
  //  have no such neighbour rule to match).
  if (ffN > 0) {
    for (i32 fi = 0; fi < 6; fi++) {
      double a = 0; for (const SayeNode &s : E.face[fi]) a += (double)s.w;
      if (fabs(a - 1.0) > 1e-6) continue;                 // cut, or empty
      const i32 d = fi/2, side = fi%2;
      const i32 t1 = (d == 0) ? 1 : 0, t2 = (d == 2) ? 1 : 2;
      std::vector<SayeNode> g; g.reserve((size_t)ffN*ffN);
      for (i32 b2 = 0; b2 < ffN; b2++)
        for (i32 a2 = 0; a2 < ffN; a2++) {
          SayeNode s{};
          s.x[d]  = (real)(side ? 1.0 : 0.0);
          s.x[t1] = (real)(0.5*(ffXi[a2] + 1.0));
          s.x[t2] = (real)(0.5*(ffXi[b2] + 1.0));
          s.w     = (real)(0.25*ffW[a2]*ffW[b2]);
          s.n[0]=s.n[1]=s.n[2]=(real)0; s.n[d] = (real)(side ? 1.0 : -1.0);
          g.push_back(s);
        }
      if (cutTraceFace)
        printf("   [face] fi=%d FULL -> tensor n=%zu (was %zu)\n",
               fi, g.size(), E.face[fi].size());
      E.face[fi].swap(g);
    }
  }

  E.wallArea=0; for (const SayeNode &s : E.wall) E.wallArea += (double)s.w;

  // ---- candidate volume nodes ---------------------------------------------
  //  Potter, "Fast Construction of Efficient Cut Cell Quadratures" (Sec 4.1,
  //  docs/PotterCutCellQuadrature.pdf): EVERY method in that paper begins from
  //  the same discretisation, and it is NOT the raw Saye points --
  //      Alg 4.2  1) fit a TIGHT BOUNDING BOX to the cut region Omega;
  //               2) map that box to [-1,1]^d  (call it F);
  //               3) grid [-1,1]^d uniformly with spacing h;
  //               4) candidates = grid nodes lying inside F(Omega),
  //  with the moments carried through by the change of variables (his 4.1).
  //  His reason for step 2 is stated plainly in Sec 4.1: rescaling "allows a
  //  more even distribution of grid nodes over Omega which avoids some issues
  //  caused by tiny sliver cut cells".
  //
  //  We had only ever used the Saye nodes.  Those are feasible BY CONSTRUCTION
  //  -- they reproduce the moments they were built from, which a grid need not
  //  -- but their positions are dictated by the height-function recursion, so
  //  on a thin cell they bunch into the subdivision pattern instead of covering
  //  the region.  The candidate set is what the NNLS fit and the GCL correction
  //  below get to choose from, so a bunched set is a bad basis for both.
  //
  //  So: take the UNION.  The Saye nodes keep feasibility unconditional, the
  //  bounding-box grid supplies the even cover Potter asks for, and NNLS picks.
  //  The grid nodes enter with ZERO WEIGHT, which is what keeps the moment
  //  target b exact -- b is accumulated as sum_q w_q chi(x_q) below, so only the
  //  Saye nodes contribute to it and the change of variables is never needed:
  //  we sample in the box but integrate in the cell's own frame.
  //
  //  CUT_CANDGRID sets the grid resolution per axis; it is OFF by default, and
  //  that is a MEASURED choice, not an untested one.  On case 9 (cylinder,
  //  nlvls 1, p3) switching it on changes nothing worth paying for:
  //      worst bndIncons   4.08e-04 -> 4.08e-04 (fp32),  1.97e-10 -> 1.97e-10 (fp64)
  //      free-stream |RHS| 3.384e-07 -> 3.385e-07 (fp64)
  //      time-marched free-stream growth: the ||dU/dt||/||U|| curve at
  //      gN = 0 / 12 / 20 agrees to 3-4 significant figures at EVERY output
  //      time (5.58e-09 -> 7.13e-06 over t = 0.05..0.5 in all three).
  //  The reason is structural and worth stating so nobody re-runs this: the
  //  quantity that limits these cells is bndIncons, which is a property of the
  //  BOUNDARY rules alone -- the volume candidate set cannot move it by
  //  construction -- and the GCL correction was already closing to that floor.
  //  Potter's rescaling earns its keep where the candidate set is genuinely
  //  degenerate (a sliver whose Saye nodes cannot span the moment space); case
  //  9's thinnest cut cell is 8.8% of a cell and is nowhere near that.  Kept,
  //  documented and one env var away for the geometry that does hit it.
  std::vector<SayeNode> cand;
  // PSEUDO-2D: the candidates are the 2-D SLICE, not an extrusion of it.  The
  // whole fit -- NNLS moments and the GCL correction -- is done in the slice
  // and the result is swept into z at the end.  That is what makes the 3-D rule
  // exactly z-symmetric AND exactly GCL-consistent: with w_{q,k} = w2_q wg_k the
  // d = x,y rows factor as (2-D GCL row) x (Gauss z-moment), and the d = z rows
  // reduce to (2-D moment) x [z^k] which the two z-faces reproduce identically.
  // Fitting in 3-D and symmetrising afterwards does NOT work (see the note by
  // the mass build): the correction has already pinned G w = g, so perturbing w
  // re-opens the residual.
  if (p2dNz > 0) { SayeSet v=mkset(scratch); sayeSlice2D(phi,z0,&v,&ar,cfg);
                   cand.assign(v.p, v.p+v.n); }
  else { SayeSet v=mkset(scratch); sayeVolume(phi,&v,&ar,cfg);
         cand.assign(v.p, v.p+v.n); }
  if (cand.empty()) return false;
  {
    static const i32 gN = getenv("CUT_CANDGRID") ? atoi(getenv("CUT_CANDGRID")) : 0;
    if (gN > 1) {
      // Step 1, the tight bounding box.  Potter's Alg 4.1 refines one out of the
      // inside-outside test alone; we do not need it -- the boundary rules ARE a
      // sample of dOmega, so vol U wall U face already brackets Omega tightly and
      // costs nothing.
      double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
      auto grow = [&](const std::vector<SayeNode> &v2) {
        for (const SayeNode &s : v2) for (i32 d=0; d<3; d++) {
          lo[d] = fmin(lo[d], (double)s.x[d]); hi[d] = fmax(hi[d], (double)s.x[d]); } };
      grow(cand); grow(E.wall);
      for (i32 f=0; f<6; f++) grow(E.face[f]);
      // one grid cell of margin, clamped to the reference cell: the box must
      // CONTAIN Omega, and the sampled extent is an inner estimate of it
      bool ok = true;
      for (i32 d=0; d<3; d++) {
        if (!(lo[d] <= hi[d])) { ok = false; break; }
        const double m2 = (hi[d]-lo[d])/(double)gN;
        lo[d] = fmax(0.0, lo[d]-m2); hi[d] = fmin(1.0, hi[d]+m2);
      }
      if (ok) {
        // Steps 2-4.  Sampling uniformly in the BOX is the same set of points as
        // gridding [-1,1]^3 uniformly and mapping back, so F stays implicit.
        for (i32 a=0; a<gN; a++)
        for (i32 b2=0; b2<gN; b2++)
        for (i32 c2=0; c2<gN; c2++) {
          double X[3] = { lo[0] + (a +0.5)*(hi[0]-lo[0])/gN,
                          lo[1] + (b2+0.5)*(hi[1]-lo[1])/gN,
                          lo[2] + (c2+0.5)*(hi[2]-lo[2])/gN };
          real xr[3] = {(real)X[0], (real)X[1], (real)X[2]};
          if (phi.eval(xr) >= 0) continue;             // outside Omega
          SayeNode s{}; s.x[0]=xr[0]; s.x[1]=xr[1]; s.x[2]=xr[2]; s.w=(real)0;
          cand.push_back(s);
        }
      }
    }
  }

  // ---- basis, adapted to the CUT REGION ----------------------------------
  // This used to be scaled over the WHOLE reference cell, for a reason that no
  // longer holds and is worth recording because it inverted the trade-off:
  // a region-adapted MONOMIAL basis has u = (X-c)/s growing like
  // (cell size)/(sliver size), so psi ~ u^N explodes at the tensor solution
  // nodes -- which a nodally-stored element must evaluate -- and that blew a
  // uniform state to 1e24 in one RK step.  But the basis is ORTHONORMALISED
  // against this element's own rule (psi~ = L^-1 psi, added later), and
  // orthonormalisation normalises that scale away: MEASURED on the case-9
  // tangency wedge, cell-wide and region-adapted give the SAME max|psi~|,
  // 54.3 over the fluid rule and 2.44e+04 at the tensor nodes, to the digit.
  // The 54.3 is intrinsic -- it is the Christoffel function of a 0.083-cell-
  // thick region and no basis choice removes it.  What the region scale DOES
  // buy is the conditioning of the mass matrix itself:
  //     kappa(M)   wedge 1.22e+07 -> 5.53e+04 (220x)
  //                quarter 5.82e+03 -> 1.54e+02 (38x)
  // which is what decides whether the element can carry its own degree at all,
  // and that now matters because there is no degree ladder to fall back on.
  double cc[3];
  double wsum=0; for (const SayeNode &s : cand) wsum+=(double)s.w;
  if (wsum<=0) return false;
  E.volume = wsum;      // raw geometric volume, published EARLY so that a
                        // caller can still judge the snap criteria if the
                        // element turns out not to support its degree
  { double m1[3]={0,0,0};
    for (const SayeNode &s : cand)
      for (i32 d=0;d<3;d++) m1[d]+=(double)s.w*(double)s.x[d];
    for (i32 d=0;d<3;d++) cc[d]=m1[d]/wsum; }
  double sc=0;          // rms radius of the fluid region about its centroid
  for (const SayeNode &s : cand) { double r2=0;
    for (i32 d=0;d<3;d++){ double t=(double)s.x[d]-cc[d]; r2+=t*t; }
    sc += (double)s.w*r2; }
  sc /= wsum;
  // a degenerate rule (all points coincident) would give s = 0; fall back to
  // the old cell-wide scale rather than dividing by it
  if (!(sc > 1e-30)) { cc[0]=cc[1]=cc[2]=0.5; sc=0.75; }
  E.B.init(2*N, cc, sqrt(sc));            // moments to total degree 2N
  const i32 nm = E.B.nb;                  // number of moment constraints

  // Solution basis, SAME centre and scale -- the kernel reads both from
  // cutCen, so they must not diverge.
  CutBasis Bs; Bs.init(N, cc, sqrt(sc));
  i32 nb=Bs.nb, nG=3*nb;

  // ---- WHAT THE FIT MUST ACHIEVE ----------------------------------------
  // Two blocks of rows, and the priority between them is the whole design:
  //
  //   [G]  the 3*nb GCL rows   SUM_q w_q d(psi_m)/dx_d == CLOSED INT psi_m n_d
  //        -- these are exactly what free-stream preservation needs, and they
  //           must hold EXACTLY.
  //   [M]  the nm moment rows  SUM_q w_q chi_m == the raw Saye rule's moments
  //        -- these are what makes the rule ACCURATE, and they are negotiable.
  //
  // Fitting the moments to BOUNDARY-derived targets (the paper's Appendix 6.3)
  // does NOT work here and it is worth recording why: at a near-tangency the
  // wall rule is genuinely wrong (its area bounces 2.888..3.067 over ng=5..10),
  // so no positive measure on these nodes can reproduce boundary moments AND
  // stay accurate -- NNLS then splits the difference and leaves the residual in
  // the GCL, which is the one place it must not go.  Consistency with a
  // slightly-wrong boundary preserves free-stream; inconsistency does not.
  // So: weight [G] heavily and let [M] absorb the geometry error.
  // ---- step 1: COMPRESS for accuracy -------------------------------------
  // NNLS onto the raw Saye points, targeting the Saye rule's OWN total-degree
  // 2N moments.  Positive weights, ~nm points, and accurate by construction.
  const i32 nc=(i32)cand.size();
  {
    // PSEUDO-2D: only the kz == 0 moments are independent constraints.  With the
    // tensor form the 3-D moment of x^i y^j z^k is (2-D moment) x (Gauss z-moment)
    // and the z factor is exact and cancels against the target, so every k for a
    // given (i,j) is the SAME condition -- fitting them all would just repeat it.
    std::vector<i32> mrow;                      // moment rows actually used
    for (i32 m=0;m<nm;m++)
      if (p2dNz<=0 || cutModeKz(m, 2*N)==0) mrow.push_back(m);
    const i32 nmU=(i32)mrow.size();
    std::vector<double> A((size_t)nc*nmU), b(nmU,0.0), chi(nm), wfit;
    for (i32 q=0;q<nc;q++){
      double X[3]={(double)cand[q].x[0],(double)cand[q].x[1],(double)cand[q].x[2]};
      E.B.eval(X, chi.data());
      for (i32 t=0;t<nmU;t++){ A[(size_t)q*nmU+t]=chi[mrow[t]];
                               b[t]+=(double)cand[q].w*chi[mrow[t]]; }
    }
    // nOuter = 8*nm, not the default nm: one column is admitted per outer pass
    // and the active-set loop removes several, so the default stops the solve
    // early and leaves ~1e-4 of moment error on a rule whose whole job is to
    // reproduce those moments.  See the note in nnlsCore.
    nnls(A, b, nmU, nc, wfit, 1e-9, 8*nmU);
    E.vol.clear();
    for (i32 q=0;q<nc;q++) if (wfit[q]>1e-15){ SayeNode s=cand[q]; s.w=(real)wfit[q]; E.vol.push_back(s); }
  }
  // The correction below solves G dw = r with G of size (3*nb) x npts, so the
  // support must EXCEED the constraint count or the system is over-determined
  // and cannot be closed -- measured: 59 points against 60 constraints left
  // 6.0e-04 of free-stream residual.  NNLS returns whatever support the moments
  // need (~50), so pad with further Saye points carrying zero weight; the
  // correction is free to give them weight, and they are valid quadrature
  // points either way.  Padding by stride keeps them spread out, which keeps G
  // full-rank.
  {
    const i32 want = 2*nG + 8;
    if ((i32)E.vol.size() < want && nc > (i32)E.vol.size()) {
      std::vector<char> used(nc,0);
      for (const SayeNode &s : E.vol)
        for (i32 q=0;q<nc;q++)
          if (used[q]==0 && cand[q].x[0]==s.x[0] && cand[q].x[1]==s.x[1]
                         && cand[q].x[2]==s.x[2]) { used[q]=1; break; }
      i32 stride = nc/(want - (i32)E.vol.size() + 1); if (stride<1) stride=1;
      for (i32 q=0; q<nc && (i32)E.vol.size()<want; q+=stride)
        if (!used[q]) { SayeNode s=cand[q]; s.w=(real)0; E.vol.push_back(s); used[q]=1; }
    }
  }

  // ---- PSEUDO-2D: SWEEP the fitted slice rule into z ----------------------
  //  From here on E.vol is the 3-D tensor rule w_{q,k} = w2_q wg_k.  nSlice
  //  records the slice size so the GCL correction below can keep the tensor
  //  form (one unknown per SLICE point, not per 3-D point).
  i32 nSlice = 0;
  if (p2dNz > 0) {
    nSlice = (i32)E.vol.size();
    std::vector<SayeNode> sw; sw.reserve((size_t)nSlice*gz.n);
    for (i32 q = 0; q < nSlice; q++)
      for (i32 k = 0; k < gz.n; k++) {
        SayeNode t = E.vol[q]; t.x[2] = gz.x[k]; t.w = E.vol[q].w*gz.w[k];
        sw.push_back(t);
      }
    E.vol.swap(sw);
  }

  // ---- step 2+3: GCL correction, then the mass ---------------------------
  // A CUT ELEMENT CARRIES THE SAME DEGREE AS AN UNCUT ONE.  There is no degree
  // ladder here any more, and that is a deliberate reversal.  Three separate
  // mechanisms used to drop the order -- a thin-cell rule (mean thickness
  // < CUT_THINTOL => P0), a conditioning cap (CUT_KAPMAX), and a CUT_DEGMAX
  // override -- and each was a workaround for an instability rather than a
  // property of the geometry.  They are gone because:
  //   * the small-cell problem is STATE REDISTRIBUTION's job (Berger &
  //     Giuliani; Taylor, Wilcox & Chan), and dgsrd_test verifies ours is
  //     conservative, polynomial-exact to degree N and contractive.  Reducing
  //     the degree solves the same problem twice, and the second solution
  //     costs the order the scheme exists to deliver;
  //   * the conditioning cap was measured to be unjustified: on case 9 every
  //     one of the 12 cut elements -- INCLUDING the 0.087-volume tangency
  //     wedges -- factors a full P3 mass matrix with bndIncons ~ 1e-11.  The
  //     cap at 1e2 was collapsing all of them to P0/P1 for nothing;
  //   * a mixed-order mesh is its own problem: a P0 cut element against a P3
  //     Cartesian neighbour is a first-order wall no matter how good the
  //     interior is, and it makes every convergence study meaningless.
  // If the mass matrix will not factor at degree N, that is now a BUILD
  // FAILURE reported to the caller, not a silent order drop.
  std::vector<SayeNode> volKeep(E.vol);      // pre-correction weights
  {
    const i32 deg = N;
  Bs.init(deg, cc, sqrt(sc)); nb = Bs.nb; nG = 3*nb;
  E.vol = volKeep;                           // reset weights for this attempt
  // ---- LEAST-NORM CORRECTION so the GCL holds EXACTLY ---------------------
  // The compressed rule is accurate but not boundary-CONSISTENT: G w - g is the
  // free-stream residual, and on a near-tangency cell it is ~1e-3.  Rather than
  // re-fitting (which lets the solver pick a support too sparse to make the mass
  // matrix SPD -- measured: 17 points for a 20-function basis), perturb the
  // weights as little as possible subject to the constraint:
  //
  //     min ||dw||  s.t.  G dw = -(G w - g)   =>   dw = G^T (G G^T)^-1 (g - G w)
  //
  // G is the 3*nb x npts GCL matrix and g the boundary right-hand side.  This
  // keeps every point (so M stays SPD), makes free-stream EXACT, and moves the
  // moments only by the size of the geometric inconsistency -- which is the
  // right trade: consistency with a slightly-wrong boundary preserves
  // free-stream, inconsistency does not.
  {
    const i32 npFull=(i32)E.vol.size();
    // PSEUDO-2D: solve for a TENSOR correction dw_{q,k} = dw2_q wg_k, i.e. one
    // unknown per slice point.  The constraint row then contracts against the
    // Gauss weights, and the correction cannot break z-symmetry because dw2 is
    // z-independent by construction.
    const bool p2d = (p2dNz > 0 && nSlice > 0);
    const i32 np = p2d ? nSlice : npFull;
    const i32 nz = p2d ? gz.n : 1;
    std::vector<double> G((size_t)nG*np, 0.0), g(nG,0.0), psi(nb), dpsi((size_t)nb*3);
    for (i32 q=0;q<np;q++){
      for (i32 k=0;k<nz;k++){
        const i32 idx = p2d ? (q*nz+k) : q;
        const double zw = p2d ? (double)gz.w[k] : 1.0;
        double X[3]={(double)E.vol[idx].x[0],(double)E.vol[idx].x[1],(double)E.vol[idx].x[2]};
        Bs.grad(X,dpsi.data());
        for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++)
          G[(size_t)(3*m+d)*np+q] += zw*dpsi[3*m+d];
      }
    }
    auto accumG=[&](const double X[3], double w, const double nrm[3]){
      Bs.eval(X, psi.data());
      for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++)
        if (fabs(nrm[d])>1e-15) g[3*m+d] += w*psi[m]*nrm[d];
    };
    for (const SayeNode &s : E.wall) {
      double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
      double nr[3]={(double)s.n[0],(double)s.n[1],(double)s.n[2]};
      accumG(X,(double)s.w,nr);
    }
    for (i32 d=0;d<3;d++) for (i32 side=0;side<2;side++) {
      double nr[3]={0,0,0}; nr[d]= side?1.0:-1.0;
      for (const SayeNode &s : E.face[2*d+side]) {
        double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
        accumG(X,(double)s.w,nr);
      } }
    // How much of r is structurally uncorrectable?  d(psi_m)/dx_d coincides for
    // different (m,d) pairs, so those rows of G are IDENTICAL and G dw = r is
    // solvable only if r matches there.  The true divergence theorem forces the
    // corresponding boundary integrals equal; any difference is boundary
    // quadrature error, and it caps what the correction can achieve.
    {
      E.bndIncons=0;
      for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++) {
        i32 e1[3]; Bs.expo(m,e1); if (e1[d]==0) continue;
        i32 t1[3]={e1[0],e1[1],e1[2]}; t1[d]--;                  // derivative exponents
        double s1=(double)e1[d];
        for (i32 m2=0;m2<nb;m2++) for (i32 d2=0;d2<3;d2++) {
          if (m2==m && d2==d) continue;
          i32 e2[3]; Bs.expo(m2,e2); if (e2[d2]==0) continue;
          i32 t2[3]={e2[0],e2[1],e2[2]}; t2[d2]--;
          if (t1[0]!=t2[0]||t1[1]!=t2[1]||t1[2]!=t2[2]) continue;
          double s2=(double)e2[d2];
          double diff = fabs(g[3*m+d]/s1 - g[3*m2+d2]/s2)*fmin(s1,s2);
          if (diff>E.bndIncons) E.bndIncons=diff;
        } }
      // THE CONSTANT MODE'S ROWS, which the pairwise measure above cannot see:
      // psi_0 == 1 has no derivative row to pair with, so its three GCL rows
      //     SUM_q w_q d(psi_0)/dx_d == CLOSED INT n_d dS
      // reduce to  CLOSED INT n_d dS == 0, a pure geometry statement the volume
      // correction can NEVER touch (grad psi_0 is identically zero, so those
      // rows of G vanish).  It is condition Q_{H,d} 1 = 0 of Taylor & Chan,
      // "An Entropy Stable High-Order DG Method on Cut Meshes" (arXiv:2412.13002
      // Sec. 2.1.4, docs/CutCellEntropyStable.pdf) -- in their framework the
      // constant-differentiation condition is what free-stream preservation AND
      // entropy stability both rest on, so it belongs in the quality metric
      // that gates the epsilon repair below.
      // AT nb == 1 THERE ARE NO PAIRS AT ALL, so bndIncons was identically 0 on
      // every P0 element and the repair was silently disabled exactly where the
      // geometry needed it most: measured on the case-9 tangency wedge (vol
      // 0.087), the unperturbed rule closes to only 1.35e-02 and emits
      // |RHS| ~ 1e1 on a uniform state, while the SAME cell at eps = -1e-5
      // closes to 6.9e-13.  The ladder found that branch at every degree >= 1
      // and never even ran at degree 0.
      for (i32 d=0; d<3; d++) E.bndIncons = fmax(E.bndIncons, fabs(g[3*0+d]));
    }
    // residual r = g - G w.  In the pseudo-2D tensor form the unknown is the
    // SLICE weight w2_q (the 3-D weight is w2_q wg_k), and G has already been
    // contracted against wg, so the product must use w2 -- not the first nSlice
    // entries of the swept array, which are (q=0, k=0..nz-1) and would be
    // nonsense.  Recover w2_q from the swept rule: w_{q,0} = w2_q wg_0.
    std::vector<double> wUse(np);
    for (i32 q=0;q<np;q++)
      wUse[q] = p2d ? (double)E.vol[(size_t)q*nz].w/(double)gz.w[0]
                    : (double)E.vol[q].w;
    std::vector<double> r(nG,0.0);
    for (i32 i=0;i<nG;i++){ double s=g[i];
      for (i32 q=0;q<np;q++) s-=G[(size_t)i*np+q]*wUse[q]; r[i]=s; }
    E.momResid=0; for (double t : r) E.momResid=fmax(E.momResid,fabs(t));
    // (G G^T) y = r, with a small Tikhonov term for the rank-deficient rows
    // (the m=0 GCL rows are 0 == CLOSED INT n_d, which G cannot see)
    std::vector<double> GG((size_t)nG*nG,0.0), y(r);
    for (i32 i=0;i<nG;i++) for (i32 j=i;j<nG;j++){
      double s=0; for (i32 q=0;q<np;q++) s+=G[(size_t)i*np+q]*G[(size_t)j*np+q];
      GG[(size_t)i*nG+j]=GG[(size_t)j*nG+i]=s; }
    double tr=0; for (i32 i=0;i<nG;i++) tr+=GG[(size_t)i*nG+i];
    for (i32 i=0;i<nG;i++) GG[(size_t)i*nG+i]+=1e-12*fmax(tr/nG,1.0);
    {
      // POSITIVITY-CONSTRAINED least-norm correction, active-set style.  An
      // unconstrained dw can drive weights negative (measured: 84 of 128 on a
      // corner-degenerate cell -> indefinite mass -> degree collapse), but a
      // GLOBAL scale-back is wrong too: the padded points carry w0 = 0, so one
      // negative component there zeroed the entire correction and the GCL was
      // never enforced at all.  Instead: solve on the free set; PIN violators
      // at exactly zero (their dw = -w0 moves into the right-hand side);
      // re-solve on the rest.  Terminates in a few rounds; when the target is
      // feasible over the positive cone this reproduces it exactly.
      std::vector<double> dw(np, 0.0), rr(r);
      std::vector<char> freeQ(np, 1);
      for (i32 round = 0; round < 5; round++) {
        std::vector<double> GGf((size_t)nG*nG, 0.0), yy(rr);
        for (i32 i=0;i<nG;i++) for (i32 j=i;j<nG;j++){
          double t=0;
          for (i32 q=0;q<np;q++) if (freeQ[q]) t+=G[(size_t)i*np+q]*G[(size_t)j*np+q];
          GGf[(size_t)i*nG+j]=GGf[(size_t)j*nG+i]=t; }
        double tr2=0; for (i32 i=0;i<nG;i++) tr2+=GGf[(size_t)i*nG+i];
        for (i32 i=0;i<nG;i++) GGf[(size_t)i*nG+i]+=1e-12*fmax(tr2/nG,1.0);
        if (!srdSolveSPDLocal(GGf, yy, nG)) break;
        bool viol=false;
        for (i32 q=0;q<np;q++){
          if (!freeQ[q]) continue;
          double d=0; for (i32 i=0;i<nG;i++) d+=G[(size_t)i*np+q]*yy[i];
          dw[q]=d;
          if (wUse[q] + d < 0) {          // wUse = the SLICE weight (see above)
            // pin at zero: dw = -w0 exactly, and its constraint contribution
            // moves to the right-hand side for the next round
            viol=true; freeQ[q]=0; dw[q]=-wUse[q];
            for (i32 i=0;i<nG;i++) rr[i]-=G[(size_t)i*np+q]*dw[q];
          }
        }
        if (!viol) break;
      }
      for (i32 q=0;q<np;q++)
        for (i32 k=0;k<nz;k++){
          const i32 idx = p2d ? (q*nz+k) : q;
          const double zw = p2d ? (double)gz.w[k] : 1.0;
          E.vol[idx].w = (real)fmax((double)E.vol[idx].w + zw*dw[q], 0.0);
        }
    }
  }
  // NOTE, measured: symmetrising the FITTED weights here (averaging each point
  //  with its z-mirror, which exists by construction in the extruded candidate
  //  set) does NOT work, and the reason is worth keeping.  The least-norm
  //  correction immediately above solves G w = g so the GCL holds EXACTLY;
  //  perturbing w afterwards re-opens a residual at the size of the asymmetry
  //  (~1e-3 relative) and the run then blows up, even though bndIncons -- a
  //  BOUNDARY metric -- still reads 1.22e-12 and looks fine.
  //  Doing it properly means solving the correction IN THE SYMMETRIC SUBSPACE:
  //  one unknown per mirror pair, so the correction is symmetric by
  //  construction and the GCL is satisfied by the symmetric weights directly.
  //  Until then, --cutz2d projects the spurious z modes out of the residual,
  //  which is exact for a pseudo-2D run and costs nothing.
  { i32 neg=0; for (const SayeNode &s : E.vol) if ((double)s.w<0) neg++;
    E.nNegW=neg; }
  if (E.vol.empty()) return false;
  E.volume=0; for (const SayeNode &s : E.vol) E.volume += (double)s.w;

  // ---- dense mass matrix on the SOLUTION basis ----------------------------
  std::vector<double> M((size_t)nb*nb, 0.0), ps(nb);
  for (const SayeNode &s : E.vol) {
    double X[3]={(double)s.x[0],(double)s.x[1],(double)s.x[2]};
    Bs.eval(X, ps.data());
    double w=(double)s.w;
    for (i32 a=0;a<nb;a++) for (i32 c2=a;c2<nb;c2++) M[(size_t)a*nb+c2]+=w*ps[a]*ps[c2];
  }
  for (i32 a=0;a<nb;a++) for (i32 c2=0;c2<a;c2++) M[(size_t)a*nb+c2]=M[(size_t)c2*nb+a];
  // Cholesky in place
  bool spd = true;
  for (i32 j=0;j<nb && spd;j++){
    double d=M[(size_t)j*nb+j];
    for (i32 q=0;q<j;q++) d-=M[(size_t)j*nb+q]*M[(size_t)j*nb+q];
    if (d<=1e-12*fmax(M[0],1e-300)) { spd=false; break; }
    d=sqrt(d); M[(size_t)j*nb+j]=d;
    for (i32 i=j+1;i<nb;i++){
      double t=M[(size_t)i*nb+j];
      for (i32 q=0;q<j;q++) t-=M[(size_t)i*nb+q]*M[(size_t)j*nb+q];
      M[(size_t)i*nb+j]=t/d;
    } }
  if (!spd) return false;                  // cannot support its own degree: build failure
  E.Mchol.swap(M);
  E.B = Bs;                                // keep the SOLUTION basis actually used
  // LOW-DEGREE MODE COUNT for the modal-decay (Persson) sensor.  The basis is
  // degree-major and the Cholesky NESTS, so the first count(N-1) orthonormal
  // functions span the degree-(N-1) space exactly and
  //     eta = 1 - sum_{m < nbLo} c~_m^2 / sum_m c~_m^2
  // is the top-degree energy fraction -- Persson & Peraire's indicator, exact
  // in this frame with no mass matvec.
  // THIS WAS NEVER ASSIGNED.  nbLo defaulted to 0, its own declaration reads
  // "0 => no sensor", and the consumer guards on `k > 0` -- so the cut-cell
  // modal-decay sensor has been DEAD since the degree ladder was removed (nbLo
  // used to be set as part of that machinery).  Everything keyed on it was
  // therefore silently off: the cut-element artificial viscosity reads that
  // verdict as its theta, so AV on cut cells was identically zero, which is
  // exactly the reported symptom "at every blowup the element sensor read
  // theta = 0.001..0.018 -- IT NEVER FIRED".
  E.nbLo = (N >= 1) ? CutBasis::count(N-1) : 0;
  E.ok = true;
  return true;
  }                                        // end of the (single-degree) block
  return false;
}


// ---------------------------------------------------------------------------
//  Robust entry point: epsilon-escalation + snap.
//
//  A level set crossing a cell CORNER puts the Saye recursion on a degenerate
//  branch: measured on a mirror pair, one side returned volume wrong by 2.6%
//  and wall by 1.3% (bndIncons 7.8e-3) while its mirror was clean (8.5e-6) --
//  and raising maxDepth made it WORSE (point spiral, silent truncation).
//  Shifting phi by a tiny epsilon moves the crossing off the exact corner and
//  fixed the pair bit-for-bit at a geometry cost of epsilon itself.
//
//  Ladder: build raw; if bndIncons > tol, retry with phi +/- eps for growing
//  eps, keeping the best.  If still inconsistent AND the element is nearly
//  uncut, SNAP: drop the sub-resolution feature entirely (status in E.snap)
//  rather than keep rules that half-see it.
// ---------------------------------------------------------------------------
inline bool cutElemBuild(const PolyND &phi, i32 N, CutElemOps &E,
                         SayeArena &ar, const SayeCfg &cfg,
                         std::vector<SayeNode> &scratch,
                         double qualTol = 1e-6,
                         const std::vector<SayeNode> * const *faceOverride = nullptr,
                         const double *ffXi = nullptr, const double *ffW = nullptr,
                         i32 ffN = 0, i32 p2dNz = 0) {
  if (!cutElemBuildRaw(phi, N, E, ar, cfg, scratch, faceOverride, ffXi, ffW, ffN, p2dNz)) {
    // The element cannot carry degree N -- and with the degree ladder gone
    // there is no lower order to fall back to.  That is deliberate, but it is
    // not a reason to lose the cell: a SUB-RESOLUTION feature should be dropped
    // geometrically, which is what the snap rules already do and is the honest
    // answer anyway (a 3e-05 fluid sliver is below the mesh's resolution, so
    // representing it at ANY order is a fiction).  Anything else is a genuine
    // failure: the element needs more QUADRATURE, not less accuracy -- the
    // limit there is the rule's point count against the mode count, so the fix
    // is a denser Saye rule (cfg.ng) on that cell, never a lower degree.
    // NOTE the contract: these return true with E.ok == false and NO operators
    // built.  E.snap != 0 means "this is not a cut element" -- the caller must
    // reclassify the block (IB_FLUID / IB_DEAD) and never touch E.B, E.vol or
    // E.Mchol.  DgCutBuild.cu does exactly that.
    if (E.volume > 0.97 && E.wallArea < 0.2) { E.snap = 1; return true; }
    if (E.volume < 0.03 && E.wallArea < 0.2) { E.snap = 2; return true; }
    return false;
  }
  if (E.bndIncons > qualTol) {
    const double epsL[6] = {1e-9,-1e-9,1e-7,-1e-7,1e-5,-1e-5};
    CutElemOps best = E;
    for (double eps : epsL) {
      PolyND ps = phi; ps.at(0,0,0) = (real)((double)ps.at(0,0,0) + eps);
      CutElemOps T;
      if (!cutElemBuildRaw(ps, N, T, ar, cfg, scratch, faceOverride, ffXi, ffW, ffN, p2dNz)) continue;
      if (T.bndIncons < best.bndIncons) best = T;
      if (best.bndIncons <= qualTol) break;
    }
    E = best;
  }
  // GEOMETRIC snap, unconditional: a nearly-full cell with a pin-prick of wall
  // (the tangency GRAZE: the arc clips a corner) is not a meaningful cut cell.
  // This must NOT be gated on bndIncons -- the shared-face canonicalization
  // makes bndIncons exactly 0 on such cells, which silently un-snapped them and
  // left live P2 cells with needle walls paired against P0 wedges across the
  // tangency face (the surviving supersonic blowup pair).
  // CUT_NOSNAP=1: keep a grazing cell as a genuine cut element.  Snapping it to
  // FLUID deletes its wall, which puts a HOLE in the body at exactly the four
  // points where the surface is tangent to the grid -- and the flow goes
  // through it.  Snapping to SOLID at least keeps the body closed.
  const bool noSnapF = getenv("CUT_NOSNAP") != nullptr;
  if (!noSnapF && E.volume > 0.97 && E.wallArea < 0.05) { E.snap = 1; return true; }
  if (E.volume < 0.03 && E.wallArea < 0.05) { E.snap = 2; return true; }
  if (E.bndIncons > qualTol) {
    // still inconsistent: if the cut is marginal, drop the feature entirely
    if (E.volume > 0.97 && E.wallArea < 0.2) { E.snap = 1; return true; }
    if (E.volume < 0.03 && E.wallArea < 0.2) { E.snap = 2; return true; }
  }
  // A cell with NO wall rule is not a cut cell, whatever its quality number:
  // the graze case (solid pocket below the detection limit) leaves wall == 0
  // with vol slightly under 1, and evolving it as cut lets flow through the
  // physical wall gap -- measured as a CFL-insensitive neighbour blowup at P2.
  if (E.wallArea <= 1e-12 && E.volume > 0.9) { E.snap = 1; return true; }
  if (E.wallArea <= 1e-12 && E.volume < 0.1) { E.snap = 2; return true; }
  return true;
}

#endif
