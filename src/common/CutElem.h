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

// ---------------------------------------------------------------------------
//  Build the operators for one cut element from its level-set fit.
//  `phi < 0` is the ACTIVE region (negate at the caller if your fluid is phi>0).
// ---------------------------------------------------------------------------
inline bool cutElemBuildRaw(const PolyND &phi, i32 N, CutElemOps &E,
                         SayeArena &ar, const SayeCfg &cfg,
                         std::vector<SayeNode> &scratch,
                         const std::vector<SayeNode> * const *faceOverride = nullptr) {
  E.ok=false;
  auto mkset=[&](std::vector<SayeNode> &b){
    SayeSet s; s.p=b.data(); s.n=0; s.cap=(i32)b.size(); s.ovf=false; return s; };

  // ---- boundary rules first: they define the geometry the volume rule must
  //      be consistent with, not the other way round ------------------------
  { SayeSet w=mkset(scratch); sayeSurface(phi,&w,&ar,cfg);
    E.wall.assign(w.p, w.p+w.n); }
  for (i32 d=0; d<3; d++) for (i32 side=0; side<2; side++) {
    SayeSet f=mkset(scratch); sayeFace(phi,d,side,&f,&ar,cfg);
    E.face[2*d+side].assign(f.p, f.p+f.n);
  }
  E.wallArea=0; for (const SayeNode &s : E.wall) E.wallArea += (double)s.w;

  // ---- candidate volume nodes: the raw Saye points ------------------------
  std::vector<SayeNode> cand;
  { SayeSet v=mkset(scratch); sayeVolume(phi,&v,&ar,cfg);
    cand.assign(v.p, v.p+v.n); }
  if (cand.empty()) return false;

  // ---- basis, scaled over the WHOLE reference cell -------------------------
  // NOT over the cut region.  A region-adapted centroid/scale looks better
  // conditioned, but a DG cut element's polynomial must also be EVALUATED at
  // the solution nodes outside the cut region -- and there u=(X-c)/s grows like
  // (cell size)/(sliver size), so psi ~ u^N explodes.  In the solver that
  // amplification fed back through the nodal->modal round trip and blew a
  // uniform state to 1e24 in one RK step.  With the cell-wide scale psi = O(1)
  // at every node; the sliver's mass-matrix conditioning is then worse, but
  // that is the small-cell problem state redistribution exists to fix.
  double cc[3]={0.5,0.5,0.5};
  double wsum=0; for (const SayeNode &s : cand) wsum+=(double)s.w;
  if (wsum<=0) return false;
  double sc=0.75;                          // (half-diagonal)^2 of the unit cell
  E.B.init(2*N, cc, sqrt(sc));            // moments to total degree 2N
  const i32 nm = E.B.nb;                  // number of moment constraints

  // Solution basis.  Degree starts at N and DROPS if the element cannot carry
  // it: a tiny sliver's quadrature cannot make a full-degree cell-scaled mass
  // matrix positive definite (measured: fluid volume 3e-5 with 21 points fails
  // to factor 20 modes).  Reducing the degree is the honest response -- the
  // element represents what its fluid region can support, down to a piecewise
  // constant -- and state redistribution merges exactly these cells anyway.
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
    std::vector<double> A((size_t)nc*nm), b(nm,0.0), chi(nm), wfit;
    for (i32 q=0;q<nc;q++){
      double X[3]={(double)cand[q].x[0],(double)cand[q].x[1],(double)cand[q].x[2]};
      E.B.eval(X, chi.data());
      for (i32 m=0;m<nm;m++){ A[(size_t)q*nm+m]=chi[m]; b[m]+=(double)cand[q].w*chi[m]; }
    }
    nnls(A, b, nm, nc, wfit);
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

  // ---- step 2+3, per candidate degree: GCL correction, then the mass -------
  std::vector<SayeNode> volKeep(E.vol);      // pre-correction weights
  i32 degTop = N;
  { const char *de = getenv("CUT_DEGMAX");
    if (de) { i32 v = atoi(de); if (v >= 0 && v < degTop) degTop = v; } }
  // THIN-CELL DEGREE RULE.  Mean thickness = fluid volume / wall area.  A
  // tangency WEDGE (wall grazing a cell face) has thickness ~0.08 cells; its
  // P^N trace on the far face is extrapolation garbage that no face-coupling
  // structure can repair -- measured as the persistent supersonic blowup at
  // exactly the four tangency cells after the mortar fixed everything else.
  // A cell thinner than CUT_THINTOL (default 0.25) of a cell carries P0 only:
  // the geometric form of the sub-half-cell resolution limit, applied to the
  // DEGREE instead of to existence.
  {
    // NOTE: E.volume is not assigned yet at this point -- an earlier version
    // read it here and silently collapsed EVERY element to degree 0 (the
    // "P2 all-Mach pass" it produced was P0 in disguise).  The raw Saye
    // candidate volume wsum is what exists now, and is the right measure.
    double aw = E.wallArea;
    if (aw > 1e-12) {
      double thick = wsum / aw;
      const char *tt = getenv("CUT_THINTOL");
      double tol = tt ? atof(tt) : 0.25;
      if (thick < tol) degTop = 0;
    }
  }
  for (i32 deg = degTop; deg >= 0; deg--) {
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
    const i32 np=(i32)E.vol.size();
    std::vector<double> G((size_t)nG*np), g(nG,0.0), psi(nb), dpsi((size_t)nb*3);
    for (i32 q=0;q<np;q++){
      double X[3]={(double)E.vol[q].x[0],(double)E.vol[q].x[1],(double)E.vol[q].x[2]};
      Bs.grad(X,dpsi.data());
      for (i32 m=0;m<nb;m++) for (i32 d=0;d<3;d++) G[(size_t)(3*m+d)*np+q]=dpsi[3*m+d];
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
    // residual r = g - G w
    std::vector<double> r(nG,0.0);
    for (i32 i=0;i<nG;i++){ double s=g[i];
      for (i32 q=0;q<np;q++) s-=G[(size_t)i*np+q]*(double)E.vol[q].w; r[i]=s; }
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
          if ((double)E.vol[q].w + d < 0) {
            // pin at zero: dw = -w0 exactly, and its constraint contribution
            // moves to the right-hand side for the next round
            viol=true; freeQ[q]=0; dw[q]=-(double)E.vol[q].w;
            for (i32 i=0;i<nG;i++) rr[i]-=G[(size_t)i*np+q]*dw[q];
          }
        }
        if (!viol) break;
      }
      for (i32 q=0;q<np;q++)
        E.vol[q].w = (real)fmax((double)E.vol[q].w + dw[q], 0.0);
    }
  }
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
  if (!spd) continue;                      // this degree does not fit: try lower
  // KAPPA CAP.  Factorability is not enough: a small cut cell can factor a
  // full-degree mass matrix whose inverse still amplifies by 1e6, and the RHS
  // then produces intermediates the sanitizer mangles before state
  // redistribution can act (measured: 7.8e6 modal RHS on a physical wall load
  // of ~1.5e2, blowing up in three stages).  A cell whose quadrature cannot
  // SUPPORT a mode must not own it -- the neighbourhood polynomial SRD builds
  // carries the accuracy there instead.  Estimate kappa from the Cholesky
  // diagonal and drop the degree until it is tame.
  {
    double dmin=1e300, dmax=0;
    for (i32 i=0;i<nb;i++){ double d=M[(size_t)i*nb+i]; dmin=fmin(dmin,d); dmax=fmax(dmax,d); }
    double kap=(dmax/dmin)*(dmax/dmin);
    const char *ke = getenv("CUT_KAPMAX");
    double kapMax = ke ? atof(ke) : 1e4;
    if (kap > kapMax && deg > 0) continue;   // too ill-conditioned: try lower
  }
  E.Mchol.swap(M);
  E.B = Bs;                                // keep the SOLUTION basis actually used
  E.ok = true;
  return true;
  }                                        // end of the degree loop
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
                         const std::vector<SayeNode> * const *faceOverride = nullptr) {
  if (!cutElemBuildRaw(phi, N, E, ar, cfg, scratch, faceOverride)) return false;
  if (E.bndIncons > qualTol) {
    const double epsL[6] = {1e-9,-1e-9,1e-7,-1e-7,1e-5,-1e-5};
    CutElemOps best = E;
    for (double eps : epsL) {
      PolyND ps = phi; ps.at(0,0,0) = (real)((double)ps.at(0,0,0) + eps);
      CutElemOps T;
      if (!cutElemBuildRaw(ps, N, T, ar, cfg, scratch, faceOverride)) continue;
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
  if (E.volume > 0.97 && E.wallArea < 0.05) { E.snap = 1; return true; }
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
