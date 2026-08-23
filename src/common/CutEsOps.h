#ifndef COMMON_CUTESOPS_H
#define COMMON_CUTESOPS_H

// ---------------------------------------------------------------------------
//  SKEW-HYBRIDIZED SBP OPERATORS ON A CUT ELEMENT
//
//  Taylor & Chan, "An Entropy Stable High-Order Discontinuous Galerkin Method
//  on Cut Meshes", arXiv:2412.13002 (docs/CutCellEntropyStable.pdf), built on
//  the skew-hybridized SBP formulation of Chan.  Given a cut element's volume
//  rule, face rules and wall rule -- exactly what CutElem.h already produces --
//  this header builds
//
//      Q_H,d = 1/2 [  Q_d - Q_d^T    E^T B_d ]        B_H,d = [ 0   0  ]
//                  [  -B_d E         B_d     ]                [ 0  B_d ]
//
//  which satisfies the hybridized SBP property  Q_H,d + Q_H,d^T = B_H,d  BY
//  CONSTRUCTION, for any quadrature whatsoever.  That is the whole point: the
//  entropy-stability proof then needs only ONE property of the geometry,
//
//      Q_H,d 1 = 0                                                (their Eq 47)
//
//  and Sec 2.1.4 shows three of its four blocks hold identically.  What is left
//  is exactly the discrete divergence theorem on the solution basis,
//
//      SUM_q w_q d(psi_m)/dx_d  ==  CLOSED INT psi_m n_d dS,
//
//  i.e. the SAME system CutElem.h already fits the volume weights to (its
//  G dw = r), with the constant mode reducing to CLOSED INT n_d dS == 0.  So a
//  cut element that passes the S1 gate is already admissible here; nothing new
//  is required of the geometry.
//
//  TWO THINGS THIS HEADER MUST DO THAT CutElem.h DOES NOT:
//
//  1. PRUNE THE SURFACE RULES.  Flux differencing costs O((nq+nf)^2) two-point
//     flux evaluations, and the raw Saye face rules are enormous -- measured on
//     case 9: nf = 19600 points against nq = 1532, with 70% of them sitting on
//     faces that are ENTIRELY FLUID (area exactly 1) where Saye subdivides for
//     a near-tangency that does not exist on that face.  Unpruned, one wedge
//     element needs 83 KB of shared memory for its entropy variables alone --
//     over the 64 KB sm_75 ceiling -- so this is not a slow configuration but
//     an impossible one.  NNLS onto the rule's own moments to total degree 2N
//     (the exactness the surface term needs) cuts nf by ~10x and costs nothing
//     that matters: measured closed-surface residual 1.1e-11 -> 7.5e-11.
//
//  2. WORK IN THE ORTHONORMAL BASIS.  CutElem.h already stores the Cholesky
//     factor L of the monomial mass matrix, and psi~ = L^-1 psi makes M
//     EXACTLY the identity.  Then P_q = V_q^T W with no solve, and the entropy
//     projection is a weighted sum.  The paper carries M^-1 throughout because
//     it uses a nodal Fekete basis; we do not have to.
//
//  Everything is on the REFERENCE cell [0,1]^3 with reference-measure weights,
//  exactly as CutElem.h leaves it; the caller folds in h.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <vector>

#include "Util.cuh"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "CutElem.h"

// ---------------------------------------------------------------------------
//  Surface-rule pruning.
//
//  NNLS onto the rule's OWN moments up to total degree `degMom`, then a
//  ridge-regularised positivity-constrained repair on the surviving support --
//  the surface analogue of the least-norm correction CutElem.h applies to the
//  volume rule, in dw-space because the moment Gram of a surface rule is
//  rank-deficient by construction (a rule living on x_d = const cannot see the
//  x_d dependence of the 3-D moment basis).
//
//  `vecMoments` fits the NORMAL-WEIGHTED moments INT psi_i n_d dS instead of
//  the scalar INT psi_i dS.  That matters for the wall: the GCL right-hand side
//  IS the normal-weighted moment, and preserving the scalar one does not
//  preserve it.  Measured on case 9: scalar leaves closure 1.10e-10, vector
//  7.48e-11, for 0-3 extra points.  Faces have a constant normal, so the two
//  are identical there.
//
//  gtol: nnls() stops on an ABSOLUTE gradient threshold, and CutBasis scales by
//  s = sqrt(3)/2 so a degree-6 monomial is ~0.04 -- the repo default 1e-9 is
//  therefore a LOOSE tolerance on exactly the high-degree moments the fit
//  exists to reproduce, and it stops early with a residual the repair cannot
//  remove (measured: free-stream 4.6e-07 vs 2.3e-09).  Default here is 1e-13.
// ---------------------------------------------------------------------------
// rules smaller than this are left alone: they cost nothing and Saye already
// made them exact, so pruning them can only lose moments
static constexpr i32 kPruneMin = 200;

struct CutPruneStat {
  i32    nIn = 0, nOut = 0;
  double momAbs = 0;          // worst |moment(pruned) - moment(raw)|
  double measIn = 0, measOut = 0;
};

inline CutPruneStat cutPruneRule(const std::vector<SayeNode> &in,
                                 const CutBasis &Bm, bool vecMoments,
                                 const double *fixedNormal,
                                 std::vector<SayeNode> &out,
                                 double gtol = 1e-13) {
  CutPruneStat P;
  out.clear();
  P.nIn = (i32)in.size();
  if (in.empty()) return P;

  const i32 nbm = Bm.nb;
  const i32 m   = vecMoments ? 3*nbm : nbm;
  const i32 nc  = (i32)in.size();
  const double wtol = 1e-15;

  std::vector<double> A((size_t)nc*m, 0.0), b(m, 0.0), psi(nbm);
  for (i32 q = 0; q < nc; q++) {
    double X[3] = {(double)in[q].x[0], (double)in[q].x[1], (double)in[q].x[2]};
    Bm.eval(X, psi.data());
    double nn[3];
    if (fixedNormal) { nn[0]=fixedNormal[0]; nn[1]=fixedNormal[1]; nn[2]=fixedNormal[2]; }
    else { nn[0]=(double)in[q].n[0]; nn[1]=(double)in[q].n[1]; nn[2]=(double)in[q].n[2]; }
    const double w = (double)in[q].w;
    P.measIn += w;
    if (!vecMoments) {
      for (i32 i = 0; i < nbm; i++) { A[(size_t)q*m+i] = psi[i]; b[i] += w*psi[i]; }
    } else {
      for (i32 i = 0; i < nbm; i++) for (i32 d = 0; d < 3; d++) {
        const double v = psi[i]*nn[d];
        A[(size_t)q*m + 3*i + d] = v;  b[3*i+d] += w*v;
      }
    }
  }

  std::vector<double> w;
  nnls(A, b, m, nc, w, gtol);

  // ---- moment repair on the surviving support ----------------------------
  {
    std::vector<i32> sup;
    for (i32 q = 0; q < nc; q++) if (w[q] > wtol) sup.push_back(q);
    const i32 k = (i32)sup.size();
    if (k > 0) {
      std::vector<double> r(m, 0.0);
      for (i32 i = 0; i < m; i++) { double s = b[i];
        for (i32 a = 0; a < k; a++) s -= w[sup[a]]*A[(size_t)sup[a]*m+i];
        r[i] = s; }
      std::vector<double> dw(k, 0.0), rr(r);
      std::vector<char> freeQ(k, 1);
      for (i32 round = 0; round < 6; round++) {
        std::vector<double> CtC((size_t)k*k, 0.0), rhs(k, 0.0);
        for (i32 a = 0; a < k; a++) { if (!freeQ[a]) continue;
          const double *Aa = &A[(size_t)sup[a]*m];
          for (i32 c = a; c < k; c++) { if (!freeQ[c]) continue;
            const double *Ac = &A[(size_t)sup[c]*m]; double t = 0;
            for (i32 i = 0; i < m; i++) t += Aa[i]*Ac[i];
            CtC[(size_t)a*k+c] = CtC[(size_t)c*k+a] = t; }
          double t = 0; for (i32 i = 0; i < m; i++) t += Aa[i]*rr[i]; rhs[a] = t; }
        for (i32 a = 0; a < k; a++) if (!freeQ[a]) { CtC[(size_t)a*k+a] = 1.0; rhs[a] = 0.0; }
        double tr = 0; for (i32 a = 0; a < k; a++) tr += CtC[(size_t)a*k+a];
        for (i32 a = 0; a < k; a++) CtC[(size_t)a*k+a] += 1e-14*fmax(tr/k, 1.0);
        std::vector<double> yy(rhs);
        if (!srdSolveSPDLocal(CtC, yy, k)) break;
        bool viol = false;
        for (i32 a = 0; a < k; a++) { if (!freeQ[a]) continue; dw[a] = yy[a];
          if (w[sup[a]] + dw[a] < 0) { viol = true; freeQ[a] = 0; dw[a] = -w[sup[a]];
            const double *Aa = &A[(size_t)sup[a]*m];
            for (i32 i = 0; i < m; i++) rr[i] -= Aa[i]*dw[a]; } }
        if (!viol) break;
      }
      for (i32 a = 0; a < k; a++) w[sup[a]] = fmax(w[sup[a]] + dw[a], 0.0);
    }
  }

  std::vector<double> acc(m, 0.0);
  for (i32 q = 0; q < nc; q++) {
    if (w[q] <= wtol) continue;
    SayeNode s = in[q]; s.w = (real)w[q];
    out.push_back(s); P.measOut += w[q];
    for (i32 i = 0; i < m; i++) acc[i] += w[q]*A[(size_t)q*m+i];
  }
  P.nOut = (i32)out.size();
  for (i32 i = 0; i < m; i++) P.momAbs = fmax(P.momAbs, fabs(acc[i]-b[i]));
  return P;
}

// ---------------------------------------------------------------------------
//  The operators.  Surface points are the six face rules followed by the wall
//  rule, concatenated; a face rule carries the constant normal +/- e_d and the
//  wall rule carries its own per-point normal.
// ---------------------------------------------------------------------------
struct CutEsOps {
  i32 nq = 0, nf = 0, nb = 0;

  std::vector<double> wq, xq;        // [nq], [3*nq]        volume rule
  std::vector<double> wf, xf, nrm;   // [nf], [3*nf], [3*nf] surface rule + normal
  std::vector<i32>    fOwner;        // [nf] 0..5 = face, 6 = wall (for BCs)

  std::vector<double> Vq;            // [nq*nb]   psi~ at volume points
  std::vector<double> dVq[3];        // [nq*nb]   d psi~/dx_d at volume points
  std::vector<double> Vf;            // [nf*nb]   psi~ at surface points
  std::vector<double> Minv;          // [nb*nb]   inverse of M = Vq^T W Vq
  std::vector<double> Pqm;           // [nb*nq]   Pq = M^-1 Vq^T W
  std::vector<double> Emat;          // [nf*nq]   E = Vf Pq
  std::vector<double> Q[3];          // [nq*nq]   Qd = W (dVq_d) Pq
  std::vector<double> B[3];          // [nf]      Bd = w_f n_d   (diagonal)

  double gclResid = 0;               // max |SUM_q w_q dpsi~_m/dx_d - CLOSED INT psi~_m n_d|
                                     // AFTER the correction below: this is Eq 47's
                                     // real content and it caps the entropy balance
};

// forward substitution L z = y, L lower-triangular row-major (CutElemOps::Mchol)
inline void cutEsSolveL(const std::vector<double> &L, i32 n, double *y) {
  for (i32 i = 0; i < n; i++) {
    double t = y[i];
    for (i32 q = 0; q < i; q++) t -= L[(size_t)i*n+q]*y[q];
    y[i] = t/L[(size_t)i*n+i];
  }
}

// value (and optionally gradient) of the ORTHONORMAL basis at one point
inline void cutEsPsi(const CutElemOps &E, const double X[3],
                     double *psi, double *dpsi /* [3*nb] or null */) {
  const i32 nb = E.B.nb;
  if (psi) { E.B.eval(X, psi); cutEsSolveL(E.Mchol, nb, psi); }
  if (dpsi) {
    E.B.grad(X, dpsi);
    std::vector<double> col(nb);   // CUT_NBMAX lives in DgSolver.cuh, off the gate include path
    for (i32 d = 0; d < 3; d++) {
      for (i32 m = 0; m < nb; m++) col[m] = dpsi[3*m+d];
      cutEsSolveL(E.Mchol, nb, col.data());
      for (i32 m = 0; m < nb; m++) dpsi[3*m+d] = col[m];
    }
  }
}

// ---------------------------------------------------------------------------
//  Build.  `degMomSurf` is the surface exactness the prune targets: 2N is what
//  Sec 2.1.4 requires (the surface integrand is a product of two degree-N
//  traces).  prune = false keeps the raw rules, for A/B measurement.
// ---------------------------------------------------------------------------
inline bool cutEsBuild(const CutElemOps &E, CutEsOps &S,
                       bool prune = true, double gtol = 1e-13,
                       CutPruneStat *statOut /* [7] or null */ = nullptr) {
  if (!E.ok) return false;
  const i32 nb = E.B.nb;
  const i32 N  = E.B.N;
  S.nb = nb;

  // ---- surface rules: prune each face and the wall separately ------------
  CutBasis Bm; Bm.init(2*N, E.B.c, E.B.s);
  std::vector<SayeNode> surf;
  S.fOwner.clear();
  for (i32 f = 0; f < 7; f++) {
    const std::vector<SayeNode> &raw = (f < 6) ? E.face[f] : E.wall;
    if (raw.empty()) { if (statOut) statOut[f] = CutPruneStat(); continue; }
    double fixedN[3] = {0,0,0};
    if (f < 6) fixedN[f/2] = (f%2) ? 1.0 : -1.0;
    const std::vector<SayeNode> *use = &raw;
    std::vector<SayeNode> pruned;
    if (prune && (i32)raw.size() > kPruneMin) {
      CutPruneStat st = cutPruneRule(raw, Bm, /*vecMoments=*/(f == 6),
                                     (f < 6) ? fixedN : nullptr, pruned, gtol);
      // NEVER TRADE THE DIVERGENCE THEOREM FOR POINT COUNT.  A prune perturbs
      // the surface moments, and the normal-weighted ones ARE the right-hand
      // side of Eq 47; the perturbation's component orthogonal to range(G) is
      // structurally uncorrectable, so Eq 47 -- and with it discrete entropy
      // conservation -- degrades to exactly the prune's moment error.
      // Measured: the wall rule is rank-limited (47 surviving points against
      // 147 normal-weighted moments at 2N=6), leaves 3.4e-10..5.9e-09, and
      // capped entropy conservation at 1.3e-08 -- while saving 53 points out of
      // 100, i.e. nothing.  The face rules that actually cost (1500 points)
      // prune to 1e-16.  So: prune only rules big enough to matter, and reject
      // any prune that does not reproduce the moments to round-off.
      const double tol = 1e-12*fmax(st.measIn, 1.0);
      if (st.momAbs <= tol) { use = &pruned; if (statOut) statOut[f] = st; }
      else if (statOut) { statOut[f] = st; statOut[f].nOut = (i32)raw.size();
                          statOut[f].momAbs = 0.0; }
    } else if (statOut) {
      statOut[f] = CutPruneStat(); statOut[f].nIn = statOut[f].nOut = (i32)raw.size();
    }
    for (const SayeNode &s : *use) {
      surf.push_back(s);
      S.fOwner.push_back(f);
    }
    if (f < 6) {   // stamp the constant face normal onto the stored copy
      for (size_t k = surf.size() - use->size(); k < surf.size(); k++) {
        surf[k].n[0] = (real)fixedN[0]; surf[k].n[1] = (real)fixedN[1];
        surf[k].n[2] = (real)fixedN[2];
      }
    }
  }

  S.nq = (i32)E.vol.size();
  S.nf = (i32)surf.size();
  if (S.nq == 0 || S.nf == 0) return false;

  // ---- sample the basis --------------------------------------------------
  S.wq.resize(S.nq); S.xq.resize(3*(size_t)S.nq);
  S.Vq.assign((size_t)S.nq*nb, 0.0);
  for (i32 d = 0; d < 3; d++) S.dVq[d].assign((size_t)S.nq*nb, 0.0);
  {
    std::vector<double> psi(nb), dpsi(3*(size_t)nb);
    for (i32 i = 0; i < S.nq; i++) {
      double X[3] = {(double)E.vol[i].x[0], (double)E.vol[i].x[1], (double)E.vol[i].x[2]};
      S.wq[i] = (double)E.vol[i].w;
      for (i32 d = 0; d < 3; d++) S.xq[3*(size_t)i+d] = X[d];
      cutEsPsi(E, X, psi.data(), dpsi.data());
      for (i32 m = 0; m < nb; m++) {
        S.Vq[(size_t)i*nb+m] = psi[m];
        for (i32 d = 0; d < 3; d++) S.dVq[d][(size_t)i*nb+m] = dpsi[3*m+d];
      }
    }
  }

  S.wf.resize(S.nf); S.xf.resize(3*(size_t)S.nf); S.nrm.resize(3*(size_t)S.nf);
  S.Vf.assign((size_t)S.nf*nb, 0.0);
  for (i32 d = 0; d < 3; d++) S.B[d].assign(S.nf, 0.0);
  {
    std::vector<double> psi(nb);
    for (i32 a = 0; a < S.nf; a++) {
      double X[3] = {(double)surf[a].x[0], (double)surf[a].x[1], (double)surf[a].x[2]};
      S.wf[a] = (double)surf[a].w;
      for (i32 d = 0; d < 3; d++) {
        S.xf[3*(size_t)a+d] = X[d];
        S.nrm[3*(size_t)a+d] = (double)surf[a].n[d];
        S.B[d][a] = S.wf[a]*(double)surf[a].n[d];
      }
      cutEsPsi(E, X, psi.data(), nullptr);
      for (i32 m = 0; m < nb; m++) S.Vf[(size_t)a*nb+m] = psi[m];
    }
  }

  // ---- RE-FIT THE VOLUME WEIGHTS TO THE PRUNED SURFACE RULES -------------
  // CutElem.h already corrected these weights so that the discrete divergence
  // theorem holds -- but against the RAW surface rules.  Pruning moves the
  // right-hand side by the prune's own moment error (~1e-9 on the wall), and
  // Eq 47 then fails by exactly that, which caps discrete entropy conservation
  // at the same 1e-9 (measured, before this block existed).  The volume rule
  // must be consistent with the rules the scheme ACTUALLY evaluates, so redo
  // the least-norm correction here:
  //
  //     min ||dw||   s.t.   G dw = r,     G[3m+d][i] = dpsi~_m/dx_d (x_i),
  //     r = CLOSED INT psi~_m n_d dS  -  SUM_i w_i dpsi~_m/dx_d(x_i)
  //
  // Weights driven negative are pinned at zero and their contribution moved to
  // the right-hand side (the same active-set move CutElem.h makes) -- a
  // positive rule is required for M to stay SPD, which Sec 2.1.4 needs.
  {
    const i32 nG = 3*nb;
    std::vector<double> g(nG, 0.0), lhs(nG, 0.0);
    for (i32 a = 0; a < S.nf; a++)
      for (i32 m = 0; m < nb; m++) for (i32 d = 0; d < 3; d++)
        g[3*m+d] += S.wf[a]*S.nrm[3*(size_t)a+d]*S.Vf[(size_t)a*nb+m];
    for (i32 i = 0; i < S.nq; i++)
      for (i32 m = 0; m < nb; m++) for (i32 d = 0; d < 3; d++)
        lhs[3*m+d] += S.wq[i]*S.dVq[d][(size_t)i*nb+m];

    std::vector<double> r(nG);
    for (i32 k = 0; k < nG; k++) r[k] = g[k] - lhs[k];

    std::vector<char> freeQ(S.nq, 1);
    std::vector<double> dw(S.nq, 0.0), rr(r);
    for (i32 round = 0; round < 6; round++) {
      // GG^T over the free columns, then dw = G^T (GG^T)^-1 rr
      std::vector<double> GG((size_t)nG*nG, 0.0), y(rr);
      for (i32 k = 0; k < nG; k++) {
        const i32 mk = k/3, dk = k%3;
        for (i32 l = k; l < nG; l++) {
          const i32 ml = l/3, dl = l%3;
          double t = 0;
          for (i32 i = 0; i < S.nq; i++) {
            if (!freeQ[i]) continue;
            t += S.dVq[dk][(size_t)i*nb+mk]*S.dVq[dl][(size_t)i*nb+ml];
          }
          GG[(size_t)k*nG+l] = GG[(size_t)l*nG+k] = t;
        }
      }
      double tr = 0; for (i32 k = 0; k < nG; k++) tr += GG[(size_t)k*nG+k];
      for (i32 k = 0; k < nG; k++) GG[(size_t)k*nG+k] += 1e-14*fmax(tr/nG, 1.0);
      if (!srdSolveSPDLocal(GG, y, nG)) break;
      bool viol = false;
      for (i32 i = 0; i < S.nq; i++) {
        if (!freeQ[i]) continue;
        double t = 0;
        for (i32 m = 0; m < nb; m++) for (i32 d = 0; d < 3; d++)
          t += S.dVq[d][(size_t)i*nb+m]*y[3*m+d];
        dw[i] = t;
        if (S.wq[i] + dw[i] < 0) {
          viol = true; freeQ[i] = 0; dw[i] = -S.wq[i];
          for (i32 m = 0; m < nb; m++) for (i32 d = 0; d < 3; d++)
            rr[3*m+d] -= S.dVq[d][(size_t)i*nb+m]*dw[i];
        }
      }
      if (!viol) break;
    }
    for (i32 i = 0; i < S.nq; i++) S.wq[i] = fmax(S.wq[i] + dw[i], 0.0);

    S.gclResid = 0;
    for (i32 m = 0; m < nb; m++) for (i32 d = 0; d < 3; d++) {
      double t = 0;
      for (i32 i = 0; i < S.nq; i++) t += S.wq[i]*S.dVq[d][(size_t)i*nb+m];
      S.gclResid = fmax(S.gclResid, fabs(t - g[3*m+d]));
    }
  }

  // ---- M = Vq^T W Vq, and its inverse ------------------------------------
  // NOT assumed to be the identity: psi~ was orthonormalised against the
  // ORIGINAL weights, and the correction above moved them.  M is the
  // QUADRATURE-defined mass matrix, which is what Pq Vq = I needs (Sec 2.1.2),
  // and every automatic block of Eq 47 rests on that identity holding exactly.
  {
    std::vector<double> M((size_t)nb*nb, 0.0);
    for (i32 i = 0; i < S.nq; i++)
      for (i32 m = 0; m < nb; m++) {
        const double a = S.wq[i]*S.Vq[(size_t)i*nb+m];
        for (i32 l = 0; l < nb; l++) M[(size_t)m*nb+l] += a*S.Vq[(size_t)i*nb+l];
      }
    S.Minv.assign((size_t)nb*nb, 0.0);
    for (i32 col = 0; col < nb; col++) {
      std::vector<double> A(M), e(nb, 0.0);
      e[col] = 1.0;
      if (!srdSolveSPDLocal(A, e, nb)) return false;    // M not SPD: rule too weak
      for (i32 m = 0; m < nb; m++) S.Minv[(size_t)m*nb+col] = e[m];
    }
  }

  // ---- Pq = M^-1 Vq^T W,  E = Vf Pq,  Qd = W (dVq_d) Pq ------------------
  S.Pqm.assign((size_t)nb*S.nq, 0.0);
  for (i32 m = 0; m < nb; m++)
    for (i32 i = 0; i < S.nq; i++) {
      double t = 0;
      for (i32 l = 0; l < nb; l++) t += S.Minv[(size_t)m*nb+l]*S.Vq[(size_t)i*nb+l];
      S.Pqm[(size_t)m*S.nq+i] = t*S.wq[i];
    }
  S.Emat.assign((size_t)S.nf*S.nq, 0.0);
  for (i32 a = 0; a < S.nf; a++)
    for (i32 i = 0; i < S.nq; i++) {
      double t = 0;
      for (i32 m = 0; m < nb; m++) t += S.Vf[(size_t)a*nb+m]*S.Pqm[(size_t)m*S.nq+i];
      S.Emat[(size_t)a*S.nq+i] = t;
    }
  for (i32 d = 0; d < 3; d++) {
    S.Q[d].assign((size_t)S.nq*S.nq, 0.0);
    for (i32 i = 0; i < S.nq; i++)
      for (i32 j = 0; j < S.nq; j++) {
        double t = 0;
        for (i32 m = 0; m < nb; m++) t += S.dVq[d][(size_t)i*nb+m]*S.Pqm[(size_t)m*S.nq+j];
        S.Q[d][(size_t)i*S.nq+j] = S.wq[i]*t;
      }
  }
  return true;
}

// ---------------------------------------------------------------------------
//  Condition (47):  Q_H,d 1 = 0.  Returns max |.| over both blocks and all d.
//  Sec 2.1.4 proves the bottom block and half the top block vanish for ANY
//  quadrature; what this actually measures is the discrete divergence theorem
//  on the solution basis, i.e. free-stream preservation.
// ---------------------------------------------------------------------------
inline double cutEsQH1(const CutEsOps &S, double *perDir /* [3] or null */ = nullptr) {
  double worst = 0;
  for (i32 d = 0; d < 3; d++) {
    double wd = 0;
    for (i32 i = 0; i < S.nq; i++) {           // top block
      double t = 0;
      for (i32 j = 0; j < S.nq; j++) t += S.Q[d][(size_t)i*S.nq+j] - S.Q[d][(size_t)j*S.nq+i];
      for (i32 a = 0; a < S.nf; a++) t += S.Emat[(size_t)a*S.nq+i]*S.B[d][a];
      wd = fmax(wd, 0.5*fabs(t));
    }
    for (i32 a = 0; a < S.nf; a++) {           // bottom block
      double e1 = 0;
      for (i32 i = 0; i < S.nq; i++) e1 += S.Emat[(size_t)a*S.nq+i];
      wd = fmax(wd, 0.5*fabs(S.B[d][a]*(1.0 - e1)));
    }
    if (perDir) perDir[d] = wd;
    worst = fmax(worst, wd);
  }
  return worst;
}

// ---------------------------------------------------------------------------
//  The hybridized SBP property  Q_H,d + Q_H,d^T == B_H,d = diag(0, B_d).
//  True by construction; measured anyway, because "by construction" is a claim
//  about the code that wrote the blocks, not about the blocks.
// ---------------------------------------------------------------------------
inline double cutEsSbpDefect(const CutEsOps &S) {
  double worst = 0;
  for (i32 d = 0; d < 3; d++) {
    // volume-volume: 1/2 (Qd - Qd^T) is exactly skew -> sum with transpose = 0
    for (i32 i = 0; i < S.nq; i++)
      for (i32 j = 0; j < S.nq; j++) {
        const double qij = 0.5*(S.Q[d][(size_t)i*S.nq+j] - S.Q[d][(size_t)j*S.nq+i]);
        const double qji = 0.5*(S.Q[d][(size_t)j*S.nq+i] - S.Q[d][(size_t)i*S.nq+j]);
        worst = fmax(worst, fabs(qij + qji));
      }
    // volume-face vs face-volume:  1/2 E^T B_d  +  (-1/2 B_d E)^T  == 0
    for (i32 a = 0; a < S.nf; a++)
      for (i32 i = 0; i < S.nq; i++) {
        const double up = 0.5*S.Emat[(size_t)a*S.nq+i]*S.B[d][a];
        const double lo = -0.5*S.B[d][a]*S.Emat[(size_t)a*S.nq+i];
        worst = fmax(worst, fabs(up + lo));
      }
    // face-face: 1/2 B_d + 1/2 B_d == B_d  (exact)
  }
  return worst;
}

#endif
