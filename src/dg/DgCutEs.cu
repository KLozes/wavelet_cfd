// ===========================================================================
//  ENTROPY-STABLE CUT-CELL RHS  --  host build and device kernel.
//
//  Taylor & Chan, "An Entropy Stable High-Order Discontinuous Galerkin Method
//  on Cut Meshes" (arXiv:2412.13002, docs/CutCellEntropyStable.pdf).  The
//  operator itself is built by src/common/CutEsOps.h and gated by
//  src/dg/DgEsCutTest.cu (ES1); this file is the solver path.
//
//  WHY THIS EXISTS.  The baseline cut RHS integrates the volume term over the
//  fitted Saye rule and the surface terms over independently-built face rules.
//  Those two are not paired -- Q_d + Q_d^T != B_d -- so nothing bounds the
//  energy the volume term injects, and the measured consequence is an
//  instability whose growth rate is INVARIANT to the quadrature order (12.98 /
//  13.23 / 13.23 at ng = 6 / 10 / 16 on the case-9 free stream) and present on
//  a uniform state with a transparent wall.  Quadrature refinement shrinks the
//  seed and leaves the amplifier untouched, which is the signature of a
//  structural defect rather than an accuracy one.  The skew-hybridized form
//  below satisfies Q_H,d + Q_H,d^T = B_H,d BY CONSTRUCTION for any quadrature,
//  so the flux-differenced volume term telescopes exactly into the surface
//  term and the semi-discrete entropy rate is the sum of interface
//  dissipations, each <= 0.
//
//  MEASURED, isolated element, same seed / dt / integrator (SSP-RK3, dt 2e-4,
//  5000 steps), baseline vs this operator, ||dc||/||dc||_0 at t = 1:
//      wedge   (6,6) N=2      762   ->    54.7
//      wedge   (6,6) N=3   NON-FINITE at t=0.019  ->  940
//      quarter (7,6) N=2     1184   ->    78.1
//      quarter (7,6) N=3    1.51e9  ->   497
//  and the ES growth is POLYNOMIAL (a neutrally-stable operator responding to
//  a constant truncation forcing -- the N=1 increments are exactly linear in
//  t), not exponential.
//
//  THE ONE DESIGN CONSTRAINT THAT DECIDES EVERYTHING.  The surface rule used
//  to build Q_H must be the rule the solver actually integrates the interface
//  flux over, for two independent reasons:
//    1. the telescoping identity is between the volume term and the surface
//       term; different rules, no telescoping, no entropy statement;
//    2. a face shared with a Cartesian neighbour must be integrated
//       identically by both sides or mass stops being conserved there, and the
//       neighbour's collocated DGSEM lift IS the tensor GLL face rule.
//  So: fully-fluid faces carry the 16-point tensor GLL rule (which also cuts
//  the wedge's 1500-point Saye face rule to 16), partial faces carry the
//  canonicalized Saye rule unpruned so two cut neighbours agree point for
//  point, and the wall carries the Saye surface rule.
// ===========================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "DgSolver.cuh"
#include "DgSolverKernels.cuh"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "CutElem.h"
#include "CutEsOps.h"

// ---------------------------------------------------------------------------
//  HOST BUILD
// ---------------------------------------------------------------------------
void DgSolver::buildCutEs(const void *opsVec) {
  const std::vector<CutElemOps> &ops = *(const std::vector<CutElemOps> *)opsVec;
  const i32 n = nCutElem;
  if (!cutEs || n == 0) return;

  double wN[NNODE], xiN[NNODE];
  dgGetHostOps(wN, xiN, gauss);

  // ---- per element: swap in the runtime interface rules, then build --------
  std::vector<CutEsOps> S(n);
  std::vector<i32> nq(n), nf(n);
  size_t tq = 0, tf = 0, tq2 = 0, te = 0;
  double gclWorst = 0;
  i32 nGll = 0, nSaye = 0;
  std::vector<std::vector<i32>> nodeOf(n);

  for (i32 c = 0; c < n; c++) {
    CutElemOps E = ops[c];                       // copy: we rewrite face rules
    std::vector<i32> isGll(6, 0);
    for (i32 f = 0; f < 6; f++) {
      double a = 0;
      for (const SayeNode &s : E.face[f]) a += (double)s.w;
      if (E.face[f].empty() || fabs(a - 1.0) > 1e-6) { nSaye++; continue; }
      // FULLY FLUID FACE -> the tensor GLL rule, which is what the neighbour
      // lifts over.  Legitimate because both rules integrate psi~_m (total
      // degree <= N) exactly on a full face: measured agreement 1.1e-14 once
      // the Gauss/GLL node tables were computed rather than typed to ten
      // digits (SayeQuad.h gaussLegendre, LagrangeBasis/PolyFit gllNodes).
      const i32 d = f/2, side = f%2, t1 = (d == 0) ? 1 : 0, t2 = (d == 2) ? 1 : 2;
      std::vector<SayeNode> g;
      for (i32 fb = 0; fb < NNODE; fb++)
        for (i32 fa = 0; fa < NNODE; fa++) {
          SayeNode s{};
          s.x[d]  = (real)(side ? 1.0 : 0.0);
          s.x[t1] = (real)(0.5*(xiN[fa] + 1.0));
          s.x[t2] = (real)(0.5*(xiN[fb] + 1.0));
          s.w     = (real)(0.25*wN[fa]*wN[fb]);
          s.n[0] = s.n[1] = s.n[2] = 0;
          s.n[d] = (real)(side ? 1.0 : -1.0);
          g.push_back(s);
        }
      E.face[f] = g; isGll[f] = 1; nGll++;
    }

    // prune = false: full faces are already 16 points, partial faces must stay
    // point-for-point identical to the neighbour's copy of the same rule, and
    // the wall is never pruned by cutEsBuild anyway.
    if (!cutEsBuild(E, S[c], /*prune=*/false)) {
      printf("cutes  : element %d FAILED to build -- falling back to the "
             "baseline cut RHS\n", c);
      cutEs = 0; return;
    }
    nq[c] = S[c].nq; nf[c] = S[c].nf;
    tq += nq[c]; tf += nf[c];
    tq2 += (size_t)nq[c]*nq[c];
    te  += (size_t)nf[c]*nq[c];
    gclWorst = fmax(gclWorst, S[c].gclResid);

    // neighbour tensor-node index for every GLL surface point (else -1)
    nodeOf[c].assign(nf[c], -1);
    for (i32 a = 0; a < nf[c]; a++) {
      const i32 f = S[c].fOwner[a];
      if (f >= 6 || !isGll[f]) continue;
      const i32 d = f/2, side = f%2, t1 = (d == 0) ? 1 : 0, t2 = (d == 2) ? 1 : 2;
      i32 ia = -1, ib = -1;
      for (i32 m = 0; m < NNODE; m++) {
        if (fabs(S[c].xf[3*(size_t)a+t1] - 0.5*(xiN[m]+1.0)) < 1e-10) ia = m;
        if (fabs(S[c].xf[3*(size_t)a+t2] - 0.5*(xiN[m]+1.0)) < 1e-10) ib = m;
      }
      if (ia < 0 || ib < 0) continue;
      i32 ci3[3]; ci3[d] = side ? (NNODE-1) : 0; ci3[t1] = ia; ci3[t2] = ib;
      nodeOf[c][a] = ci3[0] + NNODE*(ci3[1] + NNODE*ci3[2]);
    }
  }

  // ---- flatten -------------------------------------------------------------
  auto allocR = [&](real **p, size_t k) {
    cudaMallocManaged(p, k*sizeof(real)); memset(*p, 0, k*sizeof(real)); };
  auto allocI = [&](i32 **p, size_t k) {
    cudaMallocManaged(p, k*sizeof(i32)); memset(*p, 0, k*sizeof(i32)); };

  allocI(&esQOff, n+1); allocI(&esFOff, n+1); allocI(&esQ2Off, n+1); allocI(&esEOff, n+1);
  for (i32 c = 0; c < n; c++) {
    esQOff[c+1]  = esQOff[c]  + nq[c];
    esFOff[c+1]  = esFOff[c]  + nf[c];
    esQ2Off[c+1] = esQ2Off[c] + nq[c]*nq[c];
    esEOff[c+1]  = esEOff[c]  + nf[c]*nq[c];
  }
  allocR(&esVq,  (size_t)tq*CUT_NBMAX_H);
  allocR(&esDVq, (size_t)3*tq*CUT_NBMAX_H);
  allocR(&esVf,  (size_t)tf*CUT_NBMAX_H);
  allocR(&esWq,  tq);
  allocR(&esWf,  tf);
  allocR(&esNrm, 3*tf);
  allocR(&esXf,  3*tf);
  allocR(&esQ,   3*tq2);
  allocR(&esEmat, te);
  allocR(&esVtil, (size_t)n*CUT_NBMAX_H*5);
  allocI(&esOwner, tf);
  allocI(&esNode,  tf);

  for (i32 c = 0; c < n; c++) {
    const CutEsOps &s = S[c];
    const i32 nb = s.nb, q0 = esQOff[c], f0 = esFOff[c];
    for (i32 i = 0; i < s.nq; i++) {
      esWq[q0+i] = (real)s.wq[i];
      for (i32 m = 0; m < nb; m++) {
        esVq[(size_t)(q0+i)*CUT_NBMAX_H + m] = (real)s.Vq[(size_t)i*nb+m];
        for (i32 d = 0; d < 3; d++)
          esDVq[((size_t)d*tq + (q0+i))*CUT_NBMAX_H + m] = (real)s.dVq[d][(size_t)i*nb+m];
      }
    }
    for (i32 a = 0; a < s.nf; a++) {
      esWf[f0+a] = (real)s.wf[a];
      esOwner[f0+a] = s.fOwner[a];
      esNode[f0+a]  = nodeOf[c][a];
      for (i32 d = 0; d < 3; d++) {
        esNrm[3*(size_t)(f0+a)+d] = (real)s.nrm[3*(size_t)a+d];
        esXf [3*(size_t)(f0+a)+d] = (real)s.xf [3*(size_t)a+d];
      }
      for (i32 m = 0; m < nb; m++)
        esVf[(size_t)(f0+a)*CUT_NBMAX_H + m] = (real)s.Vf[(size_t)a*nb+m];
    }
    for (i32 d = 0; d < 3; d++)
      for (size_t t = 0; t < (size_t)s.nq*s.nq; t++)
        esQ[(size_t)d*tq2 + esQ2Off[c] + t] = (real)s.Q[d][t];
    for (size_t t = 0; t < (size_t)s.nf*s.nq; t++)
      esEmat[esEOff[c] + t] = (real)s.Emat[t];
  }
  esGcl = gclWorst;

  const double mb = ((double)(3*tq2 + te + tq*CUT_NBMAX_H*4 + tf*CUT_NBMAX_H)
                     *sizeof(real))/1048576.0;
  i32 nqMax = 0, nfMax = 0;
  for (i32 c = 0; c < n; c++) { nqMax = max(nqMax, nq[c]); nfMax = max(nfMax, nf[c]); }
  printf("cutes  : entropy-stable RHS ON -- %d elements, nq %d..%d, nf %d..%d "
         "(%d GLL faces, %d Saye)\n", n, nq[0], nqMax, nf[0], nfMax, nGll, nSaye);
  printf("       : operator pools %.1f MB   worst Eq-47 residual %.2e\n", mb, esGcl);
  cudaDeviceSynchronize();
}

// The device kernel lives in DgSolverKernels.cu, where the __constant__ GLL
// tables (c_xi / c_w / c_winv) and the shared state helpers are visible.
