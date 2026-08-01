#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "DgSolver.cuh"
#include "DgSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

// host-side element widths per direction (device code uses dgElemSize)
static void hostElemSize(const DgSolver &g, i32 lvl, double h[3]) {
  h[0] = (double)g.domainSize[0] / ((double)(g.baseGridSize[0]/blockSize) * powi(2, lvl));
  h[1] = (double)g.domainSize[1] / ((double)(g.baseGridSize[1]/blockSize) * powi(2, lvl));
  h[2] = g.pseudo2D ? (double)g.domainSize[2]
                    : (double)g.domainSize[2] / ((double)(g.baseGridSize[2]/blockSize) * powi(2, lvl));
}
 
void DgSolver::initialize(void) {
  periodic = (bcType == 2);
  dgUploadOperators(gauss, frType);
  buildInitialGrid(true);
}

void DgSolver::buildInitialGrid(bool doPaint) {
  initializeBaseGrid();      // leafMode: interior base elements only, no exterior ring
  if (ibOn) {   // classes must exist before the first vote/fill reads them
    dgIbClassifyGeomKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgIbPromoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }
  setInitialConditions();
  cudaDeviceSynchronize();
  printf("nblocks %d\n", hashTable.nKeys);
  if (doPaint) paint();

  // refine toward the IC's features one level per pass, re-evaluating the
  // analytic IC on every new leaf (sharper than prolongated data).
  // The bootstrap ALWAYS uses the MRA detail regardless of --indicator: a
  // freshly sampled analytic IC has exactly ZERO face jumps (adjacent elements
  // share their LGL face nodes), so jump/entropy indicators are structurally
  // blind at t=0 and would start stepping on the base grid -- the coarse-grid
  // transient then deforms the solution permanently (measured: vortex error
  // pinned at 2.9e-4 regardless of the final grid).  The runtime indicator
  // takes over from the first stepping adaptation.
  i32 runIndicator = indicator;
  indicator = 0;
  for (i32 lvl = 1; lvl < nLvls; lvl++) {
    adaptLeaves();
    setInitialConditions();
    cudaDeviceSynchronize();
    printf("nblocks %d\n", hashTable.nKeys);
    if (doPaint) paint();
  }
  indicator = runIndicator;
  if (ibOn) {   // first ghost fill from the freshly sampled IC (SCRATCH is
    // zeroed at alloc, so the shocked-donor gate stays open -> full order)
    dgIbFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (ibSbm) dgIbSolidFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (mood) dgIbGhostClampKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    if (dbgChecks >= 2) {   // fill exactness probe: uniform IC must refill
      // every ghost node to the IC state exactly (all BC data vanish)
      double W[5] = {1.0, (double)machInf, 0.0, 0.0, 1.0/dgGam}, U0[5];
      U0[0]=W[0]; U0[1]=W[0]*W[1]; U0[2]=W[0]*W[2]; U0[3]=W[0]*W[3];
      U0[4]=W[4]/(dgGam-1.0)+0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
      double dev = 0; i32 devB = -1, devNd = -1, devQ = -1;
      for (i32 b = 0; b < hashTable.nKeys; b++) {
        if (bLocList[b] == kEmpty || ibClassList[b] != IB_GHOST) continue;
        for (i32 nd = 0; nd < blockSizeTot; nd++)
          for (i32 q = 0; q < 5; q++) {
            double d = fabs((double)getField(D_RHO+q)[b*blockSizeTot+nd] - U0[q]);
            if (d > dev) { dev = d; devB = b; devNd = nd; devQ = q; }
          }
      }
      if (devB >= 0) {
        i32 lvl, ib, jb, kb; decode(bLocList[devB], lvl, ib, jb, kb);
        printf("[ibfillchk] max ghost fill deviation from uniform IC = %.3e "
               "(q=%d lvl=%d elem(%d,%d) node %d)\n", dev, devQ, lvl, ib, jb, devNd);
      }
    }
  }
  // Cut-cell operators are built ONCE, here, after the grid has finished its
  // bootstrap climb: the wall band is then at its final (finest) level and is
  // treated as STATIC.  NNLS moment fitting per element is far too costly to
  // repeat per adaptation.
  if (cutOn) {
    buildCutElems();
    buildSrd();
    // The FRIB machinery ran during the build above (ibOn was still 1) and
    // FILLED ghost/dead blocks with wall-mirrored states -- including the
    // solid-side nodes of what are now cut elements.  The cut path reads those
    // nodes through the nodal->modal projection, so the fill is CONTAMINATION,
    // not helpful data (probe measured |RHS| ~ 1e7 on a uniform IC from
    // exactly this).  Reset every block to the analytic IC; ibOn is 0 now, so
    // nothing re-fills.
    setInitialConditions();
    cudaDeviceSynchronize();
  }
}

void DgSolver::setInitialConditions(void) {
  dgSetICKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void DgSolver::sortFieldData(void) {
  dgSnapshotQ0Kernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  dgSortFieldDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void DgSolver::computeImageData(i32 f) {
  dgComputeImageDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f);
}

//
// the leaf-only adaptation cascade: vote -> neighbor rule -> 2:1 grading
// fixpoint -> merge resolution -> spawn -> prolong/restrict -> prune -> sort
//
// microsecond wall clock for the perf accounting
static long nowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

void DgSolver::adaptLeaves(void) {
  if (nLvls <= 1) return;
  nAdapts++;
  long t0 = nowUs();

  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));       // all DELETE (= coarsenable)
  cudaMemset(snapValidList, 0, nBlocksMax*sizeof(i32));

  if (staticGrid) {
    dgStaticVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  } else {
    // detail normalization scales c_i (paper eq. 59), computed and consumed
    // entirely on device: host contact with the managed scale arrays costs a
    // page migration each way per adaptation (~2.7 ms -- it was 30% of runtime)
    cudaMemset(globalScale, 0, 6*sizeof(real));
    dgScalesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgFinalizeScalesKernel<<<1, 32>>>(*this);

    switch (indicator) {
      case 1:   // smoothness-sensor vote (fresh theta on the current grid)
        dgRestrictToAnchorKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        dgDetailNormKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        dgVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this, epsFinest(), 0);  // coarsen guard
        dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);
        dgSensorVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        break;
      default:  // 0: wavelet-free MRA detail (the paper's indicator)
        dgRestrictToAnchorKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        dgDetailNormKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        dgVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this, epsFinest(), 1);
        // refine-before-stabilize (subFv): the shock/smoothness sensor that
        // drives the FV blend ALSO votes REFINE, amplitude-floored, so a
        // troubled coarse cell is refined toward the finest level rather than
        // stabilized in place -- the FV blend then only fires once it is
        // finest (dgRhsKernel finest gate).  Layered on top of the validated
        // MRA detail vote so smooth accuracy is unchanged.
        if (subFv && !mood) {   // MOOD: no a priori sensor -- MRA detail (above)
          dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);   // drives refinement
          dgSensorVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        }
    }
    // pull shocks to the finest level so they never cross a coarse/fine face
    if (shockRefine)
      dgShockRefineKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this, shockThresh);
    // pin the immersed-boundary band (ghosts + donors) to the finest level
    if (ibOn)
      dgIbBandVoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgNeighborRuleKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    // one-ring fine-level buffer around every element voted REFINE (snapshot
    // first so it stays a single ring; grading then closes 2:1 around it)
    if (refineBuffer) {
      dgSnapshotRefineKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      dgRefineBufferKernel<<<cudaGridSize, cudaBlockSize>>>(*this, epsFinest());
    }
  }
  cudaDeviceSynchronize();
  long t1 = nowUs(); tVoteUs += t1 - t0;

  // 2:1 grading of the TARGET configuration and octet-merge resolution must
  // reach a JOINT fixpoint: a merge-apply promotion (DELETE -> KEEP) raises
  // that leaf's target and can re-violate grading against a neighboring
  // octet's pending merge (leaving an illegal 2:1 hole after execution).
  // All raises are monotone and can only ripple ONE level per pass, so nLvls
  // grading passes per round and nLvls rounds are hard fixpoint bounds -- run
  // them as a FIXED stream of async kernel launches (no host loop, no managed-
  // counter readback, no syncs), like the FV solver's cascade.  The --debug
  // face-topology audit verifies the bound held.
  for (i32 round = 0; round < nLvls; round++) {
    for (i32 pass = 0; pass < nLvls; pass++) {
      dgEnforceGradingKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      nGradePasses++;
    }
    dgMergeVerdictKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgMergeApplyKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    nMergeRounds++;
  }
  long t2 = nowUs(); tGradeUs += t2 - t1;

  if (dbgChecks >= 2 && ibOn) {   // IB flag trace (debug 2): probe octet
    cudaDeviceSynchronize();
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      if (bLocList[b] == kEmpty) continue;
      i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
      if (lvl == 1 && ib >= 8 && ib <= 9 && jb >= 12 && jb <= 13)
        printf("[ibtrace] pre-spawn lvl1 (%d,%d): flag=%d class=%d\n",
               ib, jb, bFlagsList[b], ibClassList[b]);
    }
  }

  cudaMemset(createdCnt, 0, sizeof(i32));
  dgSpawnKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  // structure unchanged (no block created; deletes only ever accompany creates:
  // refine deletes the parent it split, merge deletes the octet it replaced)
  // => the fills, prune, sort and positivity pass are all no-ops -- skip them
  if (*createdCnt == 0) {
    nSortsSkipped++;
    long t3 = nowUs(); tSpawnUs += t3 - t2;
    return;
  }

  dgProlongChildrenKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  dgDemoteRefinedKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  dgRestrictParentsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  nBlocks = hashTable.nKeys;
  deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  long t3 = nowUs(); tSpawnUs += t3 - t2;
  sortBlocks();

  // IB: block slots moved -- classes are pure geometry, recompute them BEFORE
  // anything reads them (positivity/AV below), then refill the ghosts from
  // the post-adapt fluid state (fresh sensor first: SCRATCH is stale)
  if (ibOn) {
    dgIbClassifyGeomKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgIbPromoteKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }

  // the exact-L2 merge of under-resolved data can overshoot to inadmissible
  // nodal states (negative rho/p); Zhang-Shu restores admissibility while
  // preserving the (exactly conserved) GLL cell means
  dgPositivityKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  if (ibOn) {
    dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);
    dgIbFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (ibSbm) dgIbSolidFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (mood) dgIbGhostClampKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  }
  cudaDeviceSynchronize();
  tSortUs += nowUs() - t3;

  if (dbgChecks) {
    cudaMemset(dbgCnt, 0, sizeof(i32));
    dgCheckLeafCoverKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    dgCheckFaceTopologyKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    // the mixed-octet invariant only holds once the band has finished its
    // one-level-per-adapt climb (bootstrap = nLvls-1 adapts); before that the
    // march itself creates transient GHOST+FLUID octets below the finest level
    if (ibOn && nAdapts >= nLvls) dgIbCheckKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    // leaves must tile the domain: exactly-once cover <=> no live ancestors
    // (kernel) AND total leaf volume == domain volume (here)
    double vol = 0, volDom = (double)domainSize[0]*domainSize[1]*domainSize[2];
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty) continue;
      i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
      double h[3]; hostElemSize(*this, lvl, h);
      vol += h[0]*h[1]*h[2];
    }
    if (*dbgCnt > 0 || fabs(vol - volDom) > 1e-6*volDom)
      printf("[leafcover] iter %d: ancestorViolations=%d  leafVol/domVol=%.12f\n",
             iter, *dbgCnt, vol/volDom);
  }
}

void DgSolver::computeDeltaT(void) {
  long t0 = nowUs();
  dgLamKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  real *lam = getField(D_LAM);
  real *minPtr = thrust::min_element(thrust::device, lam,
                                     lam + (u64)hashTable.nKeys*blockSizeTot);
  real mn;
  cudaMemcpy(&mn, minPtr, sizeof(real), cudaMemcpyDeviceToHost);
  deltaT = cfl*mn;
  tDtUs += nowUs() - t0;
}

void DgSolver::printPerf(void) {
  long tot = tVoteUs + tGradeUs + tSpawnUs + tSortUs + tDtUs + tRkUs;
  if (tot <= 0) return;
  printf("[perf] total accounted %.1f ms over %d iters (%d adapts, %d sorts skipped)\n",
         tot/1000.0, iter, nAdapts, nAdapts - nSortsSkipped);
  printf("[perf]   vote/indicator %6.1f ms (%4.1f%%)\n", tVoteUs/1000.0, 100.0*tVoteUs/tot);
  printf("[perf]   grade+merge    %6.1f ms (%4.1f%%)  [%0.1f grade passes, %0.1f merge rounds per adapt]\n",
         tGradeUs/1000.0, 100.0*tGradeUs/tot,
         nAdapts ? (double)nGradePasses/nAdapts : 0.0,
         nAdapts ? (double)nMergeRounds/nAdapts : 0.0);
  printf("[perf]   spawn/fill     %6.1f ms (%4.1f%%)\n", tSpawnUs/1000.0, 100.0*tSpawnUs/tot);
  printf("[perf]   sort+limit     %6.1f ms (%4.1f%%)\n", tSortUs/1000.0, 100.0*tSortUs/tot);
  printf("[perf]   deltaT reduce  %6.1f ms (%4.1f%%)\n", tDtUs/1000.0, 100.0*tDtUs/tot);
  printf("[perf]   RK stages+RHS  %6.1f ms (%4.1f%%)\n", tRkUs/1000.0, 100.0*tRkUs/tot);
}

real DgSolver::step(real tStep) {
  real t = 0;
  while (t < tStep) {
    if (iter % adaptEvery == 0 && nLvls > 1) adaptLeaves();

    computeDeltaT();
    // blow-up guard: a NaN/vacuum state caps velocities at the sanitizer bound
    // and collapses dt -- abort loudly (with the offending location) instead of
    // spinning forever
    static real dtFloor = getenv("DGDTFLOOR") ? (real)atof(getenv("DGDTFLOOR"))
                                              : (real)1e-12;   // env override:
    // raise to abort-with-location on a dt CRAWL (positive but collapsing dt
    // never trips the 1e-12 NaN guard; the crawl diagnosis lever)
    if (!(deltaT > (real)0.0) || deltaT < dtFloor) {
      cudaDeviceSynchronize();
      real *lam = getField(D_LAM);
      i32 cMin = 0; real vMin = 1e30;
      for (i32 c = 0; c < hashTable.nKeys*blockSizeTot; c++)
        if (lam[c] < vMin) { vMin = lam[c]; cMin = c; }
      i32 b = cMin/blockSizeTot, nd = cMin%blockSizeTot;
      i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
      printf("[blowup] iter %d simT %.6e: dt = %.3e at lvl %d elem (%d,%d,%d) node %d: "
             "Q = %.3e %.3e %.3e %.3e %.3e | sensor theta=%.3f theta*lam=%.3e\n",
             iter, (double)(simT + t), (double)deltaT, lvl, ib, jb, kb, nd,
             (double)getField(D_RHO)[cMin], (double)getField(D_RHOU)[cMin],
             (double)getField(D_RHOV)[cMin], (double)getField(D_RHOW)[cMin],
             (double)getField(D_RHOE)[cMin],
             (double)getField(D_SCRATCH)[(u64)b*blockSizeTot + 1],
             (double)getField(D_SCRATCH)[(u64)b*blockSizeTot]);
      fflush(stdout);
      exit(2);
    }
    if (t + deltaT > tStep) deltaT = tStep - t;

    long tRk0 = nowUs();
    dgCopyQ0Kernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    // Gauss-Legendre flux reconstruction (--gauss) vs collocated Lobatto DGSEM.
    // dpSbp: the per-element DP-SBP upwind parameters (Gamma) must be fresh for
    // every RHS evaluation -- they depend on the current state.
    #define DG_RHS(T) do { \
      if (dpSbp > (real)0.0) dgDpGammaKernel<<<cudaGridSize, cudaBlockSize>>>(*this); \
      if (gauss) dgRhsGaussKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this, (T)); \
      else dgRhsKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this, (T)); \
      if (cutOn) { \
        size_t shm = (5*blockSizeTot + 10*CUT_NBMAX_H)*sizeof(real); \
        dgRhsCutKernel<<<nCutElem, blockSizeTot, shm>>>(*this, (T)); } } while (0)
    for (i32 stage = 0; stage < 3; stage++) {
      // SSP-RK3 stage abscissae: t, t+dt, t+dt/2
      real stageT = simT + t + ((stage == 1) ? deltaT : ((stage == 2) ? (real)0.5*deltaT : (real)0.0));
      if (mood) {
        // a-posteriori MOOD: attempt pure DG (alpha=0), detect failed cells,
        // recompute ONLY those with the first-order FV volume (HLLC faces
        // unchanged -> local).  No a priori sensor, no AV.
        dgMoodResetKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        if (bulkC > (real)0.0)   // bulk gate needs a fresh sensor even under MOOD
          dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);
        DG_RHS(stageT);   // DG trial
        dgMoodDetectKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage, deltaT);
        DG_RHS(stageT);   // FV redo for flagged
        dgRk3StageKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage, deltaT);
        if (cutOn) applySrd();   // stage-wise state redistribution (cut cells)
        dgPositivityKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // last-resort net
        if (ibOn && (ibFillEvery == 0 || stage == 2))
          dgIbFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (ibSbm) dgIbSolidFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (mood) dgIbGhostClampKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        continue;
      }
      if (avOn || subFv || bulkC > (real)0.0)   // per-element nu (AV jump
        dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);  // penalty),
        // subcell-FV blend factor, and/or the bulk-viscosity gate (slot 1)
      DG_RHS(stageT);
      dgRk3StageKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage, deltaT);
      if (cutOn) applySrd();     // stage-wise state redistribution (cut cells)
      dgPositivityKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      if (esLim)   // ES limiter: bound the cell entropy by the RHS's slots 3/4
        dgEntropyLimitKernel<<<cudaGridSize, cudaBlockSize>>>(*this, deltaT);
      if (ibOn && (ibFillEvery == 0 || stage == 2))   // refill ghosts from the
        dgIbFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (ibSbm) dgIbSolidFillKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (mood) dgIbGhostClampKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // post-stage
        // fluid state, so the next stage's face lifts read a wall-consistent
        // trace (ibFillEvery 1: hold ghosts frozen across the step's stages)
    }
    #undef DG_RHS
    cudaDeviceSynchronize();
    tRkUs += nowUs() - tRk0;

    // ── TEMP blowup diagnostic (env DGDBG=1): track the nose detonation patch
    // cell (lvl4 elem 105,213) and its face neighbours (flag ghosts) to catch
    // the SEED -- a bad wall ghost vs the fluid patch growing on its own.
    if (getenv("DGDBG") && iter >= 22 && iter <= 34) {
      auto findBlk = [&](i32 L, i32 I, i32 J)->i32 {
        for (i32 b = 0; b < hashTable.nKeys; b++) {
          if (bLocList[b] == kEmpty) continue;
          i32 l,ii,jj,kk; decode(bLocList[b], l,ii,jj,kk);
          if (l==L && ii==I && jj==J) return b;
        } return -1; };
      auto cellStat = [&](i32 b, double &rmax, double &pmin)->void {
        rmax=-1; pmin=1e300;
        for (i32 nd=0; nd<blockSizeTot; nd++){ u64 cc=(u64)b*blockSizeTot+nd;
          double r=getField(D_RHO)[cc];
          double p=(dgGam-1.0)*(getField(D_RHOE)[cc]-0.5*(pow(getField(D_RHOU)[cc],2)
                    +pow(getField(D_RHOV)[cc],2)+pow(getField(D_RHOW)[cc],2))/r);
          if (r>rmax) rmax=r; if (p<pmin) pmin=p; } };
      i32 bT = findBlk(4,111,182);
      if (bT >= 0) {
        double r0,p0; cellStat(bT,r0,p0);
        // also the max |velocity| in the cell (this blowup is a velocity spike)
        double vmax=0; for (i32 nd=0;nd<blockSizeTot;nd++){ u64 cc=(u64)bT*blockSizeTot+nd;
          double r=fmax(getField(D_RHO)[cc],1e-30);
          double sp=sqrt(pow(getField(D_RHOU)[cc],2)+pow(getField(D_RHOV)[cc],2)+pow(getField(D_RHOW)[cc],2))/r;
          if (sp>vmax) vmax=sp; }
        double mood = getField(D_SCRATCH)[(u64)bT*blockSizeTot + 6];
        printf("[dbg] iter %d t=%.6e  (111,182) rhoMax=%.3e pMin=%.3e |v|max=%.3e mood=%.0f | nbrs:",
               iter, (double)(simT+t), r0, p0, vmax, mood);
        const i32 fslot[4] = {12,14,10,16}; const char *fn[4] = {"-x","+x","-y","+y"};
        for (i32 f = 0; f < 4; f++) {
          i32 nb = nbrIdxList[27*bT + fslot[f]];
          if (nb == (i32)bEmpty || nb < 0 || nb >= hashTable.nKeys) { printf(" %s=none", fn[f]); continue; }
          double nr,npm; cellStat(nb,nr,npm);
          const char *cl = ibClassList[nb]==IB_FLUID?"F":(ibClassList[nb]==IB_GHOST?"G":"S");
          printf(" %s[%s]r=%.2e,p=%.2e", fn[f], cl, nr, npm);
        }
        printf("\n"); fflush(stdout);
      }
    }

    // boundary-flux time average over [fluxAvgT0, fluxAvgT1]: accumulate
    // flux(t)*dt each timestep (dt-weighted -> a true time integral, so the
    // unsteady-wake fluctuations average out).  ⟨in⟩ vs ⟨out⟩ over the window
    // is the steady-state conservation check.
    real tNow = simT + t + deltaT;
    if (fluxAvgT1 > fluxAvgT0 && tNow >= fluxAvgT0 && tNow < fluxAvgT1) {
      double b[4]; boundaryMassFlux(b);
      for (i32 f = 0; f < 4; f++) fluxAvgAcc[f] += b[f]*(double)deltaT;
      fluxAvgTime += (double)deltaT;
    }

    t += deltaT;
    iter++;
  }
  simT += t;
  return t;
}

/* ════════════════════════════════════════════════════════════════════════
 * Diagnostics / validation (host-side, managed memory)
 * ════════════════════════════════════════════════════════════════════════ */

// GLL-weighted domain totals of the conserved variables
void DgSolver::dgTotalConserved(double &mass, double &momx, double &energy) {
  cudaDeviceSynchronize();
  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);
  mass = momx = energy = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    if (ibOn && ibClassList[b] != IB_FLUID) continue;   // fluid-only totals
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    double h[3]; hostElemSize(*this, lvl, h);
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      double wv = (0.5*h[0]*w[i])*(0.5*h[1]*w[j])*(0.5*h[2]*w[k]);
      i32 c = b*blockSizeTot + nd;
      mass   += wv*(double)getField(D_RHO)[c];
      momx   += wv*(double)getField(D_RHOU)[c];
      energy += wv*(double)getField(D_RHOE)[c];
    }
  }
}

// nodal line profile along y = ymid, z = zmid: (x, rho, u, p) sorted by x
void DgSolver::writeLineProfile(const char *fileName) {
  cudaDeviceSynchronize();
  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);
  double ymid = 0.5*domainSize[1], zmid = 0.5*domainSize[2];

  struct Row { double x, rho, u, p; };
  std::vector<Row> rows;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    double h[3]; hostElemSize(*this, lvl, h);
    if (ymid < jb*h[1] || ymid >= (jb+1)*h[1]) continue;
    if (!pseudo2D && (zmid < kb*h[2] || zmid >= (kb+1)*h[2])) continue;

    // nodes nearest the midlines
    i32 jS = 0, kS = pseudo2D ? blockSize/2 : 0;
    double bestJ = 1e30, bestK = 1e30;
    for (i32 m = 0; m < NNODE; m++) {
      double y = (jb + 0.5*(xi[m]+1.0))*h[1];
      if (fabs(y-ymid) < bestJ) { bestJ = fabs(y-ymid); jS = m; }
      if (!pseudo2D) {
        double z = (kb + 0.5*(xi[m]+1.0))*h[2];
        if (fabs(z-zmid) < bestK) { bestK = fabs(z-zmid); kS = m; }
      }
    }
    for (i32 i = 0; i < NNODE; i++) {
      i32 c = b*blockSizeTot + i + jS*NNODE + kS*NNODE*NNODE;
      double rho = getField(D_RHO)[c];
      double ru  = getField(D_RHOU)[c];
      double rv  = getField(D_RHOV)[c];
      double rw  = getField(D_RHOW)[c];
      double rE  = getField(D_RHOE)[c];
      double u = ru/rho;
      double p = (dgGam-1.0)*(rE - 0.5*(ru*ru+rv*rv+rw*rw)/rho);
      rows.push_back({(ib + 0.5*(xi[i]+1.0))*h[0], rho, u, p});
    }
  }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b){ return a.x < b.x; });
  FILE *f = fopen(fileName, "w");
  if (!f) { printf("[profile] cannot open %s\n", fileName); return; }
  fprintf(f, "# x rho u p\n");
  for (auto &r : rows) fprintf(f, "%.9e %.9e %.9e %.9e\n", r.x, r.rho, r.u, r.p);
  fclose(f);
  printf("[profile] wrote %zu nodes to %s\n", rows.size(), fileName);
}

// L2 density error against the exact (advected) isentropic vortex at time t
void DgSolver::computeVortexError(real t) {
  cudaDeviceSynchronize();
  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);
  double cx = 0.5*domainSize[0] + (double)vortexU0*t;
  double cy = 0.5*domainSize[1] + (double)vortexU0*t;
  // wrap the center into the periodic domain
  cx = fmod(cx, (double)domainSize[0]); if (cx < 0) cx += domainSize[0];
  cy = fmod(cy, (double)domainSize[1]); if (cy < 0) cy += domainSize[1];

  double err2 = 0, vol = 0;
  const double eps = 5.0, gm = dgGam;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    double h[3]; hostElemSize(*this, lvl, h);
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      double x = (ib + 0.5*(xi[i]+1.0))*h[0];
      double y = (jb + 0.5*(xi[j]+1.0))*h[1];
      // nearest periodic image of the center
      double dx = x-cx; if (dx > 0.5*domainSize[0]) dx -= domainSize[0]; if (dx < -0.5*domainSize[0]) dx += domainSize[0];
      double dy = y-cy; if (dy > 0.5*domainSize[1]) dy -= domainSize[1]; if (dy < -0.5*domainSize[1]) dy += domainSize[1];
      double r2 = dx*dx + dy*dy;
      double dT = -(gm-1.0)*eps*eps/(8.0*gm*M_PI*M_PI)*exp(1.0-r2);
      double rhoEx = pow(std::max(1.0+dT, 1e-6), 1.0/(gm-1.0));
      double wv = (0.5*h[0]*w[i])*(0.5*h[1]*w[j])*(0.5*h[2]*w[k]);
      double d = (double)getField(D_RHO)[b*blockSizeTot+nd] - rhoEx;
      err2 += wv*d*d;
      vol  += wv;
    }
  }
  printf("[vortex] t=%.4f  L2(rho) error = %.6e  (nblocks %d)\n",
         (double)t, sqrt(err2/vol), hashTable.nKeys);
}

// max nodal deviation from the icType-3 uniform free stream (M3 test)
// L2 velocity error + kinetic-energy retention vs the exact (stationary)
// Gresho vortex, GLL-weighted (mirrors CompressibleSolver::computeGreshoError)
void DgSolver::computeGreshoError(void) {
  cudaDeviceSynchronize();
  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);
  double cx = 0.5*domainSize[0], cy = 0.5*domainSize[1];
  double l2Vel = 0, keNum = 0, keExact = 0, area = 0;

  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    double h[3]; hostElemSize(*this, lvl, h);
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i = nd % NNODE, j = (nd/NNODE)%NNODE, k = nd/(NNODE*NNODE);
      double x = (ib + 0.5*(xi[i]+1.0))*h[0];
      double y = (jb + 0.5*(xi[j]+1.0))*h[1];
      double ddx = x-cx, ddy = y-cy, r = sqrt(ddx*ddx + ddy*ddy);
      double wang = (r < 0.2) ? 5.0 : (r < 0.4 ? 2.0/r - 5.0 : 0.0);
      double ue = -wang*ddy, ve = wang*ddx;
      i32 c = b*blockSizeTot + nd;
      double rr = getField(D_RHO)[c];
      double u = getField(D_RHOU)[c]/rr, v = getField(D_RHOV)[c]/rr;
      double wA = (0.5*h[0]*w[i])*(0.5*h[1]*w[j])*(0.5*w[k]);   // z-normalized
      l2Vel   += ((u-ue)*(u-ue) + (v-ve)*(v-ve))*wA;
      keNum   += 0.5*rr*(u*u + v*v)*wA;
      keExact += 0.5*(ue*ue + ve*ve)*wA;
      area    += wA;
    }
  }
  printf("[gresho] L2(|vel|) error = %.6e   KE/KE0 = %.4f   (nblocks %d)\n",
         sqrt(l2Vel/area), keNum/keExact, hashTable.nKeys);
}

real DgSolver::maxDeviationFromUniform(void) {
  cudaDeviceSynchronize();
  // the uniform reference state of the active IC: case-3 free stream, or the
  // case-9 freestream at Mach machInf (--mach 0 = the IB rest-state gate)
  double W[5] = {1.0, 0.3, 0.2, pseudo2D ? 0.0 : 0.1, 1.0};
  if (icType == 7) { W[1] = machInf; W[2] = W[3] = 0.0; W[4] = 1.0/dgGam; }
  double U[5];
  U[0] = W[0]; U[1] = W[0]*W[1]; U[2] = W[0]*W[2]; U[3] = W[0]*W[3];
  U[4] = W[4]/(dgGam-1.0) + 0.5*W[0]*(W[1]*W[1]+W[2]*W[2]+W[3]*W[3]);
  double dev = 0, devQ[5] = {0,0,0,0,0};
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    if (ibOn && ibClassList[b] != IB_FLUID) continue;   // fluid-only gate
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++) {
        double d = fabs((double)getField(D_RHO+q)[b*blockSizeTot+nd] - U[q]);
        dev = std::max(dev, d);
        devQ[q] = std::max(devQ[q], d);
      }
  }
  if (ibOn) printf("[ibrestQ] rho=%.2e rhou=%.2e rhov=%.2e rhow=%.2e rhoE=%.2e\n",
                   devQ[0], devQ[1], devQ[2], devQ[3], devQ[4]);
  if (ibOn) {   // where does the worst deviation live?
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      if (bLocList[b] == kEmpty || (ibOn && ibClassList[b] != IB_FLUID)) continue;
      for (i32 nd = 0; nd < blockSizeTot; nd++)
        for (i32 q = 0; q < 5; q++)
          if (fabs((double)getField(D_RHO+q)[b*blockSizeTot+nd] - U[q]) == dev) {
            i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
            double h[3]; hostElemSize(*this, lvl, h);
            i32 i = nd % NNODE, j = (nd/NNODE)%NNODE;
            double w[NNODE], xi[NNODE]; dgGetHostOps(w, xi, gauss);
            double x = (ib + 0.5*(xi[i]+1.0))*h[0], y = (jb + 0.5*(xi[j]+1.0))*h[1];
            double phi = sqrt((x-ibX)*(x-ibX) + (y-ibY)*(y-ibY)) - ibR;
            printf("[ibrestLoc] q=%d lvl=%d elem(%d,%d) node(%d,%d) x=%.3f y=%.3f "
                   "phi/h=%.2f theta=%.1fdeg\n", q, lvl, ib, jb, i, j, x, y,
                   phi/h[0], atan2(y-ibY, x-ibX)*180.0/PI);
            return (real)dev;
          }
    }
  }
  return (real)dev;
}

void DgSolver::paintPressure(const char *fileName) {
  dgPressureToScratchKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  paintField(D_SCRATCH, fileName);
}

void DgSolver::paintBrinkPhi(const char *fileName) {
  dgBrinkPhiToScratchKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  paintField(D_SCRATCH, fileName);
}

void DgSolver::paintSensor(const char *fileName) {
  // D_LAM holds the per-node dt bound after a step (a proxy for lambda*h); the
  // useful sensor view is the refinement-level map, painted alongside
  paintField(D_LAM, fileName);
}

void DgSolver::paintIbClass(const char *fileName) {
  dgIbClassToScratchKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  paintField(D_SCRATCH, fileName);
}

// outward numerical mass flux through each domain boundary (x-lo,x-hi,y-lo,
// y-hi): positive = leaving.  -sum = discrete d/dt(fluid mass); comparing to
// the actual dM/dt isolates the IB ghost non-conservation.
void DgSolver::boundaryMassFlux(double bnd[4]) {
  real *b;
  cudaMallocManaged(&b, 4*sizeof(real));
  cudaMemset(b, 0, 4*sizeof(real));
  dgBoundaryMassFluxKernel<<<cudaGridSize, cudaBlockSize>>>(*this, b);
  cudaDeviceSynchronize();
  for (i32 f = 0; f < 4; f++) bnd[f] = (double)b[f];
  cudaFree(b);
}

// troubled-element map: the shock-indicator / subcell-FV blend factor per
// element (bright = FV-blended / sensor-flagged, dark = pure high-order DG)
void DgSolver::paintTroubled(const char *fileName) {
  dgAvNuKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this);   // fresh theta_e
  dgTroubledToScratchKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  paintField(D_LAM, fileName);
}

// Cp(theta) sampled just off the surface; also prints the stagnation pressure
// against the exact Rayleigh-pitot value and integrates the pressure drag
void DgSolver::writeIbSurface(const char *fileName) {
  const i32 nTheta = 256;
  real *out;
  cudaMallocManaged(&out, 3*nTheta*sizeof(real));
  double hF[3]; hostElemSize(*this, nLvls-1, hF);
  dgIbSurfaceKernel<<<(nTheta+63)/64, 64>>>(*this, nTheta, (real)(0.5*hF[0]), out);
  cudaDeviceSynchronize();

  double pInf = 1.0/dgGam, qInf = 0.5*machInf*machInf;   // rho=1, a=1 units
  FILE *f = fopen(fileName, "w");
  fprintf(f, "# theta(deg from stagnation)  Cp  p  vt   [M=%g]\n", (double)machInf);
  for (i32 t = 0; t < nTheta; t++) {
    if (out[3*t+1] < 0) continue;                        // sample missed the grid
    double thDeg = 180.0 - out[3*t]*180.0/PI;            // 0 = windward stagnation
    if (thDeg < 0) thDeg += 360.0;
    fprintf(f, "%.4f %.6e %.6e %.6e\n", thDeg,
            ((double)out[3*t+1] - pInf)/qInf, (double)out[3*t+1], (double)out[3*t+2]);
  }
  fclose(f);
  cudaFree(out);
}

void DgSolver::computeIbGates(void) {
  cudaDeviceSynchronize();
  const i32 nTheta = 256;
  real *out;
  cudaMallocManaged(&out, 3*nTheta*sizeof(real));
  double hF[3]; hostElemSize(*this, nLvls-1, hF);
  dgIbSurfaceKernel<<<(nTheta+63)/64, 64>>>(*this, nTheta, (real)(0.5*hF[0]), out);
  cudaDeviceSynchronize();

  double pInf = 1.0/dgGam, qInf = 0.5*machInf*machInf;
  // stagnation pressure: sample nearest theta = pi (windward); exact reference
  // is the Rayleigh-pitot normal-shock stagnation pressure
  double pStag = 0, best = 1e30;
  for (i32 t = 0; t < nTheta; t++)
    if (out[3*t+1] > 0 && fabs(out[3*t] - PI) < best) {
      best = fabs(out[3*t] - PI);
      pStag = out[3*t+1];
    }
  double M = machInf, g = dgGam;
  double pPitot = pInf * pow(0.5*(g+1.0)*M*M, g/(g-1.0))
                       * pow((g+1.0)/(2.0*g*M*M - (g-1.0)), 1.0/(g-1.0));
  // pressure drag coefficient (Cd per unit span, reference length D = 2R)
  double cd = 0, dth = 2.0*PI/nTheta;
  for (i32 t = 0; t < nTheta; t++)
    if (out[3*t+1] > 0)
      cd += ((double)out[3*t+1] - pInf)*cos((double)out[3*t])*(double)ibR*dth;
  cd /= -(qInf*2.0*ibR);   // n_x = cos(theta); drag = -integral p n_x dl
  cudaFree(out);

  // bow-shock standoff: walk the stagnation line upstream of the nose and
  // find the outermost point where p exceeds twice the freestream (mid-jump);
  // reference: Billig's correlation Delta/R = 0.386 exp(4.67/M^2)
  double xNose = ibX - ibR, xShock = -1;
  i32 nS = 512;
  real *ls; cudaMallocManaged(&ls, 3*nS*sizeof(real));
  dgIbStagLineKernel<<<(nS+63)/64, 64>>>(*this, nS, ls);
  cudaDeviceSynchronize();
  for (i32 t = 0; t < nS; t++)   // samples run inflow -> nose
    if (ls[3*t+1] > 2.0*pInf) { xShock = ls[3*t]; break; }
  double xSurr = xNose;   // innermost active-fluid sample = the effective (surrogate) nose
  for (i32 t = 0; t < nS; t++)
    if (ls[3*t+1] > 0.0) xSurr = ls[3*t];   // last valid (largest-x) sample
  // shock CENTRE = steepest pressure rise (vs the 2*pinf FOOT above)
  double xShockG = -1, maxG = 0;
  for (i32 t = 1; t < nS; t++)
    if (ls[3*(t-1)+1] > 0 && ls[3*t+1] > 0) {
      double g = (ls[3*t+1] - ls[3*(t-1)+1]);
      if (g > maxG) { maxG = g; xShockG = 0.5*(ls[3*t] + ls[3*(t-1)]); }
    }
  cudaFree(ls);
  printf("[ibgeom] realNose=%.4f surrNose=%.4f (gap %.4f)  shockFoot=%.4f shockCtr=%.4f "
         "standoff/D: foot=%.4f ctr=%.4f (Billig %.4f)\n", xNose, xSurr, xSurr - xNose,
         xShock, xShockG, (xShock>0)?(xNose-xShock)/(2.0*ibR):-1,
         (xShockG>0)?(xNose-xShockG)/(2.0*ibR):-1, 0.386*exp(4.67/(M*M))*0.5);
  double standoff = (xShock > 0) ? (xNose - xShock)/(2.0*ibR) : -1;
  double billig = 0.386*exp(4.67/(M*M))*ibR/(2.0*ibR);

  printf("[ibgates] M=%.2f  p_stag=%.4f (pitot %.4f, err %+.2f%%)  "
         "standoff/D=%.4f (Billig %.4f, err %+.2f%%)  Cd=%.4f\n",
         M, pStag, pPitot, 100.0*(pStag - pPitot)/pPitot,
         standoff, billig, (standoff > 0) ? 100.0*(standoff - billig)/billig : 0.0,
         cd);

  // entropy L2 (Funada & Imamura Eq 28): sqrt(<(P/Pinf (rhoinf/rho)^gam - 1)^2>)
  // over the FLUID volume -- inviscid smooth flow conserves entropy exactly,
  // so this is the wall-accuracy gate in the shock-free regime (M < ~0.7).
  // Node-mean quadrature per element (a gate, not a paper-grade integral).
  {
    double sInt = 0, vInt = 0;
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty || ibClassList[b] != IB_FLUID) continue;
      i32 lvl, ib2, jb2, kb2;
      decode(loc, lvl, ib2, jb2, kb2);
      if (!isInteriorBlock(lvl, ib2, jb2, kb2)) continue;
      double hE[3]; hostElemSize(*this, lvl, hE);
      double vNode = hE[0]*hE[1]*(pseudo2D ? 1.0 : hE[2])/blockSizeTot;
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 c = b*blockSizeTot + nd;
        double rho = fmax((double)getField(D_RHO)[c], 1e-12);
        double ke  = 0.5*((double)getField(D_RHOU)[c]*(double)getField(D_RHOU)[c]
                        + (double)getField(D_RHOV)[c]*(double)getField(D_RHOV)[c]
                        + (double)getField(D_RHOW)[c]*(double)getField(D_RHOW)[c])/rho;
        double p   = (dgGam - 1.0)*((double)getField(D_RHOE)[c] - ke);
        double s   = p/pInf*pow(1.0/rho, (double)dgGam) - 1.0;
        sInt += vNode*s*s;
        vInt += vNode;
      }
    }
    printf("[ibentropy] L2 = %.6e  (fluid volume %.4f)\n",
           (vInt > 0) ? sqrt(sInt/vInt) : -1.0, vInt);
  }
  printf("[ibcnt] nodonor=%d lofallback=%d shockgate=%d zsMeanFloor=%d\n",
         ibCnt[IB_CNT_NODONOR], ibCnt[IB_CNT_RETRY1], ibCnt[IB_CNT_FALLBACK], ibCnt[3]);
}

bool DgSolver::selfTest(void) {
  bool ok = dgOperatorSelfTest(gauss, frType);
  printf("[selftest] blockSize=%d NNODE=%d (p=%d), nBlocksMax=%d, fields=%d\n",
         blockSize, NNODE, dgOrder, nBlocksMax, nFields);
  return ok;
}
