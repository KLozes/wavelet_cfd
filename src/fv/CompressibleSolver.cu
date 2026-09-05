#include <iostream>
#include <climits>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <thrust/extrema.h>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#include "CompressibleSolver.cuh"
#include "CutClip.h"
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include "KtauSst.h"
#include "CompressibleSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"
#ifdef USE_MGPU
#include "Comm.cuh"
#endif

void CompressibleSolver::initialize(void) {
  // (initPartition() runs in the MultiLevelSparseGrid constructor for USE_MGPU)
  periodic = (bcType == 2);   // torus refinement: keep seam edges matched (see MultiLevelSparseGrid)
  if (mdFlux && !pseudo2D) {
    printf("[warn] multiD corner flux is implemented for pseudo-2D only; disabling\n");
    mdFlux = 0;
  }
  if (mdFlux && mu > 0) {
    printf("[warn] multiD corner flux carries no viscous term; disabling (mu = %g)\n", (double)mu);
    mdFlux = 0;
  }
  // wavelet-normalization scales (device-side global maxima)
  cudaMallocManaged(&globalScale, 3*sizeof(real));
  cudaMemset(globalScale, 0, 3*sizeof(real));
  // Painting during the build cascade costs one UNIFORM-FINE frame per level
  // (nLvls frames); honour the same switch the main loop uses.
  buildInitialGrid(paintOn != 0);
#ifdef USE_MGPU
  // Z-curve mode: the uniform-weight cut over-loads the ranks whose curve
  // interval covers the refined region.  Recount the real per-base-column
  // block weights, re-cut the curve, and rebuild from scratch on the balanced
  // map (the IC is analytic, so a rebuild IS the migration).
  if (g_partMode == 1 && comm::size() > 1) {
    rebalanceWeights();          // count + allreduce + re-cut + re-derive
    resetGrid();
    buildInitialGrid(false);
    double wOwn = 0;
    for (i32 t = 0; t < part.nb[0]*part.nb[1]*part.nb[2]; t++)
      if (ownerBase[t] == part.rank) wOwn += wBase[t];
    printf("[sfc] rank %d owns %.0f blocks after balanced re-cut\n", part.rank, wOwn);
  }
#endif
  zeroAccumulator();   // the cascade dirtied the shared bank (LSRK needs 0)
  stampIbGeometry();   // geometry cache for the final grid topology
  if (p1) p1InitSlopesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // central-difference slopes of the IC
  // (cudaMemAdvise pinning of this object was tried 2026-08-26 and MEASURED
  // WORSE: 138 -> 176 s.  Remote host access every step costs more than the
  // page migration it replaces.  Do not re-add without measuring.)
}

// base grid + IC + the refine/re-IC cascade (the body of initialize(); run a
// second time in Z-curve mode after the weighted re-cut)
void CompressibleSolver::buildInitialGrid(bool doPaint) {
  initializeBaseGrid();
  setInitialConditions();
  primitiveToConservative();
  sortBlocks();
  setBoundaryConditions();
#ifdef USE_MGPU
  rebuildGhosts();            // base-level partition ghost ring (directory exchange)
  setInitialConditions();     // fill the ghost cells (analytic IC, global position)
  primitiveToConservative();
  setBoundaryConditions();
#endif
  cudaDeviceSynchronize();
  printf("nblocks %d\n", hashTable.nKeys);
  if (doPaint) paint();

  // build the adaptive grid by repeatedly transforming / refining
  for (i32 lvl=1; lvl<nLvls; lvl++) {
    // Dump the grid at every stage of the BUILD cascade (geometry only -- the
    // IC is uniform freestream, so nothing here is flow-driven).  --gridtrace.
    if (gridTrace) {
      char fn[256];
      snprintf(fn, sizeof(fn), "output/gridtrace_%d.dat", lvl-1);
      writeGridBlocks(fn);
    }
    forwardWaveletTransform();
    adaptGridConsistent();     // MGPU: cascade with per-kernel seam sync (== adaptGrid on 1 GPU)
    setInitialConditions();
    primitiveToConservative();
    setBoundaryConditions();
    sortBlocks();
#ifdef USE_MGPU
    rebuildGhosts();           // (re)create the partition ghost blocks at this level
    setInitialConditions();    // fill the new ghost cells from the analytic IC
    primitiveToConservative();
    setBoundaryConditions();
#endif
    cudaDeviceSynchronize();
    printf("nblocks %d\n", hashTable.nKeys);
    if (doPaint) paint();
  }
  if (gridTrace) {
    char fn[256];
    snprintf(fn, sizeof(fn), "output/gridtrace_%d.dat", nLvls-1);
    writeGridBlocks(fn);
  }
}

#ifdef USE_MGPU
// the per-neighbor directory/halo buffers are sized for the CURRENT neighbor
// set; any ownerBase change can change nNbr, so they must be rebuilt from
// scratch after a re-cut (buildDirectories re-allocates on null/dirSlot=0)
void CompressibleSolver::invalidateCommBuffers(void) {
  if (dirSendCnt) { cudaFree(dirSendCnt); cudaFree(dirRecvCnt); cudaFree(dirFill); }
  dirSendCnt = dirRecvCnt = dirFill = nullptr;
  if (dirSendLoc) { cudaFree(dirSendLoc); cudaFree(dirRecvLoc); cudaFree(sendBuf); cudaFree(recvBuf); }
  dirSendLoc = dirRecvLoc = nullptr; sendBuf = recvBuf = nullptr;
  dirSlot = 0;
  // the NEED/adopt buffers are per-neighbour too; a re-cut can change nNbr, so
  // free them here as well (buildDirectories reallocates all of them on demand)
  if (needCnt) { cudaFree(needCnt); cudaFree(needRecvCnt); }
  needCnt = needRecvCnt = nullptr;
  if (needLoc) { cudaFree(needLoc); cudaFree(needRecvLoc); }
  needLoc = needRecvLoc = nullptr;
  needSlot = 0;
}
#endif

#ifdef USE_MGPU
// ---------------------------------------------------------------------------
// Dynamic Z-curve rebalancing.  Every rebalanceEvery adaptations: recount the
// per-base-column weights, and if the load imbalance exceeds ~15%, re-cut the
// Morton curve and MIGRATE whole base columns to their new owners.  The old
// and new maps are replicated on every rank, so both sides of every transfer
// are known locally; blocks travel as (loc, NEVOLVE field slabs).  The
// loopback comm backend barriers globally in neighborExchange, so ALL ranks
// call this together -- the decision is deterministic from replicated data.
// ---------------------------------------------------------------------------
void CompressibleSolver::migrateBlocks(const i32 *newOwner) {
  i32 P = comm::size();
  i32 nCol = part.nb[0]*part.nb[1]*part.nb[2];

  // partner set = ranks I lose columns to, plus ranks I gain columns from
  std::vector<i32> partnerOf(P, -1);
  std::vector<i32> mig;
  for (i32 c = 0; c < nCol; c++) {
    i32 o = ownerBase[c], w = newOwner[c];
    if (o == w) continue;
    if (o == part.rank && partnerOf[w] < 0) { partnerOf[w] = (i32)mig.size(); mig.push_back(w); }
    if (w == part.rank && partnerOf[o] < 0) { partnerOf[o] = (i32)mig.size(); mig.push_back(o); }
  }
  i32 nm = (i32)mig.size();

  // my departing blocks, grouped by destination (host scan; managed memory)
  cudaDeviceSynchronize();
  std::vector<std::vector<i32>> outIdx(nm);
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl = (i32)(loc >> 60);
    i32 kb  = (i32)((loc >> 40) & ((1<<20)-1)) - 1;
    i32 jb  = (i32)((loc >> 20) & ((1<<20)-1)) - 1;
    i32 ib  = (i32)( loc        & ((1<<20)-1)) - 1;
    // exterior-ring blocks clamp to their edge column (ownerPE semantics) so a
    // departing boundary column ships its domain-exterior ghosts along with it
    // -- the new owner needs them immediately (level-0 boundary cells have no
    // parent to interpolate from; the ring is otherwise only rebuilt at the
    // next adaptGrid).
    i32 ci = ib >> lvl, cj = jb >> lvl, ck = kb >> lvl;
    if (ci < 0) ci = 0;  if (ci >= part.nb[0]) ci = part.nb[0]-1;
    if (cj < 0) cj = 0;  if (cj >= part.nb[1]) cj = part.nb[1]-1;
    if (ck < 0) ck = 0;  if (ck >= part.nb[2]) ck = part.nb[2]-1;
    i32 c = ci + part.nb[0]*(cj + part.nb[1]*ck);
    if (ownerBase[c] == part.rank && newOwner[c] != part.rank)
      outIdx[partnerOf[newOwner[c]]].push_back(b);
  }

  // counts, then payloads
  std::vector<i32> sc(nm), rc(nm);
  for (i32 t = 0; t < nm; t++) sc[t] = (i32)outIdx[t].size();
  std::vector<void*> sp(nm), rp(nm); std::vector<size_t> sb(nm), rb(nm);
  for (i32 t = 0; t < nm; t++) { sp[t]=&sc[t]; rp[t]=&rc[t]; sb[t]=sizeof(i32); rb[t]=sizeof(i32); }
  comm::neighborExchange(nm, mig.data(), sp.data(), sb.data(), rp.data(), rb.data());

  size_t slab = (size_t)NEVOLVE*blockSizeTot;                  // reals per block
  size_t blkB = sizeof(u64) + slab*sizeof(real);               // shipped bytes per block
  std::vector<char*> sBuf(nm, nullptr), rBuf(nm, nullptr);
  for (i32 t = 0; t < nm; t++) {
    if (sc[t]) cudaMallocManaged(&sBuf[t], (size_t)sc[t]*blkB);
    if (rc[t]) cudaMallocManaged(&rBuf[t], (size_t)rc[t]*blkB);
  }
  for (i32 t = 0; t < nm; t++)
    for (i32 x = 0; x < sc[t]; x++) {
      i32 b = outIdx[t][x];
      char *dst = sBuf[t] + (size_t)x*blkB;
      *(u64*)dst = bLocList[b];
      real *fd = (real*)(dst + sizeof(u64));
      for (i32 f = 0; f < NEVOLVE; f++)
        memcpy(fd + (size_t)f*blockSizeTot, getField(f) + (size_t)b*blockSizeTot,
               blockSizeTot*sizeof(real));
    }
  for (i32 t = 0; t < nm; t++) {
    sp[t]=sBuf[t]; rp[t]=rBuf[t];
    sb[t]=(size_t)sc[t]*blkB; rb[t]=(size_t)rc[t]*blkB;
  }
  comm::neighborExchange(nm, mig.data(), sp.data(), sb.data(), rp.data(), rb.data());

  // insert the received blocks (device hash), then land their field data
  i32 R = 0; for (i32 t = 0; t < nm; t++) R += rc[t];
  if (R) {
    u64 *locs; i32 *slots;
    cudaMallocManaged(&locs,  (size_t)R*sizeof(u64));
    cudaMallocManaged(&slots, (size_t)R*sizeof(i32));
    i32 x = 0;
    for (i32 t = 0; t < nm; t++)
      for (i32 y = 0; y < rc[t]; y++) locs[x++] = *(u64*)(rBuf[t] + (size_t)y*blkB);
    migrateInsertKernel<<<(R+255)/256, 256>>>(*this, locs, R, slots);
    cudaDeviceSynchronize();
    x = 0;
    i32 dropped = 0;
    for (i32 t = 0; t < nm; t++)
      for (i32 y = 0; y < rc[t]; y++, x++) {
        i32 sIdx = slots[x];
        if (sIdx == bEmpty) { dropped++; continue; }
        real *fd = (real*)(rBuf[t] + (size_t)y*blkB + sizeof(u64));
        for (i32 f = 0; f < NEVOLVE; f++)
          memcpy(getField(f) + (size_t)sIdx*blockSizeTot, fd + (size_t)f*blockSizeTot,
                 blockSizeTot*sizeof(real));
      }
    if (dropped) printf("[sfc] rank %d: %d migrated blocks DROPPED (pool full)\n", part.rank, dropped);
    cudaFree(locs); cudaFree(slots);
  }
  for (i32 t = 0; t < nm; t++) { if (sBuf[t]) cudaFree(sBuf[t]); if (rBuf[t]) cudaFree(rBuf[t]); }
  nBlocks = hashTable.nKeys;   // inserts bumped the key count
}

// periodic rebalance driver (call with the grid consistent, between steps)
void CompressibleSolver::rebalancePartition(void) {
  if (comm::size() == 1 || g_partMode != 1) return;
  i32 P = comm::size();
  i32 nCol = part.nb[0]*part.nb[1]*part.nb[2];
  // fresh replicated weights
  if (!wBase) cudaMallocManaged(&wBase, (size_t)nCol*sizeof(double));
  cudaMemset(wBase, 0, (size_t)nCol*sizeof(double));
  countBaseWeightsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  comm::allreduceSum(wBase, nCol);
  // imbalance test (identical on every rank: replicated inputs)
  std::vector<double> wr(P, 0.0);
  double total = 0;
  for (i32 c = 0; c < nCol; c++) { wr[ownerBase[c]] += wBase[c]; total += wBase[c]; }
  double mx = 0; for (i32 r = 0; r < P; r++) if (wr[r] > mx) mx = wr[r];
  if (mx * P < 1.15 * total || total == 0) return;   // balanced enough
  // new cut; skip if the map did not change
  if (!ownerScratch) cudaMallocManaged(&ownerScratch, (size_t)nCol*sizeof(i32));
  partitionByWeight(wBase, ownerScratch);
  bool changed = false;
  for (i32 c = 0; c < nCol && !changed; c++) changed = (ownerScratch[c] != ownerBase[c]);
  if (!changed) return;
  printf("[sfc] rank %d rebalancing at iter %d (max/avg = %.2f)\n", part.rank, iter, mx * P / total);
  migrateBlocks(ownerScratch);   // uses the OLD ownerBase to find departures
  memcpy(ownerBase, ownerScratch, (size_t)nCol*sizeof(i32));
  derivePartition();
  invalidateCommBuffers();      // neighbor set may have changed with the map
  sortBlocks();                 // index/flag the migrated owned set
  rebuildGhosts();              // ghost halo + fresh directories under the new map
  haloExchange(0, NEVOLVE);     // fill ghost live data so the pre-cascade restrict/
  setBoundaryConditions();      // forward transform see valid seam neighbours
  zeroAccumulator();            // the shared bank must be clean for LSRK stage 1
  comm::barrier();
  // Migration ships only the LIVE fields (0..7) of the departing owned blocks;
  // the very next adaptation cascade (which runs immediately after this in step)
  // re-settles everything else exactly as a normal cycle: copyToOld regenerates
  // F_OLD from the migrated live data, adaptGridConsistent rebuilds the cross-
  // rank 2:1/support closure (owned-target + NEED/adopt), and reconstituteOld-
  // Snapshot fills any new blocks.  No special re-settle is needed -- that was
  // the missing piece the seam protocol now provides.
}
#endif

#ifdef USE_MGPU
// Count this rank's OWNED blocks (all levels) per level-0 base column,
// allreduce so every rank has the global weights, and re-cut the Morton curve.
void CompressibleSolver::rebalanceWeights(void) {
  i32 n = part.nb[0]*part.nb[1]*part.nb[2];
  if (!wBase) cudaMallocManaged(&wBase, (size_t)n*sizeof(double));
  cudaMemset(wBase, 0, (size_t)n*sizeof(double));
  countBaseWeightsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  comm::allreduceSum(wBase, n);
  partitionByWeight(wBase);
  derivePartition();
  invalidateCommBuffers();     // neighbor set may have changed with the map
  comm::barrier();
}
#endif

// zero the shared bank so the LSRK accumulation (A_1 = 0) starts clean; the
// NEVOLVE fields are contiguous slabs of fieldData
void CompressibleSolver::zeroAccumulator(void) {
  // kernel over active cells, NOT a 56 MB memset of the max allocation -- see
  // zeroAccumulatorKernel.  Newly created blocks are stamped ACTIVE before any
  // stage runs, and the cell loop covers every allocated slot up to nKeys.
  zeroAccumulatorKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  // The face-flux scatter atomicAdds missing-neighbour contributions into the
  // bEmpty TRASH block's slice of this bank, and bEmpty sits OUTSIDE the cell
  // loop -- so the kernel above never reaches it.  The old full-bank cudaMemset
  // did, silently; dropping it let the trash block accumulate without bound
  // (and NaN*0 = NaN defeats the S-roll), which is what broke the immersed RANS
  // path.  Zero it explicitly.
  for (i32 f = 0; f < NEVOLVE; f++)
    zeroTrashBlockKernel<<<1, blockSizeTot>>>(*this, F_RHS + f);
  if (p1)
    for (i32 s = 0; s < 2*P1_NV; s++)
      zeroTrashBlockKernel<<<1, blockSizeTot>>>(*this, F_P1SR + s);
}


real CompressibleSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;
  Timer<std::chrono::microseconds, std::chrono::steady_clock> sub;   // profiling sub-timer

  while (t < tStep) {

    if (maxIter > 0 && iter >= maxIter) break;

    clock.tick();
    // dynamic wavelet adaptation; skipped for a static (fixed) refinement grid
    // Adaptation costs ~5 host-device syncs (sortBlocks 4 + adaptGrid 1): the
    // thrust sort and the hash rebuild need nKeys on the host, so it cannot be
    // made async.  Every 4 steps was the historical default; for a march to
    // steady state the grid moves far slower than that.  --adaptevery.
    if (iter % adaptEvery == 0 && nLvls > 1 && !staticGrid) {
#ifdef USE_MGPU
      // dynamic load rebalance (experimental; off unless --rebalance > 0).  The
      // replicated decision means every rank takes this branch on the same
      // iterations, so the collective comm inside stays in lockstep.
      if (rebalanceEvery > 0 && iter > 0 && iter % (4*rebalanceEvery) == 0)
        rebalancePartition();
      haloExchange(0, NEVOLVE);   // fill last cycle's ghosts before the detail computation
#endif
      if (p1) p1MarkKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // existing cells: marker 1
      restrictFields();
      if (dbgChecks >= 2) scanNonFinite("restrictFields");
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tick(); }
      forwardWaveletTransform();
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tock(); tForwardUs += sub.duration().count(); }
      // refinement cascade; under MGPU this exchanges block activity across rank
      // seams after every create kernel so grading/support close consistently
      if (dbgChecks >= 2) scanNonFinite("forwardWavelet");
      adaptGridConsistent();
      if (dbgChecks >= 2) scanNonFinite("adaptGridConsistent");
#ifdef USE_MGPU
      // the inverse reconstructs each new fine block from its (coarse) parent's
      // F_OLD; rebuild that snapshot for every block created this cycle (owned
      // fills + halo to ghosts, coarse->fine) before the inverse reads it
      reconstituteOldSnapshot();
#endif
      setBoundaryConditions(F_OLD);
      if (dbgChecks >= 2) scanNonFinite("setBC(F_OLD)");
      inverseWaveletTransform();
      if (dbgChecks >= 2) scanNonFinite("inverseWavelet");
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tick(); }
      sortBlocks();
      if (dbgChecks >= 2) scanNonFinite("sortBlocks");
      if (p1)   // blocks created this cycle: slopes from the parent polynomial, coarse to fine
        for (i32 L = 1; L < nLvls; L++) p1ProlongNewKernel<<<cudaGridSize, cudaBlockSize>>>(*this, L);
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tock(); tSortUs += sub.duration().count(); }
#ifdef USE_MGPU
      rebuildGhosts();            // prune the stale ghosts, recreate the 2-ring from neighbors
      topoCheck(2);               // debug: post-rebuild bindings
      haloExchange(0, NEVOLVE);   // fill the fresh ghost blocks
#endif
      setBoundaryConditions();
      // the wavelet snapshot / sort buffer dirtied the shared bank; the LSRK
      // accumulator must be zero when stage 1 begins (A_1 = 0)
      zeroAccumulator();
      stampIbGeometry();   // new blocks appeared: stamp them (moves are carried by the sort)
      if (dbgChecks >= 2) scanNonFinite("stampIbGeometry");
      if (dbgChecks) {   // debug census: owned-interior block count must track the 1-GPU count
        cudaDeviceSynchronize();
        double nOwn = 0;
        for (i32 b = 0; b < hashTable.nKeys; b++) {
          u64 loc = bLocList[b];
          if (loc == kEmpty) continue;
          i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
          if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
#ifdef USE_MGPU
          if (!isOwnedBlock(lvl, ib, jb, kb)) continue;
#endif
          nOwn += 1;
        }
#ifdef USE_MGPU
        comm::allreduceSum(&nOwn, 1);
        if (part.rank == 0)
#endif
        printf("[census] iter %d ownedInteriorBlocks=%.0f\n", iter, nOwn);
      }
    }
    if (dbgChecks) cudaDeviceSynchronize();   // timing sync: debug only
    clock.tock();
    tGrid += clock.duration().count();

    clock.tick();
    // The dt reduction is a HARD host-device round trip (sync + thrust min +
    // host read) and it ran EVERY step, so the GPU queue drained every step:
    // profiled, kernels are ~3 ms/step but wall was ~12 ms/step, the rest
    // being this ping-pong.  dt changes slowly (cfl 0.9 against a measured
    // stability edge of 1.6 leaves 44% margin), so recompute it every
    // dtEvery steps and let the queue run ahead in between.
    if (dtEvery <= 1 || iter % dtEvery == 0) computeDeltaT();
    dtScale = 1.0;
    if (t + deltaT > tStep) {
      // clamp onto the output time; under LTS the local steps take the same
      // proportional cut so the whole field lands on one pseudo-time level
      if (lts && deltaT > 0) dtScale = (tStep - t)/deltaT;
      deltaT = tStep - t;
    }

    // Also sample on the LAST iteration of this call: the field dumps that
    // follow read residCell by block index, and adaptation (sortBlocks) has
    // renumbered the blocks any number of times since the last cadence sample,
    // so a stale sample paints the residual on the wrong cells.
    const bool lastOfCall  = (t + deltaT >= tStep);
    const bool residSample = (residEvery > 0 && (iter % residEvery == 0 || lastOfCall));
    if (residSample) snapshotResidualQ();

    if (mdFlux == 2) {
      // CTU-Hancock: fully-discrete predictor-corrector.  The corrector is
      // FUSED into multiDRhsKernel (it updates q in place, conservative), so
      // there is no primitiveToConservative/updateFields here; the shared
      // bank holds the half-step predicted primitives during the RHS.
      conservativeToPrimitive();
      setBoundaryConditions(0, 1);
      if (recon == 5) computeShockSensor();
      computeRightHandSide();
      setBoundaryConditions();           // the fused corrector left q CONSERVATIVE
      if (nLvls > 1) {
        restrictFields();
        interpolateFields();
        setBoundaryConditions();
      }
    }
    else
    for (i32 stage = 0; stage < nRkStages; stage++) {
      conservativeToPrimitive();
#ifdef USE_MGPU
      haloExchange(0, NEVOLVE);   // ghosts get owners' primitives (+G) before the RHS reads them
#endif
      setBoundaryConditions(0, 1);   // AFTER halo: periodic exterior ghosts copy the freshly-haloed wrap image
#ifdef USE_MGPU
      if (dbgChecks && stage == 0) {   // debug: near-vacuum states entering the flux = data-protocol bug
        unsigned long long z=0; cudaMemcpyToSymbol(g_vacOwned,&z,sizeof(z)); cudaMemcpyToSymbol(g_vacGhost,&z,sizeof(z));
        dbgVacKernel<<<cudaGridSize,cudaBlockSize>>>(*this); cudaDeviceSynchronize();
        unsigned long long vo=0,vg=0; cudaMemcpyFromSymbol(&vo,g_vacOwned,sizeof(vo)); cudaMemcpyFromSymbol(&vg,g_vacGhost,sizeof(vg));
        double vod=(double)vo, vgd=(double)vg; comm::allreduceSum(&vod,1); comm::allreduceSum(&vgd,1);
        if ((vod > 0 || vgd > 0) && part.rank==0)
          printf("[vac] iter %d pre-rhs vacuum-owned=%.0f vacuum-ghost=%.0f\n", iter, vod, vgd);
      }
#endif
      if (ffVortex && stage == 0 && (iter % ffEvery) == 0) updateFarFieldVortex();
      // --cutpi advances every live cell instead, so there is nothing to
      // reconstruct: dropping this is what makes the fixed point conservative.
      // A CUT-CELL scheme has no ghost states: every face flux is built from the
      // stored apertures and the wall flux from the discrete-GCL normal, and the
      // dead-cell stencil guards (inboard remap for tvdRec, dead-tap skips in the
      // recon-6 gradient) are already unconditional under ibRccm.  Ghost-filling
      // under --cutpi OVERWROTE live cut cells after they had been advanced by
      // their true flux, which is exactly the conservation leak that shows up in
      // the closed-box test.  Never ghost-fill when the cut geometry is active.
      else if (immerserdBcType && !ibRccm && !rans) applyWallGhosts();   // Euler: slip ghosts
      if (shash >= 2 && iter >= shashFrom && iter <= shashTo) stateHash(stage?"ghostS12":"ghost", iter);
      if (rans) {
        if (wallGeom || immerserdBcType) applyWallGhosts();
        computeTurbClosure();
#ifdef USE_MGPU
        haloExchange(F_MUT, 2);     // ghosts need mu_t and F1 on both sides of a seam face
#endif
      }
      if (recon == 5) computeShockSensor();
      computeRightHandSide();
      if (shash >= 2 && iter >= shashFrom && iter <= shashTo) stateHash(stage?"rhsS12":"rhs", iter);
      primitiveToConservative();
      if (srdOn && (srdPerStage || stage == 0)) srdSnapshot();   // U^n for the UM-SRD indicator
      updateFields(stage);
      if (srdOn && (srdPerStage || stage == nRkStages-1))
        applySrd();            // state redistribution: conservative sliver fix
      setBoundaryConditions();
      if (shash >= 2 && iter >= shashFrom && iter <= shashTo) stateHash(stage?"updS12":"upd", iter);


      // --leaf: no interior ghost rims and idle parents -- nothing to restrict or
      // interpolate per stage; the level jumps were fluxed by mortarFluxKernel
      if (nLvls > 1 && !leafFlux) {
        restrictFields();
#ifdef USE_MGPU
        haloExchange(0, NEVOLVE);   // refresh ghosts after the coarse/fine reconstruction
#endif
        interpolateFields();
        setBoundaryConditions();
        if (shash >= 2 && iter >= shashFrom && iter <= shashTo) stateHash(stage?"interpS12":"interp", iter);
        // The coarse/fine reconstruction has just overwritten fine halo cells
        // from their PARENTS, and a parent that overlaps the body holds immersed
        // GHOST data (a mirror built at the COARSE cell size).  Interpolating
        // that into fine cells puts a wrong near-wall state on the fine level,
        // and the error grows with every extra level -- which is why nLvls 6 was
        // stable and nLvls 7 was not.  Re-impose the immersed ghosts here, at
        // the level they belong to, before the state is used again.
        if (immerserdBcType && !ibRccm) applyWallGhosts();
      }
    }
    cudaDeviceSynchronize();
    if (residSample) {
      // dq/dt over EVERY fluid cell -- the residual of the whole update.
      resid = computeResidual();
      if (resid0 <= 0) resid0 = resid;
      // No time printed: step()'s t is segment-local, and under --lts the global
      // clock is not physical -- iteration is the meaningful convergence axis.
      printf("[resid] iter %7d  R = %.6e  R/R0 = %.3e   R(>%gh from wall) = %.3e"
             "   max = %.3e at %.1f h from wall\n",
             iter, (double)resid, (double)(resid0 > 0 ? resid/resid0 : 0),
             (double)resFar, (double)residFar, (double)residMax, (double)residMaxDw);
    }
    if (dbgChecks >= 2) scanNonFinite("rkStages");
    clock.tock();
    tSolver += clock.duration().count();

    if (shash && iter >= shashFrom && iter <= shashTo) stateHash("step", iter);

    t += deltaT;
    iter++;
  }

  // pseudo2D ran only the k=0 plane; refresh the rest before any host-side
  // reader (paint samples k = blockSize/2, writeSolution/error norms download
  // whole blocks).  Once per step(), not per substep -- nothing inside a step
  // reads k != 0.
  broadcastZ();

  return t;
}

void CompressibleSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  // --p1: the slope DOFs and the new-block marker are block payload too
  if (p1)
    for (i32 s = 0; s <= 2*P1_NV; s++) {
      const i32 f = (s < 2*P1_NV) ? F_P1S + s : F_P1NEW;
      copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f, F_SCRATCH);
      gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, f);
    }
  // The geometry cache is block payload: carry it through the sort with the
  // flow variables (staged through F_SCRATCH -- the F_OLD bank has exactly
  // NEVOLVE slots).  Same-stream launches, so no explicit sync is needed
  // between each snapshot and its gather.
  if (immerserdBcType != 0) {
    copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_PHI, F_SCRATCH);
    gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, F_PHI);
    copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_IBM, F_SCRATCH);
    gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, F_IBM);
    if (ibRccm)
      for (i32 f : {F_CUTA, F_CUTAX, F_CUTAY, F_CUTCX, F_CUTCY, F_CUTNX, F_CUTNY, F_CUTTX, F_CUTTY}) {
        copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f, F_SCRATCH);
        gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, f);
      }
    if (ibBrink)
      for (i32 f : {F_BRINKX, F_BRINKY}) {
        copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, f, F_SCRATCH);
        gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, f);
      }
  }
}

void CompressibleSolver::setInitialConditions(void) {
  setInitialConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  broadcastZ();          // the IC kernel is cell-looped, so it filled k=0 only
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(i32 fOff, i32 prim) {
  setBoundaryConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, fOff, prim);
}

static i32 g_convDbg = 0;
static void convCoverage(CompressibleSolver &S, const char *tag) {
  cudaDeviceSynchronize();
  real *Pp = S.getField(F_RHOE); i32 nPrim = 0, nCons = 0, nOther = 0, firstPrim = -1, lastPrim = -1, firstCons = -1, lastCons = -1;
  for (i32 b = 0; b < S.hashTable.nKeys; b++) { if (S.bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; S.decode(S.bLocList[b], lvl, ib, jb, kb); if (!S.isInteriorBlock(lvl, ib, jb, kb)) continue;
    for (i32 cc = 0; cc < blockSize*blockSize; cc++) { const i32 c = b*blockSizeTot + cc; if (S.cFlagsList[c] != ACTIVE) continue;
      const double v = Pp[c];
      if (fabs(v - 7.9365079365) < 1e-4) { nPrim++; if (firstPrim < 0) firstPrim = b; lastPrim = b; }
      else if (fabs(v - 20.3412698413) < 1e-3) { nCons++; if (firstCons < 0) firstCons = b; lastCons = b; }
      else nOther++; } }
  printf("[convdbg] %-14s prim %5d (blocks %d..%d)  cons %5d (blocks %d..%d)  other %d   nLeaf %d nExt %d\n", tag, nPrim, firstPrim, lastPrim, nCons, firstCons, lastCons, nOther, S.nLeafBlocks, S.nExtBlocks);
}

void CompressibleSolver::conservativeToPrimitive(void) {
  if (cutDbg && g_convDbg < 6) convCoverage(*this, "before c2p");
  conservativeToPrimitiveKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  if (cutDbg && g_convDbg < 6) { convCoverage(*this, "after c2p"); g_convDbg++; }
}

void CompressibleSolver::primitiveToConservative(void) {
  if (cutDbg && g_convDbg < 6) convCoverage(*this, "before p2c");
  primitiveToConservativeKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  if (cutDbg && g_convDbg < 6) { convCoverage(*this, "after p2c"); g_convDbg++; }
}

void CompressibleSolver::forwardWaveletTransform(void) {
  // thresholding scales: domain maxima of {|rho|, |mom|, |rhoE|, |grad|},
  // reduced entirely device-side and stream-ordered -- no host round-trip.
  // cudaMemset is construction-only (it also serializes the stream); zero by
  // kernel so the launch queues like everything else.
  zeroScalesKernel<<<1, 32>>>(*this);
  computeGlobalScalesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
#ifdef USE_MGPU
  // the wavelet threshold must use the SAME normalization on every PE, else
  // partitions refine against inconsistent scales -> take the domain-wide max.
  cudaDeviceSynchronize();
  comm::allreduceMax(globalScale, 3);
#endif

  zeroFlagsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  // The face-flux scatter atomicAdds missing-neighbor contributions into the
  // bEmpty trash block's slice of the ACCUMULATOR bank -- which is aliased
  // with this snapshot bank.  bEmpty sits outside the cell loop, so neither
  // the S-roll in updateFields nor copyToOld ever cleans it (and NaN*0 = NaN
  // defeats the *=0 roll anyway).  waveletPredict reads the snapshot through
  // the same missing-neighbor path, so clear the trash before it is read.
  for (i32 f = 0; f < NEVOLVE; f++) {
    zeroTrashBlockKernel<<<1, blockSizeTot>>>(*this, F_OLD + f);
    // live-field trash slice too: interpolate/restrict (and a missing-parent
    // prediction via bEmpty's self-pointing neighbor row) read live fields
    // through parent taps, and nothing else ever cleans this slice.
    zeroTrashBlockKernel<<<1, blockSizeTot>>>(*this, f);
  }
  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  forwardWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  waveletThresholdingKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::inverseWaveletTransform(void) {
  inverseWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

#ifdef USE_MGPU
// Reconstitute the F_OLD snapshot for every block created this cycle, before the
// inverse reads it.  Dense levels 0-1 already carry a valid snapshot (copyToOld);
// process finer levels coarse->fine so each new block's parent is already valid:
// fill OWNED new blocks by wavelet-predicting from the parent (fillOldSnapshot),
// then halo so the neighbours' ghosts pick up the owners' freshly-filled snapshot
// before that level becomes a parent.  This replaces the persistent (blanket-kept)
// ghost layer as the source of the inverse's coarse-parent data.
void CompressibleSolver::reconstituteOldSnapshot(void) {
  haloExchange(F_OLD, NEVOLVE);                 // ghosts of the dense levels get owners' snapshot
  setBoundaryConditions(F_OLD);                 // exterior taps must be BC-filled BEFORE prolongation reads them
  for (i32 L = 1; L < nLvls; L++) {             // L=1: level 1 is adaptive now, new level-1 blocks need F_OLD prolonged from the dense level 0
    if (dbgChecks) {                            // debug: every fill target must have valid parent support
      cudaDeviceSynchronize(); cudaMemset(dbgCnt, 0, sizeof(i32));
      checkFillSupportKernel<<<cudaGridSize, cudaBlockSize>>>(*this, L);
      cudaDeviceSynchronize();
      double v = (double)dbgCnt[0]; comm::allreduceSum(&v, 1);
      if (v > 0 && part.rank == 0) printf("[fillchk] level %d: %.0f SUPPORT VIOLATIONS\n", L, v);
    }
    fillOldSnapshotKernel<<<cudaGridSize, cudaBlockSize>>>(*this, L);
    haloExchange(F_OLD, NEVOLVE);               // owners' level-L fill -> neighbours' level-L ghosts
    setBoundaryConditions(F_OLD);               // refresh exterior images of the freshly-filled level
  }
}
#endif

void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  real *dmin = thrust::min_element(thrust::device, getField(F_SCRATCH), getField(F_SCRATCH)+hashTable.nKeys*blockSizeTot);
#ifdef USE_MGPU
  // fieldData may be device-only (symmetric heap): copy the min to host rather
  // than dereferencing on the host, then take the min across all PEs.
  real localMin;
  cudaMemcpy(&localMin, dmin, sizeof(real), cudaMemcpyDefault);
  comm::allreduceMin(&localMin, 1);
  deltaT = localMin * cfl;
#else
  // NEVER dereference managed memory from the host in the hot path: *dmin
  // touches a page of the 232 MB fieldData allocation and MIGRATES it
  // CPU-ward, so the next kernel that reads that field faults it back.
  // Profiled at 727 us per sync.  An explicit D2H copy of 4 bytes moves no
  // pages (this is what the MGPU branch above always did).
  real localMin;
  cudaMemcpy(&localMin, dmin, sizeof(real), cudaMemcpyDeviceToHost);
  deltaT = localMin * cfl;
#endif
  // dt-dip forensics (--dtdip T): the inclined-plate runs show a limit cycle
  // whose only host-visible trace is the stable step collapsing 10-100x for a
  // few hundred steps and recovering -- always BETWEEN outputs, so the --debug
  // argmin report (which runs at output time) always sees a healthy cell.
  // Catch it here, where every dt reduction lands.  F_SCRATCH still holds the
  // per-cell DeltaT this deltaT was reduced from.
  if (envCheck) envCheckStep();   // piggyback on this call's existing sync
  if (dtDipThresh > 0 && deltaT < dtDipThresh && dtDipPrints < 60) {
    if (dtDipCooldown > 0) { dtDipCooldown--; }
    else { dtDipCooldown = 25; dtDipPrints++; reportDtMinCell("dip"); }
  }
  if (lts) {
    stampLocalDtKernel<<<cudaGridSize, cudaBlockSize>>>(*this, deltaT, ltsRatio*deltaT);
    broadcastZ(F_DTL);      // pseudo2D: k>0 is stale after a cell-looped write
  }
}

// Locate the cell that owns the current F_SCRATCH dt minimum and print enough
// of its state to identify WHICH limit is biting: the acoustic, viscous and
// tau~ candidate steps are all recomputed from the cell state and printed next
// to the actual DeltaT -- whichever matches is the limiter.  Host-side only
// (managed memory), so keep it OUT of the hot path: callers gate it.
void CompressibleSolver::reportDtMinCell(const char *tag) {
  real *F = getField(F_SCRATCH);
  const size_t n = (size_t)hashTable.nKeys*blockSizeTot;
  real *dmin = thrust::min_element(thrust::device, F, F+n);
  const size_t idx = (size_t)(dmin - F);
  const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
  u64 loc = bLocList[b];
  i32 lvl=-1, ib=0, jb=0, kb=0;
  if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
  const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize;
  const double dxL = (lvl >= 0) ? domainSize[0]/(baseGridSize[0]*powi(2,lvl)) : 1.0;
  const double dyL = (lvl >= 0) ? domainSize[1]/(baseGridSize[1]*powi(2,lvl)) : 1.0;
  const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
  const double r  = fmax((double)getField(F_RHO)[idx], 1e-30);
  const double uu = (double)getField(F_RHOU)[idx]/r, vv = (double)getField(F_RHOV)[idx]/r;
  const double ke = 0.5*r*(uu*uu+vv*vv);
  const double p  = ((double)gam-1.0)*((double)getField(F_RHOE)[idx] - ke);
  const double cs = sqrt(fmax((double)gam*p/r, 0.0));
  const double tt = (double)getField(F_RHOTAU)[idx]/r;
  const double mt = (double)getField(F_MUT)[idx];
  const double nuE= ((double)mu + fmax(mt,0.0))/r;
  const double hL = fmin(dxL, dyL);
  const double dtAc  = hL/(sqrt(uu*uu+vv*vv)+cs+1e-32);
  const double dtVis = hL*hL/(4.0*nuE + 1e-32);
  const double dtTau = tt/0.09;
  const double ls = (immerserdBcType != 0) ? (double)getBoundaryLevelSet(Vec3((real)x,(real)y,(real)0)) : -1.0;
  printf("  [dtdip:%s] lvl=%d x=%.5f y=%.5f  DeltaT=%.3e | acoustic=%.3e visc=%.3e tau=%.3e | rho=%.4e u=%+.4e v=%+.4e p=%.4e tau~=%.3e muT/mu=%.3e phiLS=%+.4e\n",
         tag, lvl, x, y, (double)F[idx], dtAc, dtVis, dtTau,
         r, uu, vv, p, tt, mt/fmax((double)mu,1e-30), ls);
}

// RANS: fill F_MUT / F_TF1 for the face loop and accumulate the cell-local k~ and
// tau~ sources.  Must run AFTER setBoundaryConditions (it reads the exterior
// primitives) and BEFORE computeRightHandSide (which reads what it writes).
// Uniform freestream box (testCase 10): with no gradients and no wall, F1 = 0 and
// the model collapses to a pair of ODEs whose solution is known in closed form,
//   dk~/dt = -beta* k~/tau~,   dtau~/dt = beta2
//   =>  tau~ = tau0 + beta2 t,   k~ = k0 (tau~/tau0)^(-beta*/beta2)
// and with the Eq. (32) sustaining terms ON the two source pairs cancel EXACTLY,
// so the state must be frozen.  Either way the convective and diffusive fluxes
// must contribute exactly nothing, which is what makes this a free-stream
// preservation test for them as well.
void CompressibleSolver::computeRansDecayError(real t) {
  real kEx, tEx;
  if (ransSustain) { kEx = kInf; tEx = tauInf; }
  else {
    tEx = tauInf + ktau::beta2*t;
    kEx = kInf*pow(tEx/tauInf, -ktau::betaStar/ktau::beta2);
  }
  const size_t n = (size_t)hashTable.nKeys*blockSizeTot;
  real *F = getField(F_SCRATCH);

  // (1) uniformity: the fluxes must cancel EXACTLY on a uniform state
  real kHi, kLo;
  ransDecayErrorKernel<<<cudaGridSize, cudaBlockSize>>>(*this, kEx, tEx, 1);
  cudaDeviceSynchronize();
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  cudaMemcpy(&kHi, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
  ransDecayErrorKernel<<<cudaGridSize, cudaBlockSize>>>(*this, kEx, tEx, 2);
  cudaDeviceSynchronize();
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  cudaMemcpy(&kLo, thrust::min_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
  const real spread = (kHi - kLo)/fmax(fabs(kHi), (real)1e-300);

  // (2) accuracy vs the exact ODE solution
  ransDecayErrorKernel<<<cudaGridSize, cudaBlockSize>>>(*this, kEx, tEx, 0);
  cudaDeviceSynchronize();
  real emax;
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  cudaMemcpy(&emax, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);

  printf("---- RANS uniform-box source gate (sustain=%d, t=%g) ----\n", ransSustain, (double)t);
  printf("  exact  k~ = %.12e   tau~ = %.12e\n", (double)kEx, (double)tEx);
  printf("  k~ range [%.12e, %.12e]\n", (double)kLo, (double)kHi);
  printf("  k~ spread across the domain        = %.6e   %s   (flux cancellation)\n",
         (double)spread, (spread < (real)1e-12) ? "ok" : "FAIL");
  // With the sustaining terms on, the two source pairs cancel term by term and
  // the state is a fixed point of the ODE -- so this must be exact.  With them
  // off it is a genuine time integration and the error is RK3's: O(dt^3),
  // measured at 3.06 over an 8x cfl sweep, so the bound here is cfl-dependent.
  const real tol = ransSustain ? (real)1e-12 : (real)1e-4;
  printf("  max relative error vs the exact ODE = %.6e   %s%s\n", (double)emax,
         (emax < tol) ? "ok" : "FAIL",
         ransSustain ? "   (exact fixed point)" : "   (RK3-limited; refine --cfl for 3rd order)");
}

// Frozen-shear production probe (testCase 11).  Builds ONE right-hand side and
// compares its k~ component against the same source evaluated with the analytic
// vorticity -- isolating the discrete S/Omega stencil, which the uniform box
// cannot exercise because Omega is identically zero there.
void CompressibleSolver::computeRansShearProbe(void) {
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  zeroAccumulator();
  computeTurbClosure();
  computeRightHandSide();      // must contribute exactly nothing: k~ is uniform
  cudaDeviceSynchronize();
  const real ky = (real)(2.0*3.14159265358979323846)/domainSize[1];
  ransShearProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, vortexAdvect, ky);
  cudaDeviceSynchronize();
  const size_t n = (size_t)hashTable.nKeys*blockSizeTot;
  real *F = getField(F_SCRATCH);
  real emax;
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  cudaMemcpy(&emax, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
  printf("---- RANS frozen-shear production probe (u0 = %g) ----\n", (double)vortexAdvect);
  printf("  cells = %d   dy = %.6e\n", hashTable.nKeys*blockSizeTot,
         (double)(domainSize[1]/(baseGridSize[1]*powi(2, nLvls-1))));
  printf("  max |Rhs(k~) - source(exact Omega)| / (beta* rho k~/tau~) = %.6e\n", (double)emax);
}

// Near-wall equilibrium probe (testCase 12).  One right-hand side on the analytic
// similarity solution: both equations must balance, which puts the Appendix-A
// tau~ diffusion -- its L/R flux pair, its scatter signs and its face
// coefficients -- under test THROUGH the solver, not just in ktau_test.
void CompressibleSolver::computeRansWallProbe(void) {
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  zeroAccumulator();
  if (wallGeom) applyWallGhosts();
  computeTurbClosure();
  const size_t n = (size_t)hashTable.nKeys*blockSizeTot;
  real *F = getField(F_SCRATCH);
  // no zeroing: wallUtauKernel assigns Sc[cIdx] for EVERY cell (0 by default)
  computeRightHandSide();
  cudaDeviceSynchronize();
  printf("---- RANS near-wall equilibrium probe (u_tau = %g, nu = %g) ----\n",
         (double)vortexAdvect, (double)mu);
  if (wallGeom == 1) {
    real utMax;
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    cudaMemcpy(&utMax, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
    printf("  wall model: u_tau recovered = %.8e   input = %.8e   rel err = %.3e\n",
           (double)utMax, (double)vortexAdvect,
           (double)(fabs(utMax - vortexAdvect)/fmax(vortexAdvect,(real)1e-30)));
  }
  printf("  dy = %.4e   y+ per cell = %.2f\n",
         (double)(domainSize[1]/(baseGridSize[1]*powi(2, nLvls-1))),
         (double)(domainSize[1]/(baseGridSize[1]*powi(2, nLvls-1))*vortexAdvect/mu));
  const real bands[3] = {(real)30, (real)300, (real)2000};
  for (i32 b = 0; b < 3; b++) {
    real ek, et;
    ransWallProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, vortexAdvect, bands[b], 0);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    cudaMemcpy(&ek, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
    ransWallProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, vortexAdvect, bands[b], 1);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    cudaMemcpy(&et, thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
    printf("  y+ > %-5.0f   |Rhs(k~)|/(beta* rho k~/tau~) = %.4e   |Rhs(tau~)|/(beta1 rho) = %.4e\n",
           (double)bands[b], (double)ek, (double)et);
  }
}

// Overwrite the wall ghost rows with the wall-model linear profile.  Must run
// AFTER setBoundaryConditions (whose generic mirror it replaces) and BEFORE
// computeTurbClosure (whose gradients read it).
// Skin friction along the modeled wall.  The wall model already computes u_tau
// at every wall face and stamps it into F_SCRATCH, so C_f = tau_w/(q_inf) =
// 2 (u_tau/u_inf)^2 for rho_w ~ rho_inf.  Dumped as x, C_f for plotting against
// the TMR reference (C_f ~ 0.0027 at x/L = 0.97 on the flat plate).
// The wall model only acts on finest-level cells (see wallFineBand).  Any wall
// block left coarse therefore silently reverts to a slip wall, so count them and
// say so rather than let the run look healthy while a stretch of plate is inert.
i32 CompressibleSolver::wallResolutionCheck(bool verbose) {
  if (!rans || wallGeom != 1) return 0;
  // This is a MULTI-LEVEL scheme: a coarse wall block whose children exist is a
  // legitimate parent, restricted from them every stage (restrictFieldsKernel).
  // Only a coarse LEAF is a problem -- there the wall model is genuinely absent.
  // So membership of the child block decides, not the level alone.
  std::unordered_set<u64> live;
  live.reserve((size_t)hashTable.nKeys*2);
  for (i32 b = 0; b < hashTable.nKeys; b++)
    if (bLocList[b] != kEmpty) live.insert(bLocList[b]);

  i32 nCoarse = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (jb != 0 || lvl >= nLvls-1) continue;                 // not a wall block, or already finest
    if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2, lvl));
    const double xHi = (ib + 1)*blockSize*dxL;
    if (xHi <= (double)plateX0) continue;                    // upstream of the plate
    const i32 kc = pseudo2D ? kb : 2*kb;
    if (live.count(encode(lvl+1, 2*ib, 2*jb, kc))) continue; // has a child: a parent, not a leaf
    nCoarse++;
  }
  if (nCoarse && verbose)
    printf("[warn] %d wall-row block(s) are COARSER than the finest level: the wall model is "
           "inert there (slip wall).  Raise --wallband (now %g) or --nlvls.\n",
           nCoarse, (double)wallFineBand);
  return nCoarse;
}

__global__ void scanNonFiniteKernel(CompressibleSolver &grid, i32 baseOff);
__global__ void residualNormKernel(CompressibleSolver &grid, real *q0, real dtGlobal, real *rCell);
__global__ void residualSnapshotKernel(CompressibleSolver &grid, real *q0);
extern __device__ double             g_resSum;
extern __device__ unsigned long long g_resCnt;
extern __device__ double             g_resSumFar;
extern __device__ unsigned long long g_resCntFar;
extern __device__ double             g_resMax;
extern __device__ double             g_resMaxPhi;

// RMS of L(q) over live fluid cells.  Call ONLY right after
// computeRightHandSide() on stage 0, where the accumulator is exactly L(q^n).
// Snapshot q before the stages; the compare happens after them.
__global__ void srdBuildKernel(CompressibleSolver &grid);
__global__ void srdCountKernel(CompressibleSolver &grid);
__global__ void srdProjectKernel(CompressibleSolver &grid);
__global__ void srdAverageKernel(CompressibleSolver &grid, real sBlend);
__global__ void srdSnapKernel(CompressibleSolver &grid);
__global__ void srdIndicatorKernel(CompressibleSolver &grid);
__global__ void cutBroadcastKernel(CompressibleSolver &grid);
__global__ void cutCellKernel(CompressibleSolver &grid);
__global__ void mortarFluxKernel(CompressibleSolver &grid);
__global__ void cutPieceUpdateKernel(CompressibleSolver &grid, i32 stage);
extern __device__ double g_srdDU;
extern __device__ unsigned long long g_srdShort;

// Build the merge neighbourhoods.  Cheap and geometry-only, so it is called
// once after the geometry stamp (and would be re-called at the adaptation
// cadence, alongside stampIbGeometry, once AMR is supported).
void CompressibleSolver::buildSrd(void) {
  if (!srdOn) return;
  // Multi-level is supported: srdLive takes ACTIVE cells only, so a neighbourhood
  // never spans a level jump (see the kernel).  A sliver whose only same-level
  // neighbours are themselves small will simply come up short of volFrac and be
  // counted in "never reached the target" below rather than silently merging
  // across levels.  Neighbourhoods are rebuilt here, i.e. at every geometry stamp,
  // which is also every adaptation cycle.
  srdStride = (size_t)nBlocksMax*(size_t)blockSizeTot;
  if (!srdM) {
    cudaMallocManaged(&srdM,  (size_t)SRD_MAXM*srdStride*sizeof(i32));
    cudaMallocManaged(&srdMn, srdStride*sizeof(i32));
    cudaMallocManaged(&srdC,  srdStride*sizeof(i32));
    cudaMallocManaged(&srdPi, (size_t)5*srdStride*sizeof(real));
    cudaMallocManaged(&srdU0, (size_t)5*srdStride*sizeof(real));
    cudaMallocManaged(&srdS,  srdStride*sizeof(real));
    cudaMallocManaged(&srdPi0,(size_t)5*srdStride*sizeof(real));
    cudaMallocManaged(&srdTh, srdStride*sizeof(real));
  }
  unsigned long long z = 0;
  cudaMemcpyToSymbol(g_srdShort, &z, sizeof(z));
  srdBuildKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  srdCountKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&z, g_srdShort, sizeof(z));
  // census: how many cells were merged at all, and the neighbourhood sizes
  i32 nSmall = 0, nMax = 0; double mSum = 0;
  for (size_t c = 0; c < (size_t)hashTable.nKeys*blockSizeTot; c++) {
    const i32 n = srdMn[c];
    if (n > 1) { nSmall++; mSum += n; if (n > nMax) nMax = n; }
  }
  printf("[srd] reach %d, volFrac %.2f: %d cells merged (mean |M| %.2f, max %d), %llu never reached the target\n",
         srdReach, (double)srdVolFrac, nSmall, nSmall ? mSum/nSmall : 0.0, nMax, z);
}

void CompressibleSolver::srdSnapshot(void) {
  if (!srdOn || !srdU0) return;
  srdSnapKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

// S applied to the updated state, after every RK stage.
void CompressibleSolver::applySrd(void) {
  if (!srdOn || !srdM) return;
  // UM-SRD indicator.  srdLocal 1 = per-neighbourhood s (the paper); the average
  // kernel then takes s per cell as the max over the neighbourhoods containing it,
  // signalled by passing sBlend < 0.  srdLocal 0 = one global s (exactly
  // conservative under overlap; see the header).
  real sB = (real)-1;
  srdIndicatorKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  if (!srdLocal) {
    cudaDeviceSynchronize();
    double dmax = 0;
    for (size_t c = 0; c < (size_t)hashTable.nKeys*blockSizeTot; c++)
      if (srdMn[c] > 1 && (double)srdS[c] > dmax) dmax = (double)srdS[c];
    sB = (real)dmax;
  }
  srdProjectKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  srdAverageKernel<<<cudaGridSize, cudaBlockSize>>>(*this, sB);
}

void CompressibleSolver::snapshotResidualQ(void) {
  if (!residQ0) cudaMallocManaged(&residQ0, (size_t)4*(size_t)nBlocksMax*(size_t)blockSizeTot*sizeof(real));
  if (!residCell) {
    cudaMallocManaged(&residCell, (size_t)nBlocksMax*(size_t)blockSizeTot*sizeof(real));
    cudaMemset(residCell, 0, (size_t)nBlocksMax*(size_t)blockSizeTot*sizeof(real));
  }
  residualSnapshotKernel<<<cudaGridSize, cudaBlockSize>>>(*this, residQ0);
}

real CompressibleSolver::computeResidual(void) {
  if (!residQ0) return 0;
  double z = 0.0; unsigned long long zc = 0;
  cudaMemcpyToSymbol(g_resSum, &z,  sizeof(double));
  cudaMemcpyToSymbol(g_resCnt, &zc, sizeof(unsigned long long));
  cudaMemcpyToSymbol(g_resSumFar, &z,  sizeof(double));
  cudaMemcpyToSymbol(g_resCntFar, &zc, sizeof(unsigned long long));
  cudaMemcpyToSymbol(g_resMax,    &z,  sizeof(double));
  cudaMemcpyToSymbol(g_resMaxPhi, &z,  sizeof(double));
  residualNormKernel<<<cudaGridSize, cudaBlockSize>>>(*this, residQ0, deltaT, residCell);
  cudaDeviceSynchronize();
  double sum = 0.0; unsigned long long cnt = 0;
  cudaMemcpyFromSymbol(&sum, g_resSum, sizeof(double));
  cudaMemcpyFromSymbol(&cnt, g_resCnt, sizeof(unsigned long long));
  double sumF = 0.0, rmax = 0.0, rphi = 0.0; unsigned long long cntF = 0;
  cudaMemcpyFromSymbol(&sumF, g_resSumFar, sizeof(double));
  cudaMemcpyFromSymbol(&cntF, g_resCntFar, sizeof(unsigned long long));
  cudaMemcpyFromSymbol(&rmax, g_resMax,    sizeof(double));
  cudaMemcpyFromSymbol(&rphi, g_resMaxPhi, sizeof(double));
  residFar = (cntF > 0) ? (real)sqrt(sumF/(double)cntF) : (real)0;
  residMax = (real)sqrt(rmax); residMaxDw = (real)rphi;
  return (cnt > 0) ? (real)sqrt(sum/(double)cnt) : (real)0;
}

extern __device__ int g_nfCnt;
extern __device__ int g_nfCntZ;
extern __device__ int g_nfCidx;
extern __device__ int g_nfField;

// AMR debug probe (--debug 2): report the first non-finite evolved cell after a
// named phase.  Prints once per phase that newly goes bad.
void CompressibleSolver::scanNonFinite(const char *tag) { scanNonFiniteBase(tag, 0); scanNonFiniteBase(tag, F_OLD); }
void CompressibleSolver::scanNonFiniteBase(const char *tag, i32 baseOff) {
  int z = 0, big = INT_MAX;
  cudaMemcpyToSymbol(g_nfCnt,  &z,   sizeof(int));
  cudaMemcpyToSymbol(g_nfCntZ, &z,   sizeof(int));
  cudaMemcpyToSymbol(g_nfCidx, &big, sizeof(int));
  cudaMemcpyToSymbol(g_nfField,&z,   sizeof(int));
  scanNonFiniteKernel<<<cudaGridSize, cudaBlockSize>>>(*this, baseOff);
  cudaDeviceSynchronize();
  int n = 0, nz = 0, ci = 0, fld = 0;
  cudaMemcpyFromSymbol(&n,  g_nfCnt,  sizeof(int));
  cudaMemcpyFromSymbol(&nz, g_nfCntZ, sizeof(int));
  cudaMemcpyFromSymbol(&ci, g_nfCidx, sizeof(int));
  cudaMemcpyFromSymbol(&fld,g_nfField,sizeof(int));
  if (n == 0 && nz == 0) return;
  i32 bIdx = ci / blockSizeTot, idx = ci % blockSizeTot;
  i32 lvl = -1, ib = 0, jb = 0, kb = 0;
  u64 loc = bLocList[bIdx];
  decode(loc, lvl, ib, jb, kb);
  printf("[nf] iter %d  AFTER %-22s  %s k0=%d kz=%d  first: f=%d lvl=%d blk(%d,%d,%d) "
         "cell(%d,%d,%d) %s\n", iter, tag, baseOff ? "F_OLD" : "LIVE ", n, nz, fld, lvl, ib, jb, kb,
         idx % blockSize, (idx / blockSize) % blockSize, idx / blockSize / blockSize,
         isInteriorBlock(lvl, ib, jb, kb) ? "interior" : "EXTERIOR");
}

void CompressibleSolver::printRansExtremes(void) {
  if (immerserdBcType != 0) {
    unsigned long long z = 0, det = 0, flx = 0, tnt = 0;
    cudaMemcpyToSymbol(g_ibDetect, &z, sizeof(z));
    cudaMemcpyToSymbol(g_ibFailDip, &z, sizeof(z));
    cudaMemcpyToSymbol(g_ibFailSlip, &z, sizeof(z));
    cudaMemcpyToSymbol(g_ibFailIp, &z, sizeof(z));
    cudaMemcpyToSymbol(g_ibNup, &z, sizeof(z));
    { double dz = 0; cudaMemcpyToSymbol(g_ibMaxDfc, &dz, sizeof(dz));
      cudaMemcpyToSymbol(g_ibMaxLvl, &dz, sizeof(dz)); }
    cudaMemcpyToSymbol(g_ibFlux,   &z, sizeof(z));
    cudaMemcpyToSymbol(g_ipTaint,  &z, sizeof(z));
    conservativeToPrimitive();
    setBoundaryConditions(0, 1);
    computeTurbClosure();
    zeroAccumulator();
    computeRightHandSide();   // NOT the bare kernel: under the default --detflux
                              // the kernel only stages face fluxes and the
                              // gather is what fills F_RHS.  Launching the
                              // kernel alone made this census report max|Rhs|=0
                              // always.
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&det, g_ibDetect, sizeof(det));
    { unsigned long long fd=0,fs=0,fi=0,nu=0;
      cudaMemcpyFromSymbol(&fd, g_ibFailDip,  sizeof(fd));
      cudaMemcpyFromSymbol(&fs, g_ibFailSlip, sizeof(fs));
      cudaMemcpyFromSymbol(&fi, g_ibFailIp,   sizeof(fi));
      cudaMemcpyFromSymbol(&nu, g_ibNup,      sizeof(nu));
      { unsigned long long wg=0, wc=0;
        cudaMemcpyFromSymbol(&wg, g_wmGhost, sizeof(wg));
        cudaMemcpyFromSymbol(&wc, g_wmCand,  sizeof(wc));
        printf("  [rans] ghost wall function (cumulative): candidates=%llu applied=%llu\n", wc, wg); }
      printf("  [rans] ibWallFlux fail census: dIp<=dFc=%llu slipSample=%llu ipSample=%llu"
             "   (faces with body ABOVE: %llu)\n", fd, fs, fi, nu);
      double mdf=0, mlv=0;
      cudaMemcpyFromSymbol(&mdf, g_ibMaxDfc, sizeof(mdf));
      cudaMemcpyFromSymbol(&mlv, g_ibMaxLvl, sizeof(mlv));
      printf("  [rans]   worst dFc/h among failures = %.2f  at level %.0f\n", mdf, mlv); }
    cudaMemcpyFromSymbol(&flx, g_ibFlux,   sizeof(flx));
    cudaMemcpyFromSymbol(&tnt, g_ipTaint,  sizeof(tnt));
    // MUST be zero: a nonzero count means the image point is interpolating from
    // a cell the wall model itself writes, which is the feedback loop ipStandMin
    // exists to prevent.  Only counted under --debug.
    if (dbgChecks)
      printf("  [rans] IP stencil taps reading wall-degraded cells = %llu  %s\n",
             tnt, tnt ? "<-- FEEDBACK LOOP" : "(clean)");
    // is the momentum RHS actually nonzero at the wall?
    const size_t nn = (size_t)hashTable.nKeys*blockSizeTot;
    real *RU = getField(F_RHS + F_RHOU);
    real mx = 0; size_t mi = 0;
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      u64 lc = bLocList[b]; if (lc == kEmpty) continue;
      i32 L, I, J, Kb; decode(lc, L, I, J, Kb);
      if (!isInteriorBlock(L, I, J, Kb)) continue;               // interior only
      const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,L));
      const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,L));
      for (i32 c = 0; c < blockSizeTot; c++) {
        const i32 jj = (c/blockSize)%blockSize, ii = c%blockSize;
        const double yy = (J*blockSize+jj+0.5)*dyL;
        const double xx = (I*blockSize+ii+0.5)*dxL;
        // FLUID cells only, by the actual mask.  The old test (yy < ibPlane)
        // let the wall-coincident GHOST row through -- and at d_FC = 0.5h that
        // row's centre sits exactly ON the wall, so the diagnostic reported a
        // cell whose residual is discarded by the update mask anyway.
        if (!isFluidCell(Vec3((real)xx,(real)yy,(real)0), (real)fmin(dxL,dyL))) continue;
        const size_t t = (size_t)b*blockSizeTot + c;
        const real a = fabs(RU[t]); if (a > mx) { mx = a; mi = t; }
      }
    }
    const i32 mb = (i32)(mi/blockSizeTot), mc = (i32)(mi%blockSizeTot);
    i32 ml=-1, mib=0, mjb=0, mkb=0; u64 mloc = bLocList[mb];
    if (mloc != kEmpty) decode(mloc, ml, mib, mjb, mkb);
    real utMax = 0;
    for (size_t t = 0; t < nn; t++) utMax = fmax(utMax, getField(F_SCRATCH)[t]);
    printf("  [rans] IB u_tau(max, from the RHS) = %.6e   tau_w = %.4e\n",
           (double)utMax, (double)(utMax*utMax));
    {
      // WHERE it blows up, not just how big: x and y of the worst residual, its
      // wall distance in cells, and its row index above the surface.  The plain
      // "NaN cell" report elsewhere is first-in-scan-order and therefore useless
      // once the field is globally bad -- this one is a genuine argmax.
      const double mdx = (ml>=0) ? domainSize[0]/(baseGridSize[0]*powi(2,ml)) : 0;
      const double mdy = (ml>=0) ? domainSize[1]/(baseGridSize[1]*powi(2,ml)) : 0;
      const double mX = (ml>=0) ? (mib*blockSize + (mc%blockSize) + 0.5)*mdx : -1.0;
      const double mY = (ml>=0) ? (mjb*blockSize + (mc/blockSize)%blockSize + 0.5)*mdy : -1.0;
      const double mD = (ml>=0) ? (double)wallDistance(Vec3((real)mX,(real)mY,(real)0)) : -1.0;
      printf("  [rans] IB y-faces detected = %llu   flux applied = %llu\n", det, flx);
      printf("  [rans] max|Rhs(rhoU)| = %.4e  AT x=%.5f y=%.6f  (d=%.2f h, row %.0f above the wall,"
             " x/L from LE = %+.4f)\n",
             (double)mx, mX, mY, (mdy>0)? mD/mdy : -1.0,
             (mdy>0)? (mY-(double)ibPlane)/mdy - 0.5 : -1.0,
             mX - (double)plateX0);
    }
    zeroAccumulator();
    primitiveToConservative();
  }
  const size_t n = (size_t)hashTable.nKeys*blockSizeTot;
  real *F = getField(F_SCRATCH);
  real v[10];
  size_t vIdx[10];
  for (i32 w = 0; w < 10; w++) {
    ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, w);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    real *mp = thrust::max_element(thrust::device, F, F+n);
    vIdx[w] = (size_t)(mp - F);      // argmax, so the extreme can be LOCATED
    cudaMemcpy(&v[w], mp, sizeof(real), cudaMemcpyDefault);
  }
  // WHERE the turbulence extremes sit.  A magnitude alone cannot distinguish a
  // wake (physical), a stagnation point (the Eq. 39 u_tau -> 0 branch) and a
  // far-field cell that has lost its sustaining terms -- and those want opposite
  // fixes.  Reports level, position, and wall distance in local cells.
  auto locate = [&](const char *tag, size_t idx, double val) {
    const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
    if (b < 0 || b >= hashTable.nKeys) { printf("  [rans] %s: (unlocatable)\n", tag); return; }
    u64 loc = bLocList[b];
    if (loc == kEmpty) { printf("  [rans] %s: (empty block)\n", tag); return; }
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    const double ddx = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double ddy = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const double X = (ib*blockSize + (c%blockSize) + 0.5)*ddx;
    const double Y = (jb*blockSize + (c/blockSize)%blockSize + 0.5)*ddy;
    const double D = (double)wallDistance(Vec3((real)X,(real)Y,(real)0));
    printf("  [rans] %s = %.4e  AT x=%.4f y=%.4f  lvl=%d  d=%.4f (%.1f local cells)"
           "  fluid=%d\n", tag, val, X, Y, lvl, D, D/fmin(ddx,ddy),
           isFluidCell(Vec3((real)X,(real)Y,(real)0), (real)fmin(ddx,ddy)) ? 1 : 0);
  };
  locate("max k~   ", vIdx[0], (double)v[0]);
  locate("max tau~ ", vIdx[1], (double)v[1]);
  locate("max mu_t/mu", vIdx[3], (double)v[3]);
  printf("  [rans] max k~ = %.3e  tau~ in [%.3e, %.3e]  max mu_t/mu = %.3e  dt_rans = %.3e\n",
         (double)v[0], (double)(-v[2]), (double)v[1], (double)v[3], (double)(-v[4]));
  printf("  [rans] max sound speed = %.4e (freestream %.4e)   max rho = %.4e\n",
         (double)v[5], sqrt((double)(gam*fsP)), (double)v[6]);
  printf("  [rans] max |rhoV|: finest level = %.4e    coarser levels = %.4e   non-finite cells: %s\n",
         (double)v[7], (double)v[8], (v[9] > 0) ? "YES" : "none");

  // Locate the largest wall-normal momentum (mode 7): on a flat plate v should
  // be tiny, so where it is not says which face is misbehaving.
  {
    ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, 7);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    real *mx = thrust::max_element(thrust::device, F, F+n);
    const size_t idx = (size_t)(mx - F);
    const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
    u64 loc = bLocList[b];
    i32 lvl=-1, ib=0, jb=0, kb=0;
    if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
    const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize;
    const double dxL = (lvl>=0) ? domainSize[0]/(baseGridSize[0]*powi(2,lvl)) : 0;
    const double dyL = (lvl>=0) ? domainSize[1]/(baseGridSize[1]*powi(2,lvl)) : 0;
    printf("  [rans] max|rhoV| cell: x=%.5f y=%.6f (row j=%d, y/dy=%.2f)  rhoV=%.4e rhoU=%.4e  "
           "plane y=%.6f\n",
           (lvl>=0)?(ib*blockSize+ii+0.5)*dxL:-1.0, (lvl>=0)?(jb*blockSize+jj+0.5)*dyL:-1.0,
           jb*blockSize+jj, (lvl>=0)?((jb*blockSize+jj+0.5)):-1.0,
           (double)getField(F_RHOV)[idx], (double)getField(F_RHOU)[idx], (double)ibPlane);
  }

  // Locate a non-finite cell (mode 9) -- where a NaN is BORN says more than that
  // one exists.
  if (v[9] > 0) {
    ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, 9);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    real *mx = thrust::max_element(thrust::device, F, F+n);
    const size_t idx = (size_t)(mx - F);
    const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
    u64 loc = bLocList[b];
    i32 lvl=-1, ib=0, jb=0, kb=0;
    if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
    const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize;
    const double dxL = (lvl>=0) ? domainSize[0]/(baseGridSize[0]*powi(2,lvl)) : 0;
    const double dyL = (lvl>=0) ? domainSize[1]/(baseGridSize[1]*powi(2,lvl)) : 0;
    const double px = (lvl>=0)?(ib*blockSize+ii+0.5)*dxL:-1.0;
    const double py = (lvl>=0)?(jb*blockSize+jj+0.5)*dyL:-1.0;
    printf("  [rans] NaN cell: lvl=%d blk(%d,%d) cell(%d,%d) %s  x=%.5f y=%.6f  "
           "rho=%.4e rhoU=%.4e rhoV=%.4e rhoE=%.4e rhoK=%.4e rhoTau=%.4e\n",
           lvl, ib, jb, ii, jj,
           (loc==kEmpty)?"EMPTY":(isInteriorBlock(lvl,ib,jb,kb)?"interior":"EXTERIOR"),
           px, py,
           (double)getField(F_RHO)[idx], (double)getField(F_RHOU)[idx],
           (double)getField(F_RHOV)[idx], (double)getField(F_RHOE)[idx],
           (double)getField(F_RHOK)[idx], (double)getField(F_RHOTAU)[idx]);
    printf("  [rans]           plate at y=%.6f (ibPlane), leading edge x=%.4f, dy=%.6f\n",
           (double)ibPlane, (double)plateX0, dyL);
  }

  // Locate the hottest cell (mode 5), which is where an AMR/wall-model
  // inconsistency shows up first.
  ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, 5);
  cudaDeviceSynchronize();
  {
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    real *mx = thrust::max_element(thrust::device, F, F+n);
    const size_t idx = (size_t)(mx - F);
    const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
    u64 loc = bLocList[b];
    i32 lvl=-1, ib=0, jb=0, kb=0;
    if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
    const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
    const double dxL = (lvl>=0) ? domainSize[0]/(baseGridSize[0]*powi(2,lvl)) : 0;
    const double dyL = (lvl>=0) ? domainSize[1]/(baseGridSize[1]*powi(2,lvl)) : 0;
    printf("  [rans] hot cell: lvl=%d/%d blk(%d,%d,%d) cell(%d,%d,%d) %s  x=%.5f y=%.6f  a=%.3e rho=%.4e rhoE=%.4e\n",
           lvl, nLvls-1, ib, jb, kb, ii, jj, kk,
           (loc==kEmpty)?"EMPTY":(isInteriorBlock(lvl,ib,jb,kb)?"interior":"EXTERIOR"),
           (lvl>=0)?(ib*blockSize+ii+0.5)*dxL:-1.0, (lvl>=0)?(jb*blockSize+jj+0.5)*dyL:-1.0,
           (double)F[idx], (double)getField(F_RHO)[idx], (double)getField(F_RHOE)[idx]);
    printf("  [rans]           rhoU=%.6e rhoV=%.6e rhoW=%.6e rhoK=%.6e rhoTau=%.6e\n",
           (double)getField(F_RHOU)[idx], (double)getField(F_RHOV)[idx],
           (double)getField(F_RHOW)[idx], (double)getField(F_RHOK)[idx],
           (double)getField(F_RHOTAU)[idx]);
  }

  // Locate the cell that minimises the RANS dt limit (mode 4), so a pathological
  // value can be attributed to a position rather than guessed at.
  ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, 4);
  cudaDeviceSynchronize();
  {
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    real *mx = thrust::max_element(thrust::device, F, F+n);
    const size_t idx = (size_t)(mx - F);
    const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
    u64 loc = bLocList[b];
    i32 lvl=-1, ib=0, jb=0, kb=0;
    if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
    const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
    printf("  [rans] dt_rans cell: lvl=%d blk(%d,%d,%d) cell(%d,%d,%d) %s  rho=%.6e rhoTau=%.6e muT=%.6e\n",
           lvl, ib, jb, kb, ii, jj, kk,
           (loc==kEmpty) ? "EMPTY" : (isInteriorBlock(lvl,ib,jb,kb) ? "interior" : "EXTERIOR"),
           (double)getField(F_RHO)[idx], (double)getField(F_RHOTAU)[idx],
           (double)getField(F_MUT)[idx]);
  }

  // Which cell actually sets dt?  Re-run the real dt kernel and locate its min,
  // rather than inferring it from the individual limits.  (--debug: note that
  // the per-output dt the main loop prints is the REMAINDER step of that output
  // interval, not the operating dt -- this is what to read instead.)
  if (!dbgChecks) return;
  computeDeltaTKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
  real *dmin = thrust::min_element(thrust::device, F, F+n);
  const size_t idx = (size_t)(dmin - F);
  const i32 b = (i32)(idx/blockSizeTot), c = (i32)(idx%blockSizeTot);
  u64 loc = bLocList[b];
  i32 lvl=-1, ib=0, jb=0, kb=0;
  if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
  const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
  const double dxL = (lvl >= 0) ? domainSize[0]/(baseGridSize[0]*powi(2,lvl)) : 0.0;
  const double dyL = (lvl >= 0) ? domainSize[1]/(baseGridSize[1]*powi(2,lvl)) : 0.0;
  printf("  [rans] dt cell: lvl=%d %s  x=%.5f y=%.6f  DeltaT=%.3e  rhoU-1=%+.3e rhoV=%+.3e tau~=%.3e muT/mu=%.3e\n",
         lvl, (loc == kEmpty) ? "EMPTY-SLOT" : (isInteriorBlock(lvl,ib,jb,kb) ? "interior" : "EXTERIOR"),
         (lvl>=0) ? (ib*blockSize+ii+0.5)*dxL : -1.0,
         (lvl>=0) ? (jb*blockSize+jj+0.5)*dyL : -1.0,
         (double)F[idx],
         (double)getField(F_RHOU)[idx] - 1.0, (double)getField(F_RHOV)[idx],
         (double)(getField(F_RHOTAU)[idx]/fmax(getField(F_RHO)[idx],(real)1e-30)),
         (double)(getField(F_MUT)[idx]/fmax(mu,(real)1e-30)));
}

// Envelope check driver (--envcheck): launch the check, and when a cell has
// tripped it, print that cell and EVERY cell within 2.5 of its (finest) h so
// the staircase configuration around the seed is on the record.  Capped at 3
// reports; the buffer slot is dbgCnt[60].
void CompressibleSolver::envCheckStep(void) {
  if (!envCheck || envPrints >= 3) return;
  // dbgCnt is MANAGED: the device atomicCAS and this host read ping-pong its
  // page every step (measured as a ~10x whole-run crawl).  Checking every 8th
  // call still catches the first excursion within 8 steps of its birth.
  static i32 cadence = 0;
  if ((cadence++ & 7) != 0) return;
  if (envCheck == 1) { dbgCnt[60] = 0; envCheck = 2; }   // arm: the slot is never memset at allocation
  envCheckKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  const i32 hit = dbgCnt[60];
  if (hit == 0) return;
  envPrints++;
  const size_t idx = (size_t)hit - 1;
  const i32 b = (i32)(idx/blockSizeTot);
  u64 loc = bLocList[b];
  i32 lvl=-1, ib=0, jb=0, kb=0;
  if (loc != kEmpty) decode(loc, lvl, ib, jb, kb);
  const i32 c = (i32)(idx%blockSizeTot);
  const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize;
  const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
  const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
  const double x0 = (ib*blockSize+ii+0.5)*dxL, y0 = (jb*blockSize+jj+0.5)*dyL;
  const double hf = domainSize[0]/(baseGridSize[0]*powi(2,nLvls-1));
  printf("  [env] FIRST out-of-envelope cell: lvl=%d x=%.6f y=%.6f  (report %d)\n",
         lvl, x0, y0, envPrints);
  printf("  [env]   %9s %9s %5s | %9s %9s %9s %9s %9s %9s | %9s %9s\n",
         "dx", "dy", "lvl", "rho", "rhoU", "rhoV", "p", "k~", "tau~", "phiLS", "muT/mu");
  for (i32 bb = 0; bb < hashTable.nKeys; bb++) {
    u64 l2 = bLocList[bb];
    if (l2 == kEmpty) continue;
    i32 lv,i2,j2,k2; decode(l2, lv, i2, j2, k2);
    if (!isInteriorBlock(lv,i2,j2,k2)) continue;
    const double dx2 = domainSize[0]/(baseGridSize[0]*powi(2,lv));
    const double dy2 = domainSize[1]/(baseGridSize[1]*powi(2,lv));
    for (i32 cc = 0; cc < blockSizeTot; cc++) {
      const i32 i3 = cc%blockSize, j3 = (cc/blockSize)%blockSize, k3 = cc/blockSize/blockSize;
      if (k3 != 0) continue;
      const double x = (i2*blockSize+i3+0.5)*dx2, y = (j2*blockSize+j3+0.5)*dy2;
      if (fabs(x-x0) > 2.5*hf || fabs(y-y0) > 2.5*hf) continue;
      const size_t m = (size_t)bb*blockSizeTot + cc;
      const double r = (double)getField(F_RHO)[m];
      const double ru= (double)getField(F_RHOU)[m], rv=(double)getField(F_RHOV)[m];
      const double ke= 0.5*(ru*ru+rv*rv)/fmax(r,1e-30);
      const double p = ((double)gam-1.0)*((double)getField(F_RHOE)[m]-ke);
      const double ls= (double)getBoundaryLevelSet(Vec3((real)x,(real)y,(real)0));
      printf("  [env]   %+9.2e %+9.2e %5d | %9.3e %+9.2e %+9.2e %9.3e %+9.2e %+9.2e | %+9.2e %9.2e\n",
             x-x0, y-y0, lv, r, ru, rv, p,
             (double)getField(F_RHOK)[m], (double)getField(F_RHOTAU)[m], ls,
             (double)getField(F_MUT)[m]/fmax((double)mu,1e-30));
    }
  }
  dbgCnt[60] = 0;   // rearm for the next (post-cap: never) report
}

// Dump the solution for plotting: the finest-level field, and the wall-normal
// profile at one station in wall units (the canonical wall-model check -- the
// paper's Fig. 8).  u_tau comes from the wall model itself, so u+ = u/u_tau and
// y+ = d u_tau/nu are the model's own coordinates.
void CompressibleSolver::writeSolution(const char *fieldFile, const char *profFile,
                                       real xStation) {
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  if (wallGeom) applyWallGhosts();
  computeTurbClosure();                 // fills F_MUT / F_TF1
  // wallUtauKernel assigns every cell, so no pre-zero is needed
  wallUtauKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  double uTau = 0.0, bestDx = 1e30;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (lvl != nLvls-1 || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    if (immerserdBcType == 0 && jb != 0) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      if (immerserdBcType == 0 && jj != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL;
      const double ut = (double)getField(F_SCRATCH)[(size_t)b*blockSizeTot + c];
      if (ut > 0 && fabs(x-(double)xStation) < bestDx) { bestDx = fabs(x-(double)xStation); uTau = ut; }
    }
  }

  FILE *ff = fopen(fieldFile, "w");
  FILE *pf = fopen(profFile, "w");
  if (ff) fprintf(ff, "# x y u v k tau muT\n");
  if (pf) fprintf(pf, "# uTau=%.10e nu=%.10e xStation=%.6f\n# d yplus u uplus k tau muT\n",
                  uTau, (double)mu, (double)xStation);
  const double xTol = 0.5*domainSize[0]/(baseGridSize[0]*powi(2,nLvls-1));
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    // --fieldall also widens THIS dump: the finest level exists only near the
    // wall, so a finest-only profile is truncated well below the freestream and
    // cannot be normalised (measured: dump stopped at y=0.093 in a 0.5-tall box).
    if (!ibFieldAllLvls) { if (lvl != nLvls-1 || !isInteriorBlock(lvl, ib, jb, kb)) continue; }
    else if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;                       // pseudo-2D: one z layer
      const size_t m = (size_t)b*blockSizeTot + c;
      if (ibFieldAllLvls && cFlagsList[m] == PARENT) continue;   // leaves only
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      const double u = getField(F_RHOU)[m], v = getField(F_RHOV)[m];
      const double kv = getField(F_RHOK)[m], tv = getField(F_RHOTAU)[m];
      const double mt = getField(F_MUT)[m];
      if (ff) fprintf(ff, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e\n", x, y, u, v, kv, tv, mt);
      if (pf && fabs(x-(double)xStation) < xTol) {
        // Wall distance: the LEVEL SET gives it directly for an immersed body.
        // y + wallOffset is the grid-aligned special case only -- under an
        // immersed wall it measures from the domain floor, which is inside the
        // body, and the "profile" it dumps is the ghost/IC garbage down there.
        // u+ is the wall-PARALLEL speed, and only FLUID cells belong in it.
        double d, up = u;
        bool inFluid = true;
        if (immerserdBcType != 0) {
          Vec3 cp((real)x, (real)y, (real)0);
          const real hm = (real)fmin(dxL, dyL);
          inFluid = isFluidCell(cp, hm);
          d = (double)wallDistance(cp);
          Vec3 nw = wallNormal(cp, hm);
          const double vn  = u*(double)nw[0] + v*(double)nw[1];
          const double upx = u - vn*(double)nw[0], upy = v - vn*(double)nw[1];
          up = sqrt(upx*upx + upy*upy);
        } else d = y + (double)wallOffset;
        if (inFluid && d > 0.0)
          fprintf(pf, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                  d, d*uTau/(double)mu, up, (uTau>0)? up/uTau : 0.0, kv, tv, mt);
      }
    }
  }
  if (ff) fclose(ff);
  if (pf) fclose(pf);
  printf("---- solution dump ----\n  field -> %s   profile at x=%.3f -> %s   u_tau=%.6e\n",
         fieldFile, (double)xStation, profFile, uTau);
  primitiveToConservative();
}

// Surface pressure on an immersed body, sampled the way the WALL MODEL samples
// it -- along the level-set normal, not off the grid.
//
// The whole point of an immersed boundary is that the surface is resolved
// SUB-CELL: the level set carries the exact distance and normal, so the wall is
// never a staircase.  An earlier version of this routine threw that away by
// reporting each near-wall CELL CENTRE's pressure as the surface value.  Those
// centres sit at wall distances anywhere in (0, h], so around the nose -- where
// the curvature is highest and neighbouring cells straddle the surface quite
// differently -- consecutive samples came from different effective standoffs and
// the C_p distribution scattered.  That scatter was the DIAGNOSTIC, not the
// solution.
//
// Instead: walk the actual body geometry (the polyline vertices, which ARE the
// surface), step d_IP out along the level-set normal exactly as ibWallFlux does,
// and interpolate the pressure there from the finest-level fluid cells.  The
// wall model makes pressure Neumann across the near-wall layer (p_w = p_IP), so
// this IS the surface pressure, and it is parameterised by the true surface
// rather than by which cells happen to touch it.
//   Cp = (p - p_inf)/(0.5 rho_inf |u_inf|^2)
void CompressibleSolver::writeIbSurface(const char *fileName) {
  if (immerserdBcType == 0) { printf("[surf] no immersed body\n"); return; }
  if (ibPolyN < 3) { printf("[surf] surface sampling needs a polyline body\n"); return; }
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  cudaDeviceSynchronize();

  const i32 lf = nLvls - 1;
  const double dxF = domainSize[0]/(baseGridSize[0]*powi(2, lf));
  const double dyF = domainSize[1]/(baseGridSize[1]*powi(2, lf));
  const real  hF   = (real)fmin(dxF, dyF);

  // finest-level FLUID cells, keyed by global (i,j), for bilinear sampling
  std::unordered_map<long long, double> pmap;
  pmap.reserve((size_t)hashTable.nKeys*blockSizeTot);
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (lvl != lf || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const i32 gi = ib*blockSize + ii, gj = jb*blockSize + jj;
      Vec3 cp((real)((gi + 0.5)*dxF), (real)((gj + 0.5)*dyF), (real)0);
      if (!isFluidCell(cp, hF)) continue;                  // carries no solution
      pmap[((long long)gi << 32) ^ (long long)(unsigned)gj] =
        (double)getField(F_RHOE)[(size_t)b*blockSizeTot + c];
    }
  }

  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[surf] cannot open %s\n", fileName); primitiveToConservative(); return; }
  const double qInf = 0.5*((double)fsU*fsU + (double)fsV*fsV);   // rho_inf = 1
  // Standoff for the pressure sample.  d_IP = 3h (the wall model's own image
  // point) is too far for C_p: around a curved nose dp/dn = rho u_t^2 / R is
  // NOT small, so three cells out the peaks are genuinely relaxed (measured:
  // stagnation C_p 0.77 instead of ~1).  Half a cell is the closest standoff
  // with fluid support -- fluid cell centres start there -- and the bilinear
  // renormalises over whichever taps are fluid, so this is a one-sided
  // reconstruction toward the wall rather than a cell-centre lottery.
  const double dIp  = 0.5*(double)hF;
  fprintf(fp, "# x/c  yn/c  Cp  xSurf  ySurf  side   (side +1 = upper, -1 = lower,\n");
  fprintf(fp, "#   from the OUTWARD NORMAL's chord-normal component -- classifying by\n");
  fprintf(fp, "#   the sign of yn mislabels nose/tail points of a cambered section)\n");
  fprintf(fp, "# sampled at the image point d_IP = %.6e along the level-set normal\n", dIp);
  fprintf(fp, "# pInf=%.10e qInf=%.10e chord=%.6f\n", (double)fsP, qInf, (double)ibChord);

  i32 nOut = 0, nMiss = 0;
  double cpMin = 1e30, cpMax = -1e30;
  for (i32 e = 0; e < ibPolyN; e++) {
    const double xs = (double)ibPoly[2*e], ys = (double)ibPoly[2*e+1];
    Vec3 n = wallNormal(Vec3((real)xs, (real)ys, (real)0), hF);
    const double xip = xs + dIp*(double)n[0], yip = ys + dIp*(double)n[1];
    // bilinear interpolation over the finest-level fluid cells
    const double fx = xip/dxF - 0.5, fy = yip/dyF - 0.5;
    const long long i0 = (long long)floor(fx), j0 = (long long)floor(fy);
    const double tx = fx - (double)i0, ty = fy - (double)j0;
    double acc = 0, wsum = 0;
    for (i32 a = 0; a < 2; a++)
      for (i32 bb = 0; bb < 2; bb++) {
        const double w = (a ? tx : 1.0-tx)*(bb ? ty : 1.0-ty);
        if (w <= 0) continue;
        auto it = pmap.find(((i0+a) << 32) ^ (long long)(unsigned)(j0+bb));
        if (it == pmap.end()) continue;                    // solid or absent
        acc += w*it->second; wsum += w;
      }
    if (wsum <= 1e-12) { nMiss++; continue; }
    const double cp = (acc/wsum - (double)fsP)/fmax(qInf, 1e-30);
    const double ddx = xs - (double)ibOrigin[0], ddy = ys - (double)ibOrigin[1];
    const double xc = (ddx*(double)ibCosA - ddy*(double)ibSinA)/(double)ibChord;
    const double yn = (ddx*(double)ibSinA + ddy*(double)ibCosA)/(double)ibChord;
    const double nyn = (double)n[0]*(double)ibSinA + (double)n[1]*(double)ibCosA;
    fprintf(fp, "%.6e %.6e %.6e %.6e %.6e %d\n", xc, yn, cp, xs, ys,
            (nyn >= 0) ? 1 : -1);
    cpMin = fmin(cpMin, cp); cpMax = fmax(cpMax, cp);
    nOut++;
  }
  fclose(fp);
  printf("---- immersed surface ----\n  wrote %d/%d surface points to %s   Cp in [%.4f, %.4f]\n",
         nOut, (i32)ibPolyN, fileName, cpMin, cpMax);
  if (nMiss) printf("  [warn] %d surface points found no fluid at the image point\n", nMiss);
  if (cpMax < 0.9)
    printf("  [warn] max Cp = %.3f: a stagnation point should approach +1\n", cpMax);
  primitiveToConservative();
}

// Finest-level field in a window around the immersed body, for plotting.
// Solid cells are written with fluid = 0 so the plot can mask the body out
// rather than contour ghost values.
void CompressibleSolver::writeIbField(const char *fileName, real halfWidth) {
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  if (rans) computeTurbClosure();
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[field] cannot open %s\n", fileName); primitiveToConservative(); return; }
  // polyline bodies centre on their bbox; analytic bodies on ibCenter
  const double cx = (ibPolyN > 2) ? 0.5*((double)ibBox[0] + (double)ibBox[2]) : (double)ibCenter[0];
  const double cy = (ibPolyN > 2) ? 0.5*((double)ibBox[1] + (double)ibBox[3]) : (double)ibCenter[1];
  const double w  = (double)halfWidth*(double)ibChord;
  fprintf(fp, "# x/c y/c rho u v p mach cp fluid resid   (origin at the body centre)\n");
  // resid = per-cell |dq/dt| from the LAST residual sample (0 if --residevery is off)

  fprintf(fp, "# pInf=%.10e uInf=%.10e chord=%.6f\n",
          (double)fsP, sqrt((double)fsU*fsU + (double)fsV*fsV), (double)ibChord);
  const i32 lf = nLvls - 1;
  i32 nOut = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    // Default: finest level only (uniform spacing, which the existing analysis
    // scripts assume).  --fieldall writes every LEAF instead, giving the whole
    // composite AMR field -- needed to see anything away from the refined band,
    // since the finest level exists only at the wall and the shock.
    if (!ibFieldAllLvls) { if (lvl != lf || !isInteriorBlock(lvl, ib, jb, kb)) continue; }
    else if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const real hm = (real)fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      if (fabs(x - cx) > w || fabs(y - cy) > w) continue;
      const size_t m = (size_t)b*blockSizeTot + c;
      if (ibFieldAllLvls && cFlagsList[m] == PARENT) continue;   // leaves only
      const double r = (double)getField(F_RHO)[m];
      const double u = (double)getField(F_RHOU)[m], v = (double)getField(F_RHOV)[m];
      const double pp = (double)getField(F_RHOE)[m];
      const double a2 = gam*fmax(pp,1e-30)/fmax(r,1e-30);
      const double mach = sqrt((u*u + v*v)/fmax(a2,1e-30));
      const double qInf = 0.5*((double)fsU*fsU + (double)fsV*fsV);
      const double cp = (pp - (double)fsP)/fmax(qInf,1e-30);
      const i32 fl = isFluidCell(Vec3((real)x,(real)y,(real)0), hm) ? 1 : 0;
      const double rc = residCell ? (double)residCell[m] : 0.0;
      fprintf(fp, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %d %.6e\n",
              (x-cx)/(double)ibChord, (y-cy)/(double)ibChord, r, u, v, pp, mach, cp, fl, rc);
      nOut++;
    }
  }
  fclose(fp);
  printf("  field -> %s  (%d finest-level cells within %.1f chords)\n",
         fileName, nOut, (double)halfWidth);
  primitiveToConservative();
}

// Dump the block structure for plotting: one line per block with its origin,
// side length and level.  Blocks are blockSize x blockSize cells, so the cell
// size is side/blockSize -- the plot can draw either.
// Dump the CACHED geometry (F_PHI, F_IBM) next to the ANALYTIC classification,
// at every level in a window around the body.  The point is to be able to SEE
// the two disagree: F_IBM is what the solver actually keys the wall faces on,
// and a stale or unreachable entry there is exactly what produced the phantom
// wall faces (mask read at an invalid neighbour index) earlier today.
// Dump the wall-model geometry at every MODELLED face, reproducing exactly what
// ibWallFlux does: a face between a fluid and a non-fluid cell is the boundary;
// its centre sits d_FC = -phi(face) from the surface, the normal is -grad(phi)
// there, the foot point is fcPos - d_FC n, and the image point is
// fcPos + (d_IP - d_FC) n with d_IP = max(dIpFac h, d_FC + ipStandMin h).
// Only the x- and y-LOW faces are walked, matching the RHS's face loop.
void CompressibleSolver::writeIbWallFaces(const char *fileName, real halfWidth) {
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[faces] cannot open %s\n", fileName); return; }
  const double cx = 0.5*((double)ibBox[0] + (double)ibBox[2]);
  const double cy = 0.5*((double)ibBox[1] + (double)ibBox[3]);
  const double w  = (double)halfWidth*(double)ibChord;
  fprintf(fp, "# xf yf dir h lvl dFcOverH xs ys nx ny xip yip modelled\n");
  fprintf(fp, "# dIpFac=%.3f ipStandMin=%.3f plateX0=%.4f chord=%.6f\n",
          (double)dIpFac, (double)ipStandMin, (double)plateX0, (double)ibChord);
  i32 nF = 0, nSlip = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (kb != 0 || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      if (fabs(x-cx) > w || fabs(y-cy) > w) continue;
      const bool fC = isFluidCell(Vec3((real)x,(real)y,(real)0), (real)fmin(dxL,dyL));
      for (i32 dir = 0; dir < 2; dir++) {
        const double hd = dir ? dyL : dxL;
        const double xn = dir ? x : x - hd, yn = dir ? y - hd : y;
        const bool fN = isFluidCell(Vec3((real)xn,(real)yn,(real)0), (real)hd);
        if (fC == fN) continue;                       // not a wall face
        const double xf = dir ? x : x - 0.5*hd, yf = dir ? y - 0.5*hd : y;
        Vec3 fcp((real)xf, (real)yf, (real)0);
        const double dFc = fmax(-(double)getBoundaryLevelSet(fcp), 0.1*hd);
        Vec3 n = wallNormal(fcp, (real)hd);
        const double dIp = fmax((double)dIpFac*hd, dFc + (double)ipStandMin*hd);
        const double xs = xf - dFc*(double)n[0], ys = yf - dFc*(double)n[1];
        const double xi = xf + (dIp - dFc)*(double)n[0];
        const double yi = yf + (dIp - dFc)*(double)n[1];
        const i32 modelled = (xf >= (double)plateX0) ? 1 : 0;   // else slip
        if (!modelled) nSlip++;
        fprintf(fp, "%.8e %.8e %d %.8e %d %.6f %.8e %.8e %.6f %.6f %.8e %.8e %d\n",
                xf, yf, dir, hd, lvl, dFc/hd, xs, ys,
                (double)n[0], (double)n[1], xi, yi, modelled);
        nF++;
      }
    }
  }
  fclose(fp);
  printf("  wall faces -> %s  (%d faces within %.1f chords; %d slip, %d modelled)\n",
         fileName, nF, (double)halfWidth, nSlip, nF-nSlip);
}

// Dump the ghost-fill construction, reproducing ibGhostKernel's geometry
// exactly: for every filled ghost (non-fluid, phi <= 2.5h) the cell centre, its
// foot point on the surface (p + phi n), the mollified normal, and the sample
// point at the fixed standoff s* = 2h along that normal -- the "image line"
// the Neumann rho/p and the tangential velocity are drawn from.
void CompressibleSolver::writeIbGhostLines(const char *fileName, real halfWidth) {
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[glines] cannot open %s\n", fileName); return; }
  const double cx = (ibPolyN > 2) ? 0.5*((double)ibBox[0] + (double)ibBox[2]) : (double)ibCenter[0];
  const double cy = (ibPolyN > 2) ? 0.5*((double)ibBox[1] + (double)ibBox[3]) : (double)ibCenter[1];
  const double w  = (double)halfWidth*(double)ibChord;
  fprintf(fp, "# xg yg phi h lvl xs ys nx ny xip yip dS intersecting\n");
  i32 nG = 0, nItx = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (kb != 0 || lvl != nLvls-1 || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const real h = (real)fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      if (fabs(x-cx) > w || fabs(y-cy) > w) continue;
      Vec3 pc((real)x, (real)y, (real)0);
      if (isFluidCell(pc, h)) continue;                 // fluid: not a ghost
      const real phi = getBoundaryLevelSet(pc);
      if (phi > (real)2.5*h) continue;                  // deep interior: not filled
      Vec3 n = wallNormal(pc, h);
      const double xs = x + (double)phi*(double)n[0], ys = y + (double)phi*(double)n[1];
      const real sStar = (real)2*h;
      const double xi = xs + (double)sStar*(double)n[0], yi = ys + (double)sStar*(double)n[1];
      real dS = -getBoundaryLevelSet(Vec3((real)xi, (real)yi, (real)0));
      if (dS < (real)0.5*h) dS = sStar;
      const i32 itx = (phi <= 0) ? 1 : 0;
      nItx += itx;
      fprintf(fp, "%.8e %.8e %.6e %.6e %d %.8e %.8e %.6f %.6f %.8e %.8e %.6e %d\n",
              x, y, (double)phi, (double)h, lvl, xs, ys,
              (double)n[0], (double)n[1], xi, yi, (double)dS, itx);
      nG++;
    }
  }
  fclose(fp);
  printf("  ghost lines -> %s  (%d filled ghosts on the finest level; %d intersecting)\n",
         fileName, nG, nItx);
}

void CompressibleSolver::writeIbMask(const char *fileName, real halfWidth) {
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[mask] cannot open %s\n", fileName); return; }
  const double cx = 0.5*((double)ibBox[0] + (double)ibBox[2]);
  const double cy = 0.5*((double)ibBox[1] + (double)ibBox[3]);
  const double w  = (double)halfWidth*(double)ibChord;
  fprintf(fp, "# x y h lvl phi ibmCached fluidAnalytic  (domain coords)\n");
  fprintf(fp, "# chord=%.6f centre=%.6f,%.6f\n", (double)ibChord, cx, cy);
  i32 nOut = 0, nMismatch = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (kb != 0 || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const real hm = (real)fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      if (fabs(x-cx) > w || fabs(y-cy) > w) continue;
      const size_t m = (size_t)b*blockSizeTot + c;
      const double phi = (double)getField(F_PHI)[m];
      const i32 ibmC = (getField(F_IBM)[m] > (real)0.5) ? 1 : 0;
      const i32 ibmA = isFluidCell(Vec3((real)x,(real)y,(real)0), hm) ? 1 : 0;
      if (ibmC != ibmA) nMismatch++;
      fprintf(fp, "%.8e %.8e %.8e %d %.8e %d %d\n", x, y, dxL, lvl, phi, ibmC, ibmA);
      nOut++;
    }
  }
  fclose(fp);
  printf("  mask -> %s  (%d cells within %.1f chords, ALL levels)\n",
         fileName, nOut, (double)halfWidth);
  printf("  cached F_IBM vs analytic isFluidCell: %d mismatches%s\n",
         nMismatch, nMismatch ? "  <-- CACHE IS STALE" : "  (consistent)");
}

void CompressibleSolver::writeGridBlocks(const char *fileName) {
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[grid] cannot open %s\n", fileName); return; }
  fprintf(fp, "# x0 y0 side lvl interior   (block = blockSize^2 cells)\n");
  fprintf(fp, "# blockSize=%d nLvls=%d domain=%.6f x %.6f\n",
          blockSize, nLvls, (double)domainSize[0], (double)domainSize[1]);
  i32 n = 0, perLvl[16] = {0};
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (kb != 0) continue;                       // pseudo-2D: one z layer
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2, lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2, lvl));
    const i32 inr = isInteriorBlock(lvl, ib, jb, kb) ? 1 : 0;
    fprintf(fp, "%.8e %.8e %.8e %d %d\n",
            ib*blockSize*dxL, jb*blockSize*dyL, blockSize*dxL, lvl, inr);
    n++; if (lvl >= 0 && lvl < 16) perLvl[lvl]++;
  }
  fclose(fp);
  printf("  grid -> %s  (%d blocks:", fileName, n);
  for (i32 l = 0; l < nLvls; l++) printf(" L%d=%d", l, perLvl[l]);
  printf(")\n");
}

void CompressibleSolver::writeCfProfile(const char *fileName) {
  conservativeToPrimitive();
  setBoundaryConditions(0, 1);
  wallUtauKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[cf] cannot open %s\n", fileName); primitiveToConservative(); return; }
  fprintf(fp, "# x  Cf  uTau   (Cf = 2 uTau^2 / uInf^2)\n");
  real *Sc = getField(F_SCRATCH);
  i32 nOut = 0;
  double cfAt097 = 0.0, bestDx = 1e30;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2, lvl));
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2, lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c % blockSize, jj = (c/blockSize) % blockSize;
      // grid-aligned: the wall is the domain bottom row.  Immersed: the wall
      // sits inside the domain, so take whichever cell carries a stamped u_tau.
      if (immerserdBcType == 0 && jb*blockSize + jj != 0) continue;
      const double x = (ib*blockSize + ii + 0.5)*dxL;
      const double y = (jb*blockSize + jj + 0.5)*dyL;
      if (x < plateX0) continue;
      const double ut = (double)Sc[(size_t)b*blockSizeTot + c];
      if (ut <= 0) continue;
      // Cf against the freestream SPEED (the inclined case runs the stream at
      // (fsU, fsV) with |u| = 1), and the station as ARC LENGTH along the
      // plate from the leading edge (plateX0, ibPlane): the level set gives
      // this cell's wall distance d, so the along-plate coordinate is
      // sqrt(|p - LE|^2 - d^2) for a flat plate at any inclination.
      const double uInf2 = (double)fsU*(double)fsU + (double)fsV*(double)fsV;
      const double cf = 2.0*ut*ut/fmax(uInf2, 1e-30);
      double s = x - (double)plateX0;
      if (immerserdBcType != 0) {
        const double dw   = (double)wallDistance(Vec3((real)x, (real)y, (real)0));
        const double dxle = x - (double)plateX0, dyle = y - (double)ibPlane;
        s = sqrt(fmax(dxle*dxle + dyle*dyle - dw*dw, 0.0));
      }
      fprintf(fp, "%.10e %.10e %.10e\n", (double)plateX0 + s, cf, ut);
      nOut++;
      if (fabs(s - 0.97) < bestDx) { bestDx = fabs(s - 0.97); cfAt097 = cf; }
    }
  }
  fclose(fp);
  const double nuInf = (double)mu;
  const double dxF = domainSize[0]/(baseGridSize[0]*powi(2, nLvls-1));
  const double utR = sqrt(0.5*cfAt097)*(double)fsU;
  printf("---- FPTBL skin friction ----\n");
  printf("  wrote %d wall stations to %s\n", nOut, fileName);
  printf("  dx+ = %.0f   d_IP+ = %.0f   (paper Table 1: dx+ 26-210, d_IP+ 80-640)\n",
         dxF*utR/nuInf, dIpFac*dxF*utR/nuInf);
  printf("  Cf at x/L = 0.97 : %.6f    (TMR reference ~ 0.00270, paper tol 4%%  -> %+.1f%%)\n",
         cfAt097, 100.0*(cfAt097 - 0.0027)/0.0027);
  primitiveToConservative();
}

// Verify the cut geometry against the exact body: sum(alpha)*dV must equal the
// FLUID area, and the discrete divergence theorem sum_f A_f n_f + A_w n_w = 0
// must hold cell by cell (it is what makes the cut update conservative).
// Well-balancedness of the cut-cell discretisation: a UNIFORM state (rho, u=0,
// p) makes every flux pure pressure, so the residual must vanish identically --
//   sum_f p n_f A_f + p n_w A_w = p (sum_f n_f A_f + n_w A_w) = 0
// by the discrete divergence theorem.  Any nonzero residual localises the term
// that does not close, which no amount of reasoning about the algebra can.
static inline i32 hostNbrIdx(const i32 *nbrIdxList, i32 bIdx, i32 i, i32 j);   // defined with the merge code below
void CompressibleSolver::checkWellBalanced(void) {
  cudaDeviceSynchronize();
  real *Rho=getField(F_RHO), *U=getField(F_RHOU), *V=getField(F_RHOV),
       *W=getField(F_RHOW), *P=getField(F_RHOE);
  for (i32 c = 0; c < hashTable.nKeys*blockSizeTot; c++) {
    Rho[c] = 1; U[c] = 0; V[c] = 0; W[c] = 0; P[c] = 1;
  }
  cudaDeviceSynchronize();
  for (i32 f = 0; f < NEVOLVE; f++)
    cudaMemset(getField(F_RHS+f), 0, (size_t)hashTable.nKeys*blockSizeTot*sizeof(real));
  if (p1) {   // uniform state: zero slopes everywhere (cells and pieces), clean slope accumulators
    for (i32 s = 0; s < 2*P1_NV; s++) {
      cudaMemset(getField(F_P1S+s),  0, (size_t)hashTable.nKeys*blockSizeTot*sizeof(real));
      cudaMemset(getField(F_P1SR+s), 0, (size_t)hashTable.nKeys*blockSizeTot*sizeof(real));
    }
    if (cutPieceSX) { cudaMemset(cutPieceSX, 0, (size_t)2*P1_NV*cutPieceQCap*sizeof(real)); cudaMemset(cutPieceSR, 0, (size_t)2*P1_NV*cutPieceQCap*sizeof(real)); }
    if (cutPieceQ) for (i32 k = 0; k < nCutPiece; k++) { cutPieceQ[k] = 1; cutPieceQ[cutPieceQCap+k] = 0; cutPieceQ[2*(size_t)cutPieceQCap+k] = 0; cutPieceQ[3*(size_t)cutPieceQCap+k] = 0; cutPieceQ[4*(size_t)cutPieceQCap+k] = (real)(1.0/(gam-1.0)); }
    if (cutPieceS) cudaMemset(cutPieceS, 0, (size_t)NEVOLVE*cutPieceQCap*sizeof(real));
  }
  cudaDeviceSynchronize();
  computeRightHandSide();
  cudaDeviceSynchronize();
  if (p1) {
    double ms = 0, mp = 0;
    const i32 cEwb = bEmpty*blockSizeTot;
    for (i32 c = 0; c < hashTable.nKeys*blockSizeTot; c++) {
      if (cFlagsList[c] != ACTIVE || !(getField(F_CUTA)[c] > (real)ibRccmAlphaMin)) continue;
      // the domain boundary imposes the freestream WEAKLY, so a cell on it sees a
      // real flux against this rest state: judge the cut geometry on interior cells
      { const i32 b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize; bool edge = false;
        const i32 nb[4] = { hostNbrIdx(nbrIdxList, b, i-1, j), hostNbrIdx(nbrIdxList, b, i+1, j), hostNbrIdx(nbrIdxList, b, i, j-1), hostNbrIdx(nbrIdxList, b, i, j+1) };
        for (i32 q = 0; q < 4; q++) { if (nb[q] < 0 || nb[q] >= cEwb) { edge = true; break; } i32 l, ib, jb, kb; decode(bLocList[nb[q]/blockSizeTot], l, ib, jb, kb); if (!isInteriorBlock(l, ib, jb, kb)) { edge = true; break; } }
        if (edge) continue; }
      for (i32 s = 0; s < 2*P1_NV; s++) ms = fmax(ms, fabs((double)getField(F_P1SR+s)[c]));
    }
    if (cutPieceSR) for (i32 k = 0; k < nCutPiece; k++) for (i32 s = 0; s < 2*P1_NV; s++) mp = fmax(mp, fabs((double)cutPieceSR[(size_t)s*cutPieceQCap + k]));
    printf("  [p1] max |slope Rhs| over live cells = %.3e, over pieces = %.3e  (%d cut elements, %d quadrature points, %d face pieces)\n", ms, mp, nP1Elem, nP1Qpt, nP1Seg);
  }
  double mx[3] = {0,0,0}; i32 nbad = 0; double worstA = 1; i32 worstCut = -1;
  real *A = getField(F_CUTA);
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    if (bLocList[bIdx] == kEmpty) continue;
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      if (!(A[cIdx] > (real)ibRccmAlphaMin)) continue;          // dead
      if (getField(F_PHI)[cIdx] >= (real)0) continue;           // R-Cell
      if (p1) {   // --p1 imposes the freestream weakly at the domain boundary: skip the cells on it
        const i32 i = c%blockSize, j = (c/blockSize)%blockSize; bool edge = false;
        const i32 nb[4] = { hostNbrIdx(nbrIdxList, bIdx, i-1, j), hostNbrIdx(nbrIdxList, bIdx, i+1, j), hostNbrIdx(nbrIdxList, bIdx, i, j-1), hostNbrIdx(nbrIdxList, bIdx, i, j+1) };
        for (i32 q = 0; q < 4 && !edge; q++) { if (nb[q] < 0 || nb[q] >= bEmpty*blockSizeTot) { edge = true; break; } i32 l, ib, jb, kb; decode(bLocList[nb[q]/blockSizeTot], l, ib, jb, kb); if (!isInteriorBlock(l, ib, jb, kb)) edge = true; }
        if (edge) continue;
      }
      const double ru = fabs((double)getField(F_RHS+F_RHOU)[cIdx]);
      const double rv = fabs((double)getField(F_RHS+F_RHOV)[cIdx]);
      const double rr = fabs((double)getField(F_RHS+F_RHO)[cIdx]);
      if (ru > mx[1] || rv > mx[2]) { worstA = A[cIdx]; worstCut = cIdx; }
      mx[0] = fmax(mx[0], rr); mx[1] = fmax(mx[1], ru); mx[2] = fmax(mx[2], rv);
      if (fmax(ru, rv) > 1e-10) nbad++;
    }
  }
  printf("---- RCCM well-balanced test (uniform rho=1, u=0, p=1) ----\n");
  printf("  max |Rhs(rho)| = %.3e\n", mx[0]);
  printf("  max |Rhs(rhoU)|= %.3e   max |Rhs(rhoV)| = %.3e\n", mx[1], mx[2]);
  printf("  NR cells with |Rhs(mom)| > 1e-10 : %d   (worst cell alpha = %.4f)\n",
         nbad, worstA);
  printf("-----------------------------------------------------------\n");
  (void)worstCut;
}

extern __device__ unsigned long long g_rcDeadFace;
extern __device__ unsigned long long g_rcDeadGrad;
extern __device__ unsigned long long g_ibFaceRows;
extern __device__ unsigned long long g_rcLiveFace;

void CompressibleSolver::reportIbFaceRows(void) {
  unsigned long long fr = 0;
  cudaMemcpyFromSymbol(&fr, g_ibFaceRows, sizeof(fr));
  printf("[ibface] wall rows added to the gradient fit: %llu\n", fr);
}

void CompressibleSolver::reportDeadTaps(void) {
  unsigned long long df=0, dg=0, lf=0;
  cudaMemcpyFromSymbol(&df, g_rcDeadFace, sizeof(df));
  cudaMemcpyFromSymbol(&dg, g_rcDeadGrad, sizeof(dg));
  cudaMemcpyFromSymbol(&lf, g_rcLiveFace, sizeof(lf));
  printf("---- RCCM dead-cell exposure ----\n");
  printf("  faces read with a DEAD neighbour AND open aperture : %llu\n", df);
  printf("  live-cell faces examined                           : %llu\n", lf);
  printf("  gradient taps skipped because dead                 : %llu\n", dg);
  printf("---------------------------------\n");
}

void CompressibleSolver::checkCutGeometry(void) {
  cudaDeviceSynchronize();
  const i32 cE = bEmpty*blockSizeTot;
  double area = 0, wallLen = 0, gclMax = 0, stampErr = 0;
  i32 nCut = 0, nR = 0;
  real *A = getField(F_CUTA), *AX = getField(F_CUTAX), *AY = getField(F_CUTAY);
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    const double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      const i32 i = c % blockSize, j = (c/blockSize) % blockSize;
      const double al = A[cIdx];
      area += al*dxL*dyL;
      if (al > 1e-12 && al < 1.0 - 1e-12) {
        nCut++;
        if (getField(F_PHI)[cIdx] > 0) nR++;
        // GCL: A_w n = -( (aXhi-aXlo) dy , (aYhi-aYlo) dx )
        // Recompute the HIGH-face apertures straight from the level set rather
        // than reading the neighbour's stored low aperture: that keeps the
        // check independent of the block layout (and cross-checks the stamp).
        const i32 ib2 = ( loc        & ((1 << 20)-1)) - 1;
        const i32 jb2 = ((loc >> 20) & ((1 << 20)-1)) - 1;
        Vec3 cp((real)((ib2*blockSize + i + 0.5)*dxL),
                (real)((jb2*blockSize + j + 0.5)*dyL), (real)0);
        const double hx = 0.5*dxL, hy = 0.5*dyL;
        auto ls = [&](double X, double Y) {
          return (double)getBoundaryLevelSet(Vec3((real)X, (real)Y, cp[2])); };
        auto edge = [](double fa, double fb) {
          if (fa < 0 && fb < 0) return 1.0;
          if (fa >= 0 && fb >= 0) return 0.0;
          return fa < 0 ? fa/(fa-fb) : fb/(fb-fa); };
        const double f00 = ls(cp[0]-hx, cp[1]-hy), f10 = ls(cp[0]+hx, cp[1]-hy);
        const double f11 = ls(cp[0]+hx, cp[1]+hy), f01 = ls(cp[0]-hx, cp[1]+hy);
        const double axLo = edge(f00, f01), axHi = edge(f10, f11);
        const double ayLo = edge(f00, f10), ayHi = edge(f01, f11);
        stampErr = fmax(stampErr, fabs(axLo - (double)AX[cIdx]));
        stampErr = fmax(stampErr, fabs(ayLo - (double)AY[cIdx]));
        const double nwx = -(axHi - axLo)*dyL;
        const double nwy = -(ayHi - ayLo)*dxL;
        wallLen += sqrt(nwx*nwx + nwy*nwy);
        gclMax = fmax(gclMax, fabs(nwx) + fabs(nwy));
      }
    }
  }
  if (cutSplit && cutSplitCell && nCutSplit > 0) {
    // --cutsplit diagnostic: the two states across a thin wall, per split cell
    real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoV = getField(F_RHOV), *RhoE = getField(F_RHOE);
    auto pres = [&](i32 h) {
      if (h == CUT_DEAD) return -1.0;
      double r, ru, rv, rE;
      if (cutIsPiece(h)) { const i32 k = cutPieceOf(h); const size_t cap = cutPieceQCap;
        r = cutPieceQ[k]; ru = cutPieceQ[cap+k]; rv = cutPieceQ[2*cap+k]; rE = cutPieceQ[4*cap+k]; }
      else { r = Rho[h]; ru = RhoU[h]; rv = RhoV[h]; rE = RhoE[h]; }
      const double u = ru/r, v = rv/r;
      return (double)(gam - (real)1)*(rE - 0.5*r*(u*u + v*v)); };
    printf("---- split cells (x, piece-0 side by cy0, p_dof, p_piece1, owner handle, dp) ----\n");
    i32 shown = 0;
    for (i32 k = 0; k < nCutSplit; k++) {
      const CutSplitCell &S = cutSplitCell[k];
      const CutPiece &P = cutPiece[S.first];
      const i32 c = P.cell;
      if (c >= cE || cFlagsList[c] != ACTIVE) continue;
      double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy);
      const double p0 = pres(cutOwner ? cutOwner[c] : c), p1 = pres(P.owner);
      double ox = 0, oy = 0; if (P.owner != CUT_DEAD) { double odx, ody; cellGeomHost(cutIsPiece(P.owner) ? cutPiece[cutPieceOf(P.owner)].cell : P.owner, ox, oy, odx, ody); }
      shown++;
      printf("  x=%.4f y=%.4f  p0side=%s a0=%.3f  p0=%.5f  p1=%.5f (owner %d at y=%.4f, a1=%.3f)  dp(below-above)=%+.5f\n",
               px, py, S.cy0 > 0 ? "above" : "below", (double)S.a0, p0, p1, P.owner, oy, (double)P.a,
               S.cy0 > 0 ? p1 - p0 : p0 - p1);
    }
  }
  printf("---- RCCM cut geometry ----\n");
  printf("  fluid area   = %.10f\n", area);
  {
    const double domA = (double)domainSize[0]*(double)domainSize[1];
    if (clipSegN > 0)
      printf("  exact (segments) = %.10f   err = %.3e\n", domA - clipArea, area - (domA - clipArea));
    if (immerserdBcType == 3) {
      const double ex = domA - M_PI*(double)ibRadius*(double)ibRadius;
      printf("  exact (circle)   = %.10f   err = %.3e\n", ex, area - ex);
    }
    if (immerserdBcType == 6 && ibPolyN > 2) {
      double A2 = 0;
      for (i32 e = 0; e < ibPolyN; e++) {
        const i32 f = (e + 1 == ibPolyN) ? 0 : e + 1;
        A2 += (double)ibPoly[2*e]*(double)ibPoly[2*f+1] - (double)ibPoly[2*f]*(double)ibPoly[2*e+1];
      }
      const double ex = domA - 0.5*fabs(A2);
      printf("  exact (polyline) = %.10f   err = %.3e\n", ex, area - ex);
    }
  }
  printf("  wall length  = %.10f\n", wallLen);
  printf("  cut cells    = %d  (R-Cells, centre outside: %d)\n", nCut, nR);
  printf("  max |A_w n|  = %.3e   stored-vs-exact aperture err = %.2e\n", gclMax, stampErr);
  printf("---------------------------\n");
}

// --cutgeom 2: replace the LINEAR corner cut with the CURVED Q2 moment-fitted
// geometry on every cut cell.  Runs on the HOST, once per geometry stamp: the
// Saye arena is ~10 KB, which as a per-thread device array would reserve that
// frame for every thread of the stamp kernel and exhaust local memory at launch.
// Only alpha / the two low-face apertures / the fluid centroid are replaced --
// the wall segment stays A_w n = -sum_f A_f n_f, so conservation is untouched.
void CompressibleSolver::stampCutGeomCurved(void) {
  if (cutGeom != 2 || !ibRccm || immerserdBcType == 0) return;
  cudaDeviceSynchronize();
  real *A = getField(F_CUTA), *AX = getField(F_CUTAX), *AY = getField(F_CUTAY);
  real *CX = getField(F_CUTCX), *CY = getField(F_CUTCY);
  real *TX = getField(F_CUTTX), *TY = getField(F_CUTTY);
  i32 nCut = 0, nFail = 0, nOvf = 0, nEmpty = 0;
  const i32 NBUF = 262144;                      // heap arena: ~15 MB, host only, once per stamp
  SayeNode *sbuf = (SayeNode*)malloc((size_t)NBUF*sizeof(SayeNode));
  if (!sbuf) { printf("[cutgeom] arena alloc failed; keeping the linear cut\n"); return; }
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    const real dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const real dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const real hR  = fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const size_t m = (size_t)b*blockSizeTot + c;
      const real a0 = A[m];
      if (!(a0 > (real)0) || a0 >= (real)1 - (real)1e-12) continue;   // uncut or dead
      const real px = (ib*blockSize+ii+(real)0.5)*dxL;
      const real py = (jb*blockSize+jj+(real)0.5)*dyL;
      real f33[3][3];
      for (i32 sj = 0; sj < 3; sj++)
        for (i32 si = 0; si < 3; si++)
          f33[sj][si] = getBoundaryLevelSet(
              Vec3(px + ((real)0.5*si - (real)0.5)*dxL,
                   py + ((real)0.5*sj - (real)0.5)*dyL, (real)0));
      real al, ax, ay, cx = (real)0.5, cy = (real)0.5, tx = 0, ty = 0;
      i32 why = 0;
      if (cutGeomMoment(f33, sbuf, NBUF, al, ax, ay, &cx, &cy, &tx, &ty, &why)) {
        A[m] = al; AX[m] = ax; AY[m] = ay; TX[m] = tx; TY[m] = ty;
        CX[m] = (cx - (real)0.5)*dxL/hR;
        CY[m] = (cy - (real)0.5)*dyL/hR;
        nCut++;
      } else { nFail++; if (why == 1) nOvf++; else nEmpty++; }
    }
  }
  free(sbuf);
  printf("[cutgeom] curved Q2 moment fitting on %d cut cells (%d fell back: %d arena overflow, %d empty rule)\n",
         nCut, nFail, nOvf, nEmpty);
}

// --cutgeom 3: the cell geometry from the body SEGMENTS themselves (CutClip.h).
// No level set is involved: the polyline is clipped to the cell box and the
// fluid loops are walked, which is exact for the polyline as given (so it is at
// least as accurate as --cutgeom 2, whose Q2 fit approximates this same
// polyline), and it is the only route that sees a body thinner than the cell:
// a cell crossed twice comes back with two fluid loops.  Under the current
// one-DOF-per-cell scheme those loops are merged (area-weighted) into the nine
// stamped fields; the split-cell/agglomeration layer will keep them apart.
struct ClipRec { i32 cell; double px, py, dx, dy; ClipResult R; };

void CompressibleSolver::buildClipSegments(void) {
  if (clipSegN > 0) return;
  std::vector<double> xy;
  bool fluidInside = false;
  if (immerserdBcType == 6 && ibPolyN > 2) {
    xy.resize((size_t)2*ibPolyN);
    for (i32 i = 0; i < 2*ibPolyN; i++) xy[i] = (double)ibPoly[i];
    fluidInside = ibPolyFluidInside != 0;
  } else if (immerserdBcType == 3) {
    const i32 N = cutSeg < 8 ? 8 : cutSeg;
    xy.resize((size_t)2*N);
    for (i32 k = 0; k < N; k++) {
      const double th = 2.0*M_PI*(double)k/(double)N;
      xy[2*k]   = (double)ibCenter[0] + (double)ibRadius*cos(th);
      xy[2*k+1] = (double)ibCenter[1] + (double)ibRadius*sin(th);
    }
  } else {
    printf("[cutclip] body type %d has no segment geometry; --cutgeom 3 keeps the linear cut\n",
           immerserdBcType);
    return;
  }
  const i32 n = (i32)(xy.size()/2);
  double A2 = 0;
  clipBox[0] = clipBox[2] = xy[0]; clipBox[1] = clipBox[3] = xy[1];
  for (i32 e = 0; e < n; e++) {
    const i32 f = (e + 1 == n) ? 0 : e + 1;
    A2 += xy[2*e]*xy[2*f+1] - xy[2*f]*xy[2*e+1];
    clipBox[0] = fmin(clipBox[0], xy[2*e]); clipBox[2] = fmax(clipBox[2], xy[2*e]);
    clipBox[1] = fmin(clipBox[1], xy[2*e+1]); clipBox[3] = fmax(clipBox[3], xy[2*e+1]);
  }
  const bool ccw = A2 > 0;
  clipFluidLeftFwd = (ccw == fluidInside);
  clipArea = 0.5*fabs(A2);
  clipSeg = (double*)malloc((size_t)2*n*sizeof(double));
  for (i32 i = 0; i < 2*n; i++) clipSeg[i] = xy[i];
  clipSegN = n;
  printf("[cutclip] %d segments (%s, %s inside), enclosed area %.12f\n", n,
         ccw ? "ccw" : "cw", fluidInside ? "fluid" : "solid", clipArea);
}

void CompressibleSolver::stampCutGeomClip(void) {
  if (cutGeom != 3 || !ibRccm || immerserdBcType == 0) return;
  buildClipSegments();
  if (clipSegN == 0) return;
  cudaDeviceSynchronize();
  real *A = getField(F_CUTA), *AX = getField(F_CUTAX), *AY = getField(F_CUTAY);
  real *CX = getField(F_CUTCX), *CY = getField(F_CUTCY);
  real *TX = getField(F_CUTTX), *TY = getField(F_CUTTY);
  i32 nCell = 0, nSplit = 0, nHole = 0, nOvf = 0, nBad = 0, nChanged = 0;
  double dAlphaMax = 0;
  ClipResult R;
  std::vector<ClipRec> recs;               // --cutsplit: every clipped cell, for the piece/face tables
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    const double dxL = (double)domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = (double)domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const double hR  = fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const size_t m = (size_t)b*blockSizeTot + c;
      cutclip::Box B;
      B.x0 = (ib*blockSize+ii)*dxL;  B.x1 = B.x0 + dxL;
      B.y0 = (jb*blockSize+jj)*dyL;  B.y1 = B.y0 + dyL;
      B.dx = dxL; B.dy = dyL; B.P = 2*(dxL + dyL);
      // cheap reject: the cell box does not meet the body box (closed)
      if (B.x1 < clipBox[0] || B.x0 > clipBox[2] || B.y1 < clipBox[1] || B.y0 > clipBox[3]) continue;
      cutclip::clipCell(B, clipSeg, clipSegN, clipFluidLeftFwd, R);
      if (R.overflow) { nOvf++; continue; }
      if (R.bad)      { nBad++; continue; }
      const double px = 0.5*(B.x0 + B.x1), py = 0.5*(B.y0 + B.y1), cellA = dxL*dyL;
      double area, mx, my, fl0 = 0, fm0 = 0, fl2 = 0, fm2 = 0;
      if (R.nLoop == 0) {
        if (R.nHole == 0) {
          // the body does not cross this cell: whole cell is fluid or solid
          const bool solid = getBoundaryLevelSet(Vec3((real)px, (real)py, (real)0)) > (real)0;
          area = solid ? 0 : cellA; mx = area*px; my = area*py;
          fl0 = solid ? 0 : dyL; fm0 = fl0*py;
          fl2 = solid ? 0 : dxL; fm2 = fl2*px;
        } else {
          nHole++;
          area = cellA - R.holeArea; mx = cellA*px - R.holeMx; my = cellA*py - R.holeMy;
          fl0 = dyL; fm0 = fl0*py; fl2 = dxL; fm2 = fl2*px;
        }
      } else {
        nCell++; if (R.nLoop > 1) nSplit++;
        if (cutSplit || p1) recs.push_back({(i32)m, px, py, dxL, dyL, R});
        area = mx = my = 0;
        for (i32 l = 0; l < R.nLoop; l++) {
          const ClipLoop &L = R.loop[l];
          area += L.area; mx += L.mx; my += L.my;
          fl0 += L.faceLen[0]; fm0 += L.faceMom[0];
          fl2 += L.faceLen[2]; fm2 += L.faceMom[2];
        }
      }
      const double al = fmin(fmax(area/cellA, 0.0), 1.0);
      dAlphaMax = fmax(dAlphaMax, fabs(al - (double)A[m]));
      if (fabs(al - (double)A[m]) > 1e-12) nChanged++;
      A[m]  = (real)al;
      AX[m] = (real)fmin(fmax(fl0/dyL, 0.0), 1.0);
      AY[m] = (real)fmin(fmax(fl2/dxL, 0.0), 1.0);
      CX[m] = (real)((area > 0 ? mx/area - px : 0.0)/hR);
      CY[m] = (real)((area > 0 ? my/area - py : 0.0)/hR);
      TX[m] = (real)(fl0 > 0 ? (fm0/fl0 - py)/dyL : 0.0);
      TY[m] = (real)(fl2 > 0 ? (fm2/fl2 - px)/dxL : 0.0);
    }
  }
  printf("[cutclip] segment clipping: %d cells crossed by the body (%d split into >1 fluid loop, "
         "%d holes), %d overflow, %d bad; alpha changed on %d cells, max |dalpha| vs linear = %.3e\n",
         nCell, nSplit, nHole, nOvf, nBad, nChanged, dAlphaMax);
  if (cutSplit) buildCutSplit(recs);
  if (p1) { if (!clipRecs) clipRecs = new std::vector<ClipRec>(); clipRecs->swap(recs); }
}

// host mirror of getNbrIdx for the k = 0 plane (the merge runs on the host)
static inline i32 hostNbrIdx(const i32 *nbrIdxList, i32 bIdx, i32 i, i32 j) {
  i += blockSize; j += blockSize;
  const i32 ib = i/blockSize, jb = j/blockSize;
  const i32 nb = (ib == 1 && jb == 1) ? bIdx : nbrIdxList[27*bIdx + ib + 3*jb + 9];
  return blockSizeTot*nb + (i%blockSize) + (j%blockSize)*blockSize;
}

// --cutmerge: permanent agglomeration of small cut cells.  Union-find over the
// ACTIVE cells: every element below cutMergeFrac joins the LARGEST element
// face-adjacent to any of its members (ties to the lower index), repeated until
// nothing is small or nothing can move.  Same level only -- a cross-level lookup
// lands in a non-ACTIVE ghost cell, which is excluded -- so V is a common factor
// and the element volume is simply the alpha sum.  Then every member gets the
// element's alpha and the element's centroid (relative to its own centre), so
// the reconstruction of every face of the element extrapolates from the point
// the element average actually lives at.
void CompressibleSolver::buildCutMerge(void) {
  if (!cutMerge || !ibRccm || immerserdBcType == 0) return;
  cudaDeviceSynchronize();
  const size_t stride = (size_t)nBlocksMax*(size_t)blockSizeTot;
  if (!cutOwner) {
    cudaMallocManaged(&cutOwner,  stride*sizeof(i32));
    cudaMallocManaged(&cutAlphaE, stride*sizeof(real));
  }
  real *A = getField(F_CUTA), *CX = getField(F_CUTCX), *CY = getField(F_CUTCY);
  const i32 nC = hashTable.nKeys*blockSizeTot;
  const i32 cE = bEmpty*blockSizeTot;
  const i32 nP = (cutSplit && cutPiece) ? nCutPiece : 0;
  const i32 nN = nC + nP;                       // NODES: every cell (its piece 0), then the extra pieces
  for (size_t c = 0; c < stride; c++) { cutOwner[c] = (i32)c; cutAlphaE[c] = (c < (size_t)nC) ? A[c] : (real)0; }
  auto splitOf  = [&](i32 c) -> i32 { return (cutSplit && cutSplitId) ? cutSplitId[c] : -1; };
  auto nodeCell = [&](i32 n) -> i32 { return n < nC ? n : cutPiece[n-nC].cell; };
  auto nodeVol  = [&](i32 n) -> double {
    if (n >= nC) return (double)cutPiece[n-nC].a;
    const i32 sp = splitOf(n); return sp >= 0 ? (double)cutSplitCell[sp].a0 : (double)A[n]; };
  auto cellOk   = [&](i32 m) { return m >= 0 && m < nC && m < cE && cFlagsList[m] == ACTIVE; };
  auto nodeLive = [&](i32 n) { return cellOk(nodeCell(n)) && nodeVol(n) > 0; };
  auto pieceNode = [&](i32 c, i32 p) -> i32 {
    if (p == 0) return c;
    const i32 sp = splitOf(c); return (sp < 0 || p > cutSplitCell[sp].n) ? -1 : nC + cutSplitCell[sp].first + p - 1; };
  auto nodePiece = [&](i32 n) -> i32 {
    if (n < nC) return 0;
    const i32 sp = splitOf(cutPiece[n-nC].cell); return n - nC - cutSplitCell[sp].first + 1; };
  // face-adjacent nodes of a node: through the segment table where a face has
  // one, else piece 0 <-> piece 0 across the plain face
  // fn(neighbour node, shared open face length as a fraction of the face)
  real *AXf = getField(F_CUTAX), *AYf = getField(F_CUTAY);
  auto forEachNbr = [&](i32 node, const std::function<void(i32, double)> &fn) {
    const i32 c = nodeCell(node), p = nodePiece(node);
    const i32 b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize;
    const i32 l1 = hostNbrIdx(nbrIdxList, b, i-1, j), r1 = hostNbrIdx(nbrIdxList, b, i+1, j);
    const i32 d1 = hostNbrIdx(nbrIdxList, b, i, j-1), u1 = hostNbrIdx(nbrIdxList, b, i, j+1);
    struct Fc { i32 owner, other, dir; bool cIsOwner; };
    const Fc fc[4] = { {c, l1, 0, true}, {r1, c, 0, false}, {c, d1, 1, true}, {u1, c, 1, false} };
    for (i32 q = 0; q < 4; q++) {
      const Fc &f = fc[q];
      if (f.owner < 0 || f.owner >= cE || f.other < 0 || f.other >= cE) continue;
      const i32 id = (cutSplit && cutFaceId) ? cutFaceId[f.owner] : -1;
      const i32 nS = id < 0 ? 0 : (f.dir == 0 ? cutFace[id].nX : cutFace[id].nY);
      if (nS == 0) {
        if (p == 0) {
          const i32 nb = f.cIsOwner ? f.other : f.owner;
          const double len = (f.dir == 0) ? (double)AXf[f.owner] : (double)AYf[f.owner];   // the face's open fraction
          if (nodeLive(nb) && len > 0) fn(nb, len);
        }
        continue;
      }
      const CutFaceSeg *sg = (f.dir == 0) ? cutFace[id].sx : cutFace[id].sy;
      for (i32 s2 = 0; s2 < nS; s2++) {
        if (sg[s2].len <= 0) continue;
        const i32 mine = f.cIsOwner ? sg[s2].pC : sg[s2].pN;
        if (mine != p) continue;
        const i32 nbCell = f.cIsOwner ? f.other : f.owner, nbP = f.cIsOwner ? sg[s2].pN : sg[s2].pC;
        const i32 nb = pieceNode(nbCell, nbP);
        if (nb >= 0 && nodeLive(nb)) fn(nb, (double)sg[s2].len);
      }
    }
  };
  std::vector<i32> root(nN), best(nN); std::vector<double> elemA(nN), bestLen(nN);
  for (i32 n = 0; n < nN; n++) { root[n] = n; elemA[n] = nodeLive(n) ? nodeVol(n) : 0; }
  auto findRoot = [&](i32 c) { while (root[c] != c) c = root[c]; return c; };
  const double frac = (double)cutMergeFrac;
  i32 rounds = 0;
  for (rounds = 0; rounds < 32; rounds++) {
    for (i32 n = 0; n < nN; n++) { best[n] = -1; bestLen[n] = -1; }
    // an element (a cell or a piece, each its own DOF) must move if it is below
    // the target.  Target: the neighbour across the LARGEST shared open face (the
    // usual cut-cell merging rule -- it keeps a sliver with the cell it is best
    // connected to, and stops the last upper sliver of a thin body joining the
    // TIP cell across a partial face instead of the fully open cell above it);
    // element volume, then the lower index, only break ties.
    // a piece-rooted element (no cell in it yet) has its own, lower target: the
    // smaller side of a split cell is at most half a cell, and it should keep its
    // DOF unless it is genuinely small
    auto target = [&](i32 r) { return r >= nC ? (double)cutPieceFrac : frac; };
    for (i32 n = 0; n < nN; n++) {
      if (!nodeLive(n)) continue;
      const i32 r = findRoot(n);
      if (elemA[r] >= target(r)) continue;
      forEachNbr(n, [&](i32 m, double len) {
        const i32 rm = findRoot(m);
        if (rm == r) return;
        const bool better = best[r] < 0 || len > bestLen[r] + 1e-12 ||
          (fabs(len - bestLen[r]) <= 1e-12 && (elemA[rm] > elemA[best[r]] || (elemA[rm] == elemA[best[r]] && rm < best[r])));
        if (better) { best[r] = rm; bestLen[r] = len; }
      });
    }
    i32 nMerged = 0;
    for (i32 r = 0; r < nN; r++) {
      if (root[r] != r || best[r] < 0) continue;
      if (elemA[r] >= target(r)) continue;
      const i32 ra = findRoot(r), rb = findRoot(best[r]);
      if (ra == rb) continue;
      root[ra] = rb; nMerged++;
    }
    for (i32 n = 0; n < nN; n++) elemA[n] = 0;
    for (i32 n = 0; n < nN; n++) if (nodeLive(n)) elemA[findRoot(n)] += nodeVol(n);
    if (nMerged == 0) break;
  }
  // owner of every element: its largest node, cell or piece (ties: lowest node index)
  std::vector<i32> ownerOf(nN, -1); std::vector<double> ownerVol(nN, -1.0);
  for (i32 n = 0; n < nN; n++) {
    if (!nodeLive(n)) continue;
    const i32 r = findRoot(n); const double v = nodeVol(n);
    if (v > ownerVol[r] || (v == ownerVol[r] && n < ownerOf[r])) { ownerVol[r] = v; ownerOf[r] = n; }
  }
  auto handleOfNode = [&](i32 n) -> i32 { return n < 0 ? CUT_DEAD : (n < nC ? n : cutHandle(n - nC)); };
  // element centroids over all member nodes
  std::vector<double> cxE(nN, 0.0), cyE(nN, 0.0); std::vector<i32> nMem(nN, 0);
  for (i32 n = 0; n < nN; n++) {
    if (!nodeLive(n)) continue;
    const i32 r = findRoot(n), c = nodeCell(n);
    double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy); const double hR = fmin(dx, dy);
    const double cx = (n < nC) ? (double)CX[c] : (double)cutPiece[n-nC].cx;
    const double cy = (n < nC) ? (double)CY[c] : (double)cutPiece[n-nC].cy;
    cxE[r] += nodeVol(n)*(px + cx*hR); cyE[r] += nodeVol(n)*(py + cy*hR); nMem[r]++;
  }
  i32 nMergedCells = 0, nMergedElem = 0, maxMem = 0, nBlocked = 0, nDeadPiece = 0, nPieceDof = 0, nPieceOwner = 0; double minAE = 1e30;
  for (i32 c = 0; c < nC; c++) {
    if (!nodeLive(c)) continue;
    const i32 r = findRoot(c), o = ownerOf[r];
    cutOwner[c] = (o >= 0) ? handleOfNode(o) : c; cutAlphaE[c] = (real)elemA[r];
    if (nMem[r] > 1) {
      double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy); const double hR = fmin(dx, dy);
      CX[c] = (real)((cxE[r]/elemA[r] - px)/hR); CY[c] = (real)((cyE[r]/elemA[r] - py)/hR);
      nMergedCells++;
    }
    if (o == c) {
      if (nMem[r] > 1) { nMergedElem++; if (nMem[r] > maxMem) maxMem = nMem[r]; }
      if (elemA[r] < minAE) minAE = elemA[r];
      if (elemA[r] < frac) nBlocked++;
    }
  }
  for (i32 k = 0; k < nP; k++) {
    const i32 n = nC + k;
    CutPiece &Pk = cutPiece[k];
    if (!nodeLive(n)) { Pk.owner = CUT_DEAD; nDeadPiece++; if (cutPieceAlphaE) cutPieceAlphaE[k] = 0; continue; }
    const i32 r = findRoot(n), o = ownerOf[r];
    Pk.owner = handleOfNode(o);
    if (cutPieceAlphaE) cutPieceAlphaE[k] = (real)elemA[r];
    double px, py, dx, dy; cellGeomHost(Pk.cell, px, py, dx, dy); const double hR = fmin(dx, dy);
    if (nMem[r] > 1) { Pk.ecx = (real)((cxE[r]/elemA[r] - px)/hR); Pk.ecy = (real)((cyE[r]/elemA[r] - py)/hR); }
    else { Pk.ecx = Pk.cx; Pk.ecy = Pk.cy; }
    if (o == n) { nPieceDof++; if (nMem[r] > 1) nPieceOwner++; if (elemA[r] < minAE) minAE = elemA[r]; if (nMem[r] > 1) { nMergedElem++; if (nMem[r] > maxMem) maxMem = nMem[r]; } }
  }
  if (cutSplit && cutFace) {
    auto ownerHandleOfNode = [&](i32 n) -> i32 { return (n >= 0 && nodeLive(n)) ? handleOfNode(ownerOf[findRoot(n)]) : CUT_DEAD; };
    for (i32 f = 0; f < nCutFace; f++) {
      CutFace &F = cutFace[f];
      const i32 c = F.cell, b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize;
      const i32 l1 = hostNbrIdx(nbrIdxList, b, i-1, j), d1 = hostNbrIdx(nbrIdxList, b, i, j-1);
      for (i32 s2 = 0; s2 < F.nX; s2++) { F.sx[s2].ownC = ownerHandleOfNode(pieceNode(c, F.sx[s2].pC)); F.sx[s2].ownN = (l1 < cE) ? ownerHandleOfNode(pieceNode(l1, F.sx[s2].pN)) : CUT_DEAD; }
      for (i32 s2 = 0; s2 < F.nY; s2++) { F.sy[s2].ownC = ownerHandleOfNode(pieceNode(c, F.sy[s2].pC)); F.sy[s2].ownN = (d1 < cE) ? ownerHandleOfNode(pieceNode(d1, F.sy[s2].pN)) : CUT_DEAD; }
    }
  }
  printf("[cutmerge] frac %.2f (pieces %.2f): %d cells in %d merged elements (max %d nodes), %d rounds, "
         "min alpha_E %.3e, %d elements still below the target; %d extra pieces: %d own DOFs (%d of them element owners), %d dead\n",
         frac, (double)cutPieceFrac, nMergedCells, nMergedElem, maxMem, rounds, minAE, nBlocked, nP, nPieceDof, nPieceOwner, nDeadPiece);
}

// ---- split cells: piece records and face segments (--cutsplit) --------------
//
// Built on the host from the clipper's loops, once per geometry stamp.  The
// LARGEST loop of a cell is piece 0 and keeps the cell's DOF; every other loop
// is an extra piece (CutPiece) that buildCutMerge attaches to a neighbour
// element on its own side of the wall.  A face whose open part is not one
// single (piece 0 <-> piece 0) interval gets a CutFace entry under the cell
// whose LOW face it is: the segments are the (pC, pN) groups of the elementary
// sub-intervals, classified by midpoint against BOTH cells' loops -- both cells
// clipped the same segments, so the endpoints agree to roundoff.

void CompressibleSolver::cellGeomHost(i32 c, double &px, double &py, double &dx, double &dy) {
  const i32 b = c/blockSizeTot, cc = c%blockSizeTot;
  i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
  dx = (double)domainSize[0]/(baseGridSize[0]*powi(2,lvl));
  dy = (double)domainSize[1]/(baseGridSize[1]*powi(2,lvl));
  px = (ib*blockSize + cc%blockSize + 0.5)*dx;
  py = (jb*blockSize + (cc/blockSize)%blockSize + 0.5)*dy;
}

void CompressibleSolver::buildCutSplit(std::vector<ClipRec> &recs) {
  const size_t stride = (size_t)nBlocksMax*(size_t)blockSizeTot;
  if (!cutSplitId) { cudaMallocManaged(&cutSplitId, stride*sizeof(i32)); cudaMallocManaged(&cutFaceId, stride*sizeof(i32)); }
  for (size_t c = 0; c < stride; c++) { cutSplitId[c] = -1; cutFaceId[c] = -1; }
  nCutSplit = nCutPiece = nCutFace = 0;
  i32 nIntFace = 0;
  auto growSplit = [&]() { if (nCutSplit + 1 > cutSplitCap) { i32 nc = cutSplitCap ? 2*cutSplitCap : 256; CutSplitCell *n; cudaMallocManaged(&n, (size_t)nc*sizeof(CutSplitCell)); if (cutSplitCell) { memcpy(n, cutSplitCell, (size_t)nCutSplit*sizeof(CutSplitCell)); cudaFree(cutSplitCell); } cutSplitCell = n; cutSplitCap = nc; } };
  auto growPiece = [&]() { if (nCutPiece + 1 > cutPieceCap) { i32 nc = cutPieceCap ? 2*cutPieceCap : 512; CutPiece *n; cudaMallocManaged(&n, (size_t)nc*sizeof(CutPiece)); if (cutPiece) { memcpy(n, cutPiece, (size_t)nCutPiece*sizeof(CutPiece)); cudaFree(cutPiece); } cutPiece = n; cutPieceCap = nc; } };
  auto growFace  = [&]() { if (nCutFace + 1 > cutFaceCap) { i32 nc = cutFaceCap ? 2*cutFaceCap : 512; CutFace *n; cudaMallocManaged(&n, (size_t)nc*sizeof(CutFace)); if (cutFace) { memcpy(n, cutFace, (size_t)nCutFace*sizeof(CutFace)); cudaFree(cutFace); } cutFace = n; cutFaceCap = nc; } };
  real *A = getField(F_CUTA), *CX = getField(F_CUTCX), *CY = getField(F_CUTCY);
  const i32 cE = bEmpty*blockSizeTot, nC = hashTable.nKeys*blockSizeTot;
  std::unordered_map<i32, i32> recOf;
  std::vector<std::vector<i32>> order(recs.size());
  for (size_t k = 0; k < recs.size(); k++) {
    const ClipRec &r = recs[k]; recOf[r.cell] = (i32)k;
    order[k].resize(r.R.nLoop);
    for (i32 l = 0; l < r.R.nLoop; l++) order[k][l] = l;
    std::sort(order[k].begin(), order[k].end(), [&](i32 a, i32 b){ return r.R.loop[a].area > r.R.loop[b].area; });
    if (r.R.nLoop <= 1) continue;
    const double cellA = r.dx*r.dy, hR = fmin(r.dx, r.dy);
    growSplit();
    CutSplitCell &S = cutSplitCell[nCutSplit];
    auto fill = [&](const ClipLoop &L, real &a, real &cx, real &cy, real &wnx, real &wny, real &wcx, real &wcy) {
      a  = (real)(L.area/cellA);
      cx = (real)((L.mx/L.area - r.px)/hR); cy = (real)((L.my/L.area - r.py)/hR);
      // the WALL edges' own outward normal sum (exact for the polygon); equals
      // -(faces + internal face) by the per-piece GCL
      wnx = (real)(L.wallVx/r.dy);
      wny = (real)(L.wallVy/r.dx);
      if (L.wallLen > 0) { wcx = (real)((L.wallMx/L.wallLen - r.px)/hR); wcy = (real)((L.wallMy/L.wallLen - r.py)/hR); }
      else { wcx = cx; wcy = cy; }
    };
    fill(r.R.loop[order[k][0]], S.a0, S.cx0, S.cy0, S.wnx0, S.wny0, S.wcx0, S.wcy0);
    S.first = nCutPiece; S.n = r.R.nLoop - 1;
    // internal face: the two loops that carry an extension (normal from the
    // lower-piece-index one into the other)
    S.iLen = 0; S.icx = S.icy = S.inx = S.iny = 0; S.iPa = S.iPb = -1;
    for (i32 pa = 0; pa < r.R.nLoop && S.iPa < 0; pa++) {
      const ClipLoop &La = r.R.loop[order[k][pa]];
      if (La.intLen <= 0) continue;
      for (i32 pb = pa + 1; pb < r.R.nLoop; pb++) {
        const ClipLoop &Lb = r.R.loop[order[k][pb]];
        if (Lb.intLen <= 0) continue;
        S.iPa = pa; S.iPb = pb; S.iLen = (real)La.intLen;
        S.icx = (real)((La.intMx/La.intLen - r.px)/hR); S.icy = (real)((La.intMy/La.intLen - r.py)/hR);
        const double nl = sqrt(La.intNx*La.intNx + La.intNy*La.intNy);
        S.inx = (real)(nl > 0 ? La.intNx/nl : 0); S.iny = (real)(nl > 0 ? La.intNy/nl : 0);
        nIntFace++;
        break;
      }
    }
    for (i32 p = 1; p < r.R.nLoop; p++) {
      growPiece();
      CutPiece &P = cutPiece[nCutPiece++];
      P.cell = r.cell; P.owner = CUT_DEAD;
      fill(r.R.loop[order[k][p]], P.a, P.cx, P.cy, P.wnx, P.wny, P.wcx, P.wcy);
      P.ecx = P.cx; P.ecy = P.cy;
    }
    cutSplitId[r.cell] = nCutSplit++;
    CX[r.cell] = S.cx0; CY[r.cell] = S.cy0;      // the DOF piece's centroid, not the union's
  }
  // ---- faces ----------------------------------------------------------------
  struct Iv { double lo, hi; i32 piece; };
  // open intervals of `cell` on its face f (0 lo-x 1 hi-x 2 lo-y 3 hi-y); false if no such cell
  auto intervals = [&](i32 cell, i32 f, std::vector<Iv> &out) -> bool {
    out.clear();
    if (cell < 0 || cell >= nC || cell >= cE) return false;
    auto it = recOf.find(cell);
    if (it == recOf.end()) { if (A[cell] > (real)0) out.push_back({-1e300, 1e300, 0}); return true; }
    const ClipRec &r = recs[it->second];
    if (r.R.nLoop == 0) { if (A[cell] > (real)0) out.push_back({-1e300, 1e300, 0}); return true; }   // hole-only cell
    for (i32 pi = 0; pi < r.R.nLoop; pi++) {
      const ClipLoop &L = r.R.loop[order[it->second][pi]];
      for (i32 q = 0; q < L.nIv[f]; q++) out.push_back({L.iv[f][q][0], L.iv[f][q][1], pi});
    }
    return true;
  };
  std::unordered_set<i64> done;
  i32 nDrop = 0, nOvf = 0, nSegTot = 0;
  auto doFace = [&](i32 ownerCell, i32 dir, i32 otherCell) {
    if (ownerCell < 0 || ownerCell >= nC || ownerCell >= cE) return;
    const i64 key = (i64)ownerCell*2 + dir;
    if (!done.insert(key).second) return;
    std::vector<Iv> ivA, ivB;
    if (!intervals(ownerCell, dir == 0 ? 0 : 2, ivA)) return;
    if (!intervals(otherCell, dir == 0 ? 1 : 3, ivB)) return;
    if (ivA.empty() || ivB.empty()) return;                      // closed face
    double px, py, dx, dy; cellGeomHost(ownerCell, px, py, dx, dy);
    const double fLo = (dir == 0) ? py - 0.5*dy : px - 0.5*dx;
    const double fHi = (dir == 0) ? py + 0.5*dy : px + 0.5*dx;
    const double fLen = fHi - fLo, fMid = 0.5*(fLo + fHi), tol = 1e-12*fLen;
    std::vector<double> pts = {fLo, fHi};
    for (auto &v : ivA) { pts.push_back(fmin(fmax(v.lo, fLo), fHi)); pts.push_back(fmin(fmax(v.hi, fLo), fHi)); }
    for (auto &v : ivB) { pts.push_back(fmin(fmax(v.lo, fLo), fHi)); pts.push_back(fmin(fmax(v.hi, fLo), fHi)); }
    std::sort(pts.begin(), pts.end());
    struct G { i32 pA, pB; double len, mom; };
    std::vector<G> groups;
    auto pieceAt = [&](const std::vector<Iv> &iv, double s) { for (auto &v : iv) if (s >= v.lo - tol && s <= v.hi + tol) return v.piece; return -1; };
    for (size_t q = 0; q + 1 < pts.size(); q++) {
      const double s0 = pts[q], s1 = pts[q+1];
      if (s1 - s0 <= tol) continue;
      const double mid = 0.5*(s0 + s1);
      const i32 pA = pieceAt(ivA, mid), pB = pieceAt(ivB, mid);
      if (pA < 0 || pB < 0) { if (pA >= 0 || pB >= 0) nDrop++; continue; }
      bool found = false;
      for (auto &g : groups) if (g.pA == pA && g.pB == pB) { g.len += s1 - s0; g.mom += (s1 - s0)*mid; found = true; break; }
      if (!found) groups.push_back({pA, pB, s1 - s0, (s1 - s0)*mid});
    }
    if (groups.empty()) return;
    if (groups.size() == 1 && groups[0].pA == 0 && groups[0].pB == 0) return;   // plain face
    i32 id = cutFaceId[ownerCell];
    if (id < 0) { growFace(); id = nCutFace++; CutFace &F = cutFace[id]; F.nX = F.nY = 0; F.cell = ownerCell; cutFaceId[ownerCell] = id; }
    CutFace &F = cutFace[id];
    CutFaceSeg *sg = (dir == 0) ? F.sx : F.sy;
    i32 &n = (dir == 0) ? F.nX : F.nY;
    sg[0] = {0, 0, 0, 0, -1, -1}; n = 1;
    for (auto &g : groups) {
      const real len = (real)(g.len/fLen), cen = (real)((g.mom/g.len - fMid)/fLen);
      if (g.pA == 0 && g.pB == 0) { sg[0].len = len; sg[0].cen = cen; }
      else if (n < 4) { sg[n++] = {len, cen, g.pA, g.pB, -1, -1}; }
      else nOvf++;
    }
    nSegTot += n;
  };
  for (auto &r : recs) {
    const i32 b = r.cell/blockSizeTot, cc = r.cell%blockSizeTot;
    const i32 i = cc%blockSize, j = (cc/blockSize)%blockSize;
    doFace(r.cell, 0, hostNbrIdx(nbrIdxList, b, i-1, j));
    doFace(r.cell, 1, hostNbrIdx(nbrIdxList, b, i, j-1));
    doFace(hostNbrIdx(nbrIdxList, b, i+1, j), 0, r.cell);
    doFace(hostNbrIdx(nbrIdxList, b, i, j+1), 1, r.cell);
  }
  // piece-resident DOFs: (re)allocate with the piece capacity and seed every
  // piece with its cell's state.  (A re-stamp after adaptation re-seeds from
  // the cell: piece states do not yet survive a sort.)
  if (nCutPiece > 0) {
    if (cutPieceQCap < cutPieceCap) {
      if (cutPieceQ) { cudaFree(cutPieceQ); cudaFree(cutPieceS); cudaFree(cutPieceAlphaE); }
      cutPieceQCap = cutPieceCap;
      cudaMallocManaged(&cutPieceQ, (size_t)NEVOLVE*cutPieceQCap*sizeof(real));
      cudaMallocManaged(&cutPieceS, (size_t)NEVOLVE*cutPieceQCap*sizeof(real));
      cudaMallocManaged(&cutPieceAlphaE, (size_t)cutPieceQCap*sizeof(real));
      if (p1) {
        if (cutPieceSX) { cudaFree(cutPieceSX); cudaFree(cutPieceSR); }
        cudaMallocManaged(&cutPieceSX, (size_t)2*P1_NV*cutPieceQCap*sizeof(real));
        cudaMallocManaged(&cutPieceSR, (size_t)2*P1_NV*cutPieceQCap*sizeof(real));
      }
    }
    if (p1 && cutPieceSX)
      for (i32 k = 0; k < nCutPiece; k++)
        for (i32 s = 0; s < 2*P1_NV; s++) { cutPieceSX[(size_t)s*cutPieceQCap + k] = 0; cutPieceSR[(size_t)s*cutPieceQCap + k] = 0; }
    for (i32 k = 0; k < nCutPiece; k++)
      for (i32 f = 0; f < NEVOLVE; f++) {
        cutPieceQ[(size_t)f*cutPieceQCap + k] = getField(f)[cutPiece[k].cell];
        cutPieceS[(size_t)f*cutPieceQCap + k] = 0;
      }
  }
  i32 nAct = 0; for (i32 k = 0; k < nCutSplit; k++) { const i32 c = cutPiece[cutSplitCell[k].first].cell; if (c < cE && cFlagsList[c] == ACTIVE) nAct++; }
  printf("[cutsplit] %d split cells (%d ACTIVE, %d with an internal tip face), %d extra pieces, %d cells with segmented faces (%d segments), %d one-sided slivers dropped, %d overflow\n",
         nCutSplit, nAct, nIntFace, nCutPiece, nCutFace, nSegTot, nDrop, nOvf);
}

// ---- --p1 cut elements --------------------------------------------------------
// Built on the host after the merge (it needs the owners), once per geometry
// stamp, from the clipper's loop polygons.  Every DOF element with cut geometry
// -- a clipped cell, a merged element (owner cell or piece with its members),
// an uncut cell that absorbed a sliver -- gets a P1Elem: the union of its
// loops' polygons gives the centroid, the fluid area and the second moments
// (the slope mass matrix, stored inverted), a fan of 3-point triangle rules
// from each loop's centroid gives the volume rule (exact for quadratics), and
// every wall edge gets 2 Gauss points carrying its outward normal * length.
// Every open face interval of a clipped cell becomes a P1Seg between the two
// elements on its sides (a high face only when the neighbour is not clipped:
// it emits that face as its low face); the plain faces of an uncut owner and
// the slit-tip internal faces are segments too.  The per-cell flag p1Irr marks
// every cell whose faces live in this table, so the regular kernel skips them.
void CompressibleSolver::buildP1Cut(void) {
  if (!p1 || !ibRccm || immerserdBcType == 0 || cutGeom != 3 || !clipRecs) return;
  cudaDeviceSynchronize();
  const size_t stride = (size_t)nBlocksMax*(size_t)blockSizeTot;
  if (!p1ElemOfCell) { cudaMallocManaged(&p1ElemOfCell, stride*sizeof(i32)); cudaMallocManaged(&p1Irr, stride*sizeof(i32)); }
  for (size_t c = 0; c < stride; c++) { p1ElemOfCell[c] = -1; p1Irr[c] = 0; }
  if (cutPieceCap > p1PieceCap) {
    if (p1ElemOfPiece) cudaFree(p1ElemOfPiece);
    cudaMallocManaged(&p1ElemOfPiece, (size_t)cutPieceCap*sizeof(i32)); p1PieceCap = cutPieceCap;
  }
  for (i32 k = 0; k < p1PieceCap; k++) p1ElemOfPiece[k] = -1;
  const std::vector<ClipRec> &recs = *clipRecs;
  real *A = getField(F_CUTA);
  const i32 nC = hashTable.nKeys*blockSizeTot, cE = bEmpty*blockSizeTot;
  std::unordered_map<i32, i32> recOf;
  std::vector<std::vector<i32>> order(recs.size());
  for (size_t k = 0; k < recs.size(); k++) {
    const ClipRec &r = recs[k]; recOf[r.cell] = (i32)k;
    order[k].resize(r.R.nLoop);
    for (i32 l = 0; l < r.R.nLoop; l++) order[k][l] = l;
    std::sort(order[k].begin(), order[k].end(), [&](i32 a, i32 b){ return r.R.loop[a].area > r.R.loop[b].area; });
  }
  auto ownerCell = [&](i32 c) -> i32 { return (cutMerge && cutOwner) ? cutOwner[c] : c; };
  auto handleOf  = [&](i32 c, i32 p) -> i32 {          // DOF handle of (cell, piece rank)
    if (p == 0) return ownerCell(c);
    const i32 sp = (cutSplit && cutSplitId) ? cutSplitId[c] : -1;
    if (sp < 0 || p > cutSplitCell[sp].n) return CUT_DEAD;
    return cutPiece[cutSplitCell[sp].first + p - 1].owner; };
  auto cellOfHandle = [&](i32 h) -> i32 { return cutIsPiece(h) ? cutPiece[cutPieceOf(h)].cell : h; };
  auto liveCell = [&](i32 c) { return c >= 0 && c < nC && c < cE && cFlagsList[c] == ACTIVE; };
  // ---- irregular cells --------------------------------------------------------
  std::vector<char> hasMember(nC, 0);
  if (cutMerge && cutOwner)
    for (i32 c = 0; c < nC; c++) { const i32 o = cutOwner[c]; if (o != c && o >= 0 && o < nC) hasMember[o] = 1; }
  if (cutSplit && cutPiece)      // a cell that absorbed a split-cell piece owns cut geometry too
    for (i32 k = 0; k < nCutPiece; k++) { const i32 o = cutPiece[k].owner; if (o >= 0 && o < nC) hasMember[o] = 1; }
  for (i32 c = 0; c < nC && c < cE; c++) {
    const bool inRec  = recOf.count(c) > 0;
    const bool dead   = !(A[c] > (real)0);
    const bool member = cutOwner ? (cutOwner[c] != c) : false;
    if (inRec || dead || member || hasMember[c]) p1Irr[c] = 1;
  }
  // ---- element polygons -----------------------------------------------------
  struct Poly { std::vector<double> x, y; std::vector<signed char> k; double area = 0, mx = 0, my = 0; };
  std::vector<i32> elemHandle; std::unordered_map<i32, i32> elemOf; std::vector<std::vector<Poly>> elemLoops;
  auto elemFor = [&](i32 h) -> i32 {
    auto it = elemOf.find(h); if (it != elemOf.end()) return it->second;
    const i32 e = (i32)elemHandle.size(); elemHandle.push_back(h); elemOf[h] = e; elemLoops.emplace_back(); return e; };
  auto polyBox = [&](i32 c, Poly &P) {
    double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy);
    P.x = {px-0.5*dx, px+0.5*dx, px+0.5*dx, px-0.5*dx}; P.y = {py-0.5*dy, py-0.5*dy, py+0.5*dy, py+0.5*dy}; P.k = {1,1,1,1}; };
  i32 nOvfLoop = 0, nHoleCell = 0, nBadM = 0;
  for (i32 c = 0; c < nC && c < cE; c++) {
    if (!liveCell(c) || !p1Irr[c] || !(A[c] > (real)0)) continue;
    auto it = recOf.find(c);
    if (it == recOf.end()) {                       // uncut owner of a merged element: the full box
      const i32 h = ownerCell(c); if (h == CUT_DEAD) continue;
      Poly P; polyBox(c, P); elemLoops[elemFor(h)].push_back(P); continue;
    }
    const ClipRec &r = recs[it->second];
    if (r.R.nLoop == 0) {                          // hole-only cell: the box (the hole is ignored by the slope rule)
      nHoleCell++; const i32 h = ownerCell(c); if (h == CUT_DEAD) continue;
      Poly P; polyBox(c, P); elemLoops[elemFor(h)].push_back(P); continue;
    }
    for (i32 p = 0; p < r.R.nLoop; p++) {
      const ClipLoop &L = r.R.loop[order[it->second][p]];
      const i32 h = handleOf(c, p); if (h == CUT_DEAD) continue;
      if (L.vOvf || L.nv < 3) { nOvfLoop++; continue; }
      Poly P; P.x.assign(L.vx, L.vx + L.nv); P.y.assign(L.vy, L.vy + L.nv); P.k.assign(L.ek, L.ek + L.nv);
      elemLoops[elemFor(h)].push_back(P);
    }
  }
  // ---- per element: area, centroid, second moments, quadrature -----------------
  nP1Elem = (i32)elemHandle.size();
  if (nP1Elem > p1ElemCap) { if (p1Elem) cudaFree(p1Elem); p1ElemCap = 256; while (p1ElemCap < nP1Elem) p1ElemCap *= 2; cudaMallocManaged(&p1Elem, (size_t)p1ElemCap*sizeof(P1Elem)); }
  std::vector<P1Qpt> qpts;
  const double G1 = 0.28867513459481287;
  double aErrMax = 0;
  for (i32 e = 0; e < nP1Elem; e++) {
    P1Elem &E = p1Elem[e]; E.handle = elemHandle[e];
    double area = 0, mx = 0, my = 0, ixx = 0, iyy = 0, ixy = 0;
    for (auto &P : elemLoops[e]) {
      const size_t n = P.x.size(); double a = 0, m1 = 0, m2 = 0;
      for (size_t v = 0; v < n; v++) {
        const size_t w = (v + 1) % n;
        const double xa = P.x[v], ya = P.y[v], xb = P.x[w], yb = P.y[w], cr = xa*yb - xb*ya;
        a += 0.5*cr; m1 += (xa + xb)*cr/6.0; m2 += (ya + yb)*cr/6.0;
        ixx += (xa*xa + xa*xb + xb*xb)*cr/12.0;
        iyy += (ya*ya + ya*yb + yb*yb)*cr/12.0;
        ixy += (xa*yb + 2.0*xa*ya + 2.0*xb*yb + xb*ya)*cr/24.0;
      }
      P.area = a; P.mx = m1; P.my = m2; area += a; mx += m1; my += m2;
    }
    const double gx = mx/area, gy = my/area;
    const double Ixx = ixx - area*gx*gx, Iyy = iyy - area*gy*gy, Ixy = ixy - area*gx*gy;
    double px, py, dx, dy; cellGeomHost(cellOfHandle(E.handle), px, py, dx, dy); const double h = fmin(dx, dy);
    const double M11 = Ixx/(h*h), M22 = Iyy/(h*h), M12 = Ixy/(h*h), det = M11*M22 - M12*M12;
    E.gx = (real)gx; E.gy = (real)gy; E.h = (real)h; E.area = (real)area;
    if (area > 0 && det > 1e-12*(M11*M22 + 1e-300)) { E.m11 = (real)(M22/det); E.m22 = (real)(M11/det); E.m12 = (real)(-M12/det); }
    else { E.m11 = E.m22 = (real)(12.0/(dx*dy)); E.m12 = 0; nBadM++; }
    // consistency: the polygon area must be the stamped element volume
    const double aStamp = cutIsPiece(E.handle) ? (double)cutPieceAlphaE[cutPieceOf(E.handle)]*dx*dy : (double)(cutAlphaE ? cutAlphaE[E.handle] : A[E.handle])*dx*dy;
    if (fabs(area - aStamp)/(dx*dy) > fmax(aErrMax, 1e-8) && cutDbg) {
      printf("[p1cut] element %d handle %d cell %d: polygon area %.6f cells vs stamped %.6f, %zu loops:\n", e, E.handle, cellOfHandle(E.handle), area/(dx*dy), aStamp/(dx*dy), elemLoops[e].size());
      for (auto &P : elemLoops[e]) { printf("   loop area %.6f:", P.area/(dx*dy)); for (size_t v = 0; v < P.x.size(); v++) printf(" (%.4f,%.4f)k%d", (P.x[v]-px)/dx, (P.y[v]-py)/dy, (i32)P.k[v]); printf("\n"); }
    }
    aErrMax = fmax(aErrMax, fabs(area - aStamp)/(dx*dy));
    E.q0 = (i32)qpts.size();
    for (auto &P : elemLoops[e]) {
      const size_t n = P.x.size(); const double ax = P.mx/P.area, ay = P.my/P.area;   // apex: the loop centroid
      for (size_t v = 0; v < n; v++) {
        const size_t w = (v + 1) % n;
        const double xa = P.x[v], ya = P.y[v], xb = P.x[w], yb = P.y[w];
        const double t = 0.5*((xa - ax)*(yb - ay) - (xb - ax)*(ya - ay));
        if (t != 0) {
          qpts.push_back({(real)(0.5*(ax + xa)), (real)(0.5*(ay + ya)), (real)(t/3.0), 0, 0});
          qpts.push_back({(real)(0.5*(xa + xb)), (real)(0.5*(ya + yb)), (real)(t/3.0), 0, 0});
          qpts.push_back({(real)(0.5*(xb + ax)), (real)(0.5*(yb + ay)), (real)(t/3.0), 0, 0});
        }
        qpts.push_back({(real)xa, (real)ya, 0, 0, 0});   // polygon vertex: a limiter CHECK point (w = 0, n = 0: inert in the RHS)
        if (P.k[v] == 0) {                       // wall edge: 2 Gauss points, outward normal (ey,-ex) * half length
          const double ex = xb - xa, ey = yb - ya;
          for (i32 g = 0; g < 2; g++) {
            const double sg = 0.5 + (g ? G1 : -G1);
            qpts.push_back({(real)(xa + sg*ex), (real)(ya + sg*ey), 0, (real)(0.5*ey), (real)(-0.5*ex)});
          }
        }
      }
    }
    E.nq = (i32)qpts.size() - E.q0;
    if (cutIsPiece(E.handle)) p1ElemOfPiece[cutPieceOf(E.handle)] = e; else p1ElemOfCell[E.handle] = e;
  }
  // ---- face segments ---------------------------------------------------------
  std::vector<P1Seg> segs; i32 nDrop = 0;
  auto pieceAtFace = [&](i32 n, i32 f, double mid, i32 &pOut) -> bool {
    auto it = recOf.find(n);
    if (it == recOf.end() || recs[it->second].R.nLoop == 0) { pOut = 0; return A[n] > (real)0; }
    const ClipRec &r = recs[it->second]; const double tol = 1e-12*(r.dx + r.dy);
    for (i32 q = 0; q < r.R.nLoop; q++) {
      const ClipLoop &L = r.R.loop[order[it->second][q]];
      for (i32 v = 0; v < L.nIv[f]; v++)
        if (mid >= L.iv[f][v][0] - tol && mid <= L.iv[f][v][1] + tol) { pOut = q; return true; }
    }
    return false; };
  auto emit = [&](i32 hLow, i32 hHigh, i32 dir, double xf, double lo, double hi) {
    if (hLow == CUT_DEAD || hHigh == CUT_DEAD || hLow == hHigh || hi - lo <= 0) return;
    P1Seg S;
    if (dir == 0) { S.x0 = S.x1 = (real)xf; S.y0 = (real)lo; S.y1 = (real)hi; S.nx = 1; S.ny = 0; }
    else          { S.y0 = S.y1 = (real)xf; S.x0 = (real)lo; S.x1 = (real)hi; S.nx = 0; S.ny = 1; }
    S.hA = hLow; S.hB = hHigh; segs.push_back(S); };
  for (auto &r : recs) {
    const i32 c = r.cell; if (c < 0 || c >= cE) continue;
    const i32 b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize;
    const i32 nb[4] = { hostNbrIdx(nbrIdxList, b, i-1, j), hostNbrIdx(nbrIdxList, b, i+1, j),
                        hostNbrIdx(nbrIdxList, b, i, j-1), hostNbrIdx(nbrIdxList, b, i, j+1) };
    const i32 rk = recOf[c];
    for (i32 f = 0; f < 4; f++) {
      const i32 n = nb[f]; if (n < 0 || n >= cE) continue;
      if ((f == 1 || f == 3) && recOf.count(n)) continue;          // the clipped neighbour emits it as its low face
      if (cFlagsList[c] != ACTIVE && cFlagsList[n] != ACTIVE) continue;
      const i32 fo = f ^ 1, dir = (f <= 1) ? 0 : 1;
      const double xf = (f == 0) ? r.px - 0.5*r.dx : (f == 1) ? r.px + 0.5*r.dx : (f == 2) ? r.py - 0.5*r.dy : r.py + 0.5*r.dy;
      for (i32 p = 0; p < r.R.nLoop; p++) {
        const ClipLoop &L = r.R.loop[order[rk][p]];
        for (i32 v = 0; v < L.nIv[f]; v++) {
          const double lo = L.iv[f][v][0], hi = L.iv[f][v][1], mid = 0.5*(lo + hi);
          i32 pn; if (!pieceAtFace(n, fo, mid, pn)) { nDrop++; continue; }
          const i32 hc = handleOf(c, p), hn = handleOf(n, pn);
          if (f == 0 || f == 2) emit(hn, hc, dir, xf, lo, hi); else emit(hc, hn, dir, xf, lo, hi);
        }
      }
    }
  }
  // uncut owners of merged elements: their plain faces (a clipped neighbour emitted its own)
  for (i32 c = 0; c < nC && c < cE; c++) {
    if (!liveCell(c) || !p1Irr[c] || recOf.count(c) || !(A[c] > (real)0)) continue;
    const i32 b = c/blockSizeTot, cc = c%blockSizeTot, i = cc%blockSize, j = (cc/blockSize)%blockSize;
    const i32 nb[4] = { hostNbrIdx(nbrIdxList, b, i-1, j), hostNbrIdx(nbrIdxList, b, i+1, j),
                        hostNbrIdx(nbrIdxList, b, i, j-1), hostNbrIdx(nbrIdxList, b, i, j+1) };
    double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy);
    for (i32 f = 0; f < 4; f++) {
      const i32 n = nb[f]; if (n < 0 || n >= cE || recOf.count(n) || !(A[n] > (real)0)) continue;
      if (p1Irr[n] && n < c) continue;                              // the lower-index irregular cell emits
      const i32 dir = (f <= 1) ? 0 : 1;
      const double xf = (f == 0) ? px - 0.5*dx : (f == 1) ? px + 0.5*dx : (f == 2) ? py - 0.5*dy : py + 0.5*dy;
      const double lo = (dir == 0) ? py - 0.5*dy : px - 0.5*dx, hi = (dir == 0) ? py + 0.5*dy : px + 0.5*dx;
      const i32 hc = ownerCell(c), hn = ownerCell(n);
      if (f == 0 || f == 2) emit(hn, hc, dir, xf, lo, hi); else emit(hc, hn, dir, xf, lo, hi);
    }
  }
  // slit-tip internal faces
  i32 nTip = 0;
  if (cutSplit && cutSplitCell)
    for (i32 sp = 0; sp < nCutSplit; sp++) {
      const CutSplitCell &Sc = cutSplitCell[sp];
      if (Sc.iLen <= (real)0 || Sc.iPa < 0) continue;
      const i32 c = cutPiece[Sc.first].cell; if (!liveCell(c)) continue;
      double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy); const double hR = fmin(dx, dy);
      const double cx = px + Sc.icx*hR, cy = py + Sc.icy*hR, tx = -Sc.iny, ty = Sc.inx, L = Sc.iLen;
      const i32 hA = handleOf(c, Sc.iPa), hB = handleOf(c, Sc.iPb);
      if (hA == CUT_DEAD || hB == CUT_DEAD || hA == hB) continue;
      P1Seg S; S.x0 = (real)(cx - 0.5*L*tx); S.y0 = (real)(cy - 0.5*L*ty); S.x1 = (real)(cx + 0.5*L*tx); S.y1 = (real)(cy + 0.5*L*ty);
      S.nx = Sc.inx; S.ny = Sc.iny; S.hA = hA; S.hB = hB; segs.push_back(S); nTip++;
    }
  // ---- consistency: every element's open boundary must be covered by face pieces ----
  // (open cell-boundary edges of its loops, kind 1, plus its share of the internal faces)
  {
    std::vector<double> openLen(nP1Elem, 0.0), segLen(nP1Elem, 0.0);
    for (i32 e = 0; e < nP1Elem; e++)
      for (auto &P : elemLoops[e]) { const size_t n = P.x.size();
        for (size_t v = 0; v < n; v++) { if (P.k[v] != 1) continue; const size_t w = (v+1)%n;
          openLen[e] += sqrt((P.x[w]-P.x[v])*(P.x[w]-P.x[v]) + (P.y[w]-P.y[v])*(P.y[w]-P.y[v])); } }
    // an uncut owner's box edges against a MEMBER of its own element are internal, not open:
    // subtract them (they were counted as kind 1 by polyBox)
    for (auto &Sg : segs) { const double L = sqrt((double)(Sg.x1-Sg.x0)*(Sg.x1-Sg.x0) + (double)(Sg.y1-Sg.y0)*(Sg.y1-Sg.y0));
      auto it = elemOf.find(Sg.hA); if (it != elemOf.end()) segLen[it->second] += L;
      it = elemOf.find(Sg.hB); if (it != elemOf.end()) segLen[it->second] += L; }
    double worst = 0; i32 we = -1;
    for (i32 e = 0; e < nP1Elem; e++) { const double d = fabs(openLen[e] - segLen[e])/fmax(p1Elem[e].h, 1e-300); if (d > worst) { worst = d; we = e; } }
    printf("[p1cut] open-boundary coverage: worst |open - pieces| = %.3e h (element %d, handle %d: open %.4f h, pieces %.4f h)\n",
           worst, we, we >= 0 ? p1Elem[we].handle : -1, we >= 0 ? openLen[we]/p1Elem[we].h : 0.0, we >= 0 ? segLen[we]/p1Elem[we].h : 0.0);
  }
  // ---- element adjacency through the face pieces (limiter) ----------------------
  {
    std::vector<std::vector<i32>> nb(nP1Elem);
    for (auto &Sg : segs) {
      auto ia = elemOf.find(Sg.hA), ib2 = elemOf.find(Sg.hB);
      if (ia != elemOf.end()) nb[ia->second].push_back(Sg.hB);
      if (ib2 != elemOf.end()) nb[ib2->second].push_back(Sg.hA);
    }
    i32 tot = 0; for (auto &v : nb) tot += (i32)v.size();
    if (nP1Elem + 1 > p1ElemCap + 1 || !p1ElemNbrOff) { if (p1ElemNbrOff) cudaFree(p1ElemNbrOff); cudaMallocManaged(&p1ElemNbrOff, (size_t)(p1ElemCap + 1)*sizeof(i32)); }
    if (tot > p1NbrCap) { if (p1ElemNbr) cudaFree(p1ElemNbr); p1NbrCap = 1024; while (p1NbrCap < tot) p1NbrCap *= 2; cudaMallocManaged(&p1ElemNbr, (size_t)p1NbrCap*sizeof(i32)); }
    i32 o = 0;
    for (i32 e = 0; e < nP1Elem; e++) { p1ElemNbrOff[e] = o; for (i32 h : nb[e]) p1ElemNbr[o++] = h; }
    p1ElemNbrOff[nP1Elem] = o;
  }
  // ---- upload -------------------------------------------------------------------
  nP1Seg = (i32)segs.size();
  if (nP1Seg > p1SegCap) { if (p1Seg) cudaFree(p1Seg); p1SegCap = 1024; while (p1SegCap < nP1Seg) p1SegCap *= 2; cudaMallocManaged(&p1Seg, (size_t)p1SegCap*sizeof(P1Seg)); }
  for (i32 s = 0; s < nP1Seg; s++) p1Seg[s] = segs[s];
  nP1Qpt = (i32)qpts.size();
  if (nP1Qpt > p1QptCap) { if (p1Qpt) cudaFree(p1Qpt); p1QptCap = 4096; while (p1QptCap < nP1Qpt) p1QptCap *= 2; cudaMallocManaged(&p1Qpt, (size_t)p1QptCap*sizeof(P1Qpt)); }
  for (i32 q = 0; q < nP1Qpt; q++) p1Qpt[q] = qpts[q];
  i32 nIrr = 0; for (i32 c = 0; c < nC && c < cE; c++) nIrr += p1Irr[c] != 0;
  {   // element thickness census: sqrt(12 lambda_min(M)/A), the length the explicit step must resolve
    double tmin = 1e30, amin = 0; i32 emin = -1, nThin = 0;
    for (i32 e = 0; e < nP1Elem; e++) {
      const P1Elem &E = p1Elem[e];
      const double tr = E.m11 + E.m22, det = E.m11*E.m22 - E.m12*E.m12, disc = sqrt(fmax(tr*tr - 4.0*det, 0.0));
      const double t = E.h*sqrt(12.0/fmax(0.5*(tr + disc)*E.area, 1e-300))/E.h;   // in cells
      if (t < 0.5) nThin++;
      if (t < tmin) { tmin = t; emin = e; amin = E.area/(E.h*E.h); }
    }
    printf("[p1cut] element thickness (cells): min %.4f (element %d, handle %d, area %.4f cells), %d elements thinner than 0.5\n",
           tmin, emin, emin >= 0 ? p1Elem[emin].handle : -1, amin, nThin);
  }
  printf("[p1cut] %d cut elements (%d quadrature points), %d face pieces (%d tip faces), %d irregular cells; "
         "max |polygon area - stamped| = %.2e cells; %d loops overflowed, %d hole-only cells, %d singular mass matrices, %d slivers dropped\n",
         nP1Elem, nP1Qpt, nP1Seg, nTip, nIrr, aErrMax, nOvfLoop, nHoleCell, nBadM, nDrop);
}

// --cutdump N: every ACTIVE cell within N cells of (xc, yc): geometry, merge
// ownership, split pieces with their owners, face segments, and the pressure
// of each DOF involved.  Fields are CONSERVATIVE at the end of a run.
void CompressibleSolver::writeCutWindow(const char *fileName, double xc, double yc, i32 nh) {
  cudaDeviceSynchronize();
  FILE *fp = fopen(fileName, "w");
  if (!fp) { printf("[cutdump] cannot open %s\n", fileName); return; }
  real *A = getField(F_CUTA), *AX = getField(F_CUTAX), *AY = getField(F_CUTAY);
  real *CX = getField(F_CUTCX), *CY = getField(F_CUTCY);
  real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoV = getField(F_RHOV), *RhoE = getField(F_RHOE);
  auto pres = [&](i32 h) {
    if (h == CUT_DEAD) return -1.0;
    double r, ru, rv, rE;
    if (cutIsPiece(h)) { const i32 k = cutPieceOf(h); const size_t cap = cutPieceQCap;
      r = cutPieceQ[k]; ru = cutPieceQ[cap+k]; rv = cutPieceQ[2*cap+k]; rE = cutPieceQ[4*cap+k]; }
    else { r = Rho[h]; ru = RhoU[h]; rv = RhoV[h]; rE = RhoE[h]; }
    const double u = ru/r, v = rv/r;
    return (double)(gam - (real)1)*(rE - 0.5*r*(u*u + v*v)); };
  const i32 cE = bEmpty*blockSizeTot;
  fprintf(fp, "# window centre %.6f %.6f, half-width %d cells; owner < 0 is a piece handle (-k-1)\n", xc, yc, nh);
  fprintf(fp, "# CELL c x y dx dy alpha aXlo aYlo cx cy owner alphaE p flag\n");
  fprintf(fp, "# SPLIT c a0 cx0 cy0 wnx0 wny0 wcx0 wcy0 nExtra iLen icx icy inx iny iPa iPb\n");
  fprintf(fp, "# PIECE c k a cx cy wnx wny wcx wcy owner p_owner handle\n");
  fprintf(fp, "# FACE c dir s len cen pC pN ownC ownN\n");
  i32 n = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    for (i32 cc = 0; cc < blockSizeTot; cc++) {
      const i32 c = b*blockSizeTot + cc;
      if (cc/blockSize/blockSize != 0 || c >= cE || cFlagsList[c] != ACTIVE) continue;
      double px, py, dx, dy; cellGeomHost(c, px, py, dx, dy);
      if (fabs(px - xc) > (nh + 0.5)*dx || fabs(py - yc) > (nh + 0.5)*dy) continue;
      n++;
      const i32 own = cutOwner ? cutOwner[c] : c;
      fprintf(fp, "CELL %d %.8f %.8f %.8f %.8f %.8e %.8e %.8e %.6f %.6f %d %.6f %.8f %d\n",
              c, px, py, dx, dy, (double)A[c], (double)AX[c], (double)AY[c], (double)CX[c], (double)CY[c],
              own, cutAlphaE ? (double)cutAlphaE[c] : (double)A[c], pres(own), (i32)cFlagsList[c]);
      const i32 sp = (cutSplit && cutSplitId) ? cutSplitId[c] : -1;
      if (sp >= 0) {
        const CutSplitCell &S = cutSplitCell[sp];
        fprintf(fp, "SPLIT %d %.8e %.6f %.6f %.6f %.6f %.6f %.6f %d %.8e %.6f %.6f %.6f %.6f %d %d\n", c, (double)S.a0, (double)S.cx0, (double)S.cy0,
                (double)S.wnx0, (double)S.wny0, (double)S.wcx0, (double)S.wcy0, S.n,
                (double)S.iLen, (double)S.icx, (double)S.icy, (double)S.inx, (double)S.iny, S.iPa, S.iPb);
        for (i32 k = 0; k < S.n; k++) {
          const CutPiece &P = cutPiece[S.first + k];
          fprintf(fp, "PIECE %d %d %.8e %.6f %.6f %.6f %.6f %.6f %.6f %d %.8f %d\n", c, k + 1, (double)P.a, (double)P.cx, (double)P.cy,
                  (double)P.wnx, (double)P.wny, (double)P.wcx, (double)P.wcy, P.owner, pres(P.owner), cutHandle(S.first + k));
        }
      }
      const i32 fid = (cutSplit && cutFaceId) ? cutFaceId[c] : -1;
      if (fid >= 0) {
        const CutFace &F = cutFace[fid];
        for (i32 s2 = 0; s2 < F.nX; s2++) fprintf(fp, "FACE %d 0 %d %.8e %.6f %d %d %d %d\n", c, s2, (double)F.sx[s2].len, (double)F.sx[s2].cen, F.sx[s2].pC, F.sx[s2].pN, F.sx[s2].ownC, F.sx[s2].ownN);
        for (i32 s2 = 0; s2 < F.nY; s2++) fprintf(fp, "FACE %d 1 %d %.8e %.6f %d %d %d %d\n", c, s2, (double)F.sy[s2].len, (double)F.sy[s2].cen, F.sy[s2].pC, F.sy[s2].pN, F.sy[s2].ownC, F.sy[s2].ownN);
      }
    }
  }
  fclose(fp);
  printf("[cutdump] %d cells around (%.4f, %.4f) -> %s\n", n, xc, yc, fileName);
}

// --reconfar: the two-cell band around the body where the 1-D MUSCL stencil
// would tap a shifted centroid, a dead cell or a merged member.  Everything
// outside it reconstructs with the ordinary 1-D scheme (reconFar).
void CompressibleSolver::buildCutNear(void) {
  // the band of cells whose stencil is irregular: near a cut/dead/merged cell
  // (--reconfar) and, under --leaf, next to a level jump (a missing same-level
  // tap or a covered PARENT tap).  Band cells take the least-squares gradient.
  const bool wantCut = ibRccm && immerserdBcType != 0 && reconFar >= 0;
  if (!wantCut && !leafFlux) return;
  cudaDeviceSynchronize();
  const size_t stride = (size_t)nBlocksMax*(size_t)blockSizeTot;
  if (!cutNear) cudaMallocManaged(&cutNear, stride*sizeof(i32));
  real *A = getField(F_CUTA);
  const i32 nC = hashTable.nKeys*blockSizeTot, cE = bEmpty*blockSizeTot;
  const bool cutOn = ibRccm && immerserdBcType != 0;
  auto irregular = [&](i32 m) {
    if (leafFlux && (m < 0 || m >= nC || m >= cE)) return true;         // missing same-level tap: a coarser neighbour
    if (m < 0 || m >= nC || m >= cE) return false;
    if (leafFlux && cFlagsList[m] == PARENT) return true;               // a finer neighbour
    if (!cutOn) return false;
    if (A[m] < (real)1 - (real)1e-12) return true;                       // cut or dead
    if (cutSplit && cutSplitId && cutSplitId[m] >= 0) return true;       // split (alpha may be 1)
    if (cutMerge && cutOwner && cutOwner[m] != m) return true;           // merged member
    if (cutMerge && cutAlphaE && cutAlphaE[m] != A[m]) return true;      // element owner (centroid shifted)
    return false;
  };
  i32 nNear = 0;
  for (size_t c = 0; c < stride; c++) cutNear[c] = 1;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    for (i32 cc = 0; cc < blockSizeTot; cc++) {
      const i32 c = b*blockSizeTot + cc;
      if (cc/blockSize/blockSize != 0) continue;
      const i32 i = cc%blockSize, j = (cc/blockSize)%blockSize;
      bool near = false;
      // covered PARENT cells are never evolved (restriction overwrites them), so
      // their reconstruction is dead work: give them the cheap scheme
      if (c < cE && cFlagsList[c] == PARENT) { cutNear[c] = 0; continue; }
      for (i32 dj = -2; dj <= 2 && !near; dj++)
        for (i32 di = -2; di <= 2 && !near; di++)
          if (irregular(hostNbrIdx(nbrIdxList, b, i+di, j+dj))) near = true;
      cutNear[c] = near ? 1 : 0;
      if (near && c < cE && cFlagsList[c] == ACTIVE) nNear++;
    }
  }
  if (reconFar >= 0)
    printf("[reconfar] %d: %d ACTIVE cells keep the least-squares gradient (two-cell band), the rest use the 1-D scheme\n", reconFar, nNear);
  else
    printf("[leaf] %d ACTIVE cells in the level-jump band (least-squares reconstruction)\n", nNear);
}

// --leaf: what the sort produced
void CompressibleSolver::leafCensus(void) {
  if (!leafFlux) return;
  cudaDeviceSynchronize();
  i32 nBand = 0;
  if (cutNear) for (i32 c = 0; c < nLeafBlocks*blockSizeTot; c++) nBand += (cutNear[c] != 0 && cFlagsList[c] == ACTIVE);
  printf("[leaf] blocks: %d leaf-bearing + %d exterior live, %d fully covered idle (of %d); %d mortar faces; %d band cells\n",
         nLeafBlocks, nExtBlocks, hashTable.nKeys - nLeafBlocks - nExtBlocks, hashTable.nKeys, nMortars, nBand);
  if (dbgChecks && nMortars > 0) {   // geometry of a few mortars: coarse centre, fine centres, sub-face centroids
    for (i32 m = 0; m < nMortars && m < 4; m++) {
      const Mortar &M = mortarList[m];
      double cx, cy, cdx, cdy, f0x, f0y, fdx, fdy, f1x, f1y;
      cellGeomHost(M.coarse, cx, cy, cdx, cdy); cellGeomHost(M.fine[0], f0x, f0y, fdx, fdy); cellGeomHost(M.fine[1], f1x, f1y, fdx, fdy);
      printf("[leaf] mortar %d dir %d side %d: coarse (%.5f,%.5f) h=%.5f flag %d | fine (%.5f,%.5f) (%.5f,%.5f) h=%.5f flags %d %d | faces (%.5f,%.5f) (%.5f,%.5f)\n",
             m, M.dir, M.side, cx, cy, cdx, cFlagsList[M.coarse], f0x, f0y, f1x, f1y, fdx, cFlagsList[M.fine[0]], cFlagsList[M.fine[1]],
             (double)M.cen[0][0], (double)M.cen[0][1], (double)M.cen[1][0], (double)M.cen[1][1]);
    }
  }
}

void CompressibleSolver::stampIbGeometry(void) {
  if (immerserdBcType == 0) { if (leafFlux) { buildCutNear(); leafCensus(); } return; }
  ibStampGeometryKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  stampCutGeomCurved();   // --cutgeom 2: curved geometry over the linear cut
  stampCutGeomClip();     // --cutgeom 3: exact segment clipping over the linear cut
  buildCutMerge();        // --cutmerge: agglomerate small cells (indices: after EVERY sort)
  buildP1Cut();           // --p1: element polygons, quadrature and face segments (needs the owners)
  buildCutNear();         // --reconfar / --leaf: the band that keeps the least-squares reconstruction
  leafCensus();
  buildSrd();   // neighbourhoods follow the geometry
}


// ---- Jacobian-free Newton-Krylov: residual and matrix-free product ---------

// R(q) for the turbulence pair with everything else frozen.  The caller must
// leave the fields PRIMITIVE and the state scattered; this reproduces exactly
// the sequence a stage uses, so the residual is the one the explicit march sees.
// R(q) for the WHOLE system.  The state vector is CONSERVATIVE (that is what the
extern __device__ i32 g_p1LimPieces;
extern __device__ unsigned long long g_qUsed;
extern __device__ unsigned long long g_qDecl;
extern __device__ unsigned long long g_rcDeadFace;
extern __device__ unsigned long long g_rcDeadGrad;
extern __device__ unsigned long long g_rcLiveFace;
extern __device__ unsigned long long g_qGhostTap;

// Is the implicit ghost quadratic actually being used, and is it really taking
// ghost taps?  "Improved by 26% with iterations changing nothing" has two very
// different explanations -- converged in one sweep, or the coupling is inert --
// and they are indistinguishable without this.
void CompressibleSolver::reportGhostQuad(void) {
  unsigned long long u=0,d=0,g=0;
  cudaMemcpyFromSymbol(&u, g_qUsed, sizeof(u));
  cudaMemcpyFromSymbol(&d, g_qDecl, sizeof(d));
  cudaMemcpyFromSymbol(&g, g_qGhostTap, sizeof(g));
  printf("[ghostquad] quadratic used %llu, declined %llu (%.2f%%), ghost taps consumed %llu"
         "  -> %.2f ghost taps per evaluation\n",
         u, d, (u+d) ? 100.0*double(d)/double(u+d) : 0.0, g,
         u ? double(g)/double(u) : 0.0);
}

// Per-cell Ducros-like compression sensor into F_RHOK (recon 5 only; the
// bank is free because recon 5 is refused under RANS).  Runs on PRIMITIVES,
// after the ghosts and before the RHS reads the traces.

void CompressibleSolver::computeShockSensor(void) {
  shockSensorKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::applyWallGhosts(void) {
  if (immerserdBcType != 0) {
    if (!ibGhostFree) {
      // With the natural mirror the interpolation stencil contains ghosts, so
      // one pass is a single Jacobi sweep of a coupled system.  Iterate it.
      // The interface-cell mode couples the first fluid layer to itself as
      // well (its image-point stencil contains the cell being set), so it
      // needs at least two sweeps to resolve the self term.
      i32 nit = (ibGMirror && ibGIter > 1) ? ibGIter : 1;
      if (ibIface) nit = max(ibGIter, 2);
      for (i32 it = 0; it < nit; it++) {
        ibGhostKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
        if (ibIface) ibIfaceKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      }
    }
  }
  else wallGhostKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::computeTurbClosure(void) {
  turbClosureKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

// ---- far-field point vortex: measure C_l, refresh the circulation ---------
// Gamma = 0.5 V_inf c C_l, with C_l from the surface pressure force projected
// perpendicular to the freestream.  The state must be PRIMITIVE (the kernel
// reads p out of the F_RHOE slot), which is why this is called from the same
// place the RHS is assembled.
void CompressibleSolver::updateFarFieldVortex(void) {
  if (!ffVortex || immerserdBcType == 0) return;
  double z = 0;
  cudaMemcpyToSymbol(g_ibFx, &z, sizeof(z));
  cudaMemcpyToSymbol(g_ibFy, &z, sizeof(z));
  ibForceKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  double Fx = 0, Fy = 0;
  cudaMemcpyFromSymbol(&Fx, g_ibFx, sizeof(Fx));
  cudaMemcpyFromSymbol(&Fy, g_ibFy, sizeof(Fy));
  const double Vx = (double)fsU, Vy = (double)fsV;
  const double V2 = Vx*Vx + Vy*Vy;
  const double V  = sqrt(V2);
  if (!(V > 0) || !(ibChord > 0)) return;
  // lift = force component perpendicular to the freestream (rho_inf = 1)
  const double L  = (-Fx*Vy + Fy*Vx)/V;
  const double Cl = L/(0.5*V2*(double)ibChord);
  if (!std::isfinite(Cl)) return;
  ffCl = (real)Cl;
  // Relax: the boundary must not chase a transient.  A first-order lag also
  // keeps the BC from feeding back on itself (the vortex raises the lift,
  // which raises the vortex ...), which is the classic way this correction
  // goes unstable.
  const real gNew = (real)(0.5*V*(double)ibChord*Cl);
  ffGamma = (real)0.7*ffGamma + (real)0.3*gNew;
  if (ffPrints < 6) { ffPrints++;
    printf("[ff] iter %d  Cl(surface) = %+.4f  Gamma = %+.5f  (Fx %+.4e Fy %+.4e)\n",
           iter, Cl, (double)ffGamma, Fx, Fy); }
}

void CompressibleSolver::stateHash(const char *tag, i32 it) {
  for (i32 f = 0; f < 10; f++) dbgCnt[40+f] = 0;
  stateHashKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  printf("[shash] it=%d %-10s q %08x %08x %08x %08x %08x | S %08x %08x %08x %08x %08x\n", it, tag,
         (unsigned)dbgCnt[40], (unsigned)dbgCnt[41], (unsigned)dbgCnt[42],
         (unsigned)dbgCnt[43], (unsigned)dbgCnt[44],
         (unsigned)dbgCnt[45], (unsigned)dbgCnt[46], (unsigned)dbgCnt[47],
         (unsigned)dbgCnt[48], (unsigned)dbgCnt[49]);
}

void CompressibleSolver::computeRightHandSide(void) {
  // --detflux: resolve the flag once (Brinkman face weights and the multiD
  // path are incompatible) and keep the face banks sized to the hash table.
  if (detFlux && (mdFlux || ibRccm || ibBrink)) {
    // The deterministic gather stores raw face fluxes and sums them per cell,
    // so it bypasses the per-face weights entirely -- which under RCCM are the
    // cut apertures and the 1/alpha.  Leaving it on applied the wall flux
    // against UNWEIGHTED Cartesian faces: measured as a uniform-state momentum
    // residual of exactly p/(alpha dx), i.e. one whole uncancelled aperture.
    printf("[detflux] disabled: incompatible with %s\n",
           mdFlux ? "multiD flux" : (ibRccm ? "--ibrccm" : "--ibbrink"));
    detFlux = 0;
  }
  if (detFlux) {
    const u64 need = (u64)hashTable.nKeys*blockSizeTot;
    if (need > ffN) {
      if (ffBuf) cudaFree(ffBuf);
      ffN = need + need/2;
      // Bank map is ABSOLUTE: mean 0-14 (x/y/z), turbulence 15+4d..18+4d.
      // pseudo2D leaves 10-14 and the d=2 turb slots unwritten but the map
      // stays valid; under-allocating by the 2-D count while indexing the
      // absolute slots was an OOB write (caught by compute-sanitizer as a
      // poisoned context that then threw a bogus "not compiled for SM 89").
      const i32 nb = rans ? 27 : (pseudo2D ? 10 : 15);
      if (cudaMalloc(&ffBuf, (size_t)nb*ffN*sizeof(real)) != cudaSuccess) {
        printf("[detflux] face-bank alloc failed (%.1f MB); falling back to atomics\n",
               nb*ffN*sizeof(real)/1048576.0);
        ffBuf = nullptr; ffN = 0; detFlux = 0;
      }
    }
  }
  if (mdFlux) {
    if (mdFlux == 2) {
      // CTU-Hancock: half-step-predict all cells into the Old bank (free until
      // updateFields), fill its halos, then assemble the multiD fluxes on the
      // time-centred field
      hancockPredictKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      setBoundaryConditions(F_OLD, 1);   // the Old bank holds PREDICTED PRIMITIVES here
    }
    multiDRhsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    return;
  }
  if (p1) {   // modal P1 DG: own kernels, no reconstruction stencil
    p1RhsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    if (leafFlux && nMortars > 0)
      p1MortarKernel<<<(nMortars + 127)/128, 128>>>(*this);
    if (nP1Seg > 0)  p1SegKernel<<<(nP1Seg + 127)/128, 128>>>(*this);     // cut elements: open face pieces
    if (nP1Elem > 0) p1ElemKernel<<<(nP1Elem + 127)/128, 128>>>(*this);   // cut elements: volume rule + wall
    return;
  }
  computeRightHandSideKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  if (leafFlux && nMortars > 0)
    mortarFluxKernel<<<(nMortars + 127)/128, 128>>>(*this);   // level-jump faces: one HLLC per fine sub-face, both sides
  if (ibRccm && immerserdBcType != 0)
    cutCellKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // wall fluxes, extra face segments, piece walls, tip faces
  if (detFlux) gatherFaceFluxKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  static const i32 cutDbgIt0 = getenv("CUTDBG_IT0") ? atoi(getenv("CUTDBG_IT0")) : -1;   // scan only once iter >= this
  if (cutDbg > 0 && iter >= cutDbgIt0) {   // --cutdbg: who is exploding in the first RHS?
    cudaDeviceSynchronize();
    real *R0 = getField(F_RHS), *R1 = getField(F_RHS+1), *A = getField(F_CUTA), *Ph = getField(F_PHI);
    i32 shown = 0, nBig = 0;
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      u64 loc = bLocList[b]; if (loc == kEmpty) continue;
      i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
      for (i32 cc = 0; cc < blockSizeTot; cc++) {
        const i32 c = b*blockSizeTot + cc;
        const double m = fmax(fabs((double)R0[c]), fabs((double)R1[c]));
        if (m < cutDbgThr) continue;   // NaN fails the test and is counted as bad
        nBig++;
        if (shown++ < 25) {
          const i32 o = cutOwner ? cutOwner[c] : c;
          const i32 oc = cutIsPiece(o) ? cutPiece[cutPieceOf(o)].cell : o;
          double pxd, pyd, dxd, dyd; cellGeomHost(c, pxd, pyd, dxd, dyd);
          const i32 mor[4] = { cellMortar ? cellMortar[(size_t)c*4+0] : -1, cellMortar ? cellMortar[(size_t)c*4+1] : -1, cellMortar ? cellMortar[(size_t)c*4+2] : -1, cellMortar ? cellMortar[(size_t)c*4+3] : -1 };
          printf("[cutdbg] cell %d at (%.4f,%.4f) mortars(%d,%d,%d,%d) band %d blk %d lvl %d flag %d (i,j)=(%d,%d) |Rhs|=%.3e alpha=%.3e alphaE=%.3e owner=%d (cell %d flag %d alpha %.3e) phi/h=%.3f\n",
                 c, pxd, pyd, mor[0], mor[1], mor[2], mor[3], cutNear ? cutNear[c] : -1, b, lvl, (i32)cFlagsList[c], cc%blockSize, (cc/blockSize)%blockSize, m,
                 (double)A[c], cutAlphaE ? (double)cutAlphaE[c] : -1.0, o, oc, (i32)cFlagsList[oc],
                 (double)A[oc], (double)Ph[c]/(double)(domainSize[0]/(baseGridSize[0]*powi(2,lvl))));
        }
      }
    }
    printf("[cutdbg] stage %d: %d cells with |Rhs| > %.1e\n", stage, nBig, cutDbgThr);
    if (leafFlux) {   // conversion coverage: at this point every live cell must be CONSERVATIVE
      real *Pp = getField(F_RHOE); i32 nPrim = 0, nCons = 0, firstPrim = -1, lastPrim = -1;
      for (i32 b = 0; b < hashTable.nKeys; b++) { if (bLocList[b] == kEmpty) continue;
        i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb); if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
        for (i32 cc = 0; cc < blockSize*blockSize; cc++) { const i32 c = b*blockSizeTot + cc; if (cFlagsList[c] != ACTIVE) continue;
          const double v = Pp[c];
          if (fabs(v - 7.9365079365) < 1e-4) { nPrim++; if (firstPrim < 0) firstPrim = b; lastPrim = b; }
          else if (fabs(v - 20.3412698413) < 1e-3) nCons++; } }
      printf("[cutdbg] live ACTIVE interior cells: %d still primitive (blocks %d..%d), %d conservative; nLeaf %d nExt %d nKeys %d\n",
             nPrim, firstPrim, lastPrim, nCons, nLeafBlocks, nExtBlocks, hashTable.nKeys);
      // sort-order sanity: recompute the group of a few slots from the hash
      for (i32 b : {0, nLeafBlocks-1, nLeafBlocks, nLeafBlocks+nExtBlocks-1, nLeafBlocks+nExtBlocks, hashTable.nKeys-1}) {
        if (b < 0 || b >= hashTable.nKeys) continue;
        i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
        i32 nAct = 0, nPar = 0; for (i32 cc = 0; cc < blockSize*blockSize; cc++) { nAct += cFlagsList[b*blockSizeTot+cc] == ACTIVE; nPar += cFlagsList[b*blockSizeTot+cc] == PARENT; }
        printf("[cutdbg]   slot %d: lvl %d interior %d  ACTIVE %d PARENT %d\n", b, lvl, (i32)isInteriorBlock(lvl, ib, jb, kb), nAct, nPar);
      }
    }
    if (nBig > 0 && leafFlux) {   // the first offender's 3x3 neighbourhood
      i32 c0 = -1;
      for (i32 b = 0; b < hashTable.nKeys && c0 < 0; b++) { if (bLocList[b] == kEmpty) continue;
        for (i32 cc = 0; cc < blockSizeTot && c0 < 0; cc++) { const i32 c = b*blockSizeTot + cc;
          if (!(fmax(fabs((double)R0[c]), fabs((double)R1[c])) < cutDbgThr)) c0 = c; } }
      const i32 b0 = c0/blockSizeTot, cc0 = c0%blockSizeTot, i0 = cc0%blockSize, j0 = (cc0/blockSize)%blockSize;
      real *Rho = getField(F_RHO), *U = getField(F_RHOU), *P = getField(F_RHOE);
      for (i32 dj = 1; dj >= -1; dj--) {
        printf("[cutdbg]   ");
        for (i32 di = -1; di <= 1; di++) {
          const i32 m = hostNbrIdx(nbrIdxList, b0, i0+di, j0+dj);
          const i32 mb = m/blockSizeTot;
          const bool ok = m < bEmpty*blockSizeTot;
          printf("[%6d blk %4d grp %d flag %d rho %.4f u %.4f p %.4f] ", m, mb,
                 ok ? (mb < nLeafBlocks ? 0 : (mb < nLeafBlocks + nExtBlocks ? 1 : 2)) : -1,
                 ok ? (i32)cFlagsList[m] : -1, ok ? (double)Rho[m] : 0.0, ok ? (double)U[m] : 0.0, ok ? (double)P[m] : 0.0);
        }
        printf("\n");
      }
    }
    cutDbg--;
  }
  updateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage);
  // piece-resident DOFs advance with the same LSRK stage (before the limiters read them)
  if (cutSplit && nCutPiece > 0) cutPieceUpdateKernel<<<(nCutPiece + 127)/128, 128>>>(*this, stage);
  if (p1 && gradLim > 0) {
    p1LimitKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // slopes against the neighbour means
    static const bool noCutLim = getenv("P1_NOCUTLIM") != nullptr;   // debug switch
    static bool pieceLimSet = false;
    if (!pieceLimSet) { pieceLimSet = true; i32 v = getenv("P1_NOPIECELIM") ? 0 : 1; cudaMemcpyToSymbol(g_p1LimPieces, &v, sizeof(i32)); }
    if (nP1Elem > 0 && !noCutLim) p1LimitCutKernel<<<(nP1Elem + 127)/128, 128>>>(*this);
  }
  // merged elements: the owner advanced, the members take its state
  if (cutMerge && cutOwner) cutBroadcastKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

#ifdef USE_MGPU
// Refresh partition-boundary ghost blocks from their owning PEs.  Bracketed by
// barriers so every PE has finished writing its owned data before anyone reads
// it, and no PE overwrites before all reads complete.
// Build and exchange the per-neighbor boundary directories (once per adaptation).
// Each PE lists the loc codes of its owned blocks whose 2-ring reaches into each
// neighbor, sizes the globally-uniform per-neighbor slot, and exchanges the
// counts then the loc payloads via comm::neighborExchange.
void CompressibleSolver::buildDirectories(void) {
  if (comm::size() == 1 || nNbr == 0) return;
  if (!dirSendCnt) {
    cudaMallocManaged(&dirSendCnt, nNbr*sizeof(i32));
    cudaMallocManaged(&dirRecvCnt, nNbr*sizeof(i32));
    cudaMallocManaged(&dirFill,    nNbr*sizeof(i32));
  }
  cudaMemset(dirSendCnt, 0, nNbr*sizeof(i32));
  countDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  i32 maxc = 0; for (i32 s = 0; s < nNbr; s++) if (dirSendCnt[s] > maxc) maxc = dirSendCnt[s];
  real mc = (real)maxc; comm::allreduceMax(&mc, 1);            // slot uniform on every PE
  i32 needSlot = (i32)mc + 8;
  if (needSlot > dirSlot) {
    cudaFree(dirSendLoc); cudaFree(dirRecvLoc); cudaFree(sendBuf); cudaFree(recvBuf);
    dirSlot = needSlot;
    size_t fs = (size_t)NEVOLVE*blockSizeTot;
    cudaMallocManaged(&dirSendLoc, (size_t)nNbr*dirSlot*sizeof(u64));
    cudaMallocManaged(&dirRecvLoc, (size_t)nNbr*dirSlot*sizeof(u64));
    cudaMallocManaged(&sendBuf,    (size_t)nNbr*dirSlot*fs*sizeof(real));
    cudaMallocManaged(&recvBuf,    (size_t)nNbr*dirSlot*fs*sizeof(real));
    // NEED lists (adopt mechanism): every near-seam KEEP block can request up to
    // its full 27-target ring in the neighbour's territory, so size by 27x the
    // directory scale (loc codes only -- cheap)
    cudaFree(this->needLoc); cudaFree(needRecvLoc);
    this->needSlot = 27*dirSlot;
    cudaMallocManaged(&this->needLoc, (size_t)nNbr*this->needSlot*sizeof(u64));
    cudaMallocManaged(&needRecvLoc,   (size_t)nNbr*this->needSlot*sizeof(u64));
    if (!this->needCnt) { cudaMallocManaged(&this->needCnt, nNbr*sizeof(i32)); cudaMemset(this->needCnt, 0, nNbr*sizeof(i32)); }
    if (!needRecvCnt)   { cudaMallocManaged(&needRecvCnt, nNbr*sizeof(i32)); }
  }
  comm::barrier();
  cudaMemset(dirFill, 0, nNbr*sizeof(i32));
  fillDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  // exchange counts, then the loc payloads
  std::vector<void*> ss(nNbr), rr(nNbr); std::vector<size_t> sb(nNbr), rb(nNbr);
  for (i32 s = 0; s < nNbr; s++) { ss[s]=&dirSendCnt[s]; rr[s]=&dirRecvCnt[s]; sb[s]=sizeof(i32); rb[s]=sizeof(i32); }
  comm::neighborExchange(nNbr, nbrRank, ss.data(), sb.data(), rr.data(), rb.data());
  for (i32 s = 0; s < nNbr; s++) {
    ss[s]=&dirSendLoc[(size_t)s*dirSlot]; rr[s]=&dirRecvLoc[(size_t)s*dirSlot];
    sb[s]=(size_t)dirSendCnt[s]*sizeof(u64); rb[s]=(size_t)dirRecvCnt[s]*sizeof(u64);
  }
  comm::neighborExchange(nNbr, nbrRank, ss.data(), sb.data(), rr.data(), rb.data());
  comm::barrier();
}

// Halo exchange: pack this PE's directory blocks per neighbor, exchange the
// packed buffers with comm::neighborExchange (one contiguous message per
// neighbor), and unpack into the ghost blocks.  Reuses the directories from
// buildDirectories; nothing reaches into a peer's memory.
void CompressibleSolver::haloExchange(i32 fOff, i32 nf) {
  if (comm::size() == 1 || nNbr == 0) return;
  size_t fs = (size_t)nf*blockSizeTot;
  cudaDeviceSynchronize();
  comm::barrier();
  packDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this, fOff, nf);
  cudaDeviceSynchronize();
  std::vector<void*> ss(nNbr), rr(nNbr); std::vector<size_t> sb(nNbr), rb(nNbr);
  for (i32 s = 0; s < nNbr; s++) {
    ss[s]=&sendBuf[(size_t)s*dirSlot*fs]; rr[s]=&recvBuf[(size_t)s*dirSlot*fs];
    sb[s]=(size_t)dirSendCnt[s]*fs*sizeof(real); rb[s]=(size_t)dirRecvCnt[s]*fs*sizeof(real);
  }
  comm::neighborExchange(nNbr, nbrRank, ss.data(), sb.data(), rr.data(), rb.data());
  unpackDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this, fOff, nf);
  cudaDeviceSynchronize();
  comm::barrier();
}

// Prune the stale partition ghosts and recreate the 2-ring from the neighbors'
// directories, then re-sort.  Called after the post-adaptGrid sortBlocks: the
// ghost layer tracks — and prunes with — the moving grid, so each PE holds only
// its subdomain + a thin halo (a real decomposition, not the full grid).
void CompressibleSolver::rebuildGhosts(void) {
  if (comm::size() == 1) return;
  buildDirectories();                                           // exchange the boundary directories
  cudaDeviceSynchronize();
  markGhostsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);      // old ghosts -> DELETE; owned/exterior -> KEEP
  cudaDeviceSynchronize();
  consumeDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this);      // create ghosts from directories (un-delete)
  cudaDeviceSynchronize();
  // NB: no keepLocalSupport pass.  Under the owned-target + NEED/adopt protocol
  // every non-owned interior block has an owner and is directory-backed; a ghost
  // no directory names is a genuine zombie (its owner coarsened it) and MUST be
  // pruned -- keeping it alive leaves a permanently-unfillable zero block whose
  // presence flips owned boundary cells from GHOST to ACTIVE (vacuum feeder).
  nBlocks = hashTable.nKeys;
  deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);      // drop the ghosts no longer needed
  cudaDeviceSynchronize();
  sortBlocks();                                                  // compact + rebuild indices
  comm::barrier();
}
#endif

// Debug census: owned interior block count (allreduced), printed with a stage tag.
void CompressibleSolver::censusPrint(const char *tag) {
  if (!dbgChecks) return;
  cudaDeviceSynchronize();
  double nOwn = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    if (bFlagsList[b] == DELETE) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;
#endif
    nOwn += 1;
  }
#ifdef USE_MGPU
  comm::allreduceSum(&nOwn, 1);
  if (part.rank == 0)
    printf("[stage] %-14s ownedInteriorSum=%.0f\n", tag, nOwn);
#else
  printf("[stage] %-14s ownedInteriorSum=%.0f\n", tag, nOwn);
#endif
}

// Debug-mode integrity check: run the topology assert kernel, allreduce the
// violation counter, and report.  Gated by --debug (dbgChecks); zero cost when off.
void CompressibleSolver::topoCheck(i32 phaseTag) {
  if (!dbgChecks) return;
  cudaDeviceSynchronize();
  cudaMemset(dbgCnt, 0, sizeof(i32));
  checkTopologyKernel<<<cudaGridSize, cudaBlockSize>>>(*this, phaseTag);
  cudaDeviceSynchronize();
  double v = (double)dbgCnt[0];
#ifdef USE_MGPU
  comm::allreduceSum(&v, 1);
#endif
  if (v > 0
#ifdef USE_MGPU
      && part.rank == 0
#endif
     ) printf("[topo] phase %d: %.0f TOPOLOGY VIOLATIONS (see [topo] lines)\n", phaseTag, v);
}

#ifdef USE_MGPU
// Block-activity exchange: publish this rank's owned blocks to its neighbors and
// import (create as ghosts) the neighbors' blocks that fall within our 2-ring.
// STRUCTURE ONLY -- no field data moves, so the F_OLD wavelet snapshot survives
// (unlike rebuildGhosts, which sorts and clobbers the shared bank).  Run after
// every kernel that creates blocks so a neighbor's fresh refinement is visible
// before our next grading / reconstruction pass -- the cross-seam analogue of
// the single-GPU cascade closing in one sweep.
void CompressibleSolver::exchangeStructure(void) {
  if (comm::size() == 1 || nNbr == 0) return;
  // Full seam SYNC, both directions (the periodic-boundary analogue):
  //  - NEED lists first: support targets our closure required in a neighbour's
  //    territory are sent to their owner, who creates them as owned ("adopt");
  //  - directories: every owned block near the seam -> a GHOST on the neighbour
  //    (this same pass exports the just-adopted blocks back to the requester);
  //  - stale ghosts (no longer in any neighbour's directory and not local
  //    support) are DELETED so the layer never accumulates.
  // No sortBlocks -- that clobbers the F_OLD snapshot bank mid-cascade.
  if (needCnt && needSlot > 0) {
    cudaDeviceSynchronize();
    // overflowed counts mean silently-dropped support blocks (missing-block
    // vacuum) -- report loudly, then clamp; the buffers grow with dirSlot
    for (i32 s = 0; s < nNbr; s++) {
      if (needCnt[s] > needSlot) {
        printf("[need] rank %d: %d NEEDs for nbr %d OVERFLOW slot %d -- support dropped%s\n",
               part.rank, needCnt[s], s, needSlot, dbgChecks ? " (FATAL under --debug)" : "");
        if (dbgChecks) { fflush(stdout); abort(); }
        needCnt[s] = needSlot;
      }
    }
    std::vector<void*> ss(nNbr), rr(nNbr); std::vector<size_t> sb(nNbr), rb(nNbr);
    for (i32 s = 0; s < nNbr; s++) { ss[s]=&needCnt[s]; rr[s]=&needRecvCnt[s]; sb[s]=sizeof(i32); rb[s]=sizeof(i32); }
    comm::neighborExchange(nNbr, nbrRank, ss.data(), sb.data(), rr.data(), rb.data());
    for (i32 s = 0; s < nNbr; s++) {
      ss[s]=&needLoc[(size_t)s*needSlot]; rr[s]=&needRecvLoc[(size_t)s*needSlot];
      sb[s]=(size_t)needCnt[s]*sizeof(u64); rb[s]=(size_t)needRecvCnt[s]*sizeof(u64);
    }
    comm::neighborExchange(nNbr, nbrRank, ss.data(), sb.data(), rr.data(), rb.data());
    consumeNeedKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // adopt: create them as owned
    cudaDeviceSynchronize();
    cudaMemset(needCnt, 0, nNbr*sizeof(i32));                    // reset for the next create kernel
  }
  buildDirectories();
  consumeDirKernel<<<cudaGridSize, cudaBlockSize>>>(*this);      // create/touch ghosts of the peers' SURVIVING blocks
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;
  // NB: no markGhosts / deleteData here.  markGhosts raises EVERY owned block to
  // KEEP -- mid-cascade that overwrites thresholding's DELETE decisions on every
  // exchange, so no block can ever coarsen (a one-way ratchet to a dense grid;
  // the seam band was this ratchet's local form).  Ghost lifecycle without it:
  // all flags start the cycle DELETE (memset in the forward transform);
  // consumeDir touches ghosts named in the survivor directories (NEW -> KEEP);
  // ghosts of coarsened peer blocks stay DELETE, are never grading sources, and
  // are pruned once by the tail deleteData.  A mid-cascade deleteData would also
  // ZERO the data of blocks a later exchange legitimately re-imports.
}
#endif

// The refinement cascade.  On a single GPU this is exactly adaptGrid().
//
// Under MGPU the cascade maintains the invariant that each rank's local grid is
// a WINDOW of the single-GPU union grid: owned blocks + the union-grid blocks in
// the Chebyshev-2 same-level ring of the owned set (which covers every stencil:
// the +-2-cell flux ring, the 27-tap wavelet prediction on the parent level, and
// the one-sided interpolation parents).  Mechanics per create kernel:
//   - OWNED targets only; a target in a neighbour's territory is sent as a NEED
//     and the owner creates it (adopt) -- no rank manufactures blocks it cannot
//     fill, and the refined set stays partition-independent;
//   - exchangeStructure after each kernel: NEED adopt + directory publication of
//     owned SURVIVING (flag != DELETE) blocks + consumeDir ghost import, so the
//     next collective kernel grades against the neighbours' fresh refinement.
//     Publishing only survivors is what lets blocks coarsen: ghosts of deleted
//     peer blocks stay DELETE and are pruned at the tail;
//   - recon/boundary stages iterate to a GLOBAL fixed point (createdCnt
//     allreduce) to close multi-hop adoption chains;
//   - tail: prune, rebuild fresh (corpse-free) directories for the F_OLD halo,
//     refresh prnt/nbr index lists that the prolongation/BC/inverse traverse.
void CompressibleSolver::adaptGridConsistent(void) {
  if (nLvls <= 1) return;
#ifndef USE_MGPU
  adaptGrid();
#else
  // single pass each: children of owned are owned (no NEEDs possible), and the
  // grading ring of the union KEEP set is exactly what one pass + adopt builds
  // (repeating addAdjacent would grade rings-of-rings -- over-refinement vs 1 GPU)
  censusPrint("pre-cascade");
  addFineBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  exchangeStructure();
  setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);   // imported ghosts -> KEEP (grading sources)
  censusPrint("post-fine");
  addAdjacentBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  exchangeStructure();
  setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  censusPrint("post-adjacent");
  // reconstruction + boundary: settle each stage to a GLOBAL fixed point.  The
  // recon kernel has no level filter, so on one rank the fixed point equals the
  // old per-level loop; across ranks it also closes multi-hop chains where an
  // adopted support block needs support of its own on yet another rank.  The
  // quiescence test is "did ANY rank CREATE a block this pass" (createdCnt is
  // bumped only by activateBlock's create branch, incl. adoptions).
  const i32 passCap = nLvls + comm::size() + 4;
  for (i32 stage = 0; stage < 2; stage++) {
    i32 pass = 0;
    while (true) {
      cudaDeviceSynchronize();
      cudaMemset(createdCnt, 0, sizeof(i32));
      if (stage == 0) addReconstructionBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      else            addBoundaryBlocksKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      exchangeStructure();
      setBlocksKeepKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      cudaDeviceSynchronize();
      real created = (real)createdCnt[0];
      comm::allreduceMax(&created, 1);
      if (created == 0.0) break;
      if (++pass >= passCap) {
        printf("[adapt] rank %d: stage %d settlement exceeded %d passes (still creating)\n",
               part.rank, stage, passCap);
        break;
      }
    }
  }
  // tail: final prune, then FRESH directories (corpse-free: the pre-prune lists
  // may name just-deleted blocks, and packing those ships zeros over the peers'
  // valid ghost F_OLD), then the index refresh the prolongation/BC/inverse
  // traverse.  Single-GPU tolerates stale indices only because its support is
  // built a cycle ahead of use; the seam layer is not.
  censusPrint("post-settle");
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;
  deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  buildDirectories();
  updatePrntIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  updateNbrIndicesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  censusPrint("post-prune");
  topoCheck(1);   // debug: cascade tail -- bindings must be loc-consistent before prolongation
#endif
}

void CompressibleSolver::restrictFields(void) {
  // Finest -> coarsest, one launch per level.  A single all-level launch races:
  // a lvl-1 parent written by its lvl-2 children is concurrently READ as a
  // child by the lvl-0 restriction, so the grandparent sees old-or-new by warp
  // schedule -- the last nondeterminism source after the det flux gather
  // (bisected via --shash: state banks diverge in the interp phase, RHS clean).
  for (i32 l = nLvls - 1; l >= 1; l--)
    restrictFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, l);
}

void CompressibleSolver::interpolateFields(void) {
  // Coarsest -> finest for the same reason: a lvl-1 GHOST halo cell being
  // interpolated is a trilinear source for lvl-2 ghosts.
  for (i32 l = 1; l < nLvls; l++)
    interpolateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, l);
}

//
// Dump a 1D line profile along x (through the y/z center of the domain) of the
// primitive variables.  Used to validate against the exact Sod solution.
// Assumes the fields are in conservative form on entry (as after step()).
//
void CompressibleSolver::writeLineProfile(const char *fileName) {
  cudaDeviceSynchronize();

  i32 jbT = (baseGridSize[1] / blockSize) / 2;
  i32 jT  = blockSize / 2;
  i32 kbT = (baseGridSize[2] / blockSizeZ) / 2;
  i32 kT  = blockSizeZ / 2;

  real *Rho  = getField(0);
  real *RhoU = getField(1);
  real *RhoV = getField(2);
  real *RhoW = getField(3);
  real *RhoE = getField(4);

  std::vector<std::array<real,4>> rows; // x, rho, u, p

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;

    if (lvl != 0) continue; // nLvls == 1 validation path
    if (jb != jbT || kb != kbT) continue;

    i32 gridX = baseGridSize[0] / blockSize;
    if (ib < 0 || ib >= gridX) continue; // interior only

    real dx = domainSize[0] / real(baseGridSize[0]);
    for (i32 i = 0; i < blockSize; i++) {
      i32 cIdx = bIdx*blockSizeTot + i + jT*blockSize + kT*blockSize*blockSize;
      real r  = Rho[cIdx];
      real u  = RhoU[cIdx]/r;
      real v  = RhoV[cIdx]/r;
      real w  = RhoW[cIdx]/r;
      real p  = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      real x  = (ib*blockSize + i + 0.5)*dx;
      rows.push_back({x, r, u, p});
    }
  }

  // pseudo-2D sanity check: the transverse (y) and collapsed (z) velocity
  // components should remain identically zero for a planar shock tube.
  real maxV = 0, maxW = 0;
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]*powi(2,lvl)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,lvl)/blockSize;
    i32 gz = baseGridSize[2]*powi(2,lvl)/blockSizeZ;
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      real r = Rho[cIdx];
      if (r <= 0) continue;
      maxV = fmax(maxV, fabs(RhoV[cIdx]/r));
      maxW = fmax(maxW, fabs(RhoW[cIdx]/r));
    }
  }
  printf("pseudo-2D check: max|v| = %.3e, max|w| = %.3e (should be ~0)\n", maxV, maxW);

  std::sort(rows.begin(), rows.end(),
            [](const std::array<real,4>&a, const std::array<real,4>&b){ return a[0] < b[0]; });

  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "# x rho u p\n");
  for (auto &r : rows) {
    fprintf(fp, "%.8e %.8e %.8e %.8e\n", r[0], r[1], r[2], r[3]);
  }
  fclose(fp);
  printf("wrote %zu line samples to %s\n", rows.size(), fileName);
}

//
// Acoustic-reflection diagnostic for the exact-simple-wave test (icType 5).
// Uses the EXACT Riemann invariants (not linearized), measured as perturbations
// from the uniform background J+-0 = +-2 c0/(gam-1):
//   dJ- = (u - 2c/(gam-1)) + 2c0/(gam-1)    left-going   (0 for a pure right-runner)
//   dJ+ = (u + 2c/(gam-1)) - 2c0/(gam-1)    right-going  (the incident/transmitted wave)
// The simple-wave IC has dJ- == 0 exactly, so a uniform grid gives dJ- ~ 0; any
// dJ- in the coarse half (x < cx) is a reflection off the centre interface.
// Reflection coefficient R = max|dJ-|_coarse / max|dJ+|.  Interface at x = cx.
//
void CompressibleSolver::computeAcousticReflection(const char *fileName) {
  cudaDeviceSynchronize();
  real *Rho  = getField(F_RHO);
  real *RhoU = getField(F_RHOU);
  real *RhoV = getField(F_RHOV);
  real *RhoW = getField(F_RHOW);
  real *RhoE = getField(F_RHOE);

  const real rho0 = 1.0, p0 = 1.0;
  const real c0   = sqrt(gam*p0/rho0);
  const real Jbg  = 2.0*c0/(gam-1.0);      // background |J+-|
  real cx    = 0.5*domainSize[0];          // interface (staticGrid==3) at the centre
  real yMid  = 0.5*domainSize[1];

  std::vector<std::array<real,6>> rows;    // x, rho, u, p, dJ+, dJ-
  real maxJp = 0, maxJmLeft = 0;

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]*powi(2,lvl)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,lvl)/blockSize;
    if (ib < 0 || jb < 0 || ib >= gx || jb >= gy) continue;   // interior only

    real dxl = domainSize[0]/real(baseGridSize[0]*powi(2,lvl));
    real dyl = domainSize[1]/real(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;               // leaf cells only
      i32 i =  c % blockSize;
      i32 j = (c/blockSize) % blockSize;
      real y = (jb*blockSize + j + 0.5)*dyl;
      if (fabs(y - yMid) > 0.5*dyl) continue;                 // centreline row only

      real r = Rho[cIdx];
      real u = RhoU[cIdx]/r;
      real v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      real p = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      real cc = sqrt(gam*fmax(p,(real)1e-30)/r);
      real x  = (ib*blockSize + i + 0.5)*dxl;
      real dJp = (u + 2.0*cc/(gam-1.0)) - Jbg;
      real dJm = (u - 2.0*cc/(gam-1.0)) + Jbg;
      rows.push_back({x, r, u, p, dJp, dJm});
      maxJp = fmax(maxJp, fabs(dJp));
      if (x < cx - 0.02*domainSize[0])                        // coarse half = reflected region
        maxJmLeft = fmax(maxJmLeft, fabs(dJm));
    }
  }

  std::sort(rows.begin(), rows.end(),
            [](const std::array<real,6>&a, const std::array<real,6>&b){ return a[0] < b[0]; });
  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "# x rho u p dJplus dJminus   (interface x=%.4f, c0=%.4f)\n", cx, c0);
  for (auto &r : rows)
    fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e\n", r[0], r[1], r[2], r[3], r[4], r[5]);
  fclose(fp);

  printf("[acoustic] interface x=%.3f  incident max|dJ+|=%.4e\n", cx, maxJp);
  printf("[acoustic] reflected max|dJ-| (coarse half) = %.4e  ->  reflection R = %.4e (%.4f%%)\n",
         maxJmLeft, (maxJp>0? maxJmLeft/maxJp : 0.0), (maxJp>0? 100.0*maxJmLeft/maxJp : 0.0));
  printf("[acoustic] wrote %zu centreline samples to %s\n", rows.size(), fileName);
}

//
// L2 velocity error for the periodic sine acoustic wave (icType 6), evaluated
// after an integer number of periods where the exact solution equals the IC
// u_exact = A sin(kx).  Uses the velocity (zero background) so float precision
// scales with the wave amplitude.  Prints absolute and amplitude-relative L2 for
// order-of-accuracy studies.
//
//
// Viscous shear-wave decay (icType 8): u = U0 sin(k y), v = w = 0 on a periodic
// box is an EXACT steady solution of the Euler equations (u varies only across
// the flow, so u du/dx = 0 and d(rho u)/dx = 0), and under constant-mu
// Navier-Stokes it decays exactly as
//     u(y,t) = U0 exp(-nu k^2 t) sin(k y),   nu = mu/rho.
// The nonlinear term stays identically zero throughout, so ANY error here is
// the viscous operator: this measures both the coefficient (is nu right?) and
// the spatial order.  Viscous heating contaminates the state at O(Ma^2); the
// test case runs at Ma ~ 0.01 to keep that far below the discretization error.
//
void CompressibleSolver::computeShearDecayError(real t) {
  cudaDeviceSynchronize();
  real *Rho  = getField(F_RHO);
  real *RhoU = getField(F_RHOU);
  real U0 = vortexAdvect;                     // reused as the shear amplitude
  real k  = 2.0*PI/domainSize[1];
  real nu = mu/1.0;                           // rho = 1 in this IC
  real decay = exp(-nu*k*k*t);

  // amp is the DISCRETE SINE COEFFICIENT (2/N) sum u_j sin(k y_j), not a max:
  // cell centres never land on the sine peak (at N=32 the nearest is 0.995 of
  // it), so a max would report that sampling artifact as scheme error.  The
  // projection is exact for a uniformly sampled full period, which makes
  // amp/(U0 exp(-nu k^2 t)) a clean check on the viscous coefficient itself.
  double err2 = 0.0, ref2 = 0.0, proj = 0.0; long n = 0;
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]*powi(2,lvl)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,lvl)/blockSize;
    if (ib < 0 || jb < 0 || ib >= gx || jb >= gy) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;
#endif
    real dyl = domainSize[1]/real(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 j = (c/blockSize) % blockSize;
      real y  = (jb*blockSize + j + 0.5)*dyl;
      real u  = RhoU[cIdx]/Rho[cIdx];
      real ue = U0*decay*sin(k*y);
      err2 += double(u-ue)*double(u-ue);
      ref2 += double(ue)*double(ue);
      proj += double(u)*double(sin(k*y));
      n++;
    }
  }
#ifdef USE_MGPU
  double red[4] = {err2, ref2, proj, (double)n}; comm::allreduceSum(red, 4);
  err2 = red[0]; ref2 = red[1]; proj = red[2]; n = (long)red[3];
#endif
  double l2 = sqrt(err2/double(n));
  printf("---- viscous shear-wave decay (mu=%g, Pr=%g, t=%g) ----\n", (double)mu, (double)Pr, (double)t);
  printf("  N=%d  nu k^2 t = %.6f   exact decay = %.6e\n",
         baseGridSize[1], (double)(nu*k*k*t), (double)decay);
  double amp = 2.0*proj/double(n);
  printf("  projected amplitude = %.6e   exact = %.6e   ratio = %.6f\n",
         amp, (double)(U0*decay), amp/double(U0*decay));
  printf("  L2(u error) = %.6e   L2 relative = %.6e\n", l2, l2/sqrt(ref2/double(n)));
}

void CompressibleSolver::computeAcousticL2Error(void) {
  cudaDeviceSynchronize();
  real *Rho  = getField(F_RHO);
  real *RhoU = getField(F_RHOU);
  real A  = vortexAdvect;
  real k  = 2.0*PI/domainSize[0];

  double err2 = 0.0; long n = 0;
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]*powi(2,lvl)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,lvl)/blockSize;
    if (ib < 0 || jb < 0 || ib >= gx || jb >= gy) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;   // owned-only: exclude ghost duplicates
#endif
    real dxl = domainSize[0]/real(baseGridSize[0]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize;
      real x = (ib*blockSize + i + 0.5)*dxl;
      real u = RhoU[cIdx]/Rho[cIdx];
      real ue = A*sin(k*x);
      err2 += double(u-ue)*double(u-ue);
      n++;
    }
  }
#ifdef USE_MGPU
  double red[2] = {err2, (double)n}; comm::allreduceSum(red, 2);   // global norm over all PEs
  err2 = red[0]; n = (long)red[1];
#endif
  double l2 = sqrt(err2/double(n));
  printf("[acoustic-conv] N=%d  L2(u) = %.6e   L2(u)/A = %.6e\n",
         baseGridSize[0], l2, l2/A);
}

//
// Scan active interior cells for pressure spikes (over/undershoots beyond the
// initial pressure range [0.1, 1.0] are unphysical for a Sod explosion) and
// for spurious z-momentum (should be ~0 in pseudo-2D).  Reports whether the
// worst spikes sit on cells adjacent to a refinement-level change.
//
void CompressibleSolver::printDiagnostics(void) {
  cudaDeviceSynchronize();
  real *Rho  = getField(0);
  real *RhoU = getField(1);
  real *RhoV = getField(2);
  real *RhoW = getField(3);
  real *RhoE = getField(4);

  real maxU = 0, maxV = 0, maxW = 0;
  real minP = 1e30, maxP = -1e30;
  i64 nActive = 0, nSpike = 0;

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]*powi(2,lvl)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,lvl)/blockSize;
    i32 gz = baseGridSize[2]*powi(2,lvl)/blockSizeZ;
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;

    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      real r = Rho[cIdx];
      if (r <= 0) continue;
      real u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      real p = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      maxU = fmax(maxU, fabs(u));
      maxV = fmax(maxV, fabs(v));
      maxW = fmax(maxW, fabs(w));
      minP = fmin(minP, p);
      maxP = fmax(maxP, p);
      // spike bounds = 2% above / below the IC pressure range [0.1, pHi]
      real pHi = (vortexAdvect > 0.0) ? vortexAdvect : 1.0;
      if (p > 1.02*pHi || p < 0.098) nSpike++;
      nActive++;
    }
  }

  real pHi = (vortexAdvect > 0.0) ? vortexAdvect : 1.0;
  printf("---- diagnostics ----\n");
  printf("  active cells : %lld\n", (long long)nActive);
  printf("  max|u| = %.4e  max|v| = %.4e  max|w| = %.4e   (w should be ~0)\n", maxU, maxV, maxW);
  printf("  pressure range: [%.4f, %.4f]   (init range [0.1, %.1f])\n", minP, maxP, pHi);
  printf("  pressure-spike cells (p>%.2f or p<0.098): %lld  (%.3f%%)\n",
         1.02*pHi, (long long)nSpike, 100.0*real(nSpike)/fmax(1.0,real(nActive)));

  // per-level z-block extent: shows whether refinement also subdivides z
  printf("  per-level interior-block z-extent (nz blocks = max kb+1):\n");
  for (i32 L = 0; L < nLvls; L++) {
    i32 nzMax = 0, nBlk = 0, nWall = 0;   // nWall: blocks touching the domain boundary
    i32 gx = baseGridSize[0]*powi(2,L)/blockSize;
    i32 gy = baseGridSize[1]*powi(2,L)/blockSize;
    i32 gz = baseGridSize[2]*powi(2,L)/blockSizeZ;
    for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
      u64 loc = bLocList[bIdx];
      if (loc == kEmpty) continue;
      i32 lvl = loc >> 60;
      if (lvl != L) continue;
      i32 kb = ((loc >> 40) & ((1 << 20)-1)) - 1;
      i32 jb = ((loc >> 20) & ((1 << 20)-1)) - 1;
      i32 ib = ( loc        & ((1 << 20)-1)) - 1;
      if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
      nzMax = max(nzMax, kb+1);
      nBlk++;
      if (ib == 0 || ib == gx-1 || jb == 0 || jb == gy-1) nWall++;   // wall-adjacent
    }
    printf("    lvl %d: %d interior blocks (%d wall-adjacent = %.1f%%), nz = %d block(s) thick\n",
           L, nBlk, nWall, nBlk ? 100.0*nWall/nBlk : 0.0, nzMax);
  }
  printf("---------------------\n");
}

//
// L2 error of the current solution against the exact STATIONARY isentropic
// vortex (icType 2, vortexAdvect 0), summed over active interior cells.  For a
// stationary exact solution the error measures how well the scheme preserves the
// equilibrium; lower is better.
//
//
// L2 error against the EXACT supersonic-vortex solution (testCase 16).
//
// Reported twice, because the two answer different questions:
//   "annulus"  -- every fluid cell between the walls.  Includes the smeared
//                 interface, which is where a penalization's model error lives.
//   "interior" -- excludes a FIXED PHYSICAL band (0.05 r_i) at each wall, so the
//                 measured region is identical on every grid.  This is the honest
//                 order-of-accuracy number; the annulus norm changes region with
//                 h whenever delta scales with h.
// Comparing them says whether the error is wall-localised or spread.
//
// L2 error vs the exact Ringleb solution, split by distance from the curved
// streamline wall -- same reporting as the annulus case so the two are directly
// comparable.
// Which KIND of boundary a non-fluid cell represents.  Default is a wall; only
// bodies that have a genuine inflow/outflow segment override it.
__host__ __device__ i32 CompressibleSolver::getBoundaryBcKind(Vec3 pos) {
  // --ibdir 1: the analytic arcs carry the exact solution (ghost path: the
  // prescribed ghosts keep their exact initial state; RCCM reads ibDirichlet
  // directly, this only classifies the ghosts)
  if (immerserdBcType == 7 && ibDirichlet) return 1;
  if (immerserdBcType == 6 && ibPolyBc) {
    // Nearest-segment tag.  Its own search -- caching the index from the
    // level-set call in a shared member would be a data race across threads.
    // Stamped once per adaptation, so the extra pass is free.  Note it never
    // inverts anything, which is the point: the per-cell hodograph inversion
    // collapsed at the q = qmin arc (disc -> 0) and silently classified the
    // inflow as a WALL, which walls off the flow.
    // Nearest WALL and nearest PRESCRIBED segment, kept separately.  A plain
    // nearest-segment tag flips discontinuously at a junction between the two,
    // and a cell that lands on the wall side there gets a mirror ghost across
    // what is actually the outlet -- which showed up as rho = 1.14 against a
    // physical maximum of 0.92 in the corner where the outlet arc meets the
    // inner wall.  Occluding an inflow/outflow face with a spurious wall is the
    // damaging error; carrying the known exact state one cell too far is not.
    // So near a junction, prefer PRESCRIBED.
    real dW = (real)1e30, dP = (real)1e30;
    for (i32 e = 0; e < ibPolyN; e++) {
      const i32 f = (e + 1 == ibPolyN) ? 0 : e + 1;
      const real ax = ibPoly[2*e], ay = ibPoly[2*e+1];
      const real bx = ibPoly[2*f], by = ibPoly[2*f+1];
      const real ex = bx - ax, ey = by - ay;
      const real L2 = ex*ex + ey*ey;
      real t = (L2 > (real)0) ? ((pos[0]-ax)*ex + (pos[1]-ay)*ey)/L2 : (real)0;
      t = fmin(fmax(t, (real)0), (real)1);
      const real qx = pos[0] - (ax + t*ex), qy = pos[1] - (ay + t*ey);
      const real d2 = qx*qx + qy*qy;
      if (ibPolyBc[e] == 0) { if (d2 < dW) dW = d2; }
      else                  { if (d2 < dP) dP = d2; }
    }
    const real tol = ibPolyBcTol*ibChord;      // junction width
    return (sqrt(dP) <= sqrt(dW) + tol) ? 1 : 0;
  }
  if (immerserdBcType == 9) {
    // Ringleb: the two streamlines are WALLS (flow is tangent), but the q = qmin
    // arc is where flow enters and leaves.  Marking that arc a wall blocks the
    // flow and no steady solution exists -- it must carry the exact state.
    real q, k;
    if (!ringlebInvert(pos[0] + ringlebX0, fabs(pos[1] + ringlebY0), q, k)) return 0;
    return (q < ringlebQmin) ? 1 : 0;
  }
  return 0;
}

void CompressibleSolver::computeRinglebError(void) {
  cudaDeviceSynchronize();
  real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoV = getField(F_RHOV);
  real *RhoW = getField(F_RHOW), *RhoE = getField(F_RHOE);
  double l2[2] = {0,0}, l2r[2] = {0,0}, ar[2] = {0,0};
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib = ( loc        & ((1 << 20)-1)) - 1;
    double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
    double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
    i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
    i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
    if (ib < 0 || jb < 0 || ib >= gx || jb >= gy) continue;
    const double band = 3.0*fmin(dxL, dyL);   // "near wall" = 3 cells
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize, j = (c/blockSize) % blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      double x = (ib*blockSize + i + 0.5)*dxL, y = (jb*blockSize + j + 0.5)*dyL;
      if (getField(F_IBM)[cIdx] == (real)0) continue;      // solid / non-fluid
      real re_, ue_, ve_, pe_;
      ringlebExact((real)x, (real)y, re_, ue_, ve_, pe_);
      double r = Rho[cIdx];
      if (!(r > 0)) continue;
      double u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      (void)w;
      double dv2 = (u-(double)ue_)*(u-(double)ue_) + (v-(double)ve_)*(v-(double)ve_);
      double dr2 = (r-(double)re_)*(r-(double)re_);
      double A = dxL*dyL;
      const double gap = -(double)getField(F_PHI)[cIdx];   // phi > 0 INSIDE the body
      for (i32 z = 0; z < 2; z++) {
        if (z == 1 && gap < band) continue;
        l2[z] += dv2*A; l2r[z] += dr2*A; ar[z] += A;
      }
    }
  }
  printf("---- Ringleb L2 error (vs EXACT) ----\n");
  for (i32 z = 0; z < 2; z++)
    printf("  %-9s L2(rho) = %.6e   L2(|u|) = %.6e   (area %.4f)\n",
           z ? "interior" : "all      ", ar[z] > 0 ? sqrt(l2r[z]/ar[z]) : 0.0,
           ar[z] > 0 ? sqrt(l2[z]/ar[z]) : 0.0, ar[z]);
  printf("-------------------------------------\n");
}

// Paper Sect. 4.2 metrics for the canal (case 18).  Mass flow rate through
// two x = const sections, Eq. (38)/(39) -- summed over the cells of a grid
// column with the WET height alpha*dy so a cut floor/ceiling cell counts its
// fluid part only -- and the pressure coefficient along the floor, taken from
// the lowest live cell of each column at its fluid centroid (the paper's Cp is
// the wall-adjacent cell value; the R-Cell state IS the wall-frame fit).
void CompressibleSolver::computeCanalMetrics(const char *cpFile) {
  cudaDeviceSynchronize();
  real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoV = getField(F_RHOV);
  real *RhoW = getField(F_RHOW), *RhoE = getField(F_RHOE), *Al = getField(F_CUTA);
  const i32 nx = baseGridSize[0];                       // base-level columns only
  std::vector<double> mdot(nx, 0.0), cpLo(nx, 0.0), yLo(nx, 1e30), xCol(nx, 0.0);
  std::vector<i32> nCol(nx, 0);
  const double q = 0.5*(double)canalRhoIn*(double)canalUin*(double)canalUin;
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    if (lvl != 0) continue;   // uniform-grid case; the metrics assume one level
    i32 kb = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]/blockSize, gy = baseGridSize[1]/blockSize;
    if (ib < 0 || jb < 0 || kb != 0 || ib >= gx || jb >= gy) continue;
    const double dxL = domainSize[0]/double(baseGridSize[0]);
    const double dyL = domainSize[1]/double(baseGridSize[1]);
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize, j = (c/blockSize) % blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const i32 col = ib*blockSize + i;
      const double x = (col + 0.5)*dxL, y = (jb*blockSize + j + 0.5)*dyL;
      double al = ibRccm ? (double)Al[cIdx] : (isFluidCell(Vec3((real)x,(real)y,0), (real)dyL) ? 1.0 : 0.0);
      if (ibRccm && al <= (double)ibRccmAlphaMin) continue;
      if (!isFluidCell(Vec3((real)x,(real)y,0), (real)fmin(dxL,dyL))) continue;
      const double r = Rho[cIdx], u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      const double p = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      mdot[col] += r*u*al*dyL;  nCol[col]++;  xCol[col] = x;
      if (y < yLo[col]) { yLo[col] = y; cpLo[col] = (p - (double)canalPin)/q; }
    }
  }
  // Eq. (39) between the sections nearest x = 0.25 L and x = 2.75 L (one
  // quarter-length in from each end, clear of the inlet/outlet ghosts)
  auto colAt = [&](double xs) { return (i32)fmin(fmax(floor(xs/domainSize[0]*nx), 0.0), nx - 1.0); };
  const i32 c1 = colAt(0.25), c2 = colAt(domainSize[0] - 0.25);
  const i32 cIn = colAt(0.5*domainSize[0]/nx), cOut = colAt(domainSize[0] - 0.5*domainSize[0]/nx);
  printf("---- canal (paper Sect. 4.2) ----\n");
  printf("  mdot(x=%.3f) = %.8f   mdot(x=%.3f) = %.8f   mass-flow-rate error Eq.(39) = %.3e\n",
         xCol[c1], mdot[c1], xCol[c2], mdot[c2], fabs(mdot[c1]-mdot[c2])/fabs(mdot[c2]));
  printf("  mdot(inlet col) = %.8f   mdot(outlet col) = %.8f   error = %.3e\n",
         mdot[cIn], mdot[cOut], fabs(mdot[cIn]-mdot[cOut])/fabs(mdot[cOut]));
  double mmin = 1e30, mmax = -1e30;
  for (i32 c = 1; c < nx-1; c++) { mmin = fmin(mmin, mdot[c]); mmax = fmax(mmax, mdot[c]); }
  printf("  mdot over all interior columns: min %.8f  max %.8f  spread %.3e\n", mmin, mmax, (mmax-mmin)/mdot[c2]);
  // floor Cp, shock location = steepest Cp rise on the bump
  double dmax = 0; i32 cs = 0;
  for (i32 c = 1; c < nx; c++) if (cpLo[c]-cpLo[c-1] > dmax) { dmax = cpLo[c]-cpLo[c-1]; cs = c; }
  printf("  floor Cp: min %.4f   max %.4f   steepest rise at x = %.4f (bump spans [1,2]; paper frame x-1 = %.3f)\n",
         *std::min_element(cpLo.begin(), cpLo.end()), *std::max_element(cpLo.begin(), cpLo.end()),
         0.5*(xCol[cs]+xCol[cs-1]), 0.5*(xCol[cs]+xCol[cs-1]) - 1.0);
  if (cpFile) {
    FILE *f = fopen(cpFile, "w");
    if (f) {
      fprintf(f, "# x  Cp_floor  mdot(x)  yLowestLiveCell   (canal case 18; bump on [1,2]; Cp = (p-p_in)/(0.5 rho_in u_in^2))\n");
      for (i32 c = 0; c < nx; c++) if (nCol[c]) fprintf(f, "%.6f %.8f %.8f %.6f\n", xCol[c], cpLo[c], mdot[c], yLo[c]);
      fclose(f);
      printf("  floor Cp / mdot(x) -> %s\n", cpFile);
    }
  }
  printf("---------------------------------\n");
}

void CompressibleSolver::computeSvortexError(void) {
  cudaDeviceSynchronize();
  real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoV = getField(F_RHOV);
  real *RhoW = getField(F_RHOW), *RhoE = getField(F_RHOE);
  double l2[2] = {0,0}, l2r[2] = {0,0}, ar[2] = {0,0};
  const double band = 0.05*(double)ibRadius;
  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib = ( loc        & ((1 << 20)-1)) - 1;
    double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
    double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
    i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
    i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
    i32 gz = pseudo2D ? baseGridSize[2]/blockSizeZ : baseGridSize[2]/blockSizeZ*powi(2,lvl);
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize, j = (c/blockSize) % blockSize;
      double x = (ib*blockSize + i + 0.5)*dxL, y = (jb*blockSize + j + 0.5)*dyL;
      const double ddx = x - (double)ibCenter[0], ddy = y - (double)ibCenter[1];
      const double rr = sqrt(ddx*ddx + ddy*ddy);
      const double gap = fmin(rr - (double)ibRadius, (double)ibRadius2 - rr);
      // Use the SOLVER'S OWN fluid classification, not just the cell-centre
      // radius: a cell whose centre is barely outside the wall can still be a
      // ghost under the UTCart "intersecting = non-fluid" rule, and it then
      // holds a mirrored state, not the exact solution.  Measured with the naive
      // geometric test: L2 = 4.6e-2 at t=0 in the wall band, against 2.3e-7 in
      // the interior -- entirely ghost cells, not scheme error.
      if (gap <= 0) continue;                       // solid, or on the wall
      if (!isFluidCell(Vec3((real)x,(real)y,(real)0), (real)fmin(dxL,dyL))) continue;
      real re_, ue_, ve_, pe_;
      svortexExact((real)x, (real)y, re_, ue_, ve_, pe_);
      double r = Rho[cIdx];
      double u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      double p = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      double dv2 = (u-(double)ue_)*(u-(double)ue_) + (v-(double)ve_)*(v-(double)ve_);
      double dr2 = (r-(double)re_)*(r-(double)re_);
      // where does the error actually live?  radius, cut fraction, cell type
      {
        static double wErr = 0; static i32 nrep = 0;
        const double e = sqrt(dv2 + dr2);
        if (dbgChecks && e > wErr) {
          wErr = e;
          if (nrep < 12) {
            const double al = (double)getField(F_CUTA)[cIdx];
            printf("  [err] r=%.4f gap=%.4f alpha=%.3f phi=%+.4f  err=%.3e "
                   "(rho %.4f vs %.4f, |u| %.4f vs %.4f)\n",
                   rr, gap, al, (double)getField(F_PHI)[cIdx], e,
                   r, (double)re_, sqrt(u*u+v*v), sqrt((double)ue_*ue_+(double)ve_*ve_));
            nrep++;
          }
        }
      }
      // debug: name the first cells that poison the norm
      static i32 nBad = 0;
      if (!(isfinite(dv2) && isfinite(dr2)) && ++nBad < 400) {
        if (nBad<6) printf("  [norm] NON-FINITE cell at (%.5f, %.5f) r_pol=%.5f gap=%.4f: "
               "rho=%.3e rhou=%.3e rhov=%.3e rhoE=%.3e | exact rho=%.3e\n",
               x, y, rr, gap, (double)Rho[cIdx], (double)RhoU[cIdx],
               (double)RhoV[cIdx], (double)RhoE[cIdx], (double)re_);
      }
      if (!(isfinite(dv2) && isfinite(dr2))) continue;  // count, then keep the sums finite
      double A = dxL*dyL;
      for (i32 z = 0; z < 2; z++) {
        if (z == 1 && gap < band) continue;
        l2[z] += dv2*A; l2r[z] += dr2*A; ar[z] += A;
      }
      (void)p; (void)pe_;
    }
  }
  { static i32 dummy=0; (void)dummy; }
  printf("---- supersonic vortex L2 error (vs EXACT) ----\n");
  for (i32 z = 0; z < 2; z++)
    printf("  %-9s L2(rho) = %.6e   L2(|u|) = %.6e   (area %.4f)\n",
           z ? "interior" : "annulus", ar[z] > 0 ? sqrt(l2r[z]/ar[z]) : 0.0,
           ar[z] > 0 ? sqrt(l2[z]/ar[z]) : 0.0, ar[z]);
  printf("-----------------------------------------------\n");

  // ---- the paper's table (Ndiaye et al. Sect. 4.4, Eqs. 40, 41, 44) --------
  // L1 / L2 / Linf of rho and p over three cell sets: W = every live cell,
  // I = cells strictly inside (uncut), B = cells on the boundary (cut).  Under
  // RCCM a cut cell's average lives at its FLUID CENTROID and weighs alpha dV,
  // so the exact solution is evaluated there and the R-Cells -- centre in the
  // solid, excluded from the norms above -- are included, as in the paper.
  // Without RCCM (ghost path) "B" is the first fluid row, gap < h.
  {
    real *CA = getField(F_CUTA);
    double s1[3][2] = {{0,0},{0,0},{0,0}}, s2[3][2] = {{0,0},{0,0},{0,0}};
    double sinf[3][2] = {{0,0},{0,0},{0,0}}, vol[3] = {0,0,0};
    i32 ncell[3] = {0,0,0};
    for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
      u64 loc = bLocList[bIdx];
      if (loc == kEmpty) continue;
      i32 lvl = loc >> 60;
      i32 kb = ((loc >> 40) & ((1 << 20)-1)) - 1;
      i32 jb = ((loc >> 20) & ((1 << 20)-1)) - 1;
      i32 ib = ( loc        & ((1 << 20)-1)) - 1;
      double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
      double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
      i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
      i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
      i32 gz = pseudo2D ? baseGridSize[2]/blockSizeZ : baseGridSize[2]/blockSizeZ*powi(2,lvl);
      if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
      for (i32 c = 0; c < blockSizeTot; c++) {
        i32 cIdx = bIdx*blockSizeTot + c;
        if (cFlagsList[cIdx] != ACTIVE) continue;
        i32 i = c % blockSize, j = (c/blockSize) % blockSize;
        double x = (ib*blockSize + i + 0.5)*dxL, y = (jb*blockSize + j + 0.5)*dyL;
        const real h = (real)fmin(dxL, dyL);
        // Under Brinkman isFluidCell is true EVERYWHERE (the body is not
        // masked), so the body interior -- where the penalization holds a
        // stagnant state that has nothing to do with the exact solution --
        // would otherwise be scored.  Use the geometry directly there.
        if (ibBrink) {
          if (getBoundaryLevelSet(Vec3((real)x,(real)y,(real)0)) >= (real)0) continue;
        } else if (!isFluidCell(Vec3((real)x,(real)y,(real)0), h)) continue;
        double w = dxL*dyL, xe = x, ye = y;
        bool bnd;
        if (ibRccm) {
          const double al = (double)CA[cIdx];
          if (!(al > (double)ibRccmAlphaMin)) continue;          // dead
          bnd = al < 1.0 - 1e-12;
          if (bnd) {
            real f[4], a2, ax2, ay2, cx2 = 0.5, cy2 = 0.5;
            f[0] = getBoundaryLevelSet(Vec3((real)(x-0.5*dxL), (real)(y-0.5*dyL), 0));
            f[1] = getBoundaryLevelSet(Vec3((real)(x+0.5*dxL), (real)(y-0.5*dyL), 0));
            f[2] = getBoundaryLevelSet(Vec3((real)(x+0.5*dxL), (real)(y+0.5*dyL), 0));
            f[3] = getBoundaryLevelSet(Vec3((real)(x-0.5*dxL), (real)(y+0.5*dyL), 0));
            rccmCutGeom(f, a2, ax2, ay2, &cx2, &cy2);
            xe = x + ((double)cx2 - 0.5)*dxL;  ye = y + ((double)cy2 - 0.5)*dyL;
          }
          w *= al;
        } else {
          const double ddx = x - (double)ibCenter[0], ddy = y - (double)ibCenter[1];
          const double rr = sqrt(ddx*ddx + ddy*ddy);
          bnd = fmin(rr - (double)ibRadius, (double)ibRadius2 - rr) < (double)h;
        }
        real re_, ue_, ve_, pe_;
        svortexExact((real)xe, (real)ye, re_, ue_, ve_, pe_);
        const double r = Rho[cIdx];
        const double u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, wz = RhoW[cIdx]/r;
        const double pp = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+wz*wz));
        const double e[2] = {fabs(r - (double)re_), fabs(pp - (double)pe_)};
        if (!(isfinite(e[0]) && isfinite(e[1]))) continue;
        const i32 regs[2] = {0, bnd ? 2 : 1};                     // W, then I or B
        for (i32 q = 0; q < 2; q++) {
          const i32 z = regs[q];
          vol[z] += w; ncell[z]++;
          for (i32 f = 0; f < 2; f++) {
            s1[z][f] += w*e[f]; s2[z][f] += w*e[f]*e[f];
            if (e[f] > sinf[z][f]) sinf[z][f] = e[f];
          }
        }
      }
    }
    printf("---- paper table: region  cells      L1(rho)      L2(rho)    Linf(rho)"
           "        L1(p)        L2(p)      Linf(p) ----\n");
    const char *nm[3] = {"W (all)", "I (uncut)", "B (cut)"};
    for (i32 z = 0; z < 3; z++)
      printf("  %-10s %7d  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n", nm[z], ncell[z],
             vol[z] > 0 ? s1[z][0]/vol[z] : 0.0, vol[z] > 0 ? sqrt(s2[z][0]/vol[z]) : 0.0,
             sinf[z][0],
             vol[z] > 0 ? s1[z][1]/vol[z] : 0.0, vol[z] > 0 ? sqrt(s2[z][1]/vol[z]) : 0.0,
             sinf[z][1]);
    printf("-----------------------------------------------\n");
  }
}

void CompressibleSolver::computeVortexError(void) {
  cudaDeviceSynchronize();
  real *Rho  = getField(F_RHO);
  real *RhoU = getField(F_RHOU);
  real *RhoV = getField(F_RHOV);
  real *RhoW = getField(F_RHOW);
  real *RhoE = getField(F_RHOE);

  const double eps = 5.0, PId = 3.14159265358979323846;
  double cx = 0.5*domainSize[0], cy = 0.5*domainSize[1];

  double l2Rho = 0, l2Vel = 0, l2P = 0, area = 0;

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    // level-aware cell size + interior test (active cells live at the finest level)
    double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
    double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
    i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
    i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
    i32 gz = pseudo2D ? baseGridSize[2]/blockSizeZ : baseGridSize[2]/blockSizeZ*powi(2,lvl);
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;   // skip ghost (halo) blocks in the error norm
#endif

    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize, j = (c/blockSize) % blockSize, k = c/blockSize/blockSize;
      double x = (ib*blockSize + i + 0.5)*dxL;
      double y = (jb*blockSize + j + 0.5)*dyL;

      // exact stationary vortex
      double ddx = x - cx, ddy = y - cy, r2 = ddx*ddx + ddy*ddy;
      double f  = eps/(2.0*PId)*exp(0.5*(1.0 - r2));
      double ue = -f*ddy, ve = f*ddx;
      double dT = -(gam-1.0)*eps*eps/(8.0*gam*PId*PId)*exp(1.0 - r2);
      double Te = fmax(1.0 + dT, 1e-6);
      double re = pow(Te, 1.0/(gam-1.0)), pe = pow(Te, gam/(gam-1.0));

      double r = Rho[cIdx];
      double u = RhoU[cIdx]/r, v = RhoV[cIdx]/r, w = RhoW[cIdx]/r;
      double p = (gam-1.0)*(RhoE[cIdx] - 0.5*r*(u*u+v*v+w*w));
      double dvel = sqrt((u-ue)*(u-ue) + (v-ve)*(v-ve));
      double cellA = dxL*dyL;
      l2Rho += (r-re)*(r-re)*cellA;
      l2Vel += dvel*dvel*cellA;
      l2P   += (p-pe)*(p-pe)*cellA;
      area  += cellA;
    }
  }
#ifdef USE_MGPU
  // global norm: each rank summed only its OWNED cells
  double red[4] = {l2Rho, l2Vel, l2P, area};
  comm::allreduceSum(red, 4);
  l2Rho = red[0]; l2Vel = red[1]; l2P = red[2]; area = red[3];
  if (part.rank != 0) return;
#endif
  printf("---- vortex L2 error (vs exact stationary) ----\n");
  printf("  L2(rho) = %.4e   L2(|u|) = %.4e   L2(p) = %.4e\n",
         sqrt(l2Rho/area), sqrt(l2Vel/area), sqrt(l2P/area));
  printf("-----------------------------------------------\n");
}

//
// Gresho vortex diagnostic: L2 velocity error vs the exact (stationary) profile
// and the kinetic-energy retention  KE(t)/KE(0).  The Gresho vortex is a steady
// state, so a low-Mach-preserving scheme keeps KE(t)/KE(0) ≈ 1 and a small L2
// error; a dissipative scheme bleeds kinetic energy (retention < 1) at a rate
// that grows as the Mach number drops.
//
void CompressibleSolver::computeGreshoError(void) {
  cudaDeviceSynchronize();
  real *Rho  = getField(F_RHO);
  real *RhoU = getField(F_RHOU);
  real *RhoV = getField(F_RHOV);

  double cx = 0.5*domainSize[0], cy = 0.5*domainSize[1];
  double l2Vel = 0, keNum = 0, keExact = 0, area = 0;

  // radial error bins over [0, 0.5] to localize error relative to the AMR shells
  const i32 NBIN = 10;
  double binErr[NBIN] = {0}, binArea[NBIN] = {0};

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    double dxL = domainSize[0]/double(baseGridSize[0]*powi(2,lvl));
    double dyL = domainSize[1]/double(baseGridSize[1]*powi(2,lvl));
    i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
    i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
    i32 gz = pseudo2D ? baseGridSize[2]/blockSizeZ : baseGridSize[2]/blockSizeZ*powi(2,lvl);
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;   // skip ghost (halo) blocks in the error norm
#endif

    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      i32 i = c % blockSize, j = (c/blockSize) % blockSize;
      double x = (ib*blockSize + i + 0.5)*dxL;
      double y = (jb*blockSize + j + 0.5)*dyL;

      // exact Gresho velocity
      double ddx = x - cx, ddy = y - cy, r = sqrt(ddx*ddx + ddy*ddy);
      double wang = (r < 0.2) ? 5.0 : (r < 0.4 ? 2.0/r - 5.0 : 0.0);
      double ue = -wang*ddy, ve = wang*ddx;

      double rr = Rho[cIdx];
      double u = RhoU[cIdx]/rr, v = RhoV[cIdx]/rr;
      double cellA = dxL*dyL;
      double e2 = ((u-ue)*(u-ue) + (v-ve)*(v-ve)) * cellA;
      l2Vel   += e2;
      keNum   += 0.5*rr*(u*u + v*v) * cellA;
      keExact += 0.5*1.0*(ue*ue + ve*ve) * cellA;
      area    += cellA;
      i32 b = (i32)(r / 0.05);           // 0.05-wide radial bins
      if (b >= 0 && b < NBIN) { binErr[b] += e2; binArea[b] += cellA; }
    }
  }
  printf("---- Gresho vortex diagnostic (Ma=%.3g, %s grid) ----\n",
         greshoP0 > 0 ? 1.0/sqrt(gam*greshoP0) : 0.0,
         staticGrid ? "static-AMR" : (nLvls > 1 ? "adaptive" : "uniform"));
  printf("  L2(vel error) = %.4e   KE retention KE(t)/KE(0) = %.5f\n",
         sqrt(l2Vel/area), keNum/keExact);
  if (staticGrid) {
    printf("  refinement shell radii R(L)=refineRadius*(nLvls-L)/(nLvls-1):");
    for (i32 L = 1; L < nLvls; L++) printf(" R(%d)=%.3f", L, refineRadius*double(nLvls-L)/double(nLvls-1));
    printf("\n  RMS(vel err) per radial bin (marks AMR-boundary bins):\n");
    for (i32 b = 0; b < NBIN; b++) {
      if (binArea[b] <= 0) continue;
      double r0 = b*0.05, r1 = r0+0.05;
      bool onBnd = false;
      for (i32 L = 2; L < nLvls; L++) { double R = refineRadius*double(nLvls-L)/double(nLvls-1); if (R >= r0 && R < r1) onBnd = true; }
      printf("    r=[%.2f,%.2f): rms=%.4e %s\n", r0, r1, sqrt(binErr[b]/binArea[b]), onBnd ? "  <-- AMR boundary" : "");
    }
  }
  printf("-------------------------------------------------------\n");
}

//
// Domain totals of the conserved variables (mass, x-momentum, energy), summed
// over active interior cells with level-dependent cell volumes.  On a closed or
// not-yet-reached-boundary problem these are exactly conserved by the physics, so
// their drift over time measures the coarse/fine interface conservation error.
//
void CompressibleSolver::totalConserved(double &mass, double &momx, double &energy) {
  cudaDeviceSynchronize();
  real *Rho = getField(F_RHO), *RhoU = getField(F_RHOU), *RhoE = getField(F_RHOE);
  mass = 0; momx = 0; energy = 0;

  for (i32 bIdx = 0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    if (loc == kEmpty) continue;
    i32 lvl = loc >> 60;
    i32 kb  = ((loc >> 40) & ((1 << 20)-1)) - 1;
    i32 jb  = ((loc >> 20) & ((1 << 20)-1)) - 1;
    i32 ib  = ( loc        & ((1 << 20)-1)) - 1;
    i32 gx = baseGridSize[0]/blockSize*powi(2,lvl);
    i32 gy = baseGridSize[1]/blockSize*powi(2,lvl);
    i32 gz = pseudo2D ? baseGridSize[2]/blockSizeZ : baseGridSize[2]/blockSizeZ*powi(2,lvl);
    if (ib < 0 || jb < 0 || kb < 0 || ib >= gx || jb >= gy || kb >= gz) continue;
#ifdef USE_MGPU
    if (!isOwnedBlock(lvl, ib, jb, kb)) continue;   // owned-only: exclude ghost duplicates
#endif
    double dV = (double)(domainSize[0]/double(baseGridSize[0]*powi(2,lvl)))
              * (double)(domainSize[1]/double(baseGridSize[1]*powi(2,lvl)))
              * (double)(pseudo2D ? domainSize[2]/double(baseGridSize[2])
                                  : domainSize[2]/double(baseGridSize[2]*powi(2,lvl)));
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = bIdx*blockSizeTot + c;
      if (cFlagsList[cIdx] != ACTIVE) continue;
      // CUT CELLS: the conserved quantity in a cut cell is rho * (alpha dV),
      // not rho * dV.  Without this the totals are dominated by the DEAD cells
      // buried in the body -- which are never advanced and just hold their
      // initial state -- so the diagnostic cannot see a wall-treatment
      // conservation error at all.  alpha lives in F_CUTA under --ibrccm.
      double aC = 1.0;
      if (ibRccm && immerserdBcType) {
        aC = (double)getField(F_CUTA)[cIdx];
        if (cutSplit && cutSplitId && cutSplitId[cIdx] >= 0) {
          // split cell: piece 0 carries this cell's state, every extra piece its owner's
          const CutSplitCell &Sc = cutSplitCell[cutSplitId[cIdx]];
          aC = (double)Sc.a0;
          for (i32 p = 0; p < Sc.n; p++) {
            const CutPiece &P = cutPiece[Sc.first + p];
            if (P.owner == CUT_DEAD) continue;
            if (cutIsPiece(P.owner)) {
              const i32 k = cutPieceOf(P.owner); const size_t cap = cutPieceQCap;
              mass   += cutPieceQ[k]       * dV * (double)P.a;
              momx   += cutPieceQ[cap + k] * dV * (double)P.a;
              energy += cutPieceQ[4*cap+k] * dV * (double)P.a;
            } else {
              mass   += Rho [P.owner] * dV * (double)P.a;
              momx   += RhoU[P.owner] * dV * (double)P.a;
              energy += RhoE[P.owner] * dV * (double)P.a;
            }
          }
        }
        if (!(aC > (double)ibRccmAlphaMin)) continue;   // dead cell: no fluid in it
      }
      mass   += Rho [cIdx] * dV * aC;
      momx   += RhoU[cIdx] * dV * aC;
      energy += RhoE[cIdx] * dV * aC;
    }
  }
}

void CompressibleSolver::paintPressure(const char *fileName) {
  computePressureKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  paintField(F_SCRATCH, fileName);   // scratch field now holds pressure
}

// Diagnostic: paint the per-cell normalized wavelet-detail indicator, exactly as
// the thresholding kernel sees it (predicting from the live fields; mutates only
// F_SCRATCH).  Pixel value saturates at the refine trigger (2*waveletThresh), so
// white == would refine.  mode: 0 = max primary, 1 = rho, 2 = momentum, 3 = rhoE.
void CompressibleSolver::paintDetail(const char *fileName, i32 mode) {
  restrictFields();                            // parents = child averages (as at adapt time)
  // cudaMemset is construction-only (it also serializes the stream); zero by
  // kernel so the launch queues like everything else.
  zeroScalesKernel<<<1, 32>>>(*this);
  computeGlobalScalesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  detailToScratchKernel<<<cudaGridSize, cudaBlockSize>>>(*this, mode);
  cudaDeviceSynchronize();
  paintField(F_SCRATCH, fileName);
}

// ---------------------------------------------------------------------------
//  device helper functions
// ---------------------------------------------------------------------------

__device__ real CompressibleSolver::lim(real &r) {
  // smooth TVD limiter
  return ((r > 0.0 && r < 1.0) ? (2.0*r + r*r*r) / (1.0 + 2.0*r*r) : r);
}

__device__ real CompressibleSolver::tvdRec(real &ul, real &uc, real &ur, real theta, i32 rc) {
  // NVD-form face reconstruction: face = ul + psi(phi)*(ur - ul), where
  // phi = (uc-ul)/(ur-ul) is the normalised variable (== the limiter ratio r).
  // The stencil (ul,uc,ur) is ordered upwind->downwind, so the same formula
  // serves both sides of a face (callers pass the mirrored stencil for qR).
  real du  = ur - ul;
  real phi = (uc - ul) / (copysign(1.0, du)*fmax(abs(du), (real)1e-32));
  real psi;
  const i32 rcx = (rc < 0) ? recon : rc;      // --reconfar: a far cell asks for its own 1-D scheme
  if (rcx == 1) {
    // ROUND (Huang et al. 2026, Eq. 4.1)
    if (phi <= 0.0)
      psi = fmax(fmin((real)0.5*phi + (real)0.5, (real)-1.5*phi - (real)1.8),
                 (real)1.5*phi);
    else if (phi <= 1.0)
      psi = fmin(fmin((real)2.5*phi, (real)(1.0/3.0) + (real)(5.0/6.0)*phi),
                 (real)(3.0/50.0)*phi + (real)(47.0/50.0));
    else
      psi = (real)0.5*phi + (real)0.5;
  }
  else if (rcx == 2) {
    // LD-ROUND (Huang et al. 2026, Eq. 4.2): ROUND blended toward the 3rd-order
    // upwind line psi = 1/3 + (5/6)phi with quartic weights about phi = 1/2
    if (phi <= 0.0) {
      psi = fmax(fmin((real)0.5*phi + (real)0.5, (real)-1.5*phi - (real)1.8),
                 (real)1.5*phi);
    } else if (phi <= 0.5) {
      real dp  = phi - (real)0.5;
      real w   = (real)1.0 / ((real)1.0 + (real)180.0*dp*dp*dp*dp);
      real hi3 = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
      psi = fmin((real)2.5*phi, w*hi3 + ((real)1.0 - w)*(real)2.5*phi);
    } else if (phi <= 1.0) {
      real dp  = phi - (real)0.5;
      real w   = (real)1.0 / ((real)1.0 + (real)600.0*dp*dp*dp*dp);
      real hi3 = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
      real hid = (real)(3.0/50.0)*phi + (real)(47.0/50.0);
      psi = fmin(hid, w*hi3 + ((real)1.0 - w)*hid);
    } else {
      psi = (real)0.5*phi + (real)0.5;
    }
  }
  else if (rcx == 4) {
    // van Leer harmonic limiter in NVD form: MUSCL face u_c + (B(r)/2)(u_c-u_l)
    // with B(r) = 2r/(1+r) maps to psi = phi(2-phi) on 0 < phi < 1 (parabola
    // through (0,0),(1,1)); outside the TVD region fall back to upwind (u_c).
    psi = (phi > 0.0 && phi < 1.0) ? phi*((real)2.0 - phi) : phi;
  }
  else if (rcx == 3) {
    // unlimited 3rd-order upwind parabola (kappa = 1/3 MUSCL): the psi-line the
    // ROUND schemes blend toward, with no limiting at all.
    //   face = -1/6 ul + 5/6 uc + 1/3 ur
    // For SMOOTH tests only -- oscillates at shocks.
    psi = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
  }
  else if (rcx == 6) {
    // gradient MUSCL: the face states are built in the RHS kernel from a
    // limited least-squares gradient, so this 1-D path is never consulted.
    psi = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
  }
  else if (rcx == 5) {
    // Ducros-blended third order: the unlimited kappa=1/3 parabola pulled
    // toward the van Leer limiter by the per-face shock sensor theta
    // (max of the two adjacent cells' compression sensors, from F_RHOK).
    // theta ~ 0 in smooth flow -> pure parabola; theta -> 1 in compression
    // -> pure van Leer.  Callers without sensor information (IB traces, RANS
    // scalar transport) use the default theta = 1 and stay safely limited.
    const real psiP = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
    const real psiV = (phi > (real)0 && phi < (real)1) ? phi*((real)2.0 - phi) : phi;
    psi = theta*psiV + ((real)1 - theta)*psiP;
  }
  else {
    // smooth TVD limiter
    psi = lim(phi);
  }
  return ul + psi * du;
}

__device__ Vec5 CompressibleSolver::prim2cons(Vec5 prim) {
  Vec5 cons;
  cons[0] = prim[0];
  cons[1] = prim[1]*prim[0];
  cons[2] = prim[2]*prim[0];
  cons[3] = prim[3]*prim[0];
  cons[4] = prim[4]/(gam-1.0) + 0.5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]);
  return cons;
}

__device__ Vec5 CompressibleSolver::cons2prim(Vec5 cons) {
  Vec5 prim;
  prim[0] = cons[0];
  prim[1] = cons[1]/cons[0];
  prim[2] = cons[2]/cons[0];
  prim[3] = cons[3]/cons[0];
  prim[4] = (gam-1.0)*(cons[4] - 0.5*prim[0]*(prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]));
  return prim;
}

//
// Dynamic viscosity.  sutherS <= 0 gives constant mu; otherwise Sutherland's
// law nondimensionalized on the reference state (T = p/rho with R = 1):
//   mu(T)/mu_ref = (T/Tref)^{3/2} (Tref + S) / (T + S)
//
__device__ real CompressibleSolver::viscosity(real T) {
  if (sutherS <= 0) return mu;
  real Tr = fmax(T/sutherTref, (real)1e-12);
  return mu * Tr*sqrt(Tr) * (sutherTref + sutherS) / (T + sutherS + (real)1e-32);
}

__device__ real CompressibleSolver::tvdRecVanLeer(real ul, real uc, real ur) {
  // NVD form of van Leer's harmonic limiter: psi = phi(2-phi) inside the TVD
  // region, upwind outside it.  (Same expression as `recon == 4` in tvdRec, but
  // reachable independently of the mean-flow reconstruction setting.)
  const real du  = ur - ul;
  const real phi = (uc - ul) / (copysign((real)1.0, du)*fmax(abs(du), (real)1e-32));
  const real psi = (phi > (real)0 && phi < (real)1) ? phi*((real)2 - phi) : phi;
  return ul + psi*du;
}

// Distance to the nearest viscous wall, for the RANS blending functions and the
// Eq. (38)/(A.5) cutoffs.  Analytic per configuration -- no field is stored, so
// it costs nothing across adaptation and is exact on every level.
__host__ __device__ real CompressibleSolver::wallDistance(Vec3 pos) {
  if (wallGeom == 1) {
    // Flat plate starting at x = plateX0, sitting wallOffset BELOW the bottom
    // domain face, so the first face is at a finite wall distance (see the
    // wallOffset comment in the header).  Upstream of the leading edge the
    // nearest wall point IS the leading edge, so the distance wraps around it,
    // which keeps F1 finite and continuous there.
    const real yw   = pos[1] + wallOffset;
    const real dxle = plateX0 - pos[0];
    if (dxle <= 0) return fmax(yw, (real)0);
    return sqrt(dxle*dxle + yw*yw);
  }
  if (wallGeom == 2) {
    // immersed level set: getBoundaryLevelSet is positive INSIDE the body, so
    // the fluid-side distance is its negation.
    return fmax(-getBoundaryLevelSet(pos), (real)0);
  }
  // no wall: d -> infinity drives Gamma1..3 -> 0, hence F1 = 0, which is exactly
  // the freestream (k-epsilon) branch the model wants there.
  return (real)1e30;
}

// Level set of the immersed body: POSITIVE INSIDE, and a signed DISTANCE (so
// |grad phi| = 1 and the normal is just -grad phi).
// Segment-list geometry with a per-segment BC tag.  setAirfoil is this with
// every tag = wall.
void CompressibleSolver::setPolyline(const real *xy, const i32 *bc, const real *st, i32 n) {
  setAirfoil(xy, n);
  if (!bc) return;
  cudaMallocManaged(&ibPolyBc, (size_t)n*sizeof(i32));
  for (i32 i = 0; i < n; i++) ibPolyBc[i] = bc[i];
  cudaDeviceSynchronize();
  if (st) {
    cudaMallocManaged(&ibPolyState, (size_t)4*n*sizeof(real));
    for (i32 i = 0; i < 4*n; i++) ibPolyState[i] = st[i];
  }
  cudaDeviceSynchronize();
  i32 nw = 0; for (i32 i = 0; i < n; i++) nw += (bc[i] == 0);
  printf("[ib] segment tags: %d wall, %d prescribed\n", nw, n - nw);
}

void CompressibleSolver::setAirfoil(const real *xy, i32 n) {
  ibPolyN = n;
  cudaMallocManaged(&ibPoly, (size_t)2*n*sizeof(real));
  for (i32 i = 0; i < 2*n; i++) ibPoly[i] = xy[i];
  ibBox[0] = ibBox[2] = xy[0];
  ibBox[1] = ibBox[3] = xy[1];
  for (i32 i = 0; i < n; i++) {
    ibBox[0] = fmin(ibBox[0], xy[2*i]);   ibBox[2] = fmax(ibBox[2], xy[2*i]);
    ibBox[1] = fmin(ibBox[1], xy[2*i+1]); ibBox[3] = fmax(ibBox[3], xy[2*i+1]);
  }
  ibChord = ibBox[2] - ibBox[0];
  cudaDeviceSynchronize();
  printf("[ib] airfoil: %d points, chord %.6f, bbox [%.4f,%.4f] x [%.4f,%.4f]\n",
         n, (double)ibChord, (double)ibBox[0], (double)ibBox[2],
         (double)ibBox[1], (double)ibBox[3]);
}

//
// ---- Ringleb flow -----------------------------------------------------------
// Exact smooth solution of the 2-D steady compressible potential equations,
// obtained in the hodograph plane.  Streamlines are k = const, so a CURVED wall
// can be placed exactly on one -- which is why it is the standard verification
// case for curved-boundary treatments: the geometry error and the flow error are
// both known analytically.  Stagnation values are rho0 = p0 = a0 = 1.
//
// With q = |u| and k the streamline label:
//   a = sqrt(1 - (g-1)/2 q^2),  rho = a^(2/(g-1)),  p = a^(2g/(g-1))/g
//   J = 1/a + 1/(3a^3) + 1/(5a^5) - (1/2) ln((1+a)/(1-a))
//   x = (1/(2 rho))(2/k^2 - 1/q^2) - J/2
//   y = +- (1/(k rho q)) sqrt(1 - (q/k)^2)
// The map (q,k) -> (x,y) is analytic; the INVERSE needs a solve, done here by
// bisection on q along the k found from the streamline relation.  Bisection (not
// Newton) because the Jacobian is singular at q = k (the streamline apex) and a
// Newton step there walks straight out of the physical range 0 < q <= k.
//
__host__ __device__ void CompressibleSolver::ringlebHodograph(real q, real k,
                        real &x, real &y, real &rho, real &p) const {
  const real g = gam;
  const real a2 = (real)1 - (real)0.5*(g - (real)1)*q*q;
  const real a  = sqrt(fmax(a2, (real)1e-12));
  rho = pow(a, (real)2/(g - (real)1));
  p   = pow(a, (real)2*g/(g - (real)1))/g;
  const real J = (real)1/a + (real)1/((real)3*a*a*a) + (real)1/((real)5*a*a*a*a*a)
               - (real)0.5*log(((real)1 + a)/fmax((real)1 - a, (real)1e-12));
  x = ((real)1/((real)2*rho))*((real)2/(k*k) - (real)1/(q*q)) - (real)0.5*J;
  const real s = (real)1 - (q*q)/(k*k);
  y = (real)1/(k*rho*q)*sqrt(fmax(s, (real)0));
}

// Invert (x,y) -> (q,k).  VERIFIED in isolation before use (round trip 1e-15 over
// the duct, |div(rho V)| ~ 2e-6 at the finite-difference floor).
//
// Two things make this robust where the previous attempt was not:
//  * It bisects on the CIRCLE IDENTITY  (x + L/2)^2 + y^2 = 1/(2 rho q^2)^2,
//    which involves q ALONE.  The old version solved for k from y first, and
//    that quadratic has two roots and goes complex (disc < 0) near the q = qmin
//    arc -- so the inversion "failed" exactly at the inflow and every such cell
//    was silently classified as a WALL.
//  * It restricts q to the SUBSONIC branch, q < q_sonic = sqrt(2/(g+1)).
//    Ringleb is genuinely double-valued -- subsonic and supersonic states share
//    (x,y) -- so without this the bisection returns whichever root the bracket
//    happens to catch, which poisoned the initial condition with near-vacuum
//    cells (rho ~ 0.05 against a physical floor of 0.55).
__host__ __device__ bool CompressibleSolver::ringlebInvert(real x, real y,
                        real &q, real &k) const {
  const real qSonic = sqrt((real)2/(gam + (real)1));
  auto f = [&](real qq) {
    const real b = sqrt(fmax((real)1 - (real)0.5*(gam-(real)1)*qq*qq, (real)1e-12));
    const real r = pow(b, (real)2/(gam-(real)1));
    const real L = (real)1/b + (real)1/((real)3*b*b*b) + (real)1/((real)5*b*b*b*b*b)
                 - (real)0.5*log(((real)1+b)/fmax((real)1-b,(real)1e-12));
    const real xc = x + (real)0.5*L;
    const real rad = (real)1/((real)2*r*qq*qq);
    return xc*xc + y*y - rad*rad;
  };
  real a = (real)0.05, bq = qSonic - (real)1e-6;
  real fa = f(a), fb = f(bq);
  if (fa*fb > (real)0) return false;
  for (i32 it = 0; it < 100; it++) {
    const real m = (real)0.5*(a + bq), fm = f(m);
    if (fa*fm <= (real)0) { bq = m; fb = fm; } else { a = m; fa = fm; }
  }
  q = (real)0.5*(a + bq);
  const real b = sqrt(fmax((real)1 - (real)0.5*(gam-(real)1)*q*q, (real)1e-12));
  const real r = pow(b, (real)2/(gam-(real)1));
  const real L = (real)1/b + (real)1/((real)3*b*b*b) + (real)1/((real)5*b*b*b*b*b)
               - (real)0.5*log(((real)1+b)/fmax((real)1-b,(real)1e-12));
  // 2/k^2 = 1/q^2 + 2 rho (x + L/2)   -- sign follows the -L/2 forward map
  const real sK = (real)1/(q*q) + (real)2*r*(x + (real)0.5*L);
  if (!(sK > (real)0)) return false;
  k = sqrt((real)2/sK);
  return true;
}

__host__ __device__ void CompressibleSolver::ringlebExact(real x, real y, real &rho,
                        real &u, real &v, real &p) const {
  // solver coordinates -> Ringleb coordinates (the map spans negative x, and the
  // physical region is centred in the box so the domain BCs stay shielded)
  const real X = x + ringlebX0, Y = y + ringlebY0;
  real q, k;
  if (!ringlebInvert(X, fabs(Y), q, k)) { rho = 1; u = 0; v = 0; p = (real)1/gam; return; }
  real xx, yy;
  ringlebHodograph(q, k, xx, yy, rho, p);
  // Velocity direction is tangent to the streamline: sin(theta) = q/k, and the
  // flow turns back on itself past the apex, so the x-sign follows dx/dq.
  // Direction is the streamline tangent.  With sin(theta) = q/k the unit tangent
  // is (cos theta, -sin theta) on the upper branch -- VERIFIED against a finite
  // difference of the hodograph parametrisation d(x,y)/dq, which is the only
  // way I trust the assignment: swapping u and v here still gives |V| = q, so
  // the mistake is invisible in every magnitude check and only shows up as a
  // non-converging L2 error against a solution that is not actually steady.
  // Direction = the streamline tangent, taken as a FINITE DIFFERENCE of the
  // forward map at fixed k.  Correct by construction: every closed-form angle
  // relation I tried (sin th = q/k either way round) disagreed with the actual
  // tangent, and the error is invisible in any magnitude check because |V| = q
  // regardless -- it only shows up as a non-zero div(rho V).
  const real hq = (real)1e-6;
  real x1, y1, r1, p1, x2, y2, r2, p2;
  ringlebHodograph(fmax(q - hq, (real)1e-4), k, x1, y1, r1, p1);
  ringlebHodograph(q + hq,                   k, x2, y2, r2, p2);
  real tx = x2 - x1, ty = y2 - y1;
  const real tn = sqrt(fmax(tx*tx + ty*ty, (real)1e-30));
  tx /= tn; ty /= tn;
  if (Y < (real)0) ty = -ty;                // lower half is the mirror
  u = q*tx;  v = q*ty;
}

__host__ __device__ void CompressibleSolver::svortexExact(real x, real y, real &rho,
                                                          real &u, real &v, real &p) {
  const real dx = x - ibCenter[0], dy = y - ibCenter[1];
  // CLAMP to the annulus.  The IC is evaluated in the SOLID as well, and there
  // |u| = M r_i / r diverges as r -> 0 at the centre of the inner body (measured
  // before this: max|u| = 98 against an exact 2.25, and negative pressure).
  // Clamping holds the solid at the nearest wall state, which is benign and
  // leaves the fluid region untouched.
  const real rRaw = sqrt(dx*dx + dy*dy);
  const real r  = fmin(fmax(rRaw, ibRadius), ibRadius2);
  const real ri = ibRadius, M = svMach;
  const real t  = (real)1 + (gam-(real)1)/(real)2*M*M*((real)1 - ri*ri/(r*r));
  rho = pow(fmax(t,(real)1e-12), (real)1/(gam-(real)1));
  p   = pow(rho, gam)/gam;
  const real ut = M*ri/r;                     // counter-clockwise
  u = -ut*dy/r;  v = ut*dx/r;
}

__host__ __device__ bool CompressibleSolver::exactState(real x, real y, real &rho,
                                                        real &u, real &v, real &p) {
  if (icType == 13) { svortexExact(x, y, rho, u, v, p); return true; }
  if (icType == 14) { ringlebExact(x, y, rho, u, v, p); return true; }
  return false;
}

__host__ __device__ real CompressibleSolver::getBoundaryLevelSet(Vec3 pos) {
  if (immerserdBcType == 1) {          // sphere
    const real dx = pos[0]-ibCenter[0], dy = pos[1]-ibCenter[1], dz = pos[2]-ibCenter[2];
    return ibRadius - sqrt(dx*dx + dy*dy + dz*dz);
  }
  if (immerserdBcType == 2) {          // half-space y < ibPlane (a flat wall)
    return ibPlane - pos[1];
  }
  if (immerserdBcType == 5) {
    // Inclined flat plate: the half-plane below a line through (plateX0, ibPlane)
    // at ibAngle to the x axis, starting at that point.  This is the paper's
    // Sec. 4.2 inclined-grid case, and the first geometry that actually
    // exercises the general normal -- for a horizontal plane the IB flux reduces
    // algebraically to the grid-aligned one, so nothing about the projection is
    // tested there.  Signed distance, positive inside.
    const real a  = ibAngle*(real)(3.14159265358979323846/180.0);
    const real ct = cos(a), st = sin(a);
    const real rx = pos[0] - plateX0, ry = pos[1] - ibPlane;
    const real sAlong = rx*ct + ry*st;        // along the plate
    const real qNorm  = -rx*st + ry*ct;       // normal to it, > 0 on the fluid side
    // Infinite plane (--ibinf 1): no tip.  A plate TIP inside the domain lets
    // fluid jet through the wedge between the domain floor and the tip, and
    // the shear layer that jet sheds is a real K-H-type limit cycle that SA
    // faithfully amplifies (measured: nu~ bursts born ~26 cells ABOVE the LE,
    // convecting down the plate, C_f at 0.97 oscillating 0.001-0.003 forever).
    // TMR's FPTBL -- and the paper's inclined case -- have NO tip: the wall
    // plane spans the domain, with a symmetry plane upstream of the LE.  The
    // slip/modelled split is wmX0's job, not the geometry's.
    if (ibInfinite) return -qNorm;
    if (sAlong >= 0) return -qNorm;           // over the plate
    if (qNorm <= 0) return -(-sAlong);        // level with it, ahead of the edge
    return -sqrt(sAlong*sAlong + qNorm*qNorm);  // round the leading edge
  }
  if (immerserdBcType == 4) {
    // Half-plane plate: solid where y < ibPlane AND x > plateX0.  Same geometry
    // as the grid-aligned wallGeom 1 case, so the IB path can be run against the
    // already-validated grid-aligned result -- same physics, two independent
    // implementations.  Signed distance, positive inside.
    const real dxle = plateX0 - pos[0];      // > 0 upstream of the leading edge
    const real dyp  = ibPlane  - pos[1];      // > 0 below the plate surface
    if (dxle <= 0) return dyp;                // over the plate: + inside, - above
    // Upstream of the leading edge the point is OUTSIDE the body, so the level
    // set is negative there whichever side of the plate's height it is on: level
    // with the plate the nearest solid point is the edge, above it the corner.
    if (dyp >= 0) return -dxle;
    return -sqrt(dxle*dxle + dyp*dyp);
  }
  if (immerserdBcType == 6 && ibPolyN > 2) {   // closed polyline (airfoil)
    const real px = pos[0], py = pos[1];
    // Cheap reject: outside the bbox by more than the margin, return the (safe,
    // negative = outside) box distance.  isFluidCell only cares about |phi| < h,
    // and the wall model never queries far cells, so this costs no accuracy
    // where it matters and skips the O(N) loop for the bulk of the domain.
    const real ex = fmax(fmax(ibBox[0]-px, px-ibBox[2]), (real)0);
    const real ey = fmax(fmax(ibBox[1]-py, py-ibBox[3]), (real)0);
    if (ex*ex + ey*ey > ibChord*ibChord*(real)0.04) {    // > 0.2c outside the box
      // NOTE the sign must follow the SAME convention as the exact branch below.
      // This early-out assumed "polyline bounds a solid body", so far == outside
      // == fluid.  A loop bounding the FLUID (a duct) inverts that, and missing
      // it here labelled two slabs of far-field as fluid while the exact branch
      // labelled them solid -- the flip has to be applied at BOTH exits.
      const real dbox = -sqrt(ex*ex + ey*ey);
      return ibPolyFluidInside ? -dbox : dbox;
    }
    real d2min = (real)1e30;
    i32  wind  = 0;
    i32  eMin  = 0;                       // nearest segment (carries the BC tag)
    for (i32 e = 0; e < ibPolyN; e++) {
      const i32 f = (e + 1 == ibPolyN) ? 0 : e + 1;
      const real ax = ibPoly[2*e], ay = ibPoly[2*e+1];
      const real bx = ibPoly[2*f], by = ibPoly[2*f+1];
      // nearest point on segment ab
      const real vx = bx-ax, vy = by-ay, wx = px-ax, wy = py-ay;
      const real vv = vx*vx + vy*vy;
      real t = (vv > (real)0) ? (wx*vx + wy*vy)/vv : (real)0;
      t = fmin(fmax(t, (real)0), (real)1);
      const real qx = px - (ax + t*vx), qy = py - (ay + t*vy);
      const real d2 = qx*qx + qy*qy;
      if (d2 < d2min) { d2min = d2; eMin = e; }
      // winding number (crossing rule) -- sign is independent of the distance
      if ((ay > py) != (by > py)) {
        const real xint = ax + (py - ay)/(by - ay)*(bx - ax);
        if (px < xint) wind++;
      }
    }
    const real dist = sqrt(d2min);
    const real sgn = (wind & 1) ? dist : -dist;   // POSITIVE INSIDE the loop
    // Convention is POSITIVE = SOLID.  A loop bounding the fluid flips it.
    return ibPolyFluidInside ? -sgn : sgn;
  }
  if (immerserdBcType == 8) {
    // Planar SLAB: fluid for ibPlane < y < ibRadius2, solid outside.  Used with
    // periodic x AND y (bcType 2) for the transient-channel gate: the periodic y
    // wrap joins solid to solid, so there is no outer boundary condition at all
    // and no inflow/outflow or blockage to contaminate the exact solution.
    return fmax(ibPlane - pos[1], pos[1] - ibRadius2);
  }
  if (immerserdBcType == 9) {
    // Ringleb: the walls ARE streamlines k = ringlebKmin / kMax, so the body is
    // "outside the band of streamlines we keep".  k is only defined inside the
    // hodograph map; points off it are solid.  phi is k-distance scaled by the
    // local streamline spacing so it behaves like a distance near the wall --
    // the IB only needs the sign and a locally-consistent magnitude.
    const real X = pos[0] + ringlebX0, Y = pos[1] + ringlebY0;
    real q, k;
    if (!ringlebInvert(X, fabs(Y), q, k)) return (real)1;   // off the map = solid
    const real w = (real)0.5*(ringlebKmax - ringlebKmin);
    const real c = (real)0.5*(ringlebKmax + ringlebKmin);
    // The level set defines the FLUID PARTITION only: fluid is the streamline
    // band AND q >= qmin.  It does NOT say what kind of boundary each side is --
    // that is getBoundaryBcKind, which marks the q = qmin arc as PRESCRIBED
    // (flow enters and leaves there) and the two streamlines as WALLS.  Both are
    // then imposed through the same ghost cells.
    const real dk = fabs(k - c) - w;          // > 0 outside the streamline band
    const real dq = ringlebQmin - q;          // > 0 inside the q = qmin arc
    return fmax(dk, dq)*(ringlebScale > 0 ? ringlebScale : (real)1);
  }
  if (immerserdBcType == 7) {          // concentric annulus (supersonic vortex)
    // Solid for r < ibRadius OR r > ibRadius2.  max() of the two half-space
    // distances is the EXACT signed distance for this geometry, positive inside
    // the solid, matching the sign convention of every other body here.
    const real dx = pos[0]-ibCenter[0], dy = pos[1]-ibCenter[1];
    const real r = sqrt(dx*dx + dy*dy);
    return fmax(ibRadius - r, r - ibRadius2);
  }
  if (immerserdBcType == 3) {          // cylinder about the z axis
    const real dx = pos[0]-ibCenter[0], dy = pos[1]-ibCenter[1];
    return ibRadius - sqrt(dx*dx + dy*dy);
  }
  if (immerserdBcType == 10) {         // canal: floor U bump disc U ceiling
    // Union of three solids -> max of the three signed distances, which is the
    // EXACT distance on the fluid side (the only side the cut apertures and the
    // R-Cell foot points read).  The disc meets the floor at the bump's two
    // corners, where both terms vanish together.
    const real dx = pos[0]-ibCenter[0], dy = pos[1]-ibCenter[1];
    const real bump = ibRadius - sqrt(dx*dx + dy*dy);
    return fmax(fmax(canalY0 - pos[1], bump), pos[1] - canalY1);
  }
  else {
    // No body.  NEGATIVE means "infinitely far outside", matching the sphere
    // branch's positive-inside convention -- returning +1e32 here would make
    // wallDistance's wallGeom == 2 branch report a wall distance of ZERO
    // everywhere (it negates this), i.e. a wall through the whole domain.
    return -1e32;
  }
}

// Unit normal pointing OUT of the body into the fluid.  phi is positive inside,
// so that direction is -grad(phi); for a signed distance |grad phi| = 1, but the
// explicit normalisation keeps this honest for level sets that are only
// approximately distances.

__host__ __device__ Vec3 CompressibleSolver::wallNormal(Vec3 pos, real h) {
  // EXACT closest-point normal (user's call): the direction from the closest
  // point on the surface to the query IS the normal, computed from the
  // geometry itself rather than by finite-differencing phi.  This beats both
  // earlier attempts at once: the raw facet normal jittered at sub-cell scale
  // wherever the polyline's vertices are denser than the grid (the
  // cosine-clustered nose), and the 0.5h-mollified difference smeared the
  // normal ACROSS the body wherever it is thinner than the step (the aft 10%).
  // The closest-point direction is continuous as the query moves -- it slides
  // through vertex fans -- and is exact on both sides of a thin body right up
  // to the medial axis.
  if (immerserdBcType == 1) {          // sphere: radial
    real dx=pos[0]-ibCenter[0], dy=pos[1]-ibCenter[1], dz=pos[2]-ibCenter[2];
    const real m = sqrt(dx*dx+dy*dy+dz*dz);
    if (m > (real)1e-30) return Vec3(dx/m, dy/m, dz/m);
  }
  if (immerserdBcType == 2) return Vec3(0, 1, 0);   // half-space y < ibPlane
  if (immerserdBcType == 3) {          // cylinder: radial in x-y
    real dx=pos[0]-ibCenter[0], dy=pos[1]-ibCenter[1];
    const real m = sqrt(dx*dx+dy*dy);
    if (m > (real)1e-30) return Vec3(dx/m, dy/m, 0);
  }
  if (immerserdBcType == 7) {          // annulus: radial, sign by nearer wall
    // ANALYTIC normal.  Without this branch the annulus fell through to the
    // mollified finite-difference fallback below -- an O(h) normal at the one
    // geometry used for every order study, feeding an O(h)|u| error straight
    // into the tangential projection of every wall treatment.  Body -> fluid:
    // +r_hat off the inner cylinder, -r_hat off the outer shell, switching at
    // the midradius exactly where the fmax() in the level set switches.
    real dx=pos[0]-ibCenter[0], dy=pos[1]-ibCenter[1];
    const real m = sqrt(dx*dx+dy*dy);
    if (m > (real)1e-30) {
      const real sg = (ibRadius - m >= m - ibRadius2) ? (real)1 : (real)-1;
      return Vec3(sg*dx/m, sg*dy/m, 0);
    }
  }
  if (immerserdBcType == 10) {         // canal: nearest of floor / bump / ceiling
    const real dx=pos[0]-ibCenter[0], dy=pos[1]-ibCenter[1];
    const real m = sqrt(dx*dx+dy*dy);
    const real fl = canalY0 - pos[1], bp = ibRadius - m, ce = pos[1] - canalY1;
    if (bp >= fl && bp >= ce && m > (real)1e-30) return Vec3(dx/m, dy/m, 0);
    return (fl >= ce) ? Vec3(0, 1, 0) : Vec3(0, -1, 0);
  }
  if (immerserdBcType == 4 || immerserdBcType == 5) {
    // quarter-plane, possibly rotated: faces at sAlong = 0 (front) and
    // qNorm = 0 (top), corner at the leading edge.
    const real a  = (immerserdBcType == 5) ? ibAngle*(real)(3.14159265358979323846/180.0)
                                           : (real)0;
    const real ct = cos(a), st = sin(a);
    const real rx = pos[0] - plateX0, ry = pos[1] - ibPlane;
    const real sA = rx*ct + ry*st;          // along the plate
    const real qN = -rx*st + ry*ct;         // normal to it, > 0 fluid side
    real ns, nq;                            // normal in (sAlong, qNorm) frame
    if (ibInfinite) { ns = 0; nq = 1;
      return Vec3(ns*ct - nq*st, ns*st + nq*ct, (real)0); }
    if (sA >= (real)0) {
      if (qN >= (real)0) { ns = 0; nq = 1; }                 // above the plate
      else {                                                 // inside: nearer face
        if (sA < -qN) { ns = -1; nq = 0; } else { ns = 0; nq = 1; }
      }
    } else {
      if (qN <= (real)0) { ns = -1; nq = 0; }                // ahead of the front face
      else {                                                 // corner fan
        const real m = sqrt(sA*sA + qN*qN);
        // same cancellation guard: at the corner itself use the fan bisector
        if (m > (real)0.05*h) { ns = sA/m; nq = qN/m; }
        else { const real r2 = (real)0.7071067811865475; ns = -r2; nq = r2; }
      }
    }
    return Vec3(ns*ct - nq*st, ns*st + nq*ct, 0);
  }
  if (immerserdBcType == 6 && ibPolyN > 2) {
    // exact closest point on the polyline + winding for the sign
    const real px = pos[0], py = pos[1];
    real d2min = (real)1e30, fx = px, fy = py, tBest = 0;
    i32 wind = 0, eBest = 0;
    for (i32 e = 0; e < ibPolyN; e++) {
      const i32 f2 = (e + 1 == ibPolyN) ? 0 : e + 1;
      const real ax = ibPoly[2*e], ay = ibPoly[2*e+1];
      const real bx = ibPoly[2*f2], by = ibPoly[2*f2+1];
      const real vx = bx-ax, vy = by-ay, wx = px-ax, wy = py-ay;
      const real vv = vx*vx + vy*vy;
      real t = (vv > (real)0) ? (wx*vx + wy*vy)/vv : (real)0;
      t = fmin(fmax(t, (real)0), (real)1);
      const real cxq = ax + t*vx, cyq = ay + t*vy;
      const real qx = px - cxq, qy = py - cyq;
      const real d2 = qx*qx + qy*qy;
      if (d2 < d2min) { d2min = d2; fx = cxq; fy = cyq; eBest = e; tBest = t; }
      if ((ay > py) != (by > py)) {
        const real xint = ax + (py - ay)/(by - ay)*(bx - ax);
        if (px < xint) wind++;
      }
    }
    const real dist = sqrt(d2min);
    // The point-difference direction (p - f)/dist is computed at domain-scale
    // coordinates (O(10) here), so in the float build the subtraction cancels
    // for dist below ~1e-5 and the "normal" is rounding junk -- measured as
    // near-surface cells whose normal just pointed vertically.  Switch to the
    // exact FEATURE normal (segment perpendicular / vertex bisector, no
    // cancellation) for anything closer than 5% of a cell; the two agree to
    // O(dist * curvature) there anyway.
    if (dist > (real)0.05*h) {
      const real sgn = (wind & 1) ? (real)-1 : (real)1;   // inside: flip to outward
      return Vec3(sgn*(px-fx)/dist, sgn*(py-fy)/dist, 0);
    }
    // NEAR the surface the closest-point direction is undefined.  If the closest
    // feature is a VERTEX (t clamped to an end), one segment's perpendicular is
    // wrong by half the fan angle -- at the cosine-clustered nose that is ~20
    // degrees (measured: the single bad normal of 726, dot 0.930, at the LE
    // apex).  Use the fan BISECTOR: the normalised sum of the two adjacent
    // segments' outward perpendiculars.  Interior of a segment: its own perp.
    {
      const real eps = (real)1e-6;
      i32 eA = eBest, eB = eBest;                     // segments to average
      if      (tBest >= (real)1 - eps) eB = (eBest + 1 == ibPolyN) ? 0 : eBest + 1;
      else if (tBest <= eps)           eA = (eBest == 0) ? ibPolyN - 1 : eBest - 1;
      real sx = 0, sy = 0;
      for (i32 pass = 0; pass < 2; pass++) {
        const i32 e = pass ? eB : eA;
        const i32 f2 = (e + 1 == ibPolyN) ? 0 : e + 1;
        const real vx = ibPoly[2*f2] - ibPoly[2*e], vy = ibPoly[2*f2+1] - ibPoly[2*e+1];
        const real m = sqrt(vx*vx + vy*vy);
        if (m > (real)1e-30) { sx += vy/m; sy += -vx/m; }   // CCW outward perp
        if (eA == eB) break;
      }
      const real m = sqrt(sx*sx + sy*sy);
      if (m > (real)1e-30) return Vec3(sx/m, sy/m, 0);
    }
  }
  // fallback: mollified finite difference of phi
  const real e = (real)0.5*h;
  const real gx = getBoundaryLevelSet(Vec3(pos[0]+e, pos[1], pos[2]))
                - getBoundaryLevelSet(Vec3(pos[0]-e, pos[1], pos[2]));
  const real gy = getBoundaryLevelSet(Vec3(pos[0], pos[1]+e, pos[2]))
                - getBoundaryLevelSet(Vec3(pos[0], pos[1]-e, pos[2]));
  const real gz = pseudo2D ? (real)0
                : getBoundaryLevelSet(Vec3(pos[0], pos[1], pos[2]+e))
                - getBoundaryLevelSet(Vec3(pos[0], pos[1], pos[2]-e));
  const real m = sqrt(gx*gx + gy*gy + gz*gz);
  if (m <= (real)0) return Vec3(0, 1, 0);
  return Vec3(-gx/m, -gy/m, -gz/m);
}

// A cell counts as FLUID only if it lies entirely outside the body: following
// UTCart, every cell the surface intersects is treated as non-fluid and carries
// no solution.  For a signed distance that is phi(centre) < -(half diagonal).
// ---- Brinkman volume penalization: porosity helpers ------------------------
// Restored 2026-09-02 (brinkman branch) from the pre-cleanup solver, verbatim:
// these are the numerically delicate part of the method and the fp32-safe
// forms below were arrived at by measurement, not preference.
// Volume fraction phi = V_fluid/V_total for the pressure-tight penalization,
// Reiss 2021 Eq. (28):  phi = eps + (1-eps)*(1 + tanh(s/delta))/2, with s the
// signed distance (positive in the fluid) and delta = ibBrinkDelta * h.
//
// The tanh is not cosmetic and a smoothstep is NOT a substitute.  The scheme
// divides the RHS by phi, so what has to stay bounded is the LOGARITHMIC slope
// d(ln phi)/ds, which the tanh caps at 2/delta -- giving a phi ratio of only
// ~2 between neighbouring cells no matter how small eps is.  A polynomial
// blend lands on the eps plateau abruptly and the same ratio reaches ~1e4.
// That bounded log-slope is the "non-stiff" of the paper's title: at
// delta = 1.5h the measured stiffness rises just 25% (paper Fig. 5, Sec 4.1.3).
__host__ __device__ real CompressibleSolver::brinkPhi(real s, real h) {
  const real x = s/fmax(ibBrinkDelta*h, (real)1e-30);
  // (1 + tanh x)/2 evaluated as the logistic sigmoid 1/(1 + exp(-2x)).  These
  // are identical in exact arithmetic, but the literal tanh form is unusable
  // here: deep in the body tanh(x) -> -1 and "1 + tanh(x)" cancels away every
  // significant digit, so phi degenerates to round-off noise below ~1e-7 in
  // fp32 -- exactly the range this method lives in.  The sigmoid underflows
  // gracefully to 0 instead and keeps full RELATIVE accuracy at every depth,
  // which is what the 1/phi division needs.
  const real g = (x > (real)0) ? (real)1/((real)1 + exp((real)-2*x))
                               : exp((real)2*x)/((real)1 + exp((real)2*x));
  return ibBrinkEps + ((real)1 - ibBrinkEps)*g;
}

__host__ __device__ real CompressibleSolver::brinkPhiFaceAvgSeg(Vec3 p0, Vec3 p1,
                                                                  real h, i32 nseg) {
  const real d = fmax(ibBrinkDelta*h, (real)1e-30);
  if (nseg < 1) nseg = 1;
  #define SP_(X) (((X) > (real)0) ? ((X) + log1p(exp(-(X)))) : log1p(exp(X)))
  real s0 = -getBoundaryLevelSet(p0);
  real tot = (real)0;
  for (i32 i = 1; i <= nseg; i++) {
    const real t = (real)i/(real)nseg;
    const real s1 = -getBoundaryLevelSet(Vec3(p0[0] + (p1[0]-p0[0])*t,
                                              p0[1] + (p1[1]-p0[1])*t,
                                              p0[2] + (p1[2]-p0[2])*t));
    const real a = (real)2*s0/d, b = (real)2*s1/d, ds = s1 - s0;
    if (fabs(ds) > (real)1e-9*d) tot += (d/(real)2)*(SP_(b) - SP_(a))/ds;
    else { const real e = exp(-fabs(a));
           tot += (a >= (real)0) ? (real)1/((real)1+e) : e/((real)1+e); }
    s0 = s1;
  }
  #undef SP_
  return ibBrinkEps + ((real)1 - ibBrinkEps)*tot/(real)nseg;
}

__host__ __device__ bool CompressibleSolver::isFluidCell(Vec3 pos, real h) {
  if (immerserdBcType == 0) return true;
  // Volume penalization does NOT mask the body: the equations are solved
  // everywhere and the wall appears through the porosity weights and the
  // p grad(phi) source, so every cell is fluid.  This one line switches off the
  // sharp-IB machinery wholesale -- no wall faces, no ghost fill, no update or
  // dt masking.
  if (ibBrink) return true;
  // RCCM keeps every cell the body only PARTIALLY covers -- that is the whole
  // point of a cut-cell discretisation, and it is the exact opposite of the
  // UTCart rule below ("any intersecting cell is non-fluid").  A cell is live
  // when any corner is in the fluid; the small ones among them become R-Cells
  // and are reconstructed rather than advanced.
  if (ibRccm) {
    const real hh = (real)0.5*h;
    for (i32 a = -1; a <= 1; a += 2)
      for (i32 b = -1; b <= 1; b += 2)
        if (getBoundaryLevelSet(Vec3(pos[0]+a*hh, pos[1]+b*hh, pos[2])) < (real)0)
          return true;
    return false;
  }
  // UTCart's rule is "every INTERSECTING cell is non-fluid", and a cell is
  // intersected exactly when the level set changes sign across its CORNERS.
  // The old test compared phi at the centre against the half-DIAGONAL, i.e. the
  // circumscribed sphere -- the correct conservative bound for an arbitrary
  // normal, but for a grid-aligned surface it over-masks by (0.7071-0.5)h:
  // it throws away a fifth of a cell of genuine fluid and lifts d_FC from
  // (0, 1]h to (0.207, 1.207]h, pushing the worst case further into the range
  // where the coupling is least stable.  The corner test is exact for any
  // orientation and costs 4 level-set evaluations in 2-D (8 in 3-D), only in
  // the narrow band where it is ever ambiguous.
  const real e = (real)0.5*h;
  const i32 nk = pseudo2D ? 1 : 2;
  for (i32 a = 0; a < 2; a++)
    for (i32 b = 0; b < 2; b++)
      for (i32 c = 0; c < nk; c++) {
        const Vec3 v(pos[0] + (a ? e : -e),
                     pos[1] + (b ? e : -e),
                     pos[2] + (pseudo2D ? (real)0 : (c ? e : -e)));
        if (getBoundaryLevelSet(v) >= (real)0) return false;   // a corner is inside
      }
  return true;
}

__device__ Vec5 CompressibleSolver::hllcFlux(Vec5 qL, Vec5 qR, Vec3 normal) {
  //
  // Compute the 3D HLLC flux through a face with unit normal (nx,ny,nz).
  // qL, qR are conservative states (rho, rhou, rhov, rhow, rhoE).
  //
  real nx = normal[0];
  real ny = normal[1];
  real nz = normal[2];

  // Left state
  real rL  = qL[0];
  real uL  = qL[1]/rL;
  real vL  = qL[2]/rL;
  real wL  = qL[3]/rL;
  real vnL = uL*nx + vL*ny + wL*nz;
  real eL  = qL[4];
  real pL  = (gam-1.0)*(eL - 0.5*rL*(uL*uL + vL*vL + wL*wL));
  real hL  = (eL + pL)/rL;
  real aL  = sqrt(abs(gam*pL/rL));

  // Right state
  real rR  = qR[0];
  real uR  = qR[1]/rR;
  real vR  = qR[2]/rR;
  real wR  = qR[3]/rR;
  real vnR = uR*nx + vR*ny + wR*nz;
  real eR  = qR[4];
  real pR  = (gam-1.0)*(eR - 0.5*rR*(uR*uR + vR*vR + wR*wR));
  real hR  = (eR + pR)/rR;
  real aR  = sqrt(abs(gam*pR/rR));

  // Roe averages for the wave-speed estimate
  real sqrL = sqrt(rL);
  real sqrR = sqrt(rR);
  real rSum = sqrL + sqrR;
  real u = (uL*sqrL + uR*sqrR) / rSum;
  real v = (vL*sqrL + vR*sqrR) / rSum;
  real w = (wL*sqrL + wR*sqrR) / rSum;
  real a2 = (aL*aL*sqrL + aR*aR*sqrR) / rSum
          + 0.5*sqrL*sqrR/(rSum*rSum)*(vnR - vnL)*(vnR - vnL);
  real a = sqrt(a2);
  real vn = u*nx + v*ny + w*nz;

  // Wave speed estimates.  Under low-Mach preconditioning the ACOUSTIC
  // dissipation must be rescaled to match the preconditioned eigenvalues: the
  // time derivative and the upwind dissipation have to see the same wave
  // speeds, or the scheme is inconsistent and stalls in a limit cycle rather
  // than converging (measured: Cf oscillating 0.0025-0.0041 for 74k iters).
  // Turkel's rescaling applied to the normal velocity: u' = (1+b2)u/2,
  // c' = sqrt(((1-b2)u/2)^2 + b2 c^2).  b2 = 1 restores the exact HLLC.
  real SL, SR;
  if (precond) {
    const real q2m = u*u + v*v + w*w;
    const real b2  = fmin(fmax(q2m/fmax(a2,(real)1e-30),
                               precondK*precondMref2), (real)1);
    const real hlf = (real)0.5*((real)1 - b2);
    const real upL = (real)0.5*((real)1 + b2)*vnL;
    const real upR = (real)0.5*((real)1 + b2)*vnR;
    const real upM = (real)0.5*((real)1 + b2)*vn;
    const real cpL = sqrt(hlf*hlf*vnL*vnL + b2*aL*aL);
    const real cpR = sqrt(hlf*hlf*vnR*vnR + b2*aR*aR);
    const real cpM = sqrt(hlf*hlf*vn*vn   + b2*a2);
    SL = fmin(upL - cpL, upM - cpM);
    SR = fmax(upR + cpR, upM + cpM);
  } else {
  SL = fmin(vnL - aL, vn - a);
  SR = fmax(vnR + aR, vn + a);
  }
  real SM = (pL - pR + rR*vnR*(SR-vnR) - rL*vnL*(SL-vnL))
          / (rR*(SR-vnR) - rL*(SL-vnL));

  // Left and Right physical fluxes
  real FL[5] = {rL*vnL, rL*vnL*uL + pL*nx, rL*vnL*vL + pL*ny, rL*vnL*wL + pL*nz, rL*vnL*hL};
  real FR[5] = {rR*vnR, rR*vnR*uR + pR*nx, rR*vnR*vR + pR*ny, rR*vnR*wR + pR*nz, rR*vnR*hR};

  // Star states (Toro)
  real fL = (SL - vnL)/(SL - SM);
  real fR = (SR - vnR)/(SR - SM);
  Vec5 qLS(fL*rL,
           fL*rL*(uL + (SM - vnL)*nx),
           fL*rL*(vL + (SM - vnL)*ny),
           fL*rL*(wL + (SM - vnL)*nz),
           fL*(eL + (SM - vnL)*(rL*SM + pL/(SL - vnL))));
  Vec5 qRS(fR*rR,
           fR*rR*(uR + (SM - vnR)*nx),
           fR*rR*(vR + (SM - vnR)*ny),
           fR*rR*(wR + (SM - vnR)*nz),
           fR*(eR + (SM - vnR)*(rR*SM + pR/(SR - vnR))));

  Vec5 Flux;
  for (i32 n = 0; n < 5; n++) {
    Flux[n] = 0.5*(FL[n] + FR[n])
            - 0.5*(abs(SL)*(qLS[n]-qL[n]) + abs(SM)*(qRS[n]-qLS[n]) + abs(SR)*(qR[n]-qRS[n]));
  }
  return Flux;
}

__device__ real CompressibleSolver::calcIbMask(real phi) {
  real dx = min(getDx(nLvls-1), min(getDy(nLvls-1), getDz(nLvls-1)));
  real eps = .5;
  return (.5 * (1 + tanh(phi / (2 * eps * dx))));
}
