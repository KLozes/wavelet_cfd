#include <iostream>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <thrust/extrema.h>
#include <unordered_set>
#include <unordered_map>

#include "CompressibleSolver.cuh"
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
}


real CompressibleSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;
  Timer<std::chrono::microseconds, std::chrono::steady_clock> sub;   // profiling sub-timer

  while (t < tStep) {

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
      restrictFields();
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tick(); }
      forwardWaveletTransform();
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tock(); tForwardUs += sub.duration().count(); }
      // refinement cascade; under MGPU this exchanges block activity across rank
      // seams after every create kernel so grading/support close consistently
      adaptGridConsistent();
#ifdef USE_MGPU
      // the inverse reconstructs each new fine block from its (coarse) parent's
      // F_OLD; rebuild that snapshot for every block created this cycle (owned
      // fills + halo to ghosts, coarse->fine) before the inverse reads it
      reconstituteOldSnapshot();
#endif
      setBoundaryConditions(F_OLD);
      inverseWaveletTransform();
      if (dbgChecks) { cudaDeviceSynchronize(); sub.tick(); }
      sortBlocks();
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

    if (mdFlux == 2) {
      // CTU-Hancock: fully-discrete predictor-corrector.  The corrector is
      // FUSED into multiDRhsKernel (it updates q in place, conservative), so
      // there is no primitiveToConservative/updateFields here; the shared
      // bank holds the half-step predicted primitives during the RHS.
      conservativeToPrimitive();
      setBoundaryConditions(0, 1);
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
      if (immerserdBcType && !rans) applyWallGhosts();   // Euler: slip ghosts
      if (rans) {
        if (wallGeom || immerserdBcType) applyWallGhosts();
        computeTurbClosure();
#ifdef USE_MGPU
        haloExchange(F_MUT, 2);     // ghosts need mu_t and F1 on both sides of a seam face
#endif
      }
      computeRightHandSide();
      primitiveToConservative();
      updateFields(stage);
      setBoundaryConditions();

      if (nLvls > 1) {
        restrictFields();
#ifdef USE_MGPU
        haloExchange(0, NEVOLVE);   // refresh ghosts after the coarse/fine reconstruction
#endif
        interpolateFields();
        setBoundaryConditions();
        // The coarse/fine reconstruction has just overwritten fine halo cells
        // from their PARENTS, and a parent that overlaps the body holds immersed
        // GHOST data (a mirror built at the COARSE cell size).  Interpolating
        // that into fine cells puts a wrong near-wall state on the fine level,
        // and the error grows with every extra level -- which is why nLvls 6 was
        // stable and nLvls 7 was not.  Re-impose the immersed ghosts here, at
        // the level they belong to, before the state is used again.
        if (immerserdBcType) applyWallGhosts();
      }
    }
    cudaDeviceSynchronize();
    clock.tock();
    tSolver += clock.duration().count();

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
  // The geometry cache is block payload: carry it through the sort with the
  // flow variables (staged through F_SCRATCH -- the F_OLD bank has exactly
  // NEVOLVE slots).  Same-stream launches, so no explicit sync is needed
  // between each snapshot and its gather.
  if (immerserdBcType != 0) {
    copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_PHI, F_SCRATCH);
    gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, F_PHI);
    copyFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_IBM, F_SCRATCH);
    gatherSortedFieldKernel<<<cudaGridSize, cudaBlockSize>>>(*this, F_SCRATCH, F_IBM);
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

void CompressibleSolver::conservativeToPrimitive(void) {
  conservativeToPrimitiveKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::primitiveToConservative(void) {
  primitiveToConservativeKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
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
  if (lts) {
    stampLocalDtKernel<<<cudaGridSize, cudaBlockSize>>>(*this, deltaT, ltsRatio*deltaT);
    broadcastZ(F_DTL);      // pseudo2D: k>0 is stale after a cell-looped write
  }
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
    computeRightHandSideKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&det, g_ibDetect, sizeof(det));
    { unsigned long long fd=0,fs=0,fi=0,nu=0;
      cudaMemcpyFromSymbol(&fd, g_ibFailDip,  sizeof(fd));
      cudaMemcpyFromSymbol(&fs, g_ibFailSlip, sizeof(fs));
      cudaMemcpyFromSymbol(&fi, g_ibFailIp,   sizeof(fi));
      cudaMemcpyFromSymbol(&nu, g_ibNup,      sizeof(nu));
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
  for (i32 w = 0; w < 10; w++) {
    ransFieldProbeKernel<<<cudaGridSize, cudaBlockSize>>>(*this, w);
    cudaDeviceSynchronize();
    broadcastZ(F_SCRATCH);   // pseudo2D: k>0 is stale after a guarded write
    cudaMemcpy(&v[w], thrust::max_element(thrust::device, F, F+n), sizeof(real), cudaMemcpyDefault);
  }
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
    if (lvl != nLvls-1 || !isInteriorBlock(lvl, ib, jb, kb)) continue;   // finest leaves
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;                       // pseudo-2D: one z layer
      const size_t m = (size_t)b*blockSizeTot + c;
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
  fprintf(fp, "# x/c y/c rho u v p mach cp fluid   (origin at the body centre)\n");
  fprintf(fp, "# pInf=%.10e uInf=%.10e chord=%.6f\n",
          (double)fsP, sqrt((double)fsU*fsU + (double)fsV*fsV), (double)ibChord);
  const i32 lf = nLvls - 1;
  i32 nOut = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b]; if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    if (lvl != lf || !isInteriorBlock(lvl, ib, jb, kb)) continue;
    const double dxL = domainSize[0]/(baseGridSize[0]*powi(2,lvl));
    const double dyL = domainSize[1]/(baseGridSize[1]*powi(2,lvl));
    const real hm = (real)fmin(dxL, dyL);
    for (i32 c = 0; c < blockSizeTot; c++) {
      const i32 ii = c%blockSize, jj = (c/blockSize)%blockSize, kk = c/blockSize/blockSize;
      if (kk != 0) continue;
      const double x = (ib*blockSize+ii+0.5)*dxL, y = (jb*blockSize+jj+0.5)*dyL;
      if (fabs(x - cx) > w || fabs(y - cy) > w) continue;
      const size_t m = (size_t)b*blockSizeTot + c;
      const double r = (double)getField(F_RHO)[m];
      const double u = (double)getField(F_RHOU)[m], v = (double)getField(F_RHOV)[m];
      const double pp = (double)getField(F_RHOE)[m];
      const double a2 = gam*fmax(pp,1e-30)/fmax(r,1e-30);
      const double mach = sqrt((u*u + v*v)/fmax(a2,1e-30));
      const double qInf = 0.5*((double)fsU*fsU + (double)fsV*fsV);
      const double cp = (pp - (double)fsP)/fmax(qInf,1e-30);
      const i32 fl = isFluidCell(Vec3((real)x,(real)y,(real)0), hm) ? 1 : 0;
      fprintf(fp, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %d\n",
              (x-cx)/(double)ibChord, (y-cy)/(double)ibChord, r, u, v, pp, mach, cp, fl);
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

void CompressibleSolver::stampIbGeometry(void) {
  if (immerserdBcType == 0) return;
  ibStampGeometryKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::applyWallGhosts(void) {
  if (immerserdBcType != 0) { if (!ibGhostFree) ibGhostKernel<<<cudaGridSize, cudaBlockSize>>>(*this); }
  else                      wallGhostKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::computeTurbClosure(void) {
  turbClosureKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::computeRightHandSide(void) {
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
  computeRightHandSideKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::updateFields(i32 stage) {
  updateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, stage);
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
  restrictFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::interpolateFields(void) {
  interpolateFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
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
      mass   += Rho [cIdx] * dV;
      momx   += RhoU[cIdx] * dV;
      energy += RhoE[cIdx] * dV;
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

__device__ real CompressibleSolver::tvdRec(real &ul, real &uc, real &ur) {
  // NVD-form face reconstruction: face = ul + psi(phi)*(ur - ul), where
  // phi = (uc-ul)/(ur-ul) is the normalised variable (== the limiter ratio r).
  // The stencil (ul,uc,ur) is ordered upwind->downwind, so the same formula
  // serves both sides of a face (callers pass the mirrored stencil for qR).
  real du  = ur - ul;
  real phi = (uc - ul) / (copysign(1.0, du)*fmax(abs(du), (real)1e-32));
  real psi;
  if (recon == 1) {
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
  else if (recon == 2) {
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
  else if (recon == 4) {
    // van Leer harmonic limiter in NVD form: MUSCL face u_c + (B(r)/2)(u_c-u_l)
    // with B(r) = 2r/(1+r) maps to psi = phi(2-phi) on 0 < phi < 1 (parabola
    // through (0,0),(1,1)); outside the TVD region fall back to upwind (u_c).
    psi = (phi > 0.0 && phi < 1.0) ? phi*((real)2.0 - phi) : phi;
  }
  else if (recon == 3) {
    // unlimited 3rd-order upwind parabola (kappa = 1/3 MUSCL): the psi-line the
    // ROUND schemes blend toward, with no limiting at all.
    //   face = -1/6 ul + 5/6 uc + 1/3 ur
    // For SMOOTH tests only -- oscillates at shocks.
    psi = (real)(1.0/3.0) + (real)(5.0/6.0)*phi;
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
    if (ex*ex + ey*ey > ibChord*ibChord*(real)0.04)      // > 0.2c outside the box
      return -sqrt(ex*ex + ey*ey);
    real d2min = (real)1e30;
    i32  wind  = 0;
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
      d2min = fmin(d2min, qx*qx + qy*qy);
      // winding number (crossing rule) -- sign is independent of the distance
      if ((ay > py) != (by > py)) {
        const real xint = ax + (py - ay)/(by - ay)*(bx - ax);
        if (px < xint) wind++;
      }
    }
    const real dist = sqrt(d2min);
    return (wind & 1) ? dist : -dist;      // POSITIVE INSIDE, matching the rest
  }
  if (immerserdBcType == 3) {          // cylinder about the z axis
    const real dx = pos[0]-ibCenter[0], dy = pos[1]-ibCenter[1];
    return ibRadius - sqrt(dx*dx + dy*dy);
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

// Darcy friction mask: 1 deep inside the body, 0 at the wall and in the fluid.
// Same profile (28) as the volume fraction but with its own, SHARPER width --
// the paper uses delta_Darcy = delta/2 "to ensure a quick decay of the
// tangential velocity" -- and retreated ibBrinkShift cells into the body.
//
// The sharper width is what makes a slip wall and a damped interior compatible.
// The mask must be ~0 wherever fluid actually flows: chi looks negligible per
// step, but it compounds, and a mask of only 5e-3 at the wall removes ~90% of
// the tangential velocity over a few hundred steps -- a silently no-slip wall.
// Halving the width drops the wall value by ~200x and buys back the slip.
//
// Written as sigmoid(-2x) rather than 1 - phi on purpose: 1 - 0.99999 in fp32
// keeps barely one significant digit, so the literal difference IS the noise
// floor exactly where the mask has to be small and accurate.
__host__ __device__ real CompressibleSolver::brinkDarcyMask(real s, real h) {
  const real dD = fmax(ibBrinkDelta*ibBrinkDarcyFac*h, (real)1e-30);
  const real x  = (s + ibBrinkShift*h)/dD;
  return (x > (real)0) ? exp((real)-2*x)/((real)1 + exp((real)-2*x))
                       : (real)1/((real)1 + exp((real)2*x));
}

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
__host__ __device__ bool CompressibleSolver::isFluidCell(Vec3 pos, real h) {
  if (immerserdBcType == 0) return true;
  // Volume penalization does NOT mask the body: the equations are solved
  // everywhere and the wall appears as a source term, so every cell is fluid.
  // This one line switches off the sharp-IB machinery wholesale -- no wall
  // faces, no ghost fill, no update/dt masking.
  if (ibBrink) return true;
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
