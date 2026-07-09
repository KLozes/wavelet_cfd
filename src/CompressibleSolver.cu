#include <iostream>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <thrust/extrema.h>

#include "CompressibleSolver.cuh"
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
  // (mdFlux == 2 with scheme 1: the Hancock predictor time-centres the slope
  // DOFs' feed -- face momentum fluxes and volume term -- making the g<->p
  // coupling a partitioned midpoint-like integration, unlike the old
  // transverse-only CTU whose pure-FE corrector was unconditionally unstable
  // for the dispersive slope modes.)
  // wavelet-normalization scales (device-side global maxima)
  cudaMallocManaged(&globalScale, 4*sizeof(real));
  cudaMemset(globalScale, 0, 4*sizeof(real));
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
  paint();

  // build the adaptive grid by repeatedly transforming / refining
  for (i32 lvl=1; lvl<nLvls; lvl++) {
    forwardWaveletTransform();
    adaptGrid();               // refines OWNED blocks (multi-GPU); domain-exterior ring
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
    paint();
  }
  zeroAccumulator();   // the cascade dirtied the shared bank (LSRK needs 0)
}

// zero the shared bank so the LSRK accumulation (A_1 = 0) starts clean; the
// NEVOLVE fields are contiguous slabs of fieldData
void CompressibleSolver::zeroAccumulator(void) {
  cudaMemset(getField(F_RHS), 0,
             (size_t)NEVOLVE*(size_t)blockSizeTot*(size_t)nBlocksMax*sizeof(real));
}

real CompressibleSolver::step(real tStep) {

  real t = 0;

  Timer<std::chrono::milliseconds, std::chrono::steady_clock> clock;
  Timer<std::chrono::microseconds, std::chrono::steady_clock> sub;   // profiling sub-timer

  while (t < tStep) {

    clock.tick();
    // dynamic wavelet adaptation; skipped for a static (fixed) refinement grid
    if (iter % 4 == 0 && nLvls > 1 && !staticGrid) {
#ifdef USE_MGPU
      haloExchange(0, NEVOLVE);   // fill last cycle's ghosts before the detail computation
#endif
      restrictFields();
      cudaDeviceSynchronize(); sub.tick();
      forwardWaveletTransform();
      cudaDeviceSynchronize(); sub.tock(); tForwardUs += sub.duration().count();
      adaptGrid();
      // The inverse fills new owned blocks from the F_OLD snapshot via the
      // parents' (valid) neighbor indices.  The old ghost layer is kept through
      // adaptGrid (flagged KEEP, not propagated since grading is owned-only) so
      // those coarse-ghost parents are present and F_OLD-valid here.  sortBlocks
      // clobbers F_OLD (permutation scratch), so it MUST come after the inverse.
      // Domain-boundary ghost blocks CREATED by this adaptGrid have an all-zero
      // F_OLD (never BC-filled), and a seam block's prediction taps read it ->
      // refresh the snapshot bank's exterior ghosts before the inverse.
      setBoundaryConditions(F_OLD);
      inverseWaveletTransform();
      cudaDeviceSynchronize(); sub.tick();
      sortBlocks();
      cudaDeviceSynchronize(); sub.tock(); tSortUs += sub.duration().count();
#ifdef USE_MGPU
      rebuildGhosts();            // prune the stale ghosts, recreate the 2-ring from neighbors
      haloExchange(0, NEVOLVE);   // fill the fresh ghost blocks
#endif
      setBoundaryConditions();
      // the wavelet snapshot / sort buffer dirtied the shared bank; the LSRK
      // accumulator must be zero when stage 1 begins (A_1 = 0)
      zeroAccumulator();
    }
    cudaDeviceSynchronize();
    clock.tock();
    tGrid += clock.duration().count();

    clock.tick();
    computeDeltaT();
    if (t + deltaT > tStep) {
      deltaT = tStep - t;
    }

    if (mdFlux == 2) {
      // CTU-Hancock: fully-discrete predictor-corrector.  The corrector is
      // FUSED into multiDRhsKernel (it updates q in place, conservative), so
      // there is no primitiveToConservative/updateFields here; the shared
      // bank holds the half-step predicted primitives during the RHS.
      conservativeToPrimitive();
      setBoundaryConditions();
      computeRightHandSide();
      setBoundaryConditions();
      if (nLvls > 1) {
        restrictFields();
        interpolateFields();
        setBoundaryConditions();
      }
    }
    else
    for (i32 stage = 0; stage < 3; stage++) {
      conservativeToPrimitive();
#ifdef USE_MGPU
      haloExchange(0, NEVOLVE);   // ghosts get owners' primitives (+G) before the RHS reads them
#endif
      setBoundaryConditions();    // AFTER halo: periodic exterior ghosts copy the freshly-haloed wrap image
      computeRightHandSide();
      primitiveToConservative();
      updateFields(stage);
      setBoundaryConditions();

      if (nLvls > 1) {
        restrictFields();
        interpolateFields();
        setBoundaryConditions();
#ifdef USE_MGPU
        haloExchange(0, NEVOLVE);   // refresh ghosts after the coarse/fine reconstruction
#endif
      }
    }
    cudaDeviceSynchronize();
    clock.tock();
    tSolver += clock.duration().count();

    t += deltaT;
    iter++;
  }

  return t;
}

void CompressibleSolver::sortFieldData(void) {
  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  sortFieldDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::setInitialConditions(void) {
  setInitialConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void CompressibleSolver::setBoundaryConditions(i32 fOff) {
  setBoundaryConditionsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, fOff);
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
  cudaMemset(globalScale, 0, 4*sizeof(real));
  computeGlobalScalesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
#ifdef USE_MGPU
  // the wavelet threshold must use the SAME normalization on every PE, else
  // partitions refine against inconsistent scales -> take the domain-wide max.
  cudaDeviceSynchronize();
  comm::allreduceMax(globalScale, 4);
#endif

  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  // The face-flux scatter atomicAdds missing-neighbor contributions into the
  // bEmpty trash block's slice of the ACCUMULATOR bank -- which is aliased
  // with this snapshot bank.  bEmpty sits outside the cell loop, so neither
  // the S-roll in updateFields nor copyToOld ever cleans it (and NaN*0 = NaN
  // defeats the *=0 roll anyway).  waveletPredict reads the snapshot through
  // the same missing-neighbor path, so clear the trash before it is read.
  for (i32 f = 0; f < NEVOLVE; f++) {
    cudaMemset(getField(F_OLD + f) + (u64)bEmpty*blockSizeTot, 0, blockSizeTot*sizeof(real));
    // live-field trash slice too: interpolate/restrict (and a missing-parent
    // prediction via bEmpty's self-pointing neighbor row) read live fields
    // through parent taps, and nothing else ever cleans this slice.
    cudaMemset(getField(f) + (u64)bEmpty*blockSizeTot, 0, blockSizeTot*sizeof(real));
  }
  copyToOldFieldsKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  forwardWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  waveletThresholdingKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::inverseWaveletTransform(void) {
  inverseWaveletTransformKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
}

void CompressibleSolver::computeDeltaT(void) {
  computeDeltaTKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  real *dmin = thrust::min_element(thrust::device, getField(F_SCRATCH), getField(F_SCRATCH)+hashTable.nKeys*blockSizeTot);
#ifdef USE_MGPU
  // fieldData may be device-only (symmetric heap): copy the min to host rather
  // than dereferencing on the host, then take the min across all PEs.
  real localMin;
  cudaMemcpy(&localMin, dmin, sizeof(real), cudaMemcpyDefault);
  comm::allreduceMin(&localMin, 1);
  deltaT = localMin * cfl;
#else
  deltaT = (*dmin) * cfl;
#endif
}

void CompressibleSolver::computeRightHandSide(void) {
  if (mdFlux) {
    if (mdFlux == 2) {
      // CTU-Hancock: half-step-predict all cells into the Old bank (free until
      // updateFields), fill its halos, then assemble the multiD fluxes on the
      // time-centred field
      hancockPredictKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      setBoundaryConditions(F_OLD);
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
  // keep (un-delete only) the existing non-owned blocks that owned stencils
  // need -- locally-manufactured support at an advancing rank seam survives
  // even when no other rank has it (see keepLocalSupportKernel)
  keepLocalSupportKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;
  deleteDataKernel<<<cudaGridSize, cudaBlockSize>>>(*this);      // drop the ghosts no longer needed
  cudaDeviceSynchronize();
  sortBlocks();                                                  // compact + rebuild indices
  comm::barrier();
}
#endif

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
  i32 kbT = (baseGridSize[2] / blockSize) / 2;
  i32 kT  = blockSize / 2;

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
    i32 gz = baseGridSize[2]*powi(2,lvl)/blockSize;
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
    i32 gz = baseGridSize[2]*powi(2,lvl)/blockSize;
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
    i32 gz = baseGridSize[2]*powi(2,L)/blockSize;
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
// equilibrium (the RT0/P0 DG's headline property); lower is better.
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
    i32 gz = pseudo2D ? baseGridSize[2]/blockSize : baseGridSize[2]/blockSize*powi(2,lvl);
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
  printf("---- vortex L2 error (vs exact stationary) ----\n");
  printf("  scheme %d   L2(rho) = %.4e   L2(|u|) = %.4e   L2(p) = %.4e\n",
         scheme, sqrt(l2Rho/area), sqrt(l2Vel/area), sqrt(l2P/area));
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
    i32 gz = pseudo2D ? baseGridSize[2]/blockSize : baseGridSize[2]/blockSize*powi(2,lvl);
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
  printf("---- Gresho vortex diagnostic (scheme %d, Ma=%.3g, %s grid) ----\n",
         scheme, greshoP0 > 0 ? 1.0/sqrt(gam*greshoP0) : 0.0,
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
    i32 gz = pseudo2D ? baseGridSize[2]/blockSize : baseGridSize[2]/blockSize*powi(2,lvl);
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
  cudaMemset(globalScale, 0, 4*sizeof(real));
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
// Pressure from P0 density/energy and the RT0 momentum cell-averages, evaluated
// in double precision.  At low Mach number the (E - ½ρ|u|²) subtraction cancels
// catastrophically and single precision loses all pressure information, so the
// internals are promoted to double (as in the reference `pressure_from_rt0`).
//
__device__ real CompressibleSolver::pressureRT(real rho, real mxa, real mya, real mza, real E) {
  double u = (double)mxa / (double)rho;
  double v = (double)mya / (double)rho;
  double w = (double)mza / (double)rho;
  double p = ((double)gam - 1.0) * ((double)E - 0.5 * (double)rho * (u*u + v*v + w*w));
  return (real)p;
}

__device__ real CompressibleSolver::getBoundaryLevelSet(Vec3 pos) {
  if (immerserdBcType == 1) {
    // sphere
    real radius = .05;
    real center[3] = {.5, .5, .5};
    return radius - sqrt((pos[0]-center[0])*(pos[0]-center[0])
                       + (pos[1]-center[1])*(pos[1]-center[1])
                       + (pos[2]-center[2])*(pos[2]-center[2]));
  }
  else {
    return 1e32;
  }
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

  // Wave speed estimates
  real SL = fmin(vnL - aL, vn - a);
  real SR = fmax(vnR + aR, vn + a);
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
