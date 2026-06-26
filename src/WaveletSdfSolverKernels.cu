#include "WaveletSdfSolverKernels.cuh"
#include "Vec3f.cuh"
#include "BvhQuery.h"


// ---------------------------------------------------------------------------
//  Tricubic-Hermite cell interpolant of the 1-jet (value + gradient).  Tensor
//  product of the 4 cubic-Hermite blending functions; each corner contributes its
//  value and its world gradient (scaled to parameter units by the edge length).
//  Matches value and gradient at every corner, so it is C1 and continuous across
//  cell faces.  (An earlier eikonal "bubble" correction was removed: enforcing
//  |grad u|=1 smoothly degrades the fit at sharp features, where the SDF is
//  genuinely non-smooth -- measured worse on sharp meshes.)
// ---------------------------------------------------------------------------

// the 4 cubic-Hermite blending functions at t in [0,1]:
// b = [psi00, psi10, psi01, psi11] (value/deriv at 0, value/deriv at 1).
__device__ inline void hermiteBasis(float t, float b[4]) {
  float t2 = t*t, t3 = t2*t;
  b[0] = 2*t3 - 3*t2 + 1;  b[1] = t3 - 2*t2 + t;  b[2] = -2*t3 + 3*t2;  b[3] = t3 - t2;
}

// tricubic Hermite at (u,v,w) in the unit cell from 8 corner jets {value,
// gradient} (gradient in world units, edges hx,hy,hz; corner order a*4+b*2+d).
__device__ inline float hermiteEval(const float *cval, const float3 *cgrad,
                                    float hx, float hy, float hz,
                                    float u, float v, float w) {
  float bx[4], by[4], bz[4];
  hermiteBasis(u, bx);  hermiteBasis(v, by);  hermiteBasis(w, bz);
  float val = 0.0f;
  for (int a = 0; a < 2; a++)
  for (int b = 0; b < 2; b++)
  for (int d = 0; d < 2; d++) {
    int idx = a*4 + b*2 + d;
    float vv = cval[idx];
    float gx = cgrad[idx].x * hx, gy = cgrad[idx].y * hy, gz = cgrad[idx].z * hz;
    val += vv*bx[2*a]*by[2*b]*bz[2*d] + gx*bx[2*a+1]*by[2*b]*bz[2*d]
         + gy*bx[2*a]*by[2*b+1]*bz[2*d] + gz*bx[2*a]*by[2*b]*bz[2*d+1];
  }
  return val;
}

// ---------------------------------------------------------------------------
//  kernels
// ---------------------------------------------------------------------------

__global__ void initWaveletSdfKernel(WaveletSdfSolver &grid) {
  // mark every NODE slot (incl. not-yet-activated blocks) unfilled, so each block's
  // 125 corner nodes are sampled exactly once when fillNodesKernel first finds it.
  i32 N = nBlocksMax * nodeSizeTot;
  for (i32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
       idx += gridDim.x * blockDim.x)
    grid.Sdf[idx] = WSDF_FAR;
}

// read the 8 corner SDF values of LOCAL cell (li,lj,lk) in block `bIdx` straight from
// that block's nodal storage (all corners are local -- no neighbour lookups), and the
// per-corner gradient by finite difference of those values (SDF-only; no stored grad).

//
// Refinement test at one on-surface mesh point `p` (true SDF = 0): if its level-
// `lvl` cell's block is active, evaluate the tricubic-Hermite (1-jet) interpolant
// at p from the cell's OWN 8 corner nodes (value,gradient) read straight from the
// grid (each corner = a cell's stored lo corner).  The error |interp - 0| is the
// reconstructed zero-contour displacement; if it exceeds thresh, split p's block
// into its 8 child blocks.  No oracle call (only a missing corner falls back).
//
__device__ inline void refineAtSurfacePoint(WaveletSdfSolver &grid,
    const BvhNode *nodes, const i32 *order, const TriFeat *tris, real orient,
    real thresh, i32 lvl, real dx, real dy, real dz, i32 gx, i32 gy, i32 gz, float3 p) {
  i32 I = (i32)floorf(p.x / dx);                  // global level-lvl cell index
  i32 J = (i32)floorf(p.y / dy);
  i32 K = (i32)floorf(p.z / dz);
  if (I < 0 || J < 0 || K < 0 || I >= gx || J >= gy || K >= gz) return;

  i32 ib = I/blockSize, jb = J/blockSize, kb = K/blockSize;
  i32 bIdx = grid.hashTable.getValue(grid.encode(lvl, ib, jb, kb));
  if (bIdx == bEmpty) return;                                   // not active

  (void)bIdx;
  // the cell's 8 corner 1-jets for the tricubic-Hermite interpolant.  Refinement runs
  // at BUILD time with oracle access, so it takes the EXACT (value, gradient) from the
  // oracle -- the tricubic is exact only with exact gradients, and any finite-diff of
  // the stored SDF over-refines.  (Storage itself stays SDF value-only: this gradient
  // is used here and discarded, never stored.)
  float  cval[8];
  float3 cgrad[8];
  for (i32 a=0;a<2;a++) for (i32 b=0;b<2;b++) for (i32 d=0;d<2;d++) {
    float3 cp = make_float3((I+a)*dx, (J+b)*dy, (K+d)*dz);
    cval[a*4+b*2+d] = signedDistanceGrad(nodes, order, tris, orient, cp, cgrad[a*4+b*2+d]);
  }

  // tricubic-Hermite interpolant of the 1-jet over the cell (matches value +
  // gradient at the 8 corners).  True f(p)=0, so |interp| is the reconstructed
  // zero-contour displacement.  Local coords of p within the cell.
  float u = fminf(fmaxf(p.x/dx - I, 0.0f), 1.0f);
  float v = fminf(fmaxf(p.y/dy - J, 0.0f), 1.0f);
  float w = fminf(fmaxf(p.z/dz - K, 0.0f), 1.0f);
  float err = fabsf(hermiteEval(cval, cgrad, dx, dy, dz, u, v, w));

  if (err > thresh) {
    for (i32 dk = 0; dk < 2; dk++)
    for (i32 dj = 0; dj < 2; dj++)
    for (i32 di = 0; di < 2; di++)
      grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj, 2*kb+dk);
  }
}

// per-level boilerplate shared by the two refinement kernels
#define REFINE_SETUP(grid, lvl)                                           \
  const BvhNode *nodes = grid.dNodes;                                     \
  const i32     *order = grid.dOrder;                                     \
  const TriFeat *tris  = grid.dTris;                                      \
  const real     orient = grid.orient;                                    \
  const real     thresh = grid.thresh;                                    \
  if (lvl >= grid.nLvls - 1) return;                                      \
  real dx = grid.getDx(lvl), dy = grid.getDy(lvl), dz = grid.getDz(lvl);  \
  i32 gx = grid.baseGridSize[0]*powi(2,lvl);                              \
  i32 gy = grid.baseGridSize[1]*powi(2,lvl);                              \
  i32 gz = grid.baseGridSize[2]*powi(2,lvl)

//
// Refinement driven by the mesh, focused on the zero contour.  Two kernels split
// the on-surface refinement points so no work is repeated: the welded mesh
// VERTICES (each touched once, not once per incident triangle) and the per-
// triangle FACE CENTERS.  Both lie exactly on the surface (true SDF = 0).
//
__global__ void flagRefineVertsKernel(WaveletSdfSolver &grid, i32 lvl) {
  REFINE_SETUP(grid, lvl);
  const float3 *verts = grid.dVerts;
  const i32 nVerts = grid.nVerts;
  for (i32 i = blockIdx.x * blockDim.x + threadIdx.x; i < nVerts;
       i += gridDim.x * blockDim.x)
    refineAtSurfacePoint(grid, nodes, order, tris, orient, thresh, lvl, dx, dy, dz, gx, gy, gz, verts[i]);
}

__global__ void flagRefineCentersKernel(WaveletSdfSolver &grid, i32 lvl) {
  REFINE_SETUP(grid, lvl);
  const i32 nTris = grid.nTris;
  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    float3 c = (T.v0 + T.v1 + T.v2) * (1.0f/3.0f);
    refineAtSurfacePoint(grid, nodes, order, tris, orient, thresh, lvl, dx, dy, dz, gx, gy, gz, c);
  }
}

// Sign-consistency refinement.  The mesh-point criterion above can leave a coarse
// leaf un-refined when the surface only GRAZES one of its faces/edges: the cell's 8
// corners are all one sign, so it gets no dual-contour vertex, yet a finer neighbour
// straddles the shared face -> the T-junction fan can't close = a crack.  Here one
// thread per cell: for each near-surface LEAF cell whose corners don't straddle,
// sample the would-be child corners (the 19 cell/face/edge sub-nodes) with the
// oracle; if any flips sign the corners are hiding the surface, so split the block.
// This is gated to the thin "grazing shell" (corners all one sign AND within a cell
// diagonal of the surface), so the extra oracle calls stay bounded.
__global__ void flagRefineSignFlipKernel(WaveletSdfSolver &grid, i32 lvl) {
  const BvhNode *nodes = grid.dNodes;
  const i32     *order = grid.dOrder;
  const TriFeat *tris  = grid.dTris;
  const real     orient = grid.orient;
  if (lvl >= grid.nLvls - 1) return;
  real dx = grid.getDx(lvl), dy = grid.getDy(lvl), dz = grid.getDz(lvl);

  START_CELL_LOOP
    GET_CELL_INDICES
    i32 blvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], blvl, ib, jb, kb);
    if (blvl == lvl && grid.isInteriorBlock(lvl, ib, jb, kb) &&
        grid.hashTable.getValue(grid.encode(lvl+1, 2*ib, 2*jb, 2*kb)) == bEmpty) {  // leaf block
      i32 I = ib*blockSize + i, J = jb*blockSize + j, K = kb*blockSize + k;
      const real *S = grid.Sdf + (size_t)bIdx*nodeSizeTot;     // local nodal storage
      i32 nNeg = 0; float minAbs = 1e30f;
      for (i32 a=0;a<2;a++) for (i32 b=0;b<2;b++) for (i32 d=0;d<2;d++) {
        float val = S[WaveletSdfSolver::nodeIdx(i+a, j+b, k+d)];
        if (val < 0.0f) nNeg++;
        minAbs = fminf(minAbs, fabsf(val));
      }
      real diag = sqrtf(dx*dx + dy*dy + dz*dz);
      if ((nNeg == 0 || nNeg == 8) && minAbs < diag) {     // doesn't straddle, but near the surface
        float cs = (nNeg == 8) ? -1.0f : 1.0f;
        bool flip = false;
        for (i32 a=0;a<3 && !flip;a++)
        for (i32 b=0;b<3 && !flip;b++)
        for (i32 d=0;d<3 && !flip;d++) {
          if (((a&1)==0) && ((b&1)==0) && ((d&1)==0)) continue;   // a corner, not a sub-node
          float3 p = make_float3((I+0.5f*a)*dx, (J+0.5f*b)*dy, (K+0.5f*d)*dz);
          float3 gg;
          if ((signedDistanceGrad(nodes, order, tris, orient, p, gg) < 0.0f ? -1.0f : 1.0f) != cs)
            flip = true;
        }
        if (flip)
          for (i32 dk=0;dk<2;dk++) for (i32 dj=0;dj<2;dj++) for (i32 di=0;di<2;di++)
            grid.activateBlock(lvl+1, 2*ib+di, 2*jb+dj, 2*kb+dk);
      }
    }
  END_CELL_LOOP
}

//
// One 2:1-balance (grading) pass.  Ported from ../TensorTrain/rs meshwave.rs
// balance(): no leaf may be adjacent to a region more than one level finer.  We
// drive it from REFINED blocks (those with children = one level finer on their
// side): for each of a refined level-L block's 6 face-neighbor regions, find the
// deepest existing block covering it (level nl); if nl < L that neighbor leaf is
// >= 2 levels coarser than this block's children, so split it (activate its 8
// children at nl+1).  We only ever split an EXISTING block, whose parent exists by
// construction, so no orphans are created; the fixpoint (re-run until no new
// blocks) ripples a coarse leaf down one level at a time to within 2:1.
//
__global__ void gradeKernel(WaveletSdfSolver &grid) {
  START_BLOCK_LOOP

    i32 L, ib, jb, kb;
    u64 loc = grid.bLocList[bIdx];
    grid.decode(loc, L, ib, jb, kb);

    // a refined block has children one level finer (created as full octants, so
    // testing the 000 child is enough); finest-level blocks are never refined.
    bool refinedBlk = loc != kEmpty && L < grid.nLvls - 1 &&
                      grid.isInteriorBlock(L, ib, jb, kb) &&
                      grid.hashTable.getValue(grid.encode(L+1, 2*ib, 2*jb, 2*kb)) != bEmpty;

    if (refinedBlk) {
      for (i32 axis = 0; axis < 3; axis++)
      for (i32 dir = -1; dir <= 1; dir += 2) {
        i32 nb[3] = {ib, jb, kb};
        nb[axis] += dir;
        if (grid.isExteriorBlock(L, nb[0], nb[1], nb[2])) continue;

        // deepest existing block covering the neighbor region (monotone: parent
        // chains are complete, and the level-0 base grid always covers it)
        i32 nl = -1, ni = 0, nj = 0, nk = 0;
        for (i32 m = L; m >= 0; m--) {
          i32 s = L - m;
          i32 mi = nb[0] >> s, mj = nb[1] >> s, mk = nb[2] >> s;
          if (grid.hashTable.getValue(grid.encode(m, mi, mj, mk)) != bEmpty) {
            nl = m; ni = mi; nj = mj; nk = mk; break;
          }
        }

        if (nl >= 0 && nl < L) {                 // neighbor leaf too coarse: split it
          for (i32 dk = 0; dk < 2; dk++)
          for (i32 dj = 0; dj < 2; dj++)
          for (i32 di = 0; di < 2; di++)
            grid.activateBlock(nl+1, 2*ni+di, 2*nj+dj, 2*nk+dk);
        }
      }
    }

  END_BLOCK_LOOP
}

// Sample the oracle once at every just-activated interior cell (those still holding
// the WSDF_FAR sentinel), storing the (value, gradient) at the cell's LO CORNER node
// -- the node it uniquely owns.  Already-filled cells are skipped, so each node is
// sampled exactly once across the whole build, no matter how many times this runs.
__global__ void fillNodesKernel(WaveletSdfSolver &grid) {
  real *Sdf = grid.Sdf;
  const BvhNode *nodes = grid.dNodes;
  const i32     *order = grid.dOrder;
  const TriFeat *tris  = grid.dTris;
  const real     orient = grid.orient;
  const i32 nslots = grid.hashTable.nKeys * nodeSizeTot;     // 125 corner nodes per active block

  for (i32 idx = blockIdx.x*blockDim.x + threadIdx.x; idx < nslots; idx += gridDim.x*blockDim.x) {
    if (Sdf[idx] != WSDF_FAR) continue;                       // already filled (sample once)
    i32 b  = idx / nodeSizeTot, ln = idx % nodeSizeTot;
    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[b], lvl, ib, jb, kb);
    if (!grid.isInteriorBlock(lvl, ib, jb, kb)) continue;
    i32 ni = ln % blockSizeNode, nj = (ln/blockSizeNode) % blockSizeNode, nk = ln/(blockSizeNode*blockSizeNode);
    real dx = grid.getDx(lvl), dy = grid.getDy(lvl), dz = grid.getDz(lvl);
    float3 node = make_float3((ib*blockSize+ni)*dx, (jb*blockSize+nj)*dy, (kb*blockSize+nk)*dz);
    float3 g;
    Sdf[idx] = signedDistanceGrad(nodes, order, tris, orient, node, g);   // value only (g discarded)
  }
}
