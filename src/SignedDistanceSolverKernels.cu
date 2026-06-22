#include "SignedDistanceSolverKernels.cuh"
#include "Vec3f.cuh"

//
// device geometry helpers (closest point on triangle + pseudonormal sign),
// reused unchanged from the single-level narrowband solver.
//

// Keep the int16 code at `addr` with the smaller magnitude.  `val` is the fp32
// candidate distance; it is quantized to int16 (round(val*invQ), clamped to the
// real range [-32767, 32767]) only on store.  Because the quantum is positive,
// comparing the integer codes' magnitudes is equivalent to comparing the fp32
// distances' magnitudes, and the unreached sentinel INT16_MIN has magnitude
// 32768 so it always loses to the first real candidate.  The CAS runs on the
// underlying 16-bit word (sm_70+ supports 16-bit atomicCAS); the initial read is
// `volatile` so strict aliasing cannot drop the cross-thread store from
// initSdfKernel.
__device__ inline void atomicMinMag(i16 *addr, float val, float invQ) {
  int q = __float2int_rn(val * invQ);
  q = max(-32767, min(32767, q));                     // leave INT16_MIN as sentinel
  unsigned short vbits = (unsigned short)(i16)q;
  unsigned short *p = reinterpret_cast<unsigned short*>(addr);
  unsigned short old = *(volatile unsigned short*)p;
  while (abs((int)(i16)old) > abs(q)) {               // current loses
    unsigned short assumed = old;
    old = atomicCAS(p, assumed, vbits);
    if (old == assumed) break;                        // success
  }
}

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision
// Detection). `region`: 0,1,2 = vertices a,b,c; 3,4,5 = edges ab,bc,ca; 6 = face.
__host__ __device__ inline float3 closestPtTriangle(
    float3 p, float3 a, float3 b, float3 c, int &region) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) { region = 0; return a; }

  float3 bp = p - b;
  float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) { region = 1; return b; }

  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = d1 / (d1 - d3); region = 3; return a + ab * v;
  }

  float3 cp = p - c;
  float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) { region = 2; return c; }

  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = d2 / (d2 - d6); region = 5; return a + ac * w;
  }

  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); region = 4; return b + (c - b) * w;
  }

  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom, w = vc * denom;
  region = 6; return a + ab * v + ac * w;
}

// Pseudonormal of the feature selected by `region` (Baerentzen & Aanaes sign).
__device__ inline float3 selectPN(const TriFeat &T, int region) {
  switch (region) {
    case 0:  return T.vn0;
    case 1:  return T.vn1;
    case 2:  return T.vn2;
    case 3:  return T.en0;
    case 4:  return T.en1;
    case 5:  return T.en2;
    default: return T.fn;
  }
}

// ---------------------------------------------------------------------------

__global__ void initSdfKernel(SignedDistanceSolver &grid) {
  i16 *Sdf = grid.Sdf;
  START_CELL_LOOP
    Sdf[cIdx] = SDF_FAR;   // sentinel; the distance sweep fills reached cells
  END_CELL_LOOP
}

//
// Pass 1: activate every block touched by a triangle's narrowband.  One thread
// per triangle sweeps the cells in its AABB grown by `band` and activates the
// block of any cell within `band` of the surface.  Memory therefore scales with
// the surface area, not the domain volume (Roosing, Strickson & Nikiforakis,
// CiCP 2019).
//
__global__ void registerCellsSdfKernel(SignedDistanceSolver &grid) {
  const TriFeat *tris = grid.dTris;
  i32 nTris = grid.nTris;
  real dx = grid.getDx(0);
  i32 gx = grid.baseGridSize[0], gy = grid.baseGridSize[1], gz = grid.baseGridSize[2];

  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    float3 lo = fmin3(fmin3(T.v0, T.v1), T.v2);
    float3 hi = fmax3(fmax3(T.v0, T.v1), T.v2);
    real grow = grid.band;

    i32 i0 = max(0,    (i32)floorf((lo.x - grow) / dx));
    i32 j0 = max(0,    (i32)floorf((lo.y - grow) / dx));
    i32 k0 = max(0,    (i32)floorf((lo.z - grow) / dx));
    i32 i1 = min(gx-1, (i32)floorf((hi.x + grow) / dx) + 1);
    i32 j1 = min(gy-1, (i32)floorf((hi.y + grow) / dx) + 1);
    i32 k1 = min(gz-1, (i32)floorf((hi.z + grow) / dx) + 1);

    for (i32 k = k0; k <= k1; ++k)
      for (i32 j = j0; j <= j1; ++j)
        for (i32 i = i0; i <= i1; ++i) {
          float3 p = make_float3((i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx);
          int region;
          float3 q = closestPtTriangle(p, T.v0, T.v1, T.v2, region);
          if (norm(p - q) <= grid.band)
            grid.activateBlock(0, i/blockSize, j/blockSize, k/blockSize);
        }
  }
}

//
// Pass 2: exact signed distance for every cell of an active block (triangle-
// parallel, atomic min-magnitude, no band clamp -> stores the true signed
// distance).  A block was activated (pass 1) because one of its cells lies
// within `band` of the surface, so every cell of an active block is within
// band + blockDiag of the surface.  Growing each triangle's reach by that radius
// therefore guarantees every active-block cell is visited by its true nearest
// triangle, so the band cutoff can be dropped and the whole active block is
// filled exactly.  The hash lookup is done first so the closest-point test runs
// only for cells that actually land in an active block.
//
__global__ void computeSdfKernel(SignedDistanceSolver &grid) {
  i16 *Sdf = grid.Sdf;
  const float invQ = grid.sdfInvQuantum;
  const TriFeat *tris = grid.dTris;
  i32 nTris = grid.nTris;
  real dx = grid.getDx(0);
  i32 gx = grid.baseGridSize[0], gy = grid.baseGridSize[1], gz = grid.baseGridSize[2];
  const real grow = grid.band + blockSize * dx * 1.7320508f;   // band + block diag

  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    float3 lo = fmin3(fmin3(T.v0, T.v1), T.v2);
    float3 hi = fmax3(fmax3(T.v0, T.v1), T.v2);

    i32 i0 = max(0,    (i32)floorf((lo.x - grow) / dx));
    i32 j0 = max(0,    (i32)floorf((lo.y - grow) / dx));
    i32 k0 = max(0,    (i32)floorf((lo.z - grow) / dx));
    i32 i1 = min(gx-1, (i32)floorf((hi.x + grow) / dx) + 1);
    i32 j1 = min(gy-1, (i32)floorf((hi.y + grow) / dx) + 1);
    i32 k1 = min(gz-1, (i32)floorf((hi.z + grow) / dx) + 1);

    for (i32 k = k0; k <= k1; ++k)
      for (i32 j = j0; j <= j1; ++j)
        for (i32 i = i0; i <= i1; ++i) {
          i32 ib = i/blockSize, jb = j/blockSize, kb = k/blockSize;
          i32 bIdx = grid.hashTable.getValue(grid.encode(0, ib, jb, kb));
          if (bIdx == bEmpty) continue;          // cell not in an active block
          float3 p = make_float3((i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx);
          int region;
          float3 q = closestPtTriangle(p, T.v0, T.v1, T.v2, region);
          float3 dvec = p - q;
          float dist = norm(dvec);
          float s = (dot(dvec, selectPN(T, region)) >= 0.0f) ? 1.0f : -1.0f;
          i32 cell = bIdx * blockSizeTot + (i - ib*blockSize)
                   + (j - jb*blockSize) * blockSize
                   + (k - kb*blockSize) * blockSize * blockSize;
          atomicMinMag(&Sdf[cell], s * dist, invQ);
        }
  }
}

//
// Mark every reached cell of an interior block ACTIVE.  The narrowband fills a
// whole block exactly (pass 2), so there is no reconstruction halo to leave as
// GHOST; the cell-flag pass from sortBlocks would otherwise ghost band-edge
// cells (its halo neighbors are missing).  Unreached cells (none, in practice)
// stay GHOST so the slice image and the report only count real distances.
//
__global__ void flagBandCellsActiveSdfKernel(SignedDistanceSolver &grid) {
  i16 *Sdf = grid.Sdf;
  START_CELL_LOOP

    u64 loc = grid.bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    grid.decode(loc, lvl, ib, jb, kb);

    bool reached = grid.isInteriorBlock(lvl, ib, jb, kb)
                && Sdf[cIdx] != SDF_FAR;
    grid.cFlagsList[cIdx] = reached ? ACTIVE : GHOST;

  END_CELL_LOOP
}

