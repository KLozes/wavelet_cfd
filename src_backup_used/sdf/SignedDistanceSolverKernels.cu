#include "SignedDistanceSolverKernels.cuh"
#include "Vec3f.cuh"

//
// device geometry helpers (closest point on triangle + pseudonormal sign),
// reused unchanged from the single-level narrowband solver.
//

// Keep the fp32 distance at `addr` with the smaller magnitude.  The unreached
// sentinel SDF_FAR has the largest magnitude, so it always loses to the first
// real candidate.  The CAS runs on the underlying 32-bit word; the initial read
// is `volatile` so strict aliasing cannot drop the cross-thread store from
// initSdfKernel.
__device__ inline void atomicMinMag(real *addr, real val) {
  unsigned int  vbits = __float_as_uint(val);
  unsigned int *p     = reinterpret_cast<unsigned int*>(addr);
  unsigned int  old   = *(volatile unsigned int*)p;
  while (fabsf(__uint_as_float(old)) > fabsf(val)) {  // current loses
    unsigned int assumed = old;
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
  real *Sdf = grid.Sdf;
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
__global__ void registerCellsSdfKernel(SignedDistanceSolver &grid, i32 lvl, real band) {
  const TriFeat *tris = grid.dTris;
  i32 nTris = grid.nTris;
  real dx = grid.getDx(lvl);
  i32 gx = grid.baseGridSize[0]*powi(2,lvl);
  i32 gy = grid.baseGridSize[1]*powi(2,lvl);
  i32 gz = grid.baseGridSize[2]*powi(2,lvl);

  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    float3 lo = fmin3(fmin3(T.v0, T.v1), T.v2);
    float3 hi = fmax3(fmax3(T.v0, T.v1), T.v2);

    i32 i0 = max(0,    (i32)floorf((lo.x - band) / dx));
    i32 j0 = max(0,    (i32)floorf((lo.y - band) / dx));
    i32 k0 = max(0,    (i32)floorf((lo.z - band) / dx));
    i32 i1 = min(gx-1, (i32)floorf((hi.x + band) / dx) + 1);
    i32 j1 = min(gy-1, (i32)floorf((hi.y + band) / dx) + 1);
    i32 k1 = min(gz-1, (i32)floorf((hi.z + band) / dx) + 1);

    for (i32 k = k0; k <= k1; ++k)
      for (i32 j = j0; j <= j1; ++j)
        for (i32 i = i0; i <= i1; ++i) {
          float3 p = make_float3((i + 0.5f) * dx, (j + 0.5f) * dx, (k + 0.5f) * dx);
          int region;
          float3 q = closestPtTriangle(p, T.v0, T.v1, T.v2, region);
          if (norm(p - q) <= band)
            grid.activateBlock(lvl, i/blockSize, j/blockSize, k/blockSize);
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
__global__ void computeSdfKernel(SignedDistanceSolver &grid, i32 lvl, real band) {
  real *Sdf = grid.Sdf;
  const TriFeat *tris = grid.dTris;
  i32 nTris = grid.nTris;
  real dx = grid.getDx(lvl);
  i32 gx = grid.baseGridSize[0]*powi(2,lvl);
  i32 gy = grid.baseGridSize[1]*powi(2,lvl);
  i32 gz = grid.baseGridSize[2]*powi(2,lvl);
  const real grow = band + blockSize * dx * 1.7320508f;   // band + block diag

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
          i32 bIdx = grid.hashTable.getValue(grid.encode(lvl, ib, jb, kb));
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
          atomicMinMag(&Sdf[cell], s * dist);
        }
  }
}

// level-0 coarse full grid: brute force every interior level-0 cell against all
// triangles (cell-parallel, register min-magnitude, single store).  The coarse
// grid is small, so this gives the real far field for free; finer levels are
// filled by the narrowband computeSdfKernel above.
__global__ void computeSdfCoarseKernel(SignedDistanceSolver &grid) {
  real *Sdf = grid.Sdf;
  const TriFeat *tris = grid.dTris;
  i32 nTris = grid.nTris;

  START_CELL_LOOP
    GET_CELL_INDICES

    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);

    if (lvl == 0 && grid.isInteriorBlock(lvl, ib, jb, kb)) {
      Vec3 pos = grid.getCellPos(0, ib, jb, kb, i, j, k);
      float3 p = make_float3(pos[0], pos[1], pos[2]);
      float best = 1e30f, bestSigned = 1e30f;
      for (i32 t = 0; t < nTris; t++) {
        TriFeat T = tris[t];
        int region;
        float3 q = closestPtTriangle(p, T.v0, T.v1, T.v2, region);
        float3 dvec = p - q;
        float d = norm(dvec);
        if (d < best) {
          best = d;
          float s = (dot(dvec, selectPN(T, region)) >= 0.0f) ? 1.0f : -1.0f;
          bestSigned = s * d;
        }
      }
      Sdf[cIdx] = bestSigned;
    }

  END_CELL_LOOP
}


