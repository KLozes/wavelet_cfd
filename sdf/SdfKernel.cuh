#ifndef SDF_KERNEL_CUH
#define SDF_KERNEL_CUH

// GPU narrowband signed-distance transform.
//
// Following Roosing, Strickson & Nikiforakis (CiCP 2019), work is parallelised
// over surface features: one thread per triangle loops over the cells in the
// triangle's bounding volume (its AABB grown by the band width) and writes the
// smallest-magnitude signed distance with an atomic compare-and-swap. Only
// cells within the band are touched, and they live in a SingleLevelSparseGrid
// backed by the repo's GPU HashTable, so memory scales with the surface area
// rather than the domain volume.
//
// The sign comes from the angle-weighted pseudonormal of whichever feature
// (face / edge / vertex) carries the closest point (Baerentzen & Aanaes), which
// is exact for the convex / concave / saddle / ruff cases discussed in the
// paper.

#include "SingleLevelSparseGrid.cuh"
#include "Vec3f.cuh"
#include "Features.h"

// Keep the value at `addr` with the smaller magnitude (the float bit pattern is
// updated via integer CAS, the doc's atomic write strategy).
__device__ inline void atomicMinMag(float* addr, float val) {
  int* iaddr = reinterpret_cast<int*>(addr);
  int old = *iaddr, assumed;
  do {
    assumed = old;
    float cur = __int_as_float(assumed);
    if (fabsf(cur) <= fabsf(val)) break;   // existing value already wins
    old = atomicCAS(iaddr, assumed, __float_as_int(val));
  } while (assumed != old);
}

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision
// Detection). `region` reports which feature owns the closest point:
// 0,1,2 = vertices a,b,c; 3,4,5 = edges ab,bc,ca; 6 = face interior.
__host__ __device__ inline float3 closestPtTriangle(
    float3 p, float3 a, float3 b, float3 c, int& region) {
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

// Pseudonormal of the feature selected by `region`.
__device__ inline float3 selectPN(const TriFeat& T, int region) {
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

// Cell index range (inclusive) of a triangle's AABB grown by `grow` (world
// units), clamped to the grid.
__device__ inline void cellRange(const TriFeat& T, SingleLevelSparseGrid& grid,
                                 float grow,
                                 i32& i0, i32& i1, i32& j0, i32& j1,
                                 i32& k0, i32& k1) {
  float3 lo = fmin3(fmin3(T.v0, T.v1), T.v2);
  float3 hi = fmax3(fmax3(T.v0, T.v1), T.v2);
  float dx = grid.dx, b = grow;
  i0 = max(0, (i32)floorf((lo.x - b - grid.domainOrigin[0]) / dx));
  j0 = max(0, (i32)floorf((lo.y - b - grid.domainOrigin[1]) / dx));
  k0 = max(0, (i32)floorf((lo.z - b - grid.domainOrigin[2]) / dx));
  i1 = min(grid.gridSize[0] - 1, (i32)floorf((hi.x + b - grid.domainOrigin[0]) / dx) + 1);
  j1 = min(grid.gridSize[1] - 1, (i32)floorf((hi.y + b - grid.domainOrigin[1]) / dx) + 1);
  k1 = min(grid.gridSize[2] - 1, (i32)floorf((hi.z + b - grid.domainOrigin[2]) / dx) + 1);
}

// Pass 1: activate every block touched by a triangle's narrowband.
__global__ void registerCellsKernel(const TriFeat* tris, i32 nTris,
                                    SingleLevelSparseGrid& grid) {
  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    i32 i0, i1, j0, j1, k0, k1;
    cellRange(T, grid, grid.band, i0, i1, j0, j1, k0, k1);
    for (i32 k = k0; k <= k1; ++k)
      for (i32 j = j0; j <= j1; ++j)
        for (i32 i = i0; i <= i1; ++i) {
          int region;
          float3 q = closestPtTriangle(grid.getCellPos(i, j, k),
                                       T.v0, T.v1, T.v2, region);
          if (norm(grid.getCellPos(i, j, k) - q) <= grid.band)
            grid.activateBlock(i / blockSize, j / blockSize, k / blockSize);
        }
  }
}

// Pass 2: compute the signed distance for *every* cell of an active block and
// keep the closest over all triangles.
//
// A block was activated (pass 1) because one of its cells lies within `band` of
// the surface, so every cell of an active block is within `band + blockDiag` of
// the surface (blockDiag = the block's cell-centre diagonal). Growing each
// triangle's reach by that radius therefore guarantees every active-block cell
// is visited by its true nearest triangle, so the band cutoff can be dropped and
// the whole active block is filled exactly. The hash lookup is done first so the
// expensive closest-point test runs only for cells that actually land in an
// active block (the grown range also sweeps cells in inactive blocks).
__global__ void computeSdfKernel(const TriFeat* tris, i32 nTris,
                                 SingleLevelSparseGrid& grid) {
  const float blockDiag = blockSize * grid.dx * 1.7320508f;  // sqrt(3)
  const float grow = grid.band + blockDiag;
  for (i32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nTris;
       t += gridDim.x * blockDim.x) {
    TriFeat T = tris[t];
    i32 i0, i1, j0, j1, k0, k1;
    cellRange(T, grid, grow, i0, i1, j0, j1, k0, k1);
    for (i32 k = k0; k <= k1; ++k)
      for (i32 j = j0; j <= j1; ++j)
        for (i32 i = i0; i <= i1; ++i) {
          i32 bIdx = grid.hashTable.getValue(
              grid.encodeBlock(i / blockSize, j / blockSize, k / blockSize));
          if (bIdx == bEmpty) continue;       // cell not in an active block
          float3 p = grid.getCellPos(i, j, k);
          int region;
          float3 q = closestPtTriangle(p, T.v0, T.v1, T.v2, region);
          float3 d = p - q;
          float dist = norm(d);
          float s = (dot(d, selectPN(T, region)) >= 0.0f) ? 1.0f : -1.0f;
          i32 cell = grid.cellIndex(bIdx, i % blockSize, j % blockSize, k % blockSize);
          atomicMinMag(&grid.sdf[cell], s * dist);
        }
  }
}

// Initialise every cell of every active block to the far-field sentinel.
__global__ void initSdfKernel(SingleLevelSparseGrid& grid) {
  i64 n = (i64)grid.nBlocks * blockCells;
  for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (i64)gridDim.x * blockDim.x) {
    grid.sdf[i] = SDF_FAR;
  }
}

#endif
