#include <algorithm>
#include <cassert>
#include <vector>
#include <png++/png.hpp>
#include "SingleLevelSparseGrid.cuh"

SingleLevelSparseGrid::SingleLevelSparseGrid(real *domainOrigin_, i32 *gridSize_,
                                             real dx_, real band_) {
  for (i32 d = 0; d < 3; ++d) {
    domainOrigin[d] = domainOrigin_[d];
    gridSize[d]     = gridSize_[d];
  }
  dx   = dx_;
  band = band_;
  nBlocks = 0;

  // block indices must fit in 21 bits of the location code
  i32 nbMax = std::max({gridSize[0], gridSize[1], gridSize[2]}) / blockSize + 1;
  assert(nbMax < (1 << 21));

  cudaMallocManaged(&cLocList, blockCapacity * sizeof(u64));
  cudaMallocManaged(&sdf,      (size_t)blockCapacity * blockCells * sizeof(real));
  cudaMemset(cLocList, 0, blockCapacity * sizeof(u64));
  cudaMemset(sdf,      0, (size_t)blockCapacity * blockCells * sizeof(real));
  cudaDeviceSynchronize();
}

SingleLevelSparseGrid::~SingleLevelSparseGrid(void) {
  cudaDeviceSynchronize();
  cudaFree(cLocList);
  cudaFree(sdf);
}

__host__ __device__ u64 SingleLevelSparseGrid::encodeBlock(i32 ib, i32 jb, i32 kb) {
  return ((u64)kb << 42) | ((u64)jb << 21) | (u64)ib;
}

__host__ __device__ void SingleLevelSparseGrid::decodeBlock(u64 loc, i32 &ib, i32 &jb, i32 &kb) {
  const u64 m = (1ull << 21) - 1;
  ib = (i32)(loc & m);
  jb = (i32)((loc >> 21) & m);
  kb = (i32)((loc >> 42) & m);
}

__host__ __device__ i32 SingleLevelSparseGrid::cellIndex(i32 bIdx, i32 li, i32 lj, i32 lk) {
  return bIdx * blockCells + li + lj * blockSize + lk * blockSize * blockSize;
}

__host__ __device__ bool SingleLevelSparseGrid::isInterior(i32 i, i32 j, i32 k) {
  return i >= 0 && j >= 0 && k >= 0 &&
         i < gridSize[0] && j < gridSize[1] && k < gridSize[2];
}

__device__ float3 SingleLevelSparseGrid::getCellPos(i32 i, i32 j, i32 k) {
  return make_float3(domainOrigin[0] + (i + 0.5f) * dx,
                     domainOrigin[1] + (j + 0.5f) * dx,
                     domainOrigin[2] + (k + 0.5f) * dx);
}

__device__ void SingleLevelSparseGrid::activateBlock(i32 ib, i32 jb, i32 kb) {
  u64 loc = encodeBlock(ib, jb, kb);
  i32 idx = hashTable.insert(loc);
  if (idx != bEmpty) {
    cLocList[idx] = loc;
  }
}

// Gather one axis-aligned slice of the sparse field into a dense image and write
// it as a normalized 16-bit grayscale PNG (same idea as the compressible
// solver's paint(): rescale [min,max] -> [0,65535]). Cells with no narrowband
// value read as +band (the far-field sentinel).
void SingleLevelSparseGrid::writeSlicePNG(const std::string &path, i32 axis, i32 sliceIdx) {
  cudaDeviceSynchronize();

  i32 w, h;
  if      (axis == 0) { w = gridSize[1]; h = gridSize[2]; }  // YZ plane
  else if (axis == 1) { w = gridSize[0]; h = gridSize[2]; }  // XZ plane
  else                { w = gridSize[0]; h = gridSize[1]; }  // XY plane

  std::vector<real> img((size_t)w * h, band);
  for (i32 b = 0; b < nBlocks; ++b) {
    i32 ib, jb, kb;
    decodeBlock(cLocList[b], ib, jb, kb);
    for (i32 lk = 0; lk < blockSize; ++lk)
      for (i32 lj = 0; lj < blockSize; ++lj)
        for (i32 li = 0; li < blockSize; ++li) {
          i32 i = ib * blockSize + li, j = jb * blockSize + lj, k = kb * blockSize + lk;
          if (!isInterior(i, j, k)) continue;
          i32 a = (axis == 0) ? i : (axis == 1) ? j : k;
          if (a != sliceIdx) continue;
          real v = sdf[cellIndex(b, li, lj, lk)];
          if (v >= 0.5f * SDF_FAR) continue;      // unfilled: leave background
          i32 px = (axis == 0) ? j : i;
          i32 py = (axis == 2) ? j : k;
          img[(size_t)py * w + px] = v;
        }
  }

  real mn = 1e30f, mx = -1e30f;
  for (real v : img) { mn = fminf(mn, v); mx = fmaxf(mx, v); }

  png::image<png::gray_pixel_16> im(w, h);
  for (i32 y = 0; y < h; ++y)
    for (i32 x = 0; x < w; ++x)
      im[h - 1 - y][x] = png::gray_pixel_16(
          (img[(size_t)y * w + x] - mn) / (mx - mn + 1e-16f) * 65535);
  im.write(path);
}

void SingleLevelSparseGrid::paintSlices(const std::string &prefix) {
  writeSlicePNG(prefix + "_yz.png", 0, gridSize[0] / 2);
  writeSlicePNG(prefix + "_xz.png", 1, gridSize[1] / 2);
  writeSlicePNG(prefix + "_xy.png", 2, gridSize[2] / 2);
}
