// Narrowband signed distance field generator.
//
// Reads a triangulated surface (STL) and computes a narrowband signed distance
// field on a uniform 3D Cartesian grid using the GPU, following the
// Characteristic/Scan-Conversion approach of Roosing, Strickson & Nikiforakis
// (CiCP 26(3), 2019), see docs/NarrowBandGpuSDF.md.
//
//   usage:  ./narrowbandSDF [file.stl] [resolution] [band_cells]
//
//     file.stl     input mesh (default: assets/wing.stl)
//     resolution   cells along the longest bounding-box axis (default 128)
//     band_cells   narrowband half-width in cells           (default 5)
//
// Output is a legacy VTK STRUCTURED_POINTS file (<name>_sdf.vtk) holding the
// signed distance at each cell centre; cells outside the band are set to +band.

#include <sys/stat.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "Stl.h"
#include "Features.h"
#include "SingleLevelSparseGrid.cuh"
#include "SdfKernel.cuh"

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::string baseName(const std::string& path) {
  size_t s = path.find_last_of("/\\");
  std::string f = (s == std::string::npos) ? path : path.substr(s + 1);
  size_t d = f.find_last_of('.');
  return (d == std::string::npos) ? f : f.substr(0, d);
}

static void writeFloatBE(std::ofstream& os, float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  uint8_t b[4] = {uint8_t(u >> 24), uint8_t(u >> 16), uint8_t(u >> 8), uint8_t(u)};
  os.write(reinterpret_cast<char*>(b), 4);
}

// Build a dense field from the sparse grid and write a legacy VTK file.
static void writeVTK(const std::string& path, SingleLevelSparseGrid* grid) {
  const i32 nx = grid->gridSize[0], ny = grid->gridSize[1], nz = grid->gridSize[2];
  const size_t n = (size_t)nx * ny * nz;

  std::vector<float> dense(n, grid->band);   // far field = +band
  for (i32 b = 0; b < grid->nBlocks; ++b) {
    i32 ib, jb, kb;
    grid->decodeBlock(grid->cLocList[b], ib, jb, kb);
    for (i32 lk = 0; lk < blockSize; ++lk)
      for (i32 lj = 0; lj < blockSize; ++lj)
        for (i32 li = 0; li < blockSize; ++li) {
          i32 i = ib * blockSize + li, j = jb * blockSize + lj, k = kb * blockSize + lk;
          if (!grid->isInterior(i, j, k)) continue;
          float v = grid->sdf[grid->cellIndex(b, li, lj, lk)];
          if (v < 0.5f * SDF_FAR)         // skip only the unfilled sentinel
            dense[(size_t)k * nx * ny + (size_t)j * nx + i] = v;
        }
  }

  std::ofstream os(path, std::ios::binary);
  os << "# vtk DataFile Version 3.0\n";
  os << "narrowband signed distance field\n";
  os << "BINARY\n";
  os << "DATASET STRUCTURED_POINTS\n";
  os << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
  os << "ORIGIN " << grid->domainOrigin[0] + 0.5f * grid->dx << " "
                  << grid->domainOrigin[1] + 0.5f * grid->dx << " "
                  << grid->domainOrigin[2] + 0.5f * grid->dx << "\n";
  os << "SPACING " << grid->dx << " " << grid->dx << " " << grid->dx << "\n";
  os << "POINT_DATA " << n << "\n";
  os << "SCALARS sdf float 1\n";
  os << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) writeFloatBE(os, dense[i]);
}

int main(int argc, char* argv[]) {
  // ---- arguments ---------------------------------------------------------
  std::string stlPath = (argc > 1) ? argv[1] : "";
  int res       = (argc > 2) ? std::atoi(argv[2]) : 128;
  float bandCells = (argc > 3) ? std::atof(argv[3]) : 5.0f;

  std::vector<StlTri> tris;
  if (stlPath.empty()) {
    const char* candidates[] = {"assets/wing.stl", "../assets/wing.stl"};
    for (const char* c : candidates)
      if (readStl(c, tris)) { stlPath = c; break; }
  } else {
    readStl(stlPath, tris);
  }
  if (tris.empty()) {
    fprintf(stderr, "error: could not read an STL mesh (tried '%s')\n",
            stlPath.empty() ? "assets/wing.stl" : stlPath.c_str());
    return 1;
  }
  printf("mesh: %s  (%zu triangles)\n", stlPath.c_str(), tris.size());

  // ---- build geometric features (CPU) ------------------------------------
  double t0 = now_ms();
  std::vector<TriFeat> feats;
  int nVerts, nEdges;
  float3 bmin, bmax;
  buildFeatures(tris, feats, nVerts, nEdges, bmin, bmax);
  double tFeat = now_ms() - t0;
  printf("features: %d unique vertices, %d unique edges  (%.1f ms)\n",
         nVerts, nEdges, tFeat);

  // ---- grid geometry -----------------------------------------------------
  float3 ext = bmax - bmin;
  float maxExt = fmaxf(ext.x, fmaxf(ext.y, ext.z));
  float dx = maxExt / float(res);
  float band = bandCells * dx;
  float pad = band + dx;
  float origin[3] = {bmin.x - pad, bmin.y - pad, bmin.z - pad};
  i32 gridSize[3] = {
      (i32)ceilf((ext.x + 2 * pad) / dx),
      (i32)ceilf((ext.y + 2 * pad) / dx),
      (i32)ceilf((ext.z + 2 * pad) / dx)};
  printf("bbox: [%.4g %.4g %.4g] .. [%.4g %.4g %.4g]\n",
         bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
  printf("grid: %d x %d x %d  dx=%.4g  band=%.4g (%.1f cells)\n",
         gridSize[0], gridSize[1], gridSize[2], dx, band, bandCells);

  // ---- upload features ---------------------------------------------------
  TriFeat* dTris = nullptr;
  i32 nTris = (i32)feats.size();
  cudaMalloc(&dTris, nTris * sizeof(TriFeat));
  cudaMemcpy(dTris, feats.data(), nTris * sizeof(TriFeat), cudaMemcpyHostToDevice);

  SingleLevelSparseGrid* grid = new SingleLevelSparseGrid(origin, gridSize, dx, band);

  // ---- pass 1: activate narrowband blocks --------------------------------
  cudaDeviceSynchronize();
  double t1 = now_ms();
  registerCellsKernel<<<cudaGridSize, cudaBlockSize>>>(dTris, nTris, *grid);
  cudaDeviceSynchronize();
  grid->nBlocks = grid->hashTable.nKeys;
  double tReg = now_ms() - t1;

  if (grid->nBlocks >= blockCapacity) {
    fprintf(stderr, "error: narrowband (%d blocks) exceeds capacity (%d).\n"
                    "       lower the resolution/band or raise nCellsMax in Settings.cuh\n",
            grid->nBlocks, blockCapacity);
    delete grid; cudaFree(dTris);
    return 1;
  }

  // ---- pass 2: compute signed distances ----------------------------------
  initSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*grid);
  cudaDeviceSynchronize();
  double t2 = now_ms();
  computeSdfKernel<<<cudaGridSize, cudaBlockSize>>>(dTris, nTris, *grid);
  cudaDeviceSynchronize();
  double tSdf = now_ms() - t2;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    delete grid; cudaFree(dTris);
    return 1;
  }

  // ---- report ------------------------------------------------------------
  float vmin = 1e30f, vmax = -1e30f;
  i64 nFilled = 0, nBand = 0;
  for (i64 c = 0; c < (i64)grid->nBlocks * blockCells; ++c) {
    float v = grid->sdf[c];
    if (v >= 0.5f * SDF_FAR) continue;     // unfilled sentinel
    vmin = fminf(vmin, v); vmax = fmaxf(vmax, v); ++nFilled;
    if (fabsf(v) <= grid->band) ++nBand;
  }
  double dense = double(gridSize[0]) * gridSize[1] * gridSize[2];
  printf("active: %d blocks (%d cells), %lld filled, %lld narrowband cells (%.2f%% of %.0f)\n",
         grid->nBlocks, grid->nBlocks * blockCells, (long long)nFilled,
         (long long)nBand, 100.0 * nBand / dense, dense);
  printf("sdf range: [%.4g, %.4g]\n", vmin, vmax);
  printf("timing: activate %.1f ms, sdf %.1f ms\n", tReg, tSdf);

  // ---- output (everything under output/, like the compressible solver) ---
  mkdir("output", 0755);
  std::string outPath = "output/" + baseName(stlPath) + "_sdf.vtk";
  writeVTK(outPath, grid);
  printf("wrote %s\n", outPath.c_str());

  // center-plane slice images, like the compressible solver's paint()
  grid->paintSlices("output/" + baseName(stlPath));
  printf("wrote output/%s_{xy,xz,yz}.png\n", baseName(stlPath).c_str());

  delete grid;
  cudaFree(dTris);
  cudaDeviceReset();
  return 0;
}
