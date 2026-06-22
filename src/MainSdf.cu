// Narrowband signed distance field generator.
//
// Reads a triangulated surface (STL) and computes a narrowband signed distance
// field on a single-level sparse grid: blocks are stored only where the surface
// passes within `band`, so memory scales with the surface area rather than the
// domain volume (Roosing, Strickson & Nikiforakis, CiCP 2019).  Cells outside
// the band have no block and read as the +band far field on output.
//
//   usage:  ./wavesdf [file.stl] [res] [band_cells] [margin]
//
//     file.stl     input mesh (default: assets/wing.stl)
//     res          cells along the longest bounding-box axis (default 128)
//     band_cells   narrowband half-width in cells           (default 5)
//     margin       empty domain padding on each side, as a fraction of each
//                  axis's mesh extent (default 0.5 -> domain 2x the mesh per
//                  axis).  The VTK output is sparse (active cells only), so it is
//                  unaffected by the domain size; only the 2D slice PNG grows.

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "Stl.h"
#include "Features.h"
#include "SignedDistanceSolver.cuh"

static std::string baseName(const std::string &path) {
  size_t s = path.find_last_of("/\\");
  std::string f = (s == std::string::npos) ? path : path.substr(s + 1);
  size_t d = f.find_last_of('.');
  return (d == std::string::npos) ? f : f.substr(0, d);
}

int main(int argc, char *argv[]) {
  std::string stlPath = (argc > 1) ? argv[1] : "";
  i32   res       = (argc > 2) ? std::atoi(argv[2]) : 128;
  float bandCells = (argc > 3) ? std::atof(argv[3]) : 5.0f;
  float margin    = (argc > 4) ? std::atof(argv[4]) : 0.5f;

  // ---- read mesh ---------------------------------------------------------
  std::vector<StlTri> tris;
  if (stlPath.empty()) {
    const char *candidates[] = {"assets/wing.stl", "../assets/wing.stl"};
    for (const char *c : candidates)
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

  // ---- features (CPU) ----------------------------------------------------
  std::vector<TriFeat> feats;
  int nVerts, nEdges;
  float3 bmin, bmax;
  buildFeatures(tris, feats, nVerts, nEdges, bmin, bmax);
  printf("features: %d unique vertices, %d unique edges\n", nVerts, nEdges);

  // ---- grid geometry: single level, `res` cells on the longest axis ------
  float3 ext = bmax - bmin;
  float maxExt = fmaxf(ext.x, fmaxf(ext.y, ext.z));
  real  dx   = maxExt / float(res);
  real  band = bandCells * dx;

  real  bminArr[3] = {bmin.x, bmin.y, bmin.z};
  real  extArr[3]  = {ext.x, ext.y, ext.z};
  real  padArr[3], origin[3], domainSize[3];
  i32   baseGridSize[3];
  for (i32 d = 0; d < 3; d++) {
    // empty padding on each side: a fraction of THIS axis's extent (so the box
    // scales uniformly instead of a thin axis ballooning), but never less than
    // the band + one cell (mesh stays strictly interior and the band fits).
    padArr[d] = fmaxf(margin * extArr[d], band + dx);
    origin[d] = bminArr[d] - padArr[d];
    i32 nB = (i32)ceilf((extArr[d] + 2*padArr[d]) / dx / blockSize);
    baseGridSize[d] = blockSize * nB;
    domainSize[d]   = baseGridSize[d] * dx;
  }
  printf("bbox: [%.4g %.4g %.4g] .. [%.4g %.4g %.4g]\n",
         bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
  printf("grid: %dx%dx%d cells  dx=%.4g  band=%.4g (%.1f cells)  margin=%.2f\n",
         baseGridSize[0], baseGridSize[1], baseGridSize[2], dx, band, bandCells, margin);
  printf("      domain %.4g x %.4g x %.4g\n",
         domainSize[0], domainSize[1], domainSize[2]);

  // ---- shift mesh into the grid frame (grid runs 0..domainSize) ----------
  float3 shift = make_float3(padArr[0] - bmin.x, padArr[1] - bmin.y, padArr[2] - bmin.z);
  for (auto &f : feats) { f.v0 += shift; f.v1 += shift; f.v2 += shift; }

  TriFeat *dTris = nullptr;
  i32 nTris = (i32)feats.size();
  cudaMalloc(&dTris, nTris * sizeof(TriFeat));
  cudaMemcpy(dTris, feats.data(), nTris * sizeof(TriFeat), cudaMemcpyHostToDevice);

  // ---- build the narrowband SDF ------------------------------------------
  SignedDistanceSolver *solver = new SignedDistanceSolver(domainSize, baseGridSize, 1);
  solver->band  = band;
  solver->dTris = dTris;
  solver->nTris = nTris;
  for (i32 d = 0; d < 3; d++) solver->domainOrigin[d] = origin[d];

  auto wall0 = std::chrono::steady_clock::now();
  solver->initialize();
  auto wall1 = std::chrono::steady_clock::now();
  double wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();

  // ---- report: blocks + compression --------------------------------------
  i64 nGridBlocks = (i64)(baseGridSize[0]/blockSize)*(baseGridSize[1]/blockSize)
                  * (baseGridSize[2]/blockSize);
  printf("---- build: %d blocks in %.1f ms ----\n", solver->hashTable.nKeys, wallMs);
  printf("  blocks: %d activated / %lld in full grid (%.2f%%)\n",
         solver->hashTable.nKeys, (long long)nGridBlocks,
         100.0*real(solver->hashTable.nKeys)/real(nGridBlocks));

  // sdf range over active (reached) cells
  real vmin = 1e30f, vmax = -1e30f; i64 nBandCells = 0, nActive = 0;
  for (i32 b = 0; b < solver->hashTable.nKeys; b++) {
    if (solver->bLocList[b] == kEmpty) continue;
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 cIdx = b*blockSizeTot + c;
      if (solver->cFlagsList[cIdx] != ACTIVE) continue;
      real v = solver->Sdf[cIdx] * solver->sdfQuantum; nActive++;
      vmin = fminf(vmin, v); vmax = fmaxf(vmax, v);
      if (fabsf(v) < band) nBandCells++;
    }
  }
  // compression of the stored narrowband (active cells) vs a dense field over
  // (a) the full padded domain and (b) just the mesh's axis-aligned bbox.
  i64 nDomainCells = (i64)baseGridSize[0]*baseGridSize[1]*baseGridSize[2];
  i64 nAabbCells   = (i64)ceilf(ext.x/dx) * (i64)ceilf(ext.y/dx) * (i64)ceilf(ext.z/dx);
  printf("  active cells: %lld (%lld within band)   sdf range [%.4g, %.4g]\n",
         (long long)nActive, (long long)nBandCells, vmin, vmax);
  printf("  compression (cells stored vs dense):\n");
  printf("    full domain: %lld / %lld = %.2f%%  (%.1fx)\n",
         (long long)nActive, (long long)nDomainCells,
         100.0*real(nActive)/real(nDomainCells), real(nDomainCells)/real(nActive));
  printf("    mesh AABB:   %lld / %lld = %.2f%%  (%.1fx)\n",
         (long long)nActive, (long long)nAabbCells,
         100.0*real(nActive)/real(nAabbCells), real(nAabbCells)/real(nActive));

  // ---- output ------------------------------------------------------------
  mkdir("output", 0755);
  std::string name = baseName(stlPath);

  // compressed ImageData output over the narrowband bbox: implicit geometry
  // (no per-voxel points/connectivity) + zlib-compressed int16 scalar, so the
  // file is a small fraction of the old unstructured-grid output.
  std::string vtk = "output/" + name + "_sdf.vti";
  solver->writeVTK(vtk.c_str());
  printf("wrote %s\n", vtk.c_str());

  std::string slicePrefix = "output/" + name;
  solver->writeSlices(slicePrefix.c_str());   // orthogonal x-y / x-z / y-z cross sections
  printf("wrote %s_{xy,xz,yz}.png\n", slicePrefix.c_str());

  cudaDeviceSynchronize();
  delete solver;
  cudaFree(dTris);
  cudaDeviceReset();
  return 0;
}
