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
//   `res` is the FINEST resolution; the solver builds a coarse base grid and
//   refines a narrowband toward the surface up to that resolution (multilevel).
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

  // ---- grid geometry: multilevel.  `res` sets the FINEST resolution; we build a
  // coarse base grid (level 0) over the whole domain and refine toward the surface
  // up to the finest level.  nLvls is auto-picked so the coarse grid is roughly
  // COARSE_CELLS on its long axis.
  float3 ext = bmax - bmin;
  float maxExt = fmaxf(ext.x, fmaxf(ext.y, ext.z));
  real  dxFine = maxExt / float(res);          // finest cell size (target)
  real  band   = bandCells * dxFine;           // narrowband half-width at finest level

  // finest grid extent (cells), same construction as the old single-level path
  real bminArr[3] = {bmin.x, bmin.y, bmin.z};
  real extArr[3]  = {ext.x, ext.y, ext.z};
  real padArr[3], origin[3];
  i32  gridFine[3];
  for (i32 d = 0; d < 3; d++) {
    padArr[d] = fmaxf(margin * extArr[d], band + dxFine);
    origin[d] = bminArr[d] - padArr[d];
    gridFine[d] = blockSize * (i32)ceilf((extArr[d] + 2*padArr[d]) / dxFine / blockSize);
  }

  // pick nLvls so the coarse long axis is ~COARSE_CELLS (level fits in 4 bits)
  const i32 COARSE_CELLS = 16, MAX_LVLS = 10;
  i32 maxFine = gridFine[0];
  if (gridFine[1] > maxFine) maxFine = gridFine[1];
  if (gridFine[2] > maxFine) maxFine = gridFine[2];
  i32 nLvls = 1;
  while (nLvls < MAX_LVLS && (maxFine >> nLvls) >= COARSE_CELLS) nLvls++;
  i32 cf = 1 << (nLvls - 1);                    // coarse->fine refinement factor

  // coarse base grid: round each fine axis up to a multiple of blockSize*cf so it
  // refines evenly; coarse = fine/cf, domain spans the (rounded) fine grid
  i32  baseGridSize[3];
  real domainSize[3];
  for (i32 d = 0; d < 3; d++) {
    i32 unit  = blockSize * cf;
    i32 fineR = ((gridFine[d] + unit - 1) / unit) * unit;
    baseGridSize[d] = fineR / cf;
    domainSize[d]   = fineR * dxFine;
  }
  printf("bbox: [%.4g %.4g %.4g] .. [%.4g %.4g %.4g]\n",
         bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
  printf("multilevel: %d levels  coarse %dx%dx%d -> fine %dx%dx%d  dxFine=%.4g  band=%.4g (%.1f cells)\n",
         nLvls, baseGridSize[0], baseGridSize[1], baseGridSize[2],
         baseGridSize[0]*cf, baseGridSize[1]*cf, baseGridSize[2]*cf, dxFine, band, bandCells);
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
  SignedDistanceSolver *solver = new SignedDistanceSolver(domainSize, baseGridSize, nLvls);
  solver->band  = band;
  solver->dTris = dTris;
  solver->nTris = nTris;
  for (i32 d = 0; d < 3; d++) solver->domainOrigin[d] = origin[d];

  auto wall0 = std::chrono::steady_clock::now();
  solver->initialize();
  auto wall1 = std::chrono::steady_clock::now();
  double wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();

  // ---- report: blocks per level + compression vs a full fine grid --------
  i32 lvlBlocks[MAX_LVLS] = {0};
  i64 nCells = 0;
  real vmin = 1e30f, vmax = -1e30f;
  for (i32 b = 0; b < solver->hashTable.nKeys; b++) {
    u64 loc = solver->bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; solver->decode(loc, lvl, ib, jb, kb);
    i32 sx=(baseGridSize[0]/blockSize)<<lvl, sy=(baseGridSize[1]/blockSize)<<lvl, sz=(baseGridSize[2]/blockSize)<<lvl;
    if (ib<0||jb<0||kb<0||ib>=sx||jb>=sy||kb>=sz) continue;   // skip exterior blocks
    if (lvl < nLvls) lvlBlocks[lvl]++;
    for (i32 c = 0; c < blockSizeTot; c++) {
      real v = solver->Sdf[(size_t)b*blockSizeTot + c];
      if (v == SDF_FAR) continue;
      nCells++;
      vmin = fminf(vmin, v); vmax = fmaxf(vmax, v);
    }
  }
  printf("---- build: %d blocks, %d levels, %.1f ms ----\n",
         solver->hashTable.nKeys, nLvls, wallMs);
  for (i32 l = 0; l < nLvls; l++) printf("    level %d: %d blocks\n", l, lvlBlocks[l]);
  i64 nFineFull = (i64)(baseGridSize[0]*cf)*(baseGridSize[1]*cf)*(baseGridSize[2]*cf);
  printf("  cells stored: %lld / %lld fine-full = %.2f%% (%.1fx)   sdf range [%.4g, %.4g]\n",
         (long long)nCells, (long long)nFineFull,
         100.0*real(nCells)/real(nFineFull), real(nFineFull)/fmaxf(1.0,real(nCells)), vmin, vmax);

  // ---- output ------------------------------------------------------------
  mkdir("output", 0755);
  std::string name = baseName(stlPath);

  // output the octree as a vtkHyperTreeGrid (.htg): a single connected dataset
  // with a first-class ParaView representation -- renders + contours directly.
  std::string htg = "output/" + name + "_sdf.htg";
  solver->writeHtg(htg.c_str());
  printf("wrote %s\n", htg.c_str());

  std::string slicePrefix = "output/" + name;
  solver->writeSlices(slicePrefix.c_str());   // orthogonal x-y / x-z / y-z cross sections
  printf("wrote %s_{xy,xz,yz}.png\n", slicePrefix.c_str());

  cudaDeviceSynchronize();
  delete solver;
  cudaFree(dTris);
  cudaDeviceReset();
  return 0;
}
