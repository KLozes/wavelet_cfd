// Adaptive interpolating-wavelet signed distance field generator.
//
// Reads a triangulated surface (STL) and computes a signed distance field on a
// cell-centered sparse octree.  A point-sampling ORACLE -- a triangle BVH that
// returns (signed distance, gradient) via an exact closest-feature distance
// signed by a fast winding number -- drives the refinement: starting coarse, a
// cell is subdivided (into 8) where the tricubic-Hermite interpolant of the 1-jet
// (the surrounding cell-center value+gradient samples) mispredicts an on-surface
// mesh point by more than `thresh`.  Storage scales with
// surface features rather than the domain volume.  Ported from the Rust reference
// ../TensorTrain/rs (meshwave.rs + mesh.rs).
//
//   usage:  ./wavewsdf [file.stl] [res] [thresh_cells] [margin] [grade]
//
//     file.stl     input mesh (default: assets/wing.stl)
//     res          FINEST resolution: cells along the longest bbox axis (default 256)
//     thresh_cells max Hermite error at on-surface mesh points, as a multiple of the
//                  finest cell size (default 0.5)
//     margin       empty domain padding per side, as a fraction of mesh extent (0.5)
//     grade        2:1-balance the octree (no face-neighbor level jump > 1); 1=on (default)

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "Stl.h"
#include "Features.h"
#include "Bvh.h"
#include "DualContourGpu.cuh"
#include "WaveletSdfSolver.cuh"

static std::string baseName(const std::string &path) {
  size_t s = path.find_last_of("/\\");
  std::string f = (s == std::string::npos) ? path : path.substr(s + 1);
  size_t d = f.find_last_of('.');
  return (d == std::string::npos) ? f : f.substr(0, d);
}
// host nodal-octree path (NodalOctree.cu): SDF-only, minimal-oracle, GPU sampling, two .vtu files
void runNodalOctree(const std::vector<TriFeat> &feats, const BvhNode *bnodes, int nBvhNodes,
                    const i32 *border, real orient, const double origin[3], const double domainSize[3],
                    const int baseGrid[3], int nLvls, double thresh, const char *name);

int main(int argc, char *argv[]) {
  std::string stlPath = (argc > 1) ? argv[1] : "";
  i32   res        = (argc > 2) ? std::atoi(argv[2]) : 256;
  float threshCells= (argc > 3) ? std::atof(argv[3]) : 0.5f;
  float margin     = (argc > 4) ? std::atof(argv[4]) : 0.5f;
  i32   grade      = (argc > 5) ? std::atoi(argv[5]) : 1;     // 2:1 balance (default on)
  i32   dcMethod   = (argc > 6) ? std::atoi(argv[6]) : 0;     // 0 = gradient DC, 1 = Carrera (SDF-only)
  i32   dcOuter    = (argc > 7) ? std::atoi(argv[7]) : 0;     // Carrera outer iterations (0 = QEF baseline)
  i32   dcInner    = (argc > 8) ? std::atoi(argv[8]) : 0;     // Carrera inner (distance-energy) iterations

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

  // ---- features (CPU): welded triangle vertices + face normals -----------
  std::vector<TriFeat> feats;
  std::vector<float3> verts;          // welded unique vertices (refinement points)
  int nVerts, nEdges;
  float3 bmin, bmax;
  buildFeatures(tris, feats, nVerts, nEdges, bmin, bmax, &verts);
  printf("features: %d unique vertices, %d unique edges\n", nVerts, nEdges);

  // ---- grid geometry: multilevel.  `res` sets the FINEST resolution; a coarse
  // base grid (level 0) covers the whole domain and is refined toward the surface
  // up to the finest level.  nLvls is auto-picked so the coarse grid is ~COARSE
  // cells on its long axis.
  float3 ext = bmax - bmin;
  float maxExt = fmaxf(ext.x, fmaxf(ext.y, ext.z));
  real  dxFine = maxExt / float(res);          // finest cell size (target)

  real bminArr[3] = {bmin.x, bmin.y, bmin.z};
  real extArr[3]  = {ext.x, ext.y, ext.z};
  real padArr[3], origin[3];
  i32  gridFine[3];
  for (i32 d = 0; d < 3; d++) {
    padArr[d] = fmaxf(margin * extArr[d], 4*dxFine);
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
  real thresh = threshCells * dxFine;           // absolute detail threshold (world units)
  printf("bbox: [%.4g %.4g %.4g] .. [%.4g %.4g %.4g]\n",
         bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
  printf("multilevel: %d levels  coarse %dx%dx%d -> fine %dx%dx%d  dxFine=%.4g\n",
         nLvls, baseGridSize[0], baseGridSize[1], baseGridSize[2],
         baseGridSize[0]*cf, baseGridSize[1]*cf, baseGridSize[2]*cf, dxFine);
  printf("      domain %.4g x %.4g x %.4g  thresh=%.4g (%.2f cells)\n",
         domainSize[0], domainSize[1], domainSize[2], thresh, threshCells);

  // ---- shift mesh into the grid frame (grid runs 0..domainSize) ----------
  float3 shift = make_float3(padArr[0] - bmin.x, padArr[1] - bmin.y, padArr[2] - bmin.z);
  for (auto &f : feats) { f.v0 += shift; f.v1 += shift; f.v2 += shift; }
  for (auto &v : verts) v += shift;

  // ---- build the BVH oracle on the shifted triangles, upload to device ----
  auto bvh0 = std::chrono::steady_clock::now();
  Bvh bvh = buildBvh(feats);
  auto bvh1 = std::chrono::steady_clock::now();
  printf("bvh: %zu nodes, orient %+.0f  (%.1f ms)\n", bvh.nodes.size(), bvh.orient,
         std::chrono::duration<double,std::milli>(bvh1-bvh0).count());

  // ---- dcMethod 2: host serial NODAL OCTREE (no GPU grid) -----------------
  if (dcMethod == 2) {
    double originD[3]={origin[0],origin[1],origin[2]};
    double domD[3]={domainSize[0],domainSize[1],domainSize[2]};
    int    bg[3]={baseGridSize[0],baseGridSize[1],baseGridSize[2]};
    auto t0 = std::chrono::steady_clock::now();
    runNodalOctree(feats, bvh.nodes.data(), (int)bvh.nodes.size(), bvh.order.data(), bvh.orient,
                   originD, domD, bg, nLvls, (double)thresh, baseName(stlPath).c_str());
    printf("  nodal total %.1f ms\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
    return 0;
  }

  i32 nTris  = (i32)feats.size();
  i32 nNodes = (i32)bvh.nodes.size();
  i32 nVrt   = (i32)verts.size();
  BvhNode *dNodes = nullptr;  cudaMalloc(&dNodes, nNodes*sizeof(BvhNode));
  i32     *dOrder = nullptr;  cudaMalloc(&dOrder, nTris*sizeof(i32));
  TriFeat *dTris  = nullptr;  cudaMalloc(&dTris,  nTris*sizeof(TriFeat));
  float3  *dVerts = nullptr;  cudaMalloc(&dVerts, nVrt*sizeof(float3));
  cudaMemcpy(dNodes, bvh.nodes.data(), nNodes*sizeof(BvhNode), cudaMemcpyHostToDevice);
  cudaMemcpy(dOrder, bvh.order.data(), nTris*sizeof(i32),      cudaMemcpyHostToDevice);
  cudaMemcpy(dTris,  feats.data(),     nTris*sizeof(TriFeat),  cudaMemcpyHostToDevice);
  cudaMemcpy(dVerts, verts.data(),     nVrt*sizeof(float3),    cudaMemcpyHostToDevice);

  // ---- build the wavelet SDF ---------------------------------------------
  WaveletSdfSolver *solver = new WaveletSdfSolver(domainSize, baseGridSize, nLvls);
  solver->thresh  = thresh;
  solver->grade   = (grade != 0);
  solver->dNodes  = dNodes;  solver->nNodes = nNodes;
  solver->dOrder  = dOrder;
  solver->dTris   = dTris;   solver->nTris  = nTris;
  solver->dVerts  = dVerts;  solver->nVerts = nVrt;
  solver->orient  = bvh.orient;
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
      if (v == WSDF_FAR) continue;
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

  std::string htg = "output/" + name + "_wsdf.htg";
  solver->writeHtg(htg.c_str());
  printf("wrote %s\n", htg.c_str());

  std::string slicePrefix = "output/" + name + "_w";
  solver->writeSlices(slicePrefix.c_str());   // orthogonal x-y / x-z / y-z cross sections
  printf("wrote %s_{xy,xz,yz}.png\n", slicePrefix.c_str());

  // feature-preserving surface mesh via dual contouring of the STORED corner data
  // (GPU, no oracle), on the finest octree cells; written as legacy VTK PolyData.
  double dch[3], dcorigin[3];
  for (i32 d = 0; d < 3; d++) {
    dch[d] = (double)domainSize[d] / (baseGridSize[d]*cf);
    dcorigin[d] = origin[d];
  }
  // estimate the surface (straddling finest) cell count to size the DC vertex/hash.
  double surfArea = 0.0;
  for (const auto &f : feats) surfArea += 0.5 * norm(cross(f.v1 - f.v0, f.v2 - f.v0));
  int dcMaxVerts = (int)fmin(48.0e6, fmax(65536.0, 4.0 * surfArea / (dch[0]*dch[1])));
  std::string dcvtk = "output/" + name + "_dc.vtk";
  auto dc0 = std::chrono::steady_clock::now();
  if (dcMethod == 1)
    carreraDc(solver, dch, dcorigin, dcMaxVerts, dcvtk.c_str(), dcOuter, dcInner);
  else
    dualContourGpu(solver, dch, dcorigin, dcMaxVerts, dcvtk.c_str());
  cudaDeviceSynchronize();
  printf("  dc build %.1f ms\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dc0).count());

  cudaDeviceSynchronize();
  delete solver;
  cudaFree(dNodes);
  cudaFree(dOrder);
  cudaFree(dTris);
  cudaFree(dVerts);
  cudaDeviceReset();
  return 0;
}
