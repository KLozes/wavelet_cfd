#include <cstdint>
#include <cstring>
#include <climits>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <png++/png.hpp>

#include "WaveletSdfSolver.cuh"
#include "WaveletSdfSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

//
// Build the adaptive surface-fitting SDF:
//   1. materialize the full coarse base grid (level 0) over the whole domain,
//   2. refine level by level (triangle-parallel): split a level-l block into its 8
//      children wherever its tricubic-Hermite interpolant of the 1-jet mispredicts an on-surface
//      mesh point (vertex / face center, true SDF = 0) by more than `thresh`,
//   3. 2:1-balance the octree (optional),
//   4. fill: store the exact oracle signed distance at every active cell center.
// The criterion uses only the oracle's corner samples (the mesh points are known-
// zero), so no level needs to be filled before the next is decided -- refinement
// runs to completion first, then a single exact fill.  The result is a graded
// octree: coarse away from the surface, fine on it, real signed distance everywhere.
//
void WaveletSdfSolver::initialize(void) {
  auto clk = []{ cudaDeviceSynchronize(); return std::chrono::steady_clock::now(); };
  auto ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
  double refineMs[16] = {0};

  // mark every cell slot unfilled (sentinel), then build the coarse base grid and
  // fill its cells.  Blocks keep their activation-order memory index for the whole
  // build (no per-level re-sort), so a cell sampled once stays put and is read back
  // straight from the grid.
  initWaveletSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  auto t0 = clk();
  initializeBaseGrid();
  fillNodesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  refineMs[0] = ms(t0, clk());

  // refine toward the surface, one level at a time.  Refinement reads the level's
  // cell-center samples (already filled) via the block hashTable; newly activated
  // child blocks are then filled once (fillNodesKernel skips already-filled
  // cells).  On-surface points are split across two kernels (welded vertices, then
  // face centers) so shared vertices are tested once, not once per triangle.  No
  // sortBlocks: indices stay stable so the stored samples persist.
  for (i32 lvl = 0; lvl < nLvls-1; lvl++) {
    auto ta = clk();
    flagRefineVertsKernel<<<cudaGridSize, cudaBlockSize>>>(*this, lvl);
    flagRefineCentersKernel<<<cudaGridSize, cudaBlockSize>>>(*this, lvl);
    flagRefineSignFlipKernel<<<cudaGridSize, cudaBlockSize>>>(*this, lvl);
    fillNodesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    refineMs[lvl+1] = ms(ta, clk());
  }

  // 2:1-balance (grade) the octree: split any block whose face-neighbor region is
  // more than one level coarser.  Iterate to a fixpoint (each pass ripples a coarse
  // leaf one level finer), then fill the cells the grading added.
  i32 nBeforeGrade = hashTable.nKeys;
  i32 nGradePasses = 0;
  double gradeMs = 0.0;
  if (grade) {
    auto tg = clk();
    i32 prev = -1;
    while (hashTable.nKeys != prev && nGradePasses < 2*nLvls) {
      prev = hashTable.nKeys;
      gradeKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
      cudaDeviceSynchronize();
      nGradePasses++;
    }
    fillNodesKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
    gradeMs = ms(tg, clk());
  }
  cudaDeviceSynchronize();

  printf("wavelet sdf: %d blocks across %d levels (thresh %.4g)\n",
         hashTable.nKeys, nLvls, thresh);
  for (i32 l = 0; l < nLvls; l++)
    printf("    level %d: refine+fill %7.1f ms\n", l, refineMs[l]);
  if (grade)
    printf("    grade: %7.1f ms  (2:1 balance: +%d blocks / %d = +%.0f%% over %d passes)\n",
           gradeMs, hashTable.nKeys - nBeforeGrade, nBeforeGrade,
           100.0 * (hashTable.nKeys - nBeforeGrade) / fmaxf(1, nBeforeGrade), nGradePasses);
  if (hashTable.nDropped > 0) {
    printf("ERROR: block capacity (%d = nCellsMax/%d) exceeded (%d blocks dropped).\n"
           "       result is under-refined -- lower res, raise thresh, or raise\n"
           "       nCellsMax in src/Settings.cuh and rebuild.\n",
           nBlocksMax, blockSizeTot, hashTable.nDropped);
  }
}

// Required MultiLevelSparseGrid override, but a no-op here: sortBlocks runs before
// the fill, which (re)fills every cell by oracle sample, so there is no field data
// worth carrying across the block reorder.
void WaveletSdfSolver::sortFieldData(void) {}

// ---------------------------------------------------------------------------
//  output: the octree as a vtkHyperTreeGrid (.htg)
// ---------------------------------------------------------------------------
//
// Our multilevel grid is a branch-2 cell octree (each level doubles the
// resolution, so every cell splits into 8), which maps natively onto a
// vtkHyperTreeGrid: the level-0 coarse cells are the root cells (one tree each),
// and a cell is "refined" iff the finer block covering it exists.  Identical to
// SignedDistanceSolver::writeHtg -- it only reads the `Sdf` cell field and walks
// the octree, independent of how the field was produced.
//
void WaveletSdfSolver::writeHtg(const char *fileName) {
  cudaDeviceSynchronize();
  const i32  bs  = blockSize;
  const real dxC = domainSize[0] / real(baseGridSize[0]);        // coarse cell size (uniform)
  const i32  ncx = baseGridSize[0], ncy = baseGridSize[1], ncz = baseGridSize[2];  // coarse cells/axis

  // block location code -> block memory index (value lookups + child existence)
  std::unordered_map<u64,i32> locToBlock;
  locToBlock.reserve((size_t)hashTable.nKeys*2);
  for (i32 b=0;b<hashTable.nKeys;b++) if (bLocList[b]!=kEmpty) locToBlock[bLocList[b]]=b;

  // cell value for the HTG = average of the cell's 8 corner nodes (cell-center
  // estimate).  The block owns all 8 corners (5x5x5 nodal storage), so they are read
  // LOCALLY from the cell's own block -- de-shifts the half-cell nodal offset.
  auto value = [&](i32 L,i32 i,i32 j,i32 k)->real {
    auto it = locToBlock.find(encode(L, i/bs, j/bs, k/bs));
    if (it == locToBlock.end()) return 0;
    const real *S = Sdf + (size_t)it->second*nodeSizeTot;
    i32 li=i%bs, lj=j%bs, lk=k%bs;
    real sum = 0; int n = 0;
    for (i32 a=0;a<2;a++) for (i32 b=0;b<2;b++) for (i32 c=0;c<2;c++) {
      real v = S[WaveletSdfSolver::nodeIdx(li+a, lj+b, lk+c)];
      if (v != WSDF_FAR) { sum += v; n++; }
    }
    return n ? sum/n : 0;
  };
  auto refined = [&](i32 L,i32 i,i32 j,i32 k)->bool {
    return L+1 < nLvls &&
           locToBlock.find(encode(L+1, (2*i)/bs, (2*j)/bs, (2*k)/bs)) != locToBlock.end();
  };

  // accumulate global HTG arrays, trees in i-fastest root-id order
  std::vector<real> sdf;           // CellData (all cells, BF per tree)
  std::string      desc;           // Descriptors: ascii refinement bits
  std::vector<i64> nvpd, treeIds;  // NumberOfVerticesPerDepth, TreeIds
  std::vector<i32> depthPerTree;
  size_t nDesc = 0;
  sdf.reserve((size_t)hashTable.nKeys*blockSizeTot);

  struct C { i32 L,i,j,k; bool ref; };
  std::vector<C> bf;               // per-tree breadth-first cells (reused)
  for (i32 ck=0; ck<ncz; ck++)
  for (i32 cj=0; cj<ncy; cj++)
  for (i32 ci=0; ci<ncx; ci++) {
    bf.clear();
    bf.push_back({0,ci,cj,ck,false});
    i32 maxL = 0;
    for (size_t h=0; h<bf.size(); h++) {           // BFS (FIFO via head index)
      C c = bf[h];
      bool r = refined(c.L,c.i,c.j,c.k);
      bf[h].ref = r;
      if (c.L > maxL) maxL = c.L;
      if (r) for (i32 cc=0; cc<8; cc++)
        bf.push_back({c.L+1, 2*c.i+(cc&1), 2*c.j+((cc>>1)&1), 2*c.k+((cc>>2)&1), false});
    }
    treeIds.push_back((i64)ci + (i64)cj*ncx + (i64)ck*ncx*ncy);
    depthPerTree.push_back(maxL+1);
    std::vector<i64> cnt(maxL+1, 0);
    for (auto &c : bf) cnt[c.L]++;
    for (i32 d=0; d<=maxL; d++) nvpd.push_back(cnt[d]);
    for (auto &c : bf) if (c.L < maxL) { desc += c.ref ? "1 " : "0 "; nDesc++; }
    for (auto &c : bf) sdf.push_back(value(c.L,c.i,c.j,c.k));
  }
  size_t nCells = sdf.size();

  std::ofstream os(fileName, std::ios::binary);
  os.precision(10);
  os << "<?xml version=\"1.0\"?>\n"
     << "<VTKFile type=\"HyperTreeGrid\" version=\"2.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
     << "  <HyperTreeGrid BranchFactor=\"2\" TransposedRootIndexing=\"0\" Dimensions=\""
     << ncx+1 << " " << ncy+1 << " " << ncz+1 << "\">\n    <Grid>\n";
  const char *cn[3] = {"XCoordinates","YCoordinates","ZCoordinates"};
  i32 nc3[3] = {ncx,ncy,ncz};
  for (i32 a=0; a<3; a++) {
    os << "      <DataArray type=\"Float64\" Name=\"" << cn[a] << "\" NumberOfTuples=\""
       << nc3[a]+1 << "\" format=\"ascii\">\n        ";
    for (i32 i=0; i<=nc3[a]; i++) os << (domainOrigin[a] + i*dxC) << " ";
    os << "\n      </DataArray>\n";
  }
  os << "    </Grid>\n    <Trees>\n";
  os << "      <DataArray type=\"Bit\" Name=\"Descriptors\" NumberOfTuples=\"" << nDesc
     << "\" format=\"ascii\">\n        " << desc << "\n      </DataArray>\n";
  auto i64Array = [&](const char *name, std::vector<i64> &v){
    os << "      <DataArray type=\"Int64\" Name=\"" << name << "\" NumberOfTuples=\"" << v.size()
       << "\" format=\"ascii\">\n        ";
    for (i64 x : v) os << x << " ";
    os << "\n      </DataArray>\n";
  };
  i64Array("NumberOfVerticesPerDepth", nvpd);
  i64Array("TreeIds", treeIds);
  os << "      <DataArray type=\"UInt32\" Name=\"DepthPerTree\" NumberOfTuples=\"" << depthPerTree.size()
     << "\" format=\"ascii\">\n        ";
  for (i32 x : depthPerTree) os << x << " ";
  os << "\n      </DataArray>\n    </Trees>\n    <CellData>\n"
     << "      <DataArray type=\"Float32\" Name=\"sdf\" NumberOfTuples=\"" << nCells
     << "\" format=\"appended\" offset=\"0\"/>\n    </CellData>\n  </HyperTreeGrid>\n"
     << "  <AppendedData encoding=\"raw\">\n_";
  uint64_t nbytes = (uint64_t)nCells * sizeof(real);
  os.write((const char*)&nbytes, sizeof(nbytes));
  os.write((const char*)sdf.data(), nbytes);
  os << "\n  </AppendedData>\n</VTKFile>\n";
  os.close();

  std::ifstream fsz(fileName, std::ios::binary | std::ios::ate);
  double mb = fsz ? double(fsz.tellg())/1e6 : 0.0;
  printf("  htg: %zu trees, %zu cells, %zu descriptor bits (%.2f MB)\n",
         treeIds.size(), nCells, nDesc, mb);
}

// ---------------------------------------------------------------------------
//  output: orthogonal cross-section PNGs (quick look, no python required)
// ---------------------------------------------------------------------------
//
// Render one axis-aligned cross section of the coarse level-0 full grid (which
// holds a real far field) into a normalized 16-bit grayscale PNG.  `axis` selects
// the constant axis (0=x -> y-z, 1=y -> x-z, 2=z -> x-y) and `sliceIdx` the cell
// layer.  A quick coarse look; fine levels are not upsampled here.
//
void WaveletSdfSolver::writeSlicePNG(const char *fileName, i32 axis, i32 sliceIdx) {
  cudaDeviceSynchronize();
  i32 nx = baseGridSize[0], ny = baseGridSize[1], nz = baseGridSize[2];

  i32 w = (axis == 0) ? ny : nx;          // image width  (in-plane horizontal)
  i32 h = (axis == 2) ? ny : nz;          // image height  (in-plane vertical)

  std::vector<real> img((size_t)w*h, 0.0f);
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    decode(loc, lvl, ib, jb, kb);
    if (lvl != 0) continue;     // coarse-level quick look (avoids fine OOB)
    for (i32 c = 0; c < blockSizeTot; c++) {
      i32 ci = c % blockSize, cj = (c / blockSize) % blockSize, ck = c / blockSize / blockSize;
      real v = Sdf[(size_t)b*nodeSizeTot + WaveletSdfSolver::nodeIdx(ci, cj, ck)];  // lo-corner node
      if (v == WSDF_FAR) continue;
      i32 i = ib*blockSize + ci;
      i32 j = jb*blockSize + cj;
      i32 k = kb*blockSize + ck;
      i32 a = (axis == 0) ? i : (axis == 1) ? j : k;
      if (a != sliceIdx) continue;
      i32 px = (axis == 0) ? j : i;
      i32 py = (axis == 2) ? j : k;
      img[(size_t)py*w + px] = v;
    }
  }

  real mn = 1e30f, mx = -1e30f;
  for (real v : img) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
  png::image<png::gray_pixel_16> im(w, h);
  for (i32 y = 0; y < h; y++)
    for (i32 x = 0; x < w; x++)
      im[h-1-y][x] = png::gray_pixel_16((img[(size_t)y*w + x] - mn) / (mx - mn + 1e-16f) * 65535);
  im.write(fileName);
}

// the three orthogonal mid-plane cross sections (x-y, x-z, y-z)
void WaveletSdfSolver::writeSlices(const char *prefix) {
  std::string p(prefix);
  writeSlicePNG((p + "_xy.png").c_str(), 2, baseGridSize[2]/2);   // planform (top)
  writeSlicePNG((p + "_xz.png").c_str(), 1, baseGridSize[1]/2);   // side
  writeSlicePNG((p + "_yz.png").c_str(), 0, baseGridSize[0]/2);   // front
}
