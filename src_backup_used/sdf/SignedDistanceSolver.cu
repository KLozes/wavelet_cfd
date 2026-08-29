#include <cstdint>
#include <cstring>
#include <climits>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <png++/png.hpp>

#include "SignedDistanceSolver.cuh"
#include "SignedDistanceSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

//
// Build the multilevel narrowband SDF:
//   1. materialize the full coarse base grid (level 0) over the whole domain,
//   2. refine a narrowband toward the surface one level at a time (each level
//      adds a bandCells-wide shell of finer blocks; the shells nest),
//   3. fill the field: brute-force the (small) coarse grid for the real far
//      field, then the narrowband at each finer level (triangle-parallel).
// The result is a graded octree: coarse far from the surface, fine at it, with a
// real signed distance everywhere (no blank / saturated far field).
//
void SignedDistanceSolver::initialize(void) {
  // GPU-synchronized phase timers: refine = register + sort to reach a level,
  // compute = fill that level's distances.
  auto clk = []{ cudaDeviceSynchronize(); return std::chrono::steady_clock::now(); };
  auto ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
  double refineMs[16] = {0}, computeMs[16] = {0};

  // level 0: full coarse grid over the whole domain
  auto t0 = clk();
  initializeBaseGrid();
  refineMs[0] = ms(t0, clk());

  // refine toward the surface, one level at a time.  The band at the child level
  // is bandCells cells wide (= band * 2^(nLvls-1-child) in world units), so the
  // shells nest and each level adds a fixed-cell-width ring of finer blocks.
  for (i32 lvl = 0; lvl < nLvls-1; lvl++) {
    real bandChild = band * real(1 << (nLvls-2 - lvl));    // band at level lvl+1
    auto ta = clk();
    registerCellsSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this, lvl+1, bandChild);
    cudaDeviceSynchronize();
    nBlocks = hashTable.nKeys;
    sortBlocks();
    refineMs[lvl+1] = ms(ta, clk());
  }

  // fill the field: the coarse full grid by brute force (the real far field), then
  // the narrowband at each finer level.  fp32 stores the exact signed distance.
  initSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  auto tc = clk();
  computeSdfCoarseKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  computeMs[0] = ms(tc, clk());
  for (i32 lvl = 1; lvl < nLvls; lvl++) {
    real bandLvl = band * real(1 << (nLvls-1 - lvl));
    auto ta = clk();
    computeSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this, lvl, bandLvl);
    computeMs[lvl] = ms(ta, clk());
  }

  printf("narrowband: %d blocks across %d levels\n", hashTable.nKeys, nLvls);
  for (i32 l = 0; l < nLvls; l++)
    printf("    level %d: refine %6.1f ms   compute %6.1f ms\n", l, refineMs[l], computeMs[l]);
  if (hashTable.nDropped > 0) {
    printf("ERROR: block capacity (%d = nCellsMax/%d) exceeded (%d blocks dropped).\n"
           "       result is under-refined -- lower res / band_cells, or raise\n"
           "       nCellsMax in src/Settings.cuh and rebuild.\n",
           nBlocksMax, blockSizeTot, hashTable.nDropped);
  }
}

// Required MultiLevelSparseGrid override, but a no-op here: sortBlocks runs
// before computeSdf, which (re)fills every cell by hash lookup, so there is no
// field data worth carrying across the block reorder.
void SignedDistanceSolver::sortFieldData(void) {}

// ---------------------------------------------------------------------------
//  output: the octree as a vtkHyperTreeGrid (.htg)
// ---------------------------------------------------------------------------
//
// Our multilevel grid is exactly a branch-2 cell octree (each level doubles the
// resolution, so every cell splits into 8), which maps natively onto a
// vtkHyperTreeGrid: the level-0 coarse cells are the root cells (one tree each),
// and a cell is "refined" iff the finer block covering it exists.  Unlike the
// OverlappingAMR, HTG is a single connected dataset with a mature, first-class
// ParaView representation -- so it renders and contours directly.
//
// .htg format (validated against vtkXMLHyperTreeGridReader): ascii grid + per-
// tree metadata + a raw-appended Float32 `sdf`.  Per tree, cells are listed
// breadth-first; Descriptors carries one refinement bit per non-deepest cell;
// NumberOfVerticesPerDepth gives the cell count at each depth.  NumberOfTuples is
// mandatory on every DataArray (the reader segfaults without it).
void SignedDistanceSolver::writeHtg(const char *fileName) {
  cudaDeviceSynchronize();
  const i32  bs  = blockSize;
  const real dxC = domainSize[0] / real(baseGridSize[0]);        // coarse cell size (uniform)
  const i32  ncx = baseGridSize[0], ncy = baseGridSize[1], ncz = baseGridSize[2];  // coarse cells/axis

  // block location code -> block memory index (value lookups + child existence)
  std::unordered_map<u64,i32> locToBlock;
  locToBlock.reserve((size_t)hashTable.nKeys*2);
  for (i32 b=0;b<hashTable.nKeys;b++) if (bLocList[b]!=kEmpty) locToBlock[bLocList[b]]=b;

  auto value = [&](i32 L,i32 i,i32 j,i32 k)->real {
    i32 b = locToBlock[encode(L, i/bs, j/bs, k/bs)];
    return Sdf[(size_t)b*blockSizeTot + (i%bs) + (j%bs)*bs + (k%bs)*bs*bs];
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
// Gather one axis-aligned cross section into a dense 2D image and write it as a
// normalized 16-bit grayscale PNG.  `axis` selects the constant axis (0=x ->
// y-z plane, 1=y -> x-z plane, 2=z -> x-y plane) and `sliceIdx` the cell layer.
// Rendered from the coarse level-0 full grid (which now holds a real far field),
// so the image is a quick coarse look; the fine levels are not upsampled here.
//
void SignedDistanceSolver::writeSlicePNG(const char *fileName, i32 axis, i32 sliceIdx) {
  cudaDeviceSynchronize();
  i32 nx = baseGridSize[0], ny = baseGridSize[1], nz = baseGridSize[2];

  i32 w = (axis == 0) ? ny : nx;          // image width  (in-plane horizontal)
  i32 h = (axis == 2) ? ny : nz;          // image height  (in-plane vertical)

  std::vector<real> img((size_t)w*h, band);   // background = +band far field
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb;
    decode(loc, lvl, ib, jb, kb);
    if (lvl != 0) continue;     // coarse-level quick look (avoids fine OOB)
    for (i32 c = 0; c < blockSizeTot; c++) {
      real v = Sdf[b*blockSizeTot + c];
      if (v == SDF_FAR) continue;
      i32 i = ib*blockSize + (c % blockSize);
      i32 j = jb*blockSize + ((c / blockSize) % blockSize);
      i32 k = kb*blockSize + (c / blockSize / blockSize);
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
void SignedDistanceSolver::writeSlices(const char *prefix) {
  std::string p(prefix);
  writeSlicePNG((p + "_xy.png").c_str(), 2, baseGridSize[2]/2);   // planform (top)
  writeSlicePNG((p + "_xz.png").c_str(), 1, baseGridSize[1]/2);   // side
  writeSlicePNG((p + "_yz.png").c_str(), 0, baseGridSize[0]/2);   // front
}
