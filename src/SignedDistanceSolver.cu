#include <cstdint>
#include <cstring>
#include <climits>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <zlib.h>
#include <png++/png.hpp>

#include "SignedDistanceSolver.cuh"
#include "SignedDistanceSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

//
// Build the single-level narrowband SDF:
//   1. register the blocks whose cells fall within `band` of the surface,
//   2. sort the blocks and build the hash / cell indices,
//   3. fill the exact signed distance for every cell of those blocks,
//   4. flag the filled cells ACTIVE for the slice image / report.
//
void SignedDistanceSolver::initialize(void) {
  // int16 quantization scale: the largest magnitude an active-block cell can
  // store is the pass-2 reach radius, band + block diagonal (see computeSdfKernel).
  // Map that onto the int16 range so every level is usable and the sentinel
  // (INT16_MIN) stays out of band; 32767 (= INT16_MAX) is the largest real code.
  real dx = domainSize[0] / real(baseGridSize[0]);
  real sdfMax = band + blockSize * dx * 1.7320508f;   // band + block diag
  sdfQuantum    = sdfMax / 32767.0f;
  sdfInvQuantum = 1.0f / sdfQuantum;

  registerBlocks();
  nBlocks = hashTable.nKeys;
  sortBlocks();
  computeSdf();
  flagBandCellsActiveSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  printf("narrowband: %d blocks\n", hashTable.nKeys);
  if (hashTable.nDropped > 0) {
    printf("ERROR: block capacity (%d = nCellsMax/%d) exceeded (%d blocks dropped).\n"
           "       result is under-refined -- lower res / band_cells, or raise\n"
           "       nCellsMax in src/Settings.cuh and rebuild.\n",
           nBlocksMax, blockSizeTot, hashTable.nDropped);
  }
}

void SignedDistanceSolver::registerBlocks(void) {
  registerCellsSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

void SignedDistanceSolver::computeSdf(void) {
  initSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  computeSdfKernel<<<cudaGridSize, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

// Required MultiLevelSparseGrid override, but a no-op here: sortBlocks runs
// before computeSdf, which (re)fills every cell by hash lookup, so there is no
// field data worth carrying across the block reorder.
void SignedDistanceSolver::sortFieldData(void) {}

// ---------------------------------------------------------------------------
//  output: compressed VTK XML ImageData (.vti) over the narrowband bbox
// ---------------------------------------------------------------------------
//
// The narrowband data lives in regular blockSize^3 bricks, so the grid geometry
// is fully regular: there is no need to store per-voxel points or hexahedron
// connectivity (which dominated the old UNSTRUCTURED_GRID output at ~89% of the
// bytes).  We instead emit a single vtkImageData whose extent is the bounding
// box of the active blocks: geometry is implicit (origin + spacing + extent) and
// only the per-cell scalar is stored.  Cells outside the narrowband are blanked
// with a `vtkGhostType` HIDDENCELL mask so contour-at-0 sees no spurious shell
// where the blank fill would otherwise cross zero.  Both arrays are zlib block-
// compressed in VTK's appended-data format, so the constant blank fill (the bbox
// is mostly empty for thin shells) collapses to almost nothing.

// VTK appended DataArray: zlib-compress `nbytes` of `data` as 32 KiB blocks and
// return the on-disk payload (UInt64 header [nBlocks, blockSize, lastPartial,
// compSize_0..] followed by the concatenated compressed blocks).
static std::vector<uint8_t> vtkCompress(const uint8_t *data, size_t nbytes) {
  const uint64_t blockSize = 32768;
  uint64_t nFull   = nbytes / blockSize;
  uint64_t partial = nbytes % blockSize;
  uint64_t nBlocks = nFull + (partial ? 1 : 0);
  if (nBlocks == 0) nBlocks = 1;                 // empty array: one empty block

  std::vector<uint64_t> compSize;
  std::vector<uint8_t>  payload;
  for (uint64_t b = 0; b < nBlocks; ++b) {
    uLong srcLen = (b < nFull) ? (uLong)blockSize : (uLong)partial;
    uLongf bound = compressBound(srcLen);
    std::vector<uint8_t> tmp(bound ? bound : 1);
    uLongf destLen = (uLongf)tmp.size();
    compress2(tmp.data(), &destLen, data + b*blockSize, srcLen, Z_DEFAULT_COMPRESSION);
    compSize.push_back(destLen);
    payload.insert(payload.end(), tmp.begin(), tmp.begin() + destLen);
  }

  std::vector<uint8_t> out;
  auto put64 = [&](uint64_t v) {                 // little-endian UInt64
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t(v >> (8*i)));
  };
  put64(nBlocks); put64(blockSize); put64(partial);
  for (uint64_t c : compSize) put64(c);
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

void SignedDistanceSolver::writeVTK(const char *fileName) {
  cudaDeviceSynchronize();

  real dx = domainSize[0] / real(baseGridSize[0]);

  // bounding box of the active blocks, in global cell indices
  i32 imin = INT_MAX, jmin = INT_MAX, kmin = INT_MAX;
  i32 imax = INT_MIN, jmax = INT_MIN, kmax = INT_MIN;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    imin = min(imin, ib*blockSize); imax = max(imax, (ib+1)*blockSize);
    jmin = min(jmin, jb*blockSize); jmax = max(jmax, (jb+1)*blockSize);
    kmin = min(kmin, kb*blockSize); kmax = max(kmax, (kb+1)*blockSize);
  }
  if (imax < imin) { imin = jmin = kmin = 0; imax = jmax = kmax = 0; }  // no blocks

  i32 nx = imax - imin, ny = jmax - jmin, nz = kmax - kmin;   // cells per axis
  size_t nCells = (size_t)nx * ny * nz;

  // dense cell arrays over the bbox: scalar (blank = +far) + hidden-cell mask
  const i16     BLANK = 32767;                   // far positive: no zero crossing
  const uint8_t HIDDENCELL = 32;                 // vtkDataSetAttributes::HIDDENCELL
  std::vector<i16>     sdf(nCells, BLANK);
  std::vector<uint8_t> ghost(nCells, HIDDENCELL);

  size_t nActive = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    for (i32 c = 0; c < blockSizeTot; c++) {
      i16 v = Sdf[b*blockSizeTot + c];
      if (v == SDF_FAR) continue;                // unreached cell: leave blanked
      i32 i = ib*blockSize + (c % blockSize)              - imin;
      i32 j = jb*blockSize + ((c / blockSize) % blockSize) - jmin;
      i32 k = kb*blockSize + (c / blockSize / blockSize)   - kmin;
      size_t idx = (size_t)k*ny*nx + (size_t)j*nx + i;
      sdf[idx]   = v;
      ghost[idx] = 0;                            // visible
      nActive++;
    }
  }

  // zlib-compress both arrays; their appended-data offsets are sequential
  std::vector<uint8_t> sdfBlob   = vtkCompress((const uint8_t*)sdf.data(),   nCells*sizeof(i16));
  std::vector<uint8_t> ghostBlob = vtkCompress(ghost.data(),                 nCells);

  std::ofstream os(fileName, std::ios::binary);
  os << "<?xml version=\"1.0\"?>\n";
  os << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\""
        " header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">\n";
  // world distance = sdf * sdfQuantum (recorded for downstream rescaling)
  os << "  <ImageData WholeExtent=\"" << imin << " " << imax << " "
     << jmin << " " << jmax << " " << kmin << " " << kmax << "\""
     << " Origin=\"" << domainOrigin[0] << " " << domainOrigin[1] << " " << domainOrigin[2] << "\""
     << " Spacing=\"" << dx << " " << dx << " " << dx << "\">\n";
  os << "    <FieldData>\n";
  os << "      <DataArray type=\"Float64\" Name=\"sdf_quantum\" NumberOfTuples=\"1\""
        " format=\"ascii\">" << sdfQuantum << "</DataArray>\n";
  os << "    </FieldData>\n";
  os << "    <Piece Extent=\"" << imin << " " << imax << " "
     << jmin << " " << jmax << " " << kmin << " " << kmax << "\">\n";
  os << "      <CellData Scalars=\"sdf\">\n";
  os << "        <DataArray type=\"Int16\" Name=\"sdf\" format=\"appended\" offset=\"0\"/>\n";
  os << "        <DataArray type=\"UInt8\" Name=\"vtkGhostType\" format=\"appended\""
        " offset=\"" << sdfBlob.size() << "\"/>\n";
  os << "      </CellData>\n";
  os << "    </Piece>\n";
  os << "  </ImageData>\n";
  os << "  <AppendedData encoding=\"raw\">\n_";
  os.write((const char*)sdfBlob.data(),   sdfBlob.size());
  os.write((const char*)ghostBlob.data(), ghostBlob.size());
  os << "\n  </AppendedData>\n";
  os << "</VTKFile>\n";

  printf("  vti: %zu active cells in %dx%dx%d bbox (%.2f MB raw -> %.2f MB compressed)\n",
         nActive, nx, ny, nz,
         (nCells*3)/1e6, (sdfBlob.size()+ghostBlob.size())/1e6);
}

// ---------------------------------------------------------------------------
//  output: orthogonal cross-section PNGs (quick look, no python required)
// ---------------------------------------------------------------------------

//
// Gather one axis-aligned cross section into a dense 2D image and write it as a
// normalized 16-bit grayscale PNG.  `axis` selects the constant axis (0=x ->
// y-z plane, 1=y -> x-z plane, 2=z -> x-y plane) and `sliceIdx` the cell layer.
// Cells outside the band have no block and read as the +band far field.
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
    for (i32 c = 0; c < blockSizeTot; c++) {
      i16 code = Sdf[b*blockSizeTot + c];
      if (code == SDF_FAR) continue;
      real v = code * sdfQuantum;
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
