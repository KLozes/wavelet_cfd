#include <cstdint>
#include <cstring>
#include <climits>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <hdf5.h>
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
//  output: single ImageData in a single VTKHDF (.vtkhdf) file
// ---------------------------------------------------------------------------
//
// VTKHDF is the modern single-file HDF5 container that ParaView/VTK read
// natively.  This emits one vtkImageData over the active-block bounding box --
// the HDF5 analogue of the .vti above.  We deliberately do NOT use a composite
// type (MultiBlock banded at the hand-overlapped ghost ring; OverlappingAMR
// loaded but ParaView 6.1's representation failed to render it): a single
// uniform grid is the most-supported, most-robust VTKHDF type, renders as one
// piece, and contours seamlessly.  The narrowband cells carry their true signed
// distance; the far field is sign-filled (a boundary flood fill saturates cells
// outside the surface to +INT16_MAX and the enclosed interior to -INT16_MIN), so
// the whole grid is a valid clamped SDF -- contour-at-0 hits only the surface,
// no ghost mask needed.  The dense bbox is chunked + gzipped, so the saturated
// far field collapses on disk.
//
// Layout:
//   /VTKHDF                Type="ImageData", WholeExtent/Origin/Spacing/Direction
//     /CellData/sdf        Int16 [nz,ny,nx]  (chunked + gzip; band = true dist,
//                          far field = +-INT16 saturated by inside/outside sign)
//     /PointData /FieldData  (empty; present so the reader's probes stay quiet)

// scalar string attribute (Type / Scalars)
static void h5StrAttr(hid_t loc, const char *name, const char *val) {
  hid_t t = H5Tcopy(H5T_C_S1);
  H5Tset_size(t, strlen(val) + 1);
  H5Tset_strpad(t, H5T_STR_NULLTERM);
  hid_t s = H5Screate(H5S_SCALAR);
  hid_t a = H5Acreate2(loc, name, t, s, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, t, val);
  H5Aclose(a); H5Sclose(s); H5Tclose(t);
}

// rank-1 integer attribute, stored as Int64LE (VTK's convention)
static void h5IntAttr(hid_t loc, const char *name, const int *val, hsize_t n) {
  hid_t s = H5Screate_simple(1, &n, NULL);
  hid_t a = H5Acreate2(loc, name, H5T_STD_I64LE, s, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, H5T_NATIVE_INT, val);
  H5Aclose(a); H5Sclose(s);
}

// rank-1 double attribute (Origin / Spacing / Direction)
static void h5DblAttr(hid_t loc, const char *name, const double *val, hsize_t n) {
  hid_t s = H5Screate_simple(1, &n, NULL);
  hid_t a = H5Acreate2(loc, name, H5T_IEEE_F64LE, s, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, H5T_NATIVE_DOUBLE, val);
  H5Aclose(a); H5Sclose(s);
}

void SignedDistanceSolver::writeVTKHDF(const char *fileName) {
  cudaDeviceSynchronize();

  const real dx       = domainSize[0] / real(baseGridSize[0]);
  const i16  INTERIOR = -32768;   // INT16_MIN: far inside  (also the unfilled sentinel)
  const i16  EXTERIOR =  32767;   // INT16_MAX: far outside

  // bounding box of the active blocks in global cell indices (same as the .vti)
  i32 imin=INT_MAX, jmin=INT_MAX, kmin=INT_MAX, imax=INT_MIN, jmax=INT_MIN, kmax=INT_MIN;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    imin = min(imin, ib*blockSize); imax = max(imax, (ib+1)*blockSize);
    jmin = min(jmin, jb*blockSize); jmax = max(jmax, (jb+1)*blockSize);
    kmin = min(kmin, kb*blockSize); kmax = max(kmax, (kb+1)*blockSize);
  }
  if (imax < imin) { imin = jmin = kmin = 0; imax = jmax = kmax = 0; }
  i32 nx = imax-imin, ny = jmax-jmin, nz = kmax-kmin;     // cells per axis
  size_t nCell = (size_t)nx*ny*nz;

  // dense cell field over the bbox.  Active-block cells get their true signed
  // distance; every other cell (object interior + bbox corners) starts at the
  // INTERIOR sentinel and is signed by the flood fill below.
  std::vector<i16> sdf(nCell, INTERIOR);
  size_t nActive = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    for (i32 c = 0; c < blockSizeTot; c++) {
      i16 v = Sdf[(size_t)b*blockSizeTot + c]; if (v == SDF_FAR) continue;
      i32 i = ib*blockSize + (c % blockSize)               - imin;
      i32 j = jb*blockSize + ((c / blockSize) % blockSize) - jmin;
      i32 k = kb*blockSize + (c / blockSize / blockSize)   - kmin;
      sdf[((size_t)k*ny + j)*nx + i] = v; nActive++;
    }
  }

  // sign the far field: flood the unfilled cells (still == INTERIOR) inward from
  // the bbox boundary, tagging everything reachable as EXTERIOR (+max).  The
  // active band is a multi-cell-thick shell of real values, so a 6-connected
  // flood cannot cross the surface; cells it never reaches are enclosed and stay
  // INTERIOR (-min).  Real distances are clamped to [-32767,32767], so the -32768
  // sentinel is unambiguous.  (Assumes a closed surface, as the SDF sign already
  // does.)
  if (nCell) {
    std::vector<size_t> stack;
    auto fill = [&](size_t idx){ if (sdf[idx] == INTERIOR) { sdf[idx] = EXTERIOR; stack.push_back(idx); } };
    for (i32 k=0;k<nz;k++) for (i32 j=0;j<ny;j++) { fill(((size_t)k*ny+j)*nx+0); fill(((size_t)k*ny+j)*nx+(nx-1)); }
    for (i32 k=0;k<nz;k++) for (i32 i=0;i<nx;i++) { fill(((size_t)k*ny+0)*nx+i); fill(((size_t)k*ny+(ny-1))*nx+i); }
    for (i32 j=0;j<ny;j++) for (i32 i=0;i<nx;i++) { fill(((size_t)0*ny+j)*nx+i); fill(((size_t)(nz-1)*ny+j)*nx+i); }
    while (!stack.empty()) {
      size_t idx = stack.back(); stack.pop_back();
      i32 i = idx % nx, j = (idx / nx) % ny, k = idx / nx / ny;
      if (i>0)    fill(idx-1);          if (i<nx-1) fill(idx+1);
      if (j>0)    fill(idx-nx);         if (j<ny-1) fill(idx+nx);
      if (k>0)    fill(idx-(size_t)nx*ny); if (k<nz-1) fill(idx+(size_t)nx*ny);
    }
  }

  // ---- single ImageData --------------------------------------------------
  hid_t file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t root = H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  int ver[2] = {2, 2};                       // current VTKHDF version
  h5IntAttr(root, "Version", ver, 2);
  h5StrAttr(root, "Type", "ImageData");
  int wext[6] = {0, nx, 0, ny, 0, nz};                     // extent starts at 0; Origin places it
  h5IntAttr(root, "WholeExtent", wext, 6);
  double origin[3]  = {domainOrigin[0] + imin*dx, domainOrigin[1] + jmin*dx, domainOrigin[2] + kmin*dx};
  double spacing[3] = {dx, dx, dx};
  double dir[9]     = {1,0,0, 0,1,0, 0,0,1};
  h5DblAttr(root, "Origin", origin, 3);
  h5DblAttr(root, "Spacing", spacing, 3);
  h5DblAttr(root, "Direction", dir, 9);
  double quantum = sdfQuantum;
  h5DblAttr(root, "sdf_quantum", &quantum, 1);              // world distance per int16 step
  // empty PointData/FieldData so the reader's probes don't spew HDF5-DIAG errors
  H5Gclose(H5Gcreate2(root, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  H5Gclose(H5Gcreate2(root, "FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  { // CellData/sdf (int16), [nz,ny,nx], chunked + gzip.  No ghost mask: the far
    // field is sign-filled, so the whole grid is a valid clamped SDF.
    hid_t cd = H5Gcreate2(root, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5StrAttr(cd, "Scalars", "sdf");
    hsize_t dims[3]  = {(hsize_t)nz, (hsize_t)ny, (hsize_t)nx};
    hsize_t chunk[3] = {(hsize_t)(nz<32 ? (nz?nz:1) : 32),
                        (hsize_t)(ny<32 ? (ny?ny:1) : 32),
                        (hsize_t)(nx<32 ? (nx?nx:1) : 32)};
    hid_t s    = H5Screate_simple(3, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (nCell) { H5Pset_chunk(dcpl, 3, chunk); H5Pset_deflate(dcpl, 6); }
    hid_t ds = H5Dcreate2(cd, "sdf", H5T_STD_I16LE, s, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (nCell) H5Dwrite(ds, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sdf.data());
    H5Dclose(ds);
    H5Pclose(dcpl); H5Sclose(s); H5Gclose(cd);
  }

  H5Gclose(root); H5Fclose(file);

  std::ifstream fsz(fileName, std::ios::binary | std::ios::ate);
  double fileMB = fsz ? double(fsz.tellg())/1e6 : 0.0;
  printf("  vtkhdf: ImageData %dx%dx%d bbox, %zu band cells + sign-filled far field (%.2f MB raw -> %.2f MB file)\n",
         nx, ny, nz, nActive, (nCell*2)/1e6, fileMB);
}

// ---------------------------------------------------------------------------
//  output: single-level OverlappingAMR in a VTKHDF (.vtkhdf) file  (experimental)
// ---------------------------------------------------------------------------
//
// Sparser alternative to the single ImageData above: every active block becomes
// one AMR box (a row of the flat AMRBox table), so only the narrowband is stored
// -- no dense bbox.  The file is correct -- bare VTK reads it as vtkOverlappingAMR
// and extracts geometry / contours across every box -- but VTK cannot itself
// *write* AMR to VTKHDF yet, and ParaView 6.1's AMR Surface representation fails
// to render the raw AMR ("UpdateInformation invoked during another request",
// only a few boxes draw).  Workaround in ParaView: set the AMR source to the
// Outline representation and apply a filter (Contour, or Merge Blocks) -- the
// filter output renders fine.  Cells an active block never reached (SDF_FAR) are
// HIDDENCELL-masked.
//
// Layout:
//   /VTKHDF              Type="OverlappingAMR", Origin=[ox,oy,oz]
//     /Level0           Spacing=[dx,dx,dx]
//       AMRBox          Int32 [nBox,6]  imin,imax,jmin,jmax,kmin,kmax (inclusive)
//       /CellData/sdf            Int16 [nCell]  concatenated box-by-box (i fastest)
//       /CellData/vtkGhostType   UInt8 [nCell]  (only when some cells are SDF_FAR)
void SignedDistanceSolver::writeVTKHDFAmr(const char *fileName) {
  cudaDeviceSynchronize();

  const real    dx       = domainSize[0] / real(baseGridSize[0]);
  const i32     bs       = blockSize;
  const i16     BLANK      = 32767;          // far positive: no zero crossing
  const uint8_t HIDDENCELL = 32;             // vtkDataSetAttributes::HIDDENCELL

  // one AMR box (6 ints) + blockSizeTot cells per active block
  std::vector<int>     amrBox;               // nBox * 6 (imin,imax,jmin,jmax,kmin,kmax)
  std::vector<i16>     sdf;                   // nCell, concatenated box-by-box
  std::vector<uint8_t> ghost;                 // nCell
  amrBox.reserve((size_t)hashTable.nKeys * 6);
  sdf.reserve((size_t)hashTable.nKeys * blockSizeTot);
  ghost.reserve((size_t)hashTable.nKeys * blockSizeTot);

  size_t nActive = 0;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    amrBox.push_back(ib*bs); amrBox.push_back(ib*bs + bs-1);   // inclusive cell bounds
    amrBox.push_back(jb*bs); amrBox.push_back(jb*bs + bs-1);
    amrBox.push_back(kb*bs); amrBox.push_back(kb*bs + bs-1);
    for (i32 c = 0; c < blockSizeTot; c++) {                   // c is i-fastest
      i16 v = Sdf[(size_t)b*blockSizeTot + c];
      if (v == SDF_FAR) { sdf.push_back(BLANK); ghost.push_back(HIDDENCELL); }
      else              { sdf.push_back(v);     ghost.push_back(0); nActive++; }
    }
  }
  size_t nBox = amrBox.size() / 6, nCell = sdf.size();

  hid_t file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t root = H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  int ver[2] = {2, 2};
  h5IntAttr(root, "Version", ver, 2);
  h5StrAttr(root, "Type", "OverlappingAMR");
  double origin[3] = {domainOrigin[0], domainOrigin[1], domainOrigin[2]};
  h5DblAttr(root, "Origin", origin, 3);
  double quantum = sdfQuantum;
  h5DblAttr(root, "sdf_quantum", &quantum, 1);
  H5Gclose(H5Gcreate2(root, "FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  hid_t lvl0 = H5Gcreate2(root, "Level0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  double spacing[3] = {dx, dx, dx};
  h5DblAttr(lvl0, "Spacing", spacing, 3);
  H5Gclose(H5Gcreate2(lvl0, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  H5Gclose(H5Gcreate2(lvl0, "FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  { // AMRBox table [nBox,6]
    hsize_t d[2] = {(hsize_t)nBox, 6};
    hid_t s  = H5Screate_simple(2, d, NULL);
    hid_t ds = H5Dcreate2(lvl0, "AMRBox", H5T_STD_I32LE, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (nBox) H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, amrBox.data());
    H5Dclose(ds); H5Sclose(s);
  }

  { // CellData: sdf + vtkGhostType, 1D concatenated, chunked + gzip
    hid_t cd = H5Gcreate2(lvl0, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5StrAttr(cd, "Scalars", "sdf");
    hsize_t d[1]     = {(hsize_t)nCell};
    size_t  chunkN   = nCell ? nCell : 1; if (chunkN > (1u<<16)) chunkN = 1u<<16;
    hsize_t chunk[1] = {(hsize_t)chunkN};
    hid_t s    = H5Screate_simple(1, d, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (nCell) { H5Pset_chunk(dcpl, 1, chunk); H5Pset_deflate(dcpl, 6); }
    hid_t ds = H5Dcreate2(cd, "sdf", H5T_STD_I16LE, s, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (nCell) H5Dwrite(ds, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sdf.data());
    H5Dclose(ds);
    if (nActive != nCell) {                    // blanking mask only if needed
      ds = H5Dcreate2(cd, "vtkGhostType", H5T_STD_U8LE, s, H5P_DEFAULT, dcpl, H5P_DEFAULT);
      H5Dwrite(ds, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, ghost.data());
      H5Dclose(ds);
    }
    H5Pclose(dcpl); H5Sclose(s); H5Gclose(cd);
  }

  H5Gclose(lvl0); H5Gclose(root); H5Fclose(file);

  std::ifstream fsz(fileName, std::ios::binary | std::ios::ate);
  double fileMB = fsz ? double(fsz.tellg())/1e6 : 0.0;
  printf("  vtkhdf: OverlappingAMR, %zu boxes, %zu active cells (%.2f MB raw -> %.2f MB file)\n",
         nBox, nActive, (nCell*3)/1e6, fileMB);
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
