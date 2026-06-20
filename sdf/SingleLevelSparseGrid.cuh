#ifndef SINGLE_LEVEL_SPARSE_GRID_H
#define SINGLE_LEVEL_SPARSE_GRID_H

#include <string>

#include "Settings.cuh"
#include "Util.cuh"
#include "HashTable.cuh"
#include "Vec3f.cuh"

// A block is a blockSize^3 brick of cells, the unit of sparse storage (just
// like MultiLevelSparseGrid). blockCapacity bounds how many blocks may be
// active; the cell value array holds blockCapacity * blockCells values.
static constexpr i32 blockCells    = blockSize * blockSize * blockSize;
static constexpr i32 blockCapacity = nCellsMax / blockCells;

// Sentinel for a cell that has not yet received a distance (far larger than any
// real signed distance in an active block). Shared by the kernel and the
// host-side writers so they can tell "unfilled" from a genuine value that
// happens to lie beyond the narrowband.
static constexpr real SDF_FAR = 1e30f;

/*
** A single-level (uniform) 3D sparse Cartesian grid.
**
** This is the flat counterpart to MultiLevelSparseGrid: there is no refinement
** hierarchy, just one resolution. As in that class, the GPU HashTable maps a
** *block* location code to a compact block index, and each block owns a
** contiguous blockSize^3 run of cells in the value array. Only blocks that
** touch the narrowband are activated, so memory scales with the surface.
*/
class SingleLevelSparseGrid : public Managed {
public:

  HashTable hashTable;

  real domainOrigin[3];   // world-space corner of cell (0,0,0)
  i32  gridSize[3];       // number of cells along x, y, z
  real dx;                // uniform cell size
  real band;              // narrowband half-width (world units)

  i32 nBlocks;            // number of active blocks

  u64  *cLocList;         // location code of each active block  [blockCapacity]
  real *sdf;              // signed distance per cell  [blockCapacity*blockCells]

  SingleLevelSparseGrid(real *domainOrigin_, i32 *gridSize_, real dx_, real band_);
  ~SingleLevelSparseGrid(void);

  // block location code <-> block index triple (21 bits / axis)
  __host__ __device__ u64 encodeBlock(i32 ib, i32 jb, i32 kb);
  __host__ __device__ void decodeBlock(u64 loc, i32 &ib, i32 &jb, i32 &kb);

  // storage index of cell (li,lj,lk) within block bIdx
  __host__ __device__ i32 cellIndex(i32 bIdx, i32 li, i32 lj, i32 lk);

  __host__ __device__ bool isInterior(i32 i, i32 j, i32 k);   // global cell index
  __device__ float3 getCellPos(i32 i, i32 j, i32 k);          // global cell centre

  // insert a block into the hash table and record its location code
  __device__ void activateBlock(i32 ib, i32 jb, i32 kb);

  // export 2D slices of the field as normalized 16-bit grayscale PNGs,
  // like MultiLevelSparseGrid::paint(). axis: 0=x (YZ plane), 1=y (XZ),
  // 2=z (XY); sliceIdx is the cell index along that axis.
  void writeSlicePNG(const std::string &path, i32 axis, i32 sliceIdx);
  void paintSlices(const std::string &prefix);
};

#endif
