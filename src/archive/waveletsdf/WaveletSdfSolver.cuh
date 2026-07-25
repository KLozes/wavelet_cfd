#ifndef WAVELET_SDF_SOLVER_H
#define WAVELET_SDF_SOLVER_H

#include "MultiLevelSparseGrid.cuh"
#include "Features.h"   // TriFeat (triangle vertices, reused for the oracle)
#include "Bvh.h"        // BvhNode (host build -> device traversal)

//
// Adaptive surface-fitting signed distance field, built on the BVH oracle of the
// Rust reference ../TensorTrain/rs (meshwave.rs + mesh.rs) over wave3d's cell-
// centered octree.
//
// A point-sampling ORACLE -- a triangle BVH that answers signedDistanceGrad(x) =
// (signed distance, unit gradient) using an exact closest-feature distance signed
// by a fast generalized winding number -- is sampled exactly ONCE per cell, at the
// cell CENTER, when the cell's block is first activated (the value + gradient are
// stored in the grid's own cells).  Refinement is driven by the MESH (the zero
// contour): the triangle vertices and face centers lie on the surface, where the
// true SDF is 0, so a cell is split wherever the tricubic-HERMITE interpolant of
// the 1-jet -- built from the 8 surrounding cell-center (value+gradient) samples
// read straight from the grid via the existing block hashTable, matching value and
// gradient at each -- mispredicts such a mesh point by more than `thresh`
// (the reconstructed zero-contour displacement).  This focuses resolution on the
// surface and never re-queries the oracle for a location already sampled.
// Refinement is block-granular: a level-l block splits into its 8 child blocks
// (every cell -> 8 cells).
//
// storage: the signed distance lives in a dedicated fp32 array `Sdf` owned by the
// solver (the grid requests zero base fields and runs in lean mode), matching the
// narrowband SignedDistanceSolver's per-block budget (1 float/cell).
static constexpr i32 nWaveletSdfFields = 0;

// sentinel for an exterior cell the fill never reaches (never written to output).
static constexpr real WSDF_FAR = 1e30f;

class WaveletSdfSolver : public MultiLevelSparseGrid {
public:

  // world-space position of the grid origin (mesh coords are shifted by
  // -domainOrigin on upload; added back when writing world-space VTK output).
  real domainOrigin[3] = {0, 0, 0};

  // refinement controls (see initialize / flagRefineKernel)
  real thresh;          // max tricubic-Hermite interp error at on-surface mesh points (world units)
  bool grade;           // 2:1-balance the octree (no face-neighbor jump > 1 level)

  // device oracle: triangle BVH (built on the host, uploaded flat) over the
  // welded triangles.  `orient` (+/-1) makes the winding-number sign robust to
  // inward-wound meshes (inside <=> orient * wind > 0.5).
  BvhNode *dNodes;      i32 nNodes;
  i32     *dOrder;
  TriFeat *dTris;       i32 nTris;
  float3  *dVerts;      i32 nVerts;   // welded unique mesh vertices (refinement points)
  real     orient;

  // NODAL SDF storage: each active block owns (blockSize+1)^3 = nodeSizeTot corner
  // node samples, computed by the oracle exactly ONCE (when the block is first
  // activated).  Boundary nodes are duplicated across neighbour blocks, so every
  // cell's 8 corners are LOCAL to its own block -- corner reads need no neighbour
  // lookups.  SDF value ONLY: gradients (for the Hermite interpolant / DC normals)
  // are computed on the fly from these node values by finite differences.
  // Block memory indices stay stable through the build, so these persist.
  real    *Sdf;         // [nBlocksMax * nodeSizeTot] signed distance at block corner nodes

  WaveletSdfSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nWaveletSdfFields, /*lean=*/true) {
      thresh = 0;
      grade = true;
      dNodes = nullptr; nNodes = 0;
      dOrder = nullptr;
      dTris = nullptr;  nTris = 0;
      dVerts = nullptr; nVerts = 0;
      orient = 1.0f;
      cudaMallocManaged(&Sdf, (size_t)nBlocksMax*nodeSizeTot*sizeof(real));
  }

  ~WaveletSdfSolver(void) { cudaFree(Sdf); }

  // local index of node (ni,nj,nk in 0..blockSize) within a block's nodeSizeTot run
  __host__ __device__ static i32 nodeIdx(i32 ni, i32 nj, i32 nk) {
    return ni + nj*blockSizeNode + nk*blockSizeNode*blockSizeNode;
  }

  void initialize(void);              // coarse grid -> adaptive refine -> exact fill
  void sortFieldData(void);           // required MultiLevelSparseGrid override (no-op)

  void writeHtg(const char *fileName);                               // octree as a vtkHyperTreeGrid
  void writeSlices(const char *prefix);                              // 3 orthogonal mid-plane PNGs
  void writeSlicePNG(const char *fileName, i32 axis, i32 sliceIdx);  // one axis-aligned slice
};

#endif
