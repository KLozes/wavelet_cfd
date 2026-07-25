#ifndef SIGNED_DISTANCE_SOLVER_H
#define SIGNED_DISTANCE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"
#include "Features.h"   // TriFeat (per-triangle geometry + pseudonormals)

//
// Multilevel narrowband signed distance field.
//
// Following Roosing, Strickson & Nikiforakis (CiCP 2019), work is parallelised
// over surface triangles.  A full coarse grid (level 0) is filled by brute force
// for the far field, then a narrowband is refined toward the surface level by
// level (each level adds a bandCells-wide shell of finer blocks).  The blocks
// live in the MultiLevelSparseGrid hash table across nLvls levels (coarse far,
// fine at the surface), so storage scales with the surface area rather than the
// domain volume, while every cell still carries a real signed distance.
//
// storage: the signed distance is kept in a dedicated fp32 array, `Sdf`, owned by
// the solver -- not in the base float fieldData (the solver requests zero base
// fields).  The exact signed distance is stored per cell (no quantization).  All
// blocks are activated (coarse grid + refinement) before any SDF is computed, so
// the field is filled once, after the final sort.
static constexpr i32 nSdfFields = 0;

// sentinel for a cell the distance sweep never reached (no active block touches
// it).  Its magnitude exceeds any real distance in the domain, so the min-
// magnitude atomic always overwrites it with a real value.
static constexpr real SDF_FAR = 1e30f;

class SignedDistanceSolver : public MultiLevelSparseGrid {
public:

  real band;            // narrowband half-width (world units)

  // world-space position of the grid origin (the grid itself runs 0..domainSize;
  // mesh coordinates are shifted by -domainOrigin on upload, so this is added
  // back when writing world-space VTK output).
  real domainOrigin[3] = {0, 0, 0};

  TriFeat *dTris;       // device array of per-triangle features
  i32      nTris;

  real    *Sdf;         // signed distance, fp32 (exact, per cell)

  SignedDistanceSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nSdfFields, /*lean=*/true) {
      band = 0;
      dTris = nullptr;
      nTris = 0;
      cudaMallocManaged(&Sdf, (size_t)nBlocksMax*blockSizeTot*sizeof(real));
  }

  ~SignedDistanceSolver(void) { cudaFree(Sdf); }

  void initialize(void);              // build the multilevel narrowband: coarse grid -> refine -> fill
  void sortFieldData(void);           // required MultiLevelSparseGrid override

  void writeHtg(const char *fileName);       // the octree as a vtkHyperTreeGrid (.htg)
  void writeSlices(const char *prefix);                       // 3 orthogonal mid-plane cross-section PNGs
  void writeSlicePNG(const char *fileName, i32 axis, i32 sliceIdx); // one axis-aligned slice
};

#endif
