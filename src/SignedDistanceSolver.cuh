#ifndef SIGNED_DISTANCE_SOLVER_H
#define SIGNED_DISTANCE_SOLVER_H

#include "MultiLevelSparseGrid.cuh"
#include "Features.h"   // TriFeat (per-triangle geometry + pseudonormals)

//
// Single-level narrowband signed distance field.
//
// Following Roosing, Strickson & Nikiforakis (CiCP 2019), work is parallelised
// over surface triangles: pass 1 activates the blocks whose cells fall within
// `band` of the surface, pass 2 fills the exact signed distance for every cell
// of those active blocks.  The blocks live in the MultiLevelSparseGrid hash
// table at a single level (nLvls == 1), so storage scales with the surface area
// rather than the domain volume; cells outside the band have no block and read
// as the +band far field on output.
//
// storage: the signed distance is kept in a dedicated signed-int16 array, `Sdf`,
// owned by the solver -- not in the base float fieldData (the solver requests
// zero base fields).  All geometry / distance math is done in fp32; the final
// per-cell distance is quantized to int16 on store as round(d / sdfQuantum),
// which halves the field memory so more narrowband blocks fit.  The quantum is
// chosen (see initialize) so the largest distance an active-block cell can hold
// (band + block diagonal) maps onto the int16 range, giving uniform sub-cell
// precision across the band.  The block sort runs before computeSdf refills
// every cell by hash lookup, so no reorder scratch is needed.
static constexpr i32 nSdfFields = 0;

// sentinel for a cell the distance sweep never reached (no active block touches
// it).  INT16_MIN sits one step below the clamp range [-32767, 32767] used for
// real distances, so it is unambiguous; its magnitude (32768) is the largest
// possible, so the min-magnitude atomic always overwrites it with a real value.
static constexpr i16 SDF_FAR = -32768;

class SignedDistanceSolver : public MultiLevelSparseGrid {
public:

  real band;            // narrowband half-width (world units)

  // world-space position of the grid origin (the grid itself runs 0..domainSize;
  // mesh coordinates are shifted by -domainOrigin on upload, so this is added
  // back when writing world-space VTK output).
  real domainOrigin[3] = {0, 0, 0};

  TriFeat *dTris;       // device array of per-triangle features
  i32      nTris;

  i16     *Sdf;         // signed distance, int16 storage (quantized; computed in fp32)
  real     sdfQuantum;     // world distance per int16 step (set in initialize)
  real     sdfInvQuantum;  // 1 / sdfQuantum (precomputed for the quantize hot path)

  SignedDistanceSolver(real *domainSize_, i32 *baseGridSize_, i32 nLvls_) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, nLvls_, nSdfFields) {
      band = 0;
      dTris = nullptr;
      nTris = 0;
      sdfQuantum = 0;
      sdfInvQuantum = 0;
      cudaMallocManaged(&Sdf, (size_t)nBlocksMax*blockSizeTot*sizeof(i16));
  }

  ~SignedDistanceSolver(void) { cudaFree(Sdf); }

  void initialize(void);              // build the narrowband: register -> sort -> fill
  void registerBlocks(void);          // pass 1: activate blocks within the band
  void computeSdf(void);              // pass 2: exact mesh distance for all band cells
  void sortFieldData(void);           // required MultiLevelSparseGrid override

  void writeVTK(const char *fileName); // sparse active-cell unstructured grid (no dense far field)
  void writeSlices(const char *prefix);                       // 3 orthogonal mid-plane cross-section PNGs
  void writeSlicePNG(const char *fileName, i32 axis, i32 sliceIdx); // one axis-aligned slice
};

#endif
