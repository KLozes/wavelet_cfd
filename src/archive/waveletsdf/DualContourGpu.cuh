#ifndef DUAL_CONTOUR_GPU_H
#define DUAL_CONTOUR_GPU_H

// GPU dual contouring straight off the solver's stored CORNER data -- no oracle.
// Each finest octree cell already holds the SDF (value, gradient) at its lo corner
// node, so a cell's 8 corners are read directly from the grid; the QEF vertex uses
// the stored corner values (crossing points) and gradients (normals).  One thread
// per cell; quads join the 4 finest cells around each sign-change edge via a small
// cell-key -> vertex-id hash.  Writes legacy VTK PolyData.

#include "WaveletSdfSolver.cuh"

struct DcGpuParams {
  float h[3];           // finest cell size per axis (grid frame)
  float origin[3];      // world origin added to vertices (= domainOrigin)
  u64 *hkeys; i32 *hvals; int hcap;   // open-addressing hash: finest cell key -> vertex id
  unsigned char *vMask; // [maxVerts] per-vertex 8-corner sign bits
  float *vertexArray;   // [3*maxVerts] world xyz per vertex
  i32   *quadArray;     // [4*maxQuads] vertex ids per quad
  int *vertCount; int *quadCount;
  int maxVerts; int maxQuads;
};

// `maxVerts` = an upper estimate of the surface (straddling finest) cell count.
void dualContourGpu(WaveletSdfSolver *solver, const double h[3], const double origin[3],
                    int maxVerts, const char *path);

// Carrera et al. 2026 variant: place the dual vertices from SDF VALUES ONLY (no
// stored gradients), iteratively correcting estimated Hermite data.  Same watertight
// minimal-edge topology as dualContourGpu; only the vertex placement differs.
void carreraDc(WaveletSdfSolver *solver, const double h[3], const double origin[3],
               int maxVerts, const char *path, int outerIters, int innerIters);

#endif
