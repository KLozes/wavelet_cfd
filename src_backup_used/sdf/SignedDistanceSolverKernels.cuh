#ifndef SIGNED_DISTANCE_SOLVER_KERNELS_H
#define SIGNED_DISTANCE_SOLVER_KERNELS_H

#include "SignedDistanceSolver.cuh"

// fill every active cell with the far-field sentinel (SDF_FAR)
__global__ void initSdfKernel(SignedDistanceSolver &grid);

// refine: activate every level-`lvl` block with a cell within `band` of the
// surface (triangle-parallel over the band-grown AABB)
__global__ void registerCellsSdfKernel(SignedDistanceSolver &grid, i32 lvl, real band);

// fine levels (>=1): exact signed distance for every cell of an active level-
// `lvl` block (triangle-parallel over the AABB grown by band + block diagonal,
// atomic min-magnitude)
__global__ void computeSdfKernel(SignedDistanceSolver &grid, i32 lvl, real band);

// level-0 coarse full grid: exact signed distance for every interior cell by
// brute force over all triangles -- this is the real far field
__global__ void computeSdfCoarseKernel(SignedDistanceSolver &grid);

#endif
