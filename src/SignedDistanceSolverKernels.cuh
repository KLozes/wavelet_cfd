#ifndef SIGNED_DISTANCE_SOLVER_KERNELS_H
#define SIGNED_DISTANCE_SOLVER_KERNELS_H

#include "SignedDistanceSolver.cuh"

// fill every active cell with the far-field sentinel (SDF_FAR)
__global__ void initSdfKernel(SignedDistanceSolver &grid);

// pass 1: activate every block that has a cell within `band` of the surface
// (triangle-parallel over the band-grown AABB)
__global__ void registerCellsSdfKernel(SignedDistanceSolver &grid);

// pass 2: exact signed distance for every cell of an active block (triangle-
// parallel over the AABB grown by band + block diagonal, atomic min-magnitude)
__global__ void computeSdfKernel(SignedDistanceSolver &grid);

// mark every reached cell of an interior block ACTIVE (the narrowband fills a
// whole block, so there is no reconstruction halo / GHOST ring); unreached cells
// stay GHOST so the image / report only count real distances
__global__ void flagBandCellsActiveSdfKernel(SignedDistanceSolver &grid);

#endif
