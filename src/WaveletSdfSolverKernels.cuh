#ifndef WAVELET_SDF_SOLVER_KERNELS_H
#define WAVELET_SDF_SOLVER_KERNELS_H

#include "WaveletSdfSolver.cuh"

// set the ENTIRE Sdf array to the sentinel WSDF_FAR (all slots, incl. not-yet-
// activated blocks), so a freshly activated cell reads as unfilled exactly once
__global__ void initWaveletSdfKernel(WaveletSdfSolver &grid);

// sample the oracle once at every active interior cell still holding the WSDF_FAR
// sentinel (i.e. just-activated cells), storing (value, gradient) at the cell lo corner
__global__ void fillNodesKernel(WaveletSdfSolver &grid);

// adaptive refinement at level `lvl`: split a level-`lvl` block into its 8 children
// where the tricubic-Hermite interpolant of the 1-jet mispredicts an on-surface point (true
// SDF = 0) by more than thresh.  Two kernels cover the surface points separately so
// no work repeats: the welded VERTICES (once each) and the per-triangle FACE CENTERS.
__global__ void flagRefineVertsKernel(WaveletSdfSolver &grid, i32 lvl);
__global__ void flagRefineCentersKernel(WaveletSdfSolver &grid, i32 lvl);

// sign-consistency refinement: split a near-surface leaf cell whose 8 corners are
// all one sign but whose sub-nodes (would-be child corners) reveal a sign flip the
// corners miss -- the grazed-face case the dual contour can't otherwise stitch.
__global__ void flagRefineSignFlipKernel(WaveletSdfSolver &grid, i32 lvl);

// one 2:1-balance (grading) pass: for every refined block, split any face-neighbor
// region that is covered by a block more than one level coarser.  Run to a
// fixpoint (no new blocks).
__global__ void gradeKernel(WaveletSdfSolver &grid);

#endif
