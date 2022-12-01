#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"

void CompressibleSolver::sortfieldArray(void) {
  sortfieldArrayKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}
