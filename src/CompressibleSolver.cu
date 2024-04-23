#include "CompressibleSolver.cuh"
#include "CompressibleSolverKernels.cuh"

void CompressibleSolver::sortFieldArray(void) {
  //sortFieldArrayKernel<<<nBlocks*blockSizeTot/cudaBlockSize+1, cudaBlockSize>>>(*this);
}
