#ifndef UTIL_H
#define UTIL_H

#include "Settings.cuh"

__host__ __device__ constexpr uint log2(uint n) {
  return ((n<2) ? 1 : 1+log2(n/2));
}

__host__ __device__ constexpr uint powi(const uint base, const uint expnt) {
    return ((expnt<=0) ? 1 : base*powi(base, expnt-1));
}

constexpr bool isPowerOf2(int v) {
    return v && ((v & (v - 1)) == 0);
}

static constexpr uint log2BlockSize = log2(blockSize);

// a simple square array data structure
template<typename T, uint size>
struct Array {
  T data[powi(size, nDim)];

  inline __host__ __device__ T& operator()(const uint i=0, const uint j=0, const uint k=0) {
    return data[size*size*k + size*j + i];
  }

  __host__ __device__ void operator=(T in) {
    for(int i=0; i<powi(size, nDim); i++) {
      data[i] = in;
    }
  }

  __host__ __device__ void operator=(Array<T, size> &arr) {
    for(int i=0; i<powi(size, nDim); i++) {
      data[i] = arr(i);
    }
  }
};

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

static constexpr uint blockSizeTot = powi(blockSize, nDim);
static constexpr uint nBlocksPerCudaBlock = cudaBlockSize/blockSizeTot;
static constexpr uint bEmpty = nBlocksMax-1;

#endif
