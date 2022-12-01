#ifndef UTIL_H
#define UTIL_H

#include "Settings.cuh"

typedef int i32;
typedef unsigned int u32;
typedef long long int i64;
typedef unsigned long long int u64;

__host__ __device__ constexpr u32 log2(u32 n) {
  return ((n<2) ? 1 : 1+log2(n/2));
}

__host__ __device__ constexpr u32 powi(const u32 base, const u32 expnt) {
    return ((expnt<=0) ? 1 : base*powi(base, expnt-1));
}

constexpr bool isPowerOf2(int v) {
    return v && ((v & (v - 1)) == 0);
}

static constexpr u32 log2BlockSize = log2(blockSize);

// a simple square array data structure
template<typename T, u32 size>
struct Array {
  T data[powi(size, nDim)];

  inline __host__ __device__ T& operator()(const u32 i=0, const u32 j=0, const u32 k=0) {
    return data[size*size*k + size*j + i];
  }

  inline __host__ __device__ T& operator[](const u32 i) {
    return data[i];
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

  __host__ __device__ bool operator==(T rhs) {
    bool result = true;
    for(int i=0; i<powi(size, nDim); i++) {
      result = result && (data[i]&rhs==rhs);
    }
    return result;
  }
};

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }
  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

typedef Array<dataType, blockSize> fieldArray;
typedef Array<u32, blockSize+2*haloSize> nbrArray;
typedef Array<u32, 2> chldArray;

static constexpr u32 blockSizeTot = powi(blockSize, nDim);
static constexpr u32 nBlocksPerCudaBlock = cudaBlockSize/blockSizeTot;
static constexpr u32 bEmpty = nBlocksMax-1;
static constexpr u64 kEmpty = 0xFFFFFFFFFFFFFFFF;
static constexpr u32 nBlocksMaxPow2 = powi(2, log2(nBlocksMax)+1);


#endif
