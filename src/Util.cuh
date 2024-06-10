#ifndef UTIL_H
#define UTIL_H

#include <limits>
#include <chrono>
#include "Settings.cuh"

typedef int i32;
typedef unsigned int u32;
typedef long long int i64;
typedef unsigned long long int u64;

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

template <class DT = std::chrono::milliseconds, class ClockT = std::chrono::steady_clock>
class Timer
{
  using timep_t = typename ClockT::time_point;
  timep_t _start = ClockT::now(), _end = {};

public:
  void tick() { 
    _end = timep_t{}; 
    _start = ClockT::now(); 
  }
  
  void tock() { _end = ClockT::now(); }
  
  template <class T = DT> 
  auto duration() const { 
    return std::chrono::duration_cast<T>(_end - _start); 
  }
};

struct Vec2 {
  dataType data[2];
  __host__ __device__ Vec2(void){};
  __host__ __device__ Vec2(dataType val0, dataType val1) {
    data[0] = val0;
    data[1] = val1;
  }
  __host__ __device__ dataType& operator[](i32 i) {
    return data[i];
  }
};

struct Vec3 {
  dataType data[3];
  __host__ __device__ Vec3(void){};
  __host__ __device__ Vec3(dataType val0, dataType val1, dataType val2) {
    data[0] = val0;
    data[1] = val1;
    data[2] = val2;
  }
  __host__ __device__ dataType& operator[](i32 i) {
    return data[i];
  }
};

struct Vec4 {
  dataType data[4];
  __host__ __device__ Vec4(void){};
  __host__ __device__ Vec4(dataType val0, dataType val1, dataType val2, dataType val3) {
    data[0] = val0;
    data[1] = val1;
    data[2] = val2;
    data[3] = val3;
  }
  __host__ __device__ dataType& operator[](i32 i) {
    return data[i];
  }

};

__host__ __device__ constexpr u32 log2(u32 n) {
  return ((n<2) ? 1 : 1+log2(n/2));
}

__host__ __device__ constexpr u32 powi(const u32 base, const u32 expnt) {
    return ((expnt<=0) ? 1 : base*powi(base, expnt-1));
}

constexpr bool isPowerOf2(int v) {
    return v && ((v & (v - 1)) == 0);
}

static constexpr u32 log2BlockSize = log2((u32)blockSize);

static constexpr i32 blockSizeTot = powi(blockSize, 2);
static constexpr i32 nBlocksMax = nCellsMax / blockSizeTot;
static constexpr i32 blockHaloSize = blockSize+2*haloSize;
static constexpr i32 blockHaloSizeTot = powi(blockHaloSize, 2);
static constexpr i32 nBlocksPerCudaBlock = cudaBlockSize/blockSizeTot;
static constexpr i32 hashTableSize = powi(2, log2((u32)nBlocksMax)+1); // closest power of 2
static constexpr i32 bEmpty = nBlocksMax-1;
static constexpr u64 kEmpty = std::numeric_limits<u64>::max();

#endif
