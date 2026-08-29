#ifndef SDF_VEC3_CUH
#define SDF_VEC3_CUH

// Minimal host/device float3 vector math used by the narrowband SDF solver.
// Operators are marked __host__ __device__ so the same routines build the
// geometry on the CPU and run the distance transform on the GPU.

#include <cuda_runtime.h>
#include <cmath>

// `real` is the solver precision (Settings.cuh; float unless -DUSE_DOUBLE).  The
// float3 math below is float either way -- the geometry (STL vertices, BVH,
// distances) is fp32 by construction -- but the typedef must agree with
// Settings.cuh so an fp64 solver (e.g. wavefem_dp) can include both.
#include "Settings.cuh"

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator*(float s, float3 a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3& operator+=(float3& a, float3 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; return a;
}

__host__ __device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float norm(float3 a) {
  return sqrtf(dot(a, a));
}
__host__ __device__ inline float3 normalize(float3 a) {
  float n = norm(a);
  return (n > 0.0f) ? a * (1.0f / n) : a;
}

__host__ __device__ inline float3 fmin3(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__host__ __device__ inline float3 fmax3(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

#endif
