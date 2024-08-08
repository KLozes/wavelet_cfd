#include <thrust/sort.h>
#include <algorithm>
#include "HashTable.cuh"

#include <inttypes.h>

HashTable::HashTable(void) {
  cudaMallocManaged(&keyList, hashTableSize*sizeof(u64));
  cudaMallocManaged(&valueList, hashTableSize*sizeof(i32));
  reset();
}

HashTable::~HashTable(void) {
  cudaDeviceSynchronize();
  cudaFree(keyList);
  cudaFree(valueList);
}

__global__ void resetHashTableKernel(HashTable &hashTable) {
  i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hashTableSize) {
    hashTable.keyList[idx] = kEmpty;
    hashTable.valueList[idx] = bEmpty;
  }
}

void HashTable::reset(void) {
  nKeys = 0;
  resetHashTableKernel<<<hashTableSize/cudaBlockSize+1, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

__device__ u64 HashTable::hash(u64 x) {
  // murmur hash
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53L;
  x ^= x >> 33;
  return x;
}

__device__ i32 HashTable::insert(u64 key) {
  u64 slot = hash(key) % hashTableSize;
  while (true) {

#ifdef __CUDA_ARCH__
    u64 prev = atomicCAS(&keyList[slot], kEmpty, key); // swap in temp key while first thread sets val
#else
    u64 prev = keyList[slot];
#endif

    if (prev == kEmpty) {
#ifdef __CUDA_ARCH__
      i32 value = atomicAdd(&nKeys, 1);
#else
      i32 value = nKeys++;
#endif
      valueList[slot] = value;
      break;
    }

    if (prev == key) {
      break;
    }
    slot = (slot + 1) % hashTableSize;
  }
  return valueList[slot];
}

__device__ i32 HashTable::insertValue(u64 key, i32 value) {
  u64 slot = hash(key) % hashTableSize;
  while (true) {

#ifdef __CUDA_ARCH__
    u64 prev = atomicCAS(&keyList[slot], kEmpty, key); // swap in temp key while first thread sets val
#else
    u64 prev = keyList[slot];
#endif

    if (prev == kEmpty) {
      valueList[slot] = value;
      break;
    }

    if (prev == key) {
      valueList[slot] = value;;
    }
    slot = (slot + 1) % hashTableSize;
  }
  return valueList[slot];
}

__device__ i32 HashTable::getValue(u64 key) {
  i32 slot = hash(key) % hashTableSize;
  while (true) {
    if (keyList[slot] == key) {
      return valueList[slot];
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % hashTableSize;
  }
}

__device__ i32 HashTable::setValue(u64 key, i32 value) {
  i32 slot = hash(key) % hashTableSize;
  while (true) {
    if (keyList[slot] == key) {
      i32 v = valueList[slot];
      valueList[slot] = value;
      return v;
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % hashTableSize;
  }
}
