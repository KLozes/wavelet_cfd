#include "HashTable.cuh"

HashTable::HashTable(void) {
  cudaMallocManaged(&keyList, nBlocksMaxPow2*sizeof(u64));
  cudaMallocManaged(&valueList, nBlocksMaxPow2*sizeof(u32));
  reset();
}

HashTable::~HashTable(void)
{
  cudaDeviceSynchronize();
  cudaFree(keyList);
  cudaFree(valueList);
}

void HashTable::reset(void) {
  nKeys = 0;
  cudaMemset(&keyList, 0x11111111, nBlocksMaxPow2*sizeof(u64));
  cudaMemset(&valueList, 0, nBlocksMaxPow2*sizeof(u32));
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

__device__ u32 HashTable::insert(u64 key) {
  u32 slot = hash(key) % nBlocksMaxPow2;
  while (true) {
      u64 prev = atomicCAS(&keyList[slot], kEmpty, key);
      if (prev == kEmpty) {
        uint value = atomicAdd(&nKeys, 1);
        valueList[slot] = value;
        return value;
      }
      if (prev == key) {
        return valueList[slot];
      }
      slot = (slot + 1) % nBlocksMaxPow2;
  }
}

__device__ u32 HashTable::getValue(u64 key) {
  u32 slot = hash(key) % nBlocksMaxPow2;
  while (true) {
    if (keyList[slot] == key) {
      return valueList[slot];
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % nBlocksMaxPow2;
  }
}

__device__ u32 HashTable::setValue(u64 key, u32 value) {
  u32 slot = hash(key) % nBlocksMaxPow2;
  while (true) {
    if (keyList[slot] == key) {
      u32 v = valueList[slot];
      valueList[slot] = value;
      return v;
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % nBlocksMaxPow2;
  }
}
