#include "HashTable.cuh"

__device__ u64 HashTable::hash(u64 x) {
  // murmur hash
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53L;
  x ^= x >> 33;
  return x;
}

__device__ u32 HashTable::hashInsert(u64 key) {
  u32 slot = hash(key) % (nBlocksMax-1);
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
      slot = (slot + 1) % (nBlocksMax-1);
  }
}

__device__ u32 HashTable::hashDelete(u64 key) {
  u32 slot = hash(key) % (nBlocksMax-1);
  while (true) {
      if (keyList[slot] == key) {
        keyList[slot] = kEmpty;
        u32 value = valueList[slot];
        valueList[slot] = bEmpty;
        atomicAdd(&nKeys, -1);
        return value;
      }
      if (keyList[slot] == kEmpty) {
        return bEmpty;
      }
      slot = (slot + 1) % (nBlocksMax - 1);
  }
}

__device__ u32 HashTable::hashGetValue(u64 key) {
  u32 slot = hash(key) % (nBlocksMax-1);
  while (true) {
    if (keyList[slot] == key) {
      return valueList[slot];
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % (nBlocksMax - 1);
  }
}

__device__ u32 HashTable::hashSetValue(u64 key, u32 value) {
  u32 slot = hash(key) % (nBlocksMax-1);
  while (true) {
    if (keyList[slot] == key) {
      u32 v = valueList[slot];
      valueList[slot] = value;
      return v;
    }
    if (keyList[slot] == kEmpty) {
      return bEmpty;
    }
    slot = (slot + 1) % (nBlocksMax - 1);
  }
}
