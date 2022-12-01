#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <thrust/sort.h>
#include <algorithm>
#include <thrust/execution_policy.h>

#include "Settings.cuh"
#include "Util.cuh"

/*
** A multilevel sparse grid data structure
*/
//struct Index {
//  u32 ind[nDim];
//};

class HashTable : public Managed {
public:
  i32 nKeys;

  u64 *keyList; // hash keys are block morton codes
  u32 *valueList; // hash values are block memory indices

  HashTable(void) {
    cudaMallocManaged(&keyList, nBlocksMaxPow2*sizeof(u64));
    cudaMallocManaged(&valueList, nBlocksMaxPow2*sizeof(u32));
    cudaMemset(&keyList, 1, nBlocksMaxPow2*sizeof(u64));
    cudaMemset(&valueList, 0, nBlocksMaxPow2*sizeof(u32));

    // initialize the hashtable keys and values to bEmpty!
    for(u32 idx = 0; idx < nBlocksMax; idx++) {
      keyList[idx] = kEmpty;
      valueList[idx] = bEmpty;
    }
  }

  ~HashTable(void)
  {
    cudaDeviceSynchronize();
    cudaFree(keyList);
    cudaFree(valueList);
  }

  __device__ u64 hash(u64 x);
  __device__ u32 hashInsert(u64 key);
  __device__ u32 hashDelete(u64 key);
  __device__ u32 hashGetValue(u64 key);
  __device__ u32 hashSetValue(u64 key, u32 value);

};
#endif
