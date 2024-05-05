#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include "Settings.cuh"
#include "Util.cuh"

/*
** A simple CUDA friendly hashtable data structure
*/
class HashTable : public Managed {
public:
  i32 nKeys;

  u64 *keyList; // hash keys are block morton codes
  u32 *valueList; // hash values are block memory indices

  HashTable(void);
  ~HashTable(void);

  void reset(void);

  __host__ __device__ u64 hash(u64 x);
  __host__ __device__ u32 insert(u64 key);
  __host__ __device__ u32 getValue(u64 key);
  __host__ __device__ u32 setValue(u64 key, u32 value);

};
#endif