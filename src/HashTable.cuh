#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include "Settings.cuh"
#include "Util.cuh"

/*
** A simple GPU friendly hashtable data structure
*/
class HashTable : public Managed {
public:
  i32 nKeys;

  u64 *keyList; 
  u32 *valueList;

  HashTable(void);
  ~HashTable(void);

  void reset(void);

  __device__ u64 hash(u64 x);
  __device__ u32 insert(u64 key);
  __device__ u32 insertValue(u64 key, u32 value);
  __device__ u32 getValue(u64 key);
  __device__ u32 setValue(u64 key, u32 value);

};
#endif
