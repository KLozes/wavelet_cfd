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
  i32 *valueList;

  HashTable(void);
  ~HashTable(void);

  void reset(void);

  __device__ u64 hash(u64 x);
  __device__ i32 insert(u64 key);
  __device__ i32 insertValue(u64 key, i32 value);
  __device__ i32 getValue(u64 key);
  __device__ i32 setValue(u64 key, i32 value);

};
#endif
