#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

void MultiLevelSparseGrid::sortBlocks(void) {
  thrust::sort_by_key(thrust::device, locList, locList+nBlocks, idxList);
  sortfieldArray();
  updateBlockIndices<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
}

/*
__device__ void MultiLevelSparseGrid::activateBlock(u32 lvl, u32 i, u32 j, u32 k) {
  u64 loc = mortonEncode(lvl, i, j);
  u32 idx = hashInsert(loc);
  locList[idx] = loc;
  idxList[idx] = idx;
}

__device__ void MultiLevelSparseGrid::deactivateBlock(u32 lvl, u32 i, u32 j, u32 k) {
  u64 loc = mortonEncode(lvl, i, j);
  u32 idx = hashDelete(loc);
  locList[idx] = kEmpty;
  idxList[idx] = bEmpty;
}

*/

__device__ void MultiLevelSparseGrid::activateBlock(u32 lvl, u32 i, u32 j, u32 k) {
  // get the base grid block
  uint prnt = getBaseIdx(i>>lvl, j>>lvl, k>>lvl);

  // search up the tree and activate blocks if they do not exist
  for(uint l = 1; l < lvl+1; l++) {
    uint lvlb = l;
    uint ib = i >> (lvl-l);
    uint jb = j >> (lvl-l);
    uint kb = k >> (lvl-l);
    uint64_t locb = mortonEncode(lvlb, ib, jb, kb);

    // get the pointer the child index
    uint *chld =  &(chldArrayList[prnt](ib&1, jb&1, kb&1));

    // swap in a temp index if it is empty
    uint prev = atomicCAS(chld, bEmpty, 0);

    // wait until temp index changes to a real index
    while(*chld == 0) {
      // if the previous value of the atomicCAS was empty,
      // increment the nBlocks counter create the child block
      if (prev == bEmpty) {
        uint idx = atomicAdd(&nBlocks, 1);
        idxList[idx] = idx;
        locList[idx] = locb;
        prntList[idx] = prnt;
        chldArrayList[idx] = bEmpty;

        *chld = idx; //atomicCAS(chld, 0, idx);
      }
    }
    prnt = *chld;
  }
}

__device__ void MultiLevelSparseGrid::deactivateBlock(uint idx) {
  // only deactivate a block if all its children are empty
  // incomplete
  if (chldArrayList[idx] == bEmpty) {
    locList[idx] = 0;
    idxList[idx] = bEmpty;
    prntList[idx] = bEmpty;
  }
}

__host__ __device__ u32 & MultiLevelSparseGrid::getBaseIdx(u32 i, u32 j, u32 k) {
  return baseIdxArray[i+j*baseGridSizeB[0] + k*baseGridSizeB[1]*baseGridSizeB[0]];
}

// seperate bits from a given integer 3 positions apart
__host__ __device__ u64 MultiLevelSparseGrid::split(u32 a) {
  u64 x = (u64)a & ((1<<20)-1); // we only look at the first 20 bits
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

// encode ijk indices and resolution level into morton code
__host__ __device__ u64 MultiLevelSparseGrid::mortonEncode(u64 lvl, u32 i, u32 j) {
  u64 morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1;
  return morton;
}

// encode ijk indices and resolution level into morton code
__host__ __device__ u64 MultiLevelSparseGrid::mortonEncode(u64 lvl, u32 i, u32 j, u32 k) {
  u64 morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1 | split(k) << 2;
  return morton;
}

// compact separated bits into into an integer
__host__ __device__ u32 MultiLevelSparseGrid::compact(u64 w) {
  w &=                  0x1249249249249249;
  w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
  w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
  w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
  w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
  w = (w ^ (w >> 32)) & 0x00000000001fffff;
  return (u32)w;
}

// decode morton code into ijk idx and resolution level
__host__ __device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j) {
  lvl = u32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
}

// decode morton code into ijk idx and resolution level
__host__ __device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j, u32 &k) {
  lvl = u32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
  k = compact(morton >> 2);
}

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1, 1>>>(*this);
}
*/
