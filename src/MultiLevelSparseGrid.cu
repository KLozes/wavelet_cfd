#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

/*
void MultiLevelSparseGrid::initGrid(void) {
  // initialize the blocks of the base grid level in cartesian order
  u32 ind = 0;
  for(u32 j=0; j<max(1,baseGridSize[1]/blockSize); j++) {
    for(u32 i=0; i<max(1,baseGridSize[0]/blockSize); i++) {
      blockLocList[ind] = mortonEncode(0, i, j);
      blockList[ind].loc = blockLocList[ind];
      blockList[ind].index = ind;
      blockList[ind].parent = bEmpty;
      blockList[ind].children = bEmpty;
      blockList[ind].neighbors = 0;
      ind++;
    }
  }

  // sort the blocks and then reassign indices
  thrust::sort_by_key(thrust::host, blockLocList, blockLocList+nBlocks, blockList);
  for(ind = 0; ind < nBlocks; ind++) {
    blockList[ind].index = ind;
    u32 lvl, i, j;
    mortonDecode(blockList[ind].loc, lvl, i, j);
    getBaseBlockIndex(i, j) = ind;
  }

  // set the Empty grid block data
  blockList[bEmpty].loc = 0;
  blockList[bEmpty].index = bEmpty;
  blockList[bEmpty].parent = bEmpty;
  blockList[bEmpty].children = bEmpty;
  blockList[bEmpty].neighbors = 0;

  sortBlocks();
}

__device__ u32& MultiLevelSparseGrid::getBaseBlockIndex(int i, int j) {
  i = max(0, min(int(baseGridSize[0]/blockSize)-1, i));
  j = max(0, min(int(baseGridSize[1]/blockSize)-1, j));
  return baseBlockIndexArray[i + baseGridSize[0]/blockSize * j];
}

__device__  u32 MultiLevelSparseGrid::getBlockIndex(u32 lvl, u32 i, u32 j) {
  // get the base grid block
  u32 bIndex = getBaseBlockIndex(i>>lvl, j>>lvl);

  // search up the tree
  for(u32 l = 1; l < lvl+1; l++) {
    u32 ib = (i >> (lvl-l)) & 1;
    u32 jb = (j >> (lvl-l)) & 1;
    bIndex = blockList[bIndex].children(ib, jb);
  }
  return bIndex;
}

__device__ void MultiLevelSparseGrid::activateBlock(u32 lvl, u32 i, u32 j) {
  // get the base grid block
  u32 parent = getBaseBlockIndex(i>>lvl, j>>lvl);

  // search up the tree and activate blocks if they do not exist
  for(u32 l = 1; l < lvl+1; l++) {
    u32 lvlb = l;
    u32 ib = i >> (lvl-l);
    u32 jb = j >> (lvl-l);
    u64 locb = mortonEncode(lvlb, ib, jb);

    // get the pointer the child index
    u32 *child = &(blockList[parent].children(ib&1, jb&1));

    // swap in a temp index if it is empty
    u32 prev = atomicCAS(child, bEmpty, 0);

    // wait until temp index changes to a real index
    while(*child == 0) {
      // if the previous value of the atomicCAS was empty,
      // increment the nBlocks counter create the child block
      if (prev == bEmpty) {
        u32 nBlocksPrev = atomicAdd(&nBlocks, 1);

        blockLocList[nBlocksPrev] = locb;
        blockList[nBlocksPrev].loc = locb;
        blockList[nBlocksPrev].index = nBlocksPrev;
        blockList[nBlocksPrev].parent = parent;
        blockList[nBlocksPrev].children = bEmpty;
        blockList[nBlocksPrev].neighbors = bEmpty;

        atomicCAS(child, 0, nBlocksPrev);
      }
    }
    parent = *child;
  }
}

__device__ void MultiLevelSparseGrid::deactivateBlock(u32 bIndex) {
  blockLocList[bIndex] = 0;
  blockList[bIndex].loc = 0;
  blockList[bIndex].index = bEmpty;
  blockList[bIndex].parent = bEmpty;
  blockList[bIndex].children = bEmpty;
  blockList[bIndex].neighbors = bEmpty;
}

__device__ void MultiLevelSparseGrid::getDijk(u32 n, u32 &di, u32 &dj) {
    di = n % 3 - 1;
    dj = n / 3 - 1;
}

__device__ dataType& MultiLevelSparseGrid::getFieldValue(u32 fIndex, u32 bIndex, u32 i, u32 j) {
  return fieldDataList[fIndex*nBlocksMax + bIndex](i, j);
}

__device__ dataType& MultiLevelSparseGrid::getParentFieldValue(u32 fIndex, u32 bIndex, u32 ib, u32 jb, u32 i, u32 j) {
  i = i/2 + ib&1*blockSize/2;
  j = j/2 + jb&1*blockSize/2;
  return fieldDataList[fIndex*nBlocksMax + blockList[bIndex].parent](i, j);
}
*/

__device__ u64 MultiLevelSparseGrid::hash(u64 x) {
  // murmur hash
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53L;
  x ^= x >> 33;
  return x;
}

__device__ void MultiLevelSparseGrid::hashInsert(u64 key, u32 value) {
    u32 slot = (hash(key) % (nBlocksMax-1));

    while (true) {
        u64 prev = atomicCAS(&hashKeyList[slot], (u64)bEmpty, key);
        if (prev == bEmpty || prev == key) {
            hashValueList[slot] = value;
            return;
        }
        slot = (slot + 1) % (nBlocksMax-1);
    }
}

__device__ void MultiLevelSparseGrid::hashDelete(u64 key) {
  u32 slot = hash(key) % (nBlocksMax-1);
  while (true) {
      if (hashKeyList[slot] == key) {
          hashKeyList[slot] = bEmpty;
          hashValueList[slot] = bEmpty;
          return;
      }
      if (hashKeyList[slot] == bEmpty) {
          return;
      }
      slot = (slot + 1) % (nBlocksMax - 1);
  }
}

__device__ u32 MultiLevelSparseGrid::hashGetValue(u64 key) {
  u32 slot = hash(key) % (nBlocksMax-1);

  while (true) {
    if (hashKeyList[slot] == key) {
        return hashValueList[slot];
    }
    if (hashKeyList[slot] == bEmpty) {
        return bEmpty;
    }
    slot = (slot + 1) % (nBlocksMax - 1);
  }
}

// seperate bits from a given integer 3 positions apart
__device__ u64 MultiLevelSparseGrid::split(u32 a) {
  u64 x = (u64)a & ((1<<20)-1); // we only look at the first 20 bits
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

// encode ijk indices and resolution level into morton code
__device__ u64 MultiLevelSparseGrid::mortonEncode(u64 lvl, u32 i, u32 j) {
  u64 morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1;
  return morton;
}

// encode ijk indices and resolution level into morton code
__device__ u64 MultiLevelSparseGrid::mortonEncode(u64 lvl, u32 i, u32 j, u32 k) {
  u64 morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1 | split(k) << 2;
  return morton;
}

// compact separated bits into into an integer
__device__ u32 MultiLevelSparseGrid::compact(u64 w) {
  w &=                  0x1249249249249249;
  w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
  w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
  w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
  w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
  w = (w ^ (w >> 32)) & 0x00000000001fffff;
  return (u32)w;
}

// decode morton code into ijk index and resolution level
__device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j) {
  lvl = u32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
}

// decode morton code into ijk index and resolution level
__device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, u32 &lvl, u32 &i, u32 &j, u32 &k) {
  lvl = u32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
  k = compact(morton >> 2);
}

/*
void MultiLevelSparseGrid::sortBlocks(void) {
  copyBlockListToBlockListOld<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);
  thrust::sort_by_key(thrust::device, blockLocList, blockLocList+nBlocks, blockList);
  updateBlockConnectivity<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);
  sortFields();
  copyBlockListOldToBlockList<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);

  findNeighbors<<<nBlocks/cudaBlockSize+1, cudaBlockSize>>>(*this);
}

void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1, 1>>>(*this);
}
*/
