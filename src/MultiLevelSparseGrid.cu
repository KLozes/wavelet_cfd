#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

void MultiLevelSparseGrid::initGrid(void) {
  // initialize the blocks of the base grid level in cartesian order
  uint ind = 0;
  for(uint j=0; j<max(1,baseGridSize[1]/blockSize); j++) {
    for(uint i=0; i<max(1,baseGridSize[0]/blockSize); i++) {
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
    uint lvl, i, j;
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

__device__ uint& MultiLevelSparseGrid::getBaseBlockIndex(int i, int j) {
  i = max(0, min(int(baseGridSize[0]/blockSize)-1, i));
  j = max(0, min(int(baseGridSize[1]/blockSize)-1, j));
  return baseBlockIndexArray[i + baseGridSize[0]/blockSize * j];
}

__device__  uint MultiLevelSparseGrid::getBlockIndex(uint lvl, uint i, uint j) {
  // get the base grid block
  uint bIndex = getBaseBlockIndex(i>>lvl, j>>lvl);

  // search up the tree
  for(uint l = 1; l < lvl+1; l++) {
    uint ib = (i >> (lvl-l)) & 1;
    uint jb = (j >> (lvl-l)) & 1;
    bIndex = blockList[bIndex].children(ib, jb);
  }
  return bIndex;
}

__device__ void MultiLevelSparseGrid::activateBlock(uint lvl, uint i, uint j) {
  // get the base grid block
  uint parent = getBaseBlockIndex(i>>lvl, j>>lvl);

  // search up the tree and activate blocks if they do not exist
  for(uint l = 1; l < lvl+1; l++) {
    uint lvlb = l;
    uint ib = i >> (lvl-l);
    uint jb = j >> (lvl-l);
    uint64_t locb = mortonEncode(lvlb, ib, jb);

    // get the pointer the child index
    uint *child = &(blockList[parent].children(ib&1, jb&1));

    // swap in a temp index if it is empty
    uint prev = atomicCAS(child, bEmpty, 0);

    // wait until temp index changes to a real index
    while(*child == 0) {
      // if the previous value of the atomicCAS was empty,
      // increment the nBlocks counter create the child block
      if (prev == bEmpty) {
        uint nBlocksPrev = atomicAdd(&nBlocks, 1);

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

__device__ void MultiLevelSparseGrid::deactivateBlock(uint bIndex) {
  blockLocList[bIndex] = 0;
  blockList[bIndex].loc = 0;
  blockList[bIndex].index = bEmpty;
  blockList[bIndex].parent = bEmpty;
  blockList[bIndex].children = bEmpty;
  blockList[bIndex].neighbors = bEmpty;
}

__device__ void MultiLevelSparseGrid::getDijk(uint n, uint &di, uint &dj) {
    di = n % 3 - 1;
    dj = n / 3 - 1;
}

__device__ dataType& MultiLevelSparseGrid::getFieldValue(uint fIndex, uint bIndex, uint i, uint j) {
  return fieldDataList[fIndex*nBlocksMax + bIndex](i, j);
}

__device__ dataType& MultiLevelSparseGrid::getParentFieldValue(uint fIndex, uint bIndex, uint ib, uint jb, uint i, uint j) {
  i = i/2 + ib&1*blockSize/2;
  j = j/2 + jb&1*blockSize/2;
  return fieldDataList[fIndex*nBlocksMax + blockList[bIndex].parent](i, j);
}

uint64_t MultiLevelSparseGrid::hash(uint64_t x) {
  // murmur hash
  h ^= h >>> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >>> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >>> 33;
  return h
}

void __device__ MultiLevelSparseGrid::hashInsert(uint64_t key, uint32_t value) {
    uint32_t slot = hash(key);

    while (true) {
        uint32_t prev = atomicCAS(&hashtable[slot].key, bEmpty, key);
        if (prev == bEmpty || prev == key) {
            hashtable[slot].value = value;
            break;
        }
        slot = (slot + 1) & (nBlocksMax-1);
    }
}

void __device__ MultiLevelSparseGrid::hashDelete(uint64_t key, uint value) {
  uint32_t slot = hash(key);

  while (true) {
      if (hashtable[slot].key == key) {
          hashtable[slot].value = bEmpty;
          return;
      }
      if (hashtable[slot].key == bEmpty) {
          return;
      }
      slot = (slot + 1) & (nBlocksMax - 1);
  }
}

uint __device__ MultiLevelSparseGrid::hashGetValue(uint64_t key) {
  uint slot = hash(key);

  while (true) {
      uint32_t prev = atomicCAS(&hashKeyList[slot].key, bEmpty, key);
      if (prev == bEmpty) {
          return bEmpty;
      }
      if (prev == key) {
          return hashValueList[slot];
      }
      slot = (slot + 1) & (kHashTableCapacity-1);
  }
}

// seperate bits from a given integer 3 positions apart
__host__ __device__ uint64_t MultiLevelSparseGrid::split(uint a) {
  uint64_t x = (uint64_t)a & ((1<<20)-1); // we only look at the first 20 bits
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

// encode ijk indices and resolution level into morton code
__host__ __device__ uint64_t MultiLevelSparseGrid::mortonEncode(uint64_t lvl, uint i, uint j) {
  uint64_t morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1;
  return morton;
}

// encode ijk indices and resolution level into morton code
__host__ __device__ uint64_t MultiLevelSparseGrid::mortonEncode(uint64_t lvl, uint i, uint j, uint k) {
  uint64_t morton = 0;
  morton |= lvl << 60 | split(i) | split(j) << 1 | split(k) << 2;
  return morton;
}

// compact separated bits into into an integer
__host__ __device__ uint MultiLevelSparseGrid::compact(uint64_t w) {
  w &=                  0x1249249249249249;
  w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
  w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
  w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
  w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
  w = (w ^ (w >> 32)) & 0x00000000001fffff;
  return (uint)w;
}

// decode morton code into ijk index and resolution level
__host__ __device__ void MultiLevelSparseGrid::mortonDecode(uint64_t morton, uint &lvl, uint &i, uint &j) {
  lvl = uint((morton & ((uint64_t)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((uint64_t)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
}

// decode morton code into ijk index and resolution level
__host__ __device__ void MultiLevelSparseGrid::mortonDecode(uint64_t morton, uint &lvl, uint &i, uint &j, uint &k) {
  lvl = uint((morton & ((uint64_t)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((uint64_t)15 << 60); // remove the last 4 bits
  i = compact(morton);
  j = compact(morton >> 1);
  k = compact(morton >> 2);
}

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
