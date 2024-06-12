#include <thrust/sort.h>
#include <algorithm>

#include <png++/png.hpp>
#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

MultiLevelSparseGrid::MultiLevelSparseGrid(real *domainSize_, u32 *baseGridSize_, u32 nLvls_, u32 nFields_) {

  domainSize[0] = domainSize_[0];
  domainSize[1] = domainSize_[1];

  baseGridSize[0] = baseGridSize_[0];
  baseGridSize[1] = baseGridSize_[1];

  nLvls = nLvls_;
  nFields = nFields_;

  imageSize[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSize[1] = (baseGridSize[1])*powi(2,nLvls-1);

  blockCounter = 0;
  imageCounter = 0;

  lock = 0;


  // grid size checking
  //assert(isPowerOf2(blockSize));
  assert(baseGridSize[0]*baseGridSize[1]/blockSize/blockSize < nBlocksMax);

  cudaMallocManaged(&bLocList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&bIdxList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&bFlagsList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&prntIdxList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&chldIdxList, 4*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&nbrIdxList, blockHaloSizeTot*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&cFlagsList, blockSizeTot*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&fieldData, nFields*blockSizeTot*nBlocksMax*sizeof(real));
  cudaMallocManaged(&imageData, imageSize[0]*imageSize[1]*sizeof(real));

  cudaMemset(bLocList, 0, nBlocksMax*sizeof(u64));
  cudaMemset(bIdxList, 0, nBlocksMax*sizeof(u32));
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(u32));
  cudaMemset(prntIdxList, 0, nBlocksMax*sizeof(u32));
  cudaMemset(chldIdxList, 0, 4*nBlocksMax*sizeof(u32));
  cudaMemset(nbrIdxList, 0, blockHaloSizeTot*nBlocksMax*sizeof(u32));
  cudaMemset(cFlagsList, 0, blockSizeTot*nBlocksMax*sizeof(u32));
  cudaMemset(fieldData, 0, nFields*blockSizeTot*nBlocksMax*sizeof(real));
  cudaMemset(imageData, 0, imageSize[0]*imageSize[1]*sizeof(real));

  cudaDeviceSynchronize();
}

MultiLevelSparseGrid::~MultiLevelSparseGrid(void) {
  cudaDeviceSynchronize();
  cudaFree(bLocList);
  cudaFree(bIdxList);
  cudaFree(prntIdxList);
  cudaFree(nbrIdxList);
  cudaFree(cFlagsList);
  cudaFree(fieldData);
  cudaFree(imageData);
}

void MultiLevelSparseGrid::initializeBaseGrid(void) {
  // fill the bLocList with base grid blocks
  initGridKernel<<<nBlocksMax/cudaBlockSize, cudaBlockSize>>>(*this);
  addBoundaryBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  // sort the data by location code
  sortBlocks();
  cudaDeviceSynchronize();
}

void MultiLevelSparseGrid::adaptGrid(void) {

  addFineBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
  addAdjacentBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  for(i32 lvl=nLvls-1; lvl>2; lvl--) {
    setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
    addReconstructionBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  }
  addBoundaryBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  deleteDataKernel<<<1000, cudaBlockSize>>>(*this);
  updatePrntIndicesKernel<<<1000, cudaBlockSize>>>(*this);
}

void MultiLevelSparseGrid::sortBlocks(void) {

  cudaDeviceSynchronize();
  thrust::sort_by_key(thrust::device, bLocList, bLocList+hashTable.nKeys, bIdxList);
  sortFieldData();
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;
  hashTable.reset();
  updateIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  updatePrntIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  updateNbrIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  flagActiveCellsKernel<<<1000, cudaBlockSize>>>(*this);
  flagParentCellsKernel<<<1000, cudaBlockSize>>>(*this); 
  cudaDeviceSynchronize();
}

__device__ void MultiLevelSparseGrid::getCellPos(i32 lvl, i32 ib, i32 jb, i32 i, i32 j, real *pos) {
  pos[0] = (ib*blockSize + i + .5)*getDx(lvl);
  pos[1] = (jb*blockSize + j + .5)*getDy(lvl);
}

__device__ u32 MultiLevelSparseGrid::getNbrIdx(u32 bIdx, i32 i, i32 j) {
  return nbrIdxList[bIdx*blockHaloSizeTot + (j+haloSize)*blockHaloSize + (i+haloSize)];
}

__device__ real MultiLevelSparseGrid::getDx(i32 lvl) {
  return real(domainSize[0])/real(baseGridSize[0]*powi(2,lvl));
}

__device__ real MultiLevelSparseGrid::getDy(i32 lvl) {
  return real(domainSize[1])/real(baseGridSize[1]*powi(2,lvl));
}

__device__ bool MultiLevelSparseGrid::isInteriorBlock(i32 lvl, i32 i, i32 j) { 
  i32 gridSize[2] = {i32(baseGridSize[0]/blockSize*powi(2,lvl)), 
                     i32(baseGridSize[1]/blockSize*powi(2,lvl))};
  return i >= 0 && j >= 0 && i < gridSize[0] && j < gridSize[1];
}

__device__ bool MultiLevelSparseGrid::isExteriorBlock(i32 lvl, i32 i, i32 j) {
  return !isInteriorBlock(lvl, i, j);
}

__host__ __device__ real* MultiLevelSparseGrid::getField(u32 f) {
  return &fieldData[f*nBlocksMax*blockSizeTot];
}

__device__ void MultiLevelSparseGrid::activateBlock(i32 lvl, i32 i, i32 j) {
  u64 loc = mortonEncode(lvl, i, j);
  u32 idx = hashTable.insert(loc);

  if (idx != bEmpty) { 
    // new key was inserted if not bEmpty
    bLocList[idx] = loc;
    bIdxList[idx] = idx;
    atomicMax(&bFlagsList[idx], NEW);
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
__device__ u64 MultiLevelSparseGrid::mortonEncode(i32 lvl, i32 i, i32 j) {
  u64 morton = 0;
  i += 1; // add one so that boundary blocks are no longer negative negative
  j += 1;
  morton |= (u64)lvl << 60 | split(i) | split(j) << 1;
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

// decode morton code into ij idx and resolution level
__device__ void MultiLevelSparseGrid::mortonDecode(u64 morton, i32 &lvl, i32 &i, i32 &j) {
  lvl = i32((morton & ((u64)15 << 60)) >> 60);   // get the level stored in the last 4 bits
  morton &= ~ ((u64)15 << 60); // remove the last 4 bits
  i = compact(morton) - 1; 
  j = compact(morton >> 1) - 1;
}

void MultiLevelSparseGrid::paint(void) {

  cudaDeviceSynchronize();
  png::image<png::gray_pixel_16> image(imageSize[0], imageSize[1]);

  for (i32 f=-1; f<4; f++) {
    //computeImageData(f);
    computeImageDataKernel<<<1000, cudaBlockSize>>>(*this, f);
    cudaDeviceSynchronize();

    // normalize image data and fill png image
    real maxVal = -1e32;
    real minVal = 1e32;

    for (i32 idx=0; idx<imageSize[0]*imageSize[1]; idx++) {
      maxVal = fmax(maxVal, imageData[idx]);
      minVal = fmin(minVal, imageData[idx]);
    }

    if (f == -1) {
      minVal = 0;
      maxVal = nLvls;
    }
 
    for (i32 j=0; j<imageSize[1]; j++) {
      for (i32 i=0; i<imageSize[0]; i++) {
        i32 idx = j*imageSize[1] + i;
        image[j][i] = (imageData[idx] - minVal) / (maxVal - minVal + 1e-16) * 65535;
      }
    }

    // output the image to a png file
    char fileName[50];
    if (f >=0) {
      sprintf(fileName, "output/image%02d_%05d.png", f, imageCounter);
    }
    else {
      sprintf(fileName, "output/grid_%05d.png", imageCounter);
    }
    image.write(fileName);
  }
  imageCounter++;
}

/*
void MultiLevelSparseGrid::computeImageData(i32 f) {

  real *U;
  if (f >= 0) {
    U = getField(f);
  }

  bool gridOn = true;

  // set the pixel values 
  for (uint bIdx=0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    i32 lvl, ib, jb;
    mortonDecode(loc, lvl, ib, jb);
    if (isInteriorBlock(lvl, ib, jb) && loc != kEmpty) {
      for (uint j = 0; j < blockSize; j++) {
        for (uint i = 0; i < blockSize; i++) {
          u32 idx = i + blockSize * j + bIdx*blockSizeTot;
          u32 nPixels = powi(2,(nLvls - 1 - lvl));
          for (uint jj=0; jj<nPixels; jj++) {
            for (uint ii=0; ii<nPixels; ii++) {
              u32 iPxl = ib*blockSize*nPixels + i*nPixels + ii;
              u32 jPxl = jb*blockSize*nPixels + j*nPixels + jj;
              if (f >= 0) {
                imageData[jPxl*imageSize[0] + iPxl] = U[idx];
              }
              else {
                u32 cFlag = cFlagsList[idx];
                imageData[jPxl*imageSize[0] + iPxl] = lvl+1 - (2-cFlag)/2;
              }
              if (gridOn && ii > 0 && jj > 0) {
                  imageData[jPxl*imageSize[0] + iPxl] = 0;
              }
            }
          }
        }
      }
    }
  }
}
*/
/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1000, cudaBlockSize>>>(*this);
}
*/
