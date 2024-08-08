#include <thrust/sort.h>
#include <algorithm>

#include <png++/png.hpp>
#include "MultiLevelSparseGrid.cuh"
#include "MultiLevelSparseGridKernels.cuh"

MultiLevelSparseGrid::MultiLevelSparseGrid(real *domainSize_, i32 *baseGridSize_, i32 nLvls_, i32 nFields_) {

  domainSize[0] = domainSize_[0];
  domainSize[1] = domainSize_[1];
  domainSize[2] = domainSize_[2];

  baseGridSize[0] = baseGridSize_[0];
  baseGridSize[1] = baseGridSize_[1];
  baseGridSize[2] = baseGridSize_[2];

  nLvls = nLvls_;
  nFields = nFields_;

  imageSize[0] = (baseGridSize[0])*powi(2,nLvls-1);  // image size is the max resolution not including boundary condition blocks
  imageSize[1] = (baseGridSize[1])*powi(2,nLvls-1);

  imageCounter = 0;

  // grid size checking
  assert(isPowerOf2(blockSize));
  assert(baseGridSize[0]*baseGridSize[1]*baseGridSize[2]/blockSizeTot < nBlocksMax);

  cudaMallocManaged(&bLocList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&bIdxList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&bFlagsList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&prntIdxList, nBlocksMax*sizeof(i32));
  cudaMallocManaged(&nbrIdxList, 27*nBlocksMax*sizeof(i32));
  cudaMallocManaged(&cFlagsList, blockSizeTot*nBlocksMax*sizeof(i32));
  cudaMallocManaged(&fieldData, nFields*blockSizeTot*nBlocksMax*sizeof(real));
  cudaMallocManaged(&imageData, imageSize[0]*imageSize[1]*sizeof(real));

  cudaMemset(bLocList, 0, nBlocksMax*sizeof(u64));
  cudaMemset(bIdxList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(prntIdxList, 0, nBlocksMax*sizeof(i32));
  cudaMemset(nbrIdxList, 0, 27*nBlocksMax*sizeof(i32));
  cudaMemset(cFlagsList, 0, blockSizeTot*nBlocksMax*sizeof(i32));
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
  dim3 cudaBlockSize3(8,8,8);
  dim3 nCudaBlocks3(baseGridSize[0]/blockSize/8+1, 
                    baseGridSize[1]/blockSize/8+1, 
                    baseGridSize[2]/blockSize/8+1);
  initGridKernel<<<nCudaBlocks3, cudaBlockSize3>>>(*this);
  addBoundaryBlocksKernel<<<1000, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;

  // sort the data by location code
  sortBlocks();
  cudaDeviceSynchronize();
}

void MultiLevelSparseGrid::adaptGrid(void) {

  if (nLvls > 1) {
    addFineBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
    addAdjacentBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    for(i32 lvl=nLvls-1; lvl>2; lvl--) {
      setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
      addReconstructionBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    }
    addBoundaryBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    nBlocks = hashTable.nKeys;
    deleteDataKernel<<<1000, cudaBlockSize>>>(*this);
    updatePrntIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  }
}

void MultiLevelSparseGrid::sortBlocks(void) {

  cudaDeviceSynchronize();
  thrust::sort_by_key(thrust::device, bLocList, bLocList+hashTable.nKeys, bIdxList);
  sortFieldData();
  cudaDeviceSynchronize();
  hashTable.reset();
  hashTable.nKeys = nBlocks;
  updateIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  updatePrntIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  updateNbrIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  flagActiveCellsKernel<<<1000, cudaBlockSize>>>(*this);
  flagParentCellsKernel<<<1000, cudaBlockSize>>>(*this); 
  cudaDeviceSynchronize();
}

__device__ Vec3 MultiLevelSparseGrid::getCellPos(i32 lvl, i32 ib, i32 jb, i32 kb, i32 i, i32 j, i32 k) {
  return Vec3((ib*blockSize + i + .5)*getDx(lvl),  
              (jb*blockSize + j + .5)*getDy(lvl), 
              (kb*blockSize + k + .5)*getDz(lvl));
}

__device__ i32 MultiLevelSparseGrid::getNbrIdx(i32 bIdx, i32 i, i32 j, i32 k) {
  i += blockSize;
  j += blockSize;
  k += blockSize;
  i32 ib = i / blockSize;
  i32 jb = j / blockSize;
  i32 kb = k / blockSize;
  i32 nbrIdx = nbrIdxList[9*bIdx + ib + 3*jb + 9*kb];
  return blockSizeTot*nbrIdx + (i%blockSize) + (j%blockSize)*blockSize + (k%blockSize)*blockSize*blockSize;
}

__device__ real MultiLevelSparseGrid::getDx(i32 lvl) {
  return real(domainSize[0])/real(baseGridSize[0]*powi(2,lvl));
}

__device__ real MultiLevelSparseGrid::getDy(i32 lvl) {
  return real(domainSize[1])/real(baseGridSize[1]*powi(2,lvl));
}

__device__ real MultiLevelSparseGrid::getDz(i32 lvl) {
  return real(domainSize[2])/real(baseGridSize[2]*powi(2,lvl));
}

__device__ bool MultiLevelSparseGrid::isInteriorBlock(i32 lvl, i32 i, i32 j, i32 k) { 
  i32 gridSize[3] = {i32(baseGridSize[0]/blockSize*powi(2,lvl)), 
                     i32(baseGridSize[1]/blockSize*powi(2,lvl)),
                     i32(baseGridSize[2]/blockSize*powi(2,lvl))};
  return i >= 0 && j >= 0 && k >= 0 && i < gridSize[0] && j < gridSize[1] && k < gridSize[2];
}

__device__ bool MultiLevelSparseGrid::isExteriorBlock(i32 lvl, i32 i, i32 j, i32 k) {
  return !isInteriorBlock(lvl, i, j, k);
}

__host__ __device__ real* MultiLevelSparseGrid::getField(i32 f) {
  return &fieldData[f*nBlocksMax*blockSizeTot];
}

__device__ void MultiLevelSparseGrid::activateBlock(i32 lvl, i32 i, i32 j, i32 k) {
  u64 loc = encode(lvl, i, j, k);
  i32 idx = hashTable.insert(loc);
  if (idx != bEmpty) { 
    // new key was inserted if not bEmpty
    bLocList[idx] = loc;
    bIdxList[idx] = idx;
    atomicMax(&bFlagsList[idx], NEW);
  }
}

// encode ijk indices and resolution level into locational code
__device__ u64 MultiLevelSparseGrid::encode(i32 lvl, i32 i, i32 j, i32 k) {
  i += 1; // add one so that boundary blocks are no longer negative negative
  j += 1;
  k += 1;
  u64 loc = 0;
  loc |= (u64)lvl << 60 | (u64) k << 40 | (u64)j << 20 | (u64)i;
  return loc;
}

// decode locational code into ij idx and resolution level
__device__ void MultiLevelSparseGrid::decode(u64 loc, i32 &lvl, i32 &i, i32 &j, i32 &k) {
  lvl = loc >> 60;
  k = ((loc >> 40) & ((1 << 20)-1)) - 1;
  j = ((loc >> 20) & ((1 << 20)-1)) - 1;
  i = (loc & ((1 << 20)-1)) - 1;
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
        i32 idx = j*imageSize[0] + i;
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

void MultiLevelSparseGrid::computeImageData(i32 f) {

  real *U;
  if (f >= 0) {
    U = getField(f);
  }

  bool gridOn = false;

  // set the pixel values 
  for (uint bIdx=0; bIdx < hashTable.nKeys; bIdx++) {
    u64 loc = bLocList[bIdx];
    i32 lvl, ib, jb, kb;
    decode(loc, lvl, ib, jb, kb);
    if (isInteriorBlock(lvl, ib, jb, kb) && loc != kEmpty) {
      for (uint j = 0; j < blockSize; j++) {
        for (uint i = 0; i < blockSize; i++) {
          i32 idx = i + blockSize * j + bIdx*blockSizeTot;
          i32 nPixels = powi(2,(nLvls - 1 - lvl));
          for (uint jj=0; jj<nPixels; jj++) {
            for (uint ii=0; ii<nPixels; ii++) {
              i32 iPxl = ib*blockSize*nPixels + i*nPixels + ii;
              i32 jPxl = jb*blockSize*nPixels + j*nPixels + jj;
              if (f >= 0) {
                imageData[jPxl*imageSize[0] + iPxl] = U[idx];
              }
              else {
                i32 cFlag = cFlagsList[idx];
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

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1000, cudaBlockSize>>>(*this);
}
*/
