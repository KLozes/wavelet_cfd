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

  nBlocks = 0;
  blockCounter = 0;
  imageCounter = 0;

  // grid size checking
  assert(isPowerOf2(blockSize));
  assert(baseGridSize[0]*baseGridSize[1]/blockSize/blockSize < nBlocksMax);

  cudaMallocManaged(&bLocList, nBlocksMax*sizeof(u64));
  cudaMallocManaged(&bIdxList, nBlocksMax*sizeof(u32));

  cudaMallocManaged(&prntIdxList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&prntIdxListOld, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&chldIdxList, 4*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&chldIdxListOld, 4*nBlocksMax*sizeof(u32));
  cudaMallocManaged(&nbrIdxList, 9*nBlocksMax*sizeof(u32));

  cudaMallocManaged(&bFlagsList, nBlocksMax*sizeof(u32));
  cudaMallocManaged(&cFlagsList, blockSizeTot*nBlocksMax*sizeof(u32));

  cudaMallocManaged(&fieldData, nFields*blockSizeTot*nBlocksMax*sizeof(real));
  cudaMallocManaged(&imageData, imageSize[0]*imageSize[1]*sizeof(real));

  cudaMemset(bLocList, 0, nBlocksMax*sizeof(u64));
  cudaMemset(bIdxList, 0, nBlocksMax*sizeof(u32));

  cudaMemset(prntIdxList, 0, nBlocksMax*sizeof(u32));
  cudaMemset(prntIdxListOld, 0, nBlocksMax*sizeof(u32));
  cudaMemset(chldIdxList, 0, 4*nBlocksMax*sizeof(u32));
  cudaMemset(chldIdxListOld, 0, 4*nBlocksMax*sizeof(u32));
  cudaMemset(nbrIdxList, 0, 9*nBlocksMax*sizeof(u32));

  cudaMemset(bFlagsList, 0, nBlocksMax*sizeof(u32));
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

  initTreeKernel<<<nBlocksMax/cudaBlockSize+1, cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();

  // fill tree with base grid blocks
  nBlocks = 0;
  for(i32 j=0; j<=baseGridSize[1]/blockSize+1; j++) {
    for (i32 i=0; i<=baseGridSize[0]/blockSize+1; i++) {
      bLocList[nBlocks] = encode(0,i,j);
      bIdxList[nBlocks] = nBlocks;
      nBlocks++; 
    }
  }
  nBlocksNew = nBlocks;
}

void MultiLevelSparseGrid::adaptGrid(void) {

  if (nLvls > 1) {
    addFineBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
    addAdjacentBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    for(i32 lvl=nLvls-1; lvl>0; lvl--) {
      setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
      addReconstructionBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    }
    addBoundaryBlocksKernel<<<1000, cudaBlockSize>>>(*this);
    setBlocksKeepKernel<<<1000, cudaBlockSize>>>(*this);
    cudaDeviceSynchronize();
    nBlocksNew = nBlocks;
    deleteDataKernel<<<1000, cudaBlockSize>>>(*this);
  }
}

void MultiLevelSparseGrid::sortBlocks(void) {

  cudaDeviceSynchronize();
  thrust::sort_by_key(thrust::device, bLocList, bLocList+nBlocks, bIdxList);
  sortFieldData();
  resetOldTreeKernel<<<1000,cudaBlockSize>>>(*this);
  cudaDeviceSynchronize();
  nBlocks = nBlocksNew;
  updateOldTreeKernel<<<1000, cudaBlockSize>>>(*this);
  copyTreeFromOldKernel<<<1000, cudaBlockSize>>>(*this);
  nBlocks = nBlocksNew;
  updateNbrIndicesKernel<<<1000, cudaBlockSize>>>(*this);
  flagActiveCellsKernel<<<1000, cudaBlockSize>>>(*this);
  flagParentCellsKernel<<<1000, cudaBlockSize>>>(*this); 
  cudaDeviceSynchronize();
}

__device__ void MultiLevelSparseGrid::getCellPos(i32 lvl, i32 ib, i32 jb, i32 i, i32 j, real *pos) {
  ib -= powi(2, lvl);
  jb -= powi(2, lvl);
  pos[0] = (ib*blockSize + i + .5)*getDx(lvl);
  pos[1] = (jb*blockSize + j + .5)*getDy(lvl);
}

__device__ u32 MultiLevelSparseGrid::getNbrIdx(u32 bIdx, i32 i, i32 j) {
  i += blockSize;
  j += blockSize;
  i32 ib = i / blockSize;
  i32 jb = j / blockSize;
  i32 nbrIdx = nbrIdxList[9*bIdx + ib + 3*jb];
  return blockSizeTot*nbrIdx + (i%blockSize) + (j%blockSize)*blockSize;
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
  return i >= powi(2,lvl) && j >= powi(2,lvl) && i < gridSize[0] + powi(2,lvl) && j < gridSize[1] + powi(2,lvl);
}

__device__ bool MultiLevelSparseGrid::isExteriorBlock(i32 lvl, i32 i, i32 j) {
  return !isInteriorBlock(lvl, i, j);
}

__host__ __device__ real* MultiLevelSparseGrid::getField(u32 f) {
  return &fieldData[f*nBlocksMax*blockSizeTot];
}

__device__ void MultiLevelSparseGrid::activateBlock(i32 lvl, i32 i, i32 j) {

  if (i < 0 || j < 0 || i >= (baseGridSize[0]+2)*powi(2,lvl) || j >= (baseGridSize[1]+2)*powi(2,lvl)) {
    return;
  }

  // start at the base grid level
  i32 iBase = i / powi(2,lvl); 
  i32 jBase = j / powi(2,lvl) ; 
  i32 bIdx = iBase + jBase * (baseGridSize[0]/blockSize+2);

  for(i32 l = 1; l <= lvl; l++) {
    i32 ib = i / powi(2, lvl-l);
    i32 jb = j / powi(2, lvl-l);
    u64 loc = encode(l, ib, jb);
    u32 cIdx = 4*bIdx + 2*(jb%2) + ib%2;

    // swap in a temp index if it is empty
    uint prev = atomicCAS(&chldIdxList[cIdx], bEmpty, bEmpty-1);

    // wait until temp index changes to a real index
    while(chldIdxList[cIdx] == bEmpty-1) {
      // if the previous value of the atomicCAS was empty,
      // increment the nBlocks counter create the child block
      if (prev == bEmpty) {
        u32 idx = atomicAdd(&nBlocks, 1);
        bIdxList[idx] = idx;
        bLocList[idx] = loc;
        prntIdxList[idx] = bIdx;
        chldIdxList[4*idx] = bEmpty;
        chldIdxList[4*idx+1] = bEmpty;
        chldIdxList[4*idx+2] = bEmpty;
        chldIdxList[4*idx+3] = bEmpty;
        chldIdxList[cIdx] = idx;
      }
    }
    bIdx = chldIdxList[cIdx];
    atomicMax(&bFlagsList[bIdx], NEW);
  }
}

__device__ u32 MultiLevelSparseGrid::getBlockIdx(i32 lvl, i32 i, i32 j) {

  if (i < 0 || j < 0 || i >= (baseGridSize[0]/blockSize+2)*powi(2,lvl) || j >= (baseGridSize[1]/blockSize+2)*powi(2,lvl)) {
    return bEmpty;
  }

  // search up the tree starting from the base
  i32 iBase = i / powi(2,lvl); 
  i32 jBase = j / powi(2,lvl);
  i32 bIdx = iBase + jBase * (baseGridSize[0]/blockSize+2);
  
  for(i32 l = 1; l <= lvl; l++) {
    i32 ib = i / powi(2, lvl-l);
    i32 jb = j / powi(2, lvl-l);
    bIdx = chldIdxList[4*bIdx + 2*(jb%2) + ib%2];
    if (bIdx == bEmpty) {
      break;
    }
  }
  return bIdx;
}

// encode ijk indices and resolution level into locational code
__device__ u64 MultiLevelSparseGrid::encode(i32 lvl, i32 i, i32 j) {
  u64 loc = 0;
  loc |= (u64)lvl << 60 | (u64)j << 20 | (u64)i;
  return loc;
}

// decode locational code into ij idx and resolution level
__device__ void MultiLevelSparseGrid::decode(u64 loc, i32 &lvl, i32 &i, i32 &j) {
  lvl = loc >> 60;
  j = ((loc >> 20) & ((1 << 20)-1));
  i = (loc & ((1 << 20)-1));
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

void MultiLevelSparseGrid::computeImageData(i32 f) {

  real *U;
  if (f >= 0) {
    U = getField(f);
  }

  bool gridOn = false;

  // set the pixel values 
  for (uint bIdx=0; bIdx < nBlocks; bIdx++) {
    u64 loc = bLocList[bIdx];
    i32 lvl, ib, jb;
    decode(loc, lvl, ib, jb);
    if (isInteriorBlock(lvl, ib, jb) && loc != kEmpty ) {
      ib -= powi(2, lvl);
      jb -= powi(2, lvl);
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

/*
void MultiLevelSparseGrid::resetBlockCounter(void) {
  zeroBlockCounter<<<1000, cudaBlockSize>>>(*this);
}
*/
