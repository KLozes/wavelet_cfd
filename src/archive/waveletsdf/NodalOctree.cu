// ===========================================================================
// Host serial NODAL OCTREE (pointerless / index-linked).
//
//  * minimal oracle: every grid node is sampled EXACTLY ONCE (deduped by its
//    finest-grid integer coordinate), SDF VALUE only -- no gradients stored.
//  * build: start from a base grid of level-0 roots; each mesh surface point
//    DESCENDS root->leaf, and a leaf whose trilinear interpolation mispredicts
//    the on-surface value (true 0) by more than `thresh` is split (8 children,
//    only the 19 new nodes sampled).  No 2:1 grading.
//  * DC: dual contour the valid (8-node) leaf cells -- QEF vertices with finite-
//    difference normals + the watertight minimal-edge recursion (ungraded).
//  * output: two VTK UnstructuredGrid (.vtu) files -- leaf hexahedra + nodal SDF,
//    and the dual-contour surface triangles.
// ===========================================================================
#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "Bvh.h"
#include "BvhQuery.h"
#include "Vec3f.cuh"

namespace {

// canonical adaptive-DC tables (corner c = (x<<2)|(y<<1)|z), validated equal to the
// GPU/recursion connectivity.
const int edgevmap[12][2] = {
  {0,4},{1,5},{2,6},{3,7}, {0,2},{1,3},{4,6},{5,7}, {0,1},{2,3},{4,5},{6,7} };
const int cellProcFaceMask[12][3] = {
  {0,4,0},{1,5,0},{2,6,0},{3,7,0}, {0,2,1},{4,6,1},{1,3,1},{5,7,1}, {0,1,2},{2,3,2},{4,5,2},{6,7,2} };
const int cellProcEdgeMask[6][5] = {
  {0,1,2,3,0},{4,5,6,7,0}, {0,4,1,5,1},{2,6,3,7,1}, {0,2,4,6,2},{1,3,5,7,2} };
const int faceProcFaceMask[3][4][3] = {
  {{4,0,0},{5,1,0},{6,2,0},{7,3,0}}, {{2,0,1},{6,4,1},{3,1,1},{7,5,1}}, {{1,0,2},{3,2,2},{5,4,2},{7,6,2}} };
const int faceProcEdgeMask[3][4][6] = {
  {{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
  {{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
  {{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}} };
const int edgeProcEdgeMask[3][2][5] = {
  {{3,2,1,0,0},{7,6,5,4,0}}, {{5,1,4,0,1},{7,3,6,2,1}}, {{6,4,2,0,2},{7,5,3,1,2}} };
const int processEdgeMask[3][4] = { {3,2,1,0},{7,5,6,4},{11,10,9,8} };
const int faceProcOrders[2][4] = { {0,0,1,1}, {0,1,0,1} };
const int VTK_HEX[8] = { 0,4,6,2,1,5,7,3 };   // my corner order (a*4+b*2+d) -> VTK hex order

static void qefSolve(const std::vector<std::array<double,3>> &N,
                     const std::vector<std::array<double,3>> &P,
                     const double c[3], double out[3]) {
  double ata[6]={0,0,0,0,0,0}, atb[3]={0,0,0};
  for (size_t i=0;i<N.size();i++){
    double n0=N[i][0],n1=N[i][1],n2=N[i][2];
    double d=n0*P[i][0]+n1*P[i][1]+n2*P[i][2];
    ata[0]+=n0*n0;ata[1]+=n0*n1;ata[2]+=n0*n2;ata[3]+=n1*n1;ata[4]+=n1*n2;ata[5]+=n2*n2;
    atb[0]+=n0*d;atb[1]+=n1*d;atb[2]+=n2*d;
  }
  double lam=1e-3*(ata[0]+ata[3]+ata[5])/3.0+1e-9;
  double M[6]={ata[0]+lam,ata[1],ata[2],ata[3]+lam,ata[4],ata[5]+lam};
  double r[3]={atb[0]+lam*c[0],atb[1]+lam*c[1],atb[2]+lam*c[2]};
  double det=M[0]*(M[3]*M[5]-M[4]*M[4])-M[1]*(M[1]*M[5]-M[4]*M[2])+M[2]*(M[1]*M[4]-M[3]*M[2]);
  if(fabs(det)<1e-18){out[0]=c[0];out[1]=c[1];out[2]=c[2];return;}
  double iv[6];
  iv[0]=(M[3]*M[5]-M[4]*M[4])/det; iv[1]=(M[2]*M[4]-M[1]*M[5])/det; iv[2]=(M[1]*M[4]-M[2]*M[3])/det;
  iv[3]=(M[0]*M[5]-M[2]*M[2])/det; iv[4]=(M[2]*M[1]-M[0]*M[4])/det; iv[5]=(M[0]*M[3]-M[1]*M[1])/det;
  out[0]=iv[0]*r[0]+iv[1]*r[1]+iv[2]*r[2];
  out[1]=iv[1]*r[0]+iv[3]*r[1]+iv[4]*r[2];
  out[2]=iv[2]*r[0]+iv[4]*r[1]+iv[5]*r[2];
}

// one thread per node: sample the BVH oracle (SDF value only).
__global__ void nodeSampleKernel(const BvhNode *nodes, const i32 *order, const TriFeat *tris,
                                 real orient, const float3 *coord, float *val, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) { float3 g; val[i] = signedDistanceGrad(nodes, order, tris, orient, coord[i], g); }
}

// ===========================================================================
// GPU dual contour for the nodal octree.  Cells are uploaded as flat SoA arrays +
// a device cell-key hash; QEF vertices and the watertight minimal-edge connectivity
// (the validated block-grid logic) run as per-cell kernels.  mode 0 emits triangles,
// mode 1 collects the non-straddling cells at open edges (the crack fixpoint).
// ---------------------------------------------------------------------------
struct NoDcP {
  const int *clvl,*ci,*cj,*ck,*ccorner,*cchild; int nCells;
  const float *nval;
  u64 *ckeys; i32 *cvals; int ccap;
  int *cvid; unsigned char *cmask;
  float *vertX; int *triA; int *openA; int *cnt;   // cnt[0]=verts cnt[1]=tris cnt[2]=open
  int maxVert, maxTri, maxOpen;
  double ox,oy,oz, dsx,dsy,dsz; int bgx,bgy,bgz, nLvls, mode;
};
__constant__ int noEdge[12][2]={{0,4},{1,5},{2,6},{3,7},{0,2},{1,3},{4,6},{5,7},{0,1},{2,3},{4,5},{6,7}};
__constant__ int noProc[3][4]={{3,2,1,0},{7,5,6,4},{11,10,9,8}};
__constant__ int noQc[12]={3,2,1,0,3,1,2,0,3,2,1,0};
__constant__ int noQe[12][2]={{0,4},{2,6},{1,5},{3,7},{0,2},{4,6},{1,3},{5,7},{0,1},{4,5},{2,3},{6,7}};

__device__ inline u64 noCk(int l,int i,int j,int k){ return ((u64)l<<60)|((u64)(u32)(i+1))|((u64)(u32)(j+1)<<20)|((u64)(u32)(k+1)<<40); }
__device__ inline u64 noMur(u64 x){ x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;x*=0xc4ceb9fe1a85ec53ULL;x^=x>>33;return x; }
__device__ inline void noPut(u64*keys,i32*vals,int cap,u64 key,i32 v){ u64 h=noMur(key)&(cap-1);
  for(;;){ u64 pr=atomicCAS((unsigned long long*)&keys[h],0xFFFFFFFFFFFFFFFFULL,(unsigned long long)key);
    if(pr==0xFFFFFFFFFFFFFFFFULL||pr==key){ vals[h]=v; return; } h=(h+1)&(cap-1); } }
__device__ inline i32 noGet(const u64*keys,const i32*vals,int cap,u64 key){ u64 h=noMur(key)&(cap-1);
  for(;;){ u64 kk=keys[h]; if(kk==key)return vals[h]; if(kk==0xFFFFFFFFFFFFFFFFULL)return -1; h=(h+1)&(cap-1); } }
__device__ inline int  noLk(const NoDcP&p,int l,int i,int j,int k){ return noGet(p.ckeys,p.cvals,p.ccap,noCk(l,i,j,k)); }
__device__ inline bool noFiner(const NoDcP&p,int l,int i,int j,int k){ return l<p.nLvls-1 && noLk(p,l+1,2*i,2*j,2*k)>=0; }
__device__ inline bool noLeaf(const NoDcP&p,int l,int i,int j,int k){ int c=noLk(p,l,i,j,k); return c>=0 && p.cchild[c]<0; }
__device__ inline int  noAsc(const NoDcP&p,int l,int i,int j,int k){ for(int L=l;L>=0;L--){int s=l-L; int c=noLk(p,L,i>>s,j>>s,k>>s); if(c>=0)return c;} return -1; }
__device__ inline bool noLo(int ax,int ay,int az,int bx,int by,int bz){ if(az!=bz)return az<bz; if(ay!=by)return ay<by; return ax<bx; }

__global__ void noHashKernel(NoDcP p){ int c=blockIdx.x*blockDim.x+threadIdx.x; if(c<p.nCells) noPut(p.ckeys,p.cvals,p.ccap,noCk(p.clvl[c],p.ci[c],p.cj[c],p.ck[c]),c); }

__global__ void noQefKernel(NoDcP p){
  int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=p.nCells)return;
  if(p.cchild[c]>=0){ p.cvid[c]=-1; return; }
  int lvl=p.clvl[c],I=p.ci[c],J=p.cj[c],K=p.ck[c];
  double csx=p.dsx/(double)(p.bgx<<lvl),csy=p.dsy/(double)(p.bgy<<lvl),csz=p.dsz/(double)(p.bgz<<lvl);
  float cv[8]; unsigned char m=0;
  for(int q=0;q<8;q++){ cv[q]=p.nval[p.ccorner[8*c+q]]; if(cv[q]<0)m|=(1<<q); }
  p.cmask[c]=m; if(m==0||m==0xFF){ p.cvid[c]=-1; return; }
  double ata[6]={0,0,0,0,0,0},atb[3]={0,0,0},mass[3]={0,0,0}; int cnt=0;
  for(int e=0;e<12;e++){ int c0=noQe[e][0],c1=noQe[e][1]; float va=cv[c0],vb=cv[c1]; if((va<0)==(vb<0))continue;
    double pa[3]={(I+(c0>>2))*csx,(J+((c0>>1)&1))*csy,(K+(c0&1))*csz};
    double pb[3]={(I+(c1>>2))*csx,(J+((c1>>1)&1))*csy,(K+(c1&1))*csz};
    double t=va/(double)(va-vb); double pp[3]={pa[0]+t*(pb[0]-pa[0]),pa[1]+t*(pb[1]-pa[1]),pa[2]+t*(pb[2]-pa[2])};
    double u=pp[0]/csx-I,v=pp[1]/csy-J,w=pp[2]/csz-K, gx=0,gy=0,gz=0;
    for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int d=0;d<2;d++){ double val=cv[a*4+b*2+d];
      gx+=val*(a?1:-1)*(b?v:1-v)*(d?w:1-w); gy+=val*(a?u:1-u)*(b?1:-1)*(d?w:1-w); gz+=val*(a?u:1-u)*(b?v:1-v)*(d?1:-1); }
    gx/=csx;gy/=csy;gz/=csz; double gn=sqrt(gx*gx+gy*gy+gz*gz)+1e-20; gx/=gn;gy/=gn;gz/=gn;
    double dd=gx*pp[0]+gy*pp[1]+gz*pp[2];
    ata[0]+=gx*gx;ata[1]+=gx*gy;ata[2]+=gx*gz;ata[3]+=gy*gy;ata[4]+=gy*gz;ata[5]+=gz*gz;
    atb[0]+=gx*dd;atb[1]+=gy*dd;atb[2]+=gz*dd; mass[0]+=pp[0];mass[1]+=pp[1];mass[2]+=pp[2];cnt++; }
  if(!cnt){ p.cvid[c]=-1; return; }
  double ctr[3]={mass[0]/cnt,mass[1]/cnt,mass[2]/cnt};
  double lam=1e-3*(ata[0]+ata[3]+ata[5])/3.0+1e-9;
  double M[6]={ata[0]+lam,ata[1],ata[2],ata[3]+lam,ata[4],ata[5]+lam};
  double r[3]={atb[0]+lam*ctr[0],atb[1]+lam*ctr[1],atb[2]+lam*ctr[2]};
  double det=M[0]*(M[3]*M[5]-M[4]*M[4])-M[1]*(M[1]*M[5]-M[4]*M[2])+M[2]*(M[1]*M[4]-M[3]*M[2]); double x[3];
  if(fabs(det)<1e-18){x[0]=ctr[0];x[1]=ctr[1];x[2]=ctr[2];}
  else{ double iv[6];
    iv[0]=(M[3]*M[5]-M[4]*M[4])/det;iv[1]=(M[2]*M[4]-M[1]*M[5])/det;iv[2]=(M[1]*M[4]-M[2]*M[3])/det;
    iv[3]=(M[0]*M[5]-M[2]*M[2])/det;iv[4]=(M[2]*M[1]-M[0]*M[4])/det;iv[5]=(M[0]*M[3]-M[1]*M[1])/det;
    x[0]=iv[0]*r[0]+iv[1]*r[1]+iv[2]*r[2];x[1]=iv[1]*r[0]+iv[3]*r[1]+iv[4]*r[2];x[2]=iv[2]*r[0]+iv[4]*r[1]+iv[5]*r[2]; }
  double lo[3]={I*csx,J*csy,K*csz},he[3]={csx,csy,csz};
  for(int a=0;a<3;a++) x[a]=fmin(fmax(x[a],lo[a]-he[a]),lo[a]+2*he[a]);
  int vid=atomicAdd(&p.cnt[0],1);
  if(vid<p.maxVert){ p.vertX[3*vid]=(float)x[0];p.vertX[3*vid+1]=(float)x[1];p.vertX[3*vid+2]=(float)x[2]; }  // grid frame
  p.cvid[c]=vid;
}
__device__ inline void noTri(NoDcP&p,int a,int b,int c){ if(a==b||b==c||a==c)return; int t=atomicAdd(&p.cnt[1],1); if(t<p.maxTri){p.triA[3*t]=a;p.triA[3*t+1]=b;p.triA[3*t+2]=c;} }

__global__ void noConnKernel(NoDcP p){
  int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=p.nCells||p.cchild[c]>=0)return;
  int myVid=p.cvid[c]; if(myVid<0)return;
  int lvl=p.clvl[c],I=p.ci[c],J=p.cj[c],K=p.ck[c]; unsigned char m=p.cmask[c];
  for(int e=0;e<12;e++){ int c0=noEdge[e][0],c1=noEdge[e][1]; if(((m>>c0)&1)==((m>>c1)&1))continue;
    int dir=e/4,px[4],py[4],pz[4];
    if(dir==0){int by=(e>>1)&1,bz=e&1; px[0]=I;py[0]=J+by-1;pz[0]=K+bz-1; px[1]=I;py[1]=J+by-1;pz[1]=K+bz; px[2]=I;py[2]=J+by;pz[2]=K+bz-1; px[3]=I;py[3]=J+by;pz[3]=K+bz; }
    else if(dir==1){int ee=e-4,bx=(ee>>1)&1,bz=ee&1; px[0]=I+bx-1;py[0]=J;pz[0]=K+bz-1; px[1]=I+bx;py[1]=J;pz[1]=K+bz-1; px[2]=I+bx-1;py[2]=J;pz[2]=K+bz; px[3]=I+bx;py[3]=J;pz[3]=K+bz; }
    else{int ee=e-8,bx=(ee>>1)&1,by=ee&1; px[0]=I+bx-1;py[0]=J+by-1;pz[0]=K; px[1]=I+bx-1;py[1]=J+by;pz[1]=K; px[2]=I+bx;py[2]=J+by-1;pz[2]=K; px[3]=I+bx;py[3]=J+by;pz[3]=K; }
    int qC=noQc[e]; bool minimal=true,owner=true;
    for(int q=0;q<4&&minimal;q++) if(noFiner(p,lvl,px[q],py[q],pz[q]))minimal=false;
    if(!minimal)continue;
    for(int q=0;q<4;q++){ if(q==qC)continue; if(noLeaf(p,lvl,px[q],py[q],pz[q])&&noLo(px[q],py[q],pz[q],I,J,K)){owner=false;break;} }
    if(!owner)continue;
    int idx[4]; bool allv=true;
    for(int q=0;q<4;q++){ int cc=noAsc(p,lvl,px[q],py[q],pz[q]); idx[q]=cc<0?-1:p.cvid[cc]; if(idx[q]<0)allv=false; }
    if(!allv){ if(p.mode==1) for(int q=0;q<4;q++){ int cc=noAsc(p,lvl,px[q],py[q],pz[q]);
        if(cc>=0&&p.cvid[cc]<0&&p.cchild[cc]<0&&p.clvl[cc]<p.nLvls-1){ int o=atomicAdd(&p.cnt[2],1); if(o<p.maxOpen)p.openA[o]=cc; } }
      continue; }
    if(p.mode==1)continue;
    int e2=noProc[dir][qC]; bool flip=((m>>noEdge[e2][0])&1)==1;
    if(flip){ noTri(p,idx[0],idx[3],idx[1]); noTri(p,idx[0],idx[2],idx[3]); }
    else    { noTri(p,idx[0],idx[1],idx[3]); noTri(p,idx[0],idx[3],idx[2]); }
  }
}

struct NodalOctree {
  const BvhNode *bnodes; const i32 *border; const TriFeat *btris; real orient;
  double origin[3], domainSize[3];
  int baseGrid[3]; int nLvls; double thresh;
  double fs[3];                              // finest node spacing per axis

  std::vector<float> nval;                   // SDF per node (sampled once)
  std::vector<std::array<int,3>> nfijk;      // finest-grid integer coords (for world pos)
  std::unordered_map<uint64_t,int> nmap;     // finest key -> node index
  long oracleCalls = 0;

  // GPU node sampling: defer each node's value during the level-by-level build, then
  // sample the whole level's new nodes in one parallel kernel.
  BvhNode *dBN=nullptr; i32 *dBO=nullptr; TriFeat *dBT=nullptr; int nBvh=0, nTri=0;
  float3 *dCoord=nullptr; float *dVal=nullptr; int sampCap=0, sampledCount=0; bool deferring=false;

  void setupGpu(const BvhNode *hN, int nN, int nT) {
    nBvh=nN; nTri=nT;
    cudaMalloc(&dBN,(size_t)nN*sizeof(BvhNode)); cudaMemcpy(dBN,hN,(size_t)nN*sizeof(BvhNode),cudaMemcpyHostToDevice);
    cudaMalloc(&dBO,(size_t)nT*sizeof(i32));     cudaMemcpy(dBO,border,(size_t)nT*sizeof(i32),cudaMemcpyHostToDevice);
    cudaMalloc(&dBT,(size_t)nT*sizeof(TriFeat)); cudaMemcpy(dBT,btris,(size_t)nT*sizeof(TriFeat),cudaMemcpyHostToDevice);
  }
  void teardownGpu(){ cudaFree(dBN);cudaFree(dBO);cudaFree(dBT); if(dCoord)cudaFree(dCoord); if(dVal)cudaFree(dVal); }
  void sampleNewNodesGpu() {                 // sample nval[sampledCount .. size) on the GPU
    int n0=sampledCount, n1=(int)nval.size(), m=n1-n0; if(m<=0) return;
    if(m>sampCap){ if(dCoord)cudaFree(dCoord); if(dVal)cudaFree(dVal);
                   sampCap=m+m/2; cudaMalloc(&dCoord,(size_t)sampCap*sizeof(float3)); cudaMalloc(&dVal,(size_t)sampCap*sizeof(float)); }
    std::vector<float3> co(m);
    for(int i=0;i<m;i++){ auto&f=nfijk[n0+i]; co[i]=make_float3((float)(f[0]*fs[0]),(float)(f[1]*fs[1]),(float)(f[2]*fs[2])); }
    cudaMemcpy(dCoord,co.data(),(size_t)m*sizeof(float3),cudaMemcpyHostToDevice);
    nodeSampleKernel<<<(m+255)/256,256>>>(dBN,dBO,dBT,orient,dCoord,dVal,m);
    std::vector<float> vb(m); cudaMemcpy(vb.data(),dVal,(size_t)m*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<m;i++) nval[n0+i]=vb[i];
    oracleCalls += m; sampledCount=n1;
  }

  struct Cell { int lvl,i,j,k; int corner[8]; int firstChild; };
  std::vector<Cell> cells;
  std::unordered_map<uint64_t,int> cmap;     // (lvl,i,j,k) -> cell index

  // DC scratch
  std::vector<int> cvid;                      // cell -> DC vertex id (-1 none)
  std::vector<unsigned char> cmask;           // cell -> corner sign mask
  std::vector<std::array<double,3>> dverts;   // DC vertices (grid frame)
  std::vector<std::array<int,3>> dtris;       // DC triangles
  long nUnclosed = 0;
  bool collecting = false;                    // recursion collects open-edge cells instead of emitting
  std::vector<int> toRefine;

  uint64_t nkey(int fi,int fj,int fk) const {
    return (uint64_t)(uint32_t)fi | ((uint64_t)(uint32_t)fj<<21) | ((uint64_t)(uint32_t)fk<<42); }
  uint64_t ckey(int l,int i,int j,int k) const {
    return ((uint64_t)l<<60)|((uint64_t)(uint32_t)(i+1))|((uint64_t)(uint32_t)(j+1)<<20)|((uint64_t)(uint32_t)(k+1)<<40); }
  double cs(int lvl,int ax) const { return domainSize[ax]/(double)(baseGrid[ax]<<lvl); }

  int getNode(int lvl,int i,int j,int k) {
    int s=(nLvls-1)-lvl; int fi=i<<s, fj=j<<s, fk=k<<s;
    uint64_t key=nkey(fi,fj,fk);
    auto it=nmap.find(key); if(it!=nmap.end()) return it->second;
    int idx=(int)nval.size(); nfijk.push_back({fi,fj,fk}); nmap[key]=idx;
    if(deferring){ nval.push_back(0.0f); }              // sampled later in batch on the GPU
    else { float3 p=make_float3((float)(fi*fs[0]),(float)(fj*fs[1]),(float)(fk*fs[2])); float3 g;
           nval.push_back(signedDistanceGrad(bnodes,border,btris,orient,p,g)); oracleCalls++; }
    return idx;
  }
  int makeCell(int lvl,int i,int j,int k) {
    Cell c; c.lvl=lvl;c.i=i;c.j=j;c.k=k;c.firstChild=-1;
    for(int corner=0;corner<8;corner++){int a=corner>>2,b=(corner>>1)&1,d=corner&1; c.corner[corner]=getNode(lvl,i+a,j+b,k+d);}
    int idx=(int)cells.size(); cells.push_back(c); cmap[ckey(lvl,i,j,k)]=idx; return idx;
  }
  void refine(int ci) {
    int l=cells[ci].lvl,i=cells[ci].i,j=cells[ci].j,k=cells[ci].k, fc=(int)cells.size();
    for(int ch=0;ch<8;ch++){int a=ch>>2,b=(ch>>1)&1,d=ch&1; makeCell(l+1,2*i+a,2*j+b,2*k+d);}
    cells[ci].firstChild=fc;                 // re-access by index (vector may have realloc'd)
  }
  int childContaining(int ci,double px,double py,double pz) {
    Cell &c=cells[ci];
    int a=(px>=(c.i+0.5)*cs(c.lvl,0))?1:0, b=(py>=(c.j+0.5)*cs(c.lvl,1))?1:0, d=(pz>=(c.k+0.5)*cs(c.lvl,2))?1:0;
    return cells[ci].firstChild + (a*4+b*2+d);
  }
  double trilin(int ci,double px,double py,double pz) {
    Cell &c=cells[ci];
    double u=px/cs(c.lvl,0)-c.i, v=py/cs(c.lvl,1)-c.j, w=pz/cs(c.lvl,2)-c.k;
    u=u<0?0:(u>1?1:u); v=v<0?0:(v>1?1:v); w=w<0?0:(w>1?1:w);
    double val=0;
    for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int d=0;d<2;d++)
      val += (a?u:1-u)*(b?v:1-v)*(d?w:1-w) * nval[c.corner[a*4+b*2+d]];
    return val;
  }
  // LEVEL-BY-LEVEL build: a cell is split iff some mesh point inside it mispredicts the
  // surface (trilinear error > thresh) -- identical octree to the on-demand descent, but
  // ordered so each level's new nodes are sampled together (one GPU kernel per level).
  void build(const std::vector<TriFeat>&feats) {
    for(int ax=0;ax<3;ax++) fs[ax]=domainSize[ax]/(double)(baseGrid[ax]<<(nLvls-1));
    std::vector<std::array<double,3>> pts; pts.reserve(feats.size()*4);
    for(const auto&f:feats){
      pts.push_back({f.v0.x,f.v0.y,f.v0.z}); pts.push_back({f.v1.x,f.v1.y,f.v1.z}); pts.push_back({f.v2.x,f.v2.y,f.v2.z});
      float3 c=(f.v0+f.v1+f.v2)*(1.0f/3.0f); pts.push_back({c.x,c.y,c.z});
    }
    deferring=true;
    for(int k=0;k<baseGrid[2];k++)for(int j=0;j<baseGrid[1];j++)for(int i=0;i<baseGrid[0];i++) makeCell(0,i,j,k);
    sampleNewNodesGpu();
    std::vector<int> pc(pts.size());                    // each point's current leaf cell
    for(size_t p=0;p<pts.size();p++){
      int ri=(int)floor(pts[p][0]/cs(0,0)),rj=(int)floor(pts[p][1]/cs(0,1)),rk=(int)floor(pts[p][2]/cs(0,2));
      pc[p]=(ri<0||rj<0||rk<0||ri>=baseGrid[0]||rj>=baseGrid[1]||rk>=baseGrid[2])?-1:idxOf(0,ri,rj,rk);
    }
    for(int L=0;L<nLvls-1;L++){
      int ncells=(int)cells.size();
      std::vector<char> flag(ncells,0);
      for(size_t p=0;p<pts.size();p++){ int ci=pc[p]; if(ci<0||ci>=ncells) continue;
        if(cells[ci].lvl!=L||cells[ci].firstChild>=0) continue;
        if(fabs(trilin(ci,pts[p][0],pts[p][1],pts[p][2]))>thresh) flag[ci]=1; }
      bool any=false;
      for(int ci=0;ci<ncells;ci++) if(flag[ci]&&cells[ci].firstChild<0){ refine(ci); any=true; }
      if(any) sampleNewNodesGpu();
      for(size_t p=0;p<pts.size();p++){ int ci=pc[p]; if(ci<0) continue;
        if(cells[ci].firstChild>=0) pc[p]=childContaining(ci,pts[p][0],pts[p][1],pts[p][2]); }
    }
    deferring=false;                                    // crack-reduce + DC sample the few new nodes on host
  }

  // close grazed-transition cracks with (almost) no extra oracle: a non-straddling
  // leaf whose corners are all one sign, but where an ALREADY-SAMPLED finer node on
  // its boundary flips the sign, hides a crossing -> split it (the shared finer nodes
  // are reused; only the genuinely-new interior nodes are sampled).  Iterated to a
  // fixpoint so it matches the finest neighbour at the surface.
  long reduceCracks() {
    long before = oracleCalls; bool changed=true; int passes=0;
    while (changed && passes < 2*nLvls) {
      changed=false; passes++;
      int n=(int)cells.size();
      for (int ci=0;ci<n;ci++){
        if (cells[ci].firstChild>=0) continue;
        Cell c=cells[ci];
        if (c.lvl>=nLvls-1) continue;
        bool anyNeg=false,anyPos=false;
        for(int q=0;q<8;q++){ if(nval[c.corner[q]]<0)anyNeg=true; else anyPos=true; }
        if (anyNeg && anyPos) continue;           // straddles -> has a vertex, not a crack source
        bool cNeg=anyNeg; int s=(nLvls-1)-(c.lvl+1); bool flip=false;
        for(int a=0;a<3 && !flip;a++)for(int b=0;b<3 && !flip;b++)for(int d=0;d<3 && !flip;d++){
          if((a&1)==0 && (b&1)==0 && (d&1)==0) continue;   // a corner of C, not a sub-node
          auto it=nmap.find(nkey((2*c.i+a)<<s,(2*c.j+b)<<s,(2*c.k+d)<<s));
          if(it==nmap.end()) continue;            // not sampled -> don't sample (minimal oracle)
          if((nval[it->second]<0)!=cNeg) flip=true;
        }
        if(flip){ refine(ci); changed=true; }
      }
    }
    return oracleCalls - before;
  }

  // ---- dual contour -------------------------------------------------------
  bool exists(int l,int i,int j,int k) const { return cmap.count(ckey(l,i,j,k))>0; }
  int  idxOf (int l,int i,int j,int k) const { auto it=cmap.find(ckey(l,i,j,k)); return it==cmap.end()?-1:it->second; }
  bool isLeaf(int l,int i,int j,int k) const { int c=idxOf(l,i,j,k); return c>=0 && cells[c].firstChild<0; }
  int  vidOf (int l,int i,int j,int k) const { int c=idxOf(l,i,j,k); return c<0?-1:cvid[c]; }

  void computeVertices() {
    cvid.assign(cells.size(),-1); cmask.assign(cells.size(),0); dverts.clear();
    for(int ci=0;ci<(int)cells.size();ci++){
      if(cells[ci].firstChild>=0) continue;
      Cell &c=cells[ci];
      double csx=cs(c.lvl,0),csy=cs(c.lvl,1),csz=cs(c.lvl,2);
      float cv[8]; unsigned char m=0;
      for(int q=0;q<8;q++){ cv[q]=nval[c.corner[q]]; if(cv[q]<0) m|=(1<<q); }
      cmask[ci]=m;
      if(m==0||m==0xFF) continue;
      std::vector<std::array<double,3>> Ns,Ps; double mass[3]={0,0,0}; int cnt=0;
      const int E[12][2]={{0,4},{2,6},{1,5},{3,7},{0,2},{4,6},{1,3},{5,7},{0,1},{4,5},{2,3},{6,7}};
      for(int e=0;e<12;e++){
        int c0=E[e][0],c1=E[e][1]; float va=cv[c0],vb=cv[c1];
        if((va<0)==(vb<0)) continue;
        double pa[3]={(c.i+(c0>>2))*csx,(c.j+((c0>>1)&1))*csy,(c.k+(c0&1))*csz};
        double pb[3]={(c.i+(c1>>2))*csx,(c.j+((c1>>1)&1))*csy,(c.k+(c1&1))*csz};
        double t=va/(double)(va-vb);
        double pp[3]={pa[0]+t*(pb[0]-pa[0]),pa[1]+t*(pb[1]-pa[1]),pa[2]+t*(pb[2]-pa[2])};
        double u=pp[0]/csx-c.i,v=pp[1]/csy-c.j,w=pp[2]/csz-c.k;          // trilinear FD normal
        double gx=0,gy=0,gz=0;
        for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int d=0;d<2;d++){
          double val=cv[a*4+b*2+d];
          gx+=val*(a?1:-1)*(b?v:1-v)*(d?w:1-w);
          gy+=val*(a?u:1-u)*(b?1:-1)*(d?w:1-w);
          gz+=val*(a?u:1-u)*(b?v:1-v)*(d?1:-1);
        }
        gx/=csx;gy/=csy;gz/=csz; double gn=sqrt(gx*gx+gy*gy+gz*gz)+1e-20;
        Ns.push_back({gx/gn,gy/gn,gz/gn}); Ps.push_back({pp[0],pp[1],pp[2]});
        mass[0]+=pp[0];mass[1]+=pp[1];mass[2]+=pp[2]; cnt++;
      }
      if(!cnt) continue;
      double ctr[3]={mass[0]/cnt,mass[1]/cnt,mass[2]/cnt}, x[3];
      qefSolve(Ns,Ps,ctr,x);
      double lo[3]={c.i*csx,c.j*csy,c.k*csz};
      for(int a=0;a<3;a++) x[a]=std::fmin(std::fmax(x[a],lo[a]-(a==0?csx:a==1?csy:csz)),lo[a]+2*(a==0?csx:a==1?csy:csz));
      cvid[ci]=(int)dverts.size(); dverts.push_back({x[0],x[1],x[2]});
    }
  }
  void emit(int idx[4], bool flip) {
    int t[2][3];
    if(flip){t[0][0]=idx[0];t[0][1]=idx[3];t[0][2]=idx[1]; t[1][0]=idx[0];t[1][1]=idx[2];t[1][2]=idx[3];}
    else    {t[0][0]=idx[0];t[0][1]=idx[1];t[0][2]=idx[3]; t[1][0]=idx[0];t[1][1]=idx[3];t[1][2]=idx[2];}
    for(int r=0;r<2;r++){ if(t[r][0]==t[r][1]||t[r][1]==t[r][2]||t[r][0]==t[r][2])continue;
      dtris.push_back({t[r][0],t[r][1],t[r][2]}); }
  }
  // n[] = 4 cells (lvl,i,j,k) around a minimal edge (raster order); processEdge etc.
  struct C4 { int l,i,j,k; };
  void processEdge(const C4 n[4],int dir) {
    int minLvl=-1,mi=0,idx[4];
    for(int q=0;q<4;q++){ idx[q]=vidOf(n[q].l,n[q].i,n[q].j,n[q].k); if(n[q].l>minLvl){minLvl=n[q].l;mi=q;} }
    if(idx[0]<0||idx[1]<0||idx[2]<0||idx[3]<0){
      int e=processEdgeMask[dir][mi], cc=idxOf(n[mi].l,n[mi].i,n[mi].j,n[mi].k);
      bool sc = cc>=0 && ((cmask[cc]>>edgevmap[e][0])&1)!=((cmask[cc]>>edgevmap[e][1])&1);
      if(sc){
        if(collecting){                          // record the non-straddling leaf(s) for splitting
          for(int q=0;q<4;q++) if(idx[q]<0){
            int cq=idxOf(n[q].l,n[q].i,n[q].j,n[q].k);
            if(cq>=0 && cells[cq].firstChild<0 && cells[cq].lvl<nLvls-1) toRefine.push_back(cq);
          }
        } else nUnclosed++;
      }
      return;
    }
    int e=processEdgeMask[dir][mi],cc=idxOf(n[mi].l,n[mi].i,n[mi].j,n[mi].k);
    int m0=(cmask[cc]>>edgevmap[e][0])&1, m1=(cmask[cc]>>edgevmap[e][1])&1;
    if(m0==m1) return;
    if(!collecting) emit(idx, m0==1);
  }
  C4 child(const C4 &c,int nn) const { return { c.l+1, 2*c.i+((nn>>2)&1), 2*c.j+((nn>>1)&1), 2*c.k+(nn&1) }; }
  void edgeProc(const C4 n[4],int dir) {
    if(!exists(n[0].l,n[0].i,n[0].j,n[0].k)||!exists(n[1].l,n[1].i,n[1].j,n[1].k)||
       !exists(n[2].l,n[2].i,n[2].j,n[2].k)||!exists(n[3].l,n[3].i,n[3].j,n[3].k)) return;
    bool L0=isLeaf(n[0].l,n[0].i,n[0].j,n[0].k),L1=isLeaf(n[1].l,n[1].i,n[1].j,n[1].k),
         L2=isLeaf(n[2].l,n[2].i,n[2].j,n[2].k),L3=isLeaf(n[3].l,n[3].i,n[3].j,n[3].k);
    if(L0&&L1&&L2&&L3){ processEdge(n,dir); return; }
    for(int s=0;s<2;s++){ C4 e[4];
      for(int q=0;q<4;q++){ bool lf=isLeaf(n[q].l,n[q].i,n[q].j,n[q].k); e[q]=lf?n[q]:child(n[q],edgeProcEdgeMask[dir][s][q]); }
      edgeProc(e,edgeProcEdgeMask[dir][s][4]); }
  }
  void faceProc(const C4 n[2],int dir) {
    if(!exists(n[0].l,n[0].i,n[0].j,n[0].k)||!exists(n[1].l,n[1].i,n[1].j,n[1].k)) return;
    if(isLeaf(n[0].l,n[0].i,n[0].j,n[0].k)&&isLeaf(n[1].l,n[1].i,n[1].j,n[1].k)) return;
    for(int s=0;s<4;s++){ C4 f[2];
      for(int q=0;q<2;q++){ int c=faceProcFaceMask[dir][s][q]; f[q]=isLeaf(n[q].l,n[q].i,n[q].j,n[q].k)?n[q]:child(n[q],c); }
      faceProc(f,faceProcFaceMask[dir][s][2]); }
    for(int s=0;s<4;s++){ const int *m=faceProcEdgeMask[dir][s]; const int *ord=faceProcOrders[m[0]]; C4 e[4];
      for(int q=0;q<4;q++){ e[q]=isLeaf(n[ord[q]].l,n[ord[q]].i,n[ord[q]].j,n[ord[q]].k)?n[ord[q]]:child(n[ord[q]],m[1+q]); }
      edgeProc(e,m[5]); }
  }
  void cellProc(const C4 &c) {
    if(!exists(c.l,c.i,c.j,c.k)||isLeaf(c.l,c.i,c.j,c.k)) return;
    C4 ch[8]; for(int q=0;q<8;q++) ch[q]=child(c,q);
    for(int q=0;q<8;q++) cellProc(ch[q]);
    for(int q=0;q<12;q++){ C4 f[2]={ch[cellProcFaceMask[q][0]],ch[cellProcFaceMask[q][1]]}; faceProc(f,cellProcFaceMask[q][2]); }
    for(int q=0;q<6;q++){ C4 e[4]={ch[cellProcEdgeMask[q][0]],ch[cellProcEdgeMask[q][1]],ch[cellProcEdgeMask[q][2]],ch[cellProcEdgeMask[q][3]]}; edgeProc(e,cellProcEdgeMask[q][4]); }
  }
  void runForest() {
    int nx=baseGrid[0],ny=baseGrid[1],nz=baseGrid[2];
    for(int K=0;K<nz;K++)for(int J=0;J<ny;J++)for(int I=0;I<nx;I++) cellProc({0,I,J,K});
    for(int K=0;K<nz;K++)for(int J=0;J<ny;J++)for(int I=0;I<nx;I++){
      if(I+1<nx){C4 f[2]={{0,I,J,K},{0,I+1,J,K}};faceProc(f,0);}
      if(J+1<ny){C4 f[2]={{0,I,J,K},{0,I,J+1,K}};faceProc(f,1);}
      if(K+1<nz){C4 f[2]={{0,I,J,K},{0,I,J,K+1}};faceProc(f,2);}
    }
    for(int K=0;K<nz;K++)for(int J=1;J<ny;J++)for(int I=1;I<nx;I++){ C4 e[4]={{0,I-1,J-1,K},{0,I-1,J,K},{0,I,J-1,K},{0,I,J,K}}; edgeProc(e,2); }
    for(int J=0;J<ny;J++)for(int K=1;K<nz;K++)for(int I=1;I<nx;I++){ C4 e[4]={{0,I-1,J,K-1},{0,I,J,K-1},{0,I-1,J,K},{0,I,J,K}}; edgeProc(e,1); }
    for(int I=0;I<nx;I++)for(int K=1;K<nz;K++)for(int J=1;J<ny;J++){ C4 e[4]={{0,I,J-1,K-1},{0,I,J-1,K},{0,I,J,K-1},{0,I,J,K}}; edgeProc(e,0); }
  }
  // Iterate: contour, split every leaf that lacks a vertex at a sign-change edge
  // (the exact crack cells), and re-contour, to a fixpoint -- then emit.  Splitting
  // reuses the already-sampled finer nodes; only new interior nodes cost an oracle.
  long dualContour() {
    long before = oracleCalls;
    for(int it=0; it<nLvls+2; it++){
      computeVertices();
      collecting=true; toRefine.clear();
      runForest();
      collecting=false;
      if(toRefine.empty()) break;
      std::sort(toRefine.begin(),toRefine.end());
      toRefine.erase(std::unique(toRefine.begin(),toRefine.end()),toRefine.end());
      for(int cq:toRefine) if(cells[cq].firstChild<0) refine(cq);
    }
    dtris.clear(); nUnclosed=0;
    computeVertices();
    runForest();                                  // collecting=false -> emit triangles + count residual
    return oracleCalls - before;
  }

  // GPU dual contour: same crack fixpoint, but each pass runs the QEF + minimal-edge
  // connectivity as kernels over uploaded SoA cells.  Splits happen on the host (few
  // cells) between passes; the cells/nodes are re-uploaded.
  long dualContourGpu() {
    long before=oracleCalls;
    int cap=(int)cells.size()+(int)cells.size()/8+200000;
    int nodeCap=(int)nval.size()+(int)nval.size()/8+200000;
    int ccap=1; while(ccap<(int)(2.5*cap)) ccap<<=1;
    int maxVert=cap, maxTri=3*cap, maxOpen=cap;
    int *dlvl,*di,*dj,*dk,*dcorner,*dchild,*dcvid,*dtriA,*dopenA,*dcnt; unsigned char*dcmask;
    float*dnval,*dvertX; u64*dckeys; i32*dcvals;
    cudaMalloc(&dlvl,cap*sizeof(int));cudaMalloc(&di,cap*sizeof(int));cudaMalloc(&dj,cap*sizeof(int));cudaMalloc(&dk,cap*sizeof(int));
    cudaMalloc(&dcorner,8*(size_t)cap*sizeof(int));cudaMalloc(&dchild,cap*sizeof(int));cudaMalloc(&dcvid,cap*sizeof(int));cudaMalloc(&dcmask,cap);
    cudaMalloc(&dnval,(size_t)nodeCap*sizeof(float));
    cudaMalloc(&dckeys,(size_t)ccap*sizeof(u64));cudaMalloc(&dcvals,(size_t)ccap*sizeof(i32));
    cudaMalloc(&dvertX,3*(size_t)maxVert*sizeof(float));cudaMalloc(&dtriA,3*(size_t)maxTri*sizeof(int));cudaMalloc(&dopenA,(size_t)maxOpen*sizeof(int));
    cudaMalloc(&dcnt,3*sizeof(int));
    std::vector<int> hl,hi,hj,hk,hcorner,hchild;
    NoDcP p{}; p.clvl=dlvl;p.ci=di;p.cj=dj;p.ck=dk;p.ccorner=dcorner;p.cchild=dchild; p.nval=dnval;
    p.ckeys=dckeys;p.cvals=dcvals;p.ccap=ccap; p.cvid=dcvid;p.cmask=dcmask;
    p.vertX=dvertX;p.triA=dtriA;p.openA=dopenA;p.cnt=dcnt; p.maxVert=maxVert;p.maxTri=maxTri;p.maxOpen=maxOpen;
    p.dsx=domainSize[0];p.dsy=domainSize[1];p.dsz=domainSize[2]; p.bgx=baseGrid[0];p.bgy=baseGrid[1];p.bgz=baseGrid[2]; p.nLvls=nLvls;
    auto upload=[&](){ int nC=(int)cells.size();
      hl.resize(nC);hi.resize(nC);hj.resize(nC);hk.resize(nC);hcorner.resize(8*nC);hchild.resize(nC);
      for(int c=0;c<nC;c++){ auto&cc=cells[c]; hl[c]=cc.lvl;hi[c]=cc.i;hj[c]=cc.j;hk[c]=cc.k;hchild[c]=cc.firstChild; for(int q=0;q<8;q++)hcorner[8*c+q]=cc.corner[q]; }
      cudaMemcpy(dlvl,hl.data(),nC*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(di,hi.data(),nC*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dj,hj.data(),nC*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dk,hk.data(),nC*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dcorner,hcorner.data(),8*(size_t)nC*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dchild,hchild.data(),nC*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dnval,nval.data(),nval.size()*sizeof(float),cudaMemcpyHostToDevice); p.nCells=nC; };
    auto runPass=[&](int mode){ int blk=(p.nCells+255)/256;
      cudaMemset(dckeys,0xFF,(size_t)ccap*sizeof(u64)); noHashKernel<<<blk,256>>>(p);
      int z3[3]={0,0,0}; cudaMemcpy(dcnt,z3,3*sizeof(int),cudaMemcpyHostToDevice);
      noQefKernel<<<blk,256>>>(p); p.mode=mode; noConnKernel<<<blk,256>>>(p);
      int cnt[3]; cudaMemcpy(cnt,dcnt,3*sizeof(int),cudaMemcpyDeviceToHost); return std::array<int,3>{cnt[0],cnt[1],cnt[2]}; };
    for(int pass=0; pass<nLvls+2 && (int)cells.size()<cap && (int)nval.size()<nodeCap; pass++){
      upload(); auto cnt=runPass(1);
      if(cnt[2]==0) break;
      std::vector<int> op(std::min(cnt[2],maxOpen)); cudaMemcpy(op.data(),dopenA,op.size()*sizeof(int),cudaMemcpyDeviceToHost);
      std::sort(op.begin(),op.end()); op.erase(std::unique(op.begin(),op.end()),op.end());
      for(int o:op) if(o<(int)cells.size()&&cells[o].firstChild<0) refine(o);   // host split (samples few new nodes)
    }
    upload(); auto cnt=runPass(0);
    int nV=cnt[0]<maxVert?cnt[0]:maxVert, nT=cnt[1]<maxTri?cnt[1]:maxTri;
    std::vector<float> vx(3*nV); cudaMemcpy(vx.data(),dvertX,3*(size_t)nV*sizeof(float),cudaMemcpyDeviceToHost);
    std::vector<int> tx(3*nT); cudaMemcpy(tx.data(),dtriA,3*(size_t)nT*sizeof(int),cudaMemcpyDeviceToHost);
    dverts.resize(nV); for(int v=0;v<nV;v++) dverts[v]={vx[3*v],vx[3*v+1],vx[3*v+2]};
    dtris.resize(nT);  for(int t=0;t<nT;t++) dtris[t]={tx[3*t],tx[3*t+1],tx[3*t+2]};
    cudaFree(dlvl);cudaFree(di);cudaFree(dj);cudaFree(dk);cudaFree(dcorner);cudaFree(dchild);cudaFree(dcvid);cudaFree(dcmask);
    cudaFree(dnval);cudaFree(dckeys);cudaFree(dcvals);cudaFree(dvertX);cudaFree(dtriA);cudaFree(dopenA);cudaFree(dcnt);
    return oracleCalls-before;
  }

  // ---- VTK UnstructuredGrid (.vtu, XML ascii) -----------------------------
  void writeGridVtu(const char *path) {
    std::vector<int> leaves; for(int ci=0;ci<(int)cells.size();ci++) if(cells[ci].firstChild<0) leaves.push_back(ci);
    std::ofstream o(path); o.precision(7);
    o<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n <UnstructuredGrid>\n";
    o<<"  <Piece NumberOfPoints=\""<<nval.size()<<"\" NumberOfCells=\""<<leaves.size()<<"\">\n";
    o<<"   <Points>\n    <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(size_t n=0;n<nval.size();n++)
      o<<"     "<<(origin[0]+nfijk[n][0]*fs[0])<<" "<<(origin[1]+nfijk[n][1]*fs[1])<<" "<<(origin[2]+nfijk[n][2]*fs[2])<<"\n";
    o<<"    </DataArray>\n   </Points>\n   <Cells>\n    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for(int ci:leaves){ o<<"     "; for(int v=0;v<8;v++) o<<cells[ci].corner[VTK_HEX[v]]<<" "; o<<"\n"; }
    o<<"    </DataArray>\n    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n     ";
    for(size_t c=0;c<leaves.size();c++) o<<(8*(c+1))<<" "; o<<"\n";
    o<<"    </DataArray>\n    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n     ";
    for(size_t c=0;c<leaves.size();c++) o<<"12 "; o<<"\n";
    o<<"    </DataArray>\n   </Cells>\n   <PointData Scalars=\"sdf\">\n    <DataArray type=\"Float32\" Name=\"sdf\" format=\"ascii\">\n     ";
    for(size_t n=0;n<nval.size();n++) o<<nval[n]<<" "; o<<"\n";
    o<<"    </DataArray>\n   </PointData>\n  </Piece>\n </UnstructuredGrid>\n</VTKFile>\n";
  }
  void writeDcVtu(const char *path) {
    std::ofstream o(path); o.precision(7);
    o<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n <UnstructuredGrid>\n";
    o<<"  <Piece NumberOfPoints=\""<<dverts.size()<<"\" NumberOfCells=\""<<dtris.size()<<"\">\n";
    o<<"   <Points>\n    <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(auto&v:dverts) o<<"     "<<(origin[0]+v[0])<<" "<<(origin[1]+v[1])<<" "<<(origin[2]+v[2])<<"\n";
    o<<"    </DataArray>\n   </Points>\n   <Cells>\n    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for(auto&t:dtris) o<<"     "<<t[0]<<" "<<t[1]<<" "<<t[2]<<"\n";
    o<<"    </DataArray>\n    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n     ";
    for(size_t c=0;c<dtris.size();c++) o<<(3*(c+1))<<" "; o<<"\n";
    o<<"    </DataArray>\n    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n     ";
    for(size_t c=0;c<dtris.size();c++) o<<"5 "; o<<"\n";
    o<<"    </DataArray>\n   </Cells>\n  </Piece>\n </UnstructuredGrid>\n</VTKFile>\n";
  }
};

} // namespace

// build the host nodal octree, dual contour it, write <name>_grid.vtu + <name>_dc.vtu.
void runNodalOctree(const std::vector<TriFeat> &feats, const BvhNode *bnodes, int nBvhNodes,
                    const i32 *border, real orient, const double origin[3], const double domainSize[3],
                    const int baseGrid[3], int nLvls, double thresh, const char *name) {
  NodalOctree oc;
  oc.bnodes=bnodes; oc.border=border; oc.btris=feats.data(); oc.orient=orient;
  for(int d=0;d<3;d++){ oc.origin[d]=origin[d]; oc.domainSize[d]=domainSize[d]; oc.baseGrid[d]=baseGrid[d]; }
  oc.nLvls=nLvls; oc.thresh=thresh;
  oc.setupGpu(bnodes, nBvhNodes, (int)feats.size());

  auto clk=[]{ return std::chrono::steady_clock::now(); };
  auto ms=[](auto a,auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

  auto t0=clk(); oc.build(feats);                        long nbBuild=oc.oracleCalls;
  auto t1=clk(); long crA=0;                             // reduceCracks subsumed by the GPU crack fixpoint
  auto t2=clk(); long crB=oc.dualContourGpu();
  auto t3=clk();
  std::string g=std::string("output/")+name+"_grid.vtu", d=std::string("output/")+name+"_dc.vtu";
  oc.writeGridVtu(g.c_str());
  auto t4=clk(); oc.writeDcVtu(d.c_str());
  auto t5=clk();

  long leaves=0; for(auto&c:oc.cells) if(c.firstChild<0) leaves++;
  printf("  timing:  build(refine+sample) %.0f ms  reduceCracks %.0f ms  dualContour %.0f ms"
         "  write_grid %.0f ms  write_dc %.0f ms\n", ms(t0,t1),ms(t1,t2),ms(t2,t3),ms(t3,t4),ms(t4,t5));
  printf("  oracle:  build %ld  +crack %ld (reduce %ld, contour %ld)\n", nbBuild, crA+crB, crA, crB);
  printf("  nodal octree: %zu nodes (%ld oracle calls), %zu cells (%ld leaves)\n",
         oc.nval.size(), oc.oracleCalls, oc.cells.size(), leaves);
  printf("  dc: %zu vertices, %zu triangles%s -> %s , %s\n",
         oc.dverts.size(), oc.dtris.size(),
         oc.nUnclosed? (" ["+std::to_string(oc.nUnclosed)+" open edges]").c_str():"", g.c_str(), d.c_str());
  oc.teardownGpu();
}
