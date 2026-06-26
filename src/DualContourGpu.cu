#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include "DualContourGpu.cuh"

static constexpr u64 DC_EMPTY = 0xFFFFFFFFFFFFFFFFULL;

// cell key includes the level (leaf cells live at mixed levels)
__host__ __device__ inline u64 dcCellKey(i32 lvl, i32 i, i32 j, i32 k) {
  return ((u64)lvl << 60) | ((u64)(u32)(k+1) << 40) | ((u64)(u32)(j+1) << 20) | (u64)(u32)(i+1);
}
__device__ inline u64 dcMurmur(u64 x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x;
}
__device__ inline void dcHashPut(u64 *keys, i32 *vals, int cap, u64 key, i32 val) {
  u64 h = dcMurmur(key) & (u64)(cap - 1);                 // each cell is unique -> no value race
  for (;;) {
    u64 prev = atomicCAS((unsigned long long*)&keys[h], (unsigned long long)DC_EMPTY,
                         (unsigned long long)key);
    if (prev == DC_EMPTY || prev == key) { vals[h] = val; return; }
    h = (h + 1) & (u64)(cap - 1);
  }
}
__device__ inline i32 dcHashGet(const u64 *keys, const i32 *vals, int cap, u64 key) {
  u64 h = dcMurmur(key) & (u64)(cap - 1);
  for (;;) {
    u64 kk = keys[h];
    if (kk == key)      return vals[h];
    if (kk == DC_EMPTY) return -1;
    h = (h + 1) & (u64)(cap - 1);
  }
}

// the 12 cell edges as pairs of corner indices (corner ci = a*4+b*2+d)
__constant__ int DC_EDGES[12][2] = {
  {0,4},{2,6},{1,5},{3,7}, {0,2},{4,6},{1,3},{5,7}, {0,1},{4,5},{2,3},{6,7}
};

// device copies of the canonical adaptive-DC edge tables (see DcHost) for the flat
// GPU connectivity kernel.  dcEdgevmapD: the 12 edges (x,y,z groups); dcProcEdgeD:
// per direction the 4 cells' edge index; dcQcD[e]: which of the 4 canonical cells
// around edge e is the OWNER cell itself.
__constant__ int dcEdgevmapD[12][2] = {
  {0,4},{1,5},{2,6},{3,7}, {0,2},{1,3},{4,6},{5,7}, {0,1},{2,3},{4,5},{6,7} };
__constant__ int dcProcEdgeD[3][4] = { {3,2,1,0},{7,5,6,4},{11,10,9,8} };
__constant__ int dcQcD[12] = { 3,2,1,0, 3,1,2,0, 3,2,1,0 };

// a leaf cell's block has no children (its child block is absent), or is finest.
__device__ inline bool dcIsLeaf(WaveletSdfSolver &grid, i32 lvl, i32 ib, i32 jb, i32 kb) {
  if (lvl == grid.nLvls - 1) return true;
  return grid.hashTable.getValue(grid.encode(lvl+1, 2*ib, 2*jb, 2*kb)) == bEmpty;
}

// One thread per cell.  For each LEAF cell whose 8 corner nodes (at the cell's own
// level) are all stored and straddle the surface, place its QEF vertex from the
// corner crossings, and register (level,cell key -> vertex id).  STORED_GRAD picks
// the edge normal: true = interpolate the stored corner gradients (gradient / "true
// Hermite" DC); false = trilinear finite difference of the corner VALUES (Carrera,
// SDF values only -- needs no stored gradients).
template<bool STORED_GRAD>
__global__ void dcVertexKernelT(WaveletSdfSolver &grid, DcGpuParams p) {
  START_CELL_LOOP
    GET_CELL_INDICES
    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);
    if (grid.isInteriorBlock(lvl, ib, jb, kb) && dcIsLeaf(grid, lvl, ib, jb, kb)) {
      i32 I = ib*blockSize + i, J = jb*blockSize + j, K = kb*blockSize + k;   // cell index at `lvl`
      float hx = grid.getDx(lvl), hy = grid.getDy(lvl), hz = grid.getDz(lvl);
      const real *S = grid.Sdf + (size_t)bIdx*nodeSizeTot;    // 8 corners are LOCAL to this block
      float cv[8]; float3 cg[8]; unsigned char mask = 0;
      for (int c=0;c<8;c++) {
        int a=c>>2, b=(c>>1)&1, d=c&1;
        float v = S[WaveletSdfSolver::nodeIdx(i+a, j+b, k+d)];
        cv[c]=v; if (v < 0.0f) mask |= (unsigned char)(1<<c);
      }
      if (STORED_GRAD) for (int c=0;c<8;c++) {                // finite-diff corner gradients
        int a=c>>2, b=(c>>1)&1, d=c&1;
        cg[c] = make_float3((cv[4+(b*2+d)]-cv[(b*2+d)])/hx,
                            (cv[(a*4)+2+d]-cv[(a*4)+d])/hy,
                            (cv[(a*4)+(b*2)+1]-cv[(a*4)+(b*2)])/hz);
      }
      if (mask != 0 && mask != 0xFF) {
        double ata[6]={0,0,0,0,0,0}, atb[3]={0,0,0}, mass[3]={0,0,0}; int cnt=0;
        for (int e=0;e<12;e++) {
          int c0=DC_EDGES[e][0], c1=DC_EDGES[e][1];
          float va=cv[c0], vb=cv[c1];
          if ((va<0.0f)==(vb<0.0f)) continue;
          float3 pa = make_float3((I+(c0>>2))*hx, (J+((c0>>1)&1))*hy, (K+(c0&1))*hz);
          float3 pb = make_float3((I+(c1>>2))*hx, (J+((c1>>1)&1))*hy, (K+(c1&1))*hz);
          float t = va/(va-vb);
          float3 pp = pa + (pb-pa)*t;
          float3 nrm;
          if (STORED_GRAD) {
            nrm = cg[c0] + (cg[c1]-cg[c0])*t;             // interp stored corner gradients
          } else {
            float uu=pp.x/hx-I, vv=pp.y/hy-J, ww=pp.z/hz-K;   // trilinear FD gradient of values
            float gx=0,gy=0,gz=0;
            for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int d=0;d<2;d++){
              float val=cv[a*4+b*2+d];
              gx += val*(a?1:-1)*(b?vv:1-vv)*(d?ww:1-ww);
              gy += val*(a?uu:1-uu)*(b?1:-1)*(d?ww:1-ww);
              gz += val*(a?uu:1-uu)*(b?vv:1-vv)*(d?1:-1);
            }
            gx/=hx; gy/=hy; gz/=hz;
            float gn=sqrtf(gx*gx+gy*gy+gz*gz)+1e-20f;
            nrm=make_float3(gx/gn,gy/gn,gz/gn);
          }
          double n0=nrm.x,n1=nrm.y,n2=nrm.z, dd=n0*pp.x+n1*pp.y+n2*pp.z;
          ata[0]+=n0*n0; ata[1]+=n0*n1; ata[2]+=n0*n2; ata[3]+=n1*n1; ata[4]+=n1*n2; ata[5]+=n2*n2;
          atb[0]+=n0*dd; atb[1]+=n1*dd; atb[2]+=n2*dd;
          mass[0]+=pp.x; mass[1]+=pp.y; mass[2]+=pp.z; cnt++;
        }
        if (cnt > 0) {
          double c[3]={mass[0]/cnt, mass[1]/cnt, mass[2]/cnt};
          double lam = 1e-3*(ata[0]+ata[3]+ata[5])/3.0 + 1e-9;
          double M[6]={ata[0]+lam,ata[1],ata[2],ata[3]+lam,ata[4],ata[5]+lam};
          double r[3]={atb[0]+lam*c[0], atb[1]+lam*c[1], atb[2]+lam*c[2]};
          double det = M[0]*(M[3]*M[5]-M[4]*M[4]) - M[1]*(M[1]*M[5]-M[4]*M[2]) + M[2]*(M[1]*M[4]-M[3]*M[2]);
          double x[3];
          if (fabs(det) < 1e-18) { x[0]=c[0]; x[1]=c[1]; x[2]=c[2]; }
          else {
            double iv[6];
            iv[0]=(M[3]*M[5]-M[4]*M[4])/det; iv[1]=(M[2]*M[4]-M[1]*M[5])/det; iv[2]=(M[1]*M[4]-M[2]*M[3])/det;
            iv[3]=(M[0]*M[5]-M[2]*M[2])/det; iv[4]=(M[2]*M[1]-M[0]*M[4])/det; iv[5]=(M[0]*M[3]-M[1]*M[1])/det;
            x[0]=iv[0]*r[0]+iv[1]*r[1]+iv[2]*r[2];
            x[1]=iv[1]*r[0]+iv[3]*r[1]+iv[4]*r[2];
            x[2]=iv[2]*r[0]+iv[4]*r[1]+iv[5]*r[2];
          }
          double lo[3]={I*(double)hx, J*(double)hy, K*(double)hz};
          double he[3]={hx,hy,hz};
          for (int a=0;a<3;a++) x[a] = fmin(fmax(x[a], lo[a]), lo[a]+he[a]);
          int vid = atomicAdd(p.vertCount, 1);
          if (vid < p.maxVerts) {
            p.vertexArray[3*vid]   = (float)(x[0]+p.origin[0]);
            p.vertexArray[3*vid+1] = (float)(x[1]+p.origin[1]);
            p.vertexArray[3*vid+2] = (float)(x[2]+p.origin[2]);
            p.vMask[vid] = mask;
            dcHashPut(p.hkeys, p.hvals, p.hcap, dcCellKey(lvl,I,J,K), vid);
          }
        }
      }
    }
  END_CELL_LOOP
}

// ===========================================================================
// Flat GPU minimal-edge connectivity -- the parallel equivalent of the host
// recursion below.  One thread per leaf cell; the cell emits the minimal sign-
// change edges it OWNS (owner = lowest-index leaf at the finest level among the 4
// cells around the edge), ascending to coarser covering leaves at 2:1 transitions.
// Produces the same triangles as the recursion, fully on the GPU.
// ---------------------------------------------------------------------------

// does a FINER cell (level L+1) cover position (I,J,K)?  -> edge not minimal at L
__device__ inline bool dcFinerExists(WaveletSdfSolver &g, i32 L, i32 I, i32 J, i32 K) {
  if (L >= g.nLvls-1) return false;
  i32 cib=(2*I)/blockSize, cjb=(2*J)/blockSize, ckb=(2*K)/blockSize;
  if (!g.isInteriorBlock(L+1, cib,cjb,ckb)) return false;
  return g.hashTable.getValue(g.encode(L+1, cib,cjb,ckb)) != bEmpty;
}
// is (I,J,K) an active level-L leaf (a candidate owner)?
__device__ inline bool dcPosLeaf(WaveletSdfSolver &g, i32 L, i32 I, i32 J, i32 K) {
  if (I<0||J<0||K<0) return false;
  i32 ib=I/blockSize, jb=J/blockSize, kb=K/blockSize;
  if (!g.isInteriorBlock(L, ib,jb,kb)) return false;
  if (g.hashTable.getValue(g.encode(L, ib,jb,kb)) == bEmpty) return false;
  return !dcFinerExists(g, L, I, J, K);
}
// vertex id of the leaf covering (I,J,K) at level L, ascending to coarser leaves
__device__ inline int dcAscendVid(const DcGpuParams &p, i32 L, i32 I, i32 J, i32 K) {
  for (i32 l=L; l>=0; l--){ i32 s=L-l;
    i32 v=dcHashGet(p.hkeys,p.hvals,p.hcap, dcCellKey(l, I>>s, J>>s, K>>s)); if(v>=0) return v; }
  return -1;
}
__device__ inline bool dcLowerIdx(i32 ax,i32 ay,i32 az, i32 bx,i32 by,i32 bz) {
  if (az!=bz) return az<bz; if (ay!=by) return ay<by; return ax<bx;
}
__device__ inline void dcTri(DcGpuParams &p, int a, int b, int c) {
  if (a==b||b==c||a==c) return;                               // degenerate (T-junction fan)
  int t = atomicAdd(p.quadCount, 1);
  if (t >= p.maxQuads) return;
  p.quadArray[3*t]=a; p.quadArray[3*t+1]=b; p.quadArray[3*t+2]=c;
}

__global__ void dcQuadKernelGpu(WaveletSdfSolver &grid, DcGpuParams p, int *nUnclosed) {
  START_CELL_LOOP
    GET_CELL_INDICES
    i32 lvl, ib, jb, kb;
    grid.decode(grid.bLocList[bIdx], lvl, ib, jb, kb);
    if (grid.isInteriorBlock(lvl,ib,jb,kb) && dcIsLeaf(grid,lvl,ib,jb,kb)) {
      i32 I=ib*blockSize+i, J=jb*blockSize+j, K=kb*blockSize+k;
      int myVid = dcHashGet(p.hkeys,p.hvals,p.hcap, dcCellKey(lvl,I,J,K));
      if (myVid >= 0) {
        unsigned char m = p.vMask[myVid];
        for (int e=0;e<12;e++) {
          int c0=dcEdgevmapD[e][0], c1=dcEdgevmapD[e][1];
          if (((m>>c0)&1)==((m>>c1)&1)) continue;            // no sign change on this edge
          int dir=e/4;
          i32 px[4],py[4],pz[4];
          if (dir==0){ i32 by=(e>>1)&1, bz=e&1;
            px[0]=I; py[0]=J+by-1; pz[0]=K+bz-1;  px[1]=I; py[1]=J+by-1; pz[1]=K+bz;
            px[2]=I; py[2]=J+by;   pz[2]=K+bz-1;  px[3]=I; py[3]=J+by;   pz[3]=K+bz;
          } else if (dir==1){ i32 ee=e-4, bx=(ee>>1)&1, bz=ee&1;
            px[0]=I+bx-1; py[0]=J; pz[0]=K+bz-1;  px[1]=I+bx; py[1]=J; pz[1]=K+bz-1;
            px[2]=I+bx-1; py[2]=J; pz[2]=K+bz;    px[3]=I+bx; py[3]=J; pz[3]=K+bz;
          } else { i32 ee=e-8, bx=(ee>>1)&1, by=ee&1;
            px[0]=I+bx-1; py[0]=J+by-1; pz[0]=K;  px[1]=I+bx-1; py[1]=J+by; pz[1]=K;
            px[2]=I+bx;   py[2]=J+by-1; pz[2]=K;  px[3]=I+bx;   py[3]=J+by; pz[3]=K;
          }
          int qC=dcQcD[e];
          bool minimal=true, owner=true;
          for (int q=0;q<4 && minimal;q++) if (dcFinerExists(grid,lvl,px[q],py[q],pz[q])) minimal=false;
          if (!minimal) continue;
          for (int q=0;q<4;q++){                              // owner = lowest-index level-lvl leaf
            if (q==qC) continue;
            if (dcPosLeaf(grid,lvl,px[q],py[q],pz[q]) && dcLowerIdx(px[q],py[q],pz[q], I,J,K)) { owner=false; break; }
          }
          if (!owner) continue;
          int idx[4]; bool allv=true;
          for (int q=0;q<4;q++){ idx[q]=dcAscendVid(p,lvl,px[q],py[q],pz[q]); if(idx[q]<0) allv=false; }
          if (!allv) { atomicAdd(nUnclosed,1); continue; }    // a touching leaf has no vertex (sub-finest feature)
          int e2=dcProcEdgeD[dir][qC];
          bool flip = ((m>>dcEdgevmapD[e2][0])&1)==1;
          if (flip){ dcTri(p,idx[0],idx[3],idx[1]); dcTri(p,idx[0],idx[2],idx[3]); }
          else     { dcTri(p,idx[0],idx[1],idx[3]); dcTri(p,idx[0],idx[3],idx[2]); }
        }
      }
    }
  END_CELL_LOOP
}

// ===========================================================================
// Watertight ADAPTIVE connectivity: recursive minimal-edge dual contouring
// (Ju et al. 2002).  cellProc/faceProc/edgeProc descend the octree and visit
// every MINIMAL edge exactly once with its 4 distinct leaf cells, so a coarse
// cell's face stitches watertightly to its finer neighbours (T-junction fans).
// The expensive per-cell QEF vertices come from the GPU kernel above; this is
// pure topology (which cell vertices form each triangle), O(surface cells), so
// it runs on the host over the octree structure + the GPU-built vertex map.
// ---------------------------------------------------------------------------
// Corner index c = (x<<2)|(y<<1)|z, matching dcVertexKernel's mask; the tables
// below are the canonical adaptive-DC masks for that numbering.

namespace {
const int edgevmap[12][2] = {                              // edge -> its 2 corners
  {0,4},{1,5},{2,6},{3,7},  {0,2},{1,3},{4,6},{5,7},  {0,1},{2,3},{4,5},{6,7} };
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

struct DcCell { i32 lvl, I, J, K; };                       // an octree cell (global indices at lvl)

// a minimal sign-change edge as enumerated by the recursion: the 4 cells it joins
// (vertex ids), which is finest, its direction, the finest cell's coords (to locate
// the edge endpoints) and orientation.  Collected for the Carrera optimizer.
struct REdge { int v[4]; int mi, dir; i32 mlvl, mI, mJ, mK; bool flip; };

// host port of the recursion, holding the octree structure + GPU vertex outputs.
struct DcHost {
  WaveletSdfSolver *g;
  std::unordered_set<u64> blocks;                          // active block loc-codes (parent + leaf)
  std::unordered_map<u64,int> vmap;                        // cell key -> vertex id
  const unsigned char *vMask;
  std::vector<i32> tris;                                   // output triangle vertex ids
  bool carrera = false;                                    // collect edges instead of emitting tris
  std::vector<REdge> redges;                               // minimal sign-change edges (Carrera)

  u64 encBlk(i32 lvl,i32 ib,i32 jb,i32 kb) const {
    return ((u64)lvl<<60) | ((u64)(ib+1)) | ((u64)(jb+1)<<20) | ((u64)(kb+1)<<40);
  }
  bool interior(i32 lvl,i32 ib,i32 jb,i32 kb) const {
    i32 nx = g->baseGridSize[0]/blockSize*powi(2,lvl);
    i32 ny = g->baseGridSize[1]/blockSize*powi(2,lvl);
    i32 nz = g->pseudo2D ? g->baseGridSize[2]/blockSize : g->baseGridSize[2]/blockSize*powi(2,lvl);
    return ib>=0&&jb>=0&&kb>=0&&ib<nx&&jb<ny&&kb<nz;
  }
  bool blockActive(i32 lvl,i32 ib,i32 jb,i32 kb) const {
    return interior(lvl,ib,jb,kb) && blocks.count(encBlk(lvl,ib,jb,kb));
  }
  bool exists(const DcCell &c) const {
    return blockActive(c.lvl, c.I/blockSize, c.J/blockSize, c.K/blockSize);
  }
  bool refined(const DcCell &c) const {                    // child block (lvl+1) present?
    if (c.lvl >= g->nLvls-1) return false;
    return blockActive(c.lvl+1, (2*c.I)/blockSize, (2*c.J)/blockSize, (2*c.K)/blockSize);
  }
  bool isLeaf(const DcCell &c) const { return exists(c) && !refined(c); }
  DcCell child(const DcCell &c,int n) const {
    return { c.lvl+1, 2*c.I+((n>>2)&1), 2*c.J+((n>>1)&1), 2*c.K+(n&1) };
  }
  int vid(const DcCell &c) const {
    auto it = vmap.find(dcCellKey(c.lvl,c.I,c.J,c.K));
    return it==vmap.end() ? -1 : it->second;
  }

  long nUnclosed=0, nUnclosedUniform=0, nUnclosedFinest=0;   // sign-change minimal edges left open
  // emit the (up to) two triangles for a minimal edge shared by 4 leaf cells.
  void processEdge(const DcCell n[4], int dir) {
    int minLvl=-1, mi=0, idx[4];
    for (int i=0;i<4;i++){ idx[i]=vid(n[i]); if (n[i].lvl>minLvl){minLvl=n[i].lvl; mi=i;} }
    if (idx[0]<0||idx[1]<0||idx[2]<0||idx[3]<0) {
      int e2=processEdgeMask[dir][mi];
      if(idx[mi]>=0 && ((vMask[idx[mi]]>>edgevmap[e2][0])&1)!=((vMask[idx[mi]]>>edgevmap[e2][1])&1)){
        nUnclosed++;
        int lo=99,hi=-1; for(int i=0;i<4;i++){lo=lo<n[i].lvl?lo:n[i].lvl; hi=hi>n[i].lvl?hi:n[i].lvl;}
        if(lo==hi)          nUnclosedUniform++;   // a uniform-region crack would mean a connectivity bug
        if(hi>=g->nLvls-1)  nUnclosedFinest++;     // touches the finest level: a genuine sub-cell feature
      }
      return;    // a touching leaf has no vertex (surface finer than the finest cell here)
    }
    int e=processEdgeMask[dir][mi], c0=edgevmap[e][0], c1=edgevmap[e][1];
    int m0=(vMask[idx[mi]]>>c0)&1, m1=(vMask[idx[mi]]>>c1)&1;
    if (m0==m1) return;                                    // no sign change on the minimal edge
    bool flip = (m0==1);                                   // mask bit set = inside (cv<0)
    if (carrera) {                                         // Carrera: collect the edge, place verts later
      redges.push_back({{idx[0],idx[1],idx[2],idx[3]}, mi, dir,
                        n[mi].lvl, n[mi].I, n[mi].J, n[mi].K, flip});
      return;
    }
    int t[2][3];
    if (flip){ t[0][0]=idx[0];t[0][1]=idx[3];t[0][2]=idx[1]; t[1][0]=idx[0];t[1][1]=idx[2];t[1][2]=idx[3]; }
    else     { t[0][0]=idx[0];t[0][1]=idx[1];t[0][2]=idx[3]; t[1][0]=idx[0];t[1][1]=idx[3];t[1][2]=idx[2]; }
    for (int r=0;r<2;r++){
      if (t[r][0]==t[r][1]||t[r][1]==t[r][2]||t[r][0]==t[r][2]) continue;  // degenerate (T-junction)
      tris.push_back(t[r][0]); tris.push_back(t[r][1]); tris.push_back(t[r][2]);
    }
  }
  void edgeProc(const DcCell n[4], int dir) {
    if (!exists(n[0])||!exists(n[1])||!exists(n[2])||!exists(n[3])) return;
    if (isLeaf(n[0])&&isLeaf(n[1])&&isLeaf(n[2])&&isLeaf(n[3])) { processEdge(n,dir); return; }
    for (int i=0;i<2;i++){
      DcCell e[4];
      for (int j=0;j<4;j++) e[j] = isLeaf(n[j]) ? n[j] : child(n[j], edgeProcEdgeMask[dir][i][j]);
      edgeProc(e, edgeProcEdgeMask[dir][i][4]);
    }
  }
  void faceProc(const DcCell n[2], int dir) {
    if (!exists(n[0])||!exists(n[1])) return;
    if (isLeaf(n[0])&&isLeaf(n[1])) return;
    for (int i=0;i<4;i++){
      DcCell f[2];
      for (int j=0;j<2;j++){ int cc=faceProcFaceMask[dir][i][j]; f[j]=isLeaf(n[j])?n[j]:child(n[j],cc); }
      faceProc(f, faceProcFaceMask[dir][i][2]);
    }
    for (int i=0;i<4;i++){
      const int *m=faceProcEdgeMask[dir][i]; const int *ord=faceProcOrders[m[0]];
      DcCell e[4];
      for (int j=0;j<4;j++) e[j] = isLeaf(n[ord[j]]) ? n[ord[j]] : child(n[ord[j]], m[1+j]);
      edgeProc(e, m[5]);
    }
  }
  void cellProc(const DcCell &c) {
    if (!exists(c) || isLeaf(c)) return;
    DcCell ch[8]; for (int i=0;i<8;i++) ch[i]=child(c,i);
    for (int i=0;i<8;i++) cellProc(ch[i]);
    for (int i=0;i<12;i++){ DcCell f[2]={ch[cellProcFaceMask[i][0]],ch[cellProcFaceMask[i][1]]}; faceProc(f,cellProcFaceMask[i][2]); }
    for (int i=0;i<6;i++){ DcCell e[4]={ch[cellProcEdgeMask[i][0]],ch[cellProcEdgeMask[i][1]],ch[cellProcEdgeMask[i][2]],ch[cellProcEdgeMask[i][3]]}; edgeProc(e,cellProcEdgeMask[i][4]); }
  }
  // forest: the base grid is a dense N0^3 grid of level-0 cells.  Recurse into each
  // and stitch the shared faces/edges BETWEEN adjacent level-0 cells.
  void run() {
    i32 nx=g->baseGridSize[0], ny=g->baseGridSize[1], nz=g->baseGridSize[2];
    for (i32 K=0;K<nz;K++) for (i32 J=0;J<ny;J++) for (i32 I=0;I<nx;I++) cellProc({0,I,J,K});
    for (i32 K=0;K<nz;K++) for (i32 J=0;J<ny;J++) for (i32 I=0;I<nx;I++){
      if (I+1<nx){ DcCell f[2]={{0,I,J,K},{0,I+1,J,K}}; faceProc(f,0); }
      if (J+1<ny){ DcCell f[2]={{0,I,J,K},{0,I,J+1,K}}; faceProc(f,1); }
      if (K+1<nz){ DcCell f[2]={{0,I,J,K},{0,I,J,K+1}}; faceProc(f,2); }
    }
    for (i32 K=0;K<nz;K++) for (i32 J=1;J<ny;J++) for (i32 I=1;I<nx;I++){       // z-edges
      DcCell e[4]={{0,I-1,J-1,K},{0,I-1,J,K},{0,I,J-1,K},{0,I,J,K}}; edgeProc(e,2); }
    for (i32 J=0;J<ny;J++) for (i32 K=1;K<nz;K++) for (i32 I=1;I<nx;I++){       // y-edges
      DcCell e[4]={{0,I-1,J,K-1},{0,I,J,K-1},{0,I-1,J,K},{0,I,J,K}}; edgeProc(e,1); }
    for (i32 I=0;I<nx;I++) for (i32 K=1;K<nz;K++) for (i32 J=1;J<ny;J++){       // x-edges
      DcCell e[4]={{0,I,J-1,K-1},{0,I,J-1,K},{0,I,J,K-1},{0,I,J,K}}; edgeProc(e,0); }
  }
};
} // namespace

static int dcNextPow2(long v) { int p=1; while ((long)p < v) p <<= 1; return p; }

// host: build the octree query from the GPU vertex map, run the minimal-edge
// recursion, write the (shared) watertight triangle mesh + diagnostics.
static void dcEmitVtk(WaveletSdfSolver *solver, DcGpuParams &p, int nV,
                      const char *path, const char *label) {
  DcHost dc; dc.g = solver; dc.vMask = p.vMask;
  i32 nKeys = solver->hashTable.nKeys;
  dc.blocks.reserve(nKeys*2);
  for (i32 b=0;b<nKeys;b++) dc.blocks.insert(solver->bLocList[b]);
  dc.vmap.reserve(nV*2);
  for (int s=0;s<p.hcap;s++) if (p.hkeys[s]!=DC_EMPTY) dc.vmap[p.hkeys[s]] = p.hvals[s];
  dc.run();
  size_t nT = dc.tris.size()/3;
  std::ofstream os(path); os.precision(7);
  os << "# vtk DataFile Version 3.0\nwavewsdf " << label << "\nASCII\nDATASET POLYDATA\n";
  os << "POINTS " << nV << " float\n";
  for (int v=0;v<nV;v++)
    os << p.vertexArray[3*v] << " " << p.vertexArray[3*v+1] << " " << p.vertexArray[3*v+2] << "\n";
  os << "POLYGONS " << nT << " " << (size_t)nT*4 << "\n";
  for (size_t t=0;t<nT;t++)
    os << "3 " << dc.tris[3*t] << " " << dc.tris[3*t+1] << " " << dc.tris[3*t+2] << "\n";
  os.close();
  printf("  %s: %d vertices, %zu triangles -> %s\n", label, nV, nT, path);
  if (dc.nUnclosed > 0)
    printf("      note: %ld edge(s) left open (uniform-region=%ld, at-finest-level=%ld)\n",
           dc.nUnclosed, dc.nUnclosedUniform, dc.nUnclosedFinest);
}

// GPU connectivity: the flat minimal-edge kernel -> triangle buffer -> VTK (no host
// recursion).  Same watertight output as dcEmitVtk, fully on the GPU.
static void dcEmitVtkGpu(WaveletSdfSolver *solver, DcGpuParams &p, int nV,
                         const char *path, const char *label) {
  p.maxQuads = 8*p.maxVerts;
  cudaMallocManaged(&p.quadArray, (size_t)3*(size_t)p.maxQuads*sizeof(i32));
  int *nUnc; cudaMallocManaged(&nUnc,sizeof(int)); *nUnc=0;
  *p.quadCount = 0;
  dcQuadKernelGpu<<<cudaGridSize,cudaBlockSize>>>(*solver, p, nUnc);
  cudaDeviceSynchronize();
  size_t nT = (*p.quadCount) < p.maxQuads ? (size_t)*p.quadCount : (size_t)p.maxQuads;
  std::ofstream os(path); os.precision(7);
  os << "# vtk DataFile Version 3.0\nwavewsdf " << label << "\nASCII\nDATASET POLYDATA\n";
  os << "POINTS " << nV << " float\n";
  for (int v=0;v<nV;v++)
    os << p.vertexArray[3*v] << " " << p.vertexArray[3*v+1] << " " << p.vertexArray[3*v+2] << "\n";
  os << "POLYGONS " << nT << " " << (size_t)nT*4 << "\n";
  for (size_t t=0;t<nT;t++)
    os << "3 " << p.quadArray[3*t] << " " << p.quadArray[3*t+1] << " " << p.quadArray[3*t+2] << "\n";
  os.close();
  printf("  %s: %d vertices, %zu triangles -> %s\n", label, nV, nT, path);
  if (*nUnc > 0) printf("      note: %d edge(s) left open at sub-finest-cell features\n", *nUnc);
  cudaFree(p.quadArray); cudaFree(nUnc);
}

// GPU dual contour: one QEF vertex per straddling leaf cell (storedGrad picks the
// edge-normal source), then the shared host minimal-edge recursion + VTK output.
static void dcGpuBaseline(WaveletSdfSolver *solver, const double origin[3], int maxVerts,
                          const char *path, bool storedGrad, const char *label) {
  DcGpuParams p{};
  for (int d=0;d<3;d++) p.origin[d]=(float)origin[d];
  p.maxVerts=maxVerts; p.maxQuads=3*maxVerts; p.hcap=dcNextPow2((long)(2.5*maxVerts));
  cudaMallocManaged(&p.hkeys,(size_t)p.hcap*sizeof(u64));
  cudaMallocManaged(&p.hvals,(size_t)p.hcap*sizeof(i32));
  cudaMemset(p.hkeys,0xFF,(size_t)p.hcap*sizeof(u64));
  cudaMallocManaged(&p.vMask,(size_t)maxVerts*sizeof(unsigned char));
  cudaMallocManaged(&p.vertexArray,(size_t)3*maxVerts*sizeof(float));
  int *cnt; cudaMallocManaged(&cnt,2*sizeof(int)); cnt[0]=cnt[1]=0;
  p.vertCount=&cnt[0]; p.quadCount=&cnt[1];
  if (storedGrad) dcVertexKernelT<true ><<<cudaGridSize,cudaBlockSize>>>(*solver,p);
  else            dcVertexKernelT<false><<<cudaGridSize,cudaBlockSize>>>(*solver,p);
  cudaDeviceSynchronize();
  int nV = cnt[0]<maxVerts?cnt[0]:maxVerts;
  if (getenv("WSDF_DC_HOST")) dcEmitVtk(solver, p, nV, path, label);   // reference recursion
  else                        dcEmitVtkGpu(solver, p, nV, path, label);
  cudaFree(p.hkeys);cudaFree(p.hvals);cudaFree(p.vMask);cudaFree(p.vertexArray);cudaFree(cnt);
}

void dualContourGpu(WaveletSdfSolver *solver, const double h[3], const double origin[3],
                    int maxVerts, const char *path) {
  (void)h;
  dcGpuBaseline(solver, origin, maxVerts, path, /*storedGrad=*/true,
                "dc (gradient/true-Hermite, stored corners + watertight adaptive)");
}

// ===========================================================================
// Carrera et al. 2026, "Dual Contouring of Signed Distance Data": place the dual
// vertices from SDF VALUES ONLY (no stored gradients / Hermite data).  Estimated
// Hermite data (linear edge crossings + trilinear finite-difference normals) is
// iteratively corrected by a local-global optimisation.  The CONNECTIVITY reuses
// our watertight minimal-edge recursion unchanged (so the mesh topology, and its
// watertightness, are identical to the gradient DC); Carrera only moves vertices.
// ---------------------------------------------------------------------------

namespace {

// host read of the stored SDF (values only) at node (ni,nj,nk) of level lvl, with
// the same ascend-to-coarser fallback readNodeDC uses at 2:1 transitions.
struct HostSdf {
  WaveletSdfSolver *g;
  std::unordered_map<u64,int> locToMem;                    // block loc code -> memory index
  void build() {
    i32 nKeys = g->hashTable.nKeys; locToMem.reserve(nKeys*2);
    for (i32 b=0;b<nKeys;b++) locToMem[g->bLocList[b]] = b;
  }
  static u64 enc(i32 lvl,i32 ib,i32 jb,i32 kb) {
    return ((u64)lvl<<60) | ((u64)(ib+1)) | ((u64)(jb+1)<<20) | ((u64)(kb+1)<<40);
  }
  bool readExact(i32 lvl,i32 ni,i32 nj,i32 nk,double &val) const {
    if (ni<0||nj<0||nk<0) return false;
    auto it = locToMem.find(enc(lvl, ni/blockSize, nj/blockSize, nk/blockSize));
    if (it==locToMem.end()) return false;
    i32 c = it->second*nodeSizeTot + WaveletSdfSolver::nodeIdx(ni%blockSize, nj%blockSize, nk%blockSize);
    double v = g->Sdf[c]; if (v==WSDF_FAR) return false; val = v; return true;
  }
  bool read(i32 lvl,i32 ni,i32 nj,i32 nk,double &val) const {
    for (i32 L=lvl; L>=0; L--){ i32 s=lvl-L; if (readExact(L, ni>>s, nj>>s, nk>>s, val)) return true; }
    return false;
  }
};

struct CCell { i32 lvl, I, J, K; double x[3]; std::vector<int> edges; };  // interesting cell + its dual vertex
struct CEdge { int v[4]; bool flip; double h[3], n[3], p0[3], p1[3]; };  // hermite + normal + edge endpoints

// smallest-eigenvector (best-fit plane normal) of a 3x3 symmetric matrix via Jacobi.
static void smallestEig(double C[3][3], double n[3]) {
  double V[3][3]={{1,0,0},{0,1,0},{0,0,1}}, A[3][3];
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)A[i][j]=C[i][j];
  for(int sweep=0;sweep<16;sweep++){
    int pp=0,qq=1; double mx=fabs(A[0][1]);
    if(fabs(A[0][2])>mx){mx=fabs(A[0][2]);pp=0;qq=2;}
    if(fabs(A[1][2])>mx){mx=fabs(A[1][2]);pp=1;qq=2;}
    if(mx<1e-20)break;
    double phi=0.5*atan2(2*A[pp][qq], A[qq][qq]-A[pp][pp]), c=cos(phi), s=sin(phi);
    for(int k=0;k<3;k++){double a=A[k][pp],b=A[k][qq]; A[k][pp]=c*a-s*b; A[k][qq]=s*a+c*b;}
    for(int k=0;k<3;k++){double a=A[pp][k],b=A[qq][k]; A[pp][k]=c*a-s*b; A[qq][k]=s*a+c*b;}
    for(int k=0;k<3;k++){double a=V[k][pp],b=V[k][qq]; V[k][pp]=c*a-s*b; V[k][qq]=s*a+c*b;}
  }
  int m=0; if(A[1][1]<A[m][m])m=1; if(A[2][2]<A[m][m])m=2;
  double nn=sqrt(V[0][m]*V[0][m]+V[1][m]*V[1][m]+V[2][m]*V[2][m])+1e-20;
  n[0]=V[0][m]/nn; n[1]=V[1][m]/nn; n[2]=V[2][m]/nn;
}

// closest point on triangle (a,b,c) to p (Ericson); also returns wa = the
// barycentric weight of vertex a at that closest point.
static void closestPtTri(const double p[3], const double a[3], const double b[3], const double c[3],
                         double out[3], double &wa) {
  double ab[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, ac[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
  double ap[3]={p[0]-a[0],p[1]-a[1],p[2]-a[2]};
  double d1=ab[0]*ap[0]+ab[1]*ap[1]+ab[2]*ap[2], d2=ac[0]*ap[0]+ac[1]*ap[1]+ac[2]*ap[2];
  if(d1<=0&&d2<=0){ out[0]=a[0];out[1]=a[1];out[2]=a[2]; wa=1; return; }
  double bp[3]={p[0]-b[0],p[1]-b[1],p[2]-b[2]};
  double d3=ab[0]*bp[0]+ab[1]*bp[1]+ab[2]*bp[2], d4=ac[0]*bp[0]+ac[1]*bp[1]+ac[2]*bp[2];
  if(d3>=0&&d4<=d3){ out[0]=b[0];out[1]=b[1];out[2]=b[2]; wa=0; return; }
  double vc=d1*d4-d3*d2;
  if(vc<=0&&d1>=0&&d3<=0){ double v=d1/(d1-d3);
    out[0]=a[0]+v*ab[0];out[1]=a[1]+v*ab[1];out[2]=a[2]+v*ab[2]; wa=1-v; return; }
  double cp[3]={p[0]-c[0],p[1]-c[1],p[2]-c[2]};
  double d5=ab[0]*cp[0]+ab[1]*cp[1]+ab[2]*cp[2], d6=ac[0]*cp[0]+ac[1]*cp[1]+ac[2]*cp[2];
  if(d6>=0&&d5<=d6){ out[0]=c[0];out[1]=c[1];out[2]=c[2]; wa=0; return; }
  double vb=d5*d2-d1*d6;
  if(vb<=0&&d2>=0&&d6<=0){ double w=d2/(d2-d6);
    out[0]=a[0]+w*ac[0];out[1]=a[1]+w*ac[1];out[2]=a[2]+w*ac[2]; wa=1-w; return; }
  double va=d3*d6-d5*d4;
  if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){ double w=(d4-d3)/((d4-d3)+(d5-d6));
    out[0]=b[0]+w*(c[0]-b[0]);out[1]=b[1]+w*(c[1]-b[1]);out[2]=b[2]+w*(c[2]-b[2]); wa=0; return; }
  double denom=1.0/(va+vb+vc), v=vb*denom, w=vc*denom;
  out[0]=a[0]+ab[0]*v+ac[0]*w; out[1]=a[1]+ab[1]*v+ac[1]*w; out[2]=a[2]+ab[2]*v+ac[2]*w;
  wa=1-v-w;
}

// per-axis cell size at a level (host port of getDx/getDy/getDz)
static void levelDx(WaveletSdfSolver *g, i32 lvl, double d[3]) {
  d[0] = g->domainSize[0]/(g->baseGridSize[0]*powi(2,lvl));
  d[1] = g->domainSize[1]/(g->baseGridSize[1]*powi(2,lvl));
  d[2] = g->pseudo2D ? g->domainSize[2]/g->baseGridSize[2]
                     : g->domainSize[2]/(g->baseGridSize[2]*powi(2,lvl));
}

// solve a regularised 3x3 QEF: minimise sum_j (n_j.(x-h_j))^2 + lam|x-c|^2.
static void qefSolve(const std::vector<std::array<double,3>> &N,
                     const std::vector<std::array<double,3>> &H,
                     const double c[3], double out[3]) {
  double ata[6]={0,0,0,0,0,0}, atb[3]={0,0,0};
  for (size_t i=0;i<N.size();i++){
    double n0=N[i][0],n1=N[i][1],n2=N[i][2];
    double d = n0*H[i][0]+n1*H[i][1]+n2*H[i][2];
    ata[0]+=n0*n0; ata[1]+=n0*n1; ata[2]+=n0*n2; ata[3]+=n1*n1; ata[4]+=n1*n2; ata[5]+=n2*n2;
    atb[0]+=n0*d; atb[1]+=n1*d; atb[2]+=n2*d;
  }
  double lam = 1e-3*(ata[0]+ata[3]+ata[5])/3.0 + 1e-9;
  double M[6]={ata[0]+lam,ata[1],ata[2],ata[3]+lam,ata[4],ata[5]+lam};
  double r[3]={atb[0]+lam*c[0], atb[1]+lam*c[1], atb[2]+lam*c[2]};
  double det = M[0]*(M[3]*M[5]-M[4]*M[4]) - M[1]*(M[1]*M[5]-M[4]*M[2]) + M[2]*(M[1]*M[4]-M[3]*M[2]);
  if (fabs(det) < 1e-18) { out[0]=c[0]; out[1]=c[1]; out[2]=c[2]; return; }
  double iv[6];
  iv[0]=(M[3]*M[5]-M[4]*M[4])/det; iv[1]=(M[2]*M[4]-M[1]*M[5])/det; iv[2]=(M[1]*M[4]-M[2]*M[3])/det;
  iv[3]=(M[0]*M[5]-M[2]*M[2])/det; iv[4]=(M[2]*M[1]-M[0]*M[4])/det; iv[5]=(M[0]*M[3]-M[1]*M[1])/det;
  out[0]=iv[0]*r[0]+iv[1]*r[1]+iv[2]*r[2];
  out[1]=iv[1]*r[0]+iv[3]*r[1]+iv[4]*r[2];
  out[2]=iv[2]*r[0]+iv[4]*r[1]+iv[5]*r[2];
}

} // namespace

void carreraDc(WaveletSdfSolver *solver, const double h[3], const double origin[3],
               int maxVerts, const char *path, int outerIters, int innerIters) {
  (void)h;
  // Fast path (the default / recommended): the SDF-only baseline -- estimated
  // Hermite + QEF -- is per-cell and runs fully on the GPU (finite-difference edge
  // normals from the corner VALUES, no stored gradients), sharing the gradient DC's
  // kernel + watertight recursion.  Matches the gradient DC's quality and speed.
  if (outerIters <= 0 && innerIters <= 0) {
    dcGpuBaseline(solver, origin, maxVerts, path, /*storedGrad=*/false,
                  "carrera-dc (SDF values only, est.Hermite+QEF, GPU)");
    return;
  }
  // Iterative path (experimental): host-side outer PCA + inner distance-energy
  // refinement.  Slower and -- for this already feature-refined octree -- not an
  // improvement; kept opt-in via the outer/inner iteration counts.

  // 1) GPU: identify the surface-straddling leaf cells (cell key -> id, corner sign
  //    mask).  Its QEF position is discarded; Carrera re-places vertices on the host.
  DcGpuParams p{};
  for (int d=0;d<3;d++){ p.origin[d]=(float)origin[d]; }
  p.maxVerts = maxVerts; p.maxQuads = 3*maxVerts;
  p.hcap = dcNextPow2((long)(2.5*maxVerts));
  cudaMallocManaged(&p.hkeys, (size_t)p.hcap*sizeof(u64));
  cudaMallocManaged(&p.hvals, (size_t)p.hcap*sizeof(i32));
  cudaMemset(p.hkeys, 0xFF, (size_t)p.hcap*sizeof(u64));
  cudaMallocManaged(&p.vMask, (size_t)maxVerts*sizeof(unsigned char));
  cudaMallocManaged(&p.vertexArray, (size_t)3*maxVerts*sizeof(float));
  int *cnt; cudaMallocManaged(&cnt, 2*sizeof(int)); cnt[0]=cnt[1]=0;
  p.vertCount=&cnt[0]; p.quadCount=&cnt[1];
  dcVertexKernelT<false><<<cudaGridSize, cudaBlockSize>>>(*solver, p);
  cudaDeviceSynchronize();
  int nV = cnt[0] < maxVerts ? cnt[0] : maxVerts;

  // 2) host: octree query + cell map, run the recursion in CARRERA mode -> redges.
  DcHost dc; dc.g = solver; dc.vMask = p.vMask; dc.carrera = true;
  i32 nKeys = solver->hashTable.nKeys;
  dc.blocks.reserve(nKeys*2);
  for (i32 b=0;b<nKeys;b++) dc.blocks.insert(solver->bLocList[b]);
  dc.vmap.reserve(nV*2);
  for (int s=0;s<p.hcap;s++) if (p.hkeys[s]!=DC_EMPTY) dc.vmap[p.hkeys[s]] = p.hvals[s];
  dc.run();

  HostSdf sdf; sdf.g = solver; sdf.build();

  // 3) decode each interesting cell (id -> lvl,I,J,K).
  std::vector<CCell> cells(nV);
  for (auto &kv : dc.vmap) {
    u64 key = kv.first; int id = kv.second;
    cells[id].lvl = (i32)(key>>60);
    cells[id].I = (i32)(key & 0xFFFFF) - 1;
    cells[id].J = (i32)((key>>20)&0xFFFFF) - 1;
    cells[id].K = (i32)((key>>40)&0xFFFFF) - 1;
  }

  // 4) estimate Hermite (point + normal) per edge, from SDF VALUES only.
  std::vector<CEdge> edges; edges.reserve(dc.redges.size());
  for (auto &re : dc.redges) {
    int e = processEdgeMask[re.dir][re.mi];
    int c0 = edgevmap[e][0], c1 = edgevmap[e][1];
    double d[3]; levelDx(solver, re.mlvl, d);
    i32 n0i=re.mI+(c0>>2), n0j=re.mJ+((c0>>1)&1), n0k=re.mK+(c0&1);
    i32 n1i=re.mI+(c1>>2), n1j=re.mJ+((c1>>1)&1), n1k=re.mK+(c1&1);
    double p0[3]={n0i*d[0], n0j*d[1], n0k*d[2]};
    double p1[3]={n1i*d[0], n1j*d[1], n1k*d[2]};
    double s0=0,s1=0; sdf.read(re.mlvl,n0i,n0j,n0k,s0); sdf.read(re.mlvl,n1i,n1j,n1k,s1);
    double t = (s0==s1) ? 0.5 : s0/(s0-s1); t = t<0?0:(t>1?1:t);
    CEdge ce; for (int j=0;j<4;j++) ce.v[j]=re.v[j]; ce.flip=re.flip;
    for (int a=0;a<3;a++){ ce.p0[a]=p0[a]; ce.p1[a]=p1[a]; ce.h[a] = p0[a] + t*(p1[a]-p0[a]); }
    // trilinear-interpolant gradient of the minimal cell's 8 corners at h (Eq 2)
    double cv[8]; bool ok=true;
    for (int a=0;a<2&&ok;a++) for (int b=0;b<2&&ok;b++) for (int c=0;c<2&&ok;c++)
      if (!sdf.read(re.mlvl, re.mI+a, re.mJ+b, re.mK+c, cv[a*4+b*2+c])) ok=false;
    double u=(ce.h[0]/d[0]-re.mI), v=(ce.h[1]/d[1]-re.mJ), w=(ce.h[2]/d[2]-re.mK);
    u=u<0?0:(u>1?1:u); v=v<0?0:(v>1?1:v); w=w<0?0:(w>1?1:w);
    double gx=0,gy=0,gz=0;
    if (ok) for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int c=0;c<2;c++){
      double val=cv[a*4+b*2+c];
      gx += val*(a?1:-1)*(b?v:1-v)*(c?w:1-w);
      gy += val*(a?u:1-u)*(b?1:-1)*(c?w:1-w);
      gz += val*(a?u:1-u)*(b?v:1-v)*(c?1:-1);
    }
    gx/=d[0]; gy/=d[1]; gz/=d[2];
    double gn = sqrt(gx*gx+gy*gy+gz*gz);
    if (gn < 1e-20) { gx=p1[0]-p0[0]; gy=p1[1]-p0[1]; gz=p1[2]-p0[2]; gn=sqrt(gx*gx+gy*gy+gz*gz)+1e-20; }
    ce.n[0]=gx/gn; ce.n[1]=gy/gn; ce.n[2]=gz/gn;
    int idx = (int)edges.size(); edges.push_back(ce);
    for (int j=0;j<4;j++) if (ce.v[j]>=0 && ce.v[j]<nV) cells[ce.v[j]].edges.push_back(idx);
  }

  // QEF re-solve for one cell over its edges' (Hermite point, normal) planes (Eq 5),
  // regularised toward the centroid; vertex may leave the cell for sharp features.
  auto solveCell = [&](int i){
    CCell &c = cells[i];
    if (c.edges.empty()) return;
    double ctr[3]={0,0,0};
    for (int ei : c.edges) for (int a=0;a<3;a++) ctr[a]+=edges[ei].h[a];
    for (int a=0;a<3;a++) ctr[a]/=c.edges.size();
    std::vector<std::array<double,3>> Ns, Hs;
    for (int ei : c.edges){ Ns.push_back({edges[ei].n[0],edges[ei].n[1],edges[ei].n[2]});
                            Hs.push_back({edges[ei].h[0],edges[ei].h[1],edges[ei].h[2]}); }
    qefSolve(Ns, Hs, ctr, c.x);
    double d[3]; levelDx(solver, c.lvl, d);
    double lo[3]={c.I*d[0], c.J*d[1], c.K*d[2]};
    for (int a=0;a<3;a++) c.x[a] = fmin(fmax(c.x[a], lo[a]-d[a]), lo[a]+2*d[a]);   // small escape margin
  };

  // 5) init each cell vertex to the centroid of its edges' Hermite points (Eq 3),
  //    then a QEF solve (Eq 5).
  for (int i=0;i<nV;i++) {
    if (cells[i].edges.empty()) continue;
    double ctr[3]={0,0,0};
    for (int ei : cells[i].edges) for (int a=0;a<3;a++) ctr[a]+=edges[ei].h[a];
    for (int a=0;a<3;a++) cells[i].x[a]=ctr[a]/cells[i].edges.size();
    solveCell(i);
  }

  // INNER LOOP (per cell): correct the vertex using the SDF VALUES themselves.  The
  // cell's surface is modelled by a local fan mesh (apex = the dual vertex x, rim =
  // the edge Hermite points); the SDF samples (here the cell's 8 corners) should lie
  // at distance |s| from it.  Linearise that distance energy (Eq 8-12) about the
  // current x -- closest point t on the fan, its sphere point q, direction d -- and
  // solve a 3x3 with the Hermite QEF (Eq 5) and an L2 regulariser.
  auto innerOpt = [&](int i){
    CCell &c = cells[i];
    if (c.edges.empty()) return;
    double d[3]; levelDx(solver, c.lvl, d);
    std::vector<std::array<double,3>> H, N;
    for(int ei:c.edges){ H.push_back({edges[ei].h[0],edges[ei].h[1],edges[ei].h[2]});
                         N.push_back({edges[ei].n[0],edges[ei].n[1],edges[ei].n[2]}); }
    struct Smp{ double u[3], s; }; std::vector<Smp> smp;
    for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int e=0;e<2;e++){ double sv;
      if(sdf.read(c.lvl, c.I+a, c.J+b, c.K+e, sv))
        smp.push_back({{(c.I+a)*d[0],(c.J+b)*d[1],(c.K+e)*d[2]}, sv}); }
    const double wH=0.1, mu=1.0;
    for(int r=0;r<innerIters;r++){
      double ata[6]={0,0,0,0,0,0}, atb[3]={0,0,0};
      auto addRow=[&](double a0,double a1,double a2,double rhs){
        ata[0]+=a0*a0;ata[1]+=a0*a1;ata[2]+=a0*a2;ata[3]+=a1*a1;ata[4]+=a1*a2;ata[5]+=a2*a2;
        atb[0]+=a0*rhs;atb[1]+=a1*rhs;atb[2]+=a2*rhs; };
      for(size_t e=0;e<H.size();e++){ double n0=wH*N[e][0],n1=wH*N[e][1],n2=wH*N[e][2];
        addRow(n0,n1,n2, n0*H[e][0]+n1*H[e][1]+n2*H[e][2]); }
      double sm=sqrt(mu); addRow(sm,0,0,sm*c.x[0]); addRow(0,sm,0,sm*c.x[1]); addRow(0,0,sm,sm*c.x[2]);
      for(auto&sp:smp){
        if(H.size()<2) break;
        double bestD=1e30,t[3]={0,0,0},alpha=1;
        for(size_t a=0;a<H.size();a++)for(size_t b=a+1;b<H.size();b++){
          double cp[3],wa; closestPtTri(sp.u, c.x, H[a].data(), H[b].data(), cp, wa);
          double dd=(cp[0]-sp.u[0])*(cp[0]-sp.u[0])+(cp[1]-sp.u[1])*(cp[1]-sp.u[1])+(cp[2]-sp.u[2])*(cp[2]-sp.u[2]);
          if(dd<bestD){bestD=dd;t[0]=cp[0];t[1]=cp[1];t[2]=cp[2];alpha=wa;}
        }
        double dir[3]={t[0]-sp.u[0],t[1]-sp.u[1],t[2]-sp.u[2]};
        double dn=sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]); if(dn<1e-12)continue;
        dir[0]/=dn;dir[1]/=dn;dir[2]/=dn;
        double as=fabs(sp.s);
        double q[3]={sp.u[0]+as*dir[0],sp.u[1]+as*dir[1],sp.u[2]+as*dir[2]};
        if(alpha<1e-4) addRow(dir[0],dir[1],dir[2], q[0]*dir[0]+q[1]*dir[1]+q[2]*dir[2]);
        else { double rest[3]={t[0]-alpha*c.x[0],t[1]-alpha*c.x[1],t[2]-alpha*c.x[2]};
          addRow(alpha*dir[0],alpha*dir[1],alpha*dir[2],
                 (q[0]-rest[0])*dir[0]+(q[1]-rest[1])*dir[1]+(q[2]-rest[2])*dir[2]); }
      }
      double det=ata[0]*(ata[3]*ata[5]-ata[4]*ata[4])-ata[1]*(ata[1]*ata[5]-ata[4]*ata[2])+ata[2]*(ata[1]*ata[4]-ata[3]*ata[2]);
      if(fabs(det)<1e-18) break;
      double iv[6];
      iv[0]=(ata[3]*ata[5]-ata[4]*ata[4])/det; iv[1]=(ata[2]*ata[4]-ata[1]*ata[5])/det; iv[2]=(ata[1]*ata[4]-ata[2]*ata[3])/det;
      iv[3]=(ata[0]*ata[5]-ata[2]*ata[2])/det; iv[4]=(ata[2]*ata[1]-ata[0]*ata[4])/det; iv[5]=(ata[0]*ata[3]-ata[1]*ata[1])/det;
      double nx=iv[0]*atb[0]+iv[1]*atb[1]+iv[2]*atb[2];
      double ny=iv[1]*atb[0]+iv[3]*atb[1]+iv[4]*atb[2];
      double nz=iv[2]*atb[0]+iv[4]*atb[1]+iv[5]*atb[2];
      double mv=fabs(nx-c.x[0])+fabs(ny-c.x[1])+fabs(nz-c.x[2]);
      c.x[0]=nx;c.x[1]=ny;c.x[2]=nz;
      if(mv<1e-7*d[0]) break;
    }
    double lo[3]={c.I*d[0],c.J*d[1],c.K*d[2]};
    for(int a=0;a<3;a++) c.x[a]=fmin(fmax(c.x[a],lo[a]-d[a]),lo[a]+2*d[a]);
  };

  // 7) OUTER LOOP: optimise every cell's vertex (inner loop), then progressively
  //    correct the estimated Hermite data -- PCA the 4 dual vertices around each
  //    edge for a best-fit plane (normal n, edge intersection y) and blend toward
  //    them (Eq 7).  Repeat.
  const double wu = 0.5;
  for (int k=0; k<outerIters; k++) {
    for (int i=0;i<nV;i++) innerOpt(i);
    for (auto &ce : edges) {
      double pts[4][3]; int np=0;
      for (int j=0;j<4;j++){ int vid=ce.v[j];
        if(vid>=0&&vid<nV){ for(int a=0;a<3;a++)pts[np][a]=cells[vid].x[a]; np++; } }
      if (np<3) continue;
      double ctr[3]={0,0,0};
      for(int i=0;i<np;i++)for(int a=0;a<3;a++)ctr[a]+=pts[i][a];
      for(int a=0;a<3;a++)ctr[a]/=np;
      double C[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for(int i=0;i<np;i++){double dx=pts[i][0]-ctr[0],dy=pts[i][1]-ctr[1],dz=pts[i][2]-ctr[2];
        C[0][0]+=dx*dx;C[0][1]+=dx*dy;C[0][2]+=dx*dz;C[1][1]+=dy*dy;C[1][2]+=dy*dz;C[2][2]+=dz*dz;}
      C[1][0]=C[0][1];C[2][0]=C[0][2];C[2][1]=C[1][2];
      double nn[3]; smallestEig(C, nn);
      if (nn[0]*ce.n[0]+nn[1]*ce.n[1]+nn[2]*ce.n[2] < 0){nn[0]=-nn[0];nn[1]=-nn[1];nn[2]=-nn[2];}
      double dir[3]={ce.p1[0]-ce.p0[0],ce.p1[1]-ce.p0[1],ce.p1[2]-ce.p0[2]};
      double denom=dir[0]*nn[0]+dir[1]*nn[1]+dir[2]*nn[2];
      if (fabs(denom)>1e-12){
        double tt=((ctr[0]-ce.p0[0])*nn[0]+(ctr[1]-ce.p0[1])*nn[1]+(ctr[2]-ce.p0[2])*nn[2])/denom;
        tt=tt<0?0:(tt>1?1:tt);
        for(int a=0;a<3;a++){ double y=ce.p0[a]+tt*dir[a]; ce.h[a]=ce.h[a]+wu*(y-ce.h[a]); }
      }
      for(int a=0;a<3;a++) ce.n[a]=ce.n[a]+wu*nn[a];
      double m=sqrt(ce.n[0]*ce.n[0]+ce.n[1]*ce.n[1]+ce.n[2]*ce.n[2])+1e-20;
      for(int a=0;a<3;a++) ce.n[a]/=m;
    }
  }

  // 6) output: same minimal-edge topology, Carrera vertex positions.
  std::vector<i32> tris;
  for (auto &ce : edges) {
    int a=ce.v[0],b=ce.v[1],cc=ce.v[2],dd=ce.v[3];
    int t[2][3];
    if (ce.flip){ t[0][0]=a;t[0][1]=dd;t[0][2]=b; t[1][0]=a;t[1][1]=cc;t[1][2]=dd; }
    else        { t[0][0]=a;t[0][1]=b;t[0][2]=dd; t[1][0]=a;t[1][1]=dd;t[1][2]=cc; }
    for (int r=0;r<2;r++){
      if (t[r][0]==t[r][1]||t[r][1]==t[r][2]||t[r][0]==t[r][2]) continue;
      tris.push_back(t[r][0]); tris.push_back(t[r][1]); tris.push_back(t[r][2]);
    }
  }
  size_t nT = tris.size()/3;
  std::ofstream os(path); os.precision(7);
  os << "# vtk DataFile Version 3.0\nwavewsdf Carrera DC (SDF values only)\nASCII\nDATASET POLYDATA\n";
  os << "POINTS " << nV << " float\n";
  for (int i=0;i<nV;i++) os << (cells[i].x[0]+origin[0]) << " " << (cells[i].x[1]+origin[1]) << " "
                            << (cells[i].x[2]+origin[2]) << "\n";
  os << "POLYGONS " << nT << " " << (size_t)nT*4 << "\n";
  for (size_t t=0;t<nT;t++) os << "3 " << tris[3*t] << " " << tris[3*t+1] << " " << tris[3*t+2] << "\n";
  os.close();
  printf("  carrera-dc: %d vertices, %zu triangles (SDF values only; est.Hermite+QEF%s) -> %s\n",
         nV, nT, outerIters>0 ? ", +outer PCA/distance refine" : "", path);

  cudaFree(p.hkeys); cudaFree(p.hvals); cudaFree(p.vMask);
  cudaFree(p.vertexArray); cudaFree(cnt);
}
