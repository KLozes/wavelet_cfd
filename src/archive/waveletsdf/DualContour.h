#ifndef DUAL_CONTOUR_H
#define DUAL_CONTOUR_H

// Feature-preserving surface extraction (Dual Contouring of Hermite data, Ju et
// al. 2002) straight from the BVH SDF oracle, on a uniform grid:
//   * find the cells the surface crosses (flood fill from mesh seeds);
//   * per cell, place ONE vertex by minimizing the QEF  sum_i (n_i . (x - p_i))^2
//     over the cell's edge-crossing Hermite samples (p_i = surface point, n_i =
//     normal) -- the minimiser snaps ONTO sharp edges/corners;
//   * emit a quad per sign-change grid edge, joining the 4 cells around it.
// The Hermite samples (crossing point + exact normal) come from the oracle, so
// sharp features are reconstructed, not smoothed.  Output: legacy VTK PolyData.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "BvhQuery.h"   // host SDF oracle: signedDistanceGrad

struct DcGrid {
  const BvhNode *nodes; const i32 *order; const TriFeat *tris; real orient;
  double h[3];        // cell size per axis (grid frame)
  double origin[3];   // world origin added on output (= domainOrigin)
  int n[3];           // cells per axis (informational)
};

// pack a grid index (i,j,k in [-1, ~2^21)) into a 64-bit key
static inline uint64_t dcKey(long i, long j, long k) {
  return (uint64_t)(uint32_t)(i + 1)
       | ((uint64_t)(uint32_t)(j + 1) << 21)
       | ((uint64_t)(uint32_t)(k + 1) << 42);
}

class DualContour {
public:
  DualContour(const DcGrid &grid) : g(grid) {}

  // Extract the surface, seeding the flood fill from the mesh triangles.
  void run(const std::vector<TriFeat> &feats) {
    std::queue<std::array<long,3>> q;
    std::unordered_set<uint64_t> seen;
    auto seed = [&](float3 p){
      long c[3] = { (long)floor(p.x/g.h[0]), (long)floor(p.y/g.h[1]), (long)floor(p.z/g.h[2]) };
      q.push({c[0],c[1],c[2]});
    };
    for (const auto &t : feats) {
      seed(t.v0); seed(t.v1); seed(t.v2);
      seed((t.v0 + t.v1 + t.v2) * (1.0f/3.0f));
    }
    while (!q.empty()) {
      auto c = q.front(); q.pop();
      uint64_t k = dcKey(c[0], c[1], c[2]);
      if (!seen.insert(k).second) continue;
      if (!straddles(c[0], c[1], c[2])) continue;     // not a surface cell
      makeVertex(c[0], c[1], c[2]);
      for (int a = 0; a < 3; a++) for (int s = -1; s <= 1; s += 2) {
        std::array<long,3> nb = c; nb[a] += s; q.push(nb);
      }
    }
    buildQuads();
  }

  // legacy VTK PolyData (.vtk): points + quad polygons.
  void writeVtk(const char *path) const {
    std::ofstream os(path);
    os.precision(7);
    os << "# vtk DataFile Version 3.0\nwavewsdf dual contour\nASCII\nDATASET POLYDATA\n";
    size_t nv = verts.size()/3, nq = quads.size()/4;
    os << "POINTS " << nv << " float\n";
    for (size_t i = 0; i < nv; i++) os << verts[3*i] << " " << verts[3*i+1] << " " << verts[3*i+2] << "\n";
    os << "POLYGONS " << nq << " " << nq*5 << "\n";
    for (size_t i = 0; i < nq; i++)
      os << "4 " << quads[4*i] << " " << quads[4*i+1] << " " << quads[4*i+2] << " " << quads[4*i+3] << "\n";
    os.close();
    printf("  dc: %zu vertices, %zu quads -> %s\n", nv, nq, path);
  }

  size_t nVerts() const { return verts.size()/3; }
  size_t nQuads() const { return quads.size()/4; }

private:
  const DcGrid &g;
  std::unordered_map<uint64_t, float> cornerVal;   // corner SDF value cache
  std::unordered_map<uint64_t, int>   cellVert;    // surface cell -> vertex id
  std::vector<float> verts;                         // xyz (world) per vertex
  std::vector<int>   quads;                         // 4 vertex ids per quad

  float3 cornerPos(long i, long j, long k) const {
    return make_float3((float)(i*g.h[0]), (float)(j*g.h[1]), (float)(k*g.h[2]));
  }
  // cached SDF value at corner (i,j,k)
  float cval(long i, long j, long k) {
    uint64_t key = dcKey(i,j,k);
    auto it = cornerVal.find(key);
    if (it != cornerVal.end()) return it->second;
    float3 gd;
    float v = signedDistanceGrad(g.nodes, g.order, g.tris, g.orient, cornerPos(i,j,k), gd);
    cornerVal[key] = v;
    return v;
  }
  bool straddles(long i, long j, long k) {
    bool neg = false, pos = false;
    for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int d=0;d<2;d++) {
      float v = cval(i+a, j+b, k+d);
      (v < 0.0f) ? neg = true : pos = true;
    }
    return neg && pos;
  }

  // the 12 cell edges as (corner0, corner1) local-offset pairs
  static constexpr int EDGES[12][6] = {
    {0,0,0, 1,0,0},{0,1,0, 1,1,0},{0,0,1, 1,0,1},{0,1,1, 1,1,1}, // x
    {0,0,0, 0,1,0},{1,0,0, 1,1,0},{0,0,1, 0,1,1},{1,0,1, 1,1,1}, // y
    {0,0,0, 0,0,1},{1,0,0, 1,0,1},{0,1,0, 0,1,1},{1,1,0, 1,1,1}, // z
  };

  // solve a symmetric 3x3  M x = r  (M = [m0 m1 m2; m1 m3 m4; m2 m4 m5]); on a
  // near-singular M, fall back to the supplied mass point.
  static void solveSym3(const double m[6], const double r[3], const double fb[3], double x[3]) {
    double det = m[0]*(m[3]*m[5]-m[4]*m[4]) - m[1]*(m[1]*m[5]-m[4]*m[2]) + m[2]*(m[1]*m[4]-m[3]*m[2]);
    if (fabs(det) < 1e-18) { x[0]=fb[0]; x[1]=fb[1]; x[2]=fb[2]; return; }
    double inv[6];                                  // symmetric inverse
    inv[0]=(m[3]*m[5]-m[4]*m[4])/det; inv[1]=(m[2]*m[4]-m[1]*m[5])/det; inv[2]=(m[1]*m[4]-m[2]*m[3])/det;
    inv[3]=(m[0]*m[5]-m[2]*m[2])/det; inv[4]=(m[2]*m[1]-m[0]*m[4])/det; inv[5]=(m[0]*m[3]-m[1]*m[1])/det;
    x[0]=inv[0]*r[0]+inv[1]*r[1]+inv[2]*r[2];
    x[1]=inv[1]*r[0]+inv[3]*r[1]+inv[4]*r[2];
    x[2]=inv[2]*r[0]+inv[4]*r[1]+inv[5]*r[2];
  }

  // place the QEF-minimising vertex of surface cell (i,j,k)
  void makeVertex(long i, long j, long k) {
    double ata[6]={0,0,0,0,0,0}, atb[3]={0,0,0}, mass[3]={0,0,0}; int cnt=0;
    for (auto &e : EDGES) {
      float va = cval(i+e[0], j+e[1], k+e[2]);
      float vb = cval(i+e[3], j+e[4], k+e[5]);
      if ((va < 0.0f) == (vb < 0.0f)) continue;       // no crossing on this edge
      float t = va / (va - vb);                        // zero-crossing fraction
      float3 pa = cornerPos(i+e[0], j+e[1], k+e[2]);
      float3 pb = cornerPos(i+e[3], j+e[4], k+e[5]);
      float3 p  = pa + (pb - pa) * t;
      float3 nrm; signedDistanceGrad(g.nodes, g.order, g.tris, g.orient, p, nrm);  // normal at crossing
      double n[3]={nrm.x,nrm.y,nrm.z}, d=n[0]*p.x+n[1]*p.y+n[2]*p.z;
      ata[0]+=n[0]*n[0]; ata[1]+=n[0]*n[1]; ata[2]+=n[0]*n[2];
      ata[3]+=n[1]*n[1]; ata[4]+=n[1]*n[2]; ata[5]+=n[2]*n[2];
      atb[0]+=n[0]*d; atb[1]+=n[1]*d; atb[2]+=n[2]*d;
      mass[0]+=p.x; mass[1]+=p.y; mass[2]+=p.z; cnt++;
    }
    if (cnt == 0) return;
    double c[3]={mass[0]/cnt, mass[1]/cnt, mass[2]/cnt};
    double lam = 1e-3*(ata[0]+ata[3]+ata[5])/3.0 + 1e-9;   // regularise toward the mass point
    double M[6]={ata[0]+lam,ata[1],ata[2],ata[3]+lam,ata[4],ata[5]+lam};
    double r[3]={atb[0]+lam*c[0], atb[1]+lam*c[1], atb[2]+lam*c[2]};
    double x[3]; solveSym3(M, r, c, x);
    // clamp into the cell, then to world coords
    double lo[3]={i*g.h[0], j*g.h[1], k*g.h[2]};
    for (int a=0;a<3;a++) x[a] = fmin(fmax(x[a], lo[a]), lo[a]+g.h[a]);
    int id = (int)(verts.size()/3);
    verts.push_back((float)(x[0]+g.origin[0]));
    verts.push_back((float)(x[1]+g.origin[1]));
    verts.push_back((float)(x[2]+g.origin[2]));
    cellVert[dcKey(i,j,k)] = id;
  }

  int vertOf(long i, long j, long k) const {
    auto it = cellVert.find(dcKey(i,j,k));
    return it == cellVert.end() ? -1 : it->second;
  }

  // one quad per sign-change grid edge, joining the 4 cells around it.  Each edge
  // is owned by exactly one cell (the +,+ cell in the two transverse axes), so no
  // duplicates.  Wound so the normal points outward (toward increasing SDF).
  void buildQuads() {
    for (auto &kv : cellVert) {
      long i = (long)((uint32_t)(kv.first        & 0x1FFFFF)) - 1;
      long j = (long)((uint32_t)((kv.first >> 21) & 0x1FFFFF)) - 1;
      long k = (long)((uint32_t)((kv.first >> 42) & 0x1FFFFF)) - 1;
      // x-edge at corner (i,j,k): cells in (y,z), CCW viewed from +x
      tryQuad(cval(i,j,k), cval(i+1,j,k),
              vertOf(i,j-1,k-1), vertOf(i,j,k-1), vertOf(i,j,k), vertOf(i,j-1,k));
      // y-edge: cells in (z,x), CCW viewed from +y
      tryQuad(cval(i,j,k), cval(i,j+1,k),
              vertOf(i-1,j,k-1), vertOf(i-1,j,k), vertOf(i,j,k), vertOf(i,j,k-1));
      // z-edge: cells in (x,y), CCW viewed from +z
      tryQuad(cval(i,j,k), cval(i,j,k+1),
              vertOf(i-1,j-1,k), vertOf(i,j-1,k), vertOf(i,j,k), vertOf(i-1,j,k));
    }
  }
  void tryQuad(float c0, float c1, int a, int b, int c, int d) {
    if ((c0 < 0.0f) == (c1 < 0.0f)) return;          // edge has no sign change
    if (a < 0 || b < 0 || c < 0 || d < 0) return;    // a surrounding cell is missing
    if (c0 < 0.0f) { quads.push_back(a); quads.push_back(b); quads.push_back(c); quads.push_back(d); }
    else           { quads.push_back(d); quads.push_back(c); quads.push_back(b); quads.push_back(a); }
  }
};

constexpr int DualContour::EDGES[12][6];

// convenience entry point
inline void dualContourToVtk(const DcGrid &g, const std::vector<TriFeat> &feats, const char *path) {
  DualContour dc(g);
  dc.run(feats);
  dc.writeVtk(path);
}

#endif
