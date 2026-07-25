#ifndef SDF_BVH_H
#define SDF_BVH_H

// Host-side AABB BVH over triangles, with the per-node aggregates needed for a
// fast generalized-winding-number sign (Barill 2018).  Ported from the Rust
// reference ../TensorTrain/rs/src/mesh.rs (build_bvh + fill_aggr): a median-split
// tree on the longest centroid axis (leaf <= 4 faces), each node carrying its
// dipole moment (sum of vector areas a*n), area-weighted centroid, and cluster
// radius.  The tree is built flat (a node array + a face permutation) so it can
// be uploaded and traversed iteratively on the device -- see
// WaveletSdfSolverKernels.cu.
//
// The winding number depends on triangle WINDING (vertex order), not on the
// re-oriented face normals, so the BVH is built from the welded triangle vertices
// in their file order.  `orient` (sign of the mesh signed volume) lets the device
// inside-test stay correct for inward-wound STLs (inside <=> orient*wind > 0.5).

#include <algorithm>
#include <vector>

#include "Features.h"   // TriFeat (triangle vertices) + float3 math (Vec3f.cuh)

// One flattened BVH node.  `l`/`r` are child node indices, or -1 for a leaf, in
// which case faces order[start .. start+count) belong to it.
struct BvhNode {
  float3 lo, hi;     // node AABB
  int    l, r;       // child indices, or -1 (leaf)
  int    start, count;
  float3 pa;         // dipole moment = sum of vector areas (area * normal)
  float3 pc;         // area-weighted centroid
  float  pr;         // cluster radius (max dist from pc to the AABB corners)
};

struct Bvh {
  std::vector<BvhNode> nodes;   // node 0 is the root
  std::vector<int>     order;   // face permutation referenced by leaf ranges
  float                orient;  // +1 outward-wound mesh, -1 inward-wound
};

// squared distance from x to triangle/face AABB corner
static inline float bvhDist2(float3 a, float3 b) { float3 d = a - b; return dot(d, d); }

// Recursive median-split builder; returns the index of the node it creates.
inline int bvhBuild(std::vector<BvhNode> &nodes, std::vector<int> &order,
                    int start, int end,
                    const std::vector<float3> &faabbLo, const std::vector<float3> &faabbHi,
                    const std::vector<float3> &fcent) {
  float3 lo = make_float3( 1e30f,  1e30f,  1e30f);
  float3 hi = make_float3(-1e30f, -1e30f, -1e30f);
  for (int k = start; k < end; k++) {
    int fi = order[k];
    lo = fmin3(lo, faabbLo[fi]);
    hi = fmax3(hi, faabbHi[fi]);
  }
  int idx = (int)nodes.size();
  nodes.push_back(BvhNode{lo, hi, -1, -1, start, end - start,
                          make_float3(0,0,0), make_float3(0,0,0), 0.0f});
  if (end - start <= 4) return idx;

  // split on the longest axis of the centroid bbox, by median centroid
  float3 clo = make_float3( 1e30f,  1e30f,  1e30f);
  float3 chi = make_float3(-1e30f, -1e30f, -1e30f);
  for (int k = start; k < end; k++) {
    float3 c = fcent[order[k]];
    clo = fmin3(clo, c);
    chi = fmax3(chi, c);
  }
  float ext[3] = {chi.x - clo.x, chi.y - clo.y, chi.z - clo.z};
  int axis = 0;
  if (ext[1] > ext[axis]) axis = 1;
  if (ext[2] > ext[axis]) axis = 2;
  if (ext[axis] < 1e-30f) return idx;   // degenerate -> keep as a (large) leaf

  auto comp = [&](int a, int b) {
    float ca = (axis == 0) ? fcent[a].x : (axis == 1) ? fcent[a].y : fcent[a].z;
    float cb = (axis == 0) ? fcent[b].x : (axis == 1) ? fcent[b].y : fcent[b].z;
    return ca < cb;
  };
  std::sort(order.begin() + start, order.begin() + end, comp);
  int mid = (start + end) / 2;
  int l = bvhBuild(nodes, order, start, mid, faabbLo, faabbHi, fcent);
  int r = bvhBuild(nodes, order, mid, end, faabbLo, faabbHi, fcent);
  nodes[idx].l = l;
  nodes[idx].r = r;
  return idx;
}

// Post-order fill of per-node dipole moment, area-centroid, and radius.  Returns
// (sum vector-area P, total area A, area-weighted centroid sum Cw) for the node.
inline void bvhFillAggr(std::vector<BvhNode> &nodes, const std::vector<int> &order,
                        const std::vector<TriFeat> &tris, int node,
                        float3 &P, float &A, float3 &Cw) {
  BvhNode nd = nodes[node];   // copy the static fields we need
  if (nd.l < 0) {
    P = make_float3(0,0,0); A = 0.0f; Cw = make_float3(0,0,0);
    for (int k = nd.start; k < nd.start + nd.count; k++) {
      const TriFeat &t = tris[order[k]];
      float3 cr = cross(t.v1 - t.v0, t.v2 - t.v0);
      float area = 0.5f * norm(cr);
      P  = P + cr * 0.5f;                 // vector area = area * normal (= cr/2)
      A += area;
      float3 c = (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
      Cw = Cw + c * area;
    }
  } else {
    float3 Pl, Pr, Cwl, Cwr; float Al, Ar;
    bvhFillAggr(nodes, order, tris, nd.l, Pl, Al, Cwl);
    bvhFillAggr(nodes, order, tris, nd.r, Pr, Ar, Cwr);
    P = Pl + Pr; A = Al + Ar; Cw = Cwl + Cwr;
  }
  float3 c = (A > 1e-30f) ? Cw * (1.0f / A) : (nd.lo + nd.hi) * 0.5f;
  float rr = 0.0f;
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int d = 0; d < 2; d++) {
        float3 corner = make_float3(a ? nd.hi.x : nd.lo.x,
                                    b ? nd.hi.y : nd.lo.y,
                                    d ? nd.hi.z : nd.lo.z);
        rr = fmaxf(rr, bvhDist2(corner, c));
      }
  nodes[node].pa = P;
  nodes[node].pc = c;
  nodes[node].pr = sqrtf(rr);
}

// Build the BVH over the welded triangles (TriFeat vertices).  The face order is
// 0..nTris (matching the device TriFeat array), permuted by order[].
inline Bvh buildBvh(const std::vector<TriFeat> &tris) {
  int nTris = (int)tris.size();
  std::vector<float3> faabbLo(nTris), faabbHi(nTris), fcent(nTris);
  double vol6 = 0.0;
  for (int f = 0; f < nTris; f++) {
    const TriFeat &t = tris[f];
    float3 lo = fmin3(fmin3(t.v0, t.v1), t.v2);
    float3 hi = fmax3(fmax3(t.v0, t.v1), t.v2);
    faabbLo[f] = lo; faabbHi[f] = hi;
    fcent[f] = (lo + hi) * 0.5f;
    // 6 * signed volume of the tetra (origin, v0, v1, v2), summed -> mesh winding
    float3 a = t.v0, b = t.v1, c = t.v2;
    vol6 += (double)(a.x * (b.y * c.z - b.z * c.y)
                   + a.y * (b.z * c.x - b.x * c.z)
                   + a.z * (b.x * c.y - b.y * c.x));
  }

  Bvh bvh;
  bvh.order.resize(nTris);
  for (int i = 0; i < nTris; i++) bvh.order[i] = i;
  bvh.orient = (vol6 >= 0.0) ? 1.0f : -1.0f;
  if (nTris > 0) {
    bvhBuild(bvh.nodes, bvh.order, 0, nTris, faabbLo, faabbHi, fcent);
    float3 P, Cw; float A;
    bvhFillAggr(bvh.nodes, bvh.order, tris, 0, P, A, Cw);
  }
  return bvh;
}

#endif
