#ifndef SDF_FEATURES_H
#define SDF_FEATURES_H

// Build the per-triangle geometric features needed for a signed distance
// transform with correct signs at faces, edges and vertices.
//
// This implements the "Data structures" stage of Roosing, Strickson &
// Nikiforakis (CiCP 2019): duplicate vertices are welded, edges are paired with
// their two incident faces, and each vertex receives an angle-weighted
// pseudonormal (Baerentzen & Aanaes). The paper does this on the GPU with
// Morton codes + Thrust for throughput; for the mesh sizes here a host-side
// hash/weld is simpler and the cost is negligible next to the SDF sweep, which
// runs on the GPU. The angle-weighted pseudonormal is what makes the inside/
// outside test exact at edges and vertices (including saddle/ruff cases).

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "Stl.h"
#include "Vec3f.cuh"

// Everything a GPU thread needs to compute the signed distance from a cell to
// one triangle: the geometry, the face normal, and the pseudonormals of the
// three vertices and three edges (edge order: v0v1, v1v2, v2v0).
struct TriFeat {
  float3 v0, v1, v2;
  float3 fn;             // face normal
  float3 vn0, vn1, vn2;  // angle-weighted vertex pseudonormals
  float3 en0, en1, en2;  // edge pseudonormals (sum of incident face normals)
};

// Interior angle of the triangle at corner `a` (between edges a->b and a->c).
inline float cornerAngle(float3 a, float3 b, float3 c) {
  float3 e1 = normalize(b - a);
  float3 e2 = normalize(c - a);
  float d = dot(e1, e2);
  d = fmaxf(-1.0f, fminf(1.0f, d));
  return std::acos(d);
}

inline void buildFeatures(const std::vector<StlTri>& tris,
                          std::vector<TriFeat>& feats,
                          int& nUniqueVerts, int& nUniqueEdges,
                          float3& bmin, float3& bmax,
                          std::vector<float3>* outVerts = nullptr) {  // welded unique vertices
  const size_t nT = tris.size();

  // --- bounding box -------------------------------------------------------
  bmin = make_float3(1e30f, 1e30f, 1e30f);
  bmax = make_float3(-1e30f, -1e30f, -1e30f);
  for (const auto& t : tris)
    for (int k = 0; k < 3; ++k) { bmin = fmin3(bmin, t.v[k]); bmax = fmax3(bmax, t.v[k]); }

  // --- weld vertices ------------------------------------------------------
  // Snap to a quantization much finer than the geometry so coincident corners
  // (stored independently per triangle in the STL) collapse to one id.
  float diag = norm(bmax - bmin);
  float q = (diag > 0.0f ? diag : 1.0f) * 1e-6f;  // weld tolerance
  auto key = [&](float3 p) -> uint64_t {
    uint64_t ix = (uint64_t)std::llround((p.x - bmin.x) / q);
    uint64_t iy = (uint64_t)std::llround((p.y - bmin.y) / q);
    uint64_t iz = (uint64_t)std::llround((p.z - bmin.z) / q);
    return (ix * 73856093ull) ^ (iy * 19349663ull) ^ (iz * 83492791ull);
  };

  std::unordered_map<uint64_t, int> vmap;
  vmap.reserve(nT * 3);
  std::vector<int> corner(nT * 3);          // welded vertex id per corner
  std::vector<float3> vpos;                 // welded vertex positions
  for (size_t t = 0; t < nT; ++t)
    for (int k = 0; k < 3; ++k) {
      uint64_t h = key(tris[t].v[k]);
      auto it = vmap.find(h);
      int id;
      if (it == vmap.end()) { id = (int)vpos.size(); vmap.emplace(h, id); vpos.push_back(tris[t].v[k]); }
      else id = it->second;
      corner[t * 3 + k] = id;
    }
  nUniqueVerts = (int)vpos.size();
  if (outVerts) *outVerts = vpos;   // welded unique vertex positions (file coords)

  // --- global orientation -------------------------------------------------
  // The CSC algorithm assumes outward-pointing normals, but STL files are often
  // wound the other way (negative signed volume). Flip all normals to outward so
  // the sign convention (negative inside) holds for any consistently-wound mesh.
  double vol6 = 0.0;
  for (size_t t = 0; t < nT; ++t) {
    float3 a = tris[t].v[0], b = tris[t].v[1], c = tris[t].v[2];
    vol6 += a.x * (b.y * c.z - b.z * c.y)
          + a.y * (b.z * c.x - b.x * c.z)
          + a.z * (b.x * c.y - b.y * c.x);
  }
  float orient = (vol6 >= 0.0) ? 1.0f : -1.0f;

  // --- per-triangle face normals (recomputed from geometry) ---------------
  std::vector<float3> faceN(nT);
  for (size_t t = 0; t < nT; ++t) {
    float3 n = cross(tris[t].v[1] - tris[t].v[0], tris[t].v[2] - tris[t].v[0]);
    if (norm(n) > 0.0f) n = normalize(n);
    else                n = normalize(tris[t].n);  // fall back to stored normal
    faceN[t] = n * orient;
  }

  // --- angle-weighted vertex pseudonormals --------------------------------
  std::vector<float3> vNorm(nUniqueVerts, make_float3(0, 0, 0));
  for (size_t t = 0; t < nT; ++t) {
    float3 a = tris[t].v[0], b = tris[t].v[1], c = tris[t].v[2];
    vNorm[corner[t * 3 + 0]] += faceN[t] * cornerAngle(a, b, c);
    vNorm[corner[t * 3 + 1]] += faceN[t] * cornerAngle(b, c, a);
    vNorm[corner[t * 3 + 2]] += faceN[t] * cornerAngle(c, a, b);
  }

  // --- edge pseudonormals (sum of the two incident face normals) ----------
  auto edgeKey = [&](int a, int b) -> uint64_t {
    uint64_t lo = (uint64_t)std::min(a, b), hi = (uint64_t)std::max(a, b);
    return (lo << 32) | hi;
  };
  std::unordered_map<uint64_t, float3> eNorm;
  eNorm.reserve(nT * 3);
  auto addEdge = [&](int a, int b, float3 n) {
    uint64_t k = edgeKey(a, b);
    auto it = eNorm.find(k);
    if (it == eNorm.end()) eNorm.emplace(k, n);
    else it->second += n;
  };
  for (size_t t = 0; t < nT; ++t) {
    int a = corner[t * 3 + 0], b = corner[t * 3 + 1], c = corner[t * 3 + 2];
    addEdge(a, b, faceN[t]);
    addEdge(b, c, faceN[t]);
    addEdge(c, a, faceN[t]);
  }
  nUniqueEdges = (int)eNorm.size();

  // --- pack per-triangle features -----------------------------------------
  feats.resize(nT);
  for (size_t t = 0; t < nT; ++t) {
    int a = corner[t * 3 + 0], b = corner[t * 3 + 1], c = corner[t * 3 + 2];
    TriFeat f;
    f.v0 = tris[t].v[0]; f.v1 = tris[t].v[1]; f.v2 = tris[t].v[2];
    f.fn  = faceN[t];
    f.vn0 = normalize(vNorm[a]);
    f.vn1 = normalize(vNorm[b]);
    f.vn2 = normalize(vNorm[c]);
    f.en0 = normalize(eNorm[edgeKey(a, b)]);
    f.en1 = normalize(eNorm[edgeKey(b, c)]);
    f.en2 = normalize(eNorm[edgeKey(c, a)]);
    feats[t] = f;
  }
}

#endif
