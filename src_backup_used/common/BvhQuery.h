#ifndef BVH_QUERY_H
#define BVH_QUERY_H

// Shared BVH SDF oracle: signedDistanceGrad(x) = (signed distance, unit gradient)
// over the triangle BVH (Bvh.h).  Exact closest-feature distance from an iterative
// BVH descent; inside/outside sign from a fast generalized winding number (Barill
// 2018) treecode.  Both traversals use an explicit stack (no recursion), so the
// same code runs on the DEVICE (the wavelet-SDF kernels) and on the HOST (the dual
// contouring surface extractor).  Ported from ../TensorTrain/rs/src/mesh.rs.

#include "Util.cuh"      // i32, real
#include "Bvh.h"        // BvhNode, TriFeat (via Features.h), float3 math (Vec3f.cuh)

static constexpr float INV4PI = 1.0f / (4.0f * 3.14159265358979f);
static constexpr float INV2PI = 1.0f / (2.0f * 3.14159265358979f);
static constexpr float WIND_BETA = 2.0f;   // far-field acceptance: |x - pc| > BETA * pr
static constexpr int   BVH_STACK = 64;     // >= tree depth (~log2(nTris))

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision
// Detection). `region`: 0,1,2 = vertices a,b,c; 3,4,5 = edges ab,bc,ca; 6 = face.
__host__ __device__ inline float3 closestPtTriangle(
    float3 p, float3 a, float3 b, float3 c, int &region) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) { region = 0; return a; }

  float3 bp = p - b;
  float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) { region = 1; return b; }

  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = d1 / (d1 - d3); region = 3; return a + ab * v;
  }

  float3 cp = p - c;
  float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) { region = 2; return c; }

  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = d2 / (d2 - d6); region = 5; return a + ac * w;
  }

  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); region = 4; return b + (c - b) * w;
  }

  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom, w = vc * denom;
  region = 6; return a + ab * v + ac * w;
}

// squared distance from x to the axis-aligned box [lo,hi]
__host__ __device__ inline float aabbDist2(float3 lo, float3 hi, float3 x) {
  float ex = fmaxf(fmaxf(lo.x - x.x, x.x - hi.x), 0.0f);
  float ey = fmaxf(fmaxf(lo.y - x.y, x.y - hi.y), 0.0f);
  float ez = fmaxf(fmaxf(lo.z - x.z, x.z - hi.z), 0.0f);
  return ex * ex + ey * ey + ez * ez;
}

// signed solid angle of a triangle seen from x (Van Oosterom-Strackee)
__host__ __device__ inline float triSolid(float3 x, float3 v0, float3 v1, float3 v2) {
  float3 a = v0 - x, b = v1 - x, c = v2 - x;
  float la = norm(a), lb = norm(b), lc = norm(c);
  float det = dot(a, cross(b, c));
  float den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
  return atan2f(det, den);
}

// nearest triangle to x via iterative BVH descent (nearest-child-first, pruned by
// the running best).  Returns best squared distance; sets closest point q and the
// nearest face id fid.
__host__ __device__ inline float nearestTri(const BvhNode *nodes, const i32 *order,
                                            const TriFeat *tris, float3 x, float3 &q, i32 &fid) {
  float best = 1e30f;
  float3 bestQ = make_float3(0, 0, 0);
  i32 bestF = 0;
  i32 stack[BVH_STACK];
  i32 sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    i32 ni = stack[--sp];
    BvhNode nd = nodes[ni];
    if (aabbDist2(nd.lo, nd.hi, x) >= best) continue;
    if (nd.l < 0) {
      for (i32 k = nd.start; k < nd.start + nd.count; k++) {
        i32 f = order[k];
        TriFeat t = tris[f];
        int region;
        float3 qq = closestPtTriangle(x, t.v0, t.v1, t.v2, region);
        float3 dv = x - qq;
        float d2 = dot(dv, dv);
        if (d2 < best) { best = d2; bestQ = qq; bestF = f; }
      }
    } else {
      float dl = aabbDist2(nodes[nd.l].lo, nodes[nd.l].hi, x);
      float dr = aabbDist2(nodes[nd.r].lo, nodes[nd.r].hi, x);
      // push the farther child first so the nearer is popped (and pruned) first
      if (dl <= dr) { stack[sp++] = nd.r; stack[sp++] = nd.l; }
      else          { stack[sp++] = nd.l; stack[sp++] = nd.r; }
    }
  }
  q = bestQ; fid = bestF;
  return best;
}

// fast generalized winding number: far clusters use the node dipole moment, near
// ones recurse to exact per-triangle solid angles.  Iterative (explicit stack).
__host__ __device__ inline float windFast(const BvhNode *nodes, const i32 *order,
                                          const TriFeat *tris, float3 x) {
  float s = 0.0f;
  i32 stack[BVH_STACK];
  i32 sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    i32 ni = stack[--sp];
    BvhNode nd = nodes[ni];
    if (nd.l < 0) {
      float ls = 0.0f;
      for (i32 k = nd.start; k < nd.start + nd.count; k++) {
        TriFeat t = tris[order[k]];
        ls += triSolid(x, t.v0, t.v1, t.v2);
      }
      s += ls * INV2PI;
    } else {
      float3 dc = nd.pc - x;
      float d = norm(dc);
      if (d > WIND_BETA * nd.pr) {
        s += INV4PI * dot(nd.pa, dc) / (d * d * d);   // dipole far field
      } else {
        stack[sp++] = nd.l;
        stack[sp++] = nd.r;
      }
    }
  }
  return s;
}

// does the ray [x, x+t*dir) for t>0 hit triangle (a,b,c)?  (Moller-Trumbore)
__host__ __device__ inline bool rayHitsTri(float3 x, float3 dir, float3 a, float3 b, float3 c) {
  float3 e1 = b - a, e2 = c - a, pv = cross(dir, e2);
  float det = dot(e1, pv);
  if (fabsf(det) < 1e-12f) return false;              // ray parallel to triangle
  float inv = 1.0f / det;
  float3 tvec = x - a;
  float u = dot(tvec, pv) * inv;            if (u < 0.0f || u > 1.0f) return false;
  float3 qv = cross(tvec, e1);
  float v = dot(dir, qv) * inv;             if (v < 0.0f || u + v > 1.0f) return false;
  return dot(e2, qv) * inv > 1e-7f;                   // forward hit
}

// does the ray from x along dir (precomputed inv = 1/dir) enter box [lo,hi] at t>=0?
__host__ __device__ inline bool rayHitsAabb(float3 lo, float3 hi, float3 x, float3 inv) {
  float t1=(lo.x-x.x)*inv.x, t2=(hi.x-x.x)*inv.x, tmn=fminf(t1,t2), tmx=fmaxf(t1,t2);
  t1=(lo.y-x.y)*inv.y; t2=(hi.y-x.y)*inv.y; tmn=fmaxf(tmn,fminf(t1,t2)); tmx=fminf(tmx,fmaxf(t1,t2));
  t1=(lo.z-x.z)*inv.z; t2=(hi.z-x.z)*inv.z; tmn=fmaxf(tmn,fminf(t1,t2)); tmx=fminf(tmx,fmaxf(t1,t2));
  return tmx >= fmaxf(tmn, 0.0f);
}

// Inside/outside by RAY-CAST PARITY: a generic-direction ray from x crosses a clean
// watertight surface an odd number of times iff x is inside.  BVH-accelerated, and
// far cheaper near the surface than the winding number (which evaluates many solid
// angles there).  Exact for non-self-intersecting watertight meshes; for dirty
// meshes use the winding number instead (signedDistanceGrad, -DWSDF_SIGN_WINDING).
__host__ __device__ inline bool insideRayCast(const BvhNode *nodes, const i32 *order,
                                              const TriFeat *tris, float3 x) {
  const float3 dir = make_float3(0.21617f, 0.51059f, 0.83203f);   // generic (avoids edge/vertex hits)
  float3 inv = make_float3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);
  int hits = 0;
  i32 stack[BVH_STACK]; i32 sp = 0; stack[sp++] = 0;
  while (sp > 0) {
    i32 ni = stack[--sp];
    BvhNode nd = nodes[ni];
    if (!rayHitsAabb(nd.lo, nd.hi, x, inv)) continue;
    if (nd.l < 0) {
      for (i32 k = nd.start; k < nd.start + nd.count; k++) {
        TriFeat t = tris[order[k]];
        if (rayHitsTri(x, dir, t.v0, t.v1, t.v2)) hits++;
      }
    } else { stack[sp++] = nd.l; stack[sp++] = nd.r; }
  }
  return (hits & 1) != 0;
}

//
// Signed distance whose SIGN comes from the angle-weighted PSEUDONORMAL of the
// closest feature (Baerentzen & Aanaes 2005) rather than from ray parity.
//
// Exact for a watertight mesh and free of the parity test's failure mode: a ray
// that grazes a shared edge is counted once instead of zero or twice, which
// flips the sign at isolated points.  On a structured loft -- long strips of
// near-coplanar triangles -- that is not rare, and each flip puts a spurious
// island or hole in the level set.  The pseudonormals are already carried in
// TriFeat, and closestPtTriangle already reports which feature won, so the sign
// costs one extra closest-point evaluation and no traversal.
//
__host__ __device__ inline float signedDistancePseudo(const BvhNode *nodes, const i32 *order,
                                                      const TriFeat *tris,
                                                      float3 x, float3 &grad) {
  float3 q; i32 fid;
  float d = sqrtf(nearestTri(nodes, order, tris, x, q, fid));
  const TriFeat &t = tris[fid];
  int region;
  float3 qq = closestPtTriangle(x, t.v0, t.v1, t.v2, region);
  float3 n = (region == 0) ? t.vn0 : (region == 1) ? t.vn1 : (region == 2) ? t.vn2
           : (region == 3) ? t.en0 : (region == 4) ? t.en1 : (region == 5) ? t.en2
           : t.fn;
  float s = (dot(x - qq, n) < 0.0f) ? -1.0f : 1.0f;
  grad = (d > 1e-7f) ? (x - qq) * (s / d) : normalize(n);
  return s * d;
}

// signed distance + unit gradient at x.  distance from the BVH nearest feature;
// sign from a ray-cast parity test by default (exact + fast for clean watertight
// meshes), or the generalized winding number with -DWSDF_SIGN_WINDING (robust to
// non-watertight / self-intersecting meshes).  gradient = sign*(x-q)/d (the eikonal
// direction); on the surface falls back to the nearest face normal.  Negative inside.
__host__ __device__ inline float signedDistanceGrad(const BvhNode *nodes, const i32 *order,
                                                    const TriFeat *tris, real orient,
                                                    float3 x, float3 &grad) {
  float3 q; i32 fid;
  float d2 = nearestTri(nodes, order, tris, x, q, fid);
  float d = sqrtf(d2);
#ifdef WSDF_SIGN_WINDING
  float s = (orient * windFast(nodes, order, tris, x) > 0.5f) ? -1.0f : 1.0f;
#else
  (void)orient;
  float s = insideRayCast(nodes, order, tris, x) ? -1.0f : 1.0f;
#endif
  if (d > 1e-7f) grad = (x - q) * (s / d);
  else           grad = tris[fid].fn;                 // outward face normal (pre-oriented)
  return s * d;
}

#endif
