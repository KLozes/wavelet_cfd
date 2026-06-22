#ifndef SDF_STL_H
#define SDF_STL_H

// Reader for STereoLithography (STL) triangle meshes. Auto-detects binary vs
// ASCII. Each output triangle stores its three vertices in the file ordering
// (counter-clockwise when viewed from outside) plus the stored facet normal.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Vec3f.cuh"

struct StlTri {
  float3 v[3];
  float3 n;   // facet normal as listed in the file (may be zero / unreliable)
};

// Read an ASCII STL. Returns false if the file does not look like ASCII STL.
inline bool readStlAscii(const std::string& path, std::vector<StlTri>& tris) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return false;

  char tok[256];
  std::vector<float3> verts;
  float3 normal = make_float3(0, 0, 0);
  bool sawFacet = false;
  tris.clear();

  while (std::fscanf(f, "%255s", tok) == 1) {
    if (std::strcmp(tok, "facet") == 0) {
      sawFacet = true;
      // expect: normal nx ny nz
      if (std::fscanf(f, "%255s %f %f %f", tok, &normal.x, &normal.y, &normal.z) != 4) break;
    } else if (std::strcmp(tok, "vertex") == 0) {
      float3 v;
      if (std::fscanf(f, "%f %f %f", &v.x, &v.y, &v.z) != 3) break;
      verts.push_back(v);
    } else if (std::strcmp(tok, "endfacet") == 0) {
      if (verts.size() == 3) {
        StlTri t;
        t.v[0] = verts[0]; t.v[1] = verts[1]; t.v[2] = verts[2];
        t.n = normal;
        tris.push_back(t);
      }
      verts.clear();
    }
  }
  std::fclose(f);
  return sawFacet && !tris.empty();
}

// Read a binary STL (80-byte header, uint32 count, 50 bytes/triangle).
inline bool readStlBinary(const std::string& path, std::vector<StlTri>& tris) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;

  uint8_t header[80];
  if (std::fread(header, 1, 80, f) != 80) { std::fclose(f); return false; }
  uint32_t nTri = 0;
  if (std::fread(&nTri, 4, 1, f) != 1) { std::fclose(f); return false; }

  tris.clear();
  tris.reserve(nTri);
  for (uint32_t t = 0; t < nTri; ++t) {
    float buf[12];          // normal(3) + 3 vertices(9)
    uint16_t attr;
    if (std::fread(buf, 4, 12, f) != 12) { std::fclose(f); return false; }
    if (std::fread(&attr, 2, 1, f) != 1) { std::fclose(f); return false; }
    StlTri tri;
    tri.n    = make_float3(buf[0], buf[1], buf[2]);
    tri.v[0] = make_float3(buf[3], buf[4], buf[5]);
    tri.v[1] = make_float3(buf[6], buf[7], buf[8]);
    tri.v[2] = make_float3(buf[9], buf[10], buf[11]);
    tris.push_back(tri);
  }
  std::fclose(f);
  return true;
}

// Detect format by file size: a valid binary STL is exactly 84 + 50*nTri bytes.
inline bool readStl(const std::string& path, std::vector<StlTri>& tris) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  uint8_t header[80] = {0};
  size_t got = std::fread(header, 1, std::min<long>(80, size), f);
  uint32_t nTri = 0;
  if (size >= 84) got += std::fread(&nTri, 4, 1, f);
  (void)got;
  std::fclose(f);

  bool looksBinary = (size >= 84) &&
                     (static_cast<long>(84 + 50ull * nTri) == size);
  if (looksBinary) return readStlBinary(path, tris);
  if (readStlAscii(path, tris)) return true;
  return readStlBinary(path, tris);  // last resort
}

#endif
