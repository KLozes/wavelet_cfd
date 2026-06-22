#!/usr/bin/env python3
# Write a UV-sphere ASCII STL for analytic SDF validation.
import numpy as np, sys

R   = 0.30
C   = np.array([0.5, 0.5, 0.5])
nu, nv = 48, 96           # stacks, slices
out = sys.argv[1] if len(sys.argv) > 1 else "assets/sphere.stl"

def p(u, v):
    th = np.pi * u / nu          # 0..pi
    ph = 2*np.pi * v / nv        # 0..2pi
    return C + R*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

tris = []
for i in range(nu):
    for j in range(nv):
        a = p(i,   j);   b = p(i+1, j); c = p(i+1, j+1); d = p(i, j+1)
        for (x, y, z) in [(a, b, c), (a, c, d)]:
            n = np.cross(y - x, z - x); n = n/(np.linalg.norm(n) + 1e-30)
            tris.append((n, x, y, z))

with open(out, "w") as f:
    f.write("solid sphere\n")
    for n, x, y, z in tris:
        f.write(f" facet normal {n[0]} {n[1]} {n[2]}\n  outer loop\n")
        for vtx in (x, y, z):
            f.write(f"   vertex {vtx[0]} {vtx[1]} {vtx[2]}\n")
        f.write("  endloop\n endfacet\n")
    f.write("endsolid sphere\n")
print(f"wrote {out}: sphere R={R} center={C.tolist()}, {len(tris)} triangles")
