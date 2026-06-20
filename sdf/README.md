# Narrowband Signed Distance Field (3D, CUDA)

Computes a narrowband signed distance field (SDF) from a triangulated surface
(STL) on a uniform 3D Cartesian grid on the GPU, following the
Characteristic / Scan-Conversion approach of Roosing, Strickson & Nikiforakis,
*Fast Distance Fields for Fluid Dynamics Mesh Generation on Graphics Hardware*,
Commun. Comput. Phys. 26(3), 2019 (see [`../docs/NarrowBandGpuSDF.md`](../docs/NarrowBandGpuSDF.md)).

## Build & run

Built from the repository root with the shared `Makefile` (which also builds the
compressible `wave3d` solver). Both executables land in the root directory.

```sh
make                 # builds narrowbandSDF and wave3d
make narrowbandSDF   # just this solver
./narrowbandSDF [file.stl] [resolution] [band_cells]
```

| argument     | meaning                                              | default          |
| ------------ | ---------------------------------------------------- | ---------------- |
| `file.stl`   | input mesh (binary or ASCII STL)                     | `assets/wing.stl`|
| `resolution` | cells across the longest bounding-box axis           | `128`            |
| `band_cells` | narrowband half-width in cells (`~5` per the paper)  | `5`              |

Example (run from the repo root):

```sh
./narrowbandSDF assets/wing.stl 256 5
```

Outputs (under `output/`, like the compressible solver):

- `output/<name>_sdf.vtk` — legacy VTK `STRUCTURED_POINTS` (binary, big-endian)
  with the signed distance at each cell centre; cells outside the active blocks
  are set to `+band`. Open in ParaView/VisIt and threshold/contour the `sdf`
  scalar.
- `output/<name>_{xy,xz,yz}.png` — center-plane slice images, written the same
  way as the compressible solver's `paint()`: each slice is gathered from the
  sparse grid into a dense image and rescaled `[min,max] -> [0,65535]` as a
  16-bit grayscale PNG (`SingleLevelSparseGrid::paintSlices`).

### Contour plots

For publication-style filled distance bands with a green zero-level (surface)
contour and a cell grid, post-process the `.vtk` with `plot_slices.py` (needs
numpy + scipy + matplotlib):

```sh
python3 sdf/plot_slices.py output/wing_sdf.vtk --axis xz --levels 16
```

The surface (zero set) comes from the narrowband; distances beyond the band are
filled with a Euclidean distance transform (scipy) so the whole domain carries
bands. The styling matches `signed_heat.py`: evenly spaced topographic
iso-distance stripes, two alternating tones per side (light/deep blue inside,
light/rose outside) with `0` on a band edge, a bold zero contour, a lime surface
outline, and a faint cell grid. Flags:

| flag         | meaning                                                          |
| ------------ | --------------------------------------------------------------- |
| `--axis`     | `xy` / `xz` / `yz` / `all` (default `all`)                      |
| `--index`    | slice index along the axis (default: center)                    |
| `--levels`   | iso-distance stripes inside the body (default 6)                |
| `--clip`     | color-scale percentile of the distance (default 99)             |
| `--supersample` | sub-cell upsampling factor for smooth bands (default 4; 1 = off) |
| `--vmax`     | clip the outer range to `vmax` instead of the percentile        |
| `--no-grid`  | hide the cell grid                                              |
| `--no-fill`  | plot the raw narrowband instead of the reconstructed full field |

A wider solver band gives a deeper interior (more inside stripes), e.g.
`./narrowbandSDF assets/wing.stl 1024 16`.

## How it works

1. **STL read** (`Stl.h`) — binary/ASCII auto-detected.
2. **Feature build** (`Features.h`, host) — duplicate vertices are welded, edges
   are paired with their two incident faces, and each vertex gets an
   angle-weighted pseudonormal (Bærentzen & Aanæs). The paper does this on the
   GPU with Morton codes + Thrust; here it is a host-side hash/weld, which is
   negligible next to the GPU distance sweep. Each triangle is packed into a
   `TriFeat` carrying its geometry, face normal, three vertex pseudonormals, and
   three edge pseudonormals.
3. **Sparse grid** (`SingleLevelSparseGrid`) — a single-level (uniform) 3D
   sparse grid built on the repo's GPU `HashTable` (`../src`). As in
   `MultiLevelSparseGrid`, the hash table is **block-structured**: each entry
   maps a *block* location code to a block index, and each block owns a
   contiguous `blockSize³` brick of cells in the value array (`blockSize` from
   `Settings.cuh`). Only blocks that touch the narrowband are activated, so the
   hash table stays small (e.g. ~15k block entries vs. ~766k cells for the wing
   at res 512) and memory scales with the surface, not the domain volume.
4. **Distance transform** (`SdfKernel.cuh`) — one GPU thread per triangle loops
   over the cells in the triangle's bounding volume (AABB grown by the band):
   - *pass 1* activates every block touched by the band (`activateBlock`);
   - *pass 2* computes the closest point on the triangle (Ericson), takes the
     sign from the pseudonormal of the owning feature (face/edge/vertex), and
     keeps the smallest-magnitude value with an atomic min (CAS on the float
     bit pattern).
   The two-pass split makes the hash-table reads in pass 2 race-free.

The pseudonormal sign test is exact for the convex / concave / saddle / ruff
vertex cases the paper discusses, so saddle/ruff geometries need no special
casing here.

## Validation

- **Box** (planar faces, sharp edges, corner vertices): max error vs. the
  analytic box SDF was `8.7e-6` (float round-off) with **0** sign errors over
  ~195k narrowband cells.
- **Sphere** (curved, many convex features): matches the *faceted* mesh to
  within float precision; the only deviations from the *ideal* sphere SDF lie
  within the mesh's faceting error.

## Notes / limitations

- This produces a **narrowband** field: the deep interior beyond `band_cells`
  stays at `+band` (the paper's §11.3 note). A coordinate-axis flood fill can
  recover interior signs if a full field is needed.
- Active-block count must stay below `blockCapacity`
  (`nCellsMax / blockSize³`, from `Settings.cuh`); the program errors out
  otherwise. Because the table holds blocks rather than cells, this allows much
  higher resolutions than a per-cell table.
- Built for `sm_75` (GTX 1650 / CUDA 13). Change `ARCH` in the root `Makefile`
  for other GPUs.
