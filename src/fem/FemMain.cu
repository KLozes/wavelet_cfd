// CutFEM linear-elasticity solver on a block-sparse background grid.
//
// Two geometry sources:
//
//   BANK   an IDS aero bank file.  The blade row's MASTER sections are lofted
//          FULL SPAN in C++ (Bank.h, BladeGeom.h) into a closed solid, and the
//          hub cut, root fillet, tip clearance and one-pitch sector are then
//          applied as SDF algebra against the platform and shroud
//          (BladeSdf.cuh).  Doing the blends in the SDF rather than on the
//          surface is what makes the fillet a true rolling-ball blend along the
//          real (oblique) blade/platform intersection curve.
//
//   STL    any closed triangulated surface.  Same solver, all the blade-specific
//          SDF terms switched off.
//
// The method is Hansbo, Larson & Larsson, "Cut Finite Element Methods for
// Linear Elasticity Problems" (arXiv:1703.04377): weak (Nitsche) boundary
// conditions on a mesh the geometry cuts arbitrarily, plus a ghost-penalty
// stabilization on the interior faces of cut elements.
//
//   usage:  ./wavefem --bank assets/bank_v98d.txt --row "ROTOR 1" [options]
//           ./wavefem body.stl [options]
//
//   geometry (bank mode)
//     --bank FILE      IDS bank file
//     --row  LABEL     blade row, e.g. "ROTOR 1"   (--rows lists them)
//     --nsectors N     sector count; 0 = the row's own blade count
//     --platform T     platform thickness below the hub line   (default 0.5)
//     --platmargin f   platform axial overhang, as a fraction of the root axial
//                      chord                                    (default 0.25)
//     --fillet R       root fillet radius                       (default 0.05)
//     --gap G          tip clearance                            (default 0.02)
//     --nspan N        span stations in the full-span loft      (default 96)
//     --refine f       contour density vs the bank's own sections (default 1)
//     --noplatform     blade only, no platform / fillet
//     --nosector       do not cut to one pitch
//
//   discretization / solver
//     --res N          cells along the longest bounding-box axis  (default 64)
//     --case mms|load  manufactured solution, or the load case    (default load
//                      for bank input, mms for STL)
//     --E v --nu v     Young's modulus / Poisson ratio        (default 1, 0.3)
//     --rho v          density                                    (default 1)
//     --rpm v          shaft speed -> centrifugal body load    (default 0)
//     --gravity a,b,c  uniform body force per unit volume
//     --k a,b,c        MMS wave numbers                       (default 2,3,4)
//     --gammad v       Nitsche penalty beta                    (default 1000)
//     --gammaa v       ghost penalty gamma_a; 0 disables the stabilization
//     --stab 0|1       (2.20) uniform, or (2.25) weak on Neumann faces
//     --sub 1|2        cut-cell sub-division per direction        (default 1)
//     --tol v --maxit N   CG controls
//     --margin f       domain padding, as a fraction of the body extent
//     --novtu          skip the VTK output

#include <sys/stat.h>
#include <cstdint>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Stl.h"
#include "Features.h"
#include "Bvh.h"
#include "Bank.h"
#include "BladeGeom.h"
#include "CutFemSolver.cuh"
#include "CutFemSolverKernels.cuh"
#include "CutQuad.cuh"      // FEM_MAXSUB
#include "PolyFit.h"        // fitPoly3 (per-cell parabolic reconstruction, --recon)
#include "SayeQuad.h"
#include "SbmSolve.h"      // host-only shifted-boundary solver (--sbm)

// adapter: let SbmSolve.h evaluate the real BladeSdf level set
static const BladeSdf *g_bladeSdf = nullptr;
static double bladeSdfEval(double x,double y,double z){
  return (double)g_bladeSdf->phi((real)x,(real)y,(real)z);
}       // sayeSurface (3-D reconstructed boundary, --recon3d)

static std::string baseName(const std::string &path) {
  size_t s = path.find_last_of("/\\");
  std::string f = (s == std::string::npos) ? path : path.substr(s + 1);
  size_t d = f.find_last_of('.');
  return (d == std::string::npos) ? f : f.substr(0, d);
}

// "a,b,c" -> v[3]  (parsed as double, stored in the solver precision)
static void parse3(const char *s, real v[3]) {
  double a = 0, b = 0, c = 0;
  if (sscanf(s, "%lf,%lf,%lf", &a, &b, &c) == 3) {
    v[0] = (real)a; v[1] = (real)b; v[2] = (real)c;
  }
}

int main(int argc, char *argv[]) {
  std::string stlPath, bankPath, rowLabel = "ROTOR 1", caseName;
  i32   res = 64, maxit = 20000, stab = 0, sub = 1, novtu = 0, listRows = 0;
  i32   nSectors = 0, nSpanLoft = 96, noPlatform = 0, noSector = 0, doSlice = 0, cylMode = 0, noPeriodic = 0;
  i32   sbmP = 0, sbmRes = 0; std::vector<i32> sbmResList;
  double fitSliceZ = -1e30; i32 fitNF = 700, fitDeg = 2;
  double sbmBox[4] = {0,0,0,-1};   // cx,cy,cz,half : solve a SUB-BOX instead of the whole model
  double sbmK = 0;                 // MMS wave number (0 => one wavelength across the box)
  i32   femMethod = 0;
  i32   nCopies = 1, spdCheck = 0, isoN = 0, rangeTest = 0, femOrder = 1, reconP = 0, recon3dP = 0;
  real  E = 1, nu = (real)0.3, rhoMat = 1, margin = (real)0.05, refine = 1;
  real  gammaD = 1000, gammaA = -1, tol = (real)1e-10;
  real  platThick = (real)0.5, platMargin = (real)0.25;
  real  filletR = (real)0.05, tipGap = (real)0.02, rpm = 0;
  real  kw[3] = {2, 3, 4}, grav[3] = {0,0,0};

  for (i32 a = 1; a < argc; a++) {
    std::string s = argv[a];
    auto next = [&](void) -> const char* { return (a+1 < argc) ? argv[++a] : "0"; };
    if      (s == "--bank")      bankPath = next();
    else if (s == "--row")       rowLabel = next();
    else if (s == "--rows")      listRows = 1;
    else if (s == "--nsectors")  nSectors = atoi(next());
    else if (s == "--platform")  platThick = (real)atof(next());
    else if (s == "--platmargin")platMargin = (real)atof(next());
    else if (s == "--fillet")    filletR = (real)atof(next());
    else if (s == "--gap")       tipGap = (real)atof(next());
    else if (s == "--nspan")     nSpanLoft = atoi(next());
    else if (s == "--refine")    refine = (real)atof(next());
    else if (s == "--noplatform")noPlatform = 1;
    else if (s == "--nosector")  noSector = 1;
    else if (s == "--res")       res = atoi(next());
    else if (s == "--p")         femOrder = atoi(next());
    else if (s == "--method")  { std::string v=next(); femMethod = (v=="sbm") ? 1 : 0; }
    else if (s == "--case")      caseName = next();
    else if (s == "--E")         E = (real)atof(next());
    else if (s == "--nu")        nu = (real)atof(next());
    else if (s == "--rho")       rhoMat = (real)atof(next());
    else if (s == "--rpm")       rpm = (real)atof(next());
    else if (s == "--gravity")   parse3(next(), grav);
    else if (s == "--k")         parse3(next(), kw);
    else if (s == "--gammad")    gammaD = (real)atof(next());
    else if (s == "--gammaa")    gammaA = (real)atof(next());
    else if (s == "--stab")      stab = atoi(next());
    else if (s == "--sub")       sub = atoi(next());
    else if (s == "--tol")       tol = (real)atof(next());
    else if (s == "--maxit")     maxit = atoi(next());
    else if (s == "--margin")    margin = (real)atof(next());
    else if (s == "--novtu")     novtu = 1;
    else if (s == "--slice")     doSlice = 1;
    else if (s == "--cyl")       cylMode = 1;
    else if (s == "--noperiodic")noPeriodic = 1;
    else if (s == "--copies")    nCopies = atoi(next());
    else if (s == "--spd")       spdCheck = 1;
    else if (s == "--rangetest") rangeTest = 1;
    else if (s == "--iso")       isoN = atoi(next());
    else if (s == "--recon")     reconP = atoi(next());   // dump level-set vs deg-p reconstruction slices
    else if (s == "--sbm")       sbmP = atoi(next());   // host-only SBM at order p
    else if (s == "--sbmbox") { std::string v=next(); size_t q=0; int c4=0;
      while(q<v.size()&&c4<4){ size_t c=v.find(',',q);
        std::string t=v.substr(q,c==std::string::npos?c:c-q);
        if(!t.empty()) sbmBox[c4++]=atof(t.c_str());
        if(c==std::string::npos) break; q=c+1; } }
    else if (s == "--sbmk")      sbmK = atof(next());
    else if (s == "--fitslice")  fitSliceZ = atof(next());  // dump fitted vs true zero contour
    else if (s == "--fitnf")     fitNF = atoi(next());
    else if (s == "--fitdeg")    fitDeg = atoi(next());
    else if (s == "--sbmres") { std::string v=next(); size_t q=0;
      while(q<v.size()){ size_t c=v.find(',',q); std::string t=v.substr(q,c==std::string::npos?c:c-q);
        if(!t.empty()) sbmResList.push_back(atoi(t.c_str()));
        if(c==std::string::npos) break; q=c+1; } }
    else if (s == "--recon3d")   recon3dP = atoi(next()); // dump the 3-D deg-p reconstructed boundary
    else if (s[0] != '-')        stlPath = s;
    else { printf("unknown option %s\n", s.c_str()); return 1; }
  }
  bool bankMode = !bankPath.empty();
  if (caseName.empty()) caseName = bankMode ? "load" : "mms";

  // -------------------------------------------------------------------------
  //  geometry: build a closed triangle soup + the composite SDF parameters
  // -------------------------------------------------------------------------
  std::vector<StlTri> tris;
  BladeSdf sdf;                       // all blade terms default to "off"
  std::string tag;
  real domLo[3], domHi[3];            // world-space bounds of the SOLID
  double volExact = 0, areaExact = 0;
  std::vector<real> hubTab, casTab, thcTab;
  double cylArc = 0, cylH = 0;
  i32 cylNTheta = 0;
  real clampR = 0;

  if (bankMode) {
    bank::Bank B;
    if (!bank::read(bankPath, B)) {
      fprintf(stderr, "error: could not read bank file '%s'\n", bankPath.c_str());
      return 1;
    }
    printf("bank   : %s  (%zu stations, %zu blade rows)\n",
           bankPath.c_str(), B.stations.size(), B.rows.size());
    if (listRows) {
      for (const bank::Row &r : B.rows)
        printf("  %-9s blades=%3.0f  sections=%2zu  LE r = %.3f .. %.3f  z = %.3f\n",
               r.label.c_str(), r.nblades, r.sections.size(),
               r.leR.front(), r.leR.back(), r.leZ.front());
      return 0;
    }
    const bank::Row *row = B.findRow(rowLabel);
    if (!row) { fprintf(stderr, "error: no blade row '%s' (try --rows)\n", rowLabel.c_str()); return 1; }
    if (nSectors <= 0) nSectors = (i32)llround(row->nblades);
    tag = baseName(bankPath) + "_" + baseName(rowLabel);
    for (char &c : tag) if (c == ' ') c = '_';

    // Full-span loft: run it just past both walls so the SDF does the cutting,
    // not the mesh.  The overshoot is chosen from the GEOMETRY -- step outward
    // until the section fully clears the wall by `clear` -- because a fixed span
    // fraction either fails to clear a sloping wall or extrapolates so far that
    // the section balloons.
    std::vector<double> sp = blade::spanParam(*row);
    const double clear = 0.6*platThick;
    std::vector<double> mz, mr, mt;
    auto clears = [&](double f, bool below) {
      blade::Contour c = blade::interpContour(*row, sp, f);
      blade::sectionLoop(c, 0.25, mz, mr, mt);
      for (size_t k = 0; k < mz.size(); k++) {
        double d = below ? (mr[k] - (B.hubAt(mz[k]) - platThick))
                         : ((B.casAt(mz[k]) - mr[k]));
        if (d > -clear) return false;          // some point has not cleared yet
      }
      return true;
    };
    double fLo = 0.0, fHi = 1.0;
    while (fLo > -0.6 && !clears(fLo, true))  fLo -= 0.02;
    while (fHi <  1.6 && !clears(fHi, false)) fHi += 0.02;
    printf("       : loft span fraction %.3f .. %.3f (clears both walls by %.3f)\n",
           fLo, fHi, clear);
    blade::buildBladeMesh(*row, fLo, fHi, nSpanLoft, refine, tris);

    blade::meshProps(tris, volExact, areaExact);
    {
      int nDegen = 0;
      int nOpen = blade::countOpenEdges(tris, &nDegen);
      printf("blade  : loft watertight = %s (%d unpaired edges, %d degenerate tris)\n",
             nOpen == 0 ? "YES" : "NO", nOpen, nDegen);
      if (nOpen) printf("         WARNING: the ray-parity sign test is unreliable on an"
                        " open surface --\n         expect isolated sign flips in the"
                        " level set.\n");
    }
    printf("blade  : %s  %zu blades -> %d sectors, pitch %.4f deg\n",
           rowLabel.c_str(), (size_t)llround(row->nblades), nSectors, 360.0/nSectors);
    printf("       : full-span loft %zu triangles, volume %.6g (untrimmed)\n",
           tris.size(), volExact);

    // ---- wall tables over the row's axial range ---------------------------
    double zA = *std::min_element(row->leZ.begin(), row->leZ.end());
    double zB = *std::max_element(row->teZ.begin(), row->teZ.end());
    if (zB < zA) std::swap(zA, zB);
    double chord = zB - zA;
    double wz0 = zA - 1.0*chord, wz1 = zB + 1.0*chord;
    const i32 NW = 512;
    hubTab.resize(NW); casTab.resize(NW); thcTab.resize(NW);
    for (i32 i = 0; i < NW; i++) {
      double z = wz0 + (wz1 - wz0)*i/(double)(NW-1);
      hubTab[i] = (real)B.hubAt(z);
      casTab[i] = (real)B.casAt(z);
    }
    sdf.nWall = NW; sdf.wallZ0 = (real)wz0; sdf.wallDz = (real)((wz1-wz0)/(NW-1));

    // ---- sector centreline theta_c(z) = passage camber line ---------------
    //
    // Ported from omesh.py / parse_bank.sector_block_at.  The centreline is the
    // blade CAMBER: the midpoint of the suction and pressure surfaces
    // (blade::camberLine), NOT the mean of all contour vertices -- the midpoint
    // is free of the blunt LE/TE arc artefacts that put a spurious dip in a
    // vertex mean.  It is SPAN-AVERAGED because the blade twists more than a
    // pitch from root to tip (root LE and mid LE differ by ~0.25 rad > pi/N), so
    // no single-span centreline could keep every span inside the +-pi/N sector;
    // the span average does (verified: the blade stays within +-pi/N of it).
    //
    // In the platform overhangs (upstream of the LE, downstream of the TE) the
    // camber is continued by its TANGENT as a straight line in (z, theta).  The
    // tangent is taken at the ROOT section -- the overhang lives at the hub, and
    // the root blade angles are the true inflow/outflow directions there -- fit
    // over the first/last 12% of chord exactly as omesh.py does.
    {
      const int NF = 41;
      std::vector<double> sp = blade::spanParam(*row);
      std::vector<double> camAvg(NF, 0.0), zAvg(NF, 0.0);
      const int NSPAN = (int)row->sections.size();
      std::vector<double> thk, zk, rk;
      for (int s = 0; s < NSPAN; s++) {
        blade::camberLine(*row, sp, (double)s/(NSPAN-1), NF, thk, zk, rk);
        for (int k = 0; k < NF; k++) { camAvg[k] += thk[k]; zAvg[k] += zk[k]; }
      }
      for (int k = 0; k < NF; k++) { camAvg[k] /= NSPAN; zAvg[k] /= NSPAN; }
      // ROOT tangents (d theta/dz), fit over the first/last 12% of chord
      std::vector<double> thR, zR, rR;
      blade::camberLine(*row, sp, 0.0, NF, thR, zR, rR);
      int kf = std::max(1, (int)(0.12*NF));
      double sLE = (thR[kf] - thR[0])/(zR[kf] - zR[0]);
      double sTE = (thR[NF-1] - thR[NF-1-kf])/(zR[NF-1] - zR[NF-1-kf]);
      double zLEc = zAvg[0], zTEc = zAvg[NF-1];

      double dzTab = (wz1 - wz0)/(NW-1);
      auto camAt = [&](double z) {
        if (z <= zLEc) return camAvg[0]   + sLE*(z - zLEc);   // upstream tangent
        if (z >= zTEc) return camAvg[NF-1] + sTE*(z - zTEc);  // downstream tangent
        // interior: interpolate the span-averaged camber by z (zAvg is monotone)
        int k = 0; while (k < NF-2 && zAvg[k+1] < z) k++;
        double w = (z - zAvg[k])/(zAvg[k+1] - zAvg[k]);
        return camAvg[k] + w*(camAvg[k+1] - camAvg[k]);
      };
      for (i32 i = 0; i < NW; i++) thcTab[i] = (real)camAt(wz0 + i*dzTab);

      if (doSlice) {   // dump centreline + per-span camber + blade band
        FILE *df = fopen(("output/" + tag + "_thc.txt").c_str(), "w");
        double pf0 = zA - platMargin*chord, pf1 = zB + platMargin*chord;
        fprintf(df, "# z  thc  thmin  thmax  cnt  (blade z=%.4f..%.4f  platform z=%.4f..%.4f"
                "  sLE=%.4f sTE=%.4f)\n", zLEc, zTEc, pf0, pf1, sLE, sTE);
        std::vector<double> bmn(NW, 1e30), bmx(NW, -1e30);
        for (const StlTri &t : tris)
          for (i32 k = 0; k < 3; k++) {
            double zz = t.v[k].z, tt = std::atan2((double)t.v[k].y, (double)t.v[k].x);
            i32 bi = (i32)llround((zz - wz0)/dzTab);
            if (bi >= 0 && bi < NW) { bmn[bi]=std::min(bmn[bi],tt); bmx[bi]=std::max(bmx[bi],tt); }
          }
        for (i32 i = 0; i < NW; i += 3) {
          double zz = wz0 + i*dzTab;
          if (zz < pf0-0.1 || zz > pf1+0.1) continue;
          fprintf(df, "%.5f %.6f %.6f %.6f %.0f\n", zz, (double)thcTab[i],
                  bmn[i]<1e29?bmn[i]:0.0, bmn[i]<1e29?bmx[i]:0.0, bmn[i]<1e29?1.0:0.0);
        }
        fclose(df);
      }
      sdf.hubTab = hubTab.data();
      sdf.casTab = casTab.data();
      sdf.thcTab = thcTab.data();
      printf("       : passage camber centreline, ROOT tangent extensions "
             "(LE slope %.4f, TE slope %.4f rad/len)\n", sLE, sTE);
      printf("       : theta_c sweeps %.3f -> %.3f deg (%.2f pitches) across the row\n",
             camAvg[0]*180/PI, camAvg[NF-1]*180/PI, (camAvg[NF-1]-camAvg[0])/(2*PI/nSectors));
    }

    // ---- platform + blends ------------------------------------------------
    double zmid = 0.5*(zA + zB);
    double rh = B.hubAt(zmid), rc = B.casAt(zmid);
    sdf.platOn    = noPlatform ? 0 : 1;
    sdf.platThick = platThick;
    sdf.platZ0    = (real)(zA - platMargin*chord);
    sdf.platZ1    = (real)(zB + platMargin*chord);
    sdf.filletR   = noPlatform ? 0 : filletR;
    sdf.tipGap    = tipGap;
    if (!noSector) {
      sdf.halfPitch = (real)(PI/nSectors);
      sdf.pitch     = (real)(2*PI/nSectors);
      sdf.nCopies   = nCopies;    // the root spans more than one pitch
    }
    // With a periodic cylindrical sector the theta faces are GRID faces, not
    // cut surfaces: the level set must not cut in theta at all, or the boundary
    // would be handled by the immersed machinery (Nitsche / ghost penalty)
    // instead of by the cyclic node tie.  The neighbour blade copies stay --
    // they are what fills the sector where this blade has swept out of it.
    if (cylMode && !noPeriodic && !noSector) { sdf.halfPitch = 0; sdf.wrapTheta = 1; }
    clampR = (real)(rh - platThick);          // the platform underside

    printf("       : hub r = %.4f, casing r = %.4f (span %.4f) at z = %.4f\n",
           rh, rc, rc - rh, zmid);
    printf("       : platform %.4f thick, z = %.4f .. %.4f%s\n",
           (double)platThick, (double)sdf.platZ0, (double)sdf.platZ1,
           noPlatform ? "  (DISABLED)" : "");
    printf("       : root fillet %.4f, tip gap %.4f (%.2f%% span)%s\n",
           (double)sdf.filletR, (double)tipGap, 100.0*tipGap/(rc-rh),
           noSector ? "   sector cut DISABLED" : "");

    // ---- computational bounds --------------------------------------------
    double zLo = std::min((double)sdf.platZ0, zA - 0.1*chord);
    double zHi = std::max((double)sdf.platZ1, zB + 0.1*chord);
    double hp = noSector ? PI : PI/nSectors;
    double rMin = 1e30, rMax = -1e30;
    for (i32 iz = 0; iz <= 64; iz++) {
      double zz = zLo + (zHi - zLo)*iz/64.0;
      rMin = std::min(rMin, B.hubAt(zz) - platThick);
      rMax = std::max(rMax, B.casAt(zz) - tipGap);
    }
    if (cylMode) {
      // CYLINDRICAL: the swept one-pitch sector IS the computational box.
      //   q0 = r,  q1 = rRef*theta',  q2 = z
      // q1 is an arc length at the reference radius so all three coordinates
      // are lengths and the solver's cubic cells stay cubic in physical space
      // near rRef.
      sdf.coordMode = 1;
      sdf.rRef = (real)(0.5*(rMin + rMax));
      domLo[0] = (real)rMin;              domHi[0] = (real)rMax;
      domLo[1] = (real)(-sdf.rRef*hp);    domHi[1] = (real)(sdf.rRef*hp);
      domLo[2] = (real)zLo;               domHi[2] = (real)zHi;
      // The pitch has to be an exact whole number of cells so the periodic
      // boundary lands on cell faces.  Snap the cell size to arc/nTheta (nTheta
      // a multiple of blockSize) and let the r and z directions -- which have
      // padding to spare -- round up to it, so the cells stay cubic.
      cylArc = 2.0*hp*(double)sdf.rRef;
      double hWant = std::max(rMax - rMin, zHi - zLo)/res;
      i32 nb = std::max(1, (i32)llround(cylArc/(hWant*blockSize)));
      cylNTheta = nb*blockSize;
      cylH = cylArc/cylNTheta;
      printf("       : CYLINDRICAL grid  r = %.4f..%.4f, arc = %.4f..%.4f"
             " (rRef = %.4f), z = %.4f..%.4f\n",
             rMin, rMax, (double)domLo[1], (double)domHi[1], (double)sdf.rRef, zLo, zHi);
      printf("       : pitch = %d cells exactly (h = %.6g), periodic faces on"
             " cell boundaries\n", cylNTheta, cylH);
    } else {
      // CARTESIAN: bound the swept wedge by sampling it
      double lo[3] = {1e30,1e30,1e30}, hi[3] = {-1e30,-1e30,-1e30};
      for (i32 iz = 0; iz <= 64; iz++) {
        double zz = zLo + (zHi - zLo)*iz/64.0;
        double rmin = B.hubAt(zz) - platThick, rmax = B.casAt(zz) - tipGap;
        double t0 = (double)sdf.thc((real)zz);
        for (i32 it = 0; it <= 64; it++) {
          double tt = t0 + (noSector ? (-PI + 2*PI*it/64.0) : (-hp + 2*hp*it/64.0));
          for (i32 ir = 0; ir < 2; ir++) {
            double rr2 = ir ? rmax : rmin;
            double pxyz[3] = {rr2*std::cos(tt), rr2*std::sin(tt), zz};
            for (i32 d = 0; d < 3; d++) {
              lo[d] = std::min(lo[d], pxyz[d]); hi[d] = std::max(hi[d], pxyz[d]);
            }
          }
        }
      }
      for (i32 d = 0; d < 3; d++) { domLo[d] = (real)lo[d]; domHi[d] = (real)hi[d]; }
    }
  } else {
    if (stlPath.empty()) {
      const char *cand[] = {"assets/sphere.stl", "../assets/sphere.stl"};
      for (const char *c : cand) if (readStl(c, tris)) { stlPath = c; break; }
    } else {
      readStl(stlPath, tris);
    }
    if (tris.empty()) {
      fprintf(stderr, "error: could not read an STL mesh (tried '%s')\n",
              stlPath.empty() ? "assets/sphere.stl" : stlPath.c_str());
      return 1;
    }
    tag = baseName(stlPath);
    blade::meshProps(tris, volExact, areaExact);
    printf("mesh   : %s  (%zu triangles)\n", stlPath.c_str(), tris.size());
    printf("       : exact volume %.8g, surface area %.8g\n", volExact, areaExact);
    float3 lo = make_float3(1e30f,1e30f,1e30f), hi = make_float3(-1e30f,-1e30f,-1e30f);
    for (const StlTri &t : tris)
      for (i32 k = 0; k < 3; k++) { lo = fmin3(lo, t.v[k]); hi = fmax3(hi, t.v[k]); }
    domLo[0]=lo.x; domLo[1]=lo.y; domLo[2]=lo.z;
    domHi[0]=hi.x; domHi[1]=hi.y; domHi[2]=hi.z;
  }

  // -------------------------------------------------------------------------
  //  BVH over the (full-span) solid
  // -------------------------------------------------------------------------
  std::vector<TriFeat> feats;
  int nVerts, nEdges;
  float3 bmin, bmax;
  buildFeatures(tris, feats, nVerts, nEdges, bmin, bmax);
  Bvh bvh = buildBvh(feats);
  // ---- --fitslice : fitted (degree-q) level set vs TRUE level set ---------
  // Dumps a constant-z slice of BOTH phi_true and phi_fit, where phi_fit is the
  // per-cell degree-q polynomial fitted to the (q+1)^3 GLL samples of the true
  // level set -- i.e. exactly the geometry the Qp solver actually "sees".
  // Contouring both at 0 shows what the fitted geometry keeps and what it loses
  // (e.g. a sub-cell root fillet).  Host-only, before any CUDA call.
  if (fitSliceZ > -1e29) {
    sdf.bvhNodes = bvh.nodes.data();
    sdf.bvhOrder = bvh.order.data();
    sdf.bvhTris  = feats.data();
    sdf.orient   = bvh.orient;
    if (!hubTab.empty()) { sdf.hubTab = hubTab.data(); sdf.casTab = casTab.data();
                           sdf.nWall = (i32)hubTab.size(); }
    if (!thcTab.empty()) sdf.thcTab = thcTab.data();
    g_bladeSdf = &sdf; g_sdfFn = &bladeSdfEval;
    double ext3[3], L = 0;
    for (i32 d = 0; d < 3; d++) { ext3[d] = (double)(domHi[d] - domLo[d]); L = fmax(L, ext3[d]); }
    double pad = 0.03*L, lo3[3];
    for (i32 d = 0; d < 3; d++) lo3[d] = (double)domLo[d] - 0.5*(L - ext3[d]) - pad;
    L += 2*pad;
    i32 N = sbmResList.empty() ? res : sbmResList[0];
    double h = L/N;
    i32 kz = (i32)floor((fitSliceZ - lo3[2])/h);
    if (kz < 0) kz = 0; if (kz >= N) kz = N-1;
    std::string fn = "output/" + tag + "_fitslice.txt";
    FILE *fp = fopen(fn.c_str(), "w");
    fprintf(fp, "# z=%.8f  res=%d  h=%.8f  deg=%d  lo=%.8f %.8f %.8f  L=%.8f\n",
            fitSliceZ, N, h, fitDeg, lo3[0], lo3[1], lo3[2], L);
    fprintf(fp, "# x y phi_true phi_fit\n");
    // cache one fitted polynomial per (cx,cy) column at this z-layer
    std::vector<PolyND> cache((size_t)N*N);
    std::vector<char> have((size_t)N*N, 0);
    double z0c = lo3[2] + kz*h;
    for (i32 j = 0; j < fitNF; j++) {
      double y = lo3[1] + L*(j + 0.5)/fitNF;
      for (i32 i = 0; i < fitNF; i++) {
        double x = lo3[0] + L*(i + 0.5)/fitNF;
        i32 cx = (i32)floor((x - lo3[0])/h), cy = (i32)floor((y - lo3[1])/h);
        if (cx < 0) cx = 0; if (cx >= N) cx = N-1;
        if (cy < 0) cy = 0; if (cy >= N) cy = N-1;
        size_t key = (size_t)cy*N + cx;
        if (!have[key]) { cache[key] = fitSdfCell(lo3[0]+cx*h, lo3[1]+cy*h, z0c, h, fitDeg);
                          have[key] = 1; }
        real xr[3] = { (real)((x - (lo3[0]+cx*h))/h), (real)((y - (lo3[1]+cy*h))/h),
                       (real)((fitSliceZ - z0c)/h) };
        double pt = bladeSdfEval(x, y, fitSliceZ);
        double pf = (double)cache[key].eval(xr);
        fprintf(fp, "%.6f %.6f %.6e %.6e\n", x, y, pt, pf);
      }
    }
    fclose(fp);
    printf("fitslice: %d x %d samples at z=%.5f (cell layer %d, h=%.5f, deg %d) -> %s\n",
           fitNF, fitNF, fitSliceZ, kz, h, fitDeg, fn.c_str());
    return 0;
  }

  // ---- --sbm : HOST-ONLY shifted-boundary (SBM) solve ---------------------
  // Runs the verified SbmSolve.h solver on a structured surrogate mesh over the
  // real geometry (platform, root fillet, tip gap, sector cut all included via
  // BladeSdf).  Deliberately placed BEFORE the first CUDA call and using the
  // HOST bvh/wall pointers, so it works with no usable GPU.  Cubic box of side
  // max(extent) with --res cells along it -- the same convention as the octree
  // path.  Exits when done; the GPU path is untouched.
  if (sbmP >= 1) {
    sdf.bvhNodes = bvh.nodes.data();
    sdf.bvhOrder = bvh.order.data();
    sdf.bvhTris  = feats.data();
    sdf.orient   = bvh.orient;
    if (!hubTab.empty()) { sdf.hubTab = hubTab.data(); sdf.casTab = casTab.data();
                           sdf.nWall = (i32)hubTab.size(); }
    if (!thcTab.empty()) sdf.thcTab = thcTab.data();
    g_bladeSdf = &sdf;
    g_sdfFn = &bladeSdfEval;
    double ext3[3], L = 0, lo3[3];
    if (sbmBox[3] > 0) {                 // explicit sub-box (e.g. around the fillet)
      L = 2*sbmBox[3];
      for (i32 d = 0; d < 3; d++) lo3[d] = sbmBox[d] - sbmBox[3];
    } else {
      for (i32 d = 0; d < 3; d++) { ext3[d] = (double)(domHi[d] - domLo[d]); L = fmax(L, ext3[d]); }
      double pad = 0.03*L;
      for (i32 d = 0; d < 3; d++) lo3[d] = (double)domLo[d] - 0.5*(L - ext3[d]) - pad;
      L += 2*pad;
    }
    // MMS wave number: default to one wavelength across the box so the
    // manufactured solution is resolvable at the resolutions actually used.
    KK = (sbmK > 0) ? sbmK : (2.0*PI/L);
    i32 Nfirst = sbmResList.empty() ? res : sbmResList[0];
    g_fdEps = L/(200.0*Nfirst);          // FD step << cell size
    printf("SBM (host-only, GSBM Eq.35)  p=%d  box side %.5f, origin (%.5f %.5f %.5f)  k_mms=%.4f\n",
           sbmP, L, lo3[0], lo3[1], lo3[2], (double)KK);
    printf("  %5s  %10s  %12s  %6s  %8s  %8s  %8s\n",
           "res","nDof","L2rel","ord","its","nElem","nFaceBC");
    double prevE = 0, prevH = 0;
    if (sbmResList.empty()) sbmResList.push_back(res);
    for (i32 rr = 0; rr < (i32)sbmResList.size(); rr++) {
      i32 Nr = sbmResList[rr];
      SbmOut o = sbmSolveOne(sbmP, Nr, lo3, L);
      double ord = (prevE > 0) ? log(prevE/o.l2abs)/log(prevH/o.h) : 0.0;
      printf("  %5d  %10ld  %12.4e  %6.2f  %8d  %8d  %8d\n",
             Nr, o.nd3, o.l2rel, ord, o.iters, o.nE, o.nBF);
      prevE = o.l2abs; prevH = o.h;
    }
    return 0;
  }

  BvhNode *dNodes = nullptr; i32 *dOrder = nullptr; TriFeat *dTris = nullptr;
  real *dHub = nullptr, *dCas = nullptr;
  cudaMallocManaged(&dNodes, bvh.nodes.size()*sizeof(BvhNode));
  cudaMallocManaged(&dOrder, bvh.order.size()*sizeof(i32));
  cudaMallocManaged(&dTris,  feats.size()*sizeof(TriFeat));
  memcpy(dNodes, bvh.nodes.data(), bvh.nodes.size()*sizeof(BvhNode));
  memcpy(dOrder, bvh.order.data(), bvh.order.size()*sizeof(i32));
  memcpy(dTris,  feats.data(),     feats.size()*sizeof(TriFeat));
  sdf.bvhNodes = dNodes; sdf.bvhOrder = dOrder; sdf.bvhTris = dTris;
  sdf.orient = bvh.orient;
  real *dThc = nullptr;
  if (!hubTab.empty()) {
    cudaMallocManaged(&dHub, hubTab.size()*sizeof(real));
    cudaMallocManaged(&dCas, casTab.size()*sizeof(real));
    cudaMallocManaged(&dThc, thcTab.size()*sizeof(real));
    memcpy(dHub, hubTab.data(), hubTab.size()*sizeof(real));
    memcpy(dCas, casTab.data(), casTab.size()*sizeof(real));
    memcpy(dThc, thcTab.data(), thcTab.size()*sizeof(real));
    sdf.hubTab = dHub; sdf.casTab = dCas; sdf.thcTab = dThc;
  }

  // -------------------------------------------------------------------------
  //  background grid: cubic cells covering the solid's world bounds
  // -------------------------------------------------------------------------
  real ext[3];
  real maxExt = 0;
  for (i32 d = 0; d < 3; d++) { ext[d] = domHi[d] - domLo[d]; maxExt = fmax(maxExt, ext[d]); }
  real h = (cylH > 0) ? (real)cylH : maxExt/(real)res;
  real origin[3], domainSize[3];
  i32  baseGridSize[3];
  for (i32 d = 0; d < 3; d++) {
    if (cylH > 0 && d == 1) {
      // theta: EXACTLY one pitch, no padding -- the periodic faces are the
      // domain faces and must sit on cell boundaries
      baseGridSize[d] = cylNTheta;
      domainSize[d]   = (real)cylArc;
      origin[d]       = domLo[d];
      continue;
    }
    real pad = fmax(margin*ext[d], 2*h);       // >= 2 cells so cut elements
    origin[d] = domLo[d] - pad;                // always have a neighbour
    i32 nc = blockSize*(i32)ceil((ext[d] + 2*pad)/h/blockSize);
    baseGridSize[d] = nc;
    domainSize[d]   = nc*h;
  }
  for (i32 d = 0; d < 3; d++) sdf.org[d] = origin[d];
  printf("grid   : %dx%dx%d cells (h = %.6g), %d background blocks\n",
         baseGridSize[0], baseGridSize[1], baseGridSize[2], (double)h,
         (baseGridSize[0]/blockSize)*(baseGridSize[1]/blockSize)*(baseGridSize[2]/blockSize));

  // -------------------------------------------------------------------------
  //  solver
  // -------------------------------------------------------------------------
  CutFemSolver *S = new CutFemSolver(domainSize, baseGridSize);
  for (i32 d = 0; d < 3; d++) S->domainOrigin[d] = origin[d];
  S->ls = sdf;

  S->prob.caseId = (caseName == "load")    ? CASE_LOAD
                 : (caseName == "mmscyl")  ? CASE_MMS_CYL   // periodic-compatible MMS
                 :                           CASE_MMS;
  S->prob.mu  = E/(2*(1+nu));
  S->prob.lam = E*nu/((1+nu)*(1-2*nu));
  S->prob.rho = rhoMat;
  S->prob.omega = (real)(rpm*2*PI/60.0);
  S->prob.bcMode   = bankMode ? 1 : 0;
  S->prob.hubTab   = sdf.hubTab;
  S->prob.nWall    = sdf.nWall;
  S->prob.wallZ0   = sdf.wallZ0;
  S->prob.wallDz   = sdf.wallDz;
  S->prob.platThick= sdf.platThick;
  S->prob.clampTol = (real)0.05*h;             // a sliver above the underside
  (void)clampR;
  for (i32 d = 0; d < 3; d++) { S->prob.kw[d] = kw[d]; S->prob.gvec[d] = grav[d]; S->prob.trac[d] = 0; }
  S->prob.clampX = domLo[0] + (real)0.15*ext[0];
  S->prob.tracX0 = domLo[0] + (real)0.85*ext[0];

  S->femOrder    = (femOrder < 1) ? 1 : femOrder;
  S->femMethod   = femMethod;
  S->outTag      = tag;
  S->wantVtu     = !novtu;
  S->spdCheck    = spdCheck;
  S->rangeTest   = rangeTest;
  S->periodic    = (cylMode && !noPeriodic && !noSector) ? 1 : 0;
  S->nThetaCells = cylNTheta;
  S->pitchAngle  = (real)(2*PI/std::max(1, nSectors));
  S->gammaD   = gammaD;
  S->gammaA   = gammaA;
  S->stabMode = stab;
  S->cutSub   = (sub < 1) ? 1 : (sub > FEM_MAXSUB ? FEM_MAXSUB : sub);
  S->cgTol    = tol;
  S->cgMaxIt  = maxit;
  S->volExact = bankMode ? 0 : volExact;       // the sector volume is not the loft's
  S->areaExact = bankMode ? 0 : areaExact;

  printf("material: E = %g, nu = %g, rho = %g  ->  mu = %g, lambda = %g\n",
         (double)E, (double)nu, (double)rhoMat, (double)S->prob.mu, (double)S->prob.lam);
  if (rpm > 0) printf("load   : %g rpm  ->  omega = %g rad/s (centrifugal)\n",
                      (double)rpm, (double)S->prob.omega);
  if (S->prob.caseId == CASE_LOAD)
    printf("bc     : clamped %s\n", bankMode
           ? "on the platform underside" : "on the low-x face");
  printf("cutfem : gamma_D = %g, gamma_a = %s, stab mode %d, cut sub %d\n",
         (double)gammaD,
         gammaA < 0 ? "(2mu+lam)*1e-4" : (gammaA == 0 ? "0 (DISABLED)" : "custom"),
         stab, S->cutSub);

  // ---- optional: raw SDF slices, for inspecting the CSG geometry ---------
  //
  // Sampling the composite level set directly is the only way to SEE what the
  // fillet / gap / sector algebra actually produced -- the surface output only
  // shows where phi crossed zero on the cut cells.
  if (doSlice) {
    mkdir("output", 0755);
    const i32 NA = 800, NB = 800;
    std::vector<float> buf((size_t)NA*NB), gbuf((size_t)NA*NB);
    std::string sp = "output/" + tag + "_slice.bin";
    FILE *f = fopen(sp.c_str(), "wb");
    double zA = sdf.platZ0, zB = sdf.platZ1, zc0 = 0.5*(zA + zB);
    double rh = (double)sdf.hubR((real)zc0), rc = (double)sdf.casR((real)zc0);
    double pt = (double)sdf.platThick;
    // half-pitch for the theta slice window: halfPitch is 0 in periodic mode
    // (the sector cut is off, the domain IS one pitch), so fall back to pi/N
    double hp = (sdf.halfPitch > 0) ? (double)sdf.halfPitch
              : (sdf.pitch > 0 ? 0.5*(double)sdf.pitch : 0.35);

    double zr1 = zA + 0.35*(zB - zA), zr2 = zA + 0.65*(zB - zA);
    double rh1 = (double)sdf.hubR((real)zr1), rh2 = (double)sdf.hubR((real)zr2);
    double rc1 = (double)sdf.casR((real)zr1);
    struct Plane { i32 kind; double at; double x0, x1, y0, y1; const char *name; };
    Plane pl[4] = {
      {0, 0.0, zA - 0.08*(zB-zA), zB + 0.08*(zB-zA), rh - pt - 0.1, rc + 0.1,
       "meridional (z,r) on the swept sector centreline"},
      {1, zr1, -hp, hp, rh1 - pt - 0.05, rh1 + 0.40,
       "ROOT FILLET (theta-thc, r) at 35% chord"},
      {1, zr2, -hp, hp, rh2 - pt - 0.05, rh2 + 0.40,
       "ROOT FILLET (theta-thc, r) at 65% chord"},
      {1, zr1, -hp, hp, rc1 - 0.30, rc1 + 0.04,
       "TIP GAP (theta-thc, r) at 35% chord"},
    };
    i32 hdr[4] = {4, NA, NB, 0};
    fwrite(hdr, sizeof(i32), 4, f);
    for (i32 q = 0; q < 4; q++) {
      // cell size expressed in THIS plane's own u and v axis units -- the theta
      // axes are plotted in radians while the grid spacing is a length, so the
      // plot needs the conversion or it draws only some of the lines
      double ucell = h, vcell = h;
      if (cylMode && pl[q].kind == 1) ucell = h/(double)sdf.rRef;
      double meta[8] = {(double)pl[q].kind, pl[q].x0, pl[q].x1, pl[q].y0, pl[q].y1,
                        (double)h, ucell, vcell};
      fwrite(meta, sizeof(double), 8, f);
      i32 nl = (i32)strlen(pl[q].name);
      fwrite(&nl, sizeof(i32), 1, f);
      fwrite(pl[q].name, 1, (size_t)nl, f);
      for (i32 j = 0; j < NB; j++)
      for (i32 i = 0; i < NA; i++) {
        double u = pl[q].x0 + (pl[q].x1-pl[q].x0)*i/(NA-1.0);
        double v = pl[q].y0 + (pl[q].y1-pl[q].y0)*j/(NB-1.0);
        // GRID coordinates of the sample (phi and the cell layout both live
        // there); how they map from the plane's (u,v) axes depends on the
        // coordinate system
        double qg[3];
        if (cylMode) {
          if (pl[q].kind == 0) { qg[0] = v; qg[1] = 0.0;                  qg[2] = u; }
          else                 { qg[0] = v; qg[1] = (double)sdf.rRef*u;   qg[2] = pl[q].at; }
          for (i32 d = 0; d < 3; d++) qg[d] -= origin[d];
        } else {
          double X, Y, Z;
          if (pl[q].kind == 0) {
            double tc = (double)sdf.thc((real)u);
            X = v*std::cos(tc); Y = v*std::sin(tc); Z = u;
          } else {
            double tt = u + (double)sdf.thc((real)pl[q].at);
            X = v*std::cos(tt); Y = v*std::sin(tt); Z = pl[q].at;
          }
          qg[0] = X - origin[0]; qg[1] = Y - origin[1]; qg[2] = Z - origin[2];
        }
        buf[(size_t)j*NA + i] = (float)sdf.phi((real)qg[0], (real)qg[1], (real)qg[2]);
        // Distance to the nearest background-cell face, as a DIMENSIONLESS
        // fraction of a cell, so the plot can draw the grid the solver actually
        // uses.  Only the two IN-PLANE directions count: the third is constant
        // over the plane and would otherwise clamp the whole field.
        double gd = 1e30;
        for (i32 d = 0; d < 3; d++) {
          if (pl[q].kind == 0 && d == 1 && cylMode) continue;   // q1 constant
          if (pl[q].kind == 1 && d == 2) continue;              // z constant
          double t2 = qg[d]/h;
          double fr = t2 - std::floor(t2);
          gd = std::min(gd, std::min(fr, 1.0 - fr));
        }
        gbuf[(size_t)j*NA + i] = (float)gd;
      }
      fwrite(buf.data(),  sizeof(float), (size_t)NA*NB, f);
      fwrite(gbuf.data(), sizeof(float), (size_t)NA*NB, f);
    }
    fclose(f);
    printf("wrote %s  (4 planes, %dx%d, phi + grid)\n", sp.c_str(), NA, NB);
  }

  // ---- optional: the level set itself, as a structured grid ---------------
  //
  // Samples phi on the computational box and writes it with PHYSICAL point
  // coordinates, so ParaView can contour phi = 0 directly.  This is the
  // geometry seen exactly as the solver sees it, with none of the cut-cell
  // machinery in between -- which is what you want when the question is
  // "is the level set right" rather than "is the extraction right".
  if (isoN > 0) {
    mkdir("output", 0755);
    double L[3] = {(double)domainSize[0], (double)domainSize[1], (double)domainSize[2]};
    double Lm = std::max(L[0], std::max(L[1], L[2]));
    i32 N0 = std::max(8, (i32)llround(isoN*L[0]/Lm));
    i32 N1 = std::max(8, (i32)llround(isoN*L[1]/Lm));
    i32 N2 = std::max(8, (i32)llround(isoN*L[2]/Lm));
    size_t nP = (size_t)(N0+1)*(N1+1)*(N2+1);
    float *xyz = nullptr, *phiv = nullptr;
    cudaMallocManaged(&xyz, 3*nP*sizeof(float));
    cudaMallocManaged(&phiv, nP*sizeof(float));
    femIsoSampleKernel<<<2048, 128>>>(sdf, (real)(L[0]/N0), (real)(L[1]/N1),
                                      (real)(L[2]/N2), N0, N1, N2, xyz, phiv);
    cudaDeviceSynchronize();

    std::string fn = "output/" + tag + "_ls.vts";
    FILE *f = fopen(fn.c_str(), "wb");
    fprintf(f, "<?xml version=\"1.0\"?>\n<VTKFile type=\"StructuredGrid\" version=\"1.0\""
               " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
               "  <StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n"
               "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
               "      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\""
               " format=\"appended\" offset=\"0\"/>\n      </Points>\n"
               "      <PointData Scalars=\"phi\">\n        <DataArray type=\"Float32\""
               " Name=\"phi\" format=\"appended\" offset=\"%llu\"/>\n      </PointData>\n"
               "    </Piece>\n  </StructuredGrid>\n  <AppendedData encoding=\"raw\">\n_",
            N0, N1, N2, N0, N1, N2,
            (unsigned long long)(sizeof(uint64_t) + 3*nP*sizeof(float)));
    uint64_t nb = 3*nP*sizeof(float);
    fwrite(&nb, sizeof(uint64_t), 1, f);
    fwrite(xyz, sizeof(float), 3*nP, f);
    nb = nP*sizeof(float);
    fwrite(&nb, sizeof(uint64_t), 1, f);
    fwrite(phiv, sizeof(float), nP, f);
    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);

    // periodicity of the level set across the pitch, and how close phi is to a
    // true distance function near the interface
    double perMax = 0;
    for (i32 k = 0; k <= N2; k++)
    for (i32 i = 0; i <= N0; i++) {
      double a2 = phiv[(size_t)(k*(N1+1) + 0 )*(N0+1) + i];
      double b2 = phiv[(size_t)(k*(N1+1) + N1)*(N0+1) + i];
      perMax = std::max(perMax, std::fabs(a2-b2));
    }
    double gmin = 1e30, gmax = -1e30, gsum = 0; i64 gn = 0;
    auto P = [&](i32 i, i32 j, i32 k) {
      return (double)phiv[(size_t)(k*(N1+1) + j)*(N0+1) + i];
    };
    double d0 = L[0]/N0, d1 = L[1]/N1, d2 = L[2]/N2;
    for (i32 k = 1; k < N2; k++)
    for (i32 j = 1; j < N1; j++)
    for (i32 i = 1; i < N0; i++) {
      double pv = P(i,j,k);
      if (std::fabs(pv) > 2*std::max(d0, std::max(d1,d2))) continue;
      double gx = (P(i+1,j,k)-P(i-1,j,k))/(2*d0);
      double gy = (P(i,j+1,k)-P(i,j-1,k))/(2*d1);
      double gz = (P(i,j,k+1)-P(i,j,k-1))/(2*d2);
      double g = std::sqrt(gx*gx + gy*gy + gz*gz);
      gmin = std::min(gmin, g); gmax = std::max(gmax, g); gsum += g; gn++;
    }
    printf("levelset: wrote %s  (%d x %d x %d points)\n", fn.c_str(), N0+1, N1+1, N2+1);
    printf("        : |grad phi| near the interface  min %.3f  mean %.3f  max %.3f"
           "   (1 = true distance)\n", gmin, gn ? gsum/gn : 0.0, gmax);
    printf("        : periodicity  max |phi(-pitch/2) - phi(+pitch/2)| = %.3e\n", perMax);
    cudaFree(xyz); cudaFree(phiv);
  }

  // ---- optional: level-set vs per-cell degree-p reconstruction slices --------
  //
  // On constant-radius airfoil planes (arc, z), dump the TRUE level set (oracle)
  // and the per-cell degree-p polynomial reconstruction that the Qp cut
  // quadrature fits (fitPoly3, same as runQp).  Overlaying their zero contours
  // shows exactly where the smooth polynomial fit matches the geometry and where
  // it oscillates -- the creases (sharp TE/tip, CSG kinks).
  if (reconP >= 1) {
    mkdir("output", 0755);
    i32 pr = reconP; i32 nr = pr + 1;
    real tr[PNC]; gllNodes(pr, tr);
    const i32 NA = 800, NB = 560;
    const int NSL = 3; double spanFrac[NSL] = {0.30, 0.55, 0.80};
    std::string fn = "output/" + tag + "_recon.bin";
    FILE *f = fopen(fn.c_str(), "wb");
    int hdr[2] = {NSL, pr}; fwrite(hdr, sizeof(int), 2, f);
    i32 ncj = baseGridSize[1], nck = baseGridSize[2];
    for (int sl = 0; sl < NSL; sl++) {
      double physR = domLo[0] + spanFrac[sl]*(domHi[0] - domLo[0]);
      double q0 = physR - origin[0];
      i32 ci = (i32)floor(q0/h);
      // per-cell polynomial cache (ci fixed on this slice)
      std::vector<PolyND> cellPoly((size_t)ncj*nck);
      std::vector<char>   have((size_t)ncj*nck, 0);
      auto poly = [&](i32 cj, i32 ck) -> PolyND& {
        size_t id = (size_t)cj*nck + ck;
        if (!have[id]) {
          real v[PNC*PNC*PNC];
          for (i32 kk = 0; kk < nr; kk++)
          for (i32 jj = 0; jj < nr; jj++)
          for (i32 ii = 0; ii < nr; ii++)
            v[ii + nr*(jj + nr*kk)] =
              sdf.phi((ci+tr[ii])*h, (cj+tr[jj])*h, (ck+tr[kk])*h);
          cellPoly[id] = fitPoly3(pr, v); have[id] = 1;
        }
        return cellPoly[id];
      };
      std::vector<float> bt((size_t)NA*NB), br((size_t)NA*NB);
      double q1hi = domainSize[1], q2hi = domainSize[2];
      for (i32 jb = 0; jb < NB; jb++)
      for (i32 ia = 0; ia < NA; ia++) {
        double g1 = q1hi*ia/(NA-1.0), g2 = q2hi*jb/(NB-1.0);
        double trv = (double)sdf.phi((real)q0, (real)g1, (real)g2);
        i32 cj = (i32)floor(g1/h), ck = (i32)floor(g2/h);
        double rc = trv;
        if (cj >= 0 && cj < ncj && ck >= 0 && ck < nck) {
          real xr[3] = {(real)(q0/h - ci), (real)(g1/h - cj), (real)(g2/h - ck)};
          rc = (double)poly(cj, ck).eval(xr);
        }
        bt[(size_t)jb*NA + ia] = (float)trv;
        br[(size_t)jb*NA + ia] = (float)rc;
      }
      int dims[2] = {NA, NB}; fwrite(dims, sizeof(int), 2, f);
      double meta[5] = {q1hi, q2hi, (double)h, physR, (double)pr};
      fwrite(meta, sizeof(double), 5, f);
      fwrite(bt.data(), sizeof(float), (size_t)NA*NB, f);
      fwrite(br.data(), sizeof(float), (size_t)NA*NB, f);
      printf("recon  : slice %d at r=%.4f (%.0f%% span): true + deg-%d reconstruction\n",
             sl, physR, 100*spanFrac[sl], pr);
    }
    fclose(f);
    printf("wrote %s  (%d constant-radius airfoil slices, %dx%d)\n", fn.c_str(), NSL, NA, NB);
  }

  // ---- optional: the 3-D degree-p reconstructed boundary (Saye surface) --------
  //
  // For every cut cell, fit a degree-p polynomial to the oracle level set at the
  // (p+1)^3 GLL nodes and run the Saye SURFACE rule -> points that lie on the
  // reconstructed boundary.  Each point is tagged with |phi| there (its distance
  // from the TRUE surface), so creases -- the sharp trailing edge, where a single
  // polynomial cannot follow the kink -- light up.  This is the 3-D analogue of
  // the --recon airfoil slices, on the real geometry.
  if (recon3dP >= 1) {
    mkdir("output", 0755);
    i32 pr = recon3dP, nr = pr + 1;
    real tr[PNC]; gllNodes(pr, tr);
    static SayeNode arena[1<<18], obuf[1<<16];
    std::string fn = "output/" + tag + "_recon3d_p" + std::to_string(recon3dP) + ".txt";
    FILE *f = fopen(fn.c_str(), "w");
    long np = 0, nCut = 0;
    for (i32 ck = 0; ck < baseGridSize[2]; ck++)
    for (i32 cj = 0; cj < baseGridSize[1]; cj++)
    for (i32 ci = 0; ci < baseGridSize[0]; ci++) {
      // cheap prune: skip cells whose centre is far from the surface
      real pc0 = sdf.phi((ci+(real)0.5)*h, (cj+(real)0.5)*h, (ck+(real)0.5)*h);
      if (fabs(pc0) > (real)1.5*h) continue;
      real v[PNC*PNC*PNC];
      bool anyNeg=false, anyPos=false;
      for (i32 k=0;k<nr;k++) for(i32 j=0;j<nr;j++) for(i32 i=0;i<nr;i++){
        real p = sdf.phi((ci+tr[i])*h, (cj+tr[j])*h, (ck+tr[k])*h);
        v[i+nr*(j+nr*k)] = p; if(p<0)anyNeg=true; else anyPos=true;
      }
      if(!anyNeg || !anyPos) continue;
      nCut++;
      PolyND poly = fitPoly3(pr, v);
      SayeArena ar; ar.buf=arena; ar.cap=1<<18; ar.top=0;
      SayeSet o; o.p=obuf; o.n=0; o.cap=1<<16; o.ovf=false;
      sayeSurface(poly, &o, &ar);
      for (i32 q=0;q<o.n;q++){
        real gx=(ci+o.p[q].x[0])*h, gy=(cj+o.p[q].x[1])*h, gz=(ck+o.p[q].x[2])*h;
        real X,Y,Z; sdf.toPhys(gx,gy,gz,X,Y,Z);
        real dev = fabs(sdf.phi(gx,gy,gz));      // SDF value at the reconstructed pt
        fprintf(f,"%.5f %.5f %.5f %.6f\n",(double)X,(double)Y,(double)Z,(double)dev);
        np++;
      }
    }
    fclose(f);
    printf("recon3d: %ld boundary points from %ld cut cells (deg %d) -> %s\n", np, nCut, pr, fn.c_str());
    delete S; cudaDeviceReset(); return 0;    // diagnostic dump only; skip the solve
  }

  auto t0 = std::chrono::steady_clock::now();
  S->run();
  auto t1 = std::chrono::steady_clock::now();
  printf("total  : %.1f ms\n",
         std::chrono::duration<double,std::milli>(t1-t0).count());

  if (!novtu && S->femOrder == 1 && S->femMethod == 0) {   // Qp/SBM paths write their own VTU
    mkdir("output", 0755);
    std::string v = "output/" + tag + "_fem.vtu";
    std::string sf = "output/" + tag + "_fem_surf.vtu";
    S->writeVtu(v.c_str());
    S->writeSurfaceVtu(sf.c_str());
    printf("wrote %s and %s\n", v.c_str(), sf.c_str());
  }

  delete S;
  cudaFree(dNodes); cudaFree(dOrder); cudaFree(dTris);
  cudaFree(dHub); cudaFree(dCas); cudaFree(dThc);
  cudaDeviceReset();
  return 0;
}
