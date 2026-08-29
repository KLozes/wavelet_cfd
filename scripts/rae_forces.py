#!/usr/bin/env python3
"""CL/CD from the surface Cp dump by closed-contour integration.

Main.cu rotates the SECTION by -aoa, so the freestream is along +x in the solver
frame: force_y is lift and force_x is drag directly, with no wind-axis rotation.
Surface dump columns: x/c yn/c Cp xSurf ySurf side.
"""
import sys, numpy as np
d = np.loadtxt(sys.argv[1])
cp, xs, ys, side = d[:,2], d[:,3], d[:,4], d[:,5]
# order the closed contour: upper TE->LE then lower LE->TE
up, lo = side > 0, side < 0
xu, yu, cu = xs[up], ys[up], cp[up]; o = np.argsort(-xu); xu, yu, cu = xu[o], yu[o], cu[o]
xl, yl, cl_ = xs[lo], ys[lo], cp[lo]; o = np.argsort(xl); xl, yl, cl_ = xl[o], yl[o], cl_[o]
X = np.concatenate([xu, xl]); Y = np.concatenate([yu, yl]); C = np.concatenate([cu, cl_])
X = np.append(X, X[0]); Y = np.append(Y, Y[0]); C = np.append(C, C[0])
dx, dy = np.diff(X), np.diff(Y)
cm = 0.5*(C[:-1] + C[1:])
# outward normal of a CCW contour is (dy, -dx); force = -Cp n ds
chord = X.max() - X.min()
CD = -np.sum(cm*dy)/chord
CL =  np.sum(cm*dx)/chord
print(f"  CL = {CL:+.4f}   CD = {CD:+.5f}   ({len(d)} surface points, chord {chord:.4f})")
