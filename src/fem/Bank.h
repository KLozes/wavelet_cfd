#ifndef FEM_BANK_H
#define FEM_BANK_H

//
// Reader for an IDS aero "bank" file (e.g. assets/bank_v98d.txt).
//
// Port of the parsing half of scripts/parse_bank.py, so the C++ geometry
// pipeline can consume the design data directly instead of going through an
// intermediate STL.
//
// File layout (reverse engineered in parse_bank.py)
// ------------------------------------------------
// * A few "global" throughflow stations, then the blade rows, each delimited by
//   `BEGIN <VANE|ROTOR> <idx> ...` and closed by a final `END`.
// * A station is introduced by a header line whose first token is one of
//   FREE / INSI / VANE / ROTOR followed by z_hub r_hub z_tip r_tip ..., and
//   carries labelled 13-value blocks (RADIUS, Z, STREAMLINE, ...) -- one value
//   per streamline.
// * Each blade row also carries 13 MASTER sections, one per streamline.  A
//   MASTER block is a closed airfoil contour stored as three concatenated
//   equal-length arrays: Z (axial), R (radius), T (tangential, = r*theta).
//   The contour is laid out  surface | edge-arc | surface | edge-arc  with
//   n_surface and n_edge points each (80 and 27 in this file), so the blunt
//   leading and trailing edges get a quarter of the points.
//
// Numbers are written in a fixed-width format that lets consecutive values run
// together without a separator ("-8.60E-02-8.91E-02"), so tokens are scanned
// with an explicit float grammar rather than split on whitespace.
//

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace bank {

struct Section {
  int index = 0;
  std::vector<double> z, r, t;   // as stored
  std::vector<double> zg;        // z shifted into the global meridional frame
  int leIdx = 0, teIdx = 0;      // mid nodes of the two edge arcs
};

struct Station {
  std::string row, kind;
  std::vector<double> header, R, Z, SL;
};

struct Row {
  std::string label, type;
  int index = 0;
  double nblades = 0;
  // shaft speed from the row's own RPM record.  Signed in the file (the sign is
  // the direction of rotation); the centrifugal load goes as omega^2 so only the
  // magnitude matters here.  0 = the file did not state one.
  double rpm = 0;
  int nEdge = 27, nSurface = 80;
  std::vector<Section> sections;
  std::vector<double> leZ, leR;   // leading-edge locus, per section
  std::vector<double> teZ, teR;
  // blade metal angles per streamline (hub..tip), radians from the meridional:
  // INLET ANGL (leading edge) and EXIT ANGLE (trailing edge).  Used to set the
  // camber-tangent direction of the passage centreline at the root.
  std::vector<double> inletAngle, exitAngle;
};

struct Bank {
  std::vector<Station> stations;
  std::vector<Row> rows;
  // flow-path walls, sorted by z
  std::vector<double> hubZ, hubR, casZ, casR;

  const Row *findRow(const std::string &label) const {
    for (const Row &r : rows) if (r.label == label) return &r;
    return nullptr;
  }
  double hubAt(double z) const { return interpWall(hubZ, hubR, z); }
  double casAt(double z) const { return interpWall(casZ, casR, z); }

  // Rescale every LENGTH in the bank by s (angles/theta are untouched).  The IDS
  // format states no unit; for assets/bank_v98d.txt two independent checks say
  // INCHES -- ROTOR 1's tip radius 6.77369 gives U_tip = 450 m/s at its own
  // 24970 rpm (a transonic HPC rotor; cm would give 177 m/s), and the AXIAL VEL
  // block only makes sense as ft/s (119-143 m/s; as m/s it would be supersonic
  // axial).  Pass 0.0254 to work in metres so SI material constants mean what
  // they say -- centrifugal stress goes as rho*omega^2*L^2, so the length unit
  // is squared into the answer and getting it wrong is not a small error.
  void scaleLengths(double s) {
    if (s == 1.0) return;
    for (Station &st : stations) {
      for (double &v : st.R) v *= s;
      for (double &v : st.Z) v *= s;
      for (double &v : st.SL) v *= s;
    }
    for (Row &r : rows) {
      for (Section &sc : r.sections) {
        for (double &v : sc.z) v *= s;
        for (double &v : sc.r) v *= s;    // sc.t is an ANGLE: leave it
        for (double &v : sc.zg) v *= s;
      }
      for (double &v : r.leZ) v *= s;  for (double &v : r.leR) v *= s;
      for (double &v : r.teZ) v *= s;  for (double &v : r.teR) v *= s;
    }
    for (double &v : hubZ) v *= s;  for (double &v : hubR) v *= s;
    for (double &v : casZ) v *= s;  for (double &v : casR) v *= s;
  }

  static double interpWall(const std::vector<double> &X,
                           const std::vector<double> &Y, double x) {
    if (X.empty()) return 0;
    if (x <= X.front()) return Y.front();
    if (x >= X.back())  return Y.back();
    size_t i = (size_t)(std::upper_bound(X.begin(), X.end(), x) - X.begin());
    double w = (x - X[i-1])/(X[i] - X[i-1]);
    return Y[i-1] + w*(Y[i] - Y[i-1]);
  }
};

// ---------------------------------------------------------------------------
//  scanning
// ---------------------------------------------------------------------------

inline bool isLabelLine(const std::string &s) {
  for (char c : s) {
    if (std::isspace((unsigned char)c)) continue;
    return std::isalpha((unsigned char)c) != 0;
  }
  return false;
}

// Pull every float out of a line.  Matches  [-+]?\d*\.\d+([eE][-+]?\d+)? --
// the same grammar parse_bank.py uses, which is what lets run-together
// fixed-width columns be separated correctly.
inline void scanFloats(const std::string &s, std::vector<double> &out) {
  size_t i = 0, n = s.size();
  while (i < n) {
    size_t j = i;
    if (s[j] == '+' || s[j] == '-') j++;
    size_t d0 = j;
    while (j < n && std::isdigit((unsigned char)s[j])) j++;
    if (j >= n || s[j] != '.') { i = (j > i) ? j : i + 1; continue; }
    j++;
    size_t f0 = j;
    while (j < n && std::isdigit((unsigned char)s[j])) j++;
    if (j == f0) { i = j; continue; }              // needs digits after the dot
    (void)d0;
    if (j < n && (s[j] == 'e' || s[j] == 'E')) {   // optional exponent
      size_t k = j + 1;
      if (k < n && (s[k] == '+' || s[k] == '-')) k++;
      size_t e0 = k;
      while (k < n && std::isdigit((unsigned char)s[k])) k++;
      if (k > e0) j = k;
    }
    out.push_back(std::atof(s.substr(i, j - i).c_str()));
    i = j;
  }
}

// consecutive pure-data lines starting at `i`; returns the next label index
inline size_t readBlock(const std::vector<std::string> &L, size_t i,
                        std::vector<double> &out) {
  size_t j = i;
  while (j < L.size() && !isLabelLine(L[j])) {
    scanFloats(L[j], out);
    j++;
  }
  return j;
}

inline std::vector<std::string> tokens(const std::string &s) {
  std::vector<std::string> t;
  std::istringstream is(s);
  std::string w;
  while (is >> w) t.push_back(w);
  return t;
}

// ---------------------------------------------------------------------------
//  register: move every MASTER airfoil into the global meridional frame
// ---------------------------------------------------------------------------
//
// A MASTER contour holds 2*n_surface + 2*n_edge points; the mid node of each
// edge arc is the true leading / trailing edge, one upstream and one
// downstream.  Aligning the LE arc to the row's leading-edge QO station drops
// the TE arc onto the trailing-edge QO (the chord matches the QO spacing).
//
inline void registerRows(Bank &B) {
  for (Row &row : B.rows) {
    // the row's stations that carry per-streamline R/Z, ordered by mean z
    std::vector<const Station*> rs;
    for (const Station &s : B.stations)
      if (s.row == row.label && !s.R.empty() && !s.Z.empty()) rs.push_back(&s);
    std::sort(rs.begin(), rs.end(), [](const Station *a, const Station *b) {
      double ma = 0, mb = 0;
      for (double v : a->Z) ma += v; ma /= a->Z.size();
      for (double v : b->Z) mb += v; mb /= b->Z.size();
      return ma < mb;
    });
    if (rs.empty() || row.sections.empty()) continue;
    const Station *le = rs[0];

    std::sort(row.sections.begin(), row.sections.end(),
              [](const Section &a, const Section &b) { return a.index < b.index; });

    int a = row.nSurface + row.nEdge/2;
    int b = 2*row.nSurface + row.nEdge + row.nEdge/2;
    int leI = a, teI = b;
    if (row.sections[0].z[(size_t)a] > row.sections[0].z[(size_t)b]) { leI = b; teI = a; }

    std::vector<double> dz;
    for (const Section &s : row.sections) {
      int k = s.index - 1;
      if (k >= 0 && k < (int)le->Z.size()) dz.push_back(le->Z[(size_t)k] - s.z[(size_t)leI]);
    }
    std::sort(dz.begin(), dz.end());
    double shift = dz.empty() ? 0.0 : dz[dz.size()/2];      // median

    row.leZ.clear(); row.leR.clear(); row.teZ.clear(); row.teR.clear();
    for (Section &s : row.sections) {
      s.zg.resize(s.z.size());
      for (size_t i = 0; i < s.z.size(); i++) s.zg[i] = s.z[i] + shift;
      s.leIdx = leI; s.teIdx = teI;
      row.leZ.push_back(s.zg[(size_t)leI]); row.leR.push_back(s.r[(size_t)leI]);
      row.teZ.push_back(s.zg[(size_t)teI]); row.teR.push_back(s.r[(size_t)teI]);
    }
  }

  // flow-path walls from the station endpoints
  std::vector<std::pair<double,double>> hub, cas;
  for (const Station &s : B.stations) {
    if (s.R.empty() || s.Z.empty()) continue;
    hub.push_back({s.Z.front(), s.R.front()});
    cas.push_back({s.Z.back(),  s.R.back()});
  }
  std::sort(hub.begin(), hub.end());
  std::sort(cas.begin(), cas.end());
  for (auto &p : hub) { B.hubZ.push_back(p.first); B.hubR.push_back(p.second); }
  for (auto &p : cas) { B.casZ.push_back(p.first); B.casR.push_back(p.second); }
}

// ---------------------------------------------------------------------------
//  parse
// ---------------------------------------------------------------------------
inline bool read(const std::string &path, Bank &B) {
  std::ifstream fh(path);
  if (!fh) return false;
  std::vector<std::string> L;
  std::string line;
  while (std::getline(fh, line)) L.push_back(line);

  Row *cur = nullptr;
  Station *st = nullptr;
  int nEdge = 27, nSurface = 80;

  size_t i = 0;
  while (i < L.size()) {
    if (!isLabelLine(L[i])) { i++; continue; }
    std::vector<std::string> tok = tokens(L[i]);
    if (tok.empty()) { i++; continue; }
    const std::string &head = tok[0];

    if (head == "BEGIN" && tok.size() > 2) {
      B.rows.push_back(Row());
      cur = &B.rows.back();
      cur->label = tok[1] + " " + tok[2];
      cur->type  = tok[1];
      cur->index = std::atoi(tok[2].c_str());
      cur->nEdge = nEdge; cur->nSurface = nSurface;
      st = nullptr;
      i++;
      continue;
    }
    if (head == "END") break;

    if (head == "MASTER") {
      std::vector<double> arr;
      size_t j = readBlock(L, i + 1, arr);
      size_t n = arr.size()/3;
      if (cur && n) {
        Section s;
        s.index = (tok.size() > 1) ? std::atoi(tok[1].c_str()) : (int)cur->sections.size() + 1;
        s.z.assign(arr.begin(), arr.begin() + n);
        s.r.assign(arr.begin() + n, arr.begin() + 2*n);
        s.t.assign(arr.begin() + 2*n, arr.begin() + 3*n);
        cur->sections.push_back(s);
      }
      i = j;
      continue;
    }
    if (head == "NO." && tok.size() > 2 && tok[1] == "BLADES") {
      if (cur && cur->nblades == 0) cur->nblades = std::atof(tok[2].c_str());
      i++;
      continue;
    }
    if (head == "RPM" && tok.size() > 1) {
      if (cur && cur->rpm == 0) cur->rpm = std::atof(tok[1].c_str());
      i++;
      continue;
    }
    if (head == "NO." && tok.size() > 3 && tok[1] == "COORDS") {
      nEdge = std::atoi(tok[2].c_str());
      nSurface = std::atoi(tok[3].c_str());
      if (cur) { cur->nEdge = nEdge; cur->nSurface = nSurface; }
      i++;
      continue;
    }

    // blade metal-angle blocks (per-streamline), attached to the current row.
    // Labels are two tokens ("INLET ANGL", "EXIT ANGLE"); take the first
    // occurrence in each row (the blade's own metal angles).
    if (head == "INLET" && tok.size() > 1 && tok[1] == "ANGL") {
      std::vector<double> arr; size_t j = readBlock(L, i + 1, arr);
      if (cur && cur->inletAngle.empty()) cur->inletAngle = arr;
      i = j;
      continue;
    }
    if (head == "EXIT" && tok.size() > 1 && tok[1] == "ANGLE") {
      std::vector<double> arr; size_t j = readBlock(L, i + 1, arr);
      if (cur && cur->exitAngle.empty()) cur->exitAngle = arr;
      i = j;
      continue;
    }

    // a station header carries its own coordinates on the same line
    if (head == "FREE" || head == "INSI" || head == "VANE" || head == "ROTOR") {
      std::vector<double> hv;
      scanFloats(L[i], hv);
      if (hv.size() >= 4) {
        B.stations.push_back(Station());
        st = &B.stations.back();
        st->row = cur ? cur->label : "GLOBAL";
        st->kind = head;
        st->header = hv;
        i++;
        continue;
      }
    }

    // labelled data block belonging to the current station
    std::vector<double> arr;
    size_t j = readBlock(L, i + 1, arr);
    if (st && !arr.empty()) {
      if      (head == "RADIUS")     st->R = arr;
      else if (head == "Z")          st->Z = arr;
      else if (head == "STREAMLINE") st->SL = arr;
    }
    i = j;
  }

  registerRows(B);
  return !B.rows.empty();
}

}  // namespace bank

#endif
