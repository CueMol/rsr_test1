#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>
#include <qlib/Vector4D.hpp>

#include "minimize.hpp"
#include "mol.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;
using qlib::Vector4D;


void MiniTargCPU::calcChirEng()
{
  int i;
  int nchir = m_pMol->m_nChirs;
  int ncrd = m_pMol->m_nCrds;
  m_Echir = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Chir> chirs = m_pMol->m_chirs;

  Vector4D rc, r1, r2, r3, v1, v2, v3;

  for (i=0; i<nchir; ++i) {
    const int ai = chirs[i].atom_i*3;
    const int aj = chirs[i].atom_j*3;
    const int ak = chirs[i].atom_k*3;
    const int al = chirs[i].atom_l*3;

    rc.x() = crds[ai+0];
    rc.y() = crds[ai+1];
    rc.z() = crds[ai+2];
    r1.x() = crds[aj+0];
    r1.y() = crds[aj+1];
    r1.z() = crds[aj+2];
    r2.x() = crds[ak+0];
    r2.y() = crds[ak+1];
    r2.z() = crds[ak+2];
    r3.x() = crds[al+0];
    r3.y() = crds[al+1];
    r3.z() = crds[al+2];

    v1 = r1 - rc;
    v2 = r2 - rc;
    v3 = r3 - rc;

    realnum_t vol = v1.dot(v2.cross(v3));

    realnum_t delta = chirs[i].r0 + chirs[i].fsgn * vol;
    realnum_t resid = chirs[i].kf * delta * delta;

#ifdef DEBUG_PRINT
    //printf("%d %d %d %d chir vol: %f resid: %f\n", ai/3, aj/3, ak/3, al/3, vol, resid);
#endif

    m_Echir += resid;
  }

  m_energy += m_Echir;

#ifdef DEBUG_PRINT
  printf("CPU Echir: %f\n", m_Echir);
#endif

}

void MiniTargCPU::calcChirFce()
{
  int i;
  int nchir = m_pMol->m_nChirs;
  int ncrd = m_pMol->m_nCrds;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Chir> chirs = m_pMol->m_chirs;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  Vector4D rc, r1, r2, r3, v1, v2, v3, f1, f2, f3, fc;

  for (i=0; i<nchir; ++i) {
    const int ai = chirs[i].atom_i*3;
    const int aj = chirs[i].atom_j*3;
    const int ak = chirs[i].atom_k*3;
    const int al = chirs[i].atom_l*3;

    rc.x() = crds[ai+0];
    rc.y() = crds[ai+1];
    rc.z() = crds[ai+2];
    r1.x() = crds[aj+0];
    r1.y() = crds[aj+1];
    r1.z() = crds[aj+2];
    r2.x() = crds[ak+0];
    r2.y() = crds[ak+1];
    r2.z() = crds[ak+2];
    r3.x() = crds[al+0];
    r3.y() = crds[al+1];
    r3.z() = crds[al+2];

    v1 = r1 - rc;
    v2 = r2 - rc;
    v3 = r3 - rc;

    realnum_t vol = v1.dot(v2.cross(v3));

    realnum_t delta = chirs[i].r0 + chirs[i].fsgn * vol;
    //realnum_t resid = chirs[i].kf * delta * delta;

    realnum_t dE = chirs[i].fsgn * 2.0 * chirs[i].kf * delta;

    f1 = v2.cross(v3).scale(dE);
    f2 = v3.cross(v1).scale(dE);
    f3 = v1.cross(v2).scale(dE);
    fc = -f1-f2-f3;

    g[ai+0] += fc.x();
    g[ai+1] += fc.y();
    g[ai+2] += fc.z();

    g[aj+0] += f1.x();
    g[aj+1] += f1.y();
    g[aj+2] += f1.z();

    g[ak+0] += f2.x();
    g[ak+1] += f2.y();
    g[ak+2] += f2.z();

    g[al+0] += f3.x();
    g[al+1] += f3.y();
    g[al+2] += f3.z();

  }

#ifdef DEBUG_PRINT
  //  for (i=0; i<ncrd; i++) {
  //    printf("grad %d %f\n", g[i]);
  //  }
#endif

  for (i=0; i<ncrd; i++) {
    m_grad[i] += g[i];
  }
}

