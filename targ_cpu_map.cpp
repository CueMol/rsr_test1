#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>


#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void MiniTargCPU::calcMapEng()
{
  const realnum_t scale = m_pMap->m_dScale;
  const int natoms = m_pMol->m_nAtoms;
  const std::vector<float> &crds = m_pMol->m_crds;
  m_Emap=0.0f;
  int i;

  for (i=0; i<natoms; ++i) {
      //{ int i=4;
    const int ic = i*3;
    qlib::Vector4D pos(crds[ic+0],
		       crds[ic+1],
		       crds[ic+2]);

    const realnum_t wgt = - m_pMol->m_mass[i] * scale;

    //realnum_t Eden = wgt * gpMap->getDensity(pos);
    realnum_t Eden = wgt * m_pMap->getDensityCubic(pos);
    
    m_Emap += Eden;
  }

  m_energy += m_Emap;

#ifdef DEBUG_PRINT
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = m_Emap;
  printf("CPU Emap: %.16e [%x]\n", u.f, u.ui);
#endif
}

void MiniTargCPU::calcMapFce()
{
  const realnum_t scale = m_pMap->m_dScale;
  const int natoms = m_pMol->m_nAtoms;
  const std::vector<float> &crds = m_pMol->m_crds;
  Vector4D pos, dv;
  int i;

  const int ncrd = m_pMol->m_nCrds;
  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  for (i=0; i<natoms; ++i) {
      //{ int i=4;
    const int ic = i*3;
    pos.x() = crds[ic+0];
    pos.y() = crds[ic+1];
    pos.z() = crds[ic+2];

    realnum_t wgt = - m_pMol->m_mass[i] * scale;

    float den;
    m_pMap->getGrad(pos, den, dv);
      //gpMap->getGradDesc(pos, Eden, dv);
    
    den *= wgt;
    dv = dv.scale(wgt);

    /*
    m_grad[ic+0] += dv.x();
    m_grad[ic+1] += dv.y();
    m_grad[ic+2] += dv.z();
    */
    g[ic+0] += dv.x();
    g[ic+1] += dv.y();
    g[ic+2] += dv.z();
    //printf("atom %d (%f,%f,%f) Eden=%f grad=(%f,%f,%f)\n", i, pos.x(), pos.y(), pos.z(), Eden, dv.x(), dv.y(), dv.z());
  }

#ifdef DEBUG_PRINT
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = g[0];
  printf("CPU Map grad0: %.16e [%x]\n", u.f, u.ui);
#endif

  for (i=0; i<ncrd; i++)
    m_grad[i] += g[i];
}

