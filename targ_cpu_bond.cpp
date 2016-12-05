#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include "minimize.hpp"
#include "mol.hpp"
//#include "map.hpp"

//#define DEBUG_PRINT 1
using namespace std;
using qlib::LString;

void MiniTargCPU::calcBondEng()
{
#ifdef DEBUG_PRINT
  printf("CPU Ebond start\n");
#endif

  int i;
  int nbond = m_pMol->m_nBonds;
  int ncrd = m_pMol->m_nCrds;
  m_Ebond = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Bond> bonds = m_pMol->m_bonds;

  std::vector<realnum_t> xxx(ncrd/3);

  for (i=0; i<nbond; ++i) {
    const int ai = bonds[i].atom_i*3;
    const int aj = bonds[i].atom_j*3;

    const realnum_t dx = realnum_t(crds[ai+0]) - realnum_t(crds[aj+0]);
    const realnum_t dy = realnum_t(crds[ai+1]) - realnum_t(crds[aj+1]);
    const realnum_t dz = realnum_t(crds[ai+2]) - realnum_t(crds[aj+2]);

    const realnum_t sqlen = dx*dx + dy*dy + dz*dz;
    const realnum_t len = sqrt(sqlen);
    const realnum_t ss = len - bonds[i].r0;

    m_Ebond += bonds[i].kf * ss * ss;
    //m_Ebond += ss;
    //if (xxx[bonds[i].atom_i]==0.0f)
    //xxx[bonds[i].atom_i] = sqlen;
    xxx[bonds[i].atom_i] += bonds[i].kf * ss * ss;

#ifdef DEBUG_PRINT
    printf("  bond %d ss= %f, E= %f", i, ss, bonds[i].kf * ss * ss);
    printf("     = %f,%f,%f\n", crds[ai+0], crds[ai+1], crds[ai+2]);
#endif
  }

  /*
  for (int i=0; i<xxx.size(); ++i) {
    m_Ebond += xxx[i];
    union {
      realnum_t f;
      unsigned int ui;
    } u;
    u.f = xxx[i];
    printf("       %.16e [%x]\n", u.f, u.ui);
  }
  */

  m_energy += m_Ebond;
#ifdef DEBUG_PRINT
  /*
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = m_Ebond;
  printf("CPU Ebond: %.16e [%x]\n", u.f, u.ui);
  */
  printf("CPU Ebond: %f\n", m_Ebond);
#endif
}

void MiniTargCPU::calcBondFce()
{
  int i;
  int nbond = m_pMol->m_nBonds;
  int ncrd = m_pMol->m_nCrds;
  
  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Bond> bonds = m_pMol->m_bonds;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  for (i=0; i<nbond; ++i) {
    const int ai = bonds[i].atom_i*3;
    const int aj = bonds[i].atom_j*3;

    const realnum_t dx = realnum_t(crds[ai+0]) - realnum_t(crds[aj+0]);
    const realnum_t dy = realnum_t(crds[ai+1]) - realnum_t(crds[aj+1]);
    const realnum_t dz = realnum_t(crds[ai+2]) - realnum_t(crds[aj+2]);

    realnum_t sqlen = dx*dx + dy*dy + dz*dz;
    realnum_t len = sqrt(sqlen);

    realnum_t con = 2.0f * bonds[i].kf * (1.0f - bonds[i].r0/len);
    /*
    m_grad[ai+0] += con * dx;
    m_grad[ai+1] += con * dy;
    m_grad[ai+2] += con * dz;

    m_grad[aj+0] -= con * dx;
    m_grad[aj+1] -= con * dy;
    m_grad[aj+2] -= con * dz;
    */

    g[ai+0] += con * dx;
    g[ai+1] += con * dy;
    g[ai+2] += con * dz;

    g[aj+0] -= con * dx;
    g[aj+1] -= con * dy;
    g[aj+2] -= con * dz;

  }
#ifdef DEBUG_PRINT
  /*  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = g[0];
  printf("CPU Bond grad0: %.16e [%x]\n", u.f, u.ui);
  */
#endif

  for (i=0; i<ncrd; i++)
    m_grad[i] += g[i];
}

