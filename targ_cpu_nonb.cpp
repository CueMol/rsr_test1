#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include "minimize.hpp"
#include "mol.hpp"

//#define DEBUG_PRINT 1
using namespace std;
using qlib::LString;

void MiniTargCPU::calcNonbEng()
{
#ifdef DEBUG_PRINT
  printf("CPU Enonb start\n");
#endif

  int i,j;
  int natom = m_pMol->m_nAtoms;
  int ncrd = m_pMol->m_nCrds;
  m_Enonb = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Nonb> nonbs = m_pMol->m_nonbs;

  const float kf = m_pMol->m_nonb_wgt;
  for (i=0; i<natom; ++i) {
    const int ai = i*3;
    const int na2 = nonbs[i].atoms.size();
    for (j=0; j<na2; ++j) {
      const int aj = nonbs[i].atoms[j].aj*3;
      const float r0 = nonbs[i].atoms[j].r0;

      const realnum_t dx = realnum_t(crds[ai+0]) - realnum_t(crds[aj+0]);
      const realnum_t dy = realnum_t(crds[ai+1]) - realnum_t(crds[aj+1]);
      const realnum_t dz = realnum_t(crds[ai+2]) - realnum_t(crds[aj+2]);

      const realnum_t sqlen = dx*dx + dy*dy + dz*dz;
      const realnum_t len = sqrt(sqlen);
      const realnum_t ss = len-r0;
      if (ss>=0.0)
	continue;

      m_Enonb += kf * ss * ss;
      //xxx[nonbs[i].atom_i] += nonbs[i].kf * ss * ss;

#ifdef DEBUG_PRINT
      printf("  nonb %d r0=%f, r=%f, E= %f", i, r0, len, kf * ss * ss);
      printf(" xyz= %f,%f,%f\n", crds[ai+0], crds[ai+1], crds[ai+2]);
#endif
    }
  }

  m_energy += m_Enonb;

#ifdef DEBUG_PRINT
  printf("CPU Enonb: %f\n", m_Enonb);
#endif
}

void MiniTargCPU::calcNonbFce()
{
#ifdef DEBUG_PRINT
  printf("CPU gradnonb start\n");
#endif

  int i,j;
  int natom = m_pMol->m_nAtoms;
  int ncrd = m_pMol->m_nCrds;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Nonb> nonbs = m_pMol->m_nonbs;

  const float kf = m_pMol->m_nonb_wgt;

  for (i=0; i<natom; ++i) {
    const int ai = i*3;
    const int na2 = nonbs[i].atoms.size();
    for (j=0; j<na2; ++j) {
      const int aj = nonbs[i].atoms[j].aj*3;
      const float r0 = nonbs[i].atoms[j].r0;

      const realnum_t dx = realnum_t(crds[ai+0]) - realnum_t(crds[aj+0]);
      const realnum_t dy = realnum_t(crds[ai+1]) - realnum_t(crds[aj+1]);
      const realnum_t dz = realnum_t(crds[ai+2]) - realnum_t(crds[aj+2]);

      const realnum_t sqlen = dx*dx + dy*dy + dz*dz;
      const realnum_t len = sqrt(sqlen);
      //const realnum_t ss = qlib::min(len-r0, 0.0);
      const realnum_t ss = len-r0;
      if (ss>=0) continue;

      realnum_t con = 2.0f * kf * ss/len;
      g[ai+0] += con * dx;
      g[ai+1] += con * dy;
      g[ai+2] += con * dz;
      
      g[aj+0] -= con * dx;
      g[aj+1] -= con * dy;
      g[aj+2] -= con * dz;
      
    }
  }

#ifdef DEBUG_PRINT
  //  for (i=0; i<ncrd/3; i++)
  //    printf("grad %d: %f %f %f\n", i, g[i*3+0], g[i*3+1], g[i*3+2]);
#endif

  for (i=0; i<ncrd; i++)
    m_grad[i] += g[i];
}

