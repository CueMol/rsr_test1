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

void MiniTargCPU::calcAnglEng()
{
  int i;
  int nangl = m_pMol->m_nAngls;
  int ncrd = m_pMol->m_nCrds;
  m_Eangl = 0.0f;
  
  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Angl> angls = m_pMol->m_angls;

  for (i=0; i<nangl; ++i) {
    int ai = angls[i].atom_i;
    int aj = angls[i].atom_j;
    int ak = angls[i].atom_k;

    realnum_t rijx, rijy, rijz;
    realnum_t rkjx, rkjy, rkjz;
    realnum_t Rij, Rkj;

    rijx = crds[ai*3+0] - crds[aj*3+0];
    rijy = crds[ai*3+1] - crds[aj*3+1];
    rijz = crds[ai*3+2] - crds[aj*3+2];

    rkjx = crds[ak*3+0] - crds[aj*3+0];
    rkjy = crds[ak*3+1] - crds[aj*3+1];
    rkjz = crds[ak*3+2] - crds[aj*3+2];

    // distance
    Rij = sqrt(qlib::max<realnum_t>(F_EPS16, rijx*rijx + rijy*rijy + rijz*rijz));
    Rkj = sqrt(qlib::max<realnum_t>(F_EPS16, rkjx*rkjx + rkjy*rkjy + rkjz*rkjz));

    // normalization
    realnum_t eijx, eijy, eijz;
    realnum_t ekjx, ekjy, ekjz;
    eijx = rijx / Rij;
    eijy = rijy / Rij;
    eijz = rijz / Rij;

    ekjx = rkjx / Rkj;
    ekjy = rkjy / Rkj;
    ekjz = rkjz / Rkj;

    // angle
    realnum_t costh = eijx*ekjx + eijy*ekjy + eijz*ekjz;
    costh = qlib::min<realnum_t>(1.0f, qlib::max<realnum_t>(-1.0f, costh));
    realnum_t theta = (::acos(costh));
    realnum_t dtheta = (theta - angls[i].r0);
    realnum_t eng = angls[i].kf*dtheta*dtheta;

#ifdef DEBUG_PRINT
    printf("  Eangl %d : %f\n", i, eng);
#endif

    m_Eangl += eng;
  }
  m_energy += m_Eangl;

#ifdef DEBUG_PRINT
  printf("CPU Eangl: %f\n", m_Eangl);
#endif
}

void MiniTargCPU::calcAnglFce()
{
  int i;
  int nangl = m_pMol->m_nAngls;
  int ncrd = m_pMol->m_nCrds;
  
  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Angl> angls = m_pMol->m_angls;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  for (i=0; i<nangl; ++i) {
    int ai = angls[i].atom_i*3;
    int aj = angls[i].atom_j*3;
    int ak = angls[i].atom_k*3;

    realnum_t rijx, rijy, rijz;
    realnum_t rkjx, rkjy, rkjz;
    realnum_t Rij, Rkj;

    rijx = crds[ai+0] - crds[aj+0];
    rijy = crds[ai+1] - crds[aj+1];
    rijz = crds[ai+2] - crds[aj+2];

    rkjx = crds[ak+0] - crds[aj+0];
    rkjy = crds[ak+1] - crds[aj+1];
    rkjz = crds[ak+2] - crds[aj+2];

    // distance
    Rij = sqrt(qlib::max<realnum_t>(F_EPS16, rijx*rijx + rijy*rijy + rijz*rijz));
    Rkj = sqrt(qlib::max<realnum_t>(F_EPS16, rkjx*rkjx + rkjy*rkjy + rkjz*rkjz));

    // normalization
    realnum_t eijx, eijy, eijz;
    realnum_t ekjx, ekjy, ekjz;
    eijx = rijx / Rij;
    eijy = rijy / Rij;
    eijz = rijz / Rij;

    ekjx = rkjx / Rkj;
    ekjy = rkjy / Rkj;
    ekjz = rkjz / Rkj;

    // angle
    realnum_t costh = eijx*ekjx + eijy*ekjy + eijz*ekjz;
    costh = qlib::min<realnum_t>(1.0f, qlib::max<realnum_t>(-1.0f, costh));
    realnum_t theta = (::acos(costh));
    realnum_t dtheta = (theta - angls[i].r0);

    //realnum_t Eangle = angls[i].kf*dtheta*dtheta;

    // calc gradient
    realnum_t df = 2.0*(angls[i].kf)*dtheta;
    realnum_t Dij;
    realnum_t Dkj;

    realnum_t sinth = sqrt(qlib::max<realnum_t>(0.0f, 1.0f-costh*costh));
    Dij =  df/(qlib::max<realnum_t>(F_EPS16, sinth)*Rij);
    Dkj =  df/(qlib::max<realnum_t>(F_EPS16, sinth)*Rkj);

    realnum_t vec_dijx = Dij*(costh*eijx - ekjx);
    realnum_t vec_dijy = Dij*(costh*eijy - ekjy);
    realnum_t vec_dijz = Dij*(costh*eijz - ekjz);
    
    realnum_t vec_dkjx = Dkj*(costh*ekjx - eijx);
    realnum_t vec_dkjy = Dkj*(costh*ekjy - eijy);
    realnum_t vec_dkjz = Dkj*(costh*ekjz - eijz);

    g[ai+0] += vec_dijx;
    g[ai+1] += vec_dijy;
    g[ai+2] += vec_dijz;

    g[aj+0] -= vec_dijx;
    g[aj+1] -= vec_dijy;
    g[aj+2] -= vec_dijz;

    g[ak+0] += vec_dkjx;
    g[ak+1] += vec_dkjy;
    g[ak+2] += vec_dkjz;

    g[aj+0] -= vec_dkjx;
    g[aj+1] -= vec_dkjy;
    g[aj+2] -= vec_dkjz;
  }

#ifdef DEBUG_PRINT
  for (i=0; i<ncrd/3; i++)
    printf("grad %d: %f %f %f\n", i, g[i*3+0], g[i*3+1], g[i*3+2]);
#endif

  for (i=0; i<ncrd; i++)
    m_grad[i] += g[i];
}

