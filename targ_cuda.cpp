#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"

#include "com_cuda.hpp"
#include "bond_cuda.hpp"
#include "angl_cuda.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

MiniTargCUDA::~MiniTargCUDA()
{
  cleanup();
}

void MiniTargCUDA::setup(MolData *pMol, DensityMap *pMap)
{
  super_t::setup(pMol, pMap);

  printf("TargCUDA setup\n");

  // m_grad.resize(pMol->m_nCrds);
  m_gradtmp.resize(pMol->m_nCrds + 16);

  // setup CUDA common data structure
  m_pComData = new CuComData();
  m_pComData->setup(pMol->m_nAtoms);

  // setup CUDA bond calc data structure
  printf("  bBond = %d\n", m_bBond);
  if (m_bBond) {
    m_pBondData = new CuBondData();
    m_pBondData->setup(pMol);
    m_pBondData->setupCuda(m_pComData);
  }

  // setup CUDA angle calc data structure
  printf("  bAngl = %d\n", m_bAngl);
  if (m_bAngl) {
    m_pAnglData = new CuAnglData();
    m_pAnglData->setup(pMol);
    m_pAnglData->setupCuda(m_pComData);
  }

  //if (m_bMap)
  //m_pMapData = prepMapCuda1(pMap, pMol);
}

void MiniTargCUDA::cleanup()
{
  if (m_pComData!=NULL)
    delete m_pComData;

  if (m_pBondData!=NULL)
    delete m_pBondData;

  if (m_pAnglData!=NULL)
    delete m_pAnglData;
}

const std::vector<float> &MiniTargCUDA::calc(float &eng)
{
  m_energy = 0.0f;

  m_pComData->xferCrds(m_pMol->m_crds);
  m_pComData->resetGrad();

  if (m_bBond) {
    calcBond();
    //calcBondEng();
    //calcBondFce();
  }

  if (m_bAngl) {
    calcAngl();
    //calcAnglEng();
    //calcAnglFce();
  }

  //if (m_bMap)
  //calcMap();

  m_pComData->xferGrad(m_gradtmp);

  const int ncrd = m_grad.size();
  for (int i=0; i<ncrd; ++i)
    m_grad[i] = m_gradtmp[i];

  /*
  for (int i=ncrd; i<ncrd+16; ++i) {
    printf("energy %d: %f\n", i, m_gradtmp[i]);
    m_energy += m_gradtmp[i];
  }
  */
  m_energy = m_gradtmp[ncrd];

  eng = m_energy;
  return m_grad;
}


#if 0
void MiniTargCUDA::calcMap()
{
  const int ncrds = m_pMol->m_nCrds;
  float energy;
  energy = gradMapCuda1(m_pMol, m_pMap, m_pMapData, m_gradtmp);
  m_energy += energy;
  for (int i=0; i<ncrds; ++i)
    m_grad[i] += m_gradtmp[i];

#ifdef DEBUG_PRINT
  union {
    float f;
    unsigned int ui;
  } u;
  u.f = energy;
  printf("CUDA Emap: %.16e [%x]\n", u.f, u.ui);
  u.f = m_gradtmp[0];
  printf("CUDA Map grad0: %.16e [%x]\n", u.f, u.ui);
#endif
}
#endif
