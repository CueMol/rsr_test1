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
#include "map_cuda.hpp"
#include "plan_cuda.hpp"

#define DEBUG_PRINT 1

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
  m_gradtmp.resize(pMol->m_nCrds + GRAD_PAD);

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

  // setup CUDA plane calc data structure
  printf("  bPlan = %d\n", m_bPlan);
  if (m_bPlan) {
    m_pPlanData = new CuPlanData();
    m_pPlanData->setup(pMol);
    m_pPlanData->setupCuda(m_pComData);
  }

  // setup CUDA mape calc data structure
  printf("  bMap = %d\n", m_bMap);
  if (m_bMap) {
    m_pMapData = new CuMap2Data();
    m_pMapData->setup(pMol, pMap);
    m_pMapData->setupCuda(m_pComData);
  }
}

void MiniTargCUDA::cleanup()
{
  if (m_pComData!=NULL)
    delete m_pComData;

  if (m_pBondData!=NULL)
    delete m_pBondData;

  if (m_pAnglData!=NULL)
    delete m_pAnglData;

  if (m_pPlanData!=NULL)
    delete m_pPlanData;

  if (m_pMapData!=NULL)
    delete m_pMapData;
}

const std::vector<float> &MiniTargCUDA::calc(float &eng)
{
  const int ncrd = m_grad.size();
  m_energy = 0.0f;

  m_pComData->xferCrds(m_pMol->m_crds);
  m_pComData->resetGrad();

  if (m_bBond) {
    calcBond();
    //calcBondEng();
    //calcBondFce();
#ifdef DEBUG_PRINT
  printf("After bond\n");
  m_pComData->xferGrad(m_gradtmp);
  for (int i=ncrd; i<ncrd+GRAD_PAD; ++i) {
    printf("energy %d: %f\n", i, m_gradtmp[i]);
  }
#endif
  }


  if (m_bAngl) {
    calcAngl();
    //calcAnglEng();
    //calcAnglFce();
#ifdef DEBUG_PRINT
  printf("After angl\n");
  m_pComData->xferGrad(m_gradtmp);
  for (int i=ncrd; i<ncrd+GRAD_PAD; ++i) {
    printf("energy %d: %f\n", i, m_gradtmp[i]);
  }
#endif
  }


  if (m_bMap) {
    calcMap();
#ifdef DEBUG_PRINT
  printf("After map\n");
  m_pComData->xferGrad(m_gradtmp);
  for (int i=ncrd; i<ncrd+GRAD_PAD; ++i) {
    printf("energy %d: %f\n", i, m_gradtmp[i]);
  }
#endif
  }

  if (m_bPlan) {
    calcPlan();
    //calcPlanEng();
    //calcPlanFce();
  }

  m_pComData->xferGrad(m_gradtmp);

  for (int i=0; i<ncrd; ++i) {
    m_grad[i] = m_gradtmp[i];
  }

#ifdef DEBUG_PRINT
  printf("final\n");
  for (int i=ncrd; i<ncrd+32; ++i) {
    printf("energy %d: %f\n", i, m_gradtmp[i]);
  }
#endif

  m_energy = 0.0f;
  for (int i=ncrd; i<ncrd+32; ++i) {
    m_energy += m_gradtmp[i];
  }


  eng = m_energy;
  return m_grad;
}

