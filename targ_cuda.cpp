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
#include "grad_bond.hpp"
#include "grad_map.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void MiniTargCUDA::setup(MolData *pMol, DensityMap *pMap)
{
  printf("TargCPU setup\n");
  m_pMol = pMol;
  m_pMap = pMap;

  m_bBond = true;
  //m_bBond = false;

  m_bMap = true;
  //m_bMap = false;

  m_grad.resize(pMol->m_nCrds);
  m_gradtmp.resize(pMol->m_nCrds);

  if (m_bBond)
    m_pBondData = prepBondCuda(pMol);
  if (m_bMap)
    m_pMapData = prepMapCuda1(pMap, pMol);
}

const std::vector<float> &MiniTargCUDA::calc(float &eng)
{
  m_energy = 0.0f;
  const int ncrd = m_grad.size();
  for (int i=0; i<ncrd; ++i)
    m_grad[i] = 0.0f;

  if (m_bBond)
    calcBond();

  if (m_bMap)
    calcMap();

  eng = m_energy;
  return m_grad;
}

////////////////////////////////

void MiniTargCUDA::calcBond()
{
  const int ncrds = m_pMol->m_nCrds;
  float energy;
  energy = gradBondCuda(m_pMol, m_pBondData, m_gradtmp);
  m_energy += energy;
  for (int i=0; i<ncrds; ++i)
    m_grad[i] += m_gradtmp[i];

#ifdef DEBUG_PRINT
  union {
    float f;
    unsigned int ui;
  } u;
  u.f = energy;
  printf("CUDA Ebond: %.16e [%x]\n", u.f, u.ui);
  u.f = m_gradtmp[0];
  printf("CUDA Bond grad0: %.16e [%x]\n", u.f, u.ui);
#endif
}

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

