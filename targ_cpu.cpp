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

MiniTarg::MiniTarg()
{
  m_bBond = false;
  m_bAngl = false;
  m_bDihe = false;
  m_bChir = false;
  m_bPlan = false;
  m_bRama = false;
  m_bNonb = false;
  m_bMap = false;

  m_pMol = NULL;
  m_pMap = NULL;

  m_energy = 0.0;
  m_Edihe = m_Eangl = m_Ebond = m_Emap = 0.0;
  m_Echir = m_Eplan = m_Erama = m_Enonb = 0.0;
  
}

void MiniTargCPU::setup(MolData *pMol, DensityMap *pMap)
{
  printf("TargCPU setup\n");
  m_pMol = pMol;
  m_pMap = pMap;

  m_grad.resize(pMol->m_nCrds);

  m_ramaplot.setup();
  //m_ramaplot.dump();
}

realnum_t MiniTargCPU::calcEng()
{
  /*
  if (m_bDihe)
    calcDiheEng2();

  realnum_t edh2 = m_Edihe;
  */

  m_energy = 0.0f;

  if (m_bBond)
    calcBondEng();

  if (m_bAngl)
    calcAnglEng();

  if (m_bDihe) {
    calcDiheEng();
    //calcDiheEng2();
  }

  if (m_bChir)
    calcChirEng();

  if (m_bPlan)
    calcPlanEng();
  
  if (m_bRama)
    calcRamaEng();

  if (m_bMap)
    calcMapEng();

  /*
  realnum_t resid = fabs(m_Edihe-edh2)/m_Edihe * 100;
  if (resid>0.00001)
    printf("Edihe diff=%f\n", resid);
  */

  return m_energy;
}

const std::vector<float> &MiniTargCPU::calcFce()
{
  const int ncrd = m_grad.size();
  for (int i=0; i<ncrd; ++i)
    m_grad[i] = 0.0f;

  if (m_bBond)
    calcBondFce();

  if (m_bAngl)
    calcAnglFce();

  if (m_bDihe) {
    //calcDiheFce();
    calcDiheFce2();
  }

  if (m_bChir)
    calcChirFce();

  if (m_bPlan)
    calcPlanFce();

  if (m_bRama)
    calcRamaFce();

  if (m_bMap)
    calcMapFce();

  return m_grad;
}

//#define NUMERICAL_GRAD 1

const std::vector<float> &MiniTargCPU::calc(float &eng)
{
  calcEng();
  eng = m_energy;

#ifdef NUMERICAL_GRAD
  const int ncrd = m_pMol->m_nCrds;
  std::vector<realnum_t> ng(ncrd);
  const float eps = 0.001;
  for (int i=0; i<ncrd; ++i) {
    realnum_t tmp = m_pMol->m_crds[i];
    m_pMol->m_crds[i] -= eps;
    realnum_t em = calcEng();

    m_pMol->m_crds[i] += eps*2.0;
    realnum_t ep = calcEng();

    ng[i] = (ep-em)/(2.0*eps);
    m_pMol->m_crds[i] = tmp;
  }

  calcFce();

  for (int i=0; i<ncrd; ++i) {
    if (!qlib::isNear4(m_grad[i], 0.0f)) {
      realnum_t rat = ng[i]/m_grad[i];
      if (!qlib::isNear4(rat, 1.0)) {
	printf("%d ng = %f, ag = %f", i/3, ng[i], m_grad[i]);
	printf(" rat resid = %f", fabs(1.0-rat));
	printf("\n");
      }
    }
  }

  for (int i=0; i<ncrd; ++i) {
    m_grad[i] = ng[i];
  }
  return m_grad;
#else
  return calcFce();
#endif

}
