//
//
//

#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include "minimize.hpp"
#include "mol.hpp"

#include "com_cuda.hpp"
#include "plan_cuda.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void CuPlanData::setup(MolData *pMol)
{
  int i, j;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;
  const int nplan = pMol->m_nPlans;
      
  printf("prepCuPlanData nplan=%d, natom=%d\n", nplan, natom);

  m_cuplans.resize(nplan);

  int npatoms = 0;
  for (i=0; i<nplan; ++i) {
    const Plan &plane = pMol->m_plans[i];
    m_cuplans[i].natom = plane.atoms.size();
    m_cuplans[i].istart = npatoms;
    m_cuplans[i].wgt = plane.atoms[0].wgt;
    npatoms += plane.atoms.size();
#ifdef DEBUG_PRINT
    printf("plan %d natom %d ist %d wgt %f\n",
	   i, m_cuplans[i].natom, m_cuplans[i].istart, m_cuplans[i].wgt);
#endif
  }

  m_cdix.resize(npatoms);
  int k = 0;
  for (i=0; i<nplan; ++i) {
    const Plan &plane = pMol->m_plans[i];
    for (j=0; j<plane.atoms.size(); ++j) {
      m_cdix[k] = plane.atoms[j].iatom*3;
#ifdef DEBUG_PRINT
      printf("cdix %d %d\n", k, m_cdix[k]);
#endif
      ++k;
    }
  }
}

void MiniTargCUDA::calcPlan()
{
  m_pPlanData->calc();
}

