#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"

using namespace std;
using qlib::LString;

void Minimize::setup(MolData *pMol, DensityMap *pMap)
{
  printf("Minimize generic setup\n");
  //m_nMaxIter = 10;
  if (m_bUseCUDA)
    m_pMiniTarg = new MiniTargCUDA();
  else
    m_pMiniTarg = new MiniTargCPU();
  m_pMiniTarg->setup(pMol, pMap);

  //m_grad.resize(m_pMol->m_nCrds);
}
