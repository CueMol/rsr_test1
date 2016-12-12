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
#include "map.hpp"

#include "com_cuda.hpp"
#include "map_cuda.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void CuMapData::setup(MolData *pMol, DensityMap *pMap)
{
  int i;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;

  m_ncol = pMap->m_ncol;
  m_nrow = pMap->m_nrow;
  m_nsec = pMap->m_nsect;

  m_na = pMap->m_na;
  m_nb = pMap->m_nb;
  m_nc = pMap->m_nc;

  m_stcol = pMap->m_stacol;
  m_strow = pMap->m_starow;
  m_stsec = pMap->m_stasect;

  m_fracMat.resize(9);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      m_fracMat[i + j*3] = pMap->m_fracMat.aij(i+1,j+1);

  m_map = &pMap->m_data[0];

  // setup exec layout (same as crd/atom)
  calcThrBlk(natom, &m_nthr, &m_nblk, &m_nDevAtom);

  const float scale = pMap->m_dScale;
  m_wgts.resize(m_nDevAtom);
  for (int i=0; i<m_nDevAtom; ++i) {
    if (i<natom)
      m_wgts[i] = - pMol->m_mass[i] * scale;
    else
      m_wgts[i] = 0.0f;
    //printf("wgts: %d %f\n", i, pRet->wgts[i]);
  }

}

void MiniTargCUDA::calcMap()
{
  m_pMapData->calc();
}

//////////

void CuMap2Data::setup(MolData *pMol, DensityMap *pMap)
{
  int i;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;

  m_ncol = pMap->m_ncol;
  m_nrow = pMap->m_nrow;
  m_nsec = pMap->m_nsect;

  m_na = pMap->m_na;
  m_nb = pMap->m_nb;
  m_nc = pMap->m_nc;

  m_stcol = pMap->m_stacol;
  m_strow = pMap->m_starow;
  m_stsec = pMap->m_stasect;

  m_fracMat.resize(9);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      m_fracMat[i + j*3] = pMap->m_fracMat.aij(i+1,j+1);

  m_map = &pMap->m_data[0];

  m_nTermPerThr = 2;
  m_nThrPerAtom = 64/m_nTermPerThr;

  // setup exec layout
  calcThrBlk(natom * m_nThrPerAtom, &m_nthr, &m_nblk, &m_nTotThr);
  printf("Map2 kernel layout nthr=%d, nblk=%d, total=%d\n", m_nthr, m_nblk, m_nTotThr);

  const float scale = pMap->m_dScale;
  m_wgts.resize(m_nTotThr/m_nThrPerAtom);

  for (int i=0; i<m_wgts.size(); ++i) {
    if (i<natom)
      m_wgts[i] = - pMol->m_mass[i] * scale;
    else
      m_wgts[i] = 0.0f;
    //printf("wgts: %d %f\n", i, pRet->wgts[i]);
  }

}

