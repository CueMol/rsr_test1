#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include "minimize.hpp"
#include "mol.hpp"
#include "com_cuda.hpp"
#include "nonb_cuda.hpp"

#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

CuNonbData::CuNonbData()
{
}

CuNonbData::~CuNonbData()
{
  cleanupCuda();
}

void CuNonbData::setup(MolData *pMol)
{
  int i, j;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;
  const std::vector<Nonb>& nonbs = pMol->m_nonbs;
      
  printf("prepCuNonbData natom=%d\n", natom);

  std::vector< std::deque<CuNonb> > vatoms(natom);
  for (i=0; i<natom ;++i) {
    const int na2 = nonbs[i].atoms.size();
    for (j=0; j<na2; ++j) {
      const int aj = nonbs[i].atoms[j].aj;
      const float r0 = nonbs[i].atoms[j].r0;
      const float wgt = nonbs[i].atoms[j].wgt;
      vatoms[i].push_back( CuNonb(aj*3, r0, wgt) );
      vatoms[aj].push_back( CuNonb(i*3, r0, wgt) );
    }
  }

  m_nonb_max = 0;
  for (i=0; i<natom ;++i)
    m_nonb_max = qlib::max<int>(vatoms[i].size(), m_nonb_max);

  //if (m_nonb_max%32!=0)
  //m_nonb_max = (m_nonb_max/32+1)*32;

  printf("CUDA nonb calc/atom=%d\n", m_nonb_max);

  //m_nloop = m_nonb_max/32;
  if (m_nonb_max%32==0)
    m_nloop = m_nonb_max/32;
  else
    m_nloop = m_nonb_max/32+1;

  // m_nloop = m_nonb_max;

  printf("CUDA nonb calc/thr=%d\n", m_nloop);

  calcThrBlk(natom*32, &m_nthr, &m_nblk, &m_nTotalThr);
  // calcThrBlk(natom, &m_nthr, &m_nblk, &m_nTotalThr);
  printf("Nonb nThr = %d (%d x %d)\n", m_nTotalThr, m_nblk, m_nthr);

  m_nMatSize = natom * m_nonb_max +1;

  m_indmat.resize(m_nMatSize);
  m_prmmat.resize(m_nMatSize);

  for (i=0; i<m_nMatSize; ++i) {
    m_indmat[i] = 0;
    m_prmmat[i] = 0;
  }

  for (i=0; i<natom ;++i) {
    const int na2 = vatoms[i].size();
    for (j=0; j<na2; ++j) {
      CuNonb &elem = vatoms[i][j];
      m_indmat[i*m_nonb_max + j +1] = elem.idx;
      m_prmmat[i*m_nonb_max + j +1] = 1;
    }
  }

  /*
  m_mat.resize(m_nMatSize);
  printf("CuNonb array size = %fMb (%d x %d x %d)\n", m_nMatSize*sizeof(CuNonb)/1024.0/1024.0, natom, m_nonb_max, sizeof(CuNonb));

  for (i=0; i<m_nMatSize; ++i)
    m_mat[i] = CuNonb(0, 0.0f, 0.0f);

  for (i=0; i<natom ;++i) {
    const int na2 = vatoms[i].size();
    for (j=0; j<na2; ++j) {
      CuNonb &elem = vatoms[i][j];
      m_mat[i*m_nonb_max + j +1] = elem;
    }
  }
  */
}



