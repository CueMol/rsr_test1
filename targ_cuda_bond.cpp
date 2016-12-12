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
#include "bond_cuda.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void CuBondData::setup(MolData *pMol)
{
  int i;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;
  const int nbond = pMol->m_nBonds;
      
  printf("prepCuBondData nbond=%d, natom=%d\n", nbond, natom);

  m_bvec.resize(nbond*2+natom);

  std::vector<int> accum(natom);
  for (i=0; i<natom; ++i)
    accum[i] = 1;

  m_cubonds.resize(nbond);

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    m_cubonds[i].ai = atom_i*3;
    m_cubonds[i].aj = atom_j*3;
    m_cubonds[i].kf = pMol->m_bonds[i].kf;
    m_cubonds[i].r0 = pMol->m_bonds[i].r0;

    //    cubonds[i].ai_ord = accum[atom_i];
    //    cubonds[i].aj_ord = accum[atom_j];

    accum[atom_i] ++;
    accum[atom_j] ++;
  }

  m_bind.resize(natom);
  int idx = 0;
  for (i=0; i<natom; ++i) {
    m_bind[i] = idx;
    idx += accum[i];
#ifdef DEBUG_PRINT
    //printf("bind(%d) = %d, %d\n", i, m_bind[i], accum[i]);
#endif
  }

  printf("bvec size %d x 2 + %d = %d, %d\n", nbond, natom, nbond*2+natom, idx);

  // build bvec array (nterm)
  for (i=0; i<natom; ++i) {
    idx = m_bind[i];
    m_bvec[idx] = accum[i]-1;
  }

  // reset accum array
  for (i=0; i<natom; ++i)
    accum[i] = 1;

  // build bvec array (bond index)
  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    int bi = m_bind[atom_i];
    int inr_i = accum[atom_i];
    accum[atom_i] ++;
    m_bvec[bi + inr_i] = (i+1);

    int bj = m_bind[atom_j];
    int inr_j = accum[atom_j];
    accum[atom_j] ++;
    m_bvec[bj + inr_j] = -(i+1);
  }
  
#ifdef DEBUG_PRINT
  printf("bvec size=%d\n", m_bvec.size());
  for (i=0; i<natom; ++i) {
    printf("bvec %d: ", i);
    int idx = m_bind[i];
    for (int j=0; j<=m_bvec[idx]; ++j)
      printf("%d ", m_bvec[idx + j]);
    printf("\n");
  }
#endif

}

void MiniTargCUDA::calcBond()
{
  float Ebond;
  //m_pBondData->calc(&Ebond, m_grad);
  //m_pComData->resetGrad();
  m_pBondData->calc2(&Ebond, m_grad);
  m_Ebond = Ebond;

#ifdef DEBUG_PRINT
  printf("CUDA Ebond: %f\n", m_Ebond);
#endif
}

