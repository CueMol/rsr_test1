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

void CuBondData::setup2(MolData *pMol)
{
  int i;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;
  const int nbond = pMol->m_nBonds;
      
  printf("prepCuBondData nbond=%d, natom=%d\n", nbond, natom);

  m_cubonds.resize(nbond);

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    m_cubonds[i].ai = atom_i*3;
    m_cubonds[i].aj = atom_j*3;
    m_cubonds[i].kf = pMol->m_bonds[i].kf;
    m_cubonds[i].r0 = pMol->m_bonds[i].r0;
  }
}

