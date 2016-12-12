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
#include "angl_cuda.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;

void CuAnglData::setup(MolData *pMol)
{
  int i;
  const int natom = m_nAtoms = pMol->m_nAtoms;
  const int ncrd = pMol->m_nCrds;
  const int nangl = pMol->m_nAngls;
      
  printf("prepCuAnglData nangl=%d, natom=%d\n", nangl, natom);

  m_cuangls.resize(nangl);

  for (i=0; i<nangl; ++i) {
    int atom_i = pMol->m_angls[i].atom_i;
    int atom_j = pMol->m_angls[i].atom_j;
    int atom_k = pMol->m_angls[i].atom_k;

    m_cuangls[i].ai = atom_i*3;
    m_cuangls[i].aj = atom_j*3;
    m_cuangls[i].ak = atom_k*3;
    m_cuangls[i].kf = pMol->m_angls[i].kf;
    m_cuangls[i].r0 = pMol->m_angls[i].r0;

    //    cuangls[i].ai_ord = accum[atom_i];
    //    cuangls[i].aj_ord = accum[atom_j];
  }
}

void MiniTargCUDA::calcAngl()
{
  float Eangl;
  m_pAnglData->calc(&Eangl, m_grad);
  m_Eangl = Eangl;

#ifdef DEBUG_PRINT
  printf("CUDA Eangl: %f\n", m_Eangl);
#endif
}

