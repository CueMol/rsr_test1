// -*-Mode: C++;-*-
//
// Common implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"

CuComData::~CuComData()
{
  cleanup();
}

/// Allocate device memory
void CuComData::setup(int natom)
{
  m_nAtom = natom;

  if (pd_crds!=NULL)
    cleanup();

  // setup exec layout
  calcThrBlk(natom, &m_nthr, &m_nblk, &m_nDevAtom);

  printf("Atom nThr = %d (%d x %d)\n", m_nDevAtom, m_nblk, m_nthr);

  cudaMalloc((void**)&pd_crds, m_nDevAtom*3*sizeof(float));

  cudaMalloc((void**)&pd_grad, (m_nAtom*3+GRAD_PAD)*sizeof(float));

  cudaMalloc((void**)&pd_eatm, (EATM_MAX)*sizeof(float));
}

void CuComData::cleanup()
{
  cudaFree(pd_crds);
  cudaFree(pd_grad);
  cudaFree(pd_eatm);
  pd_crds = NULL;
  pd_grad = NULL;
}

/// Transfer coodinate array
void CuComData::xferCrds(const std::vector<float> &crds)
{
  gpuErrChk( cudaMemcpy( pd_crds, &crds[0], crds.size()*sizeof(float), cudaMemcpyHostToDevice) );
  //printf("xferCrds %p OK\n", pd_crds);
}

void CuComData::resetGrad()
{
  // ??? [0b,0b,0b,0b] == 0.0f in IEEE float??
  cudaMemset( pd_grad, 0, (m_nAtom*3+GRAD_PAD)*sizeof(float) );
  //cudaMemset( pd_eatm, 0, (EATM_MAX)*sizeof(float) );
}

void CuComData::xferGrad(std::vector<float> &grad) const
{
  cudaMemcpy( &grad[0], pd_grad, grad.size()*sizeof(float), cudaMemcpyDeviceToHost);
}

float CuComData::getEnergy()
{
  m_eatm.resize(EATM_MAX);
  cudaMemcpy( &m_eatm[0], pd_eatm, EATM_MAX*sizeof(float), cudaMemcpyDeviceToHost);
  float rval = 0.0f;
  for (int i=0; i<EATM_MAX; ++i)
    rval += m_eatm[i];
  return rval;
}
