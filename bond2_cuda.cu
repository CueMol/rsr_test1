// -*-Mode: C++;-*-
//
// Bond implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"
#include "bond_cuda.hpp"

//#define DEBUG_PRINT 1

CuBondData::CuBondData()
{
  pd_bond = NULL;
}

CuBondData::~CuBondData()
{
  cleanupCuda();
}

void CuBondData::setupCuda2(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nbond = m_cubonds.size();

  calcThrBlk(nbond, &m_nthr, &m_nblk, &m_nDevBond);
  printf("Bond nThr = %d (%d x %d)\n", m_nDevBond, m_nblk, m_nthr);

  // parameter array (param)
  cudaMalloc((void**)&pd_bond, m_nDevBond*sizeof(CuBond));
  cudaMemcpy(pd_bond, &m_cubonds[0], nbond*sizeof(CuBond), cudaMemcpyHostToDevice);
  printf("CUDA bonds (%d*%d) = %p\n", nbond, sizeof(CuBond), pd_bond);
}

void CuBondData::cleanupCuda()
{
  cudaFree(pd_bond);
  pd_bond = NULL;
}


//////////////////////////////////////////

#include "utility.cu"

__global__
void BondGradKern2(const float* crds, const CuBond* param,
		   float *grad, int ncrds, int nbonds)
{
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;

  float flag;
  if (ithr>=nbonds)
    flag = 0.0f;
  else
    flag = 1.0f;

  const int ibon = (ithr<nbonds)?ithr:0;

  const CuBond *pbon = &param[ibon];
  
  const int ai = pbon->ai;
  const int aj = pbon->aj;
  
  const float dx = crds[ai+0] - crds[aj+0];
  const float dy = crds[ai+1] - crds[aj+1];
  const float dz = crds[ai+2] - crds[aj+2];

  const float sqlen = dx*dx + dy*dy + dz*dz;
  //float sqlen = __fmul_rn(dx,dx) + __fmul_rn(dy,dy) + __fmul_rn(dz,dz);
  const float len = sqrt(sqlen);

  const float con = 2.0f * pbon->kf * (1.0f - pbon->r0/len) * flag;
  const float ss = (len - pbon->r0) * flag;

  atomicAdd(&grad[ai+0], con*dx);
  atomicAdd(&grad[ai+1], con*dy);
  atomicAdd(&grad[ai+2], con*dz);

  atomicAdd(&grad[aj+0], -con*dx);
  atomicAdd(&grad[aj+1], -con*dy);
  atomicAdd(&grad[aj+2], -con*dz);

  float sum = blockReduceSum(pbon->kf * ss * ss);

  const int tid = threadIdx.x;
  /*
  extern __shared__ float sdata[];
  sdata[tid] = pbon->kf * ss * ss;

  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2)  {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  */

  if (tid == 0) {
    //atomicAdd(&grad[ncrds + 0], sdata[0]);
    //atomicAdd(&grad[ncrds + blockIdx.x], sum);
    grad[ncrds+blockIdx.x] += sum;
  }
}

void CuBondData::calc2()
{
  const std::vector<CuBond> &param = m_cubonds;

  const int natom = m_nAtoms;
  const int nbond = param.size();

  const int nshmem = m_nthr*sizeof(float);

  BondGradKern2<<<m_nblk, m_nthr, nshmem>>>
    (m_pComDat->pd_crds, pd_bond, m_pComDat->pd_grad, natom*3, nbond);

#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  // XX
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
#endif

  /*
  const int nblk = m_pComDat->m_nblk;
  std::vector<float> eatoms(nblk);
  cudaMemcpy( &eatoms[0], pd_eatm, nblk*sizeof(float), cudaMemcpyDeviceToHost);

  *val = 0.0f;
  for (int i=0; i<nblk; ++i)
    *val += eatoms[i];
    */
}
