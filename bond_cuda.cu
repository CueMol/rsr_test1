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
  pd_bondres = NULL;
  pd_bind = NULL;
  pd_bvec = NULL;
  pd_eatm = NULL;
}

CuBondData::~CuBondData()
{
  cleanupCuda();
}

void CuBondData::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nbond = m_cubonds.size();

  calcThrBlk(nbond, &m_nthr, &m_nblk, &m_nDevBond);
  printf("Bond nThr = %d (%d x %d)\n", m_nDevBond, m_nblk, m_nthr);

  //////////
  // memory for 1st stage

  // parameter array (param)
  cudaMalloc((void**)&pd_bond, m_nDevBond*sizeof(CuBond));
  cudaMemcpy(pd_bond, &m_cubonds[0], nbond*sizeof(CuBond), cudaMemcpyHostToDevice);
  printf("CUDA bonds (%d*%d) = %p\n", nbond, sizeof(CuBond), pd_bond);
  
  // intermediate result array
  cudaMalloc((void**)&pd_bondres, m_nDevBond*4*sizeof(float));

  //////////
  // memory for 2nd stage (grad gather)

  // atom-->bond index (1)
  const int nbind = m_pComDat->m_nDevAtom;
  cudaMalloc((void**)&pd_bind, nbind*sizeof(int));
  cudaMemcpy( pd_bind, &m_bind[0], nbind*sizeof(int), cudaMemcpyHostToDevice);

  // atom-->bond index array (for gather phase)
  const int nbvec = m_bvec.size();
  cudaMalloc((void**)&pd_bvec, nbvec*sizeof(int));
  cudaMemcpy( pd_bvec, &m_bvec[0], nbvec*sizeof(int), cudaMemcpyHostToDevice);
  printf("CUDA ibvec (%d*%d) = %p\n", nbvec, sizeof(int), pd_bvec);

  // atom energy array
  cudaMalloc((void**)&pd_eatm, m_pComDat->m_nblk*sizeof(float));

  //////////
  // memory for 3rd stage (energy gather)
}

void CuBondData::cleanupCuda()
{
  cudaFree(pd_bond);
  cudaFree(pd_bondres);
  cudaFree(pd_bind);
  cudaFree(pd_bvec);
  cudaFree(pd_eatm);

  pd_bond = NULL;
  pd_bondres = NULL;
  pd_bind = NULL;
  pd_bvec = NULL;
  pd_eatm = NULL;
}


__global__
void BondGradKern11(const float* crds, const CuBond* param, float *res, int nbonds)
{
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;
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

  const float con = 2.0f * pbon->kf * (1.0f - pbon->r0/len);
  const float ss = (len - pbon->r0);

  res[ibon*4+0] = con*dx;
  res[ibon*4+1] = con*dy;
  res[ibon*4+2] = con*dz;
  res[ibon*4+3] = pbon->kf * ss * ss * 0.5;

}

__global__
void BondGradKern12(const float* intres,
		    const int *bindex, const int *bvec,
		    float *grad, float *eatom, int natom)
{
  // const int ithr = blockIdx.x*blockDim.x + threadIdx.x;
  const int iatom = blockIdx.x*blockDim.x + threadIdx.x;

  int ibv = bindex[iatom];
  const int nterm = (iatom<natom)?(bvec[ibv]):0;
  ++ibv;

  int i, ib;
  float gx = 0.0f , gy = 0.0f, gz = 0.0f, gw = 0.0f;
  float fdir;

  for (i=0; i<nterm; ++i, ++ibv) {
    ib = bvec[ibv];

    if (ib>0) {
      ib = ib-1;
      fdir = 1.0;
    }
    else if (ib<0) {
      ib = -ib-1;
      fdir = -1.0;
    }
    else {
      ib = 0;
      fdir = 0.0;
    }

    gx += intres[ib*4+0] * fdir;
    gy += intres[ib*4+1] * fdir;
    gz += intres[ib*4+2] * fdir;
    gw += intres[ib*4+3];
  }

  grad[iatom*3+0] = gx;
  grad[iatom*3+1] = gy;
  grad[iatom*3+2] = gz;

  extern __shared__ float sdata[];
  const int tid = threadIdx.x;
  sdata[tid] = gw;

  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2)  {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    eatom[blockIdx.x] = sdata[0];
  
}

void CuBondData::calc(float *val, std::vector<float> &grad)
{
  const std::vector<CuBond> &param = m_cubonds;

  const int natom = m_nAtoms;
  const int nbond = param.size();

  const int nshmem = m_nthr*sizeof(float);

  BondGradKern11<<<m_nblk, m_nthr>>>
    (m_pComDat->pd_crds, pd_bond, pd_bondres, nbond);

#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  cudaThreadSynchronize();
  std::vector<float> tmps(nbond*4);
  cudaMemcpy( &tmps[0], pd_bondres, nbond*4*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0; i<nbond; ++i) {
    printf("%d: %f %f %f %f\n", i, tmps[i*4+0], tmps[i*4+1], tmps[i*4+2], tmps[i*4+3]);
  }
#endif


  //////////

  BondGradKern12<<<m_pComDat->m_nblk, m_pComDat->m_nthr, nshmem>>>
    (pd_bondres, pd_bind, pd_bvec, m_pComDat->pd_grad, pd_eatm, natom);

  //cudaThreadSynchronize();

#ifdef DEBUG_PRINT
  // XX
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
#endif

  const int nblk = m_pComDat->m_nblk;
  std::vector<float> eatoms(nblk);
  cudaMemcpy( &eatoms[0], pd_eatm, nblk*sizeof(float), cudaMemcpyDeviceToHost);

  *val = 0.0f;
  for (int i=0; i<nblk; ++i)
    *val += eatoms[i];

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
    atomicAdd(&grad[ncrds + blockIdx.x], sum);
    //grad[ncrds+blockIdx.x] += sum;
  }
}

void CuBondData::calc2(float *val, std::vector<float> &grad)
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
