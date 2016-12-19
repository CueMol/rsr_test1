// -*-Mode: C++;-*-
//
// Plan implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"
#include "plan_cuda.hpp"

//#define DEBUG_PRINT 1

#include "utility.cu"

__global__
void PlanGradKern1(const float* crds, const CuPlan* param, const int *cdixs,
		   float *grad, int ncrds, int nplans)
{
  int i;
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;

  const int ipln = (ithr<nplans)?ithr:0;

  const CuPlan *ppln = &param[ipln];
  
  const int natom = ppln->natom;
  const int istart = ppln->istart;

  int idx0 = cdixs[istart];
  float4 kv = make_float4(crds[idx0+0],
			  crds[idx0+1],
			  crds[idx0+2], 1.0);
			  
  Matrix3<float> smat;
  smat.setZero();

  for (i=0; i<natom; ++i) {
    int idx = cdixs[istart+i];
    float4 v = make_float4(crds[idx+0],
			   crds[idx+1],
			   crds[idx+2], 1.0);
    v.x -= kv.x;
    v.y -= kv.y;
    v.z -= kv.z;

    smat.aij(1,1) += v.x;
    smat.aij(2,1) += v.y;
    smat.aij(3,1) += v.z;

    smat.aij(1,2) += v.x*v.x;
    smat.aij(2,2) += v.y*v.y;
    smat.aij(3,2) += v.z*v.z;
                 
    smat.aij(1,3) += v.x*v.y;
    smat.aij(2,3) += v.y*v.z;
    smat.aij(3,3) += v.z*v.x;
  }

  smat.scaleSelf(1.0/natom);

  float4 rc = make_float4(smat.aij(1,1)+kv.x,
			  smat.aij(2,1)+kv.y,
			  smat.aij(3,1)+kv.z, 1.0);

  Matrix3<float> resid_tens;
  resid_tens.aij(1,1) = smat.aij(1,2) - smat.aij(1,1)*smat.aij(1,1);
  resid_tens.aij(2,2) = smat.aij(2,2) - smat.aij(2,1)*smat.aij(2,1);
  resid_tens.aij(3,3) = smat.aij(3,2) - smat.aij(3,1)*smat.aij(3,1);
  
  resid_tens.aij(1,2) = smat.aij(1,3) - smat.aij(1,1)*smat.aij(2,1);
  resid_tens.aij(1,3) = smat.aij(3,3) - smat.aij(1,1)*smat.aij(3,1);
  resid_tens.aij(2,3) = smat.aij(2,3) - smat.aij(2,1)*smat.aij(3,1);
  
  resid_tens.aij(2,1) = resid_tens.aij(1,2);
  resid_tens.aij(3,1) = resid_tens.aij(1,3);
  resid_tens.aij(3,2) = resid_tens.aij(2,3);

  float4 evec;
  float4 evals;
  evec = mat33_diag(resid_tens, evals);

  //grad[idx0 + 0] = evals.x;
  //grad[idx0 + 1] = evals.y;
  //grad[idx0 + 2] = evals.z;

  float flag;
  float w = (ithr<nplans)?ppln->wgt:0.0f;

  float resid = 0.0;
  for (i=0; i<natom; ++i) {
    int idx = cdixs[istart+i];
    float4 v = make_float4(crds[idx+0],
			   crds[idx+1],
			   crds[idx+2], 1.0);
    v.x = v.x - rc.x;
    v.y = v.y - rc.y;
    v.z = v.z - rc.z;

    float del = evec.x*v.x + evec.y*v.y + evec.z*v.z;
    resid += w * del * del;

    float dE = 2.0 * w * del;
    atomicAdd(&grad[idx+0], evec.x*dE);
    atomicAdd(&grad[idx+1], evec.y*dE);
    atomicAdd(&grad[idx+2], evec.z*dE);
  }

  float sum = blockReduceSum(resid);

  const int tid = threadIdx.x;
  if (tid == 0) {
    //atomicAdd(&grad[ncrds + 0], sum);
    grad[ncrds+blockIdx.x] += sum;
  }
}

////////////////

CuPlanData::CuPlanData()
{
  pd_plan = NULL;
  pd_cdix = NULL;

  //pd_planres = NULL;
  //pd_ind = NULL;
  //pd_vec = NULL;
}

CuPlanData::~CuPlanData()
{
  cleanupCuda();
}

void CuPlanData::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nplan = m_cuplans.size();

  //////////

  // parameter array (param)
  cudaMalloc((void**)&pd_plan, m_nDevPlan*sizeof(CuPlan));
  cudaMemcpy(pd_plan, &m_cuplans[0], nplan*sizeof(CuPlan), cudaMemcpyHostToDevice);
  printf("CUDA plans (%d*%d) = %p\n", nplan, sizeof(CuPlan), pd_plan);
  
  // plane atom indices
  cudaMalloc((void**)&pd_cdix, m_cdix.size()*sizeof(int));
  cudaMemcpy(pd_cdix, &m_cdix[0], m_cdix.size()*sizeof(int), cudaMemcpyHostToDevice);
}

void CuPlanData::cleanupCuda()
{
  cudaFree(pd_plan);
  pd_plan = NULL;
  cudaFree(pd_cdix);
  pd_cdix = NULL;
}

void CuPlanData::calc()
{
  const std::vector<CuPlan> &param = m_cuplans;

  const int natom = m_nAtoms;
  const int nplan = param.size();

  const int nshmem = m_nthr*sizeof(float);
  
  PlanGradKern1<<<m_nblk, m_nthr, nshmem>>>
    (m_pComDat->pd_crds, pd_plan, pd_cdix, m_pComDat->pd_grad, natom*3, nplan);

#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  // XX
  std::vector<float> grad(natom*3);
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("plan grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
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
