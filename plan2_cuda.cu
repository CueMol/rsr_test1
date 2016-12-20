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
void PlanGradKern2(const float* crds, const CuPlan* param, const int *cdixs,
		   float *grad, int ncrds, int nplans)
{
  float resid=0, del=0;
  int i;
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;

  const int ipln_dev = ithr/PLAN2_NTHR;

  //if (ipln_dev>=nplans)
  //return;
  if (ipln_dev<nplans) {

    //const int ipln = (ipln_dev<nplans)?ipln_dev:0;
    const int ipln = ipln_dev;

    const CuPlan *ppln = &param[ipln];
  
    const int iatom = ithr%PLAN2_NTHR;
    const int istart = ppln->istart;
    const int natom = ppln->natom;

    int idx0 = cdixs[istart];
    float4 kv = make_float4(crds[idx0+0],
			    crds[idx0+1],
			    crds[idx0+2], 1.0);
			  
    //  Matrix3<float> smat;
    // smat.setZero();
    float4 v = make_float4(0,0,0,0);
    float4 vsum = make_float4(0,0,0,0);
    float4 vsqsum = make_float4(0,0,0,0);
    float4 vcrsum = make_float4(0,0,0,0);

    int idx = -1;
    if (iatom<natom) {
      idx = cdixs[istart+iatom];
      v.x = crds[idx+0];
      v.y = crds[idx+1];
      v.z = crds[idx+2];
      vsum.x = v.x - kv.x;
      vsum.y = v.y - kv.y;
      vsum.z = v.z - kv.z;
    }
  
    vsqsum.x = vsum.x*vsum.x;
    vsqsum.y = vsum.y*vsum.y;
    vsqsum.z = vsum.z*vsum.z;
  
    vcrsum.x = vsum.x*vsum.y;
    vcrsum.y = vsum.y*vsum.z;
    vcrsum.z = vsum.z*vsum.x;
  
    {
#pragma unroll
      for (int offset = PLAN2_NTHR/2; offset > 0; offset /= 2) {
	vsum.x += __shfl_down(vsum.x, offset);
	vsum.y += __shfl_down(vsum.y, offset);
	vsum.z += __shfl_down(vsum.z, offset);

	vsqsum.x += __shfl_down(vsqsum.x, offset);
	vsqsum.y += __shfl_down(vsqsum.y, offset);
	vsqsum.z += __shfl_down(vsqsum.z, offset);

	vcrsum.x += __shfl_down(vcrsum.x, offset);
	vcrsum.y += __shfl_down(vcrsum.y, offset);
	vcrsum.z += __shfl_down(vcrsum.z, offset);
      }
    }

    float4 evec;
    float4 evals;
    float4 rc;

    if (iatom==0) {
      vsum.x /= natom;
      vsum.y /= natom;
      vsum.z /= natom;

      vsqsum.x /= natom;
      vsqsum.y /= natom;
      vsqsum.z /= natom;

      vcrsum.x /= natom;
      vcrsum.y /= natom;
      vcrsum.z /= natom;

      rc = make_float4(vsum.x+kv.x,
		       vsum.y+kv.y,
		       vsum.z+kv.z, 1.0);
  
      Matrix3<float> resid_tens;
      resid_tens.aij(1,1) = vsqsum.x - vsum.x*vsum.x;
      resid_tens.aij(2,2) = vsqsum.y - vsum.y*vsum.y;
      resid_tens.aij(3,3) = vsqsum.z - vsum.z*vsum.z;
  
      resid_tens.aij(1,2) = vcrsum.x - vsum.x*vsum.y;
      resid_tens.aij(1,3) = vcrsum.z - vsum.x*vsum.z;
      resid_tens.aij(2,3) = vcrsum.y - vsum.y*vsum.z;
  
      resid_tens.aij(2,1) = resid_tens.aij(1,2);
      resid_tens.aij(3,1) = resid_tens.aij(1,3);
      resid_tens.aij(3,2) = resid_tens.aij(2,3);

      evec = mat33_diag(resid_tens, evals);
    }

    /*
      if (iatom==0) {
      const int idx = ipln_dev * 3;
      grad[idx + 0] = evals.x;
      grad[idx + 1] = evals.y;
      grad[idx + 2] = evals.z;
      }
    */

    {
      //const int ifrom = (ipln_dev%4)*8;
      const int ifrom = (ipln_dev%(PLAN2_NTHR/2))*PLAN2_NTHR;
      //printf("%d %d\n", ithr, ifrom);
      rc.x = __shfl(rc.x, ifrom);
      rc.y = __shfl(rc.y, ifrom);
      rc.z = __shfl(rc.z, ifrom);

      evec.x = __shfl(evec.x, ifrom);
      evec.y = __shfl(evec.y, ifrom);
      evec.z = __shfl(evec.z, ifrom);
    }

    v.x = v.x - rc.x;
    v.y = v.y - rc.y;
    v.z = v.z - rc.z;

    del = evec.x*v.x + evec.y*v.y + evec.z*v.z;

    float dE;

    if (idx>=0) {
      const float w = ppln->wgt;
      resid = w * del * del;
      dE = 2.0 * w * del;
      atomicAdd(&grad[idx + 0], evec.x*dE);
      atomicAdd(&grad[idx + 1], evec.y*dE);
      atomicAdd(&grad[idx + 2], evec.z*dE);
    }
  }

  //if (resid!=0.0)
  //printf("thr %d del=%f resid %f\n", threadIdx.x, del, resid);

  float sum = blockReduceSum(resid);

  const int tid = threadIdx.x;
  if (tid == 0) {
    grad[ncrds+blockIdx.x] += sum;
  }
}

////////////////

void CuPlan2Data::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nplan = m_cuplans.size();

  //////////

  // parameter array (param)
  cudaMalloc((void**)&pd_plan, nplan*sizeof(CuPlan));
  cudaMemcpy(pd_plan, &m_cuplans[0], nplan*sizeof(CuPlan), cudaMemcpyHostToDevice);
  printf("CUDA plans (%d*%d) = %p\n", nplan, sizeof(CuPlan), pd_plan);
  
  // plane atom indices
  cudaMalloc((void**)&pd_cdix, m_cdix.size()*sizeof(int));
  cudaMemcpy(pd_cdix, &m_cdix[0], m_cdix.size()*sizeof(int), cudaMemcpyHostToDevice);
}

void CuPlan2Data::calc()
{
  const std::vector<CuPlan> &param = m_cuplans;

  const int natom = m_nAtoms;
  const int nplan = param.size();

  const int nshmem = m_nthr*sizeof(float);
  
  PlanGradKern2<<<m_nblk, m_nthr, nshmem>>>
    (m_pComDat->pd_crds, pd_plan, pd_cdix, m_pComDat->pd_grad, natom*3, nplan);

#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  // XX
  std::vector<float> grad(natom*3 + GRAD_PAD);
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("plan grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  for (int i=0; i<GRAD_PAD; ++i) {
    printf("plan ener %d: %f\n", i, grad[natom*3 + i]);
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
