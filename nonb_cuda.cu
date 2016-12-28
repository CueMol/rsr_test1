// -*-Mode: C++;-*-
//
// Nonbonded intr implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"
#include "nonb_cuda.hpp"

//#define DEBUG_PRINT 1

void CuNonbData::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;


  /*
  const int nnonb = m_mat.size();
  // parameter array (param)
  cudaMalloc((void**)&pd_nonb, nnonb*sizeof(CuNonb));
  cudaMemcpy(pd_nonb, &m_mat[0], nnonb*sizeof(CuNonb), cudaMemcpyHostToDevice);
  printf("CUDA nonbs (%d*%d) = %p\n", nnonb, sizeof(CuNonb), pd_nonb);
  */

  const int nnonb = m_indmat.size();
  // parameter array (param)
  gpuErrChk( cudaMalloc((void**)&pd_indmat, nnonb*sizeof(int)) );
  gpuErrChk( cudaMemcpy(pd_indmat, &m_indmat[0], nnonb*sizeof(int), cudaMemcpyHostToDevice) );

  gpuErrChk( cudaMalloc((void**)&pd_prmmat, nnonb*sizeof(int)) );
  gpuErrChk( cudaMemcpy(pd_prmmat, &m_prmmat[0], nnonb*sizeof(int), cudaMemcpyHostToDevice) );

}

void CuNonbData::cleanupCuda()
{
  //cudaFree(pd_nonb);
  //pd_nonb = NULL;

  cudaFree(pd_indmat);
  pd_indmat = NULL;

  cudaFree(pd_prmmat);
  pd_prmmat = NULL;
}


//////////////////////////////////////////

#include "utility.cu"

__global__
void NonbGradKern1(const float* crds, const int* indmat, const int* prmmat,
		   float *grad, int natom, int nonb_max, int nloop)
{
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;
  const int iatom_dev = ithr/warpSize;
  const int laneid = ithr%warpSize;

  float4 g = make_float4(0,0,0,0);

  if (iatom_dev<natom) {

    const int iatom=iatom_dev;
    //const int iatom=(iatom_dev>=natom)?0:iatom_dev;

    const int ai = iatom*3;
    const int ibase = laneid*nloop;
    const int nmax = nloop+ibase;
    //const int imax = (nloop+ibase>nonb_max)?nonb_max:nloop+ibase;

    int i, aj;
    float4 d;
    float r0, wgt;

    //printf("thr %d iatom %d \n", ithr, iatom);

    const int idx_base = iatom*nonb_max;

    for (i=ibase; i<nmax; ++i) {
    //for (i=0; i<nloop; ++i) {
      /*int idx = -1;
	if (i<nonb_max) {
	idx = idx_base + i;
	}*/
      //const int idx = i*nonb_max + iatom;
      const int idx = (i<nonb_max)?(idx_base+i+1):0;

      //const CuNonb &nonb = param[idx];
      //aj = nonb.idx;
      //r0 = nonb.r0;
      //wgt = nonb.wgt;

      aj = indmat[idx];
      r0 = 3.5;
      wgt = float(prmmat[idx]);

      d.x = crds[ai+0] - crds[aj+0];
      d.y = crds[ai+1] - crds[aj+1];
      d.z = crds[ai+2] - crds[aj+2];

      const float len = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    
      float ss = len-r0;
      //ss = fminf(ss, 0.0f);
      ss = (ss<0.0f)?ss:0.0f;
    
      const float rlen = 1.0/( (0.1>len)?0.1:len );
      const float con = 2.0f * wgt * ss*rlen;
      g.x += con*d.x;
      g.y += con*d.y;
      g.z += con*d.z;
      g.w += wgt * ss * ss * 0.5;

      /*
	g.x += 1.0*wgt;
	g.y += 2.0;
	g.z += 3.0;
	g.w += 4.0;
      */
    
      //if (iatom==0)
      //printf("thr %d i=%d, idx=%d --> (%f, %f, %f, %f)\n", ithr, i, idx, g.x, g.y, g.z, g.w);
      //printf("thr %d i=%d, idx=%d --> aj=%d, len=%f ss=%f\n", ithr, i, idx, aj/3, len, ss);
    }

      
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      g.x += __shfl_down(g.x, offset);
      g.y += __shfl_down(g.y, offset);
      g.z += __shfl_down(g.z, offset);
      g.w += __shfl_down(g.w, offset);
    }

    if (laneid==0) {
      //printf("atom %d %f,%f,%f\n", ai/3, g.x, g.y, g.z);
      atomicAdd(&grad[ai+0], g.x);
      atomicAdd(&grad[ai+1], g.y);
      atomicAdd(&grad[ai+2], g.z);
    }
    else {
      g.w = 0.0;
    }
  }

  const int ncrds = natom*3;

  //printf("thr %d --> (%f, %f, %f, %f)\n", ithr, bg.x, g.y, g.z, g.w);
  float sum = blockReduceSum(g.w);
  const int tid = threadIdx.x;
  if (tid == 0) {
    atomicAdd(&grad[ncrds+blockIdx.x%GRAD_PAD], sum);
  }
}

void CuNonbData::calc()
{
  //const std::vector<CuNonb> &param = m_mat;

  const int natom = m_nAtoms;
  //const int nnonb = param.size();

  //  for (int i=0; i<nnonb; ++i) {
  //    printf("%d %d<-->%d %f %f\n", i, i/m_nonb_max, param[i].idx, param[i].r0, param[i].wgt);
  //  }

  //  NonbGradKern1<<<m_nblk, m_nthr>>>
  //    (m_pComDat->pd_crds, pd_nonb, nnonb, m_pComDat->pd_grad, natom, m_nonb_max, m_nloop);
  
  NonbGradKern1<<<m_nblk, m_nthr>>>
    (m_pComDat->pd_crds, pd_indmat, pd_prmmat, m_pComDat->pd_grad, natom, m_nonb_max, m_nloop);

#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  // XX
  std::vector<float> grad(natom*3 + GRAD_PAD);
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
#endif

}
