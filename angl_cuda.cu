// -*-Mode: C++;-*-
//
// Angl implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"
#include "angl_cuda.hpp"

//#define DEBUG_PRINT 1

CuAnglData::CuAnglData()
{
  pd_angl = NULL;
  pd_eatm = NULL;

  //pd_anglres = NULL;
  //pd_ind = NULL;
  //pd_vec = NULL;
}

CuAnglData::~CuAnglData()
{
  cleanupCuda();
}

void CuAnglData::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nangl = m_cuangls.size();

  calcThrBlk(nangl, &m_nthr, &m_nblk, &m_nDevAngl);
  printf("Angl nThr = %d (%d x %d)\n", m_nDevAngl, m_nblk, m_nthr);

  //////////
  // memory for 1st stage

  // parameter array (param)
  cudaMalloc((void**)&pd_angl, m_nDevAngl*sizeof(CuAngl));
  cudaMemcpy(pd_angl, &m_cuangls[0], nangl*sizeof(CuAngl), cudaMemcpyHostToDevice);
  printf("CUDA angls (%d*%d) = %p\n", nangl, sizeof(CuAngl), pd_angl);
  
  /*
  // intermediate result array
  cudaMalloc((void**)&pd_anglres, m_nDevAngl*4*sizeof(float));

  //////////
  // memory for 2nd stage (grad gather)

  // atom-->angl index (1)
  const int nbind = m_pComDat->m_nDevAtom;
  cudaMalloc((void**)&pd_bind, nbind*sizeof(int));
  cudaMemcpy( pd_bind, &m_bind[0], nbind*sizeof(int), cudaMemcpyHostToDevice);

  // atom-->angl index array (for gather phase)
  const int nbvec = m_bvec.size();
  cudaMalloc((void**)&pd_bvec, nbvec*sizeof(int));
  cudaMemcpy( pd_bvec, &m_bvec[0], nbvec*sizeof(int), cudaMemcpyHostToDevice);
  printf("CUDA ibvec (%d*%d) = %p\n", nbvec, sizeof(int), pd_bvec);
  */

  // atom energy array
  cudaMalloc((void**)&pd_eatm, m_pComDat->m_nblk*sizeof(float));

  //////////
  // memory for 3rd stage (energy gather)
}

void CuAnglData::cleanupCuda()
{
  cudaFree(pd_angl);
  cudaFree(pd_eatm);

  pd_angl = NULL;
  pd_eatm = NULL;
  /*
  cudaFree(pd_anglres);
  cudaFree(pd_bind);
  cudaFree(pd_bvec);

  pd_anglres = NULL;
  pd_bind = NULL;
  pd_bvec = NULL;
  */
}

#include "utility.cu"

__global__
void AnglGradKern2(const float* crds, const CuAngl* param,
		    float *grad, int ncrds, int nangls)
{
  const int ithr = blockIdx.x*blockDim.x + threadIdx.x;

  float flag;
  if (ithr>=nangls)
    flag = 0.0f;
  else
    flag = 1.0f;

  const int iang = (ithr<nangls)?ithr:0;

  const CuAngl *pang = &param[iang];
  
  const int ai = pang->ai;
  const int aj = pang->aj;
  const int ak = pang->ak;
  
  const float rijx = crds[ai+0] - crds[aj+0];
  const float rijy = crds[ai+1] - crds[aj+1];
  const float rijz = crds[ai+2] - crds[aj+2];

  const float rkjx = crds[ak+0] - crds[aj+0];
  const float rkjy = crds[ak+1] - crds[aj+1];
  const float rkjz = crds[ak+2] - crds[aj+2];

  //const float Rij = sqrt( rijx*rijx + rijy*rijy + rijz*rijz );
  //const float Rkj = sqrt( rkjx*rkjx + rkjy*rkjy + rkjz*rkjz );

  const float rRij = rnorm3df( rijx, rijy, rijz );
  const float rRkj = rnorm3df( rkjx, rkjy, rkjz );
  /*
  const float eijx = rijx / Rij;
  const float eijy = rijy / Rij;
  const float eijz = rijz / Rij;

  const float ekjx = rkjx / Rkj;
  const float ekjy = rkjy / Rkj;
  const float ekjz = rkjz / Rkj;
  */
  const float eijx = rijx * rRij;
  const float eijy = rijy * rRij;
  const float eijz = rijz * rRij;

  const float ekjx = rkjx * rRkj;
  const float ekjy = rkjy * rRkj;
  const float ekjz = rkjz * rRkj;

  float costh = eijx*ekjx + eijy*ekjy + eijz*ekjz;
  costh = (costh>1.0f)?1.0f:costh;
  costh = (costh<-1.0f)?-1.0f:costh;
  //const float sinth = sqrt( 1.0f-costh*costh );
  const float rsinth = rsqrtf( 1.0f-costh*costh );

  //costh = qlib::min<realnum_t>(1.0f, qlib::max<realnum_t>(-1.0f, costh));
  const float theta = acos(costh);
  const float dtheta = theta - pang->r0;

  const float eng = pang->kf*dtheta*dtheta * flag;
  const float df = 2.0*(pang->kf)*dtheta * flag;

  //const float Dij =  df/(sinth*Rij);
  //const float Dkj =  df/(sinth*Rkj);
  const float Dij =  df*rRij*rsinth;
  const float Dkj =  df*rRkj*rsinth;

  const float dijx = Dij*(costh*eijx - ekjx);
  const float dijy = Dij*(costh*eijy - ekjy);
  const float dijz = Dij*(costh*eijz - ekjz);
  
  const float dkjx = Dkj*(costh*ekjx - eijx);
  const float dkjy = Dkj*(costh*ekjy - eijy);
  const float dkjz = Dkj*(costh*ekjz - eijz);

  atomicAdd(&grad[ai+0], dijx);
  atomicAdd(&grad[ai+1], dijy);
  atomicAdd(&grad[ai+2], dijz);

  atomicAdd(&grad[aj+0], -dijx-dkjx);
  atomicAdd(&grad[aj+1], -dijy-dkjy);
  atomicAdd(&grad[aj+2], -dijz-dkjz);

  atomicAdd(&grad[ak+0], dkjx);
  atomicAdd(&grad[ak+1], dkjy);
  atomicAdd(&grad[ak+2], dkjz);

  const int tid = threadIdx.x;

  float sum = blockReduceSum(eng);
  /*
  extern __shared__ float sdata[];
  sdata[tid] = eng;

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
    //atomicAdd(&grad[ncrds], sum);
    atomicAdd(&grad[ncrds+blockIdx.x], sum);
    //grad[ncrds+blockIdx.x] += sum;
  }

}

void CuAnglData::calc(float *val, std::vector<float> &grad)
{
  const std::vector<CuAngl> &param = m_cuangls;

  const int natom = m_nAtoms;
  const int nangl = param.size();

  const int nshmem = m_nthr*sizeof(float);

  AnglGradKern2<<<m_nblk, m_nthr, nshmem>>>
    (m_pComDat->pd_crds, pd_angl, m_pComDat->pd_grad, natom*3, nangl);

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
