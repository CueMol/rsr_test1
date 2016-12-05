// -*-Mode: C++;-*-

#include <stdio.h>
#include <vector>
#include "cudacode.h"
 
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

//#define N (256*256)

#include "cuda_bond_kern.hpp"

void cudaBond_fdf(const std::vector<float> &crds,
		  CudaData *pDat,
		  float *val, std::vector<float> &grad)
{
  const std::vector<CuBond> &param = pDat->cubonds;
  const std::vector<int> &bvec = pDat->bvec;
  int nacc = pDat->nacc;
  const int ncrds = crds.size();
  const int nbond = param.size();

  // coordinates array (crds)
  if (pDat->pd_crds==NULL)
    cudaMalloc((void**)&pDat->pd_crds, ncrds*sizeof(float));
  cudaMemcpy( pDat->pd_crds, &crds[0], ncrds*sizeof(float), cudaMemcpyHostToDevice);
  //printf("CUDA crds (%d*%d) = %p\n", ncrds, sizeof(float), pDat->pd_crds);

  // parameter array (param)
  if (pDat->pd_bond==NULL) {
    cudaMalloc((void**)&pDat->pd_bond, nbond*sizeof(CuBond));
    cudaMemcpy( pDat->pd_bond, &param[0], nbond*sizeof(CuBond), cudaMemcpyHostToDevice);
  }
  //printf("CUDA bonds (%d*%d) = %p\n", nbond, sizeof(CuBond), pDat->pd_bond);

  // bond index array
  const int nbvec = bvec.size();
  cudaChannelFormatDesc cdesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  if (pDat->pd_bvec==NULL) {
    cudaMalloc((void**)&pDat->pd_bvec, nbvec*sizeof(int));
    cudaMemcpy( pDat->pd_bvec, &bvec[0], nbvec*sizeof(int), cudaMemcpyHostToDevice);
  }
  //printf("CUDA ibvec (%d*%d) = %p\n", nbvec, sizeof(int), pDat->pd_bvec);

  // result: grad vec
  if (pDat->pd_grad==NULL) {
    int dev_ncrds = pDat->dev_natom * 3;
    //int dev_ncrds = pDat->dev_natom * 4;
    cudaMalloc((void**)&pDat->pd_grad, dev_ncrds*sizeof(float));
    cudaMalloc((void**)&pDat->pd_eatm, pDat->nblk*sizeof(float));
  }

  BondGradKern<<<pDat->nblk, pDat->nthr, pDat->nthr*sizeof(float)>>>
    (pDat->pd_crds, pDat->pd_bond,
     pDat->pd_bvec, nacc,
     pDat->pd_grad, pDat->pd_eatm);

  //printf("kern exec OK\n");

  cudaThreadSynchronize();

  //printf("kern synch OK\n");
  const int natom = ncrds/3;

  cudaMemcpy( &grad[0], pDat->pd_grad, natom*3*sizeof(float), cudaMemcpyDeviceToHost);

#if 0
  std::vector<float> gradtmp(natom*4);
  cudaMemcpy( &gradtmp[0], pDat->pd_grad, natom*4*sizeof(float), cudaMemcpyDeviceToHost);
  //float energy = 0.0f;
  for (int i=0;i<natom; ++i) {
    grad[i*3+0] = gradtmp[i*4+0];
    grad[i*3+1] = gradtmp[i*4+1];
    grad[i*3+2] = gradtmp[i*4+2];
    //energy +=  gradtmp[i*4+3];
    /*
    union {
      float f;
      unsigned int ui;
    } u;
    u.f = gradtmp[i*4+3];
    printf("   %d  %.16e [%x]\n", i, u.f, u.ui);
    */
  }
  //*val = energy;
#endif

  std::vector<float> ebonds(pDat->nblk);
  cudaMemcpy( &ebonds[0], pDat->pd_eatm, pDat->nblk*sizeof(float), cudaMemcpyDeviceToHost);

  *val = 0.0f;
  for (int i=0; i<pDat->nblk; ++i)
    *val += ebonds[i];

  //cudaUnbindTexture(texRef1); 

  //printf("Results copy OK\n");
}


#include "cuda_bond_kern2.hpp"
void cudaBond_fdf2(const std::vector<float> &crds,
		   CudaData2 *pDat,
		   float *val, std::vector<float> &grad)
{
  const std::vector<CuBond> &param = pDat->cubonds;
  const std::vector<int> &bvec = pDat->bvec;

  const int ncrds = crds.size();
  const int nbond = param.size();

  // coordinates array (crds)
  if (pDat->pd_crds==NULL)
    cudaMalloc((void**)&pDat->pd_crds, ncrds*sizeof(float));
  cudaMemcpy( pDat->pd_crds, &crds[0], ncrds*sizeof(float), cudaMemcpyHostToDevice);

  // parameter array (param, constant)
  if (pDat->pd_bond==NULL) {
    cudaMalloc((void**)&pDat->pd_bond, nbond*sizeof(CuBond));
    cudaMemcpy( pDat->pd_bond, &param[0], nbond*sizeof(CuBond), cudaMemcpyHostToDevice);
  }

  // bond index array (constant)
  const int nbvec = bvec.size();
  if (pDat->pd_bvec==NULL) {
    cudaMalloc((void**)&pDat->pd_bvec, nbvec*sizeof(int));
    cudaMemcpy( pDat->pd_bvec, &bvec[0], nbvec*sizeof(int), cudaMemcpyHostToDevice);
  }

  // result: grad vec, Energy
  if (pDat->pd_grad==NULL) {
    cudaMalloc((void**)&pDat->pd_grad, ncrds*sizeof(float));
    //cudaMalloc((void**)&pDat->pd_grad, nbvec*sizeof(float));
    cudaMalloc((void**)&pDat->pd_eatm, pDat->nblk*sizeof(float));
  }

  BondGradKern2<<<pDat->nblk, pDat->nthr, pDat->nthr*sizeof(float)*4>>>
    (pDat->pd_crds, pDat->pd_bond,
     pDat->pd_bvec,
     pDat->pd_grad, pDat->pd_eatm);

  //printf("kern exec OK\n");

  cudaThreadSynchronize();

  //printf("kern synch OK\n");

  // Copy result: grad vec
  cudaMemcpy( &grad[0], pDat->pd_grad, ncrds*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy( &grad[0], pDat->pd_grad, nbvec*sizeof(float), cudaMemcpyDeviceToHost);
  //for (int i=0; i<nbvec/2; ++i)
  //printf("thr %d: grad y=%f z=%f\n", i, grad[i*2], grad[i*2+1]);

  const int natom = ncrds/3;
  std::vector<float> ebonds(pDat->nblk);
  cudaMemcpy( &ebonds[0], pDat->pd_eatm, pDat->nblk*sizeof(float), cudaMemcpyDeviceToHost);

  *val = 0.0f;
  for (int i=0; i<pDat->nblk; ++i)
    *val += ebonds[i];

  //printf("Results copy OK\n");
}


