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



#include "cuda_map_kern.hpp"

void cudaMap_fdf1(const std::vector<float> &crds,
		 CudaMapData *pDat,
		 float *val, std::vector<float> &grad)
{
  const int ncrds = crds.size();
  const int natom = ncrds/3;
  const int ntoth = pDat->nblk * pDat->nthr;

  // Coordinates array (crds)
  if (pDat->pd_crds==NULL)
    cudaMalloc((void**)&pDat->pd_crds, ncrds*sizeof(float));
  cudaMemcpy( pDat->pd_crds, &crds[0], ncrds*sizeof(float), cudaMemcpyHostToDevice);
  //printf("CUDA crds (%d*%d) = %p\n", ncrds, sizeof(float), pDat->pd_crds);

  // Weights array (wgts)
  if (pDat->pd_wgts==NULL) {
    cudaMalloc((void**)&pDat->pd_wgts, ntoth*sizeof(float));
    cudaMemcpy( pDat->pd_wgts, &pDat->wgts[0], ntoth*sizeof(float), cudaMemcpyHostToDevice);
    //for (int i=0; i<ntoth; ++i) {
    //printf("wgts: %d %f\n", i, pDat->wgts[i]);
    //}
    cudaMemcpyToSymbol(g_fracMat1, &pDat->fracMat[0], 9*sizeof(float));
  }

  // Density map (3D texture)
  cudaChannelFormatDesc cdesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  if (pDat->pd_map==NULL) {
    cudaExtent ext = make_cudaExtent(pDat->ncol,pDat->nrow,pDat->nsec);
    gpuErrChk(cudaMalloc3DArray((cudaArray_t *)&pDat->pd_map, &cdesc, ext));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(&pDat->p_map[0], pDat->ncol*sizeof(float), pDat->ncol,pDat->nrow);
    copyParams.dstArray = (cudaArray_t)pDat->pd_map;
    copyParams.extent   = ext;
    copyParams.kind     = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy3D(&copyParams));
    gpuErrChk(cudaBindTextureToArray(texRef1, (cudaArray_t)pDat->pd_map, cdesc)); 
  }

  // result: grad vec
  if (pDat->pd_grad==NULL) {
    //cudaMalloc((void**)&pDat->pd_grad, ncrds*sizeof(float));
    cudaMalloc((void**)&pDat->pd_grad, ntoth*3*sizeof(float));
    cudaMalloc((void**)&pDat->pd_eatm, pDat->nblk*sizeof(float));
  }
  
  const int nshmem = pDat->nthr * sizeof(float);
  MapGradKern1<<<pDat->nblk, pDat->nthr, nshmem>>>
    (pDat->pd_crds, pDat->pd_wgts,
     make_float4(pDat->na, pDat->nb, pDat->nc, 1.0),
     make_float4(pDat->stcol, pDat->strow, pDat->stsec, 1.0),
     pDat->pd_grad, pDat->pd_eatm);

  //printf("kern exec OK\n");

  cudaThreadSynchronize();

  //printf("kern synch OK\n");

  cudaMemcpy( &grad[0], pDat->pd_grad, natom*3*sizeof(float), cudaMemcpyDeviceToHost);

#if 0
  std::vector<float> gradtmp(natom*4);
  cudaMemcpy( &gradtmp[0], pDat->pd_grad, natom*4*sizeof(float), cudaMemcpyDeviceToHost);

  //*val = 0.0f;
  for (int i=0;i<natom; ++i) {
    /*union {
      float f;
      unsigned int ui;
    } u;
    u.f = grad[i*4+3];
    printf("  eng.x %d  %.16e [%x]\n", i, u.f, u.ui);*/
    grad[i*3+0] = gradtmp[i*4+0];
    grad[i*3+1] = gradtmp[i*4+1];
    grad[i*3+2] = gradtmp[i*4+2];
    //*val += gradtmp[i*4+3];
  }
#endif

  cudaMemcpy( &pDat->eatom[0], pDat->pd_eatm, pDat->nblk*sizeof(float), cudaMemcpyDeviceToHost);
  *val = 0.0f;
  for (int i=0; i<pDat->nblk; ++i)
    *val += pDat->eatom[i];


  //cudaUnbindTexture(texRef1); 

  //printf("Results copy OK\n");
}


#include "cuda_map_kern2.hpp"

void cudaMap_fdf2(const std::vector<float> &crds,
		  CudaMapData *pDat,
		  float *val, std::vector<float> &grad)
{
  const int ncrds = crds.size();
  const int natom = ncrds/3;

  // Coordinates array (crds)
  if (pDat->pd_crds==NULL)
    cudaMalloc((void**)&pDat->pd_crds, ncrds*sizeof(float));
  cudaMemcpy( pDat->pd_crds, &crds[0], ncrds*sizeof(float), cudaMemcpyHostToDevice);

  // Weights array (wgts)
  if (pDat->pd_wgts==NULL) {
    int nwgt = pDat->wgts.size();
    cudaMalloc((void**)&pDat->pd_wgts, nwgt*sizeof(float));
    cudaMemcpy( pDat->pd_wgts, &pDat->wgts[0], nwgt*sizeof(float), cudaMemcpyHostToDevice);
    //for (int i=0; i<natom; ++i) {
    //printf("wgts: %d %f\n", i, pDat->wgts[i]);
    //}

    cudaMemcpyToSymbol(g_fracMat2, &pDat->fracMat[0], 9*sizeof(float));

    float gcoef[] = {
      -0.5,  0.0, 0, 0, 0,
      0.0,  -1.5, 1, 1, 0,
      -1.5,  0.0, 1, 0, 1,
      0.0,  -0.5, 0, 0, 0,
    };
    cudaMemcpyToSymbol(g_coefMat2, gcoef, 20*sizeof(float));
  }

  // Density map (3D texture)
  cudaChannelFormatDesc cdesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  if (pDat->pd_map==NULL) {
    cudaExtent ext = make_cudaExtent(pDat->ncol,pDat->nrow,pDat->nsec);
    gpuErrChk(cudaMalloc3DArray((cudaArray_t *)&pDat->pd_map, &cdesc, ext));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(&pDat->p_map[0], pDat->ncol*sizeof(float), pDat->ncol,pDat->nrow);
    copyParams.dstArray = (cudaArray_t)pDat->pd_map;
    copyParams.extent   = ext;
    copyParams.kind     = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy3D(&copyParams));
    gpuErrChk(cudaBindTextureToArray(texRef2, (cudaArray_t)pDat->pd_map, cdesc)); 
  }

  // result: grad vec/Eatom
  if (pDat->pd_grad==NULL) {
    cudaMalloc((void**)&pDat->pd_grad, ncrds*sizeof(float));
    cudaMalloc((void**)&pDat->pd_eatm, pDat->nblk*sizeof(float));
  }
  
  const int nshmem = pDat->nthr * sizeof(float);
  MapGradKern2<<<pDat->nblk, pDat->nthr, nshmem>>>
    (pDat->pd_crds, pDat->pd_wgts,
     make_float4(pDat->na, pDat->nb, pDat->nc, 1.0),
     make_float4(pDat->stcol, pDat->strow, pDat->stsec, 1.0),
     pDat->pd_grad, pDat->pd_eatm);

  //printf("kern exec OK\n");

  cudaThreadSynchronize();

  //printf("kern synch OK\n");

  cudaMemcpy( &grad[0], pDat->pd_grad, ncrds*sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy( &pDat->eatom[0], pDat->pd_eatm, pDat->nblk*sizeof(float), cudaMemcpyDeviceToHost);

  *val = 0.0f;

  for (int i=0; i<pDat->eatom.size(); ++i) {
    /*
    int x = i%64;
    int ii = x/16;
    int jj = (x%16)/4;
    int kk = x%4;
    printf("%d %d %d %e\n", ii, jj, kk, pDat->eatom[i]);*/
    *val += pDat->eatom[i];
  }

  //cudaUnbindTexture(texRef2); 

  //printf("Results copy OK\n");
}
