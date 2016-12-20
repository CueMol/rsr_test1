// -*-Mode: C++;-*-
//
// Map implementation file for CUDA
//

#include <vector>
#include <stdio.h>

#include "com_cuda.hpp"
#include "map_cuda.hpp"

//#define DEBUG_PRINT 1

#include "utility.cu"

__constant__ float g_fracMat1[9];

texture<float, cudaTextureType3D, cudaReadModeElementType> texRef1;

__global__ void MapGradKern1(const float* crds, const float* wgts, int natoms,
			     float4 ngrid, float4 stagrid,
			     float *grad)
{
  int iatm = blockIdx.x*blockDim.x + threadIdx.x;

  int icrd = (iatm<natoms)?iatm*3:0;
  //int icrd = iatm*3;

  float4 crd = make_float4(crds[icrd+0], crds[icrd+1], crds[icrd+2], 1.0f);
  float4 frac = matprod(g_fracMat1, crd);
  frac.x *= ngrid.x;
  frac.y *= ngrid.y;
  frac.z *= ngrid.z;
  frac.x -= stagrid.x;
  frac.y -= stagrid.y;
  frac.z -= stagrid.z;

  float4 fl;
  fl.x = floor(frac.x);
  fl.y = floor(frac.y);
  fl.z = floor(frac.z);

  float4 c1;
  c1.x = frac.x - fl.x;
  c1.y = frac.y - fl.y;
  c1.z = frac.z - fl.z;

  float4 c0;
  c0.x = 1.0f - c1.x;
  c0.y = 1.0f - c1.y;
  c0.z = 1.0f - c1.z;

  float cx[4], cy[4], cz[4];
  cx[0] = -0.5f*c1.x*c0.x*c0.x;
  cx[1] = c0.x*( -1.5f*c1.x*c1.x + c1.x + 1.0f );
  cx[2] = c1.x*( -1.5f*c0.x*c0.x + c0.x + 1.0f );
  cx[3] = -0.5*c1.x*c1.x*c0.x;

  cy[0] = -0.5f*c1.y*c0.y*c0.y;
  cy[1] = c0.y*( -1.5f*c1.y*c1.y + c1.y + 1.0f );
  cy[2] = c1.y*( -1.5f*c0.y*c0.y + c0.y + 1.0f );
  cy[3] = -0.5*c1.y*c1.y*c0.y;

  cz[0] = -0.5f*c1.z*c0.z*c0.z;
  cz[1] = c0.z*( -1.5f*c1.z*c1.z + c1.z + 1.0f );
  cz[2] = c1.z*( -1.5f*c0.z*c0.z + c0.z + 1.0f );
  cz[3] = -0.5*c1.z*c1.z*c0.z;

  float gx[4], gy[4], gz[4];
  gx[0] =  c0.x*( 1.5f*c1.x - 0.5f );
  gx[1] =  c1.x*( 4.5f*c1.x - 5.0f );
  gx[2] = -c0.x*( 4.5f*c0.x - 5.0f );
  gx[3] = -c1.x*( 1.5f*c0.x - 0.5f );

  gy[0] =  c0.y*( 1.5f*c1.y - 0.5f );
  gy[1] =  c1.y*( 4.5f*c1.y - 5.0f );
  gy[2] = -c0.y*( 4.5f*c0.y - 5.0f );
  gy[3] = -c1.y*( 1.5f*c0.y - 0.5f );

  gz[0] =  c0.z*( 1.5f*c1.z - 0.5f );
  gz[1] =  c1.z*( 4.5f*c1.z - 5.0f );
  gz[2] = -c0.z*( 4.5f*c0.z - 5.0f );
  gz[3] = -c1.z*( 1.5f*c0.z - 0.5f );

  float rho;
  int i, j, k;
  int ib = int(fl.x) - 1;
  int jb = int(fl.y) - 1;
  int kb = int(fl.z) - 1;

  float s1, s2, s3, dv2, dw2, dw3;
  float4 d1;

  s1 = d1.x = d1.y = d1.z = 0.0;
  for ( i = 0; i < 4; i++ ) {
    s2 = dv2 = dw2 = 0.0;
    for ( j = 0; j < 4; j++ ) {
      s3 = dw3 = 0.0;
      for ( k = 0; k < 4; k++ ) {
	rho = tex3D(texRef1, ib+i, jb+j, kb+k);
	s3 += cz[k] * rho;
	dw3 += gz[k] * rho;
      }
      s2 += cy[j] * s3;
      dv2 += gy[j] * s3;
      dw2 += cy[j] * dw3;
    }
    s1   += cx[i] * s2;
    d1.x += gx[i] * s2;
    d1.y += cx[i] * dv2;
    d1.z += cx[i] * dw2;
  }

  d1.x *= ngrid.x;
  d1.y *= ngrid.y;
  d1.z *= ngrid.z;

  d1 = matprod_tp(g_fracMat1, d1);

  float w = wgts[iatm];
  d1.x *= w;
  d1.y *= w;
  d1.z *= w;

  grad[icrd+0] += d1.x;
  grad[icrd+1] += d1.y;
  grad[icrd+2] += d1.z;

  const int tid = threadIdx.x;

  float sum = blockReduceSum(s1 * w);

  if (tid == 0) {
    //atomicAdd(&grad[natoms*3 + 0], sum);
    grad[natoms*3+blockIdx.x] += sum;
  }
}



CuMapData::CuMapData()
{
  pd_wgts = NULL;
  pd_map = NULL;
}

CuMapData::~CuMapData()
{
  cleanupCuda();
}

void CuMapData::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int ntoth = m_nDevAtom;

  // Weights array (wgts)
  gpuErrChk( cudaMalloc((void**)&pd_wgts, ntoth*sizeof(float)) );
  gpuErrChk( cudaMemcpy( pd_wgts, &m_wgts[0], ntoth*sizeof(float), cudaMemcpyHostToDevice) );
  //for (int i=0; i<ntoth; ++i) {
  //printf("wgts: %d %f\n", i, m_wgts[i]);
  //}

  // 3x3 frac matrix
  cudaMemcpyToSymbol(g_fracMat1, &m_fracMat[0], 9*sizeof(float));

  // Density map (3D texture)
  cudaChannelFormatDesc cdesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaExtent ext = make_cudaExtent(m_ncol, m_nrow, m_nsec);
  gpuErrChk(cudaMalloc3DArray((cudaArray_t *)&pd_map, &cdesc, ext));
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(&m_map[0], m_ncol*sizeof(float), m_ncol, m_nrow);
  copyParams.dstArray = (cudaArray_t)pd_map;
  copyParams.extent   = ext;
  copyParams.kind     = cudaMemcpyHostToDevice;
  gpuErrChk(cudaMemcpy3D(&copyParams));
  gpuErrChk(cudaBindTextureToArray(texRef1, (cudaArray_t)pd_map, cdesc)); 

}

void CuMapData::cleanupCuda()
{
  cudaFree(pd_wgts);
  pd_wgts = NULL;

  cudaUnbindTexture(texRef1); 
  cudaFreeArray((cudaArray_t)pd_map);
  pd_map = NULL;
}

void CuMapData::calc()
{
  const int natom = m_nAtoms;
  const int nshmem = m_pComDat->m_nthr*sizeof(float);

  MapGradKern1<<<m_pComDat->m_nblk, m_pComDat->m_nthr, nshmem>>>
    (m_pComDat->pd_crds, pd_wgts, natom,
     make_float4(m_na, m_nb, m_nc, 1.0),
     make_float4(m_stcol, m_strow, m_stsec, 1.0),
     m_pComDat->pd_grad);
  
#ifdef DEBUG_PRINT
  printf("kern exec OK\n");

  std::vector<float> grad(natom*3);
  m_pComDat->xferGrad(grad);
  for (int i=0; i<natom; ++i) {
    printf("grad %d: %f %f %f\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
#endif
}

/*
void CuMapData::setup(MolData *pMol, DensityMap *pMap)
{
}

void CuMap2Data::setup(MolData *pMol, DensityMap *pMap)
{
}
*/

