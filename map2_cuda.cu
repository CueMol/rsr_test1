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

__constant__ float g_coefMat2[4*5*2];

__inline__ __device__
float gc(int i, int j)
{
  return g_coefMat2[j + i*5];
}

__inline__ __device__
float gg(int i, int j)
{
  return g_coefMat2[j + i*5 + 20];
}

__inline__ __device__
float4 warpReduceSum4(float4 val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
    val.z += __shfl_down(val.z, offset);
    val.w += __shfl_down(val.w, offset);
  }
  return val;
}


texture<float, cudaTextureType3D, cudaReadModeElementType> texRef1;

__global__ void MapGradKern2(const float* crds, const float* wgts, int natoms,
			     float4 ngrid, float4 stagrid,
			     float *grad)
{
  int ithr = (blockIdx.x*blockDim.x + threadIdx.x);
  int iatm = ithr/(64/2);
  int iord = ithr%(64/2);

  int icrd = (iatm<natoms)?iatm*3:0;
  //int icrd = iatm*3;

  float4 crd = make_float4(crds[icrd+0],
			   crds[icrd+1],
			   crds[icrd+2], 1.0f);
  float4 frac = matprod(g_fracMat1, crd);
  frac.x *= ngrid.x;
  frac.y *= ngrid.y;
  frac.z *= ngrid.z;
  frac.x -= stagrid.x;
  frac.y -= stagrid.y;
  frac.z -= stagrid.z;

  float4 fl = make_float4( floor(frac.x),
			   floor(frac.y),
			   floor(frac.z), 1.0f );

  int ib = int(fl.x) - 1;
  int jb = int(fl.y) - 1;
  int kb = int(fl.z) - 1;

  float4 c1 = make_float4(frac.x - fl.x,
			  frac.y - fl.y,
			  frac.z - fl.z, 1.0f);

  float4 c0 = make_float4(1.0f - c1.x,
			  1.0f - c1.y,
			  1.0f - c1.z, 1.0f);

  float4 c10 = make_float4(c1.x*c0.x,
			   c1.y*c0.y,
			   c1.z*c0.z, 1.0f);

  int i, j, k;
  i = iord/16 * 2;
  j = (iord%16)/4;
  k = iord%4;

  float4 co;
  float4 go;
  float rho;
  float4 term;

  co.x = gc(i,0)*c10.x*c0.x + gc(i,1)*c10.x*c1.x + gc(i,2)*c10.x + gc(i,3)*c0.x + gc(i,4)*c1.x;
  co.y = gc(j,0)*c10.y*c0.y + gc(j,1)*c10.y*c1.y + gc(j,2)*c10.y + gc(j,3)*c0.y + gc(j,4)*c1.y;
  co.z = gc(k,0)*c10.z*c0.z + gc(k,1)*c10.z*c1.z + gc(k,2)*c10.z + gc(k,3)*c0.z + gc(k,4)*c1.z;

  go.x = gg(i,0)*c0.x*c0.x + gg(i,1)*c1.x*c1.x + gg(i,2)*c10.x + gg(i,3)*c0.x + gg(i,4)*c1.x;
  go.y = gg(j,0)*c0.y*c0.y + gg(j,1)*c1.y*c1.y + gg(j,2)*c10.y + gg(j,3)*c0.y + gg(j,4)*c1.y;
  go.z = gg(k,0)*c0.z*c0.z + gg(k,1)*c1.z*c1.z + gg(k,2)*c10.z + gg(k,3)*c0.z + gg(k,4)*c1.z;
  
  rho = tex3D(texRef1, ib+i, jb+j, kb+k) * wgts[iatm];

  term.x = go.x * co.y * co.z * rho;
  term.y = co.x * go.y * co.z * rho;
  term.z = co.x * co.y * go.z * rho;
  term.w = co.x * co.y * co.z * rho;

  /////

  i++;

  co.x = gc(i,0)*c10.x*c0.x + gc(i,1)*c10.x*c1.x + gc(i,2)*c10.x + gc(i,3)*c0.x + gc(i,4)*c1.x;
  co.y = gc(j,0)*c10.y*c0.y + gc(j,1)*c10.y*c1.y + gc(j,2)*c10.y + gc(j,3)*c0.y + gc(j,4)*c1.y;
  co.z = gc(k,0)*c10.z*c0.z + gc(k,1)*c10.z*c1.z + gc(k,2)*c10.z + gc(k,3)*c0.z + gc(k,4)*c1.z;

  go.x = gg(i,0)*c0.x*c0.x + gg(i,1)*c1.x*c1.x + gg(i,2)*c10.x + gg(i,3)*c0.x + gg(i,4)*c1.x;
  go.y = gg(j,0)*c0.y*c0.y + gg(j,1)*c1.y*c1.y + gg(j,2)*c10.y + gg(j,3)*c0.y + gg(j,4)*c1.y;
  go.z = gg(k,0)*c0.z*c0.z + gg(k,1)*c1.z*c1.z + gg(k,2)*c10.z + gg(k,3)*c0.z + gg(k,4)*c1.z;
  
  rho = tex3D(texRef1, ib+i, jb+j, kb+k) * wgts[iatm];

  term.x += go.x * co.y * co.z * rho;
  term.y += co.x * go.y * co.z * rho;
  term.z += co.x * co.y * go.z * rho;
  term.w += co.x * co.y * co.z * rho;

  /////

  term = warpReduceSum4(term);

  term.x *= ngrid.x;
  term.y *= ngrid.y;
  term.z *= ngrid.z;

  float4 d1 = matprod_tp(g_fracMat1, term);

  {

    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane==0) {
      if (iatm<natoms) {
	grad[icrd+0] += d1.x;
	grad[icrd+1] += d1.y;
	grad[icrd+2] += d1.z;
      }
      shared[wid]=term.w;
    }

    __syncthreads();              // Wait for all partial reductions
    
    //read from shared memory only if that warp existed
    float val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid==0)
      val = warpReduceSum(val); //Final reduce within first warp

    const int tid = threadIdx.x;
    if (tid == 0) {
      atomicAdd(&grad[natoms*3 + 0], val);
      grad[natoms*3 + blockIdx.x] += val;
    }
  }

}



CuMap2Data::CuMap2Data()
{
  pd_wgts = NULL;
  pd_map = NULL;
}

CuMap2Data::~CuMap2Data()
{
  cleanupCuda();
}

void CuMap2Data::setupCuda(CuComData *pComDat)
{
  m_pComDat = pComDat;

  const int nwgts = m_wgts.size();

  // Weights array (wgts)
  gpuErrChk( cudaMalloc((void**)&pd_wgts, nwgts*sizeof(float)) );
  gpuErrChk( cudaMemcpy( pd_wgts, &m_wgts[0], nwgts*sizeof(float), cudaMemcpyHostToDevice) );
  //for (int i=0; i<ntoth; ++i) {
  //printf("wgts: %d %f\n", i, m_wgts[i]);
  //}

  // 3x3 frac matrix
  cudaMemcpyToSymbol(g_fracMat1, &m_fracMat[0], 9*sizeof(float));

  // spline coefs
  float gcoef[] = {
    -0.5,  0.0, 0, 0, 0,
    0.0,  -1.5, 1, 1, 0,
    -1.5,  0.0, 1, 0, 1,
    0.0,  -0.5, 0, 0, 0,

    0,     0, 1.5, -0.5,  0,
    0,   4.5,   0,    0, -5,
    -4.5,  0,   0,    5,  0,
    0,     0,-1.5,    0,0.5,
  };
  cudaMemcpyToSymbol(g_coefMat2, gcoef, 40*sizeof(float));

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

void CuMap2Data::cleanupCuda()
{
  cudaFree(pd_wgts);
  pd_wgts = NULL;

  cudaUnbindTexture(texRef1); 
  cudaFreeArray((cudaArray_t)pd_map);
  pd_map = NULL;
}

void CuMap2Data::calc()
{
  const int natom = m_nAtoms;
  const int nshmem = m_pComDat->m_nthr*sizeof(float);

  MapGradKern2<<<m_nblk, m_nthr, nshmem>>>
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
