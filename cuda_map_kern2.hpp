
__constant__ float g_fracMat2[9];
__constant__ float g_coefMat2[4*5];

__device__
inline float gc(int i, int j)
{
  return g_coefMat2[j + i*5];
}

texture<float, cudaTextureType3D, cudaReadModeElementType> texRef2;


__global__ void MapGradKern2(const float* crds, const float* wgts,
			     float4 ngrid, float4 stagrid,
			     float *grad, float *eatom)
{
#if 1
  int ithr = (blockIdx.x*blockDim.x + threadIdx.x);
  int iatm = ithr/64;
  int iord = ithr%64;

  /*
  float flag;
  if (iatm<natom)
    flag = 1.0f;
  else
    flag = 0.0f;
  */

  int icrd = iatm*3;

  float4 crd = make_float4(crds[icrd+0], crds[icrd+1], crds[icrd+2], 1.0f);
  float4 frac = matprod(g_fracMat2, crd);
  frac.x *= ngrid.x;
  frac.y *= ngrid.y;
  frac.z *= ngrid.z;
  frac.x -= stagrid.x;
  frac.y -= stagrid.y;
  frac.z -= stagrid.z;

  //fx = -6.417; //crds[icrd+0];
  //fy = -4.130; //crds[icrd+1];
  //fz = 10.414; //crds[icrd+2];

  float4 fl; //x, fly, flz;
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

  float4 co;
  //cx[0] = -0.5f*c1x*c0x*c0x;
  //cx[1] = -1.5f*c0x*c1x*c1x + c0x*c1x + c0x;
  //cx[2] = -1.5f*c1x*c0x*c0x + c0x*c1x + c1x;
  //cx[3] = -0.5f*c0x*c1x*c1x;

  float4 c10 = make_float4(c1.x*c0.x,
			   c1.y*c0.y,
			   c1.z*c0.z, 1.0);

  int i, j, k;
  i = iord/16;
  j = (iord%16)/4;
  k = iord%4;

  co.x = gc(i,0)*c10.x*c0.x + gc(i,1)*c10.x*c1.x + gc(i,2)*c10.x + gc(i,3)*c0.x + gc(i,4)*c1.x;
  co.y = gc(j,0)*c10.y*c0.y + gc(j,1)*c10.y*c1.y + gc(j,2)*c10.y + gc(j,3)*c0.y + gc(j,4)*c1.y;
  co.z = gc(k,0)*c10.z*c0.z + gc(k,1)*c10.z*c1.z + gc(k,2)*c10.z + gc(k,3)*c0.z + gc(k,4)*c1.z;

  /*
  cx[0] = (-0.5)*c10x*c0x + ( 0.0)*c10x*c1x + (0)*c10x + (0)*c0x + (0)*c1x;
  cx[1] = ( 0.0)*c10x*c0x + (-1.5)*c10x*c1x + (1)*c10x + (1)*c0x + (0)*c1x;
  cx[2] = (-1.5)*c10x*c0x + ( 0.0)*c10x*c1x + (1)*c10x + (0)*c0x + (1)*c1x;
  cx[3] = ( 0.0)*c10x*c0x + (-0.5)*c10x*c1x + (0)*c10x + (0)*c0x + (0)*c1x;

  float c0y2 = c0y*c0y;
  float c1y2 = c1y*c1y;
  cy[0] = -0.5f*c1y*c0y2;
  cy[1] = c0y*( -1.5f*c1y2 + c1y + 1.0f );
  cy[2] = c1y*( -1.5f*c0y2 + c0y + 1.0f );
  cy[3] = -0.5*c1y2*c0y;

  float c0z2 = c0z*c0z;
  float c1z2 = c1z*c1z;
  cz[0] = -0.5f*c1z*c0z2;
  cz[1] = c0z*( -1.5f*c1z2 + c1z + 1.0f );
  cz[2] = c1z*( -1.5f*c0z2 + c0z + 1.0f );
  cz[3] = -0.5*c1z2*c0z;
  */
  /*
  float gx[4], gy[4], gz[4];
  gx[0] =  c0x*( 1.5f*c1x - 0.5f );
  gx[1] =  c1x*( 4.5f*c1x - 5.0f );
  gx[2] = -c0x*( 4.5f*c0x - 5.0f );
  gx[3] = -c1x*( 1.5f*c0x - 0.5f );

  gy[0] =  c0y*( 1.5f*c1y - 0.5f );
  gy[1] =  c1y*( 4.5f*c1y - 5.0f );
  gy[2] = -c0y*( 4.5f*c0y - 5.0f );
  gy[3] = -c1y*( 1.5f*c0y - 0.5f );

  gz[0] =  c0z*( 1.5f*c1z - 0.5f );
  gz[1] =  c1z*( 4.5f*c1z - 5.0f );
  gz[2] = -c0z*( 4.5f*c0z - 5.0f );
  gz[3] = -c1z*( 1.5f*c0z - 0.5f );
  */
  float rho;
  int ib = int(fl.x) - 1;
  int jb = int(fl.y) - 1;
  int kb = int(fl.z) - 1;

  extern __shared__ float sdata[];
  
  const int tid = threadIdx.x;

  rho = tex3D(texRef2, ib+i, jb+j, kb+k);
  sdata[tid] = co.x * co.y * co.z * rho * wgts[iatm];

  //eatom[ithr] = cx[i] * cy[j] * cz[k] * rho * wgts[iatm];

  __syncthreads();

  unsigned int s;
  for (s=blockDim.x/2; s>0; s>>=1) { 
    if (tid < s) { 
      sdata[tid] += sdata[tid + s]; 
    } 
    __syncthreads();
  }

  if (tid == 0)
    eatom[blockIdx.x] = sdata[0];

  /*
  float su = 0.0f;
  for ( i = 0; i < 4; i++ ) {
    float sv = 0.0;
    for ( j = 0; j < 4; j++ ) {
      float sw = 0.0;
      for ( k = 0; k < 4; k++ ) {
	//sw += co[k].z() * getValue();
	rho = tex3D(texRef2, ib+i, jb+j, kb+k);
	sw += cz[k] * 
      }
      sv += cy[j] * sw;
      }
    su += cx[i] * sv;
  }
  */
  /*
  grad[icrd+0] = 0.0f;
  grad[icrd+1] = 0.0f;
  grad[icrd+2] = 0.0f;
  */
  /*
  float s1, s2, s3, du1, dv1, dv2, dw1, dw2, dw3;
  s1 = du1 = dv1 = dw1 = 0.0;
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
    s1 += cx[i] * s2;
    du1 += gx[i] * s2;
    dv1 += cx[i] * dv2;
    dw1 += cx[i] * dw2;
  }
  grad[icrd+0] = du1;
  grad[icrd+1] = dv1;
  grad[icrd+2] = dw1;
  */

}
#endif

