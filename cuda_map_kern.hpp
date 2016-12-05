
__constant__ float g_fracMat1[9];

texture<float, cudaTextureType3D, cudaReadModeElementType> texRef1;

__device__
inline float mat3(float *pmat, int i, int j)
{
  return pmat[i + j*3];
}

__device__
inline float4 matprod(float *pmat, float4 in)
{
  return make_float4(mat3(pmat, 0, 0) * in.x + 
		     mat3(pmat, 0, 1) * in.y +
		     mat3(pmat, 0, 2) * in.z,
		     mat3(pmat, 1, 0) * in.x + 
		     mat3(pmat, 1, 1) * in.y +
		     mat3(pmat, 1, 2) * in.z,
		     mat3(pmat, 2, 0) * in.x + 
		     mat3(pmat, 2, 1) * in.y +
		     mat3(pmat, 2, 2) * in.z, 1.0f);
}

__device__
inline float4 matprod_tp(float *pmat, float4 in)
{
  return make_float4(mat3(pmat, 0, 0) * in.x + 
		     mat3(pmat, 1, 0) * in.y +
		     mat3(pmat, 2, 0) * in.z,
		     mat3(pmat, 0, 1) * in.x + 
		     mat3(pmat, 1, 1) * in.y +
		     mat3(pmat, 2, 1) * in.z,
		     mat3(pmat, 0, 2) * in.x + 
		     mat3(pmat, 1, 2) * in.y +
		     mat3(pmat, 2, 2) * in.z, 1.0f);
}

__global__ void MapGradKern1(const float* crds, const float* wgts,
			     float4 ngrid, float4 stagrid,
			     float *grad, float *eatom)
{
  int iatm = blockIdx.x*blockDim.x + threadIdx.x;
  int icrd = iatm*3;

  float4 crd = make_float4(crds[icrd+0], crds[icrd+1], crds[icrd+2], 1.0f);
  float4 frac = matprod(g_fracMat1, crd);
  frac.x *= ngrid.x;
  frac.y *= ngrid.y;
  frac.z *= ngrid.z;
  frac.x -= stagrid.x;
  frac.y -= stagrid.y;
  frac.z -= stagrid.z;

  /*
  float cx, cy, cz;
  cx = crds[icrd+0];
  cy = crds[icrd+1];
  cz = crds[icrd+2];
  float fx, fy, fz;
  fx = crds[icrd+0];
  fy = crds[icrd+1];
  fz = crds[icrd+2];
  */

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

  /*
  float su = 0.0f;
  for ( i = 0; i < 4; i++ ) {
    float sv = 0.0;
    for ( j = 0; j < 4; j++ ) {
      float sw = 0.0;
      for ( k = 0; k < 4; k++ ) {
	//sw += co[k].z() * getValue();
	rho = tex3D(texRef1, ib+i, jb+j, kb+k);
	sw += cz[k] * rho;
      }
      sv += cy[j] * sw;
      }
    su += cx[i] * sv;
  }
  */

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

  d1.x *= wgts[iatm];
  d1.y *= wgts[iatm];
  d1.z *= wgts[iatm];

  //icrd = iatm*4;
  grad[icrd+0] = d1.x;
  grad[icrd+1] = d1.y;
  grad[icrd+2] = d1.z;
  //grad[icrd+3] = s1*wgts[iatm];

  extern __shared__ float sdata[];
  const int tid = threadIdx.x;

  sdata[tid] = s1 * wgts[iatm];

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
  //grad[icrd+0] = su * wgts[iatm];
  grad[icrd+0] = 0.0f;
  grad[icrd+1] = 0.0f;
  grad[icrd+2] = 0.0f;
  */

}

