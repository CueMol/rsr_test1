// -*-Mode: C++;-*-

__global__ void BondGradKern2(const float* crds, const CuBond* param,
			      const int *bvec,
			      float *grad, float *eatom)
{
  int iterm = blockIdx.x*blockDim.x + threadIdx.x;

  int i, ib, ai, aj;
  float dx, dy, dz;
  float gx,  gy, gz, gw;
  float fdir;

  ib = bvec[iterm*2];

  if (ib>0) {
    ib = ib-1;
    fdir = 1.0;
  }
  else if (ib<0) {
    ib = -ib-1;
    fdir = -1.0;
  }
  else {
    ib = 0;
    fdir = 0.0;
  }

  const CuBond *pbon = &param[ib];
    
  ai = pbon->ai;
  aj = pbon->aj;

  dx = crds[ai+0] - crds[aj+0];
  dy = crds[ai+1] - crds[aj+1];
  dz = crds[ai+2] - crds[aj+2];

  float sqlen = dx*dx + dy*dy + dz*dz;
  float len = sqrt(sqlen);

  float ss = (len - pbon->r0) * fdir;

  //float con = 2.0f * pbon->kf * (1.0f - pbon->r0/len) * fdir;
  float con = 2.0f * pbon->kf * ss /len ;

  gx = con*dx;
  gy = con*dy;
  gz = con*dz;
  gw = pbon->kf * ss * ss * 0.5;

  extern __shared__ float sdata[];

  const int tid = threadIdx.x;
  const int tid4 = tid*4;
  sdata[tid4+0] = gx;
  sdata[tid4+1] = gy;
  sdata[tid4+2] = gz;
  sdata[tid4+3] = gw;

  __syncthreads();

  unsigned int s;
  for (s=blockDim.x/2; s>0; s>>=1) { 
    if (tid < s) { 
      sdata[tid4+3] += sdata[(tid + s)*4+3]; 
    } 
    __syncthreads();
  }

  if (tid == 0)
    eatom[blockIdx.x] = sdata[3];

  int iatm = bvec[iterm*2+1];

  if (iatm>0) {
    int s;
    for (s=1; ; ++s) {
      if (bvec[(iterm+s)*2+1]>0)
	break;
      sdata[tid4+0] += sdata[(tid+s)*4+0];
      sdata[tid4+1] += sdata[(tid+s)*4+1];
      sdata[tid4+2] += sdata[(tid+s)*4+2];
    }

    const int icrd = (iatm-1) * 3;
    grad[icrd + 0] = sdata[tid4+0];
    grad[icrd + 1] = sdata[tid4+1];
    grad[icrd + 2] = sdata[tid4+2];
  }



}
