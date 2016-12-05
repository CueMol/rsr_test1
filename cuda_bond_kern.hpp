// -*-Mode: C++;-*-

//texture<int, cudaTextureType2D, cudaReadModeElementType> texRef1;

__global__ void BondGradKern(const float* crds, const CuBond* param,
			     const int *bvec, int nacc,
			     float *grad, float *eatom)
{
  int iatm = blockIdx.x*blockDim.x + threadIdx.x;
  //int iatm = threadIdx.x;

  int i, ib, ai, aj;
  float dx, dy, dz;
  float gx = 0.0f , gy = 0.0f, gz = 0.0f, gw = 0.0f;
  float fdir, efac;

  // int ibsum = 0;

  for (i=0; i<nacc; ++i) {
    //{ i=0;
    ib = bvec[nacc*iatm + i];
    //int ind = nacc*iatm + i;
    //ib = tex2D(texRef1, ind%32768, ind/32768);

    //ib = ib-1;
    //fdir = 1.0;

    if (ib>0) {
      ib = ib-1;
      fdir = 1.0;
      efac = 1.0f;
    }
    else if (ib<0) {
      ib = -ib-1;
      fdir = -1.0;
      efac = 0.0f;
    }
    else {
      continue;
    }

    const CuBond *pbon = &param[ib];
    
    ai = pbon->ai;
    aj = pbon->aj;

    // ibsum += ai;

    dx = crds[ai+0] - crds[aj+0];
    dy = crds[ai+1] - crds[aj+1];
    dz = crds[ai+2] - crds[aj+2];

    float sqlen = dx*dx + dy*dy + dz*dz;
    //float sqlen = __fmul_rn(dx,dx) + __fmul_rn(dy,dy) + __fmul_rn(dz,dz);
    float len = sqrt(sqlen);

    float con = 2.0f * pbon->kf * (1.0f - pbon->r0/len) * fdir;
    float ss = (len - pbon->r0) * fdir;

    gx += con*dx;
    gy += con*dy;
    gz += con*dz;
    gw += pbon->kf * ss * ss * efac; //0.5f;
    //if (gw==0.0f)
    //gw = sqlen*efac;

    /*
    if (fdir) {
    }
    else {
      gx -= con*dx;
      gy -= con*dy;
      gz -= con*dz;
    }
    */

  }

  //const int icrd = iatm * 4;
  const int icrd = iatm * 3;

  //grad[icrd + 0] = float( tex1Dfetch(texRef1, 0) );
  grad[icrd + 0] = gx;
  grad[icrd + 1] = gy;
  grad[icrd + 2] = gz;
  //grad[icrd + 3] = gw;

  extern __shared__ float sdata[];
  const int tid = threadIdx.x;
  sdata[tid] = gw;

  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2)  {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    eatom[blockIdx.x] = sdata[0];
}
