// -*-Mode: C++;-*-

#ifndef UTILITY_CU_INCLUDED
#define UTILITY_CU_INCLUDED

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float  blockReduceSum(float val)
{
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__inline__ __device__
float mat3(float *pmat, int i, int j)
{
  return pmat[i + j*3];
}

__inline__ __device__
float4 matprod(float *pmat, float4 in)
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

__inline__ __device__
float4 matprod_tp(float *pmat, float4 in)
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

template <typename _ValueType=float>
class Matrix3
{
public:
  typedef _ValueType value_type;
  value_type m_value[3*3];

  __device__ Matrix3() {}

  /// Element access (mutating)
  __device__ inline value_type &aij(int i, int j) {
    return m_value[(i-1) + (j-1)*3];
  }
  
  /// Element access (const)
  __device__ inline value_type aij(int i, int j) const {
    return m_value[(i-1) + (j-1)*3];
  }
  
  __device__ inline void setZero()
  {
    m_value[0] = 0;
    m_value[1] = 0;
    m_value[2] = 0;

    m_value[3] = 0;
    m_value[4] = 0;
    m_value[5] = 0;

    m_value[6] = 0;
    m_value[7] = 0;
    m_value[8] = 0;
  }

  __device__ inline void setUnit()
  {
    m_value[0] = 1;
    m_value[1] = 0;
    m_value[2] = 0;

    m_value[3] = 0;
    m_value[4] = 1;
    m_value[5] = 0;

    m_value[6] = 0;
    m_value[7] = 0;
    m_value[8] = 1;
  }

  __device__ inline void scaleSelf(value_type arg)
  {
    for (int i=0; i<3*3; ++i)
      m_value[i] *= arg;
  }

  __device__ inline float deter() const
  {
    return (aij(1,1)*aij(2,2)-aij(1,2)*aij(2,1))*aij(3,3) +
      (aij(2,1)*aij(3,2)-aij(2,2)*aij(3,1))*aij(1,3) +
      (aij(3,1)*aij(1,2)-aij(3,2)*aij(1,1))*aij(2,3);
  }

  __device__ inline void sub(const Matrix3 &arg)
  {
    for (int i=0; i<3*3; ++i)
      m_value[i] -=  arg.m_value[i];
  }
};

template <typename T>
__device__ T sqr(T val) { return val*val; }

__device__ __inline__
float4 mkevec(const Matrix3<float> &mat, float l1, float l2)
{
  float4 v1;
  v1.x = (mat.aij(1,1)-l1)*(mat.aij(1,1)-l2) + mat.aij(1,2)*mat.aij(2,1) + mat.aij(1,3)*mat.aij(3,1);
  v1.y = (mat.aij(1,1)-l1)*mat.aij(1,2) + mat.aij(1,2)*(mat.aij(2,2)-l2) + mat.aij(1,3)*mat.aij(3,2);
  v1.z = (mat.aij(1,1)-l1)*mat.aij(1,3) + mat.aij(1,2)*mat.aij(2,3) + mat.aij(1,3)*(mat.aij(3,3)-l2);
    
  double len1 = sqrt( sqr(v1.x) + sqr(v1.y) + sqr(v1.z) );
  v1.x /= len1;
  v1.y /= len1;
  v1.z /= len1;

  return v1;
}

__device__ __inline__
float4 mat33_diag(const Matrix3<float> &mat, float4 &evals)
{
  float p1 = sqr( mat.aij(1,2) ) + sqr( mat.aij(1,3) ) + sqr( mat.aij(2,3) );

  Matrix3<float> I;
  I.setZero();

  float q = (mat.aij(1,1) + mat.aij(2,2) + mat.aij(3,3))/3.0;
  float p2 = sqr(mat.aij(1,1) - q) + sqr(mat.aij(2,2) - q) + sqr(mat.aij(3,3) - q) + 2.0 * p1;
  float p = sqrt(p2 / 6.0);
  
  //B = (1 / p) * (A - q * I); // I is the identity matrix
  I.aij(1,1) = q;
  I.aij(2,2) = q;
  I.aij(3,3) = q;
  Matrix3<float> B(mat);
  B.sub(I);
  B.scaleSelf(1.0/p);

  float r = B.deter() / 2.0;

  // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
  // but computation error can leave it slightly outside this range.
  float phi;
  if (r <= -1.0)
    phi = M_PI / 3.0;
  else if (r >= 1.0)
    phi = 0.0;
  else
    phi = acos(r) / 3.0;
  
  // the eigenvalues satisfy eig3 <= eig2 <= eig1
  evals.x = q + 2.0 * p * cos(phi);
  evals.y = q + 2.0 * p * cos(phi + (2.0*M_PI/3));
  evals.z = 3.0 * q - evals.x - evals.y;     // since trace(A) = eig1 + eig2 + eig3
  
  //printf("evals.x: %f\n", evals.x());
  //printf("evals.y: %f\n", evals.y());
  //printf("evals.z: %f\n", evals.z());
  
  if (evals.x<=evals.y &&
      evals.x<=evals.z)
    return mkevec(mat, evals.y, evals.z);
  else if (evals.y<=evals.x &&
	   evals.y<=evals.z)
    return mkevec(mat, evals.z, evals.x);
  else
    return mkevec(mat, evals.x, evals.y);
  
}

#endif
