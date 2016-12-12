// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef CUDA_COM_HPP_INCLUDED
#define CUDA_COM_HPP_INCLUDED

#if defined(__CUDACC__)
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
#endif

#define THR_PER_BLK 1024

#define GRAD_PAD 32

/// setup exec layout
inline void
calcThrBlk(int nelem, int *nthr, int *nblk, int *ntotal)
{
  if (nelem<THR_PER_BLK) {
    *nblk = 1;
    *nthr = THR_PER_BLK;
  }
  else if (nelem%THR_PER_BLK==0) {
    *nblk = nelem/THR_PER_BLK;
    *nthr = THR_PER_BLK;
  }
  else {
    *nblk = nelem/THR_PER_BLK +1;
    *nthr = THR_PER_BLK;
  }

  *ntotal = *nblk * *nthr;
}

#define EATM_MAX 256

/// Common data
class CuComData
{
public:
  CuComData() : pd_crds(NULL), pd_grad(NULL), pd_eatm(NULL), m_nthr(0), m_nblk(0), m_nDevAtom(0)
  {
  }

  virtual ~CuComData();

  void cleanup();

  /// Allocate device memory
  void setup(int natom);

  /// Transfer coodinate array
  void xferCrds(const std::vector<float> &crds);

  void resetGrad();

  void xferGrad(std::vector<float> &grad) const;

  float getEnergy();

  //////////

  std::vector<float> m_eatm;

  /// size of the computation (for the coordinate/grad calculation phase)
  int m_nthr, m_nblk;

  /// size of the system (= m_nthr * m_nblk)
  int m_nDevAtom;

  int m_nAtom;

  /// Device array for coordinates (invariant, common)
  float *pd_crds;

  /// Device array for gradient vector
  float *pd_grad;
  
  /// Device array for energy vector
  float *pd_eatm;
};

#endif
