// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef CUDA_ANGL_HPP_INCLUDED
#define CUDA_ANGL_HPP_INCLUDED

struct CuAngl
{
  int ai;
  int aj;
  int ak;

  float kf;
  float r0;
};

class MolData;
class CuComData;

class CuAnglData
{
public:

  CuAnglData();
  virtual ~CuAnglData();

  void setup(MolData *pMol);

  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc(float *val, std::vector<float> &grad);

  std::vector<CuAngl> m_cuangls;

  // computation layout (1st stage)
  int m_nthr, m_nblk;
  int m_nDevAngl;
  int m_nAtoms;

  // device memory
  CuAngl *pd_angl;
  float *pd_eatm;

  // common data
  CuComData *m_pComDat;
};

#endif
