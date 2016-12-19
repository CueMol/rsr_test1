// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef CUDA_BOND_HPP_INCLUDED
#define CUDA_BOND_HPP_INCLUDED

struct CuBond
{
  int ai;
  //int ai_ord;
  int aj;
  //int aj_ord;
  float kf;
  float r0;
};

class MolData;
class CuComData;

class CuBondData
{
public:

  CuBondData();
  virtual ~CuBondData();

  void setup(MolData *pMol);

  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc(float *val, std::vector<float> &grad);

  std::vector<CuBond> m_cubonds;
  std::vector<int> m_bind;
  std::vector<int> m_bvec;

  // computation layout (1st stage)
  int m_nthr, m_nblk;
  int m_nDevBond;
  int m_nAtoms;

  // device memory
  CuBond *pd_bond;
  float *pd_bondres;
  int *pd_bind;
  int *pd_bvec;
  float *pd_eatm;

  // common data
  CuComData *m_pComDat;

  //////////

  void calc2();

  void setup2(MolData *pMol);

  void setupCuda2(CuComData *pComDat);
};


#endif
