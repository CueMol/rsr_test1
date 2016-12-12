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
  void calc2(float *val, std::vector<float> &grad);

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
};

#if 0
struct CuBondData2
{

  CuBondData2() : pd_crds(NULL), pd_bond(NULL), pd_bvec(NULL), pd_grad(NULL), pd_eatm(NULL)
  {
  }

  int nthr, nblk;

  std::vector<int> bvec;
  std::vector<CuBond> cubonds;

  // device memory
  float *pd_crds;
  CuBond *pd_bond;
  int *pd_bvec;
  float *pd_grad;
  float *pd_eatm;

};

  
void cudaBond_fdf(const std::vector<float> &crds,
		  CuBondData *pDat,
		  float *val, std::vector<float> &grad);

void cudaBond_fdf2(const std::vector<float> &crds,
		   CuBondData2 *pDat,
		   float *val, std::vector<float> &grad);

#endif

#endif
