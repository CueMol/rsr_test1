// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef NONB_CUDA_HPP_INCLUDED
#define NONB_CUDA_HPP_INCLUDED

class MolData;
class CuComData;


struct CuNonb
{

#if !defined(__CUDACC__)
  CuNonb() : idx(-1), r0(0.0), wgt(0.0) {}
  CuNonb(int ai, float ar0, float aw) : idx(ai), r0(ar0), wgt(aw) {}
#endif

  int idx;
  float r0;
  float wgt;
};

/*
struct CuNonb
{
  CuNonb() : idx(-1) {}
  CuNonb(int ai, float ar0, float aw) : idx(ai) {}

  int idx;
};
*/

class CuNonbData
{
public:

  CuNonbData();
  virtual ~CuNonbData();

  void setup(MolData *pMol);

  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc();

  // common data
  CuComData *m_pComDat;

  int m_nAtoms;

  /// nonb intr calc per atom
  int m_nonb_max;
  /// loop per thread
  int m_nloop;

  /// size of m_mat
  int m_nMatSize;

  int m_nTotalThr, m_nblk, m_nthr;

  //std::vector<CuNonb> m_mat;
  //CuNonb *pd_nonb;

  std::vector<int> m_indmat;
  int *pd_indmat;

  std::vector<int> m_prmmat;
  int *pd_prmmat;

};

#endif
