// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef CUDA_PLAN_HPP_INCLUDED
#define CUDA_PLAN_HPP_INCLUDED

struct CuPlan
{
  int natom;
  int istart;
  float wgt;
};

class MolData;
class CuComData;

class CuPlanData
{
public:

  CuPlanData();
  virtual ~CuPlanData();

  void setup(MolData *pMol);

  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc();

  std::vector<CuPlan> m_cuplans;
  std::vector<int> m_cdix;

  // computation layout
  int m_nthr, m_nblk;
  int m_nDevPlan;
  int m_nAtoms;

  // device memory
  CuPlan *pd_plan;
  int *pd_cdix;

  // common data
  CuComData *m_pComDat;
};

#define PLAN2_NTHR 16

class CuPlan2Data : public CuPlanData
{
public:

  //CuPlan2Data();
  //virtual ~CuPlan2Data();

  void setup(MolData *pMol);

  void setupCuda(CuComData *pComDat);

  //void cleanupCuda();
  
  void calc();
};

#endif
