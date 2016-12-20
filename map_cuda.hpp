// -*-Mode: C++;-*-
//
// Common include file for cpp/cu
//

#ifndef MAP_CUDA_HPP_INCLUDED
#define MAP_CUDA_HPP_INCLUDED

class MolData;
class DensityMap;
class CuComData;

class CuMapData
{
public:

  CuMapData();
  virtual ~CuMapData();

  void setup(MolData *pMol, DensityMap *pMap);
  
  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc();

  // computation layout
  int m_nthr, m_nblk;
  int m_nDevAtom;

  int m_nAtoms;

  // map parameters
  int m_ncol, m_nrow, m_nsec;
  int m_na, m_nb, m_nc;
  int m_stcol, m_strow, m_stsec;

  // fractionalization matrix
  std::vector<float> m_fracMat;

  // atom weights
  std::vector<float> m_wgts;

  // map (3D array)
  float *m_map;

  // device memory
  float *pd_wgts;
  float *pd_fracMat;
  void *pd_map;

  // common data
  CuComData *m_pComDat;
};

//////////

class CuMap2Data : public CuMapData
{
public:

  CuMap2Data();
  virtual ~CuMap2Data();

  void setup(MolData *pMol, DensityMap *pMap);

  void setupCuda(CuComData *pComDat);

  void cleanupCuda();
  
  void calc();

  // computation layout
  int m_nTotThr;

  int m_nTermPerThr;
  int m_nThrPerAtom;
};

#endif
