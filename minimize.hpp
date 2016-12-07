#ifndef MINIMIZE_HPP_INCLUDED
#define MINIMIZE_HPP_INCLUDED

class MolData;
class DensityMap;

typedef double realnum_t;

#include "RamaPlotData.hpp"

class MiniTarg
{
public:
  MiniTarg();

  bool m_bBond;
  bool m_bAngl;
  bool m_bDihe;
  bool m_bChir;
  bool m_bPlan;
  bool m_bRama;
  bool m_bMap;
  bool m_bNonb;

  MolData *m_pMol;
  DensityMap *m_pMap;

  virtual void setup(MolData *pMol, DensityMap *pMap) =0;

  realnum_t m_energy;
  realnum_t m_Edihe, m_Eangl, m_Ebond, m_Emap;
  realnum_t m_Echir, m_Eplan, m_Erama, m_Enonb;

  std::vector<float> m_grad;

  virtual const std::vector<float> &calc(float &eng) =0;

  RamaPlotData m_ramaplot;

};

class MiniTargCPU : public MiniTarg
{
public:
  virtual void setup(MolData *pMol, DensityMap *pMap);

  virtual const std::vector<float> &calc(float &eng);

  const std::vector<float> &calcFce();
  realnum_t calcEng();

  void calcMapEng();
  void calcMapFce();

  void calcBondEng();
  void calcBondFce();

  void calcAnglEng();
  void calcAnglFce();

  void calcDiheEng();
  void calcDiheFce();
  void calcDiheEng2();
  void calcDiheFce2();

  void calcChirEng();
  void calcChirFce();

  void calcPlanEng();
  void calcPlanFce();

  void calcRamaEng();
  void calcRamaFce();
};

#ifdef HAVE_CUDA
class CuComData;
class CuBondData;
class CuAnglData;
//struct CuMapData;

class MiniTargCUDA : public MiniTargCPU
{
public:
  typedef MiniTargCPU super_t;

  CuComData *m_pComData;
  CuBondData *m_pBondData;
  CuAnglData *m_pAnglData;
  //CuMapData *m_pMapData;

  std::vector<float> m_gradtmp;

  MiniTargCUDA() : super_t(), m_pComData(NULL), m_pBondData(NULL), m_pAnglData(NULL)
  {
  }

  virtual ~MiniTargCUDA();

  virtual void setup(MolData *pMol, DensityMap *pMap);

  virtual const std::vector<float> &calc(float &eng);

  void cleanup();

  // void calcMap();
  void calcBond();
  void calcAngl();
};
#endif

class Minimize
{
public:
  MiniTarg *m_pMiniTarg;

  int m_nMaxIter;

  bool m_bUseCUDA;

  virtual void setup();
  virtual void minimize() =0;
};

class MinLBFGS : public Minimize
{
public:
  //virtual void setup(MolData *pMol, DensityMap *pMap);
  virtual void minimize();
};

#ifdef HAVE_GSL
class MinGSL : public Minimize
{
public:
  //virtual void setup(MolData *pMol, DensityMap *pMap);
  virtual void minimize();
};
#endif

#endif
