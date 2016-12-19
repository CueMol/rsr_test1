#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>
#include <qlib/LineStream.hpp>
#include <qlib/PrintStream.hpp>
#include <qlib/FileStream.hpp>

//#include <gsl/gsl_multimin.h>
//#include <gsl/gsl_blas.h>

#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"

using namespace std;
using qlib::LString;

#if 1
int main(int argc, char* argv[])
{
  if (!(argc==3||argc==4)) {
    printf("usage: %s <parm> <pdb> <map>\n", argv[0]);
    return 0;
  }

  qlib::init();
  
  string parm = argv[1];
  string pdbin = argv[2];
  LString mapin;
  DensityMap *pMap = NULL;

  if (argc==4) {
    mapin = argv[3];
    pMap = new DensityMap;
    pMap->loadCNSMap(mapin);
    pMap->m_dScale = 1.0;
  }

  float dum;

  MolData *pMol = new MolData;
  pMol->loadparm(parm);
  pMol->loadPDB(pdbin);
  pMol->buildNonbData();

  //pMol->addRand(0.5);
    
  Minimize *pMin = new MinLBFGS;
  //Minimize *pMin = new MinGSL;

  pMin->m_bUseCUDA = true;
  //pMin->m_bUseCUDA = false;
  pMin->setup();
  
  //pMin->m_pMiniTarg->m_bBond = true;
  //pMin->m_pMiniTarg->m_bAngl = true;
  //pMin->m_pMiniTarg->m_bChir = true;
  //pMin->m_pMiniTarg->m_bPlan = true;
  pMin->m_pMiniTarg->m_bNonb = true;
  //pMin->m_pMiniTarg->m_bMap = true;
  pMin->m_pMiniTarg->setup(pMol, pMap);

  pMin->m_nMaxIter = 10000;

  //pMin->minimize();
  //pMol->savePDB("outout.pdb");

  {
    const std::vector<float> *pgrad;
    
    //for (int i=0; i<1000; ++i) {
    {
      const std::vector<float> &grad = pMin->m_pMiniTarg->calc(dum);
      pgrad = &grad;
    }

    for (int i=0; i<qlib::min<int>(pgrad->size()/3,20); ++i) {
      printf("grad %d: %f %f %f\n", i, (*pgrad)[i*3+0], (*pgrad)[i*3+1], (*pgrad)[i*3+2]);
    }
    printf("Etotal=%f\n", dum);

    }

}
#endif

#if 0

int main(int argc, char* argv[])
{
  if (argc!=4) {
    printf("usage: %s <parm> <pdb> <map>\n", argv[0]);
    return 0;
  }

  qlib::init();
  
  string parm = argv[1];
  string pdbin = argv[2];
  LString mapin = argv[3];

  float dum;
  DensityMap *pMap = new DensityMap;
  pMap->loadCNSMap(mapin);
  pMap->m_dScale = 1.0;
  //for (;;) {
  {
    //RamaPlotData rpd;
    //rpd.setup();
    //rpd.dump();

    MolData *pMol = new MolData;
    pMol->loadparm(parm);
    pMol->loadPDB(pdbin);
    //pMol->addRand(0.1);
    
    Minimize *pMin = new MinLBFGS;
    //Minimize *pMin = new MinGSL;

    pMin->m_bUseCUDA = false;
    pMin->setup();
    
    pMin->m_pMiniTarg->m_bBond = true;
    pMin->m_pMiniTarg->m_bAngl = true;
    pMin->m_pMiniTarg->m_bDihe = true;
    pMin->m_pMiniTarg->m_bChir = true;
    pMin->m_pMiniTarg->m_bPlan = true;
    //pMin->m_pMiniTarg->m_bRama = true;
    //pMin->m_pMiniTarg->m_bMap = true;

    pMin->m_pMiniTarg->setup(pMol, pMap);
    pMin->m_nMaxIter = 10000;

    pMin->minimize();
    //std::vector<float> grad = pMin->m_pMiniTarg->calc(dum);

    // for (int i=0; i<grad.size(); ++i)
    // pMol->m_crds[i] += grad[i];

    pMol->savePDB("outout.pdb");
    /*
    std::vector<float> grad = pMin->m_pMiniTarg->calc(dum);

    double gnorm = 0.0;
    for (int i=0; i<grad.size(); ++i)
      gnorm += grad[i]*grad[i];
    gnorm = sqrt(gnorm);
    printf("gnorm = %f\n", gnorm);
    for (int i=0; i<grad.size(); ++i)
      grad[i] /= gnorm;

    double step = 0.0001;
    int nstep = 100;
    for (int i=0; i<grad.size(); ++i)
      pMol->m_crds[i] += grad[i]*step*nstep/2.0;

    for (int j=0; j<nstep; ++j) {
      for (int i=0; i<grad.size(); ++i)
	pMol->m_crds[i] -= grad[i]*step;

      pMin->m_pMiniTarg->calc(dum);
      printf("%d %f\n", j-nstep/2, dum);
    }
    */
  }

  /*{
    MolData *pMol = new MolData;
    pMol->loadparm(parm);
    pMol->loadPDB(pdbin);

    Minimize *pMin = new MinLBFGS;
    pMin->m_bUseCUDA = true;
    //Minimize *pMin = new MinGSL;
    pMin->setup(pMol, pMap);
    pMin->m_nMaxIter = 100;
    //pMin->m_pMiniTarg->m_bBond = false;
    pMin->minimize();
    //pMin->m_pMiniTarg->calc(dum);
    pMol->savePDB("outout.pdb");
    }*/


  return 0;
}

#endif
