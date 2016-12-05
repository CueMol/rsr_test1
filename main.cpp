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

int main(int argc, char* argv[])
{
  if (argc!=4) {
    printf("usage: %s <parm> <pdb> <map>\n", argv[0]);
    return 0;
  }

  string parm = argv[1];
  string pdbin = argv[2];
  LString mapin = argv[3];

  float dum;
  DensityMap *pMap = new DensityMap;
  pMap->loadCNSMap(mapin);
  pMap->m_dScale = 1.0;
  {
    //RamaPlotData rpd;
    //rpd.setup();
    //rpd.dump();

    MolData *pMol = new MolData;
    pMol->loadparm(parm);
    pMol->loadPDB(pdbin);
    //pMol->addRand(0.1);
    
    //Minimize *pMin = new MinLBFGS;
    Minimize *pMin = new MinGSL;

    pMin->m_bUseCUDA = false;
    pMin->setup(pMol, pMap);
    pMin->m_nMaxIter = 10000;
    
    pMin->m_pMiniTarg->m_bBond = true;
    pMin->m_pMiniTarg->m_bAngl = true;
    pMin->m_pMiniTarg->m_bDihe = true;
    pMin->m_pMiniTarg->m_bChir = true;
    pMin->m_pMiniTarg->m_bPlan = true;
    //pMin->m_pMiniTarg->m_bRama = true;
    //pMin->m_pMiniTarg->m_bMap = true;

    //pMin->minimize();
    std::vector<float> grad = pMin->m_pMiniTarg->calc(dum);

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


#if 0
int main(int argc, char* argv[])
{
  if (argc!=4) {
    printf("usage: %s <parm> <pdb> <map>\n", argv[0]);
    return 0;
  }

  string parm = argv[1];
  string pdbin = argv[2];
  LString mapin = argv[3];

  MolData *pMol = new MolData;

  pMol->loadparm(parm);

  pMol->loadPDB(pdbin);

  std::vector<float> grad(pMol->m_nCrds);

  /*
  //CudaData2 *pDat = prepBondCuda2(pMol);
  CudaData *pDat = prepBondCuda(pMol);
  for (int i=0; i<100; i++) {
  //for (int i=0; i<1000*100; i++) {
    //{
    //gradBondCuda2(pMol, pDat, grad);
    gradBondCuda(pMol, pDat, grad);
    //gradBond(pMol, grad);
  }
  */


  DensityMap *pMap = new DensityMap;
  pMap->loadCNSMap(mapin);

  CudaMapData *pDat = prepMapCuda1(pMap, pMol);
  //for (int i=0; i<1000*5; i++) {
  //for (int i=0; i<1000*10; i++) {
  {
    gradMapCuda1(pMol, pMap, pDat, grad);
    //gradMap(pMol, pMap, grad);
  }

  return 0;
}
#endif

#if 0
bool gbBond = true;
bool gbAngl = true;
DensityMap *gpMap = NULL;

void calc_gradE(MolData *pMol, std::vector<float> &grad)
{
  int i;
  int nbond = pMol->m_nBonds;
  int nangl = pMol->m_nAngls;
  int ncrd = pMol->m_nAtoms * 3;

  //printf("target df nbond=%d, ncrd=%d\n", nbond, ncrd);

  for (i=0; i<ncrd; ++i)
    grad[i] = 0.0f;

  if (gbBond)
  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    float dx = pMol->m_crds[atom_i*3+0] - pMol->m_crds[atom_j*3+0];
    float dy = pMol->m_crds[atom_i*3+1] - pMol->m_crds[atom_j*3+1];
    float dz = pMol->m_crds[atom_i*3+2] - pMol->m_crds[atom_j*3+2];

    float sqlen = dx*dx + dy*dy + dz*dz;
    float len = sqrt(sqlen);

    float con = 2.0f * pMol->m_bonds[i].kf * (1.0f - pMol->m_bonds[i].r0/len);

    grad[atom_i*3+0] += con * dx;
    grad[atom_i*3+1] += con * dy;
    grad[atom_i*3+2] += con * dz;

    grad[atom_j*3+0] -= con * dx;
    grad[atom_j*3+1] -= con * dy;
    grad[atom_j*3+2] -= con * dz;

    if (atom_i==0 && atom_j==4) {
      //      printf("atom %d-%d dv=%f,%f,%f", atom_i, atom_j, dx, dy,dz);
      //      printf("  sqlen=%f, dr=%f\n", len, len-pMol->m_bonds[i].r0);
    }
  }


  if (gbAngl)
  for (i=0; i<nangl; ++i) {
    int ai = pMol->m_angls[i].atom_i;
    int aj = pMol->m_angls[i].atom_j;
    int ak = pMol->m_angls[i].atom_k;

    float rijx, rijy, rijz;
    float rkjx, rkjy, rkjz;
    float Rij, Rkj;

    rijx = pMol->m_crds[ai*3+0] - pMol->m_crds[aj*3+0];
    rijy = pMol->m_crds[ai*3+1] - pMol->m_crds[aj*3+1];
    rijz = pMol->m_crds[ai*3+2] - pMol->m_crds[aj*3+2];

    rkjx = pMol->m_crds[ak*3+0] - pMol->m_crds[aj*3+0];
    rkjy = pMol->m_crds[ak*3+1] - pMol->m_crds[aj*3+1];
    rkjz = pMol->m_crds[ak*3+2] - pMol->m_crds[aj*3+2];

    // distance
    Rij = sqrt(qlib::max<float>(F_EPS16, rijx*rijx + rijy*rijy + rijz*rijz));
    Rkj = sqrt(qlib::max<float>(F_EPS16, rkjx*rkjx + rkjy*rkjy + rkjz*rkjz));

    // normalization
    float eijx, eijy, eijz;
    float ekjx, ekjy, ekjz;
    eijx = rijx / Rij;
    eijy = rijy / Rij;
    eijz = rijz / Rij;

    ekjx = rkjx / Rkj;
    ekjy = rkjy / Rkj;
    ekjz = rkjz / Rkj;

    // angle
    float costh = eijx*ekjx + eijy*ekjy + eijz*ekjz;
    costh = qlib::min<float>(1.0f, qlib::max<float>(-1.0f, costh));
    float theta = ::acos(costh);
    float dtheta = (theta - pMol->m_angls[i].r0);

    //float Eangle = pMol->m_angls[i].kf*dtheta*dtheta;

    // calc gradient
    float df = 2.0*(pMol->m_angls[i].kf)*dtheta;
    float Dij;
    float Dkj;

    float sinth = sqrt(qlib::max<float>(0.0f, 1.0f-costh*costh));
    Dij =  df/(qlib::max<float>(F_EPS16, sinth)*Rij);
    Dkj =  df/(qlib::max<float>(F_EPS16, sinth)*Rkj);

    float vec_dijx = Dij*(costh*eijx - ekjx);
    float vec_dijy = Dij*(costh*eijy - ekjy);
    float vec_dijz = Dij*(costh*eijz - ekjz);
    
    float vec_dkjx = Dkj*(costh*ekjx - eijx);
    float vec_dkjy = Dkj*(costh*ekjy - eijy);
    float vec_dkjz = Dkj*(costh*ekjz - eijz);

    grad[ai*3+0] += vec_dijx;
    grad[ai*3+1] += vec_dijy;
    grad[ai*3+2] += vec_dijz;

    grad[aj*3+0] -= vec_dijx;
    grad[aj*3+1] -= vec_dijy;
    grad[aj*3+2] -= vec_dijz;

    grad[ak*3+0] += vec_dkjx;
    grad[ak*3+1] += vec_dkjy;
    grad[ak*3+2] += vec_dkjz;

    grad[aj*3+0] -= vec_dkjx;
    grad[aj*3+1] -= vec_dkjy;
    grad[aj*3+2] -= vec_dkjz;
  }

  if (gpMap!=NULL) {
    const float scale = gpMap->m_dScale;
    const int natoms = pMol->m_nAtoms;
    for (i=0; i<natoms; ++i) {
      //{ int i=4;

      qlib::Vector4D pos(pMol->m_crds[i*3+0],
			 pMol->m_crds[i*3+1],
			 pMol->m_crds[i*3+2]);

      float wgt = - pMol->m_mass[i] * scale;
      float Eden;
      Vector4D dv;

      gpMap->getGrad(pos, Eden, dv);
      //gpMap->getGradDesc(pos, Eden, dv);

      Eden *= wgt;
      dv = dv.scale(wgt);
      
      grad[i*3+0] += dv.x();
      grad[i*3+1] += dv.y();
      grad[i*3+2] += dv.z();

      //printf("atom %d (%f,%f,%f) Eden=%f grad=(%f,%f,%f)\n", i, pos.x(), pos.y(), pos.z(), Eden, dv.x(), dv.y(), dv.z());
    }
  }

}

void calc_E(MolData *pMol, double &rval)
{
  int i;
  int nbond = pMol->m_nBonds;
  int ncrd = pMol->m_nAtoms * 3;

  rval = 0.0;

  if (gbBond)
  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    float dx = pMol->m_crds[atom_i*3+0] - pMol->m_crds[atom_j*3+0];
    float dy = pMol->m_crds[atom_i*3+1] - pMol->m_crds[atom_j*3+1];
    float dz = pMol->m_crds[atom_i*3+2] - pMol->m_crds[atom_j*3+2];

    float sqlen = dx*dx + dy*dy + dz*dz;
    float len = sqrt(sqlen);
    float ss = len - pMol->m_bonds[i].r0;

    rval += pMol->m_bonds[i].kf * ss * ss;

    if (atom_i==0 && atom_j==4) {
      //      printf("atom %d-%d dv=%f,%f,%f", atom_i, atom_j, dx, dy,dz);
      //      printf("  sqlen=%f, dr=%f\n", len, len-pMol->m_bonds[i].r0);
    }
  }

  int nangl = pMol->m_nAngls;

  if (gbAngl)
  for (i=0; i<nangl; ++i) {
    int ai = pMol->m_angls[i].atom_i;
    int aj = pMol->m_angls[i].atom_j;
    int ak = pMol->m_angls[i].atom_k;

    float rijx, rijy, rijz;
    float rkjx, rkjy, rkjz;
    float Rij, Rkj;

    rijx = pMol->m_crds[ai*3+0] - pMol->m_crds[aj*3+0];
    rijy = pMol->m_crds[ai*3+1] - pMol->m_crds[aj*3+1];
    rijz = pMol->m_crds[ai*3+2] - pMol->m_crds[aj*3+2];

    rkjx = pMol->m_crds[ak*3+0] - pMol->m_crds[aj*3+0];
    rkjy = pMol->m_crds[ak*3+1] - pMol->m_crds[aj*3+1];
    rkjz = pMol->m_crds[ak*3+2] - pMol->m_crds[aj*3+2];

    // distance
    Rij = sqrt(qlib::max<float>(F_EPS16, rijx*rijx + rijy*rijy + rijz*rijz));
    Rkj = sqrt(qlib::max<float>(F_EPS16, rkjx*rkjx + rkjy*rkjy + rkjz*rkjz));

    // normalization
    float eijx, eijy, eijz;
    float ekjx, ekjy, ekjz;
    eijx = rijx / Rij;
    eijy = rijy / Rij;
    eijz = rijz / Rij;

    ekjx = rkjx / Rkj;
    ekjy = rkjy / Rkj;
    ekjz = rkjz / Rkj;

    // angle
    float costh = eijx*ekjx + eijy*ekjy + eijz*ekjz;
    costh = qlib::min<float>(1.0f, qlib::max<float>(-1.0f, costh));
    float theta = ::acos(costh);
    float dtheta = (theta - pMol->m_angls[i].r0);

    float Eangle = pMol->m_angls[i].kf*dtheta*dtheta;

    rval += Eangle;
  }

  if (gpMap!=NULL) {
    const float scale = gpMap->m_dScale;
    const int natoms = pMol->m_nAtoms;
    for (i=0; i<natoms; ++i) {
    //{ int i=4;
      qlib::Vector4D pos(pMol->m_crds[i*3+0],
			 pMol->m_crds[i*3+1],
			 pMol->m_crds[i*3+2]);

      float wgt = - pMol->m_mass[i] * scale;
      //float Eden = wgt * gpMap->getDensity(pos);
      float Eden = wgt * gpMap->getDensityCubic(pos);

      rval += Eden;
      
      //printf("atom %d (%f,%f,%f) Eden=%f\n", i, pos.x(), pos.y(), pos.z(), Eden);
    }
  }
}

inline void copyToGsl(gsl_vector *g, std::vector<float> grad)
{
  int i, ncrd = grad.size();
  for (i=0; i<ncrd; ++i)
    gsl_vector_set(g, i, grad[i]);
}

inline void copyToMol(MolData *pMol, const gsl_vector *x)
{
  int i, ncrd = pMol->m_nAtoms * 3;
  for (i=0; i<ncrd; ++i)
    pMol->m_crds[i] = float( gsl_vector_get(x, i) );
}

void target_df(const gsl_vector *x, void *params, gsl_vector *g)
{
  MolData *pMol = static_cast<MolData *>(params);
  int ncrd = pMol->m_nAtoms * 3;
  std::vector<float> grad(ncrd);

  copyToMol(pMol, x);

  //printf("Atom0 %f,%f,%f\n", pMol->m_crds[0], pMol->m_crds[1], pMol->m_crds[2]);
  calc_gradE(pMol, grad);

  copyToGsl(g, grad);
}

double target_f(const gsl_vector *x, void *params)
{
  MolData *pMol = static_cast<MolData *>(params);
  double rval;
  
  copyToMol(pMol, x);

  //printf("Atom0 %f,%f,%f\n", pMol->m_crds[0], pMol->m_crds[1], pMol->m_crds[2]);

  calc_E(pMol, rval);

  printf("target f E=%f\n", rval);
  return rval;
}

void target_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
  *f = target_f(x, params);
  target_df(x, params, g);
}

int main(int argc, char* argv[])
{
  if (argc!=4) {
    printf("usage: %s <parm> <pdb> <map>\n", argv[0]);
    return 0;
  }

  string parm = argv[1];
  string pdbin = argv[2];
  LString mapin = argv[3];
  /*
  DensityMap *pMap = new DensityMap;
  pMap->loadCNSMap(mapin);
  gpMap = pMap;
  */
  gpMap = NULL;

  MolData *pMol = new MolData;

  pMol->loadparm(parm);

  pMol->loadPDB(pdbin);

  /*{
    Vector4D pos(pMol->m_crds[0],pMol->m_crds[1],pMol->m_crds[2]);
    Vector4D grad0, grad1;
    float rho0, rho1;
    const double delta = 0.1;
    for (int i=0; i<100; ++i) {
      //float rho0 = pMap->getDensity(pos+Vector4D(0, 0, i*delta));
      //float rho1 = pMap->getDensityCubic(pos+Vector4D(0, 0, i*delta));
      pMap->getGrad(pos+Vector4D(0, 0, i*delta), rho0, grad0);
      pMap->getGradDesc(pos+Vector4D(0, 0, i*delta), rho1, grad1);
      //printf("%d %f %f %f %f\n", i, rho0, grad0.x(), grad0.y(), grad0.z());
      printf("%d %f %f\n", i, grad0.z(), grad1.z());
    }
    return 0;
    }*/

  //pMol->addRand(0.5);

  CudaData2 *pDat = prepBondCuda2(pMol);
  //CudaData *pDat = prepBondCuda(pMol);
  std::vector<float> grad(pMol->m_nCrds);
  for (int i=0; i<100000; i++) {
    //{
    gradBondCuda2(pMol, pDat, grad);
    //gradBondCuda(pMol, pDat, grad);
    //gradBond(pMol, grad);
  }
  return 0;

  pMol->savePDB("init.pdb");

  int ncrd = pMol->m_nAtoms * 3;

  gsl_multimin_function_fdf targ_func;

  printf("ncrd=%d, nbond=%d\n", ncrd, pMol->m_nBonds);
  targ_func.n = ncrd;
  targ_func.f = target_f;
  targ_func.df = target_df;
  targ_func.fdf = target_fdf;
  targ_func.params = pMol;

  const gsl_multimin_fdfminimizer_type *pMinType;
  gsl_multimin_fdfminimizer *pMin;

  pMinType = gsl_multimin_fdfminimizer_conjugate_pr;

  pMin = gsl_multimin_fdfminimizer_alloc(pMinType, ncrd);

  gsl_vector *x = gsl_vector_alloc(ncrd);
  copyToGsl(x, pMol->m_crds);
  float tolerance = 0.06;
  double step_size = 0.1 * gsl_blas_dnrm2(x);

  gsl_multimin_fdfminimizer_set(pMin, &targ_func, x, step_size, tolerance);

  int iter=0, status;

  do {
    printf("iter = %d\n", iter);
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(pMin);
    
    if (status)
      break;

    status = gsl_multimin_test_gradient(pMin->gradient, 1e-3);
    
    if (status == GSL_SUCCESS)
      printf("Minimum found\n");
    
  }
  while (status == GSL_CONTINUE && iter < 10000);

  printf("status = %d\n", status);
  copyToMol(pMol, pMin->x);

  //printf("Atom0 %f,%f,%f\n", pMol->m_crds[0], pMol->m_crds[1], pMol->m_crds[2]);

  gsl_multimin_fdfminimizer_free(pMin);
  gsl_vector_free(x);

  pMol->savePDB("outout.pdb");
}

#endif
