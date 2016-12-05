#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>
#include <qlib/Vector4D.hpp>

#include "minimize.hpp"
#include "mol.hpp"
//#define DEBUG_PRINT 1
#define USE_IMPR

#include "calc_dihe.hpp"

//#define DIHE_START 0000
//#define DIHE_END (DIHE_START+30)

#define DIHE_START 0
#define DIHE_END (DIHE_START+1000000)

using namespace std;
using qlib::LString;
using qlib::Vector4D;

void MiniTargCPU::calcDiheEng()
{
  int i;
  int ndihe = qlib::min(DIHE_END,m_pMol->m_nDihes);//m_pMol->m_nDihes;
  int ncrd = m_pMol->m_nCrds;
  m_Edihe = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Dihe> dihes = m_pMol->m_dihes;


  for (i=DIHE_START; i<ndihe; ++i) {
    const int ai = dihes[i].atom_i*3;
    const int aj = dihes[i].atom_j*3;
    const int ak = dihes[i].atom_k*3;
    const int al = dihes[i].atom_l*3;

    realnum_t phi= calcDihe(ai, aj, ak, al, crds);

    realnum_t ss = calcAngleDiff(phi, dihes[i].r0);

#ifdef DEBUG_PRINT
    printf("%d %d %d %d phi=%f, dphi=%f\n", ai/3, aj/3, ak/3, al/3,
	   qlib::toDegree(phi), qlib::toDegree(ss));
#endif

    const int npe = dihes[i].npe;
    if (npe<=0) {
#ifdef USE_IMPR
      m_Edihe += dihes[i].kf * ss * ss;      
#endif
    }
    else {
      m_Edihe += dihes[i].kf * 9600.0f /realnum_t(npe*npe) * (1.0f-cos(realnum_t(npe)*ss));

#ifdef DEBUG_PRINT
      printf("            1-cos=%f, E=%f\n",
	     (1.0f-cos(realnum_t(npe)*ss)),
	     dihes[i].kf * 9600.0f /realnum_t(npe*npe) * (1.0f-cos(realnum_t(npe)*ss)));
#endif
    }

  }

  m_energy += m_Edihe;
#ifdef DEBUG_PRINT
  /*
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = m_Edihe;
  printf("CPU Edihe: %.16e [%x]\n", u.f, u.ui);
  */

  printf("CPU Edihe: %f\n", m_Edihe);
#endif
}

void MiniTargCPU::calcDiheFce()
{
  int i;
  //int ndihe = m_pMol->m_nDihes;
  int ndihe = qlib::min(DIHE_END,m_pMol->m_nDihes);//m_pMol->m_nDihes;
  int ncrd = m_pMol->m_nCrds;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Dihe> dihes = m_pMol->m_dihes;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  Vector4D r1, r2, r3, r4;
  Vector4D r12, r23, r34, A, B, C;

  Vector4D eA, eB, eC, dcosdA, dcosdB, dsindB, dsindC;
  Vector4D f1, f2, f3;
  realnum_t dE;
  //realnum_t energy = 0.0f;

  for (i=DIHE_START; i<ndihe; ++i) {
    const int ai = dihes[i].atom_i*3;
    const int aj = dihes[i].atom_j*3;
    const int ak = dihes[i].atom_k*3;
    const int al = dihes[i].atom_l*3;

    realnum_t phi = calcDiheDiff(ai, aj, ak, al, crds,
				 f1, f2, f3);

    realnum_t ss = calcAngleDiff(phi, dihes[i].r0);

    const int npe = dihes[i].npe;
    if (npe<=0) {
#ifdef USE_IMPR
      dE = 2.0 * dihes[i].kf * ss;      
#else
      dE = 0.0;
#endif
    }
    else {
      //energy += dihes[i].kf * 9600.0f /realnum_t(npe*npe) * (1.0f-cos(realnum_t(npe)*ss));
      dE = dihes[i].kf * 9600.0f / realnum_t(npe) * sin(realnum_t(npe)*ss);
      //dE = 0.0;
    }

    f1 = f1.scale(dE);
    f2 = f2.scale(dE);
    f3 = f3.scale(dE);

#ifdef DEBUG_PRINT
    printf("%d %d %d %d phi=%f, ph0=%f, dphi=%f\n", ai/3, aj/3, ak/3, al/3,
	   qlib::toDegree(phi), qlib::toDegree(dihes[i].r0), qlib::toDegree(ss));
    printf("   sinph=%f, cosph=%f, dE=%f\n", sin_phi, cos_phi, dE);

    printf("  f1: %f, %f, %f\n", f1.x(), f1.y(), f1.z());
    printf("  f2: %f, %f, %f\n", f2.x(), f2.y(), f2.z());
    printf("  f3: %f, %f, %f\n", f3.x(), f3.y(), f3.z());
#endif

    g[ai+0] += f1.x();
    g[ai+1] += f1.y();
    g[ai+2] += f1.z();

    g[aj+0] -= f1.x();
    g[aj+1] -= f1.y();
    g[aj+2] -= f1.z();

    g[aj+0] += f2.x();
    g[aj+1] += f2.y();
    g[aj+2] += f2.z();

    g[ak+0] -= f2.x();
    g[ak+1] -= f2.y();
    g[ak+2] -= f2.z();
    
    g[ak+0] += f3.x();
    g[ak+1] += f3.y();
    g[ak+2] += f3.z();

    g[al+0] -= f3.x();
    g[al+1] -= f3.y();
    g[al+2] -= f3.z();
  }

#ifdef DEBUG_PRINT
  /*
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = g[0];
  printf("CPU Dihe grad0: %.16e [%x]\n", u.f, u.ui);
  */
  //  for (i=0; i<ncrd; i++) {
  //    printf("grad %d %f\n", g[i]);
  //  }
#endif

  for (i=0; i<ncrd; i++) {
    m_grad[i] += g[i];
  }
}

/////////////////////////////////////////////////////////////////////////
// 

void MiniTargCPU::calcDiheEng2()
{
  int i;
  int ndihe = qlib::min(DIHE_END,m_pMol->m_nDihes);//m_pMol->m_nDihes;
  int ncrd = m_pMol->m_nCrds;
  m_Edihe = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Dihe> dihes = m_pMol->m_dihes;

  Vector4D ri, rj, rk, rl;
  Vector4D rij, rkj, rkl, rmj, rnk, C;
  Vector4D rjk, A, B;

  for (i=DIHE_START; i<ndihe; ++i) {
    const int ai = dihes[i].atom_i*3;
    const int aj = dihes[i].atom_j*3;
    const int ak = dihes[i].atom_k*3;
    const int al = dihes[i].atom_l*3;

    realnum_t phi = calcDihe2(ai, aj, ak, al, crds);
    realnum_t ss = calcAngleDiff(phi, dihes[i].r0);

    const int npe = dihes[i].npe;
    //printf("Dihe %d %d %d %d: npe = %d\n", ai/3, aj/3, ak/3, al/3, npe);
    if (npe<=0) {
#ifdef USE_IMPR
      m_Edihe += dihes[i].kf * ss * ss;      
#endif
    }
    else {
      realnum_t resid = dihes[i].kf * 9600.0f /realnum_t(npe*npe) * (1.0f-cos(realnum_t(npe)*ss));
      m_Edihe += resid;
      //printf("Dihe %d %d %d %d: resid = %f\n", ai/3, aj/3, ak/3, al/3, resid);
    }
  }

  m_energy += m_Edihe;
#ifdef DEBUG_PRINT
  /*
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = m_Edihe;
  printf("CPU Edihe: %.16e [%x]\n", u.f, u.ui);
  */
  printf("CPU Edihe2: %f\n", m_Edihe);
#endif
}


void MiniTargCPU::calcDiheFce2()
{
  int i;
  //int ndihe = m_pMol->m_nDihes;
  int ndihe = qlib::min(DIHE_END,m_pMol->m_nDihes);//m_pMol->m_nDihes;
  int ncrd = m_pMol->m_nCrds;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Dihe> dihes = m_pMol->m_dihes;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0;

  Vector4D dPhidRi, dPhidRl, dPhidRj, dPhidRk;

  realnum_t dE;
  //realnum_t energy = 0.0f;

  for (i=DIHE_START; i<ndihe; ++i) {
    const int ai = dihes[i].atom_i*3;
    const int aj = dihes[i].atom_j*3;
    const int ak = dihes[i].atom_k*3;
    const int al = dihes[i].atom_l*3;

    realnum_t phi=calcDiheDiff2(ai, aj, ak, al, crds,
				dPhidRi, dPhidRl, dPhidRj, dPhidRk);

    realnum_t ss = calcAngleDiff(phi, dihes[i].r0);

#ifdef DEBUG_PRINT
    printf("%d %d %d %d phi = %f, ss = %f\n", ai/3, aj/3, ak/3, al/3, qlib::toDegree(phi), qlib::toDegree(ss));
#endif

    const int npe = dihes[i].npe;
    if (npe<=0) {
#ifdef USE_IMPR
      dE = 2.0 * dihes[i].kf * ss;      
#else
      dE = 0.0;
#endif
    }
    else {
      //energy += dihes[i].kf * 9600.0f /realnum_t(npe*npe) * (1.0f-cos(realnum_t(npe)*ss));
      dE = dihes[i].kf * 9600.0f / realnum_t(npe) * sin(realnum_t(npe)*ss);
      //dE = 0.0;
    }

    dPhidRi = dPhidRi.scale(dE);
    dPhidRj = dPhidRj.scale(dE);
    dPhidRk = dPhidRk.scale(dE);
    dPhidRl = dPhidRl.scale(dE);

    g[ai+0] += dPhidRi.x();
    g[ai+1] += dPhidRi.y();
    g[ai+2] += dPhidRi.z();

    g[aj+0] += dPhidRj.x();
    g[aj+1] += dPhidRj.y();
    g[aj+2] += dPhidRj.z();

    g[ak+0] += dPhidRk.x();
    g[ak+1] += dPhidRk.y();
    g[ak+2] += dPhidRk.z();

    g[al+0] += dPhidRl.x();
    g[al+1] += dPhidRl.y();
    g[al+2] += dPhidRl.z();

  }

#ifdef DEBUG_PRINT
  /*
  union {
    realnum_t f;
    unsigned int ui;
  } u;
  u.f = g[0];
  printf("CPU Dihe grad0: %.16e [%x]\n", u.f, u.ui);
  */
  //  for (i=0; i<ncrd; i++) {
  //    printf("grad %d %f\n", g[i]);
  //  }
#endif

  for (i=0; i<ncrd; i++) {
    m_grad[i] += g[i];
  }
}

