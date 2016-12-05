#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>
#include <qlib/Vector4D.hpp>

#include "minimize.hpp"
#include "mol.hpp"
//#define DEBUG_PRINT 1

#include "calc_dihe.hpp"

using namespace std;
using qlib::LString;
using qlib::Vector4D;
using qlib::Vector2D;

void MiniTargCPU::calcRamaEng()
{
  int i;
  int nrama = m_pMol->m_nRamas;
  int ncrd = m_pMol->m_nCrds;
  m_Erama = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Rama> ramas = m_pMol->m_ramas;
  int ai, aj, ak, al;
  const double scl = 10.0;
  Vector2D pos;

  for (i=0; i<nrama; ++i) {
    ai = ramas[i].phi_ai*3;
    aj = ramas[i].phi_aj*3;
    ak = ramas[i].phi_ak*3;
    al = ramas[i].phi_al*3;

    realnum_t phi= calcDihe2(ai, aj, ak, al, crds);

    ai = ramas[i].psi_ai*3;
    aj = ramas[i].psi_aj*3;
    ak = ramas[i].psi_ak*3;
    al = ramas[i].psi_al*3;

    realnum_t psi= calcDihe2(ai, aj, ak, al, crds);

    pos.x() = phi;
    pos.y() = psi;
    realnum_t val = m_ramaplot.getDensity(pos);

#ifdef DEBUG_PRINT
    printf("%d %d %d %d\n", ai/3, aj/3, ak/3, al/3);
    printf("    phi=%f psi=%f logP %f\n",
	   qlib::toDegree(phi), qlib::toDegree(psi), val);
#endif

    m_Erama += scl*val;
  }

  m_energy += m_Erama;

#ifdef DEBUG_PRINT
  printf("CPU Erama: %f\n", m_Erama);
#endif
}

void MiniTargCPU::calcRamaFce()
{
  int i;
  int nrama = m_pMol->m_nRamas;
  int ncrd = m_pMol->m_nCrds;
  m_Erama = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Rama> ramas = m_pMol->m_ramas;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0;

  const double scl = 10.0;
  Vector2D pos, rama_grad;

  Vector4D dPhidRi, dPhidRl, dPhidRj, dPhidRk;
  Vector4D dPsidRi, dPsidRl, dPsidRj, dPsidRk;

  for (i=0; i<nrama; ++i) {
    const int phi_ai = ramas[i].phi_ai*3;
    const int phi_aj = ramas[i].phi_aj*3;
    const int phi_ak = ramas[i].phi_ak*3;
    const int phi_al = ramas[i].phi_al*3;

    realnum_t phi= calcDiheDiff2(phi_ai, phi_aj, phi_ak, phi_al, crds,
				 dPhidRi, dPhidRl, dPhidRj, dPhidRk);
    
    const int psi_ai = ramas[i].psi_ai*3;
    const int psi_aj = ramas[i].psi_aj*3;
    const int psi_ak = ramas[i].psi_ak*3;
    const int psi_al = ramas[i].psi_al*3;

    realnum_t psi= calcDiheDiff2(psi_ai, psi_aj, psi_ak, psi_al, crds,
				 dPsidRi, dPsidRl, dPsidRj, dPsidRk);

    pos.x() = phi;
    pos.y() = psi;
    rama_grad = m_ramaplot.getGrad(pos);

    realnum_t kphi = scl * rama_grad.x();
    realnum_t kpsi = scl * rama_grad.y();

    dPhidRi = dPhidRi.scale(kphi);
    dPhidRj = dPhidRj.scale(kphi);
    dPhidRk = dPhidRk.scale(kphi);
    dPhidRl = dPhidRl.scale(kphi);

    dPsidRi = dPsidRi.scale(kpsi);
    dPsidRj = dPsidRj.scale(kpsi);
    dPsidRk = dPsidRk.scale(kpsi);
    dPsidRl = dPsidRl.scale(kpsi);

    g[phi_ai+0] += dPhidRi.x();
    g[phi_ai+1] += dPhidRi.y();
    g[phi_ai+2] += dPhidRi.z();

    g[phi_aj+0] += dPhidRj.x();
    g[phi_aj+1] += dPhidRj.y();
    g[phi_aj+2] += dPhidRj.z();

    g[phi_ak+0] += dPhidRk.x();
    g[phi_ak+1] += dPhidRk.y();
    g[phi_ak+2] += dPhidRk.z();

    g[phi_al+0] += dPhidRl.x();
    g[phi_al+1] += dPhidRl.y();
    g[phi_al+2] += dPhidRl.z();


    g[psi_ai+0] += dPsidRi.x();
    g[psi_ai+1] += dPsidRi.y();
    g[psi_ai+2] += dPsidRi.z();

    g[psi_aj+0] += dPsidRj.x();
    g[psi_aj+1] += dPsidRj.y();
    g[psi_aj+2] += dPsidRj.z();

    g[psi_ak+0] += dPsidRk.x();
    g[psi_ak+1] += dPsidRk.y();
    g[psi_ak+2] += dPsidRk.z();

    g[psi_al+0] += dPsidRl.x();
    g[psi_al+1] += dPsidRl.y();
    g[psi_al+2] += dPsidRl.z();

#ifdef DEBUG_PRINT
    printf("%d %d %d %d\n", phi_ai/3, phi_aj/3, phi_ak/3, phi_al/3);
    printf("    phi=%f psi=%f logP grad %f, %f\n",
	   qlib::toDegree(phi), qlib::toDegree(psi),
	   rama_grad.x(), rama_grad.y());
#endif

    //m_Erama += scl*val;
  }

  //m_energy += m_Erama;

  for (i=0; i<ncrd; i++) {
    m_grad[i] += g[i];
  }
}
