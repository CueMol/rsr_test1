#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>
#include <qlib/Vector4D.hpp>
#include <qlib/Matrix3D.hpp>
#include <qlib/Utils.hpp>

#include "minimize.hpp"
#include "mol.hpp"

//#define DEBUG_PRINT 1

using namespace std;
using qlib::LString;
using qlib::Vector4D;
using qlib::Matrix3D;

static inline int find_emin(const Vector4D &evals)
{
  if (evals.x()<=evals.y() &&
      evals.x()<=evals.z())
    return 1;
  else if (evals.y()<=evals.x() &&
	   evals.y()<=evals.z())
    return 2;
  else
    return 3;
}


void MiniTargCPU::calcPlanEng()
{
  int i, j;
  int nplan = m_pMol->m_nPlans;
  int ncrd = m_pMol->m_nCrds;
  m_Eplan = 0.0f;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Plan> &plans = m_pMol->m_plans;

  Vector4D rc, r1, ev1;
  Matrix3D resid_tens;
  Matrix3D evecs;
  Vector4D evals;

  for (i=0; i<nplan; ++i) {
    const Plan &plane = plans[i];
    const int natoms = plane.atoms.size();

    rc = Vector4D(0,0,0,0);
    realnum_t wsum = 0.0;
    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      rc.x() += w*crds[ia + 0];
      rc.y() += w*crds[ia + 1];
      rc.z() += w*crds[ia + 2];
      wsum += w;
    }
    rc = rc.scale(1.0/wsum);
    //printf("Plane %d com (%f, %f, %f)\n", i, rc.x(), rc.y(), rc.z());
    for (j=1; j<=3; ++j)
      for (int k=1; k<=3; ++k)
	resid_tens.aij(j,k) = 0.0;

    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      r1.x() = crds[ia + 0] - rc.x();
      r1.y() = crds[ia + 1] - rc.y();
      r1.z() = crds[ia + 2] - rc.z();

      resid_tens.aij(1,1) += w * r1.x() * r1.x();
      resid_tens.aij(2,2) += w * r1.y() * r1.y();
      resid_tens.aij(3,3) += w * r1.z() * r1.z();
      resid_tens.aij(1,2) += w * r1.x() * r1.y();
      resid_tens.aij(1,3) += w * r1.x() * r1.z();
      resid_tens.aij(2,3) += w * r1.y() * r1.z();
    }
    resid_tens.aij(2,1) = resid_tens.aij(1,2);
    resid_tens.aij(3,1) = resid_tens.aij(1,3);
    resid_tens.aij(3,2) = resid_tens.aij(2,3);

    /*{
      printf("resid_tens:\n");
      printf("( %.5f %.5f %.5f )\n", resid_tens.aij(1,1), resid_tens.aij(1,2), resid_tens.aij(1,3));
      printf("( %.5f %.5f %.5f )\n", resid_tens.aij(2,1), resid_tens.aij(2,2), resid_tens.aij(2,3));
      printf("( %.5f %.5f %.5f )\n", resid_tens.aij(3,1), resid_tens.aij(3,2), resid_tens.aij(3,3));
      }*/

    if (!resid_tens.diag(evecs, evals)) {
      printf("Diag failed for plane:\n");
      for (j=0; j<natoms; ++j) {
	const int ia = plane.atoms[j].iatom;
	printf("  %s\n", m_pMol->m_pfx[ia].c_str());
      }
      continue;
    }
#ifdef DEBUG_PRINT
    printf("eigenval1 = %e\n", evals.x());
    printf("eigenval2 = %e\n", evals.y());
    printf("eigenval3 = %e\n", evals.z());
#endif
    /*{
      printf("evecs:\n");
      printf("( %.5f %.5f %.5f )\n", evecs.aij(1,1), evecs.aij(1,2), evecs.aij(1,3));
      printf("( %.5f %.5f %.5f )\n", evecs.aij(2,1), evecs.aij(2,2), evecs.aij(2,3));
      printf("( %.5f %.5f %.5f )\n", evecs.aij(3,1), evecs.aij(3,2), evecs.aij(3,3));
      }*/

    int emin = find_emin(evals);;
    ev1.x() = evecs.aij(1,emin);
    ev1.y() = evecs.aij(2,emin);
    ev1.z() = evecs.aij(3,emin);
    ev1.w() = 0.0;
    //resid_tens.xform(ev1);
#ifdef DEBUG_PRINT
    printf("emin = %d ( %.5f %.5f %.5f )\n", emin, ev1.x(), ev1.y(), ev1.z());
#endif

    realnum_t resid = 0.0;
    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      r1.x() = crds[ia + 0] - rc.x();
      r1.y() = crds[ia + 1] - rc.y();
      r1.z() = crds[ia + 2] - rc.z();
      r1.w() = 0.0;

      realnum_t del = ev1.dot(r1);
#ifdef DEBUG_PRINT
      printf("delta for %d = %f\n", j, del);
#endif
      resid += w * del * del;
    }

#ifdef DEBUG_PRINT
    printf("Plane %d resid = %f\n", i, resid);
#endif
    m_Eplan += resid;
  }

  m_energy += m_Eplan;

#ifdef DEBUG_PRINT
  printf("CPU Eplan: %f\n", m_Eplan);
#endif

}

void MiniTargCPU::calcPlanFce()
{
  int i, j;
  int nplan = m_pMol->m_nPlans;
  int ncrd = m_pMol->m_nCrds;

  const std::vector<float> &crds = m_pMol->m_crds;
  const std::vector<Plan> &plans = m_pMol->m_plans;

  std::vector<realnum_t> g(ncrd);
  for (i=0; i<ncrd; i++)
    g[i] = 0.0f;

  Vector4D rc, r1, ev1;
  Matrix3D resid_tens;
  Matrix3D evecs;
  Vector4D evals;

  for (i=0; i<nplan; ++i) {
    const Plan &plane = plans[i];
    const int natoms = plane.atoms.size();

    rc = Vector4D(0,0,0,0);
    realnum_t wsum = 0.0;
    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      rc.x() += w*crds[ia + 0];
      rc.y() += w*crds[ia + 1];
      rc.z() += w*crds[ia + 2];
      wsum += w;
    }
    rc = rc.scale(1.0/wsum);
    // printf("Plane %d com (%f, %f, %f)\n", i, rc.x(), rc.y(), rc.z());
    for (j=1; j<=3; ++j)
      for (int k=1; k<=3; ++k)
	resid_tens.aij(j,k) = 0.0;

    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      r1.x() = crds[ia + 0] - rc.x();
      r1.y() = crds[ia + 1] - rc.y();
      r1.z() = crds[ia + 2] - rc.z();

      resid_tens.aij(1,1) += w * r1.x() * r1.x();
      resid_tens.aij(2,2) += w * r1.y() * r1.y();
      resid_tens.aij(3,3) += w * r1.z() * r1.z();
      resid_tens.aij(1,2) += w * r1.x() * r1.y();
      resid_tens.aij(1,3) += w * r1.x() * r1.z();
      resid_tens.aij(2,3) += w * r1.y() * r1.z();
    }
    resid_tens.aij(2,1) = resid_tens.aij(1,2);
    resid_tens.aij(3,1) = resid_tens.aij(1,3);
    resid_tens.aij(3,2) = resid_tens.aij(2,3);

    if (!resid_tens.diag(evecs, evals)) {
      printf("diag failed\n");
      continue;
    }

    int emin = find_emin(evals);;
    ev1.x() = evecs.aij(1,emin);
    ev1.y() = evecs.aij(2,emin);
    ev1.z() = evecs.aij(3,emin);
    ev1.w() = 0.0;

    //printf("emin = %d ( %.5f %.5f %.5f )\n", emin, ev1.x(), ev1.y(), ev1.z());

    for (j=0; j<natoms; ++j) {
      const realnum_t w = plane.atoms[j].wgt;
      const int ia = plane.atoms[j].iatom*3;
      r1.x() = crds[ia + 0] - rc.x();
      r1.y() = crds[ia + 1] - rc.y();
      r1.z() = crds[ia + 2] - rc.z();
      r1.w() = 0.0;

      realnum_t del = ev1.dot(r1);
      //printf("delta for %d = %f\n", j, del);

      realnum_t dE = 2.0 * w * del;
      g[ia+0] += ev1.x()*dE;
      g[ia+1] += ev1.y()*dE;
      g[ia+2] += ev1.z()*dE;
    }
  }

#ifdef DEBUG_PRINT
  for (i=0; i<ncrd; i++) {
    if (!qlib::isNear4<double>(g[i], 0.0)) {
      printf("grad %d (%s) %f\n", i, m_pMol->m_pfx[i/3].c_str(), g[i]);
    }
  }
#endif

  for (i=0; i<ncrd; i++) {
    m_grad[i] += g[i];
  }
}

