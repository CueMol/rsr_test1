#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"
//#include "grad_bond.hpp"
//#include "grad_map.hpp"

using namespace std;
using qlib::LString;

static inline void copyToGsl(gsl_vector *dst, const std::vector<float> &src)
{
  int i;
  const int ncrd = src.size();
  for (i=0; i<ncrd; ++i)
    gsl_vector_set(dst, i, src[i]);
}

inline void copyToVec(std::vector<float> &dst, const gsl_vector *src)
{
  int i;
  const int ncrd = dst.size();
  for (i=0; i<ncrd; ++i)
    dst[i] = float( gsl_vector_get(src, i) );
}



static void calc_fdf_cpu(const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
  MiniTarg *pMin = static_cast<MiniTarg *>(params);

  copyToVec(pMin->m_pMol->m_crds, x);

  float energy = 0.0f;
  const std::vector<float> &grad = pMin->calc(energy);
  //printf("calc force CPU OK\n");

  //printf("copy to gsl %p from vec %p\n", g, &grad);
  if (g!=NULL)
    copyToGsl(g, grad);
  *f = energy;

  // printf("target fdf OK\n");
}

static double calc_f_cpu(const gsl_vector *x, void *params)
{
  double energy;
  calc_fdf_cpu(x, params, &energy, NULL);
  return energy;
}

static void calc_df_cpu(const gsl_vector *x, void *params, gsl_vector *g)
{
  double dummy;
  calc_fdf_cpu(x, params, &dummy, g);
}

void MinGSL::minimize()
{
  MolData *pMol = m_pMiniTarg->m_pMol;
  int ncrd = pMol->m_nCrds;

  gsl_multimin_function_fdf targ_func;

  printf("ncrd=%d, nbond=%d\n", ncrd, pMol->m_nBonds);
  targ_func.n = ncrd;
  targ_func.f = calc_f_cpu;
  targ_func.df = calc_df_cpu;
  targ_func.fdf = calc_fdf_cpu;
  targ_func.params = m_pMiniTarg;

  const gsl_multimin_fdfminimizer_type *pMinType;
  gsl_multimin_fdfminimizer *pMin;

  pMinType = gsl_multimin_fdfminimizer_conjugate_pr;
  //pMinType = gsl_multimin_fdfminimizer_vector_bfgs2;

  pMin = gsl_multimin_fdfminimizer_alloc(pMinType, ncrd);

  gsl_vector *x = gsl_vector_alloc(ncrd);
  copyToGsl(x, pMol->m_crds);
  float tolerance = 0.06;
  double step_size = 0.1 * gsl_blas_dnrm2(x);

  printf("set step=%f, tol=%f\n", step_size, tolerance);

  gsl_multimin_fdfminimizer_set(pMin, &targ_func, x, step_size, tolerance);
  printf("set OK\n");

  int iter=0, status;

  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(pMin);
    
    if (status)
      break;

    status = gsl_multimin_test_gradient(pMin->gradient, 1e-3);
    
    if (status == GSL_SUCCESS)
      printf("Minimum found\n");
    
    printf("iter = %d energy=%f\n", iter, pMin->f);
  }
  while (status == GSL_CONTINUE && iter < m_nMaxIter);

  printf("status = %d\n", status);
  copyToVec(pMol->m_crds, pMin->x);

  //printf("Atom0 %f,%f,%f\n", pMol->m_crds[0], pMol->m_crds[1], pMol->m_crds[2]);

  gsl_multimin_fdfminimizer_free(pMin);
  gsl_vector_free(x);
}

