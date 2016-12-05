#include <stdio.h>

#include <common.h>
#include <qlib/LString.hpp>
#include <qlib/Utils.hpp>
#include <qlib/LExceptions.hpp>

#include <lbfgs.h>

#include "minimize.hpp"
#include "mol.hpp"
#include "map.hpp"
//#include "grad_bond.hpp"
//#include "grad_map.hpp"

using namespace std;
using qlib::LString;

static lbfgsfloatval_t evaluate(void *instance,
				const lbfgsfloatval_t *x,
				lbfgsfloatval_t *g,
				const int n,
				const lbfgsfloatval_t step)
{
  /*
  int i;
  lbfgsfloatval_t fx = 0.0;
  
  for (i = 0;i < n;i += 2) {
    lbfgsfloatval_t t1 = 1.0 - x[i];
    lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
    g[i+1] = 20.0 * t2;
    g[i] = -2.0 * (x[i] * g[i+1] + t1);
    fx += t1 * t1 + t2 * t2;
  }
  return fx;*/

  MiniTarg *pMin = static_cast<MiniTarg *>(instance);

  int i;
  const int ncrds = pMin->m_pMol->m_nCrds;

  for (i=0; i<ncrds; ++i)
    pMin->m_pMol->m_crds[i] = x[i];
  float fx = 0.0f;
  const std::vector<float> &grad = pMin->calc(fx);

  for (i=0; i<ncrds; ++i)
    g[i] = grad[i];

  return fx;
}

static int progress(void *instance,
		    const lbfgsfloatval_t *x,
		    const lbfgsfloatval_t *g,
		    const lbfgsfloatval_t fx,
		    const lbfgsfloatval_t xnorm,
		    const lbfgsfloatval_t gnorm,
		    const lbfgsfloatval_t step,
		    int n,
		    int k,
		    int ls)
{
  MiniTarg *pMin = static_cast<MiniTarg *>(instance);

  //{
  if (k%10==0) {
    printf("Iteration %d:\n", k);
    printf("  Etotal = %f ", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("  Ebond = %f", pMin->m_Ebond);
    printf("  Eangl = %f", pMin->m_Eangl);
    printf("  Edihe = %f\n", pMin->m_Edihe);
    printf("  Echir = %f", pMin->m_Echir);
    printf("  Eplan = %f", pMin->m_Eplan);
    printf("  Emap  = %f", pMin->m_Emap);
    printf("\n");
  }
  return 0;
}

LString errormsg(int nret)
{
  switch (nret) {
  case LBFGS_SUCCESS:
    return "LBFGS_SUCCESS";
    //  case LBFGS_CONVERGENCE:
    //    return "LBFGS_CONVERGENCE";
  case LBFGS_STOP:
    return "LBFGS_STOP";
    /** The initial variables already minimize the objective function. */
  case LBFGS_ALREADY_MINIMIZED:
    return "LBFGS_ALREADY_MINIMIZED";
    
   /** Unknown error. */
 case LBFGSERR_UNKNOWNERROR:
   return "LBFGSERR_UNKNOWNERROR";

   /** Logic error. */
 case LBFGSERR_LOGICERROR:
   return "LBFGSERR_LOGICERROR";

    /** Insufficient memory. */
 case LBFGSERR_OUTOFMEMORY:
   return "LBFGSERR_OUTOFMEMORY";

    /** The minimization process has been canceled. */
 case LBFGSERR_CANCELED:
 return "LBFGSERR_CANCELED";

    /** Invalid number of variables specified. */
 case LBFGSERR_INVALID_N:
   return "LBFGSERR_INVALID_N";

    /** Invalid number of variables (for SSE) specified. */
 case LBFGSERR_INVALID_N_SSE:
 return "LBFGSERR_INVALID_N_SSE";

    /** The array x must be aligned to 16 (for SSE). */
 case LBFGSERR_INVALID_X_SSE:
 return "LBFGSERR_INVALID_X_SSE";

    /** Invalid parameter lbfgs_parameter_t::epsilon specified. */
 case LBFGSERR_INVALID_EPSILON:
 return "LBFGSERR_INVALID_EPSILON";
    /** Invalid parameter lbfgs_parameter_t::past specified. */
 case LBFGSERR_INVALID_TESTPERIOD:
 return "LBFGSERR_INVALID_TESTPERIOD";
    /** Invalid parameter lbfgs_parameter_t::delta specified. */
 case LBFGSERR_INVALID_DELTA:
 return "LBFGSERR_INVALID_DELTA";
    /** Invalid parameter lbfgs_parameter_t::linesearch specified. */
 case LBFGSERR_INVALID_LINESEARCH:
 return "LBFGSERR_INVALID_LINESEARCH";
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
 case LBFGSERR_INVALID_MINSTEP:
 return "LBFGSERR_INVALID_MINSTEP";
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
 case LBFGSERR_INVALID_MAXSTEP:
 return "LBFGSERR_INVALID_MAXSTEP";
    /** Invalid parameter lbfgs_parameter_t::ftol specified. */
 case LBFGSERR_INVALID_FTOL:
 return "LBFGSERR_INVALID_FTOL";
    /** Invalid parameter lbfgs_parameter_t::wolfe specified. */
 case LBFGSERR_INVALID_WOLFE:
 return "LBFGSERR_INVALID_WOLFE";
    /** Invalid parameter lbfgs_parameter_t::gtol specified. */
 case LBFGSERR_INVALID_GTOL:
 return "LBFGSERR_INVALID_GTOL";
    /** Invalid parameter lbfgs_parameter_t::xtol specified. */
 case LBFGSERR_INVALID_XTOL:
 return "LBFGSERR_INVALID_XTOL";
    /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
 case LBFGSERR_INVALID_MAXLINESEARCH:
 return "LBFGSERR_INVALID_MAXLINESEARCH";
    /** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
 case LBFGSERR_INVALID_ORTHANTWISE:
 return "LBFGSERR_INVALID_ORTHANTWISE";
    /** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
 case LBFGSERR_INVALID_ORTHANTWISE_START:
 return "LBFGSERR_INVALID_ORTHANTWISE_START";
    /** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
 case LBFGSERR_INVALID_ORTHANTWISE_END:
 return "LBFGSERR_INVALID_ORTHANTWISE_END";
    /** The line-search step went out of the interval of uncertainty. */
 case LBFGSERR_OUTOFINTERVAL:
 return "LBFGSERR_OUTOFINTERVAL";
    /** A logic error occurred; alternatively: the interval of uncertainty
        became too small. */
 case LBFGSERR_INCORRECT_TMINMAX:
 return "LBFGSERR_INCORRECT_TMINMAX";
    /** A rounding error occurred; alternatively: no line-search step
        satisfies the sufficient decrease and curvature conditions. */
 case LBFGSERR_ROUNDING_ERROR:
 return "LBFGSERR_ROUNDING_ERROR";
    /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
 case LBFGSERR_MINIMUMSTEP:
 return "LBFGSERR_MINIMUMSTEP";
    /** The line-search step became larger than lbfgs_parameter_t::max_step. */
 case LBFGSERR_MAXIMUMSTEP:
 return "LBFGSERR_MAXIMUMSTEP";
    /** The line-search routine reaches the maximum number of evaluations. */
 case LBFGSERR_MAXIMUMLINESEARCH:
 return "LBFGSERR_MAXIMUMLINESEARCH";
    /** The algorithm routine reaches the maximum number of iterations. */
 case LBFGSERR_MAXIMUMITERATION:
 return "LBFGSERR_MAXIMUMITERATION";
    /** Relative width of the interval of uncertainty is at most
        lbfgs_parameter_t::xtol. */
 case LBFGSERR_WIDTHTOOSMALL:
 return "LBFGSERR_WIDTHTOOSMALL";
    /** A logic error (negative line-search step) occurred. */
 case LBFGSERR_INVALIDPARAMETERS:
 return "LBFGSERR_INVALIDPARAMETERS";
    /** The current search direction increases the objective function value. */
 case LBFGSERR_INCREASEGRADIENT:
 return "LBFGSERR_INCREASEGRADIENT";

 default:
   return "unknown";
 }
}


void MinLBFGS::minimize()
{
  printf("LBFGS mini\n");
  MolData *pMol = m_pMiniTarg->m_pMol;
  //DensityMap *pMap = m_pMiniTarg->m_pMap;

  printf("pMol=%p\n", pMol);
  int ncrds = pMol->m_nCrds;
  int i, ret = 0;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(ncrds);
  lbfgs_parameter_t param;
  
  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return;
  }
  
  /* Initialize the variables. */
  for (i = 0;i < ncrds; i++) {
    x[i] = pMol->m_crds[i];
  }

  /* Initialize the parameters for the L-BFGS optimization. */
  lbfgs_parameter_init(&param);
  //param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
  //param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  param.max_iterations = m_nMaxIter;
  printf("L-BFGS maxiter=%d\n", m_nMaxIter);

  /*
    Start the L-BFGS optimization; this will invoke the callback functions
    evaluate() and progress() when necessary.
  */
  ret = lbfgs(ncrds, x, &fx, evaluate, progress, m_pMiniTarg, &param);
  
  /* Report the result. */
  printf("L-BFGS optimization terminated with status code = %s\n", errormsg(ret).c_str());
  //printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
  
  for (i=0; i<ncrds; ++i)
    pMol->m_crds[i] = x[i];

  printf("  Etotal = %f\n", fx);
  //printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("  Ebond = %f", m_pMiniTarg->m_Ebond);
  printf("  Eangl = %f", m_pMiniTarg->m_Eangl);
  printf("  Edihe = %f\n", m_pMiniTarg->m_Edihe);
  printf("  Echir = %f", m_pMiniTarg->m_Echir);
  printf("  Eplan = %f", m_pMiniTarg->m_Eplan);
  printf("  Emap  = %f", m_pMiniTarg->m_Emap);
  printf("\n");

  lbfgs_free(x);
  return;
}
