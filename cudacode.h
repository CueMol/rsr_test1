// -*-Mode: C++;-*-

#ifndef CUDACODE_H_INCLUDED
#define CUDACODE_H_INCLUDED

//#ifdef __cplusplus
//extern "C" {
//#endif

#define THR_PER_BLK 1024

struct CuBond
{
  int ai;
  //int ai_ord;
  int aj;
  //int aj_ord;
  float kf;
  float r0;
};

struct CuAngl
{
  int ai;
  int aj;
  int ak;
  int flag;

  float kf;
  float r0;
};

struct CudaData
{

  CudaData() : pd_crds(NULL), pd_bond(NULL), pd_bvec(NULL), pd_grad(NULL), pd_eatm(NULL)
  {
  }

  int nacc;
  std::vector<int> bvec;
  std::vector<CuBond> cubonds;
  /*
  int nang_acc;
  std::vector<int> ang_ivec;
  std::vector<CuAngl> cuangls;
  */
  // device memory
  float *pd_crds;
  CuBond *pd_bond;
  int *pd_bvec;
  float *pd_grad;
  float *pd_eatm;

  int nthr, nblk;
  int dev_natom;
};

struct CudaData2
{

  CudaData2() : pd_crds(NULL), pd_bond(NULL), pd_bvec(NULL), pd_grad(NULL), pd_eatm(NULL)
  {
  }

  int nthr, nblk;

  std::vector<int> bvec;
  std::vector<CuBond> cubonds;

  // device memory
  float *pd_crds;
  CuBond *pd_bond;
  int *pd_bvec;
  float *pd_grad;
  float *pd_eatm;

};

  
void cudaBond_fdf(const std::vector<float> &crds,
		  CudaData *pDat,
		  float *val, std::vector<float> &grad);

void cudaBond_fdf2(const std::vector<float> &crds,
		   CudaData2 *pDat,
		   float *val, std::vector<float> &grad);


struct CudaMapData
{
  CudaMapData() : p_map(NULL), pd_crds(NULL), pd_wgts(NULL), pd_map(NULL), pd_grad(NULL), pd_eatm(NULL)
  {
  }

  int nthr, nblk;

  int ncol, nrow, nsec;
  int na, nb, nc;
  int stcol, strow, stsec;
  std::vector<float> fracMat;

  std::vector<float> wgts;
  float *p_map;
  std::vector<float> eatom;

  // device memory
  float *pd_crds;
  float *pd_wgts;
  float *pd_fracMat;
  void *pd_map;
  float *pd_grad;
  float *pd_eatm;
};

void cudaMap_fdf1(const std::vector<float> &crds,
		 CudaMapData *pDat,
		 float *val, std::vector<float> &grad);

void cudaMap_fdf2(const std::vector<float> &crds,
		  CudaMapData *pDat,
		  float *val, std::vector<float> &grad);


#endif
