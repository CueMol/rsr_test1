#ifndef GRAD_MAP_HPP_INCLUDED
#define GRAD_MAP_HPP_INCLUDED

//#define DEBUG_PRINT 1

void gradMap(MolData *pMol, DensityMap *pMap, std::vector<float> &grad)
{
  int i;
  int ncrd = pMol->m_nAtoms * 3;
  float Emap = 0.0f, den;

  const float scale = pMap->m_dScale;
  const int natoms = pMol->m_nAtoms;
  std::vector<float> &crds = pMol->m_crds;

  Vector4D pos;
  //Vector4D ngrid(pMap->m_na, pMap->m_nb, pMap->m_nc);
  //Vector4D stagrid(pMap->m_stacol, pMap->m_starow, pMap->m_stasect);
  //Matrix3D &fracMat = pMap->m_fracMat;

  for (i=0; i<natoms; ++i) {
    //{ int i=4;
    pos.x() = crds[i*3+0];
    pos.y() = crds[i*3+1];
    pos.z() = crds[i*3+2];


    float wgt = - pMol->m_mass[i] * scale;
    Vector4D dv;
    
    //den = pMap->getDensityCubic(pos);
    pMap->getGrad(pos, den, dv);
    //pMap->getGradDesc(pos, Eden, dv);
    dv = dv.scale(wgt);

    den *= wgt;

    //grad[i*3+0] = den;
    //grad[i*3+1] = 0;
    //grad[i*3+2] = 0;
    grad[i*3+0] = dv.x();
    grad[i*3+1] = dv.y();
    grad[i*3+2] = dv.z();

    /*
    pos = fracMat.mulvec(pos);
    pos = pos.scale(ngrid);
    pos -= stagrid;

    float xxx = pMap->getValue(int(pos.x()),
			       int(pos.y()),
			       int(pos.z()));

    grad[i*3+0] = pos.x();
    grad[i*3+1] = pos.y();
    grad[i*3+2] = pos.z();
    */

    Emap += den;

    //printf("atom %d (%f,%f,%f) Eden=%f grad=(%f,%f,%f)\n", i, pos.x(), pos.y(), pos.z(), Eden, dv.x(), dv.y(), dv.z());
  }

#ifdef DEBUG_PRINT
  int natom = pMol->m_nAtoms;
  for (i=0; i<natom; ++i) {
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  printf("CPU Emap = %f\n",Emap);
#endif
}

/////////////////////////////////////

CudaMapData *prepMapCuda1(DensityMap *pMap, MolData *pMol)
{
  const int natom = pMol->m_nAtoms;

  CudaMapData *pRet = new CudaMapData();
  pRet->ncol = pMap->m_ncol;
  pRet->nrow = pMap->m_nrow;
  pRet->nsec = pMap->m_nsect;

  pRet->na = pMap->m_na;
  pRet->nb = pMap->m_nb;
  pRet->nc = pMap->m_nc;

  pRet->stcol = pMap->m_stacol;
  pRet->strow = pMap->m_starow;
  pRet->stsec = pMap->m_stasect;

  pRet->fracMat.resize(9);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      pRet->fracMat[i + j*3] = pMap->m_fracMat.aij(i+1,j+1);

  pRet->p_map = &pMap->m_data[0];

  int nterm = natom;

  // setup exec layout
  if (nterm<THR_PER_BLK) {
    pRet->nblk = 1;
    pRet->nthr = THR_PER_BLK;
  }
  else if (nterm%THR_PER_BLK==0) {
    pRet->nblk = nterm/THR_PER_BLK;
    pRet->nthr = THR_PER_BLK;
  }
  else {
    pRet->nblk = nterm/THR_PER_BLK +1;
    pRet->nthr = THR_PER_BLK;
  }
  nterm = pRet->nblk * pRet->nthr;

  const float scale = pMap->m_dScale;
  pRet->wgts.resize(nterm);
  for (int i=0; i<nterm; ++i) {
    if (i<natom)
      pRet->wgts[i] = - pMol->m_mass[i] * scale;
    else
      pRet->wgts[i] = 0.0f;
    //printf("wgts: %d %f\n", i, pRet->wgts[i]);
  }

  pRet->eatom.resize(pRet->nblk);
  
  printf("prepMapCuda1 nblk x nthr = %d x %d\n", pRet->nblk, pRet->nthr);
  return pRet;
}


float gradMapCuda1(MolData *pMol, DensityMap *pMap, CudaMapData *pDat, std::vector<float> &grad)
{
  const int ncrds = pMol->m_nCrds;
  const int natom = pMol->m_nAtoms;

  std::vector<float> &crds = pMol->m_crds;
  Vector4D pos;
  /*
  Vector4D ngrid(pMap->m_na, pMap->m_nb, pMap->m_nc);
  Vector4D stagrid(pMap->m_stacol, pMap->m_starow, pMap->m_stasect);
  Matrix3D &fracMat = pMap->m_fracMat;
  std::vector<float> gcrds(ncrds);
  for (int i=0; i<natom; ++i) {
    pos.x() = crds[i*3+0];
    pos.y() = crds[i*3+1];
    pos.z() = crds[i*3+2];
    pos = fracMat.mulvec(pos);
    pos = pos.scale(ngrid);
    pos -= stagrid;
    gcrds[i*3+0] = pos.x();
    gcrds[i*3+1] = pos.y();
    gcrds[i*3+2] = pos.z();
  }
  */

  float Emap;
  cudaMap_fdf1(crds, pDat, &Emap, grad);

#ifdef DEBUG_PRINT
  //int natom = pMol->m_nAtoms;
  int i;
  const float scale = pMap->m_dScale;
  for (i=0; i<natom; ++i) {
    /*
    pos.x() = grad[i*3+0];
    pos.y() = grad[i*3+1];
    pos.z() = grad[i*3+2];
    pos.x() *= pMap->m_na;
    pos.y() *= pMap->m_nb;
    pos.z() *= pMap->m_nc;

    Matrix3D gradorth = pMap->m_fracMat.transpose();
    pos = gradorth.mulvec(pos);

    float wgt = - pMol->m_mass[i] * scale;
    pos = pos.scale(wgt);
    printf("%d (%f,%f,%f)\n", i, pos.x(), pos.y(), pos.z() );
    */
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  //Emap *= wgt;
  printf("GPU Emap = %f\n",Emap);
#endif

  return Emap;
}


///////////////////////////

CudaMapData *prepMapCuda2(DensityMap *pMap, MolData *pMol)
{
  const int natom = pMol->m_nAtoms;
  
  CudaMapData *pRet = new CudaMapData();
  pRet->ncol = pMap->m_ncol;
  pRet->nrow = pMap->m_nrow;
  pRet->nsec = pMap->m_nsect;

  pRet->na = pMap->m_na;
  pRet->nb = pMap->m_nb;
  pRet->nc = pMap->m_nc;

  pRet->stcol = pMap->m_stacol;
  pRet->strow = pMap->m_starow;
  pRet->stsec = pMap->m_stasect;

  pRet->fracMat.resize(9);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      pRet->fracMat[i + j*3] = pMap->m_fracMat.aij(i+1,j+1);

  pRet->p_map = &pMap->m_data[0];

  int nterm = natom * 64;

  // setup exec layout
  if (nterm<THR_PER_BLK) {
    pRet->nblk = 1;
    pRet->nthr = THR_PER_BLK;
  }
  else if (nterm%THR_PER_BLK==0) {
    pRet->nblk = nterm/THR_PER_BLK;
    pRet->nthr = THR_PER_BLK;
  }
  else {
    pRet->nblk = nterm/THR_PER_BLK +1;
    pRet->nthr = THR_PER_BLK;
  }

  const float scale = pMap->m_dScale;
  const int nwgt = pRet->nblk * pRet->nthr / 64;
  pRet->wgts.resize(nwgt);
  for (int i=0; i<nwgt; ++i) {
    if (i<natom)
      pRet->wgts[i] = - pMol->m_mass[i] * scale;
    else
      pRet->wgts[i] = 0.0f;
    //printf("wgts: %d %f\n", i, pRet->wgts[i]);
  }

  pRet->eatom.resize(pRet->nblk);
  
  printf("prepMapCuda1 nblk x nthr = %d x %d\n", pRet->nblk, pRet->nthr);
  return pRet;
}

void gradMapCuda2(MolData *pMol, DensityMap *pMap, CudaMapData *pDat, std::vector<float> &grad)
{
  const int ncrds = pMol->m_nCrds;
  const int natom = pMol->m_nAtoms;

  std::vector<float> gcrds(ncrds);
  std::vector<float> &crds = pMol->m_crds;

  Vector4D pos;
  /*  
  Vector4D ngrid(pMap->m_na, pMap->m_nb, pMap->m_nc);
  Vector4D stagrid(pMap->m_stacol, pMap->m_starow, pMap->m_stasect);
  Matrix3D &fracMat = pMap->m_fracMat;
  for (int i=0; i<natom; ++i) {
    pos.x() = crds[i*3+0];
    pos.y() = crds[i*3+1];
    pos.z() = crds[i*3+2];
    pos = fracMat.mulvec(pos);
    pos = pos.scale(ngrid);
    pos -= stagrid;
    gcrds[i*3+0] = pos.x();
    gcrds[i*3+1] = pos.y();
    gcrds[i*3+2] = pos.z();
    }*/

  float Emap;
  cudaMap_fdf2(crds, pDat, &Emap, grad);

#ifdef DEBUG_PRINT
  //int natom = pMol->m_nAtoms;
  int i;
  //const float scale = pMap->m_dScale;
  for (i=0; i<natom; ++i) {
    /*    pos.x() = grad[i*3+0];
    pos.y() = grad[i*3+1];
    pos.z() = grad[i*3+2];

    pos.x() *= pMap->m_na;
    pos.y() *= pMap->m_nb;
    pos.z() *= pMap->m_nc;

    Matrix3D gradorth = pMap->m_fracMat.transpose();
    pos = gradorth.mulvec(pos);

    float wgt = - pMol->m_mass[i] * scale;
    pos = pos.scale(wgt);
    printf("%d (%f,%f,%f)\n", i, pos.x(), pos.y(), pos.z() );
    */
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  //Emap *= wgt;
  printf("GPU Emap = %f\n",Emap);
#endif
}

#endif
